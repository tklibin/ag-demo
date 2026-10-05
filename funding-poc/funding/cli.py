"""Command line: python -m funding {generate,train,evaluate,backtest,offers,explain,demo}"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data_gen, features
from .evaluate import evaluate
from .models import FundingModels
from .policy import Policy, backtest, make_offers, naive_offers

DATA = Path("data")
HISTORY = DATA / "claims_history.csv"
PENDING = DATA / "claims_pending.csv"
MODELS = DATA / "models"
OFFERS = DATA / "offers.csv"
AS_OF = "2026-08-01"  # train on claims before this date, test on Aug–Sep 2026


# --- commands -----------------------------------------------------------------------------------

def cmd_generate(args) -> None:
    history = data_gen.generate_history(args.per_day, seed=args.seed)
    pending = data_gen.generate_pending(args.per_day, seed=args.seed + 1)
    data_gen.save(history, HISTORY)
    data_gen.save(pending, PENDING)
    print(f"Wrote {len(history)} ED claims ({history['lock_date'].min():%Y-%m-%d} to "
          f"{history['lock_date'].max():%Y-%m-%d}, {history['approved'].mean():.1%} approved) to {HISTORY}")
    print(f"Wrote {len(pending)} pending claims at the lock point to {PENDING}")


def cmd_train(args) -> None:
    models = FundingModels.train(data_gen.load(HISTORY), pd.Timestamp(args.as_of))
    models.save(MODELS)
    i = models.info
    print(f"Trained as of {i.as_of}: approval on {i.n_approval_train} claims "
          f"(validated on {i.n_approval_valid}), payment timing on {i.n_payment_train}")
    print(f"  Validation ROC-AUC (approval) {i.approval_valid_auc:.3f}")
    print("  Trees kept by early stopping  " + ", ".join(f"{k} {v}" for k, v in i.best_iterations.items()))
    print("  Clawback rate by payer       " + ", ".join(f"{k} {v:.2%}" for k, v in models.clawback_rate.items()))
    print(f"Saved models to {MODELS}/")


def cmd_evaluate(args) -> None:
    models = FundingModels.load(MODELS)
    e = evaluate(models, data_gen.load(HISTORY), pd.Timestamp(models.info.as_of))
    print(f"== Approval on {e.n_test} claims locked after {models.info.as_of} (out of time) ==")
    print(e.approval.round(4).to_string())
    print("\n== Calibration: predicted vs actual approval by band ==")
    print(e.calibration.round(3).to_string())
    print("\n== By payer ==")
    print(e.by_payer.round(3).to_string())
    print("\n== Drift watch: payer × service line, largest gaps (positive = payer stricter than the model knows) ==")
    print(e.drift.round(3).to_string())
    print("\n== Paid ratio of approved claims: mean absolute error ==")
    for name, mae in e.paid_ratio_mae.items():
        print(f"  {name:<14} {mae:.4f}")
    print("\n== Days to cash ==")
    print(f"  Median model error       {e.days['p50_mae_days']:.1f} days")
    print(f"  Paid by p50 estimate     {e.days['p50_coverage']:.1%}  (target 50%)")
    print(f"  Paid by p90 estimate     {e.days['p90_coverage']:.1%}  (target 90%)")
    print("\n== What drives approval (share of total split gain) ==")
    for name, share in e.importance:
        print(f"  {name:<34} {share:6.1%}")


def cmd_backtest(args) -> None:
    models = FundingModels.load(MODELS)
    history = data_gen.load(HISTORY)
    test = history[history["lock_date"] >= pd.Timestamp(models.info.as_of)]
    offers = make_offers(test, models.predict(features.build(test, history)))
    model_result = backtest(test, offers)
    naive_result = backtest(test, naive_offers(test))

    print(f"== Backtest on {len(test)} claims locked after {models.info.as_of} (SAR) ==")
    print(f"  {'':<26} {'model + policy':>16} {'naive 80% / 3%':>16}")
    for key, label in [("funded", "Claims funded"), ("advanced", "Advanced"), ("fees", "Fees"),
                       ("credit_loss", "Credit loss"), ("capital_cost", "Cost of capital"),
                       ("net", "Net"), ("denied_after_funding", "Funded, then denied")]:
        print(f"  {label:<26} {_num(model_result[key]):>16} {_num(naive_result[key]):>16}")
    print(f"  {'Net margin on advance':<26} {model_result['net_margin_on_advance']:>16.2%} "
          f"{naive_result['net_margin_on_advance']:>16.2%}")
    print("\n== Model policy by tier ==")
    print(model_result["by_tier"].round(3).to_string())


def cmd_offers(args) -> None:
    models = FundingModels.load(MODELS)
    pending = data_gen.load(PENDING)
    # History features for pending claims come from everything known before their lock date.
    offers = make_offers(pending, models.predict(features.build(pending, data_gen.load(HISTORY))))
    offers.to_csv(OFFERS, index=False)
    tiers = offers["tier"].value_counts()
    print(f"Scored {len(offers)} pending claims: tier A {tiers.get('A', 0)}, B {tiers.get('B', 0)}, "
          f"C {tiers.get('C', 0)}  → {OFFERS}")
    print(f"Advance offered {offers['advance'].sum():,.0f} SAR on {offers['billed_amount'].sum():,.0f} SAR billed\n")
    cols = ["claim_id", "payer", "service_line", "billed_amount", "p_approve", "days_p90", "tier",
            "advance", "fee_rate", "reason"]
    print(offers.sort_values("p_approve")[cols].head(args.top).to_string(index=False))


def cmd_explain(args) -> None:
    models = FundingModels.load(MODELS)
    pending = data_gen.load(PENDING)
    claim = pending[pending["claim_id"] == args.claim_id]
    if claim.empty:
        raise SystemExit(f"Unknown claim id {args.claim_id}")
    X = features.build(claim, data_gen.load(HISTORY))
    offer = make_offers(claim, models.predict(X)).iloc[0]
    c = claim.iloc[0]
    print(f"Claim {c['claim_id']}  {c['payer']}  {c['hospital']}  CTAS {c['acuity']}  {c['service_line']}  "
          f"billed {c['billed_amount']:,.2f} SAR")
    print(f"  P(approve) {offer['p_approve']:.3f}   paid ratio {offer['paid_ratio']:.3f}   "
          f"cash in ~{offer['days_p50']:.0f} days (p90 {offer['days_p90']:.0f})")
    print(f"  Tier {offer['tier']}: advance {offer['advance']:,.2f} SAR at fee {offer['fee_rate']:.2%}"
          + (f"  ({offer['reason']})" if offer["reason"] else ""))
    print("  Approval drivers (log-odds; negative pushes towards denial):")
    for name, value in models.explain(X):
        print(f"    {value:+.3f}  {name}")


def cmd_demo(args) -> None:
    for step in (cmd_generate, cmd_train, cmd_evaluate, cmd_backtest, cmd_offers):
        step(args)
        print()
    offers = pd.read_csv(OFFERS)
    cmd_explain(argparse.Namespace(claim_id=offers.sort_values("p_approve")["claim_id"].iloc[0]))


def _num(value) -> str:
    return f"{value:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m funding", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, help_text):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(fn=fn)
        return p

    for p in (add("generate", cmd_generate, "synthetic ED claim history + pending claims"),
              add("demo", cmd_demo, "generate → train → evaluate → backtest → offers → explain")):
        p.add_argument("--per-day", type=float, default=40)
        p.add_argument("--seed", type=int, default=42)
    for p in (add("train", cmd_train, "train the four LightGBM models"), sub.choices["demo"]):
        p.add_argument("--as-of", default=AS_OF, help="training date; claims locked after it are the test set")
    add("evaluate", cmd_evaluate, "out-of-time evaluation vs logistic regression")
    add("backtest", cmd_backtest, "P&L of the funding policy on test claims vs a naive flat advance")
    p = add("offers", cmd_offers, "score pending claims → data/offers.csv")
    p.add_argument("--top", type=int, default=10)
    sub.choices["demo"].add_argument("--top", type=int, default=10)
    p = add("explain", cmd_explain, "offer + approval drivers for one pending claim")
    p.add_argument("--claim-id", required=True)

    args = parser.parse_args()
    args.fn(args)
