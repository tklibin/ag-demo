"""Command line: python -m cdi {generate,train,analyze,worklist,insights,demo}"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from . import data_gen
from .codes import DENIAL_CODES, describe
from .engine import CdiEngine
from .models import Category, CdiReport, Claim
from .risk_model import DenialRiskModel
from .rules import ALL_RULES, run_rules

DATA = Path("data")
HISTORY = DATA / "claims_history.jsonl"
PENDING = DATA / "claims_pending.jsonl"
MODEL = DATA / "model.json"
WORKLIST = DATA / "worklist.csv"
REPORTS = DATA / "reports.json"


# --- commands -----------------------------------------------------------------------------------

def cmd_generate(args) -> None:
    history = data_gen.generate(args.n, seed=args.seed)
    # Pending claims are new, not-yet-submitted claims: same generator, outcome stripped.
    pending = [replace(c, claim_id=c.claim_id.replace("CLM", "NEW"), outcome=None, denial_code=None)
               for c in data_gen.generate(args.pending, seed=args.seed + 1)]
    data_gen.save(history, HISTORY)
    data_gen.save(pending, PENDING)
    denied = sum(c.outcome == "DENIED" for c in history)
    print(f"Wrote {len(history)} adjudicated claims to {HISTORY} ({denied / len(history):.0%} denied)")
    print(f"Wrote {len(pending)} pending claims to {PENDING}")


def cmd_train(args) -> None:
    model = DenialRiskModel.train(data_gen.load(HISTORY))
    model.save(MODEL)
    m = model.metrics
    print(f"Trained on {m.n_train} claims, tested on {m.n_test} (denial rate {m.denial_rate:.1%})")
    print(f"  ROC-AUC           {m.roc_auc:.3f}")
    print(f"  Average precision {m.average_precision:.3f}")
    print(f"Saved model to {MODEL}")


def cmd_analyze(args) -> None:
    claims = {c.claim_id: c for c in _load_all()}
    claim = claims.get(args.claim_id)
    if claim is None:
        raise SystemExit(f"Unknown claim id {args.claim_id}")
    report = _engine().analyze(claim)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(claim, report)


def cmd_worklist(args) -> None:
    engine = _engine()
    claims = data_gen.load(PENDING)
    reports = [(c, engine.analyze(c)) for c in claims]
    reports.sort(key=lambda cr: -(cr[1].denial_probability or 0))

    with WORKLIST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "payer", "cpt_codes", "dx_codes", "denial_probability", "risk_band",
                    "denial_risks", "coding_opportunities", "physician_queries", "top_issue", "suggested_codes"])
        for claim, r in reports:
            denial = [x for x in r.findings if x.category != Category.CODING_OPPORTUNITY]
            coding = [x for x in r.findings if x.category == Category.CODING_OPPORTUNITY]
            w.writerow([
                claim.claim_id, claim.payer, " ".join(claim.cpt_codes), " ".join(claim.dx_codes),
                r.denial_probability, r.risk_band, len(denial), len(coding),
                sum(x.query is not None for x in r.findings),
                r.findings[0].message if r.findings else "",
                " ".join(sorted({code for x in r.findings for code in x.suggested_codes})),
            ])
    REPORTS.write_text(json.dumps([r.to_dict() for _, r in reports], indent=2))

    bands = Counter(r.risk_band for _, r in reports)
    print(f"Reviewed {len(reports)} pending claims: "
          f"{bands['HIGH']} HIGH, {bands['MEDIUM']} MEDIUM, {bands['LOW']} LOW risk")
    print(f"Queries to send: {sum(x.query is not None for _, r in reports for x in r.findings)}")
    print(f"Wrote {WORKLIST} and {REPORTS}\n")
    print(f"{'claim':<10} {'payer':<8} {'cpt':<12} {'risk':>5}  band    top issue")
    for claim, r in reports[: args.top]:
        top = r.findings[0].message if r.findings else "-"
        print(f"{claim.claim_id:<10} {claim.payer:<8} {' '.join(claim.cpt_codes):<12} "
              f"{r.denial_probability or 0:>5.2f}  {r.risk_band:<6}  {top[:70]}")


def cmd_insights(args) -> None:
    claims = data_gen.load(HISTORY)
    flagged = {c.claim_id: {f.rule_id for f in run_rules(c)} for c in claims}
    n = len(claims)
    denied = [c for c in claims if c.outcome == "DENIED"]

    print(f"== Denials: {len(denied)}/{n} ({len(denied) / n:.1%}) ==")
    for code, count in Counter(c.denial_code for c in denied).most_common():
        print(f"  {code:<7} {count:>4}  {DENIAL_CODES[code]}")

    print("\n== Denial rate by payer ==")
    for payer in sorted({c.payer for c in claims}):
        group = [c for c in claims if c.payer == payer]
        print(f"  {payer:<8} {sum(c.outcome == 'DENIED' for c in group) / len(group):>6.1%}  (n={len(group)})")

    print("\n== Rule impact: denial rate when a rule fires vs when it doesn't ==")
    print(f"  {'rule':<30} {'fired':>5} {'denied|fired':>13} {'denied|not':>11} {'precision':>10} {'recall':>7}")
    rule_ids = sorted({r for ids in flagged.values() for r in ids})
    for rule_id in rule_ids:
        hit = [c for c in claims if rule_id in flagged[c.claim_id]]
        miss = [c for c in claims if rule_id not in flagged[c.claim_id]]
        truth = [c for c in claims if rule_id in c.injected_defects]
        tp = sum(rule_id in c.injected_defects for c in hit)
        precision = f"{tp / len(hit):.0%}" if hit else "-"
        recall = f"{tp / len(truth):.0%}" if truth else "-"
        print(f"  {rule_id:<30} {len(hit):>5} {_rate(hit):>13} {_rate(miss):>11} {precision:>10} {recall:>7}")


def cmd_demo(args) -> None:
    cmd_generate(argparse.Namespace(n=args.n, pending=args.pending, seed=args.seed))
    print()
    cmd_train(args)
    print()
    cmd_insights(args)
    print()
    cmd_worklist(argparse.Namespace(top=10))
    pending = data_gen.load(PENDING)
    engine = _engine()
    example = max(pending, key=lambda c: len(engine.analyze(c).findings))
    print()
    print_report(example, engine.analyze(example))


# --- output -------------------------------------------------------------------------------------

def print_report(claim: Claim, report: CdiReport) -> None:
    prob = "n/a" if report.denial_probability is None else f"{report.denial_probability:.2f}"
    print("=" * 88)
    print(f"Claim {claim.claim_id}  {claim.payer}  {claim.encounter_type}  DOS {claim.date_of_service}")
    print(f"  CPT: " + ", ".join(f"{c} ({describe(c)})" for c in claim.cpt_codes))
    print(f"  DX:  " + ", ".join(f"{c} ({describe(c)})" for c in claim.dx_codes))
    print(f"  Prior auth: {claim.prior_auth_number or '-'}")
    print(f"  Note: {claim.note}")
    print(f"\n  Denial risk: {prob} ({report.risk_band})")
    if report.risk_drivers:
        print("  Risk drivers: " + ", ".join(f"{d.feature} (+{d.contribution:.2f})" for d in report.risk_drivers))
    if not report.findings:
        print("\n  No documentation issues found.")
    for f in report.findings:
        tag = f" → {f.likely_denial_code}" if f.likely_denial_code else ""
        print(f"\n  [{f.severity}] {f.rule_id} ({f.category}){tag}")
        print(f"    {f.message}")
        print(f"    Action: {f.recommendation}")
        for e in f.evidence[:2]:
            print(f"    Evidence: \"{e}\"")
        if f.suggested_codes:
            print("    Suggested (if confirmed): " + ", ".join(f"{c} {describe(c)}" for c in f.suggested_codes))
        if f.query:
            print(f"    Physician query – {f.query.topic}: {f.query.question}")
            print("      Options: " + " | ".join(f.query.options))
    print("=" * 88)


# --- helpers ------------------------------------------------------------------------------------

def _engine() -> CdiEngine:
    model = DenialRiskModel.load(MODEL) if MODEL.exists() else None
    if model is None:
        print("(no trained model found – run `python -m cdi train`; using rules only)")
    return CdiEngine(model, ALL_RULES)


def _load_all() -> list[Claim]:
    return [*data_gen.load(HISTORY), *(data_gen.load(PENDING) if PENDING.exists() else [])]


def _rate(claims: list[Claim]) -> str:
    if not claims:
        return "-"
    return f"{sum(c.outcome == 'DENIED' for c in claims) / len(claims):.0%}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cdi", description="Clinical Documentation Improvement POC")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create synthetic adjudicated + pending claims")
    g.add_argument("--n", type=int, default=600)
    g.add_argument("--pending", type=int, default=50)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    sub.add_parser("train", help="train the denial-risk model").set_defaults(func=cmd_train)

    a = sub.add_parser("analyze", help="CDI report for one claim")
    a.add_argument("--claim-id", required=True)
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    w = sub.add_parser("worklist", help="score pending claims into a CDI worklist")
    w.add_argument("--top", type=int, default=10)
    w.set_defaults(func=cmd_worklist)

    sub.add_parser("insights", help="denial drivers across historical claims").set_defaults(func=cmd_insights)

    d = sub.add_parser("demo", help="run everything end to end")
    d.add_argument("--n", type=int, default=600)
    d.add_argument("--pending", type=int, default=50)
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)
