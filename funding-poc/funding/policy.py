"""Turns model predictions into funding offers, and backtests the policy on claims whose outcome
is now known.

The models predict; the policy decides. Keeping them apart means the credit team can change
thresholds, advance rates or pricing without retraining, and every offer records which policy made it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    version: str = "2026-10-v1"
    tier_a_min_p: float = 0.95      # fund before submission at advance_a
    tier_b_min_p: float = 0.85      # fund before submission at advance_b, the rest after approval
    advance_a: float = 0.80         # of expected approved amount (billed × predicted paid ratio)
    advance_b: float = 0.50
    max_open_high_alerts: int = 0   # an unresolved high-severity CDI alert drops the claim to tier C
    cost_of_capital: float = 0.12   # annual
    ops_margin: float = 0.01        # of the advance
    loss_given_denial: float = 0.30  # share of a shortfall not recovered by recourse / substitution


def make_offers(claims: pd.DataFrame, pred: pd.DataFrame, policy: Policy = Policy()) -> pd.DataFrame:
    p = pred["p_approve"]
    expected_approved = claims["billed_amount"] * pred["paid_ratio"]
    expected_collectible = expected_approved * p * (1 - pred["clawback_rate"])

    clean = claims["open_high_alerts"] <= policy.max_open_high_alerts
    tier = np.select([clean & (p >= policy.tier_a_min_p), clean & (p >= policy.tier_b_min_p)], ["A", "B"], "C")
    rate = pd.Series(tier, index=claims.index).map({"A": policy.advance_a, "B": policy.advance_b, "C": 0.0})
    advance = (expected_approved * rate).round(2)

    # Fee = capital tied up until the slow (p90) payment date + expected loss + operating margin.
    fee_rate = (policy.cost_of_capital * pred["days_p90"] / 365
                + (1 - p) * policy.loss_given_denial + pred["clawback_rate"] + policy.ops_margin)
    fee = (advance * fee_rate).round(2)

    reason = np.select(
        [tier != "C", ~clean],
        ["", "unresolved high-severity CDI alert"],
        "approval probability below " + f"{policy.tier_b_min_p:.0%}" + ": fund after approval",
    )
    return pd.DataFrame({
        "claim_id": claims["claim_id"], "payer": claims["payer"], "service_line": claims["service_line"],
        "billed_amount": claims["billed_amount"],
        "p_approve": p.round(4), "paid_ratio": pred["paid_ratio"].round(4),
        "days_p50": pred["days_p50"].round(0), "days_p90": pred["days_p90"].round(0),
        "expected_collectible": expected_collectible.round(2),
        "tier": tier, "advance": advance, "fee": fee.where(advance > 0, 0.0),
        "fee_rate": fee_rate.where(advance > 0, 0.0).round(4),
        "reason": reason, "policy_version": policy.version,
    }, index=claims.index)


def backtest(claims: pd.DataFrame, offers: pd.DataFrame, policy: Policy = Policy()) -> dict:
    """Realised P&L of funded claims, using the outcomes we now know.

    The hospital owes advance + fee out of the insurer's payment. A shortfall (denial, cut, clawback)
    is recovered from the hospital through recourse, except `loss_given_denial` of it. Capital is out
    until the insurer pays, or ~30 days after a denial for the recourse to settle.
    """
    funded = offers["advance"] > 0
    c, o = claims[funded], offers[funded]
    collected = c["approved_amount"] - c["clawback_amount"]
    shortfall = (o["advance"] + o["fee"] - collected).clip(lower=0)
    back = c["paid_date"].fillna(c["adjudicated_date"] + pd.Timedelta(days=30))
    days_out = (back - c["lock_date"]).dt.days
    capital_cost = o["advance"] * policy.cost_of_capital * days_out / 365
    loss = shortfall * policy.loss_given_denial

    result = {
        "claims": int(len(offers)),
        "funded": int(funded.sum()),
        "advanced": float(o["advance"].sum()),
        "fees": float(o["fee"].sum()),
        "credit_loss": float(loss.sum()),
        "capital_cost": float(capital_cost.sum()),
        "denied_after_funding": int((~c["approved"].astype(bool)).sum()),
    }
    result["net"] = result["fees"] - result["credit_loss"] - result["capital_cost"]
    result["net_margin_on_advance"] = result["net"] / result["advanced"] if result["advanced"] else 0.0
    result["by_tier"] = (pd.DataFrame({"tier": o["tier"], "advance": o["advance"], "fee": o["fee"],
                                       "loss": loss, "approved": c["approved"].astype(bool)})
                         .groupby("tier").agg(n=("advance", "size"), advanced=("advance", "sum"),
                                              fees=("fee", "sum"), loss=("loss", "sum"),
                                              approval_rate=("approved", "mean")))
    return result


def naive_offers(claims: pd.DataFrame, advance_rate: float = 0.80, flat_fee: float = 0.03) -> pd.DataFrame:
    """The no-model alternative: same advance on every claim's billed amount, one flat fee."""
    advance = (claims["billed_amount"] * advance_rate).round(2)
    return pd.DataFrame({"tier": "ALL", "advance": advance, "fee": (advance * flat_fee).round(2)},
                        index=claims.index)

