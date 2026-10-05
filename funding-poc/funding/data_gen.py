"""Synthetic ED claims at the lock point (coded, scrubbed, about to be submitted) with simulated
insurer outcomes: approval, line-item reductions, payment timing and post-payment clawbacks.

No real patient data. The simulation includes the kind of structure a tree model handles better
than a linear one: interactions (prior auth only matters when it is required; CT at low acuity
only matters for strict payers) and thresholds (high-cost review). It also has one policy change,
PAYER_C tightening CT approvals from July 2026, so out-of-time evaluation has drift to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Documentation quality: share of CDI alerts the hospital resolves before the lock point.
HOSPITALS = {"H1": 0.90, "H2": 0.75, "H3": 0.60}
HOSPITAL_SHARE = [0.45, 0.35, 0.20]
# Hospitals with slow paperwork (attachments, resubmissions) get paid later.
HOSPITAL_PAY_DELAY = {"H1": 1.00, "H2": 1.05, "H3": 1.20}

# base: approval log-odds for a clean claim. ct_strict: penalty for CT at low acuity (CTAS 4-5).
# high_cost: billed amount (SAR) above which the payer reviews line by line. reduction: mean share
# cut from approved claims. days: typical days from submission to payment. clawback: share of paid
# claims recouped later after audit.
PAYERS = {
    "PAYER_A": dict(share=0.30, base=3.0, ct_strict=0.5, high_cost=4000, reduction=0.04, days=35, clawback=0.010),
    "PAYER_B": dict(share=0.25, base=2.6, ct_strict=1.6, high_cost=3000, reduction=0.08, days=55, clawback=0.020),
    "PAYER_C": dict(share=0.20, base=2.8, ct_strict=0.6, high_cost=5000, reduction=0.05, days=45, clawback=0.015),
    "PAYER_D": dict(share=0.15, base=2.2, ct_strict=1.0, high_cost=2500, reduction=0.12, days=80, clawback=0.030),
    "PAYER_E": dict(share=0.10, base=3.2, ct_strict=0.2, high_cost=6000, reduction=0.03, days=30, clawback=0.005),
}
POLICY_CHANGE = ("PAYER_C", pd.Timestamp("2026-07-01"), 1.5)  # payer, effective date, extra CT penalty

SERVICE_LINES = ["LABS_ONLY", "XRAY", "CT", "MINOR_PROCEDURE", "OBSERVATION"]
# Rows: CTAS 1 (resuscitation) .. 5 (non-urgent). Columns follow SERVICE_LINES.
SERVICE_MIX = np.array([
    [0.05, 0.15, 0.35, 0.15, 0.30],
    [0.10, 0.20, 0.35, 0.15, 0.20],
    [0.25, 0.30, 0.25, 0.12, 0.08],
    [0.40, 0.35, 0.12, 0.10, 0.03],
    [0.60, 0.25, 0.07, 0.07, 0.01],
])
MEDIAN_BILLED = np.array([450, 700, 2200, 1500, 3800])  # SAR
PRIOR_AUTH_SERVICES = ["CT", "MINOR_PROCEDURE", "OBSERVATION"]  # only for CTAS 4-5; emergencies are exempt

HISTORY_START, HISTORY_END = pd.Timestamp("2025-04-01"), pd.Timestamp("2026-09-30")
PENDING_START, PENDING_END = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-07")

DATE_COLUMNS = ["lock_date", "submitted_date", "adjudicated_date", "paid_date", "clawback_date"]
OUTCOME_COLUMNS = ["submitted_date", "adjudicated_date", "approved", "approved_amount", "paid_date",
                   "days_to_cash", "clawback_amount", "clawback_date"]


def _days(values) -> pd.TimedeltaIndex:
    return pd.to_timedelta(np.asarray(values), unit="D")


def generate(start: pd.Timestamp, end: pd.Timestamp, per_day: float = 40, seed: int = 42,
             id_prefix: str = "ED") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    lock = pd.DatetimeIndex(np.repeat(days.values, rng.poisson(per_day, len(days))))
    n = len(lock)

    hospital = rng.choice(list(HOSPITALS), n, p=HOSPITAL_SHARE)
    payer_idx = rng.choice(len(PAYERS), n, p=[p["share"] for p in PAYERS.values()])
    payer = np.array(list(PAYERS))[payer_idx]
    P = {k: np.array([p[k] for p in PAYERS.values()])[payer_idx]
         for k in ("base", "ct_strict", "high_cost", "reduction", "days", "clawback")}

    acuity = rng.choice(np.arange(1, 6), n, p=[0.03, 0.15, 0.40, 0.30, 0.12])
    service_idx = (rng.random(n)[:, None] > SERVICE_MIX.cumsum(axis=1)[acuity - 1]).sum(axis=1)
    service = np.array(SERVICE_LINES)[service_idx]
    is_ct, is_obs = service == "CT", service == "OBSERVATION"

    billed = np.round(MEDIAN_BILLED[service_idx] * rng.lognormal(0, 0.45, n) * (1 + 0.08 * (3 - acuity)), 2)
    high_cost = billed > P["high_cost"]
    pa_required = (acuity >= 4) & np.isin(service, PRIOR_AUTH_SERVICES)
    has_pa = pa_required & (rng.random(n) < 0.7)
    eligible = rng.random(n) < 0.93

    # CDI alerts raised during the visit; the unresolved ones are still open at the lock point.
    quality = pd.Series(hospital).map(HOSPITALS).to_numpy()
    raised = rng.poisson(0.5 + 1.5 * (1 - quality) + 0.4 * (is_ct | is_obs))
    open_alerts = raised - rng.binomial(raised, quality)
    open_high = rng.binomial(open_alerts, 0.4)
    completeness = np.round(np.clip(rng.beta(9, 2, n) - 0.06 * open_alerts, 0, 1), 3)

    drift = (payer == POLICY_CHANGE[0]) & is_ct & (lock >= POLICY_CHANGE[1])
    z = (P["base"] - 3.0 * (pa_required & ~has_pa) - 2.0 * ~eligible
         - 1.1 * open_high - 0.35 * (open_alerts - open_high) + 4.0 * (completeness - 0.8)
         - P["ct_strict"] * (is_ct & (acuity >= 4)) - 0.8 * high_cost - POLICY_CHANGE[2] * drift)
    approved = rng.random(n) < 1 / (1 + np.exp(-z))

    submitted = lock + _days(rng.integers(0, 3, n))
    adjudicated = submitted + _days(7 + rng.poisson(P["days"] * 0.3))

    cut_mean = np.clip(P["reduction"] + 0.10 * high_cost + 0.03 * (open_alerts - open_high)
                       + 0.05 * (is_obs & (acuity >= 4)), 0.01, 0.6)
    cut = rng.beta(2, 2 * (1 - cut_mean) / cut_mean)
    approved_amount = np.where(approved, np.round(billed * (1 - cut), 2), 0.0)

    delay = pd.Series(hospital).map(HOSPITAL_PAY_DELAY).to_numpy()
    pay_days = np.round(P["days"] * delay * rng.lognormal(0, 0.35, n) + 20 * high_cost).astype(int)
    paid = pd.Series(submitted + _days(pay_days)).where(lambda s: s > adjudicated + _days(3), adjudicated + _days(3))
    paid_date = paid.where(approved)

    clawed = approved & (rng.random(n) < P["clawback"] * (1 + 1.5 * open_high))
    clawback_amount = np.where(clawed, np.round(approved_amount * rng.uniform(0.3, 1.0, n), 2), 0.0)
    clawback_date = (paid + _days(rng.integers(60, 240, n))).where(clawed)

    df = pd.DataFrame({
        "hospital": hospital, "payer": payer, "lock_date": lock,
        "acuity": acuity, "service_line": service, "patient_age": rng.integers(0, 91, n),
        "n_dx": 1 + rng.poisson(np.clip(0.4 + 0.3 * (3 - acuity), 0.1, None)),
        "billed_amount": billed, "prior_auth_required": pa_required, "has_prior_auth": has_pa,
        "eligibility_verified": eligible, "cdi_alerts_raised": raised,
        "open_high_alerts": open_high, "open_low_alerts": open_alerts - open_high,
        "doc_completeness": completeness,
        "submitted_date": submitted, "adjudicated_date": adjudicated, "approved": approved,
        "approved_amount": approved_amount, "paid_date": paid_date.to_numpy(),
        "days_to_cash": (paid_date - pd.Series(lock)).dt.days.to_numpy(),
        "clawback_amount": clawback_amount, "clawback_date": clawback_date.to_numpy(),
    })
    df = df.sort_values("lock_date", kind="stable").reset_index(drop=True)
    df.insert(0, "claim_id", [f"{id_prefix}-{i:06d}" for i in range(1, len(df) + 1)])
    return df


def generate_history(per_day: float = 40, seed: int = 42) -> pd.DataFrame:
    return generate(HISTORY_START, HISTORY_END, per_day, seed)


def generate_pending(per_day: float = 40, seed: int = 43) -> pd.DataFrame:
    """New claims at the lock point: outcomes are not known yet."""
    return generate(PENDING_START, PENDING_END, per_day, seed, id_prefix="NEW").drop(columns=OUTCOME_COLUMNS)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in DATE_COLUMNS:
        if col in df:
            df[col] = pd.to_datetime(df[col])
    return df
