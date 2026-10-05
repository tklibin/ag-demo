"""Feature frame for LightGBM: claim facts at the lock point + point-in-time payer/hospital history.

Two rules keep the features honest:
  1. Only facts known at the lock point. Outcome columns never enter the frame.
  2. History features for a claim locked on day t use only events known before t: approvals by
     their adjudication date, payments by their paid date. Using the final outcome of claims that
     were still in flight on day t is the classic leak that makes a backtest look great and
     production lose money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_gen import HOSPITALS, PAYERS, SERVICE_LINES

# Fixed category lists so codes are identical in training and scoring. LightGBM splits on
# pandas categoricals natively: no one-hot encoding needed.
CATEGORIES = {"hospital": list(HOSPITALS), "payer": list(PAYERS), "service_line": SERVICE_LINES}
BOOLEAN = ["prior_auth_required", "has_prior_auth", "eligibility_verified"]
CLAIM_FEATURES = ["hospital", "payer", "service_line", "acuity", "patient_age", "n_dx", "billed_amount",
                  *BOOLEAN, "cdi_alerts_raised", "open_high_alerts", "open_low_alerts", "doc_completeness"]
# NaN until there are enough events in the window; LightGBM learns where missing values should go.
HISTORY_FEATURES = ["payer_approval_rate_90d", "payer_service_approval_rate_90d",
                    "hospital_approval_rate_90d", "payer_mean_days_to_cash_180d"]
FEATURES = CLAIM_FEATURES + HISTORY_FEATURES

# Monotone constraints for the approval model (+1: higher value never lowers approval, -1: never
# raises it). They stop the trees learning noise such as "two open alerts are safer than one",
# which a credit committee would rightly reject.
APPROVAL_MONOTONE = {
    "eligibility_verified": 1, "doc_completeness": 1,
    "open_high_alerts": -1, "open_low_alerts": -1,
    "payer_approval_rate_90d": 1, "payer_service_approval_rate_90d": 1, "hospital_approval_rate_90d": 1,
}
MIN_EVENTS = 30


def monotone_vector(constraints: dict[str, int]) -> list[int]:
    return [constraints.get(f, 0) for f in FEATURES]


def _day_numbers(dates: pd.Series) -> np.ndarray:
    return dates.to_numpy().astype("datetime64[D]").astype(np.int64)


def _window_mean(event_key: np.ndarray, event_day: np.ndarray, event_value: np.ndarray,
                 query_key: np.ndarray, query_day: np.ndarray, window: int) -> np.ndarray:
    """Mean of event values per key over [query_day - window, query_day), strictly before the query."""
    out = np.full(len(query_day), np.nan)
    for key in np.unique(query_key):
        e = event_key == key
        order = np.argsort(event_day[e], kind="stable")
        day, value = event_day[e][order], event_value[e][order]
        csum = np.concatenate([[0.0], np.cumsum(value)])
        q = query_key == key
        hi = np.searchsorted(day, query_day[q], side="left")
        lo = np.searchsorted(day, query_day[q] - window, side="left")
        n = hi - lo
        out[q] = np.where(n >= MIN_EVENTS, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
    return out


def build(claims: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Features for `claims`, with history features computed from `history` as of each lock date."""
    X = claims[CLAIM_FEATURES].copy()
    for col, cats in CATEGORIES.items():
        X[col] = pd.Categorical(X[col].astype(str), categories=cats)
    for col in BOOLEAN:
        X[col] = X[col].astype(int)

    q_day = _day_numbers(claims["lock_date"])
    q_payer = claims["payer"].to_numpy(str)
    q_payer_service = q_payer + "|" + claims["service_line"].to_numpy(str)

    adjudicated = history[history["adjudicated_date"].notna()]
    a_day = _day_numbers(adjudicated["adjudicated_date"])
    a_value = adjudicated["approved"].to_numpy(float)
    a_payer = adjudicated["payer"].to_numpy(str)
    X["payer_approval_rate_90d"] = _window_mean(a_payer, a_day, a_value, q_payer, q_day, 90)
    X["payer_service_approval_rate_90d"] = _window_mean(
        a_payer + "|" + adjudicated["service_line"].to_numpy(str), a_day, a_value, q_payer_service, q_day, 90)
    X["hospital_approval_rate_90d"] = _window_mean(
        adjudicated["hospital"].to_numpy(str), a_day, a_value, claims["hospital"].to_numpy(str), q_day, 90)

    paid = history[history["paid_date"].notna()]
    X["payer_mean_days_to_cash_180d"] = _window_mean(
        paid["payer"].to_numpy(str), _day_numbers(paid["paid_date"]), paid["days_to_cash"].to_numpy(float),
        q_payer, q_day, 180)
    return X[FEATURES]
