"""Out-of-time evaluation: train on claims before `as_of`, test on claims locked after it.

Compares LightGBM with a logistic-regression baseline on the same features, and reports the
numbers that matter for funding: calibration (is 0.95 really 95%?), calibration per payer (drift),
paid-ratio error, and whether the 90th-percentile days estimate covers ~90% of claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import features
from .models import FundingModels, approval_mask, time_split

CALIBRATION_BINS = [0, 0.7, 0.85, 0.9, 0.95, 0.98, 1.0]  # edges line up with the funding tiers


@dataclass
class Evaluation:
    n_test: int
    approval: pd.DataFrame          # model × (ROC-AUC, Brier, log loss)
    calibration: pd.DataFrame       # predicted-probability band × (n, predicted, actual)
    by_payer: pd.DataFrame          # payer × (n, predicted, actual, gap)
    drift: pd.DataFrame             # payer × service line segments with the largest gaps
    paid_ratio_mae: dict[str, float]
    days: dict[str, float]
    importance: list[tuple[str, float]]


def logistic_baseline(history: pd.DataFrame, X: pd.DataFrame, as_of: pd.Timestamp):
    """Same features, same training rows; one-hot categoricals and imputed NaNs, as a linear model needs."""
    tr, _ = time_split(history["lock_date"], approval_mask(history, as_of))
    numeric = [f for f in features.FEATURES if f not in features.CATEGORIES]
    model = make_pipeline(
        ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(features.CATEGORIES)),
            ("num", make_pipeline(SimpleImputer(add_indicator=True), StandardScaler()), numeric),
        ]),
        LogisticRegression(max_iter=2000),
    )
    return model.fit(X[tr], history.loc[tr, "approved"].astype(int))


def evaluate(models: FundingModels, history: pd.DataFrame, as_of: pd.Timestamp) -> Evaluation:
    X = features.build(history, history)
    test = history["lock_date"] >= as_of
    y = history.loc[test, "approved"].astype(int)
    pred = models.predict(X[test])
    baseline = logistic_baseline(history, X, as_of).predict_proba(X[test])[:, 1]

    approval = pd.DataFrame([
        _scores("Logistic regression (baseline)", y, baseline),
        _scores("LightGBM raw", y, pred["p_approve_raw"]),
        _scores("LightGBM calibrated", y, pred["p_approve"]),
    ]).set_index("model")

    band = pd.cut(pred["p_approve"], CALIBRATION_BINS, include_lowest=True)
    calibration = (pd.DataFrame({"band": band, "predicted": pred["p_approve"], "actual": y})
                   .groupby("band", observed=True)
                   .agg(n=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean")))

    scored = pd.DataFrame({"payer": history.loc[test, "payer"], "service_line": history.loc[test, "service_line"],
                           "predicted": pred["p_approve"], "actual": y})
    by_payer = _gaps(scored, ["payer"])
    # A payer rule change usually hits one service line, and averages over the payer hide it.
    drift = _gaps(scored, ["payer", "service_line"])
    drift = drift[drift["n"] >= 30].sort_values("gap", ascending=False).head(5)

    # Paid ratio on approved test claims, against "use the payer's historical average".
    appr = test & history["approved"].astype(bool)
    ratio = (history["approved_amount"] / history["billed_amount"]).clip(0, 1)
    train_appr = (history["lock_date"] < as_of) & history["approved"].astype(bool)
    payer_mean = ratio[train_appr].groupby(history.loc[train_appr, "payer"]).mean()
    paid_ratio_mae = {
        "LightGBM": float(mean_absolute_error(ratio[appr], pred.loc[appr[appr].index, "paid_ratio"])),
        "Payer average": float(mean_absolute_error(ratio[appr], history.loc[appr, "payer"].map(payer_mean))),
    }

    paid = test & history["paid_date"].notna()
    actual_days = history.loc[paid, "days_to_cash"]
    days_pred = pred.loc[actual_days.index]
    days = {
        "p50_mae_days": float(mean_absolute_error(actual_days, days_pred["days_p50"])),
        "p50_coverage": float((actual_days <= days_pred["days_p50"]).mean()),
        "p90_coverage": float((actual_days <= days_pred["days_p90"]).mean()),
    }
    return Evaluation(int(test.sum()), approval, calibration, by_payer, drift, paid_ratio_mae, days,
                      models.importance())


def _gaps(scored: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = scored.groupby(keys).agg(n=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean"))
    g["gap"] = g["predicted"] - g["actual"]
    return g


def _scores(name: str, y: pd.Series, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"model": name, "roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)}
