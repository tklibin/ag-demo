"""The four LightGBM models behind a funding offer, each with a different objective:

  approval     binary         P(insurer approves the claim), then Platt calibration
  paid_ratio   cross_entropy  share of billed amount paid if approved (a label in [0, 1])
  days_p50/90  quantile       days from lock to cash: typical and conservative

Clawbacks are rare and arrive up to a year later, so the POC uses a per-payer historical rate
rather than a model.

Labels mature at different speeds, so each model trains only on claims old enough for their
outcome to be known on the training date (`as_of`). Every split is by time, never random:
the model is always validated on claims that came after the ones it learned from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import features

APPROVAL_MATURITY = pd.Timedelta(days=75)    # adjudication arrives within ~2 months of lock
PAYMENT_MATURITY = pd.Timedelta(days=180)    # 99% of approved claims are paid within ~150 days
CLAWBACK_MATURITY = pd.Timedelta(days=365)   # recoupments land 2-8 months after payment
VALIDATION = pd.Timedelta(days=60)           # the latest mature window: early stopping + calibration

PARAMS = dict(
    n_estimators=3000,         # an upper bound; early stopping picks the real number
    learning_rate=0.03,
    num_leaves=15,             # small trees: tabular data with a few thousand rows per segment
    min_child_samples=40,      # no leaf learns from a handful of claims
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=7, verbose=-1,
    # A few thousand rows gain nothing from threads, and OpenMP thread pools can stall badly when
    # two OpenMP runtimes are loaded (here: LightGBM + scikit-learn). One thread is fast and stable.
    n_jobs=1,
)
EARLY_STOPPING = 100


def time_split(lock: pd.Series, mask: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Train on mature claims except the most recent VALIDATION window, which validates."""
    cutoff = lock[mask].max() - VALIDATION
    return mask & (lock <= cutoff), mask & (lock > cutoff)


def approval_mask(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    return (history["lock_date"] < as_of - APPROVAL_MATURITY) & (history["adjudicated_date"] < as_of)


def payment_mask(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    # Claims paid after as_of are dropped. With a 180-day maturity that is a tiny, slow-payer biased
    # sliver; a survival model would use them as censored observations instead.
    return (history["lock_date"] < as_of - PAYMENT_MATURITY) & (history["paid_date"] < as_of)


def _fit(model: lgb.LGBMModel, X: pd.DataFrame, y: pd.Series, tr: pd.Series, va: pd.Series) -> lgb.Booster:
    model.fit(X[tr], y[tr], eval_X=(X[va],), eval_y=(y[va],),
              callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    return model.booster_


@dataclass
class TrainingInfo:
    as_of: str
    n_approval_train: int
    n_approval_valid: int
    n_payment_train: int
    approval_valid_auc: float
    best_iterations: dict[str, int]


class FundingModels:
    """Boosters are stored as LightGBM's own text format plus JSON: no pickle to load."""

    def __init__(self, approval: lgb.Booster, calibration: tuple[float, float], paid_ratio: lgb.Booster,
                 days_p50: lgb.Booster, days_p90: lgb.Booster, clawback_rate: dict[str, float],
                 info: TrainingInfo):
        self.approval = approval
        self.calibration = calibration  # (slope, intercept) on the raw log-odds
        self.paid_ratio = paid_ratio
        self.days_p50, self.days_p90 = days_p50, days_p90
        self.clawback_rate = clawback_rate
        self.info = info

    @classmethod
    def train(cls, history: pd.DataFrame, as_of: pd.Timestamp) -> FundingModels:
        X = features.build(history, history)
        lock = history["lock_date"]

        # 1. Approval. Monotone constraints keep the model's direction sensible on the features
        #    where we know it (more open alerts must never raise approval).
        tr, va = time_split(lock, approval_mask(history, as_of))
        y = history["approved"].astype(float)
        approval = _fit(lgb.LGBMClassifier(**PARAMS, monotone_constraints=features.monotone_vector(
            features.APPROVAL_MONOTONE)), X, y, tr, va)

        # Pricing uses the probability itself, so it has to be calibrated: of the claims scored
        # 0.95, about 95% must actually be approved. Platt scaling fits slope + intercept on the raw
        # log-odds of the validation window. Isotonic regression is the other common choice, but its
        # step function overfits a window of a few thousand claims: it scored claims 0.99 that were
        # approved 95% of the time out of time. Platt is strictly monotone, so the ranking (AUC) is
        # unchanged. (The same window chose the early-stopping round; with more data, hold out a
        # separate calibration window.)
        logit_va = approval.predict(X[va], raw_score=True).reshape(-1, 1)
        platt = LogisticRegression(C=1e6).fit(logit_va, y[va])
        calibration = (float(platt.coef_[0][0]), float(platt.intercept_[0]))

        # 2. Paid ratio of approved claims. cross_entropy accepts any label in [0, 1] and its
        #    predictions stay inside [0, 1].
        appr = history["approved"].astype(bool)
        ratio = (history["approved_amount"] / history["billed_amount"]).clip(0, 1)
        paid_ratio = _fit(lgb.LGBMRegressor(**PARAMS, objective="cross_entropy"),
                          X, ratio, tr & appr, va & appr)

        # 3. Days to cash, as two quantiles. The 90th percentile, not the mean, prices capital:
        #    it covers the slow tail that actually ties up the money.
        ptr, pva = time_split(lock, payment_mask(history, as_of))
        days = history["days_to_cash"]
        days_p50 = _fit(lgb.LGBMRegressor(**PARAMS, objective="quantile", alpha=0.5), X, days, ptr, pva)
        days_p90 = _fit(lgb.LGBMRegressor(**PARAMS, objective="quantile", alpha=0.9), X, days, ptr, pva)

        # 4. Clawback: share of approved money recouped, per payer, on a fully mature cohort.
        cohort = history[(lock < as_of - CLAWBACK_MATURITY) & appr]
        recouped = cohort["clawback_amount"].where(cohort["clawback_date"] < as_of, 0.0)
        clawback_rate = (recouped.groupby(cohort["payer"]).sum()
                         / cohort["approved_amount"].groupby(cohort["payer"]).sum()).round(4).to_dict()

        info = TrainingInfo(
            as_of=str(as_of.date()),
            n_approval_train=int(tr.sum()), n_approval_valid=int(va.sum()), n_payment_train=int(ptr.sum()),
            approval_valid_auc=round(float(roc_auc_score(y[va], logit_va.ravel())), 4),
            best_iterations={name: b.best_iteration for name, b in
                             [("approval", approval), ("paid_ratio", paid_ratio),
                              ("days_p50", days_p50), ("days_p90", days_p90)]},
        )
        return cls(approval, calibration, paid_ratio, days_p50, days_p90, clawback_rate, info)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        logit = self.approval.predict(X, raw_score=True)
        slope, intercept = self.calibration
        p50 = self.days_p50.predict(X)
        fallback = float(np.mean(list(self.clawback_rate.values())))
        return pd.DataFrame({
            "p_approve_raw": 1 / (1 + np.exp(-logit)),
            "p_approve": 1 / (1 + np.exp(-(slope * logit + intercept))),
            "paid_ratio": self.paid_ratio.predict(X),
            "days_p50": p50,
            # Two independent quantile models can cross; the 90th percentile is never below the median.
            "days_p90": np.maximum(self.days_p90.predict(X), p50),
            "clawback_rate": X["payer"].astype(str).map(self.clawback_rate).fillna(fallback).to_numpy(),
        }, index=X.index)

    def explain(self, X_row: pd.DataFrame, top_k: int = 5) -> list[tuple[str, float]]:
        """Per-feature SHAP contributions to the approval log-odds (pred_contrib, built into LightGBM).

        Negative values push towards denial. They explain the raw score; calibration is monotone,
        so the ranking of drivers carries over to the calibrated probability.
        """
        contrib = self.approval.predict(X_row, pred_contrib=True)[0][:-1]  # last column is the base value
        row = X_row.iloc[0]
        named = [(f"{f}={_fmt(row[f])}", float(c)) for f, c in zip(features.FEATURES, contrib)]
        return sorted(named, key=lambda t: -abs(t[1]))[:top_k]

    def importance(self, top_k: int = 10) -> list[tuple[str, float]]:
        gain = self.approval.feature_importance("gain")
        share = gain / gain.sum()
        return sorted(zip(features.FEATURES, share.round(4).tolist()), key=lambda t: -t[1])[:top_k]

    # --- persistence ----------------------------------------------------------------------------

    BOOSTERS = ("approval", "paid_ratio", "days_p50", "days_p90")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in self.BOOSTERS:
            # model_to_string() keeps only the trees up to the early-stopping best iteration.
            (directory / f"{name}.txt").write_text(getattr(self, name).model_to_string())
        (directory / "meta.json").write_text(json.dumps({
            "calibration": {"slope": self.calibration[0], "intercept": self.calibration[1]},
            "clawback_rate": self.clawback_rate,
            "info": asdict(self.info),
        }, indent=2))

    @classmethod
    def load(cls, directory: Path) -> FundingModels:
        b = {name: lgb.Booster(model_str=(directory / f"{name}.txt").read_text()) for name in cls.BOOSTERS}
        meta = json.loads((directory / "meta.json").read_text())
        cal = meta["calibration"]
        return cls(b["approval"], (cal["slope"], cal["intercept"]), b["paid_ratio"], b["days_p50"], b["days_p90"], meta["clawback_rate"],
                   TrainingInfo(**meta["info"]))


def _fmt(value) -> str:
    if isinstance(value, float):
        return "missing" if np.isnan(value) else f"{value:.3g}"
    return str(value)
