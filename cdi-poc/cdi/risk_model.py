"""Denial-risk model: logistic regression over rule flags + claim context.

Logistic regression keeps the model explainable — each claim's risk drivers are the features
with the largest coefficient × (value − training mean) contribution to the log-odds.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .models import Category, Claim, Finding, RiskDriver
from .rules import run_rules


def features(claim: Claim, findings: list[Finding]) -> dict[str, float]:
    # Pure coding opportunities (no likely denial code) affect revenue, not denials; leaving them
    # out stops the model from reporting spurious correlations as denial drivers.
    f: dict[str, float] = {f"rule={x.rule_id}": 1.0 for x in findings
                           if x.category != Category.CODING_OPPORTUNITY or x.likely_denial_code}
    f[f"payer={claim.payer}"] = 1.0
    f[f"encounter={claim.encounter_type}"] = 1.0
    for cpt in claim.cpt_codes:
        f[f"cpt={cpt}"] = 1.0
    f["log_billed_amount"] = math.log1p(claim.billed_amount)
    return f


@dataclass
class TrainingMetrics:
    n_train: int
    n_test: int
    denial_rate: float
    roc_auc: float
    average_precision: float


class DenialRiskModel:
    """Stored as plain JSON (feature names, coefficients, training means) — no pickle to load."""

    def __init__(self, feature_names: list[str], coef: np.ndarray, intercept: float, means: np.ndarray,
                 metrics: TrainingMetrics):
        self.feature_names = feature_names
        self.index = {name: i for i, name in enumerate(feature_names)}
        self.coef = coef
        self.intercept = intercept
        self.means = means
        self.metrics = metrics

    @classmethod
    def train(cls, claims: list[Claim], seed: int = 7) -> DenialRiskModel:
        labelled = [c for c in claims if c.outcome in ("PAID", "DENIED")]
        X_dicts = [features(c, run_rules(c)) for c in labelled]
        y = np.array([c.outcome == "DENIED" for c in labelled], dtype=int)

        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(X_dicts)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        metrics = TrainingMetrics(
            n_train=len(y_tr),
            n_test=len(y_te),
            denial_rate=float(y.mean()),
            roc_auc=float(roc_auc_score(y_te, p_te)),
            average_precision=float(average_precision_score(y_te, p_te)),
        )
        return cls(list(vec.get_feature_names_out()), model.coef_[0], float(model.intercept_[0]),
                   X_tr.mean(axis=0), metrics)

    def _vector(self, f: dict[str, float]) -> np.ndarray:
        x = np.zeros(len(self.feature_names))
        for name, value in f.items():
            if name in self.index:  # features unseen in training are ignored
                x[self.index[name]] = value
        return x

    def predict(self, claim: Claim, findings: list[Finding], top_k: int = 3) -> tuple[float, list[RiskDriver]]:
        x = self._vector(features(claim, findings))
        prob = 1.0 / (1.0 + math.exp(-(float(self.coef @ x) + self.intercept)))
        contributions = self.coef * (x - self.means)
        # Explain only with features present on this claim that push risk up.
        present = [(self.feature_names[i], float(contributions[i])) for i in range(len(x))
                   if x[i] != 0 and contributions[i] > 0]
        present.sort(key=lambda t: -t[1])
        return prob, [RiskDriver(n, round(c, 3)) for n, c in present[:top_k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "feature_names": self.feature_names,
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "means": self.means.tolist(),
            "metrics": asdict(self.metrics),
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> DenialRiskModel:
        d = json.loads(path.read_text())
        return cls(d["feature_names"], np.array(d["coef"]), d["intercept"], np.array(d["means"]),
                   TrainingMetrics(**d["metrics"]))
