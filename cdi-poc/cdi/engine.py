from __future__ import annotations

from .models import CdiReport, Claim, Severity
from .risk_model import DenialRiskModel
from .rules import Rule, ALL_RULES, run_rules

HIGH_RISK = 0.5
MEDIUM_RISK = 0.2


def risk_band(prob: float | None, has_high_finding: bool) -> str:
    if prob is None:  # no model: fall back to rule severity
        return "HIGH" if has_high_finding else "LOW"
    if prob >= HIGH_RISK:
        return "HIGH"
    if prob >= MEDIUM_RISK or has_high_finding:
        return "MEDIUM"
    return "LOW"


class CdiEngine:
    """Pre-submission CDI review: rule findings + (optional) ML denial-risk score."""

    def __init__(self, model: DenialRiskModel | None = None, rules: list[Rule] = ALL_RULES):
        self.model = model
        self.rules = rules

    def analyze(self, claim: Claim) -> CdiReport:
        findings = run_rules(claim, self.rules)
        prob, drivers = self.model.predict(claim, findings) if self.model else (None, [])
        severity_rank = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
        findings.sort(key=lambda f: severity_rank[f.severity])
        return CdiReport(
            claim_id=claim.claim_id,
            denial_probability=None if prob is None else round(prob, 3),
            risk_band=risk_band(prob, any(f.severity == Severity.HIGH for f in findings)),
            findings=findings,
            risk_drivers=drivers,
        )
