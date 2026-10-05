from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class Category(StrEnum):
    DENIAL_RISK = "DENIAL_RISK"  # likely to cause a payer rejection
    CODING_OPPORTUNITY = "CODING_OPPORTUNITY"  # documentation supports more specific / additional codes
    COMPLIANCE = "COMPLIANCE"  # documentation integrity issue


class Severity(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Claim:
    claim_id: str
    payer: str
    encounter_type: str
    date_of_service: str
    provider_id: str
    patient_age: int
    patient_sex: str
    dx_codes: list[str]
    cpt_codes: list[str]
    billed_amount: float
    note: str
    prior_auth_number: str | None = None
    # Present only on historical (adjudicated) claims.
    outcome: str | None = None  # PAID | DENIED
    denial_code: str | None = None
    # Ground truth from the synthetic generator; never used as a model feature.
    scenario: str | None = None
    injected_defects: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PhysicianQuery:
    """A compliant, non-leading query: states indicators, offers options incl. 'other' / 'undetermined'."""

    topic: str
    clinical_indicators: list[str]
    question: str
    options: list[str]


@dataclass
class Finding:
    rule_id: str
    category: Category
    severity: Severity
    message: str
    recommendation: str
    evidence: list[str] = field(default_factory=list)
    likely_denial_code: str | None = None
    suggested_codes: list[str] = field(default_factory=list)
    query: PhysicianQuery | None = None


@dataclass
class RiskDriver:
    feature: str
    contribution: float


@dataclass
class CdiReport:
    claim_id: str
    denial_probability: float | None
    risk_band: str
    findings: list[Finding]
    risk_drivers: list[RiskDriver]

    def to_dict(self) -> dict:
        return asdict(self)
