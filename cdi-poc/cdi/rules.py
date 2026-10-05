"""Deterministic CDI rules.

Each rule reads the claim (codes + clinical note) and returns zero or more Findings.
Rules are the explainable backbone of the engine; the ML model only ranks risk on top of them.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .codes import (
    LATERALITY_CODE_PREFIXES,
    PRIOR_AUTH_REQUIRED,
    SUPPORTING_DX_PREFIXES,
    describe,
)
from .models import Category, Claim, Finding, PhysicianQuery, Severity

Rule = Callable[[Claim], list[Finding]]

UNDETERMINED_OPTIONS = ["Other (please specify)", "Clinically undetermined"]


# --- text helpers -----------------------------------------------------------------------------

def _sentences(note: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", note) if s.strip()]


def _evidence(note: str, pattern: str) -> list[str]:
    rx = re.compile(pattern, re.IGNORECASE)
    return [s for s in _sentences(note) if rx.search(s)]


def _has(note: str, pattern: str) -> bool:
    return re.search(pattern, note, re.IGNORECASE) is not None


def _number(note: str, pattern: str) -> float | None:
    """Return the first numeric capture group of `pattern`, or None."""
    m = re.search(pattern, note, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _has_dx(claim: Claim, *prefixes: str) -> bool:
    return any(dx.startswith(prefixes) for dx in claim.dx_codes)


def _laterality(note: str) -> str | None:
    if _has(note, r"\bbilateral\b|\bboth knees\b"):
        return "bilateral"
    if _has(note, r"\bright\b"):
        return "right"
    if _has(note, r"\bleft\b"):
        return "left"
    return None


# --- rules --------------------------------------------------------------------------------------

def signature_missing(claim: Claim) -> list[Finding]:
    if _has(claim.note, r"electronically signed by|signed by dr"):
        return []
    return [Finding(
        rule_id="SIGNATURE_MISSING",
        category=Category.COMPLIANCE,
        severity=Severity.HIGH,
        message="Clinical note is not authenticated (no provider signature).",
        recommendation="Have the rendering provider sign the note before claim submission.",
        likely_denial_code="CO-16",
    )]


def prior_auth_missing(claim: Claim) -> list[Finding]:
    required = PRIOR_AUTH_REQUIRED.get(claim.payer, set())
    missing = [cpt for cpt in claim.cpt_codes if cpt in required]
    if not missing or claim.prior_auth_number:
        return []
    return [Finding(
        rule_id="PRIOR_AUTH_MISSING",
        category=Category.DENIAL_RISK,
        severity=Severity.HIGH,
        message=f"{claim.payer} requires prior authorisation for {', '.join(missing)} but none is on the claim.",
        recommendation="Obtain or attach the authorisation number before submitting.",
        evidence=[f"{cpt}: {describe(cpt)}" for cpt in missing],
        likely_denial_code="CO-197",
    )]


def dx_px_mismatch(claim: Claim) -> list[Finding]:
    findings = []
    for cpt in claim.cpt_codes:
        prefixes = SUPPORTING_DX_PREFIXES.get(cpt)
        if prefixes and not _has_dx(claim, *prefixes):
            findings.append(Finding(
                rule_id="DX_PX_MISMATCH",
                category=Category.DENIAL_RISK,
                severity=Severity.HIGH,
                message=f"No diagnosis on the claim supports {cpt} ({describe(cpt)}).",
                recommendation="Review the note for the condition that justified the procedure and code it.",
                evidence=[f"Claim diagnoses: {', '.join(claim.dx_codes)}"],
                likely_denial_code="CO-11",
            ))
    return findings


def medical_necessity(claim: Claim) -> list[Finding]:
    findings = []
    note = claim.note

    if "73721" in claim.cpt_codes:
        therapy = _has(note, r"physical therapy|\bNSAIDs?\b|home exercise|intra-articular injection")
        weeks = _number(note, r"(\d+)\s*weeks?")
        if not therapy or (weeks or 0) < 6:
            gaps = []
            if not therapy:
                gaps.append("conservative therapy tried")
            if (weeks or 0) < 6:
                gaps.append("symptom duration of at least 6 weeks")
            findings.append(Finding(
                rule_id="MEDICAL_NECESSITY_MRI",
                category=Category.DENIAL_RISK,
                severity=Severity.HIGH,
                message="Knee MRI lacks documented medical necessity: missing " + " and ".join(gaps) + ".",
                recommendation="Document conservative management (e.g. PT, NSAIDs) and its duration/outcome.",
                evidence=_evidence(note, r"knee|pain|weeks"),
                likely_denial_code="CO-50",
            ))

    if "29881" in claim.cpt_codes:
        mechanical = _has(note, r"locking|catching|giving way")
        imaging = _has(note, r"MRI (?:confirms|shows|demonstrates)[^.]*tear")
        if not (mechanical or imaging):
            findings.append(Finding(
                rule_id="MEDICAL_NECESSITY_ARTHROSCOPY",
                category=Category.DENIAL_RISK,
                severity=Severity.HIGH,
                message="Arthroscopic meniscectomy lacks mechanical symptoms or imaging-confirmed tear.",
                recommendation="Document mechanical symptoms and/or the MRI finding that confirms the tear.",
                evidence=_evidence(note, r"knee|meniscus|tear"),
                likely_denial_code="CO-50",
            ))
    return findings


def laterality(claim: Claim) -> list[Finding]:
    unspecified = [dx for dx in claim.dx_codes
                   if dx.startswith(LATERALITY_CODE_PREFIXES) and dx in {"M17.9", "S83.209A"}]
    if not unspecified:
        return []
    side = _laterality(claim.note)
    findings = []
    for dx in unspecified:
        if side:
            suggested = {
                ("M17.9", "right"): ["M17.11"], ("M17.9", "left"): ["M17.12"],
                ("M17.9", "bilateral"): ["M17.0"],
                ("S83.209A", "right"): ["S83.241A"], ("S83.209A", "left"): ["S83.242A"],
            }.get((dx, side), [])
            findings.append(Finding(
                rule_id="LATERALITY_NOT_CODED",
                category=Category.CODING_OPPORTUNITY,
                severity=Severity.MEDIUM,
                message=f"{dx} is unspecified but the note documents the {side} side.",
                recommendation="Code to the documented laterality.",
                evidence=_evidence(claim.note, side),
                suggested_codes=suggested,
                likely_denial_code="CO-16",
            ))
        else:
            findings.append(Finding(
                rule_id="LATERALITY_NOT_DOCUMENTED",
                category=Category.DENIAL_RISK,
                severity=Severity.MEDIUM,
                message=f"{dx} requires laterality, which is not documented in the note.",
                recommendation="Query the provider for the affected side.",
                likely_denial_code="CO-16",
                query=PhysicianQuery(
                    topic="Laterality",
                    clinical_indicators=[f"Diagnosis documented: {describe(dx)}"],
                    question="Please clarify the affected side for this condition.",
                    options=["Right", "Left", "Bilateral", *UNDETERMINED_OPTIONS],
                ),
            ))
    return findings


def diagnosis_specificity(claim: Claim) -> list[Finding]:
    note = claim.note
    findings = []

    if "E11.9" in claim.dx_codes:
        glucose = _number(note, r"glucose (?:of |level )?(\d{2,3})")
        egfr = _number(note, r"eGFR (?:of )?(\d{1,3})")
        indicators, suggested = [], []
        if _has(note, r"neuropath|numbness|monofilament"):
            indicators += _evidence(note, r"neuropath|numbness|monofilament")
            suggested.append("E11.40")
        if (glucose or 0) >= 250 or _has(note, r"hyperglyc"):
            indicators += _evidence(note, r"glucose|hyperglyc")
            suggested.append("E11.65")
        if (egfr is not None and egfr < 60) or _has(note, r"\bCKD\b"):
            indicators += _evidence(note, r"eGFR|CKD")
            suggested.append("E11.22")
        if suggested:
            findings.append(Finding(
                rule_id="DIABETES_SPECIFICITY",
                category=Category.CODING_OPPORTUNITY,
                severity=Severity.MEDIUM,
                message="Diabetes coded without complications, but the note contains indicators of complications.",
                recommendation="Query the provider to confirm any linked diabetic complication.",
                evidence=indicators,
                suggested_codes=suggested,
                query=PhysicianQuery(
                    topic="Type 2 diabetes – complications",
                    clinical_indicators=indicators,
                    question=("Based on your clinical judgment, can you clarify whether any of the findings "
                              "above are related to the patient's type 2 diabetes?"),
                    options=["Diabetic neuropathy", "Diabetes with hyperglycemia", "Diabetic CKD",
                             "Findings not related to diabetes", *UNDETERMINED_OPTIONS],
                ),
            ))

    if "I50.9" in claim.dx_codes:
        ef = _number(note, r"(?:EF|ejection fraction)(?: of| is|:)?\s*(\d{1,2})\s*%")
        indicators = _evidence(note, r"EF|ejection fraction|edema|BNP|diuresis|dyspnea")
        suggested = []
        if ef is not None and ef <= 40:
            suggested = ["I50.22", "I50.23"]
        elif (ef is not None and ef >= 50) or _has(note, r"preserved ejection|HFpEF|diastolic"):
            suggested = ["I50.32"]
        findings.append(Finding(
            rule_id="HEART_FAILURE_SPECIFICITY",
            category=Category.CODING_OPPORTUNITY,
            severity=Severity.HIGH if suggested else Severity.MEDIUM,
            message="Heart failure documented without type and acuity.",
            recommendation="Query the provider for heart failure type (systolic/diastolic) and acuity.",
            evidence=indicators,
            suggested_codes=suggested,
            query=PhysicianQuery(
                topic="Heart failure – type and acuity",
                clinical_indicators=indicators or ["Heart failure documented without further specification"],
                question="Please clarify the type and acuity of the heart failure being treated.",
                options=["Acute systolic", "Chronic systolic", "Acute on chronic systolic",
                         "Acute diastolic", "Chronic diastolic", "Acute on chronic diastolic",
                         *UNDETERMINED_OPTIONS],
            ),
        ))

    if "J18.9" in claim.dx_codes:
        organisms = {"streptococcus pneumoniae": "J13", "pseudomonas": "J15.1"}
        for organism, code in organisms.items():
            if _has(note, rf"(?:grew|positive for)[^.]*{organism}"):
                findings.append(Finding(
                    rule_id="PNEUMONIA_ORGANISM",
                    category=Category.CODING_OPPORTUNITY,
                    severity=Severity.MEDIUM,
                    message=f"Pneumonia coded as unspecified organism, but culture documents {organism}.",
                    recommendation="Ask the provider to confirm the causative organism in the assessment.",
                    evidence=_evidence(note, organism),
                    suggested_codes=[code],
                    query=PhysicianQuery(
                        topic="Pneumonia – causative organism",
                        clinical_indicators=_evidence(note, organism),
                        question="Based on the culture result, can you clarify the organism responsible for the pneumonia?",
                        options=[f"Pneumonia due to {organism.title()}", "Organism is a contaminant / not causative",
                                 *UNDETERMINED_OPTIONS],
                    ),
                ))
    return findings


def clinical_indicators_without_dx(claim: Claim) -> list[Finding]:
    note = claim.note
    findings = []

    hgb = _number(note, r"(?:hemoglobin|hgb)(?: of|:)?\s*(\d{1,2}(?:\.\d)?)")
    if hgb is not None and hgb < 10 and not _has_dx(claim, "D5", "D6"):
        indicators = _evidence(note, r"hemoglobin|hgb|transfus|bleed")
        findings.append(Finding(
            rule_id="ANEMIA_INDICATORS",
            category=Category.CODING_OPPORTUNITY,
            severity=Severity.MEDIUM,
            message=f"Hemoglobin {hgb} g/dL documented with no anemia diagnosis.",
            recommendation="Query the provider on the clinical significance of the low hemoglobin.",
            evidence=indicators,
            suggested_codes=["D64.9"],
            query=PhysicianQuery(
                topic="Low hemoglobin",
                clinical_indicators=indicators,
                question="Can you clarify the diagnosis, if any, associated with these findings?",
                options=["Acute blood loss anemia", "Anemia of chronic disease", "Iron deficiency anemia",
                         "Not clinically significant", *UNDETERMINED_OPTIONS],
            ),
        ))

    sodium = _number(note, r"sodium(?: of|:)?\s*(\d{3})")
    if sodium is not None and sodium < 135 and not _has_dx(claim, "E87.1"):
        indicators = _evidence(note, r"sodium|fluid restriction|confus")
        findings.append(Finding(
            rule_id="HYPONATREMIA_INDICATORS",
            category=Category.CODING_OPPORTUNITY,
            severity=Severity.LOW,
            message=f"Sodium {int(sodium)} mmol/L documented with no electrolyte diagnosis.",
            recommendation="Query the provider on the clinical significance of the low sodium.",
            evidence=indicators,
            suggested_codes=["E87.1"],
            query=PhysicianQuery(
                topic="Low sodium",
                clinical_indicators=indicators,
                question="Can you clarify the diagnosis, if any, associated with this lab value?",
                options=["Hyponatremia", "Not clinically significant", *UNDETERMINED_OPTIONS],
            ),
        ))

    bmi = _number(note, r"BMI(?: of|:)?\s*(\d{2}(?:\.\d)?)")
    if bmi is not None and bmi >= 40 and not _has_dx(claim, "E66"):
        indicators = _evidence(note, r"BMI|weight")
        findings.append(Finding(
            rule_id="OBESITY_INDICATORS",
            category=Category.CODING_OPPORTUNITY,
            severity=Severity.LOW,
            message=f"BMI {bmi} documented with no weight-related diagnosis.",
            recommendation="Query the provider for the associated clinical diagnosis (BMI alone cannot be coded as primary).",
            evidence=indicators,
            suggested_codes=["E66.01", "Z68.41"],
            query=PhysicianQuery(
                topic="Elevated BMI",
                clinical_indicators=indicators,
                question="Can you clarify the clinical significance of the documented BMI?",
                options=["Morbid (severe) obesity", "Obesity", "Not clinically significant", *UNDETERMINED_OPTIONS],
            ),
        ))
    return findings


def em_level_support(claim: Claim) -> list[Finding]:
    if "99215" not in claim.cpt_codes:
        return []
    minutes = _number(claim.note, r"total time[^.\d]*(\d{2,3})\s*minutes")
    high_mdm = _has(claim.note, r"MDM:?\s*high|high complexity medical decision")
    if (minutes or 0) >= 40 or high_mdm:
        return []
    return [Finding(
        rule_id="EM_LEVEL_NOT_SUPPORTED",
        category=Category.DENIAL_RISK,
        severity=Severity.MEDIUM,
        message="99215 billed but the note documents neither high-complexity MDM nor ≥40 minutes total time.",
        recommendation="Document MDM elements or total time, or bill the level the documentation supports.",
        evidence=_evidence(claim.note, r"minutes|MDM"),
        likely_denial_code="CO-150",
    )]


ALL_RULES: list[Rule] = [
    signature_missing,
    prior_auth_missing,
    dx_px_mismatch,
    medical_necessity,
    laterality,
    diagnosis_specificity,
    clinical_indicators_without_dx,
    em_level_support,
]


def run_rules(claim: Claim, rules: list[Rule] = ALL_RULES) -> list[Finding]:
    return [finding for rule in rules for finding in rule(claim)]
