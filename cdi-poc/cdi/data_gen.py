"""Synthetic claims + clinical notes with injected documentation defects and simulated payer outcomes.

No real patient data. Each claim records the defects that were injected (`injected_defects`),
which lets us measure rule precision/recall; the payer simulation turns defects into
probabilistic PAID / DENIED outcomes with CARC-style denial codes.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from .codes import PAYERS, PRIOR_AUTH_REQUIRED
from .models import Claim

# Probability that a payer denies a claim because of a given defect, and the code it uses.
DENIAL_BEHAVIOUR = {
    "PRIOR_AUTH_MISSING": (0.90, "CO-197"),
    "DX_PX_MISMATCH": (0.85, "CO-11"),
    "MEDICAL_NECESSITY_MRI": (0.70, "CO-50"),
    "MEDICAL_NECESSITY_ARTHROSCOPY": (0.70, "CO-50"),
    "SIGNATURE_MISSING": (0.60, "CO-16"),
    "EM_LEVEL_NOT_SUPPORTED": (0.45, "CO-150"),
    "LATERALITY_NOT_DOCUMENTED": (0.40, "CO-16"),
    "LATERALITY_NOT_CODED": (0.25, "CO-16"),
    "HEART_FAILURE_SPECIFICITY": (0.10, "CO-16"),
    "DIABETES_SPECIFICITY": (0.05, "CO-16"),
    "PNEUMONIA_ORGANISM": (0.05, "CO-16"),
    # Coding opportunities: revenue / severity impact, but not a reason for denial.
    "ANEMIA_INDICATORS": (0.0, None),
    "HYPONATREMIA_INDICATORS": (0.0, None),
    "OBESITY_INDICATORS": (0.0, None),
}
BASE_DENIAL = (0.04, "CO-29")  # noise unrelated to documentation, e.g. late filing
PAYER_STRICTNESS = {"PAYER_A": 1.0, "PAYER_B": 0.8, "PAYER_C": 1.15}

PROVIDERS = [f"PRV-{i:03d}" for i in range(1, 9)]
DOCTORS = ["Dr. A. Rahman", "Dr. S. Haddad", "Dr. L. Chen", "Dr. M. Okafor", "Dr. R. Patel"]


# --- scenarios ----------------------------------------------------------------------------------
# Each returns (note_lines, dx_codes, cpt_codes, billed_amount, defects).

def _diabetes_followup(rng: random.Random):
    lines, defects = ["Established patient seen for type 2 diabetes follow-up."], []
    cpt = "99215" if rng.random() < 0.4 else "99214"
    lines.append(f"HbA1c {rng.uniform(6.8, 10.5):.1f}%.")

    dx = "E11.9"
    complication = rng.choice(["neuropathy", "hyperglycemia", "ckd"]) if rng.random() < 0.45 else None
    if complication == "neuropathy":
        lines.append("Monofilament exam shows decreased sensation in both feet, consistent with peripheral neuropathy.")
        specific = "E11.40"
    elif complication == "hyperglycemia":
        lines.append(f"Fasting glucose {rng.randint(260, 340)} mg/dL today.")
        specific = "E11.65"
    elif complication == "ckd":
        lines.append(f"eGFR {rng.randint(35, 55)} on latest labs.")
        specific = "E11.22"
    else:
        lines.append(f"Fasting glucose {rng.randint(110, 160)} mg/dL. eGFR {rng.randint(75, 105)}.")
        if rng.random() < 0.15:
            # Negated finding: keyword rules without negation handling will misfire here.
            lines.append("Monofilament exam normal, no neuropathy.")
    if complication:
        if rng.random() < 0.5:
            dx = specific
        else:
            defects.append("DIABETES_SPECIFICITY")

    dx_codes = [dx]
    if rng.random() < 0.2:
        lines.append(f"BMI {rng.uniform(40.1, 44.8):.1f}.")
        if rng.random() < 0.3:
            dx_codes += ["E66.01", "Z68.41"]
        else:
            defects.append("OBESITY_INDICATORS")
    else:
        lines.append(f"BMI {rng.uniform(24.0, 36.0):.1f}.")

    lines.append("Plan: continue metformin, titrate basal insulin, dietary counselling, recheck HbA1c in 3 months.")
    if cpt == "99215":
        if rng.random() < 0.65:
            lines.append(rng.choice(["Total time spent 45 minutes including counselling.",
                                     "MDM: high complexity given insulin titration and comorbidities."]))
        else:
            if rng.random() < 0.5:
                lines.append(f"Total time spent {rng.randint(15, 29)} minutes.")
            defects.append("EM_LEVEL_NOT_SUPPORTED")
    else:
        lines.append(f"Total time spent {rng.randint(30, 39)} minutes.")
    return lines, dx_codes, [cpt], 260.0 if cpt == "99215" else 180.0, defects


def _knee_mri(rng: random.Random):
    lines, defects = [], []
    side = rng.choice(["right", "left"])
    documented_side = rng.random() < 0.85
    weeks_ok = rng.random() < 0.7
    subject = f"{side.capitalize()} knee" if documented_side else "Knee"

    if weeks_ok:
        lines.append(f"{subject} pain for {rng.randint(8, 16)} weeks with stiffness and swelling.")
        lines.append("Completed 6 weeks of physical therapy and NSAIDs without adequate relief.")
    else:
        if rng.random() < 0.5:
            lines.append(f"{subject} pain for {rng.randint(2, 4)} weeks. Started NSAIDs.")
        else:
            # The second phrasing mentions therapy without it having happened: keyword rules miss it.
            lines.append(f"{subject} pain for {rng.randint(8, 12)} weeks. " + rng.choice(
                ["No prior treatment tried.", "Physical therapy recommended but not yet started."]))
        defects.append("MEDICAL_NECESSITY_MRI")
    lines.append("X-ray shows joint space narrowing. MRI ordered to evaluate internal derangement.")

    if rng.random() < 0.08:
        dx_codes = ["E11.9"]  # knee diagnosis omitted from the claim
        defects.append("DX_PX_MISMATCH")
    elif not documented_side:
        dx_codes = ["M17.9"]
        defects.append("LATERALITY_NOT_DOCUMENTED")
    elif rng.random() < 0.75:
        dx_codes = ["M17.11" if side == "right" else "M17.12"]
    else:
        dx_codes = ["M17.9"]
        defects.append("LATERALITY_NOT_CODED")
    return lines, dx_codes, ["73721"], 1200.0, defects


def _knee_arthroscopy(rng: random.Random):
    lines, defects = [], []
    side = rng.choice(["right", "left"])
    lines.append(f"Twisting injury to the {side} knee playing football.")
    if rng.random() < 0.8:
        lines.append("Reports locking and catching. MRI shows a tear of the medial meniscus posterior horn.")
    else:
        lines.append("Persistent knee pain despite rest.")
        defects.append("MEDICAL_NECESSITY_ARTHROSCOPY")
    lines.append("Proceeded with arthroscopic partial medial meniscectomy. No complications.")

    if rng.random() < 0.8:
        dx_codes = ["S83.241A" if side == "right" else "S83.242A"]
    else:
        dx_codes = ["S83.209A"]
        defects.append("LATERALITY_NOT_CODED")
    return lines, dx_codes, ["29881"], 4500.0, defects


def _pneumonia_admission(rng: random.Random):
    lines, defects = ["Admitted with fever, productive cough and hypoxia."], []
    lines.append("Chest x-ray shows lower lobe consolidation. Started IV ceftriaxone and azithromycin.")
    dx = "J18.9"
    if rng.random() < 0.5:
        organism, code = rng.choice([("Streptococcus pneumoniae", "J13"), ("Pseudomonas aeruginosa", "J15.1")])
        lines.append(f"Sputum culture grew {organism}.")
        if rng.random() < 0.5:
            dx = code
        else:
            defects.append("PNEUMONIA_ORGANISM")
    else:
        lines.append("Sputum culture pending.")

    dx_codes = [dx]
    if rng.random() < 0.3:
        lines.append(f"Hemoglobin {rng.uniform(7.8, 9.6):.1f} g/dL.")
        if rng.random() < 0.3:
            dx_codes.append("D64.9")
        else:
            defects.append("ANEMIA_INDICATORS")
    else:
        lines.append(f"Hemoglobin {rng.uniform(12.0, 14.5):.1f} g/dL.")
    if rng.random() < 0.25:
        lines.append(f"Sodium {rng.randint(126, 133)} mmol/L, fluid restriction started.")
        if rng.random() < 0.3:
            dx_codes.append("E87.1")
        else:
            defects.append("HYPONATREMIA_INDICATORS")
    else:
        lines.append(f"Sodium {rng.randint(137, 142)} mmol/L.")

    if rng.random() < 0.05:
        dx_codes = ["E11.9", *[c for c in dx_codes if c in ("D64.9", "E87.1")]]  # pneumonia omitted
        defects = [d for d in defects if d != "PNEUMONIA_ORGANISM"] + ["DX_PX_MISMATCH"]
    return lines, dx_codes, ["99223", "71046"], 650.0, defects


def _heart_failure(rng: random.Random):
    lines, defects = ["Follow-up for heart failure with worsening dyspnea on exertion and ankle edema."], []
    ef = None
    if rng.random() < 0.6:
        ef = rng.choice([rng.randint(25, 38), rng.randint(52, 62)])
        lines.append(f"Prior echo EF {ef}%.")
    lines.append("Increased furosemide. Repeat transthoracic echo ordered to reassess function.")

    if rng.random() < 0.10:
        lines.append("Also has right knee osteoarthritis.")
        dx_codes = ["M17.11"]  # heart failure omitted from the claim
        defects.append("DX_PX_MISMATCH")
    elif ef is not None and rng.random() < 0.5:
        dx_codes = ["I50.22" if ef <= 40 else "I50.32"]
    else:
        dx_codes = ["I50.9"]
        defects.append("HEART_FAILURE_SPECIFICITY")
    lines.append(f"Total time spent {rng.randint(30, 39)} minutes.")
    return lines, dx_codes, ["99214", "93306"], 700.0, defects


SCENARIOS = {
    "diabetes_followup": (_diabetes_followup, "OUTPATIENT", 0.30),
    "knee_mri": (_knee_mri, "OUTPATIENT", 0.20),
    "knee_arthroscopy": (_knee_arthroscopy, "DAY_SURGERY", 0.15),
    "pneumonia_admission": (_pneumonia_admission, "INPATIENT", 0.20),
    "heart_failure": (_heart_failure, "OUTPATIENT", 0.15),
}


# --- payer simulation -----------------------------------------------------------------------------

def adjudicate(defects: list[str], payer: str, rng: random.Random) -> tuple[str, str | None]:
    """Simulate the payer: each defect independently may trigger a denial; highest-impact first."""
    strictness = PAYER_STRICTNESS[payer]
    ordered = sorted(defects, key=lambda d: -DENIAL_BEHAVIOUR[d][0])
    for defect in ordered:
        prob, code = DENIAL_BEHAVIOUR[defect]
        if prob and rng.random() < min(prob * strictness, 0.97):
            return "DENIED", code
    if rng.random() < BASE_DENIAL[0]:
        return "DENIED", BASE_DENIAL[1]
    return "PAID", None


# --- generator ------------------------------------------------------------------------------------

def generate_claim(i: int, rng: random.Random) -> Claim:
    names = list(SCENARIOS)
    scenario = rng.choices(names, weights=[SCENARIOS[n][2] for n in names])[0]
    build, encounter_type, _ = SCENARIOS[scenario]
    lines, dx_codes, cpt_codes, amount, defects = build(rng)
    payer = rng.choice(PAYERS)

    if rng.random() < 0.92:
        lines.append(f"Electronically signed by {rng.choice(DOCTORS)}, MD.")
    else:
        defects.append("SIGNATURE_MISSING")

    prior_auth = None
    if any(cpt in PRIOR_AUTH_REQUIRED[payer] for cpt in cpt_codes):
        if rng.random() < 0.8:
            prior_auth = f"PA-{rng.randint(100000, 999999)}"
        else:
            defects.append("PRIOR_AUTH_MISSING")

    outcome, denial_code = adjudicate(defects, payer, rng)
    dos = date(2026, 1, 1) + timedelta(days=rng.randint(0, 240))
    return Claim(
        claim_id=f"CLM-{i:05d}",
        payer=payer,
        encounter_type=encounter_type,
        date_of_service=dos.isoformat(),
        provider_id=rng.choice(PROVIDERS),
        patient_age=rng.randint(22, 85) if scenario != "knee_arthroscopy" else rng.randint(18, 45),
        patient_sex=rng.choice(["M", "F"]),
        dx_codes=dx_codes,
        cpt_codes=cpt_codes,
        billed_amount=round(amount * rng.uniform(0.9, 1.15), 2),
        note=" ".join(lines),
        prior_auth_number=prior_auth,
        outcome=outcome,
        denial_code=denial_code,
        scenario=scenario,
        injected_defects=sorted(defects),
    )


def generate(n: int = 600, seed: int = 42) -> list[Claim]:
    rng = random.Random(seed)
    return [generate_claim(i, rng) for i in range(1, n + 1)]


def save(claims: list[Claim], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in claims:
            f.write(json.dumps(c.to_dict()) + "\n")


def load(path: Path) -> list[Claim]:
    with path.open() as f:
        return [Claim.from_dict(json.loads(line)) for line in f if line.strip()]
