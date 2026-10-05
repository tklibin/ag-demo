"""Reference data for the POC: a small ICD-10 / CPT subset, payer policies and denial codes.

Everything here is illustrative. A real engine would load payer policies and code sets
from maintained sources (e.g. CCHI / NPHIES rules in KSA, CMS NCCI edits in the US).
"""

ICD10 = {
    # Diabetes
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E11.40": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
    "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
    "E11.22": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
    # Knee
    "M17.9": "Osteoarthritis of knee, unspecified",
    "M17.11": "Unilateral primary osteoarthritis, right knee",
    "M17.12": "Unilateral primary osteoarthritis, left knee",
    "M17.0": "Bilateral primary osteoarthritis of knee",
    "S83.209A": "Unspecified tear of unspecified meniscus, current injury, unspecified knee, initial encounter",
    "S83.241A": "Other tear of medial meniscus, current injury, right knee, initial encounter",
    "S83.242A": "Other tear of medial meniscus, current injury, left knee, initial encounter",
    # Pneumonia
    "J18.9": "Pneumonia, unspecified organism",
    "J13": "Pneumonia due to Streptococcus pneumoniae",
    "J15.1": "Pneumonia due to Pseudomonas",
    # Heart failure
    "I50.9": "Heart failure, unspecified",
    "I50.22": "Chronic systolic (congestive) heart failure",
    "I50.23": "Acute on chronic systolic (congestive) heart failure",
    "I50.32": "Chronic diastolic (congestive) heart failure",
    # Commonly under-documented secondary conditions
    "D64.9": "Anemia, unspecified",
    "E87.1": "Hypo-osmolality and hyponatremia",
    "E66.01": "Morbid (severe) obesity due to excess calories",
    "Z68.41": "Body mass index [BMI] 40.0-44.9, adult",
}

CPT = {
    "99214": "Office/outpatient visit, established patient, moderate MDM or 30-39 min",
    "99215": "Office/outpatient visit, established patient, high MDM or 40-54 min",
    "99223": "Initial hospital inpatient care, high MDM",
    "73721": "MRI any joint of lower extremity without contrast",
    "29881": "Arthroscopy, knee, with meniscectomy (medial OR lateral)",
    "93306": "Echocardiography, transthoracic, complete",
    "71046": "Radiologic exam, chest, 2 views",
}

# Procedure -> ICD-10 prefixes that support medical necessity.
# CPTs not listed (e.g. E/M visits) accept any diagnosis.
SUPPORTING_DX_PREFIXES = {
    "73721": ("M17", "M23", "M25.56", "S83"),
    "29881": ("M23", "S83"),
    "93306": ("I50", "I10", "R06", "R07"),
    "71046": ("J13", "J15", "J18", "R05", "R06"),
}

# Payer-specific prior authorisation lists.
PRIOR_AUTH_REQUIRED = {
    "PAYER_A": {"73721", "29881"},
    "PAYER_B": {"29881"},
    "PAYER_C": {"73721", "29881", "93306"},
}

PAYERS = tuple(PRIOR_AUTH_REQUIRED)

# Codes that are "unspecified" and have a more specific alternative when documentation supports it.
UNSPECIFIED_CODES = {"E11.9", "M17.9", "S83.209A", "J18.9", "I50.9"}

# Codes that require laterality to be documented.
LATERALITY_CODE_PREFIXES = ("M17", "S83")

# Claim adjustment reason codes (CARC-style) used in the sample data.
DENIAL_CODES = {
    "CO-11": "Diagnosis inconsistent with the procedure",
    "CO-16": "Claim lacks information or has submission/billing errors",
    "CO-50": "Not deemed a medical necessity by the payer",
    "CO-150": "Information submitted does not support this level of service",
    "CO-197": "Precertification/authorization absent",
    "CO-29": "Time limit for filing has expired",
}


def describe(code: str) -> str:
    return ICD10.get(code) or CPT.get(code) or code
