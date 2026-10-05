from cdi.models import Category, Claim
from cdi.rules import run_rules

SIGNED = " Electronically signed by Dr. A. Rahman, MD."


def claim(note: str, dx: list[str], cpt: list[str], payer: str = "PAYER_B", pa: str | None = None) -> Claim:
    return Claim(
        claim_id="T-1", payer=payer, encounter_type="OUTPATIENT", date_of_service="2026-03-01",
        provider_id="PRV-001", patient_age=50, patient_sex="F", dx_codes=dx, cpt_codes=cpt,
        billed_amount=200.0, note=note, prior_auth_number=pa,
    )


def rule_ids(c: Claim) -> set[str]:
    return {f.rule_id for f in run_rules(c)}


def test_clean_claim_has_no_findings():
    c = claim("Right knee pain for 10 weeks. Completed 6 weeks of physical therapy." + SIGNED,
              ["M17.11"], ["73721"], payer="PAYER_B")
    assert rule_ids(c) == set()


def test_missing_signature():
    assert "SIGNATURE_MISSING" in rule_ids(claim("Routine visit.", ["E11.9"], ["99214"]))


def test_prior_auth_is_payer_specific():
    note = "Right knee pain for 10 weeks. Completed physical therapy." + SIGNED
    assert "PRIOR_AUTH_MISSING" in rule_ids(claim(note, ["M17.11"], ["73721"], payer="PAYER_A"))
    assert "PRIOR_AUTH_MISSING" not in rule_ids(claim(note, ["M17.11"], ["73721"], payer="PAYER_B"))
    assert "PRIOR_AUTH_MISSING" not in rule_ids(claim(note, ["M17.11"], ["73721"], payer="PAYER_A", pa="PA-1"))


def test_dx_px_mismatch():
    c = claim("Echo ordered." + SIGNED, ["M17.11"], ["99214", "93306"])
    findings = [f for f in run_rules(c) if f.rule_id == "DX_PX_MISMATCH"]
    assert len(findings) == 1 and findings[0].likely_denial_code == "CO-11"


def test_mri_medical_necessity_requires_therapy_and_duration():
    short = claim("Right knee pain for 3 weeks. Started NSAIDs." + SIGNED, ["M17.11"], ["73721"])
    untreated = claim("Right knee pain for 10 weeks. No prior treatment tried." + SIGNED, ["M17.11"], ["73721"])
    assert "MEDICAL_NECESSITY_MRI" in rule_ids(short)
    assert "MEDICAL_NECESSITY_MRI" in rule_ids(untreated)


def test_laterality_documented_but_not_coded_suggests_specific_code():
    c = claim("Left knee pain for 10 weeks. Completed physical therapy." + SIGNED, ["M17.9"], ["73721"])
    [f] = [f for f in run_rules(c) if f.rule_id == "LATERALITY_NOT_CODED"]
    assert f.category == Category.CODING_OPPORTUNITY and f.suggested_codes == ["M17.12"]


def test_laterality_not_documented_raises_query():
    c = claim("Knee pain for 10 weeks. Completed physical therapy." + SIGNED, ["M17.9"], ["73721"])
    [f] = [f for f in run_rules(c) if f.rule_id == "LATERALITY_NOT_DOCUMENTED"]
    assert f.query is not None and "Clinically undetermined" in f.query.options


def test_diabetes_specificity_from_lab_values():
    c = claim("Fasting glucose 301 mg/dL today. eGFR 44 on latest labs." + SIGNED, ["E11.9"], ["99214"])
    [f] = [f for f in run_rules(c) if f.rule_id == "DIABETES_SPECIFICITY"]
    assert set(f.suggested_codes) == {"E11.65", "E11.22"}


def test_diabetes_specificity_not_raised_for_normal_labs():
    c = claim("Fasting glucose 120 mg/dL. eGFR 90." + SIGNED, ["E11.9"], ["99214"])
    assert "DIABETES_SPECIFICITY" not in rule_ids(c)


def test_heart_failure_low_ef_suggests_systolic():
    c = claim("Prior echo EF 30%. Ankle edema." + SIGNED, ["I50.9"], ["99214"])
    [f] = [f for f in run_rules(c) if f.rule_id == "HEART_FAILURE_SPECIFICITY"]
    assert "I50.22" in f.suggested_codes


def test_clinical_indicators_without_diagnosis():
    note = "Hemoglobin 8.1 g/dL. Sodium 129 mmol/L. BMI 42.5." + SIGNED
    assert {"ANEMIA_INDICATORS", "HYPONATREMIA_INDICATORS", "OBESITY_INDICATORS"} <= rule_ids(
        claim(note, ["J18.9"], ["99223"]))
    coded = claim(note, ["J18.9", "D64.9", "E87.1", "E66.01"], ["99223"])
    assert not {"ANEMIA_INDICATORS", "HYPONATREMIA_INDICATORS", "OBESITY_INDICATORS"} & rule_ids(coded)


def test_em_level_support():
    unsupported = claim("Total time spent 20 minutes." + SIGNED, ["E11.9"], ["99215"])
    by_time = claim("Total time spent 45 minutes." + SIGNED, ["E11.9"], ["99215"])
    by_mdm = claim("MDM: high complexity." + SIGNED, ["E11.9"], ["99215"])
    assert "EM_LEVEL_NOT_SUPPORTED" in rule_ids(unsupported)
    assert "EM_LEVEL_NOT_SUPPORTED" not in rule_ids(by_time) | rule_ids(by_mdm)
