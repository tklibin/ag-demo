# CDI Engine – Proof of Concept

Pre-submission **Clinical Documentation Improvement** for insurance claims. For each claim (codes + clinical note) the engine:

1. **Finds documentation gaps that cause denials** – missing prior auth, diagnosis not supporting the procedure, weak medical necessity, unsupported E/M level, missing laterality or signature.
2. **Finds coding opportunities** – the note supports a more specific or additional diagnosis (diabetes complications, heart-failure type, pneumonia organism, anemia, hyponatremia, obesity).
3. **Drafts compliant physician queries** – states the clinical indicators, offers multiple choices incl. *Other* and *Clinically undetermined* (non-leading, per ACDIS/AHIMA guidance).
4. **Scores denial risk** with a model trained on historical claim outcomes (paid / denied), and explains the top risk drivers.

> All data is **synthetic**. No real patient information is used or needed.

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# or: python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m cdi demo          # generate data → train → insights → worklist → sample report
.venv/bin/python -m pytest            # tests
```

Individual commands:

| Command | What it does |
|---|---|
| `python -m cdi generate [--n 600 --pending 50 --seed 42]` | Synthetic adjudicated history + new pending claims → `data/*.jsonl` |
| `python -m cdi train` | Train the denial-risk model → `data/model.json`, prints ROC-AUC |
| `python -m cdi insights` | Denials by code / payer; per rule: denial rate when fired vs not, precision/recall vs injected defects |
| `python -m cdi worklist [--top 10]` | Score pending claims → `data/worklist.csv` (sorted by risk) + `data/reports.json` |
| `python -m cdi analyze --claim-id NEW-00027 [--json]` | Full CDI report for one claim |

## Sample output

```
Claim NEW-00009  PAYER_B  INPATIENT
  CPT: 99223, 71046 (Chest x-ray)     DX: E11.9
  Denial risk: 0.42 (MEDIUM)
  Risk drivers: rule=DX_PX_MISMATCH (+2.26)

  [HIGH] DX_PX_MISMATCH (DENIAL_RISK) → CO-11
    No diagnosis on the claim supports 71046 (Radiologic exam, chest, 2 views).
  [MEDIUM] ANEMIA_INDICATORS (CODING_OPPORTUNITY)
    Hemoglobin 8.8 g/dL documented with no anemia diagnosis.
    Physician query – Low hemoglobin: Can you clarify the diagnosis, if any, associated with these findings?
      Options: Acute blood loss anemia | Anemia of chronic disease | Iron deficiency anemia |
               Not clinically significant | Other (please specify) | Clinically undetermined
```

## How it works

```
claim + note ──► rules.py (deterministic CDI rules) ──► findings (evidence, action, likely denial code, query)
                                   │
                                   ▼
                  risk_model.py (logistic regression on rule flags + payer + CPT + amount)
                                   │
                                   ▼
                  engine.py → CdiReport (probability, risk band, findings, top risk drivers)
```

| File | Purpose |
|---|---|
| `cdi/codes.py` | ICD-10 / CPT subset, dx↔procedure support, payer prior-auth lists, denial codes (CARC-style) |
| `cdi/rules.py` | 8 rule functions → `Finding`s with sentence-level evidence and physician queries |
| `cdi/data_gen.py` | 5 clinical scenarios with injected defects + a simulated payer that denies probabilistically |
| `cdi/risk_model.py` | Train / predict / explain; stored as plain JSON (no pickle) |
| `cdi/engine.py` | `CdiEngine.analyze(claim) -> CdiReport` |
| `cdi/cli.py` | Commands above |

**Synthetic data.** Scenarios: diabetes follow-up, knee MRI, knee arthroscopy, pneumonia admission, heart failure. Three payers with different prior-auth rules and strictness. Each claim stores `injected_defects` (ground truth, never used as a model feature) so `insights` can report rule precision/recall. Two realistic traps are included on purpose – a negated finding (*"no neuropathy"*) and therapy mentioned but not done (*"PT recommended but not yet started"*) – to show where keyword rules break.

## Limitations (it's a POC)

- **The model learns the generator.** ROC-AUC ≈ 0.87 here proves the pipeline, not real-world accuracy. Real accuracy needs your remittance / claim-response history (835 files, or NPHIES responses in KSA).
- **Keyword rules have no negation or synonym handling** – visible in `insights` (diabetes precision 73%, MRI recall 75%).
- **Tiny, illustrative code set and payer policies.** Production needs licensed, maintained code sets and payer rules.
- **Suggestions only.** The engine never changes codes; a coder / CDI specialist confirms every suggestion, and codes are assigned only when documentation supports them.

## Path to production

1. **Real data** – historical claims + remittance/denial responses; retrain and re-measure per payer.
2. **Clinical NLP / LLM extraction** – replace regexes with an extraction step (negation, abbreviations, Arabic/English notes) that outputs structured facts the same rules consume. Run it on an in-country, PHI-approved deployment.
3. **Local standards** – for KSA: NPHIES claim/response formats, CCHI policies, SBS procedure codes, ICD-10-AM.
4. **Rules as data** – payer policies in a versioned table maintained by the revenue-cycle team, not code.
5. **Workflow** – CDI worklist UI, query tracking (sent / answered / agreed), audit trail, and feedback of query outcomes into the model.
6. **Metrics** – denial rate and first-pass acceptance rate before vs after, query response and agreement rates, revenue recovered.
