# Claim Funding – LightGBM Proof of Concept

Predict, **at the lock point** (ED claim coded and scrubbed, about to be submitted to the insurer), how
much cash a claim will bring in and when, then turn that into a **tiered funding offer**:

| Model | LightGBM objective | Predicts |
|---|---|---|
| Approval | `binary` + Platt calibration | P(insurer approves the claim) |
| Paid ratio | `cross_entropy` | Share of the billed amount paid if approved (insurers cut line items) |
| Days to cash | `quantile` (α = 0.5 and 0.9) | Typical and slow-case days from lock to payment |
| Clawback | per-payer historical rate | Share of paid money recouped later after audit |

```
expected collectible = billed × P(approve) × paid ratio × (1 − clawback rate)
advance              = tier rate × billed × paid ratio          (A: 80%, B: 50%, C: wait for approval)
fee rate             = cost of capital × p90 days / 365 + (1 − P(approve)) × loss given denial + clawback + ops margin
```

> All data is **synthetic**. No real patient or claim data is used.

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# macOS: LightGBM needs OpenMP. Either `brew install libomp`, or reuse the copy scikit-learn bundles:
scripts/fix_libomp_macos.sh

.venv/bin/python -m funding demo     # generate → train → evaluate → backtest → offers → explain
.venv/bin/python -m pytest           # tests
```

| Command | What it does |
|---|---|
| `python -m funding generate [--per-day 40 --seed 42]` | 18 months of ED claims with outcomes (`data/claims_history.csv`) + one week of pending claims |
| `python -m funding train [--as-of 2026-08-01]` | Trains the four models on claims mature before `as_of` → `data/models/` |
| `python -m funding evaluate` | Out-of-time test vs logistic regression; calibration; drift by payer × service line |
| `python -m funding backtest` | Realised P&L of the policy on test claims vs a naive flat 80% advance |
| `python -m funding offers [--top 10]` | Scores pending claims → `data/offers.csv` |
| `python -m funding explain --claim-id NEW-000212` | One claim's offer and its approval drivers |

## Files

```
funding/
  data_gen.py   synthetic claims + insurer behaviour (approval, cuts, timing, clawbacks, one policy change)
  features.py   feature frame: claim facts at lock + point-in-time payer/hospital history
  models.py     the four LightGBM models, time splits, calibration, explanations, save/load
  evaluate.py   out-of-time evaluation, logistic-regression baseline, calibration and drift tables
  policy.py     tiers, advance, fee; backtest; naive comparison policy
  cli.py        commands above
```

## How LightGBM is used, and why

**Native categoricals.** `payer`, `hospital` and `service_line` are pandas `category` columns with fixed
category lists ([features.py](funding/features.py)). LightGBM splits on them directly, so there's no
one-hot encoding. Category codes must be identical in training and scoring, which is why the lists are fixed.

**Missing values.** The history features (a payer's approval rate over the last 90 days, for example)
are NaN until there are enough events. LightGBM learns which branch missing values should take, so
there's no imputation step. The logistic baseline needs `SimpleImputer` for the same features.

**Early stopping on a time-based validation set.** `n_estimators=3000` is only an upper bound.
Training stops once the most recent 60 days of mature claims stop improving (100 rounds of patience),
which typically keeps 100–200 trees. Validation is always *later* than training, never a random sample.

**Monotone constraints.** The approval model is forced to never increase approval when open
high-severity CDI alerts go up, and never decrease it when documentation completeness goes up, and so on
(`APPROVAL_MONOTONE`). This prevents noisy rules such as "two alerts are safer than one", which a credit
committee would reject, and a test checks it.

**One objective per question.**
- `binary` for approve/deny.
- `cross_entropy` for the paid ratio. It accepts labels anywhere in [0, 1], and predictions stay in [0, 1].
- `quantile` for timing. Capital is priced on the **90th percentile** of days to cash, because the slow tail
  is what ties up money. The two quantile models can cross, so p90 is clipped to be at least p50.

**Calibration.** Pricing uses the probability itself, so a score of 0.95 has to mean 95% of those claims
are approved. Platt scaling (slope and intercept on the raw log-odds) is fitted on the validation window.
Isotonic regression was tried first. Its step function overfit about 2,400 validation claims: claims it
scored 0.99 were approved 94.7% of the time out of time.

**Explanations.** `booster.predict(X, pred_contrib=True)` returns per-feature SHAP values in log-odds,
built into LightGBM with no `shap` package needed. `explain` prints them for each offer, which is what
an underwriter or a hospital disputing a fee will ask for.

**No pickle.** Each booster is saved with `model_to_string()` (LightGBM's text format, trimmed to the
early-stopping best iteration) plus JSON for calibration and clawback rates.

**Small hyperparameters.** `num_leaves=15` and `min_child_samples=40` suit a few thousand rows per segment.
`n_jobs=1` because these datasets gain nothing from threads, and OpenMP thread pools can stall when two
OpenMP runtimes are loaded in one process.

## Avoiding leakage, the main way funding models lose money

1. **Point-in-time history features.** A claim locked on day *t* only sees approvals adjudicated and
   payments made before *t*. A test hides every later outcome and checks the features don't change.
2. **Label maturity.** Each model trains only on claims old enough for its label to be known on the
   training date: 75 days for approval, 180 for payment timing, 365 for clawbacks.
3. **Out-of-time evaluation.** The model is trained on claims before `as_of` and tested on the two months
   after it. Random splits hide payer drift.

## Results on the synthetic data

See [the demo run](#sample-output) below. In summary:
- **Backtest:** the model and policy are profitable on the out-of-time months, while advancing 80% of
  billed on every claim at a flat 3% fee loses money (denials and cuts outrun the fee).
- **Drift is caught.** The simulation has PAYER_C tightening CT approvals from July 2026. The drift table
  shows PAYER_C × CT with the largest gap between predicted and actual approval.
- **LightGBM doesn't beat logistic regression here.** That's expected: the simulator generates outcomes from
  an *additive* logistic formula, which is exactly the shape logistic regression assumes. On real claims
  data, where payer rules interact (thresholds per payer, service × acuity × plan rules), gradient-boosted
  trees usually win. **On real data, always keep the baseline and let the out-of-time test decide.**

## Sample output

Trimmed from `python -m funding demo`:

```
== Approval on 2398 claims locked after 2026-08-01 (out of time) ==
                                roc_auc   brier  log_loss
Logistic regression (baseline)   0.7820  0.0899    0.3117
LightGBM raw                     0.7763  0.0914    0.3158
LightGBM calibrated              0.7763  0.0916    0.3172

== Calibration: predicted vs actual approval by band ==
                 n  predicted  actual
(0.9, 0.95]    665      0.930   0.929
(0.95, 0.98]   853      0.966   0.951
(0.98, 1.0]     66      0.982   0.970

== Drift watch: payer × service line, largest gaps (positive = payer stricter than the model knows) ==
                        n  predicted  actual    gap
PAYER_C CT            111      0.806   0.667  0.139

== Days to cash ==
  Paid by p50 estimate     53.4%  (target 50%)
  Paid by p90 estimate     89.8%  (target 90%)

== Backtest on 2398 claims locked after 2026-08-01 (SAR) ==
                               model + policy   naive 80% / 3%
  Claims funded                         1,785            2,398
  Advanced                          1,342,303        2,667,099
  Fees                                 78,178           80,013
  Credit loss                          35,736          157,502
  Cost of capital                      22,633           50,004
  Net                                  19,810         -127,493
  Funded, then denied                     118              313

Claim NEW-000212  PAYER_A  H3  CTAS 5  CT  billed 1,343.85 SAR
  P(approve) 0.123   paid ratio 0.959   cash in ~43 days (p90 64)
  Tier C: advance 0.00 SAR at fee 0.00%  (unresolved high-severity CDI alert)
  Approval drivers (log-odds; negative pushes towards denial):
    -1.689  prior_auth_required=1
    -0.883  open_high_alerts=1
    -0.782  has_prior_auth=0
    -0.510  doc_completeness=0.627
```

## Next steps

- Train on real NPHIES `ClaimResponse` and `PaymentReconciliation` history (labels: approved amount,
  adjudication date, paid date, recoupments).
- A survival model for days to cash, so claims not yet paid count as censored instead of being dropped.
- A separate calibration window, recalibrated monthly; alerts when a payer × service gap passes a threshold.
- Run in shadow mode for 3–6 months: score every claim, fund none, compare against actual outcomes.
