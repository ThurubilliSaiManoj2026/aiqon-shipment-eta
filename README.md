# Shipment ETA and Delay Propagation

Predict final delivery ETA and delay magnitude at destination-port arrival,
using only information available at that checkpoint. The project includes a
synthetic journey generator, chronological evaluation, fixed baselines, an
interpretable Ridge pipeline, per-shipment explanations and input/error checks.

**Main finding:** Ridge improved validation MAE but did not outperform
carry-forward on the final synthetic holdout. No tuning followed holdout exposure.

| Predictor | Validation ETA MAE (h), n=120 | Holdout ETA MAE (h), n=60 | Holdout ETA RMSE (h) | Holdout delay MAE (h) |
| --- | ---: | ---: | ---: | ---: |
| Scheduled delivery | 10.656 | 10.679 | 16.367 | 8.782 |
| Carry-forward | 7.145 | 7.063 | 10.102 | 5.685 |
| Ridge propagation, alpha 10 | 5.328 | 7.123 | 10.404 | 5.777 |

Ridge's paired MAE improvement over carry-forward is -0.060 h on holdout;
the approximate 95% bootstrap interval is [-1.131, 1.014] h. This does not
establish an advantage. These results concern an illustrative synthetic
mechanism, not measured operational performance at real ports.

## Start here

- [Technical report, 3 pages](reports/Shipment_ETA_Technical_Report.pdf)
- [Readable report source](reports/technical_report.md)
- [Measured evidence and provenance](reports/EVIDENCE_ORIGIN.md)
- [Final technical audit](reports/FINAL_AUDIT.md)
- [Optional PDF rebuild and independent numerical audit](report_tools/README.md)
- [Data generation and ground truth](docs/DATA_SPECIFICATION.md)
- [Features and chronological splits](docs/FEATURES_AND_EVALUATION.md)
- [Explanation contract](docs/EXPLANATION_CONTRACT.md)
- [Verified environments](docs/ENVIRONMENT.md)
- [Requirement checklist](docs/REQUIREMENTS.md)

## Setup from a fresh project directory

Use 64-bit Python 3.12. Verified targets are Windows/Python 3.12.1 and
Linux/Python 3.12.14. `requirements.txt` is the 23-package snapshot used for the
Windows clean installation. Keep the existing file with the repository.
No credentials, external service, GPU, Docker or editable installation is needed.
The included `.gitattributes` preserves LF text bytes across Git checkouts so
platform line-ending conversion does not invalidate the recorded source hashes.

From the repository root, Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -W error
```

Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q -W error
```

The suite has 186 tests. Runtime varies. Explicit interpreter paths make
activation unnecessary. Preserve an existing environment and results.
A version snapshot does not guarantee identical wheels on every platform.

## Run the workflow

Run each command separately and stop on an error. Each output directory must
be new. For a completed project use [separate-output reproduction](docs/REPRODUCIBILITY.md).

```powershell
.\.venv\Scripts\python.exe -W error scripts/generate_data.py
.\.venv\Scripts\python.exe -W error scripts/prepare_development.py
.\.venv\Scripts\python.exe -W error scripts/train_models.py
.\.venv\Scripts\python.exe -W error scripts/predict.py --demo
.\.venv\Scripts\python.exe -W error scripts/review_development.py
.\.venv\Scripts\python.exe -W error scripts/freeze_evaluation.py
.\.venv\Scripts\python.exe -W error scripts/evaluate_holdout.py
```

On Linux substitute `.venv/bin/python`. Outputs are `data/generated`, then
`artifacts/stage5`, `stage6`, `stage7_demo`, `stage8`, `stage9_freeze` and
`stage9_final`. If `data/generated` already contains the supplied dataset,
keep it and begin with preparation or use a separate reproduction destination.
Do not delete evidence to bypass an existing-directory or integrity error.

Open `artifacts/stage9_final/evaluation.md` and `holdout_results.png`.
`holdout_predictions.csv` retains every prediction/actual pair;
`largest_errors.csv` retains the ten largest selected-model errors. The freeze
binds data, split IDs, model and executable source to the evaluation protocol.
Hashes detect changes relative to records; they do not authenticate untrusted files.

## Predict and explain a checkpoint CSV

After training and creating the demo input:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --input artifacts/stage7_demo/checkpoint_inputs.csv --output-dir artifacts/prediction_example
```

The demo CSV is a working 20-column schema example. Inputs require complete,
finite values and UTC timestamps. Missing/unknown columns, duplicate IDs,
future outcomes and inconsistent timestamps are rejected; there is no imputation.
Known holdout IDs use the dedicated evaluator. The demo contains the first
five training IDs and is not an accuracy evaluation.

Outputs include remaining hours, UTC ETA, signed deviation, delay magnitude,
all feature/group contributions, range flags, Markdown and a chart. Input order
is retained. Only load trusted artifacts: joblib/pickle loading can execute code.
The loader checks recorded package versions and inference-source hashes;
train on the machine where the artifact will be used.

## Method and choices

The flow is configuration -> complete journeys -> separate checkpoint inputs
and labels -> chronological splits -> candidate pipelines -> frozen fitted
model -> prediction/explanation or final scoring. Reusable logic is in
`src/shipment_eta`; `scripts` holds the seven CLI entry points.

- **Targets:** remaining time is delivery minus port arrival. ETA is checkpoint
  plus nonnegative predicted remaining time. Delay is max(0, ETA minus deadline).
- **Simulation:** 300 shipments on five fictional routes; congestion-dependent
  processing, missed dispatch slots, buffers and random disturbances. Parameters
  and seed are fixed. All values are assumptions, not empirical estimates.
- **Features:** three base attributes (planned port/inland durations and congestion)
  versus nine features adding upstream lateness, congestion-workload interaction
  and dispatch timing. Route IDs are audit labels, not encoded predictors.
- **Model:** feature builder -> StandardScaler -> Ridge(SVD). Scaling fits only
  within each training fold. Eight candidates cover two groups and alpha in
  {0.1, 1, 10, 100}; selection uses pooled MAE and a predeclared 0.05 h simplicity tolerance.
- **Evaluation:** fold training counts 114, 153, 194, each followed by 40 validation
  shipments. Training requires delivery labels strictly before the next cutoff.
  The final model uses 237 records; three unavailable labels are excluded and
  the latest 60 checkpoints form holdout. No holdout records enter fitting.
- **Explanation:** standardized linear terms contribute hours relative to the
  training-mean feature vector. Terms sum exactly to raw remaining time; duration
  clipping and the delay floor are separate. The reference can be physically
  unrealizable. Contributions are not causal stage delays.

Ridge provides a small regularized model with exact arithmetic explanations.
Its linear approximation cannot fully represent dispatch discontinuities or
future random shocks. Squared-error fitting and MAE selection optimize different
quantities. This comparison does not establish that Ridge is optimal.

## Verification and limits

The user reported 186 passing tests in the original and fresh Windows
environments, with warnings as errors. All seven reproduction commands
completed; all three dataset CSV hashes and the final summary matched.
A clean Linux source tree and environment also reproduced the workflow.
See [environment evidence](docs/ENVIRONMENT.md) for exact boundaries.

The final review repeated all 186 tests with warnings as errors and all seven
CLI stages in an isolated Linux source/output tree. Explicit CSV prediction
also passed. Independent calculations verified 432 recorded comparisons,
including metrics and bootstrap bounds; all 29 frozen application/config/protocol
files remained unchanged. See [the audit record](reports/FINAL_AUDIT.md).

Tests cover timing/dispatch boundaries, availability, input validation, target
alignment, label-aware splits, fold-local fitting, metrics, persistence,
explanations, freeze integrity and CLI failures. The 2,133 controlled development
response rows have no new labels: they test behavior, not accuracy. No directional
decreases occurred in the declared scenarios; global monotonicity is not proven.
All eight declared invalid-input cases were rejected.

Ridge's holdout P95 absolute ETA error is 28.755 h and maximum is 31.664 h.
Future port disruptions are associated with large errors in small diagnostic
slices. Congestion is assumed available without reporting lag; shipments do not
compete for shared capacity. Missing-data recovery, new-route generalization,
shared shocks and production operation are not validated. Five holdout records
have feature-range flags; these are not confidence scores. Bootstrap intervals
condition on the fitted model, assume independently resampleable shipments,
and exclude training/selection variability.

## Repository map and submission status

| Path | Contents |
| --- | --- |
| `src/shipment_eta/`, `scripts/` | Reusable implementation and CLI entry points |
| `config/`, `tests/` | Fixed TOML settings and 186 tests |
| `requirements.txt` | Existing tested Windows dependency snapshot |
| `docs/` | Contracts, verification, reproduction, history and interview notes |
| `reports/` | Three-page PDF, readable source and measured evidence |
| `report_tools/` | Optional PDF builder, embedded fonts and independent run audit |
| `data/generated/` | Synthetic dataset and provenance, when created |
| `artifacts/` | Local models/results; excluded from version control |

Report and workflow are prepared. GitHub publication, accessibility review and
submission remain pending. The optional video is not required. See the
[submission checklist](docs/SUBMISSION_CHECKLIST.md).
