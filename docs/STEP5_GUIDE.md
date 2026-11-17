# Step 5 - features, frozen splits and validation baselines

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

Prerequisites: the step-4 tests passed and `data/generated` exists. No extra
package installation and no regeneration of the dataset are required.

## Install

Extract `aiqon_step5_update.zip` into the existing project folder. Allow
replacement of these documentation files only: `README.md`,
`docs/REQUIREMENTS.md`, `docs/DATA_SPECIFICATION.md`, and `docs/ENVIRONMENT.md`.
The other delivered files are new. The ZIP does not include `.venv`,
`requirements.txt`, `data/generated`, or existing artifact outputs.

## Test

From Cursor's PowerShell terminal in the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -W error
```

There are 79 tests. Stop and inspect the error if the run fails.

## Prepare development data

After the tests pass:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_development.py
```

This reads the already-generated data and creates `artifacts/stage5/`:

| File | Purpose |
| --- | --- |
| split_manifest.json | Exact development, validation, exclusion and holdout IDs; cutoffs and dataset bindings |
| feature_schema.json | Feature names and definitions |
| development_features_base.csv | Three numeric features for 237 development shipments |
| development_features_propagation.csv | Nine numeric features for those same shipments |
| baseline_metrics.csv | Per-fold and pooled validation metrics for two rules |
| validation_predictions.csv | Predictions/errors for 120 validation shipments, once per rule |
| summary.json | Counts, feature diagnostics and pooled metrics |
| run_manifest.json | Source/environment/data/output provenance |

The final holdout is unscored. `holdout_scored` and `learned_model_fitted` should
both be false. The final printed summary is what to share for review, together
with the test result.

If the command says the output already exists, preserve it. Do not delete the
frozen split or rerun generation. An intentional reproducibility check can use
`--output-dir artifacts/stage5_repeat` after discussing why a repeat is needed.

The next stage is fitting and comparing models using the saved validation
folds. Read FEATURES_AND_EVALUATION.md for the reasoning and current limitations.
