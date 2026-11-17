# Step 6 - train and compare interpretable models

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

Start here only after the stage-5 preparation command has completed. Keep the
existing dataset, `artifacts/stage5/`, `.venv`, and `requirements.txt`.
This update needs no additional dependencies.

## 1. Apply this update

1. Download `aiqon_step6_update.zip`.
2. In Windows File Explorer, right-click it and choose **Extract All**.
3. Browse to `C:\Users\sai manoj\Projects\aiqon-shipment-eta` as the extraction
   destination. Extract into that project root, not into `src` or `.venv`.
4. Allow replacement of `README.md`, `docs/REQUIREMENTS.md`,
   `docs/ENVIRONMENT.md`, and `docs/FEATURES_AND_EVALUATION.md`.
   The remaining update files are new stage-6 files.
5. In Cursor, verify that `scripts/train_models.py` and `config/training.toml`
   are inside the existing project. Avoid an extra nested project folder.

The ZIP does not include a virtual environment, dependency file, dataset or
generated model artifacts. No earlier Python module is replaced.

## 2. Run the accumulated tests

In Cursor's PowerShell terminal:

```powershell
Set-Location -LiteralPath "C:\Users\sai manoj\Projects\aiqon-shipment-eta"
.\.venv\Scripts\python.exe -m pytest -q -W error
```

The expected collected suite is 115 tests. Wait for the complete result. If any
test fails, stop and share the entire failure output before running training.
The local result is 115 passed; that does not establish the Windows result.

## 3. Train using the frozen development partitions

Only after tests pass:

```powershell
.\.venv\Scripts\python.exe scripts/train_models.py
```

This reads the existing data and stage-5 manifest, compares eight Ridge
candidates on the three saved folds, selects according to the written rule,
and fits the selected pipeline on 237 eligible development shipments. It writes
to `artifacts/stage6/`. The final 60 shipments remain unscored.

Local reference summary:

- Candidate count: 8; fold count: 3; validation shipments per candidate: 120.
- Selected candidate: `ridge_propagation_alpha_10`.
- Selected validation ETA MAE: approximately 5.327633 hours.
- Best fixed baseline ETA MAE: approximately 7.145462 hours.
- Final training shipments: 237; reserved holdout shipments: 60.
- `holdout_scored: false`; `learned_model_fitted: true`.
- Local serialization maximum absolute difference: 0.0 hours.

Small last-digit numerical differences between platforms may occur. These
reference values are observations, not a substitute for your actual output.
Do not edit data or settings to force a match.

## 4. Inspect the generated files

| File in artifacts/stage6 | Purpose |
| --- | --- |
| summary.json | Counts, selection, baseline comparison and holdout status |
| candidate_metrics.csv | Every candidate/fold score and pooled scores |
| candidate_validation_predictions.csv | Validation predictions and errors |
| selection.json | Selected candidate and tie-rule evidence |
| training_config.json | Exact model-comparison settings used |
| fit_audit.json | Training/validation IDs for each candidate and fold |
| model.joblib | Complete locally fitted preprocessing/model pipeline |
| model_metadata.json | Training IDs, feature names/ranges and environment |
| coefficients.csv | Fitted parameters in standardized and raw feature units |
| run_manifest.json | Dataset, split, source and output-file hashes |

The binary model is not a text file; do not edit it. Treat only your own
locally created model as trusted. Per-shipment explanations come next.

## 5. Share the checkpoint

Paste the final pytest result and the full JSON summary printed by training.
If either command fails, provide the actual error instead. Do not delete an
existing `artifacts/stage6/` or regenerate the dataset to bypass a check.

For an intentional repeat after a successful run, a separate output folder can
be supplied with `--output-dir artifacts/stage6_repeat`; preserve the first run.
Keep the alpha grid, feature definitions and saved partitions unchanged during
this stage. Do not evaluate the final holdout yet.
