# Step 7 - predictions and per-shipment explanations

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

Your Windows stage-6 run passed 115 tests and trained the selected model. This
step uses that saved model. Keep the generated dataset, stage-5 splits and
stage-6 model artifacts. No new dependencies are needed.

## 1. Extract this update

Download `aiqon_step7_update.zip`, right-click it in File Explorer and select
**Extract All**. Choose this existing project folder as the destination:

    C:\Users\sai manoj\Projects\aiqon-shipment-eta

Allow replacement of `README.md`, `docs/ENVIRONMENT.md` and
`docs/REQUIREMENTS.md`. All remaining update files are new. No earlier Python
module, configuration, virtual environment, dependency file, dataset or fitted
artifact is included in this update.

In Cursor, confirm `scripts/predict.py` is alongside `scripts/train_models.py`.
There should not be an extra project folder nested inside the existing project.

## 2. Run the accumulated tests

```powershell
Set-Location -LiteralPath "C:\Users\sai manoj\Projects\aiqon-shipment-eta"
.\.venv\Scripts\python.exe -m pytest -q -W error
```

Expected suite: **142 tests**. Wait for the complete result. If any fail, stop
and share the actual failure output before running the demo. The recorded local
result is 142 passed; your Windows result must be checked separately.

## 3. Run the prediction demonstration

Only after the tests pass:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --demo
```

This loads `artifacts/stage6/model.joblib`, verifies its associated metadata,
and selects the first five saved training IDs. It reads checkpoint inputs only.
It does not retrain, regenerate data, access target files, or score predictions.
Output is written into the new `artifacts/stage7_demo/` directory.

Expected summary fields:

| Field | Expected |
| --- | --- |
| predictor | ridge_propagation_alpha_10 |
| n_predictions | 5 |
| holdout_ids_predicted | 0 |
| holdout_scored | false |
| model_refitted | false |
| performance_evaluated | false |
| max_explanation_reconstruction_error_h | 0.0 locally; tiny numerical differences are acceptable |

The first shipment's local predicted remaining duration is about 26.708 hours,
with zero predicted delay magnitude. It is predicted to arrive about 4.795 hours
before its deadline. These examples demonstrate the workflow, not accuracy.

## 4. Read the explanation

In Cursor's Explorer, expand `artifacts` then `stage7_demo`.

1. Open `explanations.md`. Use Cursor's **Open Preview** action, or press
   **Ctrl+Shift+V**, to view the formatted report.
2. Open `explanation_example.png` for the first shipment's contribution chart.
3. Read the remaining-time table: start at the reference prediction, add the
   signed group contributions, then apply any nonnegative-duration adjustment.
4. Read the ETA and delay calculation below it. Delay magnitude cannot be
   negative; signed schedule deviation can be negative for early delivery.

The other outputs are `checkpoint_inputs.csv`, `predictions.csv`,
`contributions.csv`, `explanations.json`, `summary.json` and `run_manifest.json`.
The JSON/CSV preserve full numerical precision. The chart and Markdown round
values for readability. The reference is the model prediction at the training
feature means; it is not necessarily a physically realizable shipment.

Do not describe a negative group contribution as negative transit time or a
proven operational cause. It means fewer predicted remaining hours relative
to the chosen reference. See `EXPLANATION_CONTRACT.md` for interview reasoning.

## 5. Return the verification output

Paste the final pytest result and full JSON summary printed by the demo. If
either command fails, share the error. Do not delete existing artifacts or
retrain to bypass source/environment checks. Do not run final holdout evaluation.

## Explicit input mode, for later use

The demo's checkpoint CSV can also be passed through the normal input route:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --input "artifacts/stage7_demo/checkpoint_inputs.csv" --output-dir "artifacts/prediction_example"
```

This uses the same prediction/explanation logic. An existing output folder is
refused. For an intentional repeat, choose a new `--output-dir`. The CLI rejects
the full generated dataset because it contains reserved holdout IDs; those
shipments are reserved for a later, dedicated final evaluation stage.
