# Step 8 - review errors and model behavior

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

Your stage-7 Windows run passed 142 tests and demonstrated matching predictions.
This stage reviews the model's weaknesses before final holdout evaluation.
It uses your existing data, stage-5 splits and stage-6 model. No new packages
or retraining are needed.

## 1. Apply the update

Download `aiqon_step8_update.zip`. In File Explorer, right-click it and choose
**Extract All**. Select this existing project folder as the destination:

    C:\Users\sai manoj\Projects\aiqon-shipment-eta

Allow replacement of `README.md`, `docs/ENVIRONMENT.md` and
`docs/REQUIREMENTS.md`. The remaining update files are new. It does not replace
any earlier Python module, configuration, environment, dependency file,
generated data or fitted artifact.

In Cursor, confirm that `scripts/review_development.py` appears alongside the
existing scripts. Avoid an extra nested project folder.

## 2. Run all accumulated tests

```powershell
Set-Location -LiteralPath "C:\Users\sai manoj\Projects\aiqon-shipment-eta"
.\.venv\Scripts\python.exe -m pytest -q -W error
```

Expected suite: **167 tests**. Wait for the final result. If a test fails,
stop and share the failure output before running the review.

## 3. Run the development review

After the tests pass:

```powershell
.\.venv\Scripts\python.exe scripts/review_development.py
```

This writes `artifacts/stage8/`. It verifies prior evidence, reviews the saved
validation predictions, and checks how the fixed predictor responds to valid
modified inputs. It does not refit, change the predictor, or predict/score the
reserved holdout shipments. It refuses an existing output folder.

Local reference summary:

| Field | Local result |
| --- | --- |
| validation_shipments | 120 |
| development_stress_shipments | 237 |
| stress_scenario_rows | 2133 |
| directional_decreases | 0 for each of the three families |
| invalid_input_cases / invalid_inputs_rejected | 8 / 8 |
| reserved_holdout_shipments | 60 |
| holdout_scored | false |
| model_refitted | false |
| predictor_changed | false |

There are also 584 modified rows with a feature-range flag. This is an
intentional part of the stress review, not a count of failed tests or unique
shipments. See the interpretation in `STAGE8_VALIDATION.md`.

## 4. Inspect the report

In Cursor, open `artifacts/stage8/review.md`, then press **Ctrl+Shift+V** for
Markdown preview. Open `route_errors.png` for the route comparison chart.

| File | What it contains |
| --- | --- |
| summary.json | Main counts and findings |
| slice_metrics.csv | Errors by route, congestion, arrival deviation and later disruptions |
| validation_errors.csv | Saved validation errors with diagnostic context |
| largest_errors.csv | Ten largest absolute validation ETA errors |
| stress_responses.csv | Individual responses to the controlled input changes |
| stress_checks.json | Directional comparisons and range-flag counts |
| invalid_input_checks.csv | Eight rejection cases and observed messages |
| review.md | Readable interpretation of the review |
| route_errors.png | Selected predictor versus carry-forward by route |
| run_manifest.json | File integrity and provenance |

Interpret the limits carefully: local validation MAE is 5.328 h, but its 95th
percentile absolute error is about 19.009 h and the largest error is 37.407 h.
Later port disruptions are associated with larger errors in this sample. The
future disruption flags are only diagnostic labels; they never become model
inputs. The modified stress rows have no known actual outcomes to score.

## 5. Share the verification checkpoint

Paste the final pytest result and the full JSON summary printed by the review.
If a command fails, share the error. Preserve existing artifacts and settings.
Do not delete outputs, regenerate data, retrain, or score the holdout to bypass
a check. We will review your results before final evaluation.
