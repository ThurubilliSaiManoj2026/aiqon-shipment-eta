# Stage 9 Windows walkthrough

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

## What this stage does

Stage 8 is confirmed on your Windows PC: 167 tests passed and the development
review completed. This update adds final evaluation of the existing model on
the 60 reserved shipments. It adds no new model, feature or prediction target.

The final evaluation was already executed locally after its protocol was
frozen. Ridge did not beat carry-forward on holdout: ETA MAE 7.123 versus
7.063 hours. The Windows run verifies your project and environment; it is not
a fresh independent experiment. Do not change the model or generator to
improve this known result.

## 1. Extract and copy the update

1. Download `aiqon_step9_update.zip` and choose **Extract All** in Windows.
2. Open the extracted folder. It contains `README.md` and the folders `docs`,
   `scripts`, `src`, and `tests`.
3. Copy those items into `C:\Users\sai manoj\Projects\aiqon-shipment-eta`.
   Merge folders; replace the three updated documentation files when asked:
   `README.md`, `docs/ENVIRONMENT.md`, and `docs/REQUIREMENTS.md`.
4. Confirm Cursor shows `scripts/freeze_evaluation.py` and
   `scripts/evaluate_holdout.py` directly under the existing `scripts` folder.

The ZIP contains no virtual environment, dependency snapshot, generated data
or existing artifacts. Keep your current `.venv`, `requirements.txt`,
`data/generated`, and `artifacts/stage5`, `stage6`, `stage7_demo`, `stage8`.

## 2. Run the tests

In Cursor's PowerShell terminal, paste:

```powershell
Set-Location -LiteralPath "C:\Users\sai manoj\Projects\aiqon-shipment-eta"
.\.venv\Scripts\python.exe -m pytest -q -W error
```

The local accumulated suite passed **186 tests**. Your runtime may differ.
New final-evaluation tests use a temporary synthetic dataset with another seed
and do not write to your main artifacts. If any test fails, stop and share the
complete error before running the following commands.

## 3. Freeze the evaluation

```powershell
.\.venv\Scripts\python.exe scripts/freeze_evaluation.py
```

This writes `artifacts/stage9_freeze`. Check the output:

| Field | Expected value |
| --- | --- |
| predictor | ridge_propagation_alpha_10 |
| comparison_baseline | carry_forward |
| training_shipments | 237 |
| holdout_shipments | 60 |
| excluded_unavailable_labels | 3 |
| primary_metric | eta_mae_h |
| bootstrap_resamples | 5000 |
| holdout_scored | false |

The freeze hash binds your actual local files; it need not equal a hash from
another machine. If this command reports a provenance mismatch, stop and share
the error. Do not edit saved hashes or regenerate previous outputs to bypass it.

## 4. Evaluate the frozen holdout

After the successful freeze, run:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_holdout.py
```

This writes `artifacts/stage9_final`. The summary should show
`holdout_scored: true`, `model_refitted: false`, `predictor_changed: false`,
60 holdout shipments and 237 training shipments. Changing `holdout_scored` to
true is intentional at this stage. The existing model is not retrained.

Local results for comparison, rounded to six decimals:

| Predictor | ETA MAE (h) | ETA RMSE (h) | Delay MAE (h) |
| --- | ---: | ---: | ---: |
| ridge_propagation_alpha_10 | 7.123037 | 10.403546 | 5.776577 |
| scheduled | 10.678759 | 16.367431 | 8.781805 |
| carry_forward | 7.063309 | 10.102236 | 5.685048 |

Small floating-point differences are possible across environments; unexpected
differences need investigation. A successful run does not establish real-world
accuracy, model superiority or a recruitment ranking.

If an output directory already exists, the command refuses to overwrite it.
Keep it and share the existing summary and error. Do not delete evidence or
repeatedly change paths to seek a better outcome.

## 5. Inspect and share the outputs

Open these files in Cursor:

- `artifacts/stage9_final/evaluation.md`: use **Ctrl+Shift+V** for the report.
- `artifacts/stage9_final/holdout_results.png`: comparison and scatter plot.
- `artifacts/stage9_final/largest_errors.csv`: the ten largest Ridge errors.

Other files retain full predictions, slice metrics, explanations, bootstrap
intervals and provenance. Share your test result and both complete JSON
summaries. After Windows verification, the next stage is clean reproduction
and submission documentation, including the required 2-3-page PDF and GitHub
repository preparation. No submission has been sent.
