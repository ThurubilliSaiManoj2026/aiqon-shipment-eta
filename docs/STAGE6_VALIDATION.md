# Stage 6 local verification record

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

## What was run

Linux, Python 3.12.14, NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1,
scikit-learn 1.9.1, joblib 1.6.0, pytest 9.2.0. The principal package versions
match the user's supplied Windows installation output. Python patch version
and operating system differ; Windows verification is pending.

`python -m pytest -q -W error --tb=short`: **115 passed** locally.
This includes all earlier tests and 36 additional stage-6 cases. Passing tests
establishes the checked behavior, not an error-free system or predictive accuracy.

The training CLI was executed using the unchanged verified dataset and saved
stage-5 split manifest. It completed successfully. No final holdout score was
computed. A separate clean ZIP-overlay installation is also checked before
delivery of the update.

## Measured pooled validation results

All rows use the same 120 disjoint validation checkpoints, with 114, 153 and
194 eligible training shipments in successive folds. Hours are rounded to
six decimal places below; full precision is exported by the training command.

| Predictor | ETA MAE | ETA RMSE | ETA bias | Delay MAE | Clipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scheduled baseline | 10.656094 | 15.098328 | -7.294930 | 8.975512 | 1 |
| Carry-forward baseline | 7.145462 | 9.808328 | 1.005674 | 5.603137 | 0 |
| Base Ridge, alpha 0.1 | 5.429484 | 8.179179 | 0.497817 | 4.231975 | 0 |
| Base Ridge, alpha 1 | 5.425498 | 8.179329 | 0.496584 | 4.227225 | 0 |
| Base Ridge, alpha 10 | 5.392874 | 8.193493 | 0.484064 | 4.182482 | 0 |
| Base Ridge, alpha 100 | 6.175897 | 8.819981 | 0.373339 | 4.834935 | 0 |
| Propagation Ridge, alpha 0.1 | 5.408006 | 8.249233 | 0.191446 | 4.248324 | 0 |
| Propagation Ridge, alpha 1 | 5.397526 | 8.240893 | 0.184756 | 4.239342 | 0 |
| Propagation Ridge, alpha 10 | 5.327633 | 8.194474 | 0.130206 | 4.171589 | 0 |
| Propagation Ridge, alpha 100 | 5.580417 | 8.395590 | -0.099445 | 4.299868 | 0 |

The predeclared rule selects `ridge_propagation_alpha_10`. It attains the
smallest candidate ETA MAE. The best base model is 0.065241 hours worse, outside
the predeclared 0.05-hour simplicity tolerance. The selected model improves
ETA MAE over carry-forward by 1.817829 hours, approximately 25.44% on this
validation sample. The validation recommendation is therefore this Ridge.

The feature-group difference is only approximately 3.91 minutes. That is a
small observed benefit, not evidence of a reliable or statistically significant
improvement. Base alpha 10 has marginally lower RMSE than propagation alpha 10;
ETA MAE was the predeclared selection metric. Report this tradeoff honestly.
The three selected-candidate fold ETA MAEs are 4.259614, 6.222557 and 5.500728
hours, so performance is not uniform across windows.

After selection, a new pipeline was fitted on all 237 eligible development
shipments. Three earlier shipments with unavailable labels remain excluded;
the 60 final holdout shipments are neither fitted nor scored. The saved
pipeline's local save/reload maximum prediction difference was 0.0 hours on
the serialization check sample. A test also verifies loading and prediction
in a separate Python process.

## Verification coverage

- StandardScaler means, variances and sample count match training rows; running
  predictions does not alter fitted parameters.
- An independent regularized normal-equation calculation matches the fitted
  Ridge solution; exported coefficients reconstruct raw predictions.
- Shuffling target row order preserves ID-aligned training. Prediction ordering
  follows input IDs. Future columns and invalid checkpoint values are rejected.
- Actual calls to model fitting use only saved training IDs: 24 candidate/fold
  fits and one final fit. Availability cutoffs and holdout exclusion are checked.
- Every candidate's exported validation IDs match the saved windows; metrics
  recompute from exported errors. Selection and baseline recommendation follow
  the recorded rule, including a test where the simpler model wins a near tie.
- Changed dataset/stage-5 evidence, incompatible model environments, changed
  inference source and corrupted model bytes are rejected. Unsafe model bytes
  are not deserialized before compatibility/integrity checks.
- Repeating training reproduces selection, candidate metrics, coefficients and
  validation predictions without modifying dataset or stage-5 files. Existing
  output directories are preserved and rejected as repeat destinations.
- Both the training CLI and saved-model loader work from a different directory.

Hashes only check consistency against locally trusted manifests; they do not
authenticate a model from an unknown source. Use only project-created artifacts.
The saved feature transformer calls project source code, so the model loader
also verifies the source files needed for inference against recorded hashes.

## Limits and remaining work

These are validation scores used for selection, on one small synthetic dataset
whose mechanism was designed by the project. They are optimistic selected
scores and do not establish real-world accuracy, robustness, calibrated
uncertainty, causal effects, or a top-5% ranking. The generator was not changed
to obtain the reported improvement. More candidate searches are not added
after observing these scores in this stage.

Per-shipment explanations, a standalone prediction command, robustness and
slice analysis, final holdout evaluation after freezing decisions, final
reproducibility review, GitHub packaging and the required PDF remain future
work. Coefficients alone do not complete the explanation requirement.
