# Stage 9 verification and first final holdout result

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

## Verification scope

Local environment: Linux, Python 3.12.14; NumPy 2.5.4, pandas 3.1.1,
SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2, joblib 1.6.0,
pytest 9.2.0. Before scoring the main holdout, the accumulated suite completed:
**186 passed in 34.67 seconds**, with warnings treated as errors.
Stage-9 execution on the user's Windows Python 3.12.1 remains unverified.

New tests exercise a separate synthetic seed (314159). They check freeze
creation without predictions, preservation of prior evidence, changed/missing
freeze rejection, artifact integrity, refusal to overwrite, exact holdout ID
selection, checkpoint-only model inputs, absence of fitting, timestamp-based
metric recomputation, explanation reconstruction, hand-calculated errors,
paired bootstrap alignment/reproducibility and command-line execution from
another directory. Passing tests demonstrates these checked behaviors; it
does not establish that every defect is absent or that a model predicts well.

## Frozen main experiment

The original seed-42 synthetic dataset, stage-5 chronological split, unchanged
stage-6 model and stage-8 development review were verified. The evaluation
protocol and executable source were hashed before scoring. The training set
contains 237 label-available shipments; three earlier shipments remain
excluded because their labels were unavailable at the cutoff. The 60 frozen
holdout shipments were then evaluated for the first time. No refitting,
feature changes, seed search or model reselection followed.

Local evidence directories are `artifacts/stage9_freeze_verified` and
`artifacts/stage9_final_verified`. On Windows, the default paths are
`artifacts/stage9_freeze` and `artifacts/stage9_final`. Their freeze hashes may
differ because their upstream artifact provenance includes the environment.

## Measured final results

All values below are hours, evaluated on the same 60 synthetic shipments.

| Predictor | ETA MAE | ETA RMSE | ETA bias | Delay MAE | P95 absolute ETA error | Maximum absolute ETA error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge propagation, alpha 10 | 7.123037 | 10.403546 | -1.379439 | 5.776577 | 28.754722 | 31.663878 |
| Scheduled | 10.678759 | 16.367431 | -6.884851 | 8.781805 | 27.725443 | 62.247973 |
| Carry-forward | 7.063309 | 10.102236 | -0.128694 | 5.685048 | 21.941458 | 30.775772 |

Negative ETA bias means predicting delivery earlier than its actual timestamp.
Ridge and carry-forward clipped zero remaining-time predictions; scheduled
delivery clipped one because its scheduled ETA was already before the checkpoint.
Ridge improved absolute ETA error on 29 shipments and worsened it on 31,
relative to carry-forward, with no ties under the existing tolerance.

The validation advantage (5.328 h versus 7.145 h MAE) **did not persist** in
this holdout sample. Ridge is worse by 0.059728 h in average absolute ETA error,
about 3.58 minutes. It also has worse observed RMSE, delay MAE and tail error
than carry-forward here. Both improve mean absolute error over scheduled
delivery, but this result does not demonstrate Ridge superiority over the
stronger fixed baseline. The evaluated model remains the validation-selected
Ridge; do not silently substitute a holdout-selected winner.

## Conditional uncertainty and diagnostic limits

The predeclared 5,000 paired shipment bootstrap resamples (PCG64 seed 2026)
give approximate 95% percentile intervals:

- Ridge ETA MAE: **[5.329043, 9.143119] h**.
- Carry-forward MAE minus Ridge MAE: **[-1.130719, 1.013604] h**, point estimate
  -0.059728 h. The interval spans zero and does not establish an advantage.

These intervals assume independently resampleable shipments and condition on
the fitted model and observed sample. They exclude model-training/selection
variability and unseen shared disruptions. They are not individual prediction
intervals or operational delay probabilities, and spanning zero does not prove
equivalence between predictors.

The selected model's diagnostic slices show MAE 4.145 h among 47 shipments
without later port/inland disruptions, 20.521 h among seven port-only cases,
7.127 h among four inland-only cases, and 30.207 h among two cases with both.
These very small outcome-defined slices are descriptive. The future flags are
never prediction inputs, and these observations do not justify tuning against
the holdout. The largest miss, SHP-000243, underpredicts final ETA by 31.664 h.

Five holdout shipments have at least one feature beyond its training min/max.
These flags are not probabilities or a complete out-of-distribution test.
Maximum explanation reconstruction error is 7.105427357601002e-15 h, consistent
with floating-point rounding and below the specified tolerance. Exact
reconstruction explains model arithmetic, not causal truth or predictive accuracy.

## Decision and remaining work

Preserve all scores, the frozen model and the original evaluation artifacts.
This is an illustrative synthetic study with a clear baseline comparison and
limitations. The evidence does not support operational deployment or claims
that learned complexity is necessary. Potential future work requires new
independent data and a separately declared experiment; it must not be presented
as an improvement established on this exposed holdout.

Next: verify this stage on Windows, reproduce the project in a clean
environment, review setup/dependencies and requirements, prepare the required
2-3-page report and repository, and complete the submission checklist. The
assignment has not been submitted, and no ranking or interview outcome is claimed.
