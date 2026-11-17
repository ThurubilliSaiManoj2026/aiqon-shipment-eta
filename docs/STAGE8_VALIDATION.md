# Stage 8 local verification and engineering findings

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

## Verification boundaries

The user supplied Windows stage-7 evidence: 142 tests passed with warnings
treated as errors, and five demonstration predictions matched local results.
The 3.55e-15-hour reconstruction difference is normal floating-point rounding.

Stage 8 was verified locally on Linux, Python 3.12.14, NumPy 2.5.4, pandas
3.1.1, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, Matplotlib 3.11.2 and
pytest 9.2.0. Windows stage-8 execution remains pending.

`python -m pytest -q -W error --tb=short`: **167 passed** locally.
The suite includes 25 additional cases for coherent perturbations, invalid
settings, exact fixed-rule responses, deliberately reversed model responses,
slice boundaries, empty/small groups, hand-calculated metrics, paired ID
alignment, corrupted saved predictions, artifact integrity, no-refit/read-only
behavior, holdout exclusion, reproducibility and the command-line workflow.

The deliberately incorrect-response test verifies that the audit reports all
expected directional reversals, instead of silently declaring good behavior.
The full update is also checked after overlaying the incremental ZIPs into a
clean folder. Earlier source modules remain byte-identical to stage 7.

## Measured validation findings

These use the same 120 saved out-of-fold predictions that informed selection.
The final model's predictions on its fitted training data are not substituted.
Values below are rounded; generated CSV/JSON retain numerical precision.

| Quantity | Selected propagation Ridge, alpha 10 |
| --- | ---: |
| ETA MAE | 5.327633 h |
| 95th percentile absolute ETA error | 19.008718 h |
| Maximum absolute ETA error | 37.406903 h |
| Smaller absolute error than carry-forward | 77 of 120 |
| Larger absolute error than carry-forward | 43 of 120 |
| Tied absolute error | 0 of 120 |
| Mean absolute-error improvement over carry-forward | 1.817829 h |
| Median paired absolute-error improvement | 1.661896 h |

The 95th percentile describes errors in this validation sample. It is not a
95% confidence or prediction interval for a future shipment. The model improves
the aggregate metric while still making large individual errors.

| Route | Validation n | Selected ETA MAE (h) |
| --- | ---: | ---: |
| R01 | 17 | 3.700 |
| R02 | 31 | 4.006 |
| R03 | 19 | 5.804 |
| R04 | 26 | 6.405 |
| R05 | 27 | 6.497 |

The selected predictor has lower validation MAE than carry-forward on each of
these five route slices. This is descriptive evidence from small samples, not
proof that the improvement generalizes to each real-world route.

| Later disruption flags, used only for diagnosis | n | ETA MAE (h) | ETA bias (h) |
| --- | ---: | ---: | ---: |
| Neither port nor inland disruption | 102 | 3.752 | +2.634 |
| Port disruption only | 11 | 20.090 | -20.090 |
| Inland disruption only | 6 | 2.211 | -1.615 |
| Both | 1 | 22.308 | -22.308 |

The port-only group shows substantial underprediction. The generator includes
future random processing disruptions not observed at the port-arrival
checkpoint. These findings are consistent with that information limit, but
the slice comparison is not a causal attribution of each error. Small groups
especially limit conclusions; the low inland-only MAE should not be treated
as evidence that inland disruptions are easy to predict.

## Controlled-response findings

All 237 eligible development rows were used as starting points. Nine modified
scenarios per row produced 2,133 response rows. There are no actual labels for
these modified scenarios and no accuracy metrics are calculated for them.

- Congestion sweep: 948 adjacent-level comparisons; zero decreases in served
  remaining time as congestion increased over the five declared levels.
- Inland plan +6 h, with deadline also +6 h: 237 comparisons; zero decreases
  in remaining time. Mean predicted remaining-time increase was 5.957216 h.
- Port-arrival +1, +6 and +12 h with consistent voyage/as-of changes: 711
  comparisons; zero cases of an earlier predicted final ETA.
- Eight invalid-input cases were rejected. An unexpected acceptance would stop
  the review command rather than be reported as a successful check.
- 584 of the 2,133 modified response rows were outside at least one fitted
  feature range. These are repeated scenarios, not 584 unique shipments.
  Congestion endpoints 0 and 1 account for 474 flags because neither endpoint
  occurred in the fitted training inputs; other shifts add the remaining flags.

Directional checks passing on these finite scenarios does not prove global
monotonicity, calibration, robustness under all conditions, or deployment
readiness. Range flags do not tell us how inaccurate a prediction is. The chart
and report were visually inspected; both label the results as development
validation rather than final holdout performance.

## Decision before final evaluation

Retain the previously selected predictor and its existing fitted artifact for
the planned final evaluation. No hyperparameters, feature definitions, data
generation settings or model coefficients were changed in response to this
review. The observed baseline improvement and directional behavior support
continuing the evaluation; large-error cases remain an explicit limitation.

Do not claim the model can identify an unobserved future random disruption.
Operational improvement would require additional informative observations,
later updates or an appropriately validated uncertainty method; those are
future extensions rather than hidden additions to the current assignment scope.

The final 60-shipment holdout is still unscored. Review the Windows results
before the next stage freezes the final evaluation and executes it. Report
the eventual holdout result honestly, even if worse than validation. Final
reproduction, the required PDF, repository review and submission remain.
