# Stage 7 local verification record

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

## Previously verified Windows checkpoint

The user supplied a Windows stage-6 run with 115 passing tests in 20.32 seconds,
with warnings treated as errors. Training selected propagation Ridge, alpha 10,
with validation ETA MAE 5.327633161963231 hours and the same counts and baseline
comparison as the local run. The saved model used 237 eligible development
shipments. The 60-shipment holdout remained unscored.

## Local stage-7 checks

Linux, Python 3.12.14; NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1,
scikit-learn 1.9.1, joblib 1.6.0, Matplotlib 3.11.2 and pytest 9.2.0.

`python -m pytest -q -W error --tb=short`: **142 passed** locally.
The 27 new cases cover:

- Hand-calculated feature contributions, grouped totals, ETA and delay using
  a controlled model with known parameters, independent of the training result.
- Negative and zero remaining durations, exactly-on-schedule delivery, late
  delivery and a checkpoint already beyond the delivery deadline.
- Every exported term/group reconstructs the saved model's raw prediction.
  Input row order and input data remain unchanged.
- If a fixed baseline is recommended, that rule is served and explained using
  its exact identity instead of silently using Ridge.
- Training-range flags preserve predictions; invalid metadata is rejected.
- Extra future columns, duplicate headers/IDs, missing columns/values, empty
  input tables and naive timestamps are rejected.
- Demo and explicit CSV modes agree, with no model refitting or changes to
  input data, the split artifact or fitted-model artifacts.
- Prediction works with only a checkpoint input file; target and full-journey
  files are not needed. Known holdout IDs are rejected before prediction.
- Changed split/data evidence and existing output directories are rejected.
- The CLI runs from another directory, fails on an existing output directory,
  and requires exactly one input mode.

The update is also tested after overlaying all incremental ZIPs into a clean
project folder. Earlier source modules are byte-identical to stage 6, preserving
the stored model's source integrity checks. No model retraining is required.

## Observed demo result

The demo selected the first five saved training IDs without inspecting outputs:
SHP-000001, SHP-000002, SHP-000004, SHP-000007, SHP-000003. This order follows
the chronological checkpoint order, not numerical ID order.

| Check | Local result |
| --- | --- |
| Predictor | ridge_propagation_alpha_10 |
| Prediction count | 5 |
| Holdout IDs predicted | 0 |
| Model refitted | false |
| Performance evaluated | false |
| Holdout scored | false |
| Remaining-time clipping count | 0 |
| Shipments outside an individual training feature range | 0 |
| Maximum explanation reconstruction error | 0.0 hours |

For SHP-000001, the local prediction is:

- Remaining time: 26.708018804624636 hours.
- Final ETA: 2025-01-04T23:12:16.687365574+00:00.
- Signed schedule deviation: -4.795364620673889 hours.
- Delay magnitude: 0.0 hours, because the ETA precedes the scheduled deadline.

The approximately 40.825-hour reference prediction plus contributions of
-7.212 hours (port processing/congestion), -5.714 hours (inland plan),
-1.455 hours (dispatch timetable) and +0.264 hours (upstream arrival deviation)
reconstructs approximately 26.708 remaining hours. Display values are rounded;
the exported JSON/CSV retain precision.

The Markdown report and rendered PNG were inspected. Labels distinguish model
contributions from physical durations and causal effects. All five examples
appear in the readable report; all nine feature terms per example are exported.

These are training-record demonstrations, not new accuracy results. No holdout
predictions or outcome comparisons were produced. Small last-digit floating
differences on Windows can occur; explanation tolerances are specified in
`EXPLANATION_CONTRACT.md`. Windows stage-7 execution is still pending.

## Remaining assignment work

Robustness and slice analysis, frozen final evaluation, clean-environment
reproduction, final documentation, the 2-3-page PDF and GitHub submission remain.
No real-world reliability or submission ranking is established by these tests.
