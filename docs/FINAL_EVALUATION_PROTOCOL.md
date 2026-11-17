# Stage 9 final evaluation protocol

This protocol is written before the final holdout is predicted or scored.
Stage-8 development review retained the original selected propagation Ridge,
alpha 10. The user verified that review on Windows with 167 passing tests and
matching findings. No model choice is made using the final holdout.

## Freeze before scoring

1. Verify the dataset, stage-5 split artifacts, stage-6 model and stage-8 review.
2. Record their hashes, selected predictor, training/holdout IDs, feature
   schema, comparison baseline, source/script hashes and this protocol's settings.
3. Save this snapshot in a new `artifacts/stage9_freeze/` directory without
   predicting any holdout records. The final evaluator requires this snapshot
   and rejects changes to the recorded evidence or executable source.

The main dataset remains the original 300 shipments generated with seed 42.
Use the existing model trained on 237 shipments whose delivery labels were
available before the holdout cutoff. Keep the three unavailable-label earlier
shipments excluded. Evaluate exactly the 60 frozen holdout IDs, in saved order.
Do not refit during evaluation or refit on all 300 records afterwards and
silently describe that different artifact as the evaluated model.

Unit/integration tests of the new final-evaluation workflow use a separate
synthetic fixture with seed 314159, not the main dataset's held-out outcomes.
Existing development tests continue to exercise their original fixtures.

## Predictors and metrics

Evaluate the saved validation recommendation and both fixed baselines:
scheduled delivery and carry-forward. If the recommendation is itself a
baseline, report unique predictors without duplication. The comparison baseline
is the better baseline selected from validation, not whichever happens to win
on holdout. For the main dataset this is carry-forward.

Primary metric: served ETA MAE in hours. Also report ETA RMSE, signed ETA bias,
delay-magnitude MAE, clipping count, 95th percentile and maximum absolute ETA
error. Keep prediction/actual pairs by shipment ID for independent checking.
The nonnegative remaining-time and delay rules remain unchanged.

Report descriptive route, congestion, arrival-deviation and future-disruption
slices using the boundaries already defined in stage 8. Flag n<10 as small
sample. Future disruption labels are diagnostic only; all predictions use the
strict checkpoint-input schema. Show the ten largest errors, even if unfavorable.
Do not use these diagnostic results for further feature/model selection.

## Conditional resampling uncertainty

Use 5,000 paired shipment bootstrap resamples, NumPy PCG64 seed 2026. Each
resample draws 60 positions with replacement for the main holdout. Apply the
same positions to selected-model and comparison-baseline absolute ETA errors.
Use the 2.5th and 97.5th percentiles (linear quantile interpolation) for:

- The selected model's mean absolute ETA error.
- Mean paired improvement: baseline absolute error minus selected-model
  absolute error. Positive favors the selected model; negative favors baseline.

These are approximate 95% percentile-bootstrap intervals, conditional on the
fitted model and observed synthetic sample, assuming independently resampleable
shipments. They do not account for retraining/model-selection variability,
unobserved rare regimes, real-world shared shocks or dependency between shipments.
They are not individual ETA intervals, numerical delay probabilities or promises
of exact statistical coverage. With 60 shipments, uncertainty can be substantial.
Report the interval even if it spans zero or favors the baseline. No p-value or
new hypothesis-selection process is introduced.

## Outputs and reporting discipline

Save full predictions/errors, all metrics, paired comparison, bootstrap
intervals, slices, worst errors, selected-model explanations, a readable report,
summary, comparison/scatter chart and provenance in `artifacts/stage9_final/`.
The freeze snapshot and earlier artifacts remain unchanged. Existing output
directories are refused. A same-protocol reproduction is verification, not a
new independent test; it must never be used to search for a better seed or model.

If evaluation reveals a correctness bug, disclose it, preserve the original
run, fix the implementation and label any rescoring as a correction after
holdout exposure. Do not call an already examined holdout untouched again.
No acceptance threshold was specified by the company. Report results without
fabricating a pass mark, real-world reliability claim or submission ranking.

## References

- https://scikit-learn.org/stable/modules/cross_validation.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html

The bootstrap is implemented directly with NumPy to make paired indices and
quantile settings explicit; the SciPy documentation describes the percentile
resampling method. These choices add evaluation uncertainty reporting, not a
new prediction target or an operational probability feature.
