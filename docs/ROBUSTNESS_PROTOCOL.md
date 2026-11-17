# Stage 8 development robustness review protocol

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

Written before running the stage-8 review. The dataset, feature sets, model
selection, fitted pipeline and split IDs remain unchanged. The final holdout
is not predicted or scored. No ranking or performance guarantee is inferred.

## Validation error review

Use only the existing out-of-fold validation predictions: 120 distinct shipments
per predictor. Recheck artifact hashes, dataset/split bindings, fold membership,
ETA/delay reconstruction and errors against the validation outcomes. Do not
replace these predictions with the final model's fitted-data predictions.

Review the saved validation recommendation and both fixed baselines. Report
ETA MAE, RMSE, signed bias, delay MAE, clipping count, and the 95th percentile
and maximum absolute ETA error. These are selected validation results, not
final generalization estimates.

Define slices before examining their scores:

- Route: each observed route ID.
- Port congestion: low [0, 0.4), medium [0.4, 0.7), high [0.7, 1].
- Port arrival deviation: early/on time <=0 h, late (0, 12] h, very late >12 h.
- Future disruption: none, port only, inland only, both. These simulator flags
  are used **only after outcomes for error diagnosis**, never as model inputs,
  feature construction, example selection for training or operational predictors.

Show counts for every slice, including empty slices. Flag n<10 as small sample;
10 is a display heuristic, not a statistical sufficiency guarantee. No
significance tests or confidence intervals are claimed. Slices overlap across
dimensions and must not be added together.

Export the ten largest absolute ETA errors of the selected predictor, chosen
explicitly for error diagnosis. Compare its per-shipment absolute errors with
the better fixed baseline, counting better/worse/tied predictions with 1e-8 h
tolerance. Report mean improvement without calling it an accuracy percentage.

## Controlled input changes

Use the 237 eligible development checkpoint inputs and the already saved final
predictor. No outcomes are compared. These are synthetic conditional-response
checks, not additional training, causal experiments or performance evaluation.
All modified inputs must pass the existing input schema. Features, including
interactions and dispatch phase, are rebuilt from the modified raw inputs.

1. Set congestion to 0, 0.25, 0.5, 0.75 and 1 for every development row. Count
   decreases in served remaining time between adjacent levels. All other raw
   inputs remain fixed; the congestion-processing interaction is rebuilt.
2. Add 6 h to planned inland duration and scheduled final delivery together.
   Count decreases in served remaining time. Do not expect delay magnitude to
   rise when the deadline also moves.
3. Move actual port arrival and as-of forward by 1, 6 and 12 h, and increase
   actual voyage duration by the same amount. Scheduled events and actual
   departure stay fixed. Count cases where the predicted final ETA moves
   earlier despite the later port arrival. Remaining time can legitimately
   decrease when an initial waiting period shrinks; the check uses final ETA.

The simulator's mechanisms motivate these directional checks with other
disturbances held fixed. Ridge has no monotonicity constraint, so observed
violations are documented findings, not suppressed or treated as code-test
success. No finite sweep proves global monotonicity. A fixed baseline may show
zero response to congestion, which reflects its limited information use.

Flags identify engineered features outside the final fitted training min/max.
They do not detect all unusual joint combinations, imply confidence, or ensure
that perturbed conditions are typical. No counterfactual actual delivery labels
are invented for modified rows.

## Invalid-input checks and artifacts

Using one development checkpoint, check rejection of missing congestion,
infinite congestion, out-of-bounds congestion, a future-outcome column, a
negative inland duration, duplicate ID, a naive timestamp and inconsistent
voyage duration. Acceptance of any of these is a contract failure and stops the
review command. Expected direction violations in valid-input sweeps are instead
reported as model behavior requiring review.

Save error slices, validation errors, largest errors, stress responses,
invalid-input check outcomes, summary, readable report, route comparison plot
and provenance. Refuse existing output directories. Preserve all earlier data
and artifacts. Review findings before freezing the final evaluation step.

## References

- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/partial_dependence.html

These motivate keeping final test data separate and interpreting feature
perturbations cautiously. The particular slices and stress checks are project
choices; no assumption of independent engineered features is made.
