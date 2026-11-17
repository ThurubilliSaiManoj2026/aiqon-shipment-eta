# Stage 7 prediction and explanation contract

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

This stage consumes the existing stage-6 fitted artifact. It does not change
features, model selection, coefficients, data generation or frozen partitions.
The saved validation recommendation determines whether Ridge or a fixed rule
is served. The current verified recommendation is propagation Ridge, alpha 10.

## Exact Ridge decomposition

For engineered feature x_j, training mean mu_j, scaler scale s_j and fitted
standardized coefficient beta_j:

    contribution_j = beta_j * (x_j - mu_j) / s_j
    raw_remaining_hours = intercept + sum(contribution_j)

The reference is the vector of fitted training means. Its raw prediction is
the fitted intercept. This reference may not describe a physically realizable
shipment: interactions and sine/cosine coordinates need not match any actual
input at their separate means. It is a declared arithmetic reference, not a
no-congestion scenario or a typical observed shipment.

Positive contributions increase remaining time relative to that reference;
negative contributions decrease it. They are not stage durations, probabilities,
or causal delay estimates. Correlated features and engineered interactions
prevent interpreting one coefficient as an independent intervention effect.
This exact linear decomposition is not described as SHAP.

For readability, disjoint groups sum related feature contributions:

| Group | Features |
| --- | --- |
| Port processing and congestion | planned_port_processing_h, port_congestion_index, congestion_processing_h |
| Inland travel plan | planned_inland_h |
| Upstream arrival deviation | port_arrival_deviation_h |
| Dispatch timetable | hours_until_first_dispatch, dispatch_interval_h, dispatch_phase_sin, dispatch_phase_cos |

Only features present in the selected group are included. All feature terms
are exported; group rankings use absolute signed group totals, without turning
these into percentages. Opposing contributions within a group can cancel.
Grouping improves readability, not causal identifiability.

## ETA and delay conversion

Let A be the port-arrival checkpoint and S the scheduled final delivery:

    remaining_adjustment = max(0, -raw_remaining_hours)
    served_remaining = raw_remaining_hours + remaining_adjustment
    predicted_ETA = A + served_remaining
    deadline_offset_hours = (A - S) in hours
    predicted_signed_deviation = deadline_offset_hours + served_remaining
    predicted_delay_magnitude = max(0, predicted_signed_deviation)

Both nonlinear adjustments are explicit. Feature contributions explain raw
remaining time; they are not relabelled as additive contributions to clipped
delay magnitude. Delay is zero when the ETA is on or before the deadline.
Negative raw duration is clipped to zero, flagged, and never hidden.

The implementation compares the reconstructed raw prediction to pipeline
prediction and verifies the ETA/deviation conversion. Numerical tolerance is
1e-8 hours for arithmetic, with a tiny relative allowance for raw reconstruction.
Full precision is retained in JSON/CSV; Markdown and charts round for display.

If a baseline is the saved recommendation, it is served with its direct rule
identity: scheduled remaining time, plus signed port-arrival deviation for
carry-forward. A fitted Ridge is not silently substituted for that baseline.

## Input and output contract

Inputs are exactly the existing 20 checkpoint columns defined in
`src/shipment_eta/schema.py`. Complete, finite inputs and explicit UTC timestamps
are required. Unknown/missing columns, future outcomes, duplicate shipment IDs,
invalid values and inconsistent timestamps are rejected. There is no imputation.
The input CSV parser additionally rejects duplicated headers before pandas can
rename them. Column order may vary; shipment ID order is preserved.

Features outside the final fitted training min/max are flagged without altering
the prediction. This check is not a confidence estimate or a comprehensive
out-of-distribution detector; joint novelty and unseen operational regimes can
exist inside individual feature ranges. No uncertainty probability is claimed.

The command reads a trusted local model, checks its existing integrity and
environment guards, and verifies the split manifest hash. Inputs containing
any reserved holdout shipment ID are rejected at this stage. This guard protects
the recorded evaluation workflow; renaming IDs is not a valid way to bypass it.

`--demo` chooses the first five saved training IDs before inspecting predictions.
It reads only their checkpoint inputs, not target files. This is a demonstration
on training records, not an accuracy evaluation or a selection of favorable
examples. `--input` accepts a checkpoint-only CSV and uses the same predictor.
The first five results are shown in Markdown; JSON/CSV contain all input rows.

Existing output directories are refused. No dataset, prior artifact or model
is overwritten. Model-version incompatibility is resolved by restoring the
recorded environment or an explicit retraining workflow, not bypassing guards.

## References and limits

- https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html
- https://scikit-learn.org/stable/model_persistence.html

These references support interpretation and persistence cautions. Grouping,
reference choice and this prediction workflow are explicit project decisions.
Accuracy, stress tests, final holdout evaluation and the submission report remain
separate work; explanation arithmetic is not evidence of predictive accuracy.
