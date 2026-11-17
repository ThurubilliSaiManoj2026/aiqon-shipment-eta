# Features and evaluation contract - fixed in stage 5

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

## Data accepted from stage 4

The user supplied a Windows run with 48 passing tests and a validated 300-row
dataset, matching the structural audit: five routes with 60 rows each, no
missing values, 237 label-available development shipments, three exclusions,
and 60 reserved holdout shipments. No generator parameter has been changed.

The preparation command verifies every dataset-file hash against the original
manifest, validates the complete journeys, and compares the input/target files
with their projections from the full table. It never regenerates or edits data.
Hashes detect changes relative to the manifest; they are not authentication of
an untrusted manifest. This workflow uses the project's own generated data.

## Feature groups fixed before model fitting

Each row is processed independently. No scaler, imputer, encoder, route average
or other learned preprocessing is fitted by feature construction. The stage-6
pipeline fits its scaler only within each fold's training partition.

| Feature | Base | Propagation | Definition and reason |
| --- | --- | --- | --- |
| planned_port_processing_h | Yes | Yes | Scheduled handling/customs duration; route-related workload |
| planned_inland_h | Yes | Yes | Scheduled downstream travel duration |
| port_congestion_index | Yes | Yes | Current observed congestion |
| port_arrival_deviation_h | No | Yes | Actual minus scheduled arrival; accumulated upstream lateness |
| hours_until_first_dispatch | No | Yes | max(0, scheduled cargo-readiness/first dispatch minus current arrival); time until the earliest booked slot |
| dispatch_interval_h | No | Yes | Known gap between dispatch opportunities |
| congestion_processing_h | No | Yes | Congestion multiplied by scheduled processing duration; a workload interaction |
| dispatch_phase_sin | No | Yes | sin(2*pi*arrival phase in the known dispatch cycle) |
| dispatch_phase_cos | No | Yes | cos(2*pi*arrival phase in the known dispatch cycle) |

Arrival phase is ((actual port arrival - scheduled first dispatch) modulo
dispatch interval) divided by the dispatch interval. Integer nanoseconds are
used for modulo. Sine and cosine represent the cycle without a discontinuity at
the end/start of a period. The actual future readiness or dispatch time is
never used in this calculation. These phase terms are hypotheses, not a claim
that a fitted model will perfectly reproduce dispatch mechanics.

Route ID is retained for audit and grouping, not fed to the model alongside
route-constant duration attributes. IDs and absolute timestamps are not feature
columns. The simulator has no seasonal mechanism, so calendar features are not
introduced. Departure and voyage delay are not separately added to cumulative
port lateness. The delivery buffer determines the scheduled deadline and delay
calculation; it does not itself alter actual remaining transit in this simulator,
so it is not a separate remaining-time model feature.

Candidate groups have 3 and 9 numeric features. Development-only matrix-rank
diagnostics check for exact linear redundancy; full rank does not establish
good conditioning, predictive value, or causal interpretability. Feature benefit
is measured in stage 6 on identical validation folds.

## Frozen splits and availability

`config/evaluation.toml` specifies the design before inspecting baseline scores:
60 holdout checkpoints, three validation windows of 40 shipments, minimum 80
eligible training rows per fold. Sorting is by port-arrival checkpoint and then
shipment ID. A partition boundary that splits identical checkpoint timestamps
is rejected rather than silently creating a temporal ambiguity.

The final holdout comprises the latest 60 checkpoint timestamps. Earlier
shipments can enter development only if their actual delivery is strictly before
the first holdout checkpoint. Equality is excluded. For each validation fold,
training is an expanding prefix filtered again by actual delivery strictly
before the validation window's first checkpoint.

| Partition | Training/development | Validation/holdout | Unavailable earlier labels |
| --- | ---: | ---: | ---: |
| Final development/holdout | 237 | 60 | 3 |
| Validation fold 1 | 114 | 40 | 3 |
| Validation fold 2 | 153 | 40 | 4 |
| Validation fold 3 | 194 | 40 | 3 |

Validation sets are disjoint, totaling 120 shipments. A shipment validated in
an earlier window may enter a later training set once its label is available;
this is the intended expanding-window design. Training labels near each cutoff
are excluded, even when the shipment's departure or port arrival was earlier.

The JSON split manifest stores exact shipment IDs, cutoffs, exclusions,
evaluation settings, feature-group names and dataset hashes. Preserve this
manifest through model selection and final evaluation. Preparation refuses an
existing output directory. A deliberate repeat uses a separate destination.

Time-aware evaluation is a conservative deployment-style choice, not a claim
that the simulator contains time trends. Sample windows have equal shipment
counts, not necessarily equal calendar duration. A standard time splitter
alone would not enforce our additional delivery-label-availability constraint.

## Two fixed baselines

Let A1 be actual port arrival, S1 scheduled port arrival and S3 scheduled final
delivery. Both predictions are clipped so ETA cannot precede the checkpoint.

1. Scheduled: ETA = max(A1, S3).
2. Carry forward: ETA = max(A1, S3 + (A1-S1)).

The second rule transfers signed arrival deviation, including early arrival.
Its unadjusted remaining time algebraically equals S3-S1, or planned processing
plus inland duration plus delivery buffer. This is tested explicitly. It ignores
new congestion-driven processing, unforeseen disruptions and dispatch-slot effects.

Negative raw remaining-time predictions are adjusted to zero; raw prediction,
adjustment magnitude and clipping flag are retained. Metrics use the served
ETA, not an impossible raw timestamp. Delay magnitude is max(0, ETA-S3).

## Metrics and alignment

- Join actuals to predictions by unique shipment ID. Reject missing, duplicate
  or mismatched IDs and inconsistent target/timestamp values.
- ETA error = predicted ETA minus actual delivery in hours.
- MAE = mean absolute ETA error; lower is better.
- RMSE = square root of mean squared ETA error; emphasizes large misses.
- Bias = mean signed ETA error; positive means predicting later than reality.
- Delay MAE compares nonnegative predicted and actual delay magnitudes.
- Report clipping count and sample count.

Pooled metrics are computed from all 120 validation errors per baseline, not by
averaging fold RMSEs. Per-fold results and per-shipment predictions are exported.
The final 60-shipment holdout has no predictions or scores in this stage.
Reading label timestamps for validation/split integrity is not model training.

## Local validation baseline results

| Rule | Validation n | ETA MAE (h) | ETA RMSE (h) | ETA bias (h) | Delay MAE (h) | Clipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scheduled | 120 | 10.656094 | 15.098328 | -7.294930 | 8.975512 | 1 |
| Carry forward | 120 | 7.145462 | 9.808328 | 1.005674 | 5.603137 | 0 |

These are measured validation results on synthetic data. The stronger rule
provides the fixed reference for learned models. These baseline scores were
recorded before fitting any learned model. Stage-6 candidate results are in
`STAGE6_VALIDATION.md`; final holdout performance remains unknown.

## References

- Scikit-learn: https://scikit-learn.org/stable/common_pitfalls.html
- Scikit-learn: https://scikit-learn.org/stable/modules/cross_validation.html

These support training-only preprocessing and time-aware evaluation. The exact
features, label-availability filter and synthetic assumptions are project design
choices, not external requirements or claims of empirically optimal practice.
