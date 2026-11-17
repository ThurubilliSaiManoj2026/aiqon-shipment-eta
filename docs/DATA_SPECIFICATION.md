# Synthetic shipment specification - implemented version 1

## Purpose and scope

Create 300 distinct shipment journeys with five fictional routes (60 shipments
per route). Predict final delivery ETA and delay magnitude immediately after
actual destination-port arrival. Handling and customs form one aggregate
cargo-readiness stage. This avoids claiming that they are always sequential.
One row per shipment is used for modeling; multiple snapshots are out of scope.

All values in `config/simulation.toml` are engineering assumptions. They have
not been fitted to real logistics data. The generator implements these rules;
its validation and audit run before any model fitting. The default parameters
have not been changed after observing the first dataset. Any later change must
be documented and must not be tuned to produce favorable held-out model scores.

## Units, clock and reproducibility

- All timestamps are timezone-aware UTC; durations are floating-point hours.
- Keep full precision during computation and serialize timestamps consistently.
- Scheduled departures begin at 2025-01-01 00:00 UTC, one every 12 hours.
- Assign 60 copies of each route, then shuffle them with a dedicated seeded RNG.
- Use `SeedSequence(seed).spawn(2)[0]` with `Generator(PCG64(child))`, not global random state.
- Fix and document RNG draw order in the generator. Persist config, package
  versions, seed, and generated data. A seed alone is not a full reproducibility guarantee.
- Use distinct child RNG streams for complete journey generation and later
  missing-observation experiments so missingness does not change ground truth.

Implementation version 1.0.0 spawns two SeedSequence children. Child 0 drives
journeys and child 1 is reserved for a later missing-observation experiment.
The generator shuffles balanced route assignments first, then draws each
shipment in this order: departure normal/flag/magnitude; completed-voyage
weather/voyage noise; congestion/processing noise/flag/magnitude; inland
noise/flag/magnitude. A disruption magnitude is drawn even when its flag is zero.
Missing-observation experiments themselves are not implemented at this stage.

Dispatch selection uses integer nanoseconds rather than floating-point ceil;
this implements the stated boundary convention exactly at timestamp precision.
Derived durations and targets are recomputed from the resulting timestamps.

## Schedule

For each route define planned voyage V, port/readiness processing P, inland
travel L, delivery buffer B, and dispatch spacing H from the TOML file.
Let S0 be planned origin departure, S1 planned port arrival, S2 planned cargo
readiness, and S3 planned final delivery:

    S1 = S0 + V
    S2 = S1 + P
    S3 = S2 + L + B

The planned first dispatch is S2; additional dispatch slots are S2 + k*H for
nonnegative integer k. This is a shipment-specific illustrative dispatch
schedule, not a calibrated port-wide timetable or a shared-capacity simulation.

## Generate actual journeys

Notation: N(mu, sigma) is Normal with standard deviation sigma; U(a,b) is
Uniform; Bern(p) is a zero/one Bernoulli draw. Different noise and disruption
draws are independent unless explicitly linked below. Congestion C and voyage
weather W are dimensionless indicators in [0,1]. Route affects base durations.

1. Departure deviation:

       D0 = max(-2, N(2,4)) + Bern(0.10)*U(12,36)
       A0 = S0 + D0

2. Completed voyage weather and actual voyage duration:

       W ~ Beta(2,5)
       V_actual = max(0.75*V, V + 8*W + N(0,0.04*V))
       A1 = A0 + V_actual

   W represents the completed voyage, available upon arrival; it is not future
   realized weather. The prediction timestamp is exactly A1.

3. Current port congestion and future cargo-readiness processing:

       C ~ Beta(2,3)
       P_actual = max(1, P*(0.75 + 1.10*C) + N(0,2)
                         + Bern(0.10)*U(12,36))
       A2 = A1 + P_actual

   C is assumed measured without reporting lag at A1. The future disturbance,
   P_actual and A2 are unavailable at inference. This idealized congestion
   measurement is a limitation to disclose.

4. Dispatch timing and inland journey:

       k = max(0, ceil((A2 - S2)/H))
       dispatch_actual = S2 + k*H
       L_actual = max(1, L + N(0,0.15*L) + Bern(0.08)*U(4,12))
       A3 = dispatch_actual + L_actual

   Readiness exactly at a slot can catch that slot. Readiness before S2 waits
   for S2. The future inland disturbance and all actual downstream timestamps
   are unavailable at A1. Test numerical precision near slot boundaries.

The TOML file is the parameter source; numeric values above explain version 1.
Keep configuration and documentation synchronized if any parameter changes.

## Propagation and recovery examples

Let S1=0, P=12, L=8, B=8 and H=6 hours, so S2=12 and S3=28.

| Scenario | A1 | Actual processing | A2 | Actual dispatch | Actual inland | A3 | Final signed deviation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream lateness absorbed | 6 | 8 | 14 | 18 | 8 | 26 | -2 |
| Downstream amplification | 6 | 20 | 26 | 30 | 8 | 38 | 10 |

These are hand-calculated illustrations, not sampled records or model results.
They show why six hours of port-arrival lateness need not become exactly six
hours of final delay. No cumulative stage deviations should be added together:
doing so counts upstream lateness more than once.

## Data contract and availability

Create separate logical views: a complete journey table for labels and audit,
and an allowlisted prediction-input table. Never train by taking every column
except the target. The implemented file layout is listed below.

| Fields | Type/unit | Availability and role |
| --- | --- | --- |
| shipment_id | unique string | Identifier; never a model feature |
| route_id, origin, destination_port, destination | categorical strings | Known; avoid redundant encodings unless justified |
| scheduled_departure_utc | UTC timestamp | Known at prediction |
| scheduled_port_arrival_utc | UTC timestamp | Known at prediction |
| scheduled_ready_utc | UTC timestamp | Known at prediction |
| scheduled_delivery_utc | UTC timestamp | Known at prediction |
| planned_voyage_h, planned_port_processing_h, planned_inland_h | positive hours | Known schedule attributes |
| delivery_buffer_h, dispatch_interval_h | hours | Known, buffer >=0 and interval >0 |
| actual_departure_utc | UTC timestamp | Observed before prediction; may be hidden in robustness experiments |
| actual_port_arrival_utc | UTC timestamp | Required checkpoint; as_of_utc equals this |
| voyage_weather_index | number in [0,1] | Completed-voyage observation, not future weather |
| port_congestion_index | number in [0,1] | Current observation at the checkpoint |
| actual_voyage_h | hours | Derived from observed departure and arrival |
| actual_port_processing_h, actual_ready_utc | hours / UTC | Future outcomes; excluded from prediction inputs |
| actual_dispatch_utc, dispatch_wait_h | UTC / hours | Future outcomes; excluded from prediction inputs |
| actual_inland_h, actual_delivery_utc | hours / UTC | Future outcomes; excluded from prediction inputs |
| future disruption flags and random draws | simulation-only | Audit only, never input features |

The complete base dataset has no missing fields. Missing or inconsistent
checkpoint inputs are rejected; no masking or imputation experiment is included.
Route labels are validated as nonempty strings and are not encoded by the model.
New labels with valid numeric inputs can be accepted, but performance on unseen
routes is unverified. Range flags do not establish route generalization.

## Targets and prediction contract

    remaining_h = (A3 - A1) in hours
    signed_arrival_deviation_h = (A3 - S3) in hours
    delay_magnitude_h = max(0, signed_arrival_deviation_h)

Primary regression target: remaining_h. Predicted ETA is A1 plus predicted
remaining hours; predicted delay magnitude is max(0, predicted ETA minus S3).
Retain signed deviation to distinguish early delivery from exactly on time.

An invalid negative raw remaining-time prediction will be clipped to zero at
the output boundary and flagged. Report the clipping frequency and evaluate
the served predictions. Explanations must disclose that adjustment separately
if it changes the raw model output. Physical consistency is not evidence of
predictive accuracy. No calibrated probability will be claimed.

## Candidate feature groups

1. Schedule and route attributes, including expected remaining duration.
2. Observed departure deviation, port-arrival deviation, and voyage overrun.
3. Schedule allowance remaining at A1: (S3-A1) - P - L.
4. Current congestion and its interaction with planned processing duration.
5. Selected interactions between arrival deviation, buffer and congestion.

Candidate features are hypotheses to evaluate; they are not a commitment to
include every term. Avoid exact linear redundancies and double counting.
Any learned route summaries, imputers, encoders and scalers fit on training only.
No absolute-date feature is justified by default because this generator has no
seasonality. No future schedule updates are supplied retrospectively.

## Evaluation contract

- Freeze a chronological holdout of the latest 60 port-arrival checkpoints.
- At the earliest holdout checkpoint T, training is limited to other shipments
  with actual_delivery_utc < T. Report excluded/ineligible shipments and all
  split sizes. A fixed model is evaluated on the entire holdout window.
- Use expanding chronological folds within eligible training data for model
  choices, honoring label availability at each validation boundary. Confirm
  fold sizes in the data audit before fixing the number of folds.
- Keep the final holdout out of feature/model/hyperparameter selection.
- Baseline 1: ETA=max(A1,S3), updating any schedule already in the past.
- Baseline 2: ETA=max(A1,S3+(A1-S1)), carrying forward signed port lateness.
- Primary metric: final ETA MAE in hours. Also report RMSE, signed prediction
  error (predicted minus actual; positive means pessimistic), delay-magnitude
  MAE, route/severity errors with counts, and any boundary adjustments.
- Compare base and propagation candidates on the same validation folds. The
  final protocol evaluates only the validation-selected predictor and fixed
  baselines on holdout, preserving holdout separation from candidate selection.
  This narrows the earlier exploratory proposal; no unselected base candidate
  holdout result is claimed. Do not select by repeatedly inspecting holdout.
- Use predeclared perturbation scenarios to examine behavior. Simulation
  mechanisms are not guarantees that a learned model is monotone.
- Small per-route holdout samples limit conclusions. Synthetic performance
  is evidence about this simulator, not operational performance at real ports.

## Generator acceptance checks

- Exactly 300 unique IDs; exactly 60 records per route for the default config.
- Valid timezone-aware UTC timestamps and finite numeric values.
- A0 < A1 < A2 <= dispatch_actual < A3; scheduled stages ordered.
- Positive actual stage durations; nonnegative dispatch wait, remaining time
  and delay magnitude. Signed arrival deviation may be negative.
- Indicators inside [0,1]; dispatch aligns with its allowed schedule.
- Target recomputation exactly matches stored labels within serialization tolerance.
- Identical config, seed, versions and draw order reproduce the dataset.
- Different seeds change outcomes. Masking experiments remain later work and
  must leave the complete truth unchanged when implemented.
- Feature allowlist excludes every future field, future random draw and target.
- Hand-checked examples cover recovery, missed dispatch, exact dispatch
  boundary, early arrival, severe delay, and missing/invalid inputs.
- Record distributions, early/on-time/late counts and upper tails. Do not
  promise any counts or errors before actually generating and inspecting data.

## Generated files and current limits

Run `python scripts/generate_data.py` from the project root using the project
interpreter. The command validates in memory, checks each exported CSV round
trip, validates the reloaded inputs and journeys, then publishes a new output
directory. It refuses an existing output directory.

| File | Contents |
| --- | --- |
| journeys.csv | Complete journey, outcomes, three disturbance flags and labels; audit/training-label source only |
| checkpoint_inputs.csv | Strict input allowlist; no future outcomes, labels or disturbance flags |
| targets.csv | Shipment ID, label-availability timestamp (actual delivery), and three targets |
| audit.json | Structural counts, distributions, recovery/amplification examples and split feasibility |
| simulation_config.json | Resolved, validated generation parameters |
| manifest.json | Python/platform/NumPy/pandas versions, source/config/output hashes, RNG recipe, schema columns |

Target-table columns must never be fed to the model. A model input ID is kept
for joins only, and will be excluded from feature construction.
The generation audit previews the 60-checkpoint holdout and label-availability
exclusions; it does not fit models, calculate prediction metrics, or finalize
fold manifests. Stage 5 separately persists these splits, constructs two feature
groups, and scores the two fixed baselines on development validation windows.
See FEATURES_AND_EVALUATION.md for exact definitions and validation results.
Final holdout evaluation is now completed and reproduced; see the technical
report and STAGE9_VALIDATION.md. Historical earlier-stage records retain their
then-unscored status.
There is no service-level on-time tolerance in the brief: audit counts use
strict zero signed deviation and therefore may have no exactly-on-time rows
with continuous durations. No late-probability calibration is attempted.

The implementation rejects missing values and supplies fitted explanations,
strict prediction-input validation and feature-range flags. Optional-input
imputation and unseen-route performance evaluation are not implemented. The
simulator is a set of independent shipments, not a queueing network with
inter-shipment capacity competition.
