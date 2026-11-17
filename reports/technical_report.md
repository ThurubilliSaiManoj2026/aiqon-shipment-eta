# Shipment ETA and Delay Propagation

Shipment ETA and Delay Propagation

Sai Manoj | Aiqon technical assignment | Destination-port arrival checkpoint

<b>Purpose.</b> Support downstream delivery planning by estimating final ETA and delay magnitude from information available when a shipment reaches its destination port. The implementation combines a synthetic journey simulator, chronological evaluation and exact explanations of a small regression model.

## 1. Prediction contract

Let A₁ be actual port arrival, A₃ actual delivery and S₃ scheduled delivery. The target r is remaining transit time; r<sub rise="2.2" size="7.2">raw</sub> is the model output. All durations are hours and timestamps are UTC.

r = A₃ - A₁<br/>ETA = A₁ + max(0, r<sub rise="2.2" size="7.2">raw</sub>)<br/>Signed deviation = ETA - S₃;  delay magnitude = max(0, ETA - S₃)

A 20-column input allowlist admits schedules, route attributes, observed departure/arrival, completed-voyage weather and current congestion. Actual future readiness, dispatch, delivery, processing durations and disruption flags are excluded. Negative raw remaining time is clipped and flagged; no numerical delay probability is claimed.

## 2. Synthetic journeys and ground truth

The dataset contains 300 shipments on five fictional routes, 60 per route. Planned voyage V, port processing P, inland travel L, buffer B and dispatch interval H define S₁ = S₀ + V, S₂ = S₁ + P and S₃ = S₂ + L + B. Handling and customs form one aggregate readiness stage.

| Stage | Implemented mechanism |
| --- | --- |
| Departure | A₀ = S₀ + max(-2, N(2, 4)) + Bern(0.10) × U(12, 36). |
| Voyage | W ~ Beta(2, 5); duration = max(0.75V, V + 8W + N(0, 0.04V)); A₁ = A₀ + duration. |
| Port / customs | C ~ Beta(2, 3); processing = max(1, P(0.75 + 1.10C) + N(0, 2) + Bern(0.10) × U(12, 36)); A₂ = A₁ + processing. |
| Dispatch / inland | Dispatch at the first S₂ + kH ≥ A₂, integer k ≥ 0. Inland duration = max(1, L + N(0, 0.15L) + Bern(0.08) × U(4, 12)); A₃ = dispatch + inland duration. |

N(mean, standard deviation), U(uniform) and Bern(Bernoulli) use independent draws unless linked above. Seed 42, SeedSequence child 0, PCG64 and a fixed draw order make generation reproducible with the recorded versions. Integer-nanosecond dispatch arithmetic includes exact-slot readiness. These parameters are assumptions, not measured port statistics. Congestion has no reporting lag; shipments do not share capacity.

## 3. Delay propagation and recovery

For S₁ = 0, P = 12, L = 8, B = 8 and H = 6, the deadline is S₃ = 28. Arrival at A₁ = 6 with 8 h processing gives dispatch at 18 and delivery at 26: 2 h early. With 20 h processing, dispatch is 30 and delivery 38: 10 h late. Buffers absorb lateness; missed slots amplify it without double-counting upstream delays.

The generated audit contains 89 early and 211 late deliveries; 53 late port arrivals recover by delivery and 110 amplify final lateness. Full equations, parameters and schemas are in docs/DATA_SPECIFICATION.md and config/simulation.toml.


---

Model and evaluation design

Time-aware selection | Explicit label availability | Reconstructable explanations

## 4. Features and model choice

The base model uses planned processing, planned inland duration and observed congestion. The nine-feature propagation model adds signed port-arrival deviation, time until first dispatch, dispatch interval, congestion × planned processing, and sine/cosine dispatch phase. Route IDs are audit labels; absolute dates and future outcomes are not features.

Each scikit-learn pipeline applies deterministic features, training-only StandardScaler and Ridge(SVD) [1, 2]. Eight candidates span two feature groups and α values {0.1, 1, 10, 100}. Selection minimizes pooled validation ETA MAE; within 0.05 h of the minimum, it prefers fewer features, then larger α. Ridge is recommended only if it beats the best validation baseline by more than 0.05 h.

Propagation Ridge (α = 10) was selected. Base versus propagation MAE was 5.393 versus 5.328 h: a small 0.065 h (3.91 min) gain. Ridge limits complexity and supports exact explanations, but cannot fully represent dispatch discontinuities. Squared-error fitting and MAE selection optimize different quantities.

## 5. Chronological evaluation

| Partition | Training | Validation / holdout | Unavailable labels |
| --- | --- | --- | --- |
| Validation 1 | 114 | 40 | 3 |
| Validation 2 | 153 | 40 | 4 |
| Validation 3 | 194 | 40 | 3 |
| Final fit / holdout | 237 | 60 | 3 |

Records are ordered by port-arrival checkpoint. Training requires delivery strictly before the next window starts; three of the 240 earlier shipments lack available labels at the final cutoff. The three validation windows contain 120 distinct records. The selected model was fitted on 237 shipments, then held fixed across 60 holdout records. Data, IDs, model, source and metric definitions were frozen before scoring.

## 6. Per-shipment explanation

For feature x<sub rise="2.2" size="7.2">j</sub>, fitted training mean μ<sub rise="2.2" size="7.2">j</sub>, scale s<sub rise="2.2" size="7.2">j</sub> and standardized coefficient β<sub rise="2.2" size="7.2">j</sub>, the contribution is c<sub rise="2.2" size="7.2">j</sub> = β<sub rise="2.2" size="7.2">j</sub>(x<sub rise="2.2" size="7.2">j</sub> - μ<sub rise="2.2" size="7.2">j</sub>)/s<sub rise="2.2" size="7.2">j</sub>. The intercept plus Σ<sub rise="2.2" size="7.2">j</sub>c<sub rise="2.2" size="7.2">j</sub> equals raw remaining time. This training-mean reference need not describe a feasible shipment; contributions explain model arithmetic, not causal stage delays [3].

| First holdout checkpoint: SHP-000241 | Contribution / hours |
| --- | --- |
| Training-mean reference | 40.825 |
| Port processing and congestion | +10.497 |
| Inland travel plan | +2.229 |
| Dispatch timetable | -1.585 |
| Upstream arrival deviation | +0.066 |
| Predicted remaining time; no duration clipping | 52.032 |

Port processing/congestion is the largest positive group relative to the reference. The deadline is 47.188 h after arrival; predicted delay is 4.845 h and ETA is 9 May 2025, 08:50:40 UTC. Actual remaining time is 71.406 h: ETA error is -19.373 h. This earliest holdout record was not chosen for accuracy. Display values are rounded.

Duration clipping, deadline offset and delay flooring are reported separately. Maximum holdout reconstruction error was 7.11e-15 h; exact reconstruction is numerical consistency, not prediction accuracy.


---

Results and engineering assessment

Frozen synthetic holdout | Same 60 shipments for every predictor

## 7. Baselines and measured results

Scheduled ETA = max(A₁, S₃). Carry-forward ETA = max(A₁, S₃ + A₁ - S₁); its remaining time equals planned processing + inland duration + buffer. The comparison baseline was selected on validation. All table values below are hours; lower error is better.

| Predictor | Validation ETA MAE | Holdout ETA MAE | Holdout ETA RMSE | Holdout delay MAE |
| --- | --- | --- | --- | --- |
| Scheduled | 10.656 | 10.679 | 16.367 | 8.782 |
| Carry-forward | 7.145 | 7.063 | 10.102 | 5.685 |
| Selected Ridge | 5.328 | 7.123 | 10.404 | 5.777 |

Validation n = 120; holdout n = 60. With e = predicted ETA - actual delivery, MAE = mean(|e|), RMSE = √mean(e²), and bias = mean(e). Delay MAE compares nonnegative predicted and actual delay. Ridge and carry-forward clipped zero durations; scheduled ETA clipped one.

<b>The validation advantage did not persist on holdout.</b> Ridge MAE is 0.060 h (3.58 min) worse than carry-forward, with smaller absolute error on 29 shipments and larger error on 31. RMSE, delay MAE and tail error are also worse. The preselected model and all comparisons are retained; learned-model superiority is not established.

## 8. Uncertainty and failure analysis

The predeclared paired percentile bootstrap uses 5,000 shipment resamples and PCG64 seed 2026 [4]. Approximate 95% intervals are [5.329, 9.143] h for Ridge MAE and [-1.131, 1.014] h for carry-forward minus Ridge MAE. Spanning zero establishes neither superiority nor equivalence. Intervals condition on the fitted model, assume independently resampleable shipments and exclude training/selection variability; they are not individual ETA intervals.

Ridge bias is -1.379 h, P95 absolute error 28.755 h and maximum error 31.664 h. Port-only future-disruption cases have MAE 20.521 h (n = 7); cases with both port and inland disruptions have 30.207 h (n = 2). These small, outcome-defined slices diagnose errors; their future flags never enter prediction.

Across 2,133 controlled development response rows, no directional decreases occurred in the declared scenarios; this is a behavior check, not accuracy or global monotonicity. All 8 declared invalid-input cases were rejected. 5 holdout shipments had feature-range flags, which are not confidence scores.

Missing inputs are rejected, not imputed. Unseen-route accuracy, shared queues, reporting lags and operational drift are unverified. Independent real data and a new evaluation are needed before deployment or further model selection; the exposed holdout is not an optimization target.

## 9. Verification and reproducibility

All 186 tests passed again with warnings as errors on Linux/Python 3.12.14. The seven CLI stages and explicit CSV prediction completed; dataset hashes and the final summary matched the original run. Independent calculations passed 432 recorded checks. The supplied Windows/Python 3.12.1 clean-environment run also passed 186 tests and reproduced the workflow. These are repeatability checks, not new independent holdout evaluations.

README.md provides setup and execution. reports/FINAL_AUDIT.md records verification scope; reports/evidence retains numerical inputs. The report builder and pinned optional dependencies are in report_tools. No feature, model, simulation or evaluation setting changed after holdout exposure.

References: <link href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html" color="black">1. Ridge</link> | <link href="https://scikit-learn.org/stable/common_pitfalls.html" color="black">2. Leakage prevention</link> | <link href="https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html" color="black">3. Linear coefficients</link> | <link href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html" color="black">4. Bootstrap methods</link>
