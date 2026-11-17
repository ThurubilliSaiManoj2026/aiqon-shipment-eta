# Interview preparation

Explain these decisions in your own words and demonstrate their evidence.

**What is predicted?** Remaining hours at actual destination-port arrival.
ETA is that checkpoint plus nonnegative remaining time; delay is max(0, ETA
minus scheduled delivery). Signed deviation preserves early deliveries.
Magnitude satisfies the assignment; no calibrated probability is claimed.

**How do delays propagate?** Congestion changes processing duration, readiness
can miss a dispatch slot, and schedule buffers can absorb upstream lateness.
Adding cumulative stage deviations would double-count upstream effects.
Use the hand-calculated examples in DATA_SPECIFICATION.md.

**How is leakage prevented?** Explicit input allowlist; future processing,
readiness, dispatch, delivery and disruption flags excluded. Scaling fits only
within each training fold. Delivery labels must be available strictly before
the next validation/holdout cutoff; sorting by arrival alone is insufficient.

**Why 237 training rows?** The latest 60 checkpoints are held out. Three of
the earlier 240 shipments had not delivered by the holdout cutoff, so their
labels would be unavailable. Fold training counts are 114/153/194, each followed
by 40 validation shipments. Validation windows are disjoint.

**Why Ridge?** A small regularized model supports exact explanation arithmetic.
Scaling makes penalties comparable across features. The eight candidates cover
two feature groups and four alphas. This is a focused comparison, not proof of
optimality; dispatch discontinuities and rare shocks limit a linear model.
Squared-error fitting differs from MAE-based model selection.

**Did propagation features help?** Best base validation MAE was 5.392874 h;
propagation was 5.327633 h, a small 0.065241 h (3.91 minute) difference.
The base model was just outside the predeclared 0.05 h simplicity tolerance.
This does not establish a stable or causal benefit. Final holdout compares the
selected candidate with fixed baselines, not every candidate again.

**Why is carry-forward competitive?** Its remaining duration algebraically
equals planned processing + inland time + delivery buffer: a strong schedule
prior for this mechanism. Holdout MAE is 7.063 h versus Ridge's 7.123 h.
Complexity did not earn a demonstrated advantage over that baseline.

**Why keep Ridge after holdout?** It was selected before scoring. Switching
the evaluated artifact based on holdout would turn test data into selection
data. Keep all results; later redesign needs a separately declared experiment
and new independent data. Replaying the same seed is not a fresh test.

**What do contributions mean?** Each term is coefficient * (feature - training
mean) / training scale. Intercept plus terms equals raw remaining time. The
reference can be physically unrealizable. Terms are not causal stage delays;
ETA offset and both floors are explicit. This decomposition is not called SHAP.

**Where does it fail?** Future random disruptions are unavailable at prediction.
Holdout P95 absolute error is 28.755 h and maximum is 31.664 h. The first held-out
example underpredicts by 19.373 h despite exact explanation reconstruction.
Outcome-defined slices are small and descriptive, never prediction inputs.

**What does the uncertainty interval mean?** The 5,000 paired resamples give
[-1.131, 1.014] h for baseline minus Ridge MAE. Spanning zero establishes neither
superiority nor equivalence. It assumes independently resampleable shipments,
excludes refitting/selection variability and is not a per-shipment ETA interval.

**What next with real data?** Check knowledge-time versus event-time availability,
reporting lag, missingness, route coverage and shared capacity. Establish external
validation before choosing nonlinear, robust or quantile methods. Investigate
business error costs and whether repeated checkpoint updates are needed; do not
promise gains before measuring them. Deployment and monitoring remain future work.

For a short walkthrough, show target definitions, the input allowlist, split
policy, Pipeline, an explanation with its actual error, final comparison and
reproduction evidence. Keep any optional recorded demo within five minutes.
