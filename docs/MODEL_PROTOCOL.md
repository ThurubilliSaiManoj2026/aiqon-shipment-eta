# Stage 6 model-selection protocol

> Historical stage record: statements about pending work describe that stage.
> For the completed implementation, current results and verified environments,
> see [README](../README.md), [ENVIRONMENT](ENVIRONMENT.md) and
> [REQUIREMENTS](REQUIREMENTS.md). Preserve pre-evaluation decisions as history.

This protocol and config/training.toml were written before running any learned
model comparison. The simulator, dataset, feature definitions and stage-5
partitions remain fixed. No final holdout scores are calculated in this stage.

## Candidates

Eight candidates: Ridge regression with alpha in {0.1, 1, 10, 100}, each using
either the three base features or the nine propagation features. This compact
grid is appropriate to the available sample size and supports an interpretable
comparison. No broader search or nonlinear model is included in this stage.

Each fitted object is a scikit-learn Pipeline containing the deterministic
checkpoint feature builder, StandardScaler, and Ridge(solver="svd"). The
scaler and regression coefficients are fitted separately on each fold's
label-available training records. The prediction target is remaining hours.
Feature scales matter to Ridge's coefficient penalty; using a scaler inside
the pipeline keeps that preprocessing local to each training fold.

## Selection rule

1. Evaluate each candidate on the same three saved validation windows.
2. Compute pooled ETA MAE over the same 120 shipments, after the documented
   nonnegative remaining-time adjustment. Report all candidate and fold scores.
3. Find the smallest candidate MAE. Among candidates no more than 0.05 hours
   (three minutes) above it, prefer fewer features, then a larger alpha, then
   candidate ID as a deterministic final tie-break.
4. This tolerance is an engineering tie-break, not a confidence interval or
   significance test. Report the selected candidate's gap to the absolute minimum.
5. Compare that selected Ridge with the better fixed baseline. Recommend Ridge
   only if its validation MAE improves on the baseline by more than 0.05 hours;
   otherwise retain the baseline as the validation recommendation while keeping
   the selected Ridge artifact for transparent inspection.
6. Refit the selected Ridge on all 237 eligible development shipments. Save
   its complete pipeline, training IDs, feature metadata, coefficient table,
   package versions and provenance. Confirm predictions survive serialization.

The model-selection scores are optimistically selected validation scores, not
unbiased generalization estimates. The final 60-shipment holdout remains for a
later assessment after implementation and selection decisions are frozen.

## Feature comparison

Report the base and propagation scores at each identical alpha. Also report
each group's best validation score. Neither comparison establishes causality.
Additional features are useful only if their observed predictive benefit and
interpretability justify them. Do not add new feature terms to chase the
reported validation score during this stage.

## Verification and exclusions

Verify the saved stage-5 artifacts and dataset hashes before training. Verify
stored split IDs against the declared chronological/label-availability policy.
No holdout IDs may occur in fit records or validation predictions. Retain both
baseline and candidate errors, clipping counts and sample counts. Existing
artifact directories are never overwritten.

Feature coefficients are descriptive fitted parameters; they are not yet the
per-shipment explanation feature requested by the assignment. Local explanations,
robustness analysis, final holdout evaluation and the PDF report are later work.

## Technical references

- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- https://scikit-learn.org/stable/model_persistence.html

The alpha grid, tolerance and feature groups are explicit project choices, not
requirements or empirically optimal settings supplied by Aiqon.
