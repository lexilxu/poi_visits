# Frozen analysis protocol

Frozen before computing the new stratum or asymmetric-loss results, 2026-11-16.

## Inputs and scope

Reuse the complete-horizon T=14, H=7 Appendix A experiment: training labels October 1--November 19, validation November 20--December 3, December 4--31 test and frozen January 1--31 test. Use the archived selected learning rates and all five seeds 42--46. Deterministic historical average and AR-ridge each have one prediction archive. Evaluate saved nonnegative predictions, without training, inference, recalibration or model selection. Preserve all source files. Run numerical analysis on AutoDL4 CPU.

Display original GraphGRU prediction archives under the manuscript name EvolveGCN. Include historical average, AR-ridge, Shared LSTM, Matched LSTM, MTGNN (NAICS-2), and GraphGRU (Identity, NAICS-2, NAICS-4). The two LSTM specifications are existing Appendix A comparisons.

## Forecast-time strata

For each forecast origin, use the mean of the preceding 14 daily input values. Missing histories retain the zero placeholders used by the fitted models. Neighbor activity is the unweighted mean history across other POIs in the same NAICS group, excluding the target. Evaluate the common set with at least one non-self neighbor under both NAICS-2 and NAICS-4. Fix target and graph-specific neighbor thresholds at the medians pooled over that set and the 30 complete training origins (indices 14 through 43). Ties enter Low. Apply the same thresholds to December and January. All seven leads inherit their origin-specific group.

Define two separate partitions using NAICS-2 and NAICS-4 neighbor histories. Within each partition and cell, evaluate Identity, NAICS-2 and NAICS-4 on exactly the same valid events. Report event and POI counts, mean absolute error (MAE), positive-visit recall, false activation rate (FAR), and demand magnitude recovery (DMR). Activation is prediction >= 0.5; positive observed visits means y>0. DMR is predicted volume on positive events divided by observed volume on those events. Positive MAE gain means Identity MAE minus graph MAE. Report all seeds, mean and sample standard deviation. These are forecast-time observable conditional comparisons, not an independently deployed adjacency-selection policy.

## Asymmetric loss

For lambda in {0.5,1,2,3,5}, compute mean[lambda*max(y-p,0)+max(p-y,0)] on identical events. Evaluate All POIs and the existing training-defined Low-history cohort: positive training mean <= the 25th percentile among positive training means and observed-zero share >= 0.7, using the first 50 days. Report underprediction and overprediction components, per-seed loss, seed mean and standard deviation, and rankings. At lambda=1 require agreement with independently calculated MAE. Report the paired MTGNN-minus-Identity loss difference. Treat weights as forecasting preference scenarios for observed visits.

## Uncertainty and validation

Average event losses across the fixed five seeds, not predictions. Use 2,000 paired bootstrap replicates combining circular blocks of seven consecutive origins and resampled 3 km spatial cells (Seattle latitude 47.6 degrees; longitude factor 111.320*cos(47.6), latitude factor 110.574). Retain all seven leads together. Report pointwise 95% intervals conditional on these seeds; keep per-seed variation separately. Use random seed 421.

Before analysis, verify input SHA256, seed sets, POI IDs, dates, horizon, valid masks and observed values against the approved panel. Require 22 December and 25 January origins. Reproduce all eight rounded Appendix A.1 MAE means and standard deviations. Verify each partition exhausts its cohort, pooled cell MAE reconstructs overall MAE, same-model contrasts are zero, and asymmetric loss at lambda=1 equals MAE. Hash inputs again at completion. Write versioned aggregate output and an audit manifest; keep all event-level data private on AutoDL4.
