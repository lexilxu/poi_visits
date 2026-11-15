# External payment-based parking validation

Protocol fixed after source-integrity/coverage auditing and before comparing
model predictions with parking outcomes. Source audit: 398,027 observations,
1,343 eligible blockfaces and 3,569 blockface/clock-time pairs. A pair requires
15 observed training dates during October 1-November 19, 2020. Coordinates and
capacity summaries use that training period. Observations occur at 10:00,
14:00 and 18:00 local clock time. Missing observations retain their missing status.

## Estimand and periods

The response is paid occupancy divided by recorded parking capacity. Paid
occupancy is inferred from payments. Values above capacity remain valid
measurements of this payment-based response. The study evaluates a monitoring
screen for high payment-based pressure, defined as a response of at least 0.85.
It reports measured prediction and screening outcomes.

The primary forecast lead is seven days; one and three days are secondary.
The transport mapping is fitted on observations through November 19, starting
October 1. Its ridge penalty is selected by November 20-December 3 validation
MAE at lead seven. December 4-31 and January 1-31 are test target periods.
The saved POI predictions determine available origins: complete seven-day
windows yield 22 December and 25 January origins. All fitted transport
coefficients, normalizations and penalty selections remain fixed in January.

## Spatial matching and transport mapping

Each blockface is matched to POIs within 500 m by great-circle distance.
Radii of 250 and 1,000 m are prespecified sensitivities. The POI cohort and
coordinates come from the archived benchmark. A radius with zero matched
POIs gives zero POI-demand covariate; all eligible blockfaces remain evaluated.
The analysis reports match counts and distance coverage.

The parking baselines are the training blockface/weekday/clock-time mean,
the seven-day lag with training blockface/clock-time fallback, and pooled ridge
regression. Calendar means receive five pseudo-observations at the training
blockface/clock-time mean. Ridge features comprise that calendar forecast,
seven- and fourteen-day lags, their missingness indicators, and clock-time
and weekday indicators. Features are standardized using training rows;
the intercept is unpenalized. Penalties are 0.0001, 0.01, 1 and 100 for the
mean squared training objective plus the squared-coefficient penalty.
All predictions are truncated below at zero.

The augmented ridge mapping adds log(1 + total POI visits within the radius).
Training uses observed contemporaneous POI totals; validation and test use
saved POI forecasts. This two-stage mapping learns the association between
observed visits and paid occupancy, then substitutes forecast visits.
Its forecast-input error is assessed by the held-out outcomes. Missing POI
training counts use the same zero input convention as the benchmark.

The fixed T14 MAE-trained inputs are Shared-LSTM, Matched-LSTM, identity
GraphGRU, NAICS-2 GraphGRU, identity Fixed-MTGNN and NAICS-2 Fixed-MTGNN, each
with seeds 42-46. NLinear-ridge supplies an additional deterministic input.
Every model/radius result is reported. The transport penalty is selected
separately for each input model and seed from the same validation rows.

## Metrics and comparisons

Primary error is MAE of paid-occupancy ratio, with count MAE as a secondary
metric. A monitoring budget selects the highest-scored 10% or 20% of available
blockfaces separately for each target day and clock time. Report recall and
precision for the fixed 0.85 pressure definition, and the share of measured
paid occupancy captured within the selected budget. Ties use stable block IDs.
No monetary benefits are inferred.

Model losses and screening outcomes are averaged across the five fitted seeds.
Paired comparisons use the same observed events and 2,000 resamples of seven
consecutive target dates and occupied 3-km spatial cells; 1- and 5-km cells
provide spatial sensitivity. Retain all three clock times within each sampled
date. Confidence intervals are conditional on the fitted models and this
Seattle sample. Primary contrasts compare each augmented mapping with the
parking-only ridge model, plus each NAICS-2 model with its identity counterpart.
The primary endpoint is ratio MAE at lead seven and radius 500 m. Secondary
leads, radii and screening endpoints are exploratory sensitivity analyses.

All fitting, prediction and statistical evaluation execute on AutoDL4.
Source downloads and deterministic file packaging may execute on Windows.
