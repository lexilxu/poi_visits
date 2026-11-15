# Substantive revision after the complete frozen benchmark

The 497-fit benchmark remains archived at GitHub commit
`85f8d803b393d4b868a0dd01c6b8dcfddb1f846f`. Supplementary experiments use new
output directories and source archives. All fitting and evaluation use AutoDL4.

## Architectural control, fixed before fitting

Matched-LSTM uses the same scalar-to-32 input projection, 32-dimensional
recurrent state, last-observation projection and additive readout as GraphGRU
with identity adjacency. The recurrent cell is LSTM. The existing Shared-LSTM
and GraphGRU results remain the paired comparators. Parameter counts and actual
fitting time will be reported for each architecture.

Four configurations cross input lengths 14/28 with MAE/MSE training. Each uses
the existing seven-day forecast protocol, batch size four, Adam with weight
decay 0.0001, clipping norm five, at most 100 epochs and patience ten. Learning
rates 0.0003, 0.001 and 0.003 are compared at seed 42 by validation MAE. The
selected seed-42 fit is retained and seeds 43–46 use the selected rate: 28 fits,
20 selected runs. The original cohort, training/validation/test dates,
normalization, low-history definition and threshold rules are preserved.
January evaluation retains fitted weights and November validation calibration.

Primary comparisons are Matched-LSTM versus Shared-LSTM and identity GraphGRU
at the same input length and loss, with five-seed summaries, paired temporal
and spatial intervals, magnitude error and matched-FAR detection. Every
configuration is reported.

## Resolved parent representation, fixed before fitting

The archived frequency grouping retains two named parent identities and pools
the remaining identities. The supplementary GNN-parent representation uses all
233 recorded parent identifiers. Its eight parent dimensions contain a binary
presence indicator and a seven-dimensional identity embedding. An absent
parent has indicator zero and a fixed zero identity embedding. Other fields,
the total 52-dimensional static input and the recurrent architecture retain
their original definitions. Parent identifiers are coded by sorted labels.

Four T14 MAE configurations retain the full parent block, remove identity only,
remove presence only, or remove the complete parent block. They use the same
three-rate validation selection and five seeds as the architectural control:
28 fits and 20 selected runs. Each configuration is evaluated in December and
January, with overall, low-history and recorded-parent cohorts. All four
results are retained. The metadata snapshot is the existing retrospective
snapshot; identifier resolution changes its representation.
The identity block also distinguishes an absent parent through its zero vector.
Block-removal contrasts measure predictive utility of these encodings; they
provide no orthogonal decomposition of the information in parent presence.

## Longer-history data audit

The identified July–September 2020 monthly source shards extend the preceding
history. Downloads preserve source IDs, byte counts and SHA-256. Windows performs
deterministic CSV selection and compression after verifying each complete raw
stream and gzip CRC. AutoDL verifies the packaged input hashes and performs all
panel construction, statistical evaluation and model fitting. This transport
revision follows measured slow upload and a direct Google Drive connection
timeout from AutoDL. Immutable raw sources remain locally. Extraction retains the original 9,884 placekeys and all Seattle WA and
Portland OR records. Outcome values play no role in this extraction rule.

Coverage, identifier consistency and record dates are audited before fixing
any supplementary cohort or forecast split. Static files currently available
start in November 2020. Historical snapshot dates and actual historical release
availability are recorded separately.

## Earlier-period and second-city temporal controls, fixed before fitting

July-September 2020 panels define each city cohort from all July records with
city/region Seattle/WA or Portland/OR. Later missing monthly records remain
missing. Cohort inclusion uses July membership and precedes validation/test.
Training targets run July 1-August 19, validation August 20-September 2 and
test September 3-30. The same complete H7 windows, T14 history, five seeds,
three-rate selection, normalization and optimization settings apply.

Identity GraphGRU and Matched-LSTM are fitted in each city, giving four
configurations, 28 fits and 20 selected runs. These experiments assess the
temporal architectural comparison in an earlier period and another city.
Deterministic comparators are zero, calendar mean, same-weekday training mean,
seasonal naive, validation-selected recent mean and NLinear-ridge. Report all
POI and training-defined low-history MAE, and paired seven-origin-block
intervals for GraphGRU versus Matched-LSTM and each versus NLinear-ridge.
The original October-January cohort and predictions remain archived separately.

## External transportation data audit

Seattle official paid-parking datasets `wtpb-jp8d` (2020) and `jb6y-98nr`
(2021) provide timestamps, blockface IDs, paid occupancy, recorded capacity,
area and point geometry. These are payment-based occupancy estimates.
Missing records remain missing. Calendar coverage, operating-hour coverage,
capacity changes and spatial matching will be audited before fixing the
external forecast and monitoring-prioritization protocol. The protocol will
specify training-only mappings and thresholds before test comparisons.

## Interpretation

The paper distinguishes fitted-model comparisons, fixed-model input sensitivity,
gradient attribution and predictive utility. Negative results remain reported.
