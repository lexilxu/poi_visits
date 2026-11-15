# Seattle POI forecasting

The revision diagnoses seven-day visit forecasts for 9,884 Seattle points of interest using temporal controls, attribution stability and independent paid-parking observations. Earlier July-defined Seattle and Portland cohorts provide additional temporal comparisons.

- [Revised paper](paper/revision/LLM_POI_Revision.pdf)
- [Experiment protocol and reproduction guide](experiments/revision/README.md)
- [Required evidence checklist](experiments/revision/REQUIREMENTS.md)
- [Aggregate results and numerical audits](artifacts/revision_20261114/final_results)
- [Release source and PDF checksums](paper/revision/RELEASE_AUDIT.json)
- [Preserved original manuscript](paper/original)

The frozen queues contain 497 fits across 71 configurations and 355 selected seed runs. The continuation retains 115 earlier fits and completes 382 additional fits on AutoDL4. December has 375 evaluation records and January has 315; density-subset refits are evaluated in December.

The [supplementary protocol](experiments/extension_20261114/PROTOCOL.md) adds 84 AutoDL4 fits across 12 configurations and 60 selected seed runs: matched LSTM controls, resolved parent encoding, and July–September city comparisons. Cumulative totals are 581 fits, 83 configurations and 415 selected runs. The original cohort adds 40 records in each evaluation month; the two city cohorts have 20 separate September neural records. [Extension aggregates](artifacts/revision_20261114/final_results/extension_20261114) preserve all five seeds.

External validation combines saved POI forecasts with official Seattle payment-based parking observations. Its 93 input/seed/radius combinations select among 16 distinct fitted ridge candidates. The attribution-origin diagnostic compares paired double-precision evaluations of stored parameters; its archive separately records cross-device single-precision replay discrepancies. The detailed graph figure uses validation-defined evidence bins and stored predictions.

The study reports forecasting accuracy, intermittent-visit detection, graph-input sensitivity, feature removal, attribution stability and observed-visit coverage. January uses the same Seattle cohort. Licensed input records, event predictions and fitted checkpoints remain in private evidence storage.

The source snapshots used for fitting are preserved under `final_results/fitted_source_archives`; the current analysis code documents the accepted numerical replay revision.

Revision Overleaf: https://www.overleaf.com/project/6af81bebf43f69875df8b640
