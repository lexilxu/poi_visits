# Seattle POI forecasting

The completed revision evaluates seven-day forecasts for 9,884 Seattle points of interest using two input lengths and a fixed January temporal holdout.

- [Revised paper](paper/revision/LLM_POI_Revision.pdf)
- [Experiment protocol and reproduction guide](experiments/revision/README.md)
- [Required evidence checklist](experiments/revision/REQUIREMENTS.md)
- [Aggregate results and numerical audits](artifacts/revision_20261114/final_results)
- [Release source and PDF checksums](paper/revision/RELEASE_AUDIT.json)
- [Preserved original manuscript](paper/original)

The frozen queues contain 497 fits across 71 configurations and 355 selected seed runs. The continuation retains 115 earlier fits and completes 382 additional fits on AutoDL4. December has 375 evaluation records and January has 315; density-subset refits are evaluated in December.

The study reports forecasting accuracy, intermittent-visit detection, graph-input sensitivity, feature removal, attribution stability and observed-visit coverage. January uses the same Seattle cohort. Licensed input records, event predictions and fitted checkpoints remain in private evidence storage.

The source snapshots used for fitting are preserved under `final_results/fitted_source_archives`; the current analysis code documents the accepted numerical replay revision.

Revision Overleaf: https://www.overleaf.com/project/6af81bebf43f69875df8b640
