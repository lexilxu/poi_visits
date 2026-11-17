> Historical experiment protocol. Source paths below have been updated; run counts and dated output locations describe that release. Use the current [experiment guide](README.md) for entry points.

# Required experiments and acceptance criteria

Source: the user's complete “TR-C 投稿前的完整修稿方案”, uploaded as
“粘贴的文本 (1)(1).txt” on 2026-11-14, 2,055 extracted lines.
This checklist follows that document, including its optional/conditional items.
Existing results are not evidence that a new control has been completed.

| ID | Required evidence | Current status | Paper destination |
|---|---|---|---|
| P0-01 | Every window's input/target dates, graph/normalization dates; disjoint scored labels | Complete seven-day protocol implemented, tested, and applied throughout the revision | Parameter Settings |
| P0-02 | Complete-horizon model selection; multiple origins; per-lead errors | 497 frozen fits and 71 five-seed configurations complete; 115 preserved fits and 382 AutoDL4 fits; December and January evaluations complete | Parameter Settings; Input Windows and Forecast Leads |
| P0-03 | Actual graph implementation, adaptive/fixed distinction, identity isolation | Fixed and adaptive implementations audited; all fits and both-period evaluations complete | Forecasting Models; Adjacency Matrices |
| P0-04 | HA, weekly baseline, AR-ridge, fair LSTM, DLinear or NLinear | Baselines and both five-seed LSTM input windows complete in December and January | Forecasting Performance |
| P0-05 | BysGNN if reasonably reproducible; intermittent-demand baseline | TSB complete. Official BysGNN attention resource probe on AutoDL4 passes a 32-node forward/backward check and exhausts 23.52-GiB GPU memory at 9,884 POIs and batch size 4; no BysGNN fit produced | Forecasting Performance |
| P0-06 | NAICS-4/top-category edge and prediction/table provenance | Identical graph partition verified; both-period event provenance and tables complete | Alternative Adjacency Matrices |
| P1-01 | Validation-only affine calibration; equal-FAR Recall comparison | Calibration and matched-FAR records complete for 375 December and 315 January records | Low-History POIs |
| P1-02 | MAE, DMR, FAR, Recall, Precision/AP at 0.1/0.25/0.5/1/2 | Threshold diagnostics complete for 375 December and 315 January records | Forecast Evaluation; Low-History POIs |
| P1-03 | Explain observed zero visits versus latent actual demand | Definition added to Dataset and POI Attributes | Dataset and POI Attributes |
| P1-04 | Degree-controlled NAICS-2/4 and random graphs, k=5/10/20 | All 45 graphs, five-seed controlled fits, and both-period evaluations complete | Alternative Adjacency Matrices |
| P1-05 | Mutually exclusive transfer assignment; pre-outcome evidence strata | Complete in both months; paired block intervals and inspected figures added to the local manuscript | Related-POI Perturbation Analysis |
| P1-06 | Matched-size active/correlated/random/recent-active perturbations | All 20 cases complete on the same 1,935 targets; five-seed summaries, intervals, tables and figures checked | Related-POI Perturbation |
| P1-07 | Activity-matched history replacement and disconnected negative control | All 20 cases complete; disconnected-node changes are zero; replacement results included in the local manuscript | Related-POI Perturbation |
| P1-08 | Train-standardized continuous static inputs; refit GNN-static | Train-standardized five-seed T14 and T28 refits complete; original attribution corrections retained | Feature Attribution and POI Regimes |
| P1-09 | IG with defined legal references; shares/rank/regime agreement | Complete in both months at all five seeds, with two observed references and passing numerical completeness checks; low method agreement reported in the manuscript; parent-regime membership reflects missing-parent encoding | POI Attribution Regimes |
| P1-10 | Full, without location/identity/parent/others, history-only, all retrained | All required retrained variants complete in both periods; table and manuscript text updated and inspected | Explaining POI Visit Predictions |
| P1-11 | Parent ablation on 581 POIs, categories, attribution-error relationship | Both-period 581-POI removal, category effects, metadata sensitivity and attribution-error summaries complete; manuscript updated and inspected | Parent-Place Information |
| P1-12 | Four single semantic inputs; all jointly; without NAICS-2/4 | All semantic variants complete in both periods; table and paired December effects added to manuscript | Category and Industry Granularity |
| P2-01 | Five fixed training seeds; mean and SD, no best-seed selection | All 71 recipes retain seeds 42-46 with validation-only selection; 497 fits independently audited | Empirical Results |
| P2-02 | Paired origin blocks, spatial sensitivity, no IID event inference | 122 paired comparisons complete in each period, including temporal blocks and spatial sensitivity | Empirical Results |
| P2-03 | Independent period/city, core findings repeated | 315 January records complete on the same Seattle cohort; 19 cases use the approved numerical verification revision | Empirical Results |
| P2-04 | Optional meaningful activity-prioritization application | Coverage and paired intervals complete for 28 configurations in each period | Empirical Results; Conclusions and future work |
| R-01 | Event records with model, adjacency, seed, origin, lead, POI, y, prediction, split | 355 selected December predictions and 295 January neural predictions preserved; 12 density subsets are December-only | Reproducibility outputs |
| R-02 | Preserve original, minimum section changes, synchronized code and paper | Original 17-file hash check passed; complete revision compiles to 41 pages; final publication evidence is recorded in RELEASE_AUDIT.json | Entire project |
| R-03 | Fresh GPT-6 Sol Max review, no history and no target disclosed | Fresh independent GPT-6 Sol Max review completed on the complete 41-page manuscript; original review retained privately; mathematical and presentation clarifications checked in a follow-up | Independent assessment |

Completion requires code, a successful actual run, inspected outputs, and matching
paper text. A script or a limitation sentence alone does not complete an experiment.

## Source implementations

- NLinear: https://github.com/cure-lab/LTSF-Linear . The local CPU variant uses
  shared weights and exact regularized MSE fitting, with validation MAE selection.
  Its solver and regularization differ from the original stochastic recipe.
- BysGNN: https://github.com/USC-InfoLab/busyness-graph-neural-network . Do not
  replace it with an unrelated GNN under the same name.
- TSB: https://doi.org/10.1016/j.ejor.2011.05.018 . Missing observations are skipped;
  zeros remain observed zeros. Initialization and validation smoothing grid are saved.

## Current execution

All training and experiment evaluation run on AutoDL4 through Windows SSH.
Windows handles source editing, artifact transfer, manuscript rendering and file verification.
The frozen v1/v2 queues are complete. Archived inputs, predictions, checkpoints and selections remain preserved.
Completed aggregate evidence and fit provenance are in `artifacts/revision_20261114/final_results`.
The released manuscript and source hashes are recorded in `paper/revision/RELEASE_AUDIT.json`.
