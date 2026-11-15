# Seattle POI forecasting revision

The final study uses complete seven-day forecast windows, input lengths of 14 and 28 days, 8 validation origins, 22 December origins and 25 January origins. The licensed panel contains 9,884 POIs. Raw records, event predictions and fitted weights remain in private storage.

## Completed evidence

- 497 fits across 71 configurations, with 355 selected seed runs. Each configuration tunes learning rate at seed 42 using validation MAE and retains seeds 42-46.
- The continuation preserves 115 earlier fits and completes 382 additional fits on AutoDL4. Fitting provenance, GPU environments and checkpoint hashes are in `artifacts/revision_20261114/final_results/fitting_inventory.csv`.
- December has 375 evaluation records: 355 neural seed runs and 20 deterministic baseline records. January has 315 records: 295 neural seed runs and 20 baseline records. The 12 density-subset configurations have December-only evaluation.
- Both periods include paired temporal and spatial uncertainty, degree-controlled graph comparisons, spatial activity coverage, semantic-feature and parent removal, attribution checks, and matched neighbor interventions.
- The original paper snapshot remains under `paper/original`; `paper/revision/outline.tex` is the revised manuscript. `paper/revision/main.tex` forwards to that file.

## Protocol and numerical verification

Training targets end on November 19, 2020; validation labels span November 20-December 3 and test labels December 4-31. January uses the original weights, model-selection choices, and validation predictions. The training-only low-history cohort and density subsets remain fixed.

The default checkpoint replay check uses relative and absolute tolerances of 0.0002. Its historical special handling is confined to T14 fixed MTGNN clique propagation. Nineteen additional checkpoints received a documented numerical verification revision after operator-level diagnosis: deterministic graph evaluation with FP64 accumulation produced identical full predictions in two independent processes. Cohort/per-lead MAE changes are bounded by 0.00005 visits and FAR/recall/precision changes by 0.01 percentage points. The measured maxima are 0.00000144 visits and 0.00750 percentage points. Original strict-check outcomes are retained separately from numerical equivalence verification.

`deterministic_graph_replay.py` contains the evaluated operators. The operational `audit_all_replays_autodl.py`, `verify_replay_candidates_autodl.py`, `summarize_replay_error_autodl.py`, and `finalize_january_autodl.py` record the diagnosis, independent repetition, metric sensitivity, and final consolidation. These operational scripts address the archived AutoDL directory structure. The public numerical audits describe the exact accepted scope.

BysGNN has a resource audit rather than a fitted score. Its official per-POI recurrent units contain 979 million parameters. On the 23.52-GiB AutoDL4 GPU, the official attention module passes a small forward/backward check and exhausts memory at the full cohort and common batch size four. The source blob hash and actual allocation trace are published in `bysgnn_resource_probe.json`.

## Published result tables

`artifacts/revision_20261114/final_results` contains the final aggregate tables, content hashes, numerical audits and two immutable training-source archives. CSV fields and content are checked for event-level identifiers before release. The archives contain Python source files only.

The scientific comparisons average per-seed metrics; they do not ensemble the seed predictions. Paired intervals resample seven-origin blocks and occupied 1/3/5-km cells. Coverage measures observed visits captured by a fixed spatial budget. The January cohort is the same Seattle cohort in a later month.

## Render the paper evidence

From the repository root, with NumPy, pandas and Matplotlib installed:

```sh
python experiments/revision/render_revision_results.py --results artifacts/revision_20261114/final_results --part forecasting
python experiments/revision/render_revision_results.py --results artifacts/revision_20261114/final_results --part controls
python experiments/revision/render_revision_results.py --results artifacts/revision_20261114/final_results --part ablations
python experiments/revision/render_revision_results.py --results artifacts/revision_20261114/final_results --part density
python experiments/revision/render_revision_results.py --results artifacts/revision_20261114/final_results --part diagnostics
python experiments/revision/render_revision_results.py --results artifacts/revision_20261114/final_results --part activity
```

The private input bundle is required to regenerate fitted models and event-level analyses. Its expected SHA-256 is enforced by `audit_protocol.py`. Frozen training code is preserved in `final_results/fitted_source_archives`; current analysis code records the corrections separately. Archive extraction and the documented input layout precede a fresh reproduction run.

The current continuation uses AutoDL4 for training and experiment evaluation. Windows handles SSH, artifact transfer, source editing, manuscript rendering and verification. Historical local-GPU setup helpers remain for provenance.

## Preservation and delivery

- Original Overleaf: https://www.overleaf.com/project/6953dd9e9db9f3ef2cddbd36
- Revision Overleaf: https://www.overleaf.com/project/6af81bebf43f69875df8b640
- Repository destination: `lexilxu/poi_visits`, branch `main`.

The original 17-file snapshot is checked against `paper/original_manifest.json`. Release auditing checks the six main sections, references, result inputs, figures, rendered PDF and published file contents. Local private archives retain the complete diagnostic history and newly generated checkpoints and predictions.
