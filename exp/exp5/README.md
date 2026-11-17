# exp5: Attribution regimes, maps, stability and faithfulness

Analyze attribution profiles, spatial regimes, origin sensitivity and feature-removal faithfulness. Two checkpoint families have distinct entry points.

`reference-centered` reproduces the revised original GNN-static attribution method. It uses the archived static checkpoint, saved original forecasts, training-derived references and the independent original FP64 GPU audit. The CPU calculation checks prediction/gradient invariance before assigning regimes. `regimes.py` contains the shared standardized-share assignment and representative-POI export. No external `summarize_formal.py` is required.

```bash
python -m exp exp5 reference-centered --formal-source data/frozen_original --bundle data/approved_panel.zip --source data/original_static --gpu-audit data/original_fp64_audit --out results/exp5/reference_centered
python -m exp exp5 reference-centered-figures --results results/exp5/reference_centered --cartography data/cartography
```

The two outputs remain `attribution_fingerprint.pdf` and `attribution_cluster_map.pdf`. Cartography inputs are `king_county_areawater_2020.zip`, `kitsap_county_areawater_2020.zip` and `seattle_streets_tiger2020.npz`; `prepare-cartography` prepares the alternate compact geometry used by `map`.

`evaluate` and `faithfulness` operate on corrected EvolveGCN-H static checkpoints, with `own_gradients.py` and `feature_worlds.py`. Generate their receipts with `checks verify-own-gradients` and `checks verify-feature-worlds`, then use `summarize`, `fingerprint`, `export-map-inputs` and `map`. `method-audit`, `summarize-stability`, `compare-periods`, `embedding-origin` and `location-origin` preserve the associated method and temporal checks. [ORIGIN_PROTOCOL.md](ORIGIN_PROTOCOL.md) records the origin diagnostic.

## Source files

| File | Responsibility |
| --- | --- |
| [`compare_periods.py`](compare_periods.py) | Compare completed five-seed attribution consensus across evaluation months. |
| [`embedding_origin.py`](embedding_origin.py) | Translate parent embeddings and compensate bias, preserving the predictor. |
| [`evaluate.py`](evaluate.py) | Replay a corrected static checkpoint and export exact own-POI attribution. |
| [`export_map_inputs.py`](export_map_inputs.py) | Export mapped counts and positions on AutoDL4 for the original spatial figure. |
| [`faithfulness.py`](faithfulness.py) | Own-feature sensitivity versus full-model responses at fixed sampled events. |
| [`feature_worlds.py`](feature_worlds.py) | Full EvolveGCN-H worlds for one target feature-group intervention each. |
| [`fingerprint.py`](fingerprint.py) | Render the original attribution fingerprint from audited remote summaries. |
| [`location_origin.py`](location_origin.py) | Prediction-preserving coordinate-origin test of the original static checkpoint. |
| [`map.py`](map.py) | Map completed attribution profiles with the existing public basemap geometry. |
| [`method_audit.py`](method_audit.py) | Gradient x Input and reference-sensitive integrated gradients for GNN-static. |
| [`own_gradients.py`](own_gradients.py) | Exact own-POI Jacobian blocks for identity-graph EvolveGCN-H. |
| [`plot_map.py`](plot_map.py) | Map formal attribution groups and report their local concentration. |
| [`plot_origin.py`](plot_origin.py) | Publication figure from the audited AutoDL attribution-origin aggregates. |
| [`prepare_cartography.py`](prepare_cartography.py) | Prepare public 2020 TIGER/Line geometry, without any POI records. |
| [`reference_centered.py`](reference_centered.py) | CPU-only reference-centered attribution for the archived original GNN-static fit. |
| [`reference_centered_figures.py`](reference_centered_figures.py) | Render the two existing attribution figures with recalculated assignments. |
| [`regimes.py`](regimes.py) | Shared attribution-regime assignment and representative-POI export. |
| [`summarize.py`](summarize.py) | Produce original attribution table and figure inputs from corrected profiles. |
| [`summarize_origin.py`](summarize_origin.py) | Summarize translated attribution with the original valid-seed consensus rule. |
| [`summarize_stability.py`](summarize_stability.py) | Cross-seed attribution stability, feature-removal utility, and parent diagnostics. |
| [`tables.py`](tables.py) | Format four original tables from verified full primary-seed attribution outputs. |

For the full option list, use `python -m exp exp5 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
