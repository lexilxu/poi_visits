# Reorganization validation

The checks below validate the consolidated source layout. Full POI model fitting and all checkpoint-dependent scientific analyses were not rerun.

| Check | Result |
| --- | --- |
| Existing protocol, control, mechanism and adaptive-model tests | 22 passed |
| New path, manifest compatibility, recipe and subprocess-routing tests | 4 passed |
| Command-line action discovery and `--help` | 104 of 104 passed; no data or GPU required |
| Python parsing and local package imports | 137 source files parse; no unresolved local module imports |
| Shared control-model CPU smoke | GraphGRU, GNN-static, GNN-parent, Shared-LSTM, Matched-LSTM and Fixed-MTGNN forward/backward and identity isolation passed |
| Corrected EvolveGCN-H CUDA equivalence | Passed against vendored author implementation, feature widths 1, 5, 32 and 53 |
| EvolveGCN training stochastic-activation equivalence | Maximum error 0 |
| Checkpoint save/load | Bitwise equal predictions |
| AMP and sparse/dense graph checks | Identity, grouped and sparse graph cases passed |
| Descriptive reductions on the original approved panel | Daily means and category quantiles match the previous CSVs exactly (maximum difference 0) |
| Figure exports | Daily/category, temporal sparsity, forecasting horizons and recovery/activation PDFs, SVGs and PNGs rendered successfully |

The CUDA gate used PyTorch 2.7.1+cu128 on an NVIDIA GeForce RTX 5060 Laptop GPU. Its largest FP64 forward discrepancy was approximately 1.8e-15 and largest gradient discrepancy approximately 5.7e-14. See the [numerical receipt](tests/receipts/evolvegcn_cuda.json) and [CLI receipt](tests/receipts/entrypoints.json).

The corrected model and deterministic group operator are byte-identical to their imported source snapshots:

- `common/evolvegcn.py`: `5114f817da2b718f200ea28de496193d9e07ea7304b54755f08fe12e7f58dfc0`
- `common/deterministic_group.py`: `cc14316eda0851e8b73208de5d7a229745e1cbf95835ca26f0cf0f400ea46b75`

The original `train_model`, `make_windows` and GraphGRU class have identical abstract syntax trees to the frozen source. The attribution-regime exporter was extracted without changing its function body. Source line endings under `exp/` are fixed to LF to keep hashes stable across checkouts.

Licensed records, checkpoints and event-level validation outputs were not added to Git. Existing paper files, public aggregate results and upstream model files were not modified. Repository history and the migration ledger retain the earlier source paths.
