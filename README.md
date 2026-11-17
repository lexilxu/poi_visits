# Seattle POI forecasting

Code for the LLM-POI empirical study. All experiment code lives under [`exp/`](exp/README.md), organized by the existing `exp0`–`exp6` figure groups and three supplementary experiment groups.

| Folder | Experiment |
| --- | --- |
| [`exp0/`](exp/exp0/) | Data preparation and temporal sparsity |
| [`exp1/`](exp/exp1/) | Forecasting performance and model comparisons |
| [`exp2/`](exp/exp2/) | Visit density and low-history demand recovery |
| [`exp3/`](exp/exp3/) | Graph information transfer and adjacency controls |
| [`exp4/`](exp/exp4/) | Neighbor perturbations and activity-matched replacement |
| [`exp5/`](exp/exp5/) | Attribution regimes, maps, stability and faithfulness |
| [`exp6/`](exp/exp6/) | Parent-place information and semantic granularity |
| [`exp7/`](exp/exp7/) | Temporal validation and matched architecture controls |
| [`exp8/`](exp/exp8/) | Independent parking validation |
| [`exp9/`](exp/exp9/) | Decision losses and spatial monitoring coverage |

`exp/common/` contains shared models, data loading, graph operators, training and evaluation. `exp/tests/` contains protocol tests and numerical replay checks. Each experiment folder has its own README and executable analysis scripts; shared training implementations are not duplicated.

From the repository root, use Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m exp --list
python -m exp exp5 --help
python -m exp exp5 reference-centered --help
python -m unittest discover -s exp/tests -t . -v
```

Install a CUDA-compatible PyTorch build for full training and GPU replay. The corrected EvolveGCN numerical gate was developed with PyTorch 2.7.1. CPU tests, input preparation and aggregate plots do not require a GPU. Run commands from this checkout; `pip install -e .` is also supported for local development.

The two training protocols have separate entry points: `original_masked` retains the original 57-day training period and horizon masks; complete-window controls use 50 training days and validation ending at day 64. The corrected EvolveGCN-H model, the earlier GraphGRU control, and their checkpoint formats are explicitly distinguished in the [reproduction guide](exp/README.md).

Licensed inputs and saved checkpoints are supplied separately. Place them under ignored `data/` and `runs/` directories, or pass their paths explicitly. New outputs go under ignored `results/`. `POI_DATA_ROOT`, `POI_RUNS_ROOT`, and `POI_RESULTS_ROOT` override defaults. The [data contract](exp/DATA.md) documents panel schemas and the frozen bundle checksum.

Existing [aggregate results](artifacts/revision_20261114/final_results/), [paper files](paper/), and upstream model implementations in [`external/`](external/) remain available. Published aggregates describe their recorded experiment version; reorganizing source code does not regenerate results or revise the manuscript. Historical fitting snapshots in `artifacts/revision_20261114/final_results/fitted_source_archives/` remain preserved.

The [migration ledger](exp/MIGRATION.json) maps previous script paths to the consolidated modules and records retired prototypes. Git history preserves removed code. The [validation record](exp/VALIDATION.md) reports the checks performed on this layout.
