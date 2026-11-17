# exp3: Graph information transfer and adjacency controls

Evaluate graph information transfer and separate target history from neighbor activity. `evaluate` uses corrected EvolveGCN checkpoints and an identity reference. `evidence-grid` is the saved-prediction evidence analysis for the complete-window controls; its bins are fixed from validation data. [PROTOCOL.md](PROTOCOL.md) records those definitions.

```bash
python -m exp exp3 prepare-graphs --help
python -m exp exp3 train-controls --plan-out results/exp3/control_plan.json
python -m exp exp3 evaluate --help
python -m exp exp3 evidence-grid --help
python -m exp exp3 plot --help
```

`prepare-graphs` constructs equal-degree category and random controls with paired seeds. Training uses the shared control queue. `tables` integrates corrected transfer aggregates into a separately written manuscript copy. All experiment-specific transformations are in this folder; common checkpoint replay and graph operators are in `exp/common/`.

## Source files

| File | Responsibility |
| --- | --- |
| [`evaluate.py`](evaluate.py) | Original descriptive transfer strata using corrected frozen predictions. |
| [`evidence_grid.py`](evidence_grid.py) | Aggregate stored predictions into validation-defined evidence cells on AutoDL. |
| [`plot.py`](plot.py) | Render the audited evidence grid in the reference manuscript figure style. |
| [`prepare_graphs.py`](prepare_graphs.py) | Build the requested k=5,10,20 matched graph controls without training. |
| [`tables.py`](tables.py) | Format the two original graph-contrast tables from audited primary-seed data. |

For the full option list, use `python -m exp exp3 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
