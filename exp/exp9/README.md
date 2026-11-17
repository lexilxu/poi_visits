# exp9: Decision losses and spatial monitoring coverage

Analyze asymmetric forecast losses, forecast-time information strata and spatial monitoring coverage. These are reductions of stored predictions, with no model fitting. [PROTOCOL.md](PROTOCOL.md) records the analysis definitions.

```bash
python -m exp exp9 analyze --base runs/poi_continuation_20261114 --extension runs/poi_extension_20261114 --bundle data/approved_panel.zip --out results/exp9/metrics
python -m exp exp9 verify --base runs/poi_continuation_20261114 --extension runs/poi_extension_20261114 --bundle data/approved_panel.zip --results results/exp9/metrics
python -m exp exp9 plot --results results/exp9/metrics --out results/exp9/report
python -m exp exp9 spatial-coverage --help
```

`analyze` retains the source-archive alignment checks and expected original aggregate checks. Its historical `GraphGRU_*` checkpoint inputs are now labeled GraphGRU explicitly. The label correction does not change any forecast or loss calculation. `verify` independently recomputes scalar reductions and recovery metrics. `plot` writes the figure, tables and audit bundle to the chosen directory, without using a personal Downloads folder.

## Source files

| File | Responsibility |
| --- | --- |
| [`analyze.py`](analyze.py) | Read-only forecast reductions on AutoDL4; no training framework required. |
| [`plot.py`](plot.py) | Format verified remote aggregates into additions-only Markdown and a paper figure. |
| [`spatial_coverage.py`](spatial_coverage.py) | Paired origin-block uncertainty for observed-visit spatial prioritization. |
| [`verify.py`](verify.py) | Independent scalar reductions and history construction for the saved analysis. |

For the full option list, use `python -m exp exp9 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
