# exp1: Forecasting performance and model comparisons

Compare forecasts across models, graph definitions and horizons. The canonical original queue uses corrected EvolveGCN-H. The control queue retains the earlier GraphGRU architecture under its own name.

```bash
python -m exp checks verify-evolvegcn --out runs/exp1/verification_minimal.json
python -m exp exp1 queue --root runs/exp1 --bundle data/approved_panel.zip --phase all --plan-only
python -m exp exp1 queue --root runs/exp1 --bundle data/approved_panel.zip --phase all
python -m exp exp1 queue-controls --stage core --plan-out results/exp1/control_plan.json
```

Use `train` or `train-complete` for individual fits. The `original` queue phase has 125 fits; `all` includes 35 complete-H7 confirmation fits through `train_complete.py`. Choose the phase before starting an immutable queue. For a previously completed original-only queue, confirmation fits can be invoked individually with `train-complete` into that queue's `runs/` directory.

`baselines`, `complete-baselines`, `extended-baselines` and `train-mtgnn` retain the corresponding baseline implementations. `summarize` and `export-figure-inputs` need corrected runs plus the original baseline archive (`--legacy`). `tables` reads an input manuscript and writes a separate generated copy. The Nature-inspired horizon plot uses exported `07_horizon_skill.csv`:

```bash
python -m exp exp1 plot --data results/figure_inputs/07_horizon_skill.csv --out results/exp1/art
```

## Source files

| File | Responsibility |
| --- | --- |
| [`baselines.py`](baselines.py) | Reproduce simple baselines on the archived forecast windows. |
| [`complete_baselines.py`](complete_baselines.py) | Save remaining simple and direct per-POI AR-ridge baselines on complete horizons. |
| [`export_figure_inputs.py`](export_figure_inputs.py) | Original primary-seed figure inputs from verified corrected predictions. |
| [`extended_baselines.py`](extended_baselines.py) | NLinear, TSB, validation calibration, and observed-activity prioritization. |
| [`plot.py`](plot.py) | Render the horizons figure from verified aggregate data. |
| [`queue.py`](queue.py) | Serial AutoDL4 scheduler with pinned configurations and verified reuse. |
| [`queue_controls.py`](queue_controls.py) | Bounded, resumable job queue for the prepared controlled neural experiments. |
| [`summarize.py`](summarize.py) | Recompute original-table metrics on AutoDL4 from hash-verified predictions. |
| [`tables.py`](tables.py) | Prepare exact EvolveGCN row replacements while retaining original comparators. |
| [`train.py`](train.py) | Use the archived training loop with the corrected EvolveGCN-H constructor. |
| [`train_complete.py`](train_complete.py) | Run the prior complete-window trainer with an EvolveGCN-H model factory. |
| [`train_mtgnn.py`](train_mtgnn.py) | Run the pinned MTGNN temporal model with exact sparse fixed-graph propagation. |

For the full option list, use `python -m exp exp1 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
