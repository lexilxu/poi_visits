# exp2: Visit density and low-history demand recovery

Test whether removing sparse POIs changes forecast accuracy, and measure positive-demand recovery versus false activations for low-history POIs. Density uses the three original training-only rankings: overall activity, recent activity and persistence, each with exclusions of 10%, 25%, 50% and 75%.

```bash
python -m exp exp2 train --root runs/exp1 --bundle data/approved_panel.zip --plan-only
python -m exp exp2 train --root runs/exp1 --bundle data/approved_panel.zip
python -m exp exp2 summarize-density --source . --root runs/exp1 --bundle data/approved_panel.zip --out results/exp2/density
python -m exp exp2 plot --data results/figure_inputs/01_recovery_activation.csv --out results/exp2/art
```

The 60 subset fits use the final original-loop recipe (T28/H14, Identity, batch 6, legacy AMP); five full-cohort runs are reused from exp1. `density_protocol.py` owns subset selection and stable tie handling. The separate `train-controls` and `summarize-complete-density` actions concern complete-window GraphGRU controls. Low-history recovery inputs are exported by `exp1 export-figure-inputs`; do not combine cohorts or predictions across the two protocols.

## Source files

| File | Responsibility |
| --- | --- |
| [`density_protocol.py`](density_protocol.py) | Original density rankings with fixed 57-day observed-label denominators. |
| [`plot.py`](plot.py) | Render the recovery figure from verified aggregate data. |
| [`summarize_complete_density.py`](summarize_complete_density.py) | Compare each retrained density subset with HA on its identical POIs and events. |
| [`summarize_density.py`](summarize_density.py) | Verify and reduce all corrected density fits, reusing the full-cohort runs. |
| [`train.py`](train.py) | Refit the 12 prespecified visit-density subsets using the original training loop. |

For the full option list, use `python -m exp exp2 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
