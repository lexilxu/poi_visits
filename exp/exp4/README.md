# exp4: Neighbor perturbations and activity-matched replacement

Run fixed-checkpoint neighbor interventions, including activity-matched replacement. `evaluate` is the corrected EvolveGCN evaluator; `reused_worlds.py` reuses identical counterfactual worlds after an equivalence gate. `audit` and `summarize` retain the complete-window GraphGRU intervention analysis.

```bash
python -m exp checks verify-counterfactuals --help
python -m exp checks verify-reused-worlds --help
python -m exp exp4 evaluate --help
python -m exp exp4 plot --help
```

Supply both the graph and identity checkpoint directories, the frozen panel, and `--source .`. Optional `--world-reuse-gate` must match the current helper hashes. POI order, forecast origins and paired predictions are validated before reductions. The evaluation keeps the replacement and external matched-control definitions and the paired uncertainty calculations from the latest main-branch revision.

## Source files

| File | Responsibility |
| --- | --- |
| [`audit.py`](audit.py) | Matched-size interventions on fitted semantic GraphGRU models. |
| [`evaluate.py`](evaluate.py) | Original eight neighbor groups plus prespecified matched-history controls. |
| [`plot.py`](plot.py) | Render the original paired-marker neighbor diagnostic, including empty sets. |
| [`reused_worlds.py`](reused_worlds.py) | Reuse identical full-input interventions with the fixed eight-world arithmetic. |
| [`summarize.py`](summarize.py) | Five-seed summaries of matched neighbor interventions in December and January. |

For the full option list, use `python -m exp exp4 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
