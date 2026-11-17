# Shared experiment components

| Module | Responsibility |
| --- | --- |
| `original_protocol.py` | Original panel loading, masked windows, GraphGRU and original optimization loop |
| `evolvegcn.py` | Corrected EvolveGCN-H and its graph operators |
| `controlled_training.py` | Complete-window control trainer and optional injected model factory |
| `evolvegcn_training.py` | Earlier standalone corrected-model trainer and shared graph construction |
| `adaptive_mtgnn.py`, `lstm.py` | Adaptive MTGNN and LSTM components |
| `static_features.py`, `adjacency.py`, `inputs.py` | Static attributes, graph construction and controlled inputs |
| `protocol.py`, `audit.py`, `controls.py` | Time windows, cohort rules and experiment invariants |
| `checkpoints.py`, `evolvegcn_io.py` | Saved-control and corrected-checkpoint loading |
| `replay.py`, `deterministic_group.py` | Numerical replay and deterministic graph reductions |
| `evaluate.py`, `summarize.py`, `paired.py` | Evaluation, aggregate reduction and paired inference |
| `original_analysis.py` | Original archived multi-experiment reductions |
| `figures.py`, `primary_figures.py`, `compact_figures.py`, `forecast_figures.py` | Shared plot implementations and styles |
| `paths.py`, `provenance.py` | Portable paths and historical manifest name resolution |

Use the experiment-specific entry points first. These shared modules are not independent claims of equivalence between the original masked and complete-window protocols. `original_protocol.py` is retained because trained checkpoints and control experiments depend on it; it is not an obsolete duplicate of EvolveGCN-H.
