> Historical experiment protocol. Source paths below have been updated; run counts and dated output locations describe that release. Use the current [experiment guide](../README.md) for entry points.

# Preserved run sources

These Python files were copied without modification from the local
`paper_figures/mechanism_figures/formal_2020q4` source set, with the supporting
`plot_figures.py` used to render its figures. They document the archived runs.

The model historically labeled `EvolveGCN` in these outputs is the `GraphGRU`
class in `train_formal.py`, not the original EvolveGCN architecture. The revised
manuscript corrects that name. MTGNN uses fixed graphs with graph learning off.
See `experiments/revision/README.md` for audited findings and new experiments.

The training scripts require CUDA and the local licensed input bundle. They are
preserved as provenance, not presented as newly rerun or corrected experiments.
