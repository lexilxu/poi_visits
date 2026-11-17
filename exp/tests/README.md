# Validation

Run dataset-free tests with `python -m unittest discover -s exp/tests -t . -v`.

`test_protocol.py`, `test_controls.py`, `test_mechanisms.py` and `test_adaptive.py` cover split boundaries, future-label exclusion, density ties, tied-score calibration, paired inference, ablations, graph controls and sparse/dense adaptive operators. `test_layout.py` checks source relocation and queue routing.

The executable `verify_*`, `audit_group_replay` and `register_group_replay` modules run through `python -m exp checks <action>`. EvolveGCN equivalence requires CUDA but no licensed panel. The replay, counterfactual and derivative checks require the saved inputs indicated by `--help`; they are not part of the dataset-free unit suite.

After changes to code pinned by a numerical receipt, regenerate that receipt. Historical path aliases do not skip content hashes or relax error tolerances.
