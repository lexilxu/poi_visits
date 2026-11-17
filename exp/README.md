# Experiment guide

Run `python -m exp <group> <action> [options]` from the repository root. `python -m exp --list` lists groups; `python -m exp exp4 --help` lists that group's actions. Append `--help` to an action to see its exact inputs. The equivalent direct module command, for example `python -m exp.exp4.evaluate`, is supported.

| Group | Purpose | Main flow |
| --- | --- | --- |
| [exp0](exp0/README.md) | Dataset and sparsity description | profile → prepare/package → describe/plot-sparsity |
| [exp1](exp1/README.md) | Forecasting benchmarks and horizons | numerical gate → queue/train → summarize → plot |
| [exp2](exp2/README.md) | Density sensitivity and low-history recovery | density train → summarize-density; exp1 exports → plot |
| [exp3](exp3/README.md) | Information transfer | prepare-graphs → train-controls → evaluate/evidence-grid → plot |
| [exp4](exp4/README.md) | Neighbor interventions | replay verification → evaluate → summarize/plot |
| [exp5](exp5/README.md) | Attribution regimes | reference-centered or evaluate → summarize → fingerprint/map |
| [exp6](exp6/README.md) | Parent and semantic features | prepare-metadata → train/train-parent → shared evaluation → tables |
| [exp7](exp7/README.md) | Temporal and architecture checks | prepare holdouts → evaluate; matched/city queues |
| [exp8](exp8/README.md) | Parking validation | download/audit → evaluate → verify/tables |
| [exp9](exp9/README.md) | Decision implications | saved forecasts → analyze → verify → plot |

## Protocol and model boundaries

`common/evolvegcn.py` implements the corrected **EvolveGCN-H** with evolving graph-convolution weights. `common/original_protocol.py` retains **GraphGRU** and the original training/window functions needed by archived fits. The complete-window control trainer in `common/controlled_training.py` also supports static GNNs, resolved parent encoding, shared/matched LSTMs, fixed MTGNN and adaptive MTGNN. These architectures are not interchangeable.

| Protocol | Training / validation boundary | Main entry | Selection and arithmetic |
| --- | --- | --- | --- |
| Original masked | 57 / 64 days, 92-day panel | `exp1 train` | Original MSE loop, batch 6, AMP training/validation, FP32 test predictions; horizon masks |
| Complete H7 confirmation | 50 / 64 days | `exp1 train-complete` | Earlier complete-window loop with EvolveGCN-H factory; MAE, batch 4, FP32 |
| Complete-window controls | 50 / 64 days | `exp1 queue-controls` | Validation-only learning-rate selection, five fixed seeds, disjoint target windows |

The fixed seed set is 42–46. The original EvolveGCN queue has 125 configurations including seeds, and the complete-H7 confirmation adds 35. Density refits add 60 fits while reusing the five full-cohort T28/H14 Identity fits. `exp1 queue --plan-only` and `exp2 train --plan-only` create plans without fitting models. These counts describe recipes, not claims that training was rerun during the reorganization.

`exp1 queue-controls --stage ...` provides the shared control queue. Experiment-specific presets fix the stage: `exp2 train-controls` = density; `exp3 train-controls` = graph; `exp6 train` = ablation; `exp6 train-parent` = parent_resolved; `exp7 train-matched` and `train-cities` select their respective stages. The default is planning only; execution requires `--execute`, input paths and explicit time/fit budgets.

## Files and provenance

See [DATA.md](DATA.md) for input schemas. New experiments should write to `runs/<experiment>/` and `results/<experiment>/`. Do not mix old GraphGRU predictions with corrected EvolveGCN checkpoints or substitute one training protocol's low-history labels for another's.

Historical manifests may use old source filenames. `common/provenance.py` translates names while retaining executed-code hash validation. A moved helper whose bytes changed must pass its numerical gate again before producing a new receipt. Existing checkpoint/prediction checksums, POI ordering, window alignment and replay tolerances remain enforced.

For a historical source-path lookup, consult [MIGRATION.json](MIGRATION.json). The original and complete-window release specifications are retained in [ORIGINAL_PROTOCOL.md](common/ORIGINAL_PROTOCOL.md) and [CONTROL_PROTOCOL.md](common/CONTROL_PROTOCOL.md). Their dated output paths and historical run totals are records of those releases.

## Validation

```bash
python -m unittest discover -s exp/tests -t . -v
python -m exp common controlled-training --smoke --device cpu
python -m exp checks verify-evolvegcn --out results/checks/evolvegcn.json
```

The last command requires CUDA and compares the corrected model with the vendored author implementation. Other `checks` actions validate full-world interventions, own-input gradients, feature counterfactuals, deterministic group replay and world reuse against saved inputs. Their `--help` output specifies the required checkpoints and previous numerical receipts.
