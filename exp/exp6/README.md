# exp6: Parent-place information and semantic granularity

Evaluate parent-place information and semantic/category granularity. The feature-ablation recipes are maintained once in the shared control queue and feature construction once in `common/inputs.py`.

```bash
python -m exp exp6 prepare-metadata --help
python -m exp exp6 train --plan-out results/exp6/ablation_plan.json
python -m exp exp6 train-parent --plan-out results/exp6/parent_plan.json
python -m exp common evaluate --help
python -m exp exp6 tables --source results/extension --out results/exp6/tables
```

The ablation stage covers history-only and feature-group removals, category/subcategory/NAICS-2/NAICS-4-only variants, and NAICS removals. The parent stage separates recorded parent presence from resolved parent identity. Full models are reused from the core control queue. Evaluation and January prediction reuse the same shared evaluation tools as exp1/exp7. `categories.py` retains the original embedded-POI group definition; corresponding parent-attribution and semantic summaries are exported by exp5 and the original shared analysis.

## Source files

| File | Responsibility |
| --- | --- |
| [`categories.py`](categories.py) | Broad parent-place categories used in the original analysis. |
| [`prepare_metadata.py`](prepare_metadata.py) | Prepare sparse spatial graphs and pre-January metadata sensitivity inputs. |
| [`tables.py`](tables.py) | Render source-backed AutoDL aggregate values without fitting or rescoring. |

For the full option list, use `python -m exp exp6 <action> --help`. Shared input schemas are documented in [DATA.md](../DATA.md).
