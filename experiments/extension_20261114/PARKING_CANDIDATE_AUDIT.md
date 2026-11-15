# Candidate-set correction after the first parking evaluation

Source inspection identified that the screening calculation ranked blockfaces
with observed target labels. A prospective monitoring allocation ranks the
training-defined eligible blockface/clock-time pairs. The corrected calculation
ranks that fixed candidate set first, then applies target-label availability
when scoring selected locations. It reports both planned selections and
selections with observed outcomes. Missing labels retain their missing status.

All 93 selected transport mappings and saved predictions remain unchanged.
The original evaluation directory is preserved. A new screening audit directory
contains corrected screening metrics and paired intervals, together with a
recomputation of every saved prediction MAE. The protocol endpoint, radii,
thresholds, budgets, models and training/validation boundaries remain fixed.
The earlier screening results are superseded by this prospective-candidate
calculation. This amendment was triggered by inspection of ranking inputs.
