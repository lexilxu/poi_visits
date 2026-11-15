# Prediction-preserving parent-embedding origin test

This diagnostic follows the resolved-parent ablation. It tests whether the
original parent attribution depends on the arbitrary embedding origin, while
retaining the fitted predictive function. It uses all five archived T14
GNN-static full-model checkpoints, the same eight December test origins,
all seven leads, all POIs and the archived Gradient x Input calculation.

Let e0 be the fitted embedding for the missing-parent category and Wp the
parent columns of the input projection. Replace every parent embedding e
by e - e0 and the projection bias b by b + Wp e0. Thus Wp(e - e0) + b + Wp e0
equals Wp e + b. All other parameters remain fixed. The original checkpoints
remain immutable; translation is applied in memory to an evaluation copy.

Recompute all archived validation/December predictions and require the original
fitted batch partition. Record the cross-device archive check at the January
replay criterion (relative and absolute 0.0002). Evaluate the original and
translated fitted functions as an FP64 pair, requiring relative and absolute
agreement of 1e-10. Record maximum and mean absolute prediction differences.
Recompute Gradient x Input at the archived sampled events. Compare parent shares,
eligible regimes and missing-parent regime counts before and after translation.
Report every seed and normalize each seed separately before averaging shares.
The translated origin is fixed by the missing-parent category, without using
forecast errors or desired attribution outcomes. All computation uses AutoDL4.

The first attempt used the older attribution loader, whose contiguous batch
partition and 0.00002 archive check differed from the fitted prediction partition
and the January replay protocol. It stopped before translation. The corrected
loader reconstructs the fitted batch partition and uses the existing January
archive criterion. Both attribution copies are evaluated in the same process;
the initial failed trace remains archived.

The corrected FP32 loader also differed from 217 of 2,075,640 archived values
for the first checkpoint under the 0.0002 criterion (maximum violating
difference 0.00061846). That trace is preserved. The final diagnostic therefore
targets algebraic invariance of the stored parameters under FP64 evaluation,
and recomputes both attribution profiles in that arithmetic. It retains the
cross-device FP32 discrepancy as a diagnostic and makes no claim of exact
archive replay. Original benchmark scores and attribution files are unchanged.
