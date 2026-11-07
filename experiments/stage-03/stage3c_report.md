# Knowledge-to-Weights Stage-3c Confirmatory Report

## Design
- Representation: algebra-aware variable-length tokens, `sum(coefficient * primitive_embedding) -> tanh -> weights`.
- Initialization: primitive embedding initialized with the same `1/sqrt(D)` scale as the equivalent dense first layer.
- Target network: 1 -> 32 tanh -> 1 (97 parameters).
- Generator parameters: 6978.
- Meta horizon: 10; training inner LR: 0.02.
- 10 independent training seeds.
- Calibration mask is disjoint from five confirmatory OOD masks.
- Checkpoint and adaptation LR selection use calibration only.

## Primary result
MetaInit minus loss-matched DirectFit relative log-AUC averaged within seed over the five confirmatory masks:
- mean difference: -0.428313
- 95% seed bootstrap CI: [-0.559554, -0.303497]
- MetaInit wins: 10/10 seeds.

MetaInit minus FO-MAML:
- mean difference: -0.413573
- 95% CI: [-0.543826, -0.290228]

## Robustness
- Calibration matching within 10%: 10/10 seeds.
- Maximum calibration log-gap: 0.060359.
- Permutation-invariance maximum absolute weight difference: 4.768e-07.
- Post-hoc final-task subset with initial MSE within 10%: 821 tasks across 10 seeds.
- Relative log-AUC difference on this subset: -0.318662, 95% CI [-0.411167, -0.226986], Meta wins 10/10 seeds.

## Absolute performance caveat
The fully trained DirectFit model remains substantially better in absolute MSE than MetaInit. Stage-3c therefore supports optimization-friendliness, not global superiority of MetaInit.

## Decision
**PASS** for the claim that, after matching initialization scale and calibration loss, an algebra-aware token-conditioned generator can produce parameter states that adapt faster on unseen compositions than a same-family DirectFit checkpoint and shared FO-MAML.

## Remaining limitation
Primitive identity is still represented by a learned embedding table. This does not yet establish generalization from the semantic content of a previously unseen primitive. Stage-4 must remove this dependency.
