# Knowledge-to-Weights — Stage-2 Final Experimental Report

## Configuration
- Date: 2026-09-01
- Primitive count: 8
- Target network: 1 -> 32 tanh -> 1, 97 parameters
- Generator: 8 -> 64 tanh -> 97 plus shared base
- Meta objective: FO-MAML-style task-conditioned weight generation
- Meta horizon: 10
- Meta-training inner LR: 0.02
- Meta steps: 600
- DirectFit steps: 2400
- Meta batch: 8
- Support size per inner step: 16
- Adaptation checkpoints: 0,1,2,5,10,20,50,100
- Calibration mask: {0,2,4,6}
- Checkpoint matching and adaptation LR selection use calibration only.

## Stage-2a: confirmatory held-out composition
Final mask: {0,2,5,7}, untouched during development.

MetaInit - matched DirectFit relative log-AUC:
- mean difference: -0.308567
- 95% seed bootstrap CI: [-0.392977, -0.228734]
- direction wins: 20/20 seeds

MetaInit - FO-MAML:
- mean difference: -0.291385
- 95% CI: [-0.378814, -0.211280]
- direction wins: 20/20 seeds

Calibration loss matching within multiplicative 10%: 20/20 seeds.
Maximum calibration log-gap: 0.028542.

Median absolute metrics across training seeds:
| method                |   zero_median |   step100_median |   rel_log_auc |
|:----------------------|--------------:|-----------------:|--------------:|
| DirectFit_final       |     0.0633129 |        0.0332547 |     -0.361031 |
| DirectFit_matched_h10 |     0.41598   |        0.286357  |     -0.114219 |
| FO-MAML_shared_h10    |     1.39117   |        0.907969  |     -0.139962 |
| MetaInit_h10          |     0.364831  |        0.175297  |     -0.395625 |

Important: DirectFit_final remains much better in absolute zero-shot and post-adaptation MSE. The demonstrated advantage is adaptation rate relative to initial error against loss-matched DirectFit / shared FO-MAML, not universal absolute accuracy dominance.

## Stage-2a secondary: all 8 primitives active
MetaInit - matched DirectFit relative log-AUC:
- mean difference: -0.288099
- 95% CI: [-0.354891, -0.222055]
- direction wins: 19/20 seeds

## Stage-2b: structural breadth, no retuning
Five additional held-out masks were evaluated without using them for training, checkpoint selection, matching, or LR selection:
- far_pair_0_4
- spread_triple_1_4_7
- cross4_1_3_4_6
- fiveway_0_1_3_5_7
- sixway_0_1_2_4_6_7

Seed-level mean across all five masks, MetaInit - matched DirectFit:
- mean difference: -0.557751
- 95% CI: [-0.721269, -0.419011]
- direction wins: 10/10 seeds

Per-mask:
| mask                |      mean |     ci_lo |     ci_hi |   wins |   n |
|:--------------------|----------:|----------:|----------:|-------:|----:|
| far_pair_0_4        | -1.29733  | -1.54472  | -1.04977  |     10 |  10 |
| spread_triple_1_4_7 | -0.532513 | -0.745978 | -0.371892 |     10 |  10 |
| cross4_1_3_4_6      | -0.359112 | -0.551659 | -0.184298 |      9 |  10 |
| fiveway_0_1_3_5_7   | -0.312335 | -0.460211 | -0.183448 |      9 |  10 |
| sixway_0_1_2_4_6_7  | -0.287464 | -0.392533 | -0.193833 |     10 |  10 |

Calibration matching within multiplicative 10%: 10/10 seeds.

## Interpretation
The Stage-1 phenomenon survives scaling from 3 to 8 primitives. Under loss-matched initialization, task-conditioned MetaInit produces faster relative adaptation than both a matched DirectFit checkpoint and a shared FO-MAML initialization. Stage-2b shows that this is not confined to a single held-out structural mask.

The strongest remaining limitation is practical: a fully trained DirectFit generator still gives substantially lower absolute MSE. Thus the supported claim is about transferable optimization readiness, not best absolute function approximation.

## Decision
- Stage-2a: PASS.
- Stage-2b breadth replication: PASS.
- Claim supported: formal/task structure can be mapped to parameter states with transferable adaptation-friendly optimization geometry in this controlled function family.
- Claim not supported: MetaInit universally outperforms DirectFit in absolute error.


## Stage-2c: coefficient-magnitude distribution shift
No retraining-policy or hyperparameter retuning was performed. A new nonlocal triple mask {0,3,6} was used. Training coefficient magnitudes remain 0.50–1.50; evaluation uses:
- low OOD: 0.10–0.50
- in-distribution: 0.50–1.50
- high OOD: 1.50–2.00

MetaInit - matched DirectFit relative log-AUC:
| range              |   mean_diff |     ci_lo |      ci_hi |   wins_of_10 |
|:-------------------|------------:|----------:|-----------:|-------------:|
| amp_low_0.10_0.50  |   -1.06491  | -1.44362  | -0.688145  |           10 |
| amp_in_0.50_1.50   |   -0.359033 | -0.55041  | -0.185488  |            9 |
| amp_high_1.50_2.00 |   -0.171445 | -0.253368 | -0.0997811 |           10 |

The adaptation advantage remains statistically below zero in all three ranges, but weakens at the high-magnitude extrapolation range.

Median normalized MSE at step 100 (MSE divided by target mean-square power):
| mask               |   MetaInit_h10 |   DirectFit_matched_h10 |   DirectFit_final |   FO-MAML_shared_h10 |
|:-------------------|---------------:|------------------------:|------------------:|---------------------:|
| amp_high_1.50_2.00 |       0.166393 |                0.273966 |        0.018215   |             0.729882 |
| amp_in_0.50_1.50   |       0.107714 |                0.213962 |        0.00632868 |             0.773291 |
| amp_low_0.10_0.50  |       0.116763 |                0.203187 |        0.0119365  |             0.770551 |

Interpretation:
- Optimization-readiness transfers outside the coefficient-magnitude training range.
- High-amplitude extrapolation is materially harder; the effect size contracts.
- DirectFit_final still dominates absolute/normalized accuracy, so MetaInit's supported advantage remains adaptation efficiency, not best final predictor.

## Updated decision
- Stage-2a single untouched structure, 20 seeds: PASS.
- Stage-2b breadth across five unseen structures, 10 seeds: PASS.
- Stage-2c coefficient-range shift, 10 seeds: PASS for relative adaptation transfer; FAIL for any claim that MetaInit beats DirectFit_final in absolute accuracy.
