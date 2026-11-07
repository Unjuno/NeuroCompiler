# Knowledge-to-Weights Stage-1e: Structurally Untouched Final Composition

## Design

- Train masks: A, B, C, A+B, B+C.
- Calibration/selection mask: A+C only.
- Final untouched mask: A+B+C only. Final tasks are not used for checkpoint or LR selection.
- Meta horizons: 2, 5, 10, 20; 400 meta steps, meta batch 4.
- DirectFit: same generator, 1600 direct steps, checkpoint matched on calibration separately for each horizon.
- Shared FO-MAML baseline: 10-step horizon, 400 meta steps.
- 20 independent training seeds; 100 calibration tasks and 100 final tasks per seed.
- Evaluation steps: 0,1,2,5,10,20,50,100.
- Primary metric: taskwise mean log(MSE_k/MSE_0) over k>0, then median within seed; paired bootstrap across seeds. More negative is faster relative adaptation.

## Primary result

MetaInit_h10 − DirectFit_matched_h10 relative log-AUC:

- mean paired difference: -0.6889
- 95% bootstrap CI: [-0.9414, -0.4573]
- MetaInit wins: 17/20 seeds

This tests transfer of optimization friendliness from calibration structure A+C to a structurally untouched A+B+C final family.

## Horizon sweep

| horizon | paired mean diff vs matched DirectFit | 95% CI | Meta wins | calibration matches within 10% |
|---:|---:|---:|---:|---:|
| 2 | -0.0086 | [-0.0366, 0.0194] | 12/20 | 20/20 |
| 5 | -0.0915 | [-0.1448, -0.0400] | 15/20 | 20/20 |
| 10 | -0.6889 | [-0.9414, -0.4573] | 17/20 | 19/20 |
| 20 | -2.3790 | [-2.7114, -2.0365] | 20/20 | 3/20 |

Horizon 20 is not a valid loss-matched contrast in most seeds: its calibration match succeeds within 10% in only 3/20 seeds, because MetaInit often sacrifices zero-step accuracy beyond the range spanned by DirectFit checkpoints.

## Strong shared-initialization baseline

MetaInit_h10 − FO-MAML_shared_h10 relative log-AUC:

- mean paired difference: -0.4677
- 95% CI: [-0.7171, -0.2428]
- MetaInit wins: 16/20 seeds

## Absolute final-mask performance (seed medians)

| method | MSE step 0 | MSE step 100 | relative log-AUC |
|---|---:|---:|---:|
| MetaInit_h2 | 0.188321 | 0.058585 | -0.3359 |
| MetaInit_h5 | 0.183586 | 0.051114 | -0.4245 |
| MetaInit_h10 | 0.312977 | 0.045187 | -0.9555 |
| DirectFit_matched_h10 | 0.270162 | 0.079246 | -0.3341 |
| FO-MAML_shared | 0.985695 | 0.482204 | -0.5666 |
| DirectFit_final | 0.062631 | 0.023422 | -0.4772 |

## Final-mask initial-loss gap

Calibration matching does not guarantee equality on A+B+C. For h10, log(Meta/Direct) seed-level zero-loss ratio has mean 0.0996, 95% CI [-0.0067, 0.2135], median multiplicative ratio 1.073. This is why the preregistered primary metric normalizes by each method's own L0.

Post-hoc robustness restricted to final tasks whose initial losses are within 10%:

- matched tasks: 276 total; median 12.5 per seed; 20/20 seeds represented.
- Meta−Direct relative log-AUC mean seed-median difference: -0.5553; 95% CI [-0.7941, -0.3410].
- Meta wins 18/20 represented seeds.

This robustness analysis is post hoc and must not replace the primary result.

## Compute

Training-time medians are in stage1e_compute_summary.csv. Meta inner gradient evaluations are exactly 400 × 4 × horizon: 3,200 / 8,000 / 16,000 / 32,000 for horizons 2/5/10/20. FO-MAML_h10 also uses 16,000 inner target-gradient evaluations. DirectFit has no inner adaptation gradient evaluations during compiler training, although it still performs outer supervised gradient updates.

## Decision

- h2: FAIL/UNCERTAIN for transferred optimization advantage.
- h5: PASS for transferred optimization advantage.
- h10: PASS for transferred optimization advantage on untouched A+B+C.
- h20: INVALID as a clean loss-matched comparison in most seeds, despite fast relative adaptation, because matching fails.

The Stage-1d signal is therefore tested under a stricter structural holdout here rather than assumed to generalize.
