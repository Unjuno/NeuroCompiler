# Knowledge-to-Weights Stage-1 Experimental Report

## Executive result

Stage-1 separated zero-step function quality from optimization friendliness.

- With a **2-step meta-training horizon**, MetaInit showed **no measurable optimization advantage** over a validation-loss-matched DirectFit baseline: relative log-AUC difference MetaInit−DirectFit = 0.0015, 95% bootstrap CI [-0.0251, 0.0309], n=20 independent training seeds.
- With a **10-step meta-training horizon**, the optimization advantage became large and reproducible.
- In the strict calibration-matched experiment (Stage-1d), zero-step test MSE was effectively matched: seed-median MSE 0.3104 vs 0.3071; paired mean difference -0.0021, 95% CI [-0.0370, 0.0324].
- After 100 adaptation steps, seed-median MSE was **0.0485 (MetaInit) vs 0.0929 (DirectFit matched)**. Paired mean difference -0.1076, 95% CI [-0.2082, -0.0342].
- Initial-loss-normalized relative log-AUC difference was -0.7394, 95% CI [-1.1326, -0.4048; MetaInit better in 10/10 seeds.
- At 100 steps the relative log-loss-ratio difference was -0.9551, 95% CI [-1.4968, -0.4987; this corresponds to an average multiplicative ratio exp(diff) ≈ 0.385. Thus MetaInit retained only about 38.5% of DirectFit-matched's normalized residual at 100 steps.

## Experimental conditions

- Target network: 1 → 16 tanh → 1, 49 parameters.
- Generator: 3 → 32 tanh → 49 + learned shared base.
- Train structures: A, B, C, A+B, B+C.
- Held-out structures: A+C and A+B+C.
- Meta optimizer: Adam, lr=1e-3, meta batch=4, 400 meta steps.
- DirectFit: same generator architecture, direct 0-step behavior loss, 1600 training steps for a strong baseline.
- Meta inner LR during training: 0.05.
- Evaluation LR selected from validation candidates; all methods use identical support trajectories within each task.
- Adaptation checkpoints: 0, 1, 2, 5, 10, 20, 50, 100.
- Stage-1b: 20 independent training seeds, 200 held-out tasks/seed (100 per held-out mask).
- Stage-1d: 10 independent training seeds, separate held-out-structure calibration bank for loss matching and LR selection, then independent test bank with 100 tasks/seed.
- CPU only, PyTorch 2.10.0+cpu, Python 3.13.5.

## Stage-1b: 2-step meta horizon

The key normalized metric was taskwise mean log(MSE_k/MSE_0) across k>0, aggregated by independent training seed.

MetaInit−DirectFit matched relative log-AUC:

- mean: 0.0015
- 95% bootstrap CI: [-0.0251, 0.0309]
- n=20 independent training seeds

Interpretation: **FAIL for the strong optimization-basin claim under the original 2-step meta objective.** The Stage-0 advantage was dominated by the quality of the generated function approximation rather than faster adaptation.

## Stage-1d: 10-step meta horizon, calibration-matched

The held-out structural masks were used only in a separate calibration bank to select a DirectFit checkpoint with comparable zero-step loss and to select adaptation LR. Final metrics were measured on an independent task bank. This experiment is for causal isolation of optimization friendliness, not for a pure zero-shot structural-generalization claim.

### Loss matching

- Calibration log-gap median: 0.0150.
- All 10/10 seeds were within 10% multiplicative zero-step loss on calibration.
- Independent-test zero-step seed-median MSE: MetaInit 0.3104, DirectFit matched 0.3071.
- Paired mean zero-step difference: -0.0021, 95% CI [-0.0370, 0.0324].

### Adaptation

- 50 steps: MetaInit 0.0772, DirectFit matched 0.1606.
- 100 steps: MetaInit 0.0485, DirectFit matched 0.0929.
- Relative log-AUC MetaInit−DirectFit matched: -0.7394, 95% CI [-1.1326, -0.4048].
- MetaInit had better relative log-AUC in 10/10 seeds.
- Relative 100-step log-ratio difference: -0.9551, 95% CI [-1.4968, -0.4987].

### Threshold MSE ≤ 0.10

Median first observed crossing among the sparse checkpoints:

- MetaInit: 35.0 steps.
- DirectFit matched: 100.0 steps.
- Median censored fraction (>100 steps): MetaInit 0.095; DirectFit matched 0.465.

## Scientific interpretation

1. **Optimization friendliness is not automatic.** A short 2-step meta objective did not produce a measurable long-horizon adaptation advantage.
2. **Horizon alignment matters.** Extending the meta objective to 10 inner steps changed the result qualitatively: adaptation became substantially faster even after initial error was matched.
3. **DirectFit remains the stronger zero-shot compiler.** When trained for 1600 steps, DirectFit can achieve much lower 0-step MSE than the matched checkpoint. Therefore the research currently supports two distinct operating modes:
   - DirectFit for zero-shot weight compilation.
   - MetaInit-10 for a deliberately adaptation-friendly initialization.
4. Stage-1d uses held-out-structure calibration, so it does **not** by itself prove zero-shot compositional generalization of the matching procedure. The pure structural holdout evidence remains Stage-1c, where normalized adaptation still favored MetaInit strongly but initial test losses were not perfectly matched.

## Decision

- H(original 2-step): **FAIL**.
- H(horizon-aligned 10-step): **PASS in Stage-1d for optimization friendliness**, subject to replication at 20 seeds and a stricter calibration protocol that preserves a fully untouched structural holdout.
- Overall project status: **PROMISING / stronger than Stage-0, but not yet publication-grade proof of a general Knowledge-to-Weights Compiler.**

## Next experiment

Do not jump to 8/16/32 primitives yet. First run Stage-1e:

- train MetaInit with inner horizons {2,5,10,20};
- use one held-out composition family for calibration and a distinct held-out composition family for final test;
- preregister the primary metric as normalized relative log-AUC;
- use 20 independent training seeds;
- add compute-normalized comparison (gradient evaluations and wall-clock) and a stronger shared-initialization meta baseline using the same 10-step horizon.

If Stage-1e reproduces the 10-step advantage on a structurally untouched final mask, then proceed to Stage-2 primitive scaling.
