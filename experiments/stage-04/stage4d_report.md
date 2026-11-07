# Knowledge-to-Weights Stage-4d Confirmatory Report

## Question
Can a generator compile a primitive identity that was never active during training from a continuous semantic descriptor, rather than from a learned primitive ID?

## Design
- Primitive family: Gaussian `g_{mu,sigma}(x)`.
- Train primitive identities: centers `{-0.8,-0.4,0,0.4,0.8}` × widths `{0.18,0.30}`.
- Development-held identities: separate from confirmatory identities.
- Confirmatory-held identities: centers `{-0.2,0.6}`, including unseen width `0.24`.
- Secondary descriptor extrapolation: centers `{-1.0,1.0}`, width `0.24`.
- Semantic descriptor: `(mu, sigma, normalized-width, mu^2)`.
- Target network: `1 -> 32 tanh -> 1`, 97 parameters.
- Generator latent size: 48.
- Meta horizon: 10 inner steps, train inner LR 0.02.
- Meta training: 600 steps; DirectFit: 2400 steps.
- 10 independent training seeds.
- Calibration uses known primitive identities only. Final held identities are not used for checkpoint/LR selection.

## Primary endpoint
Relative log-AUC over adaptation checkpoints 1,2,5,10,20,50,100, normalized by each task's 0-step MSE. Negative Meta - Matched DirectFit means faster relative adaptation.

### Primary result
Semantic MetaInit - loss-matched Semantic DirectFit:
- mean seed-level difference: **-0.15393**
- 95% seed bootstrap CI: **[-0.22145, -0.09461]**
- direction: **10/10 seeds MetaInit better**

Calibration matching was valid in all 10 seeds; maximum semantic calibration log-gap was **0.01473**, well within the prespecified 10% multiplicative tolerance.

### Initial-loss matched robustness
Restricting final tasks to those where MetaInit and matched DirectFit 0-step MSE differed by <=10%:
- tasks retained: **2065** across all 10 seeds
- relative log-AUC difference: **-0.09845**
- 95% seed bootstrap CI: **[-0.16749, -0.03614]**
- 8/10 seed-level means negative

## Zero-shot unseen-identity compilation
Semantic DirectFit-final vs ID-only DirectFit-final, averaged over the six primary held-identity structures using log MSE ratio:
- mean log ratio: **-1.95037**
- 95% CI: **[-2.07099, -1.84860]**
- semantic better: **10/10 seeds**
- exp(mean log ratio) = **0.1422**

Thus semantic DirectFit-final has about 14.2% of the ID-only zero-shot MSE on a geometric scale (about 7.0x lower).

Median zero-step MSE across seeds for Semantic DirectFit-final:
- held single A: 0.0233
- held single B: 0.0314
- unseen-width primitive: 0.00990
- held pair: 0.0722
- held + known mix: 0.0730
- all four held identities: 0.0843

Corresponding ID-only DirectFit-final medians:
- 0.1769, 0.1468, 0.2220, 0.4209, 0.1786, 0.7519

## Secondary OOD
Descriptor-range extrapolation (`mu = +/-1.0`):
- Meta - matched relative log-AUC: **-0.50370**, 95% CI **[-0.66745,-0.36579]**, 10/10 negative.
- Semantic DirectFit-final zero-step median: **0.0884** vs ID-only **0.3264**.

High coefficient magnitude:
- Meta - matched relative log-AUC: **-0.17872**, 95% CI **[-0.27413,-0.07666]**, 9/10 negative.
- Semantic DirectFit-final zero-step median: **0.3836** vs ID-only **2.1898**.

## Decision
**PASS for semantic primitive-identity generalization within a known smooth parametric family.**

What is supported:
1. A primitive identity never active during training can be compiled to useful zero-shot weights from `(mu,sigma)` semantics.
2. A meta-trained semantic compiler produces a more adaptation-friendly state than a loss-matched direct semantic compiler.
3. The result survives modest descriptor and coefficient OOD.

What is not supported:
1. Arbitrary unseen mathematical operators or families.
2. AST/program semantics beyond additive composition.
3. High-frequency sinusoidal extrapolation under the previous tanh target; that experiment was confounded by a target optimization/representation floor around MSE 0.49 for sin5/6.

## Error checks
- 10 independent training seeds completed.
- Final identities were separated from development identities.
- Calibration contained known identities only.
- Semantic calibration matching passed 10/10.
- Task-level 0-step-matched robustness retained 2065 tasks.
- ID baseline uses untrained embeddings for held identities, as intended to test whether semantic descriptors carry transferable identity information.
