# Stage-5 AST Compiler Pilot

## Question
Does preserving symbolic operator/tree structure (`+`, `x`) improve direct weight compilation compared with a bag of the same semantic leaves?

## Architecture
- Leaves: Gaussian semantic descriptors.
- Tree encoder:
  - leaf embedding scaled by coefficient
  - addition: latent addition
  - multiplication: learned symmetric composition using `a+b`, `a*b`, `|a-b|`
- Bag baseline: sums leaf embeddings and deliberately ignores operators/tree structure.
- Target: `1 -> 32 tanh -> 1`.
- Development-only reduced budget: Meta 220 steps, DirectFit 700 steps, 1 seed.

## Calibration
Tree Meta vs matched Tree DirectFit calibration log-gap: **0.00683**.
Bag calibration log-gap: **0.00160**.

## Structural OOD zero-shot result
For nontrivial held ASTs, Bag DirectFit-final / Tree DirectFit-final zero-step MSE ratio:
- sum of products: **4.39x**
- product of sums: **3.65x**
- nested mix: **2.05x**
- deep mix: **4.37x**
- nested mix with unseen leaf identities: **2.64x**

This is a direct signal that operator/tree information carries predictive information that a leaf bag cannot encode.

## Meta adaptation pilot
Tree Meta - matched Tree DirectFit relative log-AUC:
- sum of products: -0.0683
- product of sums: -0.0830
- nested mix: -0.0116
- deep mix: -0.0852
- nested mix with unseen leaves: -0.0214

All listed nontrivial tasks point in the Meta-faster direction in this seed.

## Important failure found
The `triple_product` task became nearly zero because products of separated Gaussian bumps can vanish over most of the domain. Its MSE reaches ~1e-6 and is not a meaningful structural test. It must be removed/replaced in confirmatory Stage-5.

## Decision
**PROMISING PILOT / NOT CONFIRMATORY.**

Before a Stage-5 PASS claim, confirmatory tasks should use paired ASTs with the same leaf multiset but different operator/tree structure and enforce nontrivial target energy. This prevents a bag encoder from exploiting leaf differences and prevents near-zero product tasks from becoming trivial.
