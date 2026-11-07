# Knowledge-to-Weights Compiler — Stage 5/6 Experimental Report

## Executive result

Stage 5 establishes two distinct claims:

1. **AST structure is useful for zero-shot weight compilation.** With exactly the same leaf multiset and coefficients, a structure-blind Bag generator must produce the same weights for two different ASTs. A Tree generator reduces zero-shot MSE substantially relative to this Bag baseline.
2. **Optimization-friendly initialization survives nonlinear AST composition primarily as a short-horizon effect.** In a fresh confirmatory test aligned to the 10-step meta-training objective, MetaInit adapts faster than a loss-matched DirectFit checkpoint at steps 1–10; the advantage weakens by 100 steps.

Stage 6 identifies and fixes a new failure mode:

3. **A syntax Tree compiler is not invariant to semantics-preserving algebraic rewrites.** Equivalent ASTs yield different weights/predictions.
4. **A symbolic canonicalization front-end fixes this exactly for the +/× commutative-semiring identities tested, without collapsing non-equivalent expressions.**

## Environment

- Python 3.13.5
- PyTorch 2.10.0+cpu
- Linux 6.18.35 x86_64
- AMD EPYC 9V74
- 5 CPU cores allocated, 1 thread/core
- OMP/MKL/OpenBLAS/NumExpr threads fixed to 1 for stable execution

## Stage 5a — paired-AST structural test

Primary final pairs use the **same primitive identities and the same coefficients** but different operator placement/tree wiring. The Bag generator deliberately ignores operators and tree structure.

### Zero-shot structure effect

Seed-level mean log MSE ratio:

- Tree DirectFit vs Bag DirectFit: **-0.9544**
- 95% seed bootstrap CI: **[-0.9719, -0.9393]**
- Tree better: **10/10 seeds**
- exp(-0.9544) = **0.385**, so Tree zero-shot MSE is about 38.5% of Bag on a geometric-ratio scale.

Direct structural diagnostics:

- Bag maximum within-pair weight difference: **5.96e-8**
- Tree median within-pair max weight difference: **0.2275**
- Bag median predicted-pair-separation / target-pair-separation: **~5.3e-16**
- Tree median ratio: **0.372**

### Structural-blindness lower bound

For a structure-blind method that predicts the same function p for two different targets y_A and y_B,

    1/2 [ MSE(p,y_A) + MSE(p,y_B) ]
      = MSE(p,(y_A+y_B)/2) + 1/4 MSE(y_A,y_B)
      >= 1/4 MSE(y_A,y_B).

Thus a Bag encoder that maps the paired ASTs to the same weights has an unavoidable error floor proportional to target separation. The experiment directly verifies that the Bag predictions are effectively identical within each pair.

## Stage 5b — full-budget long-horizon diagnostic

The original full-horizon relative log-AUC (steps 1,2,5,10,20,50,100) did **not** robustly establish MetaInit superiority:

- Meta - matched DirectFit: **-0.02510**
- 95% CI: **[-0.05966, +0.00545]**
- Meta better: **6/10 seeds**

Therefore the claim "MetaInit remains superior throughout 100-step adaptation on nonlinear AST OOD" is not supported.

A post-hoc decomposition showed that the signal was concentrated at the earliest steps. Because that observation was post-hoc, a fresh confirmatory experiment was run.

## Stage 5d — preregistered short-horizon confirmatory

Fresh, previously unused 6-leaf paired AST families were generated. Target energy and pair separation were both thresholded at >= 0.02. No final-test information was used for checkpoint or learning-rate selection.

Primary metric was fixed **before** evaluation as the average relative log-loss over steps {1,2,5,10}.

### Primary result

- Meta - matched DirectFit early-AUC: **-0.06020**
- 95% CI: **[-0.09366, -0.02755]**
- Meta better: **8/10 seeds**

Robustness subset where final-task 0-step MSEs were within 10%:

- n = **1441 tasks** across all 10 seeds
- early-AUC difference: **-0.01700**
- 95% CI: **[-0.03118, -0.00183]**
- Meta better: **8/10 seeds**

### Stepwise result

| step | Meta - matched relative log-loss | 95% CI | Meta-better seeds |
|---:|---:|---:|---:|
| 1 | -0.06270 | [-0.09112, -0.03646] | 10/10 |
| 2 | -0.06763 | [-0.10073, -0.03615] | 9/10 |
| 5 | -0.05819 | [-0.09621, -0.02006] | 7/10 |
| 10 | -0.05228 | [-0.09421, -0.01111] | 7/10 |
| 20 | -0.05094 | [-0.09569, -0.00698] | 7/10 |
| 50 | -0.04981 | [-0.10250, +0.00020] | 7/10 |
| 100 | -0.05002 | [-0.11436, +0.00867] | 7/10 |

Interpretation: **optimization friendliness exists, but is strongest in the adaptation horizon close to the meta-training objective.** It is not a persistent 100-step dominance claim.

## Stage 5c negative diagnostic

Expanding the training grammar to energy-filtered 4-leaf ASTs did not immediately restore long-horizon superiority. Seed 0 on fresh 6-leaf OOD had a 4-family average Meta-minus-matched relative log-AUC of approximately +0.0248. This weakens the hypothesis that the Stage-5b instability was caused only by shallow train ASTs or trivial near-zero product tasks.

## Stage 6 — algebraic rewrite sensitivity

Five semantics-preserving rewrite families were tested:

- distributivity with 3 leaves
- multiplication associativity
- distributivity with an added leaf
- 4-leaf product expansion
- 5-leaf product expansion

The target functions agree numerically to at most **2.34e-15 MSE**.

Across 5 seeds, the ordinary syntax Tree compiler nevertheless showed:

- median-per-seed prediction disagreement / target energy: mean **0.03323**
- 95% exploratory bootstrap CI: **[0.02640, 0.03882]**
- rewrite-vs-compact log MSE ratio: mean **+0.08388**
- exp(0.08388) = **1.0875**, i.e. the rewrite form was about 8.7% worse on the aggregate geometric-ratio scale.

The 5-leaf product expansion showed the strongest syntax sensitivity (~32% rewrite penalty in the 5-seed diagnostic).

## Stage 6b — canonical symbolic front-end

Before neural encoding, the AST is mapped into a canonical polynomial normal form:

1. leaf c*g_i -> monomial {i: c}
2. addition -> coefficient-wise dictionary addition
3. multiplication -> monomial convolution with sorted primitive keys
4. terms are sorted and rebuilt into one deterministic AST

### Exact rewrite invariance in the tested implementation

Across all 5 seeds and all tested rewrite pairs:

- Canonical maximum generated-weight difference: **0.0**
- Canonical maximum prediction disagreement: **0.0**

This is not learned invariance; it is imposed by the symbolic compiler front-end.

### Accuracy cost

Canonical vs Syntax average rewrite-task log MSE ratio:

- mean **-0.04665**
- 95% exploratory CI **[-0.09720, +0.02172]**

Thus no degradation is detected; point estimate slightly favors Canonical but the 5-seed CI crosses zero.

### Non-equivalent structural control

Canonicalization does **not** collapse different semantics:

- median predicted-separation / target-separation: Canonical **0.640**, Syntax **0.607**
- Canonical vs Syntax structural-control log MSE ratio: **-0.03776**, exploratory 95% CI **[-0.07132, -0.00420]**

## Proof sketch: why canonicalization removes the tested rewrite dependence

Let P(e) be the polynomial dictionary induced recursively by an AST e. Under exact arithmetic,

- P(a+b) = P(a) + P(b)
- P(a*b) = P(a) convolution P(b), with monomial keys sorted.

Dictionary addition and convolution implement the commutative-semiring laws for + and ×. Therefore expressions related only by associativity, commutativity, and distributivity map to the same P(e). A deterministic reconstruction C(P(e)) then yields the same canonical AST. A deterministic neural generator G therefore satisfies

    G(C(P(e1))) = G(C(P(e2)))

for such equivalent expressions e1 and e2.

Implementation caveat: the algebraic statement is exact-arithmetic. Arbitrary floating-point coefficient evaluation can introduce order-dependent roundoff. The tested rewrite families produced exact canonical equality in float32/host arithmetic, but a production compiler should canonicalize coefficients with exact rationals/symbolic numbers or a documented quantization rule.

## Current research claim

Supported:

> Structured knowledge can be compiled into neural parameter states. Preserving AST operator structure materially improves zero-shot weight generation; a meta objective can make those states better for short-horizon adaptation; and symbolic normalization can enforce semantics-preserving rewrite invariance before neural weight generation.

Not supported:

> MetaInit remains superior for arbitrarily long adaptation, or a raw neural Tree encoder is automatically invariant to equivalent symbolic forms.

## Next experiment

The next high-information step is to add a unary operator with real scientific meaning, e.g. differentiation D, and test whether a compiler with explicit symbolic rewrite rules (linearity/product rule) can compile equivalent differential expressions consistently while preserving zero-shot/adaptation performance. This moves the project from a commutative-semiring toy algebra toward equation/PDE front-ends.
