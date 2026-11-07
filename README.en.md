# NeuroCompiler — English overview

NeuroCompiler is a **research prototype** exploring compilation from structured mathematical knowledge (expressions, operators, coefficients, and initial/boundary conditions) to neural parameters **or** a better-suited numerical solver.

The early `Knowledge → Weights` hypothesis evolved into a hybrid **canonical symbolic IR → backend selection → neural parameter generator / numerical solver → verification** architecture. The project does not establish general mathematical reasoning or superiority to classical solvers.

The archive contains the historical Stage 0–8 experiments and **Stage 9R, an independent implementation** of a 2-D nonlinear elliptic PDE study. The original Stage 9 source and raw results were not available; Stage 9R must not be represented as their reproduction.

### Representative evidence

- Stage 1: a two-step meta-training horizon showed no measured adaptation advantage; a ten-step objective did under matched-start conditions. Fully trained DirectFit still offered better zero-shot quality.
- Stage 3: generic DeepSets failed on unseen compositions. Algebra-aware tokenization recovered the advantage but remained expressively equivalent to the dense model in the tested setting.
- Stage 4: continuous descriptors generated weights for unseen Gaussian parameter identities, within a known function family.
- Stages 5–6: preserving AST operators improved compilation; a symbolic normal form enforced invariance for the tested `+`/`×` rewrites. This is **engineered**, not learned, invariance.
- Stages 7–8: robust classical numerical baselines often dominated learned weight generation in accuracy, and sometimes in speed.
- Stage 9R: physics/response losses helped slightly at equal update count but failed to beat time-matched supervised learning. Classical Galerkin methods were dramatically more accurate. The dominant neural error came from coefficient generation rather than the 36-mode representation floor.

See [Japanese research history](docs/RESEARCH_HISTORY.md), [results ledger](docs/RESULTS_AT_A_GLANCE.md), [reproducibility](docs/REPRODUCIBILITY.md), and [roadmap](docs/ROADMAP.md).

### Quick check

```bash
python3 scripts/verify_manifest.py
python3 -m pip install -r requirements-stage9r.txt
python3 -m pytest experiments/stage-09r/tests -q
```

This lightweight repository deliberately omits large task-level raw data, checkpoint files and source archives. Preserve them separately as GitHub Release assets and verify their SHA-256 hashes in [`archives/source_archives.csv`](archives/source_archives.csv). Work is historical and stage-specific; there is no unified production API or claim that all experiments execute from the lightweight snapshot alone.

**License is not yet selected.** Public visibility is not an open-source license; clear rights and choose one before publishing for reuse.
