# NeuroCompiler Stage 9R

Independent first-principles PDE experiment. Read `report_ja.md` for measured results and limitations, and `proofs_ja.md` for definitions/derivations.

## Environment
Python3.13.5, NumPy2.3.5, SciPy1.17.0, PyTorch2.10.0+cpu, pandas, matplotlib, pytest. CPU FP32 neural training/inference; FP64 numerical calculations. No GPU or network required. Torch deterministic algorithms and flush-denormal enabled. Files contain no external datasets or sensitive information.

## Reproduce
Run from this folder. Fix BLAS threads before importing NumPy:
```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
python src/experiment.py bank
python src/run_all.py
python src/evaluate.py
python src/refinement.py
python src/followup.py
python src/replay_check.py
```
`run_all.py` builds final test bank and trains 20 seeds x 4 arms. It does not tune on final tasks. Development seed100 exists in this bundle but is excluded from every main contrast. Saved seed0 Data/Response/DataTime were replayed with bit-identical state tensors.

The DataTime arm stops by wall time, so its step count can differ on other hardware. For exact replay use the saved `models/DataTime_SEED.json` step count and call `experiment.train_one(seed,'DataTime',steps=recorded_steps,seconds=None)`. This replays the same validation candidates and minibatches.

## Files
- `docs/protocol.md`, `checks/protocol_lock.json`: internal pre-evaluation freeze (not external preregistration).
- `src/`: numerical core, learning, evaluation, refinement, independent fresh-bank check, replay.
- `data/`: train/val/test/fresh banks with parameters, projected references, reference spectra, physics matrices.
- `models/`: all saved models and final predictions (unsafe pickle loading is used only for these locally generated trusted checkpoints).
- `analysis/`: task-level results, seed summaries, contrasts, training curves/times, strong nonlearned baselines.
- `checks/`: numerical tests, reference-grid refinements, environment, logs, audit and hashes.
- `figures/`: PNG visualizations (English labels).

## Important interpretation
This is a new, deliberately limited test of one PDE family. It is not a reproduction of the historical Stage9 metrics. The compiler outputs 36 basis coefficients (not arbitrary target hidden weights); semantic rewrite/AST/OOD-operator-family claims are not tested. All confidence intervals condition on fixed banks. Computational cost comparisons of training are separate from deployment cost; no full speed/accuracy Pareto claim is made.

Primary two center-OOD comparisons use97.5% bootstrap intervals (Bonferroni adjustment); other comparisons including fresh-bank check are exploratory95%. See contrasts.csv for the exact statistic (mean of paired seed-level mean log relative-L2 ratios), not a ratio of reported medians.

## Primary methodology references
- SciPy DST-I, orthonormal transform: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html
- Variational learning prior work (not an exact implementation here): E & Yu, Deep Ritz, arXiv:1710.00211, https://arxiv.org/abs/1710.00211
