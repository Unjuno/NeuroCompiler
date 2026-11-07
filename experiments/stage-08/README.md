# NeuroCompiler Stage 8 reproducibility bundle

The Japanese research report is `report_ja.md`. This is a real new experimental run, not merely a proposed roadmap.

## Layout

- `code/stage8_nonlinear.py`: nonlinear residual, analytic gradient, reference ODE solver, task banks, Meta/Direct training and classical controls.
- `code/stage8b_runner.py`: independent higher-budget follow-up.
- `code/stage8c_correction.py`: post-hoc residual-Newton analysis.
- `code/stage8d_confirm.py`: fresh-task replication of fixed correction strategies.
- `code/solver_router.py`: working symbolic IR -> classical numerical backend routing prototype.
- `code/cheb_factorial.py`, `code/benchmark_stage8.py`: sampling/precision and latency ablations.
- `code/analyze_stage8.py`, `code/finalize_stage8.py`: analyses and plots.
- `code/neurocompiler_stage7/code/`: imported legacy exact symbolic differential polynomial IR and network primitives.
- `stage8a_output/`: 10 new seeds, original lower-budget contrast, archived preliminary pilot.
- `stage8_output/`: 10 new seeds with higher-budget Meta, classical baseline, correction and fresh-task controls.
- `figures/`: visualization.
- `summary.json`, `*.csv`: key data and statistics.

## Example

Unzip bundle and run from the top-level directory. CPU-only Python 3.13, NumPy 2.3, SciPy 1.17, Pandas 2.2, PyTorch 2.10 are known to work.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd code
python stage8_nonlinear.py gradient
python solver_router.py
```

The raw results inside `stage8a_output` and `stage8_output` were generated on fixed held-out task banks using 10 independent model-training seeds. Re-running model training or the numeric baseline will execute the algorithms again, not silently synthesize a result. Files suffixed `_EVAL_ONLY.npz` contain reference solutions and **must not be used during learning or checkpoint selection**.

## Interpretation

Stage8a refutes Meta superiority under the smaller compute budget. Stage8b closes the gap but does not establish positive superiority. The strongest classical controls solve this limited 1D monotone nonlinear ODE family more accurately. Latency comparisons use *precompiled* task inputs and one CPU thread; do not claim end-to-end parse/AST execution time. Correction results are finite-sample tests, not formal accuracy certificates.
