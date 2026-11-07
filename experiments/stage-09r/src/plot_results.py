from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path("/mnt/data/neurocompiler_stage9r")
summary = pd.read_csv(root / "analysis/summary.csv")
contrasts = pd.read_csv(root / "analysis/contrasts.csv")

groups = ["id", "center", "width", "operator"]
group_labels = ["In distribution", "Unseen source\npositions", "Narrower\nsources", "Larger PDE\ncoefficients"]
methods = [
    ("Data", "Data only: 800 updates", "o"),
    ("Response", "Data + physics + response: 800 updates", "s"),
    ("DataTime", "Data only: matched training time", "^"),
    ("Galerkin36", "Numerical solve: same 36 coefficients", "D"),
    ("Galerkin100", "Numerical solve: 100 coefficients", "x"),
]

fig, ax = plt.subplots(figsize=(10.3, 5.9))
for i, (method, label, marker) in enumerate(methods):
    values = [
        100 * float(summary.query("method == @method and group == @g")["relative_l2_median"].iloc[0])
        for g in groups
    ]
    ax.scatter(np.arange(4) + (i - 2) * 0.055, values, marker=marker, s=65, label=label)
ax.set_yscale("log")
ax.set_xticks(np.arange(4), group_labels)
ax.set_ylabel("Relative L2 error (%) — logarithmic scale")
ax.set_title("Stage 9R: physics losses help slightly; numerical correction matters more")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, fontsize=9)
ax.grid(True, axis="y", which="major", alpha=0.3)
fig.text(0.08, 0.015,
         "200 tasks per group. Neural results: median of 20 seed-level task medians.\n"
         "Numerical results: task median. FP32 neural inference; FP64 numerical reference/solves.",
         fontsize=9)
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig(root / "figures/absolute_errors.png", dpi=180)
plt.show()

pairs = [
    ("Weak", "Data", "Weak residual / Data\n97.5% interval"),
    ("Response", "Data", "Weak + response / Data\n97.5% interval"),
    ("Response", "DataTime", "Weak + response / Time-matched Data\n95% interval (secondary)"),
]
fig, ax = plt.subplots(figsize=(10.2, 4.6))
for i, (num, den, label) in enumerate(pairs):
    row = contrasts.query("group == 'center' and numerator == @num and denominator == @den").iloc[0]
    ratio, lo, hi = row["ratio"], row["ratio_low"], row["ratio_high"]
    ax.errorbar(ratio, i, xerr=[[ratio - lo], [hi - ratio]], fmt="o", capsize=5)
    ax.annotate(f"{ratio:.3f} [{lo:.3f}, {hi:.3f}]", (hi + 0.003, i),
                va="center", fontsize=10)
ax.axvline(1.0, linestyle="--", linewidth=1)
ax.set_yticks(np.arange(3), [p[2] for p in pairs])
ax.invert_yaxis()
ax.set_xlim(0.95, 1.13)
ax.set_xlabel("Geometric mean error ratio (less than 1 favors numerator)")
ax.set_title("Unseen source positions: a small gain disappears after time matching")
fig.tight_layout()
fig.savefig(root / "figures/paired_contrasts.png", dpi=180)
plt.show()

# Lower bounds are measured independently of every neural training seed.
baseline = pd.read_csv(root / "analysis/baseline_tasks.csv")
fig, ax = plt.subplots(figsize=(8.5, 4.7))
for i, (method, label) in enumerate([
    ("Projection36", "Best possible in 36-mode space"),
    ("Galerkin36", "Galerkin solve in 36-mode space"),
    ("Galerkin100", "Galerkin solve in 100-mode space"),
]):
    values = [
        100 * baseline.query("method == @method and group == @g")["relative_l2"].median()
        for g in groups
    ]
    ax.scatter(np.arange(4) + (i - 1) * .08, values, s=65, label=label, marker=["o", "x", "^"][i])
ax.set_xticks(np.arange(4), group_labels)
ax.set_yscale("log")
ax.set_ylabel("Relative L2 error (%)")
ax.set_title("A generation error and a finite-basis limit are different problems")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(root / "figures/basis_floor.png", dpi=180)
plt.close(fig)

# Illustrative section of a single pre-specified representative task:
# seed 0, task nearest the median DataTime error among center-OOD tasks.
tasks = pd.read_csv(root / "analysis/task_results.csv")
subset = tasks.query("seed == 0 and method == 'DataTime' and group == 'center'")
idx = int(subset.loc[(subset.relative_l2 - subset.relative_l2.median()).abs().idxmin(), "task_id"])
bank = np.load(root / "data/test.npz")
from scipy.fft import dstn
p = bank["params"][idx]
xy = np.arange(1, 64) / 64
yy = int(np.argmin(abs(xy - p[1])))
reference = dstn(bank["ref_spectral"][idx] * 64, type=1, norm="ortho")
ii, jj = np.meshgrid(np.arange(1, 7), np.arange(1, 7), indexing="ij")
phi_slice = 2 * np.sin(np.pi * xy[:, None] * ii.ravel()) * np.sin(np.pi * xy[yy] * jj.ravel())

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(xy, reference[:, yy], label="High-resolution reference", linewidth=2)
for method, label in [("Response", "Physics + response compiler"), ("DataTime", "Time-matched data compiler")]:
    a = np.load(root / f"models/predictions_{method}_00.npz")["coefficients"][idx]
    ax.plot(xy, phi_slice @ a, label=label, linestyle="--")
ax.plot(xy, phi_slice @ bank["galerkin"][idx], label="Same 36 modes, numerical solve", linestyle=":")
ax.set_xlabel("x (dimensionless)")
ax.set_ylabel("u(x, y) (dimensionless)")
ax.set_title(f"One unseen-position task: ID {idx}, section y = {xy[yy]:.3f}")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(root / "figures/solution_section.png", dpi=180)
plt.close(fig)
print(f"4枚の図を保存しました。断面図の例はtask {idx}です。")