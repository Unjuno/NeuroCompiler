
import math, random, time, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

OUT = Path("/mnt/data")
RESULTS_PATH = OUT / "knowledge_to_weights_stage0_results.csv"
SUMMARY_PATH = OUT / "knowledge_to_weights_stage0_summary.csv"
PLOT_PATH = OUT / "knowledge_to_weights_stage0_curve.png"

# ---- exploratory Stage-0 configuration ----
SEEDS = [0, 1, 2]
H = 16
P = H + H + H + 1                # 49 target-network parameters
META_STEPS = 300
META_BATCH = 4
META_INNER_STEPS = 2
INNER_LR = 0.05
SUPPORT_N = 12
QUERY_N = 32
EVAL_TASKS_PER_MASK = 80
EVAL_STEPS = [0, 1, 3, 5, 10]
THRESHOLD_MSE = 0.10

TRAIN_MASKS = torch.tensor([
    [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1]
], dtype=torch.float32)

# Never shown as structural masks during meta-training.
TEST_MASKS = torch.tensor([
    [1,0,1], [1,1,1]
], dtype=torch.float32)

def target_forward(theta, x):
    i = 0
    W1 = theta[i:i+H].view(H,1); i += H
    b1 = theta[i:i+H].view(1,H); i += H
    W2 = theta[i:i+H].view(1,H); i += H
    b2 = theta[i:i+1].view(1,1)
    return torch.tanh(x @ W1.T + b1) @ W2.T + b2

def true_function(c, x):
    # Analytic, dimensionless primitives:
    # A = sin(pi x), B = x, C = 2x^2 - 2/3
    return (
        c[0] * torch.sin(math.pi*x)
        + c[1] * x
        + c[2] * (2.0*x*x - 2.0/3.0)
    )

def sample_coeff(mask):
    signs = torch.where(torch.rand(3) > 0.5, 1.0, -1.0)
    mags = 0.5 + torch.rand(3)
    return mask * signs * mags

def sample_xy(c, n):
    x = 2.0*torch.rand(n,1) - 1.0
    return x, true_function(c, x)

def inner_adapt(theta0, c, steps, create_graph=False):
    th = theta0
    for _ in range(steps):
        x, y = sample_xy(c, SUPPORT_N)
        loss = ((target_forward(th, x) - y)**2).mean()
        (g,) = torch.autograd.grad(loss, th, create_graph=create_graph)
        th = th - INNER_LR*g
    return th

class HyperInit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.randn(P)*0.15)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,P),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self, c):
        return self.base + self.net(c)

def train_hyper():
    model = HyperInit()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(META_STEPS):
        opt.zero_grad()
        qlosses = []
        for _ in range(META_BATCH):
            c = sample_coeff(TRAIN_MASKS[random.randrange(len(TRAIN_MASKS))])
            # First-order meta-gradient approximation for speed.
            th = inner_adapt(model(c), c, META_INNER_STEPS, create_graph=False)
            xq, yq = sample_xy(c, QUERY_N)
            qlosses.append(((target_forward(th, xq)-yq)**2).mean())
        loss = torch.stack(qlosses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model

def train_fo_maml():
    theta = torch.nn.Parameter(torch.randn(P)*0.15)
    opt = torch.optim.Adam([theta], lr=1e-3)
    for _ in range(META_STEPS):
        opt.zero_grad()
        qlosses = []
        for _ in range(META_BATCH):
            c = sample_coeff(TRAIN_MASKS[random.randrange(len(TRAIN_MASKS))])
            th = inner_adapt(theta, c, META_INNER_STEPS, create_graph=False)
            xq, yq = sample_xy(c, QUERY_N)
            qlosses.append(((target_forward(th, xq)-yq)**2).mean())
        loss = torch.stack(qlosses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 1.0)
        opt.step()
    return theta.detach()

def adapt_eval(theta0, c, steps):
    th = theta0.detach().clone().requires_grad_(True)
    for _ in range(steps):
        x, y = sample_xy(c, SUPPORT_N)
        loss = ((target_forward(th, x)-y)**2).mean()
        (g,) = torch.autograd.grad(loss, th)
        th = (th - INNER_LR*g).detach().requires_grad_(True)
    return th

def evaluate(seed, hyper, fo_maml):
    rows = []
    xq = torch.linspace(-1, 1, 256).view(-1,1)
    for mi, mask in enumerate(TEST_MASKS):
        mask_name = "A+C" if mi == 0 else "A+B+C"
        for task_id in range(EVAL_TASKS_PER_MASK):
            c = sample_coeff(mask)
            yq = true_function(c, xq)
            init = {
                "HyperInit": hyper(c).detach(),
                "FO-MAML_shared": fo_maml.detach(),
                "Random": torch.randn(P)*0.15,
            }
            for method, theta0 in init.items():
                for steps in EVAL_STEPS:
                    th = theta0 if steps == 0 else adapt_eval(theta0, c, steps)
                    mse = ((target_forward(th, xq)-yq)**2).mean().item()
                    rows.append({
                        "seed": seed, "mask": mask_name, "task_id": task_id,
                        "method": method, "adapt_steps": steps, "mse": mse,
                        "success_mse_le_0.10": mse <= THRESHOLD_MSE
                    })
    return rows

all_rows = []
train_times = []
t_start = time.time()

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    t0 = time.time()
    hyper = train_hyper()
    t_h = time.time() - t0

    t0 = time.time()
    fo_maml = train_fo_maml()
    t_m = time.time() - t0

    train_times.append((seed, t_h, t_m))
    all_rows.extend(evaluate(seed, hyper, fo_maml))

df = pd.DataFrame(all_rows)
df.to_csv(RESULTS_PATH, index=False)

summary = (
    df.groupby(["method","adapt_steps"])["mse"]
      .agg(n="count", median="median", mean="mean",
           q25=lambda s: s.quantile(0.25),
           q75=lambda s: s.quantile(0.75))
      .reset_index()
)
succ = (
    df.groupby(["method","adapt_steps"])["success_mse_le_0.10"]
      .mean().reset_index(name="success_rate")
)
summary = summary.merge(succ, on=["method","adapt_steps"])

seedmed = (
    df.groupby(["seed","method","adapt_steps"])["mse"].median().reset_index()
    .groupby(["method","adapt_steps"])["mse"]
    .agg(seed_median="median", seed_min="min", seed_max="max").reset_index()
)
summary = summary.merge(seedmed, on=["method","adapt_steps"])
summary.to_csv(SUMMARY_PATH, index=False)

fig, ax = plt.subplots(figsize=(7,4.5))
for method in ["HyperInit","FO-MAML_shared","Random"]:
    s = summary[summary.method == method].sort_values("adapt_steps")
    ax.plot(s.adapt_steps, s["median"], marker="o", label=method)
ax.set_yscale("log")
ax.set_xlabel("Adaptation gradient steps")
ax.set_ylabel("Median held-out MSE")
ax.set_title("Unseen structural compositions: A+C and A+B+C")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=180)
plt.close(fig)

elapsed = time.time() - t_start
print("=== Stage-0 completed ===")
print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()}")
print(f"Platform: {platform.platform()}")
print(f"Total wall time: {elapsed:.2f} s")
print("Training times (s):", train_times)
print()
print(summary[["method","adapt_steps","n","median","q25","q75","success_rate","seed_min","seed_max"]]
      .to_string(index=False, float_format=lambda z: f"{z:.6f}"))
