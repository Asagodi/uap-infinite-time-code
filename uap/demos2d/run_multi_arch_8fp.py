#!/usr/bin/env python3
"""
Higher-dimensional multistability + architecture generality (B-type, 3D).

Target: a continuous-time 3-bit memory  x_i' = x_i - x_i^3  (i=1,2,3), i.e. a
product of three bistables, with 2^3 = 8 stable fixed points at the cube corners
(+-1,+-1,+-1).  This is the well-conditioned (field O(1)) continuous analogue of
the 3-bit flip-flop: multistability beyond the 2D examples.

We fit the SAME target with three C^1-dense architectures named in the paper and
show all three (i) recover all 8 sinks, (ii) reproduce the orthant basins, and
(iii) achieve eps-delta closeness -- empirical support for the theory's
"the guarantee holds for any C^1-dense hypothesis class" statement:
  - smooth-activation MLP   (tanh)
  - continuous-time RNN     x' = -x + W tanh(V x + b)
  - RBF network             x' = sum_k W_k exp(-||x-c_k||^2 / 2 s_k^2)   (C^inf)

iam env, float64, CPU.  Writes figures/node_multiarch_8fp.pdf and RESULTS numbers.
"""
from uap import config
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uap.common.harness import setup_style, OKABE_ITO, panel_labels, rk4, eps_delta_curve
import numpy as np
import torch, torch.nn as nn
from itertools import product
from scipy.optimize import root

torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)

DIM = 3
DOMAIN = [(-1.5, 1.5)] * DIM
T_EVAL = 25.0
CORNERS = np.array(list(product([-1.0, 1.0], repeat=DIM)))          # (8,3)

# ---------------------------------------------------------------- target
def f_true(P):
    P = np.asarray(P, float)
    return P - P**3

def jac_true(P):
    P = np.asarray(P, float)
    N = P.shape[0]
    J = np.zeros((N, DIM, DIM))
    for i in range(DIM):
        J[:, i, i] = 1.0 - 3.0 * P[:, i]**2
    return J

def f_true_torch(P):
    return P - P**3

# ---------------------------------------------------------------- architectures
class MLPField(nn.Module):
    def __init__(self, dim=DIM, width=64, depth=2):
        super().__init__()
        L, d = [], dim
        for _ in range(depth):
            L += [nn.Linear(d, width), nn.Tanh()]; d = width
        L += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        return self.net(x)

class C1RNNField(nn.Module):
    """x' = -x + W tanh(V x + b): continuous-time RNN, single hidden layer + leak."""
    def __init__(self, dim=DIM, hidden=128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.W = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return -x + self.W(torch.tanh(self.V(x)))

class RBFField(nn.Module):
    """x' = sum_k W_k exp(-||x-c_k||^2 / 2 s_k^2): Gaussian RBF network (C^inf)."""
    def __init__(self, dim=DIM, n_centers=256, domain=DOMAIN):
        super().__init__()
        lo = torch.tensor([d[0] for d in domain])
        hi = torch.tensor([d[1] for d in domain])
        self.centers = nn.Parameter(lo + (hi - lo) * torch.rand(n_centers, dim))
        self.log_s = nn.Parameter(torch.full((n_centers,), np.log(0.45)))
        self.W = nn.Linear(n_centers, dim, bias=True)
    def forward(self, x):
        d2 = ((x[:, None, :] - self.centers[None, :, :])**2).sum(-1)    # (N,K)
        phi = torch.exp(-0.5 * d2 / (torch.exp(self.log_s)[None, :]**2))
        return self.W(phi)

ARCHS = {
    "MLP (tanh)":    lambda: MLPField(width=96, depth=2),
    "C¹ RNN":        lambda: C1RNNField(hidden=160),
    "RBF":           lambda: RBFField(n_centers=256),
}

# ---------------------------------------------------------------- generic utils
def sample(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(lo, hi, n) for (lo, hi) in DOMAIN], axis=-1)

def model_jac(model, X):
    X = X.detach().requires_grad_(True)
    Y = model(X)
    N = X.shape[0]
    J = torch.zeros(N, DIM, DIM)
    for i in range(DIM):
        g = torch.autograd.grad(Y[:, i].sum(), X, create_graph=True)[0]
        J[:, i, :] = g
    return J

def train(model, n=15000, epochs=4000, lr=2e-3, c1=1.0, seed=0, bs=2500, tag=""):
    torch.manual_seed(seed)
    Xnp = sample(n, seed)
    X = torch.tensor(Xnp)
    Yt = torch.tensor(f_true(Xnp))
    Jt = torch.tensor(jac_true(Xnp))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randint(0, n, (bs,), generator=gen)
        opt.zero_grad()
        Yh = model(X[idx])
        lf = ((Yh - Yt[idx])**2).mean()
        Jh = model_jac(model, X[idx])
        lj = ((Jh - Jt[idx])**2).mean()
        (lf + c1 * lj).backward()
        opt.step(); sch.step()
        if (ep + 1) % max(1, epochs // 4) == 0:
            print(f"    [{tag}] ep {ep+1:5d}  loss_f {lf.item():.2e}  loss_J {lj.item():.2e}")
    print(f"    [{tag}] trained in {time.time()-t0:.1f}s")
    model.eval()
    return model

def torch_field(model):
    def fh(P):
        with torch.no_grad():
            return model(torch.tensor(np.asarray(P, float))).numpy()
    return fh

def field_errors(model, n_grid=25):
    g = [np.linspace(lo, hi, n_grid) for (lo, hi) in DOMAIN]
    P = np.stack([m.ravel() for m in np.meshgrid(*g, indexing="ij")], axis=-1)
    with torch.no_grad():
        Fh = model(torch.tensor(P)).numpy()
    c0 = np.linalg.norm(Fh - f_true(P), axis=-1)
    Jh = model_jac(model, torch.tensor(P)).detach().numpy()
    c1 = np.linalg.norm((Jh - jac_true(P)).reshape(-1, DIM * DIM), axis=-1)
    return dict(eta_c0=float(c0.max()), eta_c1=float(c1.max()))

def find_sinks(model, n_start=6):
    """Newton (scipy.root) from a grid of starts; dedupe; keep sinks (all Re<0)."""
    fh = torch_field(model)
    fun = lambda z: fh(z[None, :])[0]
    starts = np.stack([m.ravel() for m in np.meshgrid(
        *[np.linspace(lo, hi, n_start) for (lo, hi) in DOMAIN], indexing="ij")], axis=-1)
    sols = []
    for s in starts:
        r = root(fun, s, method="hybr", tol=1e-10)
        if r.success and np.linalg.norm(fun(r.x)) < 1e-7:
            sols.append(r.x)
    if not sols:
        return np.zeros((0, DIM)), 0
    sols = np.array(sols)
    # dedupe
    uniq = []
    for z in sols:
        if all(np.linalg.norm(z - u) > 1e-3 for u in uniq):
            uniq.append(z)
    uniq = np.array(uniq)
    # classify: sink iff all Jacobian eigenvalues have negative real part
    Js = model_jac(model, torch.tensor(uniq)).detach().numpy()
    sinks = np.array([u for u, J in zip(uniq, Js)
                      if np.all(np.linalg.eigvals(J).real < -1e-4)])
    return sinks, len(sinks)

def basin_label(final):
    """Snap endpoints to nearest cube corner index (0..7)."""
    d = np.linalg.norm(final[:, None, :] - CORNERS[None, :, :], axis=-1)
    return d.argmin(axis=1)

def eval_model(model):
    fe = field_errors(model)
    # eps-delta over an IC grid
    X0 = sample(4000, seed=123)
    ts_t, Yt = rk4(lambda P: f_true(P), X0, T_EVAL, 0.02, save_every=5)
    ts_h, Yh = rk4(torch_field(model), X0, T_EVAL, 0.02, save_every=5)
    sup = np.linalg.norm(Yt - Yh, axis=-1).max(axis=0)      # sup_t per IC
    lab_t = basin_label(Yt[-1]); lab_h = basin_label(Yh[-1])
    basin_agree = float((lab_t == lab_h).mean())
    sinks, n_sink = find_sinks(model)
    corner_err = (np.min(np.linalg.norm(sinks[:, None, :] - CORNERS[None], axis=-1), axis=1).max()
                  if n_sink else np.nan)
    return dict(**fe, sup=sup, basin_agree=basin_agree, n_sink=n_sink,
                sinks=sinks, corner_err=corner_err)

# ---------------------------------------------------------------- run
def main():
    plt, sns = setup_style()
    print("Target: x_i' = x_i - x_i^3 (3D); 8 sinks at cube corners.")
    import pickle
    cache = os.path.join(config.MODELDIR, "cache_multiarch.pkl")
    if os.path.exists(cache):
        res = pickle.load(open(cache, "rb")); print("loaded cached results")
    else:
        res = {}
        for name, ctor in ARCHS.items():
            print(f"\n=== {name} ===")
            m = train(ctor(), epochs=3000, tag=name)
            res[name] = eval_model(m)
            r = res[name]
            print(f"    eta_C0={r['eta_c0']:.3e} sinks={r['n_sink']}/8 "
                  f"basin_agree={r['basin_agree']*100:.2f}% median sup={np.median(r['sup']):.3e}")
        pickle.dump(res, open(cache, "wb"))

    names = list(res.keys())
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))

    # Panel A: eps-delta curves, log epsilon-axis (delta -> 1 as eps -> 0)
    axA = axes[0]
    eps = np.logspace(-4, 0.0, 80)
    for k, name in enumerate(names):
        axA.plot(eps, eps_delta_curve(res[name]["sup"], eps), color=OKABE_ITO[k],
                 lw=2.4, label=name)
    axA.set_xscale("log")
    axA.set_xlabel(r"$\varepsilon$"); axA.set_ylabel(r"$\delta(\varepsilon)$")
    axA.set_ylim(-0.02, 1.02); axA.legend(fontsize=12, loc="lower left")
    axA.set_title(r"all three reach $\varepsilon$-$\delta$ closeness", fontsize=14)

    # Panel B: fixed-point recovery -- each architecture's 8 sinks, distance to
    # the nearest true corner (log y). All 8 recovered, all accurate.
    axB = axes[1]
    rng = np.random.default_rng(0)
    for k, name in enumerate(names):
        s = res[name]["sinks"]
        d = np.min(np.linalg.norm(s[:, None, :] - CORNERS[None], axis=-1), axis=1)
        x = k + rng.uniform(-0.13, 0.13, len(d))
        axB.scatter(x, d, color=OKABE_ITO[k], marker=markers[k], s=62,
                    edgecolors="w", linewidths=0.6, zorder=3)
        axB.text(k, 0.97, f"{res[name]['n_sink']}/8", ha="center", va="top",
                 fontsize=12, transform=axB.get_xaxis_transform())
    axB.set_yscale("log")
    axB.set_xticks(range(len(names))); axB.set_xticklabels(names, fontsize=12)
    axB.set_xlim(-0.5, len(names) - 0.5)
    axB.set_ylabel(r"$\|\mathrm{sink} - \mathrm{nearest\ corner}\|$")
    axB.set_title("all 8 fixed points recovered", fontsize=14)

    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    # manual panel labels at the same size as the hero row (viz_8fp.py, 17 pt)
    axA.text(-0.13, 1.10, "(D)", transform=axA.transAxes, fontsize=17, fontweight="bold")
    axB.text(-0.16, 1.10, "(E)", transform=axB.transAxes, fontsize=17, fontweight="bold")
    out = os.path.join(config.FIGDIR, "node_multiarch_8fp")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    print(f"\nsaved {out}.pdf / .png")

    print("\n--- summary ---")
    for n in names:
        r = res[n]
        print(f"{n:10s}: eta_C0={r['eta_c0']:.2e} sinks={r['n_sink']}/8 "
              f"basin={r['basin_agree']*100:.2f}% delta(0.3)={(r['sup']>0.3).mean():.4f}")

if __name__ == "__main__":
    main()
