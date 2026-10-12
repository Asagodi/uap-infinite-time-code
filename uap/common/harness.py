#!/usr/bin/env python3
"""
Shared harness for the Neural-ODE demonstrations accompanying
"Universal Approximation Theorems for Dynamical Systems with
Infinite-Time-Horizon Guarantees".

We train a Neural ODE  x' = f_hat_theta(x)  to approximate a target vector
field f in C^1 (regression on f AND on its Jacobian), then integrate both
flows and measure epsilon-delta closeness

    delta(eps) = measure{ x0 : sup_t || phi(t,x0) - phi_hat(t,x0) || > eps }.

The three targets exhibit the three error modes of the B/P/D taxonomy:
  duffing   -> B-type (basin mismatch, curved separatrix)
  vanderpol -> P-type (phase drift on a limit cycle)
  ring      -> D-type (discretization of a continuous attractor)

Everything is 2D so integration over dense initial-condition grids is cheap.
Run in the uap-infinite-time environment (see README).
"""
import os, glob, sys

# Windows console is cp1252; our prints use unicode (T̂, ≈, ...). Force UTF-8
# BEFORE importing torch (torch may write to the stream, after which
# reconfigure() would fail silently).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
DEVICE = "cpu"
DTYPE = torch.float64                     # float64 throughout (dynamics need it)
torch.set_default_dtype(DTYPE)

# ----------------------------------------------------------------------------
# matplotlib / seaborn style  (CMU font bootstrap: required on this machine)
# ----------------------------------------------------------------------------
def setup_style():
    import matplotlib
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import seaborn as sns
    _userfonts = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")
    for _f in fm.findSystemFonts(fontpaths=[_userfonts]):
        try:
            fm.fontManager.addfont(_f)
        except Exception:
            pass
    sns.set_theme(context="talk", style="ticks", font_scale=1.15)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
        "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
        "axes.linewidth": 1.2, "lines.linewidth": 2.2, "lines.markersize": 7,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.transparent": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # verify the font actually took
    got = os.path.basename(fm.findfont(fm.FontProperties(family="serif")))
    if got != "cmunrm.ttf":
        print(f"  [warn] serif font resolved to {got} (expected cmunrm.ttf)")
    return plt, sns

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def panel_labels(axes, letters="ABCDEFGH", fmt="({})", dy=6):
    """Bold panel labels flush with each panel's OUTER left edge.
    Call after fig.tight_layout(rect=[0,0,1,0.93])."""
    import matplotlib.pyplot as plt
    axes = np.atleast_1d(axes).ravel()
    fig = axes[0].figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    size = plt.rcParams["axes.titlesize"] * 1.05
    for n, ax in enumerate(axes):
        bb = ax.get_tightbbox(r).transformed(inv)
        y = ax.get_position().y1 + dy / (fig.get_size_inches()[1] * fig.dpi)
        fig.text(bb.x0, y, fmt.format(letters[n]), ha="left", va="bottom",
                 fontsize=size, fontweight="bold")

# ----------------------------------------------------------------------------
# Target vector fields:  numpy (batched) field + analytic Jacobian, torch field
# ----------------------------------------------------------------------------
class System:
    """A 2D target: field f, analytic Jacobian, a sampling/eval domain box."""
    def __init__(self, name, f_np, jac_np, f_torch, domain, T_eval, label):
        self.name = name
        self.f_np = f_np            # (N,2)->(N,2)
        self.jac_np = jac_np        # (N,2)->(N,2,2)
        self.f_torch = f_torch      # (N,2) torch ->(N,2) torch
        self.domain = domain        # ((xmin,xmax),(ymin,ymax))
        self.T_eval = T_eval        # integration horizon for eps-delta
        self.label = label


def _duffing(delta=0.5):
    def f(z):
        x, y = z[..., 0], z[..., 1]
        return np.stack([y, x - x**3 - delta * y], axis=-1)
    def jac(z):
        x, y = z[..., 0], z[..., 1]
        N = z.shape[0]
        J = np.zeros((N, 2, 2))
        J[:, 0, 0] = 0.0;          J[:, 0, 1] = 1.0
        J[:, 1, 0] = 1 - 3 * x**2; J[:, 1, 1] = -delta
        return J
    def ft(z):
        x, y = z[..., 0], z[..., 1]
        return torch.stack([y, x - x**3 - delta * y], dim=-1)
    return System("duffing", f, jac, ft, ((-2.0, 2.0), (-2.0, 2.0)), 30.0,
                  r"Duffing double well ($\dot x=y,\ \dot y=x-x^3-\delta y$)")


def _vanderpol(mu=1.0):
    def f(z):
        x, y = z[..., 0], z[..., 1]
        return np.stack([y, mu * (1 - x**2) * y - x], axis=-1)
    def jac(z):
        x, y = z[..., 0], z[..., 1]
        N = z.shape[0]
        J = np.zeros((N, 2, 2))
        J[:, 0, 0] = 0.0;                 J[:, 0, 1] = 1.0
        J[:, 1, 0] = -2 * mu * x * y - 1; J[:, 1, 1] = mu * (1 - x**2)
        return J
    def ft(z):
        x, y = z[..., 0], z[..., 1]
        return torch.stack([y, mu * (1 - x**2) * y - x], dim=-1)
    return System("vanderpol", f, jac, ft, ((-3.0, 3.0), (-3.5, 3.5)), 60.0,
                  r"Van der Pol ($\mu=%.1f$)" % mu)


def _ring():
    def f(z):
        x, y = z[..., 0], z[..., 1]
        r2 = x**2 + y**2
        return np.stack([(1 - r2) * x, (1 - r2) * y], axis=-1)
    def jac(z):
        x, y = z[..., 0], z[..., 1]
        N = z.shape[0]
        r2 = x**2 + y**2
        J = np.zeros((N, 2, 2))
        J[:, 0, 0] = 1 - r2 - 2 * x**2; J[:, 0, 1] = -2 * x * y
        J[:, 1, 0] = -2 * x * y;        J[:, 1, 1] = 1 - r2 - 2 * y**2
        return J
    def ft(z):
        x, y = z[..., 0], z[..., 1]
        r2 = x**2 + y**2
        return torch.stack([(1 - r2) * x, (1 - r2) * y], dim=-1)
    return System("ring", f, jac, ft, ((-1.6, 1.6), (-1.6, 1.6)), 40.0,
                  r"Ring attractor ($\dot z=(1-\|z\|^2)z$)")


SYSTEMS = {"duffing": _duffing, "vanderpol": _vanderpol, "ring": _ring}

# ----------------------------------------------------------------------------
# MLP vector field  (smooth activation -> C^1, as the theory requires)
# ----------------------------------------------------------------------------
class MLPField(nn.Module):
    def __init__(self, dim=2, width=64, depth=2, act=nn.Tanh):
        super().__init__()
        layers, d = [], dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# C^1 training: minimize  MSE(f_hat, f) + c1 * MSE(Jf_hat, Jf)
# ----------------------------------------------------------------------------
def sample_domain(domain, n, seed=0):
    rng = np.random.default_rng(seed)
    (xlo, xhi), (ylo, yhi) = domain
    xs = rng.uniform(xlo, xhi, n)
    ys = rng.uniform(ylo, yhi, n)
    return np.stack([xs, ys], axis=-1)


def model_jacobian(model, X):
    """Batched Jacobian of the model at rows of X (requires grad graph). (N,2,2)"""
    X = X.detach().requires_grad_(True)
    Y = model(X)                                   # (N,2)
    N = X.shape[0]
    J = torch.zeros(N, 2, 2, dtype=X.dtype)
    for i in range(2):
        g = torch.autograd.grad(Y[:, i].sum(), X, create_graph=True)[0]  # (N,2)
        J[:, i, :] = g
    return J


def train_field(system, width=64, depth=2, n_samples=8000, c1=1.0,
                epochs=4000, lr=2e-3, seed=0, verbose=True, batch_size=None):
    torch.manual_seed(seed)
    Xnp = sample_domain(system.domain, n_samples, seed=seed)
    X = torch.tensor(Xnp)
    Ytrue = torch.tensor(system.f_np(Xnp))
    Jtrue = torch.tensor(system.jac_np(Xnp))

    model = MLPField(2, width, depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    gen = torch.Generator().manual_seed(seed)

    for ep in range(epochs):
        if batch_size is None:
            idx = torch.arange(n_samples)
        else:
            idx = torch.randint(0, n_samples, (batch_size,), generator=gen)
        Xb, Yb, Jb = X[idx], Ytrue[idx], Jtrue[idx]
        opt.zero_grad()
        Yh = model(Xb)
        loss_f = ((Yh - Yb) ** 2).mean()
        if c1 > 0:
            Jh = model_jacobian(model, Xb)
            loss_j = ((Jh - Jb) ** 2).mean()
            loss = loss_f + c1 * loss_j
        else:
            loss_j = torch.tensor(0.0)
            loss = loss_f
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (ep + 1) % max(1, epochs // 8) == 0:
            print(f"    ep {ep+1:5d}  loss_f {loss_f.item():.2e}  "
                  f"loss_J {float(loss_j):.2e}")
    model.eval()
    return model


def field_errors(system, model, n_grid=200):
    """Sup-norm C^0 and C^1 vector-field errors over a dense grid (eta)."""
    (xlo, xhi), (ylo, yhi) = system.domain
    gx = np.linspace(xlo, xhi, n_grid)
    gy = np.linspace(ylo, yhi, n_grid)
    XX, YY = np.meshgrid(gx, gy)
    P = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    Ptt = torch.tensor(P)
    with torch.no_grad():
        Fh = model(Ptt).numpy()
    Ft = system.f_np(P)
    c0 = np.linalg.norm(Fh - Ft, axis=-1)
    Jh = model_jacobian(model, Ptt).detach().numpy()
    Jt = system.jac_np(P)
    c1 = np.linalg.norm((Jh - Jt).reshape(-1, 4), axis=-1)
    return dict(eta_c0_sup=c0.max(), eta_c0_mean=c0.mean(),
                eta_c1_sup=c1.max(), eta_c1_mean=c1.mean())


# ----------------------------------------------------------------------------
# Fast batched fixed-step RK4 integrator (all initial conditions at once)
# ----------------------------------------------------------------------------
def numpy_field_from_model(model):
    """Extract Linear/Tanh weights and evaluate the field in pure numpy.
    ~100x faster than calling torch per RK4 step on tiny batches."""
    Ws, bs, acts = [], [], []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            Ws.append(layer.weight.detach().numpy().copy())
            bs.append(layer.bias.detach().numpy().copy())
            acts.append(False)
        elif isinstance(layer, nn.Tanh):
            acts[-1] = True                        # activation after last Linear
    def fh(P):
        A = np.asarray(P, dtype=float)
        for W, b, a in zip(Ws, bs, acts):
            A = A @ W.T + b
            if a:
                A = np.tanh(A)
        return A
    return fh


def field_np_from_model(model):
    """Fast numpy field (weights extracted once)."""
    return numpy_field_from_model(model)


def rk4(field, X0, T, dt, save_every=1):
    """Integrate X' = field(X) for all rows of X0. Returns (times, traj) with
    traj shape (n_saved, N, 2)."""
    X = np.array(X0, dtype=float)
    n = int(round(T / dt))
    out, ts = [X.copy()], [0.0]
    t = 0.0
    for k in range(n):
        k1 = field(X)
        k2 = field(X + 0.5 * dt * k1)
        k3 = field(X + 0.5 * dt * k2)
        k4 = field(X + dt * k3)
        X = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if (k + 1) % save_every == 0:
            out.append(X.copy()); ts.append(t)
    return np.array(ts), np.array(out)


def sup_traj_error(system, model, X0, T=None, dt=0.01, save_every=5):
    """sup_t ||phi - phi_hat|| for each initial condition (row of X0)."""
    T = system.T_eval if T is None else T
    ft = lambda P: system.f_np(P)
    fh = field_np_from_model(model)
    ts, Yt = rk4(ft, X0, T, dt, save_every)
    _,  Yh = rk4(fh, X0, T, dt, save_every)
    d = np.linalg.norm(Yt - Yh, axis=-1)          # (n_saved, N)
    return ts, d.max(axis=0), Yt, Yh              # sup over time per IC


def eps_delta_curve(sup_err, eps_grid):
    """delta(eps) = fraction of ICs with sup error > eps."""
    return np.array([(sup_err > e).mean() for e in eps_grid])


if __name__ == "__main__":
    # smoke test: train small on duffing, report field error
    sys = SYSTEMS["duffing"]()
    print("smoke test: training width=32 on", sys.name)
    m = train_field(sys, width=32, depth=2, n_samples=3000, epochs=800, verbose=True)
    err = field_errors(sys, m)
    print("  field errors:", {k: round(v, 5) for k, v in err.items()})
    X0 = sample_domain(sys.domain, 200, seed=7)
    ts, se, _, _ = sup_traj_error(sys, m, X0, T=20.0)
    print(f"  sup-traj error over 200 ICs: median {np.median(se):.4f}  max {se.max():.4f}")
