#!/usr/bin/env python3
"""
Size requirement: how wide must the MLP be for a target trajectory accuracy?

The paper's fixed-point bound needs a vector-field accuracy
    eta := ||f - f_hat||   with   eta <= eta0(eps) = (lambda/K) * eps,
i.e. the required VF accuracy is PROPORTIONAL to the trajectory tolerance eps
(lambda = contraction rate, K = transient constant of the target).

C^1 universal approximation gives an achievable VF accuracy as a function of
width,  eta(W) ~ C * W^{-p}  (p = s/d for an s-smooth target in d dims).

Composing the two inverts to a WIDTH requirement:
    W(eps) >= ( C * (K/lambda) / eps )^{1/p}.

This script (1) sweeps width and fits eta(W); (2) for the Duffing case measures
the achieved trajectory accuracy eps and confirms eps ~ (K/lambda) eta linearly,
pinning the constant; (3) reports W(eps).
"""
from uap import config
import os, json, numpy as np, torch
from uap.common import harness as H

OUT = config.FIGDIR
MOD = config.MODELDIR
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)
CACHE = os.path.join(MOD, "sizelaw_cache.json")

WIDTHS = [6, 10, 16, 24, 32, 48, 64, 96]
SYS_NAMES = ["duffing", "vanderpol", "ring"]

# fast-but-converged training config for the sweep
CFG = dict(depth=2, n_samples=6000, c1=0.5, epochs=3000, lr=2e-3,
           batch_size=1024)


def eta_of_width(system, width, seed=0):
    path = os.path.join(MOD, f"sweep_{system.name}_w{width}.pt")
    model = H.MLPField(2, width, CFG["depth"])
    if os.path.exists(path):
        model.load_state_dict(torch.load(path)); model.eval()
    else:
        model = H.train_field(system, width=width, depth=CFG["depth"],
                              n_samples=CFG["n_samples"], c1=CFG["c1"],
                              epochs=CFG["epochs"], lr=CFG["lr"],
                              seed=seed, verbose=False,
                              batch_size=CFG["batch_size"])
        torch.save(model.state_dict(), path)
    err = H.field_errors(system, model, n_grid=160)
    return model, err


def duffing_traj_eps(system, model):
    """95th-pct sup-traj error over basin-agreeing ICs (isolates trajectory
    epsilon from B-type basin flips)."""
    ng = 50
    (xlo, xhi), (ylo, yhi) = system.domain
    gx = np.linspace(xlo, xhi, ng); gy = np.linspace(ylo, yhi, ng)
    XX, YY = np.meshgrid(gx, gy)
    X0 = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    _, sup, Yt, Yh = H.sup_traj_error(system, model, X0, T=25.0, dt=0.01,
                                      save_every=5)
    agree = np.sign(Yt[-1][:, 0]) == np.sign(Yh[-1][:, 0])
    return np.percentile(sup[agree], 95)


def fit_powerlaw(W, eta):
    """log eta = log C - p log W."""
    lw, le = np.log(np.array(W)), np.log(np.array(eta))
    p, logC = np.polyfit(lw, le, 1)
    return -p, np.exp(logC)      # exponent p (eta ~ C W^-p), constant C


def main():
    results = {}
    for name in SYS_NAMES:
        system = H.SYSTEMS[name]()
        etas0, etas1, npar, teps = [], [], [], []
        print(f"[{name}] sweeping widths {WIDTHS}")
        for W in WIDTHS:
            model, err = eta_of_width(system, W)
            etas0.append(err["eta_c0_sup"]); etas1.append(err["eta_c1_sup"])
            npar.append(model.n_params())
            te = duffing_traj_eps(system, model) if name == "duffing" else np.nan
            teps.append(te)
            print(f"   W={W:3d}  params={model.n_params():5d}  "
                  f"eta_C0={err['eta_c0_sup']:.4e}  eta_C1={err['eta_c1_sup']:.4e}"
                  + (f"  traj-eps={te:.4e}" if name == 'duffing' else ""))
        p, C = fit_powerlaw(WIDTHS, etas0)
        print(f"   fit: eta_C0 ~ {C:.3e} * W^(-{p:.3f})")
        results[name] = dict(widths=WIDTHS, eta0=etas0, eta1=etas1,
                             npar=npar, teps=teps, p=p, C=C)

    # ---- Duffing: eps vs eta linear fit (pins K/lambda) ----------------------
    d = results["duffing"]
    eta = np.array(d["eta0"]); eps = np.array(d["teps"])
    slope = np.polyfit(eta, eps, 1)[0]          # eps ~ slope * eta ; slope=K/lambda
    p, C = d["p"], d["C"]
    def W_of_eps(target_eps):
        return (C * slope / target_eps) ** (1.0 / p)
    print("\n  Duffing eps ~ %.2f * eta  (slope = K/lambda)" % slope)
    for te in [0.1, 0.03, 0.01, 0.003]:
        print(f"    target eps={te:6.3f}  ->  need eta<= {te/slope:.2e}"
              f"  ->  width W(eps) >= {W_of_eps(te):.1f}")
    results["duffing_slope"] = slope

    with open(CACHE, "w") as fp:
        json.dump(results, fp, indent=2, default=float)

    # ---- figure --------------------------------------------------------------
    plt, sns = H.setup_style()
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: eta(W) power laws
    ax = axes[0]
    cols = {"duffing": H.OKABE_ITO[0], "vanderpol": H.OKABE_ITO[1],
            "ring": H.OKABE_ITO[2]}
    marks = {"duffing": "o", "vanderpol": "s", "ring": "^"}
    for name in SYS_NAMES:
        r = results[name]
        ax.loglog(r["widths"], r["eta0"], marks[name], color=cols[name], ms=8,
                  label=f"{name} ($p={r['p']:.2f}$)")
        Wf = np.array(r["widths"], float)
        ax.loglog(Wf, r["C"] * Wf ** (-r["p"]), "-", color=cols[name], lw=1.6,
                  alpha=0.7)
    ax.set_xlabel(r"MLP width $W$")
    ax.set_ylabel(r"VF error $\eta=\|f-\hat f\|_{C^0}$")
    ax.legend(fontsize=12, loc="lower left")

    # Panel B: achieved trajectory eps vs VF eta (Duffing), linear -> W(eps)
    ax = axes[1]
    order = np.argsort(eta)
    ax.plot(eta[order], eps[order], "o", color=H.OKABE_ITO[0], ms=8,
            label="measured")
    xline = np.linspace(0, eta.max() * 1.05, 50)
    ax.plot(xline, slope * xline, "-", color="k", lw=1.8,
            label=r"$\varepsilon=(K/\lambda)\,\eta$, slope $=%.2f$" % slope)
    ax.set_xlabel(r"VF error $\eta$")
    ax.set_ylabel(r"trajectory accuracy $\varepsilon$")
    ax.legend(fontsize=12, loc="upper left")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    H.panel_labels(axes)
    fig.savefig(os.path.join(OUT, "node_sizelaw.pdf"))
    fig.savefig(os.path.join(OUT, "node_sizelaw.png"))
    print("  saved figures/node_sizelaw.{pdf,png}")


if __name__ == "__main__":
    main()
