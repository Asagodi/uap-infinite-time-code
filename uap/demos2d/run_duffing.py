#!/usr/bin/env python3
"""
System 1 -- B-type error on a bistable system with a CURVED separatrix.

Target: damped Duffing double well
    x' = y ,  y' = x - x^3 - delta*y        (delta = 0.5)
Two stable spirals at (+-1, 0), a saddle at the origin whose stable manifold is
the (curved) separatrix.  We train a Neural ODE to match the field in C^1 and
show:
  * epsilon-delta closeness holds (delta(eps) collapses to a thin residual),
  * the residual error is B-TYPE: it is confined to a thin collar around the
    curved separatrix and has magnitude ~ D (the inter-well distance),
  * basins agree everywhere except that collar.
"""
from uap import config
import os, numpy as np, torch
from uap.common import harness as H

OUT = config.FIGDIR
MOD = config.MODELDIR
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)

DELTA = 0.5
sys = H._duffing(DELTA)


def get_model(width=64, depth=2, epochs=5000, retrain=False):
    path = os.path.join(MOD, f"duffing_w{width}_d{depth}.pt")
    model = H.MLPField(2, width, depth)
    if os.path.exists(path) and not retrain:
        model.load_state_dict(torch.load(path)); model.eval()
        print(f"  loaded {path}")
        return model
    print(f"  training duffing width={width} depth={depth} epochs={epochs}")
    model = H.train_field(sys, width=width, depth=depth, n_samples=8000,
                          c1=1.0, epochs=epochs, lr=2e-3, seed=0)
    torch.save(model.state_dict(), path)
    return model


def trace_separatrix(field, delta=DELTA, tmax=6.0, dt=0.002):
    """Stable manifold of the saddle at origin = separatrix, by backward
    integration along the stable eigendirection."""
    J0 = np.array([[0.0, 1.0], [1.0, -delta]])
    w, V = np.linalg.eig(J0)
    vs = V[:, np.argmin(w)]                        # stable eigenvector (lambda<0)
    vs = vs / np.linalg.norm(vs)
    branches = []
    for sgn in (+1, -1):
        X0 = (1e-4 * sgn * vs)[None, :]
        # backward flow of f  ==  forward flow of -f
        _, Y = H.rk4(lambda P: -field(P), X0, tmax, dt, save_every=1)
        branches.append(Y[:, 0, :])
    return np.vstack([branches[0][::-1], branches[1]])


def basin_label(Yend):
    """Endpoint -> which well (sign of x)."""
    return np.sign(Yend[:, 0])


def main():
    model = get_model()
    err = H.field_errors(sys, model)
    print("  field errors:", {k: round(v, 5) for k, v in err.items()})

    # ---- epsilon-delta over a dense IC grid ---------------------------------
    (xlo, xhi), (ylo, yhi) = sys.domain
    ng = 90
    gx = np.linspace(xlo, xhi, ng); gy = np.linspace(ylo, yhi, ng)
    XX, YY = np.meshgrid(gx, gy)
    X0 = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    ts, sup, Yt, Yh = H.sup_traj_error(sys, model, X0, T=sys.T_eval, dt=0.01,
                                       save_every=5)
    # basin agreement (endpoints)
    bt = basin_label(Yt[-1]); bh = basin_label(Yh[-1])
    agree = (bt == bh)
    print(f"  basin agreement: {agree.mean()*100:.2f}%  "
          f"(mismatch {np.sum(~agree)}/{len(agree)} ICs)")
    D = 2.0                                          # inter-well distance
    print(f"  sup-traj error: median {np.median(sup):.4f}  "
          f"95th pct {np.percentile(sup,95):.4f}  max {sup.max():.4f}")
    print(f"  error on basin-agreeing ICs: median {np.median(sup[agree]):.4f}  "
          f"max {sup[agree].max():.4f}")
    print(f"  error on basin-mismatch ICs: median "
          f"{np.median(sup[~agree]) if (~agree).any() else float('nan'):.4f}")

    eps_grid = np.logspace(-3, np.log10(3), 60)
    delta = H.eps_delta_curve(sup, eps_grid)

    # ---- separatrices --------------------------------------------------------
    sep_true = trace_separatrix(sys.f_np)
    sep_hat = trace_separatrix(H.field_np_from_model(model))

    # ---- figure --------------------------------------------------------------
    plt, sns = H.setup_style()
    from matplotlib.ticker import MaxNLocator
    import matplotlib.colors as mcolors
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: sup-error heatmap + separatrices
    ax = axes[0]
    Z = sup.reshape(ng, ng)
    pc = ax.pcolormesh(XX, YY, Z, shading="auto", cmap="rocket",
                       norm=mcolors.LogNorm(vmin=1e-3, vmax=D))
    ax.plot(sep_true[:, 0], sep_true[:, 1], color="white", lw=2.4,
            label="true separatrix")
    ax.plot(sep_hat[:, 0], sep_hat[:, 1], color=H.OKABE_ITO[3], lw=1.8,
            ls="--", label="NODE separatrix")
    ax.plot([1, -1], [0, 0], "o", color="w", ms=9, mec="k", mew=1.2)
    ax.plot([0], [0], "s", color="w", ms=9, mec="k", mew=1.2)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.legend(loc="upper left", fontsize=12, labelcolor="k")
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\sup_t\,\|\varphi-\hat\varphi\|$", fontsize=14)

    # Panel B: eps-delta curve
    ax = axes[1]
    ax.semilogx(eps_grid, delta, color=H.OKABE_ITO[0], lw=2.6)
    ax.axhline(1 - agree.mean(), color=H.OKABE_ITO[1], ls=":", lw=2,
               label=r"basin-flip floor $=%.3f$" % (1 - agree.mean()))
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\delta(\varepsilon)=\mu\{\sup_t\|\varphi-\hat\varphi\|>\varepsilon\}$",
                  fontsize=14)
    ax.set_ylim(-0.02, max(0.1, 1.1 * delta.max()))
    ax.legend(loc="upper right", fontsize=13)

    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    # internal (A)/(B) sub-labels removed: this figure is one cell of the main-text 2x2
    fig.savefig(os.path.join(OUT, "node_duffing_btype.pdf"))
    fig.savefig(os.path.join(OUT, "node_duffing_btype.png"))
    print("  saved figures/node_duffing_btype.{pdf,png}")

    # summary numbers for the write-up
    print("\n  SUMMARY (B-type, Duffing):")
    print(f"    eta_C0(sup)={err['eta_c0_sup']:.4f}  eta_C1(sup)={err['eta_c1_sup']:.4f}")
    print(f"    basin agreement {agree.mean()*100:.2f}% ; "
          f"delta(eps=0.3)={(sup>0.3).mean():.4f} ; "
          f"delta(eps=1.0)={(sup>1.0).mean():.4f}")


if __name__ == "__main__":
    main()
