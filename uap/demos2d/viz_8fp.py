#!/usr/bin/env python3
"""Hero visualization of the 3D 8-fixed-point experiment: a trained Neural ODE
recovers all 8 attractors of x_i' = x_i - x_i^3 and their orthant basins.
Panels: (A) 3D trajectories of the LEARNED flow converging to the 8 cube
corners, colored by basin; (B) a basin slice (learned) vs the true quadrant
boundaries; (C) the eps-delta curve."""
from uap import config
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uap.common.harness import setup_style, OKABE_ITO, panel_labels, rk4, eps_delta_curve
from uap.demos2d import run_multi_arch_8fp as E
import numpy as np, torch
from itertools import product
torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
from mpl_toolkits.mplot3d import Axes3D  # noqa

CORNERS = E.CORNERS
# 8 distinct fills, Okabe-Ito but black -> gray (black reads as an empty region)
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#999999"]
def corner_idx(P):
    d = np.linalg.norm(P[:, None, :] - CORNERS[None], axis=-1)
    return d.argmin(1)

def main():
    plt, sns = setup_style()
    from matplotlib.colors import ListedColormap
    mpath = os.path.join(config.MODELDIR, "mlp_8fp_viz.pt")
    m = E.MLPField(width=64, depth=2)
    if os.path.exists(mpath):
        m.load_state_dict(torch.load(mpath)); m.eval(); print("loaded cached MLP")
    else:
        print("training MLP for the visualization...")
        m = E.train(m, n=12000, epochs=1800, bs=2500, tag="viz")
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        torch.save(m.state_dict(), mpath)
    fh = E.torch_field(m)

    fig = plt.figure(figsize=(15, 4.6))

    # ---- A: 3D trajectories of the learned flow, colored by basin ----
    axA = fig.add_subplot(1, 3, 1, projection="3d")
    rng = np.random.default_rng(3)
    X0 = rng.uniform(-1.45, 1.45, (500, 3))
    ts, Y = rk4(fh, X0, 22.0, 0.02, save_every=4)         # (nT, 500, 3)
    lab = corner_idx(Y[-1])
    for j in range(X0.shape[0]):
        axA.plot(Y[:, j, 0], Y[:, j, 1], Y[:, j, 2],
                 color=PAL[lab[j] % 8], lw=0.5, alpha=0.35)
    axA.scatter(CORNERS[:, 0], CORNERS[:, 1], CORNERS[:, 2], s=110, c="k",
                marker="o", depthshade=False, zorder=6, edgecolors="w", linewidths=0.8)
    axA.set_xlabel(r"$x_1$"); axA.set_ylabel(r"$x_2$"); axA.set_zlabel(r"$x_3$")
    axA.set_title("learned flow: 8 attractors", fontsize=15)
    axA.view_init(elev=20, azim=35)

    # ---- B: basin slice at x3 = +0.6 (learned), true quadrant lines ----
    axB = fig.add_subplot(1, 3, 2)
    g = np.linspace(-1.5, 1.5, 120)
    XX, YY = np.meshgrid(g, g)
    P = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, 0.6)], -1)
    _, Yb = rk4(fh, P, 22.0, 0.02, save_every=6)
    lb = corner_idx(Yb[-1]).reshape(XX.shape)
    axB.pcolormesh(XX, YY, lb, cmap=ListedColormap(PAL), vmin=0, vmax=7, shading="auto")
    axB.axhline(0, color="k", lw=1.6); axB.axvline(0, color="k", lw=1.6)  # true boundaries
    axB.set_xlabel(r"$x_1$"); axB.set_ylabel(r"$x_2$")
    axB.set_title(r"basins on slice $x_3{=}0.6$", fontsize=14, pad=12)

    # ---- C: eps-delta ----
    axC = fig.add_subplot(1, 3, 3)
    X0e = E.sample(4000, seed=123)
    _, Yt = rk4(lambda Q: E.f_true(Q), X0e, E.T_EVAL, 0.02, 5)
    _, Yh = rk4(fh, X0e, E.T_EVAL, 0.02, 5)
    sup = np.linalg.norm(Yt - Yh, axis=-1).max(0)
    eps = np.logspace(-4, 0.0, 70)
    axC.plot(eps, eps_delta_curve(sup, eps), color=OKABE_ITO[0], lw=2.6)
    axC.set_xscale("log")
    axC.axvline(np.median(sup), color="0.55", ls=":", lw=1.6)
    axC.text(np.median(sup) * 1.2, 0.62, "median\nsup-err", fontsize=10, color="0.4")
    axC.set_xlabel(r"$\varepsilon$"); axC.set_ylabel(r"$\delta(\varepsilon)$")
    axC.set_ylim(-0.02, 1.02)
    ba = (corner_idx(Yt[-1]) == corner_idx(Yh[-1])).mean()
    axC.set_title(f"basin agree {ba*100:.1f}%,  " + r"$\delta(0.3)=$" + f"{(sup>0.3).mean():.4f}",
                  fontsize=14, pad=12)

    sns.despine(ax=axB); sns.despine(ax=axC)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    axA.text2D(-0.04, 1.13, "(A)", transform=axA.transAxes, fontsize=17, fontweight="bold")
    axB.text(-0.20, 1.10, "(B)", transform=axB.transAxes, fontsize=17, fontweight="bold")
    axC.text(-0.20, 1.10, "(C)", transform=axC.transAxes, fontsize=17, fontweight="bold")
    out = os.path.join(config.FIGDIR, "node_8fp_hero")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    print(f"saved {out}.pdf/.png  | basin agree {ba*100:.2f}%")

if __name__ == "__main__":
    main()
