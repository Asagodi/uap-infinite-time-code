#!/usr/bin/env python3
"""Exp 2 (real trained model): fixed-point structure of the CtD-toolkit 3-bit
flip-flop network. Loads a task-trained 3BFF model, runs the toolkit's
FixedPointFinder (zero input), fits the linear latent->output readout, and shows
that the stable fixed points project to the 8 cube corners = the 8 memory states.
This is the theory's F_FP (multistable) target class in a real trained model,
the B-type analogue of the BTCA head-direction ring (D-type).

Run in the CtD environment:  python analyze_ctd_3bff.py
"""
import os
from uap import config
config.chdir_ctd()   # chdir into the CtD toolkit (set CTD_TOOLKIT_DIR in .env)
import numpy as np
from itertools import product
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
_uf = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")
for _f in fm.findSystemFonts(fontpaths=[_uf]):
    try: fm.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams.update({"font.family": "serif", "font.serif": ["CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "figure.dpi": 140, "savefig.dpi": 300})
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#999999"]
FIGDIR = config.FIGDIR
CORNERS = np.array(list(product([-1., 1.], repeat=3)))

from ctd.comparison.analysis.tt.tt import Analysis_TT
BASE = "content/trained_models/task-trained/20260712_3bff"
RUN = "latent_size=64_num_workers=0_num_workers=0_max_epochs=500_seed=0"
tt = Analysis_TT("s0", os.path.join(BASE, RUN) + "/", use_train_dm=True)

# linear readout latent -> output, fit on validation trajectories
out = tt.get_model_outputs(phase="val")
lat = out["latents"].detach().numpy().reshape(-1, 64)
ctr = out["controlled"].detach().numpy().reshape(-1, 3)
A = np.concatenate([lat, np.ones((len(lat), 1))], 1)
W, *_ = np.linalg.lstsq(A, ctr, rcond=None)              # (65,3)
readout = lambda X: np.concatenate([X, np.ones((len(X), 1))], 1) @ W
_r2 = 1 - ((ctr - A @ W) ** 2).sum() / ((ctr - ctr.mean(0)) ** 2).sum()
print(f"[check] linear readout R^2 = {_r2:.4f}  (need ~1 to trust the output projection)")
assert _r2 > 0.9, "linear readout fit too poor; output projection not trustworthy"

# FixedPointFinder, zero input (cached so re-analysis is instant)
fpcache = os.path.join(config.MODELDIR, "ctd_3bff_fps.npz")
if os.path.exists(fpcache):
    d = np.load(fpcache); xs, q, stab = d["xs"], d["q"], d["stab"]; print("loaded cached FPs")
else:
    fps = tt.compute_FPs(n_inits=512, max_iters=3000, zero_inputs=True, compute_jacobians=True)
    xs = np.asarray(fps.xstar); q = np.asarray(fps.qstar).ravel()
    stab = (np.asarray(fps.is_stable).ravel().astype(bool)
            if getattr(fps, "is_stable", None) is not None else np.ones(len(xs), bool))
    os.makedirs(os.path.dirname(fpcache), exist_ok=True)
    np.savez(fpcache, xs=xs, q=q, stab=stab)

X = xs[(q < 1e-4) & stab]                                 # stable, genuinely slow
ro = readout(X)                                          # (K,3) outputs

# memory state = binary pattern vs the per-bit midpoint (robust to +-1 vs 0/1 convention)
thr = ctr.mean(0)                                        # midpoint per bit, from task outputs
lo, hi = ctr.min(0), ctr.max(0)
fp_bits = (ro > thr).astype(int)                         # (K,3)
patterns = np.unique(fp_bits, axis=0)
import collections
cnt = collections.Counter(map(tuple, fp_bits.tolist()))
print(f"stable slow fixed points: {len(X)}")
print(f"output range per bit: lo={np.round(lo,2)} hi={np.round(hi,2)} thr={np.round(thr,2)}")
print(f"distinct memory states realized by stable FPs: {len(patterns)}/8")
print("per-state FP counts:", {k: v for k, v in sorted(cnt.items())})

fig, ax = plt.subplots(1, 1, figsize=(5.4, 5.0))
c3 = np.where(fp_bits[:, 2] > 0, OKABE[0], OKABE[1])
ax.scatter(ro[:, 0], ro[:, 1], c=c3, s=40, alpha=0.8, zorder=2)
ax.axvline(thr[0], color="0.85", lw=1); ax.axhline(thr[1], color="0.85", lw=1)
loc = np.array([ro[ro[:, j] < thr[j], j].mean() for j in range(3)])   # low level per bit
hic = np.array([ro[ro[:, j] > thr[j], j].mean() for j in range(3)])   # high level per bit
for b1 in (loc[0], hic[0]):
    for b2 in (loc[1], hic[1]):
        ax.scatter([b1], [b2], s=340, facecolors="none", edgecolors="0.4", lw=2.0, zorder=1)
ax.set_xlabel("readout bit 1"); ax.set_ylabel("readout bit 2")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "ctd_3bff_fps.pdf"))
fig.savefig(os.path.join(FIGDIR, "ctd_3bff_fps.png"))
print("saved ctd_3bff_fps.pdf/.png")
