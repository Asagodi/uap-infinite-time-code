#!/usr/bin/env python3
"""
Simulate trajectories for the 1D bistable system example.
Shows true system f(x) = x - x³ and perturbed NODE f̂(x) = x - x³ + νg(x).
"""

import os
from uap import config
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# System parameters
def f_true(x, t):
    """True bistable system: f(x) = x - x³"""
    return x - x**3

def f_perturbed(x, t, nu=0.15):
    """Perturbed NODE: f̂(x) = x - x³ - ν*cos(2x)
    cos(2x)=1 at x=0, so perturbation is maximal at separatrix.
    Negative sign shifts separatrix to the RIGHT."""
    return x - x**3 - nu * np.cos(2 * x)

# Simulation parameters
t = np.linspace(0, 5, 200)

# Initial conditions to show basin mismatch
# With nu=0.15 and -cos(2x), separatrix shifts RIGHT from 0 to ~0.15
# So x0 in (0, 0.15) will switch basins: true→right(+1), perturbed→left(-1)
# Also include points above/below the fixed points at ±1
x0_compare = [0.02, -0.05, 0.05, 0.08, 0.12, -0.12, -0.2, 0.3, -0.5, 0.5, -1.3, -1.5, 1.3, 1.5]

# Figure sized to match LaTeX panel (0.5\textwidth ≈ 3.25 inches)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
plt.rcParams.update({'font.size': 12})

# Colors for basins
color_left = '#2166ac'   # blue for x → -1
color_right = '#b2182b'  # red for x → +1

# Panel A: True system trajectories
ax1 = axes[0]
for x0 in x0_compare:
    sol = odeint(f_true, x0, t).flatten()
    color = color_left if x0 < 0 else color_right
    ax1.plot(t, sol, color=color, alpha=0.7, lw=1.5)

ax1.axhline(y=-1, color=color_left, ls='--', lw=1, alpha=0.5)
ax1.axhline(y=1, color=color_right, ls='--', lw=1, alpha=0.5)
ax1.axhline(y=0, color='gray', ls=':', lw=1, alpha=0.5)

ax1.set_xlabel('$t$', fontsize=14)
ax1.set_ylabel('$x(t)$', fontsize=14)
ax1.set_title(r'True: $\dot{x} = x - x^3$', fontsize=14)
ax1.set_ylim(-2, 2)
ax1.set_xlim(0, 5)
ax1.tick_params(labelsize=12)

# Panel B: Comparison with perturbed system
ax2 = axes[1]

for x0 in x0_compare:
    sol_true = odeint(f_true, x0, t).flatten()
    color_true = color_left if x0 < 0 else color_right
    sol_pert = odeint(f_perturbed, x0, t).flatten()

    ax2.plot(t, sol_true, color=color_true, alpha=0.35, lw=1.5)
    ax2.plot(t, sol_pert, color=color_true, alpha=0.8, lw=1.5)

ax2.axhline(y=-1, color=color_left, ls='--', lw=1, alpha=0.3)
ax2.axhline(y=1, color=color_right, ls='--', lw=1, alpha=0.3)
ax2.axhline(y=0, color='gray', ls=':', lw=1, alpha=0.5)

# Highlight basin error region
ax2.axhspan(0, 0.15, alpha=0.25, color='orange')

ax2.set_xlabel('$t$', fontsize=14)
# ax2.set_ylabel('$x(t)$', fontsize=14)
ax2.set_title(r'Perturbed ($\nu=0.15$)', fontsize=14)
ax2.set_ylim(-2, 2)
ax2.set_xlim(0, 5)
ax2.tick_params(labelsize=12)

# # Compact legend
# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], color='gray', lw=2, alpha=0.35, label='True'),
#     Line2D([0], [0], color='gray', lw=2, alpha=0.8, label='Pert.'),
# ]
# ax2.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(config.FIGDIR, "bistable_trajectories.pdf"), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(config.FIGDIR, "bistable_trajectories.png"), bbox_inches='tight', dpi=150)
print("Saved figures/bistable_trajectories.pdf and .png")
plt.show()
