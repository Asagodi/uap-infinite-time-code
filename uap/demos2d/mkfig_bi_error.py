#!/usr/bin/env python3
"""
Estimate and visualize ε-δ closeness errors for the 1D bistable system.

This script accompanies the paper on universal approximation for dynamical
systems with infinite-time guarantees. It demonstrates:

1. ERROR DECOMPOSITION (Panel A):
   - Basin error δ: fraction of initial conditions that converge to different
     attractors in true vs. perturbed system (B-type error)
   - Trajectory error ε: mean trajectory deviation for ICs that stay in the
     same basin (used in ε-δ closeness definition)

2. L^p ERROR BOUND (Panel B, Theorem 5.1):
   The theoretical bound E_{p,∞} ≤ ε^p + δ·D^p decomposes total error into:
   - ε^p: contribution from trajectory mismatch on "good" set (1-δ of domain)
   - δ·D^p: contribution from basin switching on "bad" set (δ of domain)
   where D is the basin diameter (max distance within a basin).

3. BOUND DECOMPOSITION (Panel C):
   Visualizes how the two error sources (trajectory vs. basin) contribute
   to the total L^2 error as perturbation size ν increases.

System: ẋ = x - x³ (true) vs. ẋ = x - x³ - ν·cos(2x) (perturbed)
The perturbation g(x) = -cos(2x) is maximal at x=0 (the separatrix),
causing the perturbed separatrix to shift, inducing basin switching.
"""

import os
from uap import config
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# System definitions
def f_true(x, t):
    """True bistable system: f(x) = x - x³"""
    return x - x**3

def f_perturbed(x, t, nu):
    """Perturbed NODE: f̂(x) = x - x³ - ν·cos(2x)"""
    return x - x**3 - nu * np.cos(2 * x)

def get_fp_attractor(trajectory):
    """Determine which basin a trajectory ends up in: -1 (left) or +1 (right)"""
    return -1 if trajectory[-1] < 0 else 1

def compute_errors(nu, x0_grid, t, domain_length):
    """
    Compute basin error and trajectory error for perturbation size nu.

    Returns:
        basin_error: δ = volume fraction of ICs that switch basins
        traj_error: ε = max trajectory error for non-switching ICs
        mean_traj_error: mean trajectory error for non-switching ICs
    """
    dx = domain_length / len(x0_grid)  # spacing between samples
    switch_volume = 0.0
    traj_errors = []

    for x0 in x0_grid:
        # True trajectory
        sol_true = odeint(f_true, x0, t).flatten()
        basin_true = get_fp_attractor(sol_true)

        # Perturbed trajectory
        sol_pert = odeint(lambda x, t: f_perturbed(x, t, nu), x0, t).flatten()
        basin_pert = get_fp_attractor(sol_pert)

        if basin_true != basin_pert:
            switch_volume += dx  # accumulate actual volume
        else:
            # Compute trajectory error only for non-switching ICs
            traj_err = np.max(np.abs(sol_true - sol_pert))
            traj_errors.append(traj_err)

    # Normalize by domain length to get δ (volume fraction)
    basin_error = switch_volume / domain_length
    max_traj_error = np.max(traj_errors) if traj_errors else 0
    mean_traj_error = np.mean(traj_errors) if traj_errors else 0

    return basin_error, max_traj_error, mean_traj_error

def compute_lp_error(nu, x0_grid, t, p=2):
    """
    Compute the L^p error averaged over initial conditions and time.
    E_{p,T} = (1/|X|) ∫_X (1/T ∫_0^T |φ - φ̂|^p dt) dx
    """
    dt = t[1] - t[0]
    T = t[-1]

    lp_errors = []
    for x0 in x0_grid:
        sol_true = odeint(f_true, x0, t).flatten()
        sol_pert = odeint(lambda x, t: f_perturbed(x, t, nu), x0, t).flatten()

        # Time-averaged L^p error for this IC
        err_p = np.mean(np.abs(sol_true - sol_pert)**p)
        lp_errors.append(err_p)

    return np.mean(lp_errors)  # Average over ICs

# Simulation parameters
# Time constant: at stable fixed points x*=±1, f'(x*) = 1-3x² = -2, so τ = 1/2.
# We simulate for T=10 = 20τ, ensuring trajectories are within e^(-20) ≈ 2e-9
# of their attractors, which is sufficient for accurate L^p time averages.
t = np.linspace(0, 10, 500)
# Domain: [-1.8, 1.8] contains both basins of attraction.
# The true system has basins [-1.8, 0) → -1 and (0, 1.8] → +1.
x0_min, x0_max = -1.8, 1.8
x0_grid = np.linspace(x0_min, x0_max, 1000)  # Dense grid for accurate δ estimation
domain_length = x0_max - x0_min  # = 3.6, used for normalizing basin error

# Perturbation sizes span 3 orders of magnitude on log scale
nu_values = np.logspace(-3, -0.3, 50)  # 0.001 to ~0.5
nu_reference = 0.15  # Reference value used in trajectory figure (mkfig_bi_traj.py)

# Compute errors for each nu
basin_errors = []
mean_traj_errors = []
l2_errors = []

print("Computing errors...")
for i, nu in enumerate(nu_values):
    be, _, mean_te = compute_errors(nu, x0_grid, t, domain_length)
    l2e = compute_lp_error(nu, x0_grid, t, p=2)

    basin_errors.append(be)
    mean_traj_errors.append(mean_te)
    l2_errors.append(l2e)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(nu_values)} done")

basin_errors = np.array(basin_errors)
mean_traj_errors = np.array(mean_traj_errors)
l2_errors = np.array(l2_errors)

# Basin diameter for L^p error bound (D in Theorem 5.1)
# Basins are [-1.8, 0) and (0, 1.8], each with diameter 1.8
# In this symmetric case, basin_diameter = domain_length / 2
basin_diameter = domain_length / 2  # D = 1.8

# Figure sized to match LaTeX panel (0.23\textwidth ≈ 1.5 inches)
# Use larger size for legibility, will be scaled down
fig, ax = plt.subplots(figsize=(3.5, 3.5))
plt.rcParams.update({'font.size': 14})

# Theoretical bound decomposition: E_{2,∞} ≤ ε² + δ·D²
eps_term = mean_traj_errors**2
delta_term = basin_errors * basin_diameter**2
theoretical_bound = eps_term + delta_term

ax.fill_between(nu_values, 0, eps_term, alpha=0.5, color='red', label=r'$\varepsilon^2$')
ax.fill_between(nu_values, eps_term, theoretical_bound, alpha=0.5, color='blue', label=r'$\delta D^2$')
ax.plot(nu_values, theoretical_bound, 'g--', lw=2.5, label='Bound')
ax.plot(nu_values, l2_errors, 'k-', lw=2.5, label='Empirical')
ax.axvline(x=nu_reference, color='gray', ls=':', lw=2, alpha=0.8)
ax.set_xlabel(r'$\nu$', fontsize=16)
ax.set_ylabel(r'$L^2$ error', fontsize=16)
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax.set_xscale('log')
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=13)
ax.grid(True, alpha=0.3, which='major')

plt.tight_layout()
plt.savefig(os.path.join(config.FIGDIR, "bistable_errors.pdf"), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(config.FIGDIR, "bistable_errors.png"), bbox_inches='tight', dpi=150)
print("Saved figures/bistable_errors.pdf and .png")
plt.show()
