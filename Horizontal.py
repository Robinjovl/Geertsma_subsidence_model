import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from scipy.special import j0, j1

def geertsma_displacements(r, R, D, h, delta_P, E, nu, alpha=1.0, k_max=0.02, num_k=2000):
    """
    Calculates vertical subsidence (S_z) and radial horizontal displacement magnitude (|U_r|)
    using Geertsma's (1973) disc reservoir model for a given Poisson's ratio (nu).
    """
    c_m = (alpha * (1.0 - 2.0 * nu) * (1.0 + nu)) / (E * (1.0 - nu))
    prefactor = 2.0 * c_m * h * delta_P * (1.0 - nu) * R

    r_abs = np.abs(r)
    r_flat = np.atleast_1d(r_abs)

    k = np.linspace(1e-8, k_max, num_k)
    k_R = k[:, None] * R
    k_r = k[:, None] * r_flat[None, :]
    exp_kD = np.exp(-k[:, None] * D)

    kernel_sz = j1(k_R) * j0(k_r) * exp_kD
    kernel_ur = j1(k_R) * j1(k_r) * exp_kD

    integral_sz = simpson(kernel_sz, k, axis=0)
    integral_ur = simpson(kernel_ur, k, axis=0)

    S_z = prefactor * integral_sz
    U_r_mag = prefactor * integral_ur

    return S_z, U_r_mag


if __name__ == "__main__":
    # =========================================================================
    # USER INPUT PARAMETERS
    # =========================================================================
    R = 1500.0         # Reservoir radius [m]
    D = 2150.0          # Depth to reservoir center [m]
    h = 100.0            # Reservoir thickness [m]
    delta_P = 10e6      # Pore pressure depletion [Pa] (10 MPa)
    E = 10e9            # Young's modulus [Pa] (10 GPa)
    alpha = 0.9         # Biot coefficient [-]

    # Gaussian Distribution parameters for Poisson's Ratio (nu)
    nu_mean = 0.25      # Mean
    nu_sigma = 0.05     # Standard deviation
    num_samples = 100   # Number of Monte Carlo realizations

    # Spatial Domain (-30 km to +30 km)
    x_min, x_max = -3000.0, 3000.0  # [m]
    num_points = 601
    r_vec = np.linspace(x_min, x_max, num_points)
    # =========================================================================

    # Sample Poisson's ratio from Gaussian distribution (clipped to physical bounds [0, 0.49])
    np.random.seed(42)
    nu_samples = np.random.normal(loc=nu_mean, scale=nu_sigma, size=num_samples)
    nu_samples = np.clip(nu_samples, 0.0, 0.49)

    # Compute displacements across samples
    sz_runs = []
    ur_runs = []

    for nu in nu_samples:
        S_z, U_r_mag = geertsma_displacements(r_vec, R, D, h, delta_P, E, nu, alpha)
        sz_runs.append(S_z)
        ur_runs.append(U_r_mag)

    sz_runs = np.array(sz_runs)
    ur_runs = np.array(ur_runs)

    # Compute mean and standard deviation profiles
    sz_mean = np.mean(sz_runs, axis=0)
    sz_std = np.std(sz_runs, axis=0)

    ur_mean = np.mean(ur_runs, axis=0)
    ur_std = np.std(ur_runs, axis=0)

    # =========================================================================
    # PLOTTING
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Subplot 1: Subsidence (A)
    ax1.plot(r_vec / 1000.0, sz_mean, color='tab:blue', linewidth=2, 
             label=fr"Mean Subsidence ($\mu_{{\nu}}={nu_mean}$)")
    ax1.fill_between(r_vec / 1000.0, sz_mean - sz_std, sz_mean + sz_std, 
                     color='tab:blue', alpha=0.25, label=fr"$\pm 1\sigma$ Range ($\sigma_{{\nu}}={nu_sigma}$)")
    ax1.set_title("Subsidence (A)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Subsidence (m)")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left')

    # Subplot 2: Horizontal Surface Displacement (B)
    ax2.plot(r_vec / 1000.0, ur_mean, color='tab:red', linewidth=2, 
             label=fr"Mean Horizontal Displacement ($\mu_{{\nu}}={nu_mean}$)")
    ax2.fill_between(r_vec / 1000.0, np.maximum(0, ur_mean - ur_std), ur_mean + ur_std, 
                     color='tab:red', alpha=0.25, label=fr"$\pm 1\sigma$ Range ($\sigma_{{\nu}}={nu_sigma}$)")
    ax2.set_title("Horizontal Surface Displacement (B)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Distance from Center (km)")
    ax2.set_ylabel("Displacement Magnitude (m)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    plt.tight_layout()

    # Save figure directly to results folder
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "geertsma_subsidence_displacement.png")

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Simulation completed across {num_samples} Gaussian samples of Poisson's ratio.")
    print(f"Figure updated and saved directly to: {os.path.abspath(save_path)}")