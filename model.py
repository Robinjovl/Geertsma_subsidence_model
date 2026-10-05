import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from scipy.integrate import simpson

def geertsma_surface_subsidence(r, R, D, h, delta_P, E, nu, alpha=1.0, k_max=0.01, num_k=2000):
    """
    Calculates surface subsidence using Geertsma's (1973) disc reservoir model.
    Uses vectorized Simpson integration to avoid numerical instability spikes.
    """
    c_m = (alpha * (1.0 - 2.0 * nu) * (1.0 + nu)) / (E * (1.0 - nu))
    prefactor = 2.0 * c_m * h * delta_P * (1.0 - nu) * R

    r_abs = np.abs(r)
    r_flat = np.atleast_1d(r_abs)

    # Discretize wave-number domain k
    k = np.linspace(0, k_max, num_k)
    
    # Kernel computation: shape (num_k, num_r)
    kernel = j1(k[:, None] * R) * j0(k[:, None] * r_flat[None, :]) * np.exp(-k[:, None] * D)
    
    # Integrate over k using Simpson's rule
    integral = simpson(kernel, k, axis=0)
    
    subsidence = prefactor * integral
    return subsidence[0] if np.isscalar(r) else subsidence

if __name__ == "__main__":
    # Reservoir parameters
    R = 2000.0          # Reservoir radius [m]
    D = 3000.0          # Depth to center [m]
    h = 50.0            # Thickness [m]
    delta_P = 10e6      # Pore pressure drop = 10 MPa [Pa]
    E = 10e9            # Young's modulus = 10 GPa [Pa]
    nu = 0.25           # Poisson's ratio [-]
    alpha = 1.0         # Biot coefficient [-]

    # Symmetric distance vector (-8 km to +8 km)
    r_vec = np.linspace(-8000, 8000, 401)

    # Calculate subsidence profile
    subsidence = geertsma_surface_subsidence(r_vec, R, D, h, delta_P, E, nu, alpha)

    # Plot results
    plt.figure(figsize=(9, 5))
    plt.plot(r_vec / 1000.0, subsidence * 100, 'b-', linewidth=2, label="Subsidence Profile")
    
    # Mirror boundary markers
    plt.axvline(-R / 1000.0, color='r', linestyle='--', label=f"Reservoir Edges ($r = \\pm {R/1000:.1f}$ km)")
    plt.axvline(R / 1000.0, color='r', linestyle='--')

    plt.title("Geertsma Model: Symmetric Surface Subsidence Profile", fontsize=12, fontweight='bold')
    plt.xlabel("Radial Distance from Center $r$ [km]")
    plt.ylabel("Subsidence $S_z$ [cm]")
    plt.gca().invert_yaxis()  # Invert so subsidence points downwards
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Create 'results' directory if it doesn't exist and save the plot
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "geertsma_subsidence_profile.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"Figure successfully saved to: {os.path.abspath(save_path)}")

    plt.show()