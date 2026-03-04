import numpy as np
import matplotlib.pyplot as plt

G    = 4.302e-6
M_BH = 4.0e6
V_H  = 220.0
EPS  = 0.05
R_D  = 3.0
# disc mass sets the surface density
M_DISC = 5.0e10  # M_sun  (Milky Way stellar disc mass)

def v_circ(r):
    return np.sqrt(G * M_BH / np.maximum(r, EPS) + V_H**2)

def kappa(r):
    return np.sqrt(G * M_BH / np.maximum(r, EPS)**3 + 2 * V_H**2 / r**2)

def sigma_surface(r):
    # exponential disc: Sigma(r) = M/(2*pi*R_D^2) * exp(-r/R_D)
    return M_DISC / (2 * np.pi * R_D**2) * np.exp(-r / R_D)

def toomre_Q(r, sigma_r):
    # Q = sigma_r * kappa / (pi * G * Sigma)
    # sigma_r: radial velocity dispersion [km/s]
    return sigma_r * kappa(r) / (np.pi * G * sigma_surface(r))

r_arr = np.linspace(0.5, 15, 300)

# compute sigma_r needed for marginal stability Q=1
sigma_Q1 = np.pi * G * sigma_surface(r_arr) / kappa(r_arr)

print("Radial dispersion for Q=1:")
for r_check in [2, 4, 8]:
    idx = np.argmin(np.abs(r_arr - r_check))
    print(f"  r={r_check} kpc: sigma_r = {sigma_Q1[idx]:.1f} km/s")

sigma_r_actual = 40.0  # km/s  typical Milky Way thin disc

Q_arr = toomre_Q(r_arr, sigma_r_actual)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#060a12')
ax1, ax2 = axes
for ax in axes:
    ax.set_facecolor('#060a12'); ax.tick_params(colors='#8B949E')
    for sp in ax.spines.values(): sp.set_edgecolor('#21262D')

ax1.semilogy(r_arr, sigma_surface(r_arr), color='#4CC9F0', lw=1.5)
ax1.set_xlabel("r (kpc)", color='#8B949E')
ax1.set_ylabel("Σ(r)  [M☉/kpc²]", color='#8B949E')
ax1.set_title("Exponential surface density  Σ ∝ exp(−r/Rd)", color='white', fontweight='bold')
ax1.grid(linestyle='--', alpha=0.15, color='white')

ax2.plot(r_arr, Q_arr, color='white', lw=1.6, label=f'Q  (σᵣ={sigma_r_actual} km/s)')
ax2.axhline(1.0, color='#E63946', lw=1.0, linestyle='--', label='Q = 1  (marginal stability)')
ax2.fill_between(r_arr, 0, Q_arr, where=(Q_arr < 1), alpha=0.2, color='red', label='unstable Q < 1')
ax2.fill_between(r_arr, 0, Q_arr, where=(Q_arr >= 1), alpha=0.1, color='#4CC9F0', label='stable Q ≥ 1')
ax2.set_ylim(0, 6)
ax2.set_xlabel("r (kpc)", color='#8B949E')
ax2.set_ylabel("Toomre Q", color='#8B949E')
ax2.set_title("Toomre Q stability parameter  Q = σᵣκ / πGΣ", color='white', fontweight='bold')
ax2.legend(facecolor='#0d1117', labelcolor='white', fontsize=8)
ax2.grid(linestyle='--', alpha=0.15, color='white')

plt.tight_layout()
plt.savefig("toomre_Q.png", dpi=150, bbox_inches='tight', facecolor='#060a12')
plt.show()
