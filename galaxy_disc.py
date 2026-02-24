import numpy as np
import matplotlib.pyplot as plt

G    = 4.302e-6
M_BH = 4.0e6
V_H  = 220.0
EPS  = 0.05
R_D  = 3.0       # kpc  exponential disc scale length

def v_circ(r):
    return np.sqrt(G * M_BH / np.maximum(r, EPS) + V_H**2)

def kappa(r):
    # epicyclic frequency: kappa^2 = G*M/r^3 + 2*V_H^2/r^2
    # derived from kappa^2 = r d(Omega^2)/dr + 4*Omega^2
    return np.sqrt(G * M_BH / np.maximum(r, EPS)**3 + 2 * V_H**2 / r**2)

r_arr = np.linspace(0.5, 15, 300)
Omega = v_circ(r_arr) / r_arr
kap   = kappa(r_arr)

# kappa/Omega ratio:  =1 for Keplerian (closed ellipses), =sqrt(2) for flat curve
ratio = kap / Omega
print("kappa/Omega at 2 kpc:", ratio[np.argmin(np.abs(r_arr - 2))].round(3))
print("kappa/Omega at 8 kpc:", ratio[np.argmin(np.abs(r_arr - 8))].round(3))
print("  sqrt(2) =", np.sqrt(2).round(3), "(expected for flat rotation curve)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#060a12')
ax1, ax2 = axes
for ax in axes:
    ax.set_facecolor('#060a12'); ax.tick_params(colors='#8B949E')
    for sp in ax.spines.values(): sp.set_edgecolor('#21262D')

ax1.plot(r_arr, v_circ(r_arr), 'w-', lw=1.5, label='v_circ')
ax1.plot(r_arr, Omega * r_arr, '--', color='#4CC9F0', lw=1.0, label='Omega*r (= v_circ)')
ax1.set_xlabel("r (kpc)", color='#8B949E'); ax1.set_ylabel("km/s", color='#8B949E')
ax1.set_title("Rotation curve", color='white', fontweight='bold')
ax1.legend(facecolor='#0d1117', labelcolor='white')
ax1.grid(linestyle='--', alpha=0.15, color='white')

ax2.plot(r_arr, kap,   color='#F4A261', lw=1.5, label='κ (epicyclic)')
ax2.plot(r_arr, Omega, color='#4CC9F0', lw=1.5, label='Ω (angular)')
ax2.plot(r_arr, ratio, color='white',   lw=1.2, linestyle='--', label='κ/Ω')
ax2.axhline(np.sqrt(2), color='lime', lw=0.8, linestyle=':', alpha=0.6, label='√2 (flat curve limit)')
ax2.set_xlabel("r (kpc)", color='#8B949E'); ax2.set_ylabel("rad / [kpc/(km/s)]", color='#8B949E')
ax2.set_title("Epicyclic vs angular frequency", color='white', fontweight='bold')
ax2.legend(facecolor='#0d1117', labelcolor='white', fontsize=8)
ax2.grid(linestyle='--', alpha=0.15, color='white')
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("frequencies.png", dpi=150, bbox_inches='tight', facecolor='#060a12')
plt.show()
