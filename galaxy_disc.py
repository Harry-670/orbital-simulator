import numpy as np
import matplotlib.pyplot as plt

G    = 4.302e-6  # kpc (km/s)^2 M_sun^-1
M_BH = 4.0e6     # M_sun
V_H  = 220.0     # km/s  isothermal halo flat speed (Milky Way)
EPS  = 0.05      # kpc

def v_circ(r):
    v_bh  = np.sqrt(G * M_BH / np.maximum(r, EPS))
    v_hal = V_H
    return np.sqrt(v_bh**2 + v_hal**2)

r_plot = np.linspace(0.1, 15, 300)
v_bh_arr = np.sqrt(G * M_BH / r_plot)
v_hal_arr = np.full_like(r_plot, V_H)
v_tot_arr = v_circ(r_plot)

fig, ax = plt.subplots(figsize=(9, 5), facecolor='#060a12')
ax.set_facecolor('#060a12')
ax.plot(r_plot, v_bh_arr,  '--', color='#FFD60A',  lw=1.2, label='BH (Keplerian)')
ax.plot(r_plot, v_hal_arr, '--', color='#4CC9F0',  lw=1.2, label='Isothermal halo')
ax.plot(r_plot, v_tot_arr, '-',  color='white',    lw=1.8, label='Total')
ax.set_xlabel("r (kpc)", color='#8B949E')
ax.set_ylabel("v_circ (km/s)", color='#8B949E')
ax.set_title("Milky Way rotation curve: BH + isothermal halo", color='white', fontweight='bold')
ax.tick_params(colors='#8B949E')
for sp in ax.spines.values(): sp.set_edgecolor('#21262D')
ax.legend(facecolor='#0d1117', labelcolor='white')
ax.grid(linestyle='--', alpha=0.15, color='white')
plt.tight_layout()
plt.savefig("rotation_curve.png", dpi=150, bbox_inches='tight', facecolor='#060a12')
plt.show()

print(f"v_circ(8 kpc) = {v_circ(8):.1f} km/s  (observed ~220 km/s)")
