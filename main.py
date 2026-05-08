import numpy as np
import matplotlib.pyplot as plt

from sim import physics, initial_conditions, integrator, plotting

N     = 2000
DT    = 0.004
STEPS = 1800

print("running galaxy disc simulation...")
pos, vel, r0, theta0, sigma_r0 = initial_conditions.generate(N)
frames, t_Gyr, L_arr, vel_final = integrator.run(pos, vel, STEPS, DT)
print(f"done — {STEPS} steps = {STEPS*DT*physics.KPC_KMS_TO_GYR:.2f} Gyr, {len(frames)} frames")

L0 = L_arr[0]
L_drift_max = max(abs((L - L0) / abs(L0) * 100) for L in L_arr)
print(f"Mean angular momentum drift: {L_drift_max:.4f}%")

kepler_const = 4 * np.pi**2 / (physics.G * physics.M_BH)
for rt in [0.5, 1.0, 2.0]:
    T     = 2 * np.pi * rt / np.sqrt(physics.G * physics.M_BH / rt)
    ratio = T**2 / rt**3
    print(f"  r={rt} kpc  T={T*physics.KPC_KMS_TO_GYR*1e3:.1f} Myr  T2/a3={ratio:.4e}  theory={kepler_const:.4e}")

fig, sc, vline_L, disc_title = plotting.make_analysis_figure(
    frames, t_Gyr, L_arr, r0, vel_final, STEPS, DT, N)
plt.savefig("galaxy_analysis.png", dpi=150, bbox_inches='tight', facecolor='#060a12')
print("saved galaxy_analysis.png")

ani = plotting.make_animation(fig, sc, vline_L, disc_title, frames, t_Gyr, N)
plt.show()
