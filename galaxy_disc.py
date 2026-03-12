import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G    = 4.302e-6
M_BH = 4.0e6
V_H  = 220.0
EPS  = 0.05
R_D  = 3.0

N     = 1500
DT    = 0.005    # kpc/(km/s) ~ 4.9 Myr
STEPS = 1200
KPC_KMS_TO_GYR = 0.9778

rng = np.random.default_rng(42)

# sample r from Sigma(r) ∝ r exp(-r/R_D) using CDF inversion
r_grid = np.linspace(0.3, 14.0, 50000)
pdf    = r_grid * np.exp(-r_grid / R_D)
cdf    = np.cumsum(pdf); cdf /= cdf[-1]
r = np.interp(rng.uniform(0, 1, N), cdf, r_grid)
theta = rng.uniform(0, 2 * np.pi, N)

def v_circ(r):
    return np.sqrt(G * M_BH / np.maximum(r, EPS) + V_H**2)

# initial velocities: circular + 5% radial dispersion
vc = v_circ(r)
sigma_v = 0.05 * vc
vr = rng.normal(0, sigma_v)
vt = vc + rng.normal(0, sigma_v)

pos = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
vel = np.column_stack([
    -vt * np.sin(theta) + vr * np.cos(theta),
     vt * np.cos(theta) + vr * np.sin(theta)
])

def accel(pos):
    r_mag2 = pos[:, 0]**2 + pos[:, 1]**2 + EPS**2
    r_mag  = np.sqrt(r_mag2)
    # BH + isothermal halo
    a_bh  = -G * M_BH / (r_mag2 * r_mag)[:, None] * pos
    a_hal = -V_H**2  / r_mag2[:, None] * pos
    return a_bh + a_hal

print("running disc simulation...")
acc = accel(pos)
frames = []
L_list = []

for n in range(STEPS):
    pos += vel * DT + 0.5 * acc * DT**2
    acc_new = accel(pos)
    vel += 0.5 * (acc + acc_new) * DT
    acc = acc_new

    L_list.append(np.mean(pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))

    if n % 12 == 0:
        frames.append(pos.copy())

t_total = STEPS * DT * KPC_KMS_TO_GYR
print(f"done - {STEPS} steps = {t_total:.2f} Gyr, {len(frames)} frames")

L0 = L_list[0]
L_drift = np.array([(L - L0) / abs(L0) * 100 for L in L_list])
print(f"Angular momentum drift: max {np.max(np.abs(L_drift)):.4f}%")

fig, ax = plt.subplots(figsize=(7, 7), facecolor='black')
ax.set_facecolor('black')
sc = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=0.5, c='white', alpha=0.5)
ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
ax.set_aspect('equal'); ax.axis('off')
t_txt = ax.text(-14, 13, "", color='white', fontsize=9)

def update(i):
    sc.set_offsets(frames[i])
    t_txt.set_text(f"t = {i*12*DT*KPC_KMS_TO_GYR*1e3:.0f} Myr")
    return [sc, t_txt]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
plt.show()
