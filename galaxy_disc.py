import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

G    = 4.302e-6
M_BH = 4.0e6
V_H  = 220.0
EPS  = 0.05
R_D  = 3.0

# Bar potential (quadrupole): phi_bar = -A * f(r) * cos(2*(theta - Omega_b*t))
# f(r) = (r/R_b)^2 exp(-r/R_b)  — peaks at r=2R_b, falls off beyond
A_BAR   = 1800.0   # (km/s)^2   bar strength
R_BAR   = 3.5      # kpc         bar half-length
OMEGA_B = 50.0     # km/s/kpc   bar pattern speed (corotation ~4.4 kpc)

N     = 2000
DT    = 0.004
STEPS = 1500
KPC_KMS_TO_GYR = 0.9778

rng = np.random.default_rng(7)
r_grid = np.linspace(0.3, 14.0, 50000)
pdf    = r_grid * np.exp(-r_grid / R_D)
cdf    = np.cumsum(pdf); cdf /= cdf[-1]
r = np.interp(rng.uniform(0, 1, N), cdf, r_grid)
theta = rng.uniform(0, 2 * np.pi, N)

def v_circ(r):
    return np.sqrt(G * M_BH / np.maximum(r, EPS) + V_H**2)

vc = v_circ(r)
sigma_v = 0.06 * vc
vr = rng.normal(0, sigma_v)
vt = vc + rng.normal(0, sigma_v)

pos = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
vel = np.column_stack([
    -vt * np.sin(theta) + vr * np.cos(theta),
     vt * np.cos(theta) + vr * np.sin(theta)
])

def accel(pos, t):
    x, y = pos[:, 0], pos[:, 1]
    r_sq  = x**2 + y**2 + EPS**2
    r_mag = np.sqrt(r_sq)

    # BH + halo (axisymmetric)
    a_x = -(G * M_BH / (r_sq * r_mag) + V_H**2 / r_sq) * x
    a_y = -(G * M_BH / (r_sq * r_mag) + V_H**2 / r_sq) * y

    # rotating bar: phi = -A * f(r) * cos(2*(theta - Omega_b*t))
    # f(r) = (r/R_b)^2 * exp(-r/R_b)
    # grad_x phi and grad_y phi via chain rule
    phi_bar = OMEGA_B * t        # current bar angle
    r_nosft = np.sqrt(x**2 + y**2) + 1e-9
    ang = np.arctan2(y, x)
    xi  = r_nosft / R_BAR
    f_r = xi**2 * np.exp(-xi)
    df_dr = (2/R_BAR * xi - xi**2 / R_BAR) * np.exp(-xi)
    cos2  = np.cos(2 * (ang - phi_bar))
    sin2  = np.sin(2 * (ang - phi_bar))

    # dPhi/dr  and  dPhi/dtheta / r
    dphi_dr  = -A_BAR * df_dr * cos2
    dphi_dth = -A_BAR * f_r * (-2 * sin2)     # = 2*A*f*sin2

    # convert to x,y
    cos_a = x / r_nosft; sin_a = y / r_nosft
    a_x += -(dphi_dr * cos_a - dphi_dth * sin_a / r_nosft)
    a_y += -(dphi_dr * sin_a + dphi_dth * cos_a / r_nosft)

    return np.column_stack([a_x, a_y])

print("running disc + bar simulation...")
acc = accel(pos, 0)
frames = []
t_list = []

for n in range(STEPS):
    t = n * DT
    pos += vel * DT + 0.5 * acc * DT**2
    acc_new = accel(pos, t + DT)
    vel += 0.5 * (acc + acc_new) * DT
    acc = acc_new
    if n % 10 == 0:
        frames.append(pos.copy())
        t_list.append(t * KPC_KMS_TO_GYR)

print(f"done - {STEPS} steps = {STEPS*DT*KPC_KMS_TO_GYR:.2f} Gyr")

fig, ax = plt.subplots(figsize=(7, 7), facecolor='black')
ax.set_facecolor('black')
sc = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=0.6, c='white', alpha=0.5)
ax.set_xlim(-14, 14); ax.set_ylim(-14, 14)
ax.set_aspect('equal'); ax.axis('off')
t_txt = ax.text(-13, 12, "", color='white', fontsize=10)
ax.set_title("Galaxy disc + rotating bar — spiral structure", color='white', pad=4)

def update(i):
    sc.set_offsets(frames[i])
    t_txt.set_text(f"t = {t_list[i]*1e3:.0f} Myr")
    return [sc, t_txt]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
plt.show()
