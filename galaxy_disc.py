import numpy as np
import matplotlib.pyplot as plt

# Unit system: kpc, km/s, M_sun
# G = 4.302e-3 pc (km/s)^2 M_sun^-1 = 4.302e-6 kpc (km/s)^2 M_sun^-1
G    = 4.302e-6  # kpc (km/s)^2 M_sun^-1
M_BH = 4.0e6     # M_sun  (Milky Way centre, Sgr A*)
EPS  = 0.05      # kpc  gravitational softening

N  = 200
DT = 0.005       # kpc/(km/s) ~ 4.9 Myr
STEPS = 500

rng = np.random.default_rng(0)
r     = rng.uniform(1.0, 10.0, N)
theta = rng.uniform(0, 2 * np.pi, N)

# perfect circular orbits around point mass
v_c = np.sqrt(G * M_BH / r)

x  = r * np.cos(theta);  y  = r * np.sin(theta)
vx = -v_c * np.sin(theta); vy = v_c * np.cos(theta)

pos = np.column_stack([x, y])
vel = np.column_stack([vx, vy])

def accel(pos):
    r_mag = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + EPS**2)
    return -G * M_BH / r_mag[:, None]**3 * pos

acc = accel(pos)
for _ in range(STEPS):
    pos += vel * DT + 0.5 * acc * DT**2
    acc_new = accel(pos)
    vel += 0.5 * (acc + acc_new) * DT
    acc = acc_new

fig, ax = plt.subplots(figsize=(7, 7), facecolor='black')
ax.set_facecolor('black')
ax.scatter(pos[:, 0], pos[:, 1], s=1, c='white', alpha=0.6)
ax.set_aspect('equal')
ax.set_title("Galactic disc — circular orbits (point mass only)", color='white')
ax.set_xlabel("x (kpc)", color='white'); ax.set_ylabel("y (kpc)", color='white')
ax.tick_params(colors='white')
plt.tight_layout(); plt.show()

print(f"v_c at 8 kpc = {np.sqrt(G*M_BH/8):.2f} km/s  (BH only, should be ~1.5 km/s)")
