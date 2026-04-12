import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# ── constants (kpc, km/s, M_sun) ─────────────────────────────────────────────
G    = 4.302e-6   # kpc (km/s)^2 M_sun^-1
KPC_KMS_TO_GYR = 0.9778  # 1 kpc/(km/s) = 0.978 Gyr

# ── galaxy parameters ─────────────────────────────────────────────────────────
M_BH    = 4.0e6    # M_sun  (Sgr A*)
V_H     = 220.0    # km/s   isothermal halo flat speed
M_DISC  = 5.0e10   # M_sun  total disc mass (for Toomre Sigma)
R_D     = 3.0      # kpc    exponential disc scale length
EPS     = 0.05     # kpc    gravitational softening

# bar
A_BAR   = 1800.0   # (km/s)^2
R_BAR   = 3.5      # kpc
OMEGA_B = 50.0     # km/s/kpc

N     = 2000
DT    = 0.004      # kpc/(km/s) ~ 3.9 Myr
STEPS = 1800

# ── analytical functions ──────────────────────────────────────────────────────
def v_circ(r):
    return np.sqrt(G * M_BH / np.maximum(r, EPS) + V_H**2)

def kappa(r):
    # kappa^2 = d(Omega^2)/d ln(r) + 4*Omega^2
    #         = G*M/r^3 + 2*V_H^2/r^2   (derived from the two-component potential)
    return np.sqrt(G * M_BH / np.maximum(r, EPS)**3 + 2 * V_H**2 / r**2)

def sigma_surface(r):
    return M_DISC / (2 * np.pi * R_D**2) * np.exp(-r / R_D)

def toomre_Q(r, sigma_r):
    return sigma_r * kappa(r) / (np.pi * G * sigma_surface(r))

# ── initial conditions: exponential disc ──────────────────────────────────────
rng    = np.random.default_rng(7)
r_grid = np.linspace(0.3, 14.0, 50000)
pdf    = r_grid * np.exp(-r_grid / R_D)
cdf    = np.cumsum(pdf); cdf /= cdf[-1]
r0     = np.interp(rng.uniform(0, 1, N), cdf, r_grid)
theta0 = rng.uniform(0, 2 * np.pi, N)

vc0      = v_circ(r0)
sigma_v0 = 0.06 * vc0    # 6% velocity dispersion
sigma_r0 = np.mean(sigma_v0)

vr0 = rng.normal(0, sigma_v0)
vt0 = vc0 + rng.normal(0, sigma_v0)

pos = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
vel = np.column_stack([
    -vt0 * np.sin(theta0) + vr0 * np.cos(theta0),
     vt0 * np.cos(theta0) + vr0 * np.sin(theta0)
])

# ── integrator ────────────────────────────────────────────────────────────────
def accel(pos, t):
    x, y  = pos[:, 0], pos[:, 1]
    r_sq  = x**2 + y**2 + EPS**2
    r_mag = np.sqrt(r_sq)

    a_x = -(G * M_BH / (r_sq * r_mag) + V_H**2 / r_sq) * x
    a_y = -(G * M_BH / (r_sq * r_mag) + V_H**2 / r_sq) * y

    phi_bar  = OMEGA_B * t
    r_nosft  = np.sqrt(x**2 + y**2) + 1e-9
    ang      = np.arctan2(y, x)
    xi       = r_nosft / R_BAR
    f_r      = xi**2 * np.exp(-xi)
    df_dr    = (2 / R_BAR * xi - xi**2 / R_BAR) * np.exp(-xi)
    cos2     = np.cos(2 * (ang - phi_bar))
    sin2     = np.sin(2 * (ang - phi_bar))
    dphi_dr  = -A_BAR * df_dr * cos2
    dphi_dth =  2 * A_BAR * f_r * sin2
    cos_a = x / r_nosft; sin_a = y / r_nosft
    a_x -= dphi_dr * cos_a - dphi_dth * sin_a / r_nosft
    a_y -= dphi_dr * sin_a + dphi_dth * cos_a / r_nosft

    return np.column_stack([a_x, a_y])

print("running galaxy disc simulation...")
acc    = accel(pos, 0)
frames = []
t_Gyr  = []
L_arr  = []

for n in range(STEPS):
    t = n * DT
    pos += vel * DT + 0.5 * acc * DT**2
    acc_new = accel(pos, t + DT)
    vel += 0.5 * (acc + acc_new) * DT
    acc = acc_new

    L = np.mean(pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])
    L_arr.append(L)

    if n % 6 == 0:
        frames.append(pos.copy())
        t_Gyr.append(t * KPC_KMS_TO_GYR)

print(f"done — {STEPS} steps = {STEPS*DT*KPC_KMS_TO_GYR:.2f} Gyr, {len(frames)} frames")

L0     = L_arr[0]
L_drift = [(L - L0) / abs(L0) * 100 for L in L_arr]
print(f"Mean angular momentum drift: {np.max(np.abs(L_drift)):.4f}%")

# ── Kepler verification ───────────────────────────────────────────────────────
kepler_const = 4 * np.pi**2 / (G * M_BH)
r_test = np.array([0.5, 1.0, 2.0])
for rt in r_test:
    T = 2 * np.pi * rt / np.sqrt(G * M_BH / rt)   # Keplerian only
    ratio = T**2 / rt**3
    print(f"  r={rt} kpc  T={T*KPC_KMS_TO_GYR*1e3:.1f} Myr  T2/a3={ratio:.4e}  theory={kepler_const:.4e}")

# ── measured rotation curve from final snapshot ───────────────────────────────
pos_f = frames[-1]
vel_f_r = (pos_f[:, 0] * vel[:, 0] + pos_f[:, 1] * vel[:, 1]) / (
          np.sqrt(pos_f[:, 0]**2 + pos_f[:, 1]**2) + 1e-9)
r_f   = np.sqrt(pos_f[:, 0]**2 + pos_f[:, 1]**2)
v_tan = np.abs(pos_f[:, 0] * vel[:, 1] - pos_f[:, 1] * vel[:, 0]) / (r_f + 1e-9)

r_bins  = np.linspace(0.5, 12, 25)
r_cents = 0.5 * (r_bins[:-1] + r_bins[1:])
v_meas  = np.array([
    np.median(v_tan[(r_f > r_bins[i]) & (r_f <= r_bins[i+1])])
    for i in range(len(r_bins)-1)
])

# ── FIGURES ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor='#060a12')
gs  = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

ax_disc = fig.add_subplot(gs[:2, :2])
ax_rot  = fig.add_subplot(gs[0, 2])
ax_Q    = fig.add_subplot(gs[1, 2])
ax_dens = fig.add_subplot(gs[2, 0])
ax_L    = fig.add_subplot(gs[2, 1])
ax_kap  = fig.add_subplot(gs[2, 2])

for ax in (ax_disc, ax_rot, ax_Q, ax_dens, ax_L, ax_kap):
    ax.set_facecolor('#060a12')
    ax.tick_params(colors='#8B949E', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#21262D')

# disc snapshot coloured by |r|
pos_last = frames[-1]
r_last   = np.sqrt(pos_last[:, 0]**2 + pos_last[:, 1]**2)
sc = ax_disc.scatter(pos_last[:, 0], pos_last[:, 1],
                     s=0.8, c=r_last, cmap='plasma', vmin=0, vmax=12, alpha=0.7)
ax_disc.set_xlim(-16, 16); ax_disc.set_ylim(-16, 16)
ax_disc.set_aspect('equal'); ax_disc.axis('off')
disc_title = ax_disc.text(0.5, 1.005,
    f"Galactic disc  t = {t_Gyr[-1]*1e3:.0f} Myr  (N={N}, rotating bar Ω_b={OMEGA_B} km/s/kpc)",
    transform=ax_disc.transAxes, color='white', fontsize=11, fontweight='bold',
    ha='center', va='bottom')
cb = plt.colorbar(sc, ax=ax_disc, fraction=0.03, pad=0.02)
cb.set_label("r (kpc)", color='#8B949E'); cb.ax.yaxis.set_tick_params(color='#8B949E')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8B949E')

# rotation curve
r_plot = np.linspace(0.3, 12, 300)
ax_rot.plot(r_plot, np.sqrt(G * M_BH / r_plot), '--', color='#FFD60A', lw=1.2, label='BH (Keplerian)')
ax_rot.plot(r_plot, np.full_like(r_plot, V_H),  '--', color='#4CC9F0', lw=1.2, label='Isothermal halo')
ax_rot.plot(r_plot, v_circ(r_plot), '-', color='white', lw=1.8, label='Total')
ax_rot.errorbar(r_cents, v_meas, fmt='o', color='#F4A261', ms=3, lw=0.8,
                label='Measured (simulation)', zorder=5)
ax_rot.set_xlabel("r (kpc)", color='#8B949E'); ax_rot.set_ylabel("v_circ (km/s)", color='#8B949E')
ax_rot.set_title("Rotation curve", color='white', fontsize=9, fontweight='bold')
ax_rot.legend(fontsize=6.5, facecolor='#0d1117', labelcolor='white')
ax_rot.grid(linestyle='--', alpha=0.15, color='white')

# Toomre Q
Q_arr = toomre_Q(r_plot, sigma_r0)
ax_Q.plot(r_plot, Q_arr, color='white', lw=1.5, label=f'Q  (σᵣ≈{sigma_r0:.0f} km/s)')
ax_Q.axhline(1.0, color='#E63946', lw=0.9, linestyle='--', label='Q=1 (marginal)')
ax_Q.fill_between(r_plot, 0, Q_arr, where=(Q_arr < 1), alpha=0.25, color='red')
ax_Q.set_ylim(0, 8); ax_Q.set_xlabel("r (kpc)", color='#8B949E')
ax_Q.set_ylabel("Toomre Q", color='#8B949E')
ax_Q.set_title("Toomre Q = σᵣκ / πGΣ", color='white', fontsize=9, fontweight='bold')
ax_Q.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')
ax_Q.grid(linestyle='--', alpha=0.15, color='white')

# surface density at t=0 vs t=final
bins = np.linspace(0.5, 12, 20)
cents = 0.5 * (bins[:-1] + bins[1:])
ring_area = np.pi * (bins[1:]**2 - bins[:-1]**2)
dens0 = np.histogram(r0, bins=bins)[0] / ring_area
dens_f = np.histogram(r_last, bins=bins)[0] / ring_area

# normalize by initial peak so we see relative change
dens0_n = dens0 / (dens0.max() + 1e-30)
dens_f_n = dens_f / (dens0.max() + 1e-30)

ax_dens.plot(cents, dens0_n, '--', color='#8B949E', lw=1.0, label='t = 0')
ax_dens.plot(cents, dens_f_n, '-',  color='#4CC9F0', lw=1.4, label=f't = {t_Gyr[-1]*1e3:.0f} Myr')
ax_dens.plot(r_plot, np.exp(-r_plot/R_D) / np.exp(-0.5/R_D),
             ':', color='#F4A261', lw=1.0, label=f'exp(−r/Rd), Rd={R_D} kpc')
ax_dens.set_xlabel("r (kpc)", color='#8B949E'); ax_dens.set_ylabel("Σ (normalised)", color='#8B949E')
ax_dens.set_title("Radial surface density", color='white', fontsize=9, fontweight='bold')
ax_dens.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')
ax_dens.grid(linestyle='--', alpha=0.15, color='white')

# angular momentum drift
t_step = np.arange(STEPS) * DT * KPC_KMS_TO_GYR * 1e3  # Myr
ax_L.plot(t_step, L_drift, color='#55A868', lw=1.2)
ax_L.axhline(0, color='#30363D', lw=0.8, linestyle='--')
ax_L.set_xlabel("Time (Myr)", color='#8B949E')
ax_L.set_ylabel("ΔL/L₀ (%)", color='#8B949E')
ax_L.set_title("Angular momentum conservation", color='white', fontsize=9, fontweight='bold')
ax_L.grid(linestyle='--', alpha=0.15, color='white')
ax_L.text(0.97, 0.05, f"max drift = {np.max(np.abs(L_drift)):.4f}%",
          transform=ax_L.transAxes, color='#8B949E', fontsize=7.5, ha='right')
vline_L = ax_L.axvline(t_Gyr[0] * 1e3, color='white', lw=1.0, alpha=0.75, zorder=5)

# epicyclic frequency
Omega_arr = v_circ(r_plot) / r_plot
kap_arr   = kappa(r_plot)
ax_kap.plot(r_plot, kap_arr,   color='#F4A261', lw=1.4, label='κ (epicyclic)')
ax_kap.plot(r_plot, Omega_arr, color='#4CC9F0', lw=1.4, label='Ω (angular)')
ax_kap.plot(r_plot, kap_arr / Omega_arr, color='white', lw=1.0, linestyle='--', label='κ/Ω')
ax_kap.axhline(np.sqrt(2), color='lime', lw=0.7, linestyle=':', alpha=0.6, label='√2 (flat limit)')
ax_kap.set_ylim(0, 120)
ax_kap.set_xlabel("r (kpc)", color='#8B949E'); ax_kap.set_ylabel("rad / [kpc/(km/s)]", color='#8B949E')
ax_kap.set_title("Epicyclic frequency  κ² = GM/r³ + 2V_H²/r²", color='white', fontsize=9, fontweight='bold')
ax_kap.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')
ax_kap.grid(linestyle='--', alpha=0.15, color='white')

fig.suptitle("Galactic Disc Simulator — isothermal halo + rotating bar", color='white',
             fontsize=14, fontweight='bold')
plt.savefig("galaxy_analysis.png", dpi=150, bbox_inches='tight', facecolor='#060a12')
print("saved galaxy_analysis.png")

# ── animation (reuses main figure) ────────────────────────────────────────────
sc.set_offsets(frames[0])
sc.set_array(np.sqrt(frames[0][:, 0]**2 + frames[0][:, 1]**2))

def update(i):
    p = frames[i]
    sc.set_offsets(p)
    sc.set_array(np.sqrt(p[:, 0]**2 + p[:, 1]**2))
    vline_L.set_xdata([t_Gyr[i] * 1e3, t_Gyr[i] * 1e3])
    disc_title.set_text(
        f"Galactic disc  t = {t_Gyr[i]*1e3:.0f} Myr  (N={N}, rotating bar Ω_b={OMEGA_B} km/s/kpc)")
    return [sc, vline_L, disc_title]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=16, blit=True)
plt.show()
