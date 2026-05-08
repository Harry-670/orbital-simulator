import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from .physics import (G, M_BH, V_H, R_D, OMEGA_B,
                      KPC_KMS_TO_GYR, v_circ, kappa, toomre_Q)


def _style_axes(axes):
    for ax in axes:
        ax.set_facecolor('#060a12')
        ax.tick_params(colors='#8B949E', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#21262D')


def make_analysis_figure(frames, t_Gyr, L_arr, r0, vel_final, steps, dt, N):
    pos_last = frames[-1]
    r_last   = np.sqrt(pos_last[:, 0]**2 + pos_last[:, 1]**2)

    r_f   = r_last.copy()
    v_tan = np.abs(pos_last[:, 0] * vel_final[:, 1] - pos_last[:, 1] * vel_final[:, 0]) / (r_f + 1e-9)
    r_bins  = np.linspace(0.5, 12, 25)
    r_cents = 0.5 * (r_bins[:-1] + r_bins[1:])
    v_meas  = np.array([
        np.median(v_tan[(r_f > r_bins[i]) & (r_f <= r_bins[i + 1])])
        for i in range(len(r_bins) - 1)
    ])

    L0      = L_arr[0]
    L_drift = [(L - L0) / abs(L0) * 100 for L in L_arr]
    t_step  = np.arange(steps) * dt * KPC_KMS_TO_GYR * 1e3
    sigma_r0 = float(np.mean(0.06 * v_circ(r0)))

    fig = plt.figure(figsize=(18, 12), facecolor='#060a12')
    gs  = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    ax_disc = fig.add_subplot(gs[:2, :2])
    ax_rot  = fig.add_subplot(gs[0, 2])
    ax_Q    = fig.add_subplot(gs[1, 2])
    ax_dens = fig.add_subplot(gs[2, 0])
    ax_L    = fig.add_subplot(gs[2, 1])
    ax_kap  = fig.add_subplot(gs[2, 2])
    _style_axes([ax_disc, ax_rot, ax_Q, ax_dens, ax_L, ax_kap])

    sc = ax_disc.scatter(pos_last[:, 0], pos_last[:, 1],
                         s=0.8, c=r_last, cmap='plasma', vmin=0, vmax=12, alpha=0.7)
    ax_disc.set_xlim(-16, 16); ax_disc.set_ylim(-16, 16)
    ax_disc.set_aspect('equal'); ax_disc.axis('off')
    disc_title = ax_disc.text(
        0.5, 1.005,
        f"Galactic disc  t = {t_Gyr[-1]*1e3:.0f} Myr  (N={N}, rotating bar Ω_b={OMEGA_B} km/s/kpc)",
        transform=ax_disc.transAxes, color='white', fontsize=11, fontweight='bold',
        ha='center', va='bottom')
    cb = plt.colorbar(sc, ax=ax_disc, fraction=0.03, pad=0.02)
    cb.set_label("r (kpc)", color='#8B949E')
    cb.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8B949E')

    r_plot = np.linspace(0.3, 12, 300)
    ax_rot.plot(r_plot, np.sqrt(G * M_BH / r_plot), '--', color='#FFD60A', lw=1.2, label='BH (Keplerian)')
    ax_rot.plot(r_plot, np.full_like(r_plot, V_H),  '--', color='#4CC9F0', lw=1.2, label='Isothermal halo')
    ax_rot.plot(r_plot, v_circ(r_plot), '-', color='white', lw=1.8, label='Total')
    ax_rot.errorbar(r_cents, v_meas, fmt='o', color='#F4A261', ms=3, lw=0.8, label='Measured', zorder=5)
    ax_rot.set_xlabel("r (kpc)", color='#8B949E'); ax_rot.set_ylabel("v_circ (km/s)", color='#8B949E')
    ax_rot.set_title("Rotation curve", color='white', fontsize=9, fontweight='bold')
    ax_rot.legend(fontsize=6.5, facecolor='#0d1117', labelcolor='white')
    ax_rot.grid(linestyle='--', alpha=0.15, color='white')

    Q_arr = toomre_Q(r_plot, sigma_r0)
    ax_Q.plot(r_plot, Q_arr, color='white', lw=1.5, label=f'Q  (σᵣ≈{sigma_r0:.0f} km/s)')
    ax_Q.axhline(1.0, color='#E63946', lw=0.9, linestyle='--', label='Q=1 (marginal)')
    ax_Q.fill_between(r_plot, 0, Q_arr, where=(Q_arr < 1), alpha=0.25, color='red')
    ax_Q.set_ylim(0, 8); ax_Q.set_xlabel("r (kpc)", color='#8B949E')
    ax_Q.set_ylabel("Toomre Q", color='#8B949E')
    ax_Q.set_title("Toomre Q = σᵣκ / πGΣ", color='white', fontsize=9, fontweight='bold')
    ax_Q.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')
    ax_Q.grid(linestyle='--', alpha=0.15, color='white')

    bins      = np.linspace(0.5, 12, 20)
    cents     = 0.5 * (bins[:-1] + bins[1:])
    ring_area = np.pi * (bins[1:]**2 - bins[:-1]**2)
    dens0     = np.histogram(r0,     bins=bins)[0] / ring_area
    dens_f    = np.histogram(r_last, bins=bins)[0] / ring_area
    dens0_n   = dens0  / (dens0.max() + 1e-30)
    dens_f_n  = dens_f / (dens0.max() + 1e-30)
    ax_dens.plot(cents, dens0_n,  '--', color='#8B949E', lw=1.0, label='t = 0')
    ax_dens.plot(cents, dens_f_n, '-',  color='#4CC9F0', lw=1.4, label=f't = {t_Gyr[-1]*1e3:.0f} Myr')
    ax_dens.plot(r_plot, np.exp(-r_plot / R_D) / np.exp(-0.5 / R_D),
                 ':', color='#F4A261', lw=1.0, label=f'exp(−r/Rd)')
    ax_dens.set_xlabel("r (kpc)", color='#8B949E'); ax_dens.set_ylabel("Σ (normalised)", color='#8B949E')
    ax_dens.set_title("Radial surface density", color='white', fontsize=9, fontweight='bold')
    ax_dens.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')
    ax_dens.grid(linestyle='--', alpha=0.15, color='white')

    ax_L.plot(t_step, L_drift, color='#55A868', lw=1.2)
    ax_L.axhline(0, color='#30363D', lw=0.8, linestyle='--')
    ax_L.set_xlabel("Time (Myr)", color='#8B949E')
    ax_L.set_ylabel("ΔL/L₀ (%)", color='#8B949E')
    ax_L.set_title("Angular momentum conservation", color='white', fontsize=9, fontweight='bold')
    ax_L.grid(linestyle='--', alpha=0.15, color='white')
    ax_L.text(0.97, 0.05, f"max drift = {np.max(np.abs(L_drift)):.4f}%",
              transform=ax_L.transAxes, color='#8B949E', fontsize=7.5, ha='right')
    vline_L = ax_L.axvline(t_Gyr[0] * 1e3, color='white', lw=1.0, alpha=0.75, zorder=5)

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

    fig.suptitle("Galactic Disc Simulator — isothermal halo + rotating bar",
                 color='white', fontsize=14, fontweight='bold')

    return fig, sc, vline_L, disc_title


def make_animation(fig, sc, vline_L, disc_title, frames, t_Gyr, N, interval=16):
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

    return animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
