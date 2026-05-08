import numpy as np

G               = 4.302e-6    # kpc (km/s)^2 M_sun^-1
KPC_KMS_TO_GYR  = 0.9778

M_BH    = 4.0e6
V_H     = 220.0
M_DISC  = 5.0e10
R_D     = 3.0
EPS     = 0.05

A_BAR   = 1800.0
R_BAR   = 3.5
OMEGA_B = 50.0


def v_circ(r):
    return np.sqrt(G * M_BH / np.maximum(r, EPS) + V_H**2)


def kappa(r):
    return np.sqrt(G * M_BH / np.maximum(r, EPS)**3 + 2 * V_H**2 / r**2)


def sigma_surface(r):
    return M_DISC / (2 * np.pi * R_D**2) * np.exp(-r / R_D)


def toomre_Q(r, sigma_r):
    return sigma_r * kappa(r) / (np.pi * G * sigma_surface(r))


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
    cos_a    = x / r_nosft
    sin_a    = y / r_nosft
    a_x -= dphi_dr * cos_a - dphi_dth * sin_a / r_nosft
    a_y -= dphi_dr * sin_a + dphi_dth * cos_a / r_nosft

    return np.column_stack([a_x, a_y])
