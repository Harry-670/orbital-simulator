import numpy as np
from .physics import R_D, v_circ


def generate(N, seed=7):
    rng    = np.random.default_rng(seed)
    r_grid = np.linspace(0.3, 14.0, 50000)
    pdf    = r_grid * np.exp(-r_grid / R_D)
    cdf    = np.cumsum(pdf); cdf /= cdf[-1]
    r0     = np.interp(rng.uniform(0, 1, N), cdf, r_grid)
    theta0 = rng.uniform(0, 2 * np.pi, N)

    vc0      = v_circ(r0)
    sigma_v0 = 0.06 * vc0
    sigma_r0 = float(np.mean(sigma_v0))

    vr0 = rng.normal(0, sigma_v0)
    vt0 = vc0 + rng.normal(0, sigma_v0)

    pos = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    vel = np.column_stack([
        -vt0 * np.sin(theta0) + vr0 * np.cos(theta0),
         vt0 * np.cos(theta0) + vr0 * np.sin(theta0),
    ])

    return pos, vel, r0, theta0, sigma_r0
