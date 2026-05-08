import numpy as np
from .physics import accel, KPC_KMS_TO_GYR


def run(pos, vel, steps, dt, save_every=6):
    pos = pos.copy()
    vel = vel.copy()
    acc = accel(pos, 0)

    frames = []
    t_Gyr  = []
    L_arr  = []

    for n in range(steps):
        t    = n * dt
        pos += vel * dt + 0.5 * acc * dt**2
        acc_new = accel(pos, t + dt)
        vel += 0.5 * (acc + acc_new) * dt
        acc  = acc_new

        L_arr.append(float(np.mean(pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])))

        if n % save_every == 0:
            frames.append(pos.copy())
            t_Gyr.append(t * KPC_KMS_TO_GYR)

    return frames, t_Gyr, L_arr, vel
