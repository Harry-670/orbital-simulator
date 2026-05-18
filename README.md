# Galactic Disc Simulator

A 2D test-particle simulation of a galactic disc evolving under a central supermassive black hole, an isothermal dark matter halo, and a rotating stellar bar perturbation. Integrates 2000 tracer particles over ~7 Gyr using a Velocity Verlet (leapfrog) scheme.

![Galactic disc animation](galaxy_disc.gif)

---

## Physics

### Potential Components

The total gravitational potential is the sum of three components:

$$\Phi(r,\theta,t) = \Phi_{BH}(r) + \Phi_H(r) + \Phi_\text{bar}(r,\theta,t)$$

| Component | Potential | Effect |
|---|---|---|
| Central BH (Sgr A*) | $\Phi_{BH} = -GM_{BH}/r$ | Keplerian dominance at $r < 1\,\text{kpc}$ |
| Isothermal dark matter halo | $\Phi_H = V_H^2 \ln r$ | Flat rotation curve at all radii |
| Rotating stellar bar | $\Phi_\text{bar} = -A\,f(r)\cos[2(\theta - \Omega_b t)]$ | Non-axisymmetric spiral arm driver |

The combined circular speed is flat beyond $\sim 1\,\text{kpc}$:

$$v_\text{circ}(r) = \sqrt{\frac{GM_{BH}}{r} + V_H^2} \;\approx\; 220\,\text{km/s}$$

### Toomre Q Stability

The axisymmetric stability criterion is:

$$Q(r) = \frac{\sigma_r\,\kappa(r)}{\pi G\,\Sigma(r)}$$

where $\Sigma(r)$ is the exponential disc surface density:

$$\Sigma(r) = \frac{M_\text{disc}}{2\pi R_d^2}\,e^{-r/R_d}$$

**$Q > 1$** throughout the disc indicates axisymmetric stability; the rotating bar drives non-axisymmetric spiral structure regardless.

### Epicyclic Frequency

$$\kappa^2(r) = \frac{GM_{BH}}{r^3} + \frac{2V_H^2}{r^2}$$

In the flat-curve limit $\kappa \to \sqrt{2}\,\Omega$, consistent with the isothermal sphere result.

### Corotation Resonance

The bar pattern speed places corotation at:

$$r_{CR} = \frac{V_H}{\Omega_b} = \frac{220\,\text{km/s}}{50\,\text{km/s/kpc}} = 4.4\,\text{kpc}$$

Inside corotation, particles are trapped into elongated bar-aligned orbits. Outside, trailing spiral arms propagate outward.

### Velocity Verlet Integrator

Particle trajectories are advanced with the leapfrog (Velocity Verlet) scheme, which is symplectic and conserves phase-space volume:

$$\mathbf{r}^{n+1} = \mathbf{r}^n + \mathbf{v}^n\Delta t + \tfrac{1}{2}\mathbf{a}^n\Delta t^2$$

$$\mathbf{v}^{n+1} = \mathbf{v}^n + \tfrac{1}{2}\left(\mathbf{a}^n + \mathbf{a}^{n+1}\right)\Delta t$$

---

## Parameters

| Parameter | Symbol | Value |
|---|---|---|
| Black hole mass | $M_{BH}$ | $4 \times 10^6\,M_\odot$ (Sgr A*) |
| Halo circular speed | $V_H$ | $220\,\text{km/s}$ |
| Disc scale length | $R_d$ | $3\,\text{kpc}$ |
| Total disc mass | $M_\text{disc}$ | $5 \times 10^{10}\,M_\odot$ |
| Bar pattern speed | $\Omega_b$ | $50\,\text{km/s/kpc}$ |
| Bar scale length | $R_\text{bar}$ | $3.5\,\text{kpc}$ |
| Softening length | $\varepsilon$ | $0.05\,\text{kpc}$ |
| Particles | $N$ | $2000$ |
| Timestep | $\Delta t$ | $0.004\,\text{kpc/(km/s)} \approx 3.9\,\text{Myr}$ |
| Total time | — | $1800$ steps $\approx 7\,\text{Gyr}$ |

---

## Analysis Output

Running the script produces a 6-panel figure (`galaxy_analysis.png`):

- **Disc snapshot** — particles coloured by galactocentric radius
- **Rotation curve** — BH + halo decomposition vs. measured particle velocities
- **Toomre Q** — radial stability profile $Q(r)$
- **Surface density** — initial vs. final $\Sigma(r)$ exponential profile
- **Angular momentum drift** — conservation check over the full integration
- **Epicyclic frequency** — $\kappa(r)$ and $\Omega(r)$ with the flat-curve $\sqrt{2}$ limit

---

## Running

```bash
pip install numpy matplotlib
python main.py
```

Outputs `galaxy_analysis.png` and opens an interactive animated window.

---

## Project Structure

```
├── main.py                   # entry point
├── sim/
│   ├── physics.py            # potentials, forces, constants
│   ├── initial_conditions.py # particle sampling
│   ├── integrator.py         # velocity verlet loop
│   └── plotting.py           # figures and animation
└── paper/
    ├── paper.tex             # LaTeX source
    └── paper.pdf             # compiled write-up
```

---

## Paper

A short technical write-up covering the physics, numerical methods, and results is included as [`paper/paper.pdf`](paper/paper.pdf).
