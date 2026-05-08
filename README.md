# Galactic Disc Simulator

A 2D test-particle simulation of a galactic disc evolving under a central supermassive black hole, an isothermal dark matter halo, and a rotating stellar bar perturbation. Integrates 2000 tracer particles over ~7 Gyr using a Velocity Verlet (leapfrog) scheme.

![Galactic disc animation](galaxy_disc.gif)

---

## Physics

### Potential Components

| Component | Potential | Effect |
|---|---|---|
| Central BH (Sgr A*) | Φ_BH = −GM/r | Keplerian dominance at r < 1 kpc |
| Isothermal dark matter halo | Φ_H = V_H² ln r | Flat rotation curve at all radii |
| Rotating stellar bar | Φ_bar = −A f(r) cos 2(θ − Ω_b t) | Non-axisymmetric spiral arm driver |

The combined circular speed is flat beyond ~1 kpc:

```
v_circ(r) = sqrt(GM_BH/r + V_H²)  ≈  220 km/s
```

### Toomre Q Stability

```
Q(r) = σᵣ κ(r) / (π G Σ(r))
```

where `Σ(r) = (M_disc / 2π R_d²) exp(−r/R_d)` is the exponential disc surface density and `κ(r)` is the epicyclic frequency. **Q > 1** throughout the disc indicates axisymmetric stability; the rotating bar drives non-axisymmetric spiral structure even so.

### Epicyclic Frequency

```
κ²(r) = GM/r³ + 2V_H²/r²
```

In the flat-curve limit: κ → √2 Ω, consistent with the isothermal sphere result.

### Corotation Resonance

The bar pattern speed places corotation at:

```
r_CR = V_H / Ω_b = 220 / 50 = 4.4 kpc
```

Inside corotation, particles are trapped into elongated bar-aligned orbits. Outside, trailing spiral arms propagate outward.

---

## Parameters

| Parameter | Symbol | Value |
|---|---|---|
| Black hole mass | M_BH | 4 × 10⁶ M☉ (Sgr A*) |
| Halo circular speed | V_H | 220 km/s |
| Disc scale length | R_d | 3 kpc |
| Total disc mass | M_disc | 5 × 10¹⁰ M☉ |
| Bar pattern speed | Ω_b | 50 km/s/kpc |
| Bar scale length | R_bar | 3.5 kpc |
| Softening length | ε | 0.05 kpc |
| Particles | N | 2000 |
| Timestep | Δt | 0.004 kpc/(km/s) ≈ 3.9 Myr |
| Total time | — | 1800 steps ≈ 7 Gyr |

---

## Analysis Output

Running the script produces a 6-panel figure (`galaxy_analysis.png`):

- **Disc snapshot** — particles coloured by galactocentric radius
- **Rotation curve** — BH + halo decomposition vs. measured particle velocities
- **Toomre Q** — radial stability profile
- **Surface density** — initial vs. final exponential profile
- **Angular momentum drift** — conservation check over the full integration
- **Epicyclic frequency** — κ(r) and Ω(r) with the flat-curve √2 limit

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
