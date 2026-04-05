# Galactic Disc Simulator

N-body simulation of a self-consistent galactic disc under a central black hole, an isothermal dark matter halo, and a rotating stellar bar perturbation.

## Physics

### Potential components

| Component | Potential | Circular speed |
|---|---|---|
| Central BH (Sgr A*) | Φ_BH = −GM/r | v_BH = √(GM/r) |
| Isothermal halo | Φ_H = V_H² ln r | v_H = V_H (flat) |
| Rotating bar | Φ_bar = −A f(r) cos 2(θ−Ω_b t) | — perturbation |

The total rotation curve is **flat** beyond ~1 kpc, matching observations:

```
v_circ(r) = sqrt(GM_BH/r + V_H²)  ≈  220 km/s  at  r = 8 kpc
```

### Epicyclic frequency

For the combined two-component potential:

```
κ²(r) = GM/r³ + 2V_H²/r²
```

In the flat-curve limit (V_H dominates): κ → √2 Ω, recovering the known isothermal-sphere result.

### Toomre Q stability

```
Q(r) = σᵣ κ(r) / (π G Σ(r))
```

where Σ(r) = (M_disc / 2π R_d²) exp(−r/R_d) is the exponential disc surface density.
**Q < 1** indicates gravitational instability; the rotating bar drives spiral arm excitation.

### Integration

Velocity Verlet (leapfrog) with softened gravity (ε = 0.05 kpc).
Unit system: kpc, km/s, M_sun → 1 time unit = 0.978 Gyr.

## Parameters

| Parameter | Value | Notes |
|---|---|---|
| M_BH | 4 × 10⁶ M_sun | Milky Way Sgr A* |
| V_H | 220 km/s | halo flat speed |
| R_d | 3 kpc | disc scale length |
| M_disc | 5 × 10¹⁰ M_sun | total stellar mass |
| Ω_bar | 50 km/s/kpc | bar pattern speed |
| N | 2000 | test particles |
| Δt | 0.004 kpc/(km/s) ≈ 3.9 Myr | |

## Running

```bash
python galaxy_disc.py
```

Saves `galaxy_analysis.png` (6-panel analysis), then shows the animated disc.

## Requirements

```
numpy matplotlib
```
