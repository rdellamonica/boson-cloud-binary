# Isolated displaced black-hole absorption test

Build and run from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
./build/IsolatedBH 0.2 50 4
```

The positional arguments are **mu, BH x position, physical absorber support
radius, and optional l_max** (default 6). Positions and radii are in units of
the single BH mass M=1. A negative position places it on the negative x axis;
position zero is supported for a centered control. There is no second BH.
The requested location is interpreted as the position of this sole BH, not a
binary separation. The setup is spherically symmetric about the BH in the
continuum; the numerical spherical-coordinate grid is centered at the origin.

For example, use a larger mode basis and explicitly chosen resolution:

```bash
./build/IsolatedBH 0.2 50 4 10 \
    --N 4097 --L 600 --dt 0.25 --steps 4001 --save-every 100 \
    --threads 8 --output runs/isolated_mu02_x50_h4
```

Use `--no-absorption` for a conservative control with the same parameters and
a different output directory. `--help` lists all options. Each output directory
must be new, preventing accidental replacement of an existing run.

The initial wavefunction is the nonrelativistic hydrogenic 100 state:

```text
a0 = 1 / mu^2
psi_100(x) = exp(-|x - x_BH|/a0) / sqrt(pi*a0^3)
U_lm(r) = r * integral dOmega Y_lm*(Omega) psi_100(r,Omega)
```

The C++ initializer projects this function into the existing mode basis using
Gauss–Legendre quadrature in distance from the BH and the addition theorem.
For an x-axis displacement the retained l-m-even modes contain the required
reflection symmetry. The coefficients are normalized to unit discrete
probability, `dr * sum_lm,i |U_lm(r_i)|^2 = 1`. Thus the saved config has
`M_c=mu`. The unrenormalized projected norm is printed to help monitor grid,
domain, and angular truncation errors. No Python initializer or binary input
file is needed.

The real potential is exactly the solver's projection of `-mu/|x-x_BH|`.
The BH position, mass, and phase stay fixed; both orbital frequency and binary
power are zero. The companion potential and companion absorption vanish.
This is an explicit isolated mode, since the binary solver's
`evolve_orbit=false` alone would still rotate the potentials.

Absorption uses the **same spherical kernel and area factor** as the binary
implementation, with support radius h equal to the third positional argument:
`Gamma = 16*pi*K_h`, and `H = H_conservative - i*Gamma/2`. It is active at full
strength from t=0, with no orbital startup ramp. A width below dr (including
zero) disables the kernel, following the production implementation. The BH
does not gain mass: this experiment measures field absorption on a prescribed
background. No outer mask or CAP is enabled, so the only norm loss should come
from the BH absorber.

Defaults are 2049 radial points, outer radius `|x_BH| + 20*a0 + 2*h`, 2001 time
samples, and snapshots every 100 steps. The timestep is
`min(1, 0.05/mu^3, 0.01*h^3)` for h>0; the h term is omitted at h=0.
These are starting parameters for a convergence study: decrease dt and dr,
increase l_max, and enlarge the domain as needed. In particular, a very small
absorber far from the origin may require many angular modes. The translated
analytic 100 state is not an exact eigenvector of the truncated, discretized
Hamiltonian, so small conservative transients can occur.

Outputs appear under the requested directory (or a unique `IsolatedBH_*`
directory), containing `isolated_config.toml` and the usual `Simulation_*`
directory with snapshots, modes, times, norms, fluxes, and a copied config.
`[isolated_bh]` in the config records the actual position and mass. Binary
plotting notebooks that infer body locations from q and R must instead read
these coordinates for this experiment. The final full-domain probability norm
is printed. For intermediate cloud masses, multiply the sum of `norms.txt`
columns by mu; that diagnostic covers the extraction region, which here ends
near the outer boundary. R_orb.txt records the fixed displacement magnitude,
not a binary separation. Use this executable to reproduce these configurations.

Run the focused regression:

```bash
ctest --test-dir build -R '^isolated_bh$' --output-on-failure
```

It checks the analytic centered profile, translation to positive/negative x,
potential normalization, stationary geometry/operator, conservation without
absorption, and agreement between norm loss and the discrete CN absorption
balance with one and four threads.
