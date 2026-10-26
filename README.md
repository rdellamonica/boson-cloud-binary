# Gravitational Molecule

Numerical evolution of ultralight boson clouds ("gravitational atoms") in
comparable-mass binary black holes, beyond perturbation theory.

The scalar field obeys the Schrödinger equation in the Newtonian potential of
both black holes, written in the centre-of-mass frame and expanded in
spherical harmonics,

```text
psi(t, r, theta, phi) = sum_lm U_lm(t, r) / r * Y_lm(theta, phi),
```

so that the binary's multipoles couple the radial functions `U_lm`. The coupled
system is evolved with a Crank–Nicolson scheme on a uniform radial grid. The
framework includes, without any perturbative expansion:

- **ionization** (outgoing fluxes through an extraction radius),
- **resonances** between bound states driven by the orbital motion,
- **absorption** by each black hole through an s-wave sink,
- **back-reaction** of the cloud on the orbit, together with gravitational-wave
  decay and accretion onto both bodies.

The solver is written in C++17 with OpenMP, Eigen and GSL. Python scripts build
the initial states, and a browser dashboard follows the runs live.

## Repository layout

```text
include/, src/     C++ solver (library + GravitationalMolecule and IsolatedBH executables)
initial_states/    Python scripts that write initial states in the MODESBIN format
dashboard/         Live web dashboard for running and finished simulations
examples/minimal/  A configuration that runs in about a minute
tests/             Test suite (ctest: unit, regression, physics, integration, python) and a scaling benchmark
external/toml++/   Bundled toml++ v3.4 TOML parser (header only, MIT licence)
docs/              Doxygen configuration
```

## Requirements

**Solver**

- a C++17 compiler with OpenMP (GCC ≥ 9, Clang ≥ 10, or AppleClang with `libomp`)
- CMake ≥ 3.16
- [Eigen](https://eigen.tuxfamily.org) ≥ 3.3
- [GSL](https://www.gnu.org/software/gsl/)
- Doxygen (optional, for the API documentation)

**Initial states and dashboard**

- Python ≥ 3.11
- NumPy ≥ 2.0, SciPy, SymPy and mpmath for the initial states
  ([`initial_states/requirements.txt`](initial_states/requirements.txt));
  the dashboard needs NumPy only.

## Installation

Install the dependencies, for example

```sh
# Debian / Ubuntu
sudo apt install build-essential cmake libeigen3-dev libgsl-dev

# macOS (Homebrew)
brew install cmake eigen gsl libomp
```

then build and (optionally) run the tests:

```sh
git clone https://github.com/<user>/boson-cloud-binary.git
cd boson-cloud-binary
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

This produces `build/GravitationalMolecule` (binary simulations) and
`build/IsolatedBH` (a single black hole; run it with `--help` for its options).
On macOS, CMake looks for `libomp` in the standard Homebrew locations; pass
`-DLIBOMP_ROOT=/path/to/libomp` for another installation.

Set up a Python environment for the initial states:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r initial_states/requirements.txt
```

## Minimal example

[`examples/minimal/config.toml`](examples/minimal/config.toml) describes a 2p
cloud (`n=2, l=1, m=1`, `mu = 0.2`) around the primary of a `q = 0.1` binary
at separation `R = 30`, on a fixed co-rotating orbit with absorption, for ten
orbital periods.

```sh
cd examples/minimal

# 1. Initial state: the 2p atom of the primary, translated to its COM position
python ../../initial_states/get_initial_state_COM_fast.py \
    --config config.toml --output initial_modes.bin

# 2. Evolution (writes a new Simulation_YYYY_MM_DD_HH_MM_SS/ folder)
OMP_NUM_THREADS=4 ../../build/GravitationalMolecule config.toml initial_modes.bin

# 3. Follow the run in the browser at http://127.0.0.1:8050
python ../../dashboard/live_dashboard.py .
```

The evolution takes about a minute on four cores. To continue a run, pass its
folder instead of the configuration:

```sh
../../build/GravitationalMolecule Simulation_YYYY_MM_DD_HH_MM_SS
```

## Initial states

All scripts read the simulation's `config.toml` (grid, binary and atom tables)
and write a `MODESBIN` file with one radial profile per `(l, m)` mode. The state
is normalised to one; the solver rescales it to the cloud mass `M_c`.

| Script | State |
| --- | --- |
| `get_initial_state_COM_fast.py` | One hydrogenic atom `(n, l, m)` on the primary (`--atom secondary` for the companion), with Gauss–Legendre quadrature. Recommended. |
| `get_initial_state_COM.py` | The same atom with adaptive quadrature; `--two-atoms` superposes atoms on both black holes (`[atom_secondary]` sets the second state and the relative `amplitude`). |
| `get_initial_state_molecular.py` | A "molecular" eigenstate of the full binary Hamiltonian, obtained by adiabatic continuation from large separation. See the script's header for its options. |
| `get_initial_state_gaussian.py` | A spherical Gaussian of width `--sigma` at the origin, for tests. |

The file layout is documented in [`include/InitialData.hpp`](include/InitialData.hpp).

## Configuration

Every key is optional; the defaults are in [`include/Config.hpp`](include/Config.hpp).

| Table | Key | Meaning |
| --- | --- | --- |
| `[grid]` | `N`, `L` | Number of radial nodes and outer radius |
| `[binary]` | `q`, `R`, `counter` | Mass ratio `M2/M1`, initial separation, counter-rotation |
| `[atom]` | `mu`, `M_c` | Boson mass and cloud mass (`n, l, m` are used by the initial-state scripts) |
| `[simulation]` | `dt` | Timestep during the startup ramp (and always on a fixed orbit) |
| | `num_steps` / `num_periods` | Run length in steps, or in initial orbital periods if `num_steps = 0` |
| | `save_every` | Snapshot and checkpoint interval in steps |
| | `l_max` | Largest multipole; only modes with even `l - m` are evolved |
| | `r_ex` | Extraction radius for the fluxes and for the inner-region integrals |
| | `ramp_periods` | Startup ramp length, in initial orbital periods |
| | `ramp_secondary_potential` | Ramp the companion's potential with the frequency (default `true`) |
| | `evolve_orbit` | Evolve the separation and masses (default `false`) |
| | `steps_per_period` | Post-ramp timestep `dt = P / steps_per_period` with orbital evolution (default 50) |
| | `power_average_orbits` | Orbits in the rolling average of the binary power (default 2) |
| | `absorption`, `epsilon_smear` | Enable the black-hole sinks; sink radius in units of each mass |
| | `epsilon` | Offset of the first radial node |
| `[parallel]` | `threads` | OpenMP threads; 0 uses `OMP_NUM_THREADS` |
| `[solver]` | `preconditioner` | `"radial"` (default) or `"ilut"` |
| `[mask]` | `L_factor`, `W`, `n` | Multiplicative damping layer beyond `L_factor * L` |
| `[cap]` | `L_factor`, `W`, `n` | Complex absorbing potential beyond `L_factor * L` |

## Output

Each run writes a `Simulation_YYYY_MM_DD_HH_MM_SS/` folder with

- `config.toml`, `r.txt` (radial nodes) and `modes.txt` (mode order),
- `psi/snapshot_NNNNNN.txt`: the state every `save_every` steps, as `Re Im` lines
  ordered mode by mode,
- one row per step in `t.txt`, `R_orb.txt`, `theta_orb.txt`, `Omega_orb.txt`,
  `M_1.txt`, `q.txt`, `energy.txt`, `p_bin.txt` (phase-averaged binary power),
  `p_bin_inst.txt` and `dV_bin_dt.txt`,
- one row per step and one column per mode in `flux.txt`, `energy_flux.txt`,
  `norms.txt` and `energy_modes.txt`,
- `checkpoint.bin`, the exact binary state at the last snapshot, used to resume.

## Method

**Discretisation.** The radial operator uses second-order finite differences
with Dirichlet boundaries. The binary potential is expanded in multipoles up to
`2 l_max`; its angular part is a Gaunt coefficient, precomputed once, and the
orbital phase enters as `exp(i m* theta)`. Sparse patterns are built once and
updated in place every step.

**Time stepping.** Each step solves
`(1 + i dt H_full / 2) psi_new = (1 - i dt H_full / 2) psi` with BiCGSTAB. The
default preconditioner inverts each mode's radial tridiagonal block exactly; it
falls back to a full ILUT if needed, and every solve must reach a relative
residual of `1e-9`.

**Startup.** Over `ramp_periods` initial orbital periods
`P0 = 2 pi sqrt(R^3 / (1 + q))`, the orbital frequency (and, by default, the
companion's potential) is switched on with a smooth step. Absorption and
orbital back-reaction then turn on over one further `P0`. With
`evolve_orbit = true`, the separation evolves from the end of the ramp under
gravitational-wave emission and the phase-averaged cloud power, the masses grow
by accretion, and the timestep follows the orbital period.

**Absorption.** Each black hole has a spherical compact kernel `K_a` of radius
`epsilon_smear * M_a` centred on the hole, normalised to unit volume on the grid.
The sink `Gamma_a = 16 pi M_a^2 |K_a><K_a|` acts only on the `l = 0` component
of the field about that hole, and enters as `H_full = H - i Gamma / 2`. Because
it has rank one, it is kept out of the sparse matrix and handled in the solver
with a Woodbury-corrected preconditioner.

**Diagnostics.** Fluxes are evaluated at `r_ex` on the end-of-step state with
`dpsi/dt = -i H_full psi`. The binary power is `<psi| dV/dt |psi>` inside `r_ex`,
averaged over `power_average_orbits` orbits of phase.

**Outer boundary.** Outgoing radiation can be removed by a damping mask applied
after each step (`[mask]`) or by a complex absorbing potential (`[cap]`).

**Restart.** A run folder can be resumed from its last complete snapshot. When
`checkpoint.bin` matches it, the continued run is bitwise identical to an
uninterrupted one at a fixed thread count; raising `num_steps` or `num_periods`
in the folder's `config.toml` extends a finished run.

## Tests

The suite runs in about half a minute and is also run on every push by
[`.github/workflows/tests.yml`](.github/workflows/tests.yml):

```sh
ctest --test-dir build -j 4 --output-on-failure
ctest --test-dir build -L unit            # one label only
```

| Label | What it covers |
| --- | --- |
| `unit` | Configuration parsing and defaults, the MODESBIN reader, output folders, argument validation, the rolling phase average |
| `regression` | The s-wave absorbers and their projector against an independent addition-theorem reference, the isolated black hole, the startup and back-reaction ramps, the flux/state time alignment, and single- against multi-threaded assembly and solves |
| `physics` | Hermiticity, norm conservation, hydrogenic eigenstates and their energies, the Jacobi constant on a fixed orbit and its convergence in `dt`, the mirror symmetry between co- and counter-rotation, flux continuity at `r_ex`, the gravitational-wave inspiral against Peters' law, mass balance between the cloud and the accreting holes, the damping mask and the CAP |
| `integration` | Restarting a run (bitwise from `checkpoint.bin`, from the diagnostics without one, after an interrupted write, and extending a finished run) and the two executables end to end |
| `python` | The initial-state scripts against their analytic states, and the dashboard's run discovery, payloads, snapshot viewer, zip export and HTTP endpoints. Skipped, not failed, when NumPy, SciPy, SymPy or mpmath are missing |

## Parallel performance

OpenMP parallelises the coupling construction, the potential and absorber
assembly, the Hamiltonian and Crank–Nicolson updates, the preconditioner,
sparse matrix–vector products and per-mode diagnostics. A positive
`parallel.threads` overrides `OMP_NUM_THREADS`. To compare thread counts on
your machine:

```sh
python3 tests/benchmark_parallel.py build/mode_benchmark \
    --l-max 12 --nr 512 --steps 11 --threads 1 4 8 --repeat 3
```

## API documentation

The C++ classes and functions are documented with Doxygen:

```sh
cmake --build build --target docs   # or: doxygen docs/Doxyfile
open docs/html/index.html
```

They are rebuilt and published at
<https://rdellamonica.github.io/boson-cloud-binary/> on every push to `main`
by [`.github/workflows/docs.yml`](.github/workflows/docs.yml).

## Dashboard

```sh
python dashboard/live_dashboard.py path/to/runs [--port 8050] [--depth 4]
```

The dashboard finds every run folder below the given path and serves a page at
`http://127.0.0.1:8050` with the orbit, fluxes, energies, norms and binary power,
a snapshot viewer for the radial profiles and the equatorial density, and
downloads of the data. It re-reads the files as they grow, so it can follow
runs in progress. Its fonts and d3 are bundled in `dashboard/assets` with their
licences.
