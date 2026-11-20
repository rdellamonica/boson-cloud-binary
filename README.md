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
system is evolved with a Crank–Nicolson scheme on a uniform or stretched radial grid. The
framework includes, without any perturbative expansion:

- **ionization** (outgoing fluxes through an extraction radius),
- **resonances** between bound states driven by the orbital motion,
- **absorption** by each black hole through an s-wave sink and, optionally, the
  dipolar (`l = 1`) horizon channels with spin-dependent, possibly superradiant rates,
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

## Stretched radial grid

A uniform grid spends as many nodes on the far zone, where only outgoing waves
travel, as on the cloud. A stretched grid keeps the fine spacing where the cloud
lives and widens it smoothly further out, so the box can reach much larger radii
for the same number of nodes.

**1. Choose the grid in `config.toml`.**

```toml
[grid]
type = "stretched"   # "uniform" (the default) keeps the evenly spaced grid
L = 4000.0           # outer radius
N = 1300             # radial nodes (see step 2)
r_core = 400.0       # refined region; default 4 n^2 / mu^2 from [atom]
stretch = 4.0        # far spacing / core spacing; default 4
transition = 100.0   # width of the change around r_core; default r_core / 4
```

Below `r_core` the spacing is nearly constant; beyond `r_core + transition` it
is `stretch` times larger. In between it follows
`h(r) ∝ 1 + (stretch - 1)/2 (1 + tanh((r - r_core) / transition))`. Setting
`stretch = 1` gives back an evenly spaced grid.

**2. Pick `N` from the spacing you want in the core.** For a target core
spacing `h_core`,

```text
N ≈ (r_core + (L - r_core) / stretch) / h_core + 1
```

For example, `L = 4000`, `r_core = 400`, `stretch = 4` and `h_core = 1` give
`N ≈ 1300`, where a uniform grid with the same spacing would need 4000 nodes.
The solver prints the spacing it got at startup:
`Stretched radial grid: dr = 0.96 near the centre, 3.84 at r = 4000`.

**3. Put the extraction radius inside the refined region.** The fluxes are
radial derivatives at `r_ex`, so they are only as accurate as the spacing there.
Keep `r_ex < r_core` (raise `r_core` if needed); the solver prints a note when
`r_ex` lies in the coarse zone. In the test below the flux at `r_ex` was 3% off
with `r_ex` in the coarse zone and 0.5% off with `r_core` beyond it.

**4. Generate the initial state with the same `config.toml`.** All
initial-state scripts read `[grid]`, build exactly the solver's nodes and store
them in the `MODESBIN` file:

```sh
python initial_states/get_initial_state_COM_fast.py --config config.toml --output initial_modes.bin
```

A state made for a different grid is still accepted: the solver interpolates
it onto its own nodes (cubic) and says so in the log. Files written before this
change carry no nodes and are taken to be uniform on the same `[0, L]`, so
regenerate those if `L` changed. Regenerating on the right grid also avoids the
small interpolation error.

**5. Run and monitor as usual.** Nothing else changes: the run folder's
`r.txt` lists the actual nodes, every diagnostic (norms, energies, fluxes,
binary power, sinks) uses the trapezoid weights of the stretched grid, and the
dashboard shows the grid type and the spacing range under *Grid & modes*.
The mask and CAP still act between `L_factor * L` and `L`, now in the coarse
zone; keep that layer many far-zone spacings thick (with `L = 4000` and
`L_factor = 0.85`, it is 600 wide against a spacing of about 4).

**What to expect.** With `L = 4000`, `r_core = 400` and `stretch = 4`, 1300
stretched nodes reproduced the cloud's norm, energy and binary power of a
4160-node uniform run with the same core spacing to `1e-5` or better, in a third
of the time.

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
| `get_initial_state_auger.py` | For self-gravity (gravitational Auger) runs with the hole at the origin: a 211 cloud in mode (1,1) with a small 322 seed in mode (2,2) (`--seed-fraction f`, default `1e-3` of the particles, `--phase`). By default the 211 level is made self-consistent with the cloud's own monopole and quadrupole potential at the configured `M_c` (Hartree iteration with the solver's Gaunt coefficients, discrete Green's function and weights), and 322 is the l = 2 level in that potential, so the cloud starts stationary instead of breathing at the self-energy scale (self-energy fluctuations 200 times smaller in a test run). `--bare` uses the hole's grid levels instead, which `levels.txt` also uses; use it with a nonzero `ramp_time`. It warns if `[self_gravity]` is missing, misspelt, or the box is under 40 Bohr radii. |
| `get_initial_state_seed.py` | A random fluctuating seed bound to the central hole, in every evolved mode (`l <= l_max`, `l - m` even): smoothed complex noise of correlation length `--correlation` (default `r_c / 2`) inside `--envelope` (default `10 r_c`), projected onto the bound levels `n <= --n-max` of the solver's own discretised radial Hamiltonian at t = 0 (default: the highest `n` whose levels stay inside `r_ex`, at most `l_max + 1`). The state then holds no unbound part to rounding, so nothing radiates at t = 0 and no high-n levels are left ringing (an isolated run keeps its norm inside `r_ex` to 1e-12 with zero flux, where the raw noise loses 24% by t = 3000); each level keeps a random amplitude and phase. Equal mass per mode (`--l-power P` weights them by `(l + 1)^-P`); modes with `l >= n_max` have no level and stay empty. Only the primary's monopole binds the levels, so it warns when the primary sits off the origin or the companion is on at t = 0. `--unfiltered` gives the raw noise as before, `--seed` makes it reproducible and `--zero 1,1` starts without any 211 component. |

The file layout is documented in [`include/InitialData.hpp`](include/InitialData.hpp).

## Configuration

Every key is optional; the defaults are in [`include/Config.hpp`](include/Config.hpp).

| Table | Key | Meaning |
| --- | --- | --- |
| `[grid]` | `N`, `L` | Number of radial nodes and outer radius |
| | `type` | `"uniform"` (default) or `"stretched"`: fine spacing where the cloud lives, coarser outside. See [Stretched radial grid](#stretched-radial-grid) |
| | `r_core`, `stretch`, `transition` | Stretched grid only: radius of the refined region (default `4 n^2 / mu^2`), outer-to-inner spacing ratio (default 4) and width of the change (default `r_core / 4`) |
| `[binary]` | `q`, `R`, `counter` | Mass ratio `M2/M1`, initial separation, counter-rotation |
| | `spin_1`, `spin_2` | Initial dimensionless spins `chi_a = J_a / M_a^2` of the two holes along `+z` of the simulation frame, in `[-1, 1]` (default 0). The orbit lies in the `xy` plane and `counter` reverses its sense, not the spins. They set the horizon area and the dipolar rates |
| `[atom]` | `mu`, `M_c` | Boson mass and cloud mass (`n, l, m` are used by the initial-state scripts) |
| `[simulation]` | `dt` | Timestep during the startup ramp (and always on a fixed orbit); if omitted, the initial period `P0` divided by `steps_per_period` |
| | `num_steps` / `num_periods` | Run length in steps, or in initial orbital periods if `num_steps = 0` |
| | `save_every` | Snapshot and checkpoint interval in steps |
| | `l_max` | Largest multipole; only modes with even `l - m` are evolved |
| | `r_ex` | Extraction radius for the fluxes and for the inner-region integrals |
| | `ramp_periods` | Startup ramp length, in initial orbital periods |
| | `ramp_secondary_potential` | Ramp the companion's potential with the frequency (default `true`) |
| | `secondary_potential` | `false` removes the companion altogether, to follow the cloud of an isolated hole, for example the growth of a superradiant state with `dipole_absorption` and `spin_1` (default `true`). The primary then stays at rest where the initial state has it, at `x = q R / (1 + q)`, with no companion potential or sink, no orbital motion, no startup ramp and no back-reaction; `q` and `R` only set that position and the time unit of `num_periods`. With `evolve_orbit = true` the primary's mass and spin still follow the horizon fluxes. `q = 0` is run the same way, with the primary at the origin |
| | `orbit_delay_periods` | With `evolve_orbit`, initial periods after the startup ramp during which the companion is fully on but the orbit and masses stay fixed (default 0) |
| | `orbit_ramp_periods` | With `evolve_orbit`, initial periods over which the back-reaction on the separation is switched on once that hold ends (default 1; 0 switches it on at once) |
| | `evolve_orbit` | Evolve the separation and masses (default `false`) |
| | `steps_per_period` | Post-ramp timestep `dt = P / steps_per_period` with orbital evolution (default 50) |
| | `power_average_orbits` | Orbits in the rolling average of the binary power (default 2) |
| | `power_average_window` | Shape of that window in the orbital phase: `"boxcar"` (default) or `"hann"`. A boxcar of N orbits cancels only frequencies k/N per orbit and passes about sinc(f N) of anything else, so a cloud ringing at a non-harmonic frequency makes the separation wiggle. `"hann"` tapers both ends of the window (sin^2) and passes |sinc(x)/(1 - x^2)|, x = f N, instead: for a beat at 1.15 per orbit that is 2.6% at N = 2 and 0.33% at N = 4, against 12.6% for the 1-orbit boxcar, with the same centre and lag; use N >= 2 with it, since a 1-orbit Hann window does not cancel the orbital frequency itself. Not stored in checkpoints: a resumed run uses the configured shape |
| | `absorption`, `epsilon_smear` | Enable the black-hole sinks; sink radius in units of each mass |
| | `dipole_absorption` | Add the dipolar (`l = 1` about each hole) horizon channels to the sinks (default `false`; needs `absorption = true` and `l_max >= 1`). See *Dipolar absorption* below |
| | `energy_reference` | Take each Crank-Nicolson step relative to the initial level's energy `E = -mu^3 / (2 n^2)` and restore the phase `exp(-i E dt)` exactly (default `true`). Plain CN multiplies the growth or decay rate of a level by `1 / (1 + (E dt / 2)^2)`, so without it the horizon rates (superradiant growth included) fall with dt once `|E| dt` approaches 1. The stored states and every diagnostic are unchanged |
| | `epsilon` | Offset of the first radial node |
| `[parallel]` | `threads` | OpenMP threads; 0 uses `OMP_NUM_THREADS` |
| `[solver]` | `preconditioner` | `"radial"` (default) or `"ilut"` |
| `[mask]` | `L_factor`, `W`, `n` | Multiplicative damping layer beyond `L_factor * L` |
| `[cap]` | `L_factor`, `W`, `n` | Complex absorbing potential beyond `L_factor * L` |
| | `per_mode`, `reflection`, `tune_steps`, `k_min` | Tune the CAP strength of each mode to that mode's frequency instead of using `W`, for no measurable reflection (defaults `false`, `1e-10`, `20` steps, `0` = 1 / layer width; with `per_mode` the default `n` is 6). See *Outer boundary* below |
| `[pruning]` | `enabled`, `tolerance`, `flux_tolerance`, `check_every` | Evolve only the modes that matter (defaults `true`, `1e-8`, `1e-3` and `10` steps); `enabled = false` evolves every mode but keeps the settings. See *Mode pruning* below |
| `[self_gravity]` | `enabled`, `l_max`, `recoil`, `recoil_taper`, `accretion_recoil`, `ramp_time` | Newtonian self-gravity of the cloud, for an isolated hole at the origin only (`binary.q = 0` or `secondary_potential = false` with the hole at x = 0) and without pruning. Defaults `true` when the table is present, `l_max = 2 l_max` of the modes, `recoil = true` (hole free-fall frame, needed for the Auger rate), `recoil_taper = 0` (none; otherwise the recoil field falls off from this radius to twice it), `accretion_recoil = true` (with absorption, the hole also takes up the momentum the s-wave sink absorbs), `ramp_time = 0`. See *Self-gravity* below |
| | `scheme`, `tolerance`, `max_iterations` | `"extrapolated"` (default: potential (3 W^n - W^(n-1)) / 2, one solve per step) or `"iterated"` (midpoint potential iterated to `tolerance`, default `1e-10`, at most `max_iterations` = 20 solves) |
| | `measured_reference`, `levels_n_max` | With `energy_reference`, use the initial state's measured chemical potential as E_ref (default `true`); highest n of the levels projected in `levels.txt` (default 5) |

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
- `mass_budget.txt`: one row per step with the cumulative mass absorbed by each
  black hole and the mass dropped by pruning (columns `absorbed_1 absorbed_2
  pruned`). It is kept whether or not the orbit evolves, so together with
  `norms.txt` and the time integral of `flux.txt` it closes the cloud's mass
  budget; the dashboard's *Mass budget* panel shows it,
- with `dipole_absorption`, `spins.txt`: one row per step with the spins and the
  cumulative spin angular momentum taken through the dipolar channels (columns
  `chi_1 chi_2 J_absorbed_1 J_absorbed_2`; the absorbed masses in
  `mass_budget.txt` then include the dipolar part). With `evolve_orbit` the
  dashboard shows it in a *Black-hole spins* panel, as chi or as J = chi M^2 (hidden for every other run),
- `Lz_budget.txt`: one row per step with the cumulative angular momentum along +z
  (particle units, hbar = 1) that left the cloud inside `r_ex`, accumulated on each
  Crank-Nicolson step's midpoint state (columns `ionized torque absorbed pruned
  damped`: carried out through `r_ex`, given to the cloud by the binary's potential,
  taken by the sinks, dropped by pruning, removed by a CAP or mask inside `r_ex`).
  With the cloud's `sum_lm m N_lm` from `norms.txt`, cloud + ionized + absorbed +
  pruned + damped - torque stays constant to the linear solver's tolerance
  (1e-13 to 1e-10 of the initial particle number in the tests). The dashboard's
  *Angular momentum* section plots the L_z of every reservoir in M1^2 (the cloud,
  the orbit, the hole spins, what was radiated and, on an evolving orbit, what the
  GWs carried off; as the change since t = 0 or absolute, with the orbit's change
  split into GW, the cloud feedback it was given and the accreted mass in the
  caption), the signed cloud budget over time and the cloud's L_z per mode;
  for runs written before the file existed it falls back on the diagnostics, which
  close only to O(dt). Runs resumed without it keep going without it,
- with `[self_gravity]`, one row per step in
  - `self_gravity.txt`: `E_self a_x a_y iterations` (self-energy (mu/2) int Phi mu |psi|^2
    without the recoil field, the hole's acceleration towards the cloud's dipole, and the
    solves of the step),
  - `levels.txt`: the population |c_nlm|^2 of every level of the self-consistent
    monopole Hamiltonian (hole + the cloud's own l = 0 potential), n <= `levels_n_max`,
    in the column order of `levels_index.txt` (`n l m energy`, energies at t = 0),
  - `auger_probe.txt`: `Re/Im` of the 211, 322 and 433 amplitudes (projections of the
    (1,1), (2,2), (3,3) modes on those levels), `Re/Im psi_00(0)` from a cusp fit at
    the inner boundary, and `Re/Im` of `U_33` and `U_53` at `r_ex`,
  - `absorbed_channels.txt`: the cumulative mass the hole took through the s-wave
    channel and the dipolar m = -1 and m = +1 channels,
- `checkpoint.bin`, the exact binary state at the last snapshot, used to resume,
- with pruning, `active_modes.txt`: one line `step t count flags...` at the start
  and at every change of the active set, with a 0/1 flag per mode in the order
  of `modes.txt`.

## Method

**Discretisation.** The radial operator uses second-order finite differences
with Dirichlet boundaries. The binary potential is expanded in multipoles up to
`2 l_max`; its angular part is a Gaunt coefficient, precomputed once, and the
orbital phase enters as `exp(i m* theta)`. The coupling of each pair of modes is
stored as one radial profile, recomputed only when the bodies move, so on a
fixed orbit a step only updates one phase per pair. H is applied pair by pair
without being assembled, and the solve runs on the active modes only.

**Stretched radial grid.** On unequal spacings the kinetic term uses the
three-point second difference
`U'' ≈ 2 [h- U(i+1) - (h- + h+) U(i) + h+ U(i-1)] / (h- h+ (h- + h+))`. It is
self-adjoint in the trapezoid inner product (weights `(r(i+1) - r(i-1)) / 2`),
so Crank–Nicolson still conserves the norm exactly, and norms, energies, sink
projections and pruning shares all use those weights. The nodes are equally
spaced in `x(r) = ∫ dr / h(r)`. See *Stretched radial grid* above for usage.

**Time stepping.** Each step solves
`(1 + i dt H_full / 2) psi_new = (1 - i dt H_full / 2) psi` with BiCGSTAB. The
default preconditioner inverts each mode's radial tridiagonal block exactly; it
falls back to a full ILUT if needed, and every solve must reach a relative
residual of `1e-9`.

**Startup.** Over `ramp_periods` initial orbital periods
`P0 = 2 pi sqrt(R^3 / (1 + q))`, the orbital frequency (and, by default, the
companion's potential) is switched on with a smooth step. Absorption then turns
on over one further `P0`. The orbit then stays fixed for `orbit_delay_periods`
further `P0` (none by default), after which the orbital back-reaction is switched
on over `orbit_ramp_periods` `P0` (one by default). With
`evolve_orbit = true`, the separation evolves from the end of that hold under
gravitational-wave emission and the phase-averaged cloud power, the masses grow
by accretion, and the timestep follows the orbital period.

**Absorption.** Each black hole has a spherical compact kernel `K_a` of radius
`epsilon_smear * M_a` centred on the hole, normalised to unit volume on the grid.
The sink `Gamma_a = A_h |K_a><K_a|`, with the horizon area `A_h = 8 pi M_a r_+`
(`16 pi M_a^2` without spin), acts only on the `l = 0` component
of the field about that hole, and enters as `H_full = H - i Gamma / 2`. Because
it has rank one, it is kept out of the sparse matrix and handled in the solver
with a Woodbury-corrected preconditioner.

**Dipolar absorption.** With `dipole_absorption = true` each hole also gets the
operators `gamma_1m |g_1m><g_1m|` for `m = -1, +1`, with `m` measured along `+z`
through the hole. The kernel `g_1m = exp(i mu v_a . rho) rho K(rho) Y_1m / I_1`
extracts the regular `l = 1` amplitude `A_1m` of the field in the rest frame of
the hole, which moves with velocity `v_a`, and

    gamma_1m = A_h k_+ / (4 pi mu) * [M^2 - a^2 + (2 M r_+ k_+)^2],
    k_+ = mu - m Omega_+,   Omega_+ = a / (2 M r_+),   a = chi M.

The coefficient has the sign of `k_+`: a channel with `mu < m Omega_+` is
superradiant and feeds the cloud from the hole. The `m = 0` kernel is odd under
`z -> -z`, so it has no overlap with the evolved modes (even `l - m`) and is
left out. Each step moves `dN_1m = dt gamma_1m |A_1m|^2` particles, evaluated on
the midpoint state: the hole's mass changes by `mu sum_m dN_1m` on top of the
s-wave term, and its spin `J_a = chi_a M_a^2` by `sum_m m dN_1m`. The cloud loses
exactly what the holes gain. As with accretion in general, masses and spins
only change with `evolve_orbit = true`; the cumulative transfers are recorded in
`mass_budget.txt` and `spins.txt` either way. The angular momentum the sinks
take from the cloud about the centre of mass, minus the part that goes into the
spins, is credited to the orbit.

The COM-frame weights `w_LM;1m(r) = r int dOmega Y_LM^* g_1m` are nonzero only
on the shell `|r - R_a| < epsilon_smear M_a`. They are computed once for the
hole on the `+x` axis by a quadrature over the spherical cap that meets the
kernel, and rotated to the hole's azimuth `phi_a` with the exact rule
`w(r; phi_a) = exp(-i (M - m) phi_a) w(r; 0)`. On a fixed orbit the quadrature
therefore runs once; with `evolve_orbit` it is redone when `R_a`, the kernel
radius or the speed change. The solver stores each sink on its shell only, so
an overlap costs (shell nodes) x (modes), and all six rank-one terms enter the
same Woodbury correction. The kernel is normalised on the radial grid (the
harmonic `rho Y_1m` about the hole has amplitude exactly 1), and like the s-wave
kernel it is switched off when `epsilon_smear M_a` is below the local spacing.
The dipolar rates scale with the field gradient at the hole, so the kernel
should span several radial nodes; `M_a mu^2 epsilon << 1` keeps it in the
region where the field is regular.

**Diagnostics.** Fluxes are evaluated at `r_ex` on the end-of-step state with
`dpsi/dt = -i H_full psi`. The binary power is `<psi| dV/dt |psi>` inside `r_ex`,
averaged over `power_average_orbits` orbits of phase, with a boxcar or Hann
window (`power_average_window`).

**Outer boundary.** Outgoing radiation can be removed by a damping mask applied
after each step (`[mask]`) or by a complex absorbing potential (`[cap]`).

With `per_mode = true` the outgoing radiation is absorbed with no measurable
reflection, whatever its frequency. Two things reflect a wave at a CAP: what
crosses the layer, bounces off the wall at `L` and comes back, and the start of
the layer itself, where the potential switches on. The first is removed by giving
each mode `j` its own strength, retuned every step to the frequency of the wave
in that mode:

    W_j = ln(1 / reflection) / (2 mu I_j),    I_j = sum over the layer of s(r) dr / k_j(r)

with `s(r)` the profile and `k_j(r)` the wavenumber of a wave of energy `E_j` in the
mode's centrifugal and monopole potential (the discrete one, `sin(k h) / h` at
spacing `h`). What comes back from the wall then has amplitude `reflection`
(default 1e-10, so 1e-20 of the flux), and since `W_j` grows only with
`ln(1 / reflection)` the layer stays weak. The second is set by how smoothly the
profile starts: with `n = 3` it leaves about 1e-10 of a slow wave's norm, so the
per-mode CAP defaults to `n = 6` (set `n` to override). `E_j` is measured on the
layer from the exact Crank-Nicolson phase advance of the mode over each step,
averaged over `tune_steps` steps; a mode with nothing in the layer yet starts from
the Rayleigh quotient there, and evanescent or very slow modes use `k_min`. The
strengths stay fixed within a step, so the CN step and the angular-momentum budget
are exact as before.

The reflection is measured in `cap_per_mode` against a run on a domain twice as
long without a layer (the difference inside the layer start is exactly what came
back). For packets with group velocities 0.4, 1 and 2 (about 50, 120 and 240
radians across a layer of width 120) the defaults return 2e-13, 2e-16 and 2e-20 of
their norm, at the level of the linear solver; one uniform `W = 0.3` with `n = 3`
returns 8e-5, 0.02 and 0.12. The one limit no absorber removes is a wave with only
a few wavelengths across the layer: the reflection grows quickly below
`k * width` of about 50, and the solver prints a warning for a mode whose wave in
the layer has `k * width < 30` (widen the layer by lowering `L_factor`, or enlarge
`L`). A mode that carries several frequencies at once gets the strength of their
density-weighted mean: slower components are absorbed more strongly, and a
component twice as fast as the mean comes back from the wall at about
`sqrt(reflection)` in amplitude (1e-5 with the default).

Each run writes `cap_modes.txt` (one row per step: the energy estimate of every
mode, 0 before the first one, then its strength `W_j`), and checkpoints keep the
estimates so a resumed run continues bitwise. Without `per_mode` the CAP is the
uniform one, unchanged.

**Self-gravity.** With `[self_gravity]` the cloud feels its own Newtonian
potential, Phi from Poisson's equation lap Phi = 4 pi mu |psi|^2 (G = 1, so the
term in the Schrodinger equation is mu Phi). The density is expanded in multipoles
n_LM = sum G U_j* U_k with Gaunt coefficients, each multipole is solved with the
Green's function r_<^L / r_>^(L+1) by two O(N) sweeps, and Phi couples the modes
through the same Gaunt coefficients as the binary potential. With `recoil = true`
the solver works in the hole's free-fall frame: the hole is pinned at the origin
and the cloud feels the uniform field that accelerates the hole, which removes
the hole's own dipole response (needed for the gravitational Auger rate, about
35% lower than without it). With absorption the hole is also pushed by the
momentum the s-wave sink takes from the cloud (`accretion_recoil = true`), so the
frame conserves the total momentum; without that push the frame force does work
through the sink and feeds a spurious 211 -> 322 transfer several times the Auger
rate. The potential is time-centred, extrapolated or
iterated; the iterated scheme conserves E = <H0 + V> + E_self to the solve
tolerance with `recoil = false` (the free-fall frame is not inertial). The test
`self_gravity` checks the monopole, the 211 self-energy, Hermiticity, the Auger
coefficient I_G = 0.0914 (0.143 without recoil) of Foster & Rodd
(arXiv:2611.18833), energy and norm conservation, and bitwise resume.

**Mode pruning.** Every pair of modes is coupled by the binary's multipoles, so
a step costs roughly the square of the number of modes. With a `[pruning]` table
only an active subset is evolved and the rest is held at exactly zero. Every
`check_every` steps, a mode is switched on when its share of the norm, or the
share it would receive from the active modes before the next check, exceeds
`tolerance`, or when its share of the outgoing flux exceeds `flux_tolerance`; an
active mode is switched off, and its amplitude dropped, when all three fall
below 1% of their thresholds. The received amplitude grows linearly over the
time between checks, but no longer than the inverse of the local mismatch of the
modes' centrifugal energies, after which an off-resonant response saturates. The
flux share covers the ionized field, which lives in modes holding a tiny share
of the norm but most of the flux: each coupling of an active mode is a source at
that mode's frequency shifted by `m* Omega`, and above zero energy it radiates
into the target mode at the golden-rule rate, computed with the regular solution
in the binary's monopole potential. A mode's share is the larger of its share of
this predicted flux and of the flux measured at `r_ex`. `flux_tolerance = 0`
leaves the flux out. The dropped mass is reported in the log, and the active set
is stored in `checkpoint.bin`, so pruned runs restart exactly. Without a
`[pruning]` table, or with `enabled = false` in it, every mode is evolved.

Fluxes are the most sensitive output, because the ionized field lives in
modes holding a small share of the norm. For the binary and cloud of the
minimal example with `l_max = 12` and `ramp_periods = 1`, over three orbits
(91 modes, 4 threads; flux error over the last orbit):

| `tolerance` | `flux_tolerance` | Active modes | Speedup | Orbit-averaged flux error | Energy error |
| --- | --- | --- | --- | --- | --- |
| `1e-6` | `0` | 5 | 11x | 94% | 3e-3 |
| `1e-6` | `1e-3` | 11 | 10x | 0.8% | 2e-4 |
| `1e-8` | `0` | 20 | 9x | 0.8% | 2e-5 |
| `1e-8` | `1e-3` | 20 | 9x | 0.2% | 2e-5 |
| `1e-10` | `1e-3` | 60 | 2.5x | 5e-5 | 8e-7 |

Check a new setup against a short run without pruning before relying on it.

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
| `regression` | The s-wave absorbers and their projector against an independent addition-theorem reference, the dipolar weights against a direct angular quadrature with their mass and spin budgets (including a superradiant hole), the isolated black hole, the startup and back-reaction ramps, the flux/state time alignment, and single- against multi-threaded assembly and solves |
| `physics` | Hermiticity, norm conservation, hydrogenic eigenstates and their energies, the Jacobi constant on a fixed orbit and its convergence in `dt`, the mirror symmetry between co- and counter-rotation, flux continuity at `r_ex`, the gravitational-wave inspiral against Peters' law, mass balance between the cloud and the accreting holes, the damping mask and the CAP, the per-mode CAP's reflection at three energies (against a reference run on a longer domain) compared with a uniform one, and its norm balance |
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
`http://127.0.0.1:8050` with the orbit, fluxes, energies, norms and binary
power, the modes evolved over time in a pruned run, a snapshot viewer for the
radial profiles and density slices (with a choice of colormaps), an experimental
3D view of the cloud (WebGL volume rendering that can be rotated and zoomed),
and downloads of the data. Every plot
can be saved as SVG or PDF from the download button in its corner. It re-reads
the files as they grow, so it can follow runs in progress, and it remembers its
settings (theme, axes, slice options, colormap, last run) in a browser cookie.
For a run with `secondary_potential = false` or `q = 0` the page switches to a single
hole: the binary power, separation, `t/T` and `R` axes, the companion's centre
and the `q⁻²` scaling of the ionization panels go away, the masses panel shows
the change of the hole's mass `M_1 - M_1(0)`, and the tiles, slices and 3D view
show one hole at rest.
A run with a `[self_gravity]` table (or its `auger_probe.txt`) gets a *Self-gravity &
Auger* section and its own tiles, built around the gravitational Auger test of
Foster & Rodd (arXiv:2611.18833): the 211, 322 and 433 occupations with the paper's
two-level model integrated from the initial occupations, the coefficient
|I_G| = sqrt(pi) |K| / mu^4 from K = psi_00(0) / (c_211^2 c_322*) (averaged over the
late half of the run, against 0.0914 with recoil or 0.143 without), the channel
rates (X absorbed by the hole through the s-wave sink, 322 + 322 emission in
(3,3) + (5,3) through `r_ex`, the dipolar channels) against Γ̄_H N₁²N₂ and
Γ̄_∞ N₂²N₁, the measured Γ̄'s and T_SG next to the paper's, every tracked level
occupation, the chemical potential, energy and self-energy per particle, and the
hole's acceleration. The Compare tab can overlay N_211, N_322 and |I_G| across runs.
The sidebar's *Run folder* panel shows how much disk the run takes (snapshots,
diagnostics, checkpoint) and its *Free space…* button deletes `psi/` snapshots
(and their `psi_dt/` copies): either thinning them to one in every k, or keeping a
single chosen step. The latest snapshot is always kept, since a resumed run
restarts from it with `checkpoint.bin`; a preview lists what goes before anything
is deleted, and the deletion cannot be undone.
Its fonts, d3, jsPDF and svg2pdf.js are bundled in `dashboard/assets` with their
licences.

For notebooks and publication plots, `dashboard/run_loader.py` reads a run folder
into one object, with no server: every diagnostic file, `config.toml`, the grid,
the modes, the dashboard's derived quantities (cloud mass, ionized mass, mass
budget, the signed L_z budget `Lz_budget` and the cloud's `Lz_modes`; for self-gravity
runs `levels` with `levels_index`, `E_self`, `absorbed_swave`, and `auger` with the
complex amplitudes, occupations, K and `I_G`) and one snapshot (the last, or `step=` / `time=`), with `U(l, m)` and
`slice(plane, r_plot, n)` for `|psi|^2` in a plane. `quantities=[...]` reads only
the files those quantities need, and `snapshot=False` skips the snapshot:

```python
import sys; sys.path.insert(0, "path/to/boson-cloud-binary/dashboard")
from run_loader import load_run
run = load_run("runs/Simulation_2026_11_10_10_18_06", step=400)
run.t, run.cloud_mass, run.norms[:, run.mode("2,2")], run.chi_1
x, y, rho = run.snapshot.slice("xy", r_plot=300, n=400)
```
