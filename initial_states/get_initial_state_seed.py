#!/usr/bin/env python3
"""
Random seed field in every evolved (l, m) mode, bound to the central hole, in
the MODESBIN format.

For superradiance tests: start from a small, fluctuating field instead of a
hydrogenic level, and check whether the growing 211 cloud emerges from it (for
example with a spinning, isolated hole: secondary_potential = false or q = 0,
dipole_absorption = true, spin_1 > 0).

Each mode the solver evolves (l <= l_max, l - m even; the others are odd in z
and never coupled) starts from an independent complex Gaussian random field

    F_lm(r) = W(r)  (r / (r + s))^(l+1)  [xi_lm * G_s](r),

where xi_lm is white noise, G_s a Gaussian of correlation length s that
smooths it, (r / (r + s))^(l+1) gives the regular behaviour U ~ r^(l+1) at the
origin, and W(r) = exp(-(r / r_env)^4) keeps it inside r_env.

Raw noise is mostly unbound: it would radiate away at t = 0 and leave behind
slowly decaying high-n levels. So F_lm is projected onto the bound levels
n = l + 1, ..., n_max of the solver's own radial Hamiltonian at t = 0,

    H_l = -(1 / 2 mu) d^2/dr^2 + l(l + 1) / (2 mu r^2) - mu M1 / max(r, R1),

discretised on the solver's nodes exactly as the solver does it (three-point
second difference, Dirichlet ends; M1 = 1 and R1 = q R / (1 + q) is the
primary's distance from the origin). The levels are orthonormal in the
solver's node weights, so

    U_lm = sum_n <phi_nl, F_lm> phi_nl

holds no unbound part and no level above n_max, to rounding. Each level keeps
the random amplitude and phase the noise gives it, so the seed still
fluctuates from mode to mode and level to level. --unfiltered keeps the raw
noise (the previous behaviour).

The weight c_l puts a mass proportional to (l + 1)^(-l_power) in each mode (0:
the same in every mode). The state is normalised to one on the solver's nodes,
as by the other scripts; the solver rescales it to the cloud mass M_c.

Only the primary's monopole enters H_l. With the primary off the origin (R1 >
0) its higher multipoles mix the modes slightly, and a companion present at
t = 0 (ramp_secondary_potential = false) binds the field to the pair instead;
the script warns in both cases.

Defaults follow [atom]: with the Bohr radius r_c = 1 / mu^2 (M1 = 1),
s = r_c / 2 resolves the 2p profile (peak at 4 r_c) and r_env = 10 r_c holds
it. --seed makes the field reproducible; --zero L,M (repeatable) leaves a mode
empty, for instance --zero 1,1 to start without any 211 component.

Usage:
    python get_initial_state_seed.py --config config.toml --output seed.bin [--seed 1]
        [--correlation S] [--envelope R_ENV] [--n-max N] [--l-power P] [--zero 1,1] [--unfiltered]
"""

import argparse
import struct
import sys

import numpy as np
import tomllib

from radial_grid import node_weights, radial_grid, write_modesbin_header

try:
    from scipy.linalg import eigh_tridiagonal
except ModuleNotFoundError:
    eigh_tridiagonal = None


def parse_mode(text):
    try:
        l, m = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected L,M, got {text!r}")
    return l, m


parser = argparse.ArgumentParser(description="Random seed field in every evolved mode, in MODESBIN format.")
parser.add_argument("--config", required=True, help="Simulation config.toml")
parser.add_argument("--output", required=True, help="Output MODESBIN file")
parser.add_argument("--seed", type=int, default=None, help="Random seed (default: a fresh one, printed)")
parser.add_argument("--correlation", type=float, default=None,
                    help="Radial correlation length s of the field (default: r_c / 2)")
parser.add_argument("--envelope", type=float, default=None,
                    help="Radius r_env of the envelope exp(-(r / r_env)^4) (default: 10 r_c, at most 0.8 r_ex)")
parser.add_argument("--l-power", type=float, default=0.0,
                    help="Mass in each mode proportional to (l + 1)^(-P) (default 0: equal in every mode)")
parser.add_argument("--zero", type=parse_mode, action="append", default=[], metavar="L,M",
                    help="Leave mode (L, M) empty; may be repeated")
parser.add_argument("--n-max", type=int, default=None,
                    help="Highest bound level n kept; levels n = l + 1 .. n_max of each l (default: the highest "
                         "n whose levels hold less than 1e-6 of their norm beyond r_ex, at most l_max + 1)")
parser.add_argument("--unfiltered", action="store_true",
                    help="Keep the raw noise, unbound part included (no projection on bound levels)")
args = parser.parse_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)

sim, atom = config.get("simulation", {}), config.get("atom", {})
l_max = int(sim.get("l_max", 5))
mu = float(atom.get("mu", 0.2))
r_c = 1.0 / mu ** 2
r = radial_grid(config)
L = float(r[-1])
r_ex = float(sim.get("r_ex", 0.8 * L))

s = args.correlation if args.correlation is not None else 0.5 * r_c
r_env = args.envelope if args.envelope is not None else min(10.0 * r_c, 0.8 * min(r_ex, L))
if s <= 0 or r_env <= 0:
    sys.exit("--correlation and --envelope must be positive")
if r_env > r_ex:
    print(f"warning: the envelope radius {r_env:g} lies beyond r_ex = {r_ex:g}; part of the seed starts outside "
          "the flux sphere", file=sys.stderr)
dr_max = float(np.max(np.diff(r[r <= 2 * r_env]))) if np.any(r <= 2 * r_env) else float(np.max(np.diff(r)))
if s < 3.0 * dr_max:
    print(f"warning: the correlation length {s:g} is under 3 grid spacings ({dr_max:.3g}) inside the envelope; "
          "the seed has structure the grid cannot resolve", file=sys.stderr)

n_max = args.n_max if args.n_max is not None else l_max + 1
if not args.unfiltered and n_max < 1:
    sys.exit("--n-max must be at least 1")

binary = config.get("binary", {})
q, R = float(binary.get("q", 0.001)), float(binary.get("R", 50.0))
R1 = q / (1.0 + q) * R
secondary = bool(sim.get("secondary_potential", True)) and q > 0.0
if not args.unfiltered:
    if R1 > 0.05 * r_c:
        print(f"warning: the primary sits R1 = {R1:g} from the origin ({R1 / r_c:.2g} r_c); only its monopole binds "
              "the levels, so its higher multipoles leave a small unbound part", file=sys.stderr)
    if secondary and not bool(sim.get("ramp_secondary_potential", True)):
        print("warning: the companion's potential is on at t = 0 (ramp_secondary_potential = false); the levels "
              "are those of the primary alone", file=sys.stderr)

modes = [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1) if (l - m) % 2 == 0]
zeroed = set(args.zero)
unknown = zeroed - set(modes)
if unknown:
    sys.exit(f"--zero names modes the solver does not evolve with l_max = {l_max}: {sorted(unknown)}")
if zeroed >= set(modes):
    sys.exit("--zero leaves every mode empty")
if not args.unfiltered and all(l >= n_max or (l, m) in zeroed for l, m in modes):
    sys.exit("no mode has a bound level with n <= n_max; raise --n-max")

seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2 ** 32)
rng = np.random.default_rng(seed)

# White noise on a uniform auxiliary grid finer than s, smoothed with a Gaussian
# of width s and interpolated onto the solver's nodes (uniform or stretched).
h = s / 6.0  # the smoothed field is resolved at this spacing; finer nodes interpolate it
aux = np.arange(0.0, L + 4.0 * s + h, h)
kernel_x = np.arange(-4.0 * s, 4.0 * s + h / 2, h)
kernel = np.exp(-0.5 * (kernel_x / s) ** 2)
kernel /= np.sqrt(np.sum(kernel ** 2))  # unit variance after smoothing
window = np.exp(-(r / r_env) ** 4)
weights = node_weights(r)


def smooth_noise():
    white = rng.standard_normal(aux.size) + 1j * rng.standard_normal(aux.size)
    field = np.convolve(white, kernel, mode="same") / np.sqrt(2.0)
    return np.interp(r, aux, field.real) + 1j * np.interp(r, aux, field.imag)


def bound_levels(l):
    """Energies and node values (orthonormal in the node weights) of levels n = l + 1 .. n_max of H_l."""
    count = n_max - l
    if count <= 0:
        return np.empty(0), np.zeros((r.size, 0))
    inner = r[1:-1]
    hm, hp = inner - r[:-2], r[2:] - inner
    diag = (2.0 / (hm * hp) + l * (l + 1) / inner ** 2) / (2.0 * mu) - mu / np.maximum(inner, R1)
    lower = -1.0 / (mu * hm * (hm + hp))  # H[i, i - 1]
    upper = -1.0 / (mu * hp * (hm + hp))  # H[i, i + 1]
    w = weights[1:-1]
    # sqrt(w) H / sqrt(w) is symmetric: w_i H[i, i+1] = w_(i+1) H[i+1, i] = -1 / (2 mu h).
    off = upper[:-1] * np.sqrt(w[:-1] / w[1:])
    count = min(count, inner.size)
    if eigh_tridiagonal is not None:
        energies, vectors = eigh_tridiagonal(diag, off, select="i", select_range=(0, count - 1))
    else:
        dense = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        energies, vectors = np.linalg.eigh(dense)
        energies, vectors = energies[:count], vectors[:, :count]
    phi = np.zeros((r.size, energies.size))
    phi[1:-1] = vectors / np.sqrt(w)[:, None]
    return energies, phi


def tail_beyond_r_ex(phi):
    return np.sum((weights[:, None] * np.abs(phi) ** 2)[r > r_ex], axis=0)


if not args.unfiltered and args.n_max is None:
    # The s levels reach furthest at a given n: keep the n whose levels stay inside r_ex.
    energies, phi = bound_levels(0)
    contained = (energies < 0.0) & (tail_beyond_r_ex(phi) < 1e-6)
    n_max = int(np.argmin(contained)) if not contained.all() else n_max
    if n_max < 1:
        sys.exit(f"even the 1s level reaches beyond r_ex = {r_ex:g}; enlarge r_ex or L")

levels = {} if args.unfiltered else {l: bound_levels(l) for l in range(min(l_max, n_max - 1) + 1)}
for l, (energies, phi) in levels.items():
    if np.any(energies >= 0.0):
        sys.exit(f"l = {l}: level n = {l + 1 + int(np.argmax(energies >= 0.0))} is not bound on this grid "
                 f"(E >= 0); lower --n-max or enlarge L")
for l, (energies, phi) in levels.items():
    if energies.size:
        tail = float(tail_beyond_r_ex(phi)[-1])
        if tail > 1e-6:
            print(f"warning: level n = {n_max}, l = {l} has {tail:.2g} of its norm beyond r_ex = {r_ex:g}",
                  file=sys.stderr)

U = {}
for l, m in modes:
    profile = smooth_noise() * window * (r / (r + s)) ** (l + 1)
    profile[0] = 0.0
    if not args.unfiltered:
        energies, phi = levels.get(l, (np.empty(0), np.zeros((r.size, 0))))
        coefficients = phi.T @ (weights * profile)
        profile = phi @ coefficients
        if not np.any(profile) and (l, m) not in zeroed:
            zeroed.add((l, m))  # no bound level with n <= n_max for this l
    if (l, m) in zeroed or not np.any(profile):
        profile[:] = 0.0
    else:
        mass = float(np.sum(weights * np.abs(profile) ** 2))
        profile *= np.sqrt((l + 1.0) ** (-args.l_power) / mass)
    U[(l, m)] = profile

total = sum(float(np.sum(weights * np.abs(p) ** 2)) for p in U.values())
for mode in U:
    U[mode] /= np.sqrt(total)
norm_after = sum(float(np.sum(weights * np.abs(p) ** 2)) for p in U.values())

with open(args.output, "wb") as f:
    write_modesbin_header(f, r, len(U))
    for (l, m), profile in sorted(U.items()):
        f.write(struct.pack("<ii", l, m))
        f.write(np.asarray(profile.real, dtype="<f8").tobytes(order="C"))
        f.write(np.asarray(profile.imag, dtype="<f8").tobytes(order="C"))

share = {mode: float(np.sum(weights * np.abs(p) ** 2)) for mode, p in U.items()}
print("Random seed field")
print(f"  random seed        = {seed}")
print(f"  modes              = {len(U)} (l <= {l_max}, l - m even){', empty: ' + str(sorted(zeroed)) if zeroed else ''}")
print(f"  correlation length = {s:g} M (r_c = {r_c:g} M)")
print(f"  envelope radius    = {r_env:g} M (r_ex = {r_ex:g} M, L = {L:g} M)")
print(f"  l_power            = {args.l_power:g}")
print(f"  norm               = {norm_after:.16e}")
print(f"  share of (1, 1)    = {share.get((1, 1), 0.0):.4g}")
if args.unfiltered:
    print("  filter             = none (raw noise, unbound part included)")
else:
    print(f"  filter             = bound levels n <= {n_max} of the primary (R1 = {R1:g} M); unbound part 0")
    print("  levels (n, l): E / E_hydrogenic, share of the norm")
    for l, (energies, phi) in levels.items():
        for k, E in enumerate(energies):
            n = l + 1 + k
            # Mass of this level in the saved state, summed over m.
            mass = 0.0
            for (l2, m), p in U.items():
                if l2 == l and p.any():
                    mass += abs(np.sum(weights * np.conj(phi[:, k]) * p)) ** 2
            E_h = -mu ** 3 / (2.0 * n * n)
            print(f"    ({n}, {l}): E = {E:.6g} ({E / E_h:.4f}), share {mass:.4g}")
print(f"Saved to {args.output}")
