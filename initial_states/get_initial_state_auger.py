#!/usr/bin/env python3
"""
211 cloud at the origin with a small 322 seed, for self-gravity (gravitational
Auger) runs, in the MODESBIN format.

The gravitational Auger process 211 + 211 -> 322 + X (Foster & Rodd,
arXiv:2611.18833) is stimulated by the 322 occupation. A classical field has
no spontaneous emission, so the 322 level must be seeded. This script
writes

    U_11(r) = sqrt(1 - f) f_21(r),      U_22(r) = sqrt(f) e^{i phase} f_32(r),

with every other mode empty. f is the fraction of the particles in 322
(--seed-fraction, default 1e-3). f_21 and f_32 are radial levels on the
solver's own grid, orthonormal in its node weights.

By default (--self-consistent) the levels are those of the hole plus the
cloud's own potential:

    H_l = -(1 / 2 mu) d^2/dr^2 + l(l + 1) / (2 mu r^2) - mu / r + V_l[U_11],

    V_l(r) = mu sum_{L = 0, 2} G(l l; L 0; l l) Phi_L0(r),

    Phi_L0(r) = -4 pi mu / (2L + 1) sum_j w_j r_<^L / r_>^(L+1) n_L0(r_j),
    n_L0 = G(1 1; L 0; 1 1) |U_11|^2  with  sum_j w_j |U_11|^2 = N = M_c / mu.

This is the solver's self-gravity term (src/CrankNicolsonModes_selfgravity.cpp)
restricted to the diagonal of the 211 density, with the same Gaunt
coefficients, discrete Green's function and weights. f_21 is the lowest l = 1
level of H_1 iterated to self-consistency (Hartree). f_32 is the lowest l = 2
level of H_2 in that potential. The 211 cloud then starts stationary under
self-gravity instead of breathing at the self-energy scale. The small
off-diagonal admixture into (3,1), of relative size ~ M_c/M, is left out.
The seed's own potential (order f) is neglected. --bare uses the hydrogenic
levels of the grid (no self-gravity), as written to levels.txt by the solver.

The state is normalised to one on the solver's nodes, as by the other
scripts. The solver rescales it to the cloud mass M_c, which must be the
M_c used here for the self-consistent potential (it is read from the same
config).

Usage:
    python get_initial_state_auger.py --config config.toml --output auger.bin
        [--seed-fraction 1e-3] [--phase 0] [--bare] [--tolerance 1e-12]
"""

import argparse
import math
import struct
import sys

import numpy as np
import tomllib

from radial_grid import node_weights, radial_grid, write_modesbin_header

try:
    from scipy.linalg import eigh_tridiagonal
except ModuleNotFoundError:
    eigh_tridiagonal = None

parser = argparse.ArgumentParser(description="211 at the origin with a small 322 seed, in MODESBIN format.")
parser.add_argument("--config", required=True, help="Simulation config.toml")
parser.add_argument("--output", required=True, help="Output MODESBIN file")
parser.add_argument("--seed-fraction", type=float, default=1e-3,
                    help="Fraction f of the particles in 322 (default 1e-3)")
parser.add_argument("--phase", type=float, default=0.0, help="Phase of the 322 seed in radians (default 0)")
parser.add_argument("--bare", action="store_true",
                    help="Use the hole's levels without the cloud's self-gravity (default: self-consistent 211)")
parser.add_argument("--tolerance", type=float, default=1e-12,
                    help="Self-consistency: stop when the 211 energy changes by less than this, relative")
parser.add_argument("--max-iterations", type=int, default=200, help="Self-consistency: most Hartree iterations")
args = parser.parse_args()

if not 0.0 < args.seed_fraction < 1.0:
    sys.exit("--seed-fraction must lie in (0, 1)")

with open(args.config, "rb") as f:
    config = tomllib.load(f)
sim, atom, binary = config.get("simulation", {}), config.get("atom", {}), config.get("binary", {})
l_max = int(sim.get("l_max", 5))
mu = float(atom.get("mu", 0.2))
M_c = float(atom.get("M_c", 0.0))
r = radial_grid(config)
w = node_weights(r)
r_ex = float(sim.get("r_ex", 0.8 * float(r[-1])))
a0 = 1.0 / mu ** 2

if l_max < 2:
    sys.exit("l_max must be at least 2 to hold the 322 seed (mode (2, 2))")
q, R = float(binary.get("q", 0.001)), float(binary.get("R", 50.0))
if q / (1.0 + q) * R != 0.0:
    sys.exit("The hole must sit at the origin (binary.q = 0, or secondary_potential = false with R1 = 0)")
if bool(sim.get("secondary_potential", True)) and q > 0.0:
    sys.exit("A companion is on: self-gravity runs need an isolated hole (binary.q = 0)")
sg = config.get("self_gravity")
if not isinstance(sg, dict) or not sg.get("enabled", True):
    print("warning: the config has no enabled [self_gravity] table; the run will not include self-gravity",
          file=sys.stderr)
    if "self-gravity" in config:
        print("warning: it has [self-gravity] (hyphen), which the solver ignores; rename it [self_gravity]",
              file=sys.stderr)
self_consistent = not args.bare
if self_consistent and isinstance(sg, dict) and float(sg.get("ramp_time", 0.0)) > 0.0:
    print("warning: self_gravity.ramp_time > 0 switches the potential on gradually, but this state is "
          "self-consistent with the full potential; use --bare with a ramp, or ramp_time = 0", file=sys.stderr)
if self_consistent and not M_c > 0.0:
    sys.exit("atom.M_c must be positive for the self-consistent potential")
N = M_c / mu


def wigner_3j(j1, j2, j3, m1, m2, m3):
    """Wigner 3j symbol for integer arguments (Racah formula)."""
    if m1 + m2 + m3 != 0 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    f = math.factorial
    pre = math.sqrt(f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1)
                    * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3))
    total = 0.0
    for k in range(0, j1 + j2 + j3 + 1):
        d = [k, j1 + j2 - j3 - k, j1 - m1 - k, j2 + m2 - k, j3 - j2 + m1 + k, j3 - j1 - m2 + k]
        if min(d) < 0:
            continue
        total += (-1) ** k / math.prod(f(x) for x in d)
    return (-1) ** (j1 - j2 - m3) * pre * total


def gaunt(l, m, L, M, lp, mp):
    """G(lm; LM; l'm') = int Y*_lm Y_LM Y_l'm' dOmega, as in the solver."""
    if m != M + mp:
        return 0.0
    return ((-1) ** abs(m) * math.sqrt((2 * l + 1) * (2 * L + 1) * (2 * lp + 1) / (4 * math.pi))
            * wigner_3j(l, L, lp, 0, 0, 0) * wigner_3j(l, L, lp, -m, M, mp))


def potential(n_density, L):
    """Phi_L0 on the nodes from the density n_L0, with the solver's discrete Green's function."""
    src = np.where(r > 0.0, w * n_density / np.where(r > 0.0, r, 1.0), 0.0)
    A = np.empty_like(r)
    B = np.zeros_like(r)
    A[0] = src[0]
    inner = np.zeros_like(r)
    inner[1:] = np.where(r[1:] > 0.0, (r[:-1] / r[1:]) ** (L + 1), 0.0)
    outer = (r[:-1] / r[1:]) ** L
    for i in range(1, r.size):
        A[i] = inner[i] * A[i - 1] + src[i]
    for i in range(r.size - 2, -1, -1):
        B[i] = outer[i] * (B[i + 1] + src[i + 1])
    return -4.0 * math.pi * mu / (2 * L + 1) * (A + B)


def lowest_level(l, V):
    """Lowest level of H_l + V on the interior nodes: energy and node values, sum w f^2 = 1, f > 0 inside."""
    inner = r[1:-1]
    hm, hp = inner - r[:-2], r[2:] - inner
    diag = (2.0 / (hm * hp) + l * (l + 1) / inner ** 2) / (2.0 * mu) - mu / inner + V[1:-1]
    upper = -1.0 / (mu * hp * (hm + hp))
    wi = w[1:-1]
    off = upper[:-1] * np.sqrt(wi[:-1] / wi[1:])
    if eigh_tridiagonal is not None:
        energies, vectors = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
    else:
        energies, vectors = np.linalg.eigh(np.diag(diag) + np.diag(off, 1) + np.diag(off, -1))
    f = np.zeros_like(r)
    f[1:-1] = vectors[:, 0] / np.sqrt(wi)
    f /= math.sqrt(float(np.sum(w * f * f)))
    if f[np.argmax(np.abs(f) > 1e-3 * np.abs(f).max())] < 0:
        f = -f
    return float(energies[0]), f


G11 = {L: gaunt(1, 1, L, 0, 1, 1) for L in (0, 2)}
G22 = {L: gaunt(2, 2, L, 0, 2, 2) for L in (0, 2)}


def self_potential(f21, G):
    """mu sum_L G_L Phi_L0 of the 211 density N f21^2."""
    rho = N * f21 * f21
    return mu * sum(G[L] * potential(G11[L] * rho, L) for L in (0, 2))


zero = np.zeros_like(r)
E21_bare, f21_bare = lowest_level(1, zero)
E32_bare, f32_bare = lowest_level(2, zero)
E21, f21, E32, f32 = E21_bare, f21_bare, E32_bare, f32_bare
iterations = 0
if self_consistent:
    V1 = self_potential(f21, G11)
    for iterations in range(1, args.max_iterations + 1):
        E_new, f_new = lowest_level(1, V1)
        change = abs(E_new - E21) / abs(E_new)
        E21, f21 = E_new, f_new
        V1 = 0.5 * V1 + 0.5 * self_potential(f21, G11)   # linear mixing
        if change < args.tolerance and iterations > 2:
            break
    else:
        sys.exit(f"self-consistency did not converge in {args.max_iterations} iterations (last change {change:.2g})")
    V1 = self_potential(f21, G11)
    E21, f21 = lowest_level(1, V1)
    E32, f32 = lowest_level(2, self_potential(f21, G22))

tail = float(np.sum((w * f21 ** 2)[r > r_ex]))
if tail > 1e-8:
    print(f"warning: {tail:.2g} of the 211 profile lies beyond r_ex = {r_ex:g}", file=sys.stderr)
if r[-1] < 40.0 * a0:
    print(f"warning: the box L = {r[-1]:g} is under 40 Bohr radii ({40 * a0:g}); the Auger partner and the (0,0) "
          "response need room", file=sys.stderr)

frac = args.seed_fraction
U = {(l, m): np.zeros(r.size, dtype=complex) for l in range(l_max + 1) for m in range(-l, l + 1) if (l - m) % 2 == 0}
U[(1, 1)] = math.sqrt(1.0 - frac) * f21.astype(complex)
U[(2, 2)] = math.sqrt(frac) * np.exp(1j * args.phase) * f32.astype(complex)
U = {k: v for k, v in U.items() if v.any()}
norm = sum(float(np.sum(w * np.abs(v) ** 2)) for v in U.values())

with open(args.output, "wb") as f:
    write_modesbin_header(f, r, len(U))
    for (l, m), profile in sorted(U.items()):
        f.write(struct.pack("<ii", l, m))
        f.write(np.asarray(profile.real, dtype="<f8").tobytes(order="C"))
        f.write(np.asarray(profile.imag, dtype="<f8").tobytes(order="C"))

overlap21 = float(np.sum(w * f21 * f21_bare)) ** 2
overlap32 = float(np.sum(w * f32 * f32_bare)) ** 2
E_h = lambda n: -mu ** 3 / (2.0 * n * n)
print("211 cloud with a 322 seed")
print(f"  mu = {mu:g}, M_c = {M_c:g}, N = M_c / mu = {N:g}, a0 = {a0:g} M, nodes = {r.size}, L = {r[-1]:g}")
print(f"  322 seed fraction  = {frac:g} (phase {args.phase:g})")
print(f"  levels             = {'self-consistent with the 211 self-gravity' if self_consistent else 'bare (no self-gravity)'}"
      + (f", {iterations} Hartree iterations" if self_consistent else ""))
print(f"  E_211 = {E21:.8g} (bare {E21_bare:.8g}, hydrogenic {E_h(2):.8g})")
print(f"  E_322 = {E32:.8g} (bare {E32_bare:.8g}, hydrogenic {E_h(3):.8g})")
if self_consistent:
    print(f"  self-energy shift of 211: {E21 - E21_bare:.4g} ({(E21 - E21_bare) / abs(E21_bare):.3g} relative)")
    print(f"  |<bare|self-consistent>|^2: 211 {overlap21:.10f}, 322 {overlap32:.10f}")
print(f"  norm = {norm:.16e}")
print(f"Saved to {args.output}")
