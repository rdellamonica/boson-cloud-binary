#!/usr/bin/env python3
"""
Normalised spherical Gaussian centred on the origin, in the MODESBIN format.

A quick, dependency-light initial state for tests and smoke runs.

Usage:
    python get_initial_state_gaussian.py --config config.toml --output initial_modes.bin --sigma 10
"""

import argparse
import struct
import numpy as np
import tomllib

parser = argparse.ArgumentParser(description="Generate a normalized spherical Gaussian initial state in MODESBIN format.")
parser.add_argument("--config", required=True, help="Simulation config.toml")
parser.add_argument("--output", required=True, help="Output MODESBIN file")
parser.add_argument("--sigma", required=True, type=float, help="Gaussian width sigma")
args = parser.parse_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)

N = int(config["grid"]["N"])
L = float(config["grid"]["L"])
l_max = int(config["simulation"]["l_max"])
sigma = float(args.sigma)

if sigma <= 0:
    raise ValueError("sigma must be positive.")

r = np.linspace(0.0, L, N)
modes = [(l, m) for l in range(l_max+1) for m in range(-l, l+1)]

# ----------------------------------------------------------------------
# Spherical Gaussian
#
# psi(r) = A exp[-r^2/(2 sigma^2)]
#
# Since psi = U_00(r)/r Y_00 and Y_00 = 1/sqrt(4 pi),
#
# U_00(r) = sqrt(4 pi) A r exp[-r^2/(2 sigma^2)].
#
# The overall amplitude is fixed numerically below so that
# sum_lm int |U_lm|^2 dr = 1 exactly on the simulation grid.
# ----------------------------------------------------------------------

U = {mode: np.zeros(N, dtype=np.complex128) for mode in modes}
U[(0, 0)] = np.sqrt(4*np.pi)*r*np.exp(-0.5*(r/sigma)**2)

# Normalize on the actual finite numerical grid.
norm_before = np.trapezoid(np.abs(U[(0, 0)])**2, x=r)
if norm_before <= 0:
    raise ValueError("Gaussian has zero numerical norm.")

U[(0, 0)] /= np.sqrt(norm_before)

norm_after = sum(np.trapezoid(np.abs(profile)**2, x=r) for profile in U.values())

# ----------------------------------------------------------------------
# MODESBIN writer
# ----------------------------------------------------------------------

def save_modes_binary(filename, U_dict):
    mode_items = sorted(U_dict.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    Nr = len(mode_items[0][1])
    Nmodes = len(mode_items)

    with open(filename, "wb") as f:
        f.write(b"MODESBIN")
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<q", Nr))
        f.write(struct.pack("<q", Nmodes))

        for (l, m), profile in mode_items:
            profile = np.asarray(profile, dtype=np.complex128)
            f.write(struct.pack("<i", l))
            f.write(struct.pack("<i", m))
            f.write(np.asarray(profile.real, dtype="<f8").tobytes(order="C"))
            f.write(np.asarray(profile.imag, dtype="<f8").tobytes(order="C"))

print("Spherical Gaussian initial state")
print(f"  sigma        = {sigma}")
print(f"  grid N       = {N}")
print(f"  box size L   = {L}")
print(f"  dr           = {r[1]-r[0]}")
print(f"  l_max        = {l_max}")
print(f"  norm before  = {norm_before:.16e}")
print(f"  norm after   = {norm_after:.16e}")
print(f"  |U_00| peak at r = {r[np.argmax(np.abs(U[(0,0)]))]:.6g}")

save_modes_binary(args.output, U)
print(f"Saved to {args.output}")