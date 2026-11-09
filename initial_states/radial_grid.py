"""
Radial grid of the simulation, the same nodes the C++ solver builds
(include/RadialGrid.hpp), and the MODESBIN version-2 header.

[grid] type = "uniform" (default) gives N equally spaced nodes on [0, L].

[grid] type = "stretched" gives nodes epsilon + g_i with g in [0, L], nearly
uniform inside r_core and `stretch` times sparser far outside, with spacing

    h(r) ~ 1 + (stretch - 1)/2 * (1 + tanh((r - r_core) / transition)),

placed equally in x(r) = int_0^r dr'/h(r'). Defaults: r_core = 4 n^2 / mu^2,
stretch = 4, transition = r_core / 4, epsilon = 0.001.
"""

import struct

import numpy as np


def grid_parameters(config):
    """(type, N, L, epsilon, r_core, stretch, transition) from a parsed config.toml."""
    g = config.get("grid", {})
    atom = config.get("atom", {})
    kind = str(g.get("type", "uniform"))
    N = int(g.get("N", 2048))
    L = float(g.get("L", 5000.0))
    epsilon = float(config.get("simulation", {}).get("epsilon", 0.001))
    n, mu = int(atom.get("n", 2)), float(atom.get("mu", 0.2))
    r_core = float(g.get("r_core", 4.0 * n * n / (mu * mu)))
    stretch = float(g.get("stretch", 4.0))
    transition = float(g.get("transition", 0.25 * r_core))
    if kind not in ("uniform", "stretched"):
        raise ValueError("grid.type must be uniform or stretched")
    return kind, N, L, epsilon, r_core, stretch, transition


def radial_grid(config):
    """Radial nodes of the simulation described by `config` (a parsed config.toml)."""
    kind, N, L, epsilon, r_core, stretch, transition = grid_parameters(config)
    if kind == "uniform":
        return np.linspace(0.0, L, N)
    if not (0.0 < r_core < L) or stretch < 1.0 or transition <= 0.0:
        raise ValueError("A stretched grid needs 0 < r_core < L, stretch >= 1 and transition > 0")
    M = 64 * (N - 1) + 1
    g = L * np.arange(M) / (M - 1)
    inv = 1.0 / (1.0 + 0.5 * (stretch - 1.0) * (1.0 + np.tanh((g - r_core) / transition)))
    x = np.concatenate(([0.0], np.cumsum(0.5 * np.diff(g) * (inv[:-1] + inv[1:]))))
    r = epsilon + np.interp(x[-1] * np.arange(N) / (N - 1), x, g)
    r[0], r[-1] = epsilon, epsilon + L
    return r


def is_uniform(r):
    d = np.diff(r)
    return bool(np.allclose(d, d[0], rtol=1e-12, atol=0.0))


def node_weights(r):
    """Trapezoid weights, the measure the solver's norms use."""
    w = np.empty_like(r)
    w[1:-1] = 0.5 * (r[2:] - r[:-2])
    w[0], w[-1] = 0.5 * (r[1] - r[0]), 0.5 * (r[-1] - r[-2])
    return w


def write_modesbin_header(f, r, n_modes):
    """MODESBIN version 2: magic, version, Nr, Nmodes, then the Nr nodes."""
    r = np.asarray(r, dtype="<f8")
    f.write(b"MODESBIN")
    f.write(struct.pack("<i", 2))
    f.write(struct.pack("<q", len(r)))
    f.write(struct.pack("<q", n_modes))
    f.write(r.tobytes(order="C"))
