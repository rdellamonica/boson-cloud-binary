#!/usr/bin/env python3
"""Checks for the initial-state scripts in initial_states/.

Each script is run as a subprocess on a small grid; the MODESBIN file it writes
is read back and compared with the analytic state it should contain. Exits 77
when NumPy, SciPy, SymPy or mpmath are missing, so ctest reports a skip.

Usage: test_initial_states.py [--solver path/to/GravitationalMolecule]
"""

import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, "initial_states")

try:
    import numpy as np
    import scipy  # noqa: F401  (used by the COM scripts)
    import sympy  # noqa: F401
    import mpmath  # noqa: F401
except ModuleNotFoundError as missing:
    print(f"skipping: {missing.name} is not installed")
    sys.exit(77)


CONFIG = """
[grid]
N = {N}
L = {L}

[binary]
q = {q}
R = {R}

[atom]
mu = {mu}
n = {n}
l = {l}
m = {m}
M_c = 0.01

[simulation]
dt = 2.0
num_steps = 11
save_every = 5
l_max = {l_max}
r_ex = {r_ex}
epsilon = 0.0
absorption = false
evolve_orbit = false
"""


def write_config(path, **kwargs):
    values = dict(N=129, L=64.0, q=0.1, R=20.0, mu=0.4, n=1, l=0, m=0, l_max=2, r_ex=40.0)
    values.update(kwargs)
    with open(path, "w") as f:
        f.write(CONFIG.format(**values))
    return values


def read_modesbin(path):
    """{(l, m): complex profile} from a MODESBIN file, with its header checked."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"MODESBIN", "bad magic"
    version, = struct.unpack_from("<i", data, 8)
    nr, n_modes = struct.unpack_from("<qq", data, 12)
    assert version == 1, f"unexpected version {version}"
    out, offset = {}, 28
    for _ in range(n_modes):
        l, m = struct.unpack_from("<ii", data, offset)
        offset += 8
        block = np.frombuffer(data, dtype="<f8", count=2 * nr, offset=offset)
        offset += 16 * nr
        out[(l, m)] = block[:nr] + 1j * block[nr:]
    assert offset == len(data), "trailing bytes in MODESBIN"
    return out, nr


def run_script(name, args, cwd):
    command = [sys.executable, os.path.join(SCRIPTS, name), *args]
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=script_env())
    assert done.returncode == 0, f"{name} failed:\n{done.stdout}\n{done.stderr}"
    return done.stdout


def script_env():
    env = dict(os.environ)
    # The scripts import the bundled gravatom package next to them.
    env["PYTHONPATH"] = SCRIPTS + os.pathsep + env.get("PYTHONPATH", "")
    return env


def norm(profiles, r):
    return sum(np.trapezoid(np.abs(p) ** 2, x=r) for p in profiles.values())


def test_gaussian(work):
    values = write_config(os.path.join(work, "config.toml"), N=257, L=100.0, l_max=3)
    run_script("get_initial_state_gaussian.py",
               ["--config", "config.toml", "--output", "gauss.bin", "--sigma", "10"], work)
    profiles, nr = read_modesbin(os.path.join(work, "gauss.bin"))
    r = np.linspace(0.0, values["L"], values["N"])
    assert nr == values["N"]
    expected = {(l, m) for l in range(values["l_max"] + 1) for m in range(-l, l + 1)}
    assert set(profiles) == expected, "mode list does not follow l_max"
    assert abs(norm(profiles, r) - 1.0) < 1e-12, "state is not normalised"

    # Only the monopole is populated, and U_00 is proportional to r exp(-r^2/2 sigma^2).
    for mode, profile in profiles.items():
        if mode != (0, 0):
            assert not profile.any(), f"mode {mode} is not empty"
    u = profiles[(0, 0)].real
    assert np.allclose(profiles[(0, 0)].imag, 0), "Gaussian state is not real"
    shape = r * np.exp(-0.5 * (r / 10.0) ** 2)
    scale = u[1] / shape[1]
    assert np.allclose(u, scale * shape, rtol=1e-12, atol=1e-14), "wrong radial profile"
    assert u[0] == 0, "U_00 does not vanish at the origin"

    # <r^2> = 3 sigma^2 / 2 for a density exp(-r^2 / sigma^2).
    r_rms = math.sqrt(np.trapezoid(r ** 2 * u ** 2, x=r) / np.trapezoid(u ** 2, x=r))
    assert abs(r_rms - 10.0 * math.sqrt(1.5)) < 1e-6, f"width is wrong: rms radius {r_rms}"


def test_gaussian_rejects_bad_sigma(work):
    write_config(os.path.join(work, "config.toml"))
    command = [sys.executable, os.path.join(SCRIPTS, "get_initial_state_gaussian.py"),
               "--config", "config.toml", "--output", "bad.bin", "--sigma", "-1"]
    done = subprocess.run(command, cwd=work, capture_output=True, text=True, env=script_env())
    assert done.returncode != 0, "a negative width was accepted"


def hydrogenic_1s(r, mu):
    """r R_10(r) for a hydrogenic atom of a mass-1 hole: a0 = 1 / mu^2."""
    a0 = 1.0 / mu ** 2
    return 2.0 * r * np.exp(-r / a0) / a0 ** 1.5


def test_com_fast_primary(work):
    # A 1s cloud on the primary of a q = 0.1 binary, so the atom sits at
    # x = R1 = R q / (1 + q) and the state is the translated 1s state.
    values = write_config(os.path.join(work, "config.toml"), N=257, L=80.0, mu=0.4,
                          q=0.1, R=20.0, n=1, l=0, m=0, l_max=4)
    run_script("get_initial_state_COM_fast.py",
               ["--config", "config.toml", "--output", "com.bin", "--n-quad", "48"], work)
    profiles, nr = read_modesbin(os.path.join(work, "com.bin"))
    r = np.linspace(0.0, values["L"], values["N"])
    assert nr == values["N"]
    assert abs(norm(profiles, r) - 1.0) < 1e-6, "translated state is not normalised"
    # Only l - m even modes carry the equatorial-symmetric atom.
    for (l, m), profile in profiles.items():
        if (l - m) % 2:
            assert np.abs(profile).max() < 1e-10, f"mode ({l}, {m}) breaks equatorial symmetry"
    assert np.abs(profiles[(1, 1)]).max() > 1e-3, "translation produced no dipole"

    # The field on the positive x axis is the 1s profile about x = R1.
    R1 = values["R"] * values["q"] / (1 + values["q"])
    from scipy.special import sph_harm_y
    for x in (R1 + 2.0, R1 + 6.0):
        i = int(round(x / (r[1] - r[0])))
        total = 0j
        for (l, m), profile in profiles.items():
            total += profile[i] * sph_harm_y(l, m, np.pi / 2, 0.0) / r[i]
        expected = hydrogenic_1s(abs(x - R1), values["mu"]) / (abs(x - R1) or 1) / math.sqrt(4 * math.pi)
        assert abs(abs(total) - expected) < 0.05 * expected, \
            f"reconstructed density at x={x} is {abs(total)}, expected {expected}"


def test_com_fast_secondary_is_mirrored(work):
    values = write_config(os.path.join(work, "config.toml"), N=129, L=64.0, mu=0.4, q=1.0, R=20.0,
                          n=1, l=0, m=0, l_max=2)
    run_script("get_initial_state_COM_fast.py",
               ["--config", "config.toml", "--output", "primary.bin", "--n-quad", "32"], work)
    run_script("get_initial_state_COM_fast.py",
               ["--config", "config.toml", "--output", "secondary.bin", "--atom", "secondary",
                "--n-quad", "32"], work)
    primary, _ = read_modesbin(os.path.join(work, "primary.bin"))
    secondary, _ = read_modesbin(os.path.join(work, "secondary.bin"))
    # An equal-mass binary puts the two atoms at +-R/2. The displaced 1s state
    # is axially symmetric, so a rotation by pi about z maps one onto the other:
    # U^(2)_lm = (-1)^m U^(1)_lm.
    for mode, profile in primary.items():
        sign = (-1.0) ** mode[1]
        assert np.allclose(secondary[mode], sign * profile, atol=1e-8), \
            f"mode {mode} of the companion atom is not the mirror image"


def test_solver_accepts_generated_state(work, solver):
    values = write_config(os.path.join(work, "config.toml"), N=129, L=64.0, l_max=2, r_ex=40.0)
    run_script("get_initial_state_gaussian.py",
               ["--config", "config.toml", "--output", "state.bin", "--sigma", "8"], work)
    done = subprocess.run([solver, "config.toml", "state.bin"], cwd=work, capture_output=True, text=True)
    assert done.returncode == 0, f"solver failed on a generated state:\n{done.stdout}\n{done.stderr}"
    folders = [d for d in os.listdir(work) if d.startswith("Simulation_")]
    assert len(folders) == 1, "solver did not write exactly one run folder"
    norms = np.loadtxt(os.path.join(work, folders[0], "norms.txt"), ndmin=2)
    assert norms.shape == (10, 6), f"norms.txt has shape {norms.shape}"
    mass = values["mu"] * norms.sum(axis=1)
    assert abs(mass[0] - 0.01) < 2e-3 * 0.01, f"cloud mass is {mass[0]}, not M_c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", help="GravitationalMolecule executable, for the end-to-end check")
    args = parser.parse_args()

    tests = [test_gaussian, test_gaussian_rejects_bad_sigma, test_com_fast_primary,
             test_com_fast_secondary_is_mirrored]
    failures = 0
    for test in tests:
        with tempfile.TemporaryDirectory() as work:
            try:
                test(work)
                print(f"[pass] {test.__name__}")
            except Exception as error:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"[FAIL] {test.__name__}: {error}", file=sys.stderr)
    if args.solver:
        with tempfile.TemporaryDirectory() as work:
            try:
                test_solver_accepts_generated_state(work, os.path.abspath(args.solver))
                print("[pass] test_solver_accepts_generated_state")
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"[FAIL] test_solver_accepts_generated_state: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
