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
{grid_extra}

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
    values = dict(N=129, L=64.0, q=0.1, R=20.0, mu=0.4, n=1, l=0, m=0, l_max=2, r_ex=40.0, grid_extra="")
    values.update(kwargs)
    with open(path, "w") as f:
        f.write(CONFIG.format(**values))
    return values


def read_modesbin(path, with_nodes=False):
    """{(l, m): complex profile} from a MODESBIN file, with its header checked."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"MODESBIN", "bad magic"
    version, = struct.unpack_from("<i", data, 8)
    nr, n_modes = struct.unpack_from("<qq", data, 12)
    assert version in (1, 2), f"unexpected version {version}"
    out, offset, nodes = {}, 28, None
    if version == 2:
        nodes = np.frombuffer(data, dtype="<f8", count=nr, offset=offset).copy()
        offset += 8 * nr
    for _ in range(n_modes):
        l, m = struct.unpack_from("<ii", data, offset)
        offset += 8
        block = np.frombuffer(data, dtype="<f8", count=2 * nr, offset=offset)
        offset += 16 * nr
        out[(l, m)] = block[:nr] + 1j * block[nr:]
    assert offset == len(data), "trailing bytes in MODESBIN"
    return (out, nr, nodes) if with_nodes else (out, nr)


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


def test_seed(work):
    # Raw random seed field (--unfiltered): every evolved mode filled, normalised, regular at
    # the origin, inside the envelope, reproducible with --seed, --zero honoured.
    values = write_config(os.path.join(work, "config.toml"), N=401, L=200.0, mu=0.4, l_max=4, r_ex=160.0)
    args = ["--config", "config.toml", "--seed", "7", "--envelope", "60", "--correlation", "1.5", "--unfiltered"]
    run_script("get_initial_state_seed.py", args + ["--output", "a.bin"], work)
    run_script("get_initial_state_seed.py", args + ["--output", "b.bin"], work)
    run_script("get_initial_state_seed.py", args + ["--output", "c.bin", "--zero", "1,1", "--l-power", "2"], work)
    a, nr = read_modesbin(os.path.join(work, "a.bin"))
    b, _ = read_modesbin(os.path.join(work, "b.bin"))
    c, _ = read_modesbin(os.path.join(work, "c.bin"))
    r = np.linspace(0.0, values["L"], values["N"])
    expected = {(l, m) for l in range(values["l_max"] + 1) for m in range(-l, l + 1) if (l - m) % 2 == 0}
    assert set(a) == expected, f"modes {sorted(a)}"
    assert abs(norm(a, r) - 1.0) < 1e-12 and abs(norm(c, r) - 1.0) < 1e-12, "seed is not normalised"
    assert all(np.array_equal(a[k], b[k]) for k in a), "--seed does not reproduce the field"
    masses = {k: np.trapezoid(np.abs(p) ** 2, x=r) for k, p in a.items()}
    assert max(masses.values()) / min(masses.values()) < 1.0 + 1e-9, "modes do not carry equal mass"
    assert not c[(1, 1)].any(), "--zero 1,1 left the 211 mode populated"
    c_mass = {k: np.trapezoid(np.abs(p) ** 2, x=r) for k, p in c.items() if k != (1, 1)}
    assert abs(c_mass[(0, 0)] / c_mass[(4, 0)] - 25.0) < 1e-6, "--l-power does not weight the modes"
    for (l, m), p in a.items():
        assert p[0] == 0 and np.all(np.isfinite(p)), f"mode {(l, m)} is not regular"
        assert np.max(np.abs(p[r > 120])) < 1e-6 * np.max(np.abs(p)), f"mode {(l, m)} leaks past the envelope"
        assert np.abs(p[1:6]).max() < np.abs(p).max(), f"mode {(l, m)} peaks at the origin"
    # The field fluctuates: many sign changes, independent between modes.
    assert np.sum(np.diff(np.sign(a[(1, 1)].real[r < 60])) != 0) > 5, "field does not fluctuate"
    unit = [p / np.linalg.norm(p) for p in a.values()]
    overlaps = [abs(np.vdot(unit[i], unit[j])) for i in range(len(unit)) for j in range(i)]
    assert np.mean(overlaps) < 0.3, f"modes are correlated: mean overlap {np.mean(overlaps)}"


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


def seed_levels(r, mu, l, R1=0.0):
    """Every level of the solver's radial Hamiltonian for l on a uniform grid (dense, independent)."""
    h = r[1] - r[0]
    inner = r[1:-1]
    H = (np.diag(1.0 / (mu * h * h) + l * (l + 1) / (2 * mu * inner ** 2) - mu / np.maximum(inner, R1))
         - np.diag(np.full(inner.size - 1, 0.5 / (mu * h * h)), 1)
         - np.diag(np.full(inner.size - 1, 0.5 / (mu * h * h)), -1))
    energies, vectors = np.linalg.eigh(H)
    return energies, vectors / math.sqrt(h)  # orthonormal in the node weights h


def test_seed_bound(work):
    # Default seed: projected on the bound levels n <= n_max of the central hole. No weight
    # on unbound or higher levels, every mode with a level filled, still random.
    values = write_config(os.path.join(work, "config.toml"), N=801, L=400.0, q=0.0, mu=0.4, l_max=3, r_ex=300.0)
    args = ["--config", "config.toml", "--envelope", "60", "--correlation", "1.5"]
    out = run_script("get_initial_state_seed.py", args + ["--seed", "3", "--n-max", "3", "--output", "a.bin"], work)
    run_script("get_initial_state_seed.py", args + ["--seed", "4", "--n-max", "3", "--output", "b.bin"], work)
    a, _ = read_modesbin(os.path.join(work, "a.bin"))
    b, _ = read_modesbin(os.path.join(work, "b.bin"))
    r = np.linspace(0.0, values["L"], values["N"])
    mu, h = values["mu"], r[1] - r[0]
    assert "unbound part 0" in out, out
    assert abs(norm(a, r) - 1.0) < 1e-12, "bound seed is not normalised"
    for (l, m), p in a.items():
        if l >= 3:
            assert not p.any(), f"mode {(l, m)} has no level with n <= 3 but is populated"
            continue
        assert p.any(), f"mode {(l, m)} is empty"
        energies, phi = seed_levels(r, mu, l)
        c = phi.T @ (h * p[1:-1])
        kept = 3 - l  # n = l + 1 .. 3
        assert np.all(energies[:kept] < 0), "kept levels are not bound"
        # Hydrogenic energies -mu^3 / (2 n^2) to the grid's accuracy.
        for k in range(kept):
            n = l + 1 + k
            assert abs(energies[k] / (-mu ** 3 / (2 * n * n)) - 1) < 1e-2, f"level ({n}, {l}) energy"
        outside = float(np.sum(np.abs(c[kept:]) ** 2)) / float(np.sum(np.abs(c) ** 2))
        assert outside < 1e-24, f"mode {(l, m)} has {outside:.3g} outside the bound levels n <= 3"
        assert np.min(np.abs(c[:kept])) > 1e-6 * np.max(np.abs(c[:kept])), f"mode {(l, m)} misses a level"
    # Random: another seed gives other level amplitudes and phases.
    ratios = [np.vdot(a[k], b[k]) / np.sqrt(np.vdot(a[k], a[k]).real * np.vdot(b[k], b[k]).real)
              for k in a if a[k].any() and k[0] < 2]
    assert np.mean(np.abs(ratios)) < 0.95, "different seeds give the same bound state"
    # By default, n_max is the highest n whose s level stays inside r_ex.
    out = run_script("get_initial_state_seed.py", args + ["--seed", "3", "--output", "c.bin"], work)
    n_default = int(out.split("bound levels n <= ")[1].split()[0])
    energies, phi = seed_levels(r, mu, 0)
    tail = np.sum(h * np.abs(phi[r[1:-1] > values["r_ex"]]) ** 2, axis=0)
    assert tail[n_default - 1] < 1e-6 <= tail[n_default], f"default n_max {n_default} does not match r_ex"


def test_auger(work):
    # 211 with a 322 seed at the origin: only (1, 1) and (2, 2), seed fraction f, bare levels equal
    # to the solver's radial levels, and the self-consistent 211 an eigenstate of H_1 + its own potential.
    from sympy.physics.wigner import gaunt as sympy_gaunt
    values = write_config(os.path.join(work, "config.toml"), N=801, L=400.0, q=0.0, mu=0.4, l_max=3, r_ex=300.0)
    with open(os.path.join(work, "config.toml"), "a") as f:
        f.write("\n[self_gravity]\nrecoil = true\n")
    r = np.linspace(0.0, values["L"], values["N"])
    mu, h, N = values["mu"], r[1] - r[0], 0.01 / values["mu"]
    frac = 1e-3
    run_script("get_initial_state_auger.py", ["--config", "config.toml", "--output", "bare.bin", "--bare",
                                              "--seed-fraction", str(frac), "--phase", "0.5"], work)
    out = run_script("get_initial_state_auger.py", ["--config", "config.toml", "--output", "sc.bin",
                                                    "--seed-fraction", str(frac)], work)
    bare, _ = read_modesbin(os.path.join(work, "bare.bin"))
    sc, _ = read_modesbin(os.path.join(work, "sc.bin"))
    assert set(bare) == {(1, 1), (2, 2)} and set(sc) == {(1, 1), (2, 2)}, f"modes {sorted(bare)}"
    for state in (bare, sc):
        assert abs(norm(state, r) - 1.0) < 1e-12, "state is not normalised"
        assert abs(np.trapezoid(np.abs(state[(2, 2)]) ** 2, x=r) - frac) < 1e-12, "seed fraction"
    assert abs(np.angle(bare[(2, 2)][np.argmax(np.abs(bare[(2, 2)]))]) - 0.5) < 1e-12, "seed phase"
    for (l, n) in ((1, 2), (2, 3)):
        energies, phi = seed_levels(r, mu, l)
        f = phi[:, 0] * np.sign(phi[np.argmax(np.abs(phi[:, 0])), 0])
        p = np.abs(bare[(l, l)][1:-1])
        p /= math.sqrt(np.sum(h * p ** 2))
        assert np.max(np.abs(p - np.abs(f))) < 1e-8, f"bare ({n}, {l}, {l}) is not the grid level"
    # Self-consistent 211: H_1 f + mu sum_L G_L Phi_L0[N f^2] f = E f, with an independent dense Green's function.
    G = {L: float(sympy_gaunt(1, L, 1, -1, 0, 1)) * -1 for L in (0, 2)}   # int Y*_11 Y_L0 Y_11
    f = np.real(sc[(1, 1)]) / math.sqrt(1 - frac)
    inner = r[1:-1]
    K = lambda L: np.minimum.outer(inner, inner) ** L / np.maximum.outer(inner, inner) ** (L + 1)
    V = sum(mu * G[L] * (-4 * math.pi * mu / (2 * L + 1)) * (K(L) @ (h * G[L] * N * f[1:-1] ** 2)) for L in (0, 2))
    H = (np.diag(1.0 / (mu * h * h) + 1 / (mu * inner ** 2) - mu / inner + V)
         - np.diag(np.full(inner.size - 1, 0.5 / (mu * h * h)), 1)
         - np.diag(np.full(inner.size - 1, 0.5 / (mu * h * h)), -1))
    Hf = H @ f[1:-1]
    E = float(f[1:-1] @ Hf) * h
    assert np.max(np.abs(Hf - E * f[1:-1])) < 1e-8 * np.max(np.abs(E * f[1:-1])), "211 is not self-consistent"
    assert E < seed_levels(r, mu, 1)[0][0], "self-gravity does not lower the 211 energy"
    assert "self-consistent" in out, out


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


def test_stretched_grid(work, solver):
    # A Gaussian written on a stretched grid carries its nodes, which are the
    # solver's own, so the solver copies it and starts from mass M_c.
    grid = 'type = "stretched"\nr_core = 30.0\nstretch = 4.0\ntransition = 6.0'
    values = write_config(os.path.join(work, "config.toml"), N=97, L=200.0, l_max=2, r_ex=40.0, grid_extra=grid)
    run_script("get_initial_state_gaussian.py",
               ["--config", "config.toml", "--output", "state.bin", "--sigma", "8"], work)
    profiles, nr, nodes = read_modesbin(os.path.join(work, "state.bin"), with_nodes=True)
    assert nr == values["N"] and nodes is not None, "stretched state has no nodes"
    spacing = np.diff(nodes)
    assert spacing[-1] > 3.5 * spacing[0], "grid is not stretched"
    assert abs(norm(profiles, nodes) - 1.0) < 1e-12, "state is not normalised on its nodes"
    done = subprocess.run([solver, "config.toml", "state.bin"], cwd=work, capture_output=True, text=True)
    assert done.returncode == 0, f"solver failed on a stretched grid:\n{done.stdout}\n{done.stderr}"
    assert "Interpolating" not in done.stdout, "solver did not recognise its own nodes"
    folder = os.path.join(work, [d for d in os.listdir(work) if d.startswith("Simulation_")][0])
    r = np.loadtxt(os.path.join(folder, "r.txt"))
    assert np.allclose(r, nodes, rtol=0, atol=1e-9 * values["L"]), "solver nodes differ from the script's"
    norms = np.loadtxt(os.path.join(folder, "norms.txt"), ndmin=2)
    mass = values["mu"] * norms.sum(axis=1)
    assert abs(mass[0] - 0.01) < 2e-3 * 0.01, f"cloud mass is {mass[0]}, not M_c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", help="GravitationalMolecule executable, for the end-to-end check")
    args = parser.parse_args()

    tests = [test_gaussian, test_gaussian_rejects_bad_sigma, test_seed, test_seed_bound, test_auger, test_com_fast_primary,
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
        for test in (test_solver_accepts_generated_state, test_stretched_grid):
            with tempfile.TemporaryDirectory() as work:
                try:
                    test(work, os.path.abspath(args.solver))
                    print(f"[pass] {test.__name__}")
                except Exception as error:  # noqa: BLE001
                    failures += 1
                    print(f"[FAIL] {test.__name__}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
