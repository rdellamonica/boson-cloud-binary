#!/usr/bin/env python3
"""Checks for dashboard/live_dashboard.py on a real run folder.

With --solver, a short simulation is run first; otherwise a minimal run folder
is written by hand. The dashboard's run discovery, payload builders, snapshot
viewer, zip export and HTTP endpoints are exercised against it. Exits 77 when
NumPy is missing, so ctest reports a skip.

Usage: test_dashboard.py [--solver path/to/GravitationalMolecule]
"""

import argparse
import base64
import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
except ModuleNotFoundError as missing:
    print(f"skipping: {missing.name} is not installed")
    sys.exit(77)


def load_dashboard():
    path = os.path.join(REPO, "dashboard", "live_dashboard.py")
    spec = importlib.util.spec_from_file_location("live_dashboard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = """
[grid]
N = 65
L = 64.0
[binary]
q = 0.5
R = 20.0
[atom]
mu = 0.2
n = 1
l = 0
m = 0
M_c = 0.02
[simulation]
dt = 2.0
num_steps = 21
save_every = 10
l_max = 2
r_ex = 40.0
epsilon = 0.0
evolve_orbit = true
absorption = true
epsilon_smear = 4.0
steps_per_period = 32
[parallel]
threads = 1
"""

MODES = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2)]
NR, ROWS = 65, 20


def write_modesbin(path, profiles):
    with open(path, "wb") as f:
        f.write(b"MODESBIN")
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<qq", NR, len(profiles)))
        for (l, m), profile in profiles.items():
            f.write(struct.pack("<ii", l, m))
            f.write(np.asarray(profile.real, dtype="<f8").tobytes())
            f.write(np.asarray(profile.imag, dtype="<f8").tobytes())


def real_run(root, solver):
    """A short simulation in root; returns its folder."""
    with open(os.path.join(root, "config.toml"), "w") as f:
        f.write(CONFIG)
    r = np.arange(NR, dtype=float)
    state = {(0, 0): (r * np.exp(-r / 8.0)).astype(complex),
             (1, 1): (0.3j * r ** 2 * np.exp(-r / 10.0) / 10.0)}
    scale = math.sqrt(sum(float(np.sum(np.abs(p) ** 2)) for p in state.values()))
    write_modesbin(os.path.join(root, "modes.bin"), {k: v / scale for k, v in state.items()})
    done = subprocess.run([solver, "config.toml", "modes.bin"], cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, f"solver failed:\n{done.stdout}\n{done.stderr}"
    folders = [d for d in os.listdir(root) if d.startswith("Simulation_")]
    assert len(folders) == 1
    return os.path.join(root, folders[0])


def synthetic_run(root):
    """A run folder with the same files a real run writes, filled with smooth data."""
    folder = os.path.join(root, "Simulation_2026_01_01_00_00_00")
    os.makedirs(os.path.join(folder, "psi"))
    os.makedirs(os.path.join(folder, "psi_dt"))
    with open(os.path.join(folder, "config.toml"), "w") as f:
        f.write(CONFIG)
    r = np.linspace(0.0, 64.0, NR)
    np.savetxt(os.path.join(folder, "r.txt"), r)
    with open(os.path.join(folder, "modes.txt"), "w") as f:
        for l, m in MODES:
            f.write(f"{l},{m}\n")
    t = 2.0 * np.arange(1, ROWS + 1)
    scalars = {"t": t, "R_orb": 20.0 - 0.01 * t, "theta_orb": 0.05 * t,
               "Omega_orb": np.sqrt(1.5 / (20.0 - 0.01 * t) ** 3), "M_1": 1.0 + 1e-6 * t,
               "q": 0.5 + 1e-7 * t, "energy": -0.01 - 1e-5 * t, "p_bin": -1e-6 * np.ones_like(t),
               "p_bin_inst": -1e-6 * np.cos(0.1 * t), "dV_bin_dt": 1e-7 * np.ones_like(t)}
    for name, values in scalars.items():
        np.savetxt(os.path.join(folder, name + ".txt"), values)
    rng = np.random.default_rng(3)
    for name in ("flux", "energy_flux", "norms", "energy_modes"):
        block = np.abs(rng.normal(size=(ROWS, len(MODES)))) * 1e-3
        if name == "norms":
            block[:, 0] += 0.09
        np.savetxt(os.path.join(folder, name + ".txt"), block)
    for step in (0, 10, 20):
        psi = np.zeros((len(MODES) * NR, 2))
        for j, (l, m) in enumerate(MODES):
            profile = np.exp(-((r - 10.0 - 2 * j) / 6.0) ** 2) * r / 10.0
            psi[j * NR:(j + 1) * NR, 0] = profile
            psi[j * NR:(j + 1) * NR, 1] = 0.1 * j * profile
        np.savetxt(os.path.join(folder, "psi", f"snapshot_{step:06d}.txt"), psi)
    return folder


def test_discovery(dash, root, folder):
    found = dash.list_runs(root, depth=4, ttl=0.0)
    assert folder in [os.path.realpath(f) for f in found], f"run not found below {root}"
    assert dash.is_run(folder) and not dash.is_run(root), "is_run does not identify a run folder"
    # A run folder is never descended into.
    assert not any(os.path.join(folder, "psi") == os.path.realpath(f) for f in found)
    brief = dash.run_brief(folder)
    assert brief["q"] == 0.5 and brief["mu"] == 0.2 and brief["M_c"] == 0.02, f"wrong brief {brief}"
    assert brief["evolving"] and brief["absorption"] and not brief["counter"]
    assert dash.run_brief(os.path.join(root, "nope")) == {}, "missing folder should give an empty brief"


def test_data_payload(dash, folder):
    run = dash.get_run(folder)
    assert run is dash.get_run(folder), "runs are not cached by folder"
    payload = dash.data_payload(run, window_orbits=0, max_points=4000)
    assert payload["ready"], "payload is not ready on a finished run"
    info, series, latest = payload["info"], payload["series"], payload["latest"]
    assert info["rows"] == ROWS, f"{info['rows']} rows, expected {ROWS}"
    assert info["modes"] == [list(m) for m in MODES], "mode list does not match modes.txt"
    assert info["snapshots"] == [0, 10, 20], f"snapshots {info['snapshots']}"
    assert info["initial_mode"] == [0, 0], "initial mode comes from [atom] l and m"
    assert info["setup"], "run setup table is empty"
    for key in ("t", "R", "M_1", "M_c", "q", "norms", "fractions", "energy_flux", "mass_flux"):
        assert key in series, f"series is missing {key}"
    assert len(series["t"]) == ROWS, "series was truncated"
    assert len(series["norms"]) == len(MODES), "one norm series per mode"
    assert all(abs(sum(f[i] for f in series["fractions"]) - 1) < 1e-6 for i in range(ROWS)), \
        "mode fractions do not sum to one"
    # The series is rounded for the page; `latest` carries the exact values.
    assert abs(latest["R"] - series["R"][-1]) < 1e-6 * latest["R"], "latest R does not match the series"
    assert abs(latest["t"] - series["t"][-1]) < 1e-6 * max(1.0, latest["t"]), "latest t does not match"
    assert latest["M_c"] > 0 and math.isfinite(latest["P"]), "latest block is not finite"
    assert len(payload["levels"]) == 6 and payload["levels"][0]["E"] < 0, "hydrogenic levels are wrong"

    # The averaging window keeps only the last orbits, and max_points subsamples.
    narrow = dash.data_payload(run, window_orbits=0.05, max_points=4000)
    assert len(narrow["series"]["t"]) < ROWS, "window_orbits did not restrict the series"
    coarse = dash.data_payload(run, window_orbits=0, max_points=5)
    assert len(coarse["series"]["t"]) <= 6, "max_points did not subsample"


def test_pruning_payload(dash, folder):
    # Without active_modes.txt the run was never pruned.
    if not os.path.exists(os.path.join(folder, "active_modes.txt")):
        assert dash.data_payload(dash.get_run(folder), 0, 4000)["pruning"] is None, \
            "an unpruned run reports pruning"
    with tempfile.TemporaryDirectory() as other:
        pruned = os.path.realpath(synthetic_run(other))
        flags = lambda on: " ".join("1" if j in on else "0" for j in range(len(MODES)))
        with open(os.path.join(pruned, "active_modes.txt"), "w") as f:
            f.write(f"0 0.0 2 {flags({0, 2})}\n")
            f.write(f"8 14.0 3 {flags({0, 2, 4})}\n")
            f.write("15 28.0 1 1 0")  # partial line of a run still writing
        p = dash.data_payload(dash.get_run(pruned), 0, 4000)["pruning"]
        assert p is not None, "active_modes.txt was not read"
        # Row k is written by step k + 1: the second set applies from row 7 on.
        assert p["count"] == [2] * 7 + [3] * (ROWS - 7), f"counts {p['count']}"
        assert p["active"][4] == [None] * 7 + [True] * (ROWS - 7), "mode (2, 0) switches on at step 8"
        assert p["active"][1] == [None] * ROWS, "mode (1, -1) is never evolved"
        assert (p["now"], p["total"], p["changes"], p["ever"]) == (3, len(MODES), 2, 3), f"summary {p}"


def test_spin_payload(dash, folder):
    # The spin panel is fed only by runs that evolve the spins: spins.txt
    # (dipolar absorption) on an evolving orbit.
    if not os.path.exists(os.path.join(folder, "spins.txt")):
        assert dash.data_payload(dash.get_run(folder), 0, 4000)["series"]["spins"] is None, \
            "a run without spins.txt reports spins"
    t = 2.0 * np.arange(1, ROWS + 1)
    table = np.column_stack([0.4 - 1e-7 * t, -0.3 + 2e-7 * t, -3e-8 * t, 1e-8 * t])
    with tempfile.TemporaryDirectory() as other:
        spinning = os.path.realpath(synthetic_run(other))
        with open(os.path.join(spinning, "config.toml"), "w") as f:
            f.write(CONFIG.replace("R = 20.0\n", "R = 20.0\nspin_1 = 0.4\nspin_2 = -0.3\n"))
        np.savetxt(os.path.join(spinning, "spins.txt"), table, fmt="%.12e")
        sp = dash.data_payload(dash.get_run(spinning), 0, 4000)["series"]["spins"]
        assert sp is not None, "spins.txt of an evolving run was not read"
        assert len(sp["chi_1"]) == ROWS and abs(sp["chi_1"][-1] - table[-1, 0]) < 1e-12, "chi_1 series"
        assert abs(sp["chi_2"][-1] - table[-1, 1]) < 1e-12 and abs(sp["J_1"][-1] - table[-1, 2]) < 1e-14, "chi_2 or J_1 series"
        assert sp["chi_0"] == [0.4, -0.3], f"initial spins {sp['chi_0']}"
        assert abs(sp["dchi_1"][-1] + 1e-7 * t[-1]) < 1e-12 and abs(sp["dchi_2"][-1] - 2e-7 * t[-1]) < 1e-12, \
            "spin changes are not relative to the configured spins"
    with tempfile.TemporaryDirectory() as other:
        fixed = os.path.realpath(synthetic_run(other))
        path = os.path.join(fixed, "config.toml")
        with open(path) as f:
            text = f.read()
        assert "evolve_orbit = true" in text
        with open(path, "w") as f:
            f.write(text.replace("evolve_orbit = true", "evolve_orbit = false"))
        np.savetxt(os.path.join(fixed, "spins.txt"), table, fmt="%.12e")
        assert dash.data_payload(dash.get_run(fixed), 0, 4000)["series"]["spins"] is None, \
            "a fixed orbit does not evolve the spins"


def test_alone_payload(dash):
    # secondary_potential = false: one hole at rest at x = qR/(1+q), q.txt = 0,
    # Omega = 0. The payload must stay valid JSON and drop the binary scalings.
    with tempfile.TemporaryDirectory() as other:
        folder = os.path.realpath(synthetic_run(other))
        with open(os.path.join(folder, "config.toml"), "w") as f:
            f.write(CONFIG.replace("[simulation]\n", "[simulation]\nsecondary_potential = false\n"))
        n = ROWS
        x0 = 0.5 / 1.5 * 20.0
        for name, values in {"R_orb": np.full(n, x0), "theta_orb": np.zeros(n), "Omega_orb": np.zeros(n),
                             "q": np.zeros(n), "p_bin": np.zeros(n), "p_bin_inst": np.zeros(n),
                             "dV_bin_dt": np.zeros(n), "M_1": 1.0 - 1e-11 * np.arange(n)}.items():
            np.savetxt(os.path.join(folder, name + ".txt"), values)
        run = dash.get_run(folder)
        out = dash.data_payload(run, 0, 4000)
        json.dumps(out, allow_nan=False)
        assert out["info"]["alone"], "a run without a companion is not flagged"
        s = out["series"]
        assert s["p_gw"] is None and s["M_2"] is None and s["q"] is None, "binary series sent for a lone hole"
        assert out["theory"] is None, "binary theory rates sent for a lone hole"
        assert abs(s["dM_1"][-1] + 1e-11 * (n - 1)) < 1e-15, f"mass change {s['dM_1'][-1]}"
        assert dash.scalar_payload(out)["info"]["alone"], "scalar payload drops the alone flag"
        assert dash.run_brief(folder)["alone"], "run brief drops the alone flag"
        pos = dash.black_hole_position("primary", x0, 0.0, 0.0, alone=True)
        assert abs(pos[0] - x0) < 1e-12 and abs(pos[1]) < 1e-12, f"lone hole at {pos}"
        try:
            dash.projection_payload(run, "secondary")
        except ValueError:
            pass
        else:
            raise AssertionError("a projection about the missing companion was accepted")
        sn = dash.snapshot_payload(run, 10, 30.0, 24, "all", [[1, 1]])
        assert sn["alone"], "snapshot does not flag the lone hole"
        json.dumps(sn, allow_nan=False)
        badges = " ".join(str(b) for b in dash.run_setup(run.config, "", np.linspace(0.0, 40.0, 9), 3)["badges"])
        assert "No companion" in badges, f"badges {badges}"
    # q = 0 is the same as no companion, with the hole at the origin. Older
    # runs kept a massless companion in R_orb, theta_orb and Omega_orb.
    with tempfile.TemporaryDirectory() as other:
        folder = os.path.realpath(synthetic_run(other))
        with open(os.path.join(folder, "config.toml"), "w") as f:
            f.write(CONFIG.replace("q = 0.5", "q = 0.0"))
        np.savetxt(os.path.join(folder, "q.txt"), np.zeros(ROWS))
        run = dash.get_run(folder)
        out = dash.data_payload(run, 0, 4000)
        json.dumps(out, allow_nan=False)
        assert out["info"]["alone"] and out["latest"]["R"] == 0.0, "q = 0 is not read as a lone hole at the origin"
        assert dash.run_brief(folder)["alone"], "run brief of a q = 0 run"
        assert dash.snapshot_orbit(run, 10)[1:] == (0.0, 0.0, 0.0), "q = 0 snapshot orbit"
        assert dash.snapshot_payload(run, 10, 30.0, 24, "all", [[1, 1]])["R"] == 0.0, "q = 0 snapshot position"
    # A lone hole placed at the origin with R = 0 has no initial period.
    with tempfile.TemporaryDirectory() as other:
        folder = os.path.realpath(synthetic_run(other))
        with open(os.path.join(folder, "config.toml"), "w") as f:
            f.write(CONFIG.replace("q = 0.5", "q = 0").replace("R = 20.0", "R = 0")
                    .replace("[simulation]\n", "[simulation]\nsecondary_potential = false\n"))
        for name in ("q", "R_orb", "theta_orb", "Omega_orb"):
            np.savetxt(os.path.join(folder, name + ".txt"), np.zeros(ROWS))
        run = dash.get_run(folder)
        json.dumps(dash.data_payload(run, 0, 4000), allow_nan=False)
        labels = [row["label"] for g in run.setup["groups"] for row in g["rows"]]
        assert "Initial period P₀" not in labels, "R = 0 shows an initial period"
        dash.snapshot_payload(run, 10, 30.0, 24, "all", [[1, 1]])


def test_run_loader(dash, folder):
    # dashboard/run_loader.py: every stored file, the dashboard's derived values, one snapshot.
    spec = importlib.util.spec_from_file_location("run_loader", os.path.join(REPO, "dashboard", "run_loader.py"))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    run = loader.load_run(folder)
    payload = dash.data_payload(dash.get_run(folder), 0, 10 ** 6)
    assert run.n_rows == ROWS and run.modes == [tuple(m) for m in MODES], "rows or modes"
    for key in ("t", "R_orb", "M_1", "q", "p_bin", "norms", "flux", "energy_flux", "energy_modes"):
        assert key in run.keys(), f"{key} not loaded"
    assert run.norms.shape == (ROWS, len(MODES)) and run.t.shape == (ROWS,), "array shapes"
    assert np.allclose(run.cloud_mass, np.array(payload["series"]["M_c"], float), rtol=1e-6), "cloud mass"
    assert run.snapshot.step == 20 and abs(run.snapshot.t - run.t[19]) < 1e-12, "last snapshot and its time"
    psi = dash.load_snapshot(dash.get_run(folder), 20)
    assert np.array_equal(run.snapshot.psi, psi), "snapshot differs from the dashboard's"
    assert np.array_equal(run.snapshot.U(*MODES[1]), psi[1]), "U(l, m)"
    assert loader.load_run(folder, step=10).snapshot.step == 10, "chosen step"
    assert loader.load_run(folder, time=run.t[9] + 0.1).snapshot.step == 10, "nearest time"
    assert loader.load_run(folder, snapshot=False).snapshot is None, "snapshot=False"
    # Files of different lengths: every quantity is cut to the rows they all have.
    with tempfile.TemporaryDirectory() as other:
        uneven = os.path.join(other, os.path.basename(folder))
        shutil.copytree(folder, uneven)
        np.savetxt(os.path.join(uneven, "spins.txt"), np.zeros((ROWS - 5, 4)))
        with open(os.path.join(uneven, "flux.txt"), "a") as f:
            f.write(" ".join(["1e-3"] * len(MODES)) + "\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cut = loader.load_run(uneven, snapshot=False)
        assert cut.n_rows == ROWS - 5 and any("different lengths" in str(w.message) for w in caught), "no cut or warning"
        lengths = {k: len(v) for k, v in cut.data.items()}
        lengths.update({"budget " + k: len(v) for k, v in cut.mass_budget.items() if isinstance(v, np.ndarray)})
        assert set(lengths.values()) == {ROWS - 5}, f"lengths differ: {lengths}"
    part = loader.load_run(folder, quantities=["R_orb", "cloud_mass"], snapshot=False)
    assert part.keys() == ["R_orb", "cloud_mass", "t"], f"restricted load gave {part.keys()}"
    assert np.array_equal(part.cloud_mass, run.cloud_mass), "restricted load changed a value"
    try:
        loader.load_run(folder, quantities=["no_such_file"])
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown quantity was accepted")
    try:
        loader.load_run(folder, step=7)
    except ValueError:
        pass
    else:
        raise AssertionError("a missing snapshot step was accepted")
    h, v, rho = run.snapshot.slice("zx", 30.0, 24)
    out = dash.snapshot_payload(dash.get_run(folder), 20, 30.0, 24, "all", [], plane="zx")
    density = np.frombuffer(base64.b64decode(out["density_f32"]), dtype="<f4").reshape(24, 24)
    assert np.allclose(rho, density, rtol=1e-4, atol=1e-6 * rho.max()), "slice differs from the dashboard's"


def test_Lz_payload(dash, folder):
    # The signed L_z budget: cloud + ionized + absorbed + pruned + damped + binary is
    # constant. A real run's Lz_budget.txt closes to the solver's tolerance.
    def check(run_folder, exact):
        z = dash.data_payload(dash.get_run(run_folder), 0, 4000)["series"]["Lz"]
        assert z is not None, "no L_z budget"
        assert z["exact"] is exact, f"exact = {z['exact']}, expected {exact}"
        if exact:
            assert z["closure"] < 1e-9 * max(1.0, abs(z["L0"])), f"L_z budget closes only to {z['closure']}"
        modes_sum = [sum(y[i] for y in z["modes"] if y) for i in range(len(z["cloud"]))]
        assert max(abs(a - b) for a, b in zip(modes_sum, z["cloud"])) < 1e-8, "per-mode L_z does not add up to the cloud"
        by_m = [sum(y[i] for y in z["by_m"].values()) for i in range(len(z["cloud"]))]
        assert max(abs(a - b) for a, b in zip(by_m, z["cloud"])) < 1e-8, "per-m L_z does not add up to the cloud"
        assert all(y is None for (l, m), y in zip(MODES, z["modes"]) if m == 0), "m = 0 modes carry L_z"
        g = z["ledger"]
        assert abs(g["cloud0"] - z["L0"] * z["N_ref"]) < 1e-9 * max(1.0, abs(g["cloud0"])), "ledger's initial cloud L_z"
        # q = 0.5, R = 20, M1 = 1, co-rotating: L_orb = M1 M2 sqrt(R / M) along +z.
        assert abs(g["orbit0"] - 0.5 * math.sqrt(20.0 / 1.5)) < 1e-12, f"initial orbital L_z {g['orbit0']}"
        assert len(g["cloud"]) == len(z["cloud"]) and len(g["orbit"]) == len(z["cloud"]), "ledger series lengths"
        assert g["gw"] is not None and g["gw"][-1] <= 0, "an evolving co-rotating orbit loses L_z to GW"
        return z

    if os.path.exists(os.path.join(folder, "Lz_budget.txt")):
        check(folder, True)
        spec = importlib.util.spec_from_file_location("run_loader", os.path.join(REPO, "dashboard", "run_loader.py"))
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        Lz = loader.load_run(folder, snapshot=False).Lz_budget
        assert Lz["exact"] and np.ptp(Lz["total"]) < 1e-9, f"run_loader L_z total moves by {np.ptp(Lz['total'])}"
    with tempfile.TemporaryDirectory() as other:
        run = os.path.realpath(synthetic_run(other))
        norms = np.loadtxt(os.path.join(run, "norms.txt"))
        m = np.array([mm for _, mm in MODES], dtype=float)
        cloud = norms @ m
        t = 2.0 * np.arange(1, ROWS + 1)
        torque, absorbed, pruned, damped = 1e-5 * t, -2e-6 * t, np.zeros(ROWS), 1e-7 * t
        ionized = 0.05 - cloud - absorbed - pruned - damped + torque
        np.savetxt(os.path.join(run, "Lz_budget.txt"),
                   np.column_stack([ionized, torque, absorbed, pruned, damped]), fmt="%.12e")
        z = check(run, True)
        assert z["binary"] is not None and z["pruned"] is None and z["damped"] is not None, "optional terms"
        ratio = z["binary"][-1] / z["absorbed"][-1]
        assert abs(ratio + torque[-1] / absorbed[-1]) < 1e-6, f"binary is not -torque (ratio {ratio})"
    with tempfile.TemporaryDirectory() as other:
        # Without Lz_budget.txt (older runs) the budget comes from the diagnostics, approximately.
        run = os.path.realpath(synthetic_run(other))
        check(run, False)


def test_snapshot_pruning(dash):
    # Disk usage and the snapshot cleanup: the latest snapshot always stays.
    plan = dash.snapshot_plan
    steps = [0, 10, 20, 30, 40, 50, 60]
    assert plan(steps, "thin", every=2) == [10, 30, 50], plan(steps, "thin", every=2)
    assert plan(steps, "thin", every=4) == [10, 20, 30, 50], plan(steps, "thin", every=4)
    assert plan(steps, "keep", keep=[20]) == [0, 10, 30, 40, 50], "keep one: the chosen and the latest stay"
    assert plan(steps, "keep", keep=[60]) == [0, 10, 20, 30, 40, 50]
    assert plan([], "thin", every=2) == []
    for bad in (dict(mode="thin", every=1), dict(mode="all")):
        try:
            plan(steps, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    with tempfile.TemporaryDirectory() as other:
        folder = os.path.realpath(synthetic_run(other))   # snapshots 0, 10, 20
        open(os.path.join(folder, "psi_dt", "snapshot_000000.txt"), "w").write("0 0\n")
        usage = dash.disk_usage(folder)
        assert usage["snapshot_count"] == 3 and usage["steps"] == [0, 10, 20], usage
        assert usage["total"] == sum(usage[k] for k in ("snapshots", "psi_dt", "diagnostics", "checkpoint", "other"))
        run = dash.get_run(folder)
        files, freed = dash.delete_snapshots(run, [0, 20])   # 20 is the latest: refused
        assert run.snapshot_steps() == [10, 20], run.snapshot_steps()
        assert files == 2 and freed > 0, "snapshot 0 and its psi_dt were not both removed"
        assert not os.path.exists(os.path.join(folder, "psi_dt", "snapshot_000000.txt"))
        assert dash.disk_usage(folder)["total"] == usage["total"] - freed


def test_scalar_payload(dash, folder):
    # The Compare tab's light payload: scalar series only, same samples.
    full = dash.data_payload(dash.get_run(folder), 0, 4000)
    light = dash.scalar_payload(full)
    assert light["ready"] and light["info"]["folder"] == full["info"]["folder"], "scalar payload info"
    for key in ("t", "R", "Omega", "M_c", "total_mass_flux", "p_bin", "E_s"):
        assert light["series"][key] == full["series"][key], f"scalar series {key}"
    assert not any(isinstance(v, list) and v and isinstance(v[0], list) for v in light["series"].values()), \
        "per-mode series left in the scalar payload"


def test_default_dt(dash):
    # Without dt the solver starts at P0 / steps_per_period; the setup table shows that value.
    cfg = {"binary": {"q": 0.5, "R": 20.0}, "simulation": {"steps_per_period": 32, "num_periods": 2}}
    expected = 2 * math.pi * math.sqrt(20.0 ** 3 / 1.5) / 32
    assert abs(dash.default_dt(cfg) - expected) < 1e-12 * expected, "default dt"
    dt_row = lambda c: next(row for g in dash.run_setup(c, "", np.linspace(0.0, 40.0, 9), 3)["groups"]
                            for row in g["rows"] if row.get("key") == "simulation.dt")
    row = dt_row(cfg)
    assert row["default"] and row["value"] == dash._fmt(expected), f"derived dt row {row}"
    cfg["simulation"]["dt"] = 2.0
    assert dt_row(cfg)["value"] == dash._fmt(2.0) and not dt_row(cfg)["default"], "explicit dt row"


def test_snapshot_payload(dash, folder):
    run = dash.get_run(folder)
    psi = dash.load_snapshot(run, 0)
    assert psi.shape == (len(MODES), NR), f"snapshot shape {psi.shape}"
    assert psi is dash.load_snapshot(run, 0), "snapshots are not cached"
    for plane in ("xy", "zx", "zy"):
        out = dash.snapshot_payload(run, 10, r_plot=30.0, n_grid=24, mode_set="all",
                                    radiative=[[1, 1]], plane=plane)
        assert out["step"] == 10 and out["plane"] == plane
        assert out["n"] == len(out["x"]), "grid size does not match the axis"
        density = np.frombuffer(base64.b64decode(out["density_f32"]), dtype="<f4")
        assert density.size == out["n"] ** 2, "density is not a square grid"
        assert np.all(np.isfinite(density)) and density.max() > 0, "density is empty or not finite"
        radial = np.frombuffer(base64.b64decode(out["radial_f32"]), dtype="<f4")
        assert radial.size == len(MODES) * len(out["r"]), "one radial profile per mode"
        assert out["reference"] is not None, "reference profile is missing"
    # Selecting a subset of modes lowers the density, and the cache is reused.
    full = dash.snapshot_payload(run, 10, 30.0, 24, "all", [[1, 1]])
    assert full is dash.snapshot_payload(run, 10, 30.0, 24, "all", [[1, 1]]), "payload cache missed"
    bound = dash.snapshot_payload(run, 10, 30.0, 24, "bound", [[1, 1]])
    assert bound["density_f32"] != full["density_f32"], "mode selection had no effect"
    try:
        dash.snapshot_payload(run, 10, 30.0, 24, "all", [], plane="xz")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown plane was accepted")


def test_projection(dash, folder):
    run = dash.get_run(folder)
    psi = dash.load_snapshot(run, 10)
    dr = float(run.r[1] - run.r[0])
    # About the COM the re-expansion gives back the saved modes.
    norms, ball = dash.project_about(dash.snapshot_field(run, psi), run.modes, np.zeros(3), 40.0, dr)
    direct = (np.abs(psi[:, (run.r > 0) & (run.r <= 40.0)]) ** 2).sum(axis=1) * dr
    assert np.allclose(norms, direct, rtol=1e-6), f"COM projection {norms} against {direct}"
    assert abs(ball / norms.sum() - 1) < 1e-6, "the ball holds more than the modes about the COM"
    # A Gaussian centred off the origin, carried by modes up to l = 8 about the
    # COM, is almost purely l = 0 about its own centre.
    modes = [(l, m) for l in range(9) for m in range(-l, l + 1, 2)]
    a, width = np.array([3.0, 2.0, 0.0]), 6.0
    gauss = lambda X, Y, Z: np.exp(-((X - a[0]) ** 2 + (Y - a[1]) ** 2 + Z ** 2) / (2 * width ** 2)) + 0j
    n_hat, C, F, W = dash.angular_grid(8)
    Ystar = np.stack([np.conj(dash.Y_theta(l, m, C) * np.exp(1j * m * F)) * W for l, m in modes], axis=1)
    P = run.r[:, None, None] * n_hat[None]
    U = run.r[:, None] * (gauss(*(P[..., k].ravel() for k in range(3))).reshape(len(run.r), len(W)) @ Ystar)
    gauss_field = lambda X, Y, Z: dash.field_values(U.T, modes, set(modes), dash.point_tables(run.r, X, Y, Z))
    shifted, _ = dash.project_about(gauss_field, modes, a, 30.0, dr)
    assert shifted[0] / shifted.sum() > 0.9999, f"l = 0 holds only {shifted[0] / shifted.sum()}"
    # The cached geometry (rotate, translate along z, rotate back) agrees with
    # projecting the field directly about the hole. With l_max = 2 the field
    # about a hole has much power above l_max, so both are resolved finely
    # enough that aliasing does not separate them.
    t, R, theta, q = dash.snapshot_orbit(run, 10)
    coarse = dash.angular_grid
    dash.angular_grid = lambda l_max: coarse(40)
    try:
        for center in ("primary", "secondary"):
            pos = dash.black_hole_position(center, R, theta, q)
            d = float(np.hypot(pos[0], pos[1]))
            direct, ball_direct = dash.project_about(dash.snapshot_field(run, psi), run.modes, pos, 40.0 - d, dr)
            fast, ball_fast = dash.ProjectionGeometry(run.r, run.modes, d, 40.0 - d, dr, beta_nodes=160).project(
                psi, math.atan2(pos[1], pos[0]))
            assert np.allclose(fast, direct, rtol=1e-3), f"{center}: {fast} against {direct}"
            assert abs(ball_fast / ball_direct - 1) < 1e-4, f"{center}: ball {ball_fast} against {ball_direct}"
    finally:
        dash.angular_grid = coarse
    # Every snapshot about each hole; the second request comes from the cache.
    for center in ("primary", "secondary"):
        out = dash.projection_payload(run, center)
        assert out["steps"] == [0, 10, 20] and out["remaining"] == 0, f"projected steps {out['steps']}"
        assert len(out["masses"]) == len(MODES) and len(out["masses"][0]) == 3, "projection shape"
        assert all(h >= 0 for h in out["higher"]), "negative mass above l_max"
        assert dash.projection_payload(run, center, budget=0.0)["remaining"] == 0, "projections are not cached"
    try:
        dash.projection_payload(run, "tertiary")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown centre was accepted")


def test_zip_export(dash, root, folder):
    out = os.path.join(root, "runs.zip")
    with open(out, "wb") as f:
        dash.write_runs_zip([(folder, 10, "run A"), (folder, 20, "run B")], f)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        index = zf.read("index.txt").decode()
    assert "index.txt" in names, "the zip has no index"
    assert len(index.strip().split("\n")) == 3, f"index does not list both runs:\n{index}"
    tags = {name.split("/")[0] for name in names if "/" in name}
    assert len(tags) == 2, f"runs sharing parameters were not disambiguated: {tags}"
    for wanted in ("config.toml", "t.txt", "norms.txt"):
        assert any(name.endswith("/" + wanted) for name in names), f"{wanted} is missing from the zip"
    assert any("snapshot_000010" in name for name in names), "the requested snapshot is missing"


def test_http_endpoints(dash, root, folder):
    from http.server import ThreadingHTTPServer
    dash.Handler.root = root
    dash.Handler.root_real = os.path.realpath(root)
    dash.Handler.depth = 4
    dash.Handler.log_message = lambda *args, **kwargs: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        def get(path):
            with urllib.request.urlopen(base + path, timeout=30) as response:
                return response.status, response.read()

        status, body = get("/")
        assert status == 200 and b"<html" in body.lower(), "the page did not load"
        status, body = get("/api/runs")
        listed = json.loads(body)
        assert any(os.path.realpath(entry["folder"]) == folder for entry in listed), \
            f"/api/runs does not list the run: {listed}"
        assert listed[0]["params"]["q"] == 0.5 and listed[0]["name"], "/api/runs lacks the run parameters"
        quote = urllib.parse.quote
        status, body = get("/api/data?run=" + quote(folder))
        assert status == 200 and json.loads(body)["ready"], "/api/data is not ready"
        status, body = get(f"/api/snapshot?run={quote(folder)}&step=10&rplot=30&grid=64")
        assert status == 200 and json.loads(body)["step"] == 10, "/api/snapshot returned the wrong step"
        status, body = get(f"/api/projection?run={quote(folder)}&center=secondary")
        assert status == 200 and json.loads(body)["center"] == "secondary", "/api/projection failed"
        status, body = get("/api/disk?run=" + quote(folder))
        usage = json.loads(body)
        assert status == 200 and usage["total"] > 0 and usage["snapshot_count"] >= 2, f"/api/disk: {usage}"
        before = sorted(os.listdir(os.path.join(folder, "psi")))
        request = urllib.request.Request(base + "/api/snapshots/prune", method="POST",
                                         data=json.dumps({"run": folder, "mode": "keep", "keep": [0]}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            plan = json.loads(response.read())
        assert plan["keep"][-1] == usage["steps"][-1] and plan["delete"] and plan["bytes"] > 0, f"prune plan {plan}"
        assert sorted(os.listdir(os.path.join(folder, "psi"))) == before, "a request without dry_run=false deleted files"
        # A folder outside the served root must be refused, with the error in the body.
        try:
            status, body = get("/api/data?run=" + quote(os.path.dirname(root)))
        except urllib.error.HTTPError as error:
            assert error.code == 500 and b"Unknown run folder" in error.read(), \
                f"unexpected refusal with status {error.code}"
        else:
            raise AssertionError(f"a folder outside the root was served: {status}")
    finally:
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", help="GravitationalMolecule executable; without it a folder is written by hand")
    args = parser.parse_args()
    dash = load_dashboard()

    failures = 0
    with tempfile.TemporaryDirectory() as root:
        nested = os.path.join(root, "runs")
        os.makedirs(nested)
        folder = real_run(nested, os.path.abspath(args.solver)) if args.solver else synthetic_run(nested)
        folder = os.path.realpath(folder)
        tests = [lambda: test_discovery(dash, root, folder),
                 lambda: test_data_payload(dash, folder),
                 lambda: test_pruning_payload(dash, folder),
                 lambda: test_spin_payload(dash, folder),
                 lambda: test_alone_payload(dash),
                 lambda: test_Lz_payload(dash, folder),
                 lambda: test_snapshot_pruning(dash),
                 lambda: test_default_dt(dash),
                 lambda: test_scalar_payload(dash, folder),
                 lambda: test_run_loader(dash, folder),
                 lambda: test_snapshot_payload(dash, folder),
                 lambda: test_projection(dash, folder),
                 lambda: test_zip_export(dash, root, folder),
                 lambda: test_http_endpoints(dash, root, folder)]
        names = ["discovery", "data payload", "pruning payload", "spin payload", "no-companion payload", "L_z budget payload", "snapshot cleanup", "default dt", "scalar payload", "run loader", "snapshot payload", "projection", "zip export", "http endpoints"]
        for name, test in zip(names, tests):
            try:
                test()
                print(f"[pass] {name}")
            except Exception as error:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"[FAIL] {name}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
