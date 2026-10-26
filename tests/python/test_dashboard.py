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
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
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
                 lambda: test_snapshot_payload(dash, folder),
                 lambda: test_zip_export(dash, root, folder),
                 lambda: test_http_endpoints(dash, root, folder)]
        names = ["discovery", "data payload", "snapshot payload", "zip export", "http endpoints"]
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
