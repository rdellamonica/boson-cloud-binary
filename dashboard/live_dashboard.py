#!/usr/bin/env python3
"""Live dashboard for a running GravitationalMolecule simulation.

Serves a local web page that re-reads the run folder as it grows, with the
orbit, flux, energy and power diagnostics and a viewer for individual
snapshots (radial profiles and the equatorial slice).

    python dashboard/live_dashboard.py runs                # pick any run
    python dashboard/live_dashboard.py campaign            # runs in task subfolders
    python dashboard/live_dashboard.py runs/Simulation_... # open one run

then open http://127.0.0.1:8050. Run folders are found up to --depth levels
below the given folder. Only NumPy is required; the charts use d3 from
dashboard/assets. Rows are read incrementally, so polling a long run is cheap.
"""

import argparse
import base64
from collections import OrderedDict
import io
import json
import math
import os
import re
import threading
import time
import tempfile
import tomllib
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Tabulated perturbative rates (Tomaselli et al.), shared with the initial-state helpers.
FUNCTIONS_DIR = os.path.join(SCRIPT_DIR, "..", "initial_states", "gravatom", "functions")

SCALAR_FILES = ["t", "energy", "p_bin", "p_bin_inst", "dV_bin_dt", "R_orb",
                "theta_orb", "Omega_orb", "M_1", "q"]
MODE_FILES = ["flux", "energy_flux", "energy_modes", "norms"]
SNAPSHOT_RE = re.compile(r"snapshot_(\d+)\.txt$")


# ----------------------------------------------------------------------------
# Incremental readers
# ----------------------------------------------------------------------------

class GrowingTable:
    """Rows of a whitespace-separated text file that is being appended to.

    Only complete lines are parsed. If the file shrinks (e.g. a resumed run
    cut it back to its last snapshot) the table is re-read from scratch.
    """

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.chunks = []
        self.rows = 0
        self.width = None

    def update(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.offset:
            self.__init__(self.path)
        if size == self.offset:
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        end = data.rfind(b"\n")
        if end < 0:
            return
        data = data[: end + 1]
        self.offset += len(data)
        block = np.loadtxt(io.BytesIO(data), ndmin=2)
        if block.size == 0:
            return
        if self.width is None:
            self.width = block.shape[1]
        if block.shape[1] != self.width:
            raise ValueError(f"Inconsistent column count in {self.path}")
        self.chunks.append(block)
        self.rows += block.shape[0]
        if len(self.chunks) > 64:
            self.chunks = [np.concatenate(self.chunks)]

    def array(self, n=None):
        if not self.chunks:
            return np.zeros((0, self.width or 1))
        if len(self.chunks) > 1:
            self.chunks = [np.concatenate(self.chunks)]
        a = self.chunks[0]
        return a if n is None else a[:n]


class Run:
    def __init__(self, folder):
        self.folder = folder
        self.lock = threading.Lock()
        with open(os.path.join(folder, "config.toml"), "rb") as f:
            self.config_text = f.read().decode("utf-8", "replace")
        self.config = tomllib.loads(self.config_text)
        self.r = np.loadtxt(os.path.join(folder, "r.txt"))
        with open(os.path.join(folder, "modes.txt")) as f:
            self.modes = [tuple(int(v) for v in line.split(","))
                          for line in f if line.strip()]
        self.setup = run_setup(self.config, self.config_text, self.r, len(self.modes))
        self.tables = {name: GrowingTable(os.path.join(folder, name + ".txt"))
                       for name in SCALAR_FILES + MODE_FILES}
        self.snapshot_cache = {}
        self.slice_geometry = {}            # (r_plot, n_grid) -> interpolation tables
        self.payload_cache = OrderedDict()  # finished snapshot payloads, LRU
        self.rate_history = []  # (wall time, rows) for the step rate

    def update(self):
        for table in self.tables.values():
            table.update()
        rows = min(t.rows for t in self.tables.values())
        now = time.time()
        if not self.rate_history or self.rate_history[-1][1] != rows:
            self.rate_history.append((now, rows))
        self.rate_history = [p for p in self.rate_history if now - p[0] < 600]
        return rows

    def last_modified(self):
        stamps = []
        for name in ("t.txt", "norms.txt"):
            try:
                stamps.append(os.path.getmtime(os.path.join(self.folder, name)))
            except OSError:
                pass
        return max(stamps) if stamps else 0.0

    def snapshot_steps(self):
        try:
            names = os.listdir(os.path.join(self.folder, "psi"))
        except OSError:
            return []
        steps = [int(m.group(1)) for n in names if (m := SNAPSHOT_RE.match(n))]
        return sorted(steps)


RUNS = {}
RUNS_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Run setup: every parameter the solver reads (include/Config.hpp), with its
# default when config.toml leaves it out, grouped for the page.
# ----------------------------------------------------------------------------

def _fmt(v):
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


# (group id, title, icon, rows); a row is (section, key, label, default, unit, hint).
SETUP_SCHEMA = [
    ("binary", "Binary", "orbit", [
        ("binary", "q", "Mass ratio q = M₂/M₁", 0.001, "", "initial value; M₁ = 1"),
        ("binary", "R", "Initial separation R", 50.0, "M", ""),
        ("binary", "counter", "Rotation", False, "", ""),
        ("simulation", "evolve_orbit", "Orbit", False, "", "evolving: R and q follow the back-reaction"),
    ]),
    ("cloud", "Cloud", "cloud", [
        ("atom", "mu", "Coupling α = μM₁", 0.2, "", ""),
        ("atom", "n", "n", 2, "", ""),
        ("atom", "l", "l", 1, "", ""),
        ("atom", "m", "m", 1, "", ""),
        ("atom", "a", "Spin a of M₁", 0.0, "", ""),
        ("atom", "M_c", "Cloud mass M_c", 0.0, "M", "initial"),
    ]),
    ("absorption", "Absorption & boundaries", "target", [
        ("simulation", "absorption", "Absorption", False, "", "s-wave projector at the primary"),
        ("simulation", "epsilon_smear", "Smearing radius ε_smear", 2.0, "M₁", "radius of the absorber around M₁"),
        ("mask", "L_factor", "Mask start", 1.0, "× L", ""),
        ("mask", "W", "Mask width W", 0.1, "", ""),
        ("mask", "n", "Mask order", 3, "", ""),
        ("cap", "L_factor", "CAP start", 1.0, "× L", "complex absorbing potential"),
        ("cap", "W", "CAP width W", 0.1, "", ""),
        ("cap", "n", "CAP order", 3, "", ""),
    ]),
    ("grid", "Grid & modes", "ruler", [
        ("grid", "N", "Radial points N", 2048, "", ""),
        ("grid", "L", "Box size L", 5000.0, "M", ""),
        ("simulation", "epsilon", "Inner radius r₀", 0.001, "M", "first grid point"),
        ("simulation", "l_max", "l_max", 5, "", ""),
        ("simulation", "r_ex", "Extraction radius r_ex", 1000.0, "M", "fluxes and M_c are measured inside r_ex"),
    ]),
    ("time", "Time stepping", "clock", [
        ("simulation", "dt", "Time step dt", 10.0, "M", ""),
        ("simulation", "num_steps", "Steps", 0, "", "0 means num_periods is used"),
        ("simulation", "num_periods", "Periods", 0, "", "used when num_steps is 0"),
        ("simulation", "save_every", "Save every", 50, "steps", ""),
        ("simulation", "ramp_periods", "Companion ramp-up", 0.0, "orbits", "the companion is switched on over this time"),
    ]),
    ("numerics", "Numerics", "steps", [
        ("parallel", "threads", "Threads", 0, "", "0 uses every core"),
        ("solver", "preconditioner", "Preconditioner", "radial", "", ""),
    ]),
]
SECTION_TITLES = {"atom_secondary": "Secondary atom (initial data)"}


def run_setup(cfg, text, r, n_modes):
    get = lambda sec, key, default: (cfg.get(sec, {}) or {}).get(key, default)
    used = set()

    def row(sec, key, label, default, unit="", hint=""):
        used.add((sec, key))
        present = isinstance(cfg.get(sec), dict) and key in cfg[sec]
        return {"label": label, "value": _fmt(get(sec, key, default)), "unit": unit, "hint": hint,
                "key": f"{sec}.{key}", "default": not present}

    def derived(label, value, unit="", hint=""):
        return {"label": label, "value": _fmt(value), "unit": unit, "hint": hint, "derived": True}

    q, R0 = float(get("binary", "q", 0.001)), float(get("binary", "R", 50.0))
    mu = float(get("atom", "mu", 0.2))
    n, l, m = (int(get("atom", k, d)) for k, d in (("n", 2), ("l", 1), ("m", 1)))
    counter = bool(get("binary", "counter", False))
    evolving = bool(get("simulation", "evolve_orbit", False))
    absorption = bool(get("simulation", "absorption", False))
    smear = float(get("simulation", "epsilon_smear", 2.0))
    dt = float(get("simulation", "dt", 10.0))
    steps = int(get("simulation", "num_steps", 0))
    periods = int(get("simulation", "num_periods", 0))
    save_every = int(get("simulation", "save_every", 50))
    L = float(get("grid", "L", 5000.0))
    l_max = int(get("simulation", "l_max", 5))
    P0 = 2 * math.pi * math.sqrt(R0 ** 3 / (1 + q))
    if steps <= 0 and periods > 0:
        steps = int(math.floor(periods * P0 / dt))
    r_c = 1 / mu ** 2
    has_mask, has_cap = isinstance(cfg.get("mask"), dict), isinstance(cfg.get("cap"), dict)

    groups = []
    for gid, title, icon, spec in SETUP_SCHEMA:
        rows = []
        for sec, key, label, default, unit, hint in spec:
            if sec == "mask" and not has_mask or sec == "cap" and not has_cap:
                used.add((sec, key)); continue
            if gid == "cloud" and key in ("n", "l", "m"):
                used.add((sec, key)); continue
            rw = row(sec, key, label, default, unit, hint)
            if key == "counter":
                rw["value"] = "counter-rotating" if counter else "co-rotating"
            elif key == "evolve_orbit":
                rw["value"] = "evolving" if evolving else "fixed circular"
            elif sec == "simulation" and key == "num_periods" and int(get(sec, "num_steps", 0)) > 0:
                continue
            elif sec == "simulation" and key == "num_steps" and steps != int(get(sec, key, 0)):
                continue
            rows.append(rw)
        if gid == "binary":
            rows += [derived("Initial period P₀", P0, "M", "2π √(R³/(1+q))"),
                     derived("Primary offset from COM", q * R0 / (1 + q), "M", "qR/(1+q)")]
        elif gid == "cloud":
            st = {"label": "Initial state (n, l, m)", "value": f"({n}, {l}, {m})", "unit": "", "hint": "",
                  "key": "atom.n, atom.l, atom.m",
                  "default": not all(isinstance(cfg.get("atom"), dict) and k in cfg["atom"] for k in "nlm")}
            rows.insert(1, st)
            rows += [derived("Bohr radius r_c", r_c, "M", "1/μ²"),
                     derived("Separation R / r_c", R0 / r_c, "", "")]
        elif gid == "absorption":
            if not has_mask and not has_cap:
                rows.append(derived("Outer boundary", "no mask or CAP"))
            if has_mask:
                rows.append(derived("Mask begins at r", float(get("mask", "L_factor", 1.0)) * L, "M"))
            if has_cap:
                rows.append(derived("CAP begins at r", float(get("cap", "L_factor", 1.0)) * L, "M"))
        elif gid == "grid":
            dr = float(r[1] - r[0]) if len(r) > 1 else L / max(1, int(get("grid", "N", 2048)) - 1)
            rows += [derived("Grid spacing dr", dr, "M"),
                     derived("Modes evolved", n_modes, "", "from modes.txt")]
        elif gid == "time":
            rows += [derived("Steps per initial orbit", P0 / dt, "", "P₀ / dt"),
                     derived("Planned duration", steps * dt, "M", f"≈ {steps * dt / P0:.4g} initial orbits"),
                     derived("Snapshot interval", save_every * dt, "M", f"{steps // max(1, save_every)} snapshots")]
        groups.append({"id": gid, "title": title, "icon": icon, "rows": rows})

    # Anything else in config.toml (initializer settings, new options).
    extra = {}
    for sec, val in cfg.items():
        items = val.items() if isinstance(val, dict) else [(None, val)]
        for key, v in items:
            if (sec, key) in used:
                continue
            extra.setdefault(sec, []).append({"label": key or sec, "value": _fmt(v) if not isinstance(v, (list, dict)) else json.dumps(v),
                                              "unit": "", "hint": "", "key": f"{sec}.{key}" if key else sec, "default": False})
    for sec, rows in extra.items():
        known = {s for g in SETUP_SCHEMA for s, *_ in g[3]}
        title = SECTION_TITLES.get(sec) or (f"Other [{sec}] settings" if sec in known else f"[{sec}]")
        groups.append({"id": "x-" + sec, "title": title, "icon": "info", "rows": rows})

    boundary = "mask" if has_mask else "CAP" if has_cap else None
    badges = [
        {"text": "Counter-rotating" if counter else "Co-rotating", "tone": "accent", "icon": "repeat"},
        {"text": "Evolving orbit" if evolving else "Fixed orbit", "tone": "accent", "icon": "orbit"},
        {"text": f"Absorption · ε_smear = {smear:g}" if absorption else "No absorption",
         "tone": "on" if absorption else "off", "icon": "target"},
        {"text": f"(n, l, m) = ({n}, {l}, {m})", "tone": "plain", "icon": "atom"},
        {"text": f"μ = {mu:g}", "tone": "plain"},
        {"text": f"q = {q:g}", "tone": "plain"},
        {"text": f"R₀ = {R0:g} M", "tone": "plain"},
        {"text": f"l_max = {l_max}", "tone": "plain"},
    ]
    if boundary:
        badges.append({"text": boundary.capitalize() if boundary == "mask" else boundary, "tone": "plain", "icon": "aperture"})
    summary = [
        ["Rotation", "counter" if counter else "co"],
        ["Orbit", "evolving" if evolving else "fixed"],
        ["Absorption", f"ε = {smear:g}" if absorption else "off"],
        ["Initial (n, l, m)", f"({n}, {l}, {m})"],
        ["μ", f"{mu:g}"], ["q₀", f"{q:g}"], ["Modes", str(n_modes)],
    ]
    return {"groups": groups, "badges": badges, "summary": summary, "toml": text}


def get_run(folder):
    folder = os.path.realpath(folder)
    with RUNS_LOCK:
        if folder not in RUNS:
            RUNS[folder] = Run(folder)
        return RUNS[folder]


# ----------------------------------------------------------------------------
# Physics helpers (no sympy/scipy needed)
# ----------------------------------------------------------------------------

def hydrogenic_U(n, l, r, alpha, mu):
    """r * R_nl(r), unit-normalized, for a = 1/(mu alpha)."""
    a = 1.0 / (mu * alpha)
    rho = 2.0 * r / (n * a)
    k, s = n - l - 1, 2 * l + 1
    L_prev, L = np.ones_like(rho), 1.0 + s - rho
    if k == 0:
        L = L_prev
    for j in range(1, k):
        L_prev, L = L, ((2 * j + 1 + s - rho) * L - (j + s) * L_prev) / (j + 1)
    norm = math.sqrt((2.0 / (n * a)) ** 3 * math.factorial(k)
                     / (2 * n * math.factorial(n + l)))
    return r * norm * np.exp(-rho / 2) * rho ** l * L


def Y_equator(l, m):
    """Y_lm(theta = pi/2, phi = 0) with the Condon-Shortley phase."""
    am = abs(m)
    if (l + am) % 2:
        return 0.0
    dfact = lambda x: math.prod(range(x, 0, -2)) if x > 0 else 1
    P = (-1) ** ((l + am) // 2) * dfact(l + am - 1) / dfact(l - am)
    Y = math.sqrt((2 * l + 1) / (4 * math.pi)
                  * math.factorial(l - am) / math.factorial(l + am)) * P
    return Y if m >= 0 else (-1) ** am * Y


def Y_theta(l, m, cos_t):
    """Theta part of Y_lm (Condon-Shortley phase): Y_lm(theta, phi) = Y_theta * exp(i m phi)."""
    am = abs(m)
    sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, None))
    p_mm = (-1) ** am * math.prod(range(2 * am - 1, 0, -2)) * sin_t ** am   # P_m^m
    if l == am:
        P = p_mm
    else:
        p_prev, P = p_mm, cos_t * (2 * am + 1) * p_mm                      # P_{m+1}^m
        for ll in range(am + 2, l + 1):
            p_prev, P = P, ((2 * ll - 1) * cos_t * P - (ll + am - 1) * p_prev) / (ll - am)
    Y = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - am) / math.factorial(l + am)) * P
    return Y if m >= 0 else (-1) ** am * Y


def tomaselli_rates(n, l, m, mu, q, R0, counter):
    """Perturbative (mass rate, power) from the tabulated functions; None if absent."""
    out = []
    for name, power in (("R", 4), ("P", 6)):
        path = os.path.join(FUNCTIONS_DIR, f"{name}_{n}{l}{m}.csv")
        if not os.path.exists(path):
            return None
        x, F_co, F_counter = np.genfromtxt(path, delimiter=",").T
        F = F_counter if counter else F_co
        out.append(mu ** power * q ** 2 * float(np.interp(mu ** 2 * R0, x, F, left=0, right=0)))
    return out


# ----------------------------------------------------------------------------
# Payloads
# ----------------------------------------------------------------------------

def clean(a, digits=7):
    a = np.asarray(a, dtype=float)
    return [None if not math.isfinite(v) else float(f"{v:.{digits}g}") for v in a.ravel()]


def pick(n, max_points, start=0):
    if n - start <= max_points:
        return np.arange(start, n)
    return np.unique(np.linspace(start, n - 1, max_points).round().astype(int))


def data_payload(run, window_orbits, max_points):
    with run.lock:
        rows = run.update()
        cfg = run.config
        mu = cfg["atom"]["mu"]
        q0 = cfg["binary"]["q"]
        counter = cfg["binary"].get("counter", False)
        evolving = cfg["simulation"].get("evolve_orbit", False)
        n0, l0, m0 = cfg["atom"].get("n", 2), cfg["atom"].get("l", 1), cfg["atom"].get("m", 1)
        modes = run.modes
        initial = (l0, m0)
        info = {
            "folder": run.folder,
            "name": os.path.basename(run.folder),
            "rows": rows,
            "modes": [list(md) for md in modes],
            "initial_mode": list(initial),
            "evolving": evolving,
            "mu": mu, "q0": q0,
            "save_every": cfg["simulation"].get("save_every"),
            "dt": cfg["simulation"].get("dt", 10.0),
            "num_steps": cfg["simulation"].get("num_steps"),
            "num_periods": cfg["simulation"].get("num_periods"),
            "last_modified": run.last_modified(),
            "now": time.time(),
            "snapshots": run.snapshot_steps(),
            "setup": run.setup,
        }
        hist = run.rate_history
        if len(hist) >= 2 and hist[-1][0] > hist[0][0]:
            info["steps_per_min"] = 60 * (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
        if rows < 2:
            return {"info": info, "ready": False}

        T = {k: run.tables[k].array(rows) for k in SCALAR_FILES}
        S = {k: run.tables[k].array(rows).T for k in MODE_FILES}

    t = T["t"][:, 0]
    R_orb = T["R_orb"][:, 0]
    Omega = T["Omega_orb"][:, 0]
    M_1 = T["M_1"][:, 0]
    q_arr = T["q"][:, 0]
    theta = T["theta_orb"][:, 0]
    energy = T["energy"][:, 0] * mu
    current, T_tr, energy_modes, norms_raw = S["flux"], S["energy_flux"], S["energy_modes"] * mu, S["norms"]

    Mc_raw = norms_raw.sum(axis=0)
    # eps = E/N is the energy per particle; current/mu is the number flux.
    eps = energy / mu / Mc_raw
    mass_flux = (mu + eps) / mu * current / Mc_raw
    energy_flux = (T_tr + eps * current / mu) / Mc_raw
    # Power the binary must supply to sustain the ionization seen at r_ex:
    # outgoing energy flux plus the binding energy carried off, P = -sum(T + eps*J),
    # with eps = E/N the energy per particle and J = j/mu the number flux.
    # T is already a physical energy flux (get_energy_flux divides by mu).
    p_ion = -(T_tr.sum(axis=0) + (energy / mu) / Mc_raw * current.sum(axis=0) / mu)
    norms = norms_raw * mu
    Mc = norms.sum(axis=0)

    P_last = 2 * math.pi / Omega[-1] if Omega[-1] > 0 else float("nan")

    # Mode selection: drop the initial mode, and drop modes
    # whose mean energy flux is below 1/500 of the total's.
    idx_init = modes.index(initial) if initial in modes else -1
    ef_total_mean = abs(np.mean(energy_flux.sum(axis=0)))
    selected = [i for i in range(len(modes)) if i != idx_init
                and abs(np.mean(energy_flux[i])) >= ef_total_mean / 500]
    others = [i for i in range(len(modes)) if i != idx_init]
    total_mass_flux = mass_flux[selected].sum(axis=0) if selected else np.zeros_like(t)
    total_energy_flux = energy_flux[others].sum(axis=0)

    # Radiative / bound split over the last 25 orbits.
    t_mean = t[-1] - 25 * P_last if math.isfinite(P_last) else t[0]
    i_mean = int(np.searchsorted(t, t_mean))
    i_mean = min(i_mean, len(t) - 1)
    radiative = [list(modes[k]) for k in range(len(modes)) if np.mean(energy_modes[k][i_mean:]) > 0]

    start = 0
    if window_orbits > 0 and math.isfinite(P_last):
        start = int(np.searchsorted(t, t[-1] - window_orbits * P_last))
    idx = pick(len(t), max_points, start)

    alpha = mu  # M = 1
    levels = [{"n": n, "E": -mu * alpha ** 2 / (2 * n ** 2)} for n in range(1, 7)]
    eps_b = -mu * alpha ** 2 / (2 * n0 ** 2)
    theory = tomaselli_rates(n0, l0, m0, mu, q0, R_orb[0], counter)

    with np.errstate(divide="ignore", invalid="ignore"):
        e_per_mode = energy_modes / norms

    series = {
        "t": clean(t[idx], 10),
        "R": clean(R_orb[idx], 9),
        "t_over_P": clean(t[idx] / P_last, 9) if math.isfinite(P_last) else None,
        "mass_flux": [clean(-mass_flux[i][idx] / q0 ** 2) for i in range(len(modes))],
        "energy_flux": [clean(energy_flux[i][idx] / q0 ** 2) for i in range(len(modes))],
        "total_mass_flux": clean(-total_mass_flux[idx] / q0 ** 2),
        "total_energy_flux": clean(total_energy_flux[idx] / q0 ** 2),
        "p_bin_inst": clean(T["p_bin_inst"][idx, 0]),
        "p_bin": clean(T["p_bin"][idx, 0]),
        "p_out": clean(p_ion[idx]),
        "M_1": clean(M_1[idx], 10),
        "M_2": clean((q_arr * M_1)[idx], 10),
        "M_c": clean(Mc[idx], 10),
        "q": clean(q_arr[idx], 10),
        "norms": [clean(norms[i][idx]) for i in range(len(modes))],
        "fractions": [clean((norms[i] / Mc)[idx]) for i in range(len(modes))],
        "E_s": clean((energy / Mc)[idx]),
        "E_modes": [clean(e_per_mode[i][idx]) for i in range(len(modes))],
    }

    latest = {
        "t": float(t[-1]), "R": float(R_orb[-1]), "Omega": float(Omega[-1]),
        "P": P_last, "M_c": float(Mc[-1]), "M_c0": float(Mc[0]),
        "M_1": float(M_1[-1]), "q": float(q_arr[-1]), "theta": float(theta[-1]),
        "p_bin": float(T["p_bin"][-1, 0]),
        "total_mass_flux": float(-total_mass_flux[-1] / q0 ** 2),
        "total_energy_flux": float(total_energy_flux[-1] / q0 ** 2),
    }

    return {
        "info": info, "ready": True, "series": series, "latest": latest,
        "selected": selected, "others": others, "radiative": radiative,
        "levels": levels, "eps_b": eps_b,
        "theory": None if theory is None else {
            "mass": theory[0] / q0 ** 2, "power": -theory[1] / q0 ** 2},
    }


def load_snapshot(run, step):
    key = step
    if key in run.snapshot_cache:
        return run.snapshot_cache[key]
    path = os.path.join(run.folder, "psi", f"snapshot_{step:06d}.txt")
    dat = np.loadtxt(path, ndmin=2)
    Nr, n_modes = len(run.r), len(run.modes)
    if dat.shape[0] != Nr * n_modes:
        raise ValueError("Snapshot is still being written; try again shortly.")
    psi = (dat[:, 0] + 1j * dat[:, 1]).reshape(n_modes, Nr)
    if len(run.snapshot_cache) > 4:
        run.snapshot_cache.pop(next(iter(run.snapshot_cache)))
    run.snapshot_cache[key] = psi
    return psi


# Slice planes through the centre of mass: (horizontal, vertical) axes.
PLANES = {"xy": ("x", "y"), "zx": ("x", "z"), "zy": ("y", "z")}


def slice_geometry(run, r_plot, n_grid, plane="xy"):
    """Interpolation tables for one slice plane, shared by every snapshot.

    For each pixel: the radial index and weight into run.r, 1/r (0 outside the
    radial grid), cos(theta) and phi, so a snapshot costs one gather per mode.
    The angular factors are cached per mode on first use.
    """
    key = (float(r_plot), int(n_grid), len(run.r), plane)
    g = run.slice_geometry.get(key)
    if g is None:
        r = run.r
        x = np.linspace(-r_plot, r_plot, n_grid)
        hh, vv = (a.ravel() for a in np.meshgrid(x, x, indexing="xy"))   # horizontal, vertical
        zero = np.zeros_like(hh)
        X, Y, Z = {"xy": (hh, vv, zero), "zx": (hh, zero, vv), "zy": (zero, hh, vv)}[plane]
        rr, phi = np.sqrt(X ** 2 + Y ** 2 + Z ** 2), np.arctan2(Y, X)
        cos_t = np.where(rr > 0, Z / np.maximum(rr, 1e-300), 1.0)
        inside = (rr >= r[0]) & (rr <= r[-1])
        idx = np.clip(np.searchsorted(r, rr) - 1, 0, len(r) - 2)
        w = np.clip((rr - r[idx]) / (r[idx + 1] - r[idx]), 0.0, 1.0)
        weight = np.where(inside & (rr > 0), 1.0 / np.maximum(rr, 1e-300), 0.0)
        g = {"x": x, "idx": idx, "w": w, "weight": weight, "phi": phi, "cos_t": cos_t, "eimphi": {}, "ytheta": {}}
        if len(run.slice_geometry) > 4:
            run.slice_geometry.pop(next(iter(run.slice_geometry)))
        run.slice_geometry[key] = g
    return g


def f32_b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def snapshot_payload(run, step, r_plot, n_grid, mode_set, radiative, plane="xy"):
    key = (step, float(r_plot), int(n_grid), mode_set, json.dumps(sorted(map(list, radiative))), plane)
    hit = run.payload_cache.get(key)
    if hit is not None:
        run.payload_cache.move_to_end(key)
        return hit
    out = _snapshot_payload(run, step, r_plot, n_grid, mode_set, radiative, plane)
    run.payload_cache[key] = out
    while len(run.payload_cache) > 48:
        run.payload_cache.popitem(last=False)
    return out


def _snapshot_payload(run, step, r_plot, n_grid, mode_set, radiative, plane="xy"):
    cfg = run.config
    mu = cfg["atom"]["mu"]
    psi = load_snapshot(run, step)
    r, modes = run.r, run.modes

    # Snapshot n is written by step n, i.e. diagnostic row n-1.
    with run.lock:
        rows = run.update()
        if step >= 1 and step <= rows:
            row = step - 1
            t_s = float(run.tables["t"].array()[row, 0])
            R_s = float(run.tables["R_orb"].array()[row, 0])
            th_s = float(run.tables["theta_orb"].array()[row, 0])
            q_s = float(run.tables["q"].array()[row, 0])
            norms0 = run.tables["norms"].array()[0]
        else:
            t_s, R_s, th_s, q_s = 0.0, cfg["binary"]["R"], 0.0, cfg["binary"]["q"]
            norms0 = run.tables["norms"].array()[0] if run.tables["norms"].rows else None

    ir = pick(len(r), 2500)
    radial = np.abs(psi[:, ir]) ** 2

    n0, l0 = cfg["atom"].get("n", 2), cfg["atom"].get("l", 1)
    ref = None
    if norms0 is not None:
        U0 = hydrogenic_U(n0, l0, r[ir], mu, mu)
        ref = clean(np.abs(U0) ** 2 * float(np.sum(norms0)), 5)

    # Slice through the COM: psi = sum U_LM(r)/r Y_LM(theta, phi), summed per m
    # first so each exp(i m phi) is applied once. In the xy plane theta = pi/2.
    if plane not in PLANES:
        raise ValueError(f"Unknown plane {plane!r}")
    g = slice_geometry(run, r_plot, n_grid, plane)
    x, idx, w = g["x"], g["idx"], g["w"]
    if mode_set == "radiative":
        active = {tuple(m) for m in radiative}
    elif mode_set == "bound":
        active = set(modes) - {tuple(m) for m in radiative}
    else:
        active = set(modes)
    by_m = {}
    for i, (l, m) in enumerate(modes):
        if (l, m) not in active:
            continue
        if plane == "xy":
            Y = Y_equator(l, m)
            if Y == 0.0:
                continue
        else:
            if (l, m) not in g["ytheta"]:
                g["ytheta"][(l, m)] = Y_theta(l, m, g["cos_t"])
            Y = g["ytheta"][(l, m)]
        U = psi[i, idx] * (1 - w) + psi[i, idx + 1] * w
        by_m[m] = by_m.get(m, 0) + Y * U
    zz = np.zeros(idx.shape, dtype=complex)
    for m, part in by_m.items():
        if m not in g["eimphi"]:
            g["eimphi"][m] = np.exp(1j * m * g["phi"])
        zz += part * g["eimphi"][m]
    density = (np.abs(zz * g["weight"]) ** 2).reshape(len(x), len(x))

    return {
        "step": step, "t": t_s, "R": R_s, "theta": th_s, "q": q_s,
        "r": clean(r[ir], 8), "radial_f32": f32_b64(radial), "reference": ref,
        "reference_label": f"initial |U_{n0}{l0}|² (pure state)",
        # Row-major float32, base64: a quarter of the JSON size and no parsing.
        "x": clean(x, 6), "n": len(x), "plane": plane, "density_f32": f32_b64(density),
    }


# ----------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------

def is_run(path):
    return os.path.isfile(os.path.join(path, "config.toml")) and os.path.isfile(os.path.join(path, "t.txt"))


# Folders that belong to a run, or are never worth descending into.
SKIP_DIRS = {"psi", "__pycache__", "node_modules", "env", "venv"}
_RUNS_CACHE = {"key": None, "time": 0.0, "runs": []}
_RUNS_LOCK = threading.Lock()


def list_runs(root, depth=4, ttl=10.0):
    """Run folders at or below root, newest first.

    Descends up to `depth` levels (so root/task/Simulation_* is found), does not
    look inside a run folder, and skips hidden folders. The walk is cached for
    `ttl` seconds because every request validates its run against this list
    and a shared cluster filesystem makes directory listings slow.
    """
    key = (root, depth)
    with _RUNS_LOCK:
        if _RUNS_CACHE["key"] == key and time.monotonic() - _RUNS_CACHE["time"] < ttl:
            return _RUNS_CACHE["runs"]
    if is_run(root):
        out = [root]
    else:
        out, seen = [], set()
        base = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _ in os.walk(root, followlinks=True):
            if is_run(dirpath):
                real = os.path.realpath(dirpath)       # a run linked twice is listed once
                if real not in seen:
                    seen.add(real); out.append(os.path.abspath(dirpath))
                dirnames[:] = []
                continue
            if dirpath.count(os.sep) - base >= depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS]
        out = sorted(out, key=lambda p: os.path.getmtime(os.path.join(p, "t.txt")), reverse=True)
    with _RUNS_LOCK:
        _RUNS_CACHE.update(key=key, time=time.monotonic(), runs=out)
    return out


def run_label(root, folder):
    """(group, name) for the run picker: the run's parent relative to root, and its folder name."""
    rel = os.path.relpath(folder, root)
    group = os.path.dirname(rel)
    return ("" if group in ("", ".") else group), os.path.basename(folder)


_BRIEF_CACHE = {}


def run_brief(folder):
    """Key parameters from config.toml for the run picker, cached on the file's mtime."""
    path = os.path.join(folder, "config.toml")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    hit = _BRIEF_CACHE.get(folder)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        cfg = {}
    get = lambda sec, key, default: (cfg.get(sec, {}) or {}).get(key, default)
    try:
        brief = {"q": float(get("binary", "q", 0.001)), "mu": float(get("atom", "mu", 0.2)),
                 "M_c": float(get("atom", "M_c", 0.0)), "R0": float(get("binary", "R", 50.0)),
                 "counter": bool(get("binary", "counter", False)),
                 "evolving": bool(get("simulation", "evolve_orbit", False)),
                 "absorption": bool(get("simulation", "absorption", False))}
    except (TypeError, ValueError):
        brief = {}
    _BRIEF_CACHE[folder] = (mtime, brief)
    return brief


# Left out of the diagnostics download: the restart state is large and not a diagnostic.
DOWNLOAD_SKIP = {"checkpoint.bin"}


def add_run_to_zip(zf, folder, step, prefix):
    """Add the run's top-level files (minus DOWNLOAD_SKIP) and one psi snapshot under `prefix`/.

    `step` picks psi/snapshot_<step>.txt; None takes the latest snapshot. Other
    subfolders (psi_dt, the rest of psi) are left out. Returns the step added.
    """
    for entry in sorted(os.scandir(folder), key=lambda e: e.name):
        if entry.is_file() and entry.name not in DOWNLOAD_SKIP and not entry.name.startswith("."):
            zf.write(entry.path, f"{prefix}/{entry.name}")
    if step is None:
        steps = get_run(folder).snapshot_steps()
        step = steps[-1] if steps else None
    if step is not None:
        snap = os.path.join(folder, "psi", f"snapshot_{step:06d}.txt")
        if os.path.isfile(snap):
            zf.write(snap, f"{prefix}/psi/snapshot_{step:06d}.txt")
            return step
    return None


def run_tag(folder):
    """Folder name for a run in a multi-run zip, e.g. q0.01_R40_Mc0.01_counter."""
    b = run_brief(folder)
    if not b:
        return os.path.basename(os.path.normpath(folder))
    return f"q{b['q']:g}_R{b['R0']:g}_Mc{b['M_c']:g}_{'counter' if b['counter'] else 'co'}"


def write_runs_zip(items, out):
    """Zip several runs, each (folder, step, label path), into subfolders named by run_tag.

    Runs sharing the same parameters get _2, _3, ... appended. An index.txt maps
    each subfolder back to its run folder and the snapshot included.
    """
    used, index = {}, ["# subfolder\tsnapshot step\trun folder"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for folder, step, shown in items:
            tag = run_tag(folder)
            used[tag] = used.get(tag, 0) + 1
            if used[tag] > 1:
                tag = f"{tag}_{used[tag]}"
            added = add_run_to_zip(zf, folder, step, tag)
            index.append(f"{tag}\t{'-' if added is None else added}\t{shown}")
        zf.writestr("index.txt", "\n".join(index) + "\n")


class Handler(BaseHTTPRequestHandler):
    root = "."
    root_real = "."
    max_points = 4000
    depth = 4

    def log_message(self, *args):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def run_from(self, qs):
        folder = qs.get("run", [None])[0]
        if folder is None:
            runs = list_runs(self.root, self.depth)
            if not runs:
                raise ValueError("No simulation folders found")
            folder = runs[0]
        # Any run folder inside the watched tree (checked directly, so a new run
        # works before the cached listing refreshes) or reached through a
        # symlink in it (then it is in the listing).
        real = os.path.realpath(folder)
        inside = real == self.root_real or real.startswith(self.root_real.rstrip(os.sep) + os.sep)
        if not (is_run(real) and (inside or os.path.abspath(folder) in list_runs(self.root, self.depth))):
            raise ValueError("Unknown run folder")
        return get_run(real)

    def send_zip(self, build, fname):
        # Spool to a temp file so large runs are not held in memory.
        with tempfile.TemporaryFile() as tmp:
            build(tmp)
            size = tmp.tell(); tmp.seek(0)
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            while chunk := tmp.read(1 << 20):
                self.wfile.write(chunk)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if url.path == "/api/download":
                # {"runs": [{"run": folder, "step": n or null (latest snapshot)}, ...]}
                items = []
                for it in body.get("runs", []):
                    run = self.run_from({"run": [it["run"]]})
                    step = it.get("step")
                    items.append((run.folder, None if step is None else int(step), os.path.abspath(it["run"])))
                if not items:
                    raise ValueError("No runs selected")
                self.send_zip(lambda out: write_runs_zip(items, out), f"simulations_{len(items)}runs.zip")
            else:
                self.send(404, json.dumps({"error": "not found"}))
        except Exception as exc:
            self.send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        try:
            if url.path == "/":
                with open(os.path.join(SCRIPT_DIR, "live_dashboard.html"), "rb") as f:
                    self.send(200, f.read(), "text/html; charset=utf-8")
            elif url.path.startswith("/assets/"):
                name = os.path.basename(url.path)
                path = os.path.join(SCRIPT_DIR, "assets", name)
                types = {".woff2": "font/woff2", ".svg": "image/svg+xml", ".css": "text/css", ".js": "application/javascript"}
                if not name or not os.path.isfile(path):
                    self.send(404, "not found", "text/plain")
                else:
                    with open(path, "rb") as f:
                        self.send(200, f.read(), types.get(os.path.splitext(name)[1], "application/octet-stream"))
            elif url.path == "/favicon.ico":
                self.send(204, b"", "image/x-icon")
            elif url.path == "/api/runs":
                runs = list_runs(self.root, self.depth)
                self.send(200, json.dumps([dict(zip(("group", "name"), run_label(self.root_real, p)), folder=p,
                                                params=run_brief(p)) for p in runs]))
            elif url.path == "/api/data":
                run = self.run_from(qs)
                window = float(qs.get("window", ["0"])[0] or 0)
                self.send(200, json.dumps(data_payload(run, window, self.max_points)))
            elif url.path == "/api/snapshot":
                run = self.run_from(qs)
                step = int(qs["step"][0])
                r_plot = float(qs.get("rplot", ["200"])[0])
                n_grid = max(64, min(600, int(qs.get("grid", ["300"])[0])))
                mode_set = qs.get("set", ["all"])[0]
                plane = qs.get("plane", ["xy"])[0]
                radiative = json.loads(qs.get("radiative", ["[]"])[0])
                self.send(200, json.dumps(snapshot_payload(run, step, r_plot, n_grid, mode_set, radiative, plane)))
            elif url.path == "/api/download":
                run = self.run_from(qs)
                step = qs.get("step", [""])[0]
                step = int(step) if step.strip() else None
                name = os.path.basename(os.path.normpath(run.folder))
                fname = name + (f"_step{step:06d}" if step is not None else "") + ".zip"

                def build(out):
                    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                        add_run_to_zip(zf, run.folder, step, name)
                self.send_zip(build, fname)
            else:
                self.send(404, json.dumps({"error": "not found"}))
        except Exception as exc:  # report to the page instead of dropping the request
            self.send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=os.path.join(SCRIPT_DIR, "..", "runs"),
                        help="a folder containing Simulation_* runs, possibly in task subfolders, "
                             "or one run folder (default: ../runs)")
    parser.add_argument("--depth", type=int, default=4,
                        help="how many folder levels below PATH to search for runs (default: 4)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-points", type=int, default=4000,
                        help="time samples sent per curve (the run is decimated evenly)")
    args = parser.parse_args()

    Handler.root = os.path.realpath(args.path)
    Handler.root_real = Handler.root
    Handler.depth = max(1, args.depth)
    Handler.max_points = args.max_points
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Watching {Handler.root}")
    print(f"Dashboard at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
