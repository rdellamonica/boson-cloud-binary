"""
Load a simulation run folder into one object, for notebooks and publication plots.

    import sys; sys.path.insert(0, "path/to/boson-cloud-binary/dashboard")
    from run_loader import load_run

    run = load_run("runs/Simulation_2026_11_10_10_18_06")            # last snapshot
    run = load_run(folder, step=400)                                  # snapshot of step 400
    run = load_run(folder, time=1.5e4)                                # snapshot nearest t = 1.5e4 M
    run = load_run(folder, quantities=["R_orb", "chi_1", "cloud_mass"])  # only these (and t)

    run.t, run.R_orb, run.M_1, run.norms[:, run.mode("1,1")]          # stored diagnostics
    run.cloud_mass, run.energy, run.mass_budget["absorbed_1"]         # derived, as on the dashboard
    run.Lz_budget["cloud"], run.Lz_budget["modes"][:, run.mode("2,2")]  # signed L_z, hbar per particle
    run.snapshot.U(1, 1), run.snapshot.t                              # one snapshot
    x, y, rho = run.snapshot.slice("xy", r_plot=300, n=400)           # |psi|^2 in a plane

Everything the solver writes is read: every column file at the top of the run
folder (t.txt, norms.txt, spins.txt, ...), config.toml, the radial grid, the
mode list, the evolved-mode log of pruned runs and one snapshot of psi (and of
psi_dt when it was saved for that step). Files are trimmed to the rows they all
have, so a run still being written loads consistently. Only NumPy is needed
(and tomllib, Python 3.11+); the slice reuses the dashboard's reconstruction.

Conventions (as in the solver and the dashboard): quantities are in units of the
primary mass M1 = 1. Diagnostic row k is written by step k + 1, so snapshot s is
at row s - 1 (snapshot 0 is the initial state, t = 0). norms.txt holds particle
numbers N_LM = int |U_LM|^2 dr inside r_ex, scaled to M_c / mu; masses are mu N.
"""

import importlib.util
import math
import os
import re
import tomllib
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SNAPSHOT_RE = re.compile(r"snapshot_(\d+)\.txt$")

# One value per row.
SCALAR_FILES = ["t", "energy", "p_bin", "p_bin_inst", "dV_bin_dt", "R_orb", "theta_orb",
                "Omega_orb", "M_1", "q", "Lz_absorbed_rate"]
# One column per mode, in the order of modes.txt.
MODE_FILES = ["flux", "energy_flux", "energy_modes", "norms"]
# Named columns.
COLUMN_FILES = {
    "mass_budget": ["absorbed_1", "absorbed_2", "pruned"],
    "spins": ["chi_1", "chi_2", "J_absorbed_1", "J_absorbed_2"],
    "Lz_budget": ["Lz_ionized", "Lz_torque", "Lz_absorbed", "Lz_pruned", "Lz_damped"],
}
# Not diagnostics: grid, modes and the evolved-mode log are read separately.
SPECIAL_FILES = {"r", "modes", "active_modes"}
# Derived quantities and the files each one needs.
DERIVED = {
    "cloud_mass": ["norms"], "mass_modes": ["norms"], "energy_total": ["energy"],
    "energy_modes_total": ["energy_modes"], "M_2": ["q", "M_1"], "ionized_mass": ["flux", "norms"],
    "mass_budget": ["norms", "flux", "mass_budget"], "active_modes": [],
    "Lz_cloud": ["norms"], "Lz_modes": ["norms"],
    "Lz_budget": ["norms", "flux", "mass_budget", "Lz_budget", "spins", "Lz_absorbed_rate", "p_bin_inst",
                  "Omega_orb"],
}
# The orbit at the snapshot comes from these (read for the snapshot, not returned unless asked for).
SNAPSHOT_FILES = ["R_orb", "theta_orb", "q", "M_1"]


def _dashboard():
    """The dashboard module, for its field reconstruction (imported once, no server)."""
    global _DASH
    try:
        return _DASH
    except NameError:
        spec = importlib.util.spec_from_file_location("_live_dashboard", os.path.join(_HERE, "live_dashboard.py"))
        _DASH = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_DASH)
        return _DASH


def _read_table(path):
    """Rows of a whitespace-separated file as a 2D float array; a partly written last line is dropped."""
    rows, width = [], None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                values = [float(v) for v in parts]
            except ValueError:
                continue
            if width is None:
                width = len(values)
            if len(values) == width:
                rows.append(values)
    return np.array(rows, dtype=float).reshape(-1, width or 0)


@dataclass
class Snapshot:
    """One saved state: psi[j] = U_LM(r) of mode j (complex, shape n_modes x Nr)."""
    step: int
    t: float
    psi: np.ndarray
    r: np.ndarray
    modes: list
    psi_dt: np.ndarray = None   # dU/dt, only where the solver saved it (step 0 in recent runs)
    R_orb: float = float("nan")  # orbit at this snapshot (row step - 1)
    theta_orb: float = float("nan")
    q: float = float("nan")
    M_1: float = float("nan")

    def U(self, l, m):
        """Radial profile U_lm(r) of mode (l, m)."""
        return self.psi[self.modes.index((l, m))]

    def density(self, l=None, m=None):
        """|U_lm(r)|^2 of one mode, or summed over every mode when l is None."""
        if l is None:
            return np.sum(np.abs(self.psi) ** 2, axis=0)
        return np.abs(self.U(l, m)) ** 2

    def field(self, x, y, z, modes=None):
        """psi(x, y, z) about the centre of mass, from the modes listed (default: all)."""
        dash = _dashboard()
        X, Y, Z = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y, z)))
        g = dash.point_tables(self.r, X.ravel(), Y.ravel(), Z.ravel())
        active = set(self.modes) if modes is None else {tuple(mm) for mm in modes}
        return dash.field_values(self.psi, self.modes, active, g).reshape(X.shape)

    def slice(self, plane="xy", r_plot=200.0, n=300, modes=None):
        """|psi|^2 on an n x n grid of half-width r_plot in a plane through the
        centre of mass ("xy" orbital plane, "zx" or "zy"). Returns (h, v, rho) with
        rho[i, j] at horizontal h[j], vertical v[i], ready for pcolormesh/imshow."""
        h = np.linspace(-r_plot, r_plot, n)
        H, V = np.meshgrid(h, h, indexing="xy")
        zero = np.zeros_like(H)
        xyz = {"xy": (H, V, zero), "zx": (H, zero, V), "zy": (zero, H, V)}[plane]
        return h, h.copy(), np.abs(self.field(*xyz, modes=modes)) ** 2


@dataclass
class RunData:
    """Every stored quantity of a run. Per-row arrays share the length n_rows;
    per-mode arrays have shape (n_rows, n_modes), columns in the order of `modes`."""
    folder: str
    name: str
    config: dict
    config_text: str
    r: np.ndarray
    modes: list
    n_rows: int
    snapshot_steps: list
    snapshot_times: np.ndarray
    snapshot: Snapshot = None
    data: dict = field(default_factory=dict)          # every column file, as stored
    active_modes: dict = None                         # pruned runs: steps, t, count, flags
    mass_budget: dict = None                          # derived budget (dashboard's panel)
    Lz_budget: dict = None                            # signed L_z budget (dashboard's panel)

    def __getattr__(self, name):
        # run.t, run.norms, run.chi_1 ... read from `data`.
        data = self.__dict__.get("data", {})
        if name in data:
            return data[name]
        raise AttributeError(f"{name!r}: not a stored quantity of this run (have: {', '.join(sorted(data))})")

    def mode(self, lm):
        """Column of mode (l, m), given as (l, m) or "l,m"."""
        if isinstance(lm, str):
            lm = tuple(int(v) for v in lm.split(","))
        return self.modes.index(tuple(lm))

    @property
    def mu(self):
        return float(self.config.get("atom", {}).get("mu", 0.2))

    @property
    def alone(self):
        """No companion: secondary_potential = false or q = 0."""
        sim, binary = self.config.get("simulation", {}), self.config.get("binary", {})
        return float(binary.get("q", 0.001)) == 0.0 or not sim.get("secondary_potential", True)

    def keys(self):
        return sorted(self.data)


def load_run(folder, step=None, time=None, snapshot=True, quantities=None):
    """Load the run in `folder`.

    step: snapshot step to load (default: the last complete one); time: load the
    snapshot nearest this time instead; snapshot=False skips psi altogether.
    quantities: only these, e.g. ["R_orb", "chi_1", "cloud_mass"]; default all.
    Names are files ("norms", "spins"), columns of a file ("chi_1", "absorbed_2")
    or derived quantities ("cloud_mass", "ionized_mass", "mass_budget", ...);
    only the files they need are read. t is always loaded.
    """
    folder = os.path.abspath(os.path.expanduser(folder))
    with open(os.path.join(folder, "config.toml"), "rb") as f:
        config_text = f.read().decode("utf-8", "replace")
    config = tomllib.loads(config_text)
    r = np.loadtxt(os.path.join(folder, "r.txt"))
    with open(os.path.join(folder, "modes.txt")) as f:
        modes = [tuple(int(v) for v in line.split(",")) for line in f if line.strip()]
    mu = float(config.get("atom", {}).get("mu", 0.2))

    # Every column file at the top of the folder, or those the requested quantities need.
    available = sorted(os.path.splitext(e)[0] for e in os.listdir(folder)
                       if e.endswith(".txt") and os.path.splitext(e)[0] not in SPECIAL_FILES)
    wanted = None
    if quantities is not None:
        if isinstance(quantities, str):
            quantities = [quantities]
        wanted = set(quantities) | {"t"}
        column_of = {col: name for name, cols in COLUMN_FILES.items() for col in cols}
        files = {"t"}
        for q_name in wanted:
            if q_name in DERIVED:
                files.update(DERIVED[q_name])
            elif q_name in column_of:
                files.add(column_of[q_name])
            elif q_name in available:
                files.add(q_name)
            else:
                known = sorted(set(available) | set(column_of) | set(DERIVED))
                raise ValueError(f"unknown quantity {q_name!r}; this run has: {', '.join(known)}")
        if snapshot:
            files.update(SNAPSHOT_FILES)
    raw = {}
    for name in available:
        entry = name + ".txt"
        if wanted is not None and name not in files:
            continue
        table = _read_table(os.path.join(folder, entry))
        if table.size:
            raw[name] = table
    if "t" not in raw:
        raise FileNotFoundError(f"{folder}: no t.txt, so no diagnostic rows yet")
    # Keep the rows every loaded file has, so that all per-row quantities have the
    # same length and can be plotted against each other. A run in progress has at
    # most a row of difference between files; more than that is reported.
    lengths = {k: len(v) for k, v in raw.items()}
    n_rows = min(lengths.values())
    longest = max(lengths.values())
    if longest - n_rows > 1:
        import warnings
        short = ", ".join(f"{k} ({n})" for k, n in sorted(lengths.items(), key=lambda kv: kv[1]) if n < longest)
        warnings.warn(f"{os.path.basename(folder)}: files have different lengths, so every quantity is cut "
                      f"to the {n_rows} rows they all have (longest {longest}; shorter: {short})")

    data = {}
    for name, table in raw.items():
        table = table[:n_rows]
        if name in COLUMN_FILES:
            for k, col in enumerate(COLUMN_FILES[name][:table.shape[1]]):
                data[col] = table[:, k]
            data[name] = table
        elif name in MODE_FILES or (table.shape[1] == len(modes) and name not in SCALAR_FILES):
            data[name] = table
        else:
            data[name] = table[:, 0] if table.shape[1] == 1 else table
    t = data["t"]

    # Derived quantities, with the dashboard's formulas.
    if "norms" in data:
        cloud = mu * data["norms"].sum(axis=1)
        data["mass_modes"] = mu * data["norms"]                       # M_c^LM inside r_ex
        data["cloud_mass"] = cloud                                     # M_c inside r_ex
    if "energy" in data:
        data["energy_total"] = mu * data["energy"]                     # E_s of the cloud
    if "energy_modes" in data:
        data["energy_modes_total"] = mu * data["energy_modes"]         # E_s^LM
    if "q" in data and "M_1" in data:
        data["M_2"] = data["q"] * data["M_1"]
    budget = None
    if "flux" in data and "norms" in data:
        J = data["flux"].sum(axis=1)                                   # mass flux through r_ex
        ionized = J[0] * t[0] + np.concatenate(([0.0], np.cumsum(0.5 * (J[1:] + J[:-1]) * np.diff(t))))
        data["ionized_mass"] = ionized
        absorbed_1 = data.get("absorbed_1", np.zeros(n_rows))
        absorbed_2 = data.get("absorbed_2", np.zeros(n_rows))
        pruned = data.get("pruned", np.zeros(n_rows))
        M_ref = cloud[0] + ionized[0] + absorbed_1[0] + absorbed_2[0] + pruned[0]
        if M_ref > 0:
            budget = {"M_ref": M_ref, "residual": cloud / M_ref, "ionized": ionized / M_ref,
                      "absorbed_1": absorbed_1 / M_ref, "absorbed_2": absorbed_2 / M_ref,
                      "absorbed": (absorbed_1 + absorbed_2) / M_ref, "pruned": pruned / M_ref,
                      "total": (cloud + ionized + absorbed_1 + absorbed_2 + pruned) / M_ref,
                      "inferred": "absorbed_1" not in data}
            if budget["inferred"]:   # older runs: the absorbed part is what is missing
                budget["absorbed"] = (M_ref - cloud - ionized) / M_ref

    # Signed L_z budget in hbar per initial particle (N_ref = M_ref / mu), as on the dashboard:
    # cloud + ionized + absorbed + pruned + damped + binary = const. Exact (solver's Lz_budget.txt)
    # when the run wrote it; otherwise from the diagnostics, to O(dt).
    Lz = None
    if "norms" in data:
        m = np.array([mm for _, mm in modes], dtype=float)
        data["Lz_modes"] = data["norms"] * m                              # m N_lm inside r_ex
        data["Lz_cloud"] = data["Lz_modes"].sum(axis=1)
    if budget is not None and "Lz_cloud" in data:
        N_ref = budget["M_ref"] / mu
        exact = "Lz_ionized" in data
        if exact:
            ion, torque, absorbed = data["Lz_ionized"], data["Lz_torque"], data["Lz_absorbed"]
            pruned_Lz, damped = data["Lz_pruned"], data["Lz_damped"]
        else:
            j = (data["flux"] * m).sum(axis=1) / mu
            ion = j[0] * t[0] + np.concatenate(([0.0], np.cumsum(0.5 * (j[1:] + j[:-1]) * np.diff(t))))
            dt = np.diff(np.concatenate(([0.0], t)))
            torque = np.zeros(n_rows)
            binary_cfg = config.get("binary", {}) or {}
            alone = (float(binary_cfg.get("q", 0.001)) == 0.0
                     or not (config.get("simulation", {}) or {}).get("secondary_potential", True))
            if not alone and "p_bin_inst" in data and "Omega_orb" in data:
                Om = data["Omega_orb"] * (-1.0 if binary_cfg.get("counter", False) else 1.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    torque = np.cumsum(np.where(Om != 0, data["p_bin_inst"] / np.where(Om != 0, Om, 1.0), 0.0) * dt)
            absorbed = np.cumsum(data.get("Lz_absorbed_rate", np.zeros(n_rows)) * dt)
            if "J_absorbed_1" in data:
                absorbed = absorbed + data["J_absorbed_1"] + data["J_absorbed_2"]
            pruned_Lz = damped = np.zeros(n_rows)
        spin = data["J_absorbed_1"] + data["J_absorbed_2"] if "J_absorbed_1" in data else None
        total = data["Lz_cloud"] + ion + absorbed + pruned_Lz + damped - torque
        Lz = {"N_ref": N_ref, "exact": exact, "cloud": data["Lz_cloud"] / N_ref, "ionized": ion / N_ref,
              "absorbed": absorbed / N_ref, "binary": -torque / N_ref, "pruned": pruned_Lz / N_ref,
              "damped": damped / N_ref, "total": total / N_ref, "modes": data["Lz_modes"] / N_ref,
              "spin": None if spin is None else spin / N_ref,
              "orbital": None if spin is None else (absorbed - spin) / N_ref}

    # Evolved-mode log of a pruned run: "step t count flag_0 ... flag_{n-1}".
    active = None
    path = os.path.join(folder, "active_modes.txt")
    if os.path.exists(path) and (wanted is None or "active_modes" in wanted):
        log = _read_table(path)
        if log.size and log.shape[1] == 3 + len(modes):
            active = {"step": log[:, 0].astype(int), "t": log[:, 1], "count": log[:, 2].astype(int),
                      "flags": log[:, 3:].astype(bool)}

    psi_dir = os.path.join(folder, "psi")
    steps = sorted(int(m.group(1)) for n in (os.listdir(psi_dir) if os.path.isdir(psi_dir) else [])
                   if (m := _SNAPSHOT_RE.match(n)))
    row_t = lambda s: 0.0 if s == 0 else float(t[s - 1]) if 0 < s <= n_rows else float("nan")
    times = np.array([row_t(s) for s in steps])

    run = RunData(folder=folder, name=os.path.basename(folder), config=config, config_text=config_text,
                  r=r, modes=modes, n_rows=n_rows, snapshot_steps=steps, snapshot_times=times,
                  data=data, active_modes=active, mass_budget=budget, Lz_budget=Lz)
    if snapshot and steps:
        run.snapshot = _load_snapshot(run, folder, steps, times, step, time)
    if wanted is not None:
        # Only what was asked for (and t); missing ones (spins of a run without them) are left out.
        run.data = {k: v for k, v in data.items() if k in wanted}
        if "mass_budget" not in wanted:
            run.mass_budget = None
        if "Lz_budget" not in wanted:
            run.Lz_budget = None
        missing = sorted(wanted - set(run.data) - {"mass_budget", "Lz_budget", "active_modes"})
        if missing:
            import warnings
            warnings.warn(f"{run.name}: not stored in this run: {', '.join(missing)}")
    return run


def _read_psi(path, n_modes, Nr):
    dat = _read_table(path)
    if dat.shape != (n_modes * Nr, 2):
        return None   # still being written
    return (dat[:, 0] + 1j * dat[:, 1]).reshape(n_modes, Nr)


def _load_snapshot(run, folder, steps, times, step, time):
    n_modes, Nr = len(run.modes), len(run.r)
    if time is not None:
        ok = np.isfinite(times)
        if not ok.any():
            raise ValueError("no snapshot has a time yet")
        step = steps[int(np.argmin(np.where(ok, np.abs(times - time), np.inf)))]
    if step is not None:
        if step not in steps:
            near = min(steps, key=lambda s: abs(s - step))
            raise ValueError(f"no snapshot at step {step}; the nearest is {near} (available: "
                             f"{steps[0]} … {steps[-1]}, {len(steps)} snapshots in run.snapshot_steps)")
        candidates = [step]
    else:
        candidates = steps[::-1][:2]   # the last one may still be being written
    for s in candidates:
        psi = _read_psi(os.path.join(folder, "psi", f"snapshot_{s:06d}.txt"), n_modes, Nr)
        if psi is None:
            continue
        dt_path = os.path.join(folder, "psi_dt", f"snapshot_{s:06d}.txt")
        psi_dt = _read_psi(dt_path, n_modes, Nr) if os.path.exists(dt_path) else None
        row = s - 1
        pick = lambda name, default: (float(run.data[name][row]) if name in run.data and 0 <= row < run.n_rows
                                      else default)
        cfg = run.config.get("binary", {})
        q0, R0 = float(cfg.get("q", 0.001)), float(cfg.get("R", 50.0))
        R, theta, q, M1 = pick("R_orb", R0), pick("theta_orb", 0.0), pick("q", q0), pick("M_1", 1.0)
        if run.alone and row < 0:
            R, q = q0 / (1 + q0) * R0, 0.0   # the lone hole sits where the binary's primary would
        if run.alone and q0 == 0.0:
            R, theta = 0.0, 0.0
        return Snapshot(step=s, t=float(times[steps.index(s)]), psi=psi, r=run.r, modes=run.modes, psi_dt=psi_dt,
                        R_orb=R, theta_orb=theta, q=q, M_1=M1)
    raise ValueError(f"snapshot {candidates[0]} is incomplete (still being written?)")
