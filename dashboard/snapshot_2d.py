import math

import numpy as np
from matplotlib.colors import LogNorm, Normalize


def snapshot_2d(run, ax, cmap="inferno", plane="xy", r_plot=200.0, snapshot=None, n=400,
                log=True, vmin=None, vmax=None, modes=None, holes=True, colorbar=True,
                xlabel=None, ylabel=None, title=None, cbar_label=None):
    """|psi|^2 of one snapshot of `run` on a plane through the centre of mass, drawn on `ax`.

    snapshot: a step or time already loaded (run.snapshot_at), a Snapshot, or None for run.snapshot.
    plane: "xy" (orbital plane), "zx" or "zy"; r_plot: half-width of the square, in M.
    log: logarithmic colour scale; vmin/vmax fix the range (use the same ones across panels to compare phases).
    modes: only these (l, m) modes (default all). holes: mark the black holes (in the xy plane, and on the
    horizontal axis of zx / zy when they lie on it).
    xlabel, ylabel, title, cbar_label: None keeps the default ("x [M]", ..., "t = ... M", |psi|^2);
    "" (or False) leaves that label out; any other string replaces it. title may use {t} and {step},
    e.g. title="$t = {t:.0f}\\,M$" (a title with other braces, such as LaTeX, is used as given).
    Returns the QuadMesh.
    """
    if snapshot is None:
        snap = run.snapshot
    elif hasattr(snapshot, "psi"):
        snap = snapshot
    elif isinstance(snapshot, (int, np.integer)):
        snap = run.snapshot_at(int(snapshot))
    else:
        snap = run.snapshot_at(time=float(snapshot))
    h, v, rho = snap.slice(plane, r_plot=r_plot, n=n, modes=modes)

    if log:
        positive = rho[rho > 0]
        top = vmax if vmax is not None else (positive.max() if positive.size else 1.0)
        bottom = vmin if vmin is not None else top * 1e-6
        norm = LogNorm(vmin=bottom, vmax=top, clip=True)
        rho = np.where(rho > 0, rho, bottom)
    else:
        norm = Normalize(vmin=vmin if vmin is not None else 0.0, vmax=vmax if vmax is not None else rho.max())
    mesh = ax.pcolormesh(h, v, rho, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.set_aspect("equal")
    labels = {"xy": ("x [M]", "y [M]"), "zx": ("x [M]", "z [M]"), "zy": ("y [M]", "z [M]")}[plane]
    pick = lambda given, default: default if given is None else (given or "")
    ax.set_xlabel(pick(xlabel, labels[0]))
    ax.set_ylabel(pick(ylabel, labels[1]))
    title_text = pick(title, "t = {t:.4g} M")
    try:
        title_text = title_text.format(t=snap.t, step=snap.step)
    except (KeyError, IndexError, ValueError):
        pass   # other braces (LaTeX such as \mathrm{...}): use the text as given
    ax.set_title(title_text)

    if holes and math.isfinite(snap.R_orb):
        theta, q, R = snap.theta_orb, snap.q, snap.R_orb
        if run.alone:
            spots = [(R, theta)]                                   # R_orb is the lone hole's distance
        else:
            spots = [(q / (1 + q) * R, theta), (R / (1 + q), theta + math.pi)]  # primary, companion
        for k, (d, ang) in enumerate(spots):
            x, y = d * math.cos(ang), d * math.sin(ang)
            point = {"xy": (x, y), "zx": (x, 0.0), "zy": (y, 0.0)}[plane]
            if plane != "xy" and abs({"zx": y, "zy": x}[plane]) > 1e-9 * max(1.0, d):
                continue                                           # off this plane
            ax.plot(*point, marker="o", ms=6 if k == 0 else 4, mfc="none", mec="white", mew=1.2)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=pick(cbar_label, r"$|\psi|^2$"))
    return mesh
