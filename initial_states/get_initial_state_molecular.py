#!/usr/bin/env python3
"""
get_initial_state_molecular.py

Construct a molecular initial state for the binary-cloud simulations by
adiabatically continuing an asymptotic atomic state down to the requested
simulation separation R0.

Field convention
----------------

    psi(r,theta,phi)
        = sum_lm U_lm(r)/r * Y_lm(theta,phi)

so that

    ||psi||^2
        = int dr sum_lm |U_lm(r)|^2.

The stationary problem is solved in the corotating frame,

    H_rot(R) = H_bin(R) - Omega(R) L_z,

with

    H_bin =
        - nabla^2/(2 mu)
        - mu M1 / |x-R1|
        - mu M2 / |x-R2|.

Initialization logic
--------------------

q < 1:
    By default, start from ONE asymptotic atom localized around the
    primary. R_asym is increased until this localized atomic state is
    dominated by a single eigenstate of the full binary Hamiltonian.

q ~= 1:
    Equivalent primary/secondary atomic levels become degenerate, so
    the automatic prescription switches to a coherent pair.

The selected molecular branch is then followed down to R0 using
maximum-overlap continuation. If the overlap becomes too small, the
script adaptively inserts intermediate separations.

Diagnostics
-----------

The continuation CSV contains:

    R
    Omega
    E_rot
    continuation_overlap
    residual
    spectral_gap
    nev_used

and, where meaningful,

    P_primary
    P_secondary
    two_state_capture
    molecularity
    atomic_overlap
    atom_capture_primary
    atom_capture_secondary

where

    molecularity =
        4 P_primary P_secondary / (P_primary + P_secondary)^2.

molecularity ~ 0:
    state localized on one centre.

molecularity ~ 1:
    state equally shared between the two localized reference states.

A low two_state_capture means that other atomic/molecular states have
become important.

Dependencies
------------

    numpy
    scipy
    sympy

For Python < 3.11 also install:

    tomli

Example
-------

    python get_initial_state_molecular.py \
        --config config.toml \
        --output initial_modes.bin
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from scipy.special import eval_genlaguerre, gammaln
from sympy.physics.wigner import wigner_3j


# ======================================================================
# SCIPY SPHERICAL-HARMONIC COMPATIBILITY
# ======================================================================

try:
    from scipy.special import sph_harm_y

    def Ylm(l, m, theta, phi):
        return sph_harm_y(l, m, theta, phi)

except ImportError:
    from scipy.special import sph_harm

    def Ylm(l, m, theta, phi):
        # Old scipy convention:
        #
        #     sph_harm(m, l, azimuth, polar)
        #
        return sph_harm(m, l, phi, theta)


try:
    import tomllib

except ModuleNotFoundError:
    import tomli as tomllib


# ======================================================================
# CONFIG
# ======================================================================

def load_config(path):

    with open(path, "rb") as f:
        c = tomllib.load(f)

    g = c["grid"]
    b = c["binary"]
    a = c["atom"]
    s = c["simulation"]

    a2 = c.get("atom_secondary", {})
    mi = c.get("molecular_init", {})

    nev = int(
        mi.get(
            "nev",
            12
        )
    )

    p = dict(

        # ----------------------------------------------------------
        # Grid
        # ----------------------------------------------------------

        N=int(g["N"]),
        L=float(g["L"]),

        # ----------------------------------------------------------
        # Binary
        # ----------------------------------------------------------

        M1=float(
            b.get(
                "M1",
                1.0
            )
        ),

        q=float(
            b["q"]
        ),

        R0=float(
            b["R"]
        ),

        counter=bool(
            b.get(
                "counter",
                False
            )
        ),

        # ----------------------------------------------------------
        # Primary atom
        # ----------------------------------------------------------

        mu=float(
            a["mu"]
        ),

        Mc=float(
            a.get(
                "M_c",
                1.0
            )
        ),

        n1=int(
            a["n"]
        ),

        l1=int(
            a["l"]
        ),

        m1=int(
            a["m"]
        ),

        # ----------------------------------------------------------
        # Secondary reference atom
        #
        # For q<1 this is NOT populated initially by default.
        # It is used mainly for localization diagnostics.
        # ----------------------------------------------------------

        n2=int(
            a2.get(
                "n",
                a["n"]
            )
        ),

        l2=int(
            a2.get(
                "l",
                a["l"]
            )
        ),

        m2=int(
            a2.get(
                "m",
                a["m"]
            )
        ),

        amp2=float(
            a2.get(
                "amplitude",
                1.0
            )
        ),

        # ----------------------------------------------------------
        # Angular basis
        # ----------------------------------------------------------

        lmax=int(
            s["l_max"]
        ),

        lammax=int(
            s.get(
                "lambda_max",
                2 * int(
                    s["l_max"]
                )
            )
        ),

        basis=str(
            s.get(
                "initial_basis",
                s.get(
                    "basis",
                    "z"
                )
            )
        ).lower(),

        mcut=int(
            s.get(
                "m_cut",
                s["l_max"]
            )
        ),

        # ----------------------------------------------------------
        # Molecular initialization
        # ----------------------------------------------------------

        # auto:
        #
        # q < 1       -> primary
        # q ~ 1       -> coherent_pair

        initial_state=str(
            mi.get(
                "initial_state",
                "auto"
            )
        ).lower(),

        q_equal_tol=float(
            mi.get(
                "q_equal_tol",
                1.0e-4
            )
        ),

        relative_phase=float(
            mi.get(
                "relative_phase",
                0.0
            )
        ),

        R_asym=mi.get(
            "R_asym",
            None
        ),

        R_growth=float(
            mi.get(
                "R_growth",
                1.25
            )
        ),

        max_asym_tries=int(
            mi.get(
                "max_asym_tries",
                16
            )
        ),

        # Atomic state must be represented well by current basis.

        min_capture=float(
            mi.get(
                "min_capture",
                0.95
            )
        ),

        # More physical asymptotic criterion:
        #
        # the localized atom itself must be almost one full-binary
        # eigenstate.

        asym_eig_overlap=float(
            mi.get(
                "asym_eig_overlap",
                0.995
            )
        ),

        # ----------------------------------------------------------
        # Continuation
        # ----------------------------------------------------------

        ncont=int(
            mi.get(
                "continuation_steps",
                30
            )
        ),

        nev=nev,

        nev_max=int(
            mi.get(
                "nev_max",
                max(
                    48,
                    4 * nev
                )
            )
        ),

        min_step_overlap=float(
            mi.get(
                "min_step_overlap",
                0.90
            )
        ),

        max_refinements=int(
            mi.get(
                "max_refinements",
                8
            )
        ),

        abort_on_low_overlap=bool(
            mi.get(
                "abort_on_low_overlap",
                True
            )
        ),

        # ----------------------------------------------------------
        # Angular quadrature
        # ----------------------------------------------------------

        ntheta=int(
            mi.get(
                "ntheta",
                max(
                    24,
                    4 * int(
                        s["l_max"]
                    ) + 8
                )
            )
        ),

        nphi=int(
            mi.get(
                "nphi",
                max(
                    48,
                    8 * int(
                        s["l_max"]
                    ) + 16
                )
            )
        ),

        # ----------------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------------

        diagnostics=bool(
            mi.get(
                "diagnostics",
                True
            )
        ),

        diagnostic_min_capture=float(
            mi.get(
                "diagnostic_min_capture",
                0.70
            )
        ),

        # Search for asymptotic primary-n1 / secondary-n
        # accidental degeneracies.

        resonance_nmax=int(
            mi.get(
                "resonance_nmax",
                8
            )
        ),

        resonance_detuning_tol=float(
            mi.get(
                "resonance_detuning_tol",
                5.0e-3
            )
        ),

        mass_mode=str(
            mi.get(
                "mass_mode",
                "per_atom"
            )
        ).lower(),
    )

    p["M2"] = (
        p["q"]
        * p["M1"]
    )

    if not (
        0.0
        < p["q"]
        <= 1.0
    ):
        raise ValueError(
            "This initializer assumes 0 < q <= 1 "
            "with M2 = q M1."
        )

    if p["basis"] not in {
        "x",
        "z"
    }:
        raise ValueError(
            "simulation.initial_basis/basis "
            "must be 'x' or 'z'."
        )

    if p["basis"] == "x":

        if (
            p["mcut"] < 0
            or
            p["mcut"] > p["lmax"]
        ):
            raise ValueError(
                "Require 0 <= m_cut <= l_max."
            )

    if p["N"] < 4:
        raise ValueError(
            "grid.N must be at least 4."
        )

    if p["L"] <= 0.0:
        raise ValueError(
            "grid.L must be positive."
        )

    if p["R0"] <= 0.0:
        raise ValueError(
            "binary.R must be positive."
        )

    return p


# ======================================================================
# MODE LIST
# ======================================================================

def modes_list(p):

    if p["basis"] == "z":

        return [
            (l, m)
            for l in range(
                p["lmax"] + 1
            )
            for m in range(
                -l,
                l + 1
            )
        ]

    mc = p["mcut"]

    return [
        (l, m)
        for l in range(
            p["lmax"] + 1
        )
        for m in range(
            -min(
                l,
                mc
            ),
            min(
                l,
                mc
            ) + 1
        )
    ]


# ======================================================================
# BINARY GEOMETRY
# ======================================================================

def omega(
    R,
    p
):

    sign = (
        -1.0
        if p["counter"]
        else +1.0
    )

    return (
        sign
        * math.sqrt(
            (
                p["M1"]
                + p["M2"]
            )
            / R**3
        )
    )


def positions(
    R,
    p
):
    """
    COM frame.

    Binary is placed on the x axis:

        x1 = + M2/Mtot R
        x2 = - M1/Mtot R
    """

    mt = (
        p["M1"]
        + p["M2"]
    )

    x1 = (
        +p["M2"]
        / mt
        * R
    )

    x2 = (
        -p["M1"]
        / mt
        * R
    )

    return (
        x1,
        x2
    )


# ======================================================================
# HYDROGENIC ATOMIC STATES
# ======================================================================

def hydrogenic_R(
    r,
    n,
    l,
    mu,
    M
):
    """
    Unit-normalized hydrogenic radial wavefunction R_nl(r).

    Potential:

        V = - mu M/r

    Bohr radius:

        a0 = 1/(mu^2 M)
    """

    if n <= 0:
        raise ValueError(
            "n must be > 0."
        )

    if (
        l < 0
        or
        l >= n
    ):
        raise ValueError(
            f"Invalid hydrogenic state "
            f"n={n}, l={l}."
        )

    a0 = (
        1.0
        / (
            mu
            * mu
            * M
        )
    )

    rho = (
        2.0
        * np.asarray(r)
        / (
            n
            * a0
        )
    )

    logN = (

        1.5
        * np.log(
            2.0
            / (
                n
                * a0
            )
        )

        + 0.5
        * (
            gammaln(
                n - l
            )
            - np.log(
                2 * n
            )
            - gammaln(
                n + l + 1
            )
        )
    )

    return (

        np.exp(
            logN
            - rho / 2.0
        )

        * rho**l

        * eval_genlaguerre(
            n - l - 1,
            2 * l + 1,
            rho
        )
    )


def mean_radius(
    n,
    l,
    mu,
    M
):
    """
    Hydrogenic <r>.
    """

    a0 = (
        1.0
        / (
            mu
            * mu
            * M
        )
    )

    return (
        0.5
        * a0
        * (
            3 * n * n
            - l * (
                l + 1
            )
        )
    )


def isolated_energy(
    mu,
    M,
    n
):
    """
    Leading nonrelativistic hydrogenic binding energy.
    """

    return (
        -mu**3
        * M**2
        / (
            2.0
            * n**2
        )
    )


# ======================================================================
# ANGULAR COORDINATES / QUADRATURE
# ======================================================================

def angular_grid(
    ntheta,
    nphi
):
    """
    Gauss-Legendre quadrature in cos(theta)
    and uniform quadrature in phi.
    """

    u, wu = (
        np.polynomial
        .legendre
        .leggauss(
            ntheta
        )
    )

    ph = (
        2.0
        * np.pi
        * np.arange(
            nphi
        )
        / nphi
    )

    u, ph = np.meshgrid(
        u,
        ph,
        indexing="ij"
    )

    w = (

        np.repeat(
            wu[:, None],
            nphi,
            axis=1
        )

        * (
            2.0
            * np.pi
            / nphi
        )
    )

    st = np.sqrt(
        np.maximum(
            0.0,
            1.0
            - u * u
        )
    )

    x = (
        st
        * np.cos(ph)
    )

    y = (
        st
        * np.sin(ph)
    )

    z = u

    return (
        x.ravel(),
        y.ravel(),
        z.ravel(),
        w.ravel()
    )


def angles_for_axis(
    x,
    y,
    z,
    axis
):
    """
    z quantization:

        theta_z = acos(z/r)
        phi_z   = atan2(y,x)

    x quantization:

    rotated coordinates

        x' = y
        y' = z
        z' = x

    hence

        theta_x = acos(x/r)
        phi_x   = atan2(z,y)
    """

    rr = np.sqrt(
        x * x
        + y * y
        + z * z
    )

    safe = np.where(
        rr > 0.0,
        rr,
        1.0
    )

    if axis == "z":

        theta = np.arccos(
            np.clip(
                z / safe,
                -1.0,
                1.0
            )
        )

        phi = np.mod(
            np.arctan2(
                y,
                x
            ),
            2.0 * np.pi
        )

    elif axis == "x":

        theta = np.arccos(
            np.clip(
                x / safe,
                -1.0,
                1.0
            )
        )

        phi = np.mod(
            np.arctan2(
                z,
                y
            ),
            2.0 * np.pi
        )

    else:

        raise ValueError(
            "axis must be 'x' or 'z'."
        )

    return (
        theta,
        phi
    )


# ======================================================================
# TRANSLATED ATOM -> COM SPHERICAL-HARMONIC BASIS
# ======================================================================

def project_atom(
    r,
    modes,
    basis,
    mu,
    M,
    xc,
    Om,
    n,
    l,
    m,
    ntheta,
    nphi
):
    """
    Project an atom centered at x=xc onto the COM-centered
    spherical-harmonic basis.

    The atomic m is interpreted in the same quantization convention
    as `basis`.

    A leading Galilean phase is included for an atom comoving with
    the corresponding BH.

    At t=0:

        v = Omega zhat x R,

    and R lies on x, therefore

        v_y = Omega x_c.
    """

    dx, dy, dz, w = angular_grid(
        ntheta,
        nphi
    )

    # --------------------------------------------------------------
    # COM angular basis
    # --------------------------------------------------------------

    theta_basis, phi_basis = angles_for_axis(
        dx,
        dy,
        dz,
        basis
    )

    Ybasis = np.array(
        [
            Ylm(
                L,
                Mm,
                theta_basis,
                phi_basis
            )
            for L, Mm in modes
        ]
    )

    U = np.zeros(
        (
            len(modes),
            len(r)
        ),
        dtype=complex
    )

    vy = (
        Om
        * xc
    )

    # --------------------------------------------------------------
    # Radial chunks
    # --------------------------------------------------------------

    for i0 in range(
        1,
        len(r) - 1,
        64
    ):

        i1 = min(
            i0 + 64,
            len(r) - 1
        )

        rr = r[
            i0:i1,
            None
        ]

        x = (
            rr
            * dx
        )

        y = (
            rr
            * dy
        )

        z = (
            rr
            * dz
        )

        # ----------------------------------------------------------
        # Coordinates relative to BH
        # ----------------------------------------------------------

        xl = (
            x
            - xc
        )

        yl = y
        zl = z

        rl = np.sqrt(
            xl * xl
            + yl * yl
            + zl * zl
        )

        theta_l, phi_l = angles_for_axis(
            xl,
            yl,
            zl,
            basis
        )

        radial = hydrogenic_R(
            rl,
            n,
            l,
            mu,
            M
        )

        angular = Ylm(
            l,
            m,
            theta_l,
            phi_l
        )

        # ----------------------------------------------------------
        # Comoving boost
        # ----------------------------------------------------------

        boost = np.exp(
            1j
            * mu
            * vy
            * y
        )

        psi = (
            radial
            * angular
            * boost
        )

        # ----------------------------------------------------------
        # Projection
        #
        # c_lm(r)
        #   = int dOmega Y_lm^* psi
        #
        # U_lm(r)
        #   = r c_lm(r)
        # ----------------------------------------------------------

        coeff = (

            (
                psi
                * w[
                    None,
                    :
                ]
            )

            @ np.conjugate(
                Ybasis
            ).T
        )

        U[
            :,
            i0:i1
        ] = (

            rr[
                :,
                0,
                None
            ]

            * coeff

        ).T

    # Same Dirichlet endpoints as stationary matrix.

    U[
        :,
        0
    ] = 0.0

    U[
        :,
        -1
    ] = 0.0

    return U


# ======================================================================
# NORMS / VECTOR CONVENTIONS
# ======================================================================

def radial_norm(
    U,
    r
):

    density = np.sum(
        np.abs(U)**2,
        axis=0
    )

    return float(
        np.trapezoid(
            density,
            r
        )
    )


def ivec(
    U
):
    """
    Interior points, mode-major ordering.
    """

    return (
        U[
            :,
            1:-1
        ]
        .reshape(-1)
    )


def normalize(
    v
):

    n = np.linalg.norm(
        v
    )

    if n == 0.0:

        raise ValueError(
            "Attempted to normalize "
            "a zero vector."
        )

    return (
        v / n
    )


# ======================================================================
# ASYMPTOTIC ATOMIC STATES
# ======================================================================

def atomic_state(
    R,
    p,
    modes,
    r,
    which
):

    Om = omega(
        R,
        p
    )

    x1, x2 = positions(
        R,
        p
    )

    if which == "primary":

        M = p["M1"]
        xc = x1

        n = p["n1"]
        l = p["l1"]
        m = p["m1"]

    elif which == "secondary":

        M = p["M2"]
        xc = x2

        n = p["n2"]
        l = p["l2"]
        m = p["m2"]

    else:

        raise ValueError(
            "which must be "
            "'primary' or 'secondary'."
        )

    U = project_atom(
        r,
        modes,
        p["basis"],
        p["mu"],
        M,
        xc,
        Om,
        n,
        l,
        m,
        p["ntheta"],
        p["nphi"]
    )

    capture = radial_norm(
        U,
        r
    )

    v = normalize(
        ivec(
            U
        )
    )

    return (
        U,
        v,
        capture
    )


def resolve_initial_mode(
    p
):
    """
    Default prescription:

        q < 1       -> primary
        q ~= 1      -> coherent_pair
    """

    mode = p[
        "initial_state"
    ]

    if mode == "auto":

        if (
            abs(
                1.0
                - p["q"]
            )
            <= p[
                "q_equal_tol"
            ]
        ):

            return "coherent_pair"

        return "primary"

    allowed = {
        "primary",
        "secondary",
        "coherent_pair"
    }

    if mode not in allowed:

        raise ValueError(
            "molecular_init.initial_state "
            "must be 'auto', 'primary', "
            "'secondary', or 'coherent_pair'."
        )

    return mode


def make_asymptotic_seed(
    R,
    p,
    modes,
    r,
    mode
):

    U1, v1, cap1 = atomic_state(
        R,
        p,
        modes,
        r,
        "primary"
    )

    info = {

        "cap1":
            cap1,

        "cap2":
            np.nan,

        "atomic_overlap":
            np.nan,

        "seed_norm2":
            1.0,
    }

    if mode == "primary":

        return (
            v1,
            info
        )

    U2, v2, cap2 = atomic_state(
        R,
        p,
        modes,
        r,
        "secondary"
    )

    s12 = np.vdot(
        v1,
        v2
    )

    info[
        "cap2"
    ] = cap2

    info[
        "atomic_overlap"
    ] = abs(
        s12
    )

    if mode == "secondary":

        return (
            v2,
            info
        )

    # --------------------------------------------------------------
    # Coherent pair
    # --------------------------------------------------------------

    c2 = (

        p["amp2"]

        * np.exp(
            1j
            * p[
                "relative_phase"
            ]
        )
    )

    seed_raw = (
        v1
        + c2
        * v2
    )

    info[
        "seed_norm2"
    ] = float(
        np.vdot(
            seed_raw,
            seed_raw
        ).real
    )

    return (
        normalize(
            seed_raw
        ),
        info
    )


# ======================================================================
# ASYMPTOTIC DIFFERENT-n RESONANCE WARNING
# ======================================================================

def report_asymptotic_resonances(
    p
):
    """
    Look for secondary n levels with isolated binding energy close
    to the initial primary n1 state.

    Example:

        q ~ 1/2

    gives approximately

        primary n=2
            <->
        secondary n=1.
    """

    E1 = isolated_energy(
        p["mu"],
        p["M1"],
        p["n1"]
    )

    found = []

    for n in range(
        1,
        p[
            "resonance_nmax"
        ] + 1
    ):

        E2 = isolated_energy(
            p["mu"],
            p["M2"],
            n
        )

        rel = (

            abs(
                E2
                - E1
            )

            / max(
                abs(E1),
                abs(E2),
                1.0e-30
            )
        )

        if (
            rel
            < p[
                "resonance_detuning_tol"
            ]
        ):

            found.append(
                (
                    n,
                    E2,
                    rel
                )
            )

    if found:

        print(
            "\nWARNING: possible asymptotic "
            "inter-centre hydrogenic degeneracy:"
        )

        print(
            f"  primary n={p['n1']}: "
            f"E={E1:.12e}"
        )

        for (
            n,
            E2,
            rel
        ) in found:

            print(
                f"  secondary n={n}: "
                f"E={E2:.12e}   "
                f"relative detuning="
                f"{rel:.3e}"
            )

        print(
            "If the asymptotic one-eigenstate "
            "criterion fails here, the failure may "
            "be physical. A degenerate/multibranch "
            "continuation should then be used."
        )

    return found


# ======================================================================
# ANGULAR MATRIX ELEMENTS
# ======================================================================

def gaunt_star(
    l,
    m,
    L,
    M,
    lp,
    mp
):
    """
    Integral

        int dOmega
        Y_lm^*
        Y_LM
        Y_l'm'
    """

    if (
        -m
        + M
        + mp
        != 0
    ):

        return 0.0

    if abs(M) > L:

        return 0.0

    a = float(
        wigner_3j(
            l,
            L,
            lp,
            0,
            0,
            0
        )
    )

    b = float(
        wigner_3j(
            l,
            L,
            lp,
            -m,
            M,
            mp
        )
    )

    if (
        a == 0.0
        or
        b == 0.0
    ):

        return 0.0

    return (

        (-1.0)**m

        * math.sqrt(

            (
                (2*l + 1)
                * (2*L + 1)
                * (2*lp + 1)
            )

            / (
                4.0
                * np.pi
            )
        )

        * a
        * b
    )


def C_source(
    modes,
    L,
    theta,
    phi
):
    """
    Angular matrix for the L-th multipole of a source at
    angular position (theta,phi).
    """

    C = np.zeros(
        (
            len(modes),
            len(modes)
        ),
        dtype=complex
    )

    for i, (
        l,
        m
    ) in enumerate(
        modes
    ):

        for j, (
            lp,
            mp
        ) in enumerate(
            modes
        ):

            M = (
                m
                - mp
            )

            if abs(M) > L:

                continue

            g = gaunt_star(
                l,
                m,
                L,
                M,
                lp,
                mp
            )

            if g == 0.0:

                continue

            C[
                i,
                j
            ] = (

                np.conjugate(

                    Ylm(
                        L,
                        M,
                        theta,
                        phi
                    )
                )

                * g
            )

    # Suppress tiny numerical non-Hermiticity.

    return (

        0.5

        * (
            C
            + C.conjugate().T
        )
    )


# ======================================================================
# L_z IN z- OR x-QUANTIZED BASIS
# ======================================================================

def Lz_modes(
    modes,
    basis
):

    if basis == "z":

        return sp.diags(
            [
                m
                for l, m
                in modes
            ],
            0,
            format="csr",
            dtype=complex
        )

    # --------------------------------------------------------------
    # x quantization
    #
    # L_z
    #   = (L_{x,+} - L_{x,-})/(2i)
    # --------------------------------------------------------------

    idx = {
        lm: i
        for i, lm
        in enumerate(
            modes
        )
    }

    rows = []
    cols = []
    vals = []

    for j, (
        l,
        m
    ) in enumerate(
        modes
    ):

        # ----------------------------------------------------------
        # m_x -> m_x + 1
        # ----------------------------------------------------------

        if (
            l,
            m + 1
        ) in idx:

            c = math.sqrt(
                l * (
                    l + 1
                )
                - m * (
                    m + 1
                )
            )

            rows.append(
                idx[
                    (
                        l,
                        m + 1
                    )
                ]
            )

            cols.append(
                j
            )

            vals.append(
                c
                / (
                    2.0j
                )
            )

        # ----------------------------------------------------------
        # m_x -> m_x - 1
        # ----------------------------------------------------------

        if (
            l,
            m - 1
        ) in idx:

            c = math.sqrt(
                l * (
                    l + 1
                )
                - m * (
                    m - 1
                )
            )

            rows.append(
                idx[
                    (
                        l,
                        m - 1
                    )
                ]
            )

            cols.append(
                j
            )

            vals.append(
                -c
                / (
                    2.0j
                )
            )

    A = sp.coo_matrix(
        (
            np.asarray(
                vals,
                dtype=complex
            ),
            (
                rows,
                cols
            )
        ),
        shape=(
            len(modes),
            len(modes)
        )
    ).tocsr()

    return (

        0.5

        * (
            A
            + A.getH()
        )
    )


def source_angles(
    sign,
    basis
):
    """
    Angular location of +/-x in the chosen angular system.
    """

    if basis == "z":

        if sign > 0:

            return (
                np.pi / 2.0,
                0.0
            )

        return (
            np.pi / 2.0,
            np.pi
        )

    # x quantization:
    #
    # +x = north pole
    # -x = south pole

    if sign > 0:

        return (
            0.0,
            0.0
        )

    return (
        np.pi,
        0.0
    )


# ======================================================================
# FULL TWO-CENTRE MOLECULAR HAMILTONIAN
# ======================================================================

class Hamiltonian:

    def __init__(
        self,
        p,
        modes,
        r
    ):

        self.p = p
        self.modes = modes
        self.r = r

        self.ri = r[
            1:-1
        ]

        self.nr = len(
            self.ri
        )

        self.nm = len(
            modes
        )

        dr = (
            r[1]
            - r[0]
        )

        # ----------------------------------------------------------
        # Radial second derivative
        # ----------------------------------------------------------

        D2 = sp.diags(
            [
                np.ones(
                    self.nr - 1
                ),

                -2.0
                * np.ones(
                    self.nr
                ),

                np.ones(
                    self.nr - 1
                )
            ],
            [
                -1,
                0,
                +1
            ],
            format="csr"
        ) / dr**2

        T = (

            -D2

            / (
                2.0
                * p["mu"]
            )
        )

        Im = sp.eye(
            self.nm,
            format="csr",
            dtype=complex
        )

        self.Ir = sp.eye(
            self.nr,
            format="csr",
            dtype=complex
        )

        # ----------------------------------------------------------
        # Centrifugal term
        # ----------------------------------------------------------

        cent = sp.diags(
            [
                l * (
                    l + 1
                )
                / (
                    2.0
                    * p["mu"]
                )
                for l, m
                in modes
            ],
            0,
            format="csr"
        )

        invr2 = sp.diags(
            1.0
            / self.ri**2,
            0,
            format="csr"
        )

        self.H0 = (

            sp.kron(
                Im,
                T
            )

            + sp.kron(
                cent,
                invr2
            )

        ).astype(
            complex
        )

        # ----------------------------------------------------------
        # Rotation generator
        # ----------------------------------------------------------

        self.Lz = Lz_modes(
            modes,
            p["basis"]
        )

        # ----------------------------------------------------------
        # Potential multipoles
        # ----------------------------------------------------------

        Lmax = min(
            p["lammax"],
            2 * p["lmax"]
        )

        self.Ls = list(
            range(
                Lmax + 1
            )
        )

        th1, ph1 = source_angles(
            +1,
            p["basis"]
        )

        th2, ph2 = source_angles(
            -1,
            p["basis"]
        )

        print(
            f"Precomputing potential angular matrices "
            f"for lambda=0..{Lmax} ..."
        )

        self.C1 = [
            sp.csr_matrix(
                C_source(
                    modes,
                    L,
                    th1,
                    ph1
                )
            )
            for L
            in self.Ls
        ]

        self.C2 = [
            sp.csr_matrix(
                C_source(
                    modes,
                    L,
                    th2,
                    ph2
                )
            )
            for L
            in self.Ls
        ]


    def fL(
        self,
        a,
        L
    ):
        """
        Multipole radial factor

            r_<^L / r_>^(L+1).
        """

        a = abs(
            a
        )

        rmin = np.minimum(
            self.ri,
            a
        )

        rmax = np.maximum(
            self.ri,
            a
        )

        return (
            rmin**L
            / rmax**(
                L + 1
            )
        )


    def build(
        self,
        R
    ):

        p = self.p

        x1, x2 = positions(
            R,
            p
        )

        H = self.H0.copy()

        # ----------------------------------------------------------
        # Two-centre Newtonian potential
        #
        # 1/|r-a|
        #
        # = 4pi sum_LM
        #
        #     1/(2L+1)
        #     r_<^L/r_>^(L+1)
        #     Y_LM^*(Omega_a)
        #     Y_LM(Omega)
        # ----------------------------------------------------------

        for L in self.Ls:

            fac = (
                4.0
                * np.pi
                / (
                    2 * L
                    + 1
                )
            )

            D1 = sp.diags(
                (
                    -p["mu"]
                    * p["M1"]
                    * fac
                    * self.fL(
                        x1,
                        L
                    )
                ),
                0,
                format="csr"
            )

            D2 = sp.diags(
                (
                    -p["mu"]
                    * p["M2"]
                    * fac
                    * self.fL(
                        x2,
                        L
                    )
                ),
                0,
                format="csr"
            )

            H += sp.kron(
                self.C1[L],
                D1,
                format="csr"
            )

            H += sp.kron(
                self.C2[L],
                D2,
                format="csr"
            )

        # ----------------------------------------------------------
        # Rotating-frame Hamiltonian
        #
        # H_rot = H - Omega L_z
        # ----------------------------------------------------------

        Om = omega(
            R,
            p
        )

        H -= (

            Om

            * sp.kron(
                self.Lz,
                self.Ir,
                format="csr"
            )
        )

        # Remove tiny floating-point non-Hermiticity.

        H = (

            0.5

            * (
                H
                + H.getH()
            )
        )

        return (
            H.tocsr(),
            Om
        )


# ======================================================================
# EIGENSOLVER / BRANCH SELECTION
# ======================================================================

def rayleigh(
    H,
    v
):

    v = normalize(
        v
    )

    return float(
        np.real(
            np.vdot(
                v,
                H @ v
            )
        )
    )


def eig_near(
    H,
    sigma,
    nev,
    tol=1.0e-9
):
    """
    Shift-invert eigensolve near sigma.
    """

    values, vectors = sla.eigsh(
        H,
        k=nev,
        sigma=sigma,
        which="LM",
        tol=tol,
        maxiter=4000
    )

    order = np.argsort(
        values.real
    )

    return (
        values.real[
            order
        ],
        vectors[
            :,
            order
        ]
    )


def choose(
    vectors,
    reference
):
    """
    Choose the eigenvector having maximum overlap with reference.

    Its arbitrary complex phase is then fixed so that

        <reference|state>

    is real and positive.
    """

    reference = normalize(
        reference
    )

    overlaps = (

        np.abs(

            vectors
            .conjugate()
            .T
            @ reference

        )**2
    )

    j = int(
        np.argmax(
            overlaps
        )
    )

    v = vectors[
        :,
        j
    ]

    ov_complex = np.vdot(
        reference,
        v
    )

    if abs(
        ov_complex
    ) > 0.0:

        v *= np.exp(
            -1j
            * np.angle(
                ov_complex
            )
        )

    return (
        j,
        normalize(v),
        float(
            overlaps[j]
        )
    )


def residual(
    H,
    v,
    E
):

    Hv = (
        H @ v
    )

    denom = np.linalg.norm(
        Hv
    )

    if denom == 0.0:

        return np.inf

    return float(

        np.linalg.norm(
            Hv
            - E * v
        )

        / denom
    )


def solve_near_reference(
    H,
    sigma,
    reference,
    p,
    overlap_goal=None
):
    """
    Find the eigenstate near sigma with largest overlap with reference.

    If necessary, enlarge the local spectral window from `nev`
    up to `nev_max`.
    """

    dim = H.shape[0]

    kmax_allowed = (
        dim - 2
    )

    if kmax_allowed < 2:

        raise RuntimeError(
            "Hamiltonian is too small "
            "for eigsh."
        )

    k = min(
        max(
            2,
            p["nev"]
        ),
        kmax_allowed
    )

    kmax = min(
        p["nev_max"],
        kmax_allowed
    )

    if overlap_goal is None:

        overlap_goal = p[
            "min_step_overlap"
        ]

    best = None

    while True:

        values, vectors = eig_near(
            H,
            sigma,
            k
        )

        j, state, ov = choose(
            vectors,
            reference
        )

        E = float(
            values[j]
        )

        gaps = np.abs(
            values
            - E
        )

        gaps[j] = np.inf

        spectral_gap = float(
            np.min(
                gaps
            )
        )

        candidate = {

            "state":
                state,

            "E":
                E,

            "overlap":
                ov,

            "gap":
                spectral_gap,

            "nev_used":
                k,

            "values":
                values,
        }

        if (
            best is None
            or
            ov
            > best["overlap"]
        ):

            best = candidate

        if ov >= overlap_goal:

            return candidate

        if k >= kmax:

            return best

        knew = min(
            kmax,
            max(
                k + 4,
                2 * k
            )
        )

        if knew == k:

            return best

        k = knew


# ======================================================================
# ASYMPTOTIC STARTING POINT
# ======================================================================

def asymptotic_trial(
    R,
    HH,
    p,
    modes,
    r,
    mode
):

    seed, seed_info = make_asymptotic_seed(
        R,
        p,
        modes,
        r,
        mode
    )

    H, Om = HH.build(
        R
    )

    sigma = rayleigh(
        H,
        seed
    )

    sol = solve_near_reference(
        H,
        sigma,
        seed,
        p,
        overlap_goal=p[
            "asym_eig_overlap"
        ]
    )

    sol[
        "residual"
    ] = residual(
        H,
        sol["state"],
        sol["E"]
    )

    sol[
        "Omega"
    ] = Om

    sol[
        "seed"
    ] = seed

    sol[
        "seed_info"
    ] = seed_info

    return sol


def find_R_asym_refined(
    HH,
    p,
    modes,
    r,
    mode
):
    """
    Determine R_asym using the physically relevant condition

        |<atomic seed | molecular eigenstate>|^2 -> 1.

    For q<1 this is stronger and more meaningful than simply demanding
    negligible geometric overlap between two atomic tails.
    """

    # --------------------------------------------------------------
    # User-specified R_asym
    # --------------------------------------------------------------

    if p["R_asym"] is not None:

        R = float(
            p["R_asym"]
        )

        if R <= p["R0"]:

            raise ValueError(
                "molecular_init.R_asym "
                "must exceed binary.R."
            )

        sol = asymptotic_trial(
            R,
            HH,
            p,
            modes,
            r,
            mode
        )

        info = sol[
            "seed_info"
        ]

        print(
            f"[asym] "
            f"R={R:.8g}  "
            f"capture1={info['cap1']:.6f}  "
            f"eig-overlap={sol['overlap']:.8f}  "
            f"gap={sol['gap']:.4e}  "
            f"res={sol['residual']:.2e}"
        )

        if (
            info["cap1"]
            < p["min_capture"]
        ):

            raise RuntimeError(
                "The primary asymptotic atom "
                "is not accurately represented "
                "by the present radial/angular basis."
            )

        if (
            mode
            == "coherent_pair"
            and
            info["cap2"]
            < p["min_capture"]
        ):

            raise RuntimeError(
                "The secondary asymptotic atom "
                "is not accurately represented "
                "by the present radial/angular basis."
            )

        if (
            sol["overlap"]
            < p[
                "asym_eig_overlap"
            ]
        ):

            raise RuntimeError(
                "\nThe chosen asymptotic state is not "
                "dominated by one eigenstate of the full "
                "binary Hamiltonian.\n\n"
                "Possible reasons:\n"
                "  1. R_asym is not large enough;\n"
                "  2. the system lies near an accidental "
                "primary-secondary atomic resonance;\n"
                "  3. nev/nev_max is too small;\n"
                "  4. near q=1 the asymptotic splitting is "
                "numerically unresolved.\n\n"
                "Do not arbitrarily choose one eigenvector "
                "in this situation."
            )

        return (
            R,
            sol
        )

    # --------------------------------------------------------------
    # Automatic R_asym
    # --------------------------------------------------------------

    rmean1 = mean_radius(
        p["n1"],
        p["l1"],
        p["mu"],
        p["M1"]
    )

    R = max(
        1.5
        * p["R0"],
        3.0
        * rmean1
    )

    for k in range(
        p[
            "max_asym_tries"
        ]
    ):

        sol = asymptotic_trial(
            R,
            HH,
            p,
            modes,
            r,
            mode
        )

        info = sol[
            "seed_info"
        ]

        print(
            f"[asym {k+1:02d}] "
            f"R={R:.8g}  "
            f"capture1={info['cap1']:.6f}  "
            f"eig-overlap={sol['overlap']:.8f}  "
            f"gap={sol['gap']:.4e}  "
            f"res={sol['residual']:.2e}"
        )

        # ----------------------------------------------------------
        # Increasing R moves the atom farther from the COM origin.
        # If its basis capture is already poor, increasing R is not
        # a useful solution.
        # ----------------------------------------------------------

        if (
            info["cap1"]
            < p[
                "min_capture"
            ]
        ):

            raise RuntimeError(
                "\nThe primary atom cannot be "
                "represented accurately at the required "
                "asymptotic separation.\n"
                "Increase grid.L and/or angular resolution "
                "(l_max and, for x quantization, m_cut)."
            )

        if (
            mode
            == "coherent_pair"
            and
            info["cap2"]
            < p[
                "min_capture"
            ]
        ):

            raise RuntimeError(
                "\nThe secondary atom cannot be "
                "represented accurately at the required "
                "asymptotic separation.\n"
                "Increase grid.L and/or angular resolution."
            )

        if (
            sol["overlap"]
            >= p[
                "asym_eig_overlap"
            ]
        ):

            return (
                R,
                sol
            )

        R *= p[
            "R_growth"
        ]

    raise RuntimeError(
        "\nCould not reach a clean one-eigenbranch "
        "asymptotic limit.\n"
        "If the spectral gap is becoming extremely small, "
        "this may be a genuine degenerate or near-degenerate "
        "atomic subspace. Use multibranch adiabatic transport."
    )


# ======================================================================
# LOCALIZATION / MOLECULARITY
# ======================================================================

def localization_diagnostics(
    R,
    state,
    p,
    modes,
    r
):
    """
    Project instantaneous molecular state onto localized primary and
    secondary atomic reference states.

    Since these are not generally orthogonal, use the symmetric
    Lowdin-orthogonalized two-state basis.

    Returns

        P_primary
        P_secondary

        two_state_capture =
            P_primary + P_secondary

        molecularity =
            4 P_primary P_secondary
            / (P_primary + P_secondary)^2.
    """

    nan_result = {

        "P_primary":
            np.nan,

        "P_secondary":
            np.nan,

        "two_state_capture":
            np.nan,

        "molecularity":
            np.nan,

        "atomic_overlap":
            np.nan,

        "atom_capture_primary":
            np.nan,

        "atom_capture_secondary":
            np.nan,
    }

    if not p["diagnostics"]:

        return nan_result

    _, v1, cap1 = atomic_state(
        R,
        p,
        modes,
        r,
        "primary"
    )

    _, v2, cap2 = atomic_state(
        R,
        p,
        modes,
        r,
        "secondary"
    )

    result = dict(
        nan_result
    )

    result[
        "atom_capture_primary"
    ] = cap1

    result[
        "atom_capture_secondary"
    ] = cap2

    if (
        cap1
        < p[
            "diagnostic_min_capture"
        ]
        or
        cap2
        < p[
            "diagnostic_min_capture"
        ]
    ):

        return result

    s = np.vdot(
        v1,
        v2
    )

    result[
        "atomic_overlap"
    ] = abs(
        s
    )

    # --------------------------------------------------------------
    # Overlap matrix of non-orthogonal localized site states
    # --------------------------------------------------------------

    S = np.array(
        [
            [
                1.0,
                s
            ],
            [
                np.conjugate(
                    s
                ),
                1.0
            ]
        ],
        dtype=complex
    )

    lam, Q = np.linalg.eigh(
        S
    )

    if np.min(
        lam
    ) < 1.0e-10:

        return result

    Sinvhalf = (

        Q

        @ np.diag(
            1.0
            / np.sqrt(
                lam
            )
        )

        @ Q.conjugate().T
    )

    # --------------------------------------------------------------
    # Non-orthogonal projections
    # --------------------------------------------------------------

    b = np.array(
        [
            np.vdot(
                v1,
                state
            ),
            np.vdot(
                v2,
                state
            )
        ],
        dtype=complex
    )

    # --------------------------------------------------------------
    # Amplitudes in Lowdin-orthogonalized basis
    # --------------------------------------------------------------

    a = (
        Sinvhalf
        @ b
    )

    P1 = float(
        abs(
            a[0]
        )**2
    )

    P2 = float(
        abs(
            a[1]
        )**2
    )

    Ptot = (
        P1
        + P2
    )

    result[
        "P_primary"
    ] = P1

    result[
        "P_secondary"
    ] = P2

    result[
        "two_state_capture"
    ] = Ptot

    if Ptot > 0.0:

        result[
            "molecularity"
        ] = (

            4.0
            * P1
            * P2

            / Ptot**2
        )

    return result


# ======================================================================
# ADAPTIVE CONTINUATION
# ======================================================================

def make_history_record(
    R,
    Om,
    E,
    overlap,
    res,
    gap,
    nev_used,
    state,
    p,
    modes,
    r
):

    rec = {

        "R":
            float(R),

        "Omega":
            float(Om),

        "E_rot":
            float(E),

        "continuation_overlap":
            float(overlap),

        "residual":
            float(res),

        "spectral_gap":
            float(gap),

        "nev_used":
            int(nev_used),
    }

    rec.update(

        localization_diagnostics(
            R,
            state,
            p,
            modes,
            r
        )
    )

    return rec


def advance_segment(
    HH,
    Ra,
    Rb,
    state_a,
    E_a,
    p,
    modes,
    r,
    history,
    depth=0
):
    """
    Try Ra -> Rb.

    If overlap drops below min_step_overlap, insert geometric midpoint

        Rmid = sqrt(Ra Rb)

    and recursively continue through the finer grid.
    """

    H, Om = HH.build(
        Rb
    )

    sol = solve_near_reference(
        H,
        E_a,
        state_a,
        p,
        overlap_goal=p[
            "min_step_overlap"
        ]
    )

    ov = sol[
        "overlap"
    ]

    # --------------------------------------------------------------
    # Refine in R if necessary
    # --------------------------------------------------------------

    if (
        ov
        < p[
            "min_step_overlap"
        ]
    ):

        if (
            depth
            < p[
                "max_refinements"
            ]
        ):

            Rmid = math.sqrt(
                Ra
                * Rb
            )

            if (
                abs(
                    Rmid
                    - Ra
                )
                / Ra
                > 1.0e-12

                and

                abs(
                    Rmid
                    - Rb
                )
                / Rb
                > 1.0e-12
            ):

                state_mid, E_mid = advance_segment(
                    HH,
                    Ra,
                    Rmid,
                    state_a,
                    E_a,
                    p,
                    modes,
                    r,
                    history,
                    depth=depth + 1
                )

                return advance_segment(
                    HH,
                    Rmid,
                    Rb,
                    state_mid,
                    E_mid,
                    p,
                    modes,
                    r,
                    history,
                    depth=depth + 1
                )

        msg = (
            "\nUnable to follow a unique instantaneous "
            "eigenbranch.\n"
            f"At R={Rb:.10g}, maximum overlap with "
            f"the previous state is only {ov:.6f}.\n"
            f"Local spectral gap = "
            f"{sol['gap']:.6e}.\n\n"
            "This may indicate a genuine near-degeneracy "
            "or avoided crossing requiring multibranch "
            "treatment."
        )

        if p[
            "abort_on_low_overlap"
        ]:

            raise RuntimeError(
                msg
            )

        print(
            "WARNING:",
            msg
        )

    state_b = sol[
        "state"
    ]

    E_b = sol[
        "E"
    ]

    res = residual(
        H,
        state_b,
        E_b
    )

    rec = make_history_record(
        Rb,
        Om,
        E_b,
        ov,
        res,
        sol["gap"],
        sol["nev_used"],
        state_b,
        p,
        modes,
        r
    )

    history.append(
        rec
    )

    print(
        f"R={Rb:11.6f}  "
        f"E={E_b:+.12e}  "
        f"ov={ov:.8f}  "
        f"gap={sol['gap']:.3e}  "
        f"M={rec['molecularity']:.5f}  "
        f"2state={rec['two_state_capture']:.5f}"
    )

    return (
        state_b,
        E_b
    )


# ======================================================================
# VECTOR -> MODES
# ======================================================================

def to_modes(
    v,
    nm,
    N
):

    U = np.zeros(
        (
            nm,
            N
        ),
        dtype=complex
    )

    U[
        :,
        1:-1
    ] = v.reshape(
        nm,
        N - 2
    )

    return U


# ======================================================================
# MODESBIN
# ======================================================================

def save_modesbin(
    path,
    modes,
    U
):
    """
    Format:

        8 bytes:
            MODESBIN

        int32:
            version

        int64:
            Nr

        int64:
            Nmodes

        for each mode:
            int32 l
            int32 m

            Nr float64:
                real(U)

            Nr float64:
                imag(U)
    """

    order = sorted(
        range(
            len(modes)
        ),
        key=lambda i:
            modes[i]
    )

    with open(
        path,
        "wb"
    ) as f:

        f.write(
            b"MODESBIN"
        )

        f.write(
            struct.pack(
                "<i",
                1
            )
        )

        f.write(
            struct.pack(
                "<q",
                U.shape[1]
            )
        )

        f.write(
            struct.pack(
                "<q",
                len(modes)
            )
        )

        for i in order:

            l, m = modes[i]

            a = np.asarray(
                U[i],
                dtype=complex
            )

            f.write(
                struct.pack(
                    "<ii",
                    l,
                    m
                )
            )

            f.write(
                np.asarray(
                    a.real,
                    dtype="<f8"
                ).tobytes()
            )

            f.write(
                np.asarray(
                    a.imag,
                    dtype="<f8"
                ).tobytes()
            )


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Construct a molecular initial state "
            "by adiabatic continuation from a clean "
            "asymptotic atomic configuration."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Input config.toml"
    )

    parser.add_argument(
        "--output",
        default="initial_modes.bin",
        help="Output MODESBIN file"
    )

    parser.add_argument(
        "--R-asym",
        type=float,
        default=None,
        help=(
            "Override molecular_init.R_asym"
        )
    )

    parser.add_argument(
        "--relative-phase",
        type=float,
        default=None,
        help=(
            "Override coherent-pair relative phase"
        )
    )

    parser.add_argument(
        "--continuation-steps",
        type=int,
        default=None,
        help=(
            "Override molecular_init.continuation_steps"
        )
    )

    parser.add_argument(
        "--nev",
        type=int,
        default=None,
        help=(
            "Override molecular_init.nev"
        )
    )

    parser.add_argument(
        "--initial-state",
        choices=[
            "auto",
            "primary",
            "secondary",
            "coherent_pair"
        ],
        default=None,
        help=(
            "Override molecular_init.initial_state"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Config
    # --------------------------------------------------------------

    p = load_config(
        args.config
    )

    if (
        args.relative_phase
        is not None
    ):

        p[
            "relative_phase"
        ] = (
            args.relative_phase
        )

    if (
        args.continuation_steps
        is not None
    ):

        p[
            "ncont"
        ] = (
            args.continuation_steps
        )

    if (
        args.nev
        is not None
    ):

        p[
            "nev"
        ] = args.nev

    if (
        args.R_asym
        is not None
    ):

        p[
            "R_asym"
        ] = (
            args.R_asym
        )

    if (
        args.initial_state
        is not None
    ):

        p[
            "initial_state"
        ] = (
            args.initial_state
        )

    modes = modes_list(
        p
    )

    # --------------------------------------------------------------
    # Radial grid
    # --------------------------------------------------------------

    r = np.linspace(
        0.0,
        p["L"],
        p["N"]
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "MOLECULAR INITIAL-STATE CONSTRUCTION"
    )

    print(
        "============================================================"
    )

    print(
        f"N={p['N']}  "
        f"L={p['L']}  "
        f"M1={p['M1']}  "
        f"M2={p['M2']}  "
        f"q={p['q']}  "
        f"R0={p['R0']}  "
        f"mu={p['mu']}"
    )

    print(
        f"basis={p['basis']}  "
        f"Nmodes={len(modes)}  "
        f"lmax={p['lmax']}  "
        f"m_cut="
        f"{p['mcut'] if p['basis']=='x' else '-'}  "
        f"lambda_max={p['lammax']}"
    )

    print(
        f"primary atom: "
        f"(n,l,m)="
        f"({p['n1']},{p['l1']},{p['m1']})"
    )

    print(
        f"secondary reference atom: "
        f"(n,l,m)="
        f"({p['n2']},{p['l2']},{p['m2']})"
    )

    # --------------------------------------------------------------
    # Initialization prescription
    # --------------------------------------------------------------

    mode = resolve_initial_mode(
        p
    )

    print(
        f"\nInitialization mode: "
        f"{mode}"
    )

    if mode == "primary":

        print(
            "Using one asymptotic atom "
            "localized around the primary."
        )

    elif mode == "secondary":

        print(
            "Using one asymptotic atom "
            "localized around the secondary."
        )

    else:

        print(
            "Using coherent two-atom "
            "asymptotic state:"
        )

        print(
            f"    psi ~ atom_1 "
            f"+ {p['amp2']} "
            f"exp(i {p['relative_phase']}) "
            f"atom_2"
        )

    # --------------------------------------------------------------
    # Different-n resonance warning
    # --------------------------------------------------------------

    report_asymptotic_resonances(
        p
    )

    # --------------------------------------------------------------
    # Hamiltonian
    # --------------------------------------------------------------

    print(
        "\nBuilding molecular Hamiltonian machinery ..."
    )

    HH = Hamiltonian(
        p,
        modes,
        r
    )

    # --------------------------------------------------------------
    # Find R_asym
    # --------------------------------------------------------------

    print(
        "\nSearching for a clean "
        "asymptotic starting point ..."
    )

    Rasym, asym_sol = find_R_asym_refined(
        HH,
        p,
        modes,
        r,
        mode
    )

    state = asym_sol[
        "state"
    ]

    E = asym_sol[
        "E"
    ]

    seed_info = asym_sol[
        "seed_info"
    ]

    print(
        f"\nSelected R_asym = "
        f"{Rasym:.10g}"
    )

    print(
        f"Asymptotic eigenstate overlap = "
        f"{asym_sol['overlap']:.10f}"
    )

    print(
        f"Asymptotic spectral gap = "
        f"{asym_sol['gap']:.8e}"
    )

    print(
        f"Asymptotic eigensolver residual = "
        f"{asym_sol['residual']:.8e}"
    )

    if np.isfinite(
        seed_info[
            "atomic_overlap"
        ]
    ):

        print(
            f"Asymptotic direct atomic overlap = "
            f"{seed_info['atomic_overlap']:.8e}"
        )

    # --------------------------------------------------------------
    # Initial diagnostics
    # --------------------------------------------------------------

    history = [
        make_history_record(
            Rasym,
            asym_sol["Omega"],
            E,
            asym_sol["overlap"],
            asym_sol["residual"],
            asym_sol["gap"],
            asym_sol["nev_used"],
            state,
            p,
            modes,
            r
        )
    ]

    # --------------------------------------------------------------
    # Continuation
    # --------------------------------------------------------------

    print(
        "\nContinuing selected molecular "
        "branch down to R0 ..."
    )

    Rs_base = np.geomspace(
        Rasym,
        p["R0"],
        p["ncont"]
    )

    Rprev = Rasym

    for Rnext in Rs_base[
        1:
    ]:

        state, E = advance_segment(
            HH,
            float(
                Rprev
            ),
            float(
                Rnext
            ),
            state,
            E,
            p,
            modes,
            r,
            history
        )

        Rprev = float(
            Rnext
        )

    # --------------------------------------------------------------
    # Final U_lm
    # --------------------------------------------------------------

    U = to_modes(
        state,
        len(modes),
        p["N"]
    )

    # --------------------------------------------------------------
    # Physical normalization
    #
    # rho = mu |psi|^2
    #
    # therefore
    #
    # M_cloud
    #   = mu int |psi|^2 d^3x
    #
    # and
    #
    # int dr sum_lm |U_lm|^2
    #   = M_cloud/mu.
    # --------------------------------------------------------------

    if p["mass_mode"] == "unit":

        target_norm = 1.0

    elif mode in {
        "primary",
        "secondary"
    }:

        # For q<1 this is the default:
        #
        # M_c is the cloud mass of the single
        # initially occupied atom.

        target_norm = (
            p["Mc"]
            / p["mu"]
        )

    elif mode == "coherent_pair":

        if (
            p["mass_mode"]
            == "per_atom"
        ):

            target_norm = (

                p["Mc"]
                / p["mu"]

                * seed_info[
                    "seed_norm2"
                ]
            )

        elif (
            p["mass_mode"]
            == "total"
        ):

            target_norm = (
                p["Mc"]
                / p["mu"]
            )

        else:

            raise ValueError(
                "For coherent_pair, mass_mode "
                "must be 'per_atom', 'total', "
                "or 'unit'."
            )

    else:

        raise RuntimeError(
            f"Unexpected initialization mode: "
            f"{mode}"
        )

    current_norm = radial_norm(
        U,
        r
    )

    U *= math.sqrt(
        target_norm
        / current_norm
    )

    # --------------------------------------------------------------
    # Save binary state
    # --------------------------------------------------------------

    save_modesbin(
        args.output,
        modes,
        U
    )

    # --------------------------------------------------------------
    # Save continuation diagnostics
    # --------------------------------------------------------------

    csvout = (
        str(
            Path(
                args.output
            ).with_suffix("")
        )
        + "_continuation.csv"
    )

    fieldnames = [

        "R",
        "Omega",
        "E_rot",

        "continuation_overlap",

        "residual",
        "spectral_gap",
        "nev_used",

        "P_primary",
        "P_secondary",

        "two_state_capture",
        "molecularity",

        "atomic_overlap",

        "atom_capture_primary",
        "atom_capture_secondary",
    ]

    with open(
        csvout,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in history:

            writer.writerow(
                row
            )

    # --------------------------------------------------------------
    # Final angular decomposition
    # --------------------------------------------------------------

    P = np.array(
        [
            np.trapezoid(
                np.abs(u)**2,
                r
            )
            for u
            in U
        ]
    )

    Ptot = np.sum(
        P
    )

    if Ptot > 0.0:

        P /= Ptot

    print(
        "\n"
        "============================================================"
    )

    print(
        "DONE"
    )

    print(
        "============================================================"
    )

    print(
        f"Wrote: "
        f"{args.output}"
    )

    print(
        f"Wrote: "
        f"{csvout}"
    )

    print(
        f"Final radial norm = "
        f"{radial_norm(U,r):.12e}"
    )

    print(
        f"Target radial norm = "
        f"{target_norm:.12e}"
    )

    print(
        "\nDominant final angular modes:"
    )

    for (
        l,
        m
    ), frac in sorted(

        zip(
            modes,
            P
        ),

        key=lambda x:
            x[1],

        reverse=True
    ):

        if frac > 1.0e-5:

            label = (
                "m_x"
                if p["basis"] == "x"
                else "m"
            )

            print(
                f"  "
                f"(l={l:2d}, "
                f"{label}={m:3d}) "
                f"{frac:.8e}"
            )

    # --------------------------------------------------------------
    # Final two-centre diagnostics
    # --------------------------------------------------------------

    last = history[
        -1
    ]

    if np.isfinite(
        last[
            "P_primary"
        ]
    ):

        print(
            "\nFinal two-site diagnostics:"
        )

        print(
            f"  P_primary         = "
            f"{last['P_primary']:.8e}"
        )

        print(
            f"  P_secondary       = "
            f"{last['P_secondary']:.8e}"
        )

        print(
            f"  two-state capture = "
            f"{last['two_state_capture']:.8e}"
        )

        print(
            f"  molecularity      = "
            f"{last['molecularity']:.8e}"
        )


if __name__ == "__main__":

    main()