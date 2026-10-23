#!/usr/bin/env python3
"""
Initial state with one or two hydrogenic gravitational atoms in the COM frame.

Each atom is the bound state (n, l, m) of its black hole, built at the origin,
rotated so that the displacement axis is z, translated along z by an exact
mode-by-mode projection, and rotated back. The primary atom is centred at
x = +R1 and, with --two-atoms, the secondary at x = -R2, at orbital phase zero.

The state is written in the MODESBIN format read by the C++ solver
(see include/InitialData.hpp).

Configuration keys read from the simulation TOML file:
    [grid]            N, L
    [binary]          q, R, M1 (default 1)
    [atom]            mu, n, l, m, amplitude (default 1)
    [atom_secondary]  n, l, m, amplitude (default: same as [atom])
    [simulation]      epsilon, l_max,
                      normalize_initial_state (default true), initial_norm (default 1)

Usage:
    python get_initial_state_COM.py --config config.toml --output initial_modes.bin [--two-atoms]
"""

import argparse
import struct

import numpy as np
import sympy as sp
from scipy.integrate import quad
from sympy.physics.wigner import wigner_d

from gravatom.context import Context
from gravatom.gravitational_atom import BoundState

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--config", required=True, help="Simulation config.toml")
parser.add_argument("--output", required=True, help="Output MODESBIN file")
parser.add_argument("--two-atoms", action="store_true",
                    help="Also place an atom around the secondary and superpose the two coherently.")

args = parser.parse_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)

N = config["grid"]["N"]
box_size = config["grid"]["L"]

q = config["binary"]["q"]
R = config["binary"]["R"]

# By default your units use the primary mass M1 = 1.
M1 = float(config["binary"].get("M1", 1.0))
M2 = q * M1
Mtot = M1 + M2

mu = config["atom"]["mu"]
r0 = config["simulation"]["epsilon"]
L_max = config["simulation"]["l_max"]

Omega = np.sqrt(Mtot / R**3)

# COM-frame distances.
R1 = M2 / Mtot * R
R2 = M1 / Mtot * R

# Positions at orbital phase zero:
# primary   -> +x
# secondary -> -x
theta_1 = np.pi / 2
phi_1 = 0.0

theta_2 = np.pi / 2
phi_2 = 0.0

print("Binary parameters")
print(f"  M1       = {M1}")
print(f"  M2       = {M2}")
print(f"  Mtot     = {Mtot}")
print(f"  q        = {q}")
print(f"  R1       = {R1}")
print(f"  R2       = {R2}")
print(f"  Omega    = {Omega}")

# ----------------------------------------------------------------------
# Separate gravitational-atom contexts
# ----------------------------------------------------------------------

context_1 = Context()
context_1(mu=mu, M=M1)
context_1(alpha=context_1.mu * context_1.M)
context_1(a=4 * context_1.alpha / (1 + 4 * context_1.alpha**2))

context_2 = Context()
context_2(mu=mu, M=M2)
context_2(alpha=context_2.mu * context_2.M)
context_2(a=4 * context_2.alpha / (1 + 4 * context_2.alpha**2))

print("Gravitational couplings")
print(f"  alpha_1 = M1 * mu = {M1 * mu}")
print(f"  alpha_2 = M2 * mu = {M2 * mu}")

# ----------------------------------------------------------------------
# Quantum numbers
# ----------------------------------------------------------------------

n1 = int(config["atom"]["n"])
l1 = int(config["atom"]["l"])
m1 = int(config["atom"]["m"])

# The secondary atom uses the same state unless an [atom_secondary]
# section is present in the TOML file.
secondary_config = config.get("atom_secondary", {})

n2 = int(secondary_config.get("n", n1))
l2 = int(secondary_config.get("l", l1))
m2 = int(secondary_config.get("m", m1))

bound_state_1 = BoundState(n1, l1, m1, context_1)
bound_state_2 = BoundState(n2, l2, m2, context_2)

epsilon_1 = (
    float(bound_state_1.eigenvalue.subs(context_1.get_subs())) - mu
)
epsilon_2 = (
    float(bound_state_2.eigenvalue.subs(context_2.get_subs())) - mu
)

print("Atomic states")
print(f"  Primary:   (n,l,m) = ({n1},{l1},{m1}), epsilon = {epsilon_1}")
print(f"  Secondary: (n,l,m) = ({n2},{l2},{m2}), epsilon = {epsilon_2}")

r_arr = np.linspace(0.0, box_size, N)
r_arr[0] = r0

modes = [
    (ell, emm)
    for ell in range(L_max + 1)
    for emm in range(-ell, ell + 1)
]

def integrate_complex(f, a, b, args=(), **quad_kwargs):
    """Integrate a complex-valued function of a real variable with scipy.quad."""
    re = quad(lambda x, *aa: np.real(f(x, *aa)), a, b, args=args, **quad_kwargs)[0]
    im = quad(lambda x, *aa: np.imag(f(x, *aa)), a, b, args=args, **quad_kwargs)[0]
    return re + 1j * im


def sph_harm_prefactor(l_, m_):
    """Normalisation of Y_lm, including the Condon-Shortley phase."""
    mabs = abs(m_)
    return (-1) ** mabs * sp.sqrt(
        (2 * l_ + 1) / (4 * sp.pi)
        * sp.factorial(l_ - mabs)
        / sp.factorial(l_ + mabs)
    )


def zero_mode_dict(mode_list, r_arr):
    """Return {(l, m): zeros} for every mode, on the grid r_arr."""
    return {mode: np.zeros_like(r_arr, dtype=complex) for mode in mode_list}


def wigner_D_element(l, mp, m, alpha, beta, gamma):
    """Wigner D-matrix element D^l_{mp, m}(alpha, beta, gamma)."""
    D = sp.N(wigner_d(sp.Integer(l), alpha, beta, gamma))
    m_vals = list(range(l, -l - 1, -1))
    row = m_vals.index(mp)
    col = m_vals.index(m)
    return complex(D[row, col])


def rotate_single_mode_to_aligned_full(U_0, theta_a, phi_a, inverse=False):
    """
    Rotate a single populated mode into the frame where the translation axis is aligned with z.

    Returns
    -------
    U_aligned : dict
        Rotated mode dictionary.
    coeffs : dict
        Rotation coefficients coeffs[(l, mp)] such that
        U_aligned[(l,mp)] = coeffs[(l,mp)] * original_radial
    radial0 : np.ndarray
        The original populated radial profile.
    """
    populated = [(lm, arr) for lm, arr in U_0.items() if np.any(np.abs(arr) > 0)]
    if len(populated) != 1:
        raise ValueError("U_0 must contain exactly one populated mode.")

    (l0, m0), radial0 = populated[0]
    radial0 = np.asarray(radial0, dtype=complex)

    alpha = float(phi_a)
    beta = float(theta_a)
    gamma = 0.0

    if inverse:
        alpha, beta, gamma = -gamma, -beta, -alpha

    U_aligned = {key: np.zeros_like(radial0, dtype=complex) for key in U_0.keys()}
    coeffs = {}

    for mp in range(-l0, l0 + 1):
        coeff = wigner_D_element(l0, mp, m0, alpha, beta, gamma)
        U_aligned[(l0, mp)] = coeff * radial0
        coeffs[(l0, mp)] = coeff

    return U_aligned, coeffs, radial0

def _build_translation_integrand(L, l, m, a, bound_state, context):
    """
    Integrand over theta of the projection of the (l, m) atomic mode, displaced
    by a along z, onto Y_Lm about the origin, as a numpy function of (theta, r).
    """
    theta, r = sp.symbols("theta r", real=True, positive=True)

    mabs = abs(m)

    cos_theta = sp.cos(theta)
    r_p = sp.sqrt(r**2 + a**2 - 2 * r * a * cos_theta)
    cos_theta_p = (r * cos_theta - a) / r_p

    N_Lm = sph_harm_prefactor(L, m)
    N_lm = sph_harm_prefactor(l, m)

    Yproj_conj = N_Lm * sp.assoc_legendre(L, mabs, cos_theta)
    Ysrc = N_lm * sp.assoc_legendre(l, mabs, cos_theta_p)

    radial_part = bound_state.radial.subs(bound_state.r, r_p)

    expr = 2 * sp.pi * r * sp.sin(theta) * Yproj_conj * Ysrc * radial_part
    expr = sp.simplify(expr.subs(context.get_subs()))

    return sp.lambdify((theta, r), expr, modules=["numpy"])

def translate_along_z(U_aligned, coeffs, radial0, r_arr, L_max, a, bound_state, context,
                      quad_kwargs=None):
    """
    Translate the aligned-frame state along z.

    Parameters
    ----------
    U_aligned : dict
        Rotated aligned modes.
    coeffs : dict
        Rotation coefficients returned by rotate_single_mode_to_aligned_full.
    radial0 : np.ndarray
        Original radial profile before rotation.
    """
    print("Translating along new z axis")

    if quad_kwargs is None:
        quad_kwargs = {}

    r_arr = np.asarray(r_arr)
    zero = np.zeros_like(r_arr, dtype=complex)

    m_values = sorted({m for (_, m) in U_aligned.keys()})
    U_out = {(L, m): zero.copy() for m in m_values for L in range(abs(m), L_max + 1)}

    integrand_cache = {}

    for (l, m), coeff in coeffs.items():
        if abs(coeff) == 0:
            continue

        for L in range(abs(m), L_max + 1):
            key = (L, l, m)
            print(f"Translation ({l},{m}) -> ({L},{m})")

            if key not in integrand_cache:
                integrand_cache[key] = _build_translation_integrand(
                    L=L, l=l, m=m, a=a, bound_state=bound_state, context=context
                )

            integrand = integrand_cache[key]

            profile = np.array(
                [integrate_complex(integrand, 0.0, np.pi, args=(rv,), **quad_kwargs)
                 for rv in r_arr],
                dtype=complex
            )

            U_out[(L, m)] += coeff * profile

    return U_out


def rotate_back_from_aligned_full(U_aligned_translated, L_max, r_arr,
                                  theta_a, phi_a, inverse=False):
    """
    Input:
        U_aligned_translated: dict {(L,m): radial_array}
        theta_a, phi_a: original direction of translation axis
    Output:
        U_final: dict {(L,M): radial_array} in original COM frame, with z axis
                 parallel to the original one.
    """
    print("Realigning z axis")
    r_arr = np.asarray(r_arr)
    U_final = {}

    for L in range(L_max + 1):
        for M in range(-L, L + 1):
            U_final[(L, M)] = np.zeros_like(r_arr, dtype=complex)

    alpha = float(phi_a)
    beta = float(theta_a)
    gamma = 0.0

    if not inverse:
        alpha, beta, gamma = -gamma, -beta, -alpha

    for L in range(L_max + 1):
        for M in range(-L, L + 1):
            print(f"Angular mode ({L}, {M})")
            accum = np.zeros_like(r_arr, dtype=complex)

            for mp in range(-L, L + 1):
                if (L, mp) not in U_aligned_translated:
                    continue
                coeff = wigner_D_element(L, M, mp, alpha, beta, gamma)
                accum += coeff * U_aligned_translated[(L, mp)]

            U_final[(L, M)] = accum

    return U_final

def save_modes_binary(filename, U_dict):
    """
    Save all modes to a binary file.

    U_dict:
        {(l,m): np.ndarray complex128 of shape (Nr,)}
    """

    mode_items = sorted(U_dict.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    Nmodes = len(mode_items)

    Nr = None
    for (_, _), arr in mode_items:
        arr = np.asarray(arr)
        if Nr is None:
            Nr = len(arr)
        elif len(arr) != Nr:
            raise ValueError("All radial profiles must have the same length")

    with open(filename, "wb") as f:

        # header
        f.write(b"MODESBIN")
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<q", Nr))
        f.write(struct.pack("<q", Nmodes))

        # modes
        for (l, m), profile in mode_items:

            profile = np.asarray(profile, dtype=np.complex128)

            f.write(struct.pack("<i", l))
            f.write(struct.pack("<i", m))

            f.write(profile.real.astype(np.float64).tobytes(order="C"))
            f.write(profile.imag.astype(np.float64).tobytes(order="C"))

def build_centered_atom_modes(
    modes,
    r_arr,
    bound_state,
    context,
):
    """
    Construct a gravitational atom centered at the spherical-grid origin.

    The state is represented by a single populated (l,m) mode before
    translation.
    """
    U_centered = zero_mode_dict(modes, r_arr)

    ell = int(bound_state.l)
    emm = int(bound_state.m)

    if (ell, emm) not in U_centered:
        raise ValueError(
            f"Atomic mode ({ell},{emm}) is not included in "
            f"the angular truncation l_max={max(l for l, _ in modes)}."
        )

    radial_expression = (
        bound_state.r * bound_state.radial
    ).subs(context.get_subs())

    radial_function = sp.lambdify(
        bound_state.r,
        radial_expression,
        modules=["numpy"],
    )

    U_centered[(ell, emm)] = np.asarray(
        radial_function(r_arr),
        dtype=np.complex128,
    )

    return U_centered


def initialize_translated_atom(
    modes,
    r_arr,
    L_max,
    bound_state,
    context,
    displacement,
    theta,
    phi,
    quad_kwargs=None,
):
    """
    Construct one gravitational atom and translate it from the grid
    origin to

        displacement * (sin(theta) cos(phi),
                        sin(theta) sin(phi),
                        cos(theta)).

    The intrinsic atomic angular momentum remains defined relative to
    the global z axis.
    """
    U_centered = build_centered_atom_modes(
        modes=modes,
        r_arr=r_arr,
        bound_state=bound_state,
        context=context,
    )

    U_aligned, coeffs, radial0 = rotate_single_mode_to_aligned_full(
        U_0=U_centered,
        theta_a=theta,
        phi_a=phi,
    )

    U_translated_aligned = translate_along_z(
        U_aligned=U_aligned,
        coeffs=coeffs,
        radial0=radial0,
        r_arr=r_arr,
        L_max=L_max,
        a=displacement,
        bound_state=bound_state,
        context=context,
        quad_kwargs=quad_kwargs,
    )

    U_translated = rotate_back_from_aligned_full(
        U_aligned_translated=U_translated_aligned,
        L_max=L_max,
        r_arr=r_arr,
        theta_a=theta,
        phi_a=phi,
        inverse=False,
    )

    return U_translated


def combine_mode_dicts(
    U_first,
    U_second,
    amplitude_first=1.0,
    amplitude_second=1.0,
):
    """
    Coherently superpose two sets of spherical-harmonic modes.
    """
    if U_first.keys() != U_second.keys():
        raise ValueError("The two mode dictionaries have different modes.")

    return {
        mode: (
            amplitude_first * np.asarray(U_first[mode], dtype=complex)
            + amplitude_second * np.asarray(U_second[mode], dtype=complex)
        )
        for mode in U_first
    }


def modes_norm(U_dict, r_arr):
    """
    Compute the three-dimensional norm

        integral |Psi|^2 d^3x = sum_lm integral |U_lm(r)|^2 dr

    for Psi = sum_lm U_lm(r)/r Y_lm.
    """
    return float(
        sum(
            np.trapezoid(np.abs(profile) ** 2, x=r_arr)
            for profile in U_dict.values()
        )
    )


def normalize_modes(U_dict, r_arr, target_norm=1.0):
    """Rescale all modes so that modes_norm equals target_norm."""
    current_norm = modes_norm(U_dict, r_arr)

    if current_norm <= 0.0:
        raise ValueError("Cannot normalize a state with zero norm.")

    factor = np.sqrt(target_norm / current_norm)

    return {
        mode: factor * profile
        for mode, profile in U_dict.items()
    }

# ----------------------------------------------------------------------
# Primary gravitational atom: centered at x = +R1
# ----------------------------------------------------------------------

print("\nInitializing the primary gravitational atom")

U_primary = initialize_translated_atom(
    modes=modes,
    r_arr=r_arr,
    L_max=L_max,
    bound_state=bound_state_1,
    context=context_1,
    displacement=R1,
    theta=theta_1,
    phi=phi_1,
)

# ----------------------------------------------------------------------
# Secondary gravitational atom: centered at x = -R2
# ----------------------------------------------------------------------

if args.two_atoms:
    print("\nInitializing the secondary gravitational atom")

    U_secondary = initialize_translated_atom(
        modes=modes,
        r_arr=r_arr,
        L_max=L_max,
        bound_state=bound_state_2,
        context=context_2,
        displacement=-R2,
        theta=theta_2,
        phi=phi_2,
    )

# ----------------------------------------------------------------------
# Relative amplitudes and phases
# ----------------------------------------------------------------------

primary_amplitude = float(
    config["atom"].get("amplitude", 1.0)
)

secondary_amplitude = float(
    secondary_config.get("amplitude", 1.0)
)


if args.two_atoms:
    U_final = combine_mode_dicts(
        U_first=U_primary,
        U_second=U_secondary,
        amplitude_first=primary_amplitude,
        amplitude_second=secondary_amplitude,
    )

    print("\nNorms before final normalization")
    print(f"  Primary atom:   {modes_norm(U_primary, r_arr):.16e}")
    print(f"  Secondary atom: {modes_norm(U_secondary, r_arr):.16e}")
    print(f"  Combined state: {modes_norm(U_final, r_arr):.16e}")
else:
    U_final = U_primary

# Optional normalization of the total coherent state.
normalize_total = bool(
    config["simulation"].get("normalize_initial_state", True)
)

target_norm = float(
    config["simulation"].get("initial_norm", 1.0)
)

if normalize_total:
    U_final = normalize_modes(
        U_dict=U_final,
        r_arr=r_arr,
        target_norm=target_norm,
    )

    print(
        f"  Final normalized norm: "
        f"{modes_norm(U_final, r_arr):.16e}"
    )

save_modes_binary(args.output, U_final)