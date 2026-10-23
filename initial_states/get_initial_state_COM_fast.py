#!/usr/bin/env python3
"""
Fast variant of get_initial_state_COM.py for a single translated atom.

The translation integrals use fixed Gauss-Legendre quadrature (order
--n-quad, or [simulation] translation_n_quad, default 64) and cached Wigner
matrices instead of adaptive quadrature, which is much faster on large grids.
--atom selects whether the atom sits on the primary (x = +R1, default) or on
the secondary (x = -R2). The configuration keys are those of
get_initial_state_COM.py.

Usage:
    python get_initial_state_COM_fast.py --config config.toml --output initial_modes.bin [--atom secondary]
"""

import argparse
import struct
import time
from functools import lru_cache

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy.special import lpmv, gammaln
from sympy.physics.wigner import wigner_d

from gravatom.context import Context
from gravatom.gravitational_atom import BoundState

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# ======================================================================
# COMMAND LINE
# ======================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--atom", choices=["primary", "secondary"], default="primary",
                    help="Which single atom to generate. Default: primary.")
parser.add_argument("--n-quad", type=int, default=None,
                    help="Gauss-Legendre angular quadrature order. Default: config value or 64.")
args = parser.parse_args()


# ======================================================================
# CONFIG
# ======================================================================

with open(args.config, "rb") as f:
    config = tomllib.load(f)

N = int(config["grid"]["N"])
box_size = float(config["grid"]["L"])
q = float(config["binary"]["q"])
R = float(config["binary"]["R"])
M1 = float(config["binary"].get("M1", 1.0))
M2 = q*M1
Mtot = M1 + M2
mu = float(config["atom"]["mu"])
r0 = float(config["simulation"]["epsilon"])
L_max = int(config["simulation"]["l_max"])
Omega = np.sqrt(Mtot/R**3)

# COM-frame distances
R1 = M2/Mtot*R
R2 = M1/Mtot*R

# Primary: +x
theta_1 = np.pi/2
phi_1 = 0.0

# Secondary: -x represented as negative displacement along the same
# translated axis. This preserves the convention of the original code.
theta_2 = np.pi/2
phi_2 = 0.0

n_quad = args.n_quad
if n_quad is None:
    n_quad = int(config["simulation"].get("translation_n_quad", 64))

print("Binary parameters")
print(f"  M1       = {M1}")
print(f"  M2       = {M2}")
print(f"  Mtot     = {Mtot}")
print(f"  q        = {q}")
print(f"  R        = {R}")
print(f"  R1       = {R1}")
print(f"  R2       = {R2}")
print(f"  Omega    = {Omega}")
print(f"  l_max    = {L_max}")
print(f"  N        = {N}")
print(f"  L        = {box_size}")
print(f"  n_quad   = {n_quad}")


# ======================================================================
# GRAVITATIONAL-ATOM CONTEXTS
# ======================================================================

context_1 = Context()
context_1(mu=mu, M=M1)
context_1(alpha=context_1.mu*context_1.M)
context_1(a=4*context_1.alpha/(1 + 4*context_1.alpha**2))

context_2 = Context()
context_2(mu=mu, M=M2)
context_2(alpha=context_2.mu*context_2.M)
context_2(a=4*context_2.alpha/(1 + 4*context_2.alpha**2))

print("\nGravitational couplings")
print(f"  alpha_1 = {M1*mu}")
print(f"  alpha_2 = {M2*mu}")


# ======================================================================
# QUANTUM NUMBERS
# ======================================================================

n1 = int(config["atom"]["n"])
l1 = int(config["atom"]["l"])
m1 = int(config["atom"]["m"])

secondary_config = config.get("atom_secondary", {})
n2 = int(secondary_config.get("n", n1))
l2 = int(secondary_config.get("l", l1))
m2 = int(secondary_config.get("m", m1))

bound_state_1 = BoundState(n1, l1, m1, context_1)
bound_state_2 = BoundState(n2, l2, m2, context_2)

epsilon_1 = float(bound_state_1.eigenvalue.subs(context_1.get_subs())) - mu
epsilon_2 = float(bound_state_2.eigenvalue.subs(context_2.get_subs())) - mu

print("\nAtomic states")
print(f"  Primary:   (n,l,m)=({n1},{l1},{m1}), epsilon={epsilon_1}")
print(f"  Secondary: (n,l,m)=({n2},{l2},{m2}), epsilon={epsilon_2}")


# ======================================================================
# GRID / MODES
# ======================================================================

r_arr = np.linspace(0.0, box_size, N)
r_arr[0] = r0

modes = [(ell, emm) for ell in range(L_max + 1) for emm in range(-ell, ell + 1)]


# ======================================================================
# BASIC HELPERS
# ======================================================================

def zero_mode_dict(mode_list, r):
    return {mode: np.zeros_like(r, dtype=np.complex128) for mode in mode_list}


def modes_norm(U_dict, r):
    return float(sum(np.trapezoid(np.abs(profile)**2, x=r) for profile in U_dict.values()))


def normalize_modes(U_dict, r, target_norm=1.0):
    current = modes_norm(U_dict, r)
    if current <= 0.0:
        raise ValueError("Cannot normalize a state with zero norm.")
    factor = np.sqrt(target_norm/current)
    return {mode: factor*np.asarray(profile, dtype=np.complex128) for mode, profile in U_dict.items()}


# ======================================================================
# FAST CACHED WIGNER MATRICES
# ======================================================================

@lru_cache(maxsize=None)
def wigner_D_matrix(l, alpha, beta, gamma):
    """
    Full Wigner-D matrix using the same SymPy convention as the original
    code. Cached because a given (l,angles) matrix is reused many times.
    """
    D = sp.N(wigner_d(sp.Integer(l), float(alpha), float(beta), float(gamma)))
    return np.asarray(D.tolist(), dtype=np.complex128)


def wigner_D_element(l, mp, m, alpha, beta, gamma):
    D = wigner_D_matrix(int(l), float(alpha), float(beta), float(gamma))
    return D[l-mp, l-m]


# ======================================================================
# SPHERICAL-HARMONIC THETA PREFACTOR
# ======================================================================

@lru_cache(maxsize=None)
def sph_harm_prefactor_fast(l, m):
    """
    Numerical equivalent of the prefactor used in the original code:

      (-1)^|m| sqrt[(2l+1)/(4pi) (l-|m|)!/(l+|m|)!]

    Used together with scipy.special.lpmv(|m|,l,x).
    """
    mabs = abs(int(m))
    log_ratio = gammaln(l-mabs+1) - gammaln(l+mabs+1)
    return (-1)**mabs*np.sqrt((2*l+1)/(4*np.pi)*np.exp(log_ratio))


# ======================================================================
# CENTERED ATOM
# ======================================================================

def build_centered_atom_modes(modes, r_arr, bound_state, context):
    """Construct an origin-centered atom with one populated (l,m) mode."""
    U = zero_mode_dict(modes, r_arr)
    ell, emm = int(bound_state.l), int(bound_state.m)

    if (ell, emm) not in U:
        raise ValueError(
            f"Atomic mode ({ell},{emm}) is not included in l_max="
            f"{max(l for l, _ in modes)}."
        )

    radial_expression = (bound_state.r*bound_state.radial).subs(context.get_subs())
    radial_function = sp.lambdify(bound_state.r, radial_expression, modules=["numpy"])
    U[(ell, emm)] = np.asarray(radial_function(r_arr), dtype=np.complex128)
    return U


# ======================================================================
# ROTATE SINGLE ATOMIC MODE SO TRANSLATION AXIS IS z
# ======================================================================

def rotate_single_mode_to_aligned_full(U_0, theta_a, phi_a, inverse=False, tol=1e-15):
    populated = [(lm, arr) for lm, arr in U_0.items() if np.any(np.abs(arr) > 0)]
    if len(populated) != 1:
        raise ValueError("U_0 must contain exactly one populated mode.")

    (l0, m0), radial0 = populated[0]
    radial0 = np.asarray(radial0, dtype=np.complex128)

    alpha, beta, gamma = float(phi_a), float(theta_a), 0.0
    if inverse:
        alpha, beta, gamma = -gamma, -beta, -alpha

    D = wigner_D_matrix(l0, alpha, beta, gamma)
    coeffs = {}
    U_aligned = {key: np.zeros_like(radial0) for key in U_0}

    # SymPy ordering: m=l,l-1,...,-l
    for row, mp in enumerate(range(l0, -l0-1, -1)):
        coeff = D[row, l0-m0]
        if abs(coeff) > tol:
            coeffs[(l0, mp)] = coeff
            U_aligned[(l0, mp)] = coeff*radial0

    return U_aligned, coeffs, radial0


# ======================================================================
# TRANSLATION GEOMETRY CACHE
# ======================================================================

class TranslationAlongZCache:
    """
    Cache everything depending only on:
      - radial grid
      - displacement
      - l_max
      - angular quadrature

    Translation along z conserves m.
    """

    def __init__(self, r_arr, L_max, displacement, n_quad=64):
        self.r = np.asarray(r_arr, dtype=float)
        self.L_max = int(L_max)
        self.a = float(displacement)
        self.n_quad = int(n_quad)

        # Integral over theta is converted to u=cos(theta):
        # integral_0^pi sin(theta)dtheta -> integral_-1^1 du
        self.u, self.w = leggauss(self.n_quad)

        rr = self.r[:, None]
        uu = self.u[None, :]

        # r' = |r - a zhat|
        self.rp = np.sqrt(np.maximum(rr**2 + self.a**2 - 2*rr*self.a*uu, 0.0))
        safe_rp = np.where(self.rp > 1e-14, self.rp, 1.0)

        # cos(theta') in translated coordinates
        self.cos_theta_p = np.clip((rr*uu-self.a)/safe_rp, -1.0, 1.0)

        # Lazy cache: one projection matrix for each m.
        self._projection_cache = {}

    def projection_matrix(self, m):
        """
        Matrix containing Y_Lm(theta,phi=0) theta-dependence for
        L=|m|,...,L_max at the Gauss-Legendre nodes.
        """
        m = int(m)
        if m not in self._projection_cache:
            mabs = abs(m)
            Lvals = np.arange(mabs, self.L_max+1, dtype=int)
            Yproj = np.empty((len(Lvals), self.n_quad), dtype=np.float64)

            for j, L in enumerate(Lvals):
                Yproj[j] = sph_harm_prefactor_fast(int(L), m)*lpmv(mabs, int(L), self.u)

            self._projection_cache[m] = (Lvals, Yproj)

        return self._projection_cache[m]


# ======================================================================
# FAST TRANSLATION ALONG z
# ======================================================================

def translate_along_z_fast(coeffs, r_arr, L_max, displacement, bound_state, context,
                           n_quad=64, cache=None, verbose=True):
    """
    Translate the rotated atomic state along z.

    This implements the same integral as the original code but replaces
    thousands of adaptive scipy.quad calls with vectorized Gauss-Legendre
    quadrature.

    Since the translation is along z, m is conserved.
    """
    r_arr = np.asarray(r_arr, dtype=float)

    if cache is None:
        cache = TranslationAlongZCache(r_arr, L_max, displacement, n_quad=n_quad)

    active = [(l, m, c) for (l, m), c in coeffs.items() if abs(c) > 1e-15]
    active_m = sorted(set(m for _, m, _ in active))

    U_out = {(L, m): np.zeros_like(r_arr, dtype=np.complex128)
             for m in active_m for L in range(abs(m), L_max+1)}

    if verbose:
        print("\nFast translation along aligned z axis")
        print(f"  displacement        = {displacement:g}")
        print(f"  radial points       = {len(r_arr)}")
        print(f"  quadrature points   = {n_quad}")
        print(f"  active source modes = {[(l,m) for l,m,_ in active]}")

    t0 = time.perf_counter()

    # Radial atomic wavefunction R_nl(r') evaluated ONCE over all (r,u).
    radial_expr = bound_state.radial.subs(context.get_subs())
    radial_func = sp.lambdify(bound_state.r, radial_expr, modules=["numpy"])

    if verbose:
        print("  evaluating translated radial wavefunction ...", end="", flush=True)

    radial_values = np.asarray(radial_func(cache.rp), dtype=np.complex128)

    # Some lambdified expressions may return a scalar for trivial cases.
    if radial_values.ndim == 0:
        radial_values = np.full_like(cache.rp, radial_values, dtype=np.complex128)
    else:
        radial_values = np.broadcast_to(radial_values, cache.rp.shape)

    if verbose:
        print(f" done [{time.perf_counter()-t0:.2f} s]")

    # Group rotated source modes by m.
    for im, m in enumerate(active_m, start=1):
        tm = time.perf_counter()
        sources = [(l, c) for l, mm, c in active if mm == m]
        Lvals, Yproj = cache.projection_matrix(m)
        mabs = abs(m)

        if verbose:
            print(f"  m={m:+d} ({im}/{len(active_m)}): "
                  f"{len(sources)} source mode(s) -> {len(Lvals)} output modes ...",
                  end="", flush=True)

        # Build the translated source angular dependence.
        source = np.zeros_like(cache.rp, dtype=np.complex128)
        for l, coeff in sources:
            Ysrc = sph_harm_prefactor_fast(l, m)*lpmv(mabs, l, cache.cos_theta_p)
            source += coeff*Ysrc*radial_values

        # Gauss-Legendre integral over u.
        # source*w has shape (Nr,Nquad)
        # Yproj.T has shape (Nquad,NL)
        profiles = 2*np.pi*r_arr[:, None]*((source*cache.w[None, :]) @ Yproj.T)

        for j, L in enumerate(Lvals):
            U_out[(int(L), m)] = profiles[:, j]

        if verbose:
            print(f" done [{time.perf_counter()-tm:.2f} s]")

    if verbose:
        print(f"  translation completed in {time.perf_counter()-t0:.2f} s")

    return U_out


# ======================================================================
# FAST ROTATION BACK TO GLOBAL z AXIS
# ======================================================================

def rotate_back_from_aligned_fast(U_aligned_translated, L_max, r_arr,
                                  theta_a, phi_a, inverse=False, verbose=True):
    """
    Rotate all translated modes back to the global z-quantized basis.

    For each L this is one matrix multiplication instead of nested
    (M,m') loops with repeated Wigner-D evaluations.
    """
    r_arr = np.asarray(r_arr)
    Nr = len(r_arr)
    U_final = {(L, M): np.zeros(Nr, dtype=np.complex128)
               for L in range(L_max+1) for M in range(-L, L+1)}

    alpha, beta, gamma = float(phi_a), float(theta_a), 0.0
    if not inverse:
        alpha, beta, gamma = -gamma, -beta, -alpha

    if verbose:
        print("\nRotating translated state back to global z frame")

    t0 = time.perf_counter()

    for L in range(L_max+1):
        mvals = np.arange(L, -L-1, -1)
        D = wigner_D_matrix(L, alpha, beta, gamma)

        A = np.zeros((2*L+1, Nr), dtype=np.complex128)
        for row, mp in enumerate(mvals):
            key = (L, int(mp))
            if key in U_aligned_translated:
                A[row] = U_aligned_translated[key]

        rotated = D @ A

        for row, M in enumerate(mvals):
            U_final[(L, int(M))] = rotated[row]

    if verbose:
        print(f"  rotation completed in {time.perf_counter()-t0:.2f} s")

    return U_final


# ======================================================================
# COMPLETE FAST INITIALIZER
# ======================================================================

def initialize_translated_atom_fast(modes, r_arr, L_max, bound_state, context,
                                    displacement, theta, phi, n_quad=64, verbose=True):
    """
    1. Build centered atom.
    2. Rotate translation direction onto z.
    3. Translate along z using vectorized fixed quadrature.
    4. Rotate back to original global z basis.
    """
    t0 = time.perf_counter()

    if verbose:
        print("\n============================================================")
        print("INITIALIZING TRANSLATED GRAVITATIONAL ATOM")
        print("============================================================")
        print(f"  state        = ({bound_state.n},{bound_state.l},{bound_state.m})")
        print(f"  displacement = {displacement:g}")
        print(f"  theta        = {theta:g}")
        print(f"  phi          = {phi:g}")

    if verbose:
        print("\nBuilding centered atom ...", end="", flush=True)

    U_centered = build_centered_atom_modes(
        modes=modes, r_arr=r_arr, bound_state=bound_state, context=context
    )

    if verbose:
        print(f" done [{time.perf_counter()-t0:.2f} s]")
        print("Rotating source mode into translation-aligned frame ...", end="", flush=True)

    tr = time.perf_counter()
    _, coeffs, _ = rotate_single_mode_to_aligned_full(
        U_0=U_centered, theta_a=theta, phi_a=phi
    )

    if verbose:
        print(f" done [{time.perf_counter()-tr:.2f} s]")
        print(f"  populated aligned modes = {list(coeffs.keys())}")

    cache = TranslationAlongZCache(
        r_arr=r_arr, L_max=L_max, displacement=displacement, n_quad=n_quad
    )

    U_translated_aligned = translate_along_z_fast(
        coeffs=coeffs, r_arr=r_arr, L_max=L_max, displacement=displacement,
        bound_state=bound_state, context=context, n_quad=n_quad,
        cache=cache, verbose=verbose
    )

    U_translated = rotate_back_from_aligned_fast(
        U_aligned_translated=U_translated_aligned, L_max=L_max,
        r_arr=r_arr, theta_a=theta, phi_a=phi, verbose=verbose
    )

    if verbose:
        norm0 = modes_norm(U_centered, r_arr)
        norm1 = modes_norm(U_translated, r_arr)
        print("\nTranslation diagnostics")
        print(f"  centered norm   = {norm0:.12e}")
        print(f"  translated norm = {norm1:.12e}")
        print(f"  retained norm   = {norm1/norm0:.10f}")
        print(f"  total time      = {time.perf_counter()-t0:.2f} s")

    return U_translated


# ======================================================================
# MODESBIN OUTPUT
# ======================================================================

def save_modes_binary(filename, U_dict):
    """
    MODESBIN format:
      magic      : 8 bytes "MODESBIN"
      version    : int32
      Nr         : int64
      Nmodes     : int64

      then for each sorted (l,m):
        l        : int32
        m        : int32
        real(U)  : Nr little-endian float64
        imag(U)  : Nr little-endian float64
    """
    mode_items = sorted(U_dict.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    Nmodes = len(mode_items)

    if Nmodes == 0:
        raise ValueError("No modes to write.")

    Nr = len(mode_items[0][1])
    for (_, _), arr in mode_items:
        if len(arr) != Nr:
            raise ValueError("All radial profiles must have the same length.")

    with open(filename, "wb") as f:
        f.write(b"MODESBIN")
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<q", Nr))
        f.write(struct.pack("<q", Nmodes))

        for (l, m), profile in mode_items:
            profile = np.asarray(profile, dtype=np.complex128)
            f.write(struct.pack("<i", l))
            f.write(struct.pack("<i", m))
            f.write(np.asarray(profile.real, dtype="<f8").tobytes(order="C"))
            f.write(np.asarray(profile.imag, dtype="<f8").tobytes(order="C"))

    print(f"\nSaved {Nmodes} modes x {Nr} radial points to {filename}")


# ======================================================================
# SELECT PRIMARY / SECONDARY ATOM
# ======================================================================

if args.atom == "primary":
    bound_state = bound_state_1
    context = context_1
    displacement = R1
    theta = theta_1
    phi = phi_1
    amplitude = float(config["atom"].get("amplitude", 1.0))
else:
    bound_state = bound_state_2
    context = context_2
    displacement = -R2
    theta = theta_2
    phi = phi_2
    amplitude = float(secondary_config.get("amplitude", 1.0))

print(f"\nGenerating {args.atom} atom")


# ======================================================================
# GENERATE TRANSLATED STATE
# ======================================================================

U_final = initialize_translated_atom_fast(
    modes=modes,
    r_arr=r_arr,
    L_max=L_max,
    bound_state=bound_state,
    context=context,
    displacement=displacement,
    theta=theta,
    phi=phi,
    n_quad=n_quad,
    verbose=True
)

# Apply configured amplitude.
if amplitude != 1.0:
    U_final = {mode: amplitude*profile for mode, profile in U_final.items()}
    print(f"\nApplied amplitude = {amplitude}")


# ======================================================================
# OPTIONAL FINAL NORMALIZATION
# ======================================================================

norm_before = modes_norm(U_final, r_arr)
print(f"\nNorm before final normalization: {norm_before:.16e}")

normalize_total = bool(config["simulation"].get("normalize_initial_state", True))
target_norm = float(config["simulation"].get("initial_norm", 1.0))

if normalize_total:
    U_final = normalize_modes(U_final, r_arr, target_norm=target_norm)
    print(f"Final normalized norm: {modes_norm(U_final, r_arr):.16e}")
else:
    print("Final normalization disabled.")


# ======================================================================
# SAVE
# ======================================================================

save_modes_binary(args.output, U_final)