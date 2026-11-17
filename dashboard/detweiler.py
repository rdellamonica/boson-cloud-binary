"""
Detweiler's small-alpha rates of the quasi-bound states of a massive scalar
around a Kerr black hole (Detweiler 1980; in the form of Baumann, Chia & Porto
2019, Eqs. 2.23-2.28), with G = c = hbar = 1.

    from detweiler import detweiler_rate
    G = detweiler_rate(M=1.0, chi=0.9, mu=0.2)            # 211, field-amplitude rate
    G_N = 2 * G                                            # rate of the occupation number / cloud mass
    detweiler_rate(1.0, 0.9, mu=np.linspace(0.05, 0.5, 200), n=3, l=2, m=2)

Gamma_nlm = 2 r_+/M C_nl g_lm (m Omega_H - omega_nlm) alpha^(4l+5),
    C_nl = 2^(4l+1) (n+l)! / (n^(2l+4) (n-l-1)!) [l! / ((2l)! (2l+1)!)]^2,
    g_lm = prod_{k=1..l} [k^2 (1 - chi^2) + (chi m - 2 r_+ omega)^2],
with alpha = M mu, r_+ = M (1 + sqrt(1 - chi^2)), Omega_H = chi / (2 r_+).

Sign: Gamma > 0 is growth (superradiance, m Omega_H > omega), Gamma < 0 decay.
psi ~ exp(Gamma t), so the cloud's mass changes at 2 Gamma. The result has the
units of 1/M (in code units, M1 = 1). Leading order in alpha: good for
alpha <~ 0.1-0.2; towards the peak of the 211 rate it departs from the exact
(numerical) rates by factors of a few, see Dolan (2007) for those.
"""

import math

import numpy as np


def level_frequency(M, chi, mu, n=2, l=1, m=1, order=5):
    """omega_nlm: hydrogenic level with the relativistic, fine (alpha^4) and
    hyperfine (alpha^5, spin) corrections of Baumann et al. (2019) Eq. 2.23.
    order=2 keeps only mu (1 - alpha^2 / 2n^2)."""
    alpha = M * mu
    w = 1.0 - alpha ** 2 / (2.0 * n ** 2)
    if order >= 4:
        f = -6.0 / (2 * l + 1) + 2.0 / n
        w = w - alpha ** 4 / (8.0 * n ** 4) + f * alpha ** 4 / n
    if order >= 5 and l > 0:
        h = 16.0 / (2 * l * (2 * l + 1) * (2 * l + 2))
        w = w + h * chi * m * alpha ** 5 / n ** 3
    return mu * w


def detweiler_rate(M, chi, mu, n=2, l=1, m=1, omega=None, order=5):
    """Field-amplitude rate Gamma_nlm (1/time, same units as 1/M) of the level
    (n, l, m) around a hole of mass M and dimensionless spin chi, for boson mass
    mu. Arrays broadcast. omega: use this frequency instead of level_frequency."""
    if not (n > l >= 0 and abs(m) <= l):
        raise ValueError("need n > l >= 0 and |m| <= l")
    M, chi, mu = (np.asarray(v, dtype=float) for v in (M, chi, mu))
    if np.any(np.abs(chi) > 1):
        raise ValueError("|chi| must be at most 1")
    alpha = M * mu
    r_plus = M * (1.0 + np.sqrt(1.0 - chi ** 2))
    Omega_H = chi / (2.0 * r_plus)
    w = level_frequency(M, chi, mu, n, l, m, order) if omega is None else np.asarray(omega, dtype=float)
    C = (2.0 ** (4 * l + 1) * math.factorial(n + l) / (n ** (2 * l + 4) * math.factorial(n - l - 1))
         * (math.factorial(l) / (math.factorial(2 * l) * math.factorial(2 * l + 1))) ** 2)
    g = np.ones_like(alpha)
    for k in range(1, l + 1):
        g = g * (k ** 2 * (1.0 - chi ** 2) + (chi * m - 2.0 * r_plus * w) ** 2)
    rate = 2.0 * (r_plus / M) * C * g * (m * Omega_H - w) * alpha ** (4 * l + 5)
    return rate if rate.ndim else float(rate)


if __name__ == "__main__":
    # Schwarzschild 211: the particle loss rate is (alpha^10 / 6) (1 + 16 alpha^2) / M at leading order.
    a = 0.1
    print("2 Gamma_211(chi=0) M =", 2 * detweiler_rate(1.0, 0.0, a, order=2),
          " expected", -(a ** 10 / 6) * (1 + 16 * a ** 2 * (1 - a ** 2 / 8) ** 2))
    for chi in (0.5, 0.9, 0.99):
        mus = np.linspace(0.01, 0.6, 600)
        G = detweiler_rate(1.0, chi, mus)
        i = int(np.argmax(G))
        print(f"chi = {chi}: max Gamma_211 M = {G[i]:.3e} at alpha = {mus[i]:.3f}; "
              f"superradiant for alpha < {mus[G > 0].max():.3f}")
