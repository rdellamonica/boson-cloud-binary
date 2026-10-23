/**
 * @file Definitions.hpp
 * @brief Common types: complex scalars and angular modes.
 */
#pragma once
#include <complex>

/// Double-precision complex number.
using dcomplex = std::complex<double>;
/// The imaginary unit.
extern const dcomplex I;

/// A spherical-harmonic mode (l, m).
struct mode
{
    int l;
    int m;

    bool operator==(const mode &other) const
    {
        return l == other.l && m == other.m;
    }
};
