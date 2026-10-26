/**
 * @file SphericalAbsorptionReference.hpp
 * @brief Independent reference implementations of the black-hole absorbers.
 *
 * The production code integrates the kernel over angular caps and projects it
 * with cached Gaunt coefficients. These references integrate in the distance
 * from the black hole and use the addition theorem instead.
 */
#pragma once
#include "CrankNicolsonModes.hpp"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <algorithm>
#include <cmath>
#include <vector>

/// Legendre moments 2 pi int K(s) P_l(cos gamma) dcos gamma of one body's kernel,
/// indexed l*Nr+i and normalised to unit volume on the interior grid.
inline std::vector<double> absorption_moments(const CrankNicolsonModes &cn, double smear, int body)
{
    std::vector<double> moments((2*cn.l_max+1)*cn.Nr, 0.0);
    const double width = smear*(body == 1 ? cn.M1 : cn.M2), d = body == 1 ? cn.R1 : cn.R2;
    if (width < cn.dr || width <= 0) return moments;
    auto *table = gsl_integration_glfixed_table_alloc(256);
    for (int i=1; i<cn.Nr-1; ++i) {
        const double ri=cn.r[i], low=std::abs(ri-d), high=std::min(width,ri+d);
        if (d < 1e-8*width) { // Central limit avoids cancellation in distance endpoints.
            if (ri<width) moments[i]=4*M_PI*std::exp(-1/(1-ri*ri/(width*width)));
        } else if (low<high) {
            for (int k=0;k<256;++k) {
                double s,w;
                gsl_integration_glfixed_point(low,high,k,&s,&w,table);
                const double u=std::clamp((ri*ri+d*d-s*s)/(2*ri*d),-1.0,1.0);
                for (int l=0;l<=2*cn.l_max;++l)
                    moments[l*cn.Nr+i]+=2*M_PI/(ri*d)*w*s*std::exp(-1/(1-s*s/(width*width)))*
                                             gsl_sf_legendre_Pl(l,u);
            }
        }
    }
    gsl_integration_glfixed_table_free(table);
    double norm=0;
    for(int i=1;i<cn.Nr-1;++i) norm+=cn.dr*cn.r[i]*cn.r[i]*moments[i];
    if(norm>0) for(double &v:moments) v/=norm;
    return moments;
}

inline std::vector<double> primary_absorption_moments(const CrankNicolsonModes &cn, double smear)
{
    return absorption_moments(cn, smear, 1);
}

/// Y_lm(pi/2, phi) with the Condon-Shortley phase.
inline dcomplex equatorial_harmonic(int l, int m, double phi)
{
    double y = gsl_sf_legendre_sphPlm(l, std::abs(m), 0.0);
    if (m < 0 && std::abs(m) % 2) y = -y;
    return y * std::exp(I * double(m) * phi);
}

/**
 * Projector of one body's s-wave sink, v[j*Nr+i] = r_i int K Y_j^* dOmega.
 * By the addition theorem int K Y_lm^* dOmega = K_l(r) Y_lm^*(n_BH), with the
 * primary at azimuth theta and the secondary at theta + pi.
 */
inline Eigen::VectorXcd swave_projector_reference(const CrankNicolsonModes &cn, double smear, int body)
{
    const auto moments = absorption_moments(cn, smear, body);
    const double phi = cn.theta + (body == 1 ? 0.0 : M_PI);
    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(cn.Nr*cn.n_modes);
    for (int j=0; j<cn.n_modes; ++j) {
        const auto [l, m] = cn.modes[j];
        const dcomplex direction = std::conj(equatorial_harmonic(l, m, phi));
        for (int i=1; i<cn.Nr-1; ++i)
            v[j*cn.Nr+i] = cn.r[i]*moments[l*cn.Nr+i]*direction;
    }
    return v;
}

/// Dense absorption operator Gamma = sum_a 16 pi M_a^2 ramp dr v_a v_a^dagger.
inline Eigen::MatrixXcd sink_reference(const CrankNicolsonModes &cn, double smear, double ramp)
{
    Eigen::MatrixXcd gamma = Eigen::MatrixXcd::Zero(cn.Nr*cn.n_modes, cn.Nr*cn.n_modes);
    for (int body : {1, 2}) {
        const double mass = body == 1 ? cn.M1 : cn.M2;
        const Eigen::VectorXcd v = swave_projector_reference(cn, smear, body);
        gamma += 16*M_PI*mass*mass*ramp*cn.dr * (v*v.adjoint());
    }
    return gamma;
}
