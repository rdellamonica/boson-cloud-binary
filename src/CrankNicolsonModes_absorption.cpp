/**
 * @file CrankNicolsonModes_absorption.cpp
 * @brief Spherical s-wave absorbers centred on each black hole.
 */
#include "CrankNicolsonModes.hpp"
#include <gsl/gsl_integration.h>
#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

void CrankNicolsonModes::update_absorption_profile(
    AbsorptionProfile &profile, double radius, double width)
{
    const int degree = 2 * l_max;
    const std::size_t size = static_cast<std::size_t>(degree + 1) * Nr;
    if (profile.radius == radius && profile.width == width && profile.multipoles.size() == size)
        return; // Fixed orbits only require phase/ramp updates.
    profile.radius = radius;
    profile.width = width;
    profile.multipoles.assign(size, 0.0);
    if (width < dr || width <= 0.0 || radius < 0.0)
        return;

    // K depends on s=|x-x_BH|, not on r-R_BH alone. Integrate only the angular
    // cap intersecting s<width, so a distant small ball cannot fall between nodes.
    const int nq = std::max(128, degree + 16);
    auto *table = gsl_integration_glfixed_table_alloc(nq);
    if (!table)
        throw std::bad_alloc();
    std::vector<double> nodes(nq), weights(nq);
    for (int k = 0; k < nq; ++k)
        gsl_integration_glfixed_point(0.0, 1.0, k, &nodes[k], &weights[k], table);
    gsl_integration_glfixed_table_free(table);
#pragma omp parallel for schedule(static) if (n_modes >= 8 && Nr >= 128)
    for (int i = 1; i < Nr - 1; ++i)
    {
        const double delta = r[i] - radius;
        if (std::abs(delta) >= width)
            continue;
        if (radius == 0.0)
        {
            // Central spherical absorber: exactly monopolar, with no 1/R_BH^2.
            const double x = r[i] / width;
            profile.multipoles[i] = 4.0 * M_PI * std::exp(-1.0 / (1.0 - x * x));
            continue;
        }
        const double product = 2.0 * r[i] * radius;
        const double cap = std::min(2.0, (width * width - delta * delta) / product);
        for (int k = 0; k < nq; ++k)
        {
            // z=1-cos(gamma) avoids cancellation for a narrow cap.
            const double z = cap * nodes[k];
            const double u = 1.0 - z;
            const double distance2 = delta * delta + product * z;
            const double weight = 2.0 * M_PI * cap * weights[k] *
                                  std::exp(-1.0 / (1.0 - distance2 / (width * width)));
            double previous = 1.0, current = u;
            profile.multipoles[i] += weight;
            if (degree > 0)
                profile.multipoles[Nr + i] += weight * current;
            for (int l = 2; l <= degree; ++l)
            {
                const double next = ((2 * l - 1) * u * current - (l - 1) * previous) / l;
                profile.multipoles[l * Nr + i] += weight * next;
                previous = current;
                current = next;
            }
        }
    }
    // Discrete 3D volume normalization: integral r^2 dr dOmega K = 1.
    // Only evolved interior radial samples participate in the normalization.
    double norm = 0.0;
    for (int i = 1; i < Nr - 1; ++i)
        norm += dr * r[i] * r[i] * profile.multipoles[i];
    if (norm > 0.0)
    {
        for (double &value : profile.multipoles)
            value /= norm;
    } // No interior support leaves an identically zero absorber.
}

double CrankNicolsonModes::get_absorption_rate_1(
    const Eigen::VectorXcd &psi) const
{
    if (sink1.strength == 0.0 || sink1.v.size() != psi.size())
        return 0.0;
    // mu * dr * psi^dagger Gamma_1 psi, with Gamma_1 = strength * v v^dagger.
    return mu * dr * sink1.strength * std::norm(sink1.v.dot(psi));
}

double CrankNicolsonModes::get_absorption_rate_2(
    const Eigen::VectorXcd &psi) const
{
    if (sink2.strength == 0.0 || sink2.v.size() != psi.size())
        return 0.0;
    return mu * dr * sink2.strength * std::norm(sink2.v.dot(psi));
}

Eigen::VectorXcd CrankNicolsonModes::apply_sink(const Eigen::VectorXcd &state) const
{
    Eigen::VectorXcd result = Eigen::VectorXcd::Zero(state.size());
    if (!sink_active)
        return result;
    for (const SwaveSink *sink : {&sink1, &sink2})
        if (sink->strength != 0.0 && sink->v.size() == state.size())
            result += (sink->strength * sink->v.dot(state)) * sink->v;
    return result;
}

Eigen::VectorXcd CrankNicolsonModes::schrodinger_rhs(const Eigen::VectorXcd &state) const
{
    Eigen::VectorXcd result = -I * (H * state);
    if (sink_active)
        result -= 0.5 * apply_sink(state);
    return result;
}

double CrankNicolsonModes::sink_strength(int body) const
{
    const SwaveSink &sink = body == 1 ? sink1 : sink2;
    if (!sink_active || sink.strength == 0.0)
        return 0.0;
    return sink.strength * sink.v.squaredNorm();
}

void CrankNicolsonModes::build_swave_projector(
    const AbsorptionProfile &profile, bool secondary, Eigen::VectorXcd &v) const
{
    // kappa_j(r) = integral dOmega Y_j^* K is sqrt(4 pi) times the local kernel's
    // (j, 00) matrix element, because Y_00 is constant. Reusing the potential's
    // Gaunt coefficients and orbital phases keeps the conventions identical.
    v.setZero(static_cast<Eigen::Index>(Nr) * n_modes);
    if (mode00 < 0)
        return;
    const double sqrt4pi = std::sqrt(4.0 * M_PI);
    for (const auto &pair : mode_pairs)
    {
        if (pair.col_mode != mode00)
            continue;
        const dcomplex phase = orbital_phase(pair.m_star);
        const double parity = secondary && pair.m_star % 2 != 0 ? -1.0 : 1.0;
        for (int i = 1; i < Nr - 1; ++i)
        {
            dcomplex kappa = 0.0;
            for (const auto &term : pair.terms)
                kappa += profile.multipoles[term.l_star * Nr + i] * term.angular * phase;
            v[pair.row_mode * Nr + i] = r[i] * sqrt4pi * parity * kappa;
        }
    }
}

void CrankNicolsonModes::build_spherical_absorption()
{
    update_absorption_profile(absorption_profile1, R1, epsilon_smear * M1);
    update_absorption_profile(absorption_profile2, R2, epsilon_smear * M2);

    double ramp = 1.0;

    if (!isolated_black_hole && t_i < ramp_end_time())
    {
        ramp = 0.0;
    }
    else if (!isolated_black_hole && t_i < ramp_end_time() + initial_period)
    {
        const double x = (t_i - ramp_end_time()) / initial_period;
        ramp = 3.0 * x * x - 2.0 * x * x * x;
    }

    Ah1 = 16.0 * M_PI * M1 * M1 * ramp;
    Ah2 = 16.0 * M_PI * M2 * M2 * ramp;

    // Absorption acts only on the s-wave about each black hole:
    // rate = A_h * |integral K psi d^3x|^2. Any l>0 component about the
    // black hole has zero overlap with the spherical kernel and is not absorbed.

    build_swave_projector(absorption_profile1, false, sink1.v);
    sink1.strength = Ah1 * dr;

    build_swave_projector(absorption_profile2, true, sink2.v);
    sink2.strength = Ah2 * dr;
}
