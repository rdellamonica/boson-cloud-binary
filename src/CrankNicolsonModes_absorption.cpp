/**
 * @file CrankNicolsonModes_absorption.cpp
 * @brief Horizon absorbers centred on each black hole: the s-wave sink and the dipolar channels.
 */
#include "CrankNicolsonModes.hpp"
#include <gsl/gsl_integration.h>
#include <algorithm>
#include <cmath>
#include <complex>
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
    // A kernel narrower than the local spacing is not resolved.
    double spacing = dr;
    if (!uniform_grid)
    {
        const int i = std::clamp(node_at(radius - r(0)), 0, Nr - 2);
        spacing = r(i + 1) - r(i);
    }
    if (width < spacing || width <= 0.0 || radius < 0.0)
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
        norm += node_weight(i) * r[i] * r[i] * profile.multipoles[i];
    if (norm > 0.0)
    {
        for (double &value : profile.multipoles)
            value /= norm;
    } // No interior support leaves an identically zero absorber.
}


namespace
{
/// Gauss-Legendre nodes and weights on [0, 1].
void gauss_legendre_unit(int n, std::vector<double> &nodes, std::vector<double> &weights)
{
    auto *table = gsl_integration_glfixed_table_alloc(n);
    if (!table)
        throw std::bad_alloc();
    nodes.resize(n);
    weights.resize(n);
    for (int k = 0; k < n; ++k)
        gsl_integration_glfixed_point(0.0, 1.0, k, &nodes[k], &weights[k], table);
    gsl_integration_glfixed_table_free(table);
}

/// Horizon area 8 pi M r_+ of a Kerr hole (16 pi M^2 without spin).
double horizon_area(double M, double chi)
{
    const double c = std::clamp(chi, -1.0, 1.0);
    return 16.0 * M_PI * M * M * (0.5 * (1.0 + std::sqrt(1.0 - c * c)));
}
} // namespace

double CrankNicolsonModes::dipole_coefficient(double M, double chi, int m) const
{
    // gamma_1m = A_h k_+ / (4 pi mu) * [M^2 - a^2 + (2 M r_+ k_+)^2], k_+ = mu - m Omega_+.
    if (!(M > 0.0))
        return 0.0;
    const double c = std::clamp(chi, -1.0, 1.0);
    const double spin = c * M;
    const double r_plus = M * (1.0 + std::sqrt(1.0 - c * c));
    const double k_plus = mu - m * spin / (2.0 * M * r_plus);
    const double x = 2.0 * M * r_plus * k_plus;
    return horizon_area(M, c) * k_plus / (4.0 * M_PI * mu) * (M * M - spin * spin + x * x);
}

void CrankNicolsonModes::reset_sink_support(SwaveSink &sink, double radius, double width) const
{
    const Eigen::Index size = static_cast<Eigen::Index>(Nr) * n_modes;
    const bool weighted = !uniform_grid;
    if (sink.v.size() != size || (weighted && sink.w.size() != size) || sink.first + sink.count > Nr)
    {
        sink.v.setZero(size);
        if (weighted)
            sink.w.setZero(size);
        else
            sink.w.resize(0);
    }
    else
        for (int j = 0; j < n_modes && sink.count > 0; ++j)
        {
            sink.v.segment(static_cast<Eigen::Index>(j) * Nr + sink.first, sink.count).setZero();
            if (weighted)
                sink.w.segment(static_cast<Eigen::Index>(j) * Nr + sink.first, sink.count).setZero();
        }
    // Interior nodes with |r - radius| < width.
    const double *begin = r.data() + 1, *end = r.data() + Nr - 1;
    const double *lower = std::upper_bound(begin, end, radius - width);
    const double *upper = std::lower_bound(lower, end, radius + width);
    sink.first = static_cast<int>(lower - r.data());
    sink.count = width > 0.0 ? static_cast<int>(upper - lower) : 0;
}

dcomplex CrankNicolsonModes::sink_dot(const SwaveSink &sink, const Eigen::VectorXcd &x) const
{
    const Eigen::VectorXcd &left = sink.left();
    dcomplex sum = 0.0;
    for (int j = 0; j < n_modes; ++j)
    {
        const Eigen::Index offset = static_cast<Eigen::Index>(j) * Nr + sink.first;
        sum += left.segment(offset, sink.count).dot(x.segment(offset, sink.count));
    }
    return sum;
}

std::vector<const CrankNicolsonModes::SwaveSink *> CrankNicolsonModes::active_sinks(Eigen::Index size) const
{
    std::vector<const SwaveSink *> list;
    const auto consider = [&](const SwaveSink &sink) {
        if (sink.strength != 0.0 && sink.count > 0 && sink.v.size() == size)
            list.push_back(&sink);
    };
    consider(sink1);
    consider(sink2);
    if (dipole_absorption)
        for (const SwaveSink &sink : dipole_sinks)
            consider(sink);
    return list;
}

double CrankNicolsonModes::sink_particle_rate(const SwaveSink &sink, const Eigen::VectorXcd &state) const
{
    if (sink.strength == 0.0 || sink.count <= 0 || sink.v.size() != state.size())
        return 0.0;
    // dr * psi^dagger Gamma psi, with Gamma = strength * v v^dagger.
    return (uniform_grid ? dr : 1.0) * sink.strength * std::norm(sink_dot(sink, state));
}

double CrankNicolsonModes::get_absorption_rate_1(
    const Eigen::VectorXcd &psi) const
{
    double rate = sink_particle_rate(sink1, psi);
    if (dipole_absorption)
        rate += sink_particle_rate(dipole_sinks[0], psi) + sink_particle_rate(dipole_sinks[1], psi);
    return mu * rate;
}

double CrankNicolsonModes::get_absorption_rate_2(
    const Eigen::VectorXcd &psi) const
{
    double rate = sink_particle_rate(sink2, psi);
    if (dipole_absorption)
        rate += sink_particle_rate(dipole_sinks[2], psi) + sink_particle_rate(dipole_sinks[3], psi);
    return mu * rate;
}

double CrankNicolsonModes::absorbed_spin_rate(int body, const Eigen::VectorXcd &psi) const
{
    if (!dipole_absorption)
        return 0.0;
    // Each particle taken through the channel m carries m units of spin.
    const SwaveSink *pair = dipole_sinks + (body == 1 ? 0 : 2);
    return sink_particle_rate(pair[1], psi) - sink_particle_rate(pair[0], psi);
}

dcomplex CrankNicolsonModes::dipole_amplitude(int body, int m, const Eigen::VectorXcd &state) const
{
    const SwaveSink &sink = dipole_sinks[(body == 1 ? 0 : 2) + (m > 0 ? 1 : 0)];
    if (sink.count <= 0 || sink.v.size() != state.size())
        return 0.0;
    return (uniform_grid ? dr : 1.0) * sink_dot(sink, state);
}

double CrankNicolsonModes::dipole_strength(int body, int m) const
{
    const SwaveSink &sink = dipole_sinks[(body == 1 ? 0 : 2) + (m > 0 ? 1 : 0)];
    if (!sink_active || !dipole_absorption || sink.count <= 0)
        return 0.0;
    return sink.strength / (uniform_grid ? dr : 1.0);
}

dcomplex CrankNicolsonModes::dipole_weight(int body, int m, int j, int i) const
{
    const SwaveSink &sink = dipole_sinks[(body == 1 ? 0 : 2) + (m > 0 ? 1 : 0)];
    const Eigen::Index index = static_cast<Eigen::Index>(j) * Nr + i;
    return index < sink.v.size() ? sink.v[index] : dcomplex(0.0);
}

Eigen::VectorXcd CrankNicolsonModes::apply_sink(const Eigen::VectorXcd &state) const
{
    Eigen::VectorXcd result = Eigen::VectorXcd::Zero(state.size());
    if (!sink_active)
        return result;
    for (const SwaveSink *sink : active_sinks(state.size()))
    {
        const dcomplex coefficient = sink->strength * sink_dot(*sink, state);
        for (int j = 0; j < n_modes; ++j)
        {
            const Eigen::Index offset = static_cast<Eigen::Index>(j) * Nr + sink->first;
            result.segment(offset, sink->count) += coefficient * sink->v.segment(offset, sink->count);
        }
    }
    return result;
}

Eigen::VectorXcd CrankNicolsonModes::schrodinger_rhs(const Eigen::VectorXcd &state) const
{
    Eigen::VectorXcd result = -I * hamiltonian_times(state);
    if (sink_active)
        result -= 0.5 * apply_sink(state);
    return result;
}

double CrankNicolsonModes::sink_strength(int body) const
{
    const SwaveSink &sink = body == 1 ? sink1 : sink2;
    if (!sink_active || sink.strength == 0.0)
        return 0.0;
    return uniform_grid ? sink.strength * sink.v.squaredNorm() : sink.strength * sink.left().dot(sink.v).real();
}

void CrankNicolsonModes::build_swave_projector(
    const AbsorptionProfile &profile, bool secondary, SwaveSink &sink) const
{
    // kappa_j(r) = integral dOmega Y_j^* K is sqrt(4 pi) times the local kernel's
    // (j, 00) matrix element, because Y_00 is constant. Reusing the potential's
    // Gaunt coefficients and orbital phases keeps the conventions identical.
    // The kernel vanishes outside the shell |r - R_BH| < width: only those rows are touched.
    reset_sink_support(sink, profile.radius, profile.width);
    sink.body = secondary ? 2 : 1;
    sink.m = 0;
    if (mode00 < 0 || sink.count <= 0)
        return;
    const double sqrt4pi = std::sqrt(4.0 * M_PI);
    // With pruning, the projector is restricted to the active modes (the sink
    // is the Galerkin projection of the full one), even when (0, 0) is inactive.
    for (const auto &pair : all_mode_pairs)
    {
        if (pair.col_mode != mode00 || !active_modes[pair.row_mode])
            continue;
        const dcomplex phase = orbital_phase(pair.m_star);
        const double parity = secondary && pair.m_star % 2 != 0 ? -1.0 : 1.0;
        for (int i = sink.first; i < sink.first + sink.count; ++i)
        {
            dcomplex kappa = 0.0;
            for (const auto &term : pair.terms)
                kappa += profile.multipoles[term.l_star * Nr + i] * term.angular * phase;
            const Eigen::Index index = static_cast<Eigen::Index>(pair.row_mode) * Nr + i;
            sink.v[index] = r[i] * sqrt4pi * parity * kappa;
            if (!uniform_grid) // Gamma = A_h v (W v)^dagger: <v|W psi> is the kernel integral.
                sink.w[index] = node_weight[i] * sink.v[index];
        }
    }
}

void CrankNicolsonModes::update_dipole_profile(
    DipoleProfile &profile, double radius, double width, double boost)
{
    SwaveSink shell;
    shell.v.resize(0);
    {
        // Same support as the sinks: interior nodes with |r - radius| < width.
        const double *begin = r.data() + 1, *end = r.data() + Nr - 1;
        const double *lower = std::upper_bound(begin, end, radius - width);
        const double *upper = std::lower_bound(lower, end, radius + width);
        shell.first = static_cast<int>(lower - r.data());
        shell.count = width > 0.0 ? static_cast<int>(upper - lower) : 0;
    }
    const std::size_t size = static_cast<std::size_t>(n_modes) * std::max(shell.count, 0);
    if (profile.radius == radius && profile.width == width && profile.boost == boost &&
        profile.first == shell.first && profile.count == shell.count && profile.reference[0].size() == size)
        return; // Fixed orbits only need the phase rule.
    profile.radius = radius;
    profile.width = width;
    profile.boost = boost;
    profile.first = shell.first;
    profile.count = shell.count;
    for (auto &reference : profile.reference)
        reference.assign(size, 0.0);

    // As for the s-wave kernel: a bump narrower than the local spacing is not resolved.
    double spacing = dr;
    if (!uniform_grid)
    {
        const int i = std::clamp(node_at(radius - r(0)), 0, Nr - 2);
        spacing = r(i + 1) - r(i);
    }
    int mode_minus = -1, mode_plus = -1;
    for (int j = 0; j < n_modes; ++j)
        if (modes[j].l == 1)
            (modes[j].m < 0 ? mode_minus : mode_plus) = j;
    if (width < spacing || width <= 0.0 || radius < 0.0 || profile.count <= 0 || mode_plus < 0 || mode_minus < 0)
    {
        profile.count = 0;
        for (auto &reference : profile.reference)
            reference.clear();
        return;
    }
    const int first = profile.first, count = profile.count;
    const double width2 = width * width;
    // rho Y_{1,+-1}(rho_hat) = -+ c1 (rho_x +- i rho_y).
    const double c1 = std::sqrt(3.0 / (8.0 * M_PI));

    if (radius == 0.0)
    {
        // Hole at the origin: the kernel is the (1, m) harmonic itself, w = r * (r K).
        for (int k = 0; k < count; ++k)
        {
            const double x = r[first + k] / width;
            const double value = r[first + k] * r[first + k] * std::exp(-1.0 / (1.0 - x * x));
            profile.reference[0][static_cast<std::size_t>(mode_minus) * count + k] = value;
            profile.reference[1][static_cast<std::size_t>(mode_plus) * count + k] = value;
        }
    }
    else
    {
        // w_{LM;1m}(r) = r * integral dOmega Y_LM^* e^{i boost rho_y} rho K(rho) Y_1m(rho_hat),
        // with the hole on +x and its velocity along +y. Polar angle gamma about the hole's
        // direction, z = 1 - cos(gamma) over the cap that meets the kernel, azimuth beta about
        // it. K depends on z alone and the rest is a trigonometric polynomial in beta (the boost
        // adds a rapidly converging Bessel tail), so the midpoint rule in beta is exact once it
        // resolves degree l_max + 1 plus that tail. The integrand is even under z -> -z of the
        // simulation frame (beta -> -beta): half of the beta points are enough.
        const int ng = std::max(48, l_max + 16);
        const int tail = 14 + static_cast<int>(std::ceil(4.0 * std::abs(boost) * width));
        const int nb = 2 * ((l_max + 2 + tail + 1) / 2);
        std::vector<double> nodes, weights;
        gauss_legendre_unit(ng, nodes, weights);
        std::vector<double> cos_beta(nb / 2), sin_beta(nb / 2);
        for (int b = 0; b < nb / 2; ++b)
        {
            cos_beta[b] = std::cos(2.0 * M_PI * (b + 0.5) / nb);
            sin_beta[b] = std::sin(2.0 * M_PI * (b + 0.5) / nb);
        }
        // Normalised associated Legendre functions without the sin^M factor,
        // Q_LM = Pbar_LM / sin^M(theta): Y_LM^* = Q_LM(n_z) (n_x - i n_y)^M for M >= 0.
        const int stride = l_max + 1;
        std::vector<double> rec_a(static_cast<std::size_t>(stride) * stride, 0.0), rec_b(rec_a.size(), 0.0),
            diagonal(stride, 0.0);
        diagonal[0] = std::sqrt(1.0 / (4.0 * M_PI));
        for (int m = 1; m <= l_max; ++m)
            diagonal[m] = -std::sqrt((2.0 * m + 1.0) / (2.0 * m)) * diagonal[m - 1];
        for (int m = 0; m <= l_max; ++m)
            for (int l = m + 1; l <= l_max; ++l)
            {
                rec_a[l * stride + m] = std::sqrt((4.0 * l * l - 1.0) / (static_cast<double>(l) * l - m * m));
                rec_b[l * stride + m] =
                    std::sqrt(((l - 1.0) * (l - 1.0) - m * m) / (4.0 * (l - 1.0) * (l - 1.0) - 1.0));
            }
        const double front = 2.0 * (2.0 * M_PI / nb); // beta weight, doubled for the mirror half

#pragma omp parallel if (static_cast<long>(count) * n_modes >= 256)
        {
            std::vector<double> Q(static_cast<std::size_t>(stride) * stride, 0.0);
            // sum_x, sum_y: the integrals with rho_x and rho_y in place of rho Y_1m.
            std::vector<dcomplex> power(stride), mirrored(stride), sum_x(n_modes), sum_y(n_modes);
#pragma omp for schedule(dynamic, 1)
            for (int k = 0; k < count; ++k)
            {
                const double ri = r[first + k];
                const double delta = ri - radius;
                const double product = 2.0 * ri * radius;
                const double cap = std::min(2.0, (width2 - delta * delta) / product);
                std::fill(sum_x.begin(), sum_x.end(), dcomplex(0.0));
                std::fill(sum_y.begin(), sum_y.end(), dcomplex(0.0));
                for (int g = 0; g < ng; ++g)
                {
                    const double z = cap * nodes[g];
                    const double distance2 = delta * delta + product * z;
                    if (!(distance2 < width2))
                        continue;
                    const double kernel = cap * weights[g] * front * std::exp(-1.0 / (1.0 - distance2 / width2));
                    const double cos_gamma = 1.0 - z, sin_gamma = std::sqrt(z * (2.0 - z));
                    const double rho_x = ri * cos_gamma - radius;
                    for (int b = 0; b < nb / 2; ++b)
                    {
                        const double nx = cos_gamma, ny = sin_gamma * cos_beta[b], nz = sin_gamma * sin_beta[b];
                        const double rho_y = ri * ny;
                        // power[M] = (n_x - i n_y)^M, mirrored[M] = (-1)^M (n_x + i n_y)^M
                        // (Y_{L,-M}^* = (-1)^M Y_{LM}), both times the kernel and the boost phase.
                        const dcomplex common = kernel * std::polar(1.0, boost * rho_y);
                        power[0] = mirrored[0] = common;
                        for (int m = 1; m <= l_max; ++m)
                        {
                            power[m] = power[m - 1] * dcomplex(nx, -ny);
                            mirrored[m] = mirrored[m - 1] * dcomplex(-nx, -ny);
                        }
                        for (int m = 0; m <= l_max; ++m)
                        {
                            Q[m * stride + m] = diagonal[m];
                            if (m < l_max)
                                Q[(m + 1) * stride + m] = std::sqrt(2.0 * m + 3.0) * nz * diagonal[m];
                            for (int l = m + 2; l <= l_max; ++l)
                                Q[l * stride + m] = rec_a[l * stride + m] *
                                                    (nz * Q[(l - 1) * stride + m] - rec_b[l * stride + m] * Q[(l - 2) * stride + m]);
                        }
                        for (int j = 0; j < n_modes; ++j)
                        {
                            const int M = modes[j].m, absM = std::abs(M);
                            const dcomplex term = Q[modes[j].l * stride + absM] * (M >= 0 ? power[absM] : mirrored[absM]);
                            sum_x[j] += rho_x * term;
                            sum_y[j] += rho_y * term;
                        }
                    }
                }
                for (int j = 0; j < n_modes; ++j)
                {
                    profile.reference[0][static_cast<std::size_t>(j) * count + k] = ri * c1 * (sum_x[j] - I * sum_y[j]);
                    profile.reference[1][static_cast<std::size_t>(j) * count + k] = -ri * c1 * (sum_x[j] + I * sum_y[j]);
                }
            }
        }
    }

    // Discrete normalisation in place of 1 / I_1. The harmonic field rho Y_1m about the hole
    // is, in COM modes, U_1m = r^2 and U_00 = -sqrt(4 pi) R Y_1m(pi/2, 0) r (Eq. 107). Its
    // amplitude must be the continuum value int f_1m^* e^{-i boost rho_y} rho Y_1m d^3rho,
    // which is 1 without boost.
    double target = 1.0;
    if (boost != 0.0)
    {
        std::vector<double> nodes, weights;
        gauss_legendre_unit(48, nodes, weights);
        double numerator = 0.0, denominator = 0.0;
        for (std::size_t a = 0; a < nodes.size(); ++a)
        {
            const double rho = width * nodes[a];
            const double radial = weights[a] * std::pow(rho, 4) * std::exp(-1.0 / (1.0 - nodes[a] * nodes[a]));
            double angular = 0.0; // (3/8) int_{-1}^{1} cos(boost rho c) (1 + c^2) dc
            for (std::size_t c = 0; c < nodes.size(); ++c)
            {
                const double cosine = 2.0 * nodes[c] - 1.0;
                angular += 0.75 * weights[c] * std::cos(boost * rho * cosine) * (1.0 + cosine * cosine);
            }
            numerator += radial * angular;
            denominator += radial;
        }
        target = numerator / denominator;
    }
    int mode_zero = -1;
    for (int j = 0; j < n_modes; ++j)
        if (modes[j].l == 0)
            mode_zero = j;
    for (int which = 0; which < 2; ++which)
    {
        auto &reference = profile.reference[which];
        const int own = which ? mode_plus : mode_minus;
        // sqrt(4 pi) Y_{1,+-1}(pi/2, 0) = -+ sqrt(3/2).
        const double monopole = (which ? 1.0 : -1.0) * std::sqrt(1.5) * radius;
        dcomplex norm = 0.0;
        for (int k = 0; k < count; ++k)
        {
            const double ri = r[first + k];
            dcomplex value = std::conj(reference[static_cast<std::size_t>(own) * count + k]) * ri * ri;
            if (mode_zero >= 0)
                value += std::conj(reference[static_cast<std::size_t>(mode_zero) * count + k]) * monopole * ri;
            norm += node_weight(first + k) * value;
        }
        if (std::abs(norm) > 0.0 && std::isfinite(std::abs(norm)))
        {
            const dcomplex scale = target / std::conj(norm);
            for (dcomplex &value : reference)
                value *= scale;
        }
        else
            std::fill(reference.begin(), reference.end(), dcomplex(0.0));
    }
}

void CrankNicolsonModes::fill_dipole_sinks(const DipoleProfile &profile, double azimuth, double M, double chi,
                                           double ramp, SwaveSink *pair)
{
    for (int which = 0; which < 2; ++which)
    {
        SwaveSink &sink = pair[which];
        const int m = which ? 1 : -1;
        reset_sink_support(sink, profile.radius, profile.width);
        sink.m = m;
        sink.strength = 0.0;
        if (profile.count <= 0 || profile.reference[which].empty())
        {
            sink.count = 0;
            continue;
        }
        sink.first = profile.first;
        sink.count = profile.count;
        sink.strength = dipole_coefficient(M, chi, m) * ramp * (uniform_grid ? dr : 1.0);
        const dcomplex *reference = profile.reference[which].data();
        for (int j = 0; j < n_modes; ++j)
        {
            if (!active_modes[j])
                continue; // Galerkin restriction to the evolved modes, as for the s-wave sink.
            // Rigid rotation of the hole and its velocity: w(r; phi) = e^{-i (M - m) phi} w(r; 0).
            const dcomplex phase = std::polar(1.0, -(modes[j].m - m) * azimuth);
            const Eigen::Index offset = static_cast<Eigen::Index>(j) * Nr + sink.first;
            const dcomplex *source = reference + static_cast<std::size_t>(j) * sink.count;
            for (int k = 0; k < sink.count; ++k)
            {
                sink.v[offset + k] = phase * source[k];
                if (!uniform_grid)
                    sink.w[offset + k] = node_weight[sink.first + k] * sink.v[offset + k];
            }
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

    Ah1 = horizon_area(M1, chi1) * ramp;
    Ah2 = horizon_area(M2, chi2) * ramp;

    // The spherical kernel acts only on the s-wave about each black hole:
    // rate = A_h * |integral K psi d^3x|^2. Any l>0 component about the
    // black hole has zero overlap with it; the dipolar sinks below take l=1.

    build_swave_projector(absorption_profile1, false, sink1);
    build_swave_projector(absorption_profile2, true, sink2);
    // Gamma = strength * v w^dagger, with w = v and strength = A_h dr on a uniform grid.
    sink1.strength = Ah1 * (uniform_grid ? dr : 1.0);
    sink2.strength = Ah2 * (uniform_grid ? dr : 1.0);

    if (!dipole_absorption)
        return;
    dipole_sinks[0].body = dipole_sinks[1].body = 1;
    dipole_sinks[2].body = dipole_sinks[3].body = 2;
    if (ramp == 0.0)
    {
        for (SwaveSink &sink : dipole_sinks)
            sink.strength = 0.0;
        return;
    }
    // Boost of the kernels to each hole's rest frame: mu * v_a, with v_a tangential.
    const double speed = mu * (counter ? -1.0 : 1.0) * Omega;
    update_dipole_profile(dipole_profile1, R1, epsilon_smear * M1, speed * R1);
    update_dipole_profile(dipole_profile2, R2, epsilon_smear * M2, speed * R2);
    fill_dipole_sinks(dipole_profile1, theta, M1, chi1, ramp, dipole_sinks);
    fill_dipole_sinks(dipole_profile2, theta + M_PI, M2, chi2, ramp, dipole_sinks + 2);
}

double CrankNicolsonModes::absorbed_Lz_rate(const Eigen::VectorXcd &psi) const
{
    // d<L_z>/dt|sink = -Re <psi| L_z Gamma |psi>, Gamma = g v w^dagger, with L_z = m on each mode.
    double rate = 0.0;
    for (const SwaveSink *sink : active_sinks(psi.size()))
    {
        const Eigen::VectorXcd &w = sink->left();
        const dcomplex projection = sink_dot(*sink, psi); // w^dagger psi
        dcomplex L_overlap = 0.0;                          // sum_j m_j <psi_j| w_j>
        for (int j = 0; j < n_modes; ++j)
            if (modes[j].m != 0)
            {
                const Eigen::Index offset = static_cast<Eigen::Index>(j) * Nr + sink->first;
                L_overlap += static_cast<double>(modes[j].m) *
                             psi.segment(offset, sink->count).dot(w.segment(offset, sink->count));
            }
        rate += sink->strength * (L_overlap * projection).real();
    }
    return rate * (uniform_grid ? dr : 1.0);
}
