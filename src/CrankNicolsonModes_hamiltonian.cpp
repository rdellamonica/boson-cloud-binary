/**
 * @file CrankNicolsonModes_hamiltonian.cpp
 * @brief Free Hamiltonian, multipole couplings and the time-dependent binary potential.
 */
#include "CrankNicolsonModes.hpp"
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>
#include <algorithm>
#include <cmath>
#include <vector>

double CrankNicolsonModes::F1(double r, int l_star, int /*m_star*/)
{
    double F = 0.0;

    if (r >= R1)
    {
        F += 1 / (1 + q) * pow(R1, l_star) / pow(r, l_star + 1);
    }
    else
    {
        F += 1 / (1 + q) * pow(r, l_star) / pow(R1, l_star + 1);
    }

    return F;
}

double CrankNicolsonModes::F2(double r, int l_star, int m_star)
{
    double F = 0.0;

    if (r >= R2)
    {
        F += (m_star % 2 == 0 ? 1.0 : -1.0) * q / (1 + q) * pow(R2, l_star) / pow(r, l_star + 1);
    }
    else
    {
        F += (m_star % 2 == 0 ? 1.0 : -1.0) * q / (1 + q) * pow(r, l_star) / pow(R2, l_star + 1);
    }

    return F;
}

void CrankNicolsonModes::build_stationary_hamiltonian()
{

    std::vector<Eigen::Triplet<dcomplex>> Ht(3 * n_modes * (Nr - 2));

    H0.resize(Nr * n_modes, Nr * n_modes);

    dcomplex alpha = 1.0 / (dr * dr);

#pragma omp parallel for schedule(static) if (n_modes >= 8)
    for (int j = 0; j < n_modes; j++)
    {
        for (int i = 1; i < Nr - 1; i++)
        {
            const auto offset = 3 * (j * (Nr - 2) + i - 1);
            const double centrifugal = modes[j].l * (modes[j].l + 1) / (r(i) * r(i));
            if (uniform_grid)
            {
                Ht[offset] = Eigen::Triplet<dcomplex>(j * Nr + i, j * Nr + i, -(-2.0 * alpha - centrifugal) / (2 * mu));
                Ht[offset + 1] = Eigen::Triplet<dcomplex>(j * Nr + i, j * Nr + i - 1, -alpha / (2 * mu));
                Ht[offset + 2] = Eigen::Triplet<dcomplex>(j * Nr + i, j * Nr + i + 1, -alpha / (2 * mu));
                continue;
            }
            // Three-point second difference on unequal spacings h- and h+.
            const double hm = r(i) - r(i - 1), hp = r(i + 1) - r(i);
            Ht[offset] = Eigen::Triplet<dcomplex>(j * Nr + i, j * Nr + i, (2.0 / (hm * hp) + centrifugal) / (2 * mu));
            Ht[offset + 1] = Eigen::Triplet<dcomplex>(j * Nr + i, j * Nr + i - 1, -1.0 / (mu * hm * (hm + hp)));
            Ht[offset + 2] = Eigen::Triplet<dcomplex>(j * Nr + i, j * Nr + i + 1, -1.0 / (mu * hp * (hm + hp)));
        }
    }

    H0.setFromTriplets(Ht.begin(), Ht.end());
}

void CrankNicolsonModes::build_coupling_cache()
{
    // Each worker owns a row of mode pairs. No shared hash-map insertions.
    std::vector<std::vector<ModePair>> rows(n_modes);
#pragma omp parallel for schedule(dynamic) if (n_modes >= 8)
    for (int j = 0; j < n_modes; ++j)
    {
        for (int k = 0; k < n_modes; ++k)
        {
            const int l = modes[j].l, m = modes[j].m;
            const int l1 = modes[k].l, m1 = modes[k].m;
            const int m_star = m1 - m;
            ModePair pair{j, k, m_star, {}};
            for (int l_star = std::abs(l1 - l); l_star <= l1 + l; ++l_star)
            {
                if (std::abs(m_star) > l_star || (l + l1 + l_star) % 2 != 0)
                    continue;
                const double w1 = gsl_sf_coupling_3j(2 * l_star, 2 * l1, 2 * l, 0, 0, 0);
                const double w2 = gsl_sf_coupling_3j(2 * l_star, 2 * l1, 2 * l, -2 * m_star, 2 * m1, -2 * m);
                double y = gsl_sf_legendre_sphPlm(l_star, std::abs(m_star), 0.0);
                if (m_star < 0 && std::abs(m_star) % 2 != 0)
                    y = -y;
                const double pref = (m1 % 2 == 0 ? 1.0 : -1.0) *
                                    std::sqrt((2.0 * l + 1) * (2.0 * l1 + 1) * (2.0 * l_star + 1) / (4.0 * M_PI));
                const dcomplex angular = dcomplex(pref) * dcomplex(w1) * dcomplex(w2) * dcomplex(y);
                if (std::abs(angular) != 0.0)
                    pair.terms.push_back({l_star, angular, -4.0 * M_PI / (2.0 * l_star + 1.0) * angular.real()});
            }
            if (!pair.terms.empty())
                rows[j].push_back(std::move(pair));
        }
    }
    all_mode_pairs.clear();
    for (auto &row : rows)
        for (auto &pair : row)
            all_mode_pairs.push_back(std::move(pair));
    if (active_modes.size() != static_cast<std::size_t>(n_modes))
        active_modes.assign(n_modes, 1);
}

dcomplex CrankNicolsonModes::orbital_phase(int m_star) const
{
    return std::exp(I * static_cast<double>(m_star) * theta);
}

void CrankNicolsonModes::build_active_operator()
{
    // Only couplings between active modes enter the operator. mode_pairs
    // stays grouped by row mode, in increasing order. Pairs that were already
    // active keep their coupling profiles (see refresh_couplings()).
    std::vector<int> old_index(static_cast<std::size_t>(n_modes) * n_modes, -1);
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
        old_index[mode_pairs[p].row_mode * n_modes + mode_pairs[p].col_mode] = static_cast<int>(p);
    mode_pairs.clear();
    coupling_source.clear();
    for (const auto &pair : all_mode_pairs)
        if (active_modes[pair.row_mode] && active_modes[pair.col_mode])
        {
            mode_pairs.push_back(pair);
            coupling_source.push_back(old_index[pair.row_mode * n_modes + pair.col_mode]);
        }
    couplings_stale = true;
    row_pair_start.assign(n_modes + 1, 0);
    for (const auto &pair : mode_pairs)
        ++row_pair_start[pair.row_mode + 1];
    for (int j = 0; j < n_modes; ++j)
        row_pair_start[j + 1] += row_pair_start[j];

    active_list.clear();
    mode_position.assign(n_modes, -1);
    for (int j = 0; j < n_modes; ++j)
        if (active_modes[j])
        {
            mode_position[j] = static_cast<int>(active_list.size());
            active_list.push_back(j);
        }
    self_pair.assign(n_modes, -1);
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
        if (mode_pairs[p].row_mode == mode_pairs[p].col_mode)
            self_pair[mode_pairs[p].row_mode] = static_cast<int>(p);

    // H0 is tridiagonal with a constant off-diagonal; its boundary rows are empty.
    h0_diagonal.assign(static_cast<std::size_t>(n_modes) * Nr, 0.0);
    for (int j = 0; j < n_modes; ++j)
        for (int i = 1; i < Nr - 1; ++i)
            h0_diagonal[j * Nr + i] = H0.coeff(j * Nr + i, j * Nr + i).real();
    h0_offdiagonal = Nr > 2 ? H0.coeff(1, 0).real() : 0.0;
    // On a stretched grid the off-diagonals vary along r (but not with the mode).
    h0_lower.assign(Nr, 0.0);
    h0_upper.assign(Nr, 0.0);
    for (int i = 1; i < Nr - 1; ++i)
    {
        h0_lower[i] = H0.coeff(i, i - 1).real();
        h0_upper[i] = H0.coeff(i, i + 1).real();
    }

    pair_phase.assign(mode_pairs.size(), 0.0);
    if (absorption)
    {
        absorption_profile1 = {};
        absorption_profile2 = {};
        mode00 = -1;
        for (int j = 0; j < n_modes; ++j)
            if (modes[j].l == 0 && modes[j].m == 0)
                mode00 = j;
        sink1 = {};
        sink2 = {};
        // The dipolar reference profiles cover every mode and survive a change of the active set.
        for (SwaveSink &sink : dipole_sinks)
            sink = {};
    }
    radial_f1.resize((2 * l_max + 1) * Nr);
    radial_f2.resize(radial_f1.size());
    operator_ready = true;
}

void CrankNicolsonModes::refresh_couplings()
{
    const std::size_t inner = Nr - 2;
    const std::size_t size = mode_pairs.size() * inner;
    const bool moved = coupling_R1 != R1 || coupling_R2 != R2 || coupling_q != q;
    if (moved)
    {
        coupling_R1 = R1;
        coupling_R2 = R2;
        coupling_q = q;
        // Radial profiles are independent of the mode pair and orbital phase.
#pragma omp parallel for schedule(static) if (n_modes >= 8)
        for (int l = 0; l <= 2 * l_max; ++l)
            for (int i = 1; i < Nr - 1; ++i)
            {
                radial_f1[l * Nr + i] = F1(r[i], l, 0);
                radial_f2[l * Nr + i] = F2(r[i], l, 0);
            }
    }
    if (moved || couplings_stale)
    {
        // On a fixed orbit, pairs that stay active copy their old profiles.
        std::vector<double> next1(size), next2(size);
#pragma omp parallel for schedule(static) if (n_modes >= 8)
        for (std::size_t p = 0; p < mode_pairs.size(); ++p)
        {
            const int source = moved ? -1 : coupling_source[p];
            if (source >= 0)
            {
                std::copy_n(coupling1.begin() + source * inner, inner, next1.begin() + p * inner);
                std::copy_n(coupling2.begin() + source * inner, inner, next2.begin() + p * inner);
                continue;
            }
            const auto &terms = mode_pairs[p].terms;
            for (int i = 1; i < Nr - 1; ++i)
            {
                double c1 = 0.0, c2 = 0.0;
                for (const auto &term : terms)
                {
                    c1 += term.weight * radial_f1[term.l_star * Nr + i];
                    c2 += term.weight * radial_f2[term.l_star * Nr + i];
                }
                next1[p * inner + i - 1] = c1;
                next2[p * inner + i - 1] = c2;
            }
        }
        coupling1.swap(next1);
        coupling2.swap(next2);
        couplings_stale = false;
        coupling_eff.clear();
    }

    // The companion potential follows the startup ramp unless only the
    // orbital frequency is ramped; after the ramp one profile per pair remains.
    if (coupling_eff.size() != size || coupling_eff_ramp != potential_ramp_factor)
    {
        coupling_eff_ramp = potential_ramp_factor;
        coupling_eff.resize(size);
#pragma omp parallel for schedule(static) if (n_modes >= 8)
        for (std::size_t p = 0; p < mode_pairs.size(); ++p)
        {
            const double weight = (mode_pairs[p].m_star % 2 == 0 ? 1.0 : -1.0) * potential_ramp_factor;
            for (std::size_t n = p * inner; n < (p + 1) * inner; ++n)
                coupling_eff[n] = coupling1[n] + weight * coupling2[n];
        }
    }
}

void CrankNicolsonModes::build_static_perturbation()
{
    refresh_couplings();
    // The potential of each pair is pair_phase * coupling_eff(r).
    potential_state = {theta, mu * (M1 + M2), potential_ramp_factor, Omega};
    std::vector<dcomplex> phases(4 * l_max + 1);
    for (int m = -2 * l_max; m <= 2 * l_max; ++m)
        phases[m + 2 * l_max] = potential_state.strength * orbital_phase(m);
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
        pair_phase[p] = phases[mode_pairs[p].m_star + 2 * l_max];
}

SparseComplex CrankNicolsonModes::potential_matrix(double w1, double w2, bool derivative) const
{
    const PotentialState &state = potential_state;
    const double sense = counter ? -1.0 : 1.0;
    const std::size_t inner = Nr - 2;
    std::vector<Eigen::Triplet<dcomplex>> entries;
    entries.reserve(mode_pairs.size() * inner);
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
    {
        const auto &pair = mode_pairs[p];
        dcomplex factor = state.strength * std::exp(I * static_cast<double>(pair.m_star) * state.theta);
        if (derivative)
            factor *= sense * I * static_cast<double>(pair.m_star) * state.Omega;
        const double parity = pair.m_star % 2 == 0 ? 1.0 : -1.0;
        for (int i = 1; i < Nr - 1; ++i)
        {
            const std::size_t n = p * inner + i - 1;
            entries.emplace_back(pair.row_mode * Nr + i, pair.col_mode * Nr + i,
                                 factor * (w1 * coupling1[n] + w2 * parity * coupling2[n]));
        }
    }
    SparseComplex result(Nr * n_modes, Nr * n_modes);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}

SparseComplex CrankNicolsonModes::V1() const { return potential_matrix(1.0, 0.0, false); }

SparseComplex CrankNicolsonModes::V2() const { return potential_matrix(0.0, 1.0, false); }

SparseComplex CrankNicolsonModes::V() const { return potential_matrix(1.0, potential_state.ramp, false); }

SparseComplex CrankNicolsonModes::dVdt() const
{
    return potential_matrix(1.0, potential_state.ramp, true);
}

SparseComplex CrankNicolsonModes::H() const
{
    SparseComplex result = H0 + V();
    if (sg.enabled)
        result += self_gravity_matrix();
    if (cap)
        result += CAP;
    return result;
}

void CrankNicolsonModes::build_time_dependent_perturbation(double time)
{
    ramp_factor = 1.0;
    if (isolated_black_hole)
    {
        Omega = 0.0;
    }
    else if (time < ramp_end_time())
    {
        const double x = time / (ramp_end_time());
        ramp_factor = 3.0 * x * x - 2.0 * x * x * x;
        Omega = Omega_0 * ramp_factor;
    }
    else
    {
        Omega = std::sqrt((M1 + M2) / std::pow(R, 3));
    }
    potential_ramp_factor = ramp_secondary_potential ? ramp_factor : 1.0;
    build_static_perturbation();
    if (absorption)
        build_spherical_absorption();

    // The horizon sinks are rank one each and are applied outside H.
    sink_active = absorption && time >= ramp_end_time();
}
