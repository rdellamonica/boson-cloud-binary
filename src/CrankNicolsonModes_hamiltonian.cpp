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
            Ht[offset] = Eigen::Triplet<dcomplex>(j * Nr + i, j * Nr + i, -(-2.0 * alpha - modes[j].l * (modes[j].l + 1) / (r(i) * r(i))) / (2 * mu));
            Ht[offset + 1] = Eigen::Triplet<dcomplex>(j * Nr + i, j * Nr + i - 1, -alpha / (2 * mu));
            Ht[offset + 2] = Eigen::Triplet<dcomplex>(j * Nr + i, j * Nr + i + 1, -alpha / (2 * mu));
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
            ModePair pair{j, k, m_star, {}, {}};
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
                    pair.terms.push_back({l_star, angular});
            }
            if (!pair.terms.empty())
                rows[j].push_back(std::move(pair));
        }
    }
    mode_pairs.clear();
    for (auto &row : rows)
        for (auto &pair : row)
            mode_pairs.push_back(std::move(pair));
}

dcomplex CrankNicolsonModes::orbital_phase(int m_star) const
{
    return std::exp(I * static_cast<double>(m_star) * theta);
}

void CrankNicolsonModes::build_sparse_element_caches()
{
    // Construct the pattern only once, including entries whose current value is zero.
    std::vector<Eigen::Triplet<dcomplex>> pattern;
    pattern.reserve(mode_pairs.size() * (Nr - 2));
    for (const auto &pair : mode_pairs)
        for (int i = 1; i < Nr - 1; ++i)
            pattern.emplace_back(pair.row_mode * Nr + i, pair.col_mode * Nr + i, 0.0);
    V1.resize(Nr * n_modes, Nr * n_modes);
    V1.setFromTriplets(pattern.begin(), pattern.end());
    V1.makeCompressed();
    V2 = V1;
    V = V1;
    dVdt = V1;
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
    }
#pragma omp parallel for schedule(static) if (n_modes >= 8)
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
    {
        auto &pair = mode_pairs[p];
        pair.offsets.resize(Nr - 2);
        for (int i = 1; i < Nr - 1; ++i)
        {
            const int row = pair.row_mode * Nr + i;
            const int col = pair.col_mode * Nr + i;
            const auto begin = V1.innerIndexPtr() + V1.outerIndexPtr()[row];
            const auto end = V1.innerIndexPtr() + V1.outerIndexPtr()[row + 1];
            pair.offsets[i - 1] = std::lower_bound(begin, end, col) - V1.innerIndexPtr();
        }
    }
    // Keep one fixed union pattern for H, A, B, including boundary diagonals
    // and the diagonal CAP that can be enabled after initialize().
    Id.resize(Nr * n_modes, Nr * n_modes);
    Id.setIdentity();
    H = H0 + V + Id;
    H.makeCompressed();
    stationary_values.resize(H.nonZeros());
    potential_offsets.assign(H.nonZeros(), -1);
#pragma omp parallel for schedule(static) if (n_modes >= 8)
    for (int row = 0; row < H.rows(); ++row)
    {
        for (SparseComplex::InnerIterator it(H, row); it; ++it)
        {
            const auto slot = &it.valueRef() - H.valuePtr();
            stationary_values[slot] = H0.coeff(row, it.col());
            const auto begin = V.outerIndexPtr()[row];
            const auto end = V.outerIndexPtr()[row + 1];
            const auto found = std::lower_bound(V.innerIndexPtr() + begin, V.innerIndexPtr() + end, it.col());
            if (found != V.innerIndexPtr() + end && *found == it.col())
                potential_offsets[slot] = found - V.innerIndexPtr();
        }
    }
    radial_f1.resize((2 * l_max + 1) * Nr);
    radial_f2.resize(radial_f1.size());
}

void CrankNicolsonModes::build_static_perturbation()
{
    // Radial profiles are independent of the mode pair and orbital phase.
#pragma omp parallel for schedule(static) if (n_modes >= 8)
    for (int l = 0; l <= 2 * l_max; ++l)
        for (int i = 1; i < Nr - 1; ++i)
        {
            radial_f1[l * Nr + i] = F1(r[i], l, 0);
            radial_f2[l * Nr + i] = F2(r[i], l, 0);
        }
    std::vector<dcomplex> phases(4 * l_max + 1);
    for (int m = -2 * l_max; m <= 2 * l_max; ++m)
        phases[m + 2 * l_max] = orbital_phase(m);
    const double phase_sign = counter ? -1.0 : 1.0;

    // Every pair writes distinct compressed entries. Summation over l_star stays
    // in its original order, so scheduling does not change the arithmetic.
#pragma omp parallel for schedule(static) if (n_modes >= 8)
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
    {
        const auto &pair = mode_pairs[p];
        const dcomplex phase = phases[pair.m_star + 2 * l_max];
        const double parity = pair.m_star % 2 == 0 ? 1.0 : -1.0;
        for (int i = 1; i < Nr - 1; ++i)
        {
            dcomplex v1 = 0.0, v2 = 0.0, derivative = 0.0;
            for (const auto &term : pair.terms)
            {
                const dcomplex common = -mu * (M1 + M2) * 4.0 * M_PI /
                                        (2.0 * term.l_star + 1.0) * term.angular * phase;
                const dcomplex value1 = common * radial_f1[term.l_star * Nr + i];
                const dcomplex value2 = common * (parity * radial_f2[term.l_star * Nr + i]);
                v1 += value1;
                v2 += value2;
                derivative += (value1 + potential_ramp_factor * value2) * phase_sign * I *
                              static_cast<double>(pair.m_star) * Omega;
            }
            const auto offset = pair.offsets[i - 1];
            V1.valuePtr()[offset] = v1;
            V2.valuePtr()[offset] = v2;
            // The companion potential follows the startup ramp unless only
            // the orbital frequency is ramped.
            V.valuePtr()[offset] = v1 + v2 * potential_ramp_factor;
            dVdt.valuePtr()[offset] = derivative;
        }
    }
}

void CrankNicolsonModes::assemble_hamiltonian(bool include_absorption)
{
    // The s-wave sink is rank one and is applied outside the sparse H.
    sink_active = include_absorption;
#pragma omp parallel for schedule(static) if (n_modes >= 8)
    for (int row = 0; row < H.rows(); ++row)
    {
        for (SparseComplex::InnerIterator it(H, row); it; ++it)
        {
            const auto slot = &it.valueRef() - H.valuePtr();
            dcomplex value = stationary_values[slot];
            const auto potential_slot = potential_offsets[slot];
            if (potential_slot >= 0)
                value += V.valuePtr()[potential_slot];
            if (cap && it.col() == row)
                value += CAP.coeff(row, row);
            it.valueRef() = value;
        }
    }
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

    assemble_hamiltonian(absorption && time >= ramp_end_time());
}
