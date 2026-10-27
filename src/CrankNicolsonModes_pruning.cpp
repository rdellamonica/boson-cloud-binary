/**
 * @file CrankNicolsonModes_pruning.cpp
 * @brief Adaptive selection of the evolved modes.
 */
#include "CrankNicolsonModes.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{
/// An active mode is switched off only well below the switch-on tolerance, so
/// a mode near the threshold does not flip at every check.
constexpr double switch_off_fraction = 0.01;
} // namespace

int CrankNicolsonModes::active_mode_count() const
{
    return static_cast<int>(std::count(active_modes.begin(), active_modes.end(), 1));
}

std::vector<double> CrankNicolsonModes::mode_drive(double horizon) const
{
    // The amplitude each mode j receives from the active modes k != j over the
    // horizon, sum_k V_jk psi_k * min(horizon, 1/dE_jk), with the potential of
    // the last build_static_perturbation (same radial factors, phases and ramp).
    std::vector<dcomplex> phases(4 * l_max + 1);
    for (int m = -2 * l_max; m <= 2 * l_max; ++m)
        phases[m + 2 * l_max] = orbital_phase(m);

    // all_mode_pairs is grouped by row mode.
    std::vector<std::size_t> row_start(n_modes + 1, all_mode_pairs.size());
    for (std::size_t p = all_mode_pairs.size(); p-- > 0;)
        row_start[all_mode_pairs[p].row_mode] = p;
    for (int j = n_modes - 1; j >= 0; --j)
        row_start[j] = std::min(row_start[j], row_start[j + 1]);

    const double norm = psi.squaredNorm();
    std::vector<double> drive(n_modes, 0.0);
    if (norm == 0.0)
        return drive;

#pragma omp parallel for schedule(dynamic) if (n_modes >= 8)
    for (int j = 0; j < n_modes; ++j)
    {
        Eigen::VectorXcd Vpsi = Eigen::VectorXcd::Zero(Nr);
        for (std::size_t p = row_start[j]; p < row_start[j + 1]; ++p)
        {
            const auto &pair = all_mode_pairs[p];
            if (pair.col_mode == j || !active_modes[pair.col_mode])
                continue;
            const dcomplex phase = phases[pair.m_star + 2 * l_max];
            const double parity = pair.m_star % 2 == 0 ? 1.0 : -1.0;
            const int l = modes[j].l, l1 = modes[pair.col_mode].l;
            const double centrifugal = std::abs(l * (l + 1.0) - l1 * (l1 + 1.0)) / (2.0 * mu);
            for (int i = 1; i < Nr - 1; ++i)
            {
                // The response to an off-resonant source saturates after a
                // time ~ 1/dE, with dE the local mismatch of the centrifugal
                // energies; within the horizon it grows linearly.
                const double mismatch = centrifugal / (r[i] * r[i]);
                const double time = mismatch * horizon > 1.0 ? 1.0 / mismatch : horizon;
                dcomplex value = 0.0;
                for (const auto &term : pair.terms)
                    value += -mu * (M1 + M2) * 4.0 * M_PI / (2.0 * term.l_star + 1.0) * term.angular * phase *
                             (radial_f1[term.l_star * Nr + i] +
                              potential_ramp_factor * parity * radial_f2[term.l_star * Nr + i]);
                Vpsi[i] += time * value * psi[pair.col_mode * Nr + i];
            }
        }
        drive[j] = Vpsi.squaredNorm() / norm;
    }
    return drive;
}

bool CrankNicolsonModes::update_active_modes(double horizon, bool from_scratch)
{
    const double norm = psi.squaredNorm();
    if (!pruning_enabled() || norm == 0.0)
        return false;

    const std::vector<char> before = active_modes;
    // From scratch, every mode is a candidate: the active set is the modes that
    // hold the state plus the ones they drive.
    if (from_scratch)
        for (int j = 0; j < n_modes; ++j)
            active_modes[j] = psi.segment(j * Nr, Nr).squaredNorm() > prune_tolerance * norm;

    // A mode two couplings away from the state is not driven until the mode in
    // between is populated; it is switched on at a later check, when it has only
    // reached a share of order tolerance^2.
    const std::vector<double> drive = mode_drive(horizon);
    for (int j = 0; j < n_modes; ++j)
    {
        const double relevance = std::max(psi.segment(j * Nr, Nr).squaredNorm() / norm, drive[j]);
        if (!active_modes[j] && relevance > prune_tolerance)
        {
            active_modes[j] = 1;
        }
        else if (active_modes[j] && relevance < switch_off_fraction * prune_tolerance)
            active_modes[j] = 0;
    }

    // Inactive modes are held at exactly zero.
    double dropped = 0.0;
    for (int j = 0; j < n_modes; ++j)
        if (!active_modes[j])
        {
            dropped += psi.segment(j * Nr, Nr).squaredNorm();
            psi.segment(j * Nr, Nr).setZero();
        }
    pruned_mass_total += mu * dr * dropped;

    if (active_modes == before)
        return false;
    std::cout << "\nActive modes: " << active_mode_count() << "/" << n_modes << " at t = " << t_i
              << " (pruned mass so far " << pruned_mass_total << ")" << std::endl;
    return true;
}

void CrankNicolsonModes::apply_active_set()
{
    build_sparse_element_caches();
    // H(t_i) on the new pattern; Omega is part of the evolved state, so keep it.
    const double current_Omega = Omega;
    build_time_dependent_perturbation(t_i);
    Omega = current_Omega;
    A = H;
    B = H;
    update_CN_values();
}
