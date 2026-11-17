/**
 * @file CrankNicolsonModes_pruning.cpp
 * @brief Adaptive selection of the evolved modes.
 */
#include "CrankNicolsonModes.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{
/// An active mode is switched off only well below the switch-on tolerance, so
/// a mode near the threshold does not flip at every check.
constexpr double switch_off_fraction = 0.01;
} // namespace

double CrankNicolsonModes::block_norm2(const dcomplex *u) const
{
    // On a uniform grid the constant dr cancels in every share.
    if (uniform_grid)
        return Eigen::Map<const Eigen::VectorXcd>(u, Nr).squaredNorm();
    return radial_norm2(u, 0, Nr);
}

double CrankNicolsonModes::state_norm2(const Eigen::VectorXcd &state) const
{
    if (uniform_grid)
        return state.squaredNorm();
    double sum = 0.0;
    for (int j = 0; j < n_modes; ++j)
        sum += block_norm2(state.data() + static_cast<Eigen::Index>(j) * Nr);
    return sum;
}

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
        phases[m + 2 * l_max] = mu * (M1 + M2) * orbital_phase(m);

    // all_mode_pairs is grouped by row mode.
    std::vector<std::size_t> row_start(n_modes + 1, all_mode_pairs.size());
    for (std::size_t p = all_mode_pairs.size(); p-- > 0;)
        row_start[all_mode_pairs[p].row_mode] = p;
    for (int j = n_modes - 1; j >= 0; --j)
        row_start[j] = std::min(row_start[j], row_start[j + 1]);

    const double norm = state_norm2(psi);
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
            const double companion = (pair.m_star % 2 == 0 ? 1.0 : -1.0) * potential_ramp_factor;
            const dcomplex *psi_k = psi.data() + static_cast<Eigen::Index>(pair.col_mode) * Nr;
            const int l = modes[j].l, l1 = modes[pair.col_mode].l;
            const double centrifugal = std::abs(l * (l + 1.0) - l1 * (l1 + 1.0)) / (2.0 * mu);
            for (int i = 1; i < Nr - 1; ++i)
            {
                // The response to an off-resonant source saturates after a
                // time ~ 1/dE, with dE the local mismatch of the centrifugal
                // energies; within the horizon it grows linearly.
                const double mismatch = centrifugal / (r[i] * r[i]);
                const double time = mismatch * horizon > 1.0 ? 1.0 / mismatch : horizon;
                // V_jk(r) = phase * sum_l* weight * (F1 + parity * ramp * F2), as in V.
                double coupling = 0.0;
                for (const auto &term : pair.terms)
                    coupling += term.weight * (radial_f1[term.l_star * Nr + i] +
                                               companion * radial_f2[term.l_star * Nr + i]);
                Vpsi[i] += (time * coupling) * (phase * psi_k[i]);
            }
        }
        drive[j] = block_norm2(Vpsi.data()) / norm;
    }
    return drive;
}

std::vector<double> CrankNicolsonModes::mode_ionization_flux()
{
    std::vector<double> flux(n_modes, 0.0);
    const double norm = state_norm2(psi);
    if (norm == 0.0)
        return flux;

    // Sources: the active modes holding more than the tolerance, each at its
    // current frequency.
    const Eigen::VectorXcd Hpsi = hamiltonian_times(psi);
    std::vector<double> omega(n_modes, 0.0);
    std::vector<char> source(n_modes, 0);
    for (int k = 0; k < n_modes; ++k)
    {
        const double nk = block_norm2(psi.data() + k * Nr);
        if (!active_modes[k] || nk <= prune_tolerance * norm)
            continue;
        source[k] = 1;
        // Rayleigh quotient in the grid's inner product.
        const dcomplex numerator =
            uniform_grid ? psi.segment(k * Nr, Nr).dot(Hpsi.segment(k * Nr, Nr))
                         : psi.segment(k * Nr, Nr).cwiseProduct(node_weight.cast<dcomplex>()).dot(Hpsi.segment(k * Nr, Nr));
        omega[k] = numerator.real() / nk;
    }

    // Monopole of the binary, the potential the ionized field moves in.
    std::vector<double> V0(Nr, 0.0);
    for (int i = 1; i < Nr - 1; ++i)
        V0[i] = -mu * (M1 + M2) * (radial_f1[i] + potential_ramp_factor * radial_f2[i]);

    std::vector<dcomplex> phases(4 * l_max + 1);
    for (int m = -2 * l_max; m <= 2 * l_max; ++m)
        phases[m + 2 * l_max] = mu * (M1 + M2) * orbital_phase(m);
    const double sense = counter ? -1.0 : 1.0;

#pragma omp parallel for schedule(dynamic) if (n_modes >= 8)
    for (int j = 0; j < n_modes; ++j)
    {
        const int l = modes[j].l;
        std::vector<double> phi(Nr);
        for (const auto &pair : all_mode_pairs)
        {
            const int k = pair.col_mode;
            if (pair.row_mode != j || k == j || !source[k])
                continue;
            // The coupling oscillates as exp(i m* theta): the source of mode j
            // has energy omega_k - s m* Omega, and radiates above zero.
            const double E = omega[k] - sense * pair.m_star * Omega;
            if (!(E > 0.0))
                continue;
            // Regular solution of u'' = -g u with g = 2 mu (E - V_eff): by
            // Numerov on a uniform grid, by the three-point recurrence otherwise.
            auto g = [&](int i) { return 2.0 * mu * (E - V0[i]) - l * (l + 1.0) / (r[i] * r[i]); };
            const double h2 = dr * dr / 12.0;
            phi[0] = 0.0;
            phi[1] = std::pow(uniform_grid ? dr : r[1] - r[0], l + 1);
            double g_prev = 0.0, g_cur = g(1);
            for (int i = 1; i < Nr - 2; ++i)
            {
                const double g_next = g(i + 1);
                if (uniform_grid)
                    phi[i + 1] = (2.0 * phi[i] * (1.0 - 5.0 * h2 * g_cur) - phi[i - 1] * (1.0 + h2 * g_prev)) /
                                 (1.0 + h2 * g_next);
                else
                {
                    const double hm = r[i] - r[i - 1], hp = r[i + 1] - r[i];
                    phi[i + 1] = phi[i] + hp * ((phi[i] - phi[i - 1]) / hm - 0.5 * (hm + hp) * g_cur * phi[i]);
                }
                g_prev = g_cur;
                g_cur = g_next;
                if (std::abs(phi[i + 1]) > 1e100)
                    for (int s = 0; s <= i + 1; ++s)
                        phi[s] *= 1e-100;
            }
            // Unit amplitude at infinity, from the WKB amplitude near the edge.
            const int e = Nr - 3;
            const double ge = g(e);
            if (!(ge > 0.0))
                continue; // still classically forbidden inside the box
            const double ke = std::sqrt(ge), k_inf = std::sqrt(2.0 * mu * E);
            const double dphi = uniform_grid ? (phi[e + 1] - phi[e - 1]) / (2.0 * dr)
                                             : (phi[e + 1] - phi[e - 1]) / (r[e + 1] - r[e - 1]);
            const double amplitude = std::sqrt((phi[e] * phi[e] + dphi * dphi / ge) * ke / k_inf);
            if (!(amplitude > 0.0))
                continue;

            // The pair's phase is constant along r and factors out of the overlap.
            const double companion = (pair.m_star % 2 == 0 ? 1.0 : -1.0) * potential_ramp_factor;
            dcomplex overlap = 0.0;
            for (int i = 1; i < Nr - 1; ++i)
            {
                double coupling = 0.0;
                for (const auto &term : pair.terms)
                    coupling += term.weight * (radial_f1[term.l_star * Nr + i] +
                                               companion * radial_f2[term.l_star * Nr + i]);
                overlap += (phi[i] * coupling * (uniform_grid ? 1.0 : node_weight(i))) * psi[k * Nr + i];
            }
            overlap *= phases[pair.m_star + 2 * l_max] * ((uniform_grid ? dr : 1.0) / amplitude);
            flux[j] += 4.0 * mu * mu / k_inf * std::norm(overlap);
        }
    }
    return flux;
}

bool CrankNicolsonModes::update_active_modes(double horizon, bool from_scratch)
{
    const double norm = state_norm2(psi);
    if (!pruning_enabled() || norm == 0.0)
        return false;

    const std::vector<char> before = active_modes;
    // From scratch, every mode is a candidate: the active set is the modes that
    // hold the state plus the ones they drive.
    if (from_scratch)
        for (int j = 0; j < n_modes; ++j)
            active_modes[j] = block_norm2(psi.data() + j * Nr) > prune_tolerance * norm;

    // A mode two couplings away from the state is not driven until the mode in
    // between is populated; it is switched on at a later check, when it has only
    // reached a share of order tolerance^2.
    // Each mode's share of the ionization flux driven by the active modes,
    // and, once run() has set the extraction radius, of the flux measured
    // through r_ex.
    const double flux_tol = prune_flux_tolerance;
    const std::vector<double> drive = mode_drive(horizon);
    std::vector<double> flux_share(n_modes, 0.0);
    if (flux_tol > 0.0)
    {
        const std::vector<double> predicted = mode_ionization_flux();
        double total = 0.0;
        for (double f : predicted)
            total += f;
        if (total > 0.0)
            for (int j = 0; j < n_modes; ++j)
                flux_share[j] = predicted[j] / total;
        if (idx_ex >= 1 && idx_ex < Nr - 1)
        {
            const std::vector<double> measured = get_flux(psi);
            double sum = 0.0;
            for (double f : measured)
                sum += std::abs(f);
            if (sum > 0.0)
                for (int j = 0; j < n_modes; ++j)
                    flux_share[j] = std::max(flux_share[j], std::abs(measured[j]) / sum);
        }
    }
    for (int j = 0; j < n_modes; ++j)
    {
        const double relevance = std::max(block_norm2(psi.data() + j * Nr) / norm, drive[j]);
        const double flux_relevance = flux_tol > 0.0 ? flux_share[j] / flux_tol : 0.0;
        if (!active_modes[j] && (relevance > prune_tolerance || flux_relevance > 1.0))
            active_modes[j] = 1;
        else if (active_modes[j] && relevance < switch_off_fraction * prune_tolerance &&
                 flux_relevance < switch_off_fraction)
            active_modes[j] = 0;
    }

    // Inactive modes are held at exactly zero.
    double dropped = 0.0;
    for (int j = 0; j < n_modes; ++j)
        if (!active_modes[j])
        {
            dropped += block_norm2(psi.data() + j * Nr);
            if (idx_ex > 0 && modes[j].m != 0)
                Lz_budget.pruned += modes[j].m * radial_norm2(psi.data() + j * Nr, 0, idx_ex);
            psi.segment(j * Nr, Nr).setZero();
        }
    pruned_mass_total += mu * (uniform_grid ? dr : 1.0) * dropped;

    if (active_modes == before)
        return false;
    std::cout << "\nActive modes: " << active_mode_count() << "/" << n_modes << " at t = " << t_i
              << " (pruned mass so far " << pruned_mass_total << ")" << std::endl;
    return true;
}

void CrankNicolsonModes::write_active_modes(int step_id)
{
    if (!active_modes_file.is_open())
        return;
    active_modes_file << step_id << ' ' << std::scientific << std::setprecision(12) << t_i << ' '
                      << active_mode_count();
    for (const char active : active_modes)
        active_modes_file << ' ' << static_cast<int>(active);
    active_modes_file << std::endl; // flushed: changes are rare
}

void CrankNicolsonModes::apply_active_set()
{
    build_active_operator();
    // H(t_i) on the new pattern; Omega is part of the evolved state, so keep it.
    const double current_Omega = Omega;
    build_time_dependent_perturbation(t_i);
    Omega = current_Omega;
}
