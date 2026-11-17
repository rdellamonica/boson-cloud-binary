/**
 * @file CrankNicolsonModes_outgoing.cpp
 * @brief Outer-boundary treatments: damping mask and complex absorbing potential.
 */
#include "CrankNicolsonModes.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void CrankNicolsonModes::build_damping_mask(double mask_L_factor, double mask_W, int mask_n)
{
    std::cout << "Building damping mask (L = " << mask_L_factor << " | W = " << mask_W << " | n = " << mask_n << ")" << std::endl;

    damping_mask.resize(Nr * n_modes);

    mask = true;
    int idx_mask = node_at(L * mask_L_factor);
    double r_mask = mask_L_factor * L;
    double L_mask = (1 - mask_L_factor) * L;

    for (int i = 0; i < Nr; i++)
    {
        for (int n = 0; n < n_modes; n++)
        {
            if (i >= idx_mask)
            {
                damping_mask[i + n * Nr] = std::exp(-mask_W * (std::pow((r(i) - r_mask) / L_mask, mask_n)));
            }
            else
            {
                damping_mask[i + n * Nr] = 1.0;
            }
        }
    }
}

void CrankNicolsonModes::build_cap(double cap_L_factor, double cap_W, int cap_n)
{
    std::cout << "Building CAP (L = " << cap_L_factor << " | W = " << cap_W << " | n = " << cap_n << ")" << std::endl;

    std::vector<Eigen::Triplet<dcomplex>> CAPt;

    CAP.resize(Nr * n_modes, Nr * n_modes);

    cap = true;
    int idx_cap = node_at(L * cap_L_factor);
    double r_cap = cap_L_factor * L;
    double L_cap = (1 - cap_L_factor) * L;

    for (int i = 0; i < Nr; i++)
    {
        for (int n = 0; n < n_modes; n++)
        {
            if (i >= idx_cap)
            {
                CAPt.emplace_back(i + n * Nr, i + n * Nr, -I * cap_W * (std::pow((r(i) - r_cap) / (r(i) - L_cap), cap_n)));
            }
        }
    }

    CAP.setFromTriplets(CAPt.begin(), CAPt.end());
    cap_diagonal = CAP.diagonal();

    // The profile, for a per-mode strength (enable_cap_per_mode()).
    cap_W_uniform = cap_W;
    cap_start = std::max(idx_cap, 0);
    cap_shape = Eigen::VectorXd::Zero(Nr);
    for (int i = cap_start; i < Nr; ++i)
        cap_shape[i] = std::max(0.0, std::pow((r(i) - r_cap) / (r(i) - L_cap), cap_n));
    cap_per_mode = false;
    cap_W_mode.resize(0);
}

void CrankNicolsonModes::enable_cap_per_mode(double reflection, int tune_steps, double k_min)
{
    if (!cap)
        throw std::logic_error("enable_cap_per_mode() needs build_cap() first");
    if (!(reflection > 0.0 && reflection < 1.0))
        throw std::invalid_argument("[cap] reflection must lie in (0, 1)");
    if (tune_steps < 1)
        throw std::invalid_argument("[cap] tune_steps must be at least 1");
    // The layer's interior nodes (the last node is the Dirichlet wall).
    double width = 0.0;
    bool any = false;
    for (int i = cap_start; i < Nr - 1; ++i)
        any = any || cap_shape[i] > 0.0;
    if (cap_start < Nr - 1)
        width = r(Nr - 1) - r(cap_start);
    if (!any || !(width > 0.0))
        throw std::invalid_argument("A per-mode CAP needs a layer: set [cap] L_factor below 1");

    cap_per_mode = true;
    cap_reflection = reflection;
    cap_tune_steps = tune_steps;
    cap_k_min = k_min > 0.0 ? k_min : 1.0 / width;
    cap_energy = Eigen::VectorXd::Zero(n_modes);
    cap_has_energy.assign(static_cast<std::size_t>(n_modes), 0);
    cap_W_mode = Eigen::VectorXd::Zero(n_modes);
    checkpoint_has_cap_energy = false;
    std::cout << "Per-mode CAP: reflection " << reflection << ", energies averaged over " << tune_steps
              << " steps, k_min " << cap_k_min << " (W of [cap] not used)" << std::endl;
    cap_warned.assign(static_cast<std::size_t>(n_modes), 0);
    update_cap_strengths();
}

double CrankNicolsonModes::cap_tuned_energy(int j) const
{
    if (!cap_per_mode || !cap_has_energy[static_cast<std::size_t>(j)])
        return std::numeric_limits<double>::quiet_NaN();
    return cap_energy[j];
}

double CrankNicolsonModes::cap_strength_for_energy(int j, double E) const
{
    // Round trip through the layer: |reflection| = exp(-2 mu W I), with Im k = mu W s / k
    // to first order in W and I = int s / k dr over the layer (local wavenumber k(r)).
    const double l = modes[j].l;
    const double mass = M1 + (isolated_black_hole ? 0.0 : potential_state.ramp * M2);
    double integral = 0.0;
    for (int i = cap_start; i < Nr - 1; ++i)
    {
        if (cap_shape[i] <= 0.0)
            continue;
        const double ri = r(i);
        const double V = l * (l + 1.0) / (2.0 * mu * ri * ri) - mu * mass / ri;
        const double kinetic = E - V;
        // Im k = mu W s / k_g with k_g = mu x group velocity of the discrete Laplacian at the
        // local spacing h: E - V = (1 - cos(k h)) / (mu h^2), so k_g = sin(k h) / h (sqrt(2 mu K)
        // when resolved). Above the band (cos(k h) < -1) take the largest, 1 / h.
        const double h = r(i + 1) - r(i);
        const double c = 1.0 - mu * h * h * kinetic;
        double k = cap_k_min;
        if (kinetic > 0.0)
            k = std::max(cap_k_min, c > -1.0 ? std::sqrt(1.0 - c * c) / h : 1.0 / h);
        integral += node_weight(i) * cap_shape[i] / k;
    }
    return std::log(1.0 / cap_reflection) / (2.0 * mu * integral);
}

void CrankNicolsonModes::update_cap_strengths()
{
    if (!cap_per_mode)
        return;

    // Modes with density in the layer but no estimate start from its Rayleigh quotient:
    // Re<u, H u> / <u, u> over the layer (the CAP's part is imaginary and drops out).
    bool need_rayleigh = false;
    for (int j = 0; j < n_modes; ++j)
    {
        if (!active_modes.empty() && !active_modes[j])
            cap_has_energy[j] = 0; // re-estimated when the mode comes back
        else if (!cap_has_energy[j])
            need_rayleigh = true;
    }
    if (need_rayleigh && operator_ready && psi.size() == static_cast<Eigen::Index>(Nr) * n_modes)
    {
        const Eigen::VectorXcd Hpsi = hamiltonian_times(psi);
        for (int j = 0; j < n_modes; ++j)
        {
            if (cap_has_energy[j] || (!active_modes.empty() && !active_modes[j]))
                continue;
            double num = 0.0, den = 0.0;
            for (int i = cap_start; i < Nr - 1; ++i)
            {
                const Eigen::Index n = static_cast<Eigen::Index>(j) * Nr + i;
                num += node_weight(i) * (std::conj(psi[n]) * Hpsi[n]).real();
                den += node_weight(i) * std::norm(psi[n]);
            }
            if (den > 0.0 && std::isfinite(num / den))
            {
                cap_energy[j] = num / den;
                cap_has_energy[j] = 1;
            }
        }
    }

    std::vector<Eigen::Triplet<dcomplex>> triplets;
    triplets.reserve(static_cast<std::size_t>(n_modes) * (Nr - cap_start));
    for (int j = 0; j < n_modes; ++j)
    {
        // Without an estimate (nothing in the layer yet): the slowest wave, k = k_min.
        const double E = cap_has_energy[j] ? cap_energy[j] : -std::numeric_limits<double>::infinity();
        cap_W_mode[j] = cap_strength_for_energy(j, E);
        for (int i = 0; i < Nr; ++i)
        {
            const Eigen::Index n = static_cast<Eigen::Index>(j) * Nr + i;
            cap_diagonal[n] = -I * cap_W_mode[j] * cap_shape[i];
            if (i >= cap_start && cap_shape[i] > 0.0)
                triplets.emplace_back(n, n, cap_diagonal[n]);
        }
    }
    CAP.setZero();
    CAP.setFromTriplets(triplets.begin(), triplets.end());
}

void CrankNicolsonModes::measure_cap_energies()
{
    if (!cap_per_mode)
        return;
    // Over one step a wave of energy E advances by exp(-i E_ref dt) (1 - i x) / (1 + i x),
    // x = (E - E_ref) dt / 2 (CN in the frame rotating at E_ref), so the density-weighted
    // phase of conj(psi_n) psi_n+1 over the layer gives E exactly for one frequency.
    const double alpha = 1.0 / cap_tune_steps;
    const dcomplex unrotate = std::exp(I * (reference_energy * dt));
    for (int j = 0; j < n_modes; ++j)
    {
        if (!active_modes.empty() && !active_modes[j])
            continue;
        dcomplex z = 0.0;
        for (int i = cap_start; i < Nr - 1; ++i)
        {
            const Eigen::Index n = static_cast<Eigen::Index>(j) * Nr + i;
            z += node_weight(i) * std::conj(psi_last[n]) * psi[n];
        }
        if (!(std::abs(z) > 0.0) || !std::isfinite(std::abs(z)))
            continue;
        const double phi = std::arg(z * unrotate);
        if (std::abs(phi) > 0.99 * M_PI)
            continue; // aliased: the step is too long to resolve this frequency
        const double E = reference_energy - 2.0 * std::tan(0.5 * phi) / dt;
        if (cap_has_energy[j])
            cap_energy[j] += alpha * (E - cap_energy[j]);
        else
        {
            cap_energy[j] = E;
            cap_has_energy[j] = 1;
        }
        // No absorber can take a wave with only a few wavelengths across the layer without
        // reflecting part of it: say so once per mode, when the mode carries a real wave there.
        if (!cap_warned[j] && cap_start > 0)
        {
            const double l = modes[j].l, re = r(cap_start);
            const double mass = M1 + (isolated_black_hole ? 0.0 : potential_state.ramp * M2);
            const double kinetic = E - (l * (l + 1.0) / (2.0 * mu * re * re) - mu * mass / re);
            double layer = 0.0, total = 0.0;
            for (int i = 0; i < Nr; ++i)
            {
                const double d = node_weight(i) * std::norm(psi[static_cast<Eigen::Index>(j) * Nr + i]);
                total += d;
                if (i >= cap_start)
                    layer += d;
            }
            const double k_width = std::sqrt(2.0 * mu * std::max(kinetic, 0.0)) * (r(Nr - 1) - re);
            if (kinetic > 0.0 && k_width < 30.0 && layer > 1e-3 * total)
            {
                std::cout << "\nPer-mode CAP: mode (" << modes[j].l << ", " << modes[j].m << ") has k * width = "
                          << k_width << " in the layer; widen it (k * width >~ 50) to absorb this wave fully."
                          << std::endl;
                cap_warned[j] = 1;
            }
        }
    }
}