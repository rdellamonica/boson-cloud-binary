/**
 * @file CrankNicolsonModes_extraction.cpp
 * @brief Diagnostics: fluxes at the extraction radius, energies, norms and binary power.
 */
#include "CrankNicolsonModes.hpp"
#include <algorithm>
#include <vector>

double CrankNicolsonModes::restricted_expectation(
    const Eigen::VectorXcd &psi,
    const Eigen::VectorXcd &Apsi,
    int i_cut) const
{
    dcomplex result = 0.0;

    // Inactive modes are zero and add nothing.
    for (const int mode_idx : active_list)
    {
        const int offset = mode_idx * Nr;

        if (uniform_grid)
            for (int i = 0; i <= i_cut; ++i)
            {
                const int k = offset + i;
                result += std::conj(psi[k]) * Apsi[k];
            }
        else
            for (int i = 0; i <= i_cut; ++i)
            {
                const int k = offset + i;
                result += node_weight(i) * (std::conj(psi[k]) * Apsi[k]);
            }
    }

    return result.real() * (uniform_grid ? dr : 1.0);
}

void CrankNicolsonModes::binary_power_terms(const Eigen::VectorXcd &psi, const Eigen::VectorXcd *dpsi_dt,
                                            double &power, double &dV) const
{
    // dVdt = V * i s m_* Omega pair by pair, with m_* = m_col - m_row, and
    // Omega as in the last update of V.
    const dcomplex rotation = (counter ? -1.0 : 1.0) * I * potential_state.Omega;
    const int blocks = static_cast<int>(active_list.size());
    const std::size_t inner = Nr - 2;
    const int last = std::min(idx_ex, Nr - 2);
    // Per-block sums, added in a fixed order so that results (and checkpoints)
    // do not depend on the thread schedule.
    std::vector<double> power_sums(blocks, 0.0), dV_sums(blocks, 0.0);
#pragma omp parallel for schedule(dynamic) if (blocks >= 8)
    for (int b = 0; b < blocks; ++b)
    {
        const int j = active_list[b];
        const dcomplex *psi_j = psi.data() + static_cast<Eigen::Index>(j) * Nr;
        const dcomplex *dpsi_j = dpsi_dt ? dpsi_dt->data() + static_cast<Eigen::Index>(j) * Nr : nullptr;
        for (std::size_t p = row_pair_start[j]; p < row_pair_start[j + 1]; ++p)
        {
            const dcomplex *psi_k = psi.data() + static_cast<Eigen::Index>(mode_pairs[p].col_mode) * Nr;
            const double *c = coupling_eff.data() + p * inner - 1;
            const double *weight = uniform_grid ? nullptr : node_weight.data();
            // <psi_j| V_p psi_k> and <dpsi_j| V_p psi_k> inside r_ex.
            dcomplex psi_V = 0.0, dpsi_V = 0.0;
            for (int i = 1; i <= last; ++i)
            {
                const dcomplex value = (weight ? c[i] * weight[i] : c[i]) * psi_k[i];
                psi_V += std::conj(psi_j[i]) * value;
                if (dpsi_j)
                    dpsi_V += std::conj(dpsi_j[i]) * value;
            }
            power_sums[b] += (rotation * static_cast<double>(mode_pairs[p].m_star) * pair_phase[p] * psi_V).real();
            dV_sums[b] += (pair_phase[p] * dpsi_V).real();
        }
    }
    double power_sum = 0.0, dV_sum = 0.0;
    for (int b = 0; b < blocks; ++b)
    {
        power_sum += power_sums[b];
        dV_sum += dV_sums[b];
    }
    const double measure = uniform_grid ? dr : 1.0; // the weights are already in the sums
    power = power_sum * measure;
    dV = dV_sum * measure;
}

std::vector<double> CrankNicolsonModes::get_flux(const Eigen::VectorXcd &psi)
{
    std::vector<double> fluxes(n_modes, 0.0);

#pragma omp parallel for schedule(static) if (n_modes >= 8 && Nr >= 128)
    for (int i = 0; i < n_modes; i++)
    {
        const dcomplex dU_dr = radial_derivative(psi.data() + i * Nr, idx_ex) - psi[idx_ex + i * Nr] / r[idx_ex];

        fluxes[i] = (std::conj(psi[idx_ex + i * Nr]) * dU_dr).imag();
    }

    return fluxes;
}

std::vector<double> CrankNicolsonModes::get_energy_flux(const Eigen::VectorXcd &psi, const Eigen::VectorXcd &dpsi_dt)
{
    std::vector<double> fluxes(n_modes, 0.0);

#pragma omp parallel for schedule(static) if (n_modes >= 8 && Nr >= 128)
    for (int i = 0; i < n_modes; i++)
    {
        const dcomplex dU_dr = radial_derivative(psi.data() + i * Nr, idx_ex) - psi[idx_ex + i * Nr] / r[idx_ex];

        const dcomplex dU_dt = dpsi_dt[idx_ex + i * Nr];

        fluxes[i] = (std::conj(dU_dt) * dU_dr).real() / mu;
    }

    return fluxes;
}

double CrankNicolsonModes::get_binary_power(const Eigen::VectorXcd &psi)
{
    double power = 0.0, dV = 0.0;
    binary_power_terms(psi, nullptr, power, dV);
    return power;
}

double CrankNicolsonModes::get_dVexpectation_dt(const Eigen::VectorXcd &psi, const Eigen::VectorXcd &dpsi_dt)
{
    double power = 0.0, dV = 0.0;
    binary_power_terms(psi, &dpsi_dt, power, dV);
    return dV;
}

std::vector<double> CrankNicolsonModes::get_norms(const Eigen::VectorXcd &psi)
{
    std::vector<double> norms(n_modes);

#pragma omp parallel for schedule(static) if (n_modes >= 8 && Nr >= 128)
    for (int i = 0; i < n_modes; i++)
    {
        const int off = i * Nr;
        norms[i] = radial_norm2(psi.data() + off, 0, idx_ex);
    }

    return norms;
}

double CrankNicolsonModes::get_energy(const Eigen::VectorXcd &psi)
{
    return restricted_expectation(psi, hamiltonian_times(psi), idx_ex);
}

std::vector<double> CrankNicolsonModes::get_energy_per_mode(const Eigen::VectorXcd &psi)
{
    return energy_per_mode(psi, hamiltonian_times(psi));
}

std::vector<double> CrankNicolsonModes::energy_per_mode(const Eigen::VectorXcd &psi, const Eigen::VectorXcd &Hpsi) const
{
    std::vector<double> Ej(n_modes, 0.0);

#pragma omp parallel for schedule(static) if (n_modes >= 8 && Nr >= 128)
    for (int j = 0; j < n_modes; ++j)
    {
        if (!active_modes[j])
            continue; // held at zero
        const int off = j * Nr;

        auto psi_j = psi.segment(off, idx_ex);
        auto y_j = Hpsi.segment(off, idx_ex);

        if (uniform_grid)
        {
            const dcomplex E_j = psi_j.dot(y_j);
            Ej[j] = E_j.real() * dr;
            continue;
        }
        Ej[j] = psi_j.cwiseProduct(node_weight.head(idx_ex).cast<dcomplex>()).dot(y_j).real();
    }
    return Ej;
}

double CrankNicolsonModes::step_binary_power(const Eigen::VectorXcd &psi_old, double theta_shift) const
{
    // Same sum as binary_power_terms, on the pre-step state and at the step's midpoint angle.
    // The CN step with H(t_n+1) conserves <H(t_n+1)>, so its energy change is
    // <psi_n|V(t_n+1) - V(t_n)|psi_n>. Evaluating dV/dt on the post-step state at t_n+1 instead
    // biases the power by about (1/2) Omega^2 dt <d^2V/dtheta^2>, which for a cloud held by a
    // hole off the COM is a steady positive offset of first order in dt.
    const dcomplex rotation = (counter ? -1.0 : 1.0) * I * potential_state.Omega;
    const std::size_t inner = Nr - 2;
    const int last = std::min(idx_ex, Nr - 2);
    const int blocks = static_cast<int>(active_list.size());
    std::vector<double> sums(blocks, 0.0);
#pragma omp parallel for schedule(dynamic) if (blocks >= 8)
    for (int b = 0; b < blocks; ++b)
    {
        const int j = active_list[b];
        const dcomplex *psi_j = psi_old.data() + static_cast<Eigen::Index>(j) * Nr;
        for (std::size_t p = row_pair_start[j]; p < row_pair_start[j + 1]; ++p)
        {
            const dcomplex *psi_k = psi_old.data() + static_cast<Eigen::Index>(mode_pairs[p].col_mode) * Nr;
            const double *c = coupling_eff.data() + p * inner - 1;
            const double *weight = uniform_grid ? nullptr : node_weight.data();
            dcomplex psi_V = 0.0;
            for (int i = 1; i <= last; ++i)
                psi_V += std::conj(psi_j[i]) * ((weight ? c[i] * weight[i] : c[i]) * psi_k[i]);
            const double m_star = static_cast<double>(mode_pairs[p].m_star);
            sums[b] += (rotation * m_star * pair_phase[p] * std::exp(I * m_star * theta_shift) * psi_V).real();
        }
    }
    double total = 0.0;
    for (int b = 0; b < blocks; ++b)
        total += sums[b];
    return total * (uniform_grid ? dr : 1.0);
}
