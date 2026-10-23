/**
 * @file CrankNicolsonModes_extraction.cpp
 * @brief Diagnostics: fluxes at the extraction radius, energies, norms and binary power.
 */
#include "CrankNicolsonModes.hpp"
#include <vector>

double CrankNicolsonModes::restricted_expectation(
    const Eigen::VectorXcd &psi,
    const SparseComplex &A,
    int i_cut) const
{
    const Eigen::VectorXcd Apsi = A * psi;

    dcomplex result = 0.0;

    for (int mode_idx = 0; mode_idx < n_modes; ++mode_idx)
    {
        const int offset = mode_idx * Nr;

        for (int i = 0; i <= i_cut; ++i)
        {
            const int k = offset + i;
            result += std::conj(psi[k]) * Apsi[k];
        }
    }

    return result.real() * dr;
}

std::vector<double> CrankNicolsonModes::get_flux(const Eigen::VectorXcd &psi)
{
    std::vector<double> fluxes(n_modes, 0.0);

#pragma omp parallel for schedule(static) if (n_modes >= 8 && Nr >= 128)
    for (int i = 0; i < n_modes; i++)
    {
        const dcomplex dU_dr =
            (psi[idx_ex + 1 + i * Nr] - psi[idx_ex - 1 + i * Nr]) / (2.0 * dr) - psi[idx_ex + i * Nr] / r[idx_ex];

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
        const dcomplex dU_dr =
            (psi[idx_ex + 1 + i * Nr] - psi[idx_ex - 1 + i * Nr]) / (2.0 * dr) - psi[idx_ex + i * Nr] / r[idx_ex];

        const dcomplex dU_dt = dpsi_dt[idx_ex + i * Nr];

        fluxes[i] = (std::conj(dU_dt) * dU_dr).real() / mu;
    }

    return fluxes;
}

double CrankNicolsonModes::get_binary_power(const Eigen::VectorXcd &psi)
{
    return restricted_expectation(psi, dVdt, idx_ex);
}

double CrankNicolsonModes::get_dVexpectation_dt(const Eigen::VectorXcd &psi, const Eigen::VectorXcd &dpsi_dt)
{
    const Eigen::VectorXcd Vpsi = V * psi;

    dcomplex result = 0.0;

    for (int mode_idx = 0; mode_idx < n_modes; ++mode_idx)
    {
        const int offset = mode_idx * Nr;

        for (int i = 0; i <= idx_ex; ++i)
        {
            const int k = offset + i;
            result += std::conj(dpsi_dt[k]) * Vpsi[k];
        }
    }

    return result.real() * dr;
}

std::vector<double> CrankNicolsonModes::get_norms(const Eigen::VectorXcd &psi)
{
    std::vector<double> norms(n_modes);

#pragma omp parallel for schedule(static) if (n_modes >= 8 && Nr >= 128)
    for (int i = 0; i < n_modes; i++)
    {
        const int off = i * Nr;
        auto psi_j = psi.segment(off, idx_ex);
        norms[i] = psi_j.squaredNorm() * dr;
    }

    return norms;
}

double CrankNicolsonModes::get_energy(const Eigen::VectorXcd &psi)
{
    return restricted_expectation(psi, H, idx_ex);
}

std::vector<double> CrankNicolsonModes::get_energy_per_mode(const Eigen::VectorXcd &psi)
{
    Eigen::VectorXcd y = H * psi;

    std::vector<double> Ej(n_modes, 0.0);

#pragma omp parallel for schedule(static) if (n_modes >= 8 && Nr >= 128)
    for (int j = 0; j < n_modes; ++j)
    {
        const int off = j * Nr;

        auto psi_j = psi.segment(off, idx_ex);
        auto y_j = y.segment(off, idx_ex);

        const dcomplex E_j = psi_j.dot(y_j);

        Ej[j] = E_j.real() * dr;
    }
    return Ej;
}
