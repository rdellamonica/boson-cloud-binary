/**
 * @file CrankNicolsonModes_outgoing.cpp
 * @brief Outer-boundary treatments: damping mask and complex absorbing potential.
 */
#include "CrankNicolsonModes.hpp"
#include <cmath>
#include <iostream>
#include <vector>

void CrankNicolsonModes::build_damping_mask(double mask_L_factor, double mask_W, int mask_n)
{
    std::cout << "Building damping mask (L = " << mask_L_factor << " | W = " << mask_W << " | n = " << mask_n << ")" << std::endl;

    damping_mask.resize(Nr * n_modes);

    mask = true;
    int idx_mask = (int)std::floor(L * mask_L_factor / dr);
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
    int idx_cap = (int)std::floor(L * cap_L_factor / dr);
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
}