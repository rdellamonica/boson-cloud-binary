/**
 * @file CrankNicolsonModes_evolution.cpp
 * @brief Orbital evolution: gravitational-wave decay, cloud back-reaction and accretion.
 */
#include "CrankNicolsonModes.hpp"
#include <cmath>

namespace
{
/// Quadrupole gravitational-wave decay rate of a circular orbit, dR/dt.
double dR_dt_GW(double R, double M1, double M2)
{
    return -(64.0 / 5.0) * M1 * M2 * (M1 + M2) / std::pow(R, 3);
}
} // namespace

double CrankNicolsonModes::orbit_ramp_factor(double time) const
{
    // Off during the startup ramp and the hold, then a smooth step over orbit_ramp_periods.
    const double start = orbit_start_time(), width = orbit_ramp_periods * initial_period;
    if (time < start)
        return 0.0;
    if (time >= start + width)
        return 1.0;
    const double x = (time - start) / width;
    return 3.0 * x * x - 2.0 * x * x * x;
}

void CrankNicolsonModes::update_time_dependent_increment(double orbit_dt)
{
    // Accretion rates are evaluated on the midpoint of the step.
    const Eigen::VectorXcd psi_mid = step_midpoint();

    double Mdot1 = 0.0;
    double Mdot2 = 0.0;
    double Jdot1 = 0.0, Jdot2 = 0.0;

    if (absorption)
    {
        Mdot1 = get_absorption_rate_1(psi_mid);
        Mdot2 = get_absorption_rate_2(psi_mid);
        Jdot1 = absorbed_spin_rate(1, psi_mid);
        Jdot2 = absorbed_spin_rate(2, psi_mid);
    }

    if (isolated_black_hole)
    {
        // No companion and no orbit: only the hole's mass and spin follow the horizon fluxes.
        const double J1 = chi1 * M1 * M1 + Jdot1 * orbit_dt;
        M1 += Mdot1 * orbit_dt;
        chi1 = J1 / (M1 * M1);
        return;
    }

    const double ramp = orbit_ramp_factor(t_i);

    const double dR_dt_gw = dR_dt_GW(R, M1, M2);
    const double dR_dt_cloud = -2.0 * R * R * p_bin / (M1 * M2);

    // Accretion changes the masses below; its direct drift of the separation,
    // R * (Mdot1 / M1 + Mdot2 / M2), is not included.
    const double dR_dt = (dR_dt_gw + dR_dt_cloud) * ramp;

    R += dR_dt * orbit_dt;

    // J_a = chi_a M_a^2 takes the dipolar spin flux; chi_a follows the new mass.
    const double J1 = chi1 * M1 * M1 + Jdot1 * orbit_dt;
    const double J2 = chi2 * M2 * M2 + Jdot2 * orbit_dt;

    M1 += Mdot1 * orbit_dt;
    M2 += Mdot2 * orbit_dt;

    chi1 = J1 / (M1 * M1);
    if (M2 > 0.0)
        chi2 = J2 / (M2 * M2);

    q = M2 / M1;

    const double Mtot_new = M1 + M2;

    R1 = M2 / Mtot_new * R;
    R2 = M1 / Mtot_new * R;

    Omega = std::sqrt(Mtot_new / std::pow(R, 3));

    P = 2.0 * M_PI / Omega;

    if (ramp_factor == 1.0)
    {
        dt = P / steps_per_period;
    }
}