/**
 * @file CrankNicolsonModes.cpp
 * @brief Construction, initial state, single timestep and run driver.
 */
#include "CrankNicolsonModes.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

CrankNicolsonModes::CrankNicolsonModes(int Nr, double L, double dt_i, double epsilon, int l_max, double mu, double M_c, double q, double a, double R, bool counter, bool absorption, int save_every, double ramp_periods, double epsilon_smear, std::string config_file)
    : Nr(Nr), save_every(save_every), l_max(l_max), epsilon(epsilon), mu(mu), q(q), R(R), a(a), L(L),
      dr(L / (Nr - 1)), dt(dt_i), counter(counter), M_c(M_c), ramp_periods(ramp_periods),
      initial_period(2.0 * M_PI / std::sqrt((1 + q) / std::pow(R, 3))), config_file(config_file),
      absorption(absorption), epsilon_smear(epsilon_smear)
{

    if (Nr < 3 || l_max < 0 || L <= 0.0 || dt <= 0.0 || save_every <= 0)
        throw std::invalid_argument("Require Nr >= 3, l_max >= 0, and positive L, dt, save_every");
    Omega = 0.0;
    Omega_0 = std::sqrt((1.0 + q) / pow(R, 3));
    P = initial_period;

    R1 = q / (1 + q) * R;
    R2 = 1 / (1 + q) * R;

    theta = 0.0;
    phase = 0.0;

    M1 = 1.0;
    M2 = q;

    if (q > 0.0)
        std::cout << "Orbital period = " << P << std::endl;

    if (evolve_orbit && q > 0.0)
    {
        const double t_c = 5.0 / 256.0 * pow(R, 4) / (q * (1 + q));
        std::cout << "Vacuum coalescence time = " << t_c << std::endl;
    }

    std::cout << "Building grid..." << std::endl;
    build_grid();

    std::cout << "Building modes list for l_max = " << l_max << "..." << std::endl;
    build_modes();

    std::cout << "Building stationary hamiltonian..." << std::endl;
    build_stationary_hamiltonian();
}

void CrankNicolsonModes::set_power_average_orbits(double n_orbits)
{
    power_average = RollingPhaseAverage(n_orbits); // throws unless n_orbits > 0
    orbits_average = n_orbits;
}

void CrankNicolsonModes::configure_isolated_black_hole(double x)
{
    if (!std::isfinite(x) || std::abs(x) >= L)
        throw std::invalid_argument("The isolated BH must lie inside the radial domain");
    if (A.rows() != 0)
        throw std::logic_error("Configure the isolated BH before initialize()");
    isolated_black_hole = true;
    evolve_orbit = false;
    M1 = 1.0;
    M2 = q = 0.0;
    R = R1 = std::abs(x);
    R2 = 0.0;
    theta = x < 0.0 ? M_PI : 0.0;
    phase = 0.0;
    Omega = Omega_0 = 0.0;
    ramp_periods = 0.0;
}

void CrankNicolsonModes::initialize()
{

    std::cout << "Building cache of mode couplings..." << std::endl;
    build_coupling_cache();

    std::cout << "Building sparse element caches..." << std::endl;
    build_sparse_element_caches();

    std::cout << "Building initial perturbation..." << std::endl;
    t_i = 0.0;
    build_time_dependent_perturbation(0.0);

    std::cout << "Building sparse matrices for the Crank-Nicolson Method..." << std::endl;
    build_CN_matrices();

    if (absorption)
    {
        std::cout << "Spherical absorption activated." << std::endl;
    }

    std::cout << "Solver ready to run!" << std::endl;
}

void CrankNicolsonModes::build_grid()
{
    r.resize(Nr);
    for (int i = 0; i < Nr; i++)
        r(i) = epsilon + i * dr;
}

void CrankNicolsonModes::build_modes()
{
    n_modes = 0;
    std::cout << "Modes list: " << std::endl;

    for (int l = 0; l <= l_max; ++l)
    {
        for (int m = -l; m <= l; ++m)
        {
            // Equatorial symmetry: modes with odd l - m are never excited.
            if (abs(l - m) % 2 == 1)
            {
                continue;
            }
            modes.push_back({l, m});
            std::cout << "(" << l << ", " << m << ") ";
            n_modes += 1;
        }
    }

    std::cout << std::endl;
}

void CrankNicolsonModes::set_initial_state_full(Eigen::VectorXcd psi_0)
{
    if (psi_0.size() != Nr * n_modes)
        throw std::invalid_argument("Initial state size does not match the radial grid and modes");
    std::cout << "Assigning initial state..." << std::endl;
    psi.resize(Nr * n_modes);

    for (int i = 0; i < psi.size(); i++)
        psi[i] = psi_0[i];

    psi *= std::sqrt(M_c / mu);

    psi_last = Eigen::VectorXcd::Zero(Nr * n_modes);
}

void CrankNicolsonModes::step(int step_id)
{
    // Keep the previous state for the orbit's midpoint accretion rates.
    psi_last = psi;

    const double step_start = t_i;
    t_i += dt;
    theta += (counter ? -1.0 : 1.0) * Omega * dt;
    phase += Omega * dt;

    if (!evolve_orbit)
    {
        std::cout << "\rStep: " << step_id << "/" << t.size() << " | Time " << t_i << std::flush;
    }
    else
    {
        std::cout << "\rStep: " << step_id << "/" << t.size() << " | Time " << t_i << " | R = " << R << std::flush;
    }

    t(step_id) = t_i;

    build_time_dependent_perturbation(t_i);

    update_CN_values();

    Eigen::VectorXcd rhs = B * psi;
    if (sink_active)
        rhs -= (0.25 * dt) * apply_sink(psi); // B_full = B - (dt/4) * Gamma
    solve_CN(rhs);

    if (mask)
    {
#pragma omp parallel for schedule(static) if (psi.size() >= 4096)
        for (int i = 0; i < psi.size(); i++)
        {
            psi[i] *= damping_mask[i];
        }
    }

    // The flux and snapshots use the post-mask endpoint state. Evaluate the
    // Schrodinger RHS on that same state, with the completed step's H(t_i).
    // The mask itself is a discrete damping operation, not part of this RHS.
    const Eigen::VectorXcd dpsi_dt = schrodinger_rhs(psi);

    if (!isolated_black_hole && evolve_orbit && t_i > ramp_end_time())
    {
        // The crossing step may contain both frozen and evolving time. Do not
        // apply orbital feedback to the part before the fixed activation time.
        const double orbit_dt = t_i - std::max(step_start, ramp_end_time());
        update_time_dependent_increment(orbit_dt);
    }

    auto fluxes = get_flux(psi);
    auto energy = get_energy(psi);

    const std::vector<double> energy_flux = get_energy_flux(psi, dpsi_dt);

    const double p_bin_last = get_binary_power(psi);

    // A stationary BH has no orbital phase on which to average binary power.
    p_bin = isolated_black_hole ? 0.0 : power_average.update(t_i, phase, p_bin_last);

    double dV_bin_dt = get_dVexpectation_dt(psi, dpsi_dt);

    write_quantity_line(R, R_file);
    write_quantity_line(theta, theta_file);
    write_quantity_line(Omega, Omega_file);
    write_quantity_line(M1, M_1_file);
    write_quantity_line(q, q_file);
    write_quantity_line(t_i, time_file);
    write_quantity_line_modes(energy_flux, energy_flux_file);
    write_quantity_line_modes(fluxes, flux_file);
    write_quantity_line(energy, energy_file);
    write_quantity_line(p_bin, p_bin_file);
    write_quantity_line(p_bin_last, p_bin_inst_file);
    write_quantity_line(dV_bin_dt, d_V_bin_dt_file);

    auto norms = get_norms(psi);
    write_quantity_line_modes(norms, norms_file);

    auto energies = get_energy_per_mode(psi);
    write_quantity_line_modes(energies, energy_modes_file);

    if (step_id % save_every == 0)
    {
        save_snapshot(step_id, psi, "psi");
        flush_diagnostics();
        save_checkpoint(step_id);
    }
}

void CrankNicolsonModes::run(int n_steps, double r_ex_)
{

    std::cout << "Running simulation..." << std::endl;

    set_extraction_radius(r_ex_, n_steps);
    t_i = 0.0;
    t.resize(n_steps);
    t(0) = t_i;

    output_folder = create_output_folder();
    std::cout << "Output folder: " << output_folder << std::endl;

    open_diagnostics(std::ios::out);

    save_run_info();
    // Refresh the initial operator, including a CAP enabled after initialize().
    build_time_dependent_perturbation(t_i);
    save_snapshot(0, psi, "psi");
    save_snapshot(0, schrodinger_rhs(psi), "psi_dt");
    save_checkpoint(0);

    evolve(1, n_steps);
}

void CrankNicolsonModes::set_extraction_radius(double r_ex_, int n_steps)
{
    r_ex = r_ex_;
    idx_ex = (int)std::floor(r_ex / dr);

    std::cout << "Computing fluxes at: " << r_ex_ << " [Index: " << idx_ex << "]" << std::endl;

    if (n_steps < 1 || idx_ex < 1 || idx_ex >= Nr - 1)
        throw std::invalid_argument("Require at least one time sample and an interior extraction radius");
}

void CrankNicolsonModes::evolve(int first_step, int n_steps)
{
    for (int n = first_step; n < n_steps; n++)
    {
        if (!isolated_black_hole && evolve_orbit)
        {
            if (R <= 6.0)
            {
                break;
            }
        }
        step(n);
    }

    std::cout << std::endl;
    flush_diagnostics();
    std::cout << "Simulation completed!" << std::endl;
}
