/**
 * @file CrankNicolsonModes.cpp
 * @brief Construction, initial state, single timestep and run driver.
 */
#include "CrankNicolsonModes.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
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

void CrankNicolsonModes::set_power_average_orbits(double n_orbits, RollingPhaseAverage::Window window)
{
    power_average = RollingPhaseAverage(n_orbits, window); // throws unless n_orbits > 0
    orbits_average = n_orbits;
    power_window = window;
}

void CrankNicolsonModes::configure_isolated_black_hole(double x)
{
    if (!std::isfinite(x) || std::abs(x) >= L)
        throw std::invalid_argument("The isolated BH must lie inside the radial domain");
    if (operator_ready)
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

    if (dipole_absorption && !absorption)
        throw std::invalid_argument("dipole_absorption needs absorption");
    if (dipole_absorption && l_max < 1)
        throw std::invalid_argument("dipole_absorption needs l_max >= 1");
    if (!(std::abs(chi1) <= 1.0 && std::abs(chi2) <= 1.0))
        throw std::invalid_argument("The black-hole spins must satisfy |chi| <= 1");

    std::cout << "Building cache of mode couplings..." << std::endl;
    build_coupling_cache();

    std::cout << "Building the coupling operator..." << std::endl;
    build_active_operator();

    std::cout << "Building initial perturbation..." << std::endl;
    t_i = 0.0;
    build_time_dependent_perturbation(0.0);

    std::cout << "Configuring the Crank-Nicolson solver..." << std::endl;
    build_CN_matrices();

    if (sg.enabled)
    {
        build_self_gravity_tables();
        build_levels();
    }

    if (pruning_enabled())
    {
        if (prune_check_every < 1 || !(prune_tolerance < 1.0))
            throw std::invalid_argument("Pruning needs 0 < prune_tolerance < 1 and prune_check_every >= 1");
        // Select the initial active set. A resumed run has no state yet at
        // this point and restores its own set in resume().
        if (psi.size() == static_cast<Eigen::Index>(Nr) * n_modes && update_active_modes(prune_check_every * dt, true))
            apply_active_set();
    }

    if (absorption)
    {
        std::cout << "Spherical absorption activated." << std::endl;
    }
    if (dipole_absorption)
        std::cout << "Dipolar (l = 1) horizon absorption activated, spins chi_1 = " << chi1
                  << ", chi_2 = " << chi2 << "." << std::endl;

    std::cout << "Solver ready to run!" << std::endl;
}

void CrankNicolsonModes::build_grid()
{
    r.resize(Nr);
    for (int i = 0; i < Nr; i++)
        r(i) = epsilon + i * dr;
    uniform_grid = true;
    node_weight = Eigen::VectorXd::Constant(Nr, dr);
}

void CrankNicolsonModes::set_radial_grid(const Eigen::VectorXd &nodes)
{
    if (operator_ready)
        throw std::logic_error("Set the radial grid before initialize()");
    if (nodes.size() != Nr)
        throw std::invalid_argument("The radial grid must have Nr nodes");
    for (int i = 1; i < Nr; ++i)
        if (!(nodes(i) > nodes(i - 1)))
            throw std::invalid_argument("Radial nodes must increase");
    r = nodes;
    uniform_grid = false;
    dr = (r.tail(Nr - 1) - r.head(Nr - 1)).minCoeff();
    // Trapezoid weights: the measure in which the second difference is self-adjoint.
    node_weight.resize(Nr);
    node_weight(0) = 0.5 * (r(1) - r(0));
    node_weight(Nr - 1) = 0.5 * (r(Nr - 1) - r(Nr - 2));
    for (int i = 1; i < Nr - 1; ++i)
        node_weight(i) = 0.5 * (r(i + 1) - r(i - 1));
    std::cout << "Stretched radial grid: dr = " << dr << " near the centre, "
              << r(Nr - 1) - r(Nr - 2) << " at r = " << r(Nr - 1) << std::endl;
    build_stationary_hamiltonian();
}

int CrankNicolsonModes::node_at(double offset) const
{
    if (uniform_grid)
        return static_cast<int>(std::floor(offset / dr));
    return static_cast<int>(std::upper_bound(r.data(), r.data() + Nr, r(0) + offset) - r.data()) - 1;
}

dcomplex CrankNicolsonModes::radial_derivative(const dcomplex *u, int i) const
{
    if (uniform_grid)
        return (u[i + 1] - u[i - 1]) / (2.0 * dr);
    // Second-order three-point derivative on unequal spacings.
    const double hm = r(i) - r(i - 1), hp = r(i + 1) - r(i);
    return (-hp / (hm * (hm + hp))) * u[i - 1] + ((hp - hm) / (hm * hp)) * u[i] + (hm / (hp * (hm + hp))) * u[i + 1];
}

double CrankNicolsonModes::radial_norm2(const dcomplex *u, int first, int count) const
{
    if (uniform_grid)
        return Eigen::Map<const Eigen::VectorXcd>(u + first, count).squaredNorm() * dr;
    double sum = 0.0;
    for (int i = first; i < first + count; ++i)
        sum += node_weight(i) * std::norm(u[i]);
    return sum;
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
    // Refresh the evolved modes before the step (keyed on step_id, so that a
    // resumed run checks at the same steps).
    if (pruning_enabled() && (step_id - 1) % prune_check_every == 0 &&
        update_active_modes(prune_check_every * dt))
    {
        apply_active_set();
        write_active_modes(step_id);
    }

    // A per-mode CAP is fixed over the step, from the energies measured up to now.
    update_cap_strengths();

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

    // Self-gravity: the potential over the step (extrapolated, or the first guess of the iteration).
    if (sg.enabled)
        sg_prepare_step();

    // B psi with B = 1 - (i dt/2) H on the active modes.
    const Eigen::VectorXcd state = gather_active(psi);
    Eigen::VectorXcd sink_rhs;
    if (sink_active)
        sink_rhs = (0.25 * dt) * gather_active(apply_sink(psi)); // B_full = B - (dt/4) * Gamma
    step_rotation = std::exp(I * (reference_energy * dt));
    const auto crank_nicolson = [&]() {
        Eigen::VectorXcd rhs;
        apply_hamiltonian(state, rhs, true);
        rhs = state - (I * (0.5 * dt)) * (rhs - reference_energy * state);
        if (sink_active)
            rhs -= sink_rhs;
        solve_CN(rhs);
        // The solve gives the state in the frame rotating at E_ref; turn it back.
        if (reference_energy != 0.0)
            for (const int j : active_list)
                psi.segment(static_cast<Eigen::Index>(j) * Nr, Nr) *= std::conj(step_rotation);
    };
    crank_nicolson();
    sg_last_iterations = 1;
    if (sg.enabled && sg.iterated)
    {
        // Midpoint potential (W[psi_n] + W[psi_n+1]) / 2, by fixed-point iteration.
        const double scale = sg_ramp(t_i);
        std::vector<dcomplex> next;
        for (int k = 1; k < sg.max_iterations; ++k)
        {
            const Eigen::VectorXcd previous = psi;
            sg_compute_potential(psi);
            sg_fill_profile(scale, next);
            for (std::size_t n = 0; n < next.size(); ++n)
                sg_profile[n] = 0.5 * (sg_profile_now[n] + next[n]);
            // Start the solve from the last iterate, in the rotating frame.
            if (reference_energy != 0.0)
                for (const int j : active_list)
                    psi.segment(static_cast<Eigen::Index>(j) * Nr, Nr) *= step_rotation;
            crank_nicolson();
            ++sg_last_iterations;
            if ((psi - previous).norm() <= sg.tolerance * psi.norm())
                break;
        }
    }

    // Binary power of this step: Omega_z times the torque of V(t_n+1) on the midpoint state.
    // With phi = exp(i E_ref dt) psi_n+1, CN gives phi - psi_n = -dt (i(H - E_ref) + Gamma/2) psi_mid
    // exactly, psi_mid = (psi_n + phi) / 2 (E_ref drops out of L_z and norms), so this torque minus the
    // sinks' absorbed L_z (also on psi_mid) is exactly the field's discrete Delta L_z / dt.
    if (!isolated_black_hole)
        p_bin_step = step_binary_power(step_midpoint(), 0.0);

    // Exact L_z budget of the cloud inside r_ex for this step (same midpoint identity).
    if (Lz_budget_file.is_open())
        accumulate_Lz_step(step_midpoint());

    // Each mode's frequency in the layer, for the next steps' CAP strengths.
    measure_cap_energies();

    // Mass taken up by the sinks over this step. With the rates at the
    // midpoint state this is exactly the norm the CN step removes through them.
    if (sink_active)
    {
        const Eigen::VectorXcd midpoint = step_midpoint();
        absorbed_mass_1 += get_absorption_rate_1(midpoint) * dt;
        absorbed_mass_2 += get_absorption_rate_2(midpoint) * dt;
        if (sg.enabled)
        {
            absorbed_swave_1 += mu * sink_particle_rate(sink1, midpoint) * dt;
            if (dipole_absorption)
                for (int c = 0; c < 2; ++c)
                    absorbed_dipole_1[c] += mu * sink_particle_rate(dipole_sinks[c], midpoint) * dt;
        }
        // The dipolar channels put m units of L_z per particle into the holes' spins;
        // the rest of the L_z the sinks remove (about the centre of mass) is orbital.
        const double spin_rate_1 = absorbed_spin_rate(1, midpoint), spin_rate_2 = absorbed_spin_rate(2, midpoint);
        absorbed_J_1 += spin_rate_1 * dt;
        absorbed_J_2 += spin_rate_2 * dt;
        Lz_abs_rate = absorbed_Lz_rate(midpoint) - (spin_rate_1 + spin_rate_2);
    }
    else
        Lz_abs_rate = 0.0;

    if (mask)
    {
        const double Lz_before_mask = Lz_budget_file.is_open() ? cloud_Lz(psi) : 0.0;
#pragma omp parallel for schedule(static) if (psi.size() >= 4096)
        for (int i = 0; i < psi.size(); i++)
        {
            psi[i] *= damping_mask[i];
        }
        if (Lz_budget_file.is_open())
            Lz_budget.damped += Lz_before_mask - cloud_Lz(psi);
    }

    // Self-gravity of the new state: in H for the diagnostics below and the next step.
    if (sg.enabled)
        sg_finish_step();

    // The flux and snapshots use the post-mask endpoint state. Evaluate the
    // Schrodinger RHS on that same state, with the completed step's H(t_i).
    // The mask itself is a discrete damping operation, not part of this RHS.
    // One H psi serves dpsi_dt and both energies.
    const Eigen::VectorXcd Hpsi = hamiltonian_times(psi);
    Eigen::VectorXcd dpsi_dt = -I * Hpsi;
    if (sink_active)
        dpsi_dt -= 0.5 * apply_sink(psi);

    if (evolve_orbit && t_i > orbit_start_time())
    {
        // The crossing step may contain both frozen and evolving time. Do not
        // apply orbital feedback to the part before the fixed activation time.
        const double orbit_dt = t_i - std::max(step_start, orbit_start_time());
        update_time_dependent_increment(orbit_dt);
    }

    auto fluxes = get_flux(psi);
    const double energy = restricted_expectation(psi, Hpsi, idx_ex);

    const std::vector<double> energy_flux = get_energy_flux(psi, dpsi_dt);

    double p_bin_last = 0.0, dV_bin_dt = 0.0;
    binary_power_terms(psi, &dpsi_dt, p_bin_last, dV_bin_dt);

    // A stationary BH has no orbital phase on which to average binary power.
    // The holes take the absorbed L_z; for a circular orbit the orbit gets it at
    // dE = Omega_z dL, so the power the orbit loses is reduced by Omega_z * Lz_abs_rate.
    const double p_bin_orbit = p_bin_step - (counter ? -1.0 : 1.0) * Omega * Lz_abs_rate;
    p_bin = isolated_black_hole ? 0.0 : power_average.update(t_i, phase, p_bin_orbit);

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
    write_quantity_line(p_bin_step, p_bin_inst_file);
    if (Lz_abs_file.is_open())
        write_quantity_line(Lz_abs_rate, Lz_abs_file);
    write_quantity_line(dV_bin_dt, d_V_bin_dt_file);

    auto norms = get_norms(psi);
    write_quantity_line_modes(norms, norms_file);

    auto energies = energy_per_mode(psi, Hpsi);
    write_quantity_line_modes(energies, energy_modes_file);
    if (mass_budget_file.is_open())
        write_quantity_line_modes({absorbed_mass_1, absorbed_mass_2, pruned_mass_total}, mass_budget_file);
    if (spins_file.is_open())
        write_quantity_line_modes({chi1, chi2, absorbed_J_1, absorbed_J_2}, spins_file);
    if (Lz_budget_file.is_open())
        write_quantity_line_modes({Lz_budget.ionized, Lz_budget.torque, Lz_budget.absorbed, Lz_budget.pruned,
                                   Lz_budget.damped},
                                  Lz_budget_file);
    if (sg.enabled && self_gravity_file.is_open())
        write_self_gravity_rows(psi);
    if (cap_modes_file.is_open())
    {
        std::vector<double> row(2 * static_cast<std::size_t>(n_modes));
        for (int j = 0; j < n_modes; ++j)
        {
            row[j] = cap_has_energy[j] ? cap_energy[j] : 0.0;
            row[n_modes + j] = cap_W_mode[j];
        }
        write_quantity_line_modes(row, cap_modes_file);
    }

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
    mass_budget_file.open(output_folder + "/mass_budget.txt", std::ios::out);
    Lz_abs_file.open(output_folder + "/Lz_absorbed_rate.txt", std::ios::out);
    Lz_budget = LzBudget{};
    Lz_budget_file.open(output_folder + "/Lz_budget.txt", std::ios::out);
    if (cap_per_mode)
        cap_modes_file.open(output_folder + "/cap_modes.txt", std::ios::out);
    if (dipole_absorption)
        spins_file.open(output_folder + "/spins.txt", std::ios::out);
    if (pruning_enabled())
    {
        active_modes_file.open(output_folder + "/active_modes.txt", std::ios::out);
        write_active_modes(0);
    }

    save_run_info();
    // Refresh the initial operator, including a CAP enabled after initialize().
    build_time_dependent_perturbation(t_i);
    if (sg.enabled)
    {
        sg_have_prev = false;
        sg_refresh_now();
        sg_profile = sg_profile_now;
        if (sg.measured_reference && reference_energy != 0.0)
        {
            // The cloud's phase turns at its chemical potential <psi|H0 + V + W|psi> / N.
            const Eigen::VectorXcd Hpsi = hamiltonian_times(psi);
            const double N = state_norm2(psi);
            double expectation = 0.0;
            for (int j = 0; j < n_modes; ++j)
                for (int i = 0; i < Nr; ++i)
                    expectation += node_weight[i] * (std::conj(psi[j * Nr + i]) * Hpsi[j * Nr + i]).real();
            const double weighted_N = uniform_grid ? N * dr : N;
            if (weighted_N > 0.0)
            {
                std::cout << "Self-gravity: E_ref set to the measured chemical potential " << expectation / weighted_N
                          << " (hydrogenic level " << reference_energy << ")" << std::endl;
                reference_energy = expectation / weighted_N;
            }
        }
        open_self_gravity_files(std::ios::out);
        write_levels_index();
    }
    save_snapshot(0, psi, "psi");
    save_snapshot(0, schrodinger_rhs(psi), "psi_dt");
    save_checkpoint(0);

    evolve(1, n_steps);
}

void CrankNicolsonModes::set_extraction_radius(double r_ex_, int n_steps)
{
    r_ex = r_ex_;
    idx_ex = node_at(r_ex);

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
