/**
 * @file main.cpp
 * @brief Command-line driver for binary simulations.
 *
 * Usage:
 * @code
 *   GravitationalMolecule config.toml initial_modes.bin   # new run
 *   GravitationalMolecule Simulation_YYYY_MM_DD_HH_MM_SS   # resume a run
 * @endcode
 */
#include <cmath>
#include <iostream>
#include <filesystem>
#include <Eigen/Core>
#include <omp.h>
#include "Config.hpp"
#include "CrankNicolsonModes.hpp"
#include "InitialData.hpp"

int main(int argc, char **argv)
{
    // A single directory argument resumes the run stored there, from its
    // config.toml and last snapshot.
    const bool resume = argc == 2 && std::filesystem::is_directory(argv[1]);
    if (argc != 3 && !resume)
    {
        std::cerr << "Usage: " << argv[0] << " config.toml initial_modes.bin\n"
                  << "       " << argv[0] << " simulation_folder   (resume)\n";
        return 1;
    }
    std::string config_file = "config.toml";
    std::string resume_folder;

    if (resume)
    {
        resume_folder = std::filesystem::path(argv[1]).lexically_normal().string();
        while (resume_folder.size() > 1 && resume_folder.back() == '/')
            resume_folder.pop_back();
        config_file = resume_folder + "/config.toml";
    }
    else
        config_file = argv[1];

    Config cfg = load_config(config_file);

    // A positive configuration value overrides OMP_NUM_THREADS; otherwise use
    // the OpenMP runtime's environment/default for both Eigen and our kernels.
    if (cfg.threads > 0)
        omp_set_num_threads(cfg.threads);
    Eigen::setNbThreads(omp_get_max_threads());
    std::cout << "Using up to " << omp_get_max_threads() << " OpenMP threads.\n";

    std::cout << "Loaded configuration from " << config_file << "\n";
    if (cfg.dt_from_period)
        std::cout << "Timestep dt = " << cfg.dt << " (initial period / " << cfg.steps_per_period << ")\n";

    CrankNicolsonModes solver(
        cfg.N,
        cfg.L,
        cfg.dt,
        cfg.epsilon,
        cfg.l_max,
        cfg.mu,
        cfg.M_c,
        cfg.q,
        cfg.a,
        cfg.R,
        cfg.counter,
        cfg.absorption,
        cfg.save_every,
        cfg.ramp_periods,
        cfg.epsilon_smear,
        config_file);

    if (cfg.grid.type != "uniform")
    {
        solver.set_radial_grid(make_radial_grid(cfg.N, cfg.L, cfg.epsilon, cfg.grid));
        // The fluxes are radial derivatives at r_ex, so they are only as fine as the grid there.
        if (cfg.r_ex > cfg.grid.r_core + cfg.grid.transition && cfg.r_ex < cfg.L)
            std::cout << "Note: r_ex = " << cfg.r_ex << " lies outside the refined region (r_core = " << cfg.grid.r_core
                      << "); the fluxes at r_ex use the coarser spacing there. Raise grid.r_core above r_ex for"
                      << " finer flux measurements.\n";
    }

    if (!cfg.secondary_potential)
    {
        // No companion: an isolated primary at rest where the initial state has it,
        // at distance q / (1 + q) * R from the origin on the +x axis.
        solver.configure_isolated_black_hole(cfg.q / (1.0 + cfg.q) * cfg.R);
        std::cout << "Companion switched off (" << (cfg.q == 0.0 ? "q = 0" : "secondary_potential = false")
                  << "): the primary stays at rest at x = "
                  << solver.R1 << ", with no companion potential, sink, orbital motion or back-reaction.\n";
    }

    if (!resume)
        solver.set_initial_state_full(
            load_initial_modes_binary(argv[2], solver.modes, solver.r, solver.uniform_radial_grid()));

    // Without a companion, evolve_orbit only lets the primary's mass and spin follow the horizon fluxes.
    solver.evolve_orbit = cfg.evolve_orbit;
    solver.dipole_absorption = cfg.dipole_absorption;
    // The hydrogenic level of the initial state, alpha = mu M1 with M1 = 1.
    solver.reference_energy = cfg.energy_reference ? -0.5 * cfg.mu * cfg.mu * cfg.mu / (cfg.n * cfg.n) : 0.0;
    solver.chi1 = cfg.spin_1;
    solver.chi2 = cfg.spin_2;
    if (cfg.dipole_absorption)
        std::cout << "Dipolar horizon absorption: on (initial spins " << cfg.spin_1 << ", " << cfg.spin_2 << ")\n";
    solver.ramp_secondary_potential = cfg.ramp_secondary_potential;
    solver.orbit_delay_periods = cfg.orbit_delay_periods;
    solver.orbit_ramp_periods = cfg.orbit_ramp_periods;
    solver.steps_per_period = cfg.steps_per_period;
    solver.set_power_average_orbits(cfg.power_average_orbits,
                                    cfg.power_average_window == "hann" ? RollingPhaseAverage::Window::Hann
                                                                       : RollingPhaseAverage::Window::Boxcar);
    if (cfg.secondary_potential)
        std::cout << "Startup ramp over " << cfg.ramp_periods << " initial periods: "
              << (cfg.ramp_secondary_potential ? "frequency and companion potential"
                                               : "frequency only (companion potential at full strength)")
              << "\n";
    if (cfg.evolve_orbit && !cfg.secondary_potential)
        std::cout << "The primary's mass and spin follow the horizon fluxes after "
                  << cfg.orbit_delay_periods << " initial periods\n";
    else if (cfg.evolve_orbit)
        std::cout << "Orbit held fixed for " << cfg.orbit_delay_periods
                  << " initial periods after the startup ramp, then back-reaction switched on over "
                  << cfg.orbit_ramp_periods << " initial periods\n";

    if (!cfg.pruning)
        std::cout << "Mode pruning: off (every mode is evolved)\n";
    else
    {
        solver.prune_tolerance = cfg.prune_tolerance;
        solver.prune_check_every = cfg.prune_check_every;
        solver.prune_flux_tolerance = cfg.prune_flux_tolerance;
        std::cout << "Mode pruning: tolerance " << cfg.prune_tolerance << ", flux tolerance "
                  << cfg.prune_flux_tolerance << ", active set checked every " << cfg.prune_check_every
                  << " steps\n";
    }

    solver.use_block_preconditioner = cfg.preconditioner == "radial";
    solver.initialize();

    if (cfg.mask)
    {
        solver.build_damping_mask(cfg.mask_L_factor, cfg.mask_W, cfg.mask_n);
    }

    if (cfg.cap)
    {
        solver.build_cap(cfg.cap_L_factor, cfg.cap_W, cfg.cap_n);
        if (cfg.cap_per_mode)
            solver.enable_cap_per_mode(cfg.cap_reflection, cfg.cap_tune_steps, cfg.cap_k_min);
    }

    // num_steps takes precedence; otherwise run for num_periods initial periods.
    const int num_steps = cfg.num_steps
        ? cfg.num_steps
        : static_cast<int>(std::floor(cfg.num_periods * solver.P / solver.dt));

    if (resume)
        solver.resume(resume_folder, num_steps, cfg.r_ex);
    else
        solver.run(num_steps, cfg.r_ex);

    return 0;
}