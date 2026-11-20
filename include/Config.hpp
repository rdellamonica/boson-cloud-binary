/**
 * @file Config.hpp
 * @brief Simulation configuration and its TOML loader.
 */
#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <stdexcept>
#include <toml++/toml.hpp>
#include "RadialGrid.hpp"

/**
 * @brief Run parameters read from a TOML configuration file.
 *
 * Every key is optional; defaults are those of load_config(). The tables
 * @c [mask] and @c [cap] switch on the outer damping mask and the complex
 * absorbing potential simply by being present.
 */
struct Config
{
    /// @name [grid]
    /// @{
    int N;    ///< Number of radial nodes (default 2048).
    double L; ///< Outer radius (default 5000).
    /// Grid shape: type "uniform" (default) or "stretched", with r_core
    /// (default 4 n^2 / mu^2, four times the cloud's mean radius scale),
    /// stretch (default 4) and transition (default r_core / 4).
    RadialGridSpec grid;
    /// @}

    /// @name [simulation]
    /// @{
    double dt;                          ///< Initial timestep (default: initial period / steps_per_period).
    bool dt_from_period = false;        ///< True when dt was not given and follows from the initial period.
    double num_periods;                 ///< Run length in initial orbital periods (may be fractional), used if num_steps = 0.
    int save_every;                     ///< Snapshot interval in steps (default 50).
    double ramp_periods;                ///< Startup ramp length in initial periods (default 0).
    bool ramp_secondary_potential = true; ///< Ramp the companion's potential with Omega.
    /// false removes the companion: the primary stays at rest at its initial position and
    /// only its own potential and horizon act on the cloud (default true).
    bool secondary_potential = true;
    double orbit_delay_periods = 0.0;   ///< Hold with the companion fully on and the orbit fixed, after the startup ramp, in initial periods.
    double orbit_ramp_periods = 1.0;    ///< Orbital back-reaction switch-on after that hold, in initial periods (0: at once).
    double steps_per_period = 50.0;     ///< Post-ramp timestep is P / steps_per_period.
    double power_average_orbits = 2.0;  ///< Orbits in the rolling average of the binary power.
    std::string power_average_window = "boxcar"; ///< Shape of that window: "boxcar" or "hann".
    bool evolve_orbit;                  ///< Feed the cloud power and accretion back on the orbit (default false).
    bool absorption;                    ///< Enable the black-hole absorbers (default false).
    /// Take the CN steps relative to the initial level's energy -mu^3 / (2 n^2), so that
    /// horizon growth and decay rates do not depend on dt (default true).
    bool energy_reference = true;
    bool dipole_absorption = false;     ///< Add the dipolar (l = 1) horizon channels to the absorbers (default false).
    int l_max;                          ///< Largest retained multipole (default 5).
    int num_steps;                      ///< Number of time samples; 0 uses num_periods.
    double epsilon;                     ///< Offset of the first radial node (default 0.001).
    double epsilon_smear;               ///< Absorber radius in units of each body's mass (default 2).
    double r_ex;                        ///< Extraction radius for fluxes (default 1000).
    /// @}

    /// @name [binary]
    /// @{
    double q;     ///< Mass ratio M2/M1 (default 0.001).
    double R;     ///< Initial separation (default 50).
    bool counter; ///< Counter-rotating orbit (default false).
    double spin_1 = 0.0; ///< Initial dimensionless spin of the primary along +z (default 0; atom.a stays informational).
    double spin_2 = 0.0; ///< Initial dimensionless spin of the companion along +z (default 0).
    /// @}

    /// @name [atom]
    /// @{
    double mu;  ///< Boson mass (default 0.2).
    double a;   ///< Primary spin (default 0).
    double M_c; ///< Cloud mass (default 0).
    int n;      ///< Principal quantum number of the initial state (informational).
    int l;      ///< Orbital quantum number of the initial state (informational).
    int m;      ///< Magnetic quantum number of the initial state (informational).
    /// @}

    /// @name [parallel] and [solver]
    /// @{
    int threads = 0;                     ///< OpenMP threads; 0 uses OMP_NUM_THREADS.
    std::string preconditioner = "radial"; ///< "radial" (default) or "ilut".
    /// @}

    /// @name [mask]: multiplicative damping near the outer boundary
    /// @{
    bool mask = false;
    double mask_L_factor; ///< Fraction of L where the mask starts (default 1).
    double mask_W;        ///< Strength (default 0.1).
    int mask_n;           ///< Power of the profile (default 3).
    /// @}

    /// @name [cap]: complex absorbing potential near the outer boundary
    /// @{
    bool cap = false;
    double cap_L_factor; ///< Fraction of L where the CAP starts (default 1).
    double cap_W;        ///< Strength (default 0.1).
    int cap_n;           ///< Power of the profile (default 3; 6 with per_mode).
    /// Tune each mode's strength to its frequency instead of using W (default false).
    bool cap_per_mode = false;
    double cap_reflection = 1e-10; ///< Per-mode target amplitude reflection (default 1e-10).
    int cap_tune_steps = 20;      ///< Per-mode: steps of the running energy average (default 20).
    double cap_k_min = 0.0;       ///< Per-mode: smallest wavenumber (default 0 = 1 / layer width).
    /// @}

    /// @name [self_gravity]: Newtonian self-gravity of the cloud (isolated hole at the origin)
    /// @{
    bool self_gravity = false;          ///< True when a [self_gravity] table is present and enabled is not false.
    int sg_l_max = -1;                  ///< Highest multipole of Phi (default 2 l_max).
    bool sg_recoil = true;              ///< Free-fall frame of the hole (default true).
    double sg_recoil_taper = 0.0;       ///< Taper the recoil field beyond this radius (default 0 = none).
    bool sg_accretion_recoil = true;    ///< The hole takes up the momentum the s-wave sink absorbs (default true).
    double sg_ramp_time = 0.0;          ///< Smooth switch-on time (default 0).
    std::string sg_scheme = "extrapolated"; ///< "extrapolated" (one solve per step) or "iterated".
    double sg_tolerance = 1e-10;        ///< Iterated scheme: relative change that stops the iteration.
    int sg_max_iterations = 20;         ///< Iterated scheme: most solves per step.
    bool sg_measured_reference = true;  ///< E_ref = the initial state's chemical potential (with energy_reference).
    int sg_levels_n_max = 5;            ///< Highest n written to levels.txt.
    /// @}

    /// @name [pruning]: evolve only the modes that matter (see CrankNicolsonModes)
    /// @{
    bool pruning = false;         ///< True when a [pruning] table is present and enabled is not false.
    double prune_tolerance = 0.0; ///< Norm fraction that makes a mode relevant (default 1e-8).
    int prune_check_every = 10;   ///< Steps between updates of the active set (default 10).
    /// Share of the outgoing flux that makes a mode relevant (default 1e-3; 0 ignores the flux).
    double prune_flux_tolerance = 1e-3;
    /// @}
};

/**
 * @brief Parse a TOML configuration file.
 * @param filename Path to the file.
 * @return The configuration, with defaults for missing keys.
 * @throws std::invalid_argument for invalid thread counts, preconditioners,
 *         steps_per_period or power_average_orbits.
 * @throws toml::parse_error if the file cannot be parsed.
 */
inline Config load_config(const std::string &filename)
{
    Config c;
    auto tbl = toml::parse_file(filename);

    c.threads = tbl["parallel"]["threads"].value_or(0);
    c.preconditioner = tbl["solver"]["preconditioner"].value_or(std::string("radial"));
    if (c.threads < 0)
        throw std::invalid_argument("parallel.threads must be nonnegative");
    if (c.preconditioner != "radial" && c.preconditioner != "ilut")
        throw std::invalid_argument("solver.preconditioner must be radial or ilut");

    c.N = tbl["grid"]["N"].value_or(2048);
    c.L = tbl["grid"]["L"].value_or(5000.0);

    const std::optional<double> dt = tbl["simulation"]["dt"].value<double>();
    c.num_periods = tbl["simulation"]["num_periods"].value_or(0.0);
    c.num_steps = tbl["simulation"]["num_steps"].value_or(0);
    c.save_every = tbl["simulation"]["save_every"].value_or(50);
    c.l_max = tbl["simulation"]["l_max"].value_or(5);
    c.r_ex = tbl["simulation"]["r_ex"].value_or(1000.0);
    c.ramp_periods = tbl["simulation"]["ramp_periods"].value_or(0.0);
    c.ramp_secondary_potential = tbl["simulation"]["ramp_secondary_potential"].value_or(true);
    c.secondary_potential = tbl["simulation"]["secondary_potential"].value_or(true);
    c.orbit_delay_periods = tbl["simulation"]["orbit_delay_periods"].value_or(0.0);
    if (!(c.orbit_delay_periods >= 0.0))
        throw std::invalid_argument("simulation.orbit_delay_periods must be nonnegative");
    c.orbit_ramp_periods = tbl["simulation"]["orbit_ramp_periods"].value_or(1.0);
    if (!(c.orbit_ramp_periods >= 0.0))
        throw std::invalid_argument("simulation.orbit_ramp_periods must be nonnegative");
    c.steps_per_period = tbl["simulation"]["steps_per_period"].value_or(50.0);
    if (!(c.steps_per_period > 0.0))
        throw std::invalid_argument("simulation.steps_per_period must be positive");
    c.power_average_orbits = tbl["simulation"]["power_average_orbits"].value_or(2.0);
    if (!(c.power_average_orbits > 0.0))
        throw std::invalid_argument("simulation.power_average_orbits must be positive");
    c.power_average_window = tbl["simulation"]["power_average_window"].value_or(std::string("boxcar"));
    if (c.power_average_window != "boxcar" && c.power_average_window != "hann")
        throw std::invalid_argument("simulation.power_average_window must be \"boxcar\" or \"hann\"");
    c.epsilon = tbl["simulation"]["epsilon"].value_or(0.001);
    c.epsilon_smear = tbl["simulation"]["epsilon_smear"].value_or(2.0);
    c.evolve_orbit = tbl["simulation"]["evolve_orbit"].value_or(false);
    c.absorption = tbl["simulation"]["absorption"].value_or(false);

    c.q = tbl["binary"]["q"].value_or(0.001);
    c.R = tbl["binary"]["R"].value_or(50.0);
    if (!(c.q >= 0.0))
        throw std::invalid_argument("binary.q must be nonnegative");
    // A massless companion is no companion: q = 0 runs as secondary_potential = false,
    // with the primary at rest at the origin.
    if (c.q == 0.0)
        c.secondary_potential = false;
    c.counter = tbl["binary"]["counter"].value_or(false);
    c.dipole_absorption = tbl["simulation"]["dipole_absorption"].value_or(false);
    c.energy_reference = tbl["simulation"]["energy_reference"].value_or(true);
    if (c.dipole_absorption && !c.absorption)
        throw std::invalid_argument("simulation.dipole_absorption needs simulation.absorption = true");
    if (c.dipole_absorption && c.l_max < 1)
        throw std::invalid_argument("simulation.dipole_absorption needs simulation.l_max >= 1");
    // The spins point along +z of the simulation frame whatever the sense of the orbit.
    c.spin_1 = tbl["binary"]["spin_1"].value_or(0.0);
    c.spin_2 = tbl["binary"]["spin_2"].value_or(0.0);
    if (!(std::abs(c.spin_1) <= 1.0 && std::abs(c.spin_2) <= 1.0))
        throw std::invalid_argument("binary.spin_1 and binary.spin_2 must lie in [-1, 1]");

    // Without an explicit dt, start with the fraction of the initial period
    // P0 = 2 pi sqrt(R^3 / (1 + q)) that the orbital evolution uses.
    c.dt_from_period = !dt;
    c.dt = dt ? *dt : 2.0 * M_PI * std::sqrt(std::pow(c.R, 3) / (1.0 + c.q)) / c.steps_per_period;
    // R = 0 (a lone hole at the origin) has no period to derive dt or the run length from.
    if (!(c.dt > 0.0))
        throw std::invalid_argument("simulation.dt must be positive; set it explicitly when binary.R = 0");
    if (c.num_steps <= 0 && !(c.R > 0.0))
        throw std::invalid_argument("simulation.num_periods needs binary.R > 0; set simulation.num_steps instead");

    c.mu = tbl["atom"]["mu"].value_or(0.2);
    c.n = tbl["atom"]["n"].value_or(2);
    c.l = tbl["atom"]["l"].value_or(1);
    c.m = tbl["atom"]["m"].value_or(1);
    c.a = tbl["atom"]["a"].value_or(0.0);
    c.M_c = tbl["atom"]["M_c"].value_or(0.0);

    c.grid.type = tbl["grid"]["type"].value_or(std::string("uniform"));
    if (c.grid.type != "uniform" && c.grid.type != "stretched")
        throw std::invalid_argument("grid.type must be uniform or stretched");
    if (c.grid.type == "stretched")
    {
        // Default core: four times n^2 r_B, with r_B = 1 / (mu^2 M1) and M1 = 1.
        c.grid.r_core = tbl["grid"]["r_core"].value_or(4.0 * c.n * c.n / (c.mu * c.mu));
        c.grid.stretch = tbl["grid"]["stretch"].value_or(4.0);
        c.grid.transition = tbl["grid"]["transition"].value_or(0.25 * c.grid.r_core);
        if (!(c.grid.r_core > 0.0 && c.grid.r_core < c.L))
            throw std::invalid_argument("grid.r_core must lie in (0, L)");
        if (!(c.grid.stretch >= 1.0))
            throw std::invalid_argument("grid.stretch must be at least 1");
        if (!(c.grid.transition > 0.0))
            throw std::invalid_argument("grid.transition must be positive");
    }

    if (tbl.contains("mask") && tbl["mask"].is_table())
    {
        c.mask = true;
        c.mask_L_factor = tbl["mask"]["L_factor"].value_or(1.0);
        c.mask_W = tbl["mask"]["W"].value_or(0.1);
        c.mask_n = tbl["mask"]["n"].value_or(3);
    }

    if (tbl.contains("cap") && tbl["cap"].is_table())
    {
        c.cap = true;
        c.cap_L_factor = tbl["cap"]["L_factor"].value_or(1.0);
        c.cap_W = tbl["cap"]["W"].value_or(0.1);
        c.cap_per_mode = tbl["cap"]["per_mode"].value_or(false);
        // The per-mode CAP defaults to a smoother onset: the reflection at the start of the
        // layer, not its strength, limits it once the strength is tuned (n = 3 leaves ~1e-10).
        c.cap_n = tbl["cap"]["n"].value_or(c.cap_per_mode ? 6 : 3);
        c.cap_reflection = tbl["cap"]["reflection"].value_or(1e-10);
        c.cap_tune_steps = tbl["cap"]["tune_steps"].value_or(20);
        c.cap_k_min = tbl["cap"]["k_min"].value_or(0.0);
    }

    if (tbl.contains("self_gravity") && tbl["self_gravity"].is_table())
    {
        auto sgt = tbl["self_gravity"];
        c.self_gravity = sgt["enabled"].value_or(true);
        c.sg_l_max = sgt["l_max"].value_or(-1);
        c.sg_recoil = sgt["recoil"].value_or(true);
        c.sg_recoil_taper = sgt["recoil_taper"].value_or(0.0);
        c.sg_accretion_recoil = sgt["accretion_recoil"].value_or(true);
        c.sg_ramp_time = sgt["ramp_time"].value_or(0.0);
        c.sg_scheme = sgt["scheme"].value_or(std::string("extrapolated"));
        c.sg_tolerance = sgt["tolerance"].value_or(1e-10);
        c.sg_max_iterations = sgt["max_iterations"].value_or(20);
        c.sg_measured_reference = sgt["measured_reference"].value_or(true);
        c.sg_levels_n_max = sgt["levels_n_max"].value_or(5);
        if (c.sg_scheme != "extrapolated" && c.sg_scheme != "iterated")
            throw std::invalid_argument("self_gravity.scheme must be \"extrapolated\" or \"iterated\"");
        if (!(c.sg_ramp_time >= 0.0 && c.sg_recoil_taper >= 0.0 && c.sg_tolerance > 0.0 &&
              c.sg_max_iterations >= 1 && c.sg_levels_n_max >= 1))
            throw std::invalid_argument("self_gravity: ramp_time and recoil_taper must be >= 0, tolerance > 0, "
                                        "max_iterations and levels_n_max >= 1");
        if (c.self_gravity && c.secondary_potential)
            throw std::invalid_argument("self_gravity needs an isolated hole at the origin (binary.q = 0)");
    }

    if (tbl.contains("pruning") && tbl["pruning"].is_table())
    {
        // enabled = false keeps the table (and its settings) but evolves every mode.
        c.pruning = tbl["pruning"]["enabled"].value_or(true);
        c.prune_tolerance = tbl["pruning"]["tolerance"].value_or(1e-8);
        c.prune_check_every = tbl["pruning"]["check_every"].value_or(10);
        c.prune_flux_tolerance = tbl["pruning"]["flux_tolerance"].value_or(1e-3);
        if (!(c.prune_tolerance > 0.0 && c.prune_tolerance < 1.0))
            throw std::invalid_argument("pruning.tolerance must lie in (0, 1)");
        if (c.prune_check_every < 1)
            throw std::invalid_argument("pruning.check_every must be at least 1");
        if (!(c.prune_flux_tolerance >= 0.0 && c.prune_flux_tolerance < 1.0))
            throw std::invalid_argument("pruning.flux_tolerance must lie in [0, 1)");
    }

    return c;
}

#endif
