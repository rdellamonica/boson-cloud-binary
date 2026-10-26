/**
 * @file Config.hpp
 * @brief Simulation configuration and its TOML loader.
 */
#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <iostream>
#include <string>
#include <stdexcept>
#include <toml++/toml.hpp>

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
    /// @}

    /// @name [simulation]
    /// @{
    double dt;                          ///< Initial timestep (default 10).
    double num_periods;                 ///< Run length in initial orbital periods (may be fractional), used if num_steps = 0.
    int save_every;                     ///< Snapshot interval in steps (default 50).
    double ramp_periods;                ///< Startup ramp length in initial periods (default 0).
    bool ramp_secondary_potential = true; ///< Ramp the companion's potential with Omega.
    double steps_per_period = 50.0;     ///< Post-ramp timestep is P / steps_per_period.
    double power_average_orbits = 2.0;  ///< Orbits in the rolling average of the binary power.
    bool evolve_orbit;                  ///< Feed the cloud power and accretion back on the orbit (default false).
    bool absorption;                    ///< Enable the black-hole absorbers (default false).
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
    int cap_n;           ///< Power of the profile (default 3).
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

    c.dt = tbl["simulation"]["dt"].value_or(10.0);
    c.num_periods = tbl["simulation"]["num_periods"].value_or(0.0);
    c.num_steps = tbl["simulation"]["num_steps"].value_or(0);
    c.save_every = tbl["simulation"]["save_every"].value_or(50);
    c.l_max = tbl["simulation"]["l_max"].value_or(5);
    c.r_ex = tbl["simulation"]["r_ex"].value_or(1000.0);
    c.ramp_periods = tbl["simulation"]["ramp_periods"].value_or(0.0);
    c.ramp_secondary_potential = tbl["simulation"]["ramp_secondary_potential"].value_or(true);
    c.steps_per_period = tbl["simulation"]["steps_per_period"].value_or(50.0);
    if (!(c.steps_per_period > 0.0))
        throw std::invalid_argument("simulation.steps_per_period must be positive");
    c.power_average_orbits = tbl["simulation"]["power_average_orbits"].value_or(2.0);
    if (!(c.power_average_orbits > 0.0))
        throw std::invalid_argument("simulation.power_average_orbits must be positive");
    c.epsilon = tbl["simulation"]["epsilon"].value_or(0.001);
    c.epsilon_smear = tbl["simulation"]["epsilon_smear"].value_or(2.0);
    c.evolve_orbit = tbl["simulation"]["evolve_orbit"].value_or(false);
    c.absorption = tbl["simulation"]["absorption"].value_or(false);

    c.q = tbl["binary"]["q"].value_or(0.001);
    c.R = tbl["binary"]["R"].value_or(50.0);
    c.counter = tbl["binary"]["counter"].value_or(false);

    c.mu = tbl["atom"]["mu"].value_or(0.2);
    c.n = tbl["atom"]["n"].value_or(2);
    c.l = tbl["atom"]["l"].value_or(1);
    c.m = tbl["atom"]["m"].value_or(1);
    c.a = tbl["atom"]["a"].value_or(0.0);
    c.M_c = tbl["atom"]["M_c"].value_or(0.0);

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
        c.cap_n = tbl["cap"]["n"].value_or(3);
    }

    return c;
}

#endif
