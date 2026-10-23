/**
 * @file CrankNicolsonModes.hpp
 * @brief Crank–Nicolson evolution of a scalar cloud in a binary black-hole
 *        potential, expanded in spherical-harmonic modes about the centre of mass.
 *
 * The field is written as
 * @f[
 *   \psi(t, r, \theta, \phi) = \sum_{\ell m} \frac{U_{\ell m}(t, r)}{r}\, Y_{\ell m}(\theta, \phi),
 * @f]
 * and the radial functions @f$U_{\ell m}@f$ of all retained modes are stacked
 * into one vector of length @c Nr * n_modes (mode-major: entry @c j*Nr+i is
 * mode @c j at radial node @c i). The Hamiltonian is the free radial operator
 * plus the multipole expansion of the two Newtonian potentials, which couples
 * the angular modes. Optional ingredients are an s-wave absorber on each black
 * hole, a complex absorbing potential (CAP) or damping mask at the outer
 * boundary, and back-reaction of the cloud on the orbit.
 */
#ifndef CRANK_NICOLSON_MODES_HPP
#define CRANK_NICOLSON_MODES_HPP

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include "Definitions.hpp"
#include "RollingPhase.hpp"
#include "RadialBlockPreconditioner.hpp"
#include "Utils.hpp"
#include "Config.hpp"

/// Row-major complex sparse matrix, so Eigen can parallelise sparse products.
using SparseComplex = Eigen::SparseMatrix<dcomplex, Eigen::RowMajor>;

/**
 * @brief Mode-decomposed Schrödinger solver for a gravitational atom in a binary.
 *
 * Units are @f$G = c = 1@f$ with the primary mass @f$M_1 = 1@f$ at start. The
 * primary sits at distance @c R1 and the companion (mass @c q) at @c R2 from
 * the centre of mass, separated by @c R.
 *
 * Typical use:
 * @code
 *   CrankNicolsonModes cn(N, L, dt, epsilon, l_max, mu, M_c, q, a, R, counter,
 *                         absorption, save_every, ramp_periods, epsilon_smear,
 *                         "config.toml");
 *   cn.set_initial_state_full(load_initial_modes_binary("initial_modes.bin", cn.modes));
 *   cn.initialize();
 *   cn.run(num_steps, r_ex);
 * @endcode
 *
 * Each step solves @f$(1 + i\,dt\,H_{\rm full}/2)\,\psi^{n+1} = (1 - i\,dt\,H_{\rm full}/2)\,\psi^n@f$
 * with BiCGSTAB, where @f$H_{\rm full} = H - i\Gamma/2@f$ and @f$\Gamma@f$ is the
 * rank-one s-wave sink of each black hole (kept outside the sparse @ref H).
 */
class CrankNicolsonModes
{
public:
    /**
     * @brief Build the grid, the mode list and the free Hamiltonian.
     *
     * @param Nx            Number of radial nodes (>= 3).
     * @param L             Outer radius of the radial domain.
     * @param dt            Initial timestep (kept until the startup ramp ends).
     * @param epsilon       Offset of the first radial node, @f$r_i = \epsilon + i\,dr@f$.
     * @param l_max         Largest retained multipole @f$\ell@f$.
     * @param mu            Boson mass @f$\mu@f$.
     * @param M_c           Cloud mass; the initial state is scaled to it.
     * @param q             Mass ratio @f$M_2/M_1@f$.
     * @param a             Primary spin (stored; not used by the evolution).
     * @param R             Initial binary separation.
     * @param counter       @c true for a counter-rotating orbit.
     * @param absorption    Enable the s-wave black-hole absorbers.
     * @param save_every    Snapshot / checkpoint interval in steps.
     * @param ramp_periods  Length of the startup ramp in initial orbital periods.
     * @param epsilon_smear Absorber support radius in units of each body's mass.
     * @param config_file   Configuration copied into the output folder.
     * @throws std::invalid_argument for an invalid grid or timestep.
     */
    CrankNicolsonModes(int Nx, double L, double dt, double epsilon, int l_max, double mu, double M_c, double q, double a, double R, bool counter, bool absorption, int save_every, double ramp_periods, double epsilon_smear, std::string config_file);

    /// @name Grid, modes and physical parameters
    /// @{
    int Nr;          ///< Number of radial nodes.
    int save_every;  ///< Snapshot interval in steps.
    int l_max;       ///< Largest retained multipole.
    int n_modes;     ///< Number of retained (l, m) modes.
    double epsilon;  ///< Radial offset of the first node.
    double mu;       ///< Boson mass.
    double q;        ///< Current mass ratio M2/M1.
    double R;        ///< Current binary separation.
    double theta;    ///< Orbital angle of the primary (sign follows the rotation sense).
    double phase;    ///< Accumulated orbital phase, always increasing.
    double Omega;    ///< Current orbital frequency (ramped at startup).
    double Omega_0;  ///< Keplerian frequency at the initial separation.
    double P;        ///< Current orbital period.
    double R1;       ///< Distance of the primary from the centre of mass.
    double R2;       ///< Distance of the companion from the centre of mass.
    double M1;       ///< Primary mass (grows by accretion).
    double M2;       ///< Companion mass (grows by accretion).
    double a;        ///< Primary spin (informational).
    double L;        ///< Outer radius.
    double dr;       ///< Radial spacing.
    double dt;       ///< Current timestep.
    bool counter;    ///< Counter-rotating orbit.
    double M_c;      ///< Cloud mass.
    /// @}

    /**
     * @brief Advance the state by one timestep and append all diagnostics.
     * @param step_id Index of the time sample produced by this step.
     */
    void step(int step_id);

    /**
     * @brief Start a new run in a fresh @c Simulation_<date> folder.
     * @param n_steps Number of time samples, including @f$t = 0@f$.
     * @param r_ex    Extraction radius for fluxes and inner-region integrals.
     */
    void run(int n_steps, double r_ex);

    /**
     * @brief Continue the run stored in @p folder from its last complete snapshot.
     *
     * Diagnostics are truncated to that snapshot's row and appended from there.
     * The exact state is read from @c checkpoint.bin when it matches the snapshot;
     * otherwise the orbit is rebuilt from the diagnostic files.
     */
    void resume(const std::string &folder, int n_steps, double r_ex);

    /// Feed the binary power and accretion back onto the orbit. Absorption of
    /// the field is controlled separately by the constructor's @c absorption.
    bool evolve_orbit = true;

    /// The startup ramp always takes @c Omega from 0 to @c Omega_0. When true it
    /// also ramps the companion's potential; when false the companion is
    /// present at full strength from @f$t = 0@f$.
    bool ramp_secondary_potential = true;

    /// With orbital evolution, the post-ramp timestep is @c P / steps_per_period.
    double steps_per_period = 50.0;

    /**
     * @brief Set the number of orbits in the rolling phase average of @c p_bin.
     *
     * Call before run(); a resumed run keeps the window stored in its checkpoint.
     * @throws std::invalid_argument unless @p n_orbits > 0.
     */
    void set_power_average_orbits(double n_orbits);

    /// Use the radial block preconditioner (default) instead of a full ILUT.
    bool use_block_preconditioner = true;
    double last_solver_residual = 0.0;  ///< Relative residual of the last CN solve.
    int last_solver_iterations = 0;     ///< BiCGSTAB iterations of the last CN solve.

    Eigen::VectorXd t;    ///< Time of each sample.
    Eigen::VectorXd r;    ///< Radial nodes.
    Eigen::VectorXcd psi; ///< Current state, stacked radial functions U_lm.
    bool mask = false;    ///< A damping mask is applied after every step.
    bool cap = false;     ///< A complex absorbing potential is part of H.

    /**
     * @brief Set the initial state from all modes, scaling it to the cloud mass.
     * @param psi_0 Unit-norm stacked state of size @c Nr * n_modes, ordered as @ref modes.
     * @throws std::invalid_argument if the size does not match.
     */
    void set_initial_state_full(Eigen::VectorXcd psi_0);

    /// Retained modes; only @f$\ell - m@f$ even (equatorial symmetry) are kept.
    std::vector<mode> modes;

    /**
     * @brief Build the coupling caches, the initial potential and the CN matrices.
     *
     * Call after the grid and modes exist and before step() or run(). The sparse
     * matrices must not be structurally modified afterwards, because their
     * compressed offsets are cached.
     */
    void initialize();

    /**
     * @brief Replace the binary by one stationary mass-1 black hole at (x, 0, 0).
     *
     * Call before initialize(). There is no rotation or companion, and the
     * constructor's @c epsilon_smear is then the physical absorber radius.
     * @throws std::invalid_argument if the hole lies outside the domain.
     * @throws std::logic_error if called after initialize().
     */
    void configure_isolated_black_hole(double x);

    /// @name Operators
    /// @{
    SparseComplex H;    ///< Conservative Hamiltonian H0 + V (+ CAP).
    SparseComplex H0;   ///< Free radial Hamiltonian, including the centrifugal term.
    SparseComplex A;    ///< CN left operator 1 + i dt H / 2.
    SparseComplex B;    ///< CN right operator 1 - i dt H / 2.
    SparseComplex V1;   ///< Potential of the primary.
    SparseComplex V2;   ///< Potential of the companion.
    SparseComplex V;    ///< Active potential V1 + ramp * V2.
    SparseComplex dVdt; ///< Time derivative of the active potential from the orbital rotation.
    SparseComplex Id;   ///< Identity, used to fix the union sparsity pattern.
    SparseComplex CAP;  ///< Diagonal complex absorbing potential.
    /// @}

    /// Apply the absorption operator @f$\Gamma = \sum_a g_a |v_a\rangle\langle v_a|@f$ to @p state.
    Eigen::VectorXcd apply_sink(const Eigen::VectorXcd &state) const;

    /// Return @f$-i H_{\rm full}\,\psi = -i H \psi - \Gamma\psi/2@f$, the Schrödinger right-hand side.
    Eigen::VectorXcd schrodinger_rhs(const Eigen::VectorXcd &state) const;

    /// Largest eigenvalue @f$g_a |v_a|^2@f$ of one body's sink (@p body = 1 or 2); zero when inactive.
    double sink_strength(int body) const;

    /// True once the absorbers are part of the evolution operator.
    bool absorption_sink_active() const { return sink_active; }

    /**
     * @brief Build a multiplicative damping mask for the outer region.
     *
     * For @f$r \ge r_m = f L@f$ every mode is multiplied after each step by
     * @f$\exp[-W ((r - r_m)/((1-f) L))^n]@f$.
     * @param mask_L_factor Fraction @f$f@f$ of the domain where the mask starts.
     * @param mask_W        Strength @f$W@f$.
     * @param mask_n        Power @f$n@f$.
     */
    void build_damping_mask(double mask_L_factor, double mask_W, int mask_n);

    /**
     * @brief Build a diagonal complex absorbing potential for the outer region.
     * @param cap_L_factor Fraction of the domain where the CAP starts.
     * @param cap_W        Strength.
     * @param cap_n        Power of the profile.
     */
    void build_cap(double cap_L_factor, double cap_W, int cap_n);

private:
    bool isolated_black_hole = false;
    double ramp_periods;
    /// Initial orbital period. Startup schedules use it, never the evolving @c P.
    const double initial_period;
    /// Time at which the startup ramp ends and the orbit may evolve.
    double ramp_end_time() const { return ramp_periods * initial_period; }

    Eigen::VectorXcd psi_last; ///< State at the start of the current step.
    Eigen::VectorXcd damping_mask;

    std::string config_file;

    double ramp_factor = 0.0;           ///< Smoothstep startup ramp in [0, 1].
    double potential_ramp_factor = 0.0; ///< Weight of V2: @c ramp_factor or 1.

    double r_ex;    ///< Extraction radius.
    int idx_ex;     ///< Radial index of the extraction radius.

    std::string output_folder;

    double t_i = 0.0; ///< Current simulation time.

    Eigen::BiCGSTAB<SparseComplex, RadialBlockPreconditioner> solver;
    Eigen::BiCGSTAB<SparseComplex, Eigen::IncompleteLUT<dcomplex>> ilut_solver;

    /**
     * @brief Solve the CN system @f$A_{\rm full}\psi = \mathrm{rhs}@f$ into @c psi.
     *
     * Uses the radial block preconditioner and falls back to full ILUT. When
     * the sinks are active, the preconditioner is corrected for them with the
     * Woodbury identity. Every accepted solution satisfies a relative
     * residual of at most 1e-9.
     * @throws std::runtime_error if no preconditioner reaches the tolerance.
     */
    void solve_CN(const Eigen::VectorXcd &rhs);

    double orbits_average = 2.0;
    RollingPhaseAverage power_average{orbits_average};

    /// Radial factor of the primary's multipole @p l_star, @f$r_<^{\ell}/r_>^{\ell+1}@f$ with mass weight.
    double F1(double r, int l_star, int m_star);
    /// Radial factor of the companion's multipole @p l_star, including its @f$(-1)^{m}@f$ parity.
    double F2(double r, int l_star, int m_star);

    void build_modes(); ///< Fill @ref modes with all (l, m), l <= l_max, l - m even.
    void build_grid();  ///< Fill @ref r with the radial nodes.

    /// Build @ref H0 with second-order finite differences plus the centrifugal term.
    void build_stationary_hamiltonian();

    /// Precompute the Gaunt coefficients of every coupled mode pair.
    void build_coupling_cache();
    /// Evaluate V1, V2, V and dVdt for the current orbit.
    void build_static_perturbation();
    /// Update the ramp, Omega, potentials, absorbers and H for time @p t_i.
    void build_time_dependent_perturbation(double t_i);

    /// One multipole contribution to a mode pair.
    struct AngularTerm
    {
        int l_star;       ///< Multipole order of the potential.
        dcomplex angular; ///< Angular (Gaunt) coefficient at the equator.
    };

    /// All multipoles coupling mode @c row_mode to mode @c col_mode.
    struct ModePair
    {
        int row_mode;
        int col_mode;
        int m_star; ///< Azimuthal number of the coupling, @c m_col - m_row.
        std::vector<AngularTerm> terms;
        /// Positions in the shared, compressed perturbation sparsity pattern.
        std::vector<Eigen::Index> offsets;
    };

    std::vector<ModePair> mode_pairs;
    std::vector<double> radial_f1, radial_f2;    ///< Cached F1, F2 indexed l*Nr+i.
    std::vector<dcomplex> stationary_values;     ///< H0 values in H's sparsity pattern.
    std::vector<Eigen::Index> potential_offsets; ///< Map from H slots to V slots (-1 if none).
    /// Refill H in place from H0, V and CAP; @p include_absorption activates the sinks.
    void assemble_hamiltonian(bool include_absorption);
    /// Refill A and B in place from H and the current dt.
    void update_CN_values();

    /// Build the fixed sparsity patterns of V, H, A, B and their offset caches.
    void build_sparse_element_caches();

    /// @f$e^{i m \theta}@f$ for the current orbital angle.
    dcomplex orbital_phase(int m_star) const;

    /// @name Absorption
    /// @{
    bool absorption;
    /// Legendre moments of one body's spherical kernel about the origin.
    struct AbsorptionProfile {
        double radius = -1.0, width = -1.0;
        /// @f$2\pi\int K(s)\,P_\ell(\cos\gamma)\,d\cos\gamma@f$, indexed l*Nr+i.
        std::vector<double> multipoles;
    };
    AbsorptionProfile absorption_profile1, absorption_profile2;
    /**
     * @brief Recompute a body's kernel moments if its position or width changed.
     *
     * The kernel is the compact bump @f$\exp[-1/(1 - s^2/h^2)]@f$ in the
     * distance @f$s@f$ from the body, normalised to unit volume on the grid.
     */
    void update_absorption_profile(AbsorptionProfile &profile, double radius, double width);
    double Ah1 = 0.0, Ah2 = 0.0; ///< Horizon-area factors 16 pi M_a^2 (times the absorption ramp).
    double epsilon_smear;
    /**
     * Rank-one s-wave sink of one body: @f$\Gamma = g\,v v^\dagger@f$, where
     * @f$dr\,v^\dagger U = \int K(|x - x_{BH}|)\,\psi\,d^3x@f$ and @f$g = A_h\,dr@f$.
     * It only sees the @f$\ell = 0@f$ component of @f$\psi@f$ about the black hole.
     */
    struct SwaveSink {
        Eigen::VectorXcd v;
        double strength = 0.0;
    };
    SwaveSink sink1, sink2;
    bool sink_active = false;
    int mode00 = -1; ///< Index of the (0, 0) mode, or -1.
    /// Build the projector vector @p v of one body from its kernel moments.
    void build_swave_projector(const AbsorptionProfile &profile, bool secondary, Eigen::VectorXcd &v) const;
    /// Update both absorbers for the current orbit and absorption ramp.
    void build_spherical_absorption();
    /// Mass accretion rate of the primary for @p state.
    double get_absorption_rate_1(const Eigen::VectorXcd &state) const;
    /// Mass accretion rate of the companion for @p state.
    double get_absorption_rate_2(const Eigen::VectorXcd &state) const;
    /// @}

    /**
     * @brief Evolve the orbit over @p orbit_dt.
     *
     * The separation shrinks by gravitational-wave emission and by the
     * phase-averaged cloud power @c p_bin; the masses grow by accretion. Omega,
     * P and, once the ramp is complete, dt follow the new separation.
     */
    void update_time_dependent_increment(double orbit_dt);

    double p_bin = 0.0; ///< Phase-averaged binary power.

    /// Allocate A and B and configure both iterative solvers.
    void build_CN_matrices();

    /// Write @p psi to @c output/name/snapshot_NNNNNN.txt as "re im" lines.
    void save_snapshot(int step_id, const Eigen::VectorXcd &psi, std::string name);
    /// Write config.toml, r.txt and modes.txt to the output folder.
    void save_run_info();
    /// Flush every diagnostic stream.
    void flush_diagnostics();

    /// @name Restart support
    /// @{
    /// Set @c r_ex and @c idx_ex, validating them against the grid.
    void set_extraction_radius(double r_ex_, int n_steps);
    /// Open every diagnostic file in the output folder with @p mode.
    void open_diagnostics(std::ios::openmode mode);
    /// Run steps [@p first_step, @p n_steps), stopping early if R <= 6.
    void evolve(int first_step, int n_steps);
    /// Every diagnostic stream with its file name.
    std::vector<std::pair<std::ofstream *, std::string>> diagnostic_files();
    /// Write the exact binary state to checkpoint.bin, replaced atomically.
    void save_checkpoint(int step_id) const;
    /// Restore the exact state from checkpoint.bin if it matches @p step_id.
    bool load_checkpoint(int step_id);
    /// Rebuild the orbit state from the diagnostic rows before @p step_id.
    void reconstruct_state_from_diagnostics(int step_id);
    /// @}

    /// @name Diagnostic files
    /// @{
    std::ofstream flux_file;
    std::ofstream energy_flux_file;
    std::ofstream energy_file;
    std::ofstream norms_file;
    std::ofstream time_file;
    std::ofstream energy_modes_file;
    std::ofstream R_file;
    std::ofstream theta_file;
    std::ofstream Omega_file;
    std::ofstream p_bin_file;
    std::ofstream p_bin_inst_file;
    std::ofstream d_V_bin_dt_file;
    std::ofstream M_1_file;
    std::ofstream q_file;
    /// @}

    /// Probability flux of each mode through the extraction radius.
    std::vector<double> get_flux(const Eigen::VectorXcd &psi);
    /// Energy flux of each mode through the extraction radius. @p dpsi_dt is the
    /// actual derivative @f$-iH(t)\psi(t)@f$, at the same time as @p psi.
    std::vector<double> get_energy_flux(const Eigen::VectorXcd &psi, const Eigen::VectorXcd &dpsi_dt);

    /// Real part of @f$\langle\psi|A|\psi\rangle@f$ restricted to nodes 0..@p i_cut.
    double restricted_expectation(const Eigen::VectorXcd &psi, const SparseComplex &A, int i_cut) const;

    /// @f$\mathrm{Re}\langle\dot\psi|V|\psi\rangle@f$ inside the extraction radius.
    double get_dVexpectation_dt(const Eigen::VectorXcd &psi, const Eigen::VectorXcd &dpsi_dt);

    /// Instantaneous power transferred by the binary, @f$\langle\psi|\partial_t V|\psi\rangle@f$ inside @c r_ex.
    double get_binary_power(const Eigen::VectorXcd &psi);

    /// Norm of each mode inside the extraction radius.
    std::vector<double> get_norms(const Eigen::VectorXcd &psi);
    /// Energy @f$\langle\psi|H|\psi\rangle@f$ inside the extraction radius.
    double get_energy(const Eigen::VectorXcd &psi);
    /// Energy of each mode inside the extraction radius.
    std::vector<double> get_energy_per_mode(const Eigen::VectorXcd &psi);

    /// Append one value as a line of @p quantity_file.
    void write_quantity_line(const double &quantity, std::ofstream &quantity_file);
    /// Append one value per mode as a space-separated line of @p quantity_file.
    void write_quantity_line_modes(const std::vector<double> &quantity, std::ofstream &quantity_file);
};

#endif
