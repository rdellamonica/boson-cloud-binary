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
 * rank-one s-wave sink of each black hole (kept outside @ref H()).
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
    double dr;       ///< Radial spacing (the smallest one on a stretched grid).
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

    /// Initial periods between the end of the startup ramp and the start of
    /// orbital evolution: the companion is fully on but R and the masses stay fixed.
    double orbit_delay_periods = 0.0;

    /// Once orbital evolution starts, the back-reaction (GW decay and cloud
    /// power acting on @c R) is switched on with a smooth step over this many
    /// initial periods; 0 switches it on at once.
    double orbit_ramp_periods = 1.0;

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
    /// Quadrature weight of each node: dr on a uniform grid, (r_{i+1} - r_{i-1}) / 2 otherwise.
    Eigen::VectorXd node_weight;

    /**
     * @brief Replace the uniform grid by the nodes @p nodes (e.g. from make_radial_grid()).
     *
     * Call right after construction, before the initial state is set and
     * before initialize(). On a non-uniform grid the kinetic term uses the
     * three-point second difference
     * @f$u''_i \approx \frac{2}{h_- + h_+}\bigl[\frac{u_{i+1}-u_i}{h_+} - \frac{u_i-u_{i-1}}{h_-}\bigr]@f$,
     * which is self-adjoint in the inner product weighted by @ref node_weight,
     * so Crank-Nicolson conserves @f$\sum_i w_i |U_i|^2@f$ exactly. Every norm,
     * energy and flux uses those weights.
     * @throws std::invalid_argument for a wrong size or nodes that do not increase.
     * @throws std::logic_error after initialize().
     */
    void set_radial_grid(const Eigen::VectorXd &nodes);
    /// True for the default, uniformly spaced grid.
    bool uniform_radial_grid() const { return uniform_grid; }
    Eigen::VectorXcd psi; ///< Current state, stacked radial functions U_lm.
    /**
     * @name Adaptive mode pruning
     *
     * With @c prune_tolerance > 0, only an active subset of the modes is
     * evolved: the potential couplings, the sinks and the linear solve are
     * restricted to it, so the cost of a step scales with the square of the
     * number of active modes instead of all @ref modes. Inactive modes are held
     * at exactly zero. Every @c prune_check_every steps each mode's share of
     * the norm, @f$p_j@f$, and the share it would receive from the active
     * modes before the next check, @f$d_j@f$, are compared with the
     * tolerance. The received amplitude is
     * @f$\sum_k V_{jk}\psi_k \min(\tau, 1/\Delta E_{jk})@f$, with @f$\tau@f$
     * the time between checks and @f$\Delta E_{jk}(r)@f$ the local mismatch of
     * the two modes' centrifugal energies, after which an off-resonant
     * response stops growing. The ionized field, however, lives in modes
     * that hold a tiny share of the norm while carrying most of the outgoing
     * flux, and its norm grows only slowly as it leaves. So each mode's share
     * of the outgoing flux, @f$f_j@f$, is compared with
     * @c prune_flux_tolerance: the larger of its share of the flux the active
     * modes drive into it (mode_ionization_flux()) and, once run() has set
     * @c r_ex, its share of the flux measured there. An inactive mode is
     * switched on when @f$p_j@f$ or @f$d_j@f$ exceeds the tolerance, or
     * @f$f_j@f$ the flux tolerance, and an active one is switched off, and its
     * amplitude dropped, when all three fall below 1% of their thresholds.
     * Set these before initialize().
     */
    /// @{
    double prune_tolerance = 0.0; ///< Norm fraction that makes a mode relevant; 0 evolves every mode.
    int prune_check_every = 10;   ///< Steps between updates of the active set.
    /// Share of the outgoing flux that makes a mode relevant; 0 leaves the flux out.
    double prune_flux_tolerance = 1e-3;
    /// Number of modes currently evolved.
    int active_mode_count() const;
    /// Whether mode @p j is currently evolved.
    bool mode_is_active(int j) const { return active_modes[j] != 0; }
    /// Cloud mass removed so far by switching modes off.
    double pruned_mass() const { return pruned_mass_total; }
    /// Mass absorbed so far by black hole @p body (1 or 2), whether or not the orbit evolves.
    double absorbed_mass(int body) const { return body == 1 ? absorbed_mass_1 : absorbed_mass_2; }
    /// @}

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
    SparseComplex H0;   ///< Free radial Hamiltonian, including the centrifugal term.
    SparseComplex CAP;  ///< Diagonal complex absorbing potential.
    /**
     * The evolution applies H without storing it (hamiltonian_times()); the
     * sparse matrices below are assembled on demand, for tests and analysis.
     */
    /// Conservative Hamiltonian H0 + V (+ CAP), as in the last update of V.
    SparseComplex H() const;
    /// Active potential V1 + ramp * V2, as in its last update.
    SparseComplex V() const;
    /// Potential of the primary, as in the last update of V (built on demand).
    SparseComplex V1() const;
    /// Potential of the companion, as in the last update of V (built on demand).
    SparseComplex V2() const;
    /// Time derivative of V from the orbital rotation (built on demand).
    SparseComplex dVdt() const;
    /// @}

    /// Apply the absorption operator @f$\Gamma = \sum_a g_a |v_a\rangle\langle v_a|@f$ to @p state.
    Eigen::VectorXcd apply_sink(const Eigen::VectorXcd &state) const;

    /// Return @f$-i H_{\rm full}\,\psi = -i H \psi - \Gamma\psi/2@f$, the Schrödinger right-hand side.
    Eigen::VectorXcd schrodinger_rhs(const Eigen::VectorXcd &state) const;

    /// H @p state, without assembling H.
    Eigen::VectorXcd hamiltonian_times(const Eigen::VectorXcd &state) const;

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
    /// Time at which the orbit starts evolving: the startup ramp plus the hold.
    double orbit_start_time() const { return ramp_end_time() + orbit_delay_periods * initial_period; }
    /// Weight of the orbital back-reaction at time @p time, in [0, 1].
    double orbit_ramp_factor(double time) const;

    Eigen::VectorXcd psi_last; ///< State at the start of the current step.
    Eigen::VectorXcd damping_mask;

    std::string config_file;

    double ramp_factor = 0.0;           ///< Smoothstep startup ramp in [0, 1].
    double potential_ramp_factor = 0.0; ///< Weight of V2: @c ramp_factor or 1.

    double r_ex;    ///< Extraction radius.
    int idx_ex = -1; ///< Radial index of the extraction radius (set by run()).

    std::string output_folder;

    double t_i = 0.0; ///< Current simulation time.

    RadialBlockPreconditioner block_preconditioner;
    Eigen::IncompleteLUT<dcomplex> ilut_preconditioner;

    /**
     * @brief Solve the CN system @f$A_{\rm full}\psi = \mathrm{rhs}@f$ into @c psi.
     *
     * @p rhs and the solve hold only the active modes (see gather_active());
     * @f$A = 1 + i\,dt\,H/2@f$ is applied from H without being stored.
     * Uses the radial block preconditioner and falls back to ILUT. When
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
    bool uniform_grid = true;
    /// Last node with r_i - r_0 <= @p offset (floor(offset / dr) on a uniform grid).
    int node_at(double offset) const;
    /// dU/dr at interior node @p i of the radial function starting at @p u.
    dcomplex radial_derivative(const dcomplex *u, int i) const;
    /// @f$\sum_i w_i |u_i|^2@f$ over @p count nodes starting at node @p first of a block.
    double radial_norm2(const dcomplex *u, int first, int count) const;

    /// Build @ref H0 with second-order finite differences plus the centrifugal term.
    void build_stationary_hamiltonian();

    /// Precompute the Gaunt coefficients of every coupled mode pair.
    void build_coupling_cache();
    /// Update the coupling profiles and pair phases for the current orbit.
    void build_static_perturbation();
    /// Update the ramp, Omega, potentials, absorbers and H for time @p t_i.
    void build_time_dependent_perturbation(double t_i);

    /// One multipole contribution to a mode pair.
    struct AngularTerm
    {
        int l_star;       ///< Multipole order of the potential.
        dcomplex angular; ///< Angular (Gaunt) coefficient at the equator.
        double weight;    ///< @f$-4\pi/(2\ell_*+1)@f$ times the (real) angular coefficient.
    };

    /// All multipoles coupling mode @c row_mode to mode @c col_mode.
    struct ModePair
    {
        int row_mode;
        int col_mode;
        int m_star; ///< Azimuthal number of the coupling, @c m_col - m_row.
        std::vector<AngularTerm> terms;
    };

    std::vector<ModePair> all_mode_pairs; ///< Couplings between every pair of modes, grouped by row.
    std::vector<ModePair> mode_pairs;     ///< Couplings between active modes, grouped by row.

    /// @name Adaptive mode pruning
    /// @{
    std::vector<char> active_modes;       ///< 1 for evolved modes, 0 for modes held at zero.
    double pruned_mass_total = 0.0;
    /// Cumulative mass taken up by each sink, written to mass_budget.txt.
    double absorbed_mass_1 = 0.0, absorbed_mass_2 = 0.0;
    /// True when load_checkpoint() restored the absorbed masses.
    bool checkpoint_has_budget = false;
    std::vector<char> checkpoint_active_modes; ///< Active set read from checkpoint.bin, if stored.
    /// Log of the active set, active_modes.txt: one line "step t count flags..."
    /// per change, where the set applies from that step on.
    std::ofstream active_modes_file;
    /// Append the current active set to active_modes.txt, as in effect from @p step_id.
    void write_active_modes(int step_id);
    /// Truncate active_modes.txt to the resumed snapshot and log the set the run continues with.
    void resume_active_modes_log(int start);
    bool pruning_enabled() const { return prune_tolerance > 0.0; }
    /// Norm share gained by each mode from the active ones over @p horizon, @f$d_j@f$.
    std::vector<double> mode_drive(double horizon) const;
    /// Norm of one radial block for shares of the norm: plain on a uniform grid, weighted otherwise.
    double block_norm2(const dcomplex *u) const;
    /// Sum of block_norm2() over every mode of @p state.
    double state_norm2(const Eigen::VectorXcd &state) const;
    /**
     * @brief Outgoing flux each mode would carry, driven by the active modes.
     *
     * Each coupling of an active mode k into mode j is a source at the energy
     * @f$\omega_k - s\,m_*\Omega@f$, with @f$\omega_k@f$ the mode's current
     * frequency (its Rayleigh quotient of H) and @f$s = \pm1@f$ the sense of
     * the orbit. Above zero energy the source radiates, and the steady flux
     * is @f$(4\mu^2/k)\,|\int\phi_{l_j}(r) S(r)\,dr|^2@f$ with
     * @f$\phi_{l_j}@f$ the regular solution in the monopole potential at that
     * energy, normalised to unit amplitude at infinity (golden rule). Sources
     * at different energies add incoherently.
     */
    std::vector<double> mode_ionization_flux();
    /**
     * @brief Update the active set; return true if it changed.
     * @param horizon Time until the next check.
     * @param from_scratch Treat every mode as a candidate (initial state or resume
     *        without a stored set).
     */
    bool update_active_modes(double horizon, bool from_scratch = false);
    /// Rebuild the sparse operators of the current active set at time @c t_i.
    void apply_active_set();
    /// @}
    std::vector<double> radial_f1, radial_f2;    ///< Cached F1, F2 indexed l*Nr+i.

    /// @f$e^{i m \theta}@f$ for the current orbital angle.
    dcomplex orbital_phase(int m_star) const;

    /**
     * @name Coupling profiles
     *
     * Per active mode pair and interior radial node, the radial profile of each body's coupling,
     * @f$\sum_{\ell_*} -4\pi/(2\ell_*+1)\,G_{\ell_*} F_a(r)@f$, without the
     * mass, orbital phase and parity factors. They depend only on the
     * positions of the bodies, so on a fixed orbit they are computed once and
     * each step only applies the phase @f$e^{i m_*\theta}@f$.
     */
    /// @{
    std::vector<double> coupling1, coupling2; ///< Indexed p * (Nr - 2) + i - 1.
    double coupling_R1 = -1.0, coupling_R2 = -1.0, coupling_q = -1.0;
    /// Set when the active set changed: coupling1/2 still hold the previous
    /// pairs' profiles, and @c coupling_source maps each pair to its old index (or -1).
    bool couplings_stale = true;
    std::vector<int> coupling_source;
    /// coupling1 + parity * ramp * coupling2, for the ramp @c coupling_eff_ramp.
    std::vector<double> coupling_eff;
    double coupling_eff_ramp = -1.0;
    /// @f$\mu (M_1 + M_2) e^{i m_* \theta}@f$ of each pair: V_p(r) = pair_phase[p] * coupling_eff(r).
    std::vector<dcomplex> pair_phase;
    /// Recompute the radial factors and coupling profiles if the orbit's geometry or the ramp changed.
    void refresh_couplings();
    /// Orbit state used by the last build_static_perturbation().
    struct PotentialState
    {
        double theta = 0.0, strength = 0.0, ramp = 0.0, Omega = 0.0;
    } potential_state;
    /// Sparse matrix with entries @f$e^{i m_*\theta}\mu M (w_1 C_1 + w_2 p\, C_2)@f$, times @f$i s m_*\Omega@f$ if @p derivative.
    SparseComplex potential_matrix(double w1, double w2, bool derivative) const;
    /// @}

    /**
     * @name Matrix-free Hamiltonian
     *
     * H = H0 + sum over pairs of pair_phase * coupling_eff(r) (+ CAP) is
     * applied block by block, without storing it. The CN solve works on
     * compact vectors that hold only the radial blocks of the active modes,
     * in the order of @c active_list; inactive modes are zero and uncoupled,
     * so they drop out exactly.
     */
    /// @{
    bool operator_ready = false;           ///< Set by build_active_operator().
    std::vector<int> active_list;          ///< Indices of the active modes.
    std::vector<int> mode_position;        ///< Position of each mode in active_list, or -1.
    std::vector<std::size_t> row_pair_start; ///< mode_pairs of row mode j: [start[j], start[j + 1]).
    std::vector<int> self_pair;            ///< The pair (j, j) of each mode, or -1.
    std::vector<double> h0_diagonal;       ///< Diagonal of H0, indexed j * Nr + i.
    double h0_offdiagonal = 0.0;           ///< Off-diagonal of H0, -1 / (2 mu dr^2).
    std::vector<double> h0_lower, h0_upper; ///< Off-diagonals of H0 at each node (any grid).
    Eigen::VectorXcd cap_diagonal;         ///< Diagonal of CAP, when @c cap.
    /// Select the active couplings and build the operator's index tables.
    void build_active_operator();
    /**
     * @brief @p y = H @p x, block by block.
     * @param compact Vectors hold the active blocks only (as in active_list);
     *        otherwise they hold every mode, and inactive blocks get H0 (+ CAP).
     */
    void apply_hamiltonian(const Eigen::VectorXcd &x, Eigen::VectorXcd &y, bool compact) const;
    /// The active blocks of @p full, in order.
    Eigen::VectorXcd gather_active(const Eigen::VectorXcd &full) const;
    /// Write the compact @p compact into the active blocks of @p full.
    void scatter_active(const Eigen::VectorXcd &compact, Eigen::VectorXcd &full) const;
    /// 1 + @p scale H restricted to the active modes, as a sparse matrix (ILUT fallback).
    SparseComplex active_cn_operator(dcomplex scale) const;
    /// @}

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
        /// Left vector: @f$\Gamma = g\,v\,w^\dagger@f$ with w = W v on a non-uniform
        /// grid (strength A_h), so that Gamma is self-adjoint in the weighted product;
        /// empty on a uniform grid, where w = v and g = A_h dr.
        Eigen::VectorXcd w;
        double strength = 0.0;
        const Eigen::VectorXcd &left() const { return w.size() ? w : v; }
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

    /// Configure both preconditioners.
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
    /// Cumulative absorbed (hole 1, hole 2) and pruned mass per step. Kept out of
    /// diagnostic_files() so that runs written before it existed still resume.
    std::ofstream mass_budget_file;
    /// @}

    /// Probability flux of each mode through the extraction radius.
    std::vector<double> get_flux(const Eigen::VectorXcd &psi);
    /// Energy flux of each mode through the extraction radius. @p dpsi_dt is the
    /// actual derivative @f$-iH(t)\psi(t)@f$, at the same time as @p psi.
    std::vector<double> get_energy_flux(const Eigen::VectorXcd &psi, const Eigen::VectorXcd &dpsi_dt);

    /// Real part of @f$\langle\psi|A\psi\rangle@f$ restricted to nodes 0..@p i_cut, given @p Apsi.
    double restricted_expectation(const Eigen::VectorXcd &psi, const Eigen::VectorXcd &Apsi, int i_cut) const;
    /**
     * @brief @f$\langle\psi|\partial_t V|\psi\rangle@f$ and, if @p dpsi_dt is given,
     * @f$\mathrm{Re}\langle\dot\psi|V|\psi\rangle@f$, inside @c r_ex, in one pass over V.
     */
    void binary_power_terms(const Eigen::VectorXcd &psi, const Eigen::VectorXcd *dpsi_dt,
                            double &power, double &dV) const;
    /// <psi| dV/dt |psi> with the potential rotated by @p theta_shift. Called on the step's
    /// midpoint state with V(t_n+1), it is Omega_z times the torque that, together with the
    /// sinks' absorbed L_z, gives the CN step's exact change of <L_z>.
    double step_binary_power(const Eigen::VectorXcd &psi_old, double theta_shift) const;
    double p_bin_step = 0.0;
    /// Rate at which the sinks take angular momentum L_z (about the COM) out of the field,
    /// -d<L_z>/dt from the non-Hermitian part. It belongs to the holes, i.e. to the orbit.
    double absorbed_Lz_rate(const Eigen::VectorXcd &psi) const;
    double Lz_abs_rate = 0.0;
    /// Lz_absorbed_rate.txt: Lz_abs_rate per step (absent in runs written before it existed).
    std::ofstream Lz_abs_file;

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
    /// Energy of each mode inside the extraction radius, given @p Hpsi = H @p psi.
    std::vector<double> energy_per_mode(const Eigen::VectorXcd &psi, const Eigen::VectorXcd &Hpsi) const;

    /// Append one value as a line of @p quantity_file.
    void write_quantity_line(const double &quantity, std::ofstream &quantity_file);
    /// Append one value per mode as a space-separated line of @p quantity_file.
    void write_quantity_line_modes(const std::vector<double> &quantity, std::ofstream &quantity_file);
};

#endif
