/**
 * @file CrankNicolsonModes_selfgravity.cpp
 * @brief Newtonian self-gravity of the cloud around an isolated hole at the origin,
 *        and the level/channel diagnostics used to test the gravitational Auger effect.
 *
 * Conventions (self_gravity_formalism.pdf): sum_lm int |U_lm|^2 dr = N = M_c / mu,
 * mass density mu |psi|^2, nabla^2 Phi = 4 pi mu |psi|^2, potential energy mu Phi.
 */
#include "CrankNicolsonModes.hpp"
#include <gsl/gsl_sf_coupling.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
/// G(lm; LM; l'm') = int Y*_lm Y_LM Y_l'm' dOmega (Condon-Shortley).
double gaunt(int l, int m, int L, int M, int lp, int mp)
{
    if (m != M + mp || L < std::abs(l - lp) || L > l + lp || (l + L + lp) % 2 != 0 || std::abs(M) > L)
        return 0.0;
    const double w1 = gsl_sf_coupling_3j(2 * l, 2 * L, 2 * lp, 0, 0, 0);
    const double w2 = gsl_sf_coupling_3j(2 * l, 2 * L, 2 * lp, -2 * m, 2 * M, 2 * mp);
    const double sign = (std::abs(m) % 2 == 0) ? 1.0 : -1.0;
    return sign * std::sqrt((2.0 * l + 1) * (2.0 * L + 1) * (2.0 * lp + 1) / (4.0 * M_PI)) * w1 * w2;
}

double smoothstep(double x)
{
    if (x <= 0.0)
        return 0.0;
    if (x >= 1.0)
        return 1.0;
    return x * x * (3.0 - 2.0 * x);
}
} // namespace

void CrankNicolsonModes::enable_self_gravity(const SelfGravityOptions &options)
{
    if (operator_ready)
        throw std::logic_error("Enable self-gravity before initialize()");
    if (options.ramp_time < 0.0 || options.recoil_taper < 0.0 || options.tolerance <= 0.0 ||
        options.max_iterations < 1 || options.levels_n_max < 1)
        throw std::invalid_argument("Invalid self-gravity options");
    sg = options;
}

double CrankNicolsonModes::sg_ramp(double time) const
{
    return sg.ramp_time > 0.0 ? smoothstep(time / sg.ramp_time) : 1.0;
}

void CrankNicolsonModes::build_self_gravity_tables()
{
    if (!isolated_black_hole || R1 != 0.0)
        throw std::invalid_argument("Self-gravity is implemented for an isolated hole at the origin only "
                                    "(binary.q = 0, or secondary_potential = false with the hole at x = 0)");
    if (pruning_enabled())
        throw std::invalid_argument("Self-gravity needs every mode: the Auger partners hold a tiny share of the "
                                    "norm. Remove [pruning] or set enabled = false");
    sg_L = sg.l_max < 0 ? 2 * l_max : std::min(sg.l_max, 2 * l_max);

    sg_multipoles.clear();
    sg_index.assign(static_cast<std::size_t>((sg_L + 1) * (sg_L + 1)), -1);
    for (int L = 0; L <= sg_L; ++L)
        for (int M = 0; M <= L; ++M)
            if ((L - M) % 2 == 0)
            {
                sg_index[L * (sg_L + 1) + M] = static_cast<int>(sg_multipoles.size());
                sg_multipoles.push_back({L, M});
            }

    // n_LM = sum_{j,k} G(l_k m_k; L M; l_j m_j) conj(U_j) U_k, m_k = m_j + M.
    sg_density_terms.assign(sg_multipoles.size(), {});
    for (std::size_t a = 0; a < sg_multipoles.size(); ++a)
    {
        const auto [L, M] = sg_multipoles[a];
        for (int j = 0; j < n_modes; ++j)
            for (int k = 0; k < n_modes; ++k)
            {
                const double g = gaunt(modes[k].l, modes[k].m, L, M, modes[j].l, modes[j].m);
                if (g != 0.0)
                    sg_density_terms[a].push_back({j, k, g});
            }
    }

    // W_jk = mu sum_L G(l_j m_j; L, m_j - m_k; l_k m_k) Phi_{L, m_j - m_k}, on the pairs of the operator.
    std::vector<int> pair_of(static_cast<std::size_t>(n_modes) * n_modes, -1);
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
        pair_of[mode_pairs[p].row_mode * n_modes + mode_pairs[p].col_mode] = static_cast<int>(p);
    sg_pair_terms.assign(mode_pairs.size(), {});
    for (int j = 0; j < n_modes; ++j)
        for (int k = 0; k < n_modes; ++k)
        {
            const int M = modes[j].m - modes[k].m;
            for (int L = std::abs(M); L <= sg_L; ++L)
            {
                const double g = gaunt(modes[j].l, modes[j].m, L, M, modes[k].l, modes[k].m);
                if (g == 0.0)
                    continue;
                const int p = pair_of[j * n_modes + k];
                if (p < 0)
                    throw std::logic_error("Self-gravity coupling outside the operator's mode pairs");
                const int a = sg_index[L * (sg_L + 1) + std::abs(M)];
                sg_pair_terms[p].push_back({a, g, M < 0, (M < 0 && std::abs(M) % 2 != 0) ? -1.0 : 1.0});
            }
        }

    // Ratios for the Green's-function recursions, all <= 1.
    sg_inner_ratio.assign(static_cast<std::size_t>(sg_L + 1) * Nr, 0.0);
    sg_outer_ratio.assign(static_cast<std::size_t>(sg_L + 1) * Nr, 0.0);
    for (int L = 0; L <= sg_L; ++L)
        for (int i = 0; i < Nr; ++i)
        {
            if (i > 0)
                sg_inner_ratio[L * Nr + i] = std::pow(r[i - 1] / r[i], L + 1);
            if (i < Nr - 1)
                sg_outer_ratio[L * Nr + i] = std::pow(r[i] / r[i + 1], L);
        }

    sg_phi.assign(sg_multipoles.size(), Eigen::VectorXcd::Zero(Nr));
    const std::size_t size = mode_pairs.size() * static_cast<std::size_t>(Nr - 2);
    sg_profile.assign(size, 0.0);
    sg_profile_now.assign(size, 0.0);
    sg_profile_prev.assign(size, 0.0);
    sg_have_prev = false;

    std::size_t density_terms = 0, pair_terms = 0;
    for (const auto &terms : sg_density_terms)
        density_terms += terms.size();
    for (const auto &terms : sg_pair_terms)
        pair_terms += terms.size();
    std::cout << "Self-gravity: " << sg_multipoles.size() << " multipoles up to L = " << sg_L << ", "
              << density_terms << " density terms, " << pair_terms << " coupling terms, recoil "
              << (sg.recoil ? "on" : "off") << ", " << (sg.iterated ? "iterated midpoint" : "extrapolated")
              << " potential" << (sg.ramp_time > 0.0 ? ", ramp over " + std::to_string(sg.ramp_time) : std::string())
              << std::endl;
}

void CrankNicolsonModes::sg_compute_potential(const Eigen::VectorXcd &state)
{
    const double *w = node_weight.data();
    double energy = 0.0;
    const dcomplex accretion = sg_accretion_dipole(state);
#pragma omp parallel for schedule(dynamic) reduction(+ : energy) if (sg_multipoles.size() >= 4)
    for (std::size_t a = 0; a < sg_multipoles.size(); ++a)
    {
        const int L = sg_multipoles[a].first, M = sg_multipoles[a].second;
        Eigen::VectorXcd n = Eigen::VectorXcd::Zero(Nr);
        for (const auto &term : sg_density_terms[a])
            n += term.gaunt * (state.segment(static_cast<Eigen::Index>(term.j) * Nr, Nr).conjugate().cwiseProduct(
                                  state.segment(static_cast<Eigen::Index>(term.k) * Nr, Nr)));
        // Phi_LM(r_i) = -4 pi mu / (2L + 1) sum_j w_j r_<^L / r_>^(L+1) n_j.
        Eigen::VectorXcd A(Nr), B(Nr);
        const double *inner = sg_inner_ratio.data() + static_cast<std::size_t>(L) * Nr;
        const double *outer = sg_outer_ratio.data() + static_cast<std::size_t>(L) * Nr;
        // A node at r = 0 (epsilon = 0) carries no density (U = 0 there) and is skipped in the sums.
        const auto source = [&](int i) { return r[i] > 0.0 ? dcomplex(w[i] * n[i] / r[i]) : dcomplex(0.0); };
        A[0] = source(0);
        for (int i = 1; i < Nr; ++i)
            A[i] = inner[i] * A[i - 1] + source(i);
        B[Nr - 1] = 0.0;
        for (int i = Nr - 2; i >= 0; --i)
            B[i] = outer[i] * (B[i + 1] + source(i + 1));
        Eigen::VectorXcd phi = (-4.0 * M_PI * mu / (2.0 * L + 1.0)) * (A + B);
        // (mu / 2) int Phi |psi|^2 = (mu / 2) sum_i w_i sum_LM Phi_LM conj(n_LM), M < 0 as conjugates.
        double e = 0.0;
        for (int i = 0; i < Nr; ++i)
            e += w[i] * (phi[i] * std::conj(n[i])).real();
        energy += (M == 0 ? 1.0 : 2.0) * 0.5 * mu * e;
        if (L == 1 && M == 1)
        {
            dcomplex D = 0.0;
            for (int i = 0; i < Nr; ++i)
                if (r[i] > 0.0)
                    D += w[i] * n[i] / (r[i] * r[i]);
            // The frame follows the hole, which the absorbed momentum pushes as well as the cloud's pull.
            D += accretion;
#pragma omp critical(sg_dipole_update)
            sg_dipole = D;
            if (sg.recoil)
            {
                // Free-fall frame of the hole: remove the cloud's uniform field at the origin.
                const double c = 4.0 * M_PI * mu / 3.0;
                for (int i = 0; i < Nr; ++i)
                {
                    double taper = 1.0;
                    if (sg.recoil_taper > 0.0 && r[i] > sg.recoil_taper)
                    {
                        const double x = std::min(1.0, (r[i] - sg.recoil_taper) / sg.recoil_taper);
                        taper = std::cos(0.5 * M_PI * x) * std::cos(0.5 * M_PI * x);
                    }
                    phi[i] += c * r[i] * D * taper;
                }
            }
        }
        sg_phi[a] = phi;
    }
    if (sg_L < 1)
        sg_dipole = 0.0;
    sg_accretion = sg_L >= 1 ? accretion : dcomplex(0.0);
    sg_energy = energy;
}

dcomplex CrankNicolsonModes::sg_accretion_dipole(const Eigen::VectorXcd &state) const
{
    // The s-wave sink, dpsi/dt = -Gamma psi / 2 with Gamma psi = g <v|psi> v in the (0, 0) mode, takes
    // momentum <p>_abs = Re <p psi|Gamma psi> from the cloud (p = -i grad); the hole gains it. Only the
    // l = 1 modes give <v|p|psi> != 0. With f = v / r and d_+- = d_x +- i d_y, integration by parts gives
    //   <v|d_+|psi> = -sqrt(2/3) int r U_1,-1 f' dr,  <v|d_-|psi> = sqrt(2/3) int r U_11 f' dr.
    // (The dipolar sinks' share, l = 1 kernels against l = 0 and 2 modes, is left out.)
    if (!sg.recoil || !sg.accretion_recoil || !sink_active || mode00 < 0 || sink1.strength == 0.0 ||
        sink1.count <= 0 || sink1.v.size() != state.size() || state.size() != static_cast<Eigen::Index>(Nr) * n_modes)
        return 0.0;
    int j_plus = -1, j_minus = -1;
    for (int k = 0; k < n_modes; ++k)
    {
        if (modes[k].l == 1 && modes[k].m == 1)
            j_plus = k;
        if (modes[k].l == 1 && modes[k].m == -1)
            j_minus = k;
    }
    if (j_plus < 0 && j_minus < 0)
        return 0.0;
    const dcomplex coefficient = sink1.strength * sink_dot(sink1, state); // Gamma psi = coefficient v
    const dcomplex *v = sink1.v.data() + static_cast<std::size_t>(mode00) * Nr;
    const auto f = [&](int i) { return r[i] > 0.0 ? v[i] / r[i] : dcomplex(0.0); };
    const int lo = std::max(1, sink1.first - 1), hi = std::min(Nr - 2, sink1.first + sink1.count);
    dcomplex up = 0.0, down = 0.0; // int r U f' dr for U_1,-1 and U_11
    for (int i = lo; i <= hi; ++i)
    {
        const dcomplex fp = (f(i + 1) - f(i - 1)) / (r[i + 1] - r[i - 1]);
        const double weight = node_weight[i] * r[i];
        if (j_minus >= 0)
            up += weight * std::conj(fp) * state[static_cast<Eigen::Index>(j_minus) * Nr + i];
        if (j_plus >= 0)
            down += weight * std::conj(fp) * state[static_cast<Eigen::Index>(j_plus) * Nr + i];
    }
    const double s = std::sqrt(2.0 / 3.0);
    const dcomplex d_plus = -s * up, d_minus = s * down;
    const dcomplex px = -I * 0.5 * (d_plus + d_minus); // <v|p_x|psi>
    const dcomplex py = -0.5 * (d_plus - d_minus);     // <v|p_y|psi>
    const double fx = (coefficient * std::conj(px)).real();
    const double fy = (coefficient * std::conj(py)).real();
    const double mass = M1 > 0.0 ? M1 : 1.0;
    // self_gravity_hole_acceleration(): a = c (-Re D, Im D).
    const double c = 4.0 * M_PI * mu / 3.0 * 2.0 * std::sqrt(3.0 / (8.0 * M_PI));
    return dcomplex(-fx / (mass * c), fy / (mass * c));
}

void CrankNicolsonModes::sg_fill_profile(double scale, std::vector<dcomplex> &out) const
{
    const std::size_t inner = Nr - 2;
    out.assign(mode_pairs.size() * inner, 0.0);
    const double factor = mu * scale;
#pragma omp parallel for schedule(static) if (mode_pairs.size() >= 64)
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
        for (const auto &term : sg_pair_terms[p])
        {
            const Eigen::VectorXcd &phi = sg_phi[term.multipole];
            dcomplex *o = out.data() + p * inner - 1;
            const double g = factor * term.gaunt * term.sign;
            if (term.conjugate)
                for (int i = 1; i < Nr - 1; ++i)
                    o[i] += g * std::conj(phi[i]);
            else
                for (int i = 1; i < Nr - 1; ++i)
                    o[i] += g * phi[i];
        }
}

void CrankNicolsonModes::sg_refresh_now()
{
    sg_compute_potential(psi);
    sg_fill_profile(sg_ramp(t_i), sg_profile_now);
}

void CrankNicolsonModes::sg_prepare_step()
{
    // Extrapolated midpoint potential (3 W^n - W^n-1) / 2; the first step uses W^n.
    if (!sg_have_prev)
    {
        sg_profile = sg_profile_now;
        return;
    }
    sg_profile.resize(sg_profile_now.size());
    for (std::size_t n = 0; n < sg_profile.size(); ++n)
        sg_profile[n] = 1.5 * sg_profile_now[n] - 0.5 * sg_profile_prev[n];
}

void CrankNicolsonModes::sg_finish_step()
{
    sg_profile_prev.swap(sg_profile_now);
    sg_psi_prev = psi_last;
    sg_t_prev = t_i - dt;
    sg_have_prev = true;
    sg_refresh_now();
    sg_profile = sg_profile_now;
}

Eigen::VectorXcd CrankNicolsonModes::self_gravity_apply(const Eigen::VectorXcd &density_state, const Eigen::VectorXcd &x)
{
    if (!sg.enabled || !operator_ready)
        throw std::logic_error("self_gravity_apply needs self-gravity and initialize()");
    std::vector<dcomplex> profile;
    sg_compute_potential(density_state);
    sg_fill_profile(1.0, profile);
    Eigen::VectorXcd y = Eigen::VectorXcd::Zero(x.size());
    const std::size_t inner = Nr - 2;
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
    {
        const int j = mode_pairs[p].row_mode, k = mode_pairs[p].col_mode;
        for (int i = 1; i < Nr - 1; ++i)
            y[static_cast<Eigen::Index>(j) * Nr + i] += profile[p * inner + i - 1] * x[static_cast<Eigen::Index>(k) * Nr + i];
    }
    // Leave the evolution's potential as it was.
    if (psi.size() == static_cast<Eigen::Index>(Nr) * n_modes)
        sg_compute_potential(psi);
    return y;
}

Eigen::VectorXcd CrankNicolsonModes::self_gravity_potential(const Eigen::VectorXcd &state, int L, int M)
{
    if (!sg.enabled || !operator_ready || L < 0 || L > sg_L || M < 0 || M > L || sg_index[L * (sg_L + 1) + M] < 0)
        throw std::invalid_argument("No such self-gravity multipole");
    const bool recoil = sg.recoil;
    sg.recoil = false;
    sg_compute_potential(state);
    sg.recoil = recoil;
    Eigen::VectorXcd phi = sg_phi[sg_index[L * (sg_L + 1) + M]];
    if (psi.size() == static_cast<Eigen::Index>(Nr) * n_modes)
        sg_compute_potential(psi);
    return phi;
}

std::pair<double, double> CrankNicolsonModes::self_gravity_hole_acceleration() const
{
    // Near the origin the L = 1 part of Phi is (4 pi mu / 3) 2 sqrt(3 / 8 pi) (Re D x - Im D y).
    const double c = 4.0 * M_PI * mu / 3.0 * 2.0 * std::sqrt(3.0 / (8.0 * M_PI));
    return {-c * sg_dipole.real(), c * sg_dipole.imag()};
}

SparseComplex CrankNicolsonModes::self_gravity_matrix() const
{
    SparseComplex result(Nr * n_modes, Nr * n_modes);
    if (!sg.enabled || sg_profile.size() != mode_pairs.size() * static_cast<std::size_t>(Nr - 2))
        return result;
    std::vector<Eigen::Triplet<dcomplex>> entries;
    entries.reserve(sg_profile.size());
    const std::size_t inner = Nr - 2;
    for (std::size_t p = 0; p < mode_pairs.size(); ++p)
        if (!sg_pair_terms[p].empty())
            for (int i = 1; i < Nr - 1; ++i)
                entries.emplace_back(mode_pairs[p].row_mode * Nr + i, mode_pairs[p].col_mode * Nr + i,
                                     sg_profile[p * inner + i - 1]);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}

// ---------------------------------------------------------------------------
// Levels of the hole's monopole on the grid
// ---------------------------------------------------------------------------

void CrankNicolsonModes::build_levels()
{
    sg_levels.clear();
    sg_level_columns.clear();
    const int n_inner = Nr - 2;
    for (int l = 0; l <= l_max && l + 1 <= sg.levels_n_max; ++l)
    {
        int j = -1;
        for (int k = 0; k < n_modes; ++k)
            if (modes[k].l == l && modes[k].m == l)
                j = k;
        if (j < 0 || self_pair[j] < 0)
            continue;
        // Symmetric tridiagonal form of H0 + V_BH on the interior nodes (weights folded in).
        Eigen::VectorXd d(n_inner), e(std::max(n_inner - 1, 0)), sw(n_inner);
        const std::size_t inner = Nr - 2;
        for (int i = 1; i < Nr - 1; ++i)
        {
            d[i - 1] = h0_diagonal[static_cast<std::size_t>(j) * Nr + i] +
                       (pair_phase[self_pair[j]] * coupling_eff[self_pair[j] * inner + i - 1]).real();
            sw[i - 1] = std::sqrt(node_weight[i]);
        }
        for (int i = 1; i < Nr - 2; ++i)
            e[i - 1] = uniform_grid ? h0_offdiagonal : -std::sqrt(h0_upper[i] * h0_lower[i + 1]);
        double lo = std::numeric_limits<double>::max(), hi = -lo;
        for (int i = 0; i < n_inner; ++i)
        {
            const double radius = (i > 0 ? std::abs(e[i - 1]) : 0.0) + (i < n_inner - 1 ? std::abs(e[i]) : 0.0);
            lo = std::min(lo, d[i] - radius);
            hi = std::max(hi, d[i] + radius);
        }
        // Number of eigenvalues below x (Sturm count of the LDL^T pivots).
        const auto count_below = [&](double x) {
            int count = 0;
            double q = d[0] - x;
            if (q < 0.0)
                ++count;
            for (int i = 1; i < n_inner; ++i)
            {
                if (q == 0.0)
                    q = 1e-300;
                q = d[i] - x - e[i - 1] * e[i - 1] / q;
                if (q < 0.0)
                    ++count;
            }
            return count;
        };
        for (int n = l + 1; n <= sg.levels_n_max; ++n)
        {
            const int index = n - l - 1;
            double a = lo, b = hi;
            for (int it = 0; it < 200 && b - a > 1e-15 * std::max(1.0, std::abs(a)); ++it)
            {
                const double mid = 0.5 * (a + b);
                if (count_below(mid) > index)
                    b = mid;
                else
                    a = mid;
            }
            const double lambda = 0.5 * (a + b);
            // Inverse iteration with the Thomas algorithm.
            const double shift = lambda - 1e-9 * std::max(std::abs(lambda), 1e-12);
            Eigen::VectorXd y = Eigen::VectorXd::Ones(n_inner), c(n_inner), z(n_inner);
            for (int pass = 0; pass < 3; ++pass)
            {
                double denom = d[0] - shift;
                c[0] = n_inner > 1 ? e[0] / denom : 0.0;
                z[0] = y[0] / denom;
                for (int i = 1; i < n_inner; ++i)
                {
                    denom = d[i] - shift - e[i - 1] * c[i - 1];
                    if (denom == 0.0)
                        denom = 1e-300;
                    c[i] = i < n_inner - 1 ? e[i] / denom : 0.0;
                    z[i] = (y[i] - e[i - 1] * z[i - 1]) / denom;
                }
                for (int i = n_inner - 2; i >= 0; --i)
                    z[i] -= c[i] * z[i + 1];
                y = z / z.norm();
            }
            Eigen::VectorXd f = Eigen::VectorXd::Zero(Nr);
            for (int i = 1; i < Nr - 1; ++i)
                f[i] = y[i - 1] / sw[i - 1];
            double norm2 = 0.0;
            for (int i = 0; i < Nr; ++i)
                norm2 += node_weight[i] * f[i] * f[i];
            f /= std::sqrt(norm2);
            const double peak = f.cwiseAbs().maxCoeff();
            for (int i = 0; i < Nr; ++i)
                if (std::abs(f[i]) > 1e-3 * peak)
                {
                    if (f[i] < 0.0)
                        f = -f;
                    break;
                }
            sg_levels.push_back({n, l, lambda, f});
        }
    }
    // Columns of levels.txt: every level in every evolved m.
    for (std::size_t a = 0; a < sg_levels.size(); ++a)
        for (int k = 0; k < n_modes; ++k)
            if (modes[k].l == sg_levels[a].l)
                sg_level_columns.push_back({static_cast<int>(a), k});
}

const Eigen::VectorXd &CrankNicolsonModes::level_function(int n, int l) const
{
    for (const auto &level : sg_levels)
        if (level.n == n && level.l == l)
            return level.f;
    throw std::invalid_argument("Level not computed (needs self-gravity, l <= l_max and n <= levels_n_max)");
}

double CrankNicolsonModes::level_energy(int n, int l) const
{
    for (const auto &level : sg_levels)
        if (level.n == n && level.l == l)
            return level.energy;
    throw std::invalid_argument("Level not computed (needs self-gravity, l <= l_max and n <= levels_n_max)");
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void CrankNicolsonModes::open_self_gravity_files(std::ios::openmode mode)
{
    self_gravity_file.open(output_folder + "/self_gravity.txt", mode);
    levels_file.open(output_folder + "/levels.txt", mode);
    auger_probe_file.open(output_folder + "/auger_probe.txt", mode);
    if (absorption)
        absorbed_channels_file.open(output_folder + "/absorbed_channels.txt", mode);
    if (sg.recoil)
        momentum_file.open(output_folder + "/momentum.txt", mode);
}

std::pair<double, double> CrankNicolsonModes::self_gravity_cloud_momentum(const Eigen::VectorXcd &state) const
{
    // <psi|d_+-|psi> from d_+- [f Y_lm] (f = U / r) into Y_l+1,m+-1 with (U' - (l+1) U / r) / r and
    // Y_l-1,m+-1 with (U' + l U / r) / r; the coefficients were checked on a 3D grid.
    if (state.size() != static_cast<Eigen::Index>(Nr) * n_modes)
        return {0.0, 0.0};
    const auto find = [&](int l, int m) {
        for (int k = 0; k < n_modes; ++k)
            if (modes[k].l == l && modes[k].m == m)
                return k;
        return -1;
    };
    dcomplex s_plus = 0.0, s_minus = 0.0;
    std::vector<dcomplex> up(Nr, 0.0), down(Nr, 0.0);
    for (int j = 0; j < n_modes; ++j)
    {
        const int l = modes[j].l, m = modes[j].m;
        const dcomplex *u = state.data() + static_cast<std::size_t>(j) * Nr;
        for (int i = 1; i < Nr - 1; ++i)
        {
            const dcomplex du = (u[i + 1] - u[i - 1]) / (r[i + 1] - r[i - 1]);
            const dcomplex v = r[i] > 0.0 ? u[i] / r[i] : dcomplex(0.0);
            up[i] = du - static_cast<double>(l + 1) * v;
            down[i] = du + static_cast<double>(l) * v;
        }
        const auto overlap = [&](int k, const std::vector<dcomplex> &g) {
            dcomplex sum = 0.0;
            const dcomplex *b = state.data() + static_cast<std::size_t>(k) * Nr;
            for (int i = 1; i < Nr - 1; ++i)
                sum += node_weight[i] * std::conj(b[i]) * g[i];
            return sum;
        };
        const double L2 = 2.0 * l + 1.0;
        int k;
        if ((k = find(l + 1, m + 1)) >= 0)
            s_plus += -std::sqrt((l + m + 1.0) * (l + m + 2.0) / (L2 * (2.0 * l + 3.0))) * overlap(k, up);
        if (l > 0 && (k = find(l - 1, m + 1)) >= 0)
            s_plus += std::sqrt((l - m) * (l - m - 1.0) / ((2.0 * l - 1.0) * L2)) * overlap(k, down);
        if ((k = find(l + 1, m - 1)) >= 0)
            s_minus += std::sqrt((l - m + 1.0) * (l - m + 2.0) / (L2 * (2.0 * l + 3.0))) * overlap(k, up);
        if (l > 0 && (k = find(l - 1, m - 1)) >= 0)
            s_minus += -std::sqrt((l + m) * (l + m - 1.0) / ((2.0 * l - 1.0) * L2)) * overlap(k, down);
    }
    // p_x = -i (d_+ + d_-) / 2, p_y = -(d_+ - d_-) / 2.
    return {(-I * 0.5 * (s_plus + s_minus)).real(), (-0.5 * (s_plus - s_minus)).real()};
}

void CrankNicolsonModes::write_levels_index() const
{
    std::ofstream out(output_folder + "/levels_index.txt");
    out << "# columns of levels.txt: occupation N_nlm = |sum_i w_i f_nl(r_i) U_lm(r_i)|^2 (particles); "
           "energy is the level's grid energy without self-gravity\n# n l m energy\n";
    out.precision(12);
    for (const auto &column : sg_level_columns)
    {
        const auto &level = sg_levels[column.level];
        out << level.n << " " << level.l << " " << modes[column.mode].m << " " << level.energy << "\n";
    }
}

void CrankNicolsonModes::write_self_gravity_rows(const Eigen::VectorXcd &midpoint)
{
    (void)midpoint;
    const auto [ax, ay] = self_gravity_hole_acceleration();
    write_quantity_line_modes({sg_energy, ax, ay, static_cast<double>(sg_last_iterations)}, self_gravity_file);

    if (momentum_file.is_open())
    {
        // Diagnostic: with M V the hole's momentum (V from integrating a, trapezoid rule) and the cloud's
        // inertial momentum P_frame + M_c V, the total P_frame + (M_c + M1) V is conserved when the frame
        // follows the hole consistently. Columns: P_frame (x, y), V (x, y), P_total (x, y), the accretion
        // share of M1 V (x, y).
        const double c = 4.0 * M_PI * mu / 3.0 * 2.0 * std::sqrt(3.0 / (8.0 * M_PI));
        const double acc_x = -c * sg_accretion.real(), acc_y = c * sg_accretion.imag();
        if (sg_have_last_acc && t_i > sg_last_acc_time)
        {
            const double h = 0.5 * (t_i - sg_last_acc_time);
            sg_hole_velocity[0] += h * (ax + sg_last_acc[0]);
            sg_hole_velocity[1] += h * (ay + sg_last_acc[1]);
            sg_accretion_impulse[0] += h * (acc_x + sg_last_acc_accretion[0]);
            sg_accretion_impulse[1] += h * (acc_y + sg_last_acc_accretion[1]);
        }
        sg_last_acc[0] = ax;
        sg_last_acc[1] = ay;
        sg_last_acc_accretion[0] = acc_x;
        sg_last_acc_accretion[1] = acc_y;
        sg_last_acc_time = t_i;
        sg_have_last_acc = true;
        const auto [px, py] = self_gravity_cloud_momentum(psi);
        double particles = 0.0;
        for (Eigen::Index k = 0; k < psi.size(); ++k)
            particles += node_weight[k % Nr] * std::norm(psi[k]);
        const double mass = mu * particles + M1;
        write_quantity_line_modes({px, py, sg_hole_velocity[0], sg_hole_velocity[1], px + mass * sg_hole_velocity[0],
                                   py + mass * sg_hole_velocity[1], M1 * sg_accretion_impulse[0],
                                   M1 * sg_accretion_impulse[1]},
                                  momentum_file);
    }

    const auto project = [&](const Eigen::VectorXd &f, int j) {
        dcomplex c = 0.0;
        const dcomplex *u = psi.data() + static_cast<std::size_t>(j) * Nr;
        for (int i = 1; i < Nr - 1; ++i)
            c += node_weight[i] * f[i] * u[i];
        return c;
    };
    std::vector<double> row(sg_level_columns.size());
    for (std::size_t c = 0; c < sg_level_columns.size(); ++c)
        row[c] = std::norm(project(sg_levels[sg_level_columns[c].level].f, sg_level_columns[c].mode));
    write_quantity_line_modes(row, levels_file);

    // auger_probe.txt: c_211, c_322, c_433, psi_00(0), U_33(r_ex), U_53(r_ex), as (re, im) pairs.
    const auto mode_index = [&](int l, int m) {
        for (int k = 0; k < n_modes; ++k)
            if (modes[k].l == l && modes[k].m == m)
                return k;
        return -1;
    };
    std::vector<double> probe;
    probe.reserve(12);
    for (const auto &[n, l] : {std::pair<int, int>{2, 1}, {3, 2}, {4, 3}})
    {
        dcomplex c = 0.0;
        const int j = mode_index(l, l);
        if (j >= 0)
            for (const auto &level : sg_levels)
                if (level.n == n && level.l == l)
                    c = project(level.f, j);
        probe.push_back(c.real());
        probe.push_back(c.imag());
    }
    dcomplex psi0 = 0.0;
    if (mode00 >= 0 || mode_index(0, 0) >= 0)
    {
        // U_00 = a s + b s^2 + c s^3 near the origin (Coulomb cusp), s = r - r_0 the distance from
        // the Dirichlet node; psi(0) = a / sqrt(4 pi).
        const int j = mode_index(0, 0);
        const int first = 1, count = std::min(5, Nr - 2);
        Eigen::MatrixXd A(count, 3);
        Eigen::VectorXcd b(count);
        for (int k = 0; k < count; ++k)
        {
            const double x = r[first + k] - r[0];
            A(k, 0) = x;
            A(k, 1) = x * x;
            A(k, 2) = x * x * x;
            b[k] = psi[static_cast<Eigen::Index>(j) * Nr + first + k];
        }
        const Eigen::MatrixXd pinv = (A.transpose() * A).ldlt().solve(A.transpose());
        psi0 = (pinv.row(0).cast<dcomplex>() * b)(0) / std::sqrt(4.0 * M_PI);
    }
    probe.push_back(psi0.real());
    probe.push_back(psi0.imag());
    for (const auto &[l, m] : {std::pair<int, int>{3, 3}, {5, 3}})
    {
        const int j = mode_index(l, m);
        const dcomplex u = j >= 0 && idx_ex > 0 ? psi[static_cast<Eigen::Index>(j) * Nr + idx_ex] : dcomplex(0.0);
        probe.push_back(u.real());
        probe.push_back(u.imag());
    }
    write_quantity_line_modes(probe, auger_probe_file);

    if (absorbed_channels_file.is_open())
        write_quantity_line_modes({absorbed_swave_1, absorbed_dipole_1[0], absorbed_dipole_1[1]},
                                  absorbed_channels_file);
}
