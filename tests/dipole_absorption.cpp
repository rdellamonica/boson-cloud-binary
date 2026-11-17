// Dipolar (l = 1) horizon absorption: the COM-frame weights against a direct
// angular quadrature, amplitude extraction, the signed coefficients, and the
// mass and spin budgets of the Crank-Nicolson step.
#include "Config.hpp"
#include "CrankNicolsonModes.hpp"
#include "TestUtils.hpp"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <omp.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

using test::require;

namespace {

struct Setup
{
    int Nr = 161;
    double L = 40.0, mu = 0.3, M_c = 0.05, q = 0.5, R = 20.0, smear = 3.0;
    int l_max = 4;
    bool counter = false, dipole = true;
    double chi1 = 0.0, chi2 = 0.0;
};

std::unique_ptr<CrankNicolsonModes> make(const Setup &s)
{
    std::ofstream("fixture.toml") << "# dipole fixture\n";
    auto cn = std::make_unique<CrankNicolsonModes>(s.Nr, s.L, 1.0, 0.0, s.l_max, s.mu, s.M_c, s.q, 0.0, s.R,
                                                   s.counter, true, 1000, 0.0, s.smear, "fixture.toml");
    cn->dipole_absorption = s.dipole;
    cn->chi1 = s.chi1;
    cn->chi2 = s.chi2;
    cn->evolve_orbit = false;
    cn->dt = cn->P / 16;
    cn->reference_energy = -std::pow(s.mu, 3) / 8.0; // as main sets it for the 2p level
    return cn;
}

int mode_index(const CrankNicolsonModes &cn, int l, int m)
{
    for (int j = 0; j < cn.n_modes; ++j)
        if (cn.modes[j].l == l && cn.modes[j].m == m) return j;
    return -1;
}

Eigen::VectorXcd generic_state(const CrankNicolsonModes &cn)
{
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
    for (int j = 0; j < cn.n_modes; ++j) {
        const auto [l, m] = cn.modes[j];
        const double width = 5.0 + l, centre = 10.0 + 1.5 * m;
        const dcomplex weight = std::polar(1.0 / (1.0 + l), 0.7 * m + 0.3 * l);
        for (int i = 1; i < cn.Nr - 1; ++i) {
            const double x = (cn.r[i] - centre) / width;
            psi[j * cn.Nr + i] = weight * std::exp(-x * x) * std::sin(M_PI * i / (cn.Nr - 1))
                                 * std::polar(1.0, 0.05 * cn.r[i]);
        }
    }
    return psi / std::sqrt(psi.squaredNorm() * cn.dr);
}

double cloud_mass(const CrankNicolsonModes &cn) { return cn.mu * cn.dr * cn.psi.squaredNorm(); }

/// Advance to the end of the absorption ramp (one initial period after t = 0).
void run_past_ramp(CrankNicolsonModes &cn, int total_steps)
{
    cn.set_initial_state_full(generic_state(cn));
    cn.initialize();
    cn.run(1, 0.8 * cn.L);
    cn.t.resize(total_steps + 1);
}

dcomplex Y(int l, int m, double cos_theta, double phi)
{
    double value = gsl_sf_legendre_sphPlm(l, std::abs(m), cos_theta);
    if (m < 0 && m % 2 != 0) value = -value;
    return value * std::polar(1.0, m * phi);
}

double bump(double s, double width)
{
    return s < width ? std::exp(-1.0 / (1.0 - s * s / (width * width))) : 0.0;
}

/// w_{LM;1m}(r) = r int dOmega Y_LM^* e^{i mu v.rho} rho K(rho) Y_1m(rho_hat) / I_1 by a
/// direct (theta, phi) quadrature in Cartesian components.
dcomplex reference_weight(double r, int L, int M, int m, double Ra, double phi_a, double speed, double mu, double width)
{
    static std::vector<double> nodes, weights;
    const int n_theta = 600, n_phi = 2400;
    if (nodes.empty()) {
        auto *table = gsl_integration_glfixed_table_alloc(n_theta);
        nodes.resize(n_theta);
        weights.resize(n_theta);
        for (int k = 0; k < n_theta; ++k)
            gsl_integration_glfixed_point(-1.0, 1.0, k, &nodes[k], &weights[k], table);
        gsl_integration_glfixed_table_free(table);
    }
    // I_1 = int rho^4 K d rho.
    double I1 = 0.0;
    {
        const int n = 20000;
        for (int k = 0; k < n; ++k) {
            const double rho = width * (k + 0.5) / n;
            I1 += std::pow(rho, 4) * bump(rho, width) * width / n;
        }
    }
    const double bx = Ra * std::cos(phi_a), by = Ra * std::sin(phi_a);
    const double vx = -speed * std::sin(phi_a), vy = speed * std::cos(phi_a);
    dcomplex sum = 0.0;
    for (int k = 0; k < n_theta; ++k) {
        const double c = nodes[k], s = std::sqrt(1.0 - c * c);
        for (int p = 0; p < n_phi; ++p) {
            const double phi = 2.0 * M_PI * p / n_phi;
            const double rx = r * s * std::cos(phi) - bx, ry = r * s * std::sin(phi) - by, rz = r * c;
            const double kernel = bump(std::sqrt(rx * rx + ry * ry + rz * rz), width);
            if (kernel == 0.0) continue;
            const dcomplex solid = (m > 0 ? -1.0 : 1.0) * std::sqrt(3.0 / (8.0 * M_PI)) * dcomplex(rx, m * ry);
            sum += weights[k] * (2.0 * M_PI / n_phi) * std::conj(Y(L, M, c, phi)) *
                   std::polar(1.0, mu * (vx * rx + vy * ry)) * kernel * solid;
        }
    }
    return r * sum / I1;
}

void weights_match_direct_quadrature(bool counter)
{
    test::ScratchDir dir("weights");
    Setup s;
    s.counter = counter;
    auto cn = make(s);
    run_past_ramp(*cn, 20);
    for (int step = 1; step <= 19; ++step) cn->step(step);
    require(cn->dipole_strength(1, 1) > 0 && cn->dipole_strength(2, -1) > 0, "Dipolar sinks are not active after the ramp");
    const double sense = counter ? -1.0 : 1.0;
    for (int body : {1, 2}) {
        const double Ra = body == 1 ? cn->R1 : cn->R2, width = s.smear * (body == 1 ? cn->M1 : cn->M2);
        const double phi_a = cn->theta + (body == 1 ? 0.0 : M_PI);
        const double speed = sense * cn->Omega * Ra;
        for (int m : {-1, 1}) {
            std::vector<dcomplex> expected, actual;
            for (int i = 1; i < cn->Nr - 1; ++i) {
                const bool inside = std::abs(cn->r[i] - Ra) < width;
                for (int j = 0; j < cn->n_modes; ++j) {
                    if (!inside)
                        require(cn->dipole_weight(body, m, j, i) == 0.0, "Dipolar weight outside the kernel shell");
                    else if (i % 3 == 0) { // every third shell node is enough
                        expected.push_back(reference_weight(cn->r[i], cn->modes[j].l, cn->modes[j].m, m, Ra, phi_a,
                                                            speed, s.mu, width));
                        actual.push_back(cn->dipole_weight(body, m, j, i));
                    }
                }
            }
            // The solver normalises the kernel on the radial grid, the reference in the
            // continuum: they differ by one factor per kernel, close to 1.
            dcomplex overlap = 0.0;
            double norm = 0.0, scale = 0.0, worst = 0.0;
            for (std::size_t k = 0; k < expected.size(); ++k) {
                overlap += std::conj(expected[k]) * actual[k];
                norm += std::norm(expected[k]);
                scale = std::max(scale, std::abs(expected[k]));
            }
            const dcomplex factor = overlap / norm;
            for (std::size_t k = 0; k < expected.size(); ++k)
                worst = std::max(worst, std::abs(actual[k] - factor * expected[k]));
            std::cout << "  body " << body << ", m = " << m << ": normalisation " << factor << ", max shape error "
                      << worst / scale << " of the largest weight (" << expected.size() << " weights)\n";
            require(expected.size() > 50 && worst < 1e-6 * scale, "Dipolar weights differ from the direct angular quadrature");
            require(std::abs(factor - 1.0) < (body == 1 ? 2e-5 : 1e-3), "Dipolar kernel normalisation is off");
        }
    }
}

void weights_prograde() { weights_match_direct_quadrature(false); }
void weights_retrograde() { weights_match_direct_quadrature(true); }

void amplitudes_of_local_harmonics()
{
    // psi = B + sum_m A_m rho Y_1m(rho_hat) about a static hole at x: by Eq. (107),
    // U_1m = A_m r^2 and U_00 = sqrt(4 pi) r (B - R sum_m A_m Y_1m(pi/2, phi_a)).
    for (double x : {7.0, -7.0, 0.0}) {
        test::ScratchDir dir("amplitudes");
        std::ofstream("fixture.toml") << "# dipole fixture\n";
        CrankNicolsonModes cn(161, 40.0, 0.05, 0.0, 4, 0.3, 0.3, 0, 0, 1, false, true, 1000, 0, 3.0, "fixture.toml");
        cn.configure_isolated_black_hole(x);
        cn.dipole_absorption = true;
        const dcomplex A_plus(0.7, -0.2), A_minus(-0.3, 0.5), B(0.4, 0.1);
        const double phi_a = x < 0 ? M_PI : 0.0;
        Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
        const int j00 = mode_index(cn, 0, 0), jp = mode_index(cn, 1, 1), jm = mode_index(cn, 1, -1);
        const dcomplex shift = A_plus * Y(1, 1, 0.0, phi_a) + A_minus * Y(1, -1, 0.0, phi_a);
        for (int i = 1; i < cn.Nr - 1; ++i) {
            psi[j00 * cn.Nr + i] = std::sqrt(4.0 * M_PI) * cn.r[i] * (B - std::abs(x) * shift);
            psi[jp * cn.Nr + i] = A_plus * cn.r[i] * cn.r[i];
            psi[jm * cn.Nr + i] = A_minus * cn.r[i] * cn.r[i];
        }
        cn.set_initial_state_full(psi);
        cn.initialize();
        const Eigen::VectorXcd state = cn.psi / std::sqrt(cn.M_c / cn.mu);
        // Orthogonality to the other local harmonics holds to the radial quadrature error
        // of the kernel; the amplitude of the channel's own harmonic is exact by construction.
        require(std::abs(cn.dipole_amplitude(1, 1, state) - A_plus) < 1e-5, "A_{1,+1} is not recovered");
        require(std::abs(cn.dipole_amplitude(1, -1, state) - A_minus) < 1e-5, "A_{1,-1} is not recovered");
        for (int m : {-1, 1}) {
            Eigen::VectorXcd single = Eigen::VectorXcd::Zero(state.size());
            const int own = m > 0 ? jp : jm;
            for (int i = 1; i < cn.Nr - 1; ++i) {
                single[own * cn.Nr + i] = cn.r[i] * cn.r[i];
                single[j00 * cn.Nr + i] = -std::sqrt(4.0 * M_PI) * cn.r[i] * std::abs(x) * Y(1, m, 0.0, phi_a);
            }
            require(std::abs(cn.dipole_amplitude(1, m, single) - 1.0) < 1e-12, "Kernel normalisation is not exact on the grid");
        }
        // The s-wave sink sees only B: A_h |B|^2 plus the dipolar terms.
        const double expected = 16.0 * M_PI * std::norm(B) +
                                cn.dipole_coefficient(1.0, 0.0, 1) * (std::norm(A_plus) + std::norm(A_minus));
        const double rate = cn.dr * state.dot(cn.apply_sink(state)).real();
        require(std::abs(rate - expected) < 2e-5 * expected, "Horizon rate is not A_h |B|^2 + sum gamma |A_1m|^2");
    }
}

void hydrogenic_2p_rate()
{
    // A 211 state centred on a Schwarzschild hole: psi -> A r Y_11 with |A|^2 = alpha^10 / 24
    // (M = 1), so the particle loss rate is gamma_11 |A|^2 = (alpha^10 / 6) (1 + 16 alpha^2),
    // Detweiler's 2 |Gamma_211| at leading order. The finite kernel averages
    // exp(-rho / 2 r_B) over its width, which lowers the rate slightly.
    test::ScratchDir dir("hydrogenic");
    std::ofstream("fixture.toml") << "# dipole fixture\n";
    const double mu = 0.3, bohr = 1.0 / (mu * mu);
    for (double width : {3.0, 1.5}) {
        CrankNicolsonModes cn(321, 40.0, 0.05, 0.0, 2, mu, mu, 0, 0, 1, false, true, 1000, 0, width, "fixture.toml");
        cn.configure_isolated_black_hole(0.0);
        cn.dipole_absorption = true;
        Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
        const int j = mode_index(cn, 1, 1);
        for (int i = 1; i < cn.Nr - 1; ++i)
            psi[j * cn.Nr + i] = cn.r[i] * cn.r[i] / (2.0 * std::sqrt(6.0) * std::pow(bohr, 2.5)) * std::exp(-cn.r[i] / (2.0 * bohr));
        cn.set_initial_state_full(psi);
        cn.initialize();
        const Eigen::VectorXcd state = cn.psi / std::sqrt(cn.M_c / cn.mu);
        const double rate = cn.dipole_strength(1, 1) * std::norm(cn.dipole_amplitude(1, 1, state));
        const double expected = std::pow(mu, 10) / 6.0 * (1.0 + 16.0 * mu * mu);
        // <exp(-rho / 2 r_B)> over the kernel: about exp(-0.7 width / 2 r_B).
        const double smoothing = std::exp(-0.7 * width / bohr);
        std::cout << "  2p rate / point-like value = " << rate / expected << " for kernel radius " << width << "\n";
        require(rate < expected && std::abs(rate / expected - smoothing) < 0.03, "2p horizon rate is not gamma |A_11|^2");
        require(std::abs(cn.dipole_amplitude(1, -1, state)) < 1e-14 && cn.sink_strength(1) > 0.0 &&
                std::abs(cn.dr * state.dot(cn.apply_sink(state)).real() - rate) < 1e-12 * rate,
                "A 2p state must be absorbed through the m = +1 channel only");
    }
}

void isolated_hole_with_evolving_mass_and_spin()
{
    // No companion (secondary_potential = false): a static, rapidly spinning hole with a
    // co-rotating 2p cloud. The m = +1 channel is superradiant, so the cloud grows at the
    // expense of the hole's mass and spin, and nothing else moves.
    test::ScratchDir dir("isolated_growth");
    std::ofstream("fixture.toml") << "# dipole fixture\n";
    const double mu = 0.3, bohr = 1.0 / (mu * mu), x = 5.0;
    CrankNicolsonModes cn(321, 80.0, 2.0, 0.0, 2, mu, 0.05, 0.5, 0, 15.0, false, true, 1000, 2.0, 3.0, "fixture.toml");
    cn.configure_isolated_black_hole(x);
    require(cn.is_isolated_black_hole() && !cn.evolve_orbit, "Isolated configuration was not applied");
    cn.evolve_orbit = true; // mass and spin follow the horizon fluxes
    cn.orbit_delay_periods = 0.0;
    cn.dipole_absorption = true;
    cn.chi1 = 0.95;
    // 2p state about the hole, to first order in the offset: U_11 = r^2 f(r) and the
    // monopole that removes its value at the hole.
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
    const int j11 = mode_index(cn, 1, 1), j00 = mode_index(cn, 0, 0);
    for (int i = 1; i < cn.Nr - 1; ++i) {
        const double envelope = std::exp(-cn.r[i] / (2.0 * bohr)) * std::sin(M_PI * i / (cn.Nr - 1));
        psi[j11 * cn.Nr + i] = cn.r[i] * cn.r[i] * envelope;
        psi[j00 * cn.Nr + i] = std::sqrt(1.5) * x * cn.r[i] * envelope;
    }
    cn.set_initial_state_full(psi / std::sqrt(psi.squaredNorm() * cn.dr));
    cn.initialize();
    cn.run(1, 60.0);
    cn.t.resize(31);
    const double total0 = cloud_mass(cn) + cn.M1, J0 = cn.chi1;
    for (int step = 1; step <= 30; ++step) {
        cn.step(step);
        require(std::abs(cloud_mass(cn) + cn.M1 - total0) < 1e-10, "Cloud plus hole mass is not conserved at step " + std::to_string(step));
        require(std::abs(cn.chi1 * cn.M1 * cn.M1 - J0 - cn.absorbed_spin(1)) < 1e-13, "Spin does not follow the dipolar flux");
        require(cn.R1 == x && cn.R == x && cn.theta == 0.0 && cn.Omega == 0.0 && cn.M2 == 0.0 && cn.dt == 2.0,
                "The isolated hole moved or its timestep changed");
    }
    std::cout << "  superradiant m = +1 channel: gamma = " << cn.dipole_strength(1, 1) << ", spin given to the cloud "
              << -cn.absorbed_spin(1) << ", chi_1 0.95 -> " << std::setprecision(12) << cn.chi1 << std::setprecision(6) << "\n";
    require(cn.dipole_strength(1, 1) < 0.0 && cn.absorbed_spin(1) < 0.0 && cn.chi1 < 0.95,
            "The superradiant channel did not spin the hole down");
}

void superradiant_rate_independent_of_dt()
{
    // Crank-Nicolson turns a level's growth rate g into g / (1 + (E dt / 2)^2): with
    // the 2p level's E = -mu^3 / 8 and E dt = 4 that is a factor 5. Stepping
    // relative to E_ref = E removes it, so the superradiant rate does not depend on dt.
    test::ScratchDir dir("rate_vs_dt");
    std::ofstream("fixture.toml") << "# dipole fixture\n";
    const double mu = 0.3, bohr = 1.0 / (mu * mu), level = -mu * mu * mu / 8.0, T = 4800.0;
    const auto rate = [&](double dt, double reference) {
        CrankNicolsonModes cn(801, 200.0, dt, 0.0, 2, mu, mu, 0, 0, 1, false, true, 1000, 0, 3.0, "fixture.toml");
        cn.configure_isolated_black_hole(0.0);
        cn.dipole_absorption = true;
        cn.chi1 = 0.95;
        cn.reference_energy = reference;
        Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
        const int j = mode_index(cn, 1, 1);
        for (int i = 1; i < cn.Nr - 1; ++i)
            psi[j * cn.Nr + i] = cn.r[i] * cn.r[i] * std::exp(-cn.r[i] / (2.0 * bohr));
        cn.set_initial_state_full(psi / std::sqrt(psi.squaredNorm() * cn.dr));
        cn.initialize();
        const int steps = static_cast<int>(std::lround(T / dt));
        cn.run(1, 160.0);
        cn.t.resize(steps + 1);
        const double start_mass = cn.absorbed_mass(1), start_time = cn.t[1];
        for (int step = 2; step <= steps; ++step)
            cn.step(step);
        return -(cn.absorbed_mass(1) - start_mass) / (cn.t[steps] - start_time) / cloud_mass(cn);
    };
    const double coarse_dt = 4.0 / std::abs(level);
    const double fine = rate(4.0, level), coarse = rate(coarse_dt, level), plain = rate(coarse_dt, 0.0);
    std::cout << "  superradiant rate: dt = 4 " << fine << ", E dt = 4 " << coarse << " (plain CN " << plain << ")\n";
    require(fine > 0.0 && std::abs(coarse / fine - 1.0) < 0.01, "The superradiant rate depends on dt");
    require(std::abs(plain / fine - 0.2) < 0.02, "Plain CN no longer shows the 1 / (1 + (E dt / 2)^2) factor");
}

void coefficients()
{
    test::ScratchDir dir("coefficients");
    Setup s;
    auto cn = make(s);
    const double mu = s.mu;
    // Schwarzschild: gamma = 4 M^2 (M^2 + 16 M^4 mu^2).
    require(std::abs(cn->dipole_coefficient(1.0, 0.0, 1) - 4.0 * (1.0 + 16.0 * mu * mu)) < 1e-13 &&
            cn->dipole_coefficient(1.0, 0.0, -1) == cn->dipole_coefficient(1.0, 0.0, 1),
            "Schwarzschild dipolar coefficient is wrong");
    require(std::abs(cn->dipole_coefficient(2.0, 0.0, 1) - 16.0 * (4.0 + 256.0 * mu * mu)) < 1e-11,
            "Dipolar coefficient does not scale with the mass");
    // Kerr, written out: chi = 0.6, M = 1 gives r_+ = 1.8, Omega_+ = 1/6.
    const double k = mu - 1.0 / 6.0, expected = 8.0 * M_PI * 1.8 * k / (4.0 * M_PI * mu) * (0.64 + std::pow(3.6 * k, 2));
    require(std::abs(cn->dipole_coefficient(1.0, 0.6, 1) - expected) < 1e-13, "Kerr dipolar coefficient is wrong");
    require(cn->dipole_coefficient(1.0, 0.6, -1) > cn->dipole_coefficient(1.0, 0.0, 1), "Counter-rotating channel must absorb more");
    require(cn->dipole_coefficient(1.0, 0.95, 1) < 0 && cn->dipole_coefficient(1.0, -0.95, -1) < 0 &&
            cn->dipole_coefficient(1.0, 0.95, -1) > 0, "Superradiant channel has the wrong sign");
}

/// Fixed orbit: every bit of norm the sinks remove is in the absorbed-mass counters, and the
/// absorbed spins are sum_m m gamma_1m |A_1m|^2 dt at the midpoint state.
void budgets_on_fixed_orbit(double chi1, bool counter, bool expect_superradiance)
{
    test::ScratchDir dir("fixed_orbit");
    Setup s;
    s.q = 1.0;
    s.chi1 = chi1;
    s.counter = counter;
    auto cn = make(s);
    run_past_ramp(*cn, 48);
    const double cloud0 = cloud_mass(*cn);
    double spin[2] = {0.0, 0.0}, dipolar_mass = 0.0;
    for (int step = 1; step <= 48; ++step) {
        cn->step(step);
        // The CN pair is psi_n and exp(i E_ref dt) psi_n+1.
        const Eigen::VectorXcd midpoint = cn->step_midpoint();
        for (int body : {1, 2})
            for (int m : {-1, 1}) {
                const double dN = cn->dt * cn->dipole_strength(body, m) * std::norm(cn->dipole_amplitude(body, m, midpoint));
                spin[body - 1] += m * dN;
                dipolar_mass += std::abs(cn->mu * dN);
            }
        require(std::abs(cloud_mass(*cn) + cn->absorbed_mass(1) + cn->absorbed_mass(2) - cloud0) < 1e-10,
                "Field plus absorbed mass is not conserved at step " + std::to_string(step));
    }
    for (int body : {1, 2})
        require(std::abs(cn->absorbed_spin(body) - spin[body - 1]) < 1e-12 * std::abs(spin[body - 1]) + 1e-18,
                "Absorbed spin is not sum_m m dN_1m");
    std::cout << "  chi1 = " << chi1 << ": dipolar share of the absorbed mass "
              << dipolar_mass / (cn->absorbed_mass(1) + cn->absorbed_mass(2)) << ", absorbed spins "
              << cn->absorbed_spin(1) << ", " << cn->absorbed_spin(2) << "\n";
    require(dipolar_mass > 1e-4 * cloud0, "Dipolar absorption test is vacuous");
    require(cn->M1 == 1.0 && cn->chi1 == chi1, "A fixed orbit changed the mass or spin");
    if (expect_superradiance)
        require(cn->dipole_strength(1, 1) < 0.0 && cn->dipole_strength(1, -1) > 0.0 && cn->dipole_strength(2, 1) > 0.0,
                "Superradiant channel of the spinning hole is not signed");
}

void budgets_schwarzschild() { budgets_on_fixed_orbit(0.0, false, false); }
void budgets_retrograde() { budgets_on_fixed_orbit(0.0, true, false); }
void budgets_superradiant() { budgets_on_fixed_orbit(0.95, false, true); }

void masses_and_spins_follow_the_fluxes()
{
    test::ScratchDir dir("evolving");
    Setup s;
    s.q = 1.0;
    s.chi1 = 0.3;
    s.chi2 = -0.2;
    auto cn = make(s);
    cn->evolve_orbit = true;
    cn->steps_per_period = 16;
    cn->orbit_ramp_periods = 0.0;
    run_past_ramp(*cn, 40);
    const double total0 = cloud_mass(*cn) + cn->M1 + cn->M2;
    const double J1 = s.chi1, J2 = s.chi2 * cn->M2 * cn->M2;
    for (int step = 1; step <= 40; ++step) {
        cn->step(step);
        require(std::abs(cloud_mass(*cn) + cn->M1 + cn->M2 - total0) < 1e-10,
                "Field plus black-hole mass is not conserved at step " + std::to_string(step));
        require(std::abs(cn->chi1 * cn->M1 * cn->M1 - J1 - cn->absorbed_spin(1)) < 1e-13 &&
                std::abs(cn->chi2 * cn->M2 * cn->M2 - J2 - cn->absorbed_spin(2)) < 1e-13,
                "chi M^2 does not follow the absorbed spin at step " + std::to_string(step));
    }
    require(cn->M1 > 1.0 && cn->M2 > 1.0 && cn->absorbed_spin(1) != 0.0 && cn->absorbed_spin(2) != 0.0,
            "Evolving-orbit dipolar test is vacuous");
}

void pruned_and_stretched_budget()
{
    // Stretched grid (weighted inner product) with pruning: the sinks are restricted to
    // the active modes and the budget still closes.
    test::ScratchDir dir("stretched");
    Setup s;
    s.q = 1.0;
    auto cn = make(s);
    Eigen::VectorXd nodes(cn->Nr);
    for (int i = 0; i < cn->Nr; ++i) {
        const double x = static_cast<double>(i) / (cn->Nr - 1);
        nodes[i] = s.L * (0.6 * x + 0.4 * x * x);
    }
    cn->set_radial_grid(nodes);
    cn->prune_tolerance = 1e-5;
    cn->prune_check_every = 4;
    Eigen::VectorXcd psi = generic_state(*cn);
    for (int j = 0; j < cn->n_modes; ++j)
        if (cn->modes[j].l > 2) psi.segment(j * cn->Nr, cn->Nr).setZero();
    cn->set_initial_state_full(psi);
    cn->initialize();
    cn->run(1, 0.8 * cn->L);
    cn->t.resize(41);
    const auto mass = [&] {
        double sum = 0.0;
        for (int j = 0; j < cn->n_modes; ++j)
            for (int i = 0; i < cn->Nr; ++i) sum += cn->node_weight[i] * std::norm(cn->psi[j * cn->Nr + i]);
        return cn->mu * sum;
    };
    const double cloud0 = mass();
    for (int step = 1; step <= 40; ++step) {
        cn->step(step);
        require(std::abs(mass() + cn->absorbed_mass(1) + cn->absorbed_mass(2) + cn->pruned_mass() - cloud0) < 1e-9 * cloud0,
                "Stretched-grid budget does not close at step " + std::to_string(step));
    }
    require(cn->absorbed_spin(1) != 0.0 && cn->absorbed_mass(1) > 1e-4 * cloud0, "Stretched-grid dipolar test is vacuous");
}

void config_toggle()
{
    test::ScratchDir dir("config");
    std::ofstream("off.toml") << "[simulation]\nabsorption = true\n";
    require(!load_config("off.toml").dipole_absorption, "Dipolar absorption must be off by default");
    require(load_config("off.toml").secondary_potential, "The companion must be on by default");
    std::ofstream("alone.toml") << "[simulation]\nsecondary_potential = false\n";
    require(!load_config("alone.toml").secondary_potential, "secondary_potential = false was not read");
    std::ofstream("on.toml") << "[simulation]\nabsorption = true\ndipole_absorption = true\n[binary]\nspin_1 = 0.5\nspin_2 = -0.25\n";
    const Config on = load_config("on.toml");
    require(on.dipole_absorption && on.spin_1 == 0.5 && on.spin_2 == -0.25, "Dipolar settings were not read");
    for (const char *bad : {"[simulation]\ndipole_absorption = true\n",
                            "[simulation]\nabsorption = true\ndipole_absorption = true\nl_max = 0\n",
                            "[binary]\nspin_1 = 1.5\n"}) {
        std::ofstream("bad.toml") << bad;
        bool threw = false;
        try {
            load_config("bad.toml");
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "An invalid dipolar configuration was accepted");
    }
}

} // namespace

int main()
{
    omp_set_dynamic(0);
    omp_set_num_threads(2);
    Eigen::setNbThreads(2);
    return test::run_all(std::pair{"horizon coefficients", &coefficients},
                         std::pair{"amplitudes of local harmonics", &amplitudes_of_local_harmonics},
                         std::pair{"2p state on a Schwarzschild hole", &hydrogenic_2p_rate},
                         std::pair{"weights vs direct quadrature (prograde)", &weights_prograde},
                         std::pair{"weights vs direct quadrature (retrograde)", &weights_retrograde},
                         std::pair{"mass and spin budgets, fixed orbit", &budgets_schwarzschild},
                         std::pair{"mass and spin budgets, retrograde", &budgets_retrograde},
                         std::pair{"mass and spin budgets, superradiant hole", &budgets_superradiant},
                         std::pair{"masses and spins follow the fluxes", &masses_and_spins_follow_the_fluxes},
                         std::pair{"stretched grid with pruning", &pruned_and_stretched_budget},
                         std::pair{"isolated hole, evolving mass and spin", &isolated_hole_with_evolving_mass_and_spin},
                         std::pair{"superradiant rate independent of dt", &superradiant_rate_independent_of_dt},
                         std::pair{"configuration toggle", &config_toggle});
}
