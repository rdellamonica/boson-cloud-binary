/**
 * @file cap_per_mode.cpp
 * @brief Per-mode CAP ([cap] per_mode = true): outgoing packets of very
 *        different energies are absorbed with no measurable reflection, where
 *        one uniform strength cannot do that; the strengths follow the modes'
 *        energies; the CN norm balance holds with the tuned CAP.
 */
#include "CrankNicolsonModes.hpp"
#include "TestUtils.hpp"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

namespace {
using test::require;

// Isolated hole of mass 1 (potential -mu/r), mu = 1, spacing 0.4, layer over [0.7 L, L], L = 400.
constexpr double L = 400.0, h = 0.4, mu = 1.0, cap_start = 0.7, r0 = 200.0, sigma = 12.0, dt = 0.2;

struct Packet { int l, m; double k; };
// Group velocities 0.4, 1 and 2 (about 50, 120 and 240 radians across the layer).
const Packet packets[] = {{0, 0, 0.4}, {1, 1, 1.0}, {2, 2, 2.0}};

int mode_index(const CrankNicolsonModes &cn, const Packet &p)
{
    for (int j = 0; j < cn.n_modes; ++j)
        if (cn.modes[j].l == p.l && cn.modes[j].m == p.m)
            return j;
    throw std::runtime_error("packet mode not retained");
}

/// The three packets, of equal norm, on [0, length] with spacing h.
std::unique_ptr<CrankNicolsonModes> make_solver(double length = L)
{
    const int Nr = static_cast<int>(std::lround(length / h)) + 1;
    auto cn = std::make_unique<CrankNicolsonModes>(Nr, length, dt, 0, 2, mu, 0.1, 0, 0, 1, false, false, 1000000, 0,
                                                   2, "fixture.toml");
    cn->configure_isolated_black_hole(0);
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(Nr) * cn->n_modes);
    for (const Packet &p : packets)
    {
        const int j = mode_index(*cn, p);
        for (int i = 1; i < Nr - 1; ++i)
        {
            const double x = cn->r[i] - r0;
            if (std::abs(x) < 150)
                psi[j * Nr + i] = std::exp(-x * x / (4 * sigma * sigma)) * std::exp(dcomplex(0, p.k * cn->r[i]));
        }
    }
    psi /= std::sqrt(3.0 * std::sqrt(2 * M_PI) * sigma);
    cn->set_initial_state_full(psi);
    cn->evolve_orbit = false;
    cn->initialize();
    return cn;
}

struct Result
{
    std::vector<double> per_mode, uniform; ///< Reflected norm / packet norm.
    std::vector<double> strength;          ///< Per-mode strength while each packet crosses the layer.
};

/**
 * Reflection measured exactly: the difference, inside the layer start, from a
 * reference run on a domain twice as long with no layer, taken until the
 * reference's own echo off its far wall could arrive. Whatever is not in the
 * reference (bound Rydberg parts of the packets included) cancels.
 */
Result reflections()
{
    auto tuned = make_solver(), uniform = make_solver(), reference = make_solver(2 * L);
    tuned->build_cap(cap_start, 0.0, 6);
    tuned->enable_cap_per_mode(1e-10, 20, 0); // the per_mode defaults
    // One strength that suits the slow packet lets the fast ones reach the wall.
    uniform->build_cap(cap_start, 0.3, 3);
    for (auto *cn : {tuned.get(), uniform.get(), reference.get()})
        cn->run(1, 100);
    const int edge = static_cast<int>(std::floor(cap_start * L / h));
    std::vector<double> initial; // each packet's norm (the state is scaled to the cloud mass)
    for (const Packet &p : packets)
        initial.push_back(reference->psi.segment(mode_index(*reference, p) * reference->Nr, reference->Nr).squaredNorm() * h);
    const int steps = static_cast<int>(1250 / dt);
    for (auto *cn : {tuned.get(), uniform.get(), reference.get()})
        cn->t.resize(steps + 1);
    const std::size_t n = std::size(packets);
    Result result{std::vector<double>(n, 0.0), std::vector<double>(n, 0.0), std::vector<double>(n, 0.0)};
    for (int step = 1; step <= steps; ++step)
    {
        tuned->step(step);
        uniform->step(step);
        reference->step(step);
        for (std::size_t a = 0; a < n; ++a)
        {
            const double v = packets[a].k / mu, t = step * dt;
            const int j = mode_index(*tuned, packets[a]);
            if (step == static_cast<int>((0.85 * L - r0) / v / dt)) // packet centre in the layer
                result.strength[a] = tuned->cap_strength(j);
            if (t > (4 * L - r0 - cap_start * L - 250) / v) // reference's far-wall echo, with spreading
                continue;
            const auto ref = reference->psi.segment(j * reference->Nr, edge);
            const double norm = initial[a];
            result.per_mode[a] = std::max(result.per_mode[a],
                                          (tuned->psi.segment(j * tuned->Nr, edge) - ref).squaredNorm() * h / norm);
            result.uniform[a] = std::max(result.uniform[a],
                                         (uniform->psi.segment(j * uniform->Nr, edge) - ref).squaredNorm() * h / norm);
        }
    }
    return result;
}

std::string list(const std::vector<double> &values)
{
    std::ostringstream text;
    for (double v : values) text << " " << v;
    return text.str();
}

void reflection_against_uniform()
{
    const Result r = reflections();
    std::cout << "\nReflected norm, per-mode (defaults):" << list(r.per_mode) << "\n"
              << "Reflected norm, uniform W = 0.3, n = 3:" << list(r.uniform) << "\n"
              << "Per-mode strengths in the layer:" << list(r.strength) << "\n";
    // Measured: about 2e-13, 2e-16 and 2e-20.
    for (std::size_t a = 0; a < r.per_mode.size(); ++a)
        require(r.per_mode[a] < 1e-11,
                "Per-mode CAP reflects a measurable part at k = " + std::to_string(packets[a].k));
    require(r.uniform.back() > 1e-3, "Uniform control is unexpectedly reflectionless for the fast packet");
    // Strength ~ group velocity: increasing with the packet's energy.
    require(r.strength[0] < r.strength[1] && r.strength[1] < r.strength[2],
            "Per-mode strengths do not follow the modes' energies");
}

void strength_and_balance()
{
    auto cn = make_solver();
    cn->build_cap(cap_start, 0.0, 6);
    cn->enable_cap_per_mode(1e-10, 20, 0);
    require(cn->cap_is_per_mode(), "Per-mode CAP not enabled");
    // The closed form for a resolved wave far out: W = ln(1/eps) / (2 mu I), I = int s / k dr.
    const double E = 0.02;
    const double W = cn->cap_strength_for_energy(0, E);
    require(W > 0 && std::isfinite(W), "Strength is not finite and positive");
    require(cn->cap_strength_for_energy(0, 4 * E) > 1.8 * W && cn->cap_strength_for_energy(0, 4 * E) < 2.2 * W,
            "Strength does not scale with the wave's speed");
    // An evanescent wave gets the k_min floor.
    require(cn->cap_strength_for_energy(0, -1.0) < W, "Evanescent mode not floored");

    cn->run(1, 100);
    cn->t.resize(41);
    for (int step = 1; step <= 40; ++step)
    {
        const Eigen::VectorXcd previous = cn->psi;
        cn->step(step);
        require((cn->H() - (cn->H0 + cn->V() + cn->CAP)).norm() < 1e-12, "H() misses the tuned CAP");
        const Eigen::VectorXcd mid = 0.5 * (previous + cn->psi);
        const double predicted = 2 * cn->dt * mid.dot(cn->CAP * mid).imag();
        const double change = cn->psi.squaredNorm() - previous.squaredNorm();
        require(change <= 1e-12 * previous.squaredNorm(), "Per-mode CAP increased the norm");
        require(std::abs(change - predicted) < 1e-9 * previous.squaredNorm(),
                "Per-mode CAP norm balance violated (CAP and the operator disagree)");
    }
    // The fast packet's mode has a measured energy near its discrete kinetic energy.
    const int j = mode_index(*cn, packets[2]);
    const double expected = (1 - std::cos(packets[2].k * h)) / (mu * h * h);
    test::close(cn->cap_tuned_energy(j), expected, 0.1, "Tuned energy of the fast mode");
}
} // namespace

int main()
{
    try {
        omp_set_dynamic(0);
        omp_set_num_threads(2);
        std::ofstream("fixture.toml") << "# per-mode CAP\n";
        strength_and_balance();
        reflection_against_uniform();
        std::cout << "Per-mode CAP checks passed.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
