// Self-gravity of the cloud around an isolated hole at the origin: the multipole
// potential, Hermiticity, the gravitational Auger amplitude of Foster & Rodd
// (arXiv:2611.18833, I_G = -0.0914 with recoil, +0.143 without), conservation
// over CN steps and exact resume.
#include "CrankNicolsonModes.hpp"
#include "TestUtils.hpp"
#include <Eigen/SparseLU>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <cmath>

namespace fs = std::filesystem;
using test::close;
using test::require;

namespace {

std::unique_ptr<CrankNicolsonModes> make_solver(int Nr, double L, int l_max, double mu, double M_c, double dt,
                                                bool recoil, bool iterated = false, bool absorption = false,
                                                double epsilon = 0.001, bool accretion_recoil = true)
{
    std::ofstream("config.toml") << "# self-gravity fixture\n";
    auto cn = std::make_unique<CrankNicolsonModes>(Nr, L, dt, epsilon, l_max, mu, M_c, 0.0, 0.0, 0.0, false,
                                                   absorption, 10, 0.0, 1.5, "config.toml");
    cn->configure_isolated_black_hole(0.0);
    CrankNicolsonModes::SelfGravityOptions options;
    options.enabled = true;
    options.recoil = recoil;
    options.accretion_recoil = accretion_recoil;
    options.iterated = iterated;
    options.tolerance = 1e-12;
    options.levels_n_max = 4;
    cn->enable_self_gravity(options);
    return cn;
}

int mode_index(const CrankNicolsonModes &cn, int l, int m)
{
    for (int j = 0; j < cn.n_modes; ++j)
        if (cn.modes[j].l == l && cn.modes[j].m == m)
            return j;
    throw std::runtime_error("mode not evolved");
}

double weighted_dot_real(const CrankNicolsonModes &cn, const Eigen::VectorXcd &a, const Eigen::VectorXcd &b,
                         dcomplex *full = nullptr)
{
    dcomplex sum = 0.0;
    for (int j = 0; j < cn.n_modes; ++j)
        for (int i = 0; i < cn.Nr; ++i)
            sum += cn.node_weight[i] * std::conj(a[j * cn.Nr + i]) * b[j * cn.Nr + i];
    if (full)
        *full = sum;
    return sum.real();
}

/// Unit-norm smooth state in every mode, for runs.
Eigen::VectorXcd mixed_state(const CrankNicolsonModes &cn, unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal;
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
    for (int j = 0; j < cn.n_modes; ++j)
    {
        const dcomplex a(normal(gen), normal(gen));
        const int l = cn.modes[j].l;
        for (int i = 1; i < cn.Nr - 1; ++i)
        {
            const double x = cn.r[i] * cn.mu * cn.mu;
            psi[j * cn.Nr + i] = a * std::pow(x, l + 1) * std::exp(-x / 2.0);
        }
    }
    double n2 = 0.0;
    for (int k = 0; k < psi.size(); ++k)
        n2 += cn.node_weight[k % cn.Nr] * std::norm(psi[k]);
    return psi / std::sqrt(n2);
}

void check_monopole_and_hermiticity()
{
    test::ScratchDir dir("sg_monopole");
    const double mu = 0.4, M_c = 0.05;
    auto cn = make_solver(801, 120.0, 3, mu, M_c, 1.0, true);
    cn->initialize();
    const int Nr = cn->Nr;
    // A 211 cloud with N = M_c / mu particles.
    const double N = M_c / mu;
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(Nr * cn->n_modes);
    const int j11 = mode_index(*cn, 1, 1);
    psi.segment(j11 * Nr, Nr) = std::sqrt(N) * cn->level_function(2, 1).cast<dcomplex>();
    const Eigen::VectorXcd phi00 = cn->self_gravity_potential(psi, 0, 0);
    // Phi_00 Y_00 = -mu [N(<r)/r + int_r |U|^2 / s] with the same node weights (r_i counted once).
    double worst = 0.0;
    for (int i = 5; i < Nr - 5; ++i)
    {
        double inside = 0.0, outside = 0.0;
        for (int k = 0; k < Nr; ++k)
        {
            const double u2 = cn->node_weight[k] * std::norm(psi[j11 * Nr + k]);
            if (k <= i)
                inside += u2 / cn->r[i];
            else
                outside += u2 / cn->r[k];
        }
        const double expected = -mu * (inside + outside);
        worst = std::max(worst, std::abs(phi00[i].real() / std::sqrt(4.0 * M_PI) - expected) / std::abs(expected));
    }
    require(worst < 1e-12, "Phi_00 differs from the enclosed-mass potential");
    close(phi00[Nr - 2].real() / std::sqrt(4.0 * M_PI) * cn->r[Nr - 2], -mu * N, 1e-6,
          "r Phi tends to -mu N outside the cloud");

    // First-order self-energy of 211: <mu Phi> per particle = -(F0 + F2/25) mu^3 M M_c.
    const Eigen::VectorXcd Wpsi = cn->self_gravity_apply(psi, psi);
    close(weighted_dot_real(*cn, psi, Wpsi) / N / (mu * mu * mu * M_c), -(93.0 / 512 + 45.0 / 512 / 25), 2e-3,
          "211 self-energy shift");

    // W[rho] is Hermitian in the weighted product, for a density with every multipole.
    const Eigen::VectorXcd rho = mixed_state(*cn, 1), x = mixed_state(*cn, 2), y = mixed_state(*cn, 3);
    dcomplex xWy, Wxy;
    weighted_dot_real(*cn, x, cn->self_gravity_apply(rho, y), &xWy);
    weighted_dot_real(*cn, cn->self_gravity_apply(rho, x), y, &Wxy);
    require(std::abs(xWy - Wxy) < 1e-12 * std::abs(xWy), "Self-gravity coupling is not Hermitian");
    std::cout << "monopole, 211 self-energy and Hermiticity: ok" << std::endl;
}

/// |I_G| of 211 + 211 -> 322 + X from the forced (0,0) partner at the origin.
double auger_coefficient(bool recoil)
{
    test::ScratchDir dir("sg_auger");
    const double mu = 0.4, a0 = 1.0 / (mu * mu);
    auto cn = make_solver(3201, 40.0 * a0, 4, mu, 0.01, 1.0, recoil);
    cn->initialize();
    const int Nr = cn->Nr;
    const int j11 = mode_index(*cn, 1, 1), j22 = mode_index(*cn, 2, 2), j00 = mode_index(*cn, 0, 0);
    const dcomplex c1 = 1.0, c2 = 1e-3;
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(Nr * cn->n_modes);
    psi.segment(j11 * Nr, Nr) = c1 * cn->level_function(2, 1).cast<dcomplex>();
    psi.segment(j22 * Nr, Nr) = c2 * cn->level_function(3, 2).cast<dcomplex>();
    // The (0,0) row of W[psi] psi is exactly the c1^2 conj(c2) source of the partner.
    const Eigen::VectorXcd source = cn->self_gravity_apply(psi, psi).segment(j00 * Nr, Nr);
    const double E_X = 2.0 * cn->level_energy(2, 1) - cn->level_energy(3, 2);
    // (E_X - H_00) kappa = S on the interior nodes, H_00 = H0 - mu M / r on (0,0).
    const SparseComplex H = cn->H0 + cn->V();
    SparseComplex A(Nr - 2, Nr - 2);
    std::vector<Eigen::Triplet<dcomplex>> entries;
    for (int i = 1; i < Nr - 1; ++i)
        for (int k = std::max(1, i - 1); k <= std::min(Nr - 2, i + 1); ++k)
        {
            const dcomplex h = H.coeff(j00 * Nr + i, j00 * Nr + k);
            entries.emplace_back(i - 1, k - 1, (i == k ? E_X : 0.0) - h);
        }
    A.setFromTriplets(entries.begin(), entries.end());
    Eigen::SparseMatrix<dcomplex> Ac(A);
    Eigen::SparseLU<Eigen::SparseMatrix<dcomplex>> lu(Ac);
    const Eigen::VectorXcd kappa = lu.solve(source.segment(1, Nr - 2));
    // psi_X(0) from U = a r + b r^2 + c r^3 on the first nodes.
    Eigen::MatrixXd F(5, 3);
    Eigen::VectorXcd b(5);
    for (int k = 0; k < 5; ++k)
    {
        const double x = cn->r[k + 1] - cn->r[0]; // distance from the Dirichlet node
        F(k, 0) = x;
        F(k, 1) = x * x;
        F(k, 2) = x * x * x;
        b[k] = kappa[k];
    }
    const Eigen::MatrixXd pinv = (F.transpose() * F).ldlt().solve(F.transpose());
    const dcomplex psi0 = (pinv.row(0).cast<dcomplex>() * b)(0) / std::sqrt(4.0 * M_PI);
    const dcomplex K = psi0 / (c1 * c1 * std::conj(c2));
    // A_H |psi_X(0)|^2 = Gamma_H N1^2 N2  =>  |K|^2 = (alpha mu)^3 q^2 I_G^2 / pi, alpha = q = mu.
    return std::sqrt(M_PI * std::norm(K) / (std::pow(mu * mu, 3) * mu * mu));
}

void check_auger()
{
    const double with_recoil = auger_coefficient(true), without = auger_coefficient(false);
    std::cout << "I_G = " << with_recoil << " (paper 0.0914), without recoil " << without << " (paper 0.143)"
              << std::endl;
    close(with_recoil, 0.0914, 0.0015, "Auger coefficient with recoil");
    close(without, 0.1427, 0.0015, "Auger coefficient without recoil");
}

/// Norm and energy over a short evolution; returns the final state.
Eigen::VectorXcd check_evolution(bool iterated)
{
    test::ScratchDir dir(iterated ? "sg_iterated" : "sg_extrapolated");
    const double mu = 0.4, M_c = 0.3;
    auto cn = make_solver(401, 80.0, 3, mu, M_c, 2.0, false, iterated);
    cn->reference_energy = 0.0;
    cn->set_initial_state_full(mixed_state(*cn, 7));
    cn->initialize();
    const double N0 = M_c / mu;
    cn->run(41, 79.9);
    double N = 0.0;
    for (int k = 0; k < cn->psi.size(); ++k)
        N += cn->node_weight[k % cn->Nr] * std::norm(cn->psi[k]);
    close(N / N0, 1.0, 1e-9, "Self-gravity evolution conserves the norm");
    // E = <H0 + V> + E_self from the written diagnostics: energy.txt holds <H0 + V + W> inside r_ex, so r_ex
    // is put at the box edge. Recoil is off: the hole's free-fall frame is not inertial and E is not conserved.
    const fs::path folder = test::simulation_folder(dir.path());
    std::ifstream energy_in(folder / "energy.txt"), sg_in(folder / "self_gravity.txt");
    std::vector<double> E;
    double e, es, ax, ay, it;
    while (energy_in >> e && sg_in >> es >> ax >> ay >> it)
        E.push_back(e - es);
    require(E.size() == 40, "One energy and self-gravity row per step");
    double drift = 0.0;
    for (double value : E)
        drift = std::max(drift, std::abs(value - E.front()) / std::abs(E.front()));
    std::cout << (iterated ? "iterated" : "extrapolated") << ": energy drift " << drift << ", iterations " << it
              << std::endl;
    require(drift < (iterated ? 1e-9 : 1e-4), "Energy drift with self-gravity too large");
    return cn->psi;
}

/// A grid that starts at r = 0 (epsilon = 0): the origin node must not poison the potential.
void check_origin_node()
{
    test::ScratchDir dir("sg_origin");
    const double mu = 0.4, M_c = 0.3;
    auto cn = make_solver(401, 80.0, 3, mu, M_c, 2.0, true, true, false, 0.0);
    require(cn->r[0] == 0.0, "fixture starts at r = 0");
    cn->reference_energy = -0.5 * mu * mu * mu / 4.0;
    cn->set_initial_state_full(mixed_state(*cn, 5));
    cn->initialize();
    cn->run(6, 70.0);
    require(cn->psi.allFinite(), "Self-gravity with a node at r = 0 gives a finite state");
    require(std::isfinite(cn->reference_energy) && std::isfinite(cn->self_gravity_energy()),
            "Self-gravity with a node at r = 0 gives a finite energy");
    std::cout << "origin node: ok" << std::endl;
}

void check_accretion_recoil()
{
    // With recoil and absorption the hole also takes up the momentum F = Re <p psi|Gamma psi> that the
    // s-wave sink absorbs. Its acceleration with the accretion push minus without must be F / M1, with F
    // evaluated here with the derivative on the l = 1 modes (the solver differentiates the kernel).
    test::ScratchDir dir("sg_accretion");
    const double mu = 0.4, M_c = 0.2;
    const auto accel = [&](bool on, double &fx, double &fy) {
        auto cn = make_solver(1601, 80.0, 2, mu, M_c, 1.0, true, false, true, 0.001, on);
        cn->reference_energy = -0.5 * mu * mu * mu / 4.0;
        const Eigen::VectorXcd psi = mixed_state(*cn, 21);
        cn->set_initial_state_full(psi);
        cn->initialize();
        cn->self_gravity_apply(cn->psi, cn->psi); // leaves the potential of psi in place
        const Eigen::VectorXcd G = cn->apply_sink(cn->psi); // Gamma psi, in the (0, 0) mode
        const int j0 = mode_index(*cn, 0, 0), jp = mode_index(*cn, 1, 1), jm = mode_index(*cn, 1, -1);
        dcomplex d_plus = 0.0, d_minus = 0.0;                // <Gamma psi|d_+- psi>
        const double s = std::sqrt(2.0 / 3.0);
        for (int i = 1; i < cn->Nr - 1; ++i)
        {
            const double h = cn->r[i + 1] - cn->r[i - 1];
            const auto ru = [&](int j, int k) { return cn->r[k] * cn->psi[j * cn->Nr + k]; };
            const dcomplex f = std::conj(G[j0 * cn->Nr + i]) / cn->r[i];
            d_plus += cn->node_weight[i] * s * f * (ru(jm, i + 1) - ru(jm, i - 1)) / h;
            d_minus -= cn->node_weight[i] * s * f * (ru(jp, i + 1) - ru(jp, i - 1)) / h;
        }
        fx = (-I * 0.5 * (d_plus + d_minus)).real() / cn->M1;
        fy = (-0.5 * (d_plus - d_minus)).real() / cn->M1;
        return cn->self_gravity_hole_acceleration();
    };
    double fx, fy;
    const auto on = accel(true, fx, fy), off = accel(false, fx, fy);
    const double dx = on.first - off.first, dy = on.second - off.second, f = std::hypot(fx, fy);
    require(f > 1e-3 * std::hypot(off.first, off.second), "the fixture's absorbed momentum is not negligible");
    require(std::hypot(dx - fx, dy - fy) < 2e-2 * f, "accretion push on the hole = absorbed momentum / M1");
    std::cout << "accretion recoil: ok (push " << dx << ", " << dy << " vs F/M " << fx << ", " << fy
              << ", gravity " << off.first << ", " << off.second << ")" << std::endl;
}

void check_resume()
{
    const double mu = 0.4, M_c = 0.2;
    const auto make = [&] {
        auto cn = make_solver(401, 80.0, 3, mu, M_c, 2.0, true, false, true);
        cn->dipole_absorption = true;
        cn->chi1 = 0.5;
        cn->reference_energy = -0.5 * mu * mu * mu / 4.0;
        return cn;
    };
    const auto run_in = [&](const std::string &name, int steps) {
        test::ScratchDir dir(name);
        auto cn = make();
        cn->set_initial_state_full(mixed_state(*cn, 11));
        cn->initialize();
        cn->run(steps, 70.0);
        return test::simulation_folder(dir.path());
    };
    const auto read = [](const fs::path &folder, const char *name) {
        std::ifstream in(folder / name);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    // Uninterrupted 31 samples, against 21 samples (checkpoint at step 20) resumed to 31.
    const fs::path full = run_in("sg_resume_full", 31), part = run_in("sg_resume_part", 21);
    {
        test::ScratchDir dir("sg_resume_continue");
        auto cn = make();
        cn->initialize();
        cn->resume(part.string(), 31, 70.0);
    }
    for (const char *name : {"auger_probe.txt", "levels.txt", "self_gravity.txt", "absorbed_channels.txt", "norms.txt"})
        require(read(part, name) == read(full, name), std::string("Resumed ") + name + " differs from the uninterrupted run");
    require(!read(full, "absorbed_channels.txt").empty(), "absorbed_channels.txt is written");
    std::cout << "resume: ok (bitwise)" << std::endl;
}

} // namespace

int main()
{
    try
    {
        check_monopole_and_hermiticity();
        check_auger();
        check_evolution(false);
        check_evolution(true);
        check_origin_node();
        check_accretion_recoil();
        check_resume();
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAILED: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "self_gravity: all checks passed" << std::endl;
    return 0;
}
