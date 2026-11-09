// Stretched radial grid: node generator, [grid] configuration, conservation
// laws and eigenstates on unequal spacings, and MODESBIN input on a grid
// different from the one the file was written on.
#include "Config.hpp"
#include "CrankNicolsonModes.hpp"
#include "InitialData.hpp"
#include "RadialGrid.hpp"
#include "TestUtils.hpp"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>

using test::close;
using test::require;

namespace {

RadialGridSpec stretched(double r_core, double stretch, double transition)
{
    RadialGridSpec spec;
    spec.type = "stretched";
    spec.r_core = r_core;
    spec.stretch = stretch;
    spec.transition = transition;
    return spec;
}

template <class Exception, class F>
void require_throws(F &&f, const std::string &message)
{
    try {
        f();
    } catch (const Exception &) {
        return;
    }
    throw std::runtime_error(message);
}

void generator()
{
    const Eigen::VectorXd uniform = make_radial_grid(11, 5.0, 0.1, RadialGridSpec{});
    for (int i = 0; i < 11; ++i)
        close(uniform[i], 0.1 + 0.5 * i, 1e-15, "Uniform nodes");

    const int N = 801;
    const double L = 4000.0, r_core = 300.0, s = 4.0;
    const Eigen::VectorXd r = make_radial_grid(N, L, 0.001, stretched(r_core, s, 75.0));
    require(r.size() == N, "Node count");
    close(r[0], 0.001, 1e-15, "First node");
    close(r[N - 1], 0.001 + L, 1e-12, "Last node");
    double smallest = L, largest = 0, jump = 0;
    for (int i = 1; i < N; ++i) {
        const double h = r[i] - r[i - 1];
        require(h > 0, "Nodes are not increasing");
        smallest = std::min(smallest, h);
        largest = std::max(largest, h);
        if (i > 1)
            jump = std::max(jump, std::abs(h / (r[i - 1] - r[i - 2]) - 1));
    }
    // Spacing in the core and far out differ by the stretch factor.
    close(largest / smallest, s, 2e-3, "Outer / inner spacing");
    require(jump < 0.05, "Spacing does not vary smoothly");
    // Fewer nodes than a uniform grid of the core spacing would need.
    require(L / smallest > 2.5 * (N - 1), "Stretched grid saves no nodes");
    int inside = 0;
    for (int i = 0; i < N; ++i)
        inside += r[i] < r_core;
    require(inside > (N - 1) * r_core / L * 2.5, "Core is not refined");

    // stretch = 1 is the uniform grid.
    const Eigen::VectorXd flat = make_radial_grid(101, 100.0, 0.0, stretched(30.0, 1.0, 5.0));
    for (int i = 0; i < 101; ++i)
        close(flat[i], double(i), 1e-11, "stretch = 1 is not uniform");

    require_throws<std::invalid_argument>([] { make_radial_grid(11, 5.0, 0.0, stretched(6.0, 4.0, 1.0)); },
                                          "r_core beyond L accepted");
    require_throws<std::invalid_argument>([] { make_radial_grid(11, 5.0, 0.0, stretched(2.0, 0.5, 1.0)); },
                                          "stretch below 1 accepted");
}

Config config_from(const std::string &text)
{
    std::ofstream("config.toml") << text;
    return load_config("config.toml");
}

void configuration()
{
    test::ScratchDir dir("grid_config");
    require(config_from("").grid.type == "uniform", "Grid is uniform by default");
    const Config d = config_from("[grid]\nL = 9000.0\ntype = \"stretched\"\n[atom]\nmu = 0.1\nn = 3\n");
    close(d.grid.r_core, 4.0 * 9 / 0.01, 1e-12, "Default r_core = 4 n^2 / mu^2");
    close(d.grid.stretch, 4.0, 0.0, "Default stretch");
    close(d.grid.transition, d.grid.r_core / 4, 1e-12, "Default transition = r_core / 4");
    const Config c = config_from("[grid]\ntype = \"stretched\"\nr_core = 120.0\nstretch = 6.0\ntransition = 10.0\n");
    require(c.grid.r_core == 120.0 && c.grid.stretch == 6.0 && c.grid.transition == 10.0, "Stretched grid values");
    require_throws<std::invalid_argument>([] { config_from("[grid]\ntype = \"log\"\n"); }, "Unknown grid type accepted");
    require_throws<std::invalid_argument>([] { config_from("[grid]\nL = 100.0\ntype = \"stretched\"\nr_core = 200.0\n"); },
                                          "r_core beyond L accepted");
    require_throws<std::invalid_argument>([] { config_from("[grid]\ntype = \"stretched\"\nstretch = 0.5\n"); },
                                          "stretch below 1 accepted");
    require_throws<std::invalid_argument>([] { config_from("[grid]\ntype = \"stretched\"\ntransition = 0.0\n"); },
                                          "Zero transition accepted");
}

struct Setup
{
    int Nr = 97;
    double L = 120.0, dt = 2.0, mu = 0.2, M_c = 0.1, q = 0.5, R = 20.0;
    int l_max = 2;
    bool absorption = false;
    double r_core = 40.0, stretch = 4.0, transition = 8.0, ramp_periods = 0.2;
};

std::unique_ptr<CrankNicolsonModes> make(const Setup &s)
{
    std::ofstream("fixture.toml") << "# radial grid fixture\n";
    auto cn = std::make_unique<CrankNicolsonModes>(s.Nr, s.L, s.dt, 0.0, s.l_max, s.mu, s.M_c, s.q, 0.0, s.R, false,
                                                   s.absorption, 1000, s.ramp_periods, 4.0, "fixture.toml");
    cn->set_radial_grid(make_radial_grid(s.Nr, s.L, 0.0, stretched(s.r_core, s.stretch, s.transition)));
    return cn;
}

/// Weighted norm sum_i w_i |U_i|^2, the integral of |U|^2 dr.
double norm2(const CrankNicolsonModes &cn, const Eigen::VectorXcd &psi)
{
    double sum = 0;
    for (int j = 0; j < cn.n_modes; ++j)
        for (int i = 0; i < cn.Nr; ++i)
            sum += cn.node_weight[i] * std::norm(psi[j * cn.Nr + i]);
    return sum;
}

Eigen::VectorXcd generic_state(const CrankNicolsonModes &cn)
{
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
    for (int j = 0; j < cn.n_modes; ++j) {
        const auto [l, m] = cn.modes[j];
        const double width = 6.0 + 1.5 * l, centre = 18.0 + 2.0 * m;
        const dcomplex weight = std::polar(1.0 / (1.0 + l), 0.7 * m + 0.3 * l);
        for (int i = 1; i < cn.Nr - 1; ++i) {
            const double x = (cn.r[i] - centre) / width;
            psi[j * cn.Nr + i] = weight * std::exp(-x * x) * std::polar(1.0, 0.05 * cn.r[i]);
        }
    }
    return psi / std::sqrt(norm2(cn, psi));
}

void grid_is_installed()
{
    test::ScratchDir dir("installed");
    Setup s;
    auto cn = make(s);
    require(!cn->uniform_radial_grid(), "Stretched grid reported as uniform");
    double smallest = s.L;
    for (int i = 1; i < cn->Nr; ++i)
        smallest = std::min(smallest, cn->r[i] - cn->r[i - 1]);
    close(cn->dr, smallest, 1e-15, "dr is the smallest spacing");
    close(cn->node_weight.sum(), s.L, 1e-12, "Weights do not integrate 1 over the box");
    cn->set_initial_state_full(generic_state(*cn));
    cn->initialize();
    require_throws<std::logic_error>([&] { cn->set_radial_grid(cn->r); }, "Grid changed after initialisation");
}

void unitarity()
{
    test::ScratchDir dir("unitarity");
    Setup s;
    s.l_max = 4;
    auto cn = make(s);
    cn->evolve_orbit = false;
    cn->set_initial_state_full(generic_state(*cn));
    cn->initialize();
    cn->run(1, 40);
    const double norm0 = norm2(*cn, cn->psi);
    cn->t.resize(31);
    for (int step = 1; step <= 30; ++step) {
        cn->step(step);
        close(norm2(*cn, cn->psi), norm0, 1e-9, "Weighted norm changed at step " + std::to_string(step));
    }
}

void hydrogen_eigenstate()
{
    test::ScratchDir dir("hydrogen");
    // n = 2, l = 1 about an isolated hole, on a grid refined inside 20 a0.
    Setup s;
    s.Nr = 241;
    s.L = 400.0;
    s.mu = 0.5;
    s.M_c = 0.5;
    s.dt = 0.5;
    s.r_core = 40.0;
    s.transition = 10.0;
    auto cn = make(s);
    cn->configure_isolated_black_hole(0.0);
    const double a0 = 1.0 / (s.mu * s.mu);
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn->Nr * cn->n_modes);
    int j = 0;
    while (!(cn->modes[j].l == 1 && cn->modes[j].m == 1)) ++j;
    for (int i = 0; i < cn->Nr - 1; ++i)
        psi[j * cn->Nr + i] = cn->r[i] * cn->r[i] * std::exp(-cn->r[i] / (2 * a0));
    psi /= std::sqrt(norm2(*cn, psi));
    cn->set_initial_state_full(psi);
    cn->initialize();
    const Eigen::VectorXcd initial = cn->psi;
    const Eigen::VectorXcd Hpsi = cn->H() * initial;
    dcomplex numerator = 0;
    for (int k = 0; k < initial.size(); ++k)
        numerator += cn->node_weight[k % cn->Nr] * std::conj(initial[k]) * Hpsi[k];
    const double norm = norm2(*cn, initial), energy = numerator.real() / norm, exact = -std::pow(s.mu, 3) / 8.0;
    close(energy, exact, 2e-3 * std::abs(exact), "Hydrogenic energy on a stretched grid");

    cn->run(1, 60);
    cn->t.resize(41);
    for (int step = 1; step <= 40; ++step) cn->step(step);
    dcomplex overlap = 0;
    for (int k = 0; k < initial.size(); ++k)
        overlap += cn->node_weight[k % cn->Nr] * std::conj(initial[k]) * cn->psi[k];
    require(std::abs(overlap / norm) > 1 - 1e-4, "Eigenstate is not stationary on a stretched grid");
}

void accretion_balance()
{
    test::ScratchDir dir("accretion");
    // The sinks remove from the field exactly the mass the holes gain.
    Setup s;
    s.q = 1.0;
    s.M_c = 0.05;
    s.absorption = true;
    s.ramp_periods = 0.0;
    auto cn = make(s);
    cn->evolve_orbit = true;
    cn->steps_per_period = 16;
    cn->dt = cn->P / 16;
    cn->set_initial_state_full(generic_state(*cn));
    cn->initialize();
    cn->run(1, 40);
    const double cloud0 = cn->mu * norm2(*cn, cn->psi), total0 = cloud0 + cn->M1 + cn->M2;
    cn->t.resize(41);
    for (int step = 1; step <= 40; ++step) {
        cn->step(step);
        const double total = cn->mu * norm2(*cn, cn->psi) + cn->M1 + cn->M2;
        close(total, total0, 1e-10,
              "Field plus black-hole mass is not conserved at step " + std::to_string(step));
    }
    require(cloud0 - cn->mu * norm2(*cn, cn->psi) > 1e-3 * cloud0, "Accretion test is vacuous");
}

void write_v2(const std::string &filename, const Eigen::VectorXd &r, const std::vector<std::pair<mode, Eigen::VectorXcd>> &profiles)
{
    test::write_modesbin(filename, profiles, "MODESBIN", 2);
    // Version 2 stores the nodes between the header and the profiles.
    std::ifstream in(filename, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    std::string nodes(reinterpret_cast<const char *>(r.data()), r.size() * sizeof(double));
    bytes.insert(8 + 4 + 8 + 8, nodes);
    std::ofstream(filename, std::ios::binary) << bytes;
}

void loader()
{
    test::ScratchDir dir("loader");
    const auto f = [](double r) { return dcomplex(r * std::exp(-r / 10.0), 0.3 * r * r * std::exp(-r / 8.0)); };
    const int N = 201;
    const double L = 200.0;
    const Eigen::VectorXd uniform = make_radial_grid(N, L, 0.0, RadialGridSpec{});
    const Eigen::VectorXd target = make_radial_grid(151, L, 0.0, stretched(50.0, 3.0, 10.0));
    Eigen::VectorXcd profile(N);
    for (int i = 0; i < N; ++i) profile[i] = f(uniform[i]);
    const std::vector<mode> modes{{0, 0}, {1, 1}};

    // A uniform version-1 file onto a stretched grid is interpolated.
    test::write_modesbin("v1.bin", {{{1, 1}, profile}});
    const Eigen::VectorXcd a = load_initial_modes_binary("v1.bin", modes, target, false);
    double error = 0;
    for (int i = 0; i < target.size(); ++i) {
        require(a[i] == dcomplex(0), "Missing mode is not zero");
        error = std::max(error, std::abs(a[target.size() + i] - f(target[i])));
    }
    require(error < 2e-4 * profile.cwiseAbs().maxCoeff(), "Cubic interpolation error " + std::to_string(error));

    // A version-2 file on the same nodes is copied exactly.
    Eigen::VectorXcd on_target(target.size());
    for (int i = 0; i < target.size(); ++i) on_target[i] = f(target[i]);
    write_v2("v2.bin", target, {{{1, 1}, on_target}});
    const Eigen::VectorXcd b = load_initial_modes_binary("v2.bin", modes, target, false);
    require(b.segment(target.size(), target.size()) == on_target, "Matching nodes were not copied");

    // ...and interpolated back onto the uniform grid from its own nodes.
    const Eigen::VectorXcd c = load_initial_modes_binary("v2.bin", modes, uniform, true);
    error = 0;
    for (int i = 0; i < N; ++i) error = std::max(error, std::abs(c[N + i] - f(uniform[i])));
    require(error < 1e-3 * profile.cwiseAbs().maxCoeff(), "Interpolation from stretched nodes, error " + std::to_string(error));
}
} // namespace

int main()
{
    return test::run_all(std::pair{"generator", &generator},
                         std::pair{"configuration", &configuration},
                         std::pair{"grid is installed", &grid_is_installed},
                         std::pair{"unitarity", &unitarity},
                         std::pair{"hydrogen eigenstate", &hydrogen_eigenstate},
                         std::pair{"accretion balance", &accretion_balance},
                         std::pair{"MODESBIN onto another grid", &loader});
}
