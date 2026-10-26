// Configuration parsing, MODESBIN input, output folders, grid and mode list,
// and argument validation of the solver's public interface.
#include "Config.hpp"
#include "CrankNicolsonModes.hpp"
#include "InitialData.hpp"
#include "TestUtils.hpp"
#include <filesystem>
#include <fstream>
#include <memory>

using test::require;
using test::close;

namespace {

Config config_from(const std::string &text)
{
    std::ofstream("config.toml") << text;
    return load_config("config.toml");
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

std::unique_ptr<CrankNicolsonModes> small_solver(int lmax = 4, double epsilon = 0.1)
{
    std::ofstream("fixture.toml") << "# fixture\n";
    return std::make_unique<CrankNicolsonModes>(33, 64.0, 0.5, epsilon, lmax, 0.2, 0.01, 0.1,
                                                0.0, 20.0, false, false, 10, 0.0, 2.0, "fixture.toml");
}

void config_defaults()
{
    test::ScratchDir dir("config_defaults");
    const Config c = config_from("");
    require(c.N == 2048 && c.L == 5000.0, "Grid defaults");
    require(c.dt == 10.0 && c.num_steps == 0 && c.num_periods == 0 && c.save_every == 50, "Run-length defaults");
    require(c.l_max == 5 && c.r_ex == 1000.0 && c.epsilon == 0.001 && c.epsilon_smear == 2.0, "Simulation defaults");
    require(c.ramp_periods == 0.0 && c.ramp_secondary_potential && c.steps_per_period == 50.0 &&
            c.power_average_orbits == 2.0, "Ramp and averaging defaults");
    require(!c.evolve_orbit && !c.absorption, "Physics switches default to off");
    require(c.q == 0.001 && c.R == 50.0 && !c.counter, "Binary defaults");
    require(c.mu == 0.2 && c.M_c == 0.0 && c.a == 0.0 && c.n == 2 && c.l == 1 && c.m == 1, "Atom defaults");
    require(c.threads == 0 && c.preconditioner == "radial", "Parallel and solver defaults");
    require(!c.mask && !c.cap, "Mask and CAP are off unless their tables exist");
}

void config_values()
{
    test::ScratchDir dir("config_values");
    const Config c = config_from(R"(
[grid]
N = 512
L = 300.5
[binary]
q = 0.25
R = 40.0
counter = true
[atom]
mu = 0.3
M_c = 0.05
a = 0.7
n = 3
l = 2
m = 2
[simulation]
dt = 1.5
num_steps = 77
num_periods = 4
save_every = 7
l_max = 6
r_ex = 250.0
ramp_periods = 3.5
ramp_secondary_potential = false
steps_per_period = 64.0
power_average_orbits = 1.5
epsilon = 0.0
epsilon_smear = 5.0
evolve_orbit = true
absorption = true
[parallel]
threads = 3
[solver]
preconditioner = "ilut"
[mask]
L_factor = 0.8
[cap]
L_factor = 0.9
W = 0.5
n = 4
)");
    require(c.N == 512 && c.L == 300.5, "Grid values");
    require(c.q == 0.25 && c.R == 40.0 && c.counter, "Binary values");
    require(c.mu == 0.3 && c.M_c == 0.05 && c.a == 0.7 && c.n == 3 && c.l == 2 && c.m == 2, "Atom values");
    require(c.dt == 1.5 && c.num_steps == 77 && c.num_periods == 4 && c.save_every == 7 && c.l_max == 6 &&
            c.r_ex == 250.0 && c.ramp_periods == 3.5 && !c.ramp_secondary_potential &&
            c.steps_per_period == 64.0 && c.power_average_orbits == 1.5 && c.epsilon == 0.0 &&
            c.epsilon_smear == 5.0 && c.evolve_orbit && c.absorption, "Simulation values");
    require(c.threads == 3 && c.preconditioner == "ilut", "Parallel and solver values");
    require(c.mask && c.mask_L_factor == 0.8 && c.mask_W == 0.1 && c.mask_n == 3,
            "A [mask] table enables the mask, with defaults for missing keys");
    require(c.cap && c.cap_L_factor == 0.9 && c.cap_W == 0.5 && c.cap_n == 4, "CAP values");
}

void config_validation()
{
    test::ScratchDir dir("config_validation");
    require_throws<std::invalid_argument>([] { config_from("[parallel]\nthreads = -1\n"); },
                                          "Negative thread count accepted");
    require_throws<std::invalid_argument>([] { config_from("[solver]\npreconditioner = \"lu\"\n"); },
                                          "Unknown preconditioner accepted");
    require_throws<std::invalid_argument>([] { config_from("[simulation]\nsteps_per_period = 0.0\n"); },
                                          "Non-positive steps_per_period accepted");
    require_throws<std::invalid_argument>([] { config_from("[simulation]\npower_average_orbits = -2.0\n"); },
                                          "Non-positive power_average_orbits accepted");
    require_throws<std::exception>([] { config_from("[grid\nN = 3\n"); }, "Malformed TOML accepted");
    require(config_from("[simulation]\nnum_periods = 2.5\n").num_periods == 2.5, "Fractional num_periods was lost");
    require(config_from("[simulation]\ndt = 2\nnum_periods = 3\n").dt == 2.0, "Integer value for a real key was lost");
    require_throws<std::exception>([] { load_config("does_not_exist.toml"); }, "Missing file accepted");
}

void modesbin_reader()
{
    test::ScratchDir dir("modesbin");
    const int nr = 6;
    auto profile = [&](double scale) {
        Eigen::VectorXcd p(nr);
        for (int i = 0; i < nr; ++i) p[i] = dcomplex(scale * i, -0.5 * scale * i * i);
        return p;
    };
    test::write_modesbin("state.bin", {{{1, 1}, profile(2)}, {{0, 0}, profile(1)}, {{7, 7}, profile(9)}});
    const std::vector<mode> modes{{0, 0}, {1, -1}, {1, 1}};
    const Eigen::VectorXcd psi = load_initial_modes_binary("state.bin", modes);
    require(psi.size() == nr * 3, "Loaded state has the wrong size");
    require((psi.segment(0, nr) - profile(1)).norm() == 0, "Mode (0,0) was not read exactly");
    require(psi.segment(nr, nr).norm() == 0, "A mode missing from the file must be zero");
    require((psi.segment(2 * nr, nr) - profile(2)).norm() == 0,
            "Mode (1,1) was not read exactly, independently of the file order");

    test::write_modesbin("bad_magic.bin", {{{0, 0}, profile(1)}}, "NOTMODES");
    test::write_modesbin("bad_version.bin", {{{0, 0}, profile(1)}}, "MODESBIN", 2);
    require_throws<std::runtime_error>([&] { load_initial_modes_binary("bad_magic.bin", modes); },
                                       "Wrong magic accepted");
    require_throws<std::runtime_error>([&] { load_initial_modes_binary("bad_version.bin", modes); },
                                       "Unsupported version accepted");
    require_throws<std::runtime_error>([&] { load_initial_modes_binary("missing.bin", modes); },
                                       "Missing file accepted");
}

void output_folder()
{
    test::ScratchDir dir("output_folder");
    const std::string folder = create_output_folder();
    require(folder.rfind("Simulation_", 0) == 0, "Output folder is not named Simulation_*");
    require(std::filesystem::is_directory(folder + "/psi") && std::filesystem::is_directory(folder + "/psi_dt"),
            "Output folder lacks psi/ and psi_dt/");
}

void grid_and_modes()
{
    test::ScratchDir dir("grid_and_modes");
    auto cn = small_solver(4, 0.1);
    close(cn->dr, 64.0 / 32, 1e-15, "dr = L / (N - 1)");
    for (int i = 0; i < cn->Nr; ++i)
        close(cn->r[i], 0.1 + i * cn->dr, 1e-14, "r_i = epsilon + i dr");
    require(cn->n_modes == 15 && int(cn->modes.size()) == 15, "l_max = 4 retains sum (l+1) = 15 modes");
    int previous_l = -1, previous_m = -100;
    for (const auto &m : cn->modes) {
        require((m.l - m.m) % 2 == 0 && std::abs(m.m) <= m.l, "Only l - m even modes are retained");
        require(m.l > previous_l || (m.l == previous_l && m.m > previous_m), "Modes are ordered by l, then m");
        previous_l = m.l;
        previous_m = m.m;
    }
    close(cn->P, 2 * M_PI * std::sqrt(std::pow(20.0, 3) / 1.1), 1e-14, "Initial Kepler period");
    close(cn->R1, 0.1 / 1.1 * 20.0, 1e-14, "Primary distance from the COM");
    close(cn->R2, 1.0 / 1.1 * 20.0, 1e-14, "Secondary distance from the COM");
    require(cn->M1 == 1.0 && cn->M2 == 0.1, "Masses in units of M1");
}

void argument_validation()
{
    test::ScratchDir dir("argument_validation");
    std::ofstream("fixture.toml") << "# fixture\n";
    require_throws<std::invalid_argument>([] {
        CrankNicolsonModes(2, 64.0, 0.5, 0.0, 2, 0.2, 0.01, 0.1, 0.0, 20.0, false, false, 10, 0.0, 2.0, "fixture.toml");
    }, "Nr < 3 accepted");
    require_throws<std::invalid_argument>([] {
        CrankNicolsonModes(33, 64.0, 0.0, 0.0, 2, 0.2, 0.01, 0.1, 0.0, 20.0, false, false, 10, 0.0, 2.0, "fixture.toml");
    }, "dt = 0 accepted");
    require_throws<std::invalid_argument>([] {
        CrankNicolsonModes(33, 64.0, 0.5, 0.0, 2, 0.2, 0.01, 0.1, 0.0, 20.0, false, false, 0, 0.0, 2.0, "fixture.toml");
    }, "save_every = 0 accepted");

    auto cn = small_solver(2);
    require_throws<std::invalid_argument>([&] { cn->set_initial_state_full(Eigen::VectorXcd::Zero(5)); },
                                          "Initial state of the wrong size accepted");
    require_throws<std::invalid_argument>([&] { cn->set_power_average_orbits(0.0); },
                                          "Zero averaging window accepted");
    require_throws<std::invalid_argument>([&] { cn->configure_isolated_black_hole(100.0); },
                                          "Isolated black hole outside the domain accepted");
    cn->set_initial_state_full(Eigen::VectorXcd::Zero(cn->Nr * cn->n_modes));
    cn->initialize();
    require_throws<std::logic_error>([&] { cn->configure_isolated_black_hole(1.0); },
                                     "Isolated black hole configured after initialize()");
    require_throws<std::invalid_argument>([&] { cn->run(5, 64.0); }, "Extraction radius on the boundary accepted");
}

void initial_state_scaling()
{
    test::ScratchDir dir("initial_state_scaling");
    auto cn = small_solver(2);
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn->Nr * cn->n_modes);
    for (int i = 1; i < cn->Nr - 1; ++i) psi[i] = std::sin(M_PI * i / (cn->Nr - 1));
    psi /= std::sqrt(psi.squaredNorm() * cn->dr);
    cn->set_initial_state_full(psi);
    // A unit-norm state is scaled so that the cloud mass mu * |psi|^2 dr is M_c.
    close(cn->mu * cn->psi.squaredNorm() * cn->dr, cn->M_c, 1e-13, "Initial state not scaled to M_c");
}
} // namespace

int main()
{
    return test::run_all(std::pair{"config defaults", &config_defaults},
                         std::pair{"config values", &config_values},
                         std::pair{"config validation", &config_validation},
                         std::pair{"MODESBIN reader", &modesbin_reader},
                         std::pair{"output folder", &output_folder},
                         std::pair{"grid and modes", &grid_and_modes},
                         std::pair{"argument validation", &argument_validation},
                         std::pair{"initial state scaling", &initial_state_scaling});
}
