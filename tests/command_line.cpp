// End-to-end runs of the two executables: a new run from config.toml and a
// MODESBIN file, its output folder, resuming and extending it, and the
// command-line errors. Usage: command_line <GravitationalMolecule> <IsolatedBH>
#include "Config.hpp"
#include "TestUtils.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>

namespace fs = std::filesystem;
using test::close;
using test::require;

namespace {

std::string solver_exe, isolated_exe;

const char *small_config = R"(
[grid]
N = 65
L = 64.0
[binary]
q = 0.5
R = 20.0
[atom]
mu = 0.2
M_c = 0.02
[simulation]
dt = 2.0
num_steps = 21
save_every = 10
l_max = 2
r_ex = 40.0
ramp_periods = 0.0
epsilon = 0.0
evolve_orbit = true
absorption = true
epsilon_smear = 4.0
steps_per_period = 32
[parallel]
threads = 1
[mask]
L_factor = 0.85
W = 1
n = 6
)";

/// Run a command in the current directory, with output in log.txt; return its exit code.
int run(const std::string &command)
{
    const int status = std::system((command + " > log.txt 2>&1").c_str());
#ifdef WEXITSTATUS
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#else
    return status;
#endif
}

std::string shell_quote(const std::string &text) { return "\"" + text + "\""; }

std::string contents(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

/// Write config.toml and a MODESBIN state with (0,0) and (1,1) profiles.
void write_inputs(const std::string &config = small_config)
{
    std::ofstream("config.toml") << config;
    Eigen::VectorXcd s(65), p(65);
    for (int i = 0; i < 65; ++i) {
        const double r = i;
        s[i] = r * std::exp(-r / 8.0);
        p[i] = dcomplex(0.0, 0.3) * r * r * std::exp(-r / 10.0) / 10.0;
    }
    s[64] = p[64] = 0;
    const double norm = std::sqrt(s.squaredNorm() + p.squaredNorm());
    test::write_modesbin("initial_modes.bin", {{mode{0, 0}, s / norm}, {mode{1, 1}, p / norm}});
}

std::size_t rows(const fs::path &path) { return test::read_rows(path).size(); }

void new_run_writes_complete_folder()
{
    test::ScratchDir dir("new_run");
    write_inputs();
    require(run(shell_quote(solver_exe) + " config.toml initial_modes.bin") == 0,
            "Solver failed:\n" + contents("log.txt"));
    const fs::path folder = test::simulation_folder(".");
    require(contents(folder / "config.toml") == contents("config.toml"), "config.toml was not copied verbatim");
    require(rows(folder / "r.txt") == 65, "r.txt does not list the radial nodes");
    std::vector<std::vector<int>> modes;
    {
        std::ifstream in(folder / "modes.txt");
        int l, m;
        char comma;
        while (in >> l >> comma >> m) modes.push_back({l, m});
    }
    require(modes.size() == 6, "modes.txt does not list the six l <= 2 modes");
    for (const char *name : {"t.txt", "R_orb.txt", "theta_orb.txt", "Omega_orb.txt", "M_1.txt", "q.txt",
                             "energy.txt", "p_bin.txt", "p_bin_inst.txt", "dV_bin_dt.txt"}) {
        const auto data = test::read_rows(folder / name);
        require(data.size() == 20 && data.front().size() == 1, std::string(name) + " has the wrong shape");
    }
    for (const char *name : {"flux.txt", "energy_flux.txt", "norms.txt", "energy_modes.txt"}) {
        const auto data = test::read_rows(folder / name);
        require(data.size() == 20 && data.front().size() == 6, std::string(name) + " has the wrong shape");
    }
    for (const char *snapshot : {"psi/snapshot_000000.txt", "psi/snapshot_000010.txt", "psi/snapshot_000020.txt",
                                 "psi_dt/snapshot_000000.txt", "checkpoint.bin"})
        require(fs::exists(folder / snapshot), std::string("Missing ") + snapshot);

    // The initial snapshot carries the cloud mass in the (0,0) and (1,1) modes only.
    const auto psi0 = test::read_rows(folder / "psi/snapshot_000000.txt");
    require(psi0.size() == 6 * 65, "Snapshot has the wrong length");
    double mass = 0, other = 0;
    for (std::size_t k = 0; k < psi0.size(); ++k) {
        const double density = psi0[k][0] * psi0[k][0] + psi0[k][1] * psi0[k][1];
        const auto lm = modes[k / 65];
        mass += density;
        if (!((lm[0] == 0 && lm[1] == 0) || (lm[0] == 1 && lm[1] == 1))) other += density;
    }
    close(0.2 * 1.0 * mass, 0.02, 1e-5, "Initial snapshot does not hold the cloud mass");
    require(other == 0, "Modes missing from MODESBIN were not set to zero");

    // Times increase, the orbit decays and the primary accretes.
    const auto t = test::read_rows(folder / "t.txt"), R = test::read_rows(folder / "R_orb.txt");
    for (std::size_t i = 1; i < t.size(); ++i) require(t[i][0] > t[i - 1][0], "Time is not increasing");
    require(R.back()[0] < 20.0, "Separation did not decrease with evolve_orbit = true");
    require(test::read_rows(folder / "M_1.txt").back()[0] > 1.0, "Primary did not accrete");
}

void resume_and_extend_by_folder()
{
    test::ScratchDir dir("resume");
    write_inputs();
    require(run(shell_quote(solver_exe) + " config.toml initial_modes.bin") == 0, "Initial run failed");
    const fs::path folder = test::simulation_folder(".");
    const std::string finished = contents(folder / "flux.txt");

    // Resuming a finished run leaves it unchanged; raising num_steps extends it.
    require(run(shell_quote(solver_exe) + " " + shell_quote(folder.string() + "/")) == 0,
            "Resume failed:\n" + contents("log.txt"));
    require(contents(folder / "flux.txt") == finished, "Resuming a finished run modified it");
    std::string config = contents(folder / "config.toml");
    config.replace(config.find("num_steps = 21"), 14, "num_steps = 31");
    std::ofstream(folder / "config.toml", std::ios::trunc) << config;
    require(run(shell_quote(solver_exe) + " " + shell_quote(folder.string())) == 0, "Extension failed");
    require(rows(folder / "t.txt") == 30 && fs::exists(folder / "psi/snapshot_000030.txt"),
            "Extended run has the wrong length");
    require(contents(folder / "flux.txt").rfind(finished, 0) == 0, "Extension rewrote earlier rows");
}

void run_length_in_periods()
{
    test::ScratchDir dir("periods");
    std::string config = small_config;
    config.replace(config.find("num_steps = 21"), 14, "num_steps = 0\nnum_periods = 0.5");
    config.replace(config.find("evolve_orbit = true"), 19, "evolve_orbit = false");
    write_inputs(config);
    require(run(shell_quote(solver_exe) + " config.toml initial_modes.bin") == 0, "Solver failed");
    const double P = 2 * M_PI * std::sqrt(8000.0 / 1.5);
    require(rows(test::simulation_folder(".") / "t.txt") == static_cast<std::size_t>(std::floor(0.5 * P / 2.0)) - 1,
            "num_periods does not set the run length");
}

void command_line_errors()
{
    test::ScratchDir dir("errors");
    write_inputs();
    require(run(shell_quote(solver_exe)) == 1 && contents("log.txt").find("Usage") != std::string::npos,
            "Missing arguments do not print the usage");
    require(run(shell_quote(solver_exe) + " config.toml missing.bin") != 0, "Missing MODESBIN file was accepted");
    std::ofstream("bad.toml") << "[grid]\nN = 2\n";
    require(run(shell_quote(solver_exe) + " bad.toml initial_modes.bin") != 0, "Invalid grid was accepted");
    require(run(shell_quote(solver_exe) + " not_a_folder") == 1, "A non-folder single argument was accepted");
    for (const auto &entry : fs::directory_iterator("."))
        require(entry.path().filename().string().rfind("Simulation_", 0) != 0 || fs::is_empty(entry.path() / "psi"),
                "A failed run wrote snapshots");
}

void isolated_black_hole_driver()
{
    test::ScratchDir dir("isolated");
    require(run(shell_quote(isolated_exe) + " --help") == 0 && contents("log.txt").find("Usage") != std::string::npos,
            "--help does not print the usage");
    require(run(shell_quote(isolated_exe)) == 1, "Missing arguments were accepted");
    require(run(shell_quote(isolated_exe) + " abc 0 2") == 1, "Non-numeric mu was accepted");
    require(run(shell_quote(isolated_exe) + " 0.5 0 2 --bogus 1") == 1, "Unknown option was accepted");
    require(run(shell_quote(isolated_exe) + " -0.5 0 2") == 1, "Negative mu was accepted");

    const std::string args = " 0.5 2 2 2 --N 129 --L 40 --steps 11 --save-every 5 --threads 1 --output out";
    require(run(shell_quote(isolated_exe) + args) == 0, "IsolatedBH failed:\n" + contents("log.txt"));
    const Config cfg = load_config("out/isolated_config.toml");
    require(cfg.N == 129 && cfg.L == 40 && cfg.mu == 0.5 && cfg.l_max == 2 && cfg.absorption,
            "isolated_config.toml does not record the run");
    const fs::path folder = test::simulation_folder("out");
    const auto norms = test::read_rows(folder / "norms.txt");
    require(norms.size() == 10, "IsolatedBH wrote the wrong number of rows");
    auto total = [](const std::vector<double> &row) { double s = 0; for (double x : row) s += x; return s; };
    require(total(norms.back()) < total(norms.front()), "Absorbing isolated BH did not remove norm");
    require(run(shell_quote(isolated_exe) + args) == 1, "An existing output directory was overwritten");
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: command_line <GravitationalMolecule> <IsolatedBH>\n";
        return 2;
    }
    solver_exe = fs::absolute(argv[1]).string();
    isolated_exe = fs::absolute(argv[2]).string();
    return test::run_all(std::pair{"new run writes a complete folder", &new_run_writes_complete_folder},
                         std::pair{"resume and extend by folder", &resume_and_extend_by_folder},
                         std::pair{"run length in periods", &run_length_in_periods},
                         std::pair{"command-line errors", &command_line_errors},
                         std::pair{"IsolatedBH driver", &isolated_black_hole_driver});
}
