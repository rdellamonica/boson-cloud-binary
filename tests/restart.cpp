// Resuming runs: exact continuation from checkpoint.bin, reconstruction from
// the diagnostics, recovery from interrupted output, and run extension.
#include "CrankNicolsonModes.hpp"
#include "TestUtils.hpp"
#include <omp.h>
#include <filesystem>
#include <fstream>
#include <memory>

namespace fs = std::filesystem;
using test::require;

namespace {

constexpr double r_ex = 40.0;

std::unique_ptr<CrankNicolsonModes> make_solver()
{
    std::ofstream("config.toml") << "# restart fixture\n";
    auto cn = std::make_unique<CrankNicolsonModes>(65, 64.0, 2.0, 0.0, 2, 0.2, 0.05, 0.5, 0.0, 20.0, false, true,
                                                   10, 0.5, 4.0, "config.toml");
    cn->evolve_orbit = true;
    cn->steps_per_period = 32;
    cn->set_power_average_orbits(1.0);
    return cn;
}

Eigen::VectorXcd initial_state(const CrankNicolsonModes &cn)
{
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
    for (int j = 0; j < cn.n_modes; ++j)
        for (int i = 1; i < cn.Nr - 1; ++i)
            psi[j * cn.Nr + i] = std::polar(std::exp(-std::pow((cn.r[i] - 12.0 - 3 * j) / 6.0, 2)), 0.3 * j);
    return psi / std::sqrt(psi.squaredNorm() * cn.dr);
}

/// Start a run of @p n_steps samples in a fresh directory and return its folder.
fs::path new_run(const std::string &name, int n_steps)
{
    const fs::path root = fs::current_path() / name;
    fs::create_directories(root);
    const fs::path previous = fs::current_path();
    fs::current_path(root);
    auto cn = make_solver();
    cn->set_initial_state_full(initial_state(*cn));
    cn->initialize();
    cn->run(n_steps, r_ex);
    fs::current_path(previous);
    return test::simulation_folder(root);
}

void resume_run(const fs::path &folder, int n_steps)
{
    auto cn = make_solver();
    cn->initialize();
    cn->resume(folder.string(), n_steps, r_ex);
}

fs::path copy_run(const fs::path &folder, const std::string &name)
{
    const fs::path target = fs::current_path() / name / folder.filename();
    fs::create_directories(target.parent_path());
    fs::copy(folder, target, fs::copy_options::recursive);
    return target;
}

std::string contents(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

/// Every file of @p expected exists in @p actual with identical bytes.
void require_identical(const fs::path &expected, const fs::path &actual)
{
    int compared = 0;
    for (const auto &entry : fs::recursive_directory_iterator(expected)) {
        if (!entry.is_regular_file()) continue;
        const fs::path relative = fs::relative(entry.path(), expected);
        require(fs::exists(actual / relative), "Missing " + relative.string() + " after resume");
        require(contents(entry.path()) == contents(actual / relative),
                relative.string() + " differs from the uninterrupted run");
        ++compared;
    }
    require(compared > 20, "Too few files compared");
}

/// Largest difference between the numeric rows of two files, relative to the
/// largest magnitude in the first file.
double max_difference(const fs::path &a, const fs::path &b)
{
    const auto x = test::read_rows(a), y = test::read_rows(b);
    require(x.size() == y.size(), "Row count differs in " + a.filename().string());
    double scale = 0, worst = 0;
    for (const auto &row : x)
        for (double value : row) scale = std::max(scale, std::abs(value));
    for (std::size_t i = 0; i < x.size(); ++i) {
        require(x[i].size() == y[i].size(), "Column count differs in " + a.filename().string());
        for (std::size_t k = 0; k < x[i].size(); ++k)
            worst = std::max(worst, std::abs(x[i][k] - y[i][k]) / scale);
    }
    return worst;
}

const char *diagnostics[] = {"t.txt", "R_orb.txt", "theta_orb.txt", "Omega_orb.txt", "M_1.txt", "q.txt",
                             "energy.txt", "p_bin.txt", "norms.txt", "flux.txt", "energy_flux.txt"};

void exact_resume_from_checkpoint()
{
    test::ScratchDir dir("exact_resume");
    const fs::path reference = new_run("reference", 41);
    // Stopping after step 24 leaves snapshot 20 and four rows past it.
    const fs::path interrupted = new_run("interrupted", 25);
    require(test::read_rows(interrupted / "t.txt").size() == 24, "Interrupted run has the wrong row count");
    resume_run(interrupted, 41);
    require_identical(reference, interrupted);
}

void resume_after_crash_mid_write()
{
    test::ScratchDir dir("crash");
    const fs::path reference = new_run("reference", 41);
    const fs::path crashed = new_run("crashed", 25);
    // A crash can leave a partial row and a snapshot written ahead of the rows.
    std::ofstream(crashed / "flux.txt", std::ios::app) << "0.123 0.4";
    fs::copy_file(crashed / "psi" / "snapshot_000020.txt", crashed / "psi" / "snapshot_000030.txt");
    resume_run(crashed, 41);
    require_identical(reference, crashed);
}

void resume_without_checkpoint()
{
    test::ScratchDir dir("reconstruct");
    const fs::path reference = new_run("reference", 41);
    const fs::path interrupted = new_run("interrupted", 25);
    fs::remove(interrupted / "checkpoint.bin");
    resume_run(interrupted, 41);
    for (const char *name : diagnostics) {
        const double difference = max_difference(reference / name, interrupted / name);
        // Snapshots store 6 significant digits; fluxes, built from radial
        // derivatives, amplify that rounding the most.
        require(difference < 2e-4, std::string(name) + " deviates by " + std::to_string(difference)
                                       + " after reconstruction from diagnostics");
    }
}

void skips_incomplete_snapshot()
{
    test::ScratchDir dir("incomplete");
    const fs::path reference = new_run("reference", 41);
    const fs::path interrupted = new_run("interrupted", 31);
    // Truncate the last snapshot: the run must restart from snapshot 20.
    const fs::path last = interrupted / "psi" / "snapshot_000030.txt";
    const std::string text = contents(last);
    std::ofstream(last, std::ios::trunc) << text.substr(0, text.size() / 2);
    resume_run(interrupted, 41);
    for (const char *name : diagnostics)
        require(max_difference(reference / name, interrupted / name) < 2e-4,
                std::string(name) + " deviates after restarting from an earlier snapshot");
}

void extend_finished_run()
{
    test::ScratchDir dir("extend");
    const fs::path longer = new_run("long", 61);
    const fs::path shorter = new_run("short", 41);
    resume_run(shorter, 41);
    require(test::read_rows(shorter / "t.txt").size() == 40, "Resuming a finished run changed it");
    resume_run(shorter, 61);
    require_identical(longer, shorter);
}

void rejects_mismatched_modes()
{
    test::ScratchDir dir("mismatch");
    const fs::path folder = new_run("run", 12);
    std::ofstream(folder / "modes.txt", std::ios::trunc) << "0,0\n1,1\n";
    bool threw = false;
    try {
        resume_run(folder, 20);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "Resume accepted a folder with a different mode list");

    const fs::path empty = new_run("empty", 5);
    for (const auto &entry : fs::directory_iterator(empty / "psi")) fs::remove(entry.path());
    threw = false;
    try {
        resume_run(empty, 20);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "Resume without any snapshot did not fail");
}

} // namespace

int main()
{
    omp_set_dynamic(0);
    omp_set_num_threads(2);
    Eigen::setNbThreads(2);
    return test::run_all(std::pair{"exact resume from checkpoint", &exact_resume_from_checkpoint},
                         std::pair{"resume after a crash mid-write", &resume_after_crash_mid_write},
                         std::pair{"resume without checkpoint", &resume_without_checkpoint},
                         std::pair{"skip incomplete snapshot", &skips_incomplete_snapshot},
                         std::pair{"extend a finished run", &extend_finished_run},
                         std::pair{"reject mismatched folders", &rejects_mismatched_modes});
}
