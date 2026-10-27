// Adaptive mode pruning: selection of the active set, accuracy against the
// full evolution, switching modes on and off, and exact restarts.
#include "CrankNicolsonModes.hpp"
#include "TestUtils.hpp"
#include <omp.h>
#include <filesystem>
#include <fstream>
#include <memory>

namespace fs = std::filesystem;
using test::close;
using test::require;

namespace {

constexpr double r_ex = 40.0;

std::unique_ptr<CrankNicolsonModes> make(double tolerance, bool absorption = true, int save_every = 1000)
{
    std::ofstream("config.toml") << "# pruning fixture\n";
    auto cn = std::make_unique<CrankNicolsonModes>(97, 96.0, 4.0, 0.0, 8, 0.2, 0.01, 0.3, 0.0, 16.0, false,
                                                   absorption, save_every, 0.5, 4.0, "config.toml");
    cn->evolve_orbit = false;
    cn->prune_tolerance = tolerance;
    cn->prune_check_every = 5;
    return cn;
}

int mode_index(const CrankNicolsonModes &cn, int l, int m)
{
    for (int j = 0; j < cn.n_modes; ++j)
        if (cn.modes[j].l == l && cn.modes[j].m == m) return j;
    throw std::runtime_error("missing mode");
}

/// A 2p (m = 1) cloud about the origin, plus a faint (l, m) = (2, 0) admixture.
Eigen::VectorXcd initial_state(const CrankNicolsonModes &cn, double admixture = 0.0)
{
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
    const int p = mode_index(cn, 1, 1), d = mode_index(cn, 2, 0);
    for (int i = 1; i < cn.Nr - 1; ++i) {
        const double r = cn.r[i];
        psi[p * cn.Nr + i] = r * r * std::exp(-r * 0.02);
        psi[d * cn.Nr + i] = admixture * r * r * r * std::exp(-r * 0.04);
    }
    psi[cn.Nr * p + cn.Nr - 1] = 0;
    return psi / std::sqrt(psi.squaredNorm() * cn.dr);
}

double mode_norm(const CrankNicolsonModes &cn, int j)
{
    return cn.psi.segment(j * cn.Nr, cn.Nr).squaredNorm();
}

void require_inactive_modes_zero(const CrankNicolsonModes &cn)
{
    for (int j = 0; j < cn.n_modes; ++j)
        if (!cn.mode_is_active(j))
            require(mode_norm(cn, j) == 0.0, "An inactive mode has a nonzero amplitude");
}

void disabled_by_default()
{
    test::ScratchDir dir("disabled");
    auto cn = make(0.0);
    cn->set_initial_state_full(initial_state(*cn));
    cn->initialize();
    require(cn->active_mode_count() == cn->n_modes, "Without a tolerance every mode must be evolved");
    require(cn->pruned_mass() == 0.0, "Nothing is pruned without a tolerance");
}

void initial_selection()
{
    test::ScratchDir dir("selection");
    auto cn = make(1e-8);
    cn->set_initial_state_full(initial_state(*cn, 1e-6));
    cn->initialize();
    const int active = cn->active_mode_count();
    require(active > 1 && active < cn->n_modes, "Initial active set is trivial (" + std::to_string(active) + ")");
    require(cn->mode_is_active(mode_index(*cn, 1, 1)), "The populated mode is not active");
    // The (2, 0) admixture carries a norm share of ~1e-12 < tolerance: it is
    // dropped unless the potential drives it.
    require_inactive_modes_zero(*cn);
    // The primary's dipole couples (1, 1) to l = 0 and l = 2 modes at once.
    require(cn->mode_is_active(mode_index(*cn, 0, 0)) && cn->mode_is_active(mode_index(*cn, 2, 2)),
            "Modes driven by the populated one were not switched on");
    require(!cn->mode_is_active(mode_index(*cn, 8, -8)), "A remote mode was switched on");
}

/// Final state of a run of @p steps, optionally interrupted and resumed.
Eigen::VectorXcd evolve(double tolerance, int steps, bool absorption, int *final_active = nullptr,
                        double *pruned = nullptr)
{
    auto cn = make(tolerance, absorption);
    cn->set_initial_state_full(initial_state(*cn));
    cn->initialize();
    cn->run(steps + 1, r_ex);
    if (final_active) *final_active = cn->active_mode_count();
    if (pruned) *pruned = cn->pruned_mass();
    require_inactive_modes_zero(*cn);
    return cn->psi;
}

void accuracy_against_full_evolution()
{
    test::ScratchDir dir("accuracy");
    const Eigen::VectorXcd full = evolve(0.0, 120, true);
    double previous_error = 1.0;
    for (double tolerance : {1e-4, 1e-6, 1e-8, 1e-10}) {
        int active = 0;
        const Eigen::VectorXcd pruned = evolve(tolerance, 120, true, &active);
        const double error = (pruned - full).norm() / full.norm();
        std::cout << "tolerance " << tolerance << ": " << active << " active modes, relative error " << error
                  << std::endl;
        // The truncation error is of order sqrt(tolerance) in amplitude.
        require(error < 30 * std::sqrt(tolerance), "Pruned evolution departs from the full one");
        require(error <= previous_error * 1.0001, "Error does not decrease with the tolerance");
        previous_error = error;
    }
}

void conservation_with_pruning()
{
    test::ScratchDir dir("conservation");
    // Without sinks the truncated Hamiltonian is Hermitian: the cloud mass plus
    // the mass dropped by switching modes off is conserved.
    auto cn = make(1e-8, false);
    cn->set_initial_state_full(initial_state(*cn));
    const double mass0 = cn->mu * cn->dr * cn->psi.squaredNorm();
    cn->initialize();
    cn->run(1, r_ex);
    cn->t.resize(81);
    for (int step = 1; step <= 80; ++step) {
        cn->step(step);
        close(cn->mu * cn->dr * cn->psi.squaredNorm() + cn->pruned_mass(), mass0, 1e-9,
              "Mass is not conserved with pruning at step " + std::to_string(step));
        require_inactive_modes_zero(*cn);
    }
}

void switching_on_and_off()
{
    test::ScratchDir dir("switching");
    auto cn = make(1e-6);
    cn->set_initial_state_full(initial_state(*cn));
    cn->initialize();
    cn->run(1, r_ex);
    const int initial = cn->active_mode_count();
    cn->t.resize(201);
    int step = 1;
    // The companion is ramped in, so its couplings switch modes on over time.
    for (; step <= 100; ++step) cn->step(step);
    const int grown = cn->active_mode_count();
    std::cout << "active modes: " << initial << " -> " << grown << std::endl;
    require(grown > initial, "No mode was switched on as the companion ramped in");

    // A coarser tolerance switches the faint modes off at the next check and
    // accounts for the mass they held.
    const double before = cn->mu * cn->dr * cn->psi.squaredNorm() + cn->pruned_mass();
    cn->prune_tolerance = 1e-3;
    for (; step <= 110; ++step) cn->step(step);
    require(cn->active_mode_count() < grown, "No mode was switched off");
    require(cn->pruned_mass() > 0.0, "Switched-off modes held no mass");
    require(cn->mu * cn->dr * cn->psi.squaredNorm() + cn->pruned_mass() <= before * (1 + 1e-12),
            "Pruned mass accounting gained mass");
    require_inactive_modes_zero(*cn);
}

std::string contents(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

fs::path new_run(const std::string &name, int steps)
{
    const fs::path root = fs::current_path() / name;
    fs::create_directories(root);
    const fs::path previous = fs::current_path();
    fs::current_path(root);
    auto cn = make(1e-8, true, 20);
    cn->set_initial_state_full(initial_state(*cn));
    cn->initialize();
    cn->run(steps, r_ex);
    fs::current_path(previous);
    return test::simulation_folder(root);
}

void resume_run(const fs::path &folder, int steps)
{
    auto cn = make(1e-8, true, 20);
    cn->initialize();
    cn->resume(folder.string(), steps, r_ex);
}

void exact_restart()
{
    test::ScratchDir dir("restart");
    const fs::path reference = new_run("reference", 121);
    // Stopped after step 70: the last snapshot (60) falls between two checks.
    const fs::path interrupted = new_run("interrupted", 71);
    resume_run(interrupted, 121);
    int compared = 0;
    for (const auto &entry : fs::recursive_directory_iterator(reference)) {
        if (!entry.is_regular_file()) continue;
        const fs::path relative = fs::relative(entry.path(), reference);
        require(contents(entry.path()) == contents(interrupted / relative),
                relative.string() + " differs after resuming a pruned run");
        ++compared;
    }
    require(compared > 10, "Too few files compared");

    // A checkpoint without the active set (older format) still resumes, by
    // selecting the set again from the snapshot.
    const fs::path old = new_run("old_checkpoint", 71);
    const std::string checkpoint = contents(old / "checkpoint.bin");
    const auto trailer = checkpoint.rfind("ACTIVE01");
    require(trailer != std::string::npos, "Checkpoint does not store the active set");
    std::ofstream(old / "checkpoint.bin", std::ios::binary | std::ios::trunc) << checkpoint.substr(0, trailer);
    resume_run(old, 121);
    require(test::read_rows(old / "t.txt").size() == 120, "Resume from an old checkpoint did not finish");
}

void rejects_invalid_settings()
{
    test::ScratchDir dir("invalid");
    for (double tolerance : {1.0, 2.0}) {
        auto cn = make(tolerance);
        cn->set_initial_state_full(initial_state(*cn));
        bool threw = false;
        try {
            cn->initialize();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "Tolerance >= 1 was accepted");
    }
    auto cn = make(1e-8);
    cn->prune_check_every = 0;
    cn->set_initial_state_full(initial_state(*cn));
    bool threw = false;
    try {
        cn->initialize();
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    require(threw, "check_every = 0 was accepted");
}

} // namespace

int main()
{
    omp_set_dynamic(0);
    omp_set_num_threads(2);
    Eigen::setNbThreads(2);
    return test::run_all(std::pair{"disabled by default", &disabled_by_default},
                         std::pair{"initial selection", &initial_selection},
                         std::pair{"accuracy against the full evolution", &accuracy_against_full_evolution},
                         std::pair{"mass conservation with pruning", &conservation_with_pruning},
                         std::pair{"switching modes on and off", &switching_on_and_off},
                         std::pair{"exact restart", &exact_restart},
                         std::pair{"invalid settings", &rejects_invalid_settings});
}
