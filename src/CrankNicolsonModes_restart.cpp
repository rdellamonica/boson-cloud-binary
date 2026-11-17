/**
 * @file CrankNicolsonModes_restart.cpp
 * @brief Checkpoints and resuming a run from its output folder.
 */
#include "CrankNicolsonModes.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr char checkpoint_magic[8] = {'G', 'M', 'C', 'K', 'P', 'T', '0', '1'};
// Optional trailer with the active-mode set of a pruned run. Older checkpoints
// end before it, and older readers ignore it.
constexpr char active_set_magic[8] = {'A', 'C', 'T', 'I', 'V', 'E', '0', '1'};
// Optional trailer after it with the cumulative absorbed masses.
constexpr char budget_magic[8] = {'B', 'U', 'D', 'G', 'E', 'T', '0', '1'};
// Optional trailer after it with the spins and the cumulative absorbed spins.
constexpr char spin_magic[8] = {'S', 'P', 'I', 'N', '0', '0', '0', '1'};
// Cumulative L_z budget terms (ionized, torque, absorbed, pruned, damped).
constexpr char Lz_budget_magic[8] = {'L', 'Z', 'B', 'U', 'D', 'G', '0', '1'};
// Per-mode CAP energy estimates (written only with a per-mode CAP).
constexpr char cap_energy_magic[8] = {'C', 'A', 'P', 'M', 'O', 'D', 'E', '1'};

template <class T>
void write_pod(std::ostream &out, const T &value)
{
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <class T>
void read_pod(std::istream &in, T &value)
{
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
}

// Byte offsets just past each '\n'; a trailing partial line is not counted.
std::vector<std::uintmax_t> line_ends(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open " + path);

    std::vector<std::uintmax_t> ends;
    std::vector<char> buffer(1 << 20);
    std::uintmax_t offset = 0;
    while (in)
    {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = in.gcount();
        for (std::streamsize i = 0; i < got; ++i)
            if (buffer[static_cast<std::size_t>(i)] == '\n')
                ends.push_back(offset + static_cast<std::uintmax_t>(i) + 1);
        offset += static_cast<std::uintmax_t>(got);
    }
    return ends;
}

// First n_rows values of a one-column diagnostic file.
std::vector<double> read_column(const std::string &path, int n_rows)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open " + path);

    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(n_rows));
    std::string line;
    while (static_cast<int>(values.size()) < n_rows && std::getline(in, line))
        values.push_back(std::strtod(line.c_str(), nullptr));

    if (static_cast<int>(values.size()) != n_rows)
        throw std::runtime_error("Too few rows in " + path);
    return values;
}

bool load_snapshot(const std::string &path, Eigen::VectorXcd &state, Eigen::Index size)
{
    std::ifstream in(path);
    if (!in)
        return false;

    Eigen::VectorXcd values(size);
    std::string line;
    Eigen::Index k = 0;
    while (std::getline(in, line))
    {
        if (line.empty())
            continue;
        if (k == size)
            return false;
        char *end = nullptr;
        const double re = std::strtod(line.c_str(), &end);
        char *end_im = nullptr;
        const double im = std::strtod(end, &end_im);
        if (end == line.c_str() || end_im == end)
            return false;
        values[k++] = dcomplex(re, im);
    }

    if (k != size)
        return false;
    state = values;
    return true;
}
}

std::vector<std::pair<std::ofstream *, std::string>> CrankNicolsonModes::diagnostic_files()
{
    return {
        {&flux_file, "flux.txt"},
        {&energy_flux_file, "energy_flux.txt"},
        {&energy_file, "energy.txt"},
        {&energy_modes_file, "energy_modes.txt"},
        {&norms_file, "norms.txt"},
        {&time_file, "t.txt"},
        {&p_bin_file, "p_bin.txt"},
        {&p_bin_inst_file, "p_bin_inst.txt"},
        {&d_V_bin_dt_file, "dV_bin_dt.txt"},
        {&R_file, "R_orb.txt"},
        {&Omega_file, "Omega_orb.txt"},
        {&M_1_file, "M_1.txt"},
        {&q_file, "q.txt"},
        {&theta_file, "theta_orb.txt"},
    };
}

void CrankNicolsonModes::open_diagnostics(std::ios::openmode mode)
{
    for (auto &[stream, name] : diagnostic_files())
    {
        stream->open(output_folder + "/" + name, mode);
        if (!*stream)
            throw std::runtime_error("Cannot open " + output_folder + "/" + name);
    }
}

void CrankNicolsonModes::save_checkpoint(int step_id) const
{
    namespace fs = std::filesystem;
    const std::string final_path = output_folder + "/checkpoint.bin";
    const std::string tmp_path = final_path + ".tmp";

    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            std::cerr << "Could not write checkpoint " << tmp_path << "\n";
            return;
        }

        out.write(checkpoint_magic, sizeof(checkpoint_magic));
        const std::int32_t header[3] = {step_id, Nr, n_modes};
        out.write(reinterpret_cast<const char *>(header), sizeof(header));
        for (const double value : {t_i, theta, phase, R, M1, M2, q, R1, R2, Omega, P, dt, p_bin})
            write_pod(out, value);

        const std::uint64_t size = static_cast<std::uint64_t>(psi.size());
        write_pod(out, size);
        out.write(reinterpret_cast<const char *>(psi.data()),
                  static_cast<std::streamsize>(size * sizeof(dcomplex)));
        power_average.save(out);

        out.write(active_set_magic, sizeof(active_set_magic));
        const std::int32_t count = n_modes;
        write_pod(out, count);
        out.write(active_modes.data(), static_cast<std::streamsize>(active_modes.size()));
        write_pod(out, pruned_mass_total);
        out.write(budget_magic, sizeof(budget_magic));
        write_pod(out, absorbed_mass_1);
        write_pod(out, absorbed_mass_2);
        out.write(spin_magic, sizeof(spin_magic));
        for (const double value : {chi1, chi2, absorbed_J_1, absorbed_J_2})
            write_pod(out, value);
        out.write(Lz_budget_magic, sizeof(Lz_budget_magic));
        for (const double value : {Lz_budget.ionized, Lz_budget.torque, Lz_budget.absorbed, Lz_budget.pruned,
                                   Lz_budget.damped})
            write_pod(out, value);
        if (cap_per_mode)
        {
            out.write(cap_energy_magic, sizeof(cap_energy_magic));
            write_pod(out, count);
            out.write(reinterpret_cast<const char *>(cap_energy.data()),
                      static_cast<std::streamsize>(n_modes * sizeof(double)));
            out.write(cap_has_energy.data(), static_cast<std::streamsize>(n_modes));
        }

        if (!out)
        {
            std::cerr << "Could not write checkpoint " << tmp_path << "\n";
            return;
        }
    }

    // A crash mid-write leaves the previous checkpoint intact.
    std::error_code error;
    fs::rename(tmp_path, final_path, error);
    if (error)
        std::cerr << "Could not replace checkpoint: " << error.message() << "\n";
}

bool CrankNicolsonModes::load_checkpoint(int step_id)
{
    std::ifstream in(output_folder + "/checkpoint.bin", std::ios::binary);
    if (!in)
        return false;

    char magic[sizeof(checkpoint_magic)];
    in.read(magic, sizeof(magic));
    std::int32_t header[3] = {-1, -1, -1};
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!in || std::memcmp(magic, checkpoint_magic, sizeof(magic)) != 0 ||
        header[0] != step_id || header[1] != Nr || header[2] != n_modes)
        return false;

    double values[13];
    for (double &value : values)
        read_pod(in, value);
    std::uint64_t size = 0;
    read_pod(in, size);
    if (!in || size != static_cast<std::uint64_t>(Nr) * static_cast<std::uint64_t>(n_modes))
        return false;

    Eigen::VectorXcd state(static_cast<Eigen::Index>(size));
    in.read(reinterpret_cast<char *>(state.data()),
            static_cast<std::streamsize>(size * sizeof(dcomplex)));
    if (!in)
        return false;

    RollingPhaseAverage average{orbits_average, power_window};
    try
    {
        average.load(in);
    }
    catch (const std::exception &)
    {
        return false;
    }

    // Without a stored active set, resume() selects one from the state.
    std::vector<char> active;
    double pruned = 0.0;
    char trailer[sizeof(active_set_magic)];
    std::int32_t count = -1;
    in.read(trailer, sizeof(trailer));
    read_pod(in, count);
    if (in && std::memcmp(trailer, active_set_magic, sizeof(trailer)) == 0 && count == n_modes)
    {
        active.resize(static_cast<std::size_t>(n_modes));
        in.read(active.data(), static_cast<std::streamsize>(active.size()));
        read_pod(in, pruned);
        if (!in)
            return false;
    }
    double absorbed[2] = {0.0, 0.0};
    bool has_budget = false;
    if (in)
    {
        char budget[sizeof(budget_magic)];
        in.read(budget, sizeof(budget));
        if (in && std::memcmp(budget, budget_magic, sizeof(budget)) == 0)
        {
            read_pod(in, absorbed[0]);
            read_pod(in, absorbed[1]);
            has_budget = static_cast<bool>(in);
        }
    }
    double spins[4] = {0.0, 0.0, 0.0, 0.0};
    bool has_spins = false;
    if (in && has_budget)
    {
        char spin[sizeof(spin_magic)];
        in.read(spin, sizeof(spin));
        if (in && std::memcmp(spin, spin_magic, sizeof(spin)) == 0)
        {
            for (double &value : spins)
                read_pod(in, value);
            has_spins = static_cast<bool>(in);
        }
    }
    double Lz_terms[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    bool has_Lz_budget = false;
    if (in && has_spins)
    {
        char tag[sizeof(Lz_budget_magic)];
        in.read(tag, sizeof(tag));
        if (in && std::memcmp(tag, Lz_budget_magic, sizeof(tag)) == 0)
        {
            for (double &value : Lz_terms)
                read_pod(in, value);
            has_Lz_budget = static_cast<bool>(in);
        }
    }
    Eigen::VectorXd cap_E;
    std::vector<char> cap_has;
    if (in && has_Lz_budget && cap_per_mode)
    {
        char tag[sizeof(cap_energy_magic)];
        std::int32_t modes_stored = -1;
        in.read(tag, sizeof(tag));
        read_pod(in, modes_stored);
        if (in && std::memcmp(tag, cap_energy_magic, sizeof(tag)) == 0 && modes_stored == n_modes)
        {
            cap_E.resize(n_modes);
            cap_has.resize(static_cast<std::size_t>(n_modes));
            in.read(reinterpret_cast<char *>(cap_E.data()), static_cast<std::streamsize>(n_modes * sizeof(double)));
            in.read(cap_has.data(), static_cast<std::streamsize>(n_modes));
            if (!in)
                cap_E.resize(0);
        }
    }

    t_i = values[0];
    theta = values[1];
    phase = values[2];
    R = values[3];
    M1 = values[4];
    M2 = values[5];
    q = values[6];
    R1 = values[7];
    R2 = values[8];
    Omega = values[9];
    P = values[10];
    dt = values[11];
    p_bin = values[12];
    psi = state;
    power_average = average;
    checkpoint_active_modes = active;
    pruned_mass_total = pruned;
    checkpoint_has_budget = has_budget;
    if (has_budget)
    {
        absorbed_mass_1 = absorbed[0];
        absorbed_mass_2 = absorbed[1];
    }
    checkpoint_has_spins = has_spins;
    if (has_spins)
    {
        chi1 = spins[0];
        chi2 = spins[1];
        absorbed_J_1 = spins[2];
        absorbed_J_2 = spins[3];
    }
    checkpoint_has_Lz_budget = has_Lz_budget;
    if (has_Lz_budget)
        Lz_budget = LzBudget{Lz_terms[0], Lz_terms[1], Lz_terms[2], Lz_terms[3], Lz_terms[4]};
    checkpoint_has_cap_energy = cap_E.size() == n_modes;
    if (checkpoint_has_cap_energy)
    {
        cap_energy = cap_E;
        cap_has_energy = cap_has;
    }
    return true;
}

// For runs without a matching checkpoint: rebuild the evolving scalars from
// the diagnostic rows 0..step_id-1 (row k is written by step k+1). These are
// stored with 12 significant digits, so the restart is accurate, not bitwise.
void CrankNicolsonModes::reconstruct_state_from_diagnostics(int step_id)
{
    const std::string f = output_folder + "/";
    const std::vector<double> times = read_column(f + "t.txt", step_id);
    const std::vector<double> Rs = read_column(f + "R_orb.txt", step_id);
    const std::vector<double> thetas = read_column(f + "theta_orb.txt", step_id);
    const std::vector<double> Omegas = read_column(f + "Omega_orb.txt", step_id);
    const std::vector<double> M1s = read_column(f + "M_1.txt", step_id);
    const std::vector<double> qs = read_column(f + "q.txt", step_id);
    const std::vector<double> p_bins = read_column(f + "p_bin.txt", step_id);
    const std::vector<double> p_bin_insts = read_column(f + "p_bin_inst.txt", step_id);
    // The orbit was charged p_bin_inst - Omega_z * (absorbed L_z rate); older runs have no such file.
    std::vector<double> Lz_rates(static_cast<std::size_t>(step_id), 0.0);
    if (std::filesystem::exists(f + "Lz_absorbed_rate.txt"))
        Lz_rates = read_column(f + "Lz_absorbed_rate.txt", step_id);
    const double sense = counter ? -1.0 : 1.0;

    // Before step 1, Omega comes from build_time_dependent_perturbation(0).
    double Omega_previous = isolated_black_hole ? 0.0 : (ramp_end_time() > 0.0 ? 0.0 : Omega_0);
    double t_previous = 0.0;
    phase = 0.0;
    power_average.reset();
    for (int k = 0; k < step_id; ++k)
    {
        phase += Omega_previous * (times[k] - t_previous);
        if (!isolated_black_hole)
            power_average.update(times[k], phase, p_bin_insts[k] - sense * Omegas[k] * Lz_rates[k]);
        Omega_previous = Omegas[k];
        t_previous = times[k];
    }

    if (step_id == 0)
        return; // The constructor's state is the initial state.

    const int last = step_id - 1;
    t_i = times[last];
    R = Rs[last];
    theta = thetas[last];
    M1 = M1s[last];
    q = qs[last];
    M2 = q * M1;
    R1 = M2 / (M1 + M2) * R;
    R2 = M1 / (M1 + M2) * R;
    if (isolated_black_hole)
    {
        // R_orb.txt holds the hole's distance from the origin.
        R1 = R;
        R2 = 0.0;
    }
    Omega = Omegas[last];
    p_bin = p_bins[last];

    // update_time_dependent_increment sets P and switches to dt = P/steps_per_period once
    // the orbit starts evolving; before that both keep their initial values.
    if (!isolated_black_hole && evolve_orbit && t_i > orbit_start_time())
    {
        P = 2.0 * M_PI / Omega;
        dt = P / steps_per_period;
    }
}

void CrankNicolsonModes::resume_active_modes_log(int start)
{
    // Keep the entries up to the snapshot, then log the set the continuation
    // starts with if it differs from the last kept one (no checkpoint, or
    // pruning switched on or off in config.toml).
    const std::string path = output_folder + "/active_modes.txt";
    std::vector<std::string> kept;
    {
        std::ifstream in(path);
        if (!in && !pruning_enabled())
            return; // Never pruned.
        std::string line;
        while (std::getline(in, line))
        {
            std::istringstream row(line);
            int step = 0;
            if (row >> step && step <= start)
                kept.push_back(line);
        }
    }
    std::string last_flags;
    if (!kept.empty())
    {
        std::istringstream row(kept.back());
        std::string step, time, count, flag;
        row >> step >> time >> count;
        while (row >> flag)
            last_flags += flag;
    }
    std::string flags;
    for (const char active : active_modes)
        flags += active ? '1' : '0';

    {
        std::ofstream out(path, std::ios::trunc);
        for (const auto &line : kept)
            out << line << '\n';
    }
    active_modes_file.open(path, std::ios::out | std::ios::app);
    if (flags != last_flags)
        write_active_modes(start + 1);
}

void CrankNicolsonModes::resume(const std::string &folder, int n_steps, double r_ex_)
{
    namespace fs = std::filesystem;

    std::cout << "Resuming simulation in " << folder << std::endl;

    set_extraction_radius(r_ex_, n_steps);
    output_folder = folder;

    // The stored mode ordering must match the one rebuilt from config.toml.
    {
        std::ifstream modes_in(folder + "/modes.txt");
        std::string line;
        int count = 0;
        while (std::getline(modes_in, line))
        {
            if (line.empty())
                continue;
            int l = 0, m = 0;
            char comma = 0;
            std::istringstream row(line);
            row >> l >> comma >> m;
            if (count >= n_modes || modes[count].l != l || modes[count].m != m)
                throw std::runtime_error("modes.txt does not match the modes built from config.toml");
            ++count;
        }
        if (modes_in.is_open() && count != n_modes)
            throw std::runtime_error("modes.txt does not match the modes built from config.toml");
    }

    // Complete diagnostic rows available in every file.
    std::size_t complete_rows = std::numeric_limits<std::size_t>::max();
    std::vector<std::vector<std::uintmax_t>> ends;
    for (auto &[stream, name] : diagnostic_files())
    {
        ends.push_back(line_ends(folder + "/" + name));
        complete_rows = std::min(complete_rows, ends.back().size());
    }
    // mass_budget.txt is absent in runs written before it existed.
    const std::string budget_path = folder + "/mass_budget.txt";
    const bool has_budget_file = fs::exists(budget_path);
    std::vector<std::uintmax_t> budget_ends;
    if (has_budget_file)
    {
        budget_ends = line_ends(budget_path);
        complete_rows = std::min(complete_rows, budget_ends.size());
    }

    // spins.txt is written by runs with dipolar absorption.
    const std::string spins_path = folder + "/spins.txt";
    const bool has_spins_file = fs::exists(spins_path);
    std::vector<std::uintmax_t> spins_ends;
    if (has_spins_file)
    {
        spins_ends = line_ends(spins_path);
        complete_rows = std::min(complete_rows, spins_ends.size());
    }

    const std::string Lz_path = folder + "/Lz_absorbed_rate.txt";
    const bool has_Lz_file = fs::exists(Lz_path);
    std::vector<std::uintmax_t> Lz_ends;
    if (has_Lz_file)
    {
        Lz_ends = line_ends(Lz_path);
        complete_rows = std::min(complete_rows, Lz_ends.size());
    }

    // Lz_budget.txt is absent in runs written before it existed; those resume without it.
    const std::string Lz_budget_path = folder + "/Lz_budget.txt";
    const bool has_Lz_budget_file = fs::exists(Lz_budget_path);
    std::vector<std::uintmax_t> Lz_budget_ends;
    if (has_Lz_budget_file)
    {
        Lz_budget_ends = line_ends(Lz_budget_path);
        complete_rows = std::min(complete_rows, Lz_budget_ends.size());
    }

    // cap_modes.txt exists only in runs with a per-mode CAP.
    const std::string cap_modes_path = folder + "/cap_modes.txt";
    const bool has_cap_modes_file = fs::exists(cap_modes_path);
    std::vector<std::uintmax_t> cap_modes_ends;
    if (has_cap_modes_file)
    {
        cap_modes_ends = line_ends(cap_modes_path);
        if (cap_per_mode)
            complete_rows = std::min(complete_rows, cap_modes_ends.size());
    }

    // Latest snapshot that is complete and has all of its diagnostic rows.
    std::vector<int> snapshot_steps;
    const std::regex pattern(R"(snapshot_(\d+)\.txt)");
    for (const auto &entry : fs::directory_iterator(folder + "/psi"))
    {
        std::smatch match;
        const std::string name = entry.path().filename().string();
        if (std::regex_match(name, match, pattern))
            snapshot_steps.push_back(std::stoi(match[1].str()));
    }
    std::sort(snapshot_steps.rbegin(), snapshot_steps.rend());

    int start = -1;
    for (const int candidate : snapshot_steps)
    {
        if (static_cast<std::size_t>(candidate) > complete_rows)
            continue;
        std::ostringstream path;
        path << folder << "/psi/snapshot_" << std::setw(6) << std::setfill('0') << candidate << ".txt";
        if (load_snapshot(path.str(), psi, static_cast<Eigen::Index>(Nr) * n_modes))
        {
            start = candidate;
            break;
        }
        std::cout << "Skipping incomplete snapshot " << path.str() << std::endl;
    }
    if (start < 0)
        throw std::runtime_error("No usable snapshot found in " + folder + "/psi");

    std::cout << "Restarting from snapshot " << start << std::endl;

    checkpoint_active_modes.clear();
    checkpoint_has_budget = false;
    checkpoint_has_spins = false;
    checkpoint_has_Lz_budget = false;
    checkpoint_has_cap_energy = false;
    Lz_budget = LzBudget{};
    // Spin angular momenta of the configured initial state (the masses are still the initial ones).
    const double initial_J1 = chi1 * M1 * M1, initial_J2 = chi2 * M2 * M2;
    if (load_checkpoint(start))
    {
        std::cout << "Exact state restored from checkpoint.bin." << std::endl;
    }
    else
    {
        std::cout << "No checkpoint for this snapshot: rebuilding the orbit state from the "
                     "diagnostics (snapshot and diagnostic precision limit the match)."
                  << std::endl;
        reconstruct_state_from_diagnostics(start);
    }

    // Drop everything written after the snapshot, then append.
    int file_index = 0;
    for (auto &[stream, name] : diagnostic_files())
    {
        const auto &file_ends = ends[file_index++];
        const std::uintmax_t keep = start == 0 ? 0 : file_ends[static_cast<std::size_t>(start - 1)];
        fs::resize_file(folder + "/" + name, keep);
    }
    open_diagnostics(std::ios::out | std::ios::app);
    if (has_budget_file)
    {
        // Older checkpoints do not store the absorbed masses: take them from the
        // last kept row (12 significant digits).
        if (!checkpoint_has_budget)
        {
            absorbed_mass_1 = absorbed_mass_2 = 0.0;
            if (start > 0)
            {
                std::ifstream in(budget_path);
                std::string line;
                for (int row = 0; row < start && std::getline(in, line); ++row)
                    if (row == start - 1)
                        std::istringstream(line) >> absorbed_mass_1 >> absorbed_mass_2;
            }
        }
        fs::resize_file(budget_path, start == 0 ? 0 : budget_ends[static_cast<std::size_t>(start - 1)]);
        mass_budget_file.open(budget_path, std::ios::out | std::ios::app);
    }
    if (!checkpoint_has_spins)
    {
        // Without stored spins: J_a is unchanged unless spins.txt records dipolar transfers.
        absorbed_J_1 = absorbed_J_2 = 0.0;
        bool from_file = false;
        if (has_spins_file && start > 0)
        {
            std::ifstream in(spins_path);
            std::string line;
            for (int row = 0; row < start && std::getline(in, line); ++row)
                if (row == start - 1)
                    from_file = static_cast<bool>(std::istringstream(line) >> chi1 >> chi2 >> absorbed_J_1 >> absorbed_J_2);
        }
        if (!from_file)
        {
            chi1 = initial_J1 / (M1 * M1);
            if (M2 > 0.0)
                chi2 = initial_J2 / (M2 * M2);
        }
    }
    if (has_spins_file)
    {
        fs::resize_file(spins_path, start == 0 ? 0 : spins_ends[static_cast<std::size_t>(start - 1)]);
        spins_file.open(spins_path, std::ios::out | std::ios::app);
    }
    else if (dipole_absorption)
    {
        // Dipolar absorption switched on for a run written without it: pad the kept rows.
        spins_file.open(spins_path, std::ios::out);
        for (int row = 0; row < start; ++row)
            write_quantity_line_modes({chi1, chi2, 0.0, 0.0}, spins_file);
    }
    if (has_Lz_budget_file)
    {
        // Without stored terms, take them from the last kept row (12 significant digits).
        if (!checkpoint_has_Lz_budget && start > 0)
        {
            std::ifstream in(Lz_budget_path);
            std::string line;
            for (int row = 0; row < start && std::getline(in, line); ++row)
                if (row == start - 1)
                    std::istringstream(line) >> Lz_budget.ionized >> Lz_budget.torque >> Lz_budget.absorbed >>
                        Lz_budget.pruned >> Lz_budget.damped;
        }
        fs::resize_file(Lz_budget_path, start == 0 ? 0 : Lz_budget_ends[static_cast<std::size_t>(start - 1)]);
        Lz_budget_file.open(Lz_budget_path, std::ios::out | std::ios::app);
    }
    if (cap_per_mode)
    {
        if (has_cap_modes_file && static_cast<std::size_t>(start) <= cap_modes_ends.size())
        {
            // Without stored estimates, take them from the last kept row (0 = no estimate).
            if (!checkpoint_has_cap_energy && start > 0)
            {
                std::ifstream in(cap_modes_path);
                std::string line;
                for (int row = 0; row < start && std::getline(in, line); ++row)
                    if (row == start - 1)
                    {
                        std::istringstream values(line);
                        for (int j = 0; j < n_modes; ++j)
                        {
                            double E = 0.0;
                            if (values >> E && E != 0.0)
                            {
                                cap_energy[j] = E;
                                cap_has_energy[j] = 1;
                            }
                        }
                    }
            }
            fs::resize_file(cap_modes_path, start == 0 ? 0 : cap_modes_ends[static_cast<std::size_t>(start - 1)]);
            cap_modes_file.open(cap_modes_path, std::ios::out | std::ios::app);
        }
        else
        {
            // Per-mode CAP switched on for a run written without it: pad the kept rows.
            cap_modes_file.open(cap_modes_path, std::ios::out);
            const std::vector<double> zeros(2 * static_cast<std::size_t>(n_modes), 0.0);
            for (int row = 0; row < start; ++row)
                write_quantity_line_modes(zeros, cap_modes_file);
        }
    }
    else if (has_cap_modes_file && static_cast<std::size_t>(start) <= cap_modes_ends.size())
        fs::resize_file(cap_modes_path, start == 0 ? 0 : cap_modes_ends[static_cast<std::size_t>(start - 1)]);
    if (has_Lz_file)
    {
        fs::resize_file(Lz_path, start == 0 ? 0 : Lz_ends[static_cast<std::size_t>(start - 1)]);
        Lz_abs_file.open(Lz_path, std::ios::out | std::ios::app);
    }
    else
    {
        // A run from before the file existed: pad the kept rows with zeros.
        Lz_abs_file.open(Lz_path, std::ios::out);
        for (int row = 0; row < start; ++row)
            Lz_abs_file << "0\n";
    }

    t.resize(n_steps);
    t.setZero();
    if (start < n_steps)
        t(start) = t_i;
    psi_last = Eigen::VectorXcd::Zero(psi.size());

    // H(t_n) with the restored orbit. build_time_dependent_perturbation sets
    // Omega from R, M1, M2; keep the stored value so the next step matches.
    const double restored_Omega = Omega;
    build_time_dependent_perturbation(t_i);
    Omega = restored_Omega;

    if (pruning_enabled())
    {
        // The stored active set makes the continuation exact; otherwise select
        // one from the snapshot as for a new run.
        if (!checkpoint_active_modes.empty())
            active_modes = checkpoint_active_modes;
        else
            update_active_modes(prune_check_every * dt, true);
        apply_active_set();
    }
    resume_active_modes_log(start);

    if (start + 1 >= n_steps)
    {
        std::cout << "The run had already reached its last step (" << n_steps - 1 << ")." << std::endl;
        return;
    }

    evolve(start + 1, n_steps);
}
