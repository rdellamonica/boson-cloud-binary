/**
 * @file CrankNicolsonModes_utils.cpp
 * @brief Output: run metadata, snapshots and diagnostic lines.
 */
#include "CrankNicolsonModes.hpp"
#include <limits>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#if __has_include(<version>)
#include <version>
#endif
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
#include <charconv>
#endif

namespace
{
/// Append @p value as printf would with "%.*e" (@p scientific) or "%.*g",
/// which is also what an ostream prints with std::scientific or by default.
/// std::to_chars is used where available, since it is much faster.
void append_number(std::string &out, double value, int precision, bool scientific)
{
    char buffer[64];
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                      scientific ? std::chars_format::scientific : std::chars_format::general,
                                      precision);
    out.append(buffer, result.ptr);
#else
    const int length = std::snprintf(buffer, sizeof(buffer), scientific ? "%.*e" : "%.*g", precision, value);
    out.append(buffer, static_cast<std::size_t>(length));
#endif
}
} // namespace

void CrankNicolsonModes::save_run_info()
{
    namespace fs = std::filesystem;

    fs::copy_file(config_file, output_folder + "/config.toml", fs::copy_options::overwrite_existing);

    std::stringstream filename;
    filename << output_folder << "/r.txt";

    std::ofstream out(filename.str());
    if (!out)
    {
        std::cerr << "Could not open file: " << filename.str() << "\n";
        return;
    }

    // Full precision: on a stretched grid the nodes cannot be recomputed from N and L alone.
    out.precision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < Nr; ++i)
    {
        out << r[i] << "\n";
    }

    std::stringstream modes_filename;
    modes_filename << output_folder << "/modes.txt";

    std::ofstream modes_out(modes_filename.str());
    if (!modes_out)
    {
        std::cerr << "Could not open file: " << modes_filename.str() << "\n";
        return;
    }

    for (int n = 0; n < n_modes; n++)
    {
        modes_out << modes[n].l << "," << modes[n].m << std::endl;
    }
}

void CrankNicolsonModes::save_snapshot(int step_id, const Eigen::VectorXcd &psi, std::string name)
{
    namespace fs = std::filesystem;

    std::stringstream filename;
    filename << output_folder << "/" << name << "/snapshot_" << std::setw(6)
             << std::setfill('0') << step_id << ".txt";

    std::ofstream out(filename.str());
    if (!out)
    {
        std::cerr << "Could not open file: " << filename.str() << "\n";
        return;
    }

    // Same text as "out << re << ' ' << im", formatted in one buffer.
    std::string text;
    text.reserve(static_cast<std::size_t>(psi.size()) * 28);
    for (int i = 0; i < psi.size(); i++)
    {
        append_number(text, psi[i].real(), 6, false);
        text += ' ';
        append_number(text, psi[i].imag(), 6, false);
        text += '\n';
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void CrankNicolsonModes::write_quantity_line_modes(const std::vector<double> &quantity, std::ofstream &quantity_file)
{
    std::string line;
    line.reserve(quantity.size() * 20);
    for (size_t i = 0; i < quantity.size(); i++)
    {
        append_number(line, quantity[i], 12, true);
        line += i + 1 < quantity.size() ? ' ' : '\n';
    }
    if (quantity.empty())
        line += '\n';
    quantity_file.write(line.data(), static_cast<std::streamsize>(line.size()));
}

void CrankNicolsonModes::write_quantity_line(const double &quantity, std::ofstream &quantity_file)
{
    std::string line;
    append_number(line, quantity, 12, true);
    line += '\n';
    quantity_file.write(line.data(), static_cast<std::streamsize>(line.size()));
}

void CrankNicolsonModes::flush_diagnostics()
{
    for (auto* stream : {&flux_file, &energy_flux_file, &energy_file,
                         &norms_file, &time_file, &energy_modes_file, &R_file,
                         &theta_file, &Omega_file, &p_bin_file, &p_bin_inst_file,
                         &d_V_bin_dt_file, &M_1_file, &q_file, &mass_budget_file, &Lz_abs_file})
        if (stream->is_open())
            stream->flush();
}
