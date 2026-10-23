/**
 * @file CrankNicolsonModes_utils.cpp
 * @brief Output: run metadata, snapshots and diagnostic lines.
 */
#include "CrankNicolsonModes.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

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

    for (int i = 0; i < psi.size(); i++)
    {
        out << psi[i].real() << " " << psi[i].imag() << "\n";
    }
}

void CrankNicolsonModes::write_quantity_line_modes(const std::vector<double> &quantity, std::ofstream &quantity_file)
{
    for (size_t i = 0; i < quantity.size(); i++)
    {
        quantity_file << std::scientific << std::setprecision(12) << quantity[i];
        if (i + 1 < quantity.size())
            quantity_file << " ";
    }
    quantity_file << '\n';
}

void CrankNicolsonModes::write_quantity_line(const double &quantity, std::ofstream &quantity_file)
{
    quantity_file << std::scientific << std::setprecision(12) << quantity << '\n';
}

void CrankNicolsonModes::flush_diagnostics()
{
    for (auto* stream : {&flux_file, &energy_flux_file, &energy_file,
                         &norms_file, &time_file, &energy_modes_file, &R_file,
                         &theta_file, &Omega_file, &p_bin_file, &p_bin_inst_file,
                         &d_V_bin_dt_file, &M_1_file, &q_file})
        stream->flush();
}
