/**
 * @file InitialData.cpp
 * @brief Reader for MODESBIN initial-state files.
 */
#include "InitialData.hpp"

Eigen::VectorXcd load_initial_modes_binary(
    const std::string &filename,
    const std::vector<mode> &modes)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open initial-state file " + filename);

    // header
    char magic[8];
    in.read(magic, 8);

    if (std::string(magic, 8) != "MODESBIN")
        throw std::runtime_error(filename + " is not a MODESBIN file");

    int32_t version;
    in.read(reinterpret_cast<char *>(&version), sizeof(version));

    if (version != 1)
        throw std::runtime_error("Unsupported MODESBIN version in " + filename);

    int64_t Nr;
    int64_t Nmodes;

    in.read(reinterpret_cast<char *>(&Nr), sizeof(Nr));
    in.read(reinterpret_cast<char *>(&Nmodes), sizeof(Nmodes));

    // read all stored modes
    std::map<std::pair<int, int>, Eigen::VectorXcd> stored_modes;

    for (int64_t k = 0; k < Nmodes; ++k)
    {
        int32_t l, m;

        in.read(reinterpret_cast<char *>(&l), sizeof(l));
        in.read(reinterpret_cast<char *>(&m), sizeof(m));

        std::vector<double> re(Nr);
        std::vector<double> im(Nr);

        in.read(reinterpret_cast<char *>(re.data()), Nr * sizeof(double));
        in.read(reinterpret_cast<char *>(im.data()), Nr * sizeof(double));

        Eigen::VectorXcd profile(Nr);

        for (int64_t i = 0; i < Nr; ++i)
            profile(i) = dcomplex(re[i], im[i]);

        stored_modes[{l, m}] = std::move(profile);
    }

    // build flattened vector
    Eigen::VectorXcd psi(Nr * modes.size());
    psi.setZero();

    for (size_t j = 0; j < modes.size(); ++j)
    {
        auto key = std::make_pair(modes[j].l, modes[j].m);

        auto it = stored_modes.find(key);

        if (it != stored_modes.end())
        {
            psi.segment(j * Nr, Nr) = it->second;
        }
        else
        {
            psi.segment(j * Nr, Nr).setZero();
        }
    }

    return psi;
}