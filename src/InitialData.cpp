/**
 * @file InitialData.cpp
 * @brief Reader for MODESBIN initial-state files.
 */
#include "InitialData.hpp"
#include <algorithm>
#include <cmath>

namespace
{
struct ModesFile
{
    std::vector<double> nodes; ///< Radial nodes (version 2 only).
    int64_t Nr = 0;
    std::map<std::pair<int, int>, Eigen::VectorXcd> profiles;
};

ModesFile read_modesbin(const std::string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open initial-state file " + filename);

    char magic[8];
    in.read(magic, 8);
    if (!in || std::string(magic, 8) != "MODESBIN")
        throw std::runtime_error(filename + " is not a MODESBIN file");

    int32_t version = 0;
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != 1 && version != 2)
        throw std::runtime_error("Unsupported MODESBIN version in " + filename);

    ModesFile file;
    int64_t Nmodes = 0;
    in.read(reinterpret_cast<char *>(&file.Nr), sizeof(file.Nr));
    in.read(reinterpret_cast<char *>(&Nmodes), sizeof(Nmodes));
    if (!in || file.Nr < 0 || Nmodes < 0)
        throw std::runtime_error("Truncated MODESBIN header in " + filename);
    if (version == 2)
    {
        file.nodes.resize(file.Nr);
        in.read(reinterpret_cast<char *>(file.nodes.data()), file.Nr * sizeof(double));
    }

    for (int64_t k = 0; k < Nmodes; ++k)
    {
        int32_t l, m;
        in.read(reinterpret_cast<char *>(&l), sizeof(l));
        in.read(reinterpret_cast<char *>(&m), sizeof(m));

        std::vector<double> re(file.Nr);
        std::vector<double> im(file.Nr);
        in.read(reinterpret_cast<char *>(re.data()), file.Nr * sizeof(double));
        in.read(reinterpret_cast<char *>(im.data()), file.Nr * sizeof(double));
        if (!in)
            throw std::runtime_error("Truncated MODESBIN data in " + filename);

        Eigen::VectorXcd profile(file.Nr);
        for (int64_t i = 0; i < file.Nr; ++i)
            profile(i) = dcomplex(re[i], im[i]);
        file.profiles[{l, m}] = std::move(profile);
    }
    return file;
}

/// Cubic Lagrange interpolation of (x, y) at @p t from the four nearest nodes; zero outside [x0, xN].
dcomplex interpolate(const std::vector<double> &x, const Eigen::VectorXcd &y, double t)
{
    const auto n = static_cast<Eigen::Index>(x.size());
    if (n == 0 || t < x.front() || t > x.back())
        return 0.0;
    if (n < 4)
    {
        const auto i = std::min<Eigen::Index>(std::upper_bound(x.begin(), x.end(), t) - x.begin(), n - 1);
        if (i == 0)
            return y[0];
        const double w = (t - x[i - 1]) / (x[i] - x[i - 1]);
        return (1.0 - w) * y[i - 1] + w * y[i];
    }
    const Eigen::Index i = std::upper_bound(x.begin(), x.end(), t) - x.begin() - 1;
    const Eigen::Index first = std::clamp<Eigen::Index>(i - 1, 0, n - 4);
    dcomplex value = 0.0;
    for (Eigen::Index a = first; a < first + 4; ++a)
    {
        double basis = 1.0;
        for (Eigen::Index b = first; b < first + 4; ++b)
            if (b != a)
                basis *= (t - x[b]) / (x[a] - x[b]);
        value += basis * y[a];
    }
    return value;
}
} // namespace

Eigen::VectorXcd load_initial_modes_binary(
    const std::string &filename,
    const std::vector<mode> &modes)
{
    const ModesFile file = read_modesbin(filename);
    const int64_t Nr = file.Nr;

    // build flattened vector
    Eigen::VectorXcd psi(Nr * modes.size());
    psi.setZero();

    for (size_t j = 0; j < modes.size(); ++j)
    {
        auto it = file.profiles.find(std::make_pair(modes[j].l, modes[j].m));
        if (it != file.profiles.end())
            psi.segment(j * Nr, Nr) = it->second;
    }

    return psi;
}

Eigen::VectorXcd load_initial_modes_binary(
    const std::string &filename,
    const std::vector<mode> &modes,
    const Eigen::VectorXd &r,
    bool uniform_grid)
{
    const ModesFile file = read_modesbin(filename);
    const auto Nr = static_cast<int64_t>(r.size());
    const double length = r(Nr - 1) - r(0);

    // Same grid: copy the values, as files written for a uniform grid always were.
    bool same = file.Nr == Nr && (uniform_grid || !file.nodes.empty());
    if (same && !uniform_grid)
        for (int64_t i = 0; i < Nr && same; ++i)
            same = std::abs(file.nodes[i] - r(i)) <= 1e-9 * length;
    if (same)
        return load_initial_modes_binary(filename, modes);

    // Otherwise interpolate. A version-1 file holds no nodes: it was written on
    // the uniform grid from 0 to L.
    std::vector<double> nodes = file.nodes;
    if (nodes.empty())
    {
        nodes.resize(file.Nr);
        for (int64_t i = 0; i < file.Nr; ++i)
            nodes[i] = file.Nr > 1 ? length * static_cast<double>(i) / static_cast<double>(file.Nr - 1) : 0.0;
    }
    std::cout << "Interpolating the initial state from " << file.Nr << " nodes onto the "
              << Nr << "-node radial grid." << std::endl;
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(Nr * static_cast<int64_t>(modes.size()));
    for (size_t j = 0; j < modes.size(); ++j)
    {
        auto it = file.profiles.find(std::make_pair(modes[j].l, modes[j].m));
        if (it == file.profiles.end())
            continue;
        for (int64_t i = 0; i < Nr; ++i)
            psi[j * Nr + i] = interpolate(nodes, it->second, r(i));
    }
    return psi;
}