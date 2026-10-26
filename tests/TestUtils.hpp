/**
 * @file TestUtils.hpp
 * @brief Small helpers shared by the C++ tests: checks, scratch folders and
 *        MODESBIN writing.
 */
#pragma once
#include "Definitions.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace test {

/// Throw with @p message unless @p condition holds.
inline void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

/// Require |actual - expected| <= tol * max(1, |expected|).
inline void close(double actual, double expected, double tol, const std::string &message)
{
    if (!(std::abs(actual - expected) <= tol * std::max(1.0, std::abs(expected)))) {
        std::ostringstream text;
        text << message << " (actual " << actual << ", expected " << expected << ")";
        throw std::runtime_error(text.str());
    }
}

/// Fresh empty directory that becomes the working directory while in scope.
class ScratchDir
{
public:
    explicit ScratchDir(const std::string &name)
        : previous_(std::filesystem::current_path())
    {
        path_ = previous_ / (name + "_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path_);
        std::filesystem::current_path(path_);
    }
    ~ScratchDir() { std::filesystem::current_path(previous_); }
    const std::filesystem::path &path() const { return path_; }

private:
    std::filesystem::path previous_, path_;
};

/// The only Simulation_* folder inside @p root.
inline std::filesystem::path simulation_folder(const std::filesystem::path &root)
{
    std::filesystem::path found;
    for (const auto &entry : std::filesystem::directory_iterator(root))
        if (entry.is_directory() && entry.path().filename().string().rfind("Simulation_", 0) == 0) {
            require(found.empty(), "More than one simulation folder in " + root.string());
            found = entry.path();
        }
    require(!found.empty(), "No simulation folder in " + root.string());
    return found;
}

/// All whitespace-separated numbers of a text file, row by row.
inline std::vector<std::vector<double>> read_rows(const std::filesystem::path &path)
{
    std::ifstream in(path);
    require(bool(in), "Cannot open " + path.string());
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        std::vector<double> values;
        double value;
        while (row >> value) values.push_back(value);
        rows.push_back(values);
    }
    return rows;
}

/// Write a MODESBIN file with the given (l, m) profiles.
inline void write_modesbin(const std::string &filename,
                           const std::vector<std::pair<mode, Eigen::VectorXcd>> &profiles,
                           const char *magic = "MODESBIN", std::int32_t version = 1)
{
    std::ofstream out(filename, std::ios::binary);
    out.write(magic, 8);
    out.write(reinterpret_cast<const char *>(&version), sizeof(version));
    const std::int64_t nr = profiles.empty() ? 0 : profiles.front().second.size();
    const std::int64_t count = static_cast<std::int64_t>(profiles.size());
    out.write(reinterpret_cast<const char *>(&nr), sizeof(nr));
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const auto &[lm, profile] : profiles) {
        const std::int32_t l = lm.l, m = lm.m;
        out.write(reinterpret_cast<const char *>(&l), sizeof(l));
        out.write(reinterpret_cast<const char *>(&m), sizeof(m));
        for (int part = 0; part < 2; ++part)
            for (Eigen::Index i = 0; i < profile.size(); ++i) {
                const double value = part == 0 ? profile[i].real() : profile[i].imag();
                out.write(reinterpret_cast<const char *>(&value), sizeof(value));
            }
    }
}

/// Run named test functions, report each, and return the process exit code.
template <class... Tests>
int run_all(Tests &&...tests)
{
    int failures = 0;
    auto run = [&](const std::pair<const char *, void (*)()> &test) {
        try {
            test.second();
            std::cout << "[pass] " << test.first << std::endl;
        } catch (const std::exception &error) {
            ++failures;
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << std::endl;
        }
    };
    (run(tests), ...);
    return failures == 0 ? 0 : 1;
}

} // namespace test
