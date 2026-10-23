/**
 * @file Utils.cpp
 * @brief Output-folder helper.
 */
#include "Utils.hpp"
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

std::string create_output_folder()
{
    namespace fs = std::filesystem;

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "Simulation_%Y_%m_%d_%H_%M_%S");
    std::string folder = ss.str();

    fs::create_directory(folder);
    fs::create_directory(folder + "/psi");
    fs::create_directory(folder + "/psi_dt");

    return folder;
}