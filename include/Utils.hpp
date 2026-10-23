/**
 * @file Utils.hpp
 * @brief Output-folder helper.
 */
#pragma once
#include <string>

/**
 * @brief Create a new @c Simulation_YYYY_MM_DD_HH_MM_SS folder in the working
 *        directory, with @c psi/ and @c psi_dt/ subfolders.
 * @return The folder name.
 */
std::string create_output_folder();
