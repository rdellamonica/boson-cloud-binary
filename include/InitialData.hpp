/**
 * @file InitialData.hpp
 * @brief Loading and constructing initial states.
 */
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "Definitions.hpp"

/**
 * @brief Read an initial state in the MODESBIN format.
 *
 * Layout (little-endian): the 8 bytes @c "MODESBIN", @c int32 version (= 1),
 * @c int64 Nr, @c int64 number of stored modes, then for each mode @c int32 l,
 * @c int32 m, Nr doubles with Re U_lm(r_i) and Nr doubles with Im U_lm(r_i).
 * The Python scripts in @c initial_states/ write this format.
 *
 * @param filename Path of the file.
 * @param modes    Modes of the solver; stored modes that are not listed are
 *                 ignored and listed modes that are not stored are zero.
 * @return The stacked state, of size Nr * modes.size().
 * @throws std::runtime_error if the file cannot be opened or has the wrong format.
 */
Eigen::VectorXcd load_initial_modes_binary(
    const std::string &filename,
    const std::vector<mode> &modes);

/**
 * @brief Hydrogenic 1s state of a mass-1 black hole displaced to (x_bh, 0, 0).
 *
 * Computes @f$U_{\ell m}(r) = r \int Y_{\ell m}^*\,\psi_{100}(|x - x_{BH}|)\,d\Omega@f$.
 * The result is the continuum-normalised state projected onto the supplied
 * grid and modes, with zero radial boundary values; the caller may normalise
 * its discrete norm.
 *
 * @param r     Increasing, non-negative radial nodes.
 * @param modes Modes to project on.
 * @param mu    Boson mass (Bohr radius 1/mu^2).
 * @param x_bh  Position of the black hole on the x axis.
 * @throws std::invalid_argument for invalid parameters.
 */
Eigen::VectorXcd displaced_100_state(const Eigen::VectorXd &r,
    const std::vector<mode> &modes, double mu, double x_bh);
