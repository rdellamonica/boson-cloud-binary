/**
 * @file RadialGrid.hpp
 * @brief Radial nodes: uniform, or stretched with fine spacing where the cloud lives.
 */
#ifndef RADIAL_GRID_HPP
#define RADIAL_GRID_HPP

#include <Eigen/Core>
#include <string>

/**
 * @brief Shape of the radial grid.
 *
 * A stretched grid has nearly uniform spacing @f$h_{\rm in}@f$ inside
 * @c r_core and spacing @c stretch times larger far outside it, with a
 * smooth change over a width @c transition:
 * @f[
 *   h(r) \propto 1 + \frac{s - 1}{2}\Bigl[1 + \tanh\frac{r - r_{\rm core}}{w}\Bigr].
 * @f]
 * The nodes are equally spaced in @f$x(r) = \int_0^r dr'/h(r')@f$, so the
 * spacing changes smoothly and the finite differences stay second order.
 * @c initial_states/radial_grid.py builds the same nodes.
 */
struct RadialGridSpec
{
    std::string type = "uniform"; ///< "uniform" or "stretched".
    double r_core = 0.0;          ///< Radius of the finely resolved region.
    double stretch = 4.0;         ///< Outer spacing / inner spacing (>= 1).
    double transition = 0.0;      ///< Width of the change of spacing (> 0).
};

/**
 * @brief Nodes @f$\epsilon + g_i@f$, i = 0..N-1, with @f$g_0 = 0@f$ and @f$g_{N-1} = L@f$.
 * @throws std::invalid_argument for N < 3, L <= 0 or invalid stretching parameters.
 */
Eigen::VectorXd make_radial_grid(int N, double L, double epsilon, const RadialGridSpec &spec);

#endif
