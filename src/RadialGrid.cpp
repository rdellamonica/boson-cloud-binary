/**
 * @file RadialGrid.cpp
 * @brief Uniform and stretched radial grids.
 */
#include "RadialGrid.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

Eigen::VectorXd make_radial_grid(int N, double L, double epsilon, const RadialGridSpec &spec)
{
    if (N < 3 || !(L > 0.0))
        throw std::invalid_argument("A radial grid needs N >= 3 and L > 0");
    Eigen::VectorXd r(N);
    if (spec.type == "uniform")
    {
        const double dr = L / (N - 1);
        for (int i = 0; i < N; ++i)
            r(i) = epsilon + i * dr;
        return r;
    }
    if (spec.type != "stretched")
        throw std::invalid_argument("grid.type must be uniform or stretched");
    if (!(spec.r_core > 0.0 && spec.r_core < L) || !(spec.stretch >= 1.0) || !(spec.transition > 0.0))
        throw std::invalid_argument("A stretched grid needs 0 < r_core < L, stretch >= 1 and transition > 0");

    // x(g) = int_0^g dg' / f(g') on a fine table (trapezoid rule), then the
    // nodes are the points where x reaches i / (N - 1) of its total.
    const auto f = [&](double g) {
        return 1.0 + 0.5 * (spec.stretch - 1.0) * (1.0 + std::tanh((g - spec.r_core) / spec.transition));
    };
    const long M = 64L * (N - 1) + 1;
    std::vector<double> g(M), x(M);
    double previous = 1.0 / f(0.0);
    g[0] = 0.0;
    x[0] = 0.0;
    for (long k = 1; k < M; ++k)
    {
        g[k] = L * static_cast<double>(k) / static_cast<double>(M - 1);
        const double current = 1.0 / f(g[k]);
        x[k] = x[k - 1] + 0.5 * (g[k] - g[k - 1]) * (previous + current);
        previous = current;
    }
    r(0) = epsilon;
    r(N - 1) = epsilon + L;
    long k = 0;
    for (int i = 1; i < N - 1; ++i)
    {
        const double target = x[M - 1] * static_cast<double>(i) / static_cast<double>(N - 1);
        while (x[k + 1] < target)
            ++k;
        const double w = (target - x[k]) / (x[k + 1] - x[k]);
        r(i) = epsilon + g[k] + w * (g[k + 1] - g[k]);
    }
    return r;
}
