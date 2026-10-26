#include "CrankNicolsonModes.hpp"
#include <omp.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

Eigen::VectorXcd run_case(bool absorb, double smear, double q, double radius,
                         double ramp_periods, bool expect_loss)
{
    // dr=1; equal-mass centers are exactly on nodes, including sub-grid cases.
    CrankNicolsonModes cn(65, 64.0, 1.0, 0.0, 0, 0.2, 0.1, q,
                         0.0, radius, true, absorb, 100, ramp_periods, smear, "fixture.toml");
    cn.evolve_orbit = false;
    cn.dt = cn.P/16;
    const double dt = cn.dt, period = cn.P, r1 = cn.R1, r2 = cn.R2;
    Eigen::VectorXcd initial = Eigen::VectorXcd::Zero(cn.Nr);
    for (int i = 1; i < cn.Nr-1; ++i)
        initial[i] = std::sin(M_PI*i/(cn.Nr-1));
    cn.set_initial_state_full(initial);
    cn.initialize();
    cn.run(1, 40.0);
    cn.t.resize(41);
    const double initial_norm = cn.psi.squaredNorm();
    require(cn.sink_strength(1) + cn.sink_strength(2) == 0.0,
            "Absorption must start at zero ramp amplitude");

    double total_loss = 0.0;
    double last_sink = 0.0;
    for (int step = 1; step <= 40; ++step) {
        const Eigen::VectorXcd previous = cn.psi;
        cn.step(step);
        require(cn.psi.allFinite() && Eigen::MatrixXcd(cn.H).allFinite(),
                "Empty or unresolved absorber produced nonfinite values");
        require(cn.R == radius && cn.M1 == 1.0 && cn.M2 == q &&
                cn.R1 == r1 && cn.R2 == r2 && cn.P == period && cn.dt == dt,
                "Fixed-orbit absorption changed separation, masses, period, or dt");
        const double loss = previous.squaredNorm() - cn.psi.squaredNorm();
        require(loss >= -1e-10*initial_norm, "Absorption increased the field norm");
        total_loss += loss;
        const Eigen::VectorXcd midpoint = 0.5*(previous+cn.psi);
        // CN's discrete norm balance for H_full=H_hermitian-i*Gamma/2.
        const double predicted_loss = dt*midpoint.dot(cn.apply_sink(midpoint)).real();
        require(std::abs(loss-predicted_loss) < 1e-9*initial_norm,
                "Field loss violates the CN absorption balance");
        const double sink = cn.sink_strength(1) + cn.sink_strength(2);
        if (cn.t[step] <= ramp_periods*period)
            require(sink == 0.0, "Absorption started before its scheduled ramp");
        if (cn.t[step] >= (ramp_periods+1)*period && last_sink > 0.0)
            require(std::abs(sink-last_sink) < 1e-12,
                    "Fixed-orbit absorber changed after the ramp completed");
        if (cn.t[step] >= (ramp_periods+1)*period) last_sink = sink;
    }
    require(cn.phase > 0 && cn.theta < 0 && std::abs(cn.phase+cn.theta) < 1e-12,
            "Fixed orbit must retain counter-rotating motion");
    if (ramp_periods == 0.0)
        require(std::abs(cn.phase-40*dt*cn.Omega_0) < 1e-12,
                "Fixed-orbit angular speed changed");
    if (expect_loss) {
        require(total_loss > 1e-4*initial_norm && last_sink > 0.0,
                "Resolved absorber did not remove field norm on a fixed orbit");
        // With unequal masses, only the wider of these separated bumps is resolved.
        if (q != 1.0) {
            const int disabled_body = q < 1.0 ? 2 : 1;
            const int active_body = q < 1.0 ? 1 : 2;
            require(cn.sink_strength(disabled_body) == 0.0,
                    "Sub-grid absorber was not disabled independently");
            require(cn.sink_strength(active_body) > 0.0,
                    "Disabling one absorber also disabled the resolved absorber");
        }
    } else {
        require(std::abs(total_loss) < 1e-9*initial_norm && last_sink == 0.0,
                "Disabled or unsupported absorption removed field norm");
    }
    return cn.psi;
}
}

int main()
{
    try {
        omp_set_num_threads(1);
        Eigen::setNbThreads(1);
        std::ofstream("fixture.toml") << "# fixed-orbit absorption regression\n";
        const auto control = run_case(false, 4.0, 1.0, 20.0, 0.0, false);
        run_case(true, 4.0, 1.0, 20.0, 0.0, true);
        run_case(true, 4.0, 1.0, 20.0, 1.0, true);
        for (double smear : {0.0, 0.5}) {
            const auto unresolved = run_case(true, smear, 1.0, 20.0, 0.0, false);
            require((unresolved-control).norm() < 1e-10*control.norm(),
                    "Unresolved absorbers differ from absorption=false");
        }
        run_case(true, 4.0, 0.1, 22.0, 0.0, true); // Secondary unresolved.
        run_case(true, 0.4, 10.0, 22.0, 0.0, true); // Primary unresolved; only the secondary absorbs.
        run_case(true, 4.0, 1.0, 200.0, 0.0, false); // No interior support.
        std::cout << "Fixed-orbit absorption checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
