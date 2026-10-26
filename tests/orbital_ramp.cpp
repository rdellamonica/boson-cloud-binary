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

void check_close(double actual, double expected, const char* message)
{
    require(std::abs(actual - expected) <= 1e-11 * std::max(1.0, std::abs(expected)), message);
}

void set_radius(CrankNicolsonModes& cn, double radius)
{
    // Emulate an expanded orbit without relying on a particular cloud profile.
    cn.R = radius;
    cn.R1 = cn.M2 / (cn.M1 + cn.M2) * radius;
    cn.R2 = cn.M1 / (cn.M1 + cn.M2) * radius;
    cn.Omega = std::sqrt((cn.M1 + cn.M2) / std::pow(radius, 3));
    cn.P = 2 * M_PI / cn.Omega;
}

double smoothstep(double x)
{
    return x <= 0 ? 0.0 : x >= 1 ? 1.0 : 3*x*x - 2*x*x*x;
}

// Back-reaction weight at the end of a step: zero until the startup ramp ends,
// then a smooth step over one initial period.
double backreaction_ramp(double t_end, double ramp_end, double initial_period)
{
    return smoothstep((t_end - ramp_end) / initial_period);
}

double expected_radius(const CrankNicolsonModes& cn, double duration, double ramp = 1.0)
{
    // A zero cloud leaves only the forward-Euler GW contribution.
    return cn.R - (64.0/5.0)*cn.M1*cn.M2*(cn.M1+cn.M2)/std::pow(cn.R,3)*duration*ramp;
}

void prepare(CrankNicolsonModes& cn)
{
    cn.set_initial_state_full(Eigen::VectorXcd::Zero(cn.Nr*cn.n_modes));
    cn.initialize();
    cn.run(1, 100.0);
    cn.t.resize(32);
}

void test_expanding_orbit(bool absorb)
{
    CrankNicolsonModes cn(64, 200.0, 200.0, 0.1, 0, 0.2, 0.0, 0.25,
                         0.0, 50.0, false, absorb, 100, 5.0, 15.0, "fixture.toml");
    const double initial_period = cn.P;
    const double ramp_end = 5 * initial_period;
    prepare(cn);
    cn.dt = ramp_end + 0.1*initial_period;
    cn.step(1);

    // This makes 5*P exceed the elapsed time, as in the reported staircase.
    set_radius(cn, 80.0);
    cn.dt = 0.1*initial_period;
    require(5*cn.P > cn.t[1] + cn.dt, "Fixture must cross the old moving gate");
    // 0.2 initial periods after activation: back-reaction and absorption are
    // both at smoothstep(0.2) of their final strength.
    const double ramp = backreaction_ramp(cn.t[1] + cn.dt, ramp_end, initial_period);
    const double expected = expected_radius(cn, cn.dt, ramp);
    cn.step(2);
    check_close(cn.R, expected, "Expanded orbit froze after the initial ramp completed");
    require((cn.V - (cn.V1 + cn.V2)).norm() < 1e-13,
            "Completed companion-potential ramp restarted after expansion");
    const double partial_sink = cn.sink_strength(1);
    if (absorb)
        require(partial_sink > 0, "Absorption switched off again");

    // Pass the end of the absorption ramp, then increase P again. The potential
    // and absorption must remain fully enabled, and R must update on every step.
    cn.dt = initial_period;
    cn.step(3);
    set_radius(cn, 80.0);
    for (int step = 4; step <= 8; ++step) {
        cn.dt = 0.05*initial_period;
        const double next_radius = expected_radius(cn, cn.dt);
        cn.step(step);
        check_close(cn.R, next_radius, "Orbital evolution skipped a post-ramp step");
        require((cn.V - (cn.V1 + cn.V2)).norm() < 1e-13, "Potential ramp is not permanently complete");
        if (absorb && step == 4) {
            // Same geometry as step 2 (R = 80), now at full absorption strength.
            const double full_sink = cn.sink_strength(1);
            check_close(partial_sink, full_sink*smoothstep(0.2),
                        "Absorption ramp changed when the orbital period changed");
        }
    }
}

void test_crossing_step()
{
    CrankNicolsonModes cn(32, 200.0, 200.0, 0.1, 0, 0.2, 0.0, 0.25,
                         0.0, 50.0, false, false, 100, 5.0, 15.0, "fixture.toml");
    const double initial_period = cn.P;
    prepare(cn);
    cn.dt = 4.9*initial_period;
    cn.step(1);
    check_close(cn.R, 50.0, "Orbit moved before activation");
    cn.dt = 0.2*initial_period;
    // Only the 0.1 P0 after activation counts, weighted by the back-reaction ramp.
    const double expected = expected_radius(cn, 0.1*initial_period,
                                            backreaction_ramp(5.1*initial_period, 5*initial_period, initial_period));
    cn.step(2);
    check_close(cn.R, expected, "Ramp crossing integrated time before activation");
    check_close(cn.t[2], 5.1*initial_period, "Adaptive dt changed the completed step timestamp");
}

void test_frequency_only_ramp(bool ramp_potential)
{
    CrankNicolsonModes cn(32, 200.0, 200.0, 0.1, 2, 0.2, 0.0, 0.25,
                         0.0, 50.0, false, false, 100, 5.0, 15.0, "fixture.toml");
    cn.ramp_secondary_potential = ramp_potential;
    const double initial_period = cn.P;
    prepare(cn);
    // At t = 0 Omega is zero in both modes; only the companion weight differs.
    const double start_weight = ramp_potential ? 0.0 : 1.0;
    require((cn.V - (cn.V1 + start_weight*cn.V2)).norm() < 1e-13,
            "Companion potential has the wrong weight at t = 0");
    require(cn.V2.norm() > 0, "Fixture has no companion potential");
    cn.dt = 2.0*initial_period;
    cn.step(1);
    const double x = 0.4, s = 3*x*x - 2*x*x*x;
    check_close(cn.Omega, cn.Omega_0*s, "Orbital frequency did not follow the ramp");
    const double weight = ramp_potential ? s : 1.0;
    require((cn.V - (cn.V1 + weight*cn.V2)).norm() < 1e-13*std::max(1.0, cn.V.norm()),
            "Companion potential has the wrong weight during the ramp");
    check_close(cn.R, 50.0, "Orbit moved during the ramp");
    cn.dt = 4.0*initial_period;
    cn.step(2);
    require((cn.V - (cn.V1 + cn.V2)).norm() < 1e-13, "Companion potential incomplete after the ramp");
}

void test_zero_ramp_fixed_orbit()
{
    CrankNicolsonModes cn(32, 200.0, 20.0, 0.1, 0, 0.2, 0.0, 0.25,
                         0.0, 50.0, true, false, 100, 0.0, 15.0, "fixture.toml");
    cn.evolve_orbit = false;
    prepare(cn);
    check_close(cn.Omega, cn.Omega_0, "Zero ramp left the binary stationary");
    cn.step(1);
    cn.step(2); // The phase average also requires a strictly increasing phase.
    check_close(cn.R, 50.0, "Disabled orbital evolution changed R");
    require(cn.phase > 0 && cn.theta < 0, "Counter-rotating phase did not advance");
}
}

int main()
{
    try {
        omp_set_num_threads(1);
        Eigen::setNbThreads(1);
        std::ofstream("fixture.toml") << "# orbital ramp regression\n";
        test_expanding_orbit(false);
        test_expanding_orbit(true);
        test_crossing_step();
        test_frequency_only_ramp(true);
        test_frequency_only_ramp(false);
        test_zero_ramp_fixed_orbit();
        std::cout << "Orbital ramp regression checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
