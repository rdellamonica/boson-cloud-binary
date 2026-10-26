// RollingPhaseAverage: the phase-windowed time average used for the binary power.
#include "RollingPhase.hpp"
#include "TestUtils.hpp"
#include <cmath>
#include <sstream>

using test::require;
using test::close;

namespace {

void first_sample_and_readiness()
{
    RollingPhaseAverage average(1.0);
    require(!average.ready(), "Empty average reports ready");
    close(average.update(0.0, 0.0, 3.5), 3.5, 0, "First sample is not returned as the average");
    for (int k = 1; k <= 99; ++k) average.update(k, 2 * M_PI * k / 100, 3.5);
    require(!average.ready(), "Ready before a full orbit of phase");
    average.update(100, 2 * M_PI, 3.5);
    require(average.ready(), "Not ready after a full orbit of phase");
    close(average.value(), 3.5, 1e-14, "Constant signal is not averaged to itself");
}

void linear_signal_is_exact()
{
    // For a linear signal the trapezoidal average over the window is exact,
    // also when the window edge falls between samples.
    const double rate = 0.37, n_orbits = 1.3, omega = 0.05, dt = 0.71;
    RollingPhaseAverage average(n_orbits);
    double t = 0;
    for (int k = 0; k < 2000; ++k, t += dt) {
        const double value = 2.0 + rate * t;
        const double result = average.update(t, omega * t, value);
        const double window = 2 * M_PI * n_orbits / omega;
        if (t >= window) {
            const double expected = 2.0 + rate * (t - 0.5 * window);
            close(result, expected, 1e-10, "Linear signal average is not the window midpoint value");
        }
    }
}

void oscillation_averages_out()
{
    // A signal periodic in the orbital phase averages to its mean over full orbits.
    RollingPhaseAverage average(2.0);
    double result = 0;
    for (int k = 0; k <= 4000; ++k) {
        const double phase = 2 * M_PI * k / 400;
        result = average.update(0.1 * k, phase, 1.0 + std::sin(phase) + 0.5 * std::cos(2 * phase));
    }
    close(result, 1.0, 1e-4, "Oscillation in the orbital phase did not average out");
}

void chirping_orbit()
{
    // The window is fixed in phase, not in time: with a growing frequency it spans
    // less time, and a constant signal still averages to itself.
    RollingPhaseAverage average(1.0);
    double phase = 0, result = 0;
    for (int k = 0; k <= 3000; ++k) {
        const double t = 0.5 * k, omega = 0.01 * (1 + 0.001 * k);
        result = average.update(t, phase, -2.25);
        phase += omega * 0.5;
    }
    close(result, -2.25, 1e-13, "Constant signal changed under a chirping orbit");
}

void rejects_non_monotonic_input()
{
    RollingPhaseAverage average(1.0);
    average.update(1.0, 1.0, 0.0);
    bool time_rejected = false, phase_rejected = false;
    try { average.update(1.0, 2.0, 0.0); } catch (const std::runtime_error &) { time_rejected = true; }
    try { average.update(2.0, 1.0, 0.0); } catch (const std::runtime_error &) { phase_rejected = true; }
    require(time_rejected, "Non-increasing time accepted");
    require(phase_rejected, "Non-increasing phase accepted");
    bool window_rejected = false;
    try { RollingPhaseAverage bad(0.0); } catch (const std::invalid_argument &) { window_rejected = true; }
    require(window_rejected, "Zero window accepted");
}

void reset_and_checkpoint()
{
    RollingPhaseAverage average(0.7);
    for (int k = 0; k < 300; ++k) average.update(k, 0.03 * k, std::cos(0.1 * k));

    std::stringstream buffer;
    average.save(buffer);
    RollingPhaseAverage restored(5.0); // the window is part of the saved state
    restored.load(buffer);
    for (int k = 300; k < 600; ++k) {
        const double a = average.update(k, 0.03 * k, std::cos(0.1 * k));
        const double b = restored.update(k, 0.03 * k, std::cos(0.1 * k));
        require(a == b, "Restored average differs from the original");
    }
    require(average.ready() == restored.ready(), "Restored readiness differs");

    std::stringstream truncated(buffer.str().substr(0, 20));
    bool rejected = false;
    try { restored.load(truncated); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "Truncated checkpoint accepted");

    average.reset();
    require(!average.ready(), "Reset average is still ready");
    close(average.update(1000, 50, 7.0), 7.0, 0, "Reset average kept old samples");
}
} // namespace

int main()
{
    return test::run_all(std::pair{"first sample and readiness", &first_sample_and_readiness},
                         std::pair{"linear signal is exact", &linear_signal_is_exact},
                         std::pair{"oscillation averages out", &oscillation_averages_out},
                         std::pair{"chirping orbit", &chirping_orbit},
                         std::pair{"rejects non-monotonic input", &rejects_non_monotonic_input},
                         std::pair{"reset and checkpoint", &reset_and_checkpoint});
}
