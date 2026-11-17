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
void hann_constant_and_linear()
{
    // Normalised weights: a constant is returned exactly, also on a chirping
    // orbit; at constant Omega the taper is symmetric, so a linear signal gives
    // its value at the window centre, as for the boxcar.
    RollingPhaseAverage chirp(2.0, RollingPhaseAverage::Window::Hann);
    require(chirp.window() == RollingPhaseAverage::Window::Hann, "Window shape not kept");
    double phase = 0, result = 0;
    for (int k = 0; k <= 4000; ++k) {
        const double t = 0.5 * k, omega = 0.01 * (1 + 0.001 * k);
        result = chirp.update(t, phase, -2.25);
        phase += omega * 0.5;
    }
    close(result, -2.25, 1e-13, "Hann: constant signal changed under a chirping orbit");

    const double rate = 0.37, n_orbits = 1.3, omega = 0.05, dt = 0.71;
    RollingPhaseAverage linear(n_orbits, RollingPhaseAverage::Window::Hann);
    double t = 0;
    for (int k = 0; k < 2000; ++k, t += dt) {
        const double value = 2.0 + rate * t;
        const double res = linear.update(t, omega * t, value);
        const double window = 2 * M_PI * n_orbits / omega;
        if (t >= window)
            close(res, 2.0 + rate * (t - 0.5 * window), 1e-6, "Hann: linear signal average is not the window centre value");
    }
}

void hann_suppresses_non_harmonic_beat()
{
    // A beat at 1.15 cycles per orbit, as left by a ringing cloud. A boxcar of
    // N orbits passes |sinc(1.15 N)| (12.6% at N = 1); a Hann window passes
    // |sinc(x)/(1 - x^2)|, x = 1.15 N: 2.6% at 2 orbits and 0.33% at 4. Harmonics of the orbit are removed by both.
    auto residual = [](double n_orbits, RollingPhaseAverage::Window window, double cycles) {
        RollingPhaseAverage average(n_orbits, window);
        const int per_orbit = 300;
        double worst = 0;
        for (int k = 0; k <= 40 * per_orbit; ++k) {
            const double phase = 2 * M_PI * k / per_orbit;
            const double r = average.update(k, phase, 1.0 + std::cos(cycles * phase + 0.3)) - 1.0;
            if (k > 10 * per_orbit) worst = std::max(worst, std::abs(r));
        }
        return worst;
    };
    using W = RollingPhaseAverage::Window;
    const double box1 = residual(1.0, W::Boxcar, 1.15), hann2 = residual(2.0, W::Hann, 1.15), hann4 = residual(4.0, W::Hann, 1.15);
    require(box1 > 0.1, "Boxcar leakage at 1.15 per orbit smaller than expected");
    require(hann2 < 0.035, "Hann (2 orbits) leaks more than 3.5% at 1.15 per orbit");
    require(hann4 < 0.004, "Hann (4 orbits) leaks more than 0.4% at 1.15 per orbit");
    require(residual(2.0, W::Hann, 1.0) < 1e-6 && residual(2.0, W::Hann, 2.0) < 1e-6,
            "Hann (2 orbits) does not cancel the orbital harmonics");
}

void hann_checkpoint()
{
    // The saved state is the same as for the boxcar; the shape comes from the
    // restored object (the configuration), and a resume continues exactly.
    using W = RollingPhaseAverage::Window;
    RollingPhaseAverage average(2.0, W::Hann);
    for (int k = 0; k < 900; ++k) average.update(k, 0.03 * k, std::cos(0.1 * k));
    std::stringstream buffer;
    average.save(buffer);
    RollingPhaseAverage restored(5.0, W::Hann);
    restored.load(buffer);
    for (int k = 900; k < 1500; ++k) {
        const double a = average.update(k, 0.03 * k, std::cos(0.1 * k));
        const double b = restored.update(k, 0.03 * k, std::cos(0.1 * k));
        require(a == b, "Restored Hann average differs from the original");
    }
}
} // namespace

int main()
{
    return test::run_all(std::pair{"first sample and readiness", &first_sample_and_readiness},
                         std::pair{"linear signal is exact", &linear_signal_is_exact},
                         std::pair{"oscillation averages out", &oscillation_averages_out},
                         std::pair{"chirping orbit", &chirping_orbit},
                         std::pair{"rejects non-monotonic input", &rejects_non_monotonic_input},
                         std::pair{"reset and checkpoint", &reset_and_checkpoint},
                         std::pair{"Hann: constant and linear", &hann_constant_and_linear},
                         std::pair{"Hann: suppresses a non-harmonic beat", &hann_suppresses_non_harmonic_beat},
                         std::pair{"Hann: checkpoint", &hann_checkpoint});
}
