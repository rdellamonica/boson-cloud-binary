/**
 * @file RollingPhase.hpp
 * @brief Time average over a fixed window of orbital phase.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include <iosfwd>
#include <stdexcept>

/**
 * @brief Rolling time average of a signal over the last @c n_orbits orbits of phase.
 *
 * Samples are joined by linear segments and integrated with the trapezoidal
 * rule. The window is cut exactly at @f$\phi - 2\pi n_{\rm orbits}@f$, so the
 * average is insensitive to a changing orbital period.
 *
 * The window shape is either a boxcar (every sample of the window weighs the
 * same) or a Hann taper in the phase, @f$w = \sin^2(\pi x)@f$ with @f$x@f$ the
 * position in the window, so that both ends weigh zero. A boxcar of N orbits
 * cancels only frequencies k/N per orbit; the Hann window also cancels k/N for
 * k >= 2 and suppresses everything in between with sidelobes falling as f^-3,
 * which removes beats at non-harmonic frequencies that a boxcar lets through.
 */
class RollingPhaseAverage
{
public:
    /// Shape of the averaging window in the orbital phase.
    enum class Window
    {
        Boxcar,
        Hann
    };

    /// @throws std::invalid_argument unless @p n_orbits > 0.
    RollingPhaseAverage(double n_orbits, Window window = Window::Boxcar);

    /**
     * @brief Add a sample and return the current average.
     * @throws std::runtime_error unless both time and phase increase.
     */
    double update(double time, double phase, double value);
    /// True once a full window of phase has been accumulated.
    bool ready() const;
    /// The current average.
    double value() const;
    /// Discard all samples.
    void reset();

    /// The window shape.
    Window window() const { return window_; }

    /// Write the exact state, used by restart checkpoints. The window shape is
    /// not stored: a resumed run takes it from its configuration.
    void save(std::ostream &out) const;
    /// Read a state written by save().
    /// @throws std::runtime_error if the stream is truncated.
    void load(std::istream &in);

private:
    struct Sample
    {
        double time;
        double phase;
        double value;
    };

    struct Segment
    {
        double time0;
        double time1;

        double phase0;
        double phase1;

        double value0;
        double value1;

        double integral = 0.0;
    };

    double window_phase_;
    Window window_ = Window::Boxcar;

    /// Hann-weighted mean over the current segments, cut at @p phase_cut.
    double hann_average(double phase_cut, double fallback) const;

    std::deque<Segment> segments_;

    Sample previous_{};

    bool has_previous_ = false;
    bool ready_ = false;

    double initial_phase_ = 0.0;

    double integral_ = 0.0;
    double duration_ = 0.0;
    double average_ = 0.0;
};