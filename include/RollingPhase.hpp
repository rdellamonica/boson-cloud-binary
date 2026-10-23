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
 */
class RollingPhaseAverage
{
public:
    /// @throws std::invalid_argument unless @p n_orbits > 0.
    RollingPhaseAverage(double n_orbits);

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

    /// Write the exact state, used by restart checkpoints.
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

    std::deque<Segment> segments_;

    Sample previous_{};

    bool has_previous_ = false;
    bool ready_ = false;

    double initial_phase_ = 0.0;

    double integral_ = 0.0;
    double duration_ = 0.0;
    double average_ = 0.0;
};