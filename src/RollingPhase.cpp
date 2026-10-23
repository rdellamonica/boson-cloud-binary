/**
 * @file RollingPhase.cpp
 * @brief Implementation of RollingPhaseAverage.
 */
#include "RollingPhase.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <cstdint>
#include <istream>
#include <ostream>

RollingPhaseAverage::RollingPhaseAverage(double n_orbits)
    : window_phase_(2.0 * M_PI * n_orbits)
{
    if (n_orbits <= 0.0)
    {
        throw std::invalid_argument(
            "RollingPhaseAverage: n_orbits must be positive");
    }
}

double RollingPhaseAverage::update(double time, double phase, double value)
{
    if (!has_previous_)
    {
        previous_ = {time, phase, value};
        initial_phase_ = phase;
        has_previous_ = true;

        average_ = value;
        return average_;
    }

    if (time <= previous_.time)
    {
        throw std::runtime_error(
            "RollingPhaseAverage: time must increase");
    }

    if (phase <= previous_.phase)
    {
        throw std::runtime_error(
            "RollingPhaseAverage: phase must increase");
    }

    Segment segment{
        previous_.time,
        time,
        previous_.phase,
        phase,
        previous_.value,
        value};

    segment.integral =
        0.5 * (segment.value0 + segment.value1) * (segment.time1 - segment.time0);

    segments_.push_back(segment);

    integral_ += segment.integral;
    duration_ += segment.time1 - segment.time0;

    if (phase - initial_phase_ >= window_phase_)
    {
        ready_ = true;
    }

    const double phase_cut = phase - window_phase_;

    // Remove segments lying entirely before the window.
    while (!segments_.empty() &&
           segments_.front().phase1 <= phase_cut)
    {
        const Segment &front = segments_.front();

        integral_ -= front.integral;
        duration_ -= front.time1 - front.time0;

        segments_.pop_front();
    }

    // Truncate the first segment exactly at phase_cut.
    if (!segments_.empty())
    {
        Segment &front = segments_.front();

        if (front.phase0 < phase_cut &&
            phase_cut < front.phase1)
        {
            const double fraction =
                (phase_cut - front.phase0) / (front.phase1 - front.phase0);

            const double time_cut =
                front.time0 + fraction * (front.time1 - front.time0);

            const double value_cut =
                front.value0 + fraction * (front.value1 - front.value0);

            const double removed_integral =
                0.5 * (front.value0 + value_cut) * (time_cut - front.time0);

            integral_ -= removed_integral;
            duration_ -= time_cut - front.time0;

            front.time0 = time_cut;
            front.phase0 = phase_cut;
            front.value0 = value_cut;
            front.integral =
                0.5 * (front.value0 + front.value1) * (front.time1 - front.time0);
        }
    }

    // Guard against tiny negative values from roundoff.
    duration_ = std::max(duration_, 0.0);

    if (duration_ > 0.0)
    {
        average_ = integral_ / duration_;
    }
    else
    {
        average_ = value;
    }

    previous_ = {time, phase, value};

    return average_;
}

bool RollingPhaseAverage::ready() const
{
    return ready_;
}

double RollingPhaseAverage::value() const
{
    return average_;
}

void RollingPhaseAverage::reset()
{
    segments_.clear();

    has_previous_ = false;
    ready_ = false;

    integral_ = 0.0;
    duration_ = 0.0;
    average_ = 0.0;
    initial_phase_ = 0.0;
}

namespace
{
template <class T>
void write_pod(std::ostream &out, const T &value)
{
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <class T>
void read_pod(std::istream &in, T &value)
{
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
}
}

void RollingPhaseAverage::save(std::ostream &out) const
{
    write_pod(out, window_phase_);
    write_pod(out, previous_);
    write_pod(out, has_previous_);
    write_pod(out, ready_);
    write_pod(out, initial_phase_);
    write_pod(out, integral_);
    write_pod(out, duration_);
    write_pod(out, average_);

    const std::uint64_t n = segments_.size();
    write_pod(out, n);
    for (const Segment &segment : segments_)
        write_pod(out, segment);
}

void RollingPhaseAverage::load(std::istream &in)
{
    read_pod(in, window_phase_);
    read_pod(in, previous_);
    read_pod(in, has_previous_);
    read_pod(in, ready_);
    read_pod(in, initial_phase_);
    read_pod(in, integral_);
    read_pod(in, duration_);
    read_pod(in, average_);

    std::uint64_t n = 0;
    read_pod(in, n);
    segments_.clear();
    for (std::uint64_t i = 0; i < n && in; ++i)
    {
        Segment segment;
        read_pod(in, segment);
        segments_.push_back(segment);
    }

    if (!in)
        throw std::runtime_error("RollingPhaseAverage: truncated checkpoint");
}
