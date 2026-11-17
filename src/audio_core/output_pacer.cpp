// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include "audio_core/output_pacer.h"
#include "common/trace_markers.h"

namespace AudioCore {

namespace {

constexpr double RATE = native_sample_rate;

constexpr double FramesIn(double milliseconds) {
    return milliseconds * RATE / 1000.0;
}

// The margin: what the queue should still hold at its lowest
constexpr double MARGIN_START = FramesIn(16.0);
constexpr double MARGIN_LEAST = FramesIn(10.0);
constexpr double MARGIN_MOST = FramesIn(50.0);
/// Less than this left counts as having come close to running out
constexpr double CLOSE_CALL = FramesIn(6.0);
constexpr double CLOSE_CALL_RAISE = FramesIn(2.0);
/// How often the margin is raised for close calls at the most, and reviewed for lowering
constexpr double RAISE_INTERVAL = FramesIn(500.0);
constexpr double REVIEW_INTERVAL = FramesIn(1000.0);
constexpr double REVIEW_LOWER = FramesIn(1.0);
/// The margin is lowered while the queue never held less than this much over the least margin
constexpr double REVIEW_SLACK = FramesIn(2.0);
/// How long the margin is left alone after it was raised
constexpr double RAISE_HOLD = FramesIn(10000.0);
/// A pause longer than this is not something to keep a margin for
constexpr double LONGEST_COVERED_PAUSE = FramesIn(60.0);
/// What the margin is raised by over the length of a pause it did not cover
constexpr double PAUSE_RAISE = FramesIn(5.0);
/// New sound counts as having arrived once the queue grew by this much
constexpr double ARRIVAL = FramesIn(2.0);
/// After the sound ran out it starts again with this much of the margin at the most, so as not
/// to add to the silence. The rest is made up by playing a little slower for a while.
constexpr double RESTART_MARGIN = FramesIn(12.0);

// The reading speed
constexpr double RATE_LEAST = 0.5;
constexpr double RATE_MOST = 1.05;
/// Change of speed for each frame the lowest level is off the margin
constexpr double LEVEL_GAIN = 0.001 / FramesIn(1.0);
/// With this much more than the margin in the queue the speed in use is plainly too low: it
/// is then raised by this much more for each further frame
constexpr double PLAINLY_TOO_MUCH = FramesIn(30.0);
constexpr double TOO_MUCH_GAIN = 0.004 / FramesIn(1.0);
/// Most the speed is raised to shorten the queue
constexpr double MOST_SPEEDUP = 0.6;
/// With this much more than the margin in it at its lowest, the queue is cut short rather than
/// played down, which would take seconds of sound that is audibly too fast. So it is when it
/// is about to overflow: it can hold a quarter of a second.
constexpr double FAR_TOO_MUCH = FramesIn(70.0);
constexpr double NEARLY_FULL = FramesIn(190.0);
constexpr double CUT_DOWN_TO = FramesIn(120.0);
/// How much slower the queue is read as it comes close to empty, on top of the above, and
/// what it holds when that begins: a share of the margin, and no more than a few milliseconds
constexpr double EMPTY_SLOWDOWN = 0.15;
constexpr double EMPTY_FROM = 0.6;
constexpr double EMPTY_FROM_MOST = FramesIn(8.0);
/// Requests of the device over which the queue is watched for coming close to empty: long
/// enough to span the bursts sound comes in, short enough to let go when the danger is over
constexpr std::size_t EMPTY_WINDOW = 8;
/// Slow correction of what the measured speed is off by, and its bounds
constexpr double TRIM_GAIN = 5e-7 / (FramesIn(1.0) * FramesIn(1.0));
constexpr double TRIM_MOST = 0.05;
/// Most the speed changes from one request of the device to the next, so that the pitch
/// glides rather than steps
constexpr double MOST_RATE_RISE = 0.02;
constexpr double MOST_RATE_FALL = 0.03;
/// Time over which the measured speed is followed: slowly while it is about what is in use,
/// as each measurement is off by half a percent or so, and quickly once it is clearly another
constexpr double SPEED_SMOOTHING_SLOW = FramesIn(3000.0);
constexpr double SPEED_SMOOTHING_FAST = FramesIn(300.0);
constexpr double CLEARLY_ANOTHER_SPEED = 0.03;

/// Frames over which the sound is faded out and in
constexpr std::size_t FADE_FRAMES = 96;

// Resampling: a windowed sinc, tabulated at this many places between two input frames
constexpr std::size_t TAPS = 16;
constexpr std::size_t PHASES = 256;
/// The share of the spectrum let through. What lies above would fold back when sound is read
/// faster than it was made.
constexpr double PASSBAND = 0.90;

struct FilterBank {
    std::array<std::array<float, TAPS>, PHASES + 1> taps;

    FilterBank() {
        const auto bessel = [](double x) {
            double sum = 1.0;
            double term = 1.0;
            for (int k = 1; k < 32; k++) {
                term *= (x / (2.0 * k)) * (x / (2.0 * k));
                sum += term;
            }
            return sum;
        };
        constexpr double BETA = 8.0;
        constexpr double HALF = TAPS / 2;
        for (std::size_t phase = 0; phase <= PHASES; phase++) {
            const double fraction = static_cast<double>(phase) / PHASES;
            std::array<double, TAPS> row;
            double total = 0.0;
            for (std::size_t tap = 0; tap < TAPS; tap++) {
                // Distance of this input frame from the output frame
                const double x = static_cast<double>(tap) - (HALF - 1.0) - fraction;
                const double sinc = std::abs(x) < 1e-9 ? PASSBAND
                                                       : std::sin(std::numbers::pi * PASSBAND * x) /
                                                             (std::numbers::pi * x);
                const double reach = x / HALF;
                const double window =
                    std::abs(reach) < 1.0
                        ? bessel(BETA * std::sqrt(1.0 - reach * reach)) / bessel(BETA)
                        : 0.0;
                row[tap] = sinc * window;
                total += row[tap];
            }
            for (std::size_t tap = 0; tap < TAPS; tap++) {
                taps[phase][tap] = static_cast<float>(row[tap] / total);
            }
        }
    }
};

const FilterBank& Filter() {
    static const FilterBank bank;
    return bank;
}

} // Anonymous namespace

OutputPacer::OutputPacer() : margin{MARGIN_START} {
    static_assert(FILTER_TAPS == TAPS);
    // Built here rather than on the audio thread
    Filter();
}

bool OutputPacer::TakeInputFrame() {
    if (input_next == input_end) {
        return false;
    }
    std::copy(history.begin() + 1, history.end(), history.begin());
    history.back() = {static_cast<float>(input[2 * input_next]),
                      static_cast<float>(input[2 * input_next + 1])};
    input_next++;
    return true;
}

void OutputPacer::UpdateRate(std::size_t waiting, std::size_t num_frames, double measured) {
    // The speed sound is produced at, followed smoothly
    const double difference = (measured - speed) / CLEARLY_ANOTHER_SPEED;
    const double clearly = std::min(1.0, difference * difference);
    const double smoothing =
        SPEED_SMOOTHING_SLOW + clearly * (SPEED_SMOOTHING_FAST - SPEED_SMOOTHING_SLOW);
    speed += (measured - speed) * std::min(1.0, static_cast<double>(num_frames) / smoothing);

    // The lowest the queue has been, counting what this request will leave of it: over a
    // while, which is what the margin is held against, and just now
    const double taken = static_cast<double>(num_frames) * rate;
    const double after = std::max(0.0, static_cast<double>(waiting) - taken);
    double lowest = after;
    for (const std::size_t left : recent_left) {
        lowest = std::min(lowest, static_cast<double>(left));
    }
    double lowest_now = after;
    for (std::size_t back = 1; back <= EMPTY_WINDOW; back++) {
        const std::size_t index = (recent_next + recent_left.size() - back) % recent_left.size();
        lowest_now = std::min(lowest_now, static_cast<double>(recent_left[index]));
    }
    const double off = lowest - margin;
    lowest_lately = lowest;

    if (std::abs(off) < PLAINLY_TOO_MUCH) {
        trim = std::clamp(trim + TRIM_GAIN * off * static_cast<double>(num_frames), -TRIM_MOST,
                          TRIM_MOST);
    }
    double correction = LEVEL_GAIN * off;
    if (off > PLAINLY_TOO_MUCH) {
        correction += TOO_MUCH_GAIN * (off - PLAINLY_TOO_MUCH);
    }
    correction = std::min(correction, MOST_SPEEDUP) + trim;
    // Ever slower as the queue empties, to make what is left last
    const double safe = std::min(EMPTY_FROM * margin, EMPTY_FROM_MOST);
    if (lowest_now < safe) {
        const double emptiness = (safe - lowest_now) / safe;
        correction -= EMPTY_SLOWDOWN * emptiness * emptiness;
    }
    rate_target = std::clamp(speed * (1.0 + correction), RATE_LEAST, RATE_MOST);
    rate_target = std::clamp(rate_target, rate - MOST_RATE_FALL, rate + MOST_RATE_RISE);

    CITRA_TRACE_SET_COUNTER("azahar.audio_rate_e5", static_cast<s64>(rate_target * 1e5));
    CITRA_TRACE_SET_COUNTER("azahar.audio_produced_e5", static_cast<s64>(speed * 1e5));
    CITRA_TRACE_SET_COUNTER("azahar.audio_lowest_us",
                            static_cast<s64>(lowest * 1e6 / native_sample_rate));
    CITRA_TRACE_SET_COUNTER("azahar.audio_margin_us",
                            static_cast<s64>(margin * 1e6 / native_sample_rate));
}

void OutputPacer::UpdateMargin(std::size_t left, std::size_t num_frames) {
    recent_left[recent_next] = left;
    recent_next = (recent_next + 1) % recent_left.size();

    frames_since_raise += num_frames;
    if (state == State::Playing && static_cast<double>(left) < CLOSE_CALL &&
        static_cast<double>(frames_since_raise) >= RAISE_INTERVAL) {
        margin = std::min(margin + CLOSE_CALL_RAISE, MARGIN_MOST);
        frames_since_raise = 0;
    }

    // Lowered again, slowly, once the queue has not been close to empty for a while
    lowest_since_review = std::min(lowest_since_review, left);
    frames_since_review += num_frames;
    if (static_cast<double>(frames_since_review) >= REVIEW_INTERVAL) {
        if (static_cast<double>(frames_since_raise) >= RAISE_HOLD &&
            static_cast<double>(lowest_since_review) > MARGIN_LEAST + REVIEW_SLACK) {
            margin = std::max(margin - REVIEW_LOWER, MARGIN_LEAST);
        }
        frames_since_review = 0;
        lowest_since_review = std::numeric_limits<std::size_t>::max();
    }
}

OutputPacer::Result OutputPacer::Render(OutputFifo& fifo, s16* buffer, std::size_t num_frames,
                                        double production_speed) {
    std::size_t waiting = fifo.Size() + (input_end - input_next);
    const double restart = std::min(margin, RESTART_MARGIN) + static_cast<double>(num_frames);

    if (state == State::Playing &&
        (lowest_lately > margin + FAR_TOO_MUCH || static_cast<double>(waiting) > NEARLY_FULL)) {
        // Far more sound than there should be: the emulation caught up after a slow stretch,
        // or runs unthrottled. Fade out, leave out what is too much, fade back in.
        state = State::Fading;
        skip_ahead = true;
    }

    if (state != State::Playing && gain == 0.0f) {
        // Silent. Sound starts again when there is enough of it to keep going.
        if (skip_ahead) {
            skip_ahead = false;
            // What the queue holds at its lowest over the margin is too much. The queue swings
            // by a video frame's worth of sound around that, which is to stay.
            double too_much = std::max(0.0, lowest_lately - margin);
            if (static_cast<double>(waiting) - too_much > CUT_DOWN_TO) {
                too_much = static_cast<double>(waiting) - CUT_DOWN_TO;
            }
            std::size_t to_discard = static_cast<std::size_t>(too_much);
            input_next = input_end = 0;
            input.resize(std::max<std::size_t>(input.size(), 2 * 256));
            while (to_discard != 0) {
                const std::size_t popped =
                    fifo.Pop(input.data(), std::min<std::size_t>(to_discard, 256));
                if (popped == 0) {
                    break;
                }
                to_discard -= popped;
            }
            lowest_lately = margin;
            waiting = fifo.Size();
            state = State::Waiting;
            waiting_when_silent = waiting;
            sound_arrived = true;
            frames_waiting = 0;
        }
        if (state != State::Waiting) {
            state = State::Waiting;
            waiting_when_silent = waiting;
            sound_arrived = false;
        }
        if (static_cast<double>(waiting) < restart) {
            // How long the sound stayed away, up to when more of it came
            sound_arrived |=
                static_cast<double>(waiting) > static_cast<double>(waiting_when_silent) + ARRIVAL;
            if (!sound_arrived) {
                frames_waiting += num_frames;
            }
            std::memset(buffer, 0, num_frames * 2 * sizeof(s16));
            UpdateMargin(waiting, num_frames);
            return {num_frames, rate, static_cast<std::size_t>(margin), waiting};
        }
        // The emulation paused for longer than the margin. A short pause is worth covering the
        // next time; a long one, or the silence before the first sound, is not.
        const double paused = static_cast<double>(frames_waiting);
        if (started && paused > 0.0 && paused <= LONGEST_COVERED_PAUSE) {
            margin = std::min(margin + paused + PAUSE_RAISE, MARGIN_MOST);
            frames_since_raise = 0;
        }
        started = true;
        frames_waiting = 0;
        state = State::Playing;
        rate = rate_target = std::clamp(speed, RATE_LEAST, RATE_MOST);
        recent_left.fill(waiting);
    } else if (state == State::Fading && !skip_ahead && static_cast<double>(waiting) >= restart) {
        // More came before the sound was gone
        state = State::Playing;
    }

    UpdateRate(waiting, num_frames, production_speed);

    // Bring in what this request can take at the most
    const std::size_t pending = input_end - input_next;
    const std::size_t wanted =
        static_cast<std::size_t>(std::ceil(std::max(rate, rate_target) * num_frames)) + 2;
    if (pending != 0 && input_next != 0) {
        std::memmove(input.data(), input.data() + 2 * input_next, pending * 2 * sizeof(s16));
    }
    input_next = 0;
    input_end = pending;
    if (wanted > pending) {
        input.resize(std::max(input.size(), 2 * wanted));
        input_end += fifo.Pop(input.data() + 2 * pending, wanted - pending);
    }
    const std::size_t left_in_fifo = fifo.Size();

    const FilterBank& filter = Filter();
    const double rate_step = (rate_target - rate) / static_cast<double>(num_frames);
    const float gain_step = 1.0f / FADE_FRAMES;
    std::size_t silent_frames = 0;
    for (std::size_t i = 0; i < num_frames; i++) {
        while (position >= 1.0) {
            if (!TakeInputFrame()) {
                // Nothing left: the last of the sound is held while it fades
                position = 1.0;
                state = State::Fading;
                break;
            }
            position -= 1.0;
        }
        // Fade out over the last of what there is, rather than stop on it
        const std::size_t remaining = (input_end - input_next) + left_in_fifo;
        if (state == State::Playing && remaining < FADE_FRAMES + TAPS) {
            state = State::Fading;
        }
        gain = state == State::Playing ? std::min(1.0f, gain + gain_step)
                                       : std::max(0.0f, gain - gain_step);
        if (gain == 0.0f) {
            buffer[2 * i] = 0;
            buffer[2 * i + 1] = 0;
            silent_frames++;
            continue;
        }

        const double place = std::min(position, 1.0) * PHASES;
        const std::size_t phase = std::min(static_cast<std::size_t>(place), PHASES - 1);
        const float between = static_cast<float>(place - static_cast<double>(phase));
        const auto& near = filter.taps[phase];
        const auto& far = filter.taps[phase + 1];
        float left = 0.0f;
        float right = 0.0f;
        for (std::size_t tap = 0; tap < TAPS; tap++) {
            const float weight = near[tap] + between * (far[tap] - near[tap]);
            left += weight * history[tap][0];
            right += weight * history[tap][1];
        }
        buffer[2 * i] = static_cast<s16>(std::clamp(left * gain, -32768.0f, 32767.0f));
        buffer[2 * i + 1] = static_cast<s16>(std::clamp(right * gain, -32768.0f, 32767.0f));

        if (state == State::Playing || input_next != input_end) {
            position += rate;
        }
        rate += rate_step;
    }
    rate = rate_target;
    if (state == State::Fading && gain == 0.0f) {
        frames_waiting += silent_frames;
    }

    UpdateMargin(left_in_fifo + (input_end - input_next), num_frames);
    return {silent_frames, rate, static_cast<std::size_t>(margin), waiting};
}

} // namespace AudioCore
