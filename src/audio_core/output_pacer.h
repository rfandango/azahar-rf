// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <vector>
#include "audio_core/audio_types.h"
#include "common/common_types.h"
#include "common/ring_buffer.h"

namespace AudioCore {

/// The queue between the emulation, which produces sound, and the device, which plays it
using OutputFifo = Common::RingBuffer<s16, 0x2000, 2>;

/**
 * Plays the sound the emulation produces at the rate it is produced.
 *
 * The device asks for sound at a steady rate. The emulation produces it in bursts, a video
 * frame's worth at a time, at a rate that is only nominally the device's: it runs a fraction
 * of a percent off to stay in step with the display, a few percent slow in a heavy scene, and
 * not at all for the length of a hitch. Whenever the queue between the two is empty the device
 * gets nothing to play, which is heard as a crack.
 *
 * Three things keep that from happening:
 * - The queue is read at the speed it is filled, which is the speed the emulation runs at. The
 *   reading speed is nudged so that the least the queue holds between two bursts stays at a
 *   margin. Reading slower or faster changes the pitch by as much, which goes unnoticed for
 *   small changes and is the lesser evil for large ones.
 * - The margin follows how unevenly the emulation runs. It grows when the queue came close to
 *   empty or ran out, and shrinks slowly while it never does, so that a game that runs evenly
 *   is not delayed for the sake of one that does not.
 * - When the queue does run out, the sound is faded out over its last milliseconds and faded
 *   back in once there is a margin again, instead of stopping and starting in mid-wave.
 */
class OutputPacer {
public:
    OutputPacer();

    struct Result {
        std::size_t silent_frames; ///< Frames written as silence because there was no sound
        double rate;               ///< Speed the queue was read at, relative to the nominal one
        std::size_t margin;        ///< Frames aimed for in the queue at its lowest
        std::size_t waiting;       ///< Frames that were waiting, in the queue and here
        std::size_t ahead_frames;  ///< Frames that had to be asked for, see Refill
        std::size_t spare_frames;  ///< Frames there are too many of, for the source of the
                                   ///< sound to leave out as many of what it makes next
    };

    /**
     * A way to have another frame of sound queued at once, for when the queue would run out
     * before the emulation fills it: see DspInterface::ProduceFrameAhead. `queue_frame` returns
     * whether a frame was queued.
     *
     * With it the speed is held. Without, the queue has to be read at the speed it is filled
     * at, which is heard as the pitch giving way whenever the emulation runs a little slow, and
     * a reserve has to be rebuilt after every hold-up by playing slower for a while. With it
     * the sound is read at the speed the emulation is meant to run at (`production_speed` is
     * then that, and within a fraction of a percent of one), what is missing is asked for, and
     * what is too much is reported back for the source to leave out.
     */
    struct Refill {
        bool (*queue_frame)(void* context);
        void* context;

        explicit operator bool() const {
            return queue_frame != nullptr;
        }
        bool operator()() const {
            return queue_frame(context);
        }
    };

    /// Fills `buffer` with `num_frames` stereo frames from the queue, which is being filled at
    /// `production_speed` times the nominal rate. Audio thread.
    Result Render(OutputFifo& fifo, s16* buffer, std::size_t num_frames, double production_speed,
                  Refill refill = Refill{nullptr, nullptr});

private:
    enum class State {
        Waiting, ///< Silent until the queue holds a margin again
        Playing,
        Fading, ///< Running out: playing what is left ever more quietly
    };

    static constexpr std::size_t FILTER_TAPS = 16;
    /// Requests of the device over which the lowest level of the queue is taken
    static constexpr std::size_t LOW_WINDOW = 64;

    /// Takes the next frame of the queue into the resampling filter. False if there is none.
    bool TakeInputFrame();
    void UpdateRate(std::size_t waiting, std::size_t num_frames, double production_speed);
    void UpdateMargin(std::size_t left, std::size_t num_frames);

    State state = State::Waiting;
    /// Frames popped from the queue that the filter has not taken in yet
    std::vector<s16> input;
    std::size_t input_next = 0;
    std::size_t input_end = 0;
    /// The frames the filter spans, oldest first, and the place of the next output frame
    /// between the middle two
    std::array<std::array<float, 2>, FILTER_TAPS> history{};
    double position = 0.0;

    double rate = 1.0;
    double rate_target = 1.0;
    double speed = 1.0;
    double trim = 0.0;
    float gain = 0.0f;

    /// The margin, in frames
    double margin;
    /// What was left after each of the last times the device asked, to find the lowest
    std::array<std::size_t, LOW_WINDOW> recent_left{};
    std::size_t recent_next = 0;
    std::size_t frames_since_raise = 0;
    std::size_t frames_since_review = 0;
    std::size_t lowest_since_review = 0;
    /// While silent: for how long no sound came, and what the queue held when it ran out
    std::size_t frames_waiting = 0;
    std::size_t waiting_when_silent = 0;
    bool sound_arrived = false;
    /// Whether any sound was played yet
    bool started = false;
    /// The queue grew too long: what is too much of it is left out once the sound has faded
    bool skip_ahead = false;
    /// The least the queue held lately, as the speed was last set by
    double lowest_lately = 0.0;
    /// Whether frames can be asked for, and the speed is held: see Refill
    bool steady = false;
};

} // namespace AudioCore
