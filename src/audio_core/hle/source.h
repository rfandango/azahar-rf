// Copyright 2016-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <utility>
#include <vector>
#include <boost/serialization/array.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/serialization/priority_queue.hpp>
#include <boost/serialization/vector.hpp>
#include <queue>
#include "audio_core/audio_types.h"
#include "audio_core/codec.h"
#include "audio_core/hle/common.h"
#include "audio_core/hle/filter.h"
#include "audio_core/interpolate.h"
#include "common/common_types.h"

namespace Memory {
class MemorySystem;
}

namespace AudioCore::HLE {

/**
 * This module performs:
 * - Buffer management
 * - Decoding of buffers
 * - Buffer resampling and interpolation
 * - Per-source filtering (SimpleFilter, BiquadFilter)
 * - Per-source gain
 * - Other per-source processing
 */
class Source final {
public:
    explicit Source(std::size_t source_id_) : source_id(source_id_) {
        Reset();
    }

    /// Resets internal state.
    void Reset();

    void Sleep();
    void Wakeup();

    /// Sets the memory system to read data from
    void SetMemory(Memory::MemorySystem& memory);

    /**
     * This is called once every audio frame. This performs per-source processing every frame.
     * @param config The new configuration we've got for this Source from the application.
     * @param adpcm_coeffs ADPCM coefficients to use if config tells us to use them (may contain
     * invalid values otherwise).
     * @return The current status of this Source. This is given back to the emulated application via
     * SharedMemory.
     */
    SourceStatus::Status Tick(SourceConfiguration::Configuration& config,
                              const s16_le (&adpcm_coeffs)[16]);

    /**
     * Generates a frame the application has not been asked for yet, from what it has handed
     * over so far: the emulation is held up and the device would go without sound otherwise.
     *
     * The application could not hand over more in the meantime. So a source that would use up
     * the last of what it has stays where it is and is silent for the frame, rather than be
     * reported as having run out, and it keeps waiting for a few frames once the emulation
     * runs again, which is when a stream gets its next block. A sound that was at its end
     * anyway ends that little later.
     */
    void TickAhead();

    /// As Tick, without a frame: the configuration is taken and the status given, and the
    /// source stays where it is. For a frame that is left out, see
    /// DspInterface::TakeSpareFrame.
    SourceStatus::Status TickWithoutFrame(SourceConfiguration::Configuration& config,
                                          const s16_le (&adpcm_coeffs)[16]) {
        ParseConfig(config, adpcm_coeffs);
        return GetCurrentStatus();
    }

    /// Whether the source waited in the last frame for more from the application
    bool IsWaiting() const {
        return state.enabled && waiting;
    }

    /// Whether the source is playing
    bool IsEnabled() const {
        return state.enabled;
    }

    /// Frames the source can still fill from what it has, when nothing more is queued for it.
    /// Negative when more is queued, or it is not playing.
    s32 FramesLeft() const {
        if (!state.enabled || !state.input_queue.empty()) {
            return -1;
        }
        const float per_frame = state.rate_multiplier * static_cast<float>(samples_per_frame);
        return static_cast<s32>(static_cast<float>(state.current_buffer.size()) / per_frame);
    }

    /// What became of a wait
    enum class WaitOutcome : u8 {
        None,
        WentOn, ///< The application gave it more to play
        Ended,  ///< Nothing came: it played what it had left and ran out
    };
    /// The outcome of a wait that ended since this was last asked
    WaitOutcome TakeWaitOutcome() {
        return std::exchange(wait_outcome, WaitOutcome::None);
    }

    /**
     * Mix this source's output into dest, using the gains for the `intermediate_mix_id`-th
     * intermediate mixer.
     * @param dest The QuadFrame32 to mix into.
     * @param intermediate_mix_id The id of the intermediate mix whose gains we are using.
     */
    void MixInto(QuadFrame32& dest, std::size_t intermediate_mix_id);

private:
    const std::size_t source_id;
    Memory::MemorySystem* memory_system{};
    StereoFrame16 current_frame;
    StereoFrame16 backup_frame; // TODO(PabloMK7): Check if we actually need this

    using Format = SourceConfiguration::Configuration::Format;
    using InterpolationMode = SourceConfiguration::Configuration::InterpolationMode;
    using MonoOrStereo = SourceConfiguration::Configuration::MonoOrStereo;

    /// Internal representation of a buffer for our buffer queue
    struct Buffer {
        PAddr physical_address;
        u32 length;
        u8 adpcm_ps;
        std::array<u16, 2> adpcm_yn;
        bool adpcm_dirty;
        bool is_looping;
        u16 buffer_id;

        MonoOrStereo mono_or_stereo;
        Format format;

        bool from_queue;
        u32 play_position; // = 0;
        bool has_played;   // = false;

    private:
        template <class Archive>
        void serialize(Archive& ar, const unsigned int) {
            ar & physical_address;
            ar & length;
            ar & adpcm_ps;
            ar & adpcm_yn;
            ar & adpcm_dirty;
            ar & is_looping;
            ar & buffer_id;
            ar & mono_or_stereo;
            ar & format;
            ar & from_queue;
            ar & play_position;
            ar & has_played;
        }
        friend class boost::serialization::access;
    };

    struct BufferOrder {
        bool operator()(const Buffer& a, const Buffer& b) const {
            // Lower buffer_id comes first.
            return a.buffer_id > b.buffer_id;
        }
    };

    struct SourceState {

        // State variables

        bool enabled = false;
        u16 sync_count = 0;

        // Mixing

        std::array<std::array<float, 4>, 3> gain = {};
        std::array<std::array<float, 4>, 3> gain_ramp_start = {};
        std::array<bool, 3> gain_ramp_active = {};

        // Buffer queue

        std::priority_queue<Buffer, std::vector<Buffer>, BufferOrder> input_queue = {};
        MonoOrStereo mono_or_stereo = MonoOrStereo::Mono;
        Format format = Format::ADPCM;

        // Current buffer

        u32 current_sample_number = 0;
        PAddr current_buffer_physical_address = 0;
        u32 current_buffer_length = 0;
        MonoOrStereo current_buffer_mono_or_stereo = MonoOrStereo::Mono;
        Format current_buffer_format = Format::ADPCM;
        bool current_buffer_is_looping = false;
        AudioInterp::StereoBuffer16 current_buffer = {};

        // buffer_id state

        bool buffer_update = false;
        u16 last_buffer_id = 0;
        u16 current_buffer_id = 0;

        // Decoding state

        std::array<s16, 16> adpcm_coeffs = {};
        Codec::ADPCMState adpcm_state = {};

        // Resampling state

        float rate_multiplier = 1.0;
        InterpolationMode interpolation_mode = InterpolationMode::Polyphase;
        AudioInterp::State interp_state = {};

        // Filter state

        SourceFilters filters = {};

    private:
        template <class Archive>
        void serialize(Archive& ar, const unsigned int) {
            ar & enabled;
            ar & sync_count;
            ar & gain;
            ar & gain_ramp_start;
            ar & gain_ramp_active;
            ar & input_queue;
            ar & mono_or_stereo;
            ar & format;
            ar & current_sample_number;
            ar & current_buffer_physical_address;
            ar & current_buffer_length;
            ar & current_buffer_mono_or_stereo;
            ar & current_buffer_format;
            ar & current_buffer_is_looping;
            ar & current_buffer;
            ar & buffer_update;
            ar & current_buffer_id;
            ar & adpcm_coeffs;
            ar & rate_multiplier;
            ar & interpolation_mode;
        }
        friend class boost::serialization::access;
    };

    SourceState state;
    SourceState backup_state;

    /// Frames for which the source still waits for more from the application rather than use
    /// up the last of what it has, see TickAhead. Not part of a save state: it is over within
    /// a few frames.
    u32 frames_to_wait = 0;
    /// Whether the last frame was one it waited in
    bool waiting = false;
    WaitOutcome wait_outcome = WaitOutcome::None;

    // Internal functions

    /// INTERNAL: Whether generating a frame now would use up all there is to play.
    bool WouldRunOut() const;
    /// INTERNAL: Makes the current frame one in which the source waits: silence, reached
    /// from where the last frame ended.
    void WaitFrame();

    /// INTERNAL: Update our internal state based on the current config.
    void ParseConfig(SourceConfiguration::Configuration& config, const s16_le (&adpcm_coeffs)[16]);
    /// INTERNAL: Generate the current audio output for this frame based on our internal state.
    void GenerateFrame();
    /// INTERNAL: Dequeues a buffer and does preprocessing on it (decoding, resampling). Puts it
    /// into current_buffer.
    bool DequeueBuffer();
    /// INTERNAL: Generates a SourceStatus::Status based on our internal state.
    SourceStatus::Status GetCurrentStatus();

    template <class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar & state;
        ar & backup_state;
        ar & current_frame;
        ar & backup_frame;
    }
    friend class boost::serialization::access;
};

} // namespace AudioCore::HLE
