// Copyright 2017-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>
#include <boost/serialization/access.hpp>
#include "audio_core/audio_types.h"
#include "audio_core/output_pacer.h"
#include "audio_core/time_stretch.h"
#include "common/common_types.h"
#include "common/ring_buffer.h"
#include "core/memory.h"

namespace Core {
class System;
} // namespace Core

namespace Service::DSP {
enum class InterruptType : u32;
} // namespace Service::DSP

namespace AudioCore {

class Sink;
enum class SinkType : u32;

class DspInterface {
public:
    DspInterface(Core::System& system_);
    virtual ~DspInterface();

    DspInterface(const DspInterface&) = delete;
    DspInterface(DspInterface&&) = delete;
    DspInterface& operator=(const DspInterface&) = delete;
    DspInterface& operator=(DspInterface&&) = delete;

    /**
     * Reads data from one of three DSP registers
     * @note this function blocks until the data is available
     * @param register_number the index of the register to read
     * @returns the value of the register
     */
    virtual u16 RecvData(u32 register_number) = 0;

    /**
     * Checks whether data is ready in one of three DSP registers
     * @param register_number the index of the register to check
     * @returns true if data is ready
     */
    virtual bool RecvDataIsReady(u32 register_number) const = 0;

    /**
     * Sets the DSP semaphore register
     * @param semaphore_value the value set to the semaphore register
     */
    virtual void SetSemaphore(u16 semaphore_value) = 0;

    /**
     * Reads `length` bytes from the DSP pipe identified with `pipe_number`.
     * @note Can read up to the maximum value of a u16 in bytes (65,535).
     * @note IF an error is encoutered with either an invalid `pipe_number` or `length` value, an
     * empty vector will be returned.
     * @note IF `length` is set to 0, an empty vector will be returned.
     * @note IF `length` is greater than the amount of data available, this function will only read
     * the available amount.
     * @param pipe_number a `DspPipe`
     * @param length the number of bytes to read. The max is 65,535 (max of u16).
     * @returns a vector of bytes from the specified pipe. On error, will be empty.
     */
    virtual std::vector<u8> PipeRead(DspPipe pipe_number, std::size_t length) = 0;

    /**
     * How much data is left in pipe
     * @param pipe_number The Pipe ID
     * @return The amount of data remaning in the pipe. This is the maximum length PipeRead will
     * return.
     */
    virtual std::size_t GetPipeReadableSize(DspPipe pipe_number) const = 0;

    /**
     * Write to a DSP pipe.
     * @param pipe_number The Pipe ID
     * @param buffer The data to write to the pipe.
     */
    virtual void PipeWrite(DspPipe pipe_number, std::span<const u8> buffer) = 0;

    /// Sets the handler for the interrupts we trigger
    virtual void SetInterruptHandler(
        std::function<void(Service::DSP::InterruptType type, DspPipe pipe)> handler) = 0;

    /// Loads the DSP program
    virtual void LoadComponent(std::span<const u8> buffer) = 0;

    /// Unloads the DSP program
    virtual void UnloadComponent() = 0;

    /// Select the sink to use based on sink type.
    void SetSink(SinkType sink_type, std::string_view audio_device);
    /// Get the current sink
    Sink& GetSink();
    /// Enable/Disable audio stretching.
    void EnableStretching(bool enable);

    /**
     * Adds one frame of CSND output. On hardware CSND and the DSP reach the codec chip on separate
     * I2S lines and the codec mixes them, here the frame is mixed to the DSP output.
     * Must be called once per audio frame period.
     */
    void OutputCsndFrame(const StereoFrame16& frame);

protected:
    /// Queues sound for the device. The form with Locked in its name is for a caller that
    /// holds production_mutex; `ahead` says the frame is one mixed ahead of the emulation, on
    /// the audio thread.
    void OutputFrame(StereoFrame16 frame);
    void OutputFrameLocked(StereoFrame16 frame, bool ahead = false);
    void OutputSample(std::array<s16, 2> sample);

    /**
     * Mixes one more frame from what the application has handed over so far and queues it, ahead
     * of the emulation.
     *
     * Sound is mixed as emulated time passes, so an emulation that is held up, by a pipeline
     * that has to be built or by anything else the host makes it wait for, leaves the device
     * without sound for as long as that takes. The real DSP is a processor of its own that goes
     * on playing what it was given while the application is busy, and this is what lets ours
     * do the same.
     *
     * Called on the audio thread with production_mutex held. Returns false when there is
     * nothing to mix from, or when this has gone on for too long to be a hold-up.
     */
    virtual bool ProduceFrameAhead() {
        return false;
    }

    /**
     * Whether the frame that is due should be left out: the device has more sound waiting than
     * it needs, which happens when the emulation makes up for a hold-up that frames mixed ahead
     * already covered, or when it makes more sound than time passes. Takes one frame off what
     * the output stage reported as too much. A source that can mix ahead leaves the frame out
     * rather than have the output stage play faster or cut. Emulation thread.
     */
    bool TakeSpareFrame();

    /// Guards what produces sound: the state frames are mixed from and the writing end of the
    /// queue to the device. Held by the emulation thread while it mixes or queues, and by the
    /// audio thread while it mixes a frame ahead of the emulation.
    std::mutex production_mutex;

private:
    void FlushResidualStretcherAudio();
    void OutputCallback(s16* buffer, std::size_t num_frames);
    /// For the output stage, when the queue is about to run out. Audio thread.
    bool QueueFrameAhead();
    /// Returns the number of frames it threw away
    std::size_t DrainFifoBacklog(std::size_t num_frames);
    std::size_t PopResampled(s16* buffer, std::size_t num_frames, double ratio);
    void MixCsnd(std::span<std::array<s16, 2>> samples);
    void PushSamples(std::span<const std::array<s16, 2>> samples, bool ahead = false);

    Core::System& system;

    std::atomic<bool> enable_time_stretching = false;
    std::atomic<bool> performing_time_stretching = false;
    std::atomic<bool> flushing_time_stretcher = false;
    OutputFifo fifo;
    OutputPacer output_pacer;
    std::array<s16, 2> last_frame{};
    std::size_t fifo_low_fill = std::numeric_limits<std::size_t>::max();
    std::size_t backlog_window_frames = 0;

    /// Resampling by a ratio close to one, see PopResampled: the four frames around the
    /// position of the next output frame, and that position between the middle two
    std::array<std::array<s16, 2>, 4> resample_window{};
    double resample_position = 0.0;
    bool resampling = false;
    std::vector<s16> resample_input;
    TimeStretcher time_stretcher;
    std::unique_ptr<Sink> sink;
    /// Whether frames may be mixed ahead of the emulation, as the emulation thread last found
    std::atomic<bool> ahead_allowed = false;
    /// Frames the output stage has too many of, for the source to leave out
    std::atomic<u32> spare_frames = 0;
    /// What the sink was made for, to tell whether settings that were applied anew ask for
    /// another one
    SinkType sink_type{};
    std::string sink_device;

    std::mutex csnd_mutex;
    StereoBuffer16 csnd_pending;
    std::size_t dsp_samples_since_csnd_frame = 0;
    u32 csnd_frames_without_dsp = 0;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int) {}
    friend class boost::serialization::access;
};

} // namespace AudioCore
