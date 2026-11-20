// Copyright 2017-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>

#include <boost/serialization/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include "audio_core/audio_types.h"
#include "audio_core/hle/aac_decoder.h"
#include "audio_core/hle/common.h"
#include "audio_core/hle/decoder.h"
#include "audio_core/hle/hle.h"
#include "audio_core/hle/mixers.h"
#include "audio_core/hle/shared_memory.h"
#include "audio_core/hle/source.h"
#include "audio_core/sink.h"
#include "common/archives.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"

using InterruptType = Service::DSP::InterruptType;

namespace AudioCore {

DspHle::DspHle(Core::System& system) : DspHle(system, system.Memory(), system.CoreTiming()) {}

template <class Archive>
void DspHle::serialize(Archive& ar, const unsigned int) {
    ar& boost::serialization::base_object<DspInterface>(*this);
    // The audio thread mixes from this state too, when the emulation is late
    std::scoped_lock lock{production_mutex};
    ar&* impl.get();
}
SERIALIZE_IMPL(DspHle)

// The value below is the "perfect" mathematical ratio of ARM11 cycles per audio frame, samples per
// frame * teaklite cycles per sample * 2 ARM11 cycles/teaklite cycle
// (160 * 4096 * 2) = (1310720)
//
// This value has been verified against a rough hardware test with hardware and LLE
static constexpr u64 audio_frame_ticks = samples_per_frame * 4096 * 2ull; ///< Units: ARM11 cycles

struct DspHle::Impl final {
public:
    explicit Impl(DspHle& parent, Memory::MemorySystem& memory, Core::Timing& timing);
    ~Impl();

    DspState GetDspState() const;

    u16 RecvData(u32 register_number);
    bool RecvDataIsReady(u32 register_number) const;
    std::vector<u8> PipeRead(DspPipe pipe_number, std::size_t length);
    std::size_t GetPipeReadableSize(DspPipe pipe_number) const;
    void PipeWrite(DspPipe pipe_number, std::span<const u8> buffer);

    void SetInterruptHandler(
        std::function<void(Service::DSP::InterruptType type, DspPipe pipe)> handler);

    /// See DspInterface::ProduceFrameAhead. Audio thread, with the production mutex held.
    bool MixFrameAhead();

private:
    /// Frames the audio thread may mix between two ticks of the emulation: half a second of
    /// them. An emulation that stands still for longer than that is paused, or loading a
    /// state, and sound that goes on by itself would be out of place.
    static constexpr u32 MOST_FRAMES_AHEAD = 100;
    /// The last of them, over which the sound goes down rather than be cut off at the end, and
    /// how much of it comes back with each frame once the emulation ticks again
    static constexpr u32 FADING_FRAMES_AHEAD = 30;
    static constexpr float GAIN_BACK_PER_FRAME = 0.25f;

    /// Takes a frame from the gain the last one ended at to another
    void ApplyOutputGain(StereoFrame16& frame, float target);

    void SetState(DspState state);
    void Initialize();
    void Sleep();
    void Wakeup();
    void FinishPendingSleep();

    void ResetPipes();
    void WriteU16(DspPipe pipe_number, u16 value);
    void AudioPipeWriteStructAddresses();

    std::size_t CurrentRegionIndex() const;
    HLE::SharedMemory& ReadRegion();
    HLE::SharedMemory& WriteRegion();

    StereoFrame16 GenerateCurrentFrame();
    void TickWithoutFrame();
    bool Tick();
    void AudioTickCallback(s64 cycles_late);

    /// Written with the production mutex held: the audio thread reads it
    DspState dsp_state = DspState::Off;
    /// Frames the audio thread mixed since the emulation's last tick
    u32 frames_ahead = 0;
    /// What the last frame was multiplied by at its end: one, but around the limit of frames
    /// mixed ahead
    float output_gain = 1.0f;
    /// Since they were last reported: frames mixed ahead, those of them in which a source
    /// waited for more, and ticks
    u32 reported_ahead = 0;
    u32 reported_waits = 0;
    u32 reported_went_on = 0;
    u32 reported_ended = 0;
    u32 ticks_since_report = 0;
    /// In the frames mixed ahead since the last report: the most sources that played in one,
    /// the most of them that had nothing more queued, and the fewest frames one of those had
    /// left before the frame was mixed
    u32 reported_playing = 0;
    u32 reported_unqueued = 0;
    s32 reported_closest = -1;
    /// Frames left out since the last report, see DspInterface::TakeSpareFrame
    u32 reported_left_out = 0;
    std::array<std::vector<u8>, num_dsp_pipe> pipe_data{};

    // On hardware, the DSP firmware enters sleep at its next audio frame, only then sending the
    // pipe reply and its interrupt, and sets reply register 0 two frames later. We need to emulate
    // this timing due to races in some games such as NSMB2, where replying immediately causes a
    // fatal error when opening the home menu.
    bool sleep_pending = false;
    u32 sleep_reply_frames = 0;

public:
    HLE::DspMemory* dsp_memory;

private:
    std::array<HLE::Source, HLE::num_sources> sources{{
        HLE::Source(0),  HLE::Source(1),  HLE::Source(2),  HLE::Source(3),  HLE::Source(4),
        HLE::Source(5),  HLE::Source(6),  HLE::Source(7),  HLE::Source(8),  HLE::Source(9),
        HLE::Source(10), HLE::Source(11), HLE::Source(12), HLE::Source(13), HLE::Source(14),
        HLE::Source(15), HLE::Source(16), HLE::Source(17), HLE::Source(18), HLE::Source(19),
        HLE::Source(20), HLE::Source(21), HLE::Source(22), HLE::Source(23),
    }};
    HLE::Mixers mixers{};

    HLE::DspMemory backup_dsp_memory;

    DspHle& parent;
    Core::Timing& core_timing;
    Core::TimingEventType* tick_event{};

    std::unique_ptr<HLE::DecoderBase> aac_decoder{};

    std::function<void(Service::DSP::InterruptType type, DspPipe pipe)> interrupt_handler{};

    template <class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar& boost::serialization::make_binary_object(backup_dsp_memory.raw_memory.data(),
                                                     backup_dsp_memory.raw_memory.size());
        ar & dsp_state;
        ar & pipe_data;
        ar & sleep_pending;
        ar & sleep_reply_frames;
        ar & sources;
        ar & mixers;
        // interrupt_handler is reregistered when loading state from DSP_DSP
    }
    friend class boost::serialization::access;
};

DspHle::Impl::Impl(DspHle& parent_, Memory::MemorySystem& memory, Core::Timing& timing)
    : parent(parent_), core_timing(timing) {
    dsp_memory = reinterpret_cast<HLE::DspMemory*>(memory.GetDspMemory(0));
    dsp_memory->raw_memory.fill(0);

    for (auto& source : sources) {
        source.SetMemory(memory);
    }

    aac_decoder = std::make_unique<HLE::AACDecoder>(memory);
    tick_event =
        core_timing.RegisterEvent("AudioCore::DspHle::tick_event", [this](u64, s64 cycles_late) {
            this->AudioTickCallback(cycles_late);
        });
    core_timing.ScheduleEvent(audio_frame_ticks, tick_event);
}

DspHle::Impl::~Impl() {
    core_timing.UnscheduleEvent(tick_event, 0);
}

DspState DspHle::Impl::GetDspState() const {
    return dsp_state;
}

u16 DspHle::Impl::RecvData(u32 register_number) {
    ASSERT_MSG(register_number == 0, "Unknown register_number {}", register_number);

    // Application reads this after requesting DSP shutdown, to verify the DSP has indeed shutdown
    // or slept. While a sleep is in progress it returns the value the reply will have.
    if (sleep_pending || sleep_reply_frames > 0) {
        return 1;
    }

    switch (GetDspState()) {
    case AudioCore::DspState::On:
        return 0;
    case AudioCore::DspState::Off:
    case AudioCore::DspState::Sleeping:
        return 1;
    default:
        UNREACHABLE();
        break;
    }
}

bool DspHle::Impl::RecvDataIsReady(u32 register_number) const {
    ASSERT_MSG(register_number == 0, "Unknown register_number {}", register_number);
    return !sleep_pending && sleep_reply_frames == 0;
}

std::vector<u8> DspHle::Impl::PipeRead(DspPipe pipe_number, std::size_t length) {
    const std::size_t pipe_index = static_cast<std::size_t>(pipe_number);

    if (pipe_index >= num_dsp_pipe) {
        LOG_ERROR(Audio_DSP, "pipe_number = {} invalid", pipe_index);
        return {};
    }

    if (length > UINT16_MAX) { // Can only read at most UINT16_MAX from the pipe
        LOG_ERROR(Audio_DSP, "length of {} greater than max of {}", length, UINT16_MAX);
        return {};
    }

    std::vector<u8>& data = pipe_data[pipe_index];

    if (length > data.size()) {
        LOG_WARNING(
            Audio_DSP,
            "pipe_number = {} is out of data, application requested read of {} but {} remain",
            pipe_index, length, data.size());
        length = static_cast<u32>(data.size());
    }

    if (length == 0)
        return {};

    std::vector<u8> ret(data.begin(), data.begin() + length);
    data.erase(data.begin(), data.begin() + length);
    return ret;
}

size_t DspHle::Impl::GetPipeReadableSize(DspPipe pipe_number) const {
    const std::size_t pipe_index = static_cast<std::size_t>(pipe_number);

    if (pipe_index >= num_dsp_pipe) {
        LOG_ERROR(Audio_DSP, "pipe_number = {} invalid", pipe_index);
        return 0;
    }

    return pipe_data[pipe_index].size();
}

void DspHle::Impl::PipeWrite(DspPipe pipe_number, std::span<const u8> buffer) {
    switch (pipe_number) {
    case DspPipe::Audio: {
        if (buffer.size() != 4) {
            LOG_ERROR(Audio_DSP, "DspPipe::Audio: Unexpected buffer length {} was written",
                      buffer.size());
            return;
        }

        enum class StateChange {
            Initialize = 0,
            Shutdown = 1,
            Wakeup = 2,
            Sleep = 3,
        };

        // Waking up from sleep garbles some of the structs in the memory region. (TODO:
        // Implement this.) Applications store away the state of these structs before
        // sleeping and reset it back after wakeup on behalf of the DSP.

        FinishPendingSleep();
        sleep_reply_frames = 0;

        switch (static_cast<StateChange>(buffer[0])) {
        case StateChange::Initialize:
            LOG_INFO(Audio_DSP, "Application has requested initialization of DSP hardware");
            Initialize();
            AudioPipeWriteStructAddresses();
            SetState(DspState::On);
            break;
        case StateChange::Shutdown:
            LOG_INFO(Audio_DSP, "Application has requested shutdown of DSP hardware");
            SetState(DspState::Off);
            break;
        case StateChange::Wakeup:
            LOG_INFO(Audio_DSP, "Application has requested wakeup of DSP hardware");
            Wakeup();
            AudioPipeWriteStructAddresses();
            SetState(DspState::On);
            break;
        case StateChange::Sleep:
            LOG_INFO(Audio_DSP, "Application has requested sleep of DSP hardware");
            sleep_pending = true;
            break;
        default:
            LOG_ERROR(Audio_DSP,
                      "Application has requested unknown state transition of DSP hardware {}",
                      buffer[0]);
            SetState(DspState::Off);
            break;
        }

        return;
    }
    case DspPipe::Binary: {
        // TODO(B3N30): Make this async, and signal the interrupt
        HLE::BinaryMessage request{};
        if (sizeof(request) != buffer.size()) {
            LOG_CRITICAL(Audio_DSP, "got binary pipe with wrong size {}", buffer.size());
            UNIMPLEMENTED();
            return;
        }
        std::memcpy(&request, buffer.data(), buffer.size());
        if (request.header.codec != HLE::DecoderCodec::DecodeAAC) {
            LOG_CRITICAL(Audio_DSP, "got unknown codec {}", static_cast<u16>(request.header.codec));
            UNIMPLEMENTED();
            return;
        }
        const HLE::BinaryMessage response = aac_decoder->ProcessRequest(request);
        pipe_data[static_cast<u32>(pipe_number)].resize(sizeof(response));
        std::memcpy(pipe_data[static_cast<u32>(pipe_number)].data(), &response, sizeof(response));

        interrupt_handler(InterruptType::Pipe, DspPipe::Binary);
        break;
    }
    default:
        LOG_CRITICAL(Audio_DSP, "pipe_number = {} unimplemented",
                     static_cast<std::size_t>(pipe_number));
        UNIMPLEMENTED();
        return;
    }
}

void DspHle::Impl::SetInterruptHandler(
    std::function<void(Service::DSP::InterruptType type, DspPipe pipe)> handler) {
    interrupt_handler = handler;
}

void DspHle::Impl::SetState(DspState state) {
    std::scoped_lock lock{parent.production_mutex};
    dsp_state = state;
}

void DspHle::Impl::Initialize() {
    // TODO(PabloMK7): This is NOT the right way to do this,
    // but it is close enough. This makes sure the DSP state
    // is clean and consistent every time the HW is initialized,
    // but what is exactly reset needs to be figured out.
    {
        std::scoped_lock lock{parent.production_mutex};
        dsp_memory->raw_memory.fill(0);
        mixers.Reset();
        for (auto& s : sources) {
            s.Reset();
        }
    }
    aac_decoder->Reset();
    ResetPipes();
}

void DspHle::Impl::Sleep() {
    // TODO(PabloMK7): This is NOT the right way to do this,
    // but it is close enough. What state is saved on
    // real hardware still not figured out.
    std::scoped_lock lock{parent.production_mutex};
    backup_dsp_memory.raw_memory = dsp_memory->raw_memory;
    mixers.Sleep();
    for (auto& s : sources) {
        s.Sleep();
    }
    // TODO(PabloMK7): Figure out if we need to save the state
    // of the AAC decoder, probably not.
}

void DspHle::Impl::FinishPendingSleep() {
    if (!sleep_pending) {
        return;
    }
    sleep_pending = false;
    Sleep();
    AudioPipeWriteStructAddresses();
    SetState(DspState::Sleeping);
}

void DspHle::Impl::Wakeup() {
    // TODO(PabloMK7): This is NOT the right way to do this,
    // but it is close enough. What state is restored on
    // real hardware still not figured out.
    {
        std::scoped_lock lock{parent.production_mutex};
        dsp_memory->raw_memory = backup_dsp_memory.raw_memory;
        backup_dsp_memory.raw_memory.fill(0);
        mixers.Wakeup();
        for (auto& s : sources) {
            s.Wakeup();
        }
    }
    aac_decoder->Reset();
    ResetPipes();
}

void DspHle::Impl::ResetPipes() {
    for (auto& data : pipe_data) {
        data.clear();
    }
}

void DspHle::Impl::WriteU16(DspPipe pipe_number, u16 value) {
    const std::size_t pipe_index = static_cast<std::size_t>(pipe_number);

    std::vector<u8>& data = pipe_data.at(pipe_index);
    // Little endian
    data.emplace_back(value & 0xFF);
    data.emplace_back(value >> 8);
}

void DspHle::Impl::AudioPipeWriteStructAddresses() {
    // These struct addresses are DSP dram addresses.
    // See also: DSP_DSP::ConvertProcessAddressFromDspDram
    static const std::array<u16, 15> struct_addresses = {
        0x8000 + offsetof(HLE::SharedMemory, frame_counter) / 2,
        0x8000 + offsetof(HLE::SharedMemory, source_configurations) / 2,
        0x8000 + offsetof(HLE::SharedMemory, source_statuses) / 2,
        0x8000 + offsetof(HLE::SharedMemory, adpcm_coefficients) / 2,
        0x8000 + offsetof(HLE::SharedMemory, dsp_configuration) / 2,
        0x8000 + offsetof(HLE::SharedMemory, dsp_status) / 2,
        0x8000 + offsetof(HLE::SharedMemory, final_samples) / 2,
        0x8000 + offsetof(HLE::SharedMemory, intermediate_mix_samples) / 2,
        0x8000 + offsetof(HLE::SharedMemory, compressor) / 2,
        0x8000 + offsetof(HLE::SharedMemory, dsp_debug) / 2,
        0x8000 + offsetof(HLE::SharedMemory, unknown10) / 2,
        0x8000 + offsetof(HLE::SharedMemory, unknown11) / 2,
        0x8000 + offsetof(HLE::SharedMemory, unknown12) / 2,
        0x8000 + offsetof(HLE::SharedMemory, unknown13) / 2,
        0x8000 + offsetof(HLE::SharedMemory, unknown14) / 2,
    };

    // Begin with a u16 denoting the number of structs.
    WriteU16(DspPipe::Audio, static_cast<u16>(struct_addresses.size()));
    // Then write the struct addresses.
    for (u16 addr : struct_addresses) {
        WriteU16(DspPipe::Audio, addr);
    }
    // Signal that we have data on this pipe.
    interrupt_handler(InterruptType::Pipe, DspPipe::Audio);
}

size_t DspHle::Impl::CurrentRegionIndex() const {
    // The region with the higher frame counter is chosen unless there is wraparound.
    // This function only returns a 0 or 1.
    const u16 frame_counter_0 = dsp_memory->region_0.frame_counter;
    const u16 frame_counter_1 = dsp_memory->region_1.frame_counter;

    if (frame_counter_0 == 0xFFFFu && frame_counter_1 != 0xFFFEu) {
        // Wraparound has occurred.
        return 1;
    }

    if (frame_counter_1 == 0xFFFFu && frame_counter_0 != 0xFFFEu) {
        // Wraparound has occurred.
        return 0;
    }

    return (frame_counter_0 > frame_counter_1) ? 0 : 1;
}

HLE::SharedMemory& DspHle::Impl::ReadRegion() {
    return CurrentRegionIndex() == 0 ? dsp_memory->region_0 : dsp_memory->region_1;
}

HLE::SharedMemory& DspHle::Impl::WriteRegion() {
    return CurrentRegionIndex() != 0 ? dsp_memory->region_0 : dsp_memory->region_1;
}

StereoFrame16 DspHle::Impl::GenerateCurrentFrame() {
    HLE::SharedMemory& read = ReadRegion();
    HLE::SharedMemory& write = WriteRegion();

    std::array<QuadFrame32, 3> intermediate_mixes = {};

    // Generate intermediate mixes
    for (std::size_t i = 0; i < HLE::num_sources; i++) {
        write.source_statuses.status[i] =
            sources[i].Tick(read.source_configurations.config[i], read.adpcm_coefficients.coeff[i]);
        for (std::size_t mix = 0; mix < 3; mix++) {
            sources[i].MixInto(intermediate_mixes[mix], mix);
        }
    }

    // Generate final mix
    write.dsp_status = mixers.Tick(read.dsp_configuration, read.intermediate_mix_samples,
                                   write.intermediate_mix_samples, intermediate_mixes);

    StereoFrame16 output_frame = mixers.GetOutput();

    // Write current output frame to the shared memory region
    for (std::size_t samplei = 0; samplei < output_frame.size(); samplei++) {
        for (std::size_t channeli = 0; channeli < output_frame[0].size(); channeli++) {
            write.final_samples.pcm16[samplei][channeli] = s16_le(output_frame[samplei][channeli]);
        }
    }

    return output_frame;
}

bool DspHle::Impl::Tick() {
    bool is_on = GetDspState() == DspState::On;

    if (is_on) {
        // The audio thread mixes frames too, when this one is late with them
        std::scoped_lock lock{parent.production_mutex};
        frames_ahead = 0;
#ifdef CITRA_ENABLE_TRACE_MARKERS
        if (++ticks_since_report >= 1024) {
            // Every five seconds or so
            if (reported_left_out != 0) {
                LOG_INFO(Audio_DSP, "{} frames left out: the device had more than enough",
                         reported_left_out);
                reported_left_out = 0;
            }
            if (reported_ahead != 0) {
                LOG_INFO(Audio_DSP,
                         "{} frames mixed ahead of the emulation, with a source waiting for more "
                         "in {}; of the sources that waited {} went on with more and {} ended; "
                         "up to {} sources played, up to {} of them with nothing more queued, the "
                         "closest to its end {} frames from it",
                         reported_ahead, reported_waits, reported_went_on, reported_ended,
                         reported_playing, reported_unqueued, reported_closest);
            }
            reported_ahead = 0;
            reported_waits = 0;
            reported_went_on = 0;
            reported_ended = 0;
            reported_playing = 0;
            reported_unqueued = 0;
            reported_closest = -1;
            ticks_since_report = 0;
        }
        for (auto& source : sources) {
            switch (source.TakeWaitOutcome()) {
            case HLE::Source::WaitOutcome::WentOn:
                reported_went_on++;
                break;
            case HLE::Source::WaitOutcome::Ended:
                reported_ended++;
                break;
            default:
                break;
            }
        }
#endif

        if (parent.TakeSpareFrame()) {
            // The device has more than enough waiting. The application is heard and answered,
            // and the sound stays where it is for this tick.
            TickWithoutFrame();
#ifdef CITRA_ENABLE_TRACE_MARKERS
            reported_left_out++;
#endif
            return is_on;
        }

        StereoFrame16 current_frame = {};

        // TODO: Check dsp::DSP semaphore (which indicates emulated application has finished writing
        // to shared memory region)
        current_frame = GenerateCurrentFrame();

        ApplyOutputGain(current_frame, std::min(1.0f, output_gain + GAIN_BACK_PER_FRAME));
        parent.OutputFrameLocked(std::move(current_frame));
    }

    return is_on;
}

void DspHle::Impl::ApplyOutputGain(StereoFrame16& frame, float target) {
    if (output_gain == 1.0f && target == 1.0f) {
        return;
    }
    const float step = (target - output_gain) / static_cast<float>(frame.size());
    float gain = output_gain;
    for (auto& sample : frame) {
        gain += step;
        sample[0] = static_cast<s16>(static_cast<float>(sample[0]) * gain);
        sample[1] = static_cast<s16>(static_cast<float>(sample[1]) * gain);
    }
    output_gain = target;
}

void DspHle::Impl::TickWithoutFrame() {
    HLE::SharedMemory& read = ReadRegion();
    HLE::SharedMemory& write = WriteRegion();

    for (std::size_t i = 0; i < HLE::num_sources; i++) {
        write.source_statuses.status[i] = sources[i].TickWithoutFrame(
            read.source_configurations.config[i], read.adpcm_coefficients.coeff[i]);
    }
    write.dsp_status = mixers.TickWithoutFrame(read.dsp_configuration);
}

bool DspHle::Impl::MixFrameAhead() {
    if (dsp_state != DspState::On || frames_ahead >= MOST_FRAMES_AHEAD) {
        return false;
    }
    frames_ahead++;

#ifdef CITRA_ENABLE_TRACE_MARKERS
    u32 playing = 0;
    u32 unqueued = 0;
    for (const auto& source : sources) {
        playing += source.IsEnabled() ? 1 : 0;
        if (const s32 left = source.FramesLeft(); left >= 0) {
            unqueued++;
            reported_closest = reported_closest < 0 ? left : std::min(reported_closest, left);
        }
    }
    reported_playing = std::max(reported_playing, playing);
    reported_unqueued = std::max(reported_unqueued, unqueued);
#endif

    // As GenerateCurrentFrame, without the application: its configuration is as the last tick
    // left it, and what it is told about the sources it reads at the next one
    std::array<QuadFrame32, 3> intermediate_mixes = {};
    for (auto& source : sources) {
        source.TickAhead();
        for (std::size_t mix = 0; mix < 3; mix++) {
            source.MixInto(intermediate_mixes[mix], mix);
        }
    }
    mixers.TickAhead(intermediate_mixes);

    reported_ahead++;
    if (std::ranges::any_of(sources, &HLE::Source::IsWaiting)) {
        reported_waits++;
    }

    // Down to nothing over the last frames before this stops
    const float left = static_cast<float>(MOST_FRAMES_AHEAD - frames_ahead);
    StereoFrame16 frame = mixers.GetOutput();
    ApplyOutputGain(frame, std::min(1.0f, left / static_cast<float>(FADING_FRAMES_AHEAD)));
    parent.OutputFrameLocked(std::move(frame), true);
    return true;
}

void DspHle::Impl::AudioTickCallback(s64 cycles_late) {
    if (sleep_pending) {
        FinishPendingSleep();
        sleep_reply_frames = 2;
    } else if (sleep_reply_frames > 0 && --sleep_reply_frames == 0) {
        interrupt_handler(InterruptType::Zero, DspPipe::Debug);
    }

    if (Tick()) {
        // TODO(merry): Signal all the other interrupts as appropriate.
        interrupt_handler(InterruptType::Pipe, DspPipe::Audio);
    }

    // Reschedule recurrent event
    const double time_scale =
        Core::System::GetInstance().IsAudioRealtime()
            ? std::max(0.01, // Arbitrary small value to prevent time_scale from approaching zero
                       Core::System::GetInstance().GetStableFrameTimeScale())
            : 1.0;
    s64 adjusted_ticks = static_cast<s64>(audio_frame_ticks / time_scale - cycles_late);
    core_timing.ScheduleEvent(adjusted_ticks, tick_event);
}

DspHle::DspHle(Core::System& system, Memory::MemorySystem& memory, Core::Timing& timing)
    : DspInterface(system), impl(std::make_unique<Impl>(*this, memory, timing)) {}

DspHle::~DspHle() {
    // The audio thread may be about to mix a frame: the device is only closed further up
    std::scoped_lock lock{production_mutex};
    impl.reset();
}

bool DspHle::ProduceFrameAhead() {
    return impl && impl->MixFrameAhead();
}

u16 DspHle::RecvData(u32 register_number) {
    return impl->RecvData(register_number);
}

bool DspHle::RecvDataIsReady(u32 register_number) const {
    return impl->RecvDataIsReady(register_number);
}

void DspHle::SetSemaphore(u16 semaphore_value) {
    // Do nothing in HLE
}

std::vector<u8> DspHle::PipeRead(DspPipe pipe_number, std::size_t length) {
    return impl->PipeRead(pipe_number, length);
}

size_t DspHle::GetPipeReadableSize(DspPipe pipe_number) const {
    return impl->GetPipeReadableSize(pipe_number);
}

void DspHle::PipeWrite(DspPipe pipe_number, std::span<const u8> buffer) {
    impl->PipeWrite(pipe_number, buffer);
}

void DspHle::SetInterruptHandler(
    std::function<void(Service::DSP::InterruptType type, DspPipe pipe)> handler) {
    impl->SetInterruptHandler(handler);
};

void DspHle::LoadComponent(std::span<const u8> component_data) {
    // HLE doesn't need DSP program. Only log some info here
    LOG_INFO(Service_DSP, "Firmware hash: {:#018x}",
             Common::ComputeHash64<Common::HashAlgo64::CityHash>(component_data.data(),
                                                                 component_data.size()));
    // Some versions of the firmware have the location of DSP structures listed here.
    if (component_data.size() > 0x37C) {
        LOG_INFO(
            Service_DSP, "Structures hash: {:#018x}",
            Common::ComputeHash64<Common::HashAlgo64::CityHash>(component_data.data() + 0x340, 60));
    }
}

void DspHle::UnloadComponent() {
    // Do nothing
}

} // namespace AudioCore
