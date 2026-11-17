// Copyright 2017-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <cstddef>
#include <vector>
#include "audio_core/dsp_interface.h"
#include "audio_core/sink.h"
#include "audio_core/sink_details.h"
#include "common/assert.h"
#include "common/hitch_tracker.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "core/core.h"
#include "core/dumping/backend.h"

#if defined(CITRA_ENABLE_TRACE_MARKERS) && defined(ANDROID)
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fmt/format.h>
#include <sys/system_properties.h>
#define CITRA_AUDIO_RECORDER 1
#endif

namespace AudioCore {

namespace {
/// Amount of CSND frames to wait until they are outputted on their own
/// because DSP is not producing any output.
constexpr u32 CsndFramesBeforeDspIdle = 2;
/// Max amount of CSND pending samples.
constexpr std::size_t MaxCsndPendingSamples = samples_per_frame * 4;

#ifdef CITRA_AUDIO_RECORDER
/// Keeps what is handed to the audio device for a while, to look at it afterwards. Setting
/// `debug.azahar.record_audio` to a new value records the next twelve seconds into
/// files/audio-<value>.raw of the app's external directory: 16 bit stereo at the native rate.
class OutputRecorder {
public:
    /// Called now and then on the emulation thread
    void Poll() {
        if (state.load(std::memory_order_acquire) == State::Full) {
            Save();
            state.store(State::Idle, std::memory_order_release);
        }
        char value[PROP_VALUE_MAX]{};
        __system_property_get("debug.azahar.record_audio", value);
        if (!initialized) {
            // What the property holds when the game starts is an old request
            request = value;
            initialized = true;
            return;
        }
        if (value[0] == 0 || request == value ||
            state.load(std::memory_order_acquire) != State::Idle) {
            return;
        }
        request = value;
        samples.assign(SECONDS * native_sample_rate * 2, 0);
        position.store(0, std::memory_order_relaxed);
        state.store(State::Recording, std::memory_order_release);
    }

    /// Called on the audio thread with what goes to the device
    void Write(const s16* buffer, std::size_t num_frames) {
        if (state.load(std::memory_order_acquire) != State::Recording) [[likely]] {
            return;
        }
        const std::size_t at = position.load(std::memory_order_relaxed);
        const std::size_t count = std::min(num_frames * 2, samples.size() - at);
        std::memcpy(samples.data() + at, buffer, count * sizeof(s16));
        position.store(at + count, std::memory_order_relaxed);
        if (at + count == samples.size()) {
            state.store(State::Full, std::memory_order_release);
        }
    }

private:
    enum class State : u32 { Idle, Recording, Full };
    static constexpr std::size_t SECONDS = 12;

    void Save() {
        std::string package;
        if (std::FILE* file = std::fopen("/proc/self/cmdline", "r")) {
            char data[256]{};
            const std::size_t read = std::fread(data, 1, sizeof(data) - 1, file);
            std::fclose(file);
            package.assign(data, strnlen(data, read));
        }
        const std::string path =
            fmt::format("/sdcard/Android/data/{}/files/audio-{}.raw", package, request);
        if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
            std::fwrite(samples.data(), sizeof(s16), samples.size(), file);
            std::fclose(file);
            LOG_INFO(Audio, "Recorded the audio output to {}", path);
        } else {
            LOG_WARNING(Audio, "Cannot write {}", path);
        }
        samples = {};
    }

    std::atomic<State> state{State::Idle};
    std::atomic<std::size_t> position{0};
    std::vector<s16> samples;
    std::string request;
    bool initialized = false;
};
OutputRecorder output_recorder;
u32 pushes_since_poll = 0;
#endif
} // namespace

DspInterface::DspInterface(Core::System& system_) : system(system_) {}

DspInterface::~DspInterface() = default;

void DspInterface::SetSink(AudioCore::SinkType sink_type, std::string_view audio_device) {
    // Dispose of the current sink first to avoid contention.
    sink.reset();

    sink = AudioCore::GetSinkDetails(sink_type).create_sink(audio_device);
    sink->SetCallback(
        [this](s16* buffer, std::size_t num_frames) { OutputCallback(buffer, num_frames); });
    time_stretcher.SetOutputSampleRate(sink->GetNativeSampleRate());
}

Sink& DspInterface::GetSink() {
    ASSERT(sink);
    return *sink.get();
}

void DspInterface::EnableStretching(bool enable) {
    enable_time_stretching = enable;
}

void DspInterface::OutputFrame(StereoFrame16 frame) {
    std::scoped_lock lock{csnd_mutex};
    MixCsnd(frame);
    PushSamples(frame);
}

void DspInterface::OutputSample(std::array<s16, 2> sample) {
    std::scoped_lock lock{csnd_mutex};
    MixCsnd({&sample, 1});
    PushSamples({&sample, 1});
}

void DspInterface::OutputCsndFrame(const StereoFrame16& frame) {
    std::scoped_lock lock{csnd_mutex};

    if (dsp_samples_since_csnd_frame == 0) {
        csnd_frames_without_dsp++;
    } else {
        csnd_frames_without_dsp = 0;
    }
    dsp_samples_since_csnd_frame = 0;

    csnd_pending.insert(csnd_pending.end(), frame.begin(), frame.end());

    if (csnd_frames_without_dsp >= CsndFramesBeforeDspIdle) {
        // The DSP is not running, CSND is the only output.
        const std::vector<std::array<s16, 2>> samples(csnd_pending.begin(), csnd_pending.end());
        csnd_pending.clear();
        PushSamples(samples);
        return;
    }

    // The two clocks may drift a little, don't let CSND fall too far behind the DSP output.
    while (csnd_pending.size() > MaxCsndPendingSamples) {
        csnd_pending.pop_front();
    }
}

void DspInterface::MixCsnd(std::span<std::array<s16, 2>> samples) {
    // TODO(PabloMK7): Implement SIMD mixing here if possible.
    dsp_samples_since_csnd_frame += samples.size();
    for (auto& sample : samples) {
        if (csnd_pending.empty()) {
            break;
        }
        const auto& csnd = csnd_pending.front();
        for (std::size_t i = 0; i < 2; i++) {
            sample[i] = static_cast<s16>(std::clamp(sample[i] + csnd[i], -0x8000, 0x7FFF));
        }
        csnd_pending.pop_front();
    }
}

void DspInterface::PushSamples(std::span<const std::array<s16, 2>> samples) {
    if (!sink) {
        return;
    }

    if (sink->ImmediateSubmission()) {
        sink->PushSamples(samples.data(), samples.size());
    } else {
        fifo.Push(samples.data(), samples.size());
    }
#ifdef CITRA_AUDIO_RECORDER
    if (++pushes_since_poll >= 64) {
        pushes_since_poll = 0;
        output_recorder.Poll();
    }
#endif

    auto video_dumper = system.GetVideoDumper();
    if (video_dumper && video_dumper->IsDumping()) {
        for (const auto& sample : samples) {
            video_dumper->AddAudioSample(sample);
        }
    }
}

std::size_t DspInterface::DrainFifoBacklog(std::size_t num_frames) {
    // The emulation thread produces a video frame of audio at a time, so the FIFO only needs enough
    // headroom to smooth those bursts. Audio that stays in the FIFO for a whole window without
    // being needed remains queued during a stall or at stream start. That extra data delays all
    // later audio by its length, for the entire session.
    constexpr std::size_t window_frames = native_sample_rate;
    constexpr std::size_t kept_frames = native_sample_rate / 100;

    fifo_low_fill = std::min(fifo_low_fill, fifo.Size());
    backlog_window_frames += num_frames;
    if (backlog_window_frames < window_frames) {
        return 0;
    }
    std::size_t excess = 0;
    if (fifo_low_fill > 2 * kept_frames) {
        excess = fifo_low_fill - kept_frames;
        fifo.Pop(excess);
        LOG_DEBUG(Audio, "Dropped {} frames of audio backlog", excess);
    }
    fifo_low_fill = std::numeric_limits<std::size_t>::max();
    backlog_window_frames = 0;
    return excess;
}

std::size_t DspInterface::PopResampled(s16* buffer, std::size_t num_frames, double ratio) {
    // Output frame n lies at input position n * ratio. Four input frames around that position
    // are kept, and the output is a cubic through them, between the middle two.
    if (!resampling) {
        resample_window.fill(last_frame);
        resample_position = 0.0;
        resampling = true;
    }

    const std::size_t wanted = static_cast<std::size_t>(resample_position + ratio * num_frames);
    resample_input.resize(wanted * 2);
    const std::size_t available = fifo.Pop(resample_input.data(), wanted);
    std::size_t next_input = 0;

    const auto advance = [&]() -> bool {
        while (resample_position >= 1.0) {
            if (next_input == available) {
                return false;
            }
            std::copy(resample_window.begin() + 1, resample_window.end(), resample_window.begin());
            resample_window[3] = {resample_input[2 * next_input],
                                  resample_input[2 * next_input + 1]};
            next_input++;
            resample_position -= 1.0;
        }
        return true;
    };

    std::size_t written = 0;
    for (; written < num_frames; written++) {
        if (!advance()) {
            // Ran out of input: the caller holds the last frame
            return written;
        }
        const double t = resample_position;
        for (std::size_t channel = 0; channel < 2; channel++) {
            const double y0 = resample_window[0][channel];
            const double y1 = resample_window[1][channel];
            const double y2 = resample_window[2][channel];
            const double y3 = resample_window[3][channel];
            const double c1 = 0.5 * (y2 - y0);
            const double c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
            const double c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
            const double value = ((c3 * t + c2) * t + c1) * t + y1;
            buffer[2 * written + channel] = static_cast<s16>(std::clamp(value, -32768.0, 32767.0));
        }
        resample_position += ratio;
    }
    // Take in what was popped for the position reached
    advance();
    return written;
}

void DspInterface::OutputCallback(s16* buffer, std::size_t num_frames) {
    // Determine if we should stretch based on the current emulation speed.
    // TODO: Only activate audio stretching when emulation speed goes below 95% threshold
    //       (see #2487) -OS
    if (performing_time_stretching && !enable_time_stretching) {
        // If we just stopped stretching, flush the stretcher before returning to normal output.
        flushing_time_stretcher = true;
    }
    performing_time_stretching = enable_time_stretching.load();

    std::size_t buffered = fifo.Size();
    std::size_t dropped = 0;
    std::size_t frames_written = 0;
    // Set where the sound is not simply handed over as it comes
    bool paced = false;
    std::size_t silent = 0;
    double rate = 1.0;
    std::size_t margin = 0;
    if (performing_time_stretching) {
        const std::vector<s16> in{fifo.Pop()};
        const std::size_t num_in{in.size() / 2};
        frames_written = time_stretcher.Process(in.data(), num_in, buffer, num_frames);
    } else {
        if (flushing_time_stretcher) {
            time_stretcher.Flush();
            frames_written = time_stretcher.Process(nullptr, 0, buffer, num_frames);
            flushing_time_stretcher = false;

            // Make sure any frames that did not fit are cleared from the time stretcher,
            // so that they do not bleed into the next time the stretcher is enabled.
            time_stretcher.Clear();
        }
        if (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyAudioOutput) {
            // The presentation backend may run the emulation a fraction of a percent faster
            // or slower to keep it in step with the display. The audio it produces is then
            // played at that speed too, rather than piling up or running out.
            const double ratio = Settings::values.enable_realtime_audio
                                     ? 1.0
                                     : system.frame_limiter.GetPacingScale();
            if (ratio != 1.0 && ratio > 0.98 && ratio < 1.02) {
                frames_written +=
                    PopResampled(buffer + 2 * frames_written, num_frames - frames_written, ratio);
            } else {
                resampling = false;
                frames_written += fifo.Pop(buffer, num_frames - frames_written);
            }
            dropped = DrainFifoBacklog(num_frames);
        } else {
            // Played at the speed it is produced at, with a margin for when it comes late.
            // Sound is produced as emulated time passes, or more often than that to keep it
            // in step with real time.
            const double production_speed = system.GetAudioProductionSpeed();
            const OutputPacer::Result result = output_pacer.Render(
                fifo, buffer + 2 * frames_written, num_frames - frames_written, production_speed);
            paced = true;
            silent = result.silent_frames;
            rate = result.rate;
            margin = result.margin;
            buffered = result.waiting;
            frames_written = num_frames;
            resampling = false;
        }
    }

    Common::Hitches::NoteAudioOutput({
        .sample_rate = native_sample_rate,
        .wanted = static_cast<u32>(num_frames),
        .missing = static_cast<u32>(paced ? silent : num_frames - frames_written),
        .buffered = static_cast<u32>(buffered),
        .dropped = static_cast<u32>(dropped),
        .margin = static_cast<u32>(margin),
        .rate = static_cast<float>(rate),
    });
    CITRA_TRACE_SET_COUNTER("azahar.audio_waiting_ms", buffered * 1000 / native_sample_rate);
    CITRA_TRACE_SET_COUNTER("azahar.audio_missing_frames",
                            paced ? silent : num_frames - frames_written);
    CITRA_TRACE_SET_COUNTER("azahar.audio_rate_permille", static_cast<s64>(rate * 1000.0));

    if (frames_written > 0) {
        std::memcpy(&last_frame[0], buffer + 2 * (frames_written - 1), 2 * sizeof(s16));
    }

    // Hold last emitted frame; this prevents popping.
    for (std::size_t i = frames_written; i < num_frames; i++) {
        std::memcpy(buffer + 2 * i, &last_frame[0], 2 * sizeof(s16));
    }

    // Implementation of the hardware volume slider
    // A cubic curve is used to approximate a linear change in human-perceived loudness
    const float linear_volume = std::clamp(Settings::Volume(), 0.0f, 1.0f);
    if (linear_volume != 1.0) {
        const float volume_scale_factor = linear_volume * linear_volume * linear_volume;
        for (std::size_t i = 0; i < num_frames; i++) {
            buffer[i * 2 + 0] = static_cast<s16>(buffer[i * 2 + 0] * volume_scale_factor);
            buffer[i * 2 + 1] = static_cast<s16>(buffer[i * 2 + 1] * volume_scale_factor);
        }
    }
#ifdef CITRA_AUDIO_RECORDER
    output_recorder.Write(buffer, num_frames);
#endif
}

} // namespace AudioCore
