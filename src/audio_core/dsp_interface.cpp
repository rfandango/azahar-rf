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
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/dumping/backend.h"

namespace AudioCore {

namespace {
/// Amount of CSND frames to wait until they are outputted on their own
/// because DSP is not producing any output.
constexpr u32 CsndFramesBeforeDspIdle = 2;
/// Max amount of CSND pending samples.
constexpr std::size_t MaxCsndPendingSamples = samples_per_frame * 4;
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

    auto video_dumper = system.GetVideoDumper();
    if (video_dumper && video_dumper->IsDumping()) {
        for (const auto& sample : samples) {
            video_dumper->AddAudioSample(sample);
        }
    }
}

void DspInterface::DrainFifoBacklog(std::size_t num_frames) {
    // The emulation thread produces a video frame of audio at a time, so the FIFO only needs enough
    // headroom to smooth those bursts. Audio that stays in the FIFO for a whole window without
    // being needed remains queued during a stall or at stream start. That extra data delays all
    // later audio by its length, for the entire session.
    constexpr std::size_t window_frames = native_sample_rate;
    constexpr std::size_t kept_frames = native_sample_rate / 100;

    fifo_low_fill = std::min(fifo_low_fill, fifo.Size());
    backlog_window_frames += num_frames;
    if (backlog_window_frames < window_frames) {
        return;
    }
    if (fifo_low_fill > 2 * kept_frames) {
        const std::size_t excess = fifo_low_fill - kept_frames;
        fifo.Pop(excess);
        LOG_DEBUG(Audio, "Dropped {} frames of audio backlog", excess);
    }
    fifo_low_fill = std::numeric_limits<std::size_t>::max();
    backlog_window_frames = 0;
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

    std::size_t frames_written = 0;
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
        // The presentation backend may run the emulation a fraction of a percent faster or
        // slower to keep it in step with the display. The audio it produces is then played at
        // that speed too, rather than piling up or running out.
        const double ratio =
            Settings::values.enable_realtime_audio ? 1.0 : system.frame_limiter.GetPacingScale();
        if (ratio != 1.0 && ratio > 0.98 && ratio < 1.02) {
            frames_written +=
                PopResampled(buffer + 2 * frames_written, num_frames - frames_written, ratio);
        } else {
            resampling = false;
            frames_written += fifo.Pop(buffer, num_frames - frames_written);
        }
        DrainFifoBacklog(num_frames);
    }

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
}

} // namespace AudioCore
