// Copyright 2017-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <chrono>
#include <iterator>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include "common/file_util.h"
#include "common/hitch_tracker.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "core/core_timing.h"
#include "core/perf_stats.h"
#include "video_core/gpu.h"

#ifndef _WIN32
#include <time.h>
#endif

namespace {

/// The processor time the calling thread has used so far, or zero where it cannot be told
std::chrono::microseconds ThreadCpuTime() {
#ifndef _WIN32
    timespec time{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) == 0) {
        return std::chrono::microseconds{static_cast<s64>(time.tv_sec) * 1'000'000 +
                                         time.tv_nsec / 1000};
    }
#endif
    return std::chrono::microseconds::zero();
}

} // Anonymous namespace

using namespace std::chrono_literals;
using DoubleSecs = std::chrono::duration<double, std::chrono::seconds::period>;
using std::chrono::duration_cast;
using std::chrono::microseconds;

constexpr double FRAME_LENGTH = 1.0 / SCREEN_REFRESH_RATE;
// Purposefully ignore the first five frames, as there's a significant amount of overhead in
// booting that we shouldn't account for
constexpr std::size_t IgnoreFrames = 5;

namespace Core {

bool PerfStats::game_frames_updated = true;

PerfStats::PerfStats(u64 title_id) : title_id(title_id) {}

PerfStats::~PerfStats() {
    if (!Settings::values.record_frame_times || title_id == 0) {
        return;
    }

    const std::time_t t = std::time(nullptr);
    std::ostringstream stream;
    std::copy(perf_history.begin() + IgnoreFrames, perf_history.begin() + current_index,
              std::ostream_iterator<double>(stream, "\n"));
    const std::string& path = FileUtil::GetUserPath(FileUtil::UserPath::LogDir);
    // %F Date format expanded is "%Y-%m-%d"
    const std::string filename =
        fmt::format("{}/{:%F-%H-%M}_{:016X}.csv", path, *std::localtime(&t), title_id);
    FileUtil::IOFile file(filename, "w");
    file.WriteString(stream.str());
}

void PerfStats::BeginSVCProcessing() {
    // A game makes tens of thousands of system calls a second, and reading the clock at both
    // ends of each took 4% of all CPU time (Xenoblade Chronicles 3D on a Snapdragon 865,
    // where reading the clock is a system call of the host). One call in SVC_SAMPLE_INTERVAL
    // is timed and stands for the others (one in sixteen still cost 0.6% of the emulation
    // thread in that game, which makes 88,000 calls a second: one in sixty-four is still
    // over a thousand timed a second there). The few calls that process GPU commands are
    // far longer than the rest and would make such an estimate jump: that part is timed
    // every time, and kept out of what is scaled up.
    svc_sample_interval =
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyEmulationOverheads)
            ? 1
            : SVC_SAMPLE_INTERVAL;
    svc_timed = (svc_count++ % svc_sample_interval) == 0;
    if (svc_timed) {
        start_svc_time = Clock::now();
    }
}

void PerfStats::EndSVCProcessing() {
    if (svc_timed) {
        accumulated_svc_time += (Clock::now() - start_svc_time);
        svc_timed = false;
    }
}

void PerfStats::BeginIPCProcessing() {
    if (svc_timed) {
        start_ipc_time = Clock::now();
    }
}

void PerfStats::EndIPCProcessing() {
    if (svc_timed) {
        accumulated_ipc_time += (Clock::now() - start_ipc_time);
    }
}

void PerfStats::BeginGPUProcessing() {
    start_gpu_time = Clock::now();
}

void PerfStats::EndGPUProcessing() {
    const Clock::duration taken = Clock::now() - start_gpu_time;
    accumulated_gpu_time += taken;
    if (svc_timed) {
        accumulated_gpu_time_in_timed_svc += taken;
    }
}

void PerfStats::StartSwap() {
    start_swap_time = Clock::now();
}

void PerfStats::EndSwap() {
    accumulated_swap_time += (Clock::now() - start_swap_time);
}

void PerfStats::BeginSystemFrame() {
    std::scoped_lock lock{object_mutex};

    frame_begin = Clock::now();
}

void PerfStats::EndSystemFrame() {
    std::scoped_lock lock{object_mutex};

    auto frame_end = Clock::now();
    const auto frame_time = frame_end - frame_begin;
    if (current_index < perf_history.size()) {
        perf_history[current_index++] =
            std::chrono::duration<double, std::milli>(frame_time).count();
    }
    accumulated_frametime += frame_time;
    last_system_frame_time = frame_time;
    system_frames += 1;

    // TODO: Track previous frame times in a less stupid way. -OS
    previous_previous_frame_length = previous_frame_length;

    previous_frame_length = frame_end - previous_frame_end;
    previous_frame_end = frame_end;
}

void PerfStats::EndGameFrame() {
    std::scoped_lock lock{object_mutex};

    game_frames += 1;
    total_game_frames += 1;
    PerfStats::game_frames_updated = true;
}

double PerfStats::GetMeanFrametime() const {
    std::scoped_lock lock{object_mutex};

    if (current_index <= IgnoreFrames) {
        return 0;
    }

    const double sum = std::accumulate(perf_history.begin() + IgnoreFrames,
                                       perf_history.begin() + current_index, 0.0);
    return sum / static_cast<double>(current_index - IgnoreFrames);
}

PerfStats::Clock::duration PerfStats::GetLastSystemFrameTime() const {
    std::scoped_lock lock{object_mutex};
    return last_system_frame_time;
}

u64 PerfStats::GetGameFrameCount() const {
    std::scoped_lock lock{object_mutex};
    return total_game_frames;
}

PerfStats::Results PerfStats::GetAndResetStats(microseconds current_system_time_us) {
    std::scoped_lock lock{object_mutex};

    const auto now = Clock::now();
    // Walltime elapsed since stats were reset
    const auto interval = duration_cast<DoubleSecs>(now - reset_point).count();

    const auto system_us_per_second = (current_system_time_us - reset_point_system_us) / interval;

    last_stats.system_fps = static_cast<double>(system_frames) / interval;
    last_stats.game_fps = static_cast<double>(game_frames) / interval;
    last_stats.time_vblank_interval =
        system_frames ? (duration_cast<DoubleSecs>(accumulated_frametime).count() /
                         static_cast<double>(system_frames))
                      : 0;
    // The system calls that were timed stand for all of them, but for the GPU command
    // processing in them, which is timed in every call
    const auto for_all_calls = [this](Clock::duration timed) {
        return (timed - accumulated_gpu_time_in_timed_svc) * svc_sample_interval +
               accumulated_gpu_time;
    };
    const Clock::duration svc_time = for_all_calls(accumulated_svc_time);
    const Clock::duration ipc_time = for_all_calls(accumulated_ipc_time);
    last_stats.time_hle_svc = system_frames
                                  ? (duration_cast<DoubleSecs>(svc_time - ipc_time).count() /
                                     static_cast<double>(system_frames))
                                  : 0;
    last_stats.time_hle_ipc =
        system_frames ? (duration_cast<DoubleSecs>(ipc_time - accumulated_gpu_time).count() /
                         static_cast<double>(system_frames))
                      : 0;
    last_stats.time_gpu = system_frames ? (duration_cast<DoubleSecs>(accumulated_gpu_time).count() /
                                           static_cast<double>(system_frames))
                                        : 0;
    last_stats.time_swap = system_frames
                               ? (duration_cast<DoubleSecs>(accumulated_swap_time).count() /
                                  static_cast<double>(system_frames))
                               : 0;

    last_stats.time_remaining =
        system_frames
            ? (duration_cast<DoubleSecs>((accumulated_frametime - svc_time) - accumulated_swap_time)
                   .count() /
               static_cast<double>(system_frames))
            : 0;
    last_stats.emulation_speed = system_us_per_second.count() / 1'000'000.0;
    last_stats.artic_transmitted = static_cast<double>(artic_transmitted) / interval;
    last_stats.artic_events.raw = artic_events.raw | prev_artic_event.raw;

    // Reset counters
    reset_point = now;
    reset_point_system_us = current_system_time_us;
    accumulated_frametime = Clock::duration::zero();
    system_frames = 0;
    accumulated_svc_time = Clock::duration::zero();
    accumulated_ipc_time = Clock::duration::zero();
    accumulated_gpu_time = Clock::duration::zero();
    accumulated_gpu_time_in_timed_svc = Clock::duration::zero();
    accumulated_swap_time = Clock::duration::zero();
    game_frames = 0;
    artic_transmitted = 0;
    prev_artic_event.raw &= artic_events.raw;

    return last_stats;
}

PerfStats::Results PerfStats::GetLastStats() {
    std::scoped_lock lock{object_mutex};

    return last_stats;
}

double PerfStats::GetLastFrameTimeScale() const {
    std::scoped_lock lock{object_mutex};

    return duration_cast<DoubleSecs>(previous_frame_length).count() / FRAME_LENGTH;
}

double PerfStats::GetStableFrameTimeScale() const {
    std::scoped_lock lock{object_mutex};

    const double stable_previous_frame_length =
        (duration_cast<DoubleSecs>(previous_frame_length).count() +
         duration_cast<DoubleSecs>(previous_previous_frame_length).count()) /
        2;
    return stable_previous_frame_length / FRAME_LENGTH;
}

void FrameLimiter::WaitForFrameAdvance() {
    if (wait_callback) {
        // Let the callback wait in short steps, checking for the frame advance in between
        constexpr auto check_interval = std::chrono::milliseconds(2);
        while (!frame_advance_event.IsSet()) {
            wait_callback(std::chrono::steady_clock::now() + check_interval);
        }
    }
    frame_advance_event.Wait();
    frame_advance_event.Reset();
}

void FrameLimiter::SetWaitCallback(WaitCallback callback) {
    wait_callback = std::move(callback);
}

void FrameLimiter::WaitOnce() {
    if (frame_advancing_enabled) {
        // Frame advancing is enabled: wait on event instead of doing framelimiting
        WaitForFrameAdvance();
    }
}

void FrameLimiter::DoFrameLimiting(microseconds current_system_time_us) {
    if (Common::TraceMarkers::Experiments() & Common::TraceMarkers::InjectHitches) [[unlikely]] {
        static u32 frames_since_hitch = 0;
        if (++frames_since_hitch >= 60) {
            frames_since_hitch = 0;
            std::this_thread::sleep_for(30ms);
        }
    }
    if (const auto stall = Common::TraceMarkers::Stall(); stall.milliseconds != 0) [[unlikely]] {
        static u32 vblanks_since_stall = 0;
        if (++vblanks_since_stall >= stall.every) {
            vblanks_since_stall = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds{stall.milliseconds});
        }
    }

    if (frame_advancing_enabled) {
        // Frame advancing is enabled: wait on event instead of doing framelimiting
        WaitForFrameAdvance();
        return;
    }

    auto now = Clock::now();
    last_interval_emulated = current_system_time_us - previous_system_time_us;
    last_interval_work = duration_cast<microseconds>(now - previous_walltime);
    // Of that, what the thread spent running: the rest it waited (for the GPU, for a file)
    const microseconds cpu_time = ThreadCpuTime();
    last_interval_cpu = previous_cpu_time > microseconds::zero() && cpu_time > previous_cpu_time
                            ? std::min(cpu_time - previous_cpu_time, last_interval_work)
                            : microseconds::zero();
    previous_cpu_time = cpu_time;
    double sleep_scale =
        Settings::GetFrameLimit() / 100.0 * pacing_scale.load(std::memory_order_relaxed);

    if (Settings::GetFrameLimit() == 0) {
        MeasureSpeed(now, current_system_time_us, false, 0.0);
        return;
    }

    // Max lag caused by slow frames. Shouldn't be more than the length of a frame at the current
    // speed percent or it will clamp too much and prevent this from properly limiting to that
    // percent. High values means it'll take longer after a slow frame to recover and start limiting
    const microseconds max_lag_time_us = duration_cast<microseconds>(
        std::chrono::duration<double, std::chrono::microseconds::period>(25ms / sleep_scale));
    frame_limiting_delta_err += duration_cast<microseconds>(
        std::chrono::duration<double, std::chrono::microseconds::period>(
            (current_system_time_us - previous_system_time_us) / sleep_scale));
    frame_limiting_delta_err -= duration_cast<microseconds>(now - previous_walltime);
    frame_limiting_delta_err =
        std::clamp(frame_limiting_delta_err, -max_lag_time_us, max_lag_time_us);

    const bool on_schedule = frame_limiting_delta_err > microseconds::zero();
    if (on_schedule) {
        const microseconds wanted = frame_limiting_delta_err;
        if (wait_callback) {
            wait_callback(std::chrono::steady_clock::now() + frame_limiting_delta_err);
        } else {
            std::this_thread::sleep_for(frame_limiting_delta_err);
        }
        auto now_after_sleep = Clock::now();
        frame_limiting_delta_err -= duration_cast<microseconds>(now_after_sleep - now);
        if (now_after_sleep - now > wanted) {
            Common::Hitches::MoveTime(Common::Hitches::Category::Limiter,
                                      Common::Hitches::Category::LimiterOvershoot,
                                      now_after_sleep - now - wanted);
        }
        now = now_after_sleep;
    }
    MeasureSpeed(now, current_system_time_us, on_schedule, sleep_scale);

    previous_system_time_us = current_system_time_us;
    previous_walltime = now;
}

void FrameLimiter::MeasureSpeed(Clock::time_point now, microseconds emulated_time, bool on_schedule,
                                double limit) {
    using Seconds = std::chrono::duration<double>;
    if (speed_stretch_start == Clock::time_point{} || emulated_time < speed_stretch_emulated) {
        // The first frame, or emulated time started over
        speed_stretch_start = now;
        speed_stretch_emulated = emulated_time;
        speed_previous_frame = now;
        return;
    }

    // Games do their work in some frames and next to none in others, so the emulation falls
    // behind and catches up all the time. Where it waited for real time it is not behind at
    // all, and after a frame no longer than most it is as little behind as it gets. Between
    // two such points emulated time and real time can be compared; between any two frames
    // the comparison would be off by however much the emulation happened to be behind.
    const double frame_time = Seconds(now - speed_previous_frame).count();
    speed_previous_frame = now;
    std::array<double, 8> sorted_times = speed_frame_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    const bool short_frame = frame_time <= sorted_times[sorted_times.size() / 2];
    speed_frame_times[speed_frame_count++ % speed_frame_times.size()] = frame_time;

    constexpr double SHORTEST_STRETCH = 0.1;
    constexpr double LONGEST_STRETCH = 0.3;
    const double elapsed = Seconds(now - speed_stretch_start).count();
    if (elapsed < SHORTEST_STRETCH || (!on_schedule && !short_frame && elapsed < LONGEST_STRETCH)) {
        return;
    }
    const double stretch_speed = Seconds(emulated_time - speed_stretch_emulated).count() / elapsed;
    speed_stretches[speed_stretch_count++ % speed_stretches.size()] =
        std::clamp(stretch_speed, 0.1, 8.0);
    speed_stretch_start = now;
    speed_stretch_emulated = emulated_time;

    // The average of the last few without the slowest two: a stretch with a long frame in it
    // does not make the emulation slow, most of them being slow does. Two fast stretches in a
    // row make it fast again right away.
    if (speed_stretch_count >= 3) {
        std::array<double, 9> sorted = speed_stretches;
        const std::size_t count = std::min(speed_stretch_count, sorted.size());
        std::sort(sorted.begin(), sorted.begin() + count);
        const std::size_t left_out = std::min<std::size_t>(2, count / 3);
        const double average =
            std::accumulate(sorted.begin() + left_out, sorted.begin() + count, 0.0) /
            static_cast<double>(count - left_out);
        const std::size_t size = speed_stretches.size();
        const double last = speed_stretches[(speed_stretch_count - 1) % size];
        const double before = speed_stretches[(speed_stretch_count - 2) % size];
        double speed = std::max(average, std::min(last, before));
        if (limit > 0.0) {
            // After falling behind the emulation runs faster than it is held to for a moment,
            // to catch up. It cannot keep that up, so it does not count.
            speed = std::min(speed, limit);
        }
        emulation_speed.store(speed, std::memory_order_relaxed);
    }
}

bool FrameLimiter::IsFrameAdvancing() const {
    return frame_advancing_enabled;
}

void FrameLimiter::SetFrameAdvancing(bool value) {
    const bool was_enabled = frame_advancing_enabled.exchange(value);
    if (was_enabled && !value) {
        // Set the event to let emulation continue
        frame_advance_event.Set();
    }
}

void FrameLimiter::AdvanceFrame() {
    frame_advance_event.Set();
}

} // namespace Core
