// Copyright 2019-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include <android/native_window_jni.h>
#include <fmt/format.h>
#include <sched.h>
#include "common/android_utils.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#ifdef CITRA_ENABLE_TRACE_MARKERS
#include <sys/system_properties.h>
#endif
#include "core/core.h"
#include "core/core_timing.h"
#include "input_common/main.h"
#include "jni/emu_window/emu_window.h"
#include "jni/id_cache.h"
#include "jni/input_manager.h"
#include "network/network.h"
#include "video_core/renderer_base.h"

bool EmuWindow_Android::OnSurfaceChanged(ANativeWindow* surface) {
    int temp_width = (surface == nullptr) ? 0 : ANativeWindow_getWidth(surface);
    int temp_height = (surface == nullptr) ? 0 : ANativeWindow_getHeight(surface);
    if (render_window == surface && temp_width == window_width && temp_height == window_height) {
        return false;
    }
    StopPresenting();
    window_width = temp_width;
    window_height = temp_height;
    render_window = surface;
    window_info.type = Frontend::WindowSystemType::Android;
    window_info.render_surface = surface;
    OnFramebufferSizeChanged();
    return true;
}

bool EmuWindow_Android::OnTouchEvent(int x, int y, bool pressed) {
    if (pressed) {
        return TouchPressed((unsigned)std::max(x, 0), (unsigned)std::max(y, 0));
    }

    TouchReleased();
    return true;
}

void EmuWindow_Android::OnTouchMoved(int x, int y) {
    TouchMoved((unsigned)std::max(x, 0), (unsigned)std::max(y, 0));
}

namespace {

struct CoreSets {
    cpu_set_t all;
    cpu_set_t fast;
};

/// Finds the cores outside the slowest cluster. Empty if the cores are all alike, or if fewer
/// than two would be left.
std::optional<CoreSets> FindFastCores() {
    std::vector<long> max_freqs;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        std::ifstream file{
            fmt::format("/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq", cpu)};
        long max_freq{};
        if (!(file >> max_freq)) {
            break;
        }
        max_freqs.push_back(max_freq);
    }
    if (max_freqs.empty()) {
        return std::nullopt;
    }

    const long slowest = *std::ranges::min_element(max_freqs);
    CoreSets sets;
    CPU_ZERO(&sets.all);
    CPU_ZERO(&sets.fast);
    for (std::size_t cpu = 0; cpu < max_freqs.size(); cpu++) {
        CPU_SET(cpu, &sets.all);
        if (max_freqs[cpu] > slowest) {
            CPU_SET(cpu, &sets.fast);
        }
    }
    if (CPU_COUNT(&sets.fast) < 2) {
        return std::nullopt;
    }
    return sets;
}

/// Keeps the calling thread, the emulation thread, off the slowest cores.
///
/// The thread works in bursts and sleeps in between (frame limiter, waiting for the GPU), so the
/// scheduler takes it for a light task and now and then moves it to the slow cluster, where a
/// frame's work takes several times longer. The performance hint below asks for the opposite,
/// but not every device acts on it.
void KeepOffSlowCores() {
    static const std::optional<CoreSets> cores = FindFastCores();
    if (!cores) {
        return;
    }
    static thread_local int current = -1;
    const int wanted =
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyThreadPlacement) ? 0 : 1;
    if (wanted == current) {
        return;
    }
    current = wanted;
    const cpu_set_t& set = wanted ? cores->fast : cores->all;
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        LOG_WARNING(Frontend, "Could not set the cores of the emulation thread: {}", errno);
    } else {
        LOG_INFO(Frontend, "Emulation thread runs on {} of {} cores", CPU_COUNT(&set),
                 CPU_COUNT(&cores->all));
    }
}

#ifdef CITRA_ENABLE_TRACE_MARKERS
/// Profiling builds load a save state when asked through a system property, so that a scene
/// can be reached without touching the device: `adb shell setprop debug.azahar.load_state 1`.
/// Acts when the value changes; zero asks for nothing.
void PollLoadStateRequest() {
    static int last_slot = 0;
    char value[PROP_VALUE_MAX]{};
    const int slot =
        __system_property_get("debug.azahar.load_state", value) > 0 ? std::atoi(value) : 0;
    if (slot == last_slot) {
        return;
    }
    last_slot = slot;
    if (slot > 0) {
        LOG_INFO(Frontend, "Loading save state {} on request", slot);
        Core::System::GetInstance().SendSignal(Core::System::Signal::Load, slot);
    }
}
#endif

} // Anonymous namespace

void EmuWindow_Android::ReportFrameWork() {
    if (is_secondary) {
        return;
    }
    KeepOffSlowCores();
#ifdef CITRA_ENABLE_TRACE_MARKERS
    PollLoadStateRequest();
#endif
    auto& system = Core::System::GetInstance();
    if (!system.perf_stats) {
        return;
    }

    // The way the emulator works causes Android to see the thread do a big
    // CPU burst followed by a sleep (due to the frame limiter). This makes
    // the OS see the thread sleeping for a long time, and puts it on slow
    // cores with reduced frequency, then when the next burst happens it
    // may do so in the slow core.
    // To prevent this we hint the OS that we are targetting half of what
    // the current frame time took so that it puts us in the beefy cores.
    constexpr double TARGET_RATIO = 0.5;

    const double frame_limit = Settings::GetFrameLimit();
    const double speed = frame_limit > 0 ? frame_limit / 100.0 : 1.0;
    const std::chrono::duration<double> frame_time{1.0 / (SCREEN_REFRESH_RATE * speed)};
    const auto target =
        std::chrono::duration_cast<std::chrono::nanoseconds>(frame_time * TARGET_RATIO);
    if (!performance_hint) {
        performance_hint = std::make_unique<PerformanceHintSession>(target);
    }
    performance_hint->ReportWork(system.perf_stats->GetLastSystemFrameTime(), target);
}

void EmuWindow_Android::OnFramebufferSizeChanged() {
    const bool is_portrait_mode = (AndroidUtils::IsPortraitMode() && !is_secondary);
    UpdateCurrentFramebufferLayout(window_width, window_height, is_portrait_mode);
}

EmuWindow_Android::EmuWindow_Android(ANativeWindow* surface, bool is_secondary)
    : EmuWindow{is_secondary}, host_window(surface) {
    LOG_DEBUG(Frontend, "Initializing EmuWindow_Android");
    if (!surface) {
        LOG_CRITICAL(Frontend, "surface is nullptr");
        return;
    }

    window_width = ANativeWindow_getWidth(surface);
    window_height = ANativeWindow_getHeight(surface);
}

EmuWindow_Android::~EmuWindow_Android() {
    DestroyWindowSurface();
    DestroyContext();
}

void EmuWindow_Android::MakeCurrent() {
    core_context->MakeCurrent();
}

void EmuWindow_Android::DoneCurrent() {
    core_context->DoneCurrent();
}
