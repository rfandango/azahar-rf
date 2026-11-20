// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

// What it takes to get a new shader or pipeline to the point where a draw can use it, logged
// step by step in profiling builds: which thread did each step, how long it took and how long
// the work sat in a queue before. Compiles to nothing in other builds.

#include <chrono>

#ifdef __ANDROID__
#include <sched.h>
#endif

#include "common/logging/log.h"

namespace Vulkan::ShaderTiming {

using Clock = std::chrono::steady_clock;

/// Milliseconds from `start` until now
inline double MsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// Milliseconds from `start` until `end`
inline double MsBetween(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/// The core the calling thread is on, to tell work done on a slow core; -1 where unknown
inline int CurrentCore() {
#ifdef __ANDROID__
    return sched_getcpu();
#else
    return -1;
#endif
}

} // namespace Vulkan::ShaderTiming

#ifdef CITRA_ENABLE_TRACE_MARKERS
#define LOG_SHADER_TIMING(...) LOG_INFO(Render_Vulkan, __VA_ARGS__)
#else
#define LOG_SHADER_TIMING(...) static_cast<void>(0)
#endif
