// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

// One-shot text dump of everything the emulated GPU is asked to do in a frame: command lists,
// every draw with its PICA state, fills, transfers, surface cache traffic and host render pass
// boundaries, in order. Only available in profiling builds (ENABLE_PROFILING on Android).
//
// Trigger it by changing the system property `debug.azahar.dump_frame` to a new value while a
// game is running; the next two VBlank periods are written to
// /sdcard/Android/data/<package>/files/frame-dump-<value>.txt
//
// With a value that starts with "gpu" the dump covers four VBlank periods and also says how
// long the GPU took over each render pass, copy and transfer: the renderer waits for the GPU
// after each of them, so the frame itself takes much longer than usual meanwhile. With
// "gpud" it waits after every draw instead, which then is a render pass of its own: the
// times tell the draws of a pass apart, and add up to more than the pass takes as a whole.

#include "common/common_types.h"

#ifdef CITRA_ENABLE_TRACE_MARKERS

#include <atomic>
#include <string>
#include <fmt/format.h>

namespace Pica {
struct RegsInternal;
}

namespace VideoCore::FrameStateDump {

extern std::atomic_bool active;
extern std::atomic_bool times_gpu;
extern std::atomic_bool times_draws;

inline bool IsActive() {
    return active.load(std::memory_order_relaxed);
}

/// Whether the dump being recorded also wants the GPU time of each piece of work.
inline bool TimesGpu() {
    return times_gpu.load(std::memory_order_relaxed);
}

/// Whether it wants the GPU time of each draw on its own.
inline bool TimesDraws() {
    return times_draws.load(std::memory_order_relaxed);
}

/// Appends one line to the dump being recorded.
void Line(std::string text);

/// Appends the PICA state of a draw call.
void Draw(const Pica::RegsInternal& regs, bool hardware_shader, bool is_indexed, u64 vs_hash);

/// Writes a file next to the frame dumps (used to save generated shaders for inspection).
void SaveFile(const std::string& name, const void* data, std::size_t size);

/// Called once per emulated VBlank: finishes a dump in progress or starts a requested one.
void OnVBlank();

/// A game starts (or a state was loaded, which makes the emulated GPU anew): the dumps that are
/// timed from a game's start, and the count of those taken unasked, begin again. Without it
/// the second game of a session got its first dump a second in, and the third got none.
void OnGameStart();

} // namespace VideoCore::FrameStateDump

#define CITRA_FRAME_DUMP(...)                                                                      \
    do {                                                                                           \
        if (::VideoCore::FrameStateDump::IsActive()) [[unlikely]] {                                \
            ::VideoCore::FrameStateDump::Line(fmt::format(__VA_ARGS__));                           \
        }                                                                                          \
    } while (0)

#define CITRA_FRAME_DUMP_VBLANK() ::VideoCore::FrameStateDump::OnVBlank()
#define CITRA_FRAME_DUMP_GAME_START() ::VideoCore::FrameStateDump::OnGameStart()

/// In a dump with GPU times: waits until the GPU is done with everything recorded so far and
/// notes how long that took, under the description given.
#define CITRA_FRAME_DUMP_GPU_TIME(scheduler, ...)                                                  \
    do {                                                                                           \
        if (::VideoCore::FrameStateDump::TimesGpu()) [[unlikely]] {                                \
            (scheduler).TimeGpuWork(fmt::format(__VA_ARGS__));                                     \
        }                                                                                          \
    } while (0)

/// The same after a draw, in a dump that times every draw on its own.
#define CITRA_FRAME_DUMP_GPU_TIME_OF_DRAW(scheduler)                                               \
    do {                                                                                           \
        if (::VideoCore::FrameStateDump::TimesDraws()) [[unlikely]] {                              \
            (scheduler).TimeGpuWork("draw");                                                       \
        }                                                                                          \
    } while (0)

#else

#define CITRA_FRAME_DUMP(...) static_cast<void>(0)
#define CITRA_FRAME_DUMP_VBLANK() static_cast<void>(0)
#define CITRA_FRAME_DUMP_GAME_START() static_cast<void>(0)
#define CITRA_FRAME_DUMP_GPU_TIME(scheduler, ...) static_cast<void>(0)
#define CITRA_FRAME_DUMP_GPU_TIME_OF_DRAW(scheduler) static_cast<void>(0)

#endif
