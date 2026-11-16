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

inline bool IsActive() {
    return active.load(std::memory_order_relaxed);
}

/// Appends one line to the dump being recorded.
void Line(std::string text);

/// Appends the PICA state of a draw call.
void Draw(const Pica::RegsInternal& regs, bool hardware_shader, bool is_indexed, u64 vs_hash);

/// Writes a file next to the frame dumps (used to save generated shaders for inspection).
void SaveFile(const std::string& name, const void* data, std::size_t size);

/// Called once per emulated VBlank: finishes a dump in progress or starts a requested one.
void OnVBlank();

} // namespace VideoCore::FrameStateDump

#define CITRA_FRAME_DUMP(...)                                                                      \
    do {                                                                                           \
        if (::VideoCore::FrameStateDump::IsActive()) [[unlikely]] {                                \
            ::VideoCore::FrameStateDump::Line(fmt::format(__VA_ARGS__));                           \
        }                                                                                          \
    } while (0)

#define CITRA_FRAME_DUMP_VBLANK() ::VideoCore::FrameStateDump::OnVBlank()

#else

#define CITRA_FRAME_DUMP(...) static_cast<void>(0)
#define CITRA_FRAME_DUMP_VBLANK() static_cast<void>(0)

#endif
