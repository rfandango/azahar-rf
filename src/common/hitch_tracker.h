// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

// Tracks hitches: game frames that take longer than the frames around them, on the emulation
// thread or on the screen, each with the most likely cause.
//
// The emulation thread accounts the wall time of every VBlank period to a Category through
// scopes placed where it works or waits. A game frame that takes clearly longer than the usual
// one is a hitch, and its cause is the category that grew the most, or the game itself if the
// frame only took more VBlanks of emulated time.
//
// Where the presentation backend can tell when an image really reached the display, the time
// each game frame stayed on screen is tracked as well. That catches frames that were ready on
// time but shown late.
//
// Tracking is off, and costs one atomic load per scope, until SetOverlayEnabled(true). Profiling
// builds always track.

#include <atomic>
#include <chrono>
#include <string>
#include "common/common_types.h"

namespace Common::Hitches {

/// Where the emulation thread spends the wall time of a frame.
enum class Category : u8 {
    Guest,            ///< Emulated CPU and system services: the rest of the thread's CPU time
    GpuEmulation,     ///< Emulating GPU commands
    TextureUpload,    ///< Decoding guest textures and uploading them
    ShaderCompile,    ///< Generating and compiling shaders and pipelines
    Compose,          ///< Composing the screens at VBlank
    GpuReadback,      ///< Reading surfaces back from the GPU
    GpuWait,          ///< Other waits for the GPU
    GpuBehind,        ///< Waiting for the GPU to finish earlier frames
    RenderThread,     ///< Waiting for the render thread
    PresentQueue,     ///< Waiting for a free presentation frame
    Limiter,          ///< Frame limiter sleep
    LimiterOvershoot, ///< Frame limiter sleeping for longer than it asked for
    OffCpu,           ///< Not running for no known reason: preempted, or blocked on I/O or a lock
    Count,
};

namespace Detail {
extern std::atomic_bool enabled;
} // namespace Detail

inline bool IsEnabled() {
    return Detail::enabled.load(std::memory_order_relaxed);
}

/// Turns tracking on for the overlay. Has no effect on profiling builds, which always track.
void SetOverlayEnabled(bool enabled);

/// Accounts the time until the end of the enclosing scope to a category. Nested scopes take
/// their time away from the enclosing one. Does nothing on threads other than the emulation
/// thread, unless created with AnyThread.
class [[nodiscard]] Scope {
public:
    enum class Mode : u8 {
        Exclusive, ///< Nested scopes account for themselves
        Absorb,    ///< Everything inside counts for this scope
        AnyThread, ///< Also counted on other threads, where it stands for time they are stalled
    };

    explicit Scope(Category category, Mode mode = Mode::Exclusive) {
        if (IsEnabled()) [[unlikely]] {
            Begin(category, mode);
        }
    }

    ~Scope() {
        if (state != State::Off) [[unlikely]] {
            End();
        }
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    enum class State : u8 { Off, Tracked, TrackedAbsorb, Foreign };

    void Begin(Category category, Mode mode);
    void End();

    std::chrono::steady_clock::time_point foreign_start;
    State state{State::Off};
};

/// Moves time of the current VBlank period from one category to another. Emulation thread only.
void MoveTime(Category from, Category to, std::chrono::nanoseconds time);

/// The game submitted a frame (top screen buffer swap). Emulation thread only.
void NoteGameFrame();

/// Number of the last frame the game submitted.
u32 CurrentGameFrame();

/// The work of a VBlank period is over and its image was handed to the presentation backend.
/// Called once per VBlank on the emulation thread, which is how that thread is recognised.
void EndVBlank();

/// Whether the presentation backend reports when images reached the display, the refresh
/// period of that display, and how images are queued for it (a string literal, for display).
void SetDisplayTiming(bool available, std::chrono::nanoseconds refresh_period,
                      const char* present_mode);

/// How images are being given their place on the display, for the summary.
struct Pacing {
    bool active;      ///< Each image is given a time to be displayed at
    bool locked;      ///< Emulated VBlanks are kept in step with the display's refreshes
    bool overloaded;  ///< Images keep coming too late for their refresh: shown as they come
    double speed;     ///< Emulation speed asked for, relative to the nominal one
    double lead_ms;   ///< How long before its refresh an image is handed over, on average
    double target_ms; ///< The lead aimed for
    u32 slips;        ///< Images handed over too late for their refresh
    u32 misses;       ///< Images handed over in time and still shown late
};
void SetPacing(const Pacing& pacing);

/// The image `present_id` (not zero) of the given game frame, handed over by the emulation
/// thread at `submitted`, is being queued for display.
void NotePresentQueued(u32 present_id, u32 game_frame,
                       std::chrono::steady_clock::time_point submitted);

/// A queued image reached the display at the given time (steady clock, nanoseconds), having
/// been ready `margin_ns` before the display had to take it. Images queued before it that were
/// never reported are taken as dropped.
void NotePresentShown(u32 present_id, u64 display_time_ns, u64 margin_ns);

/// Multi-line summary of recent hitches for an on-screen overlay.
std::string OverlayText();

} // namespace Common::Hitches

#define CITRA_HITCH_CONCAT_IMPL(a, b) a##b
#define CITRA_HITCH_CONCAT(a, b) CITRA_HITCH_CONCAT_IMPL(a, b)

/// Accounts the rest of the enclosing scope to a Common::Hitches::Category.
#define CITRA_HITCH_SCOPE(category)                                                                \
    const ::Common::Hitches::Scope CITRA_HITCH_CONCAT(citra_hitch_scope_, __LINE__) {              \
        ::Common::Hitches::Category::category                                                      \
    }

/// Same, with everything nested inside counting for this category too.
#define CITRA_HITCH_SCOPE_ABSORB(category)                                                         \
    const ::Common::Hitches::Scope CITRA_HITCH_CONCAT(citra_hitch_scope_, __LINE__) {              \
        ::Common::Hitches::Category::category, ::Common::Hitches::Scope::Mode::Absorb              \
    }

/// Same as CITRA_HITCH_SCOPE, also counted when another thread is the one that waits.
#define CITRA_HITCH_SCOPE_ANY_THREAD(category)                                                     \
    const ::Common::Hitches::Scope CITRA_HITCH_CONCAT(citra_hitch_scope_, __LINE__) {              \
        ::Common::Hitches::Category::category, ::Common::Hitches::Scope::Mode::AnyThread           \
    }
