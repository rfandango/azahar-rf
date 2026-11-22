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

#include <array>
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

namespace Detail {
void NoteGuestSlice(u32 core, bool idle, u32 thread, u32 priority, u64 ticks);
} // namespace Detail

/// A core of the emulated system spent `ticks` of its clock running the thread of the game
/// with the given ID and priority, or with no thread to run if `idle`. A game that takes more
/// VBlanks for a frame with one of its cores busy all the time has run out of emulated CPU time,
/// which no speed of the host makes up for. Emulation thread only.
inline void NoteGuestSlice(u32 core, bool idle, u32 thread, u32 priority, u64 ticks) {
    if (IsEnabled()) [[unlikely]] {
        Detail::NoteGuestSlice(core, idle, thread, priority, ticks);
    }
}

namespace Detail {
void NoteCommandListDone(u32 thread);
void NoteGuestThreadWaits(u32 thread);
} // namespace Detail

/// A thread of the game handed the GPU a list of commands, which the emulated GPU has now
/// worked through. From here until that thread next waits for something (see
/// NoteGuestThreadWaits) the emulated time that runs is counted: it is the work of the game
/// that a GPU emulated on a thread of its own could be done alongside. Emulation thread only.
inline void NoteCommandListDone(u32 thread) {
    if (IsEnabled()) [[unlikely]] {
        Detail::NoteCommandListDone(thread);
    }
}

/// A thread of the game gave up its core to wait for something other than the reply to a
/// call of a system service. Emulation thread only.
inline void NoteGuestThreadWaits(u32 thread) {
    if (IsEnabled()) [[unlikely]] {
        Detail::NoteGuestThreadWaits(thread);
    }
}

/// What the game's side asks of the emulated GPU in the course of a frame. Each is a point
/// where the two meet. With the GPU emulated on a thread of its own, the commands are handed
/// over and the game goes on; for the last three it has to wait until that thread has worked
/// off what it was handed before: a register it reads, and memory the GPU keeps a newer state
/// of (read) or has in use (written), which today comes to the rasterizer as FlushRegion and
/// InvalidateRegion.
enum class GpuCall : u32 {
    CommandList,
    Fill,
    Transfer, ///< Display transfers and texture copies
    OtherCommand,
    RegisterRead,
    MemoryRead,
    MemoryWrite,
    /// Of the reads: those of memory the host GPU holds a newer state of, which has to be
    /// fetched first
    MemoryReadNewer,
    /// Of the writes: those into what a cached surface covers
    MemoryWriteSurface,
    Count,
};

namespace Detail {
std::array<std::atomic<u32>, static_cast<std::size_t>(GpuCall::Count)>& GpuCalls();
} // namespace Detail

/// Counted for the summary, as so many a VBlank. Any thread.
inline void NoteGpuCall(GpuCall call) {
    if (IsEnabled()) [[unlikely]] {
        Detail::GpuCalls()[static_cast<std::size_t>(call)].fetch_add(1, std::memory_order_relaxed);
    }
}

/// Host pixels written without being drawn: what the game's transfers and copies, and the
/// renderer's own bookkeeping of surfaces, come to. Counted for the summary. Any thread.
enum class PixelMove : u32 {
    Blit,        ///< A transfer or copy that scales or converts
    Copy,        ///< A plain copy between two images
    Relayout,    ///< The same bytes read as rows of another length
    SecondImage, ///< Bringing a surface's second image up to date
    Clear,
    /// Not moved after all: a blit that waited and was given up
    Spared,
    Count,
};

namespace Detail {
std::array<std::atomic<u64>, static_cast<std::size_t>(PixelMove::Count)>& PixelMoves();
std::array<std::atomic<u32>, static_cast<std::size_t>(PixelMove::Count)>& PixelMoveCounts();
} // namespace Detail

inline void NotePixelsMoved(PixelMove kind, u64 pixels) {
    if (IsEnabled()) [[unlikely]] {
        Detail::PixelMoves()[static_cast<std::size_t>(kind)].fetch_add(pixels,
                                                                       std::memory_order_relaxed);
        Detail::PixelMoveCounts()[static_cast<std::size_t>(kind)].fetch_add(
            1, std::memory_order_relaxed);
    }
}

namespace Detail {
struct CommandListWords {
    std::atomic<u64> words{}; ///< In all lists
    std::atomic<u64> same{};  ///< Of them, as in the list last sent from the same address
    std::atomic<u32> lists{};
    std::atomic<u32> identical{}; ///< Lists that are the same throughout
};
CommandListWords& ListWords();
} // namespace Detail

/// A command list of so many 32-bit words was sent, of which `same` are what the list last
/// sent from the same address had in the same place. Tells how much of the decoding of a
/// list is decoding what was decoded a frame earlier. Any thread.
inline void NoteCommandListWords(u32 words, u32 same, bool identical) {
    if (IsEnabled()) [[unlikely]] {
        Detail::CommandListWords& counts = Detail::ListWords();
        counts.words.fetch_add(words, std::memory_order_relaxed);
        counts.same.fetch_add(same, std::memory_order_relaxed);
        counts.lists.fetch_add(1, std::memory_order_relaxed);
        counts.identical.fetch_add(identical ? 1 : 0, std::memory_order_relaxed);
    }
}

namespace Detail {
CommandListWords& CalledListWords();
} // namespace Detail

/// The same for a list that a command list calls into: a jump to another buffer of commands,
/// which a game may have built once and call every frame. Any thread.
inline void NoteCalledListWords(u32 words, u32 same, bool identical) {
    if (IsEnabled()) [[unlikely]] {
        Detail::CommandListWords& counts = Detail::CalledListWords();
        counts.words.fetch_add(words, std::memory_order_relaxed);
        counts.same.fetch_add(same, std::memory_order_relaxed);
        counts.lists.fetch_add(1, std::memory_order_relaxed);
        counts.identical.fetch_add(identical ? 1 : 0, std::memory_order_relaxed);
    }
}

/// What is counted of the draws by the place they are issued from, see NoteDrawSite
enum class DrawSite : u32 {
    Draws,
    Same,  ///< Every register as the last time a draw came from this place
    Fresh, ///< No draw came from this place before
    // In which registers the others differ (a draw can count in several)
    Rasterizer,
    Texturing,
    Framebuffer,
    Lighting,
    Geometry,
    GeometryShader,
    VertexShader,
    Count,
};

namespace Detail {
std::array<std::atomic<u64>, static_cast<std::size_t>(DrawSite::Count)>& DrawSites();
} // namespace Detail

/// Counted for the summary. Any thread.
inline void NoteDrawSite(DrawSite what) {
    Detail::DrawSites()[static_cast<std::size_t>(what)].fetch_add(1, std::memory_order_relaxed);
}

/// The game submitted a frame (top screen buffer swap). Emulation thread only.
void NoteGameFrame();

/// Number of the last frame the game submitted.
u32 CurrentGameFrame();

/// The work of a VBlank period is over and its image was handed to the presentation backend.
/// Called once per VBlank on the emulation thread, which is how that thread is recognised.
void EndVBlank();

/// The app paused the emulation, or let it go on. Any thread. The log says of every standstill
/// of the emulation whether the app asked for it.
void NotePaused(bool paused);

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
    double work_ms;   ///< Of it, the time allowed for an image to get through the GPU
    double ready_ms;  ///< How early the least ready of the recent images was, as reported
    u32 slips;        ///< Images handed over too late for their refresh
    u32 misses;       ///< Images handed over in time and still shown late
};
void SetPacing(const Pacing& pacing);

/// How busy the host's GPU was over the last stretch of time, and the resolution scale the
/// renderer draws at against the one the settings ask for.
struct HostGpu {
    float load;      ///< Share of the time it worked on what was sent to it, negative if unknown
    float waited;    ///< Share of the time the emulation waited for it to finish a frame
    u32 scale;       ///< Resolution scale in use
    u32 asked_scale; ///< Resolution scale of the settings
};
void SetHostGpu(const HostGpu& gpu);

/// The image `present_id` (not zero) of the given game frame, handed over by the emulation
/// thread at `submitted`, is being queued for display, to be shown at `desired_time_ns`
/// (steady clock, nanoseconds; zero for as soon as possible).
void NotePresentQueued(u32 present_id, u32 game_frame,
                       std::chrono::steady_clock::time_point submitted, u64 desired_time_ns = 0);

/// A queued image reached the display at the given time (steady clock, nanoseconds), having
/// been ready `margin_ns` before the display had to take it. Images queued before it that were
/// never reported are taken as dropped.
void NotePresentShown(u32 present_id, u64 display_time_ns, u64 margin_ns);

/// What the audio output did when the device asked it for sound, counted in frames (a sample
/// for each channel).
struct AudioOutput {
    u32 sample_rate; ///< Frames per second
    u32 wanted;      ///< Frames the device asked for
    u32 missing;     ///< Frames of those that were not there to give
    u32 buffered;    ///< Frames that were waiting when it asked
    u32 dropped;     ///< Frames thrown away afterwards to shorten the delay
    u32 ahead;       ///< Frames mixed ahead of the emulation, which was late with them
    u32 margin;      ///< Frames the output aims to have left at the least, zero if it has no aim
    float rate;      ///< Speed the sound was played at, relative to the nominal one
};
/// Called by the audio output each time the device asks for sound. May be called on any
/// thread and does not block.
void NoteAudioOutput(const AudioOutput& output);

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
