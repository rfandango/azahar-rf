// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

// Trace markers for system-wide profilers (Perfetto / systrace).
//
// When built with ENABLE_PROFILING on Android, these emit ATrace sections and counters for the
// app, which show up in Perfetto captures that enable app tracing for it (atrace_apps). In every
// other build they compile to nothing.
//
// The amount of detail is selected once, at first use, through the Android system property
// `debug.azahar.trace`:
//   0 - disabled
//   1 - (default) per-frame work: emulated frames, GPU command lists, command buffer
//       submission and GPU waits, shader and pipeline compilation, surface transfers,
//       presentation and frame limiting, plus per-frame event counters.
//   2 - additionally per-draw, per-SVC and per-HLE-service-call sections. This is high volume
//       and noticeably perturbs timings; use it to understand structure, not to measure.

#include <cstddef>
#include <cstdint>

namespace Common::TraceMarkers {

/// Experimental behaviours that can be switched while a game runs, to measure their effect:
/// `adb shell setprop debug.azahar.experiment <sum of values>`. Polled once per VBlank.
/// Always zero outside profiling builds.
enum Experiment : std::uint32_t {
    LegacyDepthWrite = 1,          ///< Fragment shaders always write depth, as before
    SkipEffectDraws = 2,           ///< Drop unlit draws that test depth without writing it (glows,
                                   ///< particles and other blended effects)
    SkipLitDraws = 4,              ///< Drop draws that use fragment lighting
    SkipFlatDraws = 8,             ///< Drop unlit draws without depth test (2D overlays)
    DumpShaders = 16,              ///< Save shaders generated from now on next to the frame dumps
    SkipUnlitSolidDraws = 32,      ///< Drop unlit draws that test and write depth
    LegacyAttachments = 64,        ///< Bind only the attachments each draw uses, and start a new
                                   ///< render pass whenever the area drawn to changes, as before
    LegacyTexCopyGap = 128,        ///< Texture copies load their output gap first, as before
    GenericVertexShaders = 256,    ///< Vertex shaders test bool uniforms at run time, as before
    LegacyHiddenDisplay = 512,     ///< Render to the secondary window while it is hidden, as before
    LegacyFullScaleTargets = 1024, ///< Render targets that the game shrinks when it displays
                                   ///< them get the full resolution scale, as before
    LegacyLayoutReadback = 2048,   ///< Surfaces with a different layout exchange their contents
                                   ///< through guest memory, as before
    LegacyThreadPlacement = 4096,  ///< Android: the emulation thread may run on any core, as
                                   ///< before
    LogPresents = 8192,            ///< Log every image the display reports, for the hitch tracker
    LegacyPresentRepeats = 16384,  ///< Compose and present an image at every VBlank, also when
                                   ///< nothing on the screens changed, as before
    LegacyPresentMode = 32768,     ///< Queue images as with V-Sync off, the former default on
                                   ///< Android
    NoPresentTimes = 65536,        ///< Do not ask the display for a time for each image
    NoDisplayRateLock = 131072,    ///< Do not keep emulated VBlanks in step with the display
    NoJitCodeCache = 262144,       ///< Do not keep recompiled code for reuse. Read when a title
                                   ///< starts or a save state is loaded.
    LegacyRightEyeSkip = 524288,   ///< "Disable right eye render" goes by the order of command
                                   ///< lists and transfers, as before, not by which results
                                   ///< reach the screen
    LegacyAudioOutput = 1048576,   ///< The audio device is handed what is queued as it comes,
                                   ///< and the last sample again when nothing is, as before
    InjectHitches = 2097152,       ///< Hold the emulation up for 30 ms once a second, to see how
                                   ///< what depends on its pace copes with frames that come late
    RealtimeAudio = 4194304,       ///< As the "Enable Realtime Audio" setting: sound is produced in
                                   ///< step with real time, however slowly the game runs
    LegacyFrameHandoff = 8388608,  ///< Vulkan: the emulation waits for the render thread at every
                                   ///< VBlank, and images are handed to the display on the queue
                                   ///< rendering is submitted to, as before
    LegacyOpaqueBlend = 16777216,  ///< Vulkan: a blend function that only copies the source is
                                   ///< passed to the driver as blending, as before. Read when a
                                   ///< pipeline is built, so set it before the game starts.
    LegacyScissor = 33554432,      ///< Vulkan: fragment shaders test the scissor that keeps what is
                                   ///< inside a rectangle and drop what is outside, as before
    /// Shadow maps are rendered at the resolution scale, as before
    LegacyShadowMapScale = 67108864,
    /// Vulkan: all the arithmetic of fragment shaders is marked as needing no more than half
    /// precision, to measure what that is worth. Renders incorrectly.
    RelaxedShaderPrecision = 134217728,
    /// Fragment shaders leave out fragment lighting and use a fixed colour in its place, to
    /// measure what the lighting of a lit fragment costs. Renders incorrectly.
    SkipLightingMath = 268435456,
    /// Fragment shaders work out the level of detail for every texture sample, also of
    /// textures that have one level only, as before
    LegacyTextureLod = 536870912,
    /// Vulkan: every RGBA8 surface is created with storage usage and an R32 view format, in
    /// case it becomes a shadow map, as before. Read when a surface is created, so set it
    /// before the game starts.
    LegacyStorageUsage = 1073741824,
    /// Vulkan: a draw that writes no colour keeps the fragment shader of its lighting and
    /// combiner state, as before
    LegacyColourlessShader = 2147483648,
};

/// More of the same, in a property of their own (`debug.azahar.experiment2`) now that the
/// first has no bits left.
enum Experiment2 : std::uint32_t {
    /// A game that renders faster than the display shows is left to, as before
    LegacyRenderPace = 1,
    /// Work nobody sees: copies, transfers and fills that come up more than once are told
    /// apart by counting from where the frame began, as before
    LegacyWorkNaming = 2,
    /// A title that asks for the New 3DS's 804 MHz processor clock gets as much emulated CPU
    /// time as one for the Old 3DS, as before. Read when the title starts.
    LegacyProcessorClock = 4,
    /// When the emulation is late with the sound, the device goes without until it comes, as
    /// before: no frames are mixed ahead of it
    NoAudioAhead = 8,
    /// The audio device is closed and opened again each time settings are applied anew, as
    /// before
    LegacySinkReload = 16,
};

/// A hold-up of the emulation asked for through `debug.azahar.stall`, as
/// "<milliseconds>/<every so many VBlanks>": "150/180" holds it up for 150 ms every three
/// seconds, "12/1" makes a game that runs slow all the time. For seeing how what depends on
/// the emulation's pace, the sound above all, copes with it. Zero when none is asked for.
struct InjectedStall {
    std::uint32_t milliseconds;
    std::uint32_t every;
};

#ifdef CITRA_ENABLE_TRACE_MARKERS
/// Currently active Experiment flags.
std::uint32_t Experiments();
/// Currently active Experiment2 flags.
std::uint32_t Experiments2();
/// The hold-up currently asked for.
InjectedStall Stall();
#else
constexpr std::uint32_t Experiments() {
    return 0;
}
constexpr std::uint32_t Experiments2() {
    return 0;
}
constexpr InjectedStall Stall() {
    return {};
}
#endif

} // namespace Common::TraceMarkers

#ifdef CITRA_ENABLE_TRACE_MARKERS

#include <type_traits>

namespace Common::TraceMarkers {

enum class Level : int {
    Off = 0,
    Frame = 1,
    Detail = 2,
};

/// Events counted per emulated frame and emitted as `azahar.*` counter tracks by EndFrame().
enum class Counter : std::size_t {
    HwDraws,           ///< Draw batches whose vertex shading runs on the host GPU
    SwDraws,           ///< Draw batches whose vertex shading runs on the CPU
    SwVertices,        ///< Vertices shaded on the CPU (shader JIT or interpreter)
    CommandLists,      ///< PICA command lists processed
    ShaderCompiles,    ///< Host shader modules generated and compiled
    ShaderJitCompiles, ///< PICA shaders compiled by the CPU shader JIT
    PipelineBuilds,    ///< Host graphics pipelines built by the driver
    SurfaceUploads,    ///< Guest memory -> host texture uploads
    SurfaceDownloads,  ///< Host texture -> guest memory readbacks
    SurfaceCopies,     ///< Host texture -> host texture copies
    Submits,           ///< Command buffer submissions
    GpuWaits,          ///< Times a thread blocked waiting for GPU completion
    Count,
};

/// Returns true when the app is being traced and the configured level includes `level`.
bool IsEnabled(Level level);

void BeginSection(const char* name);
void EndSection();

/// These check IsEnabled(Level::Frame) themselves.
void BeginAsyncSection(const char* name, std::int32_t cookie);
void EndAsyncSection(const char* name, std::int32_t cookie);
void SetCounterIfEnabled(const char* name, std::int64_t value);

/// Adds to a per-frame counter. Cheap no-op when not tracing.
void Increment(Counter counter, std::uint64_t amount = 1);

/// Marks the end of an emulated frame: emits and resets the per-frame counters.
void EndFrame();

class [[nodiscard]] ScopedSection {
public:
    ScopedSection(Level level, const char* name) : active{IsEnabled(level)} {
        if (active) {
            BeginSection(name);
        }
    }

    /// Takes a callable returning the name (a std::string), invoked only while tracing.
    template <typename MakeName>
        requires std::is_invocable_v<MakeName>
    ScopedSection(Level level, MakeName&& make_name) : active{IsEnabled(level)} {
        if (active) {
            BeginSection(make_name().c_str());
        }
    }

    ~ScopedSection() {
        if (active) {
            EndSection();
        }
    }

    ScopedSection(const ScopedSection&) = delete;
    ScopedSection& operator=(const ScopedSection&) = delete;

private:
    bool active;
};

} // namespace Common::TraceMarkers

#define CITRA_TRACE_CONCAT_IMPL(a, b) a##b
#define CITRA_TRACE_CONCAT(a, b) CITRA_TRACE_CONCAT_IMPL(a, b)

/// Section covering the rest of the enclosing scope, recorded at the default (frame) level.
#define CITRA_TRACE_SCOPE(name)                                                                    \
    const ::Common::TraceMarkers::ScopedSection CITRA_TRACE_CONCAT(citra_trace_scope_, __LINE__) { \
        ::Common::TraceMarkers::Level::Frame, name                                                 \
    }

/// Section covering the rest of the enclosing scope, recorded only at the detail level.
#define CITRA_TRACE_SCOPE_DETAIL(name)                                                             \
    const ::Common::TraceMarkers::ScopedSection CITRA_TRACE_CONCAT(citra_trace_scope_, __LINE__) { \
        ::Common::TraceMarkers::Level::Detail, name                                                \
    }

/// Frame-level section whose name is built with fmt::format, only while tracing. The including
/// file must provide fmt.
#define CITRA_TRACE_SCOPE_FMT(...)                                                                 \
    const ::Common::TraceMarkers::ScopedSection CITRA_TRACE_CONCAT(citra_trace_scope_, __LINE__) { \
        ::Common::TraceMarkers::Level::Frame, [&] { return fmt::format(__VA_ARGS__); }             \
    }

#define CITRA_TRACE_COUNT(counter)                                                                 \
    ::Common::TraceMarkers::Increment(::Common::TraceMarkers::Counter::counter)

#define CITRA_TRACE_COUNT_N(counter, amount)                                                       \
    ::Common::TraceMarkers::Increment(::Common::TraceMarkers::Counter::counter,                    \
                                      static_cast<std::uint64_t>(amount))

#define CITRA_TRACE_END_FRAME() ::Common::TraceMarkers::EndFrame()

/// Asynchronous section: may begin and end in different scopes; shown on its own track.
#define CITRA_TRACE_ASYNC_BEGIN(name, cookie)                                                      \
    ::Common::TraceMarkers::BeginAsyncSection(name, static_cast<std::int32_t>(cookie))
#define CITRA_TRACE_ASYNC_END(name, cookie)                                                        \
    ::Common::TraceMarkers::EndAsyncSection(name, static_cast<std::int32_t>(cookie))

/// Sets a counter track to an absolute value.
#define CITRA_TRACE_SET_COUNTER(name, value)                                                       \
    ::Common::TraceMarkers::SetCounterIfEnabled(name, static_cast<std::int64_t>(value))

#else

#define CITRA_TRACE_SCOPE(name) static_cast<void>(0)
#define CITRA_TRACE_SCOPE_DETAIL(name) static_cast<void>(0)
#define CITRA_TRACE_SCOPE_FMT(...) static_cast<void>(0)
#define CITRA_TRACE_COUNT(counter) static_cast<void>(0)
#define CITRA_TRACE_COUNT_N(counter, amount) static_cast<void>(0)
#define CITRA_TRACE_END_FRAME() static_cast<void>(0)
#define CITRA_TRACE_ASYNC_BEGIN(name, cookie) static_cast<void>(0)
#define CITRA_TRACE_ASYNC_END(name, cookie) static_cast<void>(0)
#define CITRA_TRACE_SET_COUNTER(name, value) static_cast<void>(0)

#endif
