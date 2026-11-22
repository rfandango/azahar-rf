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
    /// Vulkan: a pipeline is built in one piece, its shader stages compiled together, as
    /// before, also where the driver can build the stages one by one and keep each. Read
    /// when a pipeline is first asked for.
    MonolithicPipelines = 32,
    /// Vulkan, with shaders compiled in the background: every draw of two triangles at the
    /// most holds the render thread up until its pipeline is built, as before
    LegacySmallDrawWait = 64,
    /// Vulkan: a pipeline put together from separately built stages is drawn with for
    /// good; it is not built in one piece afterwards. On an Adreno 650 with Turnip the GPU
    /// hung in two of eight runs of ninety seconds that way, so this is for looking into
    /// that hang. Read when a pipeline is built.
    KeepPipelinesFromParts = 128,
    /// Vulkan: a pipeline whose shaders are still being compiled is not queued for building
    /// until a later draw asks for it again, as before
    LegacyPipelineQueueing = 256,
    /// A title that asks for the New 3DS's 804 MHz processor clock gets all of that emulated
    /// CPU time whether the host can deliver it in real time or not, as before
    FixedProcessorClock = 512,
    /// Work nobody sees: a draw counts as writing and reading all of its target and all of
    /// each texture, as before, not only the rows its viewport and scissor confine it to
    LegacyWholeTargetWork = 1024,
    /// Vulkan: a fill that is brought to a target as a draw gets it ready is a clear by
    /// itself, which the render pass of the draw then loads, as before; the pass does not
    /// do it as it begins
    LegacyClearBeforePass = 8192,
    /// Fragment shaders round every combiner stage's result and the final colour to 1/255
    /// also where the value is rounded already, as before. Shaders made with this are kept
    /// apart and in memory only.
    LegacyStageRounding = 16384,
    /// Every system call is timed and every switch between emulated cores sets process, page
    /// table and recompiler anew, as before
    LegacyEmulationOverheads = 65536,
    /// The recompiler's dispatcher calls out for the block of every guest return and
    /// indirect branch, as before, without looking in its table of recent ones first. Read
    /// when a recompiler is set up.
    LegacyDispatcher = 131072,
    /// The recompiled code looks every address up in the page table, as before, also where
    /// it would reach the emulated memory through the host's own address translation, a
    /// second view of each emulated address space (Common::AddressSpaceView): Android on
    /// arm64. Read when the emulated system is set up.
    LegacyMemoryAccess = 262144,
    /// Every read the game makes on a page a texture or target is cached on is announced to
    /// the rasterizer, as before, also where it holds nothing newer than the emulated memory
    /// (VideoCore::RasterizerCache::NoteCpuRead)
    LegacyCachedPageReads = 4194304,
    /// Every draw looks its render target up in the cache, as before, also one into the
    /// target of the draw before it (VideoCore::RasterizerCache::GetFramebufferSurfaces)
    LegacyTargetLookup = 8388608,
    /// Every draw looks each of its textures up in the cache, as before, also one that was
    /// looked up this frame with the cache as it still is
    /// (VideoCore::RasterizerCache::GetTextureSurface)
    LegacyTextureLookup = 16777216,
    /// Vulkan: every draw has its pipeline looked up by the hash of all its state, as
    /// before, also one that asks for the pipeline the draw before it used
    LegacyPipelineLookup = 33554432,
    /// Vulkan: every draw takes a descriptor set for its textures and writes it, as before,
    /// also one whose texture units hold what the draw before it had
    LegacyTextureSets = 134217728,
    /// Every write the game makes on a page a surface is cached on is announced to the
    /// rasterizer, as before, also one that touches nothing a surface covers
    /// (Memory::MemorySystem::RasterizerSetSurfaceCells)
    LegacyWriteAnnouncements = 268435456,
    /// Vulkan: every draw has its shaders looked up from the registers and its index data
    /// scanned, as before, also one whose registers are as they were for a draw before it
    /// (RasterizerVulkan::DrawSite)
    LegacyDrawResolve = 536870912,
    /// The command lists, fills and transfers a game sends are decoded on the emulation
    /// thread, where the game hands them over, as before; not on a thread of their own while
    /// the game goes on (VideoCore::GPU::Submit). Read when the emulated GPU is set up.
    LegacyGpuDecoding = 1073741824,
    /// With the decoder thread: the game gets the interrupt that says a command is done when
    /// the host is done with it, not when it hands the command over. As it is otherwise, the
    /// game is told at hand-over, with the delay the command at that place of its frame took
    /// the frame before, and the decoding goes on behind it: a game that waits for "done"
    /// before it goes on works on its next frame while the last is decoded. The bytes of a
    /// command list are copied at hand-over, and what the decoder reads or writes cannot be
    /// touched without waiting for it, or is copied as well where the game keeps writing it
    /// (Memory::MemorySystem::ArmGpuReferences). Read when the emulated GPU is set up.
    LateGpuCompletion = 2147483648,
    /// For measuring. Every command list, sent or called from one, is compared with what
    /// was at its address the time before, and every read and write the game makes on a
    /// cached page is looked up in the cache, for the lines "command lists", "lists called
    /// from them" and "of those" of the hitch summary. Costs a few percent of the emulation
    /// thread.
    MeasureCommandLists = 67108864,
    /// Vulkan: every draw copies its vertex and index data to a buffer of the renderer, as
    /// before, also where the emulated system's memory is a buffer of the host GPU as well,
    /// the same pages for both as on the console, and a draw takes the data from where the
    /// game has it (Vulkan::GuestMemory, Vulkan::InPlaceData): Android on arm64. Read when
    /// the emulated system and the renderer are set up.
    LegacyVertexCopies = 524288,
    /// For measuring. Vulkan: counts how much of the vertex and index data of its draws
    /// the game writes anew from frame to frame, and how much of it before the host GPU has
    /// drawn with it, which is what a draw that takes it in place would get wrong
    /// (Vulkan::GuestDataWatch). A line in the log every ten seconds.
    MeasureVertexData = 1048576,
    /// Vulkan: every readback of a small render target waits for the host GPU, as before,
    /// also one the game reads every frame (VideoCore::RasterizerCache::ServeLateReadbacks).
    LegacySmallReadbacks = 2097152,
    /// Vulkan, with multisampling: the fragment shader runs once a pixel, not once a
    /// sample. Read when a pipeline is built.
    NoSampleShading = 32768,
    /// For measuring, the picture may be wrong. Vulkan: no render pass loads what its
    /// attachments hold, which tells what the loads cost a GPU that renders in tiles
    NoAttachmentLoads = 2048,
    /// For measuring, the picture may be wrong. Vulkan: no render pass stores depth and
    /// stencil
    NoDepthStores = 4096,
};

/// A third word of them (`debug.azahar.experiment3`): the first two are full.
enum Experiment3 : std::uint32_t {
    /// A game that draws its scene twice, a part of a pixel apart, and blends the two images
    /// (its own anti-aliasing) has both passes drawn, as before, also above the console's
    /// resolution. See Pica::PicaCore::TwinPass.
    KeepSecondScenePass = 1,
    /// For finding out why a scene drawn twice is not taken for one: the first draw of a
    /// second pass that is not like its counterpart in the first is described in the log
    /// (a dozen times).
    ExplainScenePasses = 2,
    /// Vulkan: a blit between two surfaces is recorded when the game asks for it, as before,
    /// also one that fills all of its destination (Vulkan::TextureRuntime::BlitTextures).
    LegacyTransfers = 4,
    /// Vulkan: the swapchain's images are upright and the compositor turns them, as before,
    /// also where the display's panel lies turned (Vulkan::Swapchain::GetPreRotation). Read
    /// when a swapchain is made.
    LegacyDisplayRotation = 8,
    /// For seeing that the game does not wait for its storage: every write of the recompiled
    /// code that is kept (Core::JitCodeCache) is held for five seconds first, as a storage
    /// that stalls would.
    SlowCodeStorage = 16,
    /// The depth buffer is made ready with what its memory holds, as before, also for a draw
    /// that overwrites depth and stencil in all of its viewport: read back from the GPU where
    /// another surface has the newest bytes (RasterizerCache::ValidateSurface).
    LoadOverwrittenDepth = 32,
    /// A title is never given less than the Old 3DS's processor clock, as before, also where
    /// the host cannot run it in real time at that (Core::Timing::GovernProcessorClock).
    ConsoleClockFloor = 64,
    /// Vulkan: every draw copies its vertex and index data, as LegacyVertexCopies, but looked
    /// at with every draw, so that it can be turned on and off while a game runs
    /// (RasterizerVulkan::MayReadInPlace).
    CopiedVertexData = 128,
    /// Vulkan: a draw of a few vertices counts as writing all of the rectangle it is confined
    /// to, as before, not just the box around its triangles, so that the second image of its
    /// target is brought up to date in all of that rectangle (RasterizerVulkan::WrittenBy).
    WholeAreaWritten = 256,
    /// For seeing whether the model of the console's GPU is what sets a game's frame rate
    /// (Pica::DelayGenerator): the time it lets pass before a command list is reported done
    /// is a quarter of what it is otherwise.
    FasterConsoleGpu = 512,
    /// The second of two runs of draws that make the same image of the top screen, one for
    /// each eye, is drawn as before (Pica::PicaCore::RepeatedPass). ExplainScenePasses also
    /// has more lines written on why a second run was no repetition.
    KeepRepeatedPass = 1024,
    /// For seeing that neither the game nor the system log waits for the storage the log
    /// file is on: every write of the file is held up as a storage that lets 12 KiB a second
    /// through would hold it (Common::Log, the file backend's writer).
    SlowLogStorage = 2048,
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
/// Currently active Experiment3 flags.
std::uint32_t Experiments3();
/// The hold-up currently asked for.
InjectedStall Stall();
#else
constexpr std::uint32_t Experiments() {
    return 0;
}
constexpr std::uint32_t Experiments2() {
    return 0;
}
constexpr std::uint32_t Experiments3() {
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

/// Reads the experiment flags and the hold-up anew, as EndFrame() does: for what has changed
/// them before the first frame (Common::Diagnostics::LoadControlFile).
void RefreshSettings();

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
