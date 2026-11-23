# Profiling Azahar on Android

Tools for finding out why a game runs slowly on a particular Android device: a profiling build
that installs over the release build (one app, its data kept), a system trace (Perfetto) with Azahar's own markers,
CPU sampling (simpleperf), and GPU counters.

All scripts are bash and run from Linux, macOS or WSL. They need `adb`, Python 3 and the
Android NDK (for simpleperf). The device must have USB or wireless debugging enabled; root is
not required.

## The profiling build

`./build-install.sh` builds the `vanillaProfile` variant for arm64 and installs it.

| | release | profile | relWithDebInfo / debug |
|---|---|---|---|
| Package | `io.github.rfandango.azahar` | `io.github.rfandango.azahar` | `io.github.rfandango.azahar.debug` |
| Launcher name | Azahar RF | Azahar RF | Azahar RF |
| Native code | `RelWithDebInfo` (-O2, LTO) | same, plus frame pointers | same (debug: -O0) |
| Java/Kotlin | R8 | R8 | not optimized |
| Debuggable | no | no, but `profileable` from the shell | yes |
| Trace markers | no | yes | no |

The profile build runs the same native and Java code as release. It is not debuggable, so ART
and JNI behave exactly as in release. Being profileable lets simpleperf, Perfetto and Android
GPU Inspector attach on unrooted retail devices.

The `ENABLE_PROFILING` CMake option adds frame pointers (`-fno-omit-frame-pointer`, which
costs a few percent at most) and compiles in the trace markers from
`src/common/trace_markers.h`.

Packaging strips the native libraries in the APK as usual. The scripts take symbols from the
unstripped copies under `src/android/app/build/intermediates/merged_native_libs/`, so keep
the build tree that produced the installed APK.

It is a separate app with its own data. On first launch, point it at the same user folder
and games, and copy the settings you want to investigate. Its shader caches start empty, so
the first run of a game includes shader compilation; do one warm-up run before measuring
steady-state performance.

## Workflow

1. **Reproduce.** Start the game in Azahar Profile and go to the slow scene. Keep the device in
   the same state between runs: the same performance mode (many handhelds have a
   fan/performance toggle), not charging, and similar temperature. `device-info.txt` in every
   output directory records clocks, governor, thermal status and battery state.
2. **System trace:** `./perfetto.sh -d 10`. This shows which kind of bottleneck you have (see
   below). Open `trace.perfetto-trace` at <https://ui.perfetto.dev>. If the `perfetto`
   Python package is installed (`pip install perfetto`), a text summary is also printed and
   saved as `summary.txt`. You can also run `./analyze-trace.py <trace>` yourself.
3. **CPU-bound:** `./simpleperf.sh -d 15` shows where the CPU time goes. `report.html` has
   per-thread flame graphs, `symbols.txt` lists the hottest functions per thread, and
   `emu-thread.txt` is the call graph of the emulation thread.
4. **GPU-bound:** `./perfetto.sh -g` adds GPU counters and render stages. See
   [GPU analysis](#gpu-analysis).

Captures go to `build/android-profiles/<kind>-<timestamp>/` (ignored by git); `-o` overrides
this.

## Reading a system trace

The threads that matter, all in the `io.github.rfandango.azahar` process:

| Thread | Does |
|---|---|
| `NativeEmulation` | Everything guest-side: the ARM JIT, HLE kernel and services, PICA command processing, CPU vertex shading, texture decoding, recording Vulkan work |
| `VulkanWorker` | Turns recorded work into Vulkan commands and submits them (driver CPU cost) |
| `VulkanPresent` | Copies finished frames to the swapchain and presents them |
| `Shader workers` / `Pipeline worker` | Shader compilation / pipeline builds (one thread per two cores each) |

A game that can't keep up shows one of these patterns on `NativeEmulation`:

- **CPU-bound emulation:** the thread is *Running* almost all the time, `EmuFrame` slices are
  longer than 16.7 ms, and `FrameLimiter` slices are near zero. Continue with simpleperf. Also
  check which cores it runs on: time on the little cores, or low clocks on the big ones, is a
  scheduling or governor problem, not an emulator one.
- **Waiting for the GPU:** long `Vulkan::WaitGpu`, `LimitFramesAheadOfGpu`,
  `Vulkan::WaitPresentFrame` or `Vulkan::Finish` slices. `Surface::Download` (GPU to guest
  memory readbacks) forces such waits. Check the GPU clock and the GPU counters.
- **Compilation stutter:** spikes with `Vulkan::GenerateVS`, `Vulkan::WaitPipeline`,
  `Vulkan::WaitShader` or `ShaderJit::Compile`, and non-zero `azahar.shader_compiles` /
  `azahar.pipeline_builds` in those frames.
- **Starved of CPU time:** large *Runnable* time means other threads or processes hold the
  cores it wants.

### Markers

Default level (`debug.azahar.trace` unset or `1`). These are cheap enough to leave on while
measuring:

| Marker | Meaning |
|---|---|
| `EmuFrame` (async track) | Emulation work for one VBlank (1/59.83 s), excluding frame limiting and GPU throttling |
| `VBlank::SwapBuffers` | Per-VBlank presentation work on the emulation thread, including the two below |
| `FrameLimiter` | Sleeping to hold 100% speed: headroom |
| `LimitFramesAheadOfGpu` | Waiting for the GPU to be at most one guest frame behind |
| `GPU::CommandList` | Processing a PICA command list, including all its draws |
| `GPU::DisplayTransfer` / `GPU::TextureCopy` / `GPU::MemoryFill` | PICA transfer engine operations |
| `Surface::Upload` | Decoding a guest texture and uploading it |
| `Surface::Download` | Reading a GPU surface back into guest memory (usually stalls on the GPU) |
| `Vulkan::WaitGpu` | A thread blocked until the GPU finished earlier work |
| `Vulkan::Finish` | Submit, then wait for the GPU to finish |
| `Vulkan::WaitWorker` | Emulation thread waiting for `VulkanWorker` to drain its queue |
| `Vulkan::ExecuteChunk`, `vkQueueSubmit` | `VulkanWorker` recording and submitting commands |
| `Vulkan::WaitPresentFrame` | Emulation thread waiting for a free presentation frame |
| `Vulkan::CopyToSwapchain`, `vkAcquireNextImageKHR`, `vkQueuePresentKHR` | Presentation |
| `Vulkan::GenerateVS` | Translating a new PICA vertex shader to GLSL (emulation thread) |
| `Vulkan::CompileVS/FS/GS` | Compiling a host shader (worker threads) |
| `Vulkan::BuildPipeline`, `(cache probe)` | Building a pipeline / asking the driver's cache for it |
| `Vulkan::WaitPipeline`, `Vulkan::WaitShader` | Blocked on a compile that hasn't finished |
| `ShaderJit::Compile` | Compiling a PICA shader for CPU vertex shading |

Counters, reset every VBlank (shown as `azahar.*` tracks):

| Counter | Meaning |
|---|---|
| `draws_hw` / `draws_sw` | Draw batches with vertex shading on the GPU / on the CPU. CPU shading happens with geometry shaders or when hardware shaders are disabled, and is expensive |
| `vertices_sw` | Vertices shaded on the CPU |
| `command_lists` | PICA command lists processed |
| `shader_compiles`, `shader_jit_compiles`, `pipeline_builds` | Compilation activity |
| `surface_uploads`, `surface_downloads`, `surface_copies` | Rasterizer cache transfers |
| `submits`, `gpu_waits` | Command buffer submissions and blocking waits on the GPU |
| `game_frame` | Increments once per guest frame: its slope is the game's frame rate |

Detail level (`./perfetto.sh -l 2`, then restart the app) adds `PICA::DrawArrays`,
`PICA::SoftwareVertexShading`, `Surface::Copy`, `ARM::Run` (JIT time slices), every SVC by
name and every HLE service call as `<service>::<function>`. This adds thousands of markers
per frame and skews timings: use it to understand what happens inside a frame, not to
measure it. Set it back with `./perfetto.sh -l 1` (and restart).

## CPU profiles

`simpleperf.sh` samples on-CPU time (`cpu-clock`, user and kernel) for every thread of the
app, with frame-pointer call graphs.

On unrooted Android 10+ devices, simpleperf can only record after it sets
`security.perf_harden=0`, which it does by itself the first time it runs from the shell. This
relaxes the device's `perf_event_paranoid` restriction until the next reboot. The first
recording after a reboot can fail while that setting takes effect; just run it again.

Things to know when reading the results:

- Time in JIT-compiled code has no symbols. Guest ARM code (Dynarmic) and PICA shaders on the
  CPU path (shader JIT) appear as anonymous executable memory (`[anon:...]` or `//anon`), so
  large anonymous self time on `NativeEmulation` means guest code execution. Callees like
  `Memory::MemorySystem::Read32` are symbolized and show up as normal functions.
- Kernel time on `NativeEmulation` or `VulkanWorker` (for example in `kgsl_ioctl`) is driver
  work: submissions, waits, memory mapping.
- `-c dwarf` unwinds through code without frame pointers (system libraries, the GPU driver)
  at much higher overhead. `-c none` gives flat profiles only.

## GPU analysis

Start with the coarse view, which works everywhere: in the system trace, look at the GPU clock
(the `gpufreq` track, plus kgsl tracks on Adreno) next to the `Vulkan::WaitGpu` and
`Vulkan::WaitPresentFrame` slices. A GPU at its maximum clock while the CPU waits on it means
GPU-bound. A GPU at low clocks while the CPU waits points at latency or synchronization
instead: readbacks, `Finish`, or too little work in flight for the governor to ramp up.

For more detail:

- **GPU counters and render stages in the system trace:** `./perfetto.sh -g`. This uses
  Android GPU Inspector's counter producer, so AGI must be installed on the device (install it
  once with the AGI desktop app). Render stages show each render pass on a GPU timeline.
  Counters show utilization, shader ALU and texture load, bandwidth and so on, depending on
  the GPU. Counter IDs differ per GPU; `./gpu-producer.sh list` prints what the device offers,
  and `AZAHAR_GPU_COUNTERS=id,id,...` selects them. Two things to know on Adreno: the game
  has to be started after the producer for its render stages to arrive, and they stop
  arriving once a save state is loaded (the renderer is created anew and the driver's
  profiler does not follow). For a scene that is reached through a save state, use the
  counters and a timed frame dump (below) instead.
- **What the counters say a frame costs:** divide the rates by the game's frame rate. Memory
  read and written per frame (`Read Total`, `Write Total`, `Texture Memory Read BW`) next to
  `% Stalled on System Memory`, `% Texture Fetch Stall` and `% Shader ALU Capacity Utilized`
  tell a frame that is short of arithmetic from one that is short of memory bandwidth. At
  high resolution scales it is often the latter: every full-screen pass reads and writes
  the whole target, and so does every copy the renderer makes on top.
- **Android GPU Inspector (desktop app):** its System Profiler shows the same data with a GUI.
  Its Frame Profiler shows per-draw costs, but needs a debuggable app: use the
  `relWithDebInfo` build (`.debug` package), whose native code matches release.
- **Snapdragon Profiler** (Adreno) and **Arm Performance Studio** (Mali) for vendor counters.
- **RenderDoc** for inspecting what a frame renders, with the `relWithDebInfo` build.

Emulator settings worth A/B testing for GPU-bound games: internal resolution, texture
filtering, the graphics API (Vulkan/OpenGL), a custom driver (e.g. Turnip on Adreno),
SPIR-V shader generation, accurate multiplication, and asynchronous shader compilation.

## What a frame does: frame dumps

Counters say how much work a frame is; a frame dump says what the work is. Changing the
`debug.azahar.dump_frame` property to any new value makes the profiling build write the next
two VBlanks as text to the app's external files directory:

    adb shell setprop debug.azahar.dump_frame $(date +%s)
    adb pull /sdcard/Android/data/io.github.rfandango.azahar/files/frame-dump-<value>.txt

The dump lists, in order, every PICA command list, draw (render target, viewport, depth,
stencil, blend and lighting state, vertex shader, texture formats), memory fill, display
transfer and texture copy, together with what the renderer did for them: render passes begun
and ended, submissions, and every rasterizer cache copy, upload, reinterpretation and
download. A draw of two triangles at the most is followed by a `draw vertices` line: where
each vertex lands in the target, in the console's pixels, and its texture coordinates. That
shows what a full-screen pass covers and which part of a texture it samples.

With a value that starts with `gpu` the dump also has the GPU time of each piece of work:

    adb shell setprop debug.azahar.dump_frame gpu$(date +%s)     # per render pass
    adb shell setprop debug.azahar.dump_frame gpud$(date +%s)    # per draw

The renderer then waits for the GPU after every render pass, copy, transfer and fill (or
after every draw) and writes `gpu <ms>` lines, over four VBlanks instead of two. Each wait
adds a fixed 0.2 to 0.4 ms, which the lines for work that does nothing show; take that off.
The first VBlank also holds what the GPU still had to do from before. Per pass the times
add up to about the real frame. Per draw they do not: every draw is a pass of its own, and
the GPU lowers its clock over a frame that long, so compare draws that are near each other.

Things to look for:

- `cache DOWNLOAD` lines: the emulation thread waited for the GPU there. The lines before it
  show which guest operation needed the data.
- The number of `vk renderpass begin` lines per frame. On tile-based GPUs each one costs a
  load and a store of the attachments, so a frame should have about as many as the game has
  render targets, not as many as it has state changes. A pass goes on while draws stay
  within its area, and `vk renderpass does not load: ...` marks one that begins with a draw
  that overwrites all of its area (a quad that clears, a full-screen effect), for which
  nothing is loaded.
- `vk tex0 is the colour target` lines: the draw samples the target it draws to. If it
  overwrites every pixel of its area, the target `moves to its other image` and the draw
  renders there while sampling the first, at the cost of copying what lies outside the area.
  Otherwise it is `sampling a copy`, a `vk full-surface copy` of the whole target.
- How many times the scene is drawn per VBlank, and at which target sizes. Games that render
  more than one frame per VBlank are running uncapped (see the "Simulate 3DS GPU timings"
  setting).
- `left out: nobody sees its result` lines: with "Disable Right Eye Rendering" on and one eye
  on display, draws, copies, transfers and fills whose results never reach a displayed image
  or the game are not carried out (`FrameLiveness`, in `src/video_core/frame_liveness.cpp`).
  They are still listed, so a dump shows what the game asked for and what was done.

The log says what is being left out: `Leaving out work nobody sees: ...` when it starts,
`Unseen work: ...` every 300 game frames with the counts per kind, and `Work that was left
out turned out to be needed: ...` when a guess was wrong, after which everything is rendered
for a while. Experiment switch 524288 goes back to the former check, which only knew games
that render one eye after the other.

## Hitches and frame pacing

A game can run at full speed and still stutter. `Common::Hitches` (`src/common/hitch_tracker.h`)
tracks that, in every build while the hitch overlay is shown (in-game menu, Overlay Options,
"Show Hitch Overlay") and always in the profiling build. It follows two things:

- The emulation thread. Scopes account the wall time of each VBlank period to a category:
  emulated CPU, GPU emulation, texture upload, shader compilation, waits for the GPU, the
  render thread or a presentation frame, the frame limiter, and time the thread was not
  running for no known reason. A game frame that takes clearly longer than the usual one is a
  hitch. Its cause is the category that grew, or `game frame time` when the frame only took
  more VBlanks of emulated time, which the host cannot change.
- The screen (Vulkan, where `VK_GOOGLE_display_timing` exists). The display reports when each
  image was shown, so the tracker knows for how many refreshes each game frame stayed on
  screen. A frame that stays longer than usual is a hitch on screen: caused by the emulation
  if that delivered the next frame late, otherwise `shown late` or `image replaced` (the next
  image took its place in the queue before the display got to it).

The overlay and a `Hitch summary` in logcat every five seconds show the same five lines:

    hitches 10s: 65 on screen, 4 in emulation, worst +183 ms
    last 0.1s ago: on screen 50.1 ms (+16.7) over 3 refreshes, shown late
    causes 60s: shown late 239, gpu readback 5, emulated cpu 4
    screen 60.0 Hz mailbox: 29 fps, held 2x 57%, 3x 21%, 1x 20%, 59 dropped (0 new), latency 29 ms
    emu 29 fps, vblank 16.7 +-15.5 ms (max 39.0): cpu 3.7, gpu emu 1.7, readback 9.7, sleep 0.8

Where images are paced (see below) a `pacing` line follows the `screen` line:

    pacing: in step with the display, speed +0.27%, lead 23 ms (aim 23), 2 late, 0 missed

`held` is the share of game frames shown for that many refreshes: a steady game has one
value near 100%. `dropped` counts images the display never showed, `new` those that were the
first image of a game frame. `latency` is from the emulation thread handing an image over to
the display showing it. The `emu` line gives the VBlank period as the images were handed over,
with its deviation, and the time per VBlank in each category.

Every hitch is also logged with its numbers (`adb logcat | grep hitch_tracker`):

    Hitch: frame 3144 took 63.3 ms over 2 vblanks (usual 33.4 over 2): emulated cpu +37.9 ms | cpu 43.5, ...
    Hitch on screen: frame 1162 stayed 3 refreshes (50.0 ms, usual 2), next frame 1163 on time from emulation, shown late | ...

For the path of every single image, set experiment switch 8192 (`LogPresents`, see below) for
a few seconds: one line per image with when it was handed over, where in the display's refresh
cycle that was, and how long it took to be queued and shown.

In a system trace the same data is on the `azahar.frame_interval_us`, `azahar.frame_vblanks`
and `azahar.hitch_ms` counter tracks.

How images get to the screen (Vulkan with V-Sync on, the default):

- An image is only composed and queued at VBlanks where something on a screen changed, not
  at every VBlank.
- Each image is given the refresh it should be shown at, from the number of its VBlank
  (`VideoCore::FramePacer`), instead of being shown whenever the work of its VBlank happened
  to end.
- When the display's rate is within 1% of the console's (60 Hz against 59.83), the emulation
  runs at the display's rate and is kept a steady time ahead of it: `speed` is that change,
  `lead` how long before its refresh the latest of the recent images was handed over, `late`
  the images that came after their refresh was due and `missed` the ones that were on time
  and still shown late.

Experiment switches 16384, 32768, 65536 and 131072 turn these off one by one, for comparing
in the same scene. To get to a scene without touching the device, the profiling build loads a
save state when `debug.azahar.load_state` changes: `adb shell setprop debug.azahar.load_state 1`
(set it to 0 before the next request for the same slot).

## Audio

The last line of the hitch summary and overlay is about the audio output:

    audio: 17-70 ms waiting (usually 50), margin 17 ms, speed 100.0-100.6%, 0 gaps of 0 ms in
    all, 0 ms thrown away, asked for 8.6 ms every 8.6 ms (longest 20)

- `waiting` is how much sound was queued each time the audio device asked for some, over the
  last ten seconds. It swings by a video frame's worth, because the emulation produces sound
  in bursts.
- `margin` is what the output aims to still have at the lowest point of that swing, and
  `speed` how fast it played the queue to keep it there (`AudioCore::OutputPacer`). The speed
  follows the emulation's: sound is produced as emulated time passes, so a game at 90% speed
  makes 90% of the sound. With "Enable Realtime Audio", which is the default, the emulated
  sound hardware is clocked by real time instead and the speed stays near 100%.
- `gaps` are the times the queue ran out and the device got silence, with their total length.
- `asked for ... every ...` is the size and period of the device's requests, and `longest`
  the longest pause between two.

To look at the sound itself, any new value of `debug.azahar.record_audio` records the next
twelve seconds of what is handed to the device:

    adb shell setprop debug.azahar.record_audio $(date +%s)
    adb pull /sdcard/Android/data/io.github.rfandango.azahar/files/audio-<value>.raw

The file is raw 16 bit stereo at 32728 Hz. A stretch of identical samples in the middle of
sound is a gap that was filled by repeating the last sample, which is heard as a crack; a
stretch of zeros is one that was faded out and in.

Experiment switches for this: 1048576 hands the device what is queued as it comes, as before
the output was paced; 2097152 holds the emulation up for 30 ms once a second, to hear what a
late frame does; 4194304 forces "Enable Realtime Audio".

## Experiment switches

`debug.azahar.experiment` holds a sum of the `Common::TraceMarkers::Experiment` values in
`src/common/trace_markers.h`. It is polled every VBlank, so behaviours can be compared in the
same build, in the same scene, without restarting:

    adb shell setprop debug.azahar.experiment 4     # e.g. drop every draw that uses lighting
    adb shell setprop debug.azahar.experiment 0     # back to normal

Some switches turn a new code path back into the old one for A/B measurements, some drop or
simplify work to measure what it costs (those render incorrectly on purpose), and
`DumpShaders` (16) saves every shader generated from then on under `files/shaders/`.
`LegacyAttachments` (64) covers how render passes are kept going: with it a pass is restarted
whenever the attachments or the area drawn to change, always loads what its area held, and a
draw that samples its own target always gets a copy of all of it. `LegacyScissor`
(33554432) has fragment shaders test the scissor again; it changes shaders, so wait for the
pipelines to be rebuilt. `LegacyShadowMapScale` (67108864) renders shadow maps at the
resolution scale again instead of at the size the game gave them. `RelaxedShaderPrecision`
(134217728) marks all the arithmetic of fragment shaders as needing only half precision, which
renders incorrectly and is there to measure what precision costs on a GPU. `SkipLightingMath`
(268435456) leaves fragment lighting out of the shaders, to measure its share of a lit pixel.
`LegacyTextureLod` (536870912) has shaders work out the level of detail for every texture
sample again, also for textures with a single level.

Two rules for trustworthy numbers:

- Switches that change shaders make the renderer build new pipelines, which can take seconds
  each on mobile drivers. Wait until they are done before capturing (a minute or more for a
  3D scene) and check that the capture contains no `Vulkan::BuildPipeline` or
  `Vulkan::Compile*` slices.
- Compare per game frame, not per second or per VBlank: divide by the slope of the
  `game_frame` counter. A change that speeds the game up otherwise looks like more work.
