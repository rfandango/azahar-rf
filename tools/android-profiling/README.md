# Profiling Azahar on Android

Tools for finding out why a game runs slowly on a particular Android device: a profiling build
that installs next to the release build, a system trace (Perfetto) with Azahar's own markers,
CPU sampling (simpleperf), and GPU counters.

All scripts are bash and run from Linux, macOS or WSL. They need `adb`, Python 3 and the
Android NDK (for simpleperf). The device must have USB or wireless debugging enabled; root is
not required.

## The profiling build

`./build-install.sh` builds the `vanillaProfile` variant for arm64 and installs it.

| | release | profile | relWithDebInfo / debug |
|---|---|---|---|
| Package | `org.azahar_emu.azahar` | `org.azahar_emu.azahar.profile` | `org.azahar_emu.azahar.debug` |
| Launcher name | Azahar | Azahar Profile | Azahar |
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

The threads that matter, all in the `org.azahar_emu.azahar.profile` process:

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
  and `AZAHAR_GPU_COUNTERS=id,id,...` selects them.
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
    adb pull /sdcard/Android/data/org.azahar_emu.azahar.profile/files/frame-dump-<value>.txt

The dump lists, in order, every PICA command list, draw (render target, viewport, depth,
blend and lighting state, vertex shader, texture formats), memory fill, display transfer and
texture copy, together with what the renderer did for them: render passes begun and ended,
submissions, and every rasterizer cache copy, upload, reinterpretation and download. Things
to look for:

- `cache DOWNLOAD` lines: the emulation thread waited for the GPU there. The lines before it
  show which guest operation needed the data.
- The number of `vk renderpass begin` lines per frame. On tile-based GPUs each one costs a
  load and a store of the attachments, so a frame should have about as many as the game has
  render targets, not as many as it has state changes.
- How many times the scene is drawn per VBlank, and at which target sizes. Games that render
  more than one frame per VBlank are running uncapped (see the "Simulate 3DS GPU timings"
  setting).

## Experiment switches

`debug.azahar.experiment` holds a sum of the `Common::TraceMarkers::Experiment` values in
`src/common/trace_markers.h`. It is polled every VBlank, so behaviours can be compared in the
same build, in the same scene, without restarting:

    adb shell setprop debug.azahar.experiment 4     # e.g. drop every draw that uses lighting
    adb shell setprop debug.azahar.experiment 0     # back to normal

Some switches turn a new code path back into the old one for A/B measurements, some drop or
simplify work to measure what it costs (those render incorrectly on purpose), and
`DumpShaders` (16) saves every shader generated from then on under `files/shaders/`.

Two rules for trustworthy numbers:

- Switches that change shaders make the renderer build new pipelines, which can take seconds
  each on mobile drivers. Wait until they are done before capturing (a minute or more for a
  3D scene) and check that the capture contains no `Vulkan::BuildPipeline` or
  `Vulkan::Compile*` slices.
- Compare per game frame, not per second or per VBlank: divide by the slope of the
  `game_frame` counter. A change that speeds the game up otherwise looks like more work.
