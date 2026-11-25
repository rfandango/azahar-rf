// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include "video_core/frame_state_dump.h"

#ifdef CITRA_ENABLE_TRACE_MARKERS

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>
#include <android/log.h>
#include <sys/system_properties.h>
#include <unistd.h>
#include "common/diagnostics.h"
#include "common/logging/log.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/pica_types.h"

namespace VideoCore::FrameStateDump {

std::atomic_bool active{false};
std::atomic_bool times_gpu{false};
std::atomic_bool times_draws{false};

namespace {

constexpr const char* TRIGGER_PROPERTY = "debug.azahar.dump_frame";
constexpr const char* LOG_TAG = "AzaharFrameDump";
constexpr u32 VBLANKS_PER_DUMP = 5;
/// With GPU times a game frame is wanted that starts with the GPU idle, which the first is not
constexpr u32 VBLANKS_PER_TIMED_DUMP = 4;
constexpr std::string_view TIMED_PREFIX = "gpu";
constexpr std::string_view TIMED_DRAWS_PREFIX = "gpud";

std::mutex mutex;
std::string buffer;
std::string request;
/// What the trigger property held when it was last looked at
std::string last_property;
bool property_initialized = false;
/// When the game's first VBlank was, and how many of the dumps asked for ahead of time are done
std::optional<std::chrono::steady_clock::time_point> first_vblank;
std::size_t next_scheduled = 0;

// Dumps taken unasked when a game runs below its speed (Common::Diagnostics::
// AutomaticFrameDumps): up to three a session, the first no sooner than 45 s into the game and
// two minutes between them, each after ten seconds below 95% of the speed.
constexpr u32 SPEED_WINDOW_VBLANKS = 120;
/// 120 VBlanks take 2.0 s at full speed
constexpr std::chrono::milliseconds SLOW_WINDOW{2105};
constexpr u32 SLOW_STRETCHES = 5;
constexpr u32 AUTOMATIC_DUMPS = 3;
constexpr std::chrono::seconds AUTOMATIC_FIRST{45};
constexpr std::chrono::seconds AUTOMATIC_APART{120};
/// How soon after a dump that caught nothing the next may be taken
constexpr std::chrono::seconds AUTOMATIC_AGAIN{20};
std::chrono::steady_clock::time_point speed_window_start;
u32 speed_vblanks = 0;
u32 slow_stretches = 0;
u32 automatic_dumps = 0;
std::chrono::steady_clock::time_point last_automatic;
u32 vblanks_left = 0;
u32 draw_index = 0;

std::string PackageName() {
    std::string name;
    if (std::FILE* file = std::fopen("/proc/self/cmdline", "r")) {
        char data[256]{};
        const std::size_t read = std::fread(data, 1, sizeof(data) - 1, file);
        std::fclose(file);
        name.assign(data, strnlen(data, read));
    }
    return name;
}

void Write() {
    // A dump taken unasked that caught nothing being drawn (a game still starting, a menu that
    // stands still) does not count as one of the session's few, and another may follow soon
    if (request.starts_with("gpu-slow-") &&
        buffer.find("vk renderpass begin") == std::string::npos && automatic_dumps > 0) {
        automatic_dumps--;
        last_automatic -= AUTOMATIC_APART - AUTOMATIC_AGAIN;
    }

    // Written by a thread of its own: this one is the game's, the file goes to storage that
    // may hold a write up for seconds, and the lines for the log are more than its queue
    // takes at once (the thread that puts them in waits for room).
    std::thread([text = std::move(buffer), name = request] {
        const std::string dir = fmt::format("/sdcard/Android/data/{}/files", PackageName());
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        const std::string path = fmt::format("{}/frame-dump-{}.txt", dir, name);
        if (std::FILE* file = std::fopen(path.c_str(), "w")) {
            std::fwrite(text.data(), 1, text.size(), file);
            std::fclose(file);
            __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "wrote %s (%zu bytes)", path.c_str(),
                                text.size());
        } else {
            __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "cannot write %s (%s)", path.c_str(),
                                std::strerror(errno));
        }

        // And into the emulator's log, which is the one file a user shares. Every line ends
        // in a newline; see Line().
        LOG_INFO(HW_GPU, "FrameDump begin {}", name);
        const std::string_view lines{text};
        for (std::size_t start = 0, end; (end = lines.find('\n', start)) != std::string_view::npos;
             start = end + 1) {
            LOG_INFO(HW_GPU, "FrameDump: {}", lines.substr(start, end - start));
        }
        LOG_INFO(HW_GPU, "FrameDump end {}", name);
    }).detach();
    buffer.clear();
}

template <typename T>
u32 Raw(const T& value) {
    u32 raw{};
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

} // Anonymous namespace

void SaveFile(const std::string& name, const void* data, std::size_t size) {
    const std::string dir = fmt::format("/sdcard/Android/data/{}/files/shaders", PackageName());
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string path = fmt::format("{}/{}", dir, name);
    if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
        std::fwrite(data, 1, size, file);
        std::fclose(file);
    } else {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "cannot write %s (%s)", path.c_str(),
                            std::strerror(errno));
    }
}

void Line(std::string text) {
    std::scoped_lock lock{mutex};
    buffer += text;
    buffer += '\n';
}

void Draw(const Pica::RegsInternal& regs, bool hardware_shader, bool is_indexed, u64 vs_hash) {
    using Pica::f24;
    const auto& fb = regs.framebuffer.framebuffer;
    const auto& om = regs.framebuffer.output_merger;
    const auto& rast = regs.rasterizer;
    const auto& lighting = regs.lighting;
    const auto& tex = regs.texturing;

    std::string out = fmt::format(
        "draw {} {} idx={} verts={} topo={} gs={} vs={:016X}@{:X} vsb={:04X}", draw_index++,
        hardware_shader ? "hw" : "sw", is_indexed, regs.pipeline.num_vertices,
        static_cast<u32>(regs.pipeline.triangle_topology.Value()),
        static_cast<u32>(regs.pipeline.use_gs.Value()), vs_hash, regs.vs.main_offset.Value(),
        regs.vs.bool_uniforms.Value());

    out +=
        fmt::format(" | fb c={:08X} cf={} d={:08X} df={} {}x{}", fb.GetColorBufferPhysicalAddress(),
                    static_cast<u32>(fb.color_format.Value()), fb.GetDepthBufferPhysicalAddress(),
                    static_cast<u32>(fb.depth_format.Value()), fb.GetWidth(), fb.GetHeight());

    const auto vp = rast.GetViewportRect();
    out += fmt::format(" | vp {},{}-{},{} sc m={} {},{}-{},{}", vp.left, vp.bottom, vp.right,
                       vp.top, static_cast<u32>(rast.scissor_test.mode.Value()),
                       rast.scissor_test.x1.Value(), rast.scissor_test.y1.Value(),
                       rast.scissor_test.x2.Value(), rast.scissor_test.y2.Value());

    out +=
        fmt::format(" | z test={} f={} wr={} wbuf={} scale={} off={}", om.depth_test_enable.Value(),
                    static_cast<u32>(om.depth_test_func.Value()), om.depth_write_enable.Value(),
                    rast.depthmap_enable == Pica::RasterizerRegs::DepthBuffering::WBuffering,
                    f24::FromRaw(rast.viewport_depth_range).ToFloat32(),
                    f24::FromRaw(rast.viewport_depth_near_plane).ToFloat32());

    out += fmt::format(
        " | mask={}{}{}{} atest={}/{}/{} blend={} eq={}/{} f={},{},{},{} lop={} st={} cull={} "
        "fop={}",
        om.red_enable.Value(), om.green_enable.Value(), om.blue_enable.Value(),
        om.alpha_enable.Value(), om.alpha_test.enable.Value(),
        static_cast<u32>(om.alpha_test.func.Value()), om.alpha_test.ref.Value(),
        om.alphablend_enable.Value(),
        static_cast<u32>(om.alpha_blending.blend_equation_rgb.Value()),
        static_cast<u32>(om.alpha_blending.blend_equation_a.Value()),
        static_cast<u32>(om.alpha_blending.factor_source_rgb.Value()),
        static_cast<u32>(om.alpha_blending.factor_dest_rgb.Value()),
        static_cast<u32>(om.alpha_blending.factor_source_a.Value()),
        static_cast<u32>(om.alpha_blending.factor_dest_a.Value()),
        static_cast<u32>(om.logic_op.Value()), om.stencil_test.enable.Value(),
        static_cast<u32>(rast.cull_mode.Value()),
        static_cast<u32>(om.fragment_operation_mode.Value()));
    if (om.stencil_test.enable) {
        out += fmt::format(" sten f={} ref={:02X} in={:02X} wr={:02X} ops={}{}{}",
                           static_cast<u32>(om.stencil_test.func.Value()),
                           om.stencil_test.reference_value.Value(),
                           om.stencil_test.input_mask.Value(), om.stencil_test.write_mask.Value(),
                           static_cast<u32>(om.stencil_test.action_stencil_fail.Value()),
                           static_cast<u32>(om.stencil_test.action_depth_fail.Value()),
                           static_cast<u32>(om.stencil_test.action_depth_pass.Value()));
    }

    const u32 num_lights = lighting.disable ? 0 : lighting.max_light_index + 1;
    out += fmt::format(" | light n={} cfg={} bump={} shadow={} c0={:08X} c1={:08X}", num_lights,
                       static_cast<u32>(lighting.config0.config.Value()),
                       static_cast<u32>(lighting.config0.bump_mode.Value()),
                       lighting.config0.enable_shadow.Value(), Raw(lighting.config0),
                       lighting.config1.raw);
    for (u32 i = 0; i < num_lights; i++) {
        const auto& light = lighting.light[i];
        out += fmt::format(" L{}[s0={:06X} s1={:06X} d={:06X} a={:06X} c={:X}]", i,
                           Raw(light.specular_0), Raw(light.specular_1), Raw(light.diffuse),
                           Raw(light.ambient), Raw(light.config) & 0xF);
    }

    const auto textures = tex.GetTextures();
    for (u32 i = 0; i < textures.size(); i++) {
        if (!textures[i].enabled) {
            continue;
        }
        const auto& config = textures[i].config;
        out += fmt::format(
            " | t{} {:08X} {}x{} f={} type={} wrap={},{} filt={},{}", i,
            config.GetPhysicalAddress(), config.width.Value(), config.height.Value(),
            static_cast<u32>(textures[i].format), static_cast<u32>(config.type.Value()),
            static_cast<u32>(config.wrap_s.Value()), static_cast<u32>(config.wrap_t.Value()),
            static_cast<u32>(config.mag_filter.Value()),
            static_cast<u32>(config.min_filter.Value()));
    }

    out += fmt::format(" | proctex={} fog={} tev", tex.main_config.texture3_enable.Value(),
                       static_cast<u32>(tex.fog_mode.Value()));
    for (const auto& stage : tex.GetTevStages()) {
        out += fmt::format(" {:X}:{:X}:{:X}:{:X}", stage.sources_raw, stage.modifiers_raw,
                           stage.ops_raw, stage.scales_raw);
    }

    Line(std::move(out));
}

void OnGameStart() {
    Common::Diagnostics::OnGameStart();
    // When the last dump taken unasked was is kept: two are never close to each other
    first_vblank.reset();
    next_scheduled = 0;
    speed_vblanks = 0;
    slow_stretches = 0;
    automatic_dumps = 0;
}

void OnVBlank() {
    Common::Diagnostics::OnVBlank();
    if (IsActive()) {
        std::scoped_lock lock{mutex};
        if (--vblanks_left == 0) {
            active.store(false, std::memory_order_relaxed);
            times_gpu.store(false, std::memory_order_relaxed);
            times_draws.store(false, std::memory_order_relaxed);
            Write();
            buffer.clear();
        } else {
            buffer += "=== vblank\n";
        }
        return;
    }

    // A dump asked for ahead of time (dump_frames_at in diag.txt), by the time since the game's
    // first VBlank; else one asked for now by changing the property
    std::string wanted;
    const auto now = std::chrono::steady_clock::now();
    if (!first_vblank) {
        first_vblank = now;
        speed_window_start = now;
    }

    // The pace of the emulation, a stretch of VBlanks at a time. A pause makes one stretch
    // long and no more.
    if (++speed_vblanks == SPEED_WINDOW_VBLANKS) {
        slow_stretches = now - speed_window_start > SLOW_WINDOW ? slow_stretches + 1 : 0;
        speed_vblanks = 0;
        speed_window_start = now;
    }

    const auto& times = Common::Diagnostics::FrameDumpTimes();
    if (next_scheduled < times.size() &&
        now - *first_vblank >= std::chrono::seconds{times[next_scheduled]}) {
        wanted = fmt::format("gpu-at-{}s", times[next_scheduled]);
        next_scheduled++;
    } else if (Common::Diagnostics::AutomaticFrameDumps() && automatic_dumps < AUTOMATIC_DUMPS &&
               slow_stretches >= SLOW_STRETCHES && now - *first_vblank >= AUTOMATIC_FIRST &&
               (automatic_dumps == 0 || now - last_automatic >= AUTOMATIC_APART)) {
        // A game that has been running below its speed for a while: what a frame of it costs
        // the GPU, pass by pass, is what whoever gets the log needs. A game at its speed is
        // left alone, as a dump holds it up for a moment.
        wanted = fmt::format(
            "gpu-slow-{}s",
            std::chrono::duration_cast<std::chrono::seconds>(now - *first_vblank).count());
        automatic_dumps++;
        last_automatic = now;
        slow_stretches = 0;
    } else {
        const auto value = Common::Diagnostics::Property(TRIGGER_PROPERTY);
        if (!value) {
            property_initialized = true;
            return;
        }
        if (!property_initialized) {
            // Ignore a value left over from before this process started.
            property_initialized = true;
            last_property = *value;
            return;
        }
        if (last_property == *value) {
            return;
        }
        last_property = *value;
        wanted = *value;
    }

    std::scoped_lock lock{mutex};
    request = wanted;
    buffer.clear();
    buffer += "=== vblank\n";
    draw_index = 0;
    const bool timed = request.starts_with(TIMED_PREFIX);
    vblanks_left = timed ? VBLANKS_PER_TIMED_DUMP : VBLANKS_PER_DUMP;
    times_gpu.store(timed, std::memory_order_relaxed);
    times_draws.store(request.starts_with(TIMED_DRAWS_PREFIX), std::memory_order_relaxed);
    active.store(true, std::memory_order_relaxed);
}

} // namespace VideoCore::FrameStateDump

#endif
