// Copyright 2015-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <chrono>
#include "common/hitch_tracker.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/hle/service/hid/hid.h"
#include "core/movie.h"
#include "core/tracer/recorder.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/frame_state_dump.h"
#include "video_core/renderer_base.h"

namespace VideoCore {

RendererBase::RendererBase(Core::System& system_, Frontend::EmuWindow& window,
                           Frontend::EmuWindow* secondary_window_)
    : system{system_}, render_window{window}, secondary_window{secondary_window_} {}

RendererBase::~RendererBase() = default;

namespace {

/// VBlanks the GPU's load is added up over before anything is made of it
constexpr u32 LOAD_WINDOW = 30;
/// For the model of the console's GPU time (RendererBase::ModelledGpuCostsFrames): the load
/// the GPU may have with a frame at every VBlank for the model to be reined in, the load
/// from which it is given its way again, and the stretches in a row either takes
constexpr float MODEL_ROOM_LOAD = 0.85f;
constexpr float MODEL_FULL_LOAD = 0.93f;
constexpr u32 MODEL_CHANGE_WINDOWS = 2;
/// Such stretches with the emulation behind, in a row of stretches with the GPU holding it
/// back, before the scale is lowered
constexpr u32 OVERLOADED_WINDOWS = 3;
/// Stretches in a row with room for the next scale, as far as the load tells, before it is
/// raised
constexpr u32 ROOMY_WINDOWS = 10;
/// Stretches in a row without the GPU holding anything back before the next scale is tried
/// although the load does not promise room for it: to begin with, and at the most. Doubled
/// each time a raise had to be taken back.
constexpr u32 CALM_WINDOWS = 60;
constexpr u32 MAX_CALM_WINDOWS = 1200;
/// A raise that lasted this many stretches was right
constexpr u32 PROVEN_WINDOWS = 40;
/// Stretches after a change that are not judged: they hold the cost of the change itself
constexpr u32 SETTLING_WINDOWS = 2;
/// Share of the time the GPU is at work from which it can be what holds the emulation back
constexpr float BUSY_LOAD = 0.85f;
/// Share of the time the emulation waits for the GPU from which it is held back by it
constexpr float HELD_BACK = 0.10f;
/// Share of the time the GPU may be at work at the next scale, going by the pixels. The host
/// lowers the GPU's clock when it has little to do, which makes the share measured too high
/// rather than too low.
constexpr float ROOM_LOAD = 0.80f;

/// How far the governor lowers the scale asked for, in whole factors: to half of it, and never
/// to the console's own resolution
constexpr u32 MostReduction(Common::ResolutionScale asked) {
    return asked <= Common::ResolutionScale::Whole(2)
               ? 0
               : asked.Steps() / (2 * Common::ResolutionScale::STEPS);
}

constexpr Common::ResolutionScale Reduced(Common::ResolutionScale asked, u32 reduction) {
    return Common::ResolutionScale::FromSteps(asked.Steps() -
                                              reduction * Common::ResolutionScale::STEPS);
}

} // Anonymous namespace

Common::ResolutionScale RendererBase::AskedResolutionScaleFactor() {
    const auto graphics_api = Settings::GetWorkingGraphicsAPI();
    if (graphics_api == Settings::GraphicsAPI::Software) {
        // Software renderer always render at native resolution
        return {};
    }

    const float scale_factor = Settings::values.resolution_factor.GetValue();
    return scale_factor != 0.f ? Common::ResolutionScale::Closest(scale_factor)
                               : render_window.GetFramebufferLayout().GetScalingRatio();
}

Common::ResolutionScale RendererBase::GetResolutionScaleFactor() {
    const Common::ResolutionScale asked = AskedResolutionScaleFactor();
    const u32 reduction = scale_reduction.load(std::memory_order_relaxed);
    return Reduced(asked, std::min(reduction, MostReduction(asked)));
}

void RendererBase::GovernResolution() {
    const bool governs = Settings::values.dynamic_resolution.GetValue();
    // Also followed for the model of the console's GPU time, see ModelledGpuCostsFrames
    const bool models = Settings::values.simulate_3ds_gpu_timings.GetValue();
    const auto busy = TakeGpuTime(governs || models || Common::Hitches::IsEnabled() ||
                                  gpu_time_wanted.load(std::memory_order_relaxed));
    last_vblank_gpu_ns.store(busy ? busy->count() : 0, std::memory_order_relaxed);
    if (!busy) {
        scale_reduction.store(0, std::memory_order_relaxed);
        load_window_open = false;
        modelled_gpu_costs_frames.store(false, std::memory_order_relaxed);
        return;
    }
    const u64 game_frames = system.perf_stats->GetGameFrameCount();
    const auto now = std::chrono::steady_clock::now();
    const auto restart = [&] {
        load_window_open = true;
        load_window_start = now;
        load_window_vblanks = 0;
        load_window_busy = {};
        load_window_waited = {};
        load_window_work = {};
        load_window_emulated = {};
        load_window_game_frames = game_frames;
    };
    if (!load_window_open) {
        // What was just taken is of a time that was not followed
        restart();
        return;
    }
    load_window_busy += *busy;
    load_window_work += system.frame_limiter.LastIntervalWork();
    load_window_emulated += system.frame_limiter.LastIntervalEmulated();
    if (++load_window_vblanks < LOAD_WINDOW) {
        return;
    }

    const double wall = std::chrono::duration<double>(now - load_window_start).count();
    const auto share = [wall](std::chrono::nanoseconds time) {
        return wall > 0.0 ? static_cast<float>(std::chrono::duration<double>(time).count() / wall)
                          : 0.0f;
    };
    const float load = share(load_window_busy);
    const float waited = share(load_window_waited);
    const double limit = Settings::GetFrameLimit() / 100.0;
    // Behind: the host worked for longer than the time it emulated, with nothing left for the
    // frame limiter to wait for. With the GPU setting the pace that is already too slow: a
    // game half a percent behind shows a frame twice every few seconds.
    const bool behind =
        limit > 0.0 && static_cast<double>(load_window_work.count()) >
                           static_cast<double>(load_window_emulated.count()) / limit * 1.003;
    // The game's frames a VBlank in this stretch: one for a game at the display's rate
    const float frame_rate =
        static_cast<float>(game_frames - load_window_game_frames) / static_cast<float>(LOAD_WINDOW);
    restart();

    // The model of the console's GPU time. It tells the game later that its drawing is done
    // the more there was to draw, and a game with a frame rate of its own then leaves out
    // frames, as it does on the console. Where the host's GPU has room for every frame, that
    // only costs frames (Monster Hunter 4 Ultimate at 50 a second on a GPU two thirds busy):
    // the model is then kept from reaching past a VBlank. Where the host has no room, the
    // model is what lets the game shed frames instead of the emulation falling behind. Room
    // is judged by what the GPU's load would be with a frame at every VBlank.
    if (models) {
        const bool costs = modelled_gpu_costs_frames.load(std::memory_order_relaxed);
        const float at_full_rate = load / std::clamp(frame_rate, 0.25f, 1.f);
        const bool speaks_for_change = costs ? (!behind && at_full_rate < MODEL_ROOM_LOAD)
                                             : (behind || load > MODEL_FULL_LOAD);
        model_change_windows = speaks_for_change ? model_change_windows + 1 : 0;
        if (model_change_windows >= MODEL_CHANGE_WINDOWS) {
            model_change_windows = 0;
            modelled_gpu_costs_frames.store(!costs, std::memory_order_relaxed);
            LOG_INFO(Render,
                     "Host GPU {:.0f}% busy with {:.2f} frames of the game a VBlank{}: the model "
                     "of the console's GPU time {}",
                     load * 100.f, frame_rate, behind ? ", the emulation behind" : "",
                     costs ? "is kept from costing the game VBlanks from now on"
                           : "sets the game's pace from now on");
        }
    } else {
        modelled_gpu_costs_frames.store(false, std::memory_order_relaxed);
        model_change_windows = 0;
    }

    const Common::ResolutionScale asked = AskedResolutionScaleFactor();
    const u32 most_reduction = MostReduction(asked);
    u32 reduction = std::min(scale_reduction.load(std::memory_order_relaxed), most_reduction);
    const float scale = Reduced(asked, reduction).Factor();
    Common::Hitches::SetHostGpu({
        .load = load,
        .waited = waited,
        .scale = scale,
        .asked_scale = asked.Factor(),
    });

    // Only at the console's own speed: faster than that, the GPU is meant to be the limit
    if (!governs || limit <= 0.0 || limit > 1.0) {
        if (!governs) {
            scale_reduction.store(0, std::memory_order_relaxed);
        }
        overloaded_windows = roomy_windows = calm_windows = 0;
        return;
    }
    if (settling_windows != 0) {
        settling_windows--;
        return;
    }
    if (raise_patience == 0) {
        raise_patience = CALM_WINDOWS;
    }
    if (raise_unproven && ++windows_since_raise >= PROVEN_WINDOWS) {
        raise_unproven = false;
        raise_patience = CALM_WINDOWS;
    }

    // The GPU holds the emulation back: the emulation spends its time waiting for the GPU to
    // finish frames, the GPU is at work nearly all the time, and the emulation is behind. At
    // the limit it is behind in some stretches and just in time in others, so a stretch in
    // which it only waits does not take back what the others counted.
    if (waited >= HELD_BACK && load >= BUSY_LOAD) {
        roomy_windows = calm_windows = 0;
        if (behind && ++overloaded_windows >= OVERLOADED_WINDOWS && reduction < most_reduction) {
            reduction++;
            scale_reduction.store(reduction, std::memory_order_relaxed);
            overloaded_windows = 0;
            settling_windows = SETTLING_WINDOWS;
            if (raise_unproven) {
                // The raise was too early: wait twice as long before the next one
                raise_unproven = false;
                raise_patience = std::min(raise_patience * 2, MAX_CALM_WINDOWS);
            }
            LOG_INFO(Render,
                     "Resolution scale lowered to {}x of {}x: the GPU was at work {:.0f}% of "
                     "the time and the emulation waited for it {:.0f}%",
                     Reduced(asked, reduction), asked, 100.0f * load, 100.0f * waited);
        }
        return;
    }
    overloaded_windows = 0;
    if (reduction == 0) {
        roomy_windows = calm_windows = 0;
        return;
    }

    // Back up: at once if the load leaves room for the pixels of the next scale, and
    // otherwise after a while without the GPU holding anything back, as a try
    const float pixels = ((scale + 1.f) * (scale + 1.f)) / (scale * scale);
    roomy_windows = load * pixels <= ROOM_LOAD ? roomy_windows + 1 : 0;
    calm_windows = waited < HELD_BACK ? calm_windows + 1 : 0;
    const bool room = roomy_windows >= ROOMY_WINDOWS;
    if (room || calm_windows >= raise_patience) {
        reduction--;
        scale_reduction.store(reduction, std::memory_order_relaxed);
        roomy_windows = calm_windows = 0;
        settling_windows = SETTLING_WINDOWS;
        windows_since_raise = 0;
        raise_unproven = true;
        LOG_INFO(Render,
                 "Resolution scale raised to {}x of {}x ({}): the GPU was at work {:.0f}% of "
                 "the time",
                 Reduced(asked, reduction), asked, room ? "room for it" : "as a try",
                 100.0f * load);
    }
}

u8 RendererBase::GetSampleCount() const {
    return Settings::GetAntiAliasingSampleCount(Settings::values.antialiasing.GetValue());
}

void RendererBase::UpdateCurrentFramebufferLayout(bool is_portrait_mode) {
    const auto update_layout = [is_portrait_mode](Frontend::EmuWindow& window) {
        const Layout::FramebufferLayout& layout = window.GetFramebufferLayout();
        window.UpdateCurrentFramebufferLayout(layout.width, layout.height, is_portrait_mode);
    };
    update_layout(render_window);
    if (secondary_window != nullptr) {
        update_layout(*secondary_window);
    }
}

namespace {

/// VBlanks without a known change after which an image is presented at every VBlank again. A
/// game that stopped swapping buffers may be drawing into the ones on display, and what else
/// goes into the image (settings, for one) is not followed either.
constexpr u32 MAX_UNCHANGED_VBLANKS = 6;

bool SameLayout(const Layout::FramebufferLayout& a, const Layout::FramebufferLayout& b) {
    return a.width == b.width && a.height == b.height &&
           a.top_screen_enabled == b.top_screen_enabled &&
           a.bottom_screen_enabled == b.bottom_screen_enabled && a.top_screen == b.top_screen &&
           a.bottom_screen == b.bottom_screen && a.is_rotated == b.is_rotated &&
           a.additional_screen_enabled == b.additional_screen_enabled &&
           a.additional_screen_is_bottom == b.additional_screen_is_bottom &&
           a.additional_screen == b.additional_screen && a.top_opacity == b.top_opacity &&
           a.bottom_opacity == b.bottom_opacity;
}

} // Anonymous namespace

bool RendererBase::ShouldPresentImage() {
    const bool screen_changed = screen_revision != presented_revision;
    presented_revision = screen_revision;
    unchanged_vblanks = screen_changed ? 0 : unchanged_vblanks + 1;

    const Layout::FramebufferLayout& layout = render_window.GetFramebufferLayout();
    bool layout_changed = !SameLayout(layout, presented_layout);
    presented_layout = layout;
    if (secondary_window) {
        const Layout::FramebufferLayout& secondary = secondary_window->GetFramebufferLayout();
        layout_changed |= !SameLayout(secondary, presented_secondary_layout);
        presented_secondary_layout = secondary;
    }

    const bool requested = settings.screenshot_requested || settings.bg_color_update_requested ||
                           settings.shader_update_requested;
    const bool legacy =
        Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyPresentRepeats;

    return screen_changed || layout_changed || requested ||
           unchanged_vblanks >= MAX_UNCHANGED_VBLANKS || legacy;
}

void RendererBase::EndFrame() {
    EndFrameParts(true, true);
}

void RendererBase::EndEmulatedFrame() {
    EndFrameParts(true, false);
}

void RendererBase::EndGpuFrame() {
    EndFrameParts(false, true);
}

void RendererBase::EndFrameParts(bool emulation, bool gpu) {
    if (emulation) {
        // One "EmuFrame" async slice per emulated VBlank period, excluding GPU throttling and
        // frame limiting below.
        CITRA_TRACE_ASYNC_END("EmuFrame", current_frame);
        CITRA_TRACE_END_FRAME();
        CITRA_FRAME_DUMP_VBLANK();
        Common::Hitches::EndVBlank();

        current_frame++;

        system.perf_stats->EndSystemFrame();

        render_window.PollEvents();
    }

    if (gpu) {
        // Wait for the GPU to finish and to be at most one game frame behind.
        // Below 60 FPS a guest frame spans several VBlanks, and its GPU work may still run
        // while the next one is emulated.
        const u64 game_frames = system.perf_stats->GetGameFrameCount();
        if (game_frames != limited_game_frames) {
            limited_game_frames = game_frames;
            if (Settings::values.use_limit_frames_ahead_of_gpu.GetValue() ||
                AlwaysLimitsFramesAheadOfGpu()) {
                CITRA_TRACE_SCOPE("LimitFramesAheadOfGpu");
                CITRA_HITCH_SCOPE_ABSORB(GpuBehind);
                if (load_window_open) {
                    const auto wait_start = std::chrono::steady_clock::now();
                    LimitFramesAheadOfGpu();
                    load_window_waited += std::chrono::steady_clock::now() - wait_start;
                } else {
                    LimitFramesAheadOfGpu();
                }
            }
        }
    }

    if (emulation) {
        {
            CITRA_TRACE_SCOPE("FrameLimiter");
            CITRA_HITCH_SCOPE_ABSORB(Limiter);
            system.frame_limiter.DoFrameLimiting(system.CoreTiming().GetGlobalTimeUs());
        }
        const Core::Movie::PlayMode movie = system.Movie().GetPlayMode();
        system.CoreTiming().GovernProcessorClock(
            system.frame_limiter.LastIntervalEmulated(), system.frame_limiter.LastIntervalWork(),
            system.frame_limiter.LastIntervalCpu(), Settings::GetFrameLimit() / 100.0,
            movie == Core::Movie::PlayMode::Recording || movie == Core::Movie::PlayMode::Playing);
    }
    if (gpu) {
        GovernResolution();
    }

    if (emulation) {
        // On fast enough devices, the time limiter sleep spans most of the 16ms period
        // between VBlank events. If an input event lands in that period, then the change
        // will not reflect in HID until the next HID event. For applications that only
        // check the input just after VBlank, it will cause them to lose the event until
        // the next frame. Sacrifice HID timing accuracy for input latency by triggering
        // the event just after VBlank (frame limiting behaviour is not accurate anyways).
        if (const auto hid = Service::HID::GetModule(system)) {
            hid->RestartPadUpdates();
        }

        system.perf_stats->BeginSystemFrame();
        CITRA_TRACE_ASYNC_BEGIN("EmuFrame", current_frame);
    }
}

bool RendererBase::IsScreenshotPending() const {
    return settings.screenshot_requested;
}

void RendererBase::RequestScreenshot(void* data, std::function<void(bool)> callback,
                                     const Layout::FramebufferLayout& layout) {
    if (settings.screenshot_requested) {
        LOG_ERROR(Render, "A screenshot is already requested or in progress, ignoring the request");
        return;
    }
    settings.screenshot_bits = data;
    settings.screenshot_complete_callback = callback;
    settings.screenshot_framebuffer_layout = layout;
    settings.screenshot_requested = true;
}
} // namespace VideoCore
