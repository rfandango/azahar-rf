// Copyright 2015-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include "common/hitch_tracker.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/hle/service/hid/hid.h"
#include "core/tracer/recorder.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/frame_state_dump.h"
#include "video_core/renderer_base.h"

namespace VideoCore {

RendererBase::RendererBase(Core::System& system_, Frontend::EmuWindow& window,
                           Frontend::EmuWindow* secondary_window_)
    : system{system_}, render_window{window}, secondary_window{secondary_window_} {}

RendererBase::~RendererBase() = default;

u32 RendererBase::GetResolutionScaleFactor() {
    const auto graphics_api = Settings::GetWorkingGraphicsAPI();
    if (graphics_api == Settings::GraphicsAPI::Software) {
        // Software renderer always render at native resolution
        return 1;
    }

    const u32 scale_factor = Settings::values.resolution_factor.GetValue();
    return scale_factor != 0 ? scale_factor
                             : render_window.GetFramebufferLayout().GetScalingRatio();
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
    // One "EmuFrame" async slice per emulated VBlank period, excluding GPU throttling and frame
    // limiting below.
    CITRA_TRACE_ASYNC_END("EmuFrame", current_frame);
    CITRA_TRACE_END_FRAME();
    CITRA_FRAME_DUMP_VBLANK();
    Common::Hitches::EndVBlank();

    current_frame++;

    system.perf_stats->EndSystemFrame();

    render_window.PollEvents();

    // Wait for the GPU to finish and to be at most one game frame behind.
    // Below 60 FPS a guest frame spans several VBlanks, and its GPU work may still run while the
    // next one is emulated.
    const u64 game_frames = system.perf_stats->GetGameFrameCount();
    if (game_frames != limited_game_frames) {
        limited_game_frames = game_frames;
        if (Settings::values.use_limit_frames_ahead_of_gpu.GetValue() ||
            AlwaysLimitsFramesAheadOfGpu()) {
            CITRA_TRACE_SCOPE("LimitFramesAheadOfGpu");
            CITRA_HITCH_SCOPE_ABSORB(GpuBehind);
            LimitFramesAheadOfGpu();
        }
    }

    {
        CITRA_TRACE_SCOPE("FrameLimiter");
        CITRA_HITCH_SCOPE_ABSORB(Limiter);
        system.frame_limiter.DoFrameLimiting(system.CoreTiming().GetGlobalTimeUs());
    }

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
