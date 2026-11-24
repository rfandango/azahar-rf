// Copyright 2014-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <atomic>
#include <chrono>
#include <optional>
#include "common/common_types.h"
#include "common/resolution_scale.h"
#include "core/frontend/framebuffer_layout.h"
#include "video_core/rasterizer_interface.h"

namespace Frontend {
class EmuWindow;
}

namespace Core {
class System;
}

namespace VideoCore {

enum class ScreenId : u32 {
    TopLeft,
    TopRight,
    Bottom,
};

struct RendererSettings {
    // Screenshot
    std::atomic_bool screenshot_requested{false};
    void* screenshot_bits{};
    std::function<void(bool)> screenshot_complete_callback;
    Layout::FramebufferLayout screenshot_framebuffer_layout;
    // Renderer
    std::atomic_bool bg_color_update_requested{false};
    std::atomic_bool shader_update_requested{false};
};

class RendererBase : NonCopyable {
public:
    explicit RendererBase(Core::System& system, Frontend::EmuWindow& window,
                          Frontend::EmuWindow* secondary_window);
    virtual ~RendererBase();

    /// Returns the rasterizer owned by the renderer
    virtual VideoCore::RasterizerInterface* Rasterizer() = 0;

    /// Finalize rendering the guest frame and draw into the presentation texture
    virtual void SwapBuffers() = 0;

    /// Draws the latest frame to the window waiting timeout_ms for a frame to arrive (Renderer
    /// specific implementation)
    virtual void TryPresent(int timeout_ms, bool is_secondary) = 0;
    virtual void TryPresent(int timeout_ms) {
        TryPresent(timeout_ms, false);
    }

    /// Prepares for video dumping (e.g. create necessary buffers, etc)
    virtual void PrepareVideoDumping() {}

    /// Cleans up after video dumping is ended
    virtual void CleanupVideoDumping() {}

    /// This is called to notify the rendering backend of a surface change
    // if second == true then it is the second screen
    virtual void NotifySurfaceChanged(bool second) {}

    /// Returns the resolution scale factor relative to the native 3DS screen resolution: the
    /// one asked for, or a lower one while the GPU cannot keep up with it, see GovernResolution
    Common::ResolutionScale GetResolutionScaleFactor();

    /// Returns the MSAA sample count
    u8 GetSampleCount() const;

    /// Updates the framebuffer layout of the contained render window handle.
    void UpdateCurrentFramebufferLayout(bool is_portrait_mode = {});

    /// Ends the current frame
    void EndFrame();

    /// Whether ComposeFrame and EndEmulatedFrame together can stand in for SwapBuffers, for an
    /// emulated GPU that composes on another thread than the one the emulation runs on
    virtual bool ComposesApart() const {
        return false;
    }

    /// The part of SwapBuffers that is the GPU's: composes and presents the image and ends
    /// the frame for the host's GPU. On the thread that uses the rasterizer. `vblank` is
    /// GetCurrentFrame() as it was at the VBlank the image is of.
    virtual void ComposeFrame(u64 vblank) {}

    /// The rest of it: ends the frame for the emulation (speed limit, statistics, input). On
    /// the emulation thread.
    void EndEmulatedFrame();

    f32 GetCurrentFPS() const {
        return current_fps;
    }

    u64 GetCurrentFrame() const {
        return current_frame;
    }

    Frontend::EmuWindow& GetRenderWindow() {
        return render_window;
    }

    const Frontend::EmuWindow& GetRenderWindow() const {
        return render_window;
    }

    [[nodiscard]] RendererSettings& Settings() {
        return settings;
    }

    [[nodiscard]] const RendererSettings& Settings() const {
        return settings;
    }

    /// Something that shows on a screen changed: a buffer swap, a write into a buffer on
    /// display, a display register.
    void NotifyScreenChanged() {
        screen_revision++;
    }

    /// Returns true if a screenshot is being processed
    [[nodiscard]] bool IsScreenshotPending() const;

    /// Request a screenshot of the next frame
    void RequestScreenshot(void* data, std::function<void(bool)> callback,
                           const Layout::FramebufferLayout& layout);

protected:
    /// The part of EndFrame that is the GPU's, see ComposeFrame
    void EndGpuFrame();

    /// Waits until the GPU is at most one game frame behind the emulation.
    /// Called on VBlanks where the guest application submits a frame.
    virtual void LimitFramesAheadOfGpu() {}

    /// Whether the renderer needs that wait whatever the setting says, because nothing else
    /// keeps the emulation from getting frames ahead of the GPU
    virtual bool AlwaysLimitsFramesAheadOfGpu() const {
        return false;
    }

    /// The time the host's GPU worked on what was sent to it since this was last called, for
    /// a renderer that can tell and if `wanted`. Called once per VBlank.
    virtual std::optional<std::chrono::nanoseconds> TakeGpuTime(bool wanted) {
        return std::nullopt;
    }

    /// Whether the image composed at this VBlank can differ from the one composed last. To be
    /// called once per VBlank.
    bool ShouldPresentImage();

public:
    /// Whether the model of the console's GPU time is left to make the game miss VBlanks
    /// (VideoCore::GPU::PaceCompletion): only while the host has no room for more frames than
    /// the game then draws. False where the host's GPU time is not measured.
    [[nodiscard]] bool ModelledGpuCostsFrames() const noexcept {
        return modelled_gpu_costs_frames.load(std::memory_order_relaxed);
    }

protected:
    Core::System& system;
    RendererSettings settings;
    Frontend::EmuWindow& render_window;    /// Reference to the render window handle.
    Frontend::EmuWindow* secondary_window; /// Reference to the secondary render window handle.

protected:
    f32 current_fps = 0.0f; /// Current framerate, should be set by the renderer
    u64 current_frame = 0;  /// Current frame, should be set by the renderer

private:
    /// The resolution scale factor the settings ask for
    Common::ResolutionScale AskedResolutionScaleFactor();

    void EndFrameParts(bool emulation, bool gpu);

    /// Once per VBlank: follows how busy the host's GPU is and, with the dynamic resolution
    /// setting, lowers the resolution scale while the GPU holds the emulation back and raises
    /// it again when there is room.
    void GovernResolution();

    u64 limited_game_frames = 0; /// Guest frame count at the last LimitFramesAheadOfGpu call

    std::atomic<u32> scale_reduction{0}; /// Whole factors the resolution scale is lowered by
    // The stretch of VBlanks the GPU's load is being added up over
    bool load_window_open = false;
    std::chrono::steady_clock::time_point load_window_start{};
    u32 load_window_vblanks = 0;
    std::chrono::nanoseconds load_window_busy{};      /// The GPU at work
    std::chrono::nanoseconds load_window_waited{};    /// The emulation waiting for it
    std::chrono::microseconds load_window_work{};     /// The host at work on the emulation
    std::chrono::microseconds load_window_emulated{}; /// Emulated time
    u64 load_window_game_frames = 0;                  /// The game's frame count at its start
    // See ModelledGpuCostsFrames: the state, and stretches in a row that speak for the other
    std::atomic<bool> modelled_gpu_costs_frames{false};
    u32 model_change_windows = 0;
    // Stretches in a row, see GovernResolution
    u32 overloaded_windows = 0;
    u32 roomy_windows = 0;
    u32 calm_windows = 0;
    u32 settling_windows = 0;
    u32 windows_since_raise = 0;
    bool raise_unproven = false;
    u32 raise_patience = 0;

    u32 screen_revision = 1;    /// Counts the changes NotifyScreenChanged was told about
    u32 presented_revision = 0; /// Its value at the last VBlank
    u32 unchanged_vblanks = 0;  /// VBlanks since the last change
    Layout::FramebufferLayout presented_layout{};
    Layout::FramebufferLayout presented_secondary_layout{};
};

} // namespace VideoCore
