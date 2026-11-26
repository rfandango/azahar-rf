// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <vector>
#include "common/common_types.h"
#include "common/math_util.h"
#include "video_core/renderer_base.h"
#ifdef HAVE_LIBRETRO
#include "citra_libretro/libretro_vk.h"
#else
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_present_window.h"
#endif
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Core {
class System;
}

namespace Memory {
class MemorySystem;
}

namespace Pica {
class PicaCore;
}

namespace Layout {
struct FramebufferLayout;
}

namespace VideoCore {
class GPU;
}

namespace Vulkan {

struct TextureInfo {
    u32 width;
    u32 height;
    Pica::PixelFormat format;
    vk::Image image;
    vk::ImageView image_view;
    VmaAllocation allocation;
};

struct ScreenInfo {
    TextureInfo texture;
    Common::Rectangle<f32> texcoords;
    vk::ImageView image_view;
    /// Pixels of the image the view is of
    u32 image_width{};
    u32 image_height{};
};

struct PresentUniformData {
    std::array<f32, 4 * 4> modelview;
    /// The part of the image that is the screen: its lowest and highest coordinates
    Common::Vec4f screen_area;
    /// Its size in pixels, and that of a pixel
    Common::Vec4f image_size;
    int screen_id_l = 0;
    int screen_id_r = 0;
    float output_rows = 0.0f;
    int reverse_interlaced = 0;
    /// Of the upscalers, from 0 to 2
    float edge_sharpness = 0.0f;
};
static_assert(sizeof(PresentUniformData) == 116,
              "PresentUniformData does not structure in shader!");

class RendererVulkan : public VideoCore::RendererBase {
    static constexpr std::size_t PRESENT_PIPELINES = 7;
    /// Of those, the ones that enlarge a screen with Snapdragon Game Super Resolution
    static constexpr u32 SGSR_PIPELINE = 3;
    static constexpr u32 SGSR_EDGE_PIPELINE = 4;
    /// and the lighter ones
    static constexpr u32 LIGHT_PIPELINE = 5;
    static constexpr u32 CUBIC_PIPELINE = 6;

public:
    explicit RendererVulkan(Core::System& system, Pica::PicaCore& pica, Frontend::EmuWindow& window,
                            Frontend::EmuWindow* secondary_window);
    ~RendererVulkan() override;

    [[nodiscard]] VideoCore::RasterizerInterface* Rasterizer() override {
        return &rasterizer;
    }

    void NotifySurfaceChanged(bool second) override;

    void SwapBuffers() override;
    void ComposeFrame(u64 vblank) override;
    bool ComposesApart() const override {
        return true;
    }
    void TryPresent(int timeout_ms, bool is_secondary) override {}

private:
    void ReloadPipeline(Settings::StereoRenderOption render_3d);
    void CompileShaders();
    void BuildLayouts();
    void BuildPipelines();
    void ConfigureFramebufferTexture(TextureInfo& texture,
                                     const Pica::FramebufferConfig& framebuffer);
    void ConfigureRenderPipeline();
    /// Tells the rasterizer which shape the top screen is shown in, for the game's picture to be
    /// drawn for it (Pica::PicaCore::WideScales)
    void UpdateWideFactor(const Layout::FramebufferLayout& main_layout);
    /// Composes the image of the VBlank `composing_vblank` and hands it to the window
    void Compose();
    u64 composing_vblank{};
    /// By how many degrees the frame being drawn is turned, see Swapchain::GetPreRotation
    u32 frame_rotation{};
    void PrepareRendertarget();
    void LimitFramesAheadOfGpu() override;
    bool AlwaysLimitsFramesAheadOfGpu() const override;
    std::optional<std::chrono::nanoseconds> TakeGpuTime(bool wanted) override;
    void RenderScreenshot();
    void RenderScreenshotWithStagingCopy();
    bool TryRenderScreenshotWithHostMemory();
    /// Begins the drawing of the frame. Without `clear` only the given parts of it are cleared.
    void PrepareDraw(Frame* frame, const Layout::FramebufferLayout& layout, bool clear,
                     std::vector<vk::ClearRect> parts_to_clear);
    void RenderToWindow(PresentWindow& window, const Layout::FramebufferLayout& layout,
                        bool flipped);

    void DrawScreens(Frame* frame, const Layout::FramebufferLayout& layout, bool flipped);
    void DrawBottomScreen(const Layout::FramebufferLayout& layout,
                          const Common::Rectangle<u32>& bottom_screen);

    void DrawTopScreen(const Layout::FramebufferLayout& layout,
                       const Common::Rectangle<u32>& top_screen);
    /// The pipeline a screen is drawn with that is shown `width` pixels along its 240 and `height`
    /// along its length, with what that one wants to know
    u32 ChoosePresentPipeline(u32 screen_id, float width, float height);
    void DrawSingleScreen(u32 screen_id, float x, float y, float w, float h,
                          Layout::DisplayOrientation orientation);
    void DrawSingleScreenStereo(u32 screen_id_l, u32 screen_id_r, float x, float y, float w,
                                float h, Layout::DisplayOrientation orientation);

    void ApplySecondLayerOpacity(float alpha);

    /// What of the frame the screens of the layout leave free, as rectangles. Nothing if that is to be
    /// taken as not known.
    std::optional<std::vector<vk::ClearRect>> UncoveredParts(
        const Frame& frame, const Layout::FramebufferLayout& layout) const;

    /// Every so many VBlanks: what the GPU does, and what of it is showing frames, in the log
    void ReportGpuTime();

    void DrawCursor(const Layout::FramebufferLayout& layout);

    void LoadFBToScreenInfo(const Pica::FramebufferConfig& framebuffer, ScreenInfo& screen_info,
                            bool right_eye);
    void FillScreen(Common::Vec3<u8> color, const TextureInfo& texture);

private:
    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;

#ifdef HAVE_LIBRETRO
    LibRetroVKInstance instance;
#else
    Instance instance;
#endif
    Scheduler scheduler;
    RenderManager renderpass_cache;
    PresentWindow main_present_window;
    StreamBuffer vertex_buffer;
    DescriptorUpdateQueue update_queue;
    RasterizerVulkan rasterizer;
    std::unique_ptr<PresentWindow> secondary_present_window_ptr;
    DescriptorHeap present_heap;
    vk::UniquePipelineLayout present_pipeline_layout;
    /// With blending, and without it
    std::array<vk::Pipeline, PRESENT_PIPELINES> present_pipelines;
    std::array<vk::Pipeline, PRESENT_PIPELINES> opaque_present_pipelines;
    /// Whether what is drawn now lets through what is below it
    bool layer_blends{};
    /// Pixels of the frame being drawn for one of the layout
    float output_scale = 1.0f;
    GpuSpanMeter draw_meter;
    std::vector<GpuPassMeter::Sum> reported_passes;
    std::chrono::nanoseconds reported_busy{};
    std::array<GpuSpanMeter::Sum, GpuSpanMeter::KINDS> reported_draws{};
    std::array<GpuSpanMeter::Sum, 2> reported_copies{};
    u32 report_vblanks{};
    std::chrono::steady_clock::time_point report_start{};
    std::array<vk::ShaderModule, PRESENT_PIPELINES> present_shaders;
    std::array<vk::Sampler, 2> present_samplers;
    vk::ShaderModule present_vertex_shader;
    u32 current_pipeline = 0;

    std::array<ScreenInfo, 3> screen_infos{};
    PresentUniformData draw_info{};
    vk::ClearColorValue clear_color{};

    vk::ShaderModule cursor_vertex_shader{};
    vk::ShaderModule cursor_fragment_shader{};
    vk::Pipeline cursor_pipeline{};
    vk::UniquePipelineLayout cursor_pipeline_layout{};
    bool isSecondaryWindow;
    bool secondaryWindowEnabled;
    /// The factor last given to the rasterizer for a top screen shown at another shape than
    /// the console's, for saying so when it changes
    f32 last_wide_factor{1.f};
    bool screenRendered;
    u64 previous_game_frame_tick{};
};

} // namespace Vulkan
