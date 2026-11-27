// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <string>
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
#include "video_core/renderer_vulkan/vk_shader_preset.h"

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
    /// That image and its format, where it is one of the game's (in the general layout)
    vk::Image image{};
    vk::Format format{vk::Format::eUndefined};
    /// Whether the image is already the way the screen is shown, at the size it is shown at (a
    /// shader preset's)
    bool upright{};
};

struct PresentUniformData {
    std::array<f32, 4 * 4> modelview;
    /// The part of the image that is the screen: its lowest and highest coordinates
    Common::Vec4f screen_area;
    /// Its size in pixels, and that of a pixel
    Common::Vec4f image_size;
    int screen_id_l = 0;
    int screen_id_r = 0;
    /// The pixels of the frame the screen is drawn on, along its 240 and along its length
    float output_rows = 0.0f;
    int reverse_interlaced = 0;
    /// Of the upscalers, from 0 to 2
    float edge_sharpness = 0.0f;
    float output_columns = 0.0f;
    /// For a post-processing shader: 1 for the second of the two pictures of a side by side
    /// mode, 0 otherwise
    int layer = 0;
};
static_assert(sizeof(PresentUniformData) == 124,
              "PresentUniformData does not structure in shader!");

class RendererVulkan : public VideoCore::RendererBase {
    static constexpr std::size_t PRESENT_PIPELINES = 9;
    /// Of those, the ones that enlarge a screen with Snapdragon Game Super Resolution
    static constexpr u32 SGSR_PIPELINE = 3;
    static constexpr u32 SGSR_EDGE_PIPELINE = 4;
    /// and the lighter ones
    static constexpr u32 LIGHT_PIPELINE = 5;
    static constexpr u32 CUBIC_PIPELINE = 6;
    /// The post-processing shader of the user's choice, and the one for anaglyph 3D (from the
    /// "shaders" folder of the user directory, see VideoCore::GetPostProcessingShaderList). Without
    /// one, or where it does not compile, their pipelines are null.
    static constexpr u32 CUSTOM_PIPELINE = 7;
    static constexpr u32 CUSTOM_ANAGLYPH_PIPELINE = 8;

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
    /// The pipeline the screens are drawn with in that 3D mode, where nothing enlarges them
    u32 PipelineFor(Settings::StereoRenderOption render_3d) const;
    /// Compiles the post-processing shaders the settings name, where they changed
    void LoadCustomShaders();
    void CompileShaders();
    void BuildLayouts();
    void BuildPipelines();
    /// A pipeline that draws a screen with that fragment shader. Null where the driver cannot
    /// make it.
    vk::Pipeline BuildPresentPipeline(vk::ShaderModule fragment_shader, bool blends) const;
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
    /// For native output: the share of the layout's size the screens shown in it take up in
    /// the game's own pixels, the larger where they differ
    float NativeScale(const Layout::FramebufferLayout& layout) const;
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

    /// Loads the RetroArch shader preset the settings name, where it changed
    void LoadShaderPreset();
    /// Runs the shader preset over the screens the layout shows, into images of the size they
    /// are drawn at, and puts those in the screens' place (restored after drawing)
    void RunShaderPresets(const Frame& frame, const Layout::FramebufferLayout& layout);

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
    /// Pipeline switches since the last report
    PipelineCache::SwitchCounts report_switches{};
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
    /// The names of the post-processing shaders that were loaded (or not found or failed), for
    /// CUSTOM_PIPELINE and CUSTOM_ANAGLYPH_PIPELINE; empty for the built-in ones. And their text.
    std::array<std::string, 2> custom_shader_names;
    std::array<std::string, 2> custom_shader_sources;
    /// RetroArch shader presets
    std::unique_ptr<ShaderPresetRunner> preset_runner;

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
