// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include "common/assert.h"
#include "common/diagnostics.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/memory_detect.h"
#include "common/microprofile.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "video_core/gpu.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_memory_util.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include "video_core/host_shaders/vulkan_present_anaglyph_frag.h"
#include "video_core/host_shaders/vulkan_present_cubic_frag.h"
#include "video_core/host_shaders/vulkan_present_frag.h"
#include "video_core/host_shaders/vulkan_present_interlaced_frag.h"
#include "video_core/host_shaders/vulkan_present_light_frag.h"
#include "video_core/host_shaders/vulkan_present_sgsr1_frag.h"
#include "video_core/host_shaders/vulkan_present_vert.h"

#include "video_core/host_shaders/vulkan_cursor_frag.h"
#include "video_core/host_shaders/vulkan_cursor_vert.h"

#include <vk_mem_alloc.h>
#if defined(__APPLE__) && !defined(HAVE_LIBRETRO)
#include "common/apple_utils.h"
#endif

#ifdef ENABLE_SDL2
#include <SDL.h>
#endif

MICROPROFILE_DEFINE(Vulkan_RenderFrame, "Vulkan", "Render Frame", MP_RGB(128, 128, 64));

namespace Vulkan {

struct ScreenRectVertex {
    ScreenRectVertex() = default;
    ScreenRectVertex(float x, float y, float u, float v)
        : position{Common::MakeVec(x, y)}, tex_coord{Common::MakeVec(u, v)} {}

    Common::Vec2f position;
    Common::Vec2f tex_coord;
};

constexpr u32 VERTEX_BUFFER_SIZE = sizeof(ScreenRectVertex) * 8192;

constexpr std::array<f32, 4 * 4> MakeOrthographicMatrix(u32 width, u32 height) {
    // clang-format off
    return { 2.f / width, 0.f,         0.f, -1.f,
            0.f,         2.f / height, 0.f, -1.f,
            0.f,         0.f,          1.f,  0.f,
            0.f,         0.f,          0.f,  1.f};
    // clang-format on
}

/// The same, for a frame that is drawn turned by so many degrees (clockwise as seen on the
/// frame): a point at (x, y) of the upright picture goes to (-y, x), (-x, -y) or (y, -x)
constexpr std::array<f32, 4 * 4> MakeOrthographicMatrix(u32 width, u32 height, u32 rotation) {
    const f32 w = 2.f / width;
    const f32 h = 2.f / height;
    // clang-format off
    switch (rotation) {
    case 90:
        return { 0.f, -h,   0.f,  1.f,
                 w,    0.f, 0.f, -1.f,
                 0.f,  0.f, 1.f,  0.f,
                 0.f,  0.f, 0.f,  1.f};
    case 180:
        return {-w,    0.f, 0.f,  1.f,
                 0.f, -h,   0.f,  1.f,
                 0.f,  0.f, 1.f,  0.f,
                 0.f,  0.f, 0.f,  1.f};
    case 270:
        return { 0.f,  h,   0.f, -1.f,
                -w,    0.f, 0.f,  1.f,
                 0.f,  0.f, 1.f,  0.f,
                 0.f,  0.f, 0.f,  1.f};
    default:
        return MakeOrthographicMatrix(width, height);
    }
    // clang-format on
}

constexpr static std::array<vk::DescriptorSetLayoutBinding, 1> PRESENT_BINDINGS = {{
    {0, vk::DescriptorType::eCombinedImageSampler, 3, vk::ShaderStageFlagBits::eFragment},
}};

namespace {
static bool IsLowRefreshRate() {
#if (defined(__APPLE__) || defined(ENABLE_SDL2)) && !defined(HAVE_LIBRETRO)
    if (!Settings::values.use_display_refresh_rate_detection) {
        LOG_INFO(Render_Vulkan, "Refresh rate detection is currently disabled via settings");
        return false;
    }
#ifdef __APPLE__
    // Apple's low power mode sometimes limits applications to 30fps without changing the refresh
    // rate, meaning the above code doesn't catch it.
    if (AppleUtils::IsLowPowerModeEnabled()) {
        LOG_WARNING(Render_Vulkan, "Apple's low power mode is enabled, assuming low application "
                                   "framerate. FIFO will be disabled");
        return true;
    }

    const auto cur_refresh_rate = AppleUtils::GetRefreshRate();
#elif defined(ENABLE_SDL2)
    if (SDL_WasInit(SDL_INIT_VIDEO) == 0) {
        LOG_ERROR(Render_Vulkan, "Attempted to check refresh rate via SDL, but failed because "
                                 "SDL_INIT_VIDEO wasn't initialized");
        return false;
    }

    SDL_DisplayMode cur_display_mode;
    SDL_GetCurrentDisplayMode(0, &cur_display_mode); // TODO: Multimonitor handling. -OS

    const auto cur_refresh_rate = cur_display_mode.refresh_rate;
#endif // ENABLE_SDL2

    if (cur_refresh_rate < SCREEN_REFRESH_RATE) {
        LOG_WARNING(Render_Vulkan,
                    "Detected refresh rate lower than the emulated 3DS screen: {}hz. FIFO will "
                    "be disabled",
                    cur_refresh_rate);
        return true;
    } else {
        LOG_INFO(Render_Vulkan, "Refresh rate is above emulated 3DS screen: {}hz. Good.",
                 cur_refresh_rate);
    }
#endif // (defined(__APPLE__) || defined(ENABLE_SDL2)) && !defined(HAVE_LIBRETRO)

    // We have no available method of checking refresh rate. Just assume that everything is fine :)
    return false;
}
} // Anonymous namespace

RendererVulkan::RendererVulkan(Core::System& system, Pica::PicaCore& pica_,
                               Frontend::EmuWindow& window, Frontend::EmuWindow* secondary_window)
    : RendererBase{system, window, secondary_window}, memory{system.Memory()}, pica{pica_},
      instance{window, Settings::values.physical_device.GetValue()}, scheduler{instance},
      renderpass_cache{instance, scheduler},
      main_present_window{window, instance, scheduler, IsLowRefreshRate()},
      vertex_buffer{instance, scheduler, vk::BufferUsageFlagBits::eVertexBuffer,
                    VERTEX_BUFFER_SIZE},
      update_queue{instance}, rasterizer{memory,
                                         pica,
                                         system.CustomTexManager(),
                                         *this,
                                         render_window,
                                         instance,
                                         scheduler,
                                         renderpass_cache,
                                         update_queue,
                                         main_present_window.ImageCount()},
      present_heap{instance, scheduler.GetMasterSemaphore(), PRESENT_BINDINGS, 32},
      draw_meter{instance} {
    CompileShaders();
    BuildLayouts();
    BuildPipelines();
    if (secondary_window) {
        secondary_present_window_ptr = std::make_unique<PresentWindow>(
            *secondary_window, instance, scheduler, IsLowRefreshRate());
    }
}

RendererVulkan::~RendererVulkan() {
    system.frame_limiter.SetPacingScale(1.0);

    vk::Device device = instance.GetDevice();
    scheduler.Finish();
    main_present_window.WaitPresent();
    device.waitIdle();

    device.destroyShaderModule(present_vertex_shader);
    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        device.destroyPipeline(present_pipelines[i]);
        device.destroyPipeline(opaque_present_pipelines[i]);
        device.destroyShaderModule(present_shaders[i]);
    }

    for (auto& sampler : present_samplers) {
        device.destroySampler(sampler);
    }

    for (auto& info : screen_infos) {
        device.destroyImageView(info.texture.image_view);
        vmaDestroyImage(instance.GetAllocator(), info.texture.image, info.texture.allocation);
    }

    device.destroyPipeline(cursor_pipeline);
    device.destroyShaderModule(cursor_vertex_shader);
    device.destroyShaderModule(cursor_fragment_shader);
}

void RendererVulkan::PrepareRendertarget() {
    const auto& framebuffer_config = pica.regs.framebuffer_config;
    const auto& regs_lcd = pica.regs_lcd;
    for (u32 i = 0; i < 3; i++) {
        const u32 fb_id = i == 2 ? 1 : 0;
        const auto& framebuffer = framebuffer_config[fb_id];
        auto& texture = screen_infos[i].texture;

        const auto color_fill = fb_id == 0 ? regs_lcd.color_fill_top : regs_lcd.color_fill_bottom;
        if (color_fill.is_enabled) {
            screen_infos[i].image_view = texture.image_view;
            FillScreen(color_fill.AsVector(), texture);
            continue;
        }

        if (texture.width != framebuffer.width || texture.height != framebuffer.height ||
            texture.format != framebuffer.color_format) {
            ConfigureFramebufferTexture(texture, framebuffer);
        }

        LoadFBToScreenInfo(framebuffer, screen_infos[i], i == 1);
    }
}

std::optional<std::vector<vk::ClearRect>> RendererVulkan::UncoveredParts(
    const Frame& frame, const Layout::FramebufferLayout& layout) const {
    if (layout.render_3d_mode != Settings::StereoRenderOption::Off || layout.width == 0 ||
        layout.height == 0 ||
        (Common::TraceMarkers::Experiments4() & Common::TraceMarkers::LegacyScreenBlending)) {
        return std::nullopt;
    }
    struct Part {
        s32 left, top, right, bottom;
    };
    std::vector<Part> parts{
        Part{0, 0, static_cast<s32>(frame.width), static_cast<s32>(frame.height)}};
    const auto take_out = [&](const Common::Rectangle<u32>& screen) {
        // Where the screen comes to lie in the frame, see MakeOrthographicMatrix, only with pixels
        // that are covered in full
        const float left = static_cast<float>(screen.left) / static_cast<float>(layout.width);
        const float right = static_cast<float>(screen.right) / static_cast<float>(layout.width);
        const float top = static_cast<float>(screen.top) / static_cast<float>(layout.height);
        const float bottom = static_cast<float>(screen.bottom) / static_cast<float>(layout.height);
        std::array<float, 4> box{left, top, right, bottom};
        switch (frame_rotation) {
        case 90:
            box = {1.f - bottom, left, 1.f - top, right};
            break;
        case 180:
            box = {1.f - right, 1.f - bottom, 1.f - left, 1.f - top};
            break;
        case 270:
            box = {top, 1.f - right, bottom, 1.f - left};
            break;
        default:
            break;
        }
        const float width = static_cast<float>(frame.width);
        const float height = static_cast<float>(frame.height);
        const Part covered{static_cast<s32>(std::ceil(box[0] * width - 0.01f)),
                           static_cast<s32>(std::ceil(box[1] * height - 0.01f)),
                           static_cast<s32>(std::floor(box[2] * width + 0.01f)),
                           static_cast<s32>(std::floor(box[3] * height + 0.01f))};
        std::vector<Part> rest;
        for (const Part& part : parts) {
            const Part both{std::max(part.left, covered.left), std::max(part.top, covered.top),
                            std::min(part.right, covered.right),
                            std::min(part.bottom, covered.bottom)};
            if (both.left >= both.right || both.top >= both.bottom) {
                rest.push_back(part);
                continue;
            }
            const std::array pieces{Part{part.left, part.top, part.right, both.top},
                                    Part{part.left, both.bottom, part.right, part.bottom},
                                    Part{part.left, both.top, both.left, both.bottom},
                                    Part{both.right, both.top, part.right, both.bottom}};
            for (const Part& piece : pieces) {
                if (piece.left < piece.right && piece.top < piece.bottom) {
                    rest.push_back(piece);
                }
            }
        }
        parts = std::move(rest);
    };
    // (The one below may be seen through the one above)
    const bool swapped = Settings::values.swap_screen.GetValue();
    if (layout.top_screen_enabled && (!swapped || layout.top_opacity >= 1)) {
        take_out(layout.top_screen);
    }
    if (layout.bottom_screen_enabled && (swapped || layout.bottom_opacity >= 1)) {
        take_out(layout.bottom_screen);
    }
    std::vector<vk::ClearRect> rects;
    for (const Part& part : parts) {
        rects.push_back(vk::ClearRect{
            .rect = vk::Rect2D{.offset = {part.left, part.top},
                               .extent = {static_cast<u32>(part.right - part.left),
                                          static_cast<u32>(part.bottom - part.top)}},
            .baseArrayLayer = 0,
            .layerCount = 1,
        });
    }
    return rects;
}

void RendererVulkan::PrepareDraw(Frame* frame, const Layout::FramebufferLayout& layout, bool clear,
                                 std::vector<vk::ClearRect> parts_to_clear) {
    // (Snapdragon Game Super Resolution starts from the filtered picture)
    const bool linear =
        Settings::values.filter_mode.GetValue() ||
        Settings::values.screen_upscaling.GetValue() != Settings::ScreenUpscaling::Filter;
    const auto sampler = present_samplers[!linear];
    const auto present_set = present_heap.Commit();
    for (u32 index = 0; index < screen_infos.size(); index++) {
        update_queue.AddImageSampler(present_set, 0, index, screen_infos[index].image_view,
                                     sampler);
    }

    renderpass_cache.EndRendering();
    scheduler.Record([this, layout, frame, present_set,
                      renderpass = clear ? frame->renderpass : frame->renderpass_uncleared,
                      parts = std::move(parts_to_clear),
                      index = current_pipeline](vk::CommandBuffer cmdbuf) {
        draw_meter.Begin(cmdbuf);
        // (the frame is as large as the layout, or the layout lying on its side)
        const vk::Viewport viewport = {
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(frame->width),
            .height = static_cast<float>(frame->height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        const vk::Rect2D scissor = {
            .offset = {0, 0},
            .extent = {frame->width, frame->height},
        };

        cmdbuf.setViewport(0, viewport);
        cmdbuf.setScissor(0, scissor);

        const vk::ClearValue clear{.color = clear_color};
        const vk::PipelineLayout layout{*present_pipeline_layout};
        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = renderpass,
            .framebuffer = frame->framebuffer,
            .renderArea =
                vk::Rect2D{
                    .offset = {0, 0},
                    .extent = {frame->width, frame->height},
                },
            .clearValueCount = 1,
            .pClearValues = &clear,
        };

        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
        if (!parts.empty()) {
            const vk::ClearAttachment attachment = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .colorAttachment = 0,
                .clearValue = clear,
            };
            cmdbuf.clearAttachments(attachment, parts);
        }
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, present_pipelines[index]);
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, present_set, {});
    });
}

void RendererVulkan::RenderToWindow(PresentWindow& window, const Layout::FramebufferLayout& layout,
                                    bool flipped) {
    if (!Settings::values.use_skip_duplicate_frames.GetValue() ||
        Core::PerfStats::game_frames_updated) {
        const bool direct = window.DrawsIntoSwapchain();
        Frame* frame = window.GetRenderFrame();

        // Where the display's panel lies turned the frame is drawn the way it lies
        if (direct) {
            if (!frame) {
                // There is nothing to show it on
                return;
            }
            frame_rotation = frame->rotation;
        } else {
            frame_rotation = window.PreRotation();
            const bool sideways = frame_rotation == 90 || frame_rotation == 270;
            u32 frame_width = sideways ? layout.height : layout.width;
            u32 frame_height = sideways ? layout.width : layout.height;
#ifdef ANDROID
            if (Settings::values.output_resolution.GetValue() != 100) {
                frame_width = window.Extent().width;
                frame_height = window.Extent().height;
            }
#endif
            if (frame_width != frame->width || frame_height != frame->height) {
                window.WaitPresent();
                scheduler.Finish();
                window.RecreateFrame(frame, frame_width, frame_height);
            }
        }

        clear_color.float32[0] = Settings::values.bg_red.GetValue();
        clear_color.float32[1] = Settings::values.bg_green.GetValue();
        clear_color.float32[2] = Settings::values.bg_blue.GetValue();
        clear_color.float32[3] = 1.0f;

        DrawScreens(frame, layout, flipped);
        frame_rotation = 0;
        scheduler.Flush(frame->render_ready, direct ? frame->image_acquired : vk::Semaphore{});
        window.Present(frame, composing_vblank);
        if ((secondaryWindowEnabled && isSecondaryWindow) || (!secondaryWindowEnabled)) {
            Core::PerfStats::game_frames_updated = false;
            screenRendered = true;
        }
    }
}

void RendererVulkan::LoadFBToScreenInfo(const Pica::FramebufferConfig& framebuffer,
                                        ScreenInfo& screen_info, bool right_eye) {

    if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0) {
        right_eye = false;
    }

    const PAddr framebuffer_addr =
        framebuffer.active_fb == 0
            ? (right_eye ? framebuffer.address_right1 : framebuffer.address_left1)
            : (right_eye ? framebuffer.address_right2 : framebuffer.address_left2);

    LOG_TRACE(Render_Vulkan, "0x{:08x} bytes from 0x{:08x}({}x{}), fmt {:x}",
              framebuffer.stride * framebuffer.height, framebuffer_addr, framebuffer.width.Value(),
              framebuffer.height.Value(), framebuffer.format);

    const u32 bpp = Pica::BytesPerPixel(framebuffer.color_format);
    const std::size_t pixel_stride = framebuffer.stride / bpp;

    ASSERT(pixel_stride * bpp == framebuffer.stride);
    ASSERT(pixel_stride % 4 == 0);

    if (!rasterizer.AccelerateDisplay(framebuffer, framebuffer_addr, static_cast<u32>(pixel_stride),
                                      screen_info)) {
        // Reset the screen info's display texture to its own permanent texture
        screen_info.image_view = screen_info.texture.image_view;
        screen_info.texcoords = {0.f, 0.f, 1.f, 1.f};
        screen_info.image_width = screen_info.texture.width;
        screen_info.image_height = screen_info.texture.height;

        ASSERT(false);
    }
}

void RendererVulkan::CompileShaders() {
    const vk::Device device = instance.GetDevice();
    const std::string_view preamble =
        instance.IsImageArrayDynamicIndexSupported() ? "#define ARRAY_DYNAMIC_INDEX" : "";
    present_vertex_shader =
        Compile(HostShaders::VULKAN_PRESENT_VERT, vk::ShaderStageFlagBits::eVertex, device);
    present_shaders[0] = Compile(HostShaders::VULKAN_PRESENT_FRAG,
                                 vk::ShaderStageFlagBits::eFragment, device, preamble);
    present_shaders[1] = Compile(HostShaders::VULKAN_PRESENT_ANAGLYPH_FRAG,
                                 vk::ShaderStageFlagBits::eFragment, device, preamble);
    present_shaders[2] = Compile(HostShaders::VULKAN_PRESENT_INTERLACED_FRAG,
                                 vk::ShaderStageFlagBits::eFragment, device, preamble);
    present_shaders[SGSR_PIPELINE] = Compile(HostShaders::VULKAN_PRESENT_SGSR1_FRAG,
                                             vk::ShaderStageFlagBits::eFragment, device, preamble);
    const std::string edge_preamble = fmt::format("{}\n#define EDGE_DIRECTION", preamble);
    present_shaders[SGSR_EDGE_PIPELINE] =
        Compile(HostShaders::VULKAN_PRESENT_SGSR1_FRAG, vk::ShaderStageFlagBits::eFragment, device,
                edge_preamble);
    present_shaders[LIGHT_PIPELINE] = Compile(HostShaders::VULKAN_PRESENT_LIGHT_FRAG,
                                              vk::ShaderStageFlagBits::eFragment, device, preamble);
    present_shaders[CUBIC_PIPELINE] = Compile(HostShaders::VULKAN_PRESENT_CUBIC_FRAG,
                                              vk::ShaderStageFlagBits::eFragment, device, preamble);

    cursor_vertex_shader =
        Compile(HostShaders::VULKAN_CURSOR_VERT, vk::ShaderStageFlagBits::eVertex, device);
    cursor_fragment_shader =
        Compile(HostShaders::VULKAN_CURSOR_FRAG, vk::ShaderStageFlagBits::eFragment, device);

    for (std::size_t i = 0; i < present_samplers.size(); i++) {
        const bool linear = i == 0;
        const vk::Filter filter_mode = linear ? vk::Filter::eLinear : vk::Filter::eNearest;
        const vk::SamplerMipmapMode mipmap_mode =
            linear ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest;
        const vk::SamplerCreateInfo sampler_info = {
            .magFilter = filter_mode,
            .minFilter = filter_mode,
            .mipmapMode = mipmap_mode,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1.0f,
            .compareEnable = false,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = false,
        };

        present_samplers[i] = device.createSampler(sampler_info);
    }
}

void RendererVulkan::BuildLayouts() {
    const vk::PushConstantRange push_range = {
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(PresentUniformData),
    };

    const auto descriptor_set_layout = present_heap.Layout();
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    present_pipeline_layout = instance.GetDevice().createPipelineLayoutUnique(layout_info);

    const vk::PipelineLayoutCreateInfo cursor_layout_info = {};
    cursor_pipeline_layout = instance.GetDevice().createPipelineLayoutUnique(cursor_layout_info);
}

void RendererVulkan::BuildPipelines() {
    const vk::VertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(ScreenRectVertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };

    const std::array attributes = {
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ScreenRectVertex, position),
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ScreenRectVertex, tex_coord),
        },
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = vk::PrimitiveTopology::eTriangleStrip,
        .primitiveRestartEnable = false,
    };

    const vk::PipelineRasterizationStateCreateInfo raster_state = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false,
    };

    vk::PipelineColorBlendAttachmentState colorblend_attachment = {
        .blendEnable = true,
        .srcColorBlendFactor = vk::BlendFactor::eConstantAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusConstantAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eConstantAlpha,
        .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusConstantAlpha,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &colorblend_attachment,
    };

    const vk::Viewport placeholder_viewport = vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    const vk::Rect2D placeholder_scissor = vk::Rect2D{{0, 0}, {1, 1}};
    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .viewportCount = 1,
        .pViewports = &placeholder_viewport,
        .scissorCount = 1,
        .pScissors = &placeholder_scissor,
    };

    const std::array dynamic_states = {
        vk::DynamicState::eBlendConstants,
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::PipelineDepthStencilStateCreateInfo depth_info = {
        .depthTestEnable = false,
        .depthWriteEnable = false,
        .depthCompareOp = vk::CompareOp::eAlways,
        .depthBoundsTestEnable = false,
        .stencilTestEnable = false,
    };

    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        const std::array shader_stages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = present_vertex_shader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = present_shaders[i],
                .pName = "main",
            },
        };

        const vk::GraphicsPipelineCreateInfo pipeline_info = {
            .stageCount = static_cast<u32>(shader_stages.size()),
            .pStages = shader_stages.data(),
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_state,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_info,
            .layout = *present_pipeline_layout,
            .renderPass = main_present_window.Renderpass(),
        };

        for (const bool blends : {true, false}) {
            colorblend_attachment.blendEnable = blends;
            const auto [result, pipeline] =
                instance.GetDevice().createGraphicsPipeline({}, pipeline_info);
            ASSERT_MSG(result == vk::Result::eSuccess, "Unable to build present pipelines");
            (blends ? present_pipelines : opaque_present_pipelines)[i] = pipeline;
        }
    }

    // Build cursor pipeline (simple position-only, inverted color blending)
    {
        const vk::VertexInputBindingDescription cursor_binding = {
            .binding = 0,
            .stride = sizeof(float) * 2,
            .inputRate = vk::VertexInputRate::eVertex,
        };

        const vk::VertexInputAttributeDescription cursor_attribute = {
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = 0,
        };

        const vk::PipelineVertexInputStateCreateInfo cursor_vertex_input = {
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &cursor_binding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &cursor_attribute,
        };

        const vk::PipelineInputAssemblyStateCreateInfo cursor_input_assembly = {
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = false,
        };

        const vk::PipelineRasterizationStateCreateInfo cursor_raster = {
            .depthClampEnable = false,
            .rasterizerDiscardEnable = false,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eClockwise,
            .depthBiasEnable = false,
            .lineWidth = 1.0f,
        };

        const vk::PipelineMultisampleStateCreateInfo cursor_multisample = {
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = false,
        };

        const vk::PipelineColorBlendAttachmentState cursor_blend_attachment = {
            .blendEnable = true,
            .srcColorBlendFactor = vk::BlendFactor::eOneMinusDstColor,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcColor,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };

        const vk::PipelineColorBlendStateCreateInfo cursor_color_blending = {
            .logicOpEnable = false,
            .attachmentCount = 1,
            .pAttachments = &cursor_blend_attachment,
        };

        const vk::Viewport placeholder_vp = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
        const vk::Rect2D placeholder_sc = {{0, 0}, {1, 1}};
        const vk::PipelineViewportStateCreateInfo cursor_viewport = {
            .viewportCount = 1,
            .pViewports = &placeholder_vp,
            .scissorCount = 1,
            .pScissors = &placeholder_sc,
        };

        const std::array cursor_dynamic_states = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };

        const vk::PipelineDynamicStateCreateInfo cursor_dynamic = {
            .dynamicStateCount = static_cast<u32>(cursor_dynamic_states.size()),
            .pDynamicStates = cursor_dynamic_states.data(),
        };

        const vk::PipelineDepthStencilStateCreateInfo cursor_depth = {
            .depthTestEnable = false,
            .depthWriteEnable = false,
            .depthCompareOp = vk::CompareOp::eAlways,
            .depthBoundsTestEnable = false,
            .stencilTestEnable = false,
        };

        const std::array cursor_shader_stages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = cursor_vertex_shader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = cursor_fragment_shader,
                .pName = "main",
            },
        };

        const vk::GraphicsPipelineCreateInfo cursor_pipeline_info = {
            .stageCount = static_cast<u32>(cursor_shader_stages.size()),
            .pStages = cursor_shader_stages.data(),
            .pVertexInputState = &cursor_vertex_input,
            .pInputAssemblyState = &cursor_input_assembly,
            .pViewportState = &cursor_viewport,
            .pRasterizationState = &cursor_raster,
            .pMultisampleState = &cursor_multisample,
            .pDepthStencilState = &cursor_depth,
            .pColorBlendState = &cursor_color_blending,
            .pDynamicState = &cursor_dynamic,
            .layout = *cursor_pipeline_layout,
            .renderPass = main_present_window.Renderpass(),
        };

        const auto [result, pipeline] =
            instance.GetDevice().createGraphicsPipeline({}, cursor_pipeline_info);
        ASSERT_MSG(result == vk::Result::eSuccess, "Unable to build cursor pipeline");
        cursor_pipeline = pipeline;
    }
}

void RendererVulkan::ConfigureFramebufferTexture(TextureInfo& texture,
                                                 const Pica::FramebufferConfig& framebuffer) {
    vk::Device device = instance.GetDevice();
    if (texture.image_view) {
        device.destroyImageView(texture.image_view);
    }
    if (texture.image) {
        vmaDestroyImage(instance.GetAllocator(), texture.image, texture.allocation);
    }

    const VideoCore::PixelFormat pixel_format =
        VideoCore::PixelFormatFromGPUPixelFormat(framebuffer.color_format);
    const vk::Format format = instance.GetTraits(pixel_format).native;
    const vk::ImageCreateInfo image_info = {
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {framebuffer.width, framebuffer.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eSampled,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &texture.allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }
    texture.image = vk::Image{unsafe_image};

    const vk::ImageViewCreateInfo view_info = {
        .image = texture.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    texture.image_view = device.createImageView(view_info);

    texture.width = framebuffer.width;
    texture.height = framebuffer.height;
    texture.format = framebuffer.color_format;
}

void RendererVulkan::FillScreen(Common::Vec3<u8> color, const TextureInfo& texture) {
    // When loading some 3GX extensions, FillScreen may be called before texture image is available
    if (!texture.image) {
        return;
    }

    const vk::ClearColorValue clear_color = {
        .float32 =
            std::array{
                color.r() / 255.0f,
                color.g() / 255.0f,
                color.b() / 255.0f,
                1.0f,
            },
    };

    renderpass_cache.EndRendering();
    scheduler.Record([image = texture.image, clear_color](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };

        const vk::ImageMemoryBarrier pre_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        const vk::ImageMemoryBarrier post_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

        cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, clear_color, range);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eFragmentShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });
}

void RendererVulkan::ReloadPipeline(Settings::StereoRenderOption render_3d) {
    switch (render_3d) {
    case Settings::StereoRenderOption::Anaglyph:
        current_pipeline = 1;
        break;
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced:
        current_pipeline = 2;
        draw_info.reverse_interlaced = render_3d == Settings::StereoRenderOption::ReverseInterlaced;
        break;
    default:
        current_pipeline = 0;
        break;
    }
}

u32 RendererVulkan::ChoosePresentPipeline(u32 screen_id, float width, float height) {
    const ScreenInfo& screen_info = screen_infos[screen_id];
    // (The screens lie on their side in memory: left and right are along the image's height)
    const auto& texcoords = screen_info.texcoords;
    const Common::Vec4f area =
        Common::MakeVec(std::min(texcoords.top, texcoords.bottom),
                        std::min(texcoords.left, texcoords.right),
                        std::max(texcoords.top, texcoords.bottom),
                        std::max(texcoords.left, texcoords.right));
    draw_info.screen_area = area;
    const float image_width = static_cast<float>(std::max(screen_info.image_width, 1u));
    const float image_height = static_cast<float>(std::max(screen_info.image_height, 1u));
    draw_info.image_size =
        Common::MakeVec(image_width, image_height, 1.0f / image_width, 1.0f / image_height);
    draw_info.edge_sharpness =
        static_cast<float>(Settings::values.screen_upscaling_sharpness.GetValue()) / 100.0f;
    const auto upscaling = Common::Diagnostics::ComparesWithoutUpscaling()
                               ? Settings::ScreenUpscaling::Filter
                               : Settings::values.screen_upscaling.GetValue();
    if (current_pipeline != 0 || upscaling == Settings::ScreenUpscaling::Filter ||
        (screen_id == 2 && !Settings::values.screen_upscaling_bottom.GetValue())) {
        return current_pipeline;
    }
    // Only for a picture that is made larger: at its size, or smaller, there is nothing to add
    const float pixels_across = (area.z - area.x) * image_width;
    const float pixels_along = (area.w - area.y) * image_height;
    const bool enlarged = width * output_scale > pixels_across * 1.02f ||
                          height * output_scale > pixels_along * 1.02f;
    if (!enlarged) {
        return current_pipeline;
    }
    switch (upscaling) {
    case Settings::ScreenUpscaling::SGSR1:
        return SGSR_PIPELINE;
    case Settings::ScreenUpscaling::SGSR1Edge:
        return SGSR_EDGE_PIPELINE;
    case Settings::ScreenUpscaling::Light:
        return LIGHT_PIPELINE;
    case Settings::ScreenUpscaling::Cubic:
        return CUBIC_PIPELINE;
    default:
        return current_pipeline;
    }
}

void RendererVulkan::DrawSingleScreen(u32 screen_id, float x, float y, float w, float h,
                                      Layout::DisplayOrientation orientation) {
    const ScreenInfo& screen_info = screen_infos[screen_id];
    const auto& texcoords = screen_info.texcoords;
    const bool upright = orientation == Layout::DisplayOrientation::Portrait ||
                         orientation == Layout::DisplayOrientation::PortraitFlipped;
    const u32 pipeline = ChoosePresentPipeline(screen_id, upright ? w : h, upright ? h : w);

    std::array<ScreenRectVertex, 4> vertices;
    switch (orientation) {
    case Layout::DisplayOrientation::Landscape:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
        }};
        break;
    case Layout::DisplayOrientation::Portrait:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
        }};
        std::swap(h, w);
        break;
    case Layout::DisplayOrientation::LandscapeFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.left),
        }};
        break;
    case Layout::DisplayOrientation::PortraitFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.right),
        }};
        std::swap(h, w);
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown DisplayOrientation: {}", orientation);
        break;
    }

    const u64 size = sizeof(ScreenRectVertex) * vertices.size();
    auto [data, offset, invalidate] = vertex_buffer.Map(size, 16);
    std::memcpy(data, vertices.data(), size);
    vertex_buffer.Commit(size);

    draw_info.output_rows = h * output_scale;
    draw_info.screen_id_l = screen_id;

    scheduler.Record([this, offset = offset, info = draw_info,
                      pipeline = (layer_blends ? present_pipelines
                                               : opaque_present_pipelines)[pipeline]](
                         vk::CommandBuffer cmdbuf) {
        const u32 first_vertex = static_cast<u32>(offset) / sizeof(ScreenRectVertex);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdbuf.pushConstants(*present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, first_vertex, 0);
    });
}

void RendererVulkan::DrawSingleScreenStereo(u32 screen_id_l, u32 screen_id_r, float x, float y,
                                            float w, float h,
                                            Layout::DisplayOrientation orientation) {
    const ScreenInfo& screen_info_l = screen_infos[screen_id_l];
    const auto& texcoords = screen_info_l.texcoords;

    std::array<ScreenRectVertex, 4> vertices;
    switch (orientation) {
    case Layout::DisplayOrientation::Landscape:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
        }};
        break;
    case Layout::DisplayOrientation::Portrait:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
        }};
        std::swap(h, w);
        break;
    case Layout::DisplayOrientation::LandscapeFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.left),
        }};
        break;
    case Layout::DisplayOrientation::PortraitFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.right),
        }};
        std::swap(h, w);
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown DisplayOrientation: {}", orientation);
        break;
    }

    const u64 size = sizeof(ScreenRectVertex) * vertices.size();
    auto [data, offset, invalidate] = vertex_buffer.Map(size, 16);
    std::memcpy(data, vertices.data(), size);
    vertex_buffer.Commit(size);

    draw_info.output_rows = h * output_scale;
    draw_info.screen_id_l = screen_id_l;
    draw_info.screen_id_r = screen_id_r;

    scheduler.Record([this, offset = offset, info = draw_info,
                      pipeline = (layer_blends ? present_pipelines
                                               : opaque_present_pipelines)[current_pipeline]](
                         vk::CommandBuffer cmdbuf) {
        const u32 first_vertex = static_cast<u32>(offset) / sizeof(ScreenRectVertex);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdbuf.pushConstants(*present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, first_vertex, 0);
    });
}

void RendererVulkan::ApplySecondLayerOpacity(float alpha) {
    layer_blends =
        alpha < 1.0f ||
        (Common::TraceMarkers::Experiments4() & Common::TraceMarkers::LegacyScreenBlending) != 0;
    scheduler.Record([alpha](vk::CommandBuffer cmdbuf) {
        const std::array<float, 4> blend_constants = {0.0f, 0.0f, 0.0f, alpha};
        cmdbuf.setBlendConstants(blend_constants.data());
    });
}

void RendererVulkan::DrawTopScreen(const Layout::FramebufferLayout& layout,
                                   const Common::Rectangle<u32>& top_screen) {
    if (!layout.top_screen_enabled) {
        return;
    }
    int leftside, rightside;
    leftside = Settings::values.swap_eyes_3d.GetValue() ? 1 : 0;
    rightside = Settings::values.swap_eyes_3d.GetValue() ? 0 : 1;
    const float top_screen_left = static_cast<float>(top_screen.left);
    const float top_screen_top = static_cast<float>(top_screen.top);
    const float top_screen_width = static_cast<float>(top_screen.GetWidth());
    const float top_screen_height = static_cast<float>(top_screen.GetHeight());

    const auto orientation = layout.is_rotated ? Layout::DisplayOrientation::Landscape
                                               : Layout::DisplayOrientation::Portrait;
    switch (layout.render_3d_mode) {
    case Settings::StereoRenderOption::Off: {
        const int eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
        DrawSingleScreen(eye, top_screen_left, top_screen_top, top_screen_width, top_screen_height,
                         orientation);
        break;
    }
    case Settings::StereoRenderOption::SideBySide: {
        DrawSingleScreen(leftside, top_screen_left / 2, top_screen_top, top_screen_width / 2,
                         top_screen_height, orientation);
        DrawSingleScreen(rightside, static_cast<float>((top_screen_left / 2) + (layout.width / 2)),
                         top_screen_top, top_screen_width / 2, top_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::SideBySideFull: {
        DrawSingleScreen(leftside, top_screen_left, top_screen_top, top_screen_width,
                         top_screen_height, orientation);
        DrawSingleScreen(rightside, top_screen_left + layout.width / 2, top_screen_top,
                         top_screen_width, top_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::CardboardVR: {
        DrawSingleScreen(leftside, top_screen_left, top_screen_top, top_screen_width,
                         top_screen_height, orientation);
        DrawSingleScreen(
            rightside,
            static_cast<float>(layout.cardboard.top_screen_right_eye + (layout.width / 2)),
            top_screen_top, top_screen_width, top_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::Anaglyph:
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced: {
        DrawSingleScreenStereo(leftside, rightside, top_screen_left, top_screen_top,
                               top_screen_width, top_screen_height, orientation);
        break;
    }
    }
}

void RendererVulkan::DrawBottomScreen(const Layout::FramebufferLayout& layout,
                                      const Common::Rectangle<u32>& bottom_screen) {
    if (!layout.bottom_screen_enabled) {
        return;
    }

    const float bottom_screen_left = static_cast<float>(bottom_screen.left);
    const float bottom_screen_top = static_cast<float>(bottom_screen.top);
    const float bottom_screen_width = static_cast<float>(bottom_screen.GetWidth());
    const float bottom_screen_height = static_cast<float>(bottom_screen.GetHeight());

    const auto orientation = layout.is_rotated ? Layout::DisplayOrientation::Landscape
                                               : Layout::DisplayOrientation::Portrait;

    switch (layout.render_3d_mode) {
    case Settings::StereoRenderOption::Off: {
        DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                         bottom_screen_height, orientation);

        break;
    }
    case Settings::StereoRenderOption::SideBySide: // Bottom screen is identical on both sides
    {
        DrawSingleScreen(2, bottom_screen_left / 2, bottom_screen_top, bottom_screen_width / 2,
                         bottom_screen_height, orientation);
        DrawSingleScreen(2, static_cast<float>((bottom_screen_left / 2) + (layout.width / 2)),
                         bottom_screen_top, bottom_screen_width / 2, bottom_screen_height,
                         orientation);
        break;
    }
    case Settings::StereoRenderOption::SideBySideFull: {
        DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                         bottom_screen_height, orientation);
        DrawSingleScreen(2, bottom_screen_left + layout.width / 2, bottom_screen_top,
                         bottom_screen_width, bottom_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::CardboardVR: {
        DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                         bottom_screen_height, orientation);
        DrawSingleScreen(
            2, static_cast<float>(layout.cardboard.bottom_screen_right_eye + (layout.width / 2)),
            bottom_screen_top, bottom_screen_width, bottom_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::Anaglyph:
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced: {
        DrawSingleScreenStereo(2, 2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                               bottom_screen_height, orientation);
        break;
    }
    }
}

void RendererVulkan::DrawScreens(Frame* frame, const Layout::FramebufferLayout& layout,
                                 bool flipped) {
    if (settings.bg_color_update_requested.exchange(false)) {
        clear_color.float32[0] = Settings::values.bg_red.GetValue();
        clear_color.float32[1] = Settings::values.bg_green.GetValue();
        clear_color.float32[2] = Settings::values.bg_blue.GetValue();
    }
    if (settings.shader_update_requested.exchange(false)) {
        ReloadPipeline(layout.render_3d_mode);
    }

    auto uncovered = UncoveredParts(*frame, layout);
    const bool clear = !uncovered.has_value();
    PrepareDraw(frame, layout, clear,
                std::move(uncovered).value_or(std::vector<vk::ClearRect>{}));

    const auto& top_screen = layout.top_screen;
    const auto& bottom_screen = layout.bottom_screen;
    draw_info.modelview = MakeOrthographicMatrix(layout.width, layout.height, frame_rotation);
    output_scale = std::sqrt(static_cast<float>(frame->width) * static_cast<float>(frame->height) /
                             std::max(static_cast<float>(layout.width) *
                                          static_cast<float>(layout.height),
                                      1.0f));

    // Apply the initial default opacity value; Needed to avoid flickering
    ApplySecondLayerOpacity(1.0f);

    if (!Settings::values.swap_screen.GetValue()) {
        DrawTopScreen(layout, top_screen);
        if (layout.bottom_opacity < 1) {
            ApplySecondLayerOpacity(layout.bottom_opacity);
        }
        DrawBottomScreen(layout, bottom_screen);
    } else {
        DrawBottomScreen(layout, bottom_screen);
        if (layout.top_opacity < 1) {
            ApplySecondLayerOpacity(layout.top_opacity);
        }
        DrawTopScreen(layout, top_screen);
    }

    if (layout.additional_screen_enabled) {
        const auto& additional_screen = layout.additional_screen;
        if (!layout.additional_screen_is_bottom) {
            DrawTopScreen(layout, additional_screen);
        } else {
            DrawBottomScreen(layout, additional_screen);
        }
    }

    DrawCursor(layout);

    scheduler.Record([this, kind = isSecondaryWindow ? 1 : 0](vk::CommandBuffer cmdbuf) {
        cmdbuf.endRenderPass();
        draw_meter.End(cmdbuf, kind);
    });
}

void RendererVulkan::DrawCursor(const Layout::FramebufferLayout& layout) {
    const auto cursor = render_window.GetCursorInfo();
    if (!cursor.visible) {
        return;
    }

    const float buf_w = static_cast<float>(layout.width);
    const float buf_h = static_cast<float>(layout.height);

    // Convert from bottom-screen-local to layout-absolute, then to NDC
    const float abs_x = layout.bottom_screen.left + cursor.projected_x;
    const float abs_y = layout.bottom_screen.top + cursor.projected_y;
    const float cx = (abs_x / buf_w) * 2.0f - 1.0f;
    const float cy = (abs_y / buf_h) * 2.0f - 1.0f;
    const float ratio = static_cast<float>(layout.bottom_screen.GetHeight()) / 30.0f;
    const float rw = ratio / buf_w;
    const float rh = ratio / buf_h;

    // Bottom screen bounds in NDC
    const float bl = (layout.bottom_screen.left / buf_w) * 2.0f - 1.0f;
    const float bt = (layout.bottom_screen.top / buf_h) * 2.0f - 1.0f;
    const float br = (layout.bottom_screen.right / buf_w) * 2.0f - 1.0f;
    const float bb = (layout.bottom_screen.bottom / buf_h) * 2.0f - 1.0f;

    // Crosshair geometry clamped to bottom screen bounds
    const float vl = std::fmax(cx - rw / 5.0f, bl);
    const float vr = std::fmin(cx + rw / 5.0f, br);
    const float vt = std::fmax(cy - rh, bt);
    const float vb = std::fmin(cy + rh, bb);

    const float hl = std::fmax(cx - rw, bl);
    const float hr = std::fmin(cx + rw, br);
    const float ht = std::fmax(cy - rh / 5.0f, bt);
    const float hb = std::fmin(cy + rh / 5.0f, bb);

    // 12 vertices = 4 triangles (2 for vertical bar, 2 for horizontal bar)
    // clang-format off
    float vertices[] = {
        // Vertical bar
        vl, vt,  vr, vt,  vr, vb,
        vl, vt,  vr, vb,  vl, vb,
        // Horizontal bar
        hl, ht,  hr, ht,  hr, hb,
        hl, ht,  hr, hb,  hl, hb,
    };
    // clang-format on
    // Turned with the frame, see MakeOrthographicMatrix
    for (std::size_t i = 0; i + 1 < std::size(vertices); i += 2) {
        const float x = vertices[i];
        const float y = vertices[i + 1];
        if (frame_rotation == 90) {
            vertices[i] = -y;
            vertices[i + 1] = x;
        } else if (frame_rotation == 180) {
            vertices[i] = -x;
            vertices[i + 1] = -y;
        } else if (frame_rotation == 270) {
            vertices[i] = y;
            vertices[i + 1] = -x;
        }
    }

    const u64 size = sizeof(vertices);
    auto [data, offset, invalidate] = vertex_buffer.Map(size, 16);
    std::memcpy(data, vertices, size);
    vertex_buffer.Commit(size);

    scheduler.Record([this, offset = offset, pipeline = cursor_pipeline](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        const u32 first_vertex = static_cast<u32>(offset) / (sizeof(float) * 2);
        cmdbuf.draw(12, 1, first_vertex, 0);
    });
}

void RendererVulkan::SwapBuffers() {
    composing_vblank = current_frame;
    Compose();
    EndFrame();
}

void RendererVulkan::ComposeFrame(u64 vblank) {
    composing_vblank = vblank;
    Compose();
    EndGpuFrame();
}

void RendererVulkan::UpdateWideFactor(const Layout::FramebufferLayout& main_layout) {
    f32 wide_factor = 1.f;
    // Where the top screen is shown: in the main window, or on the second display alone. (With
    // both showing it, each in another shape, the one picture there is is drawn for the main one.)
    const Layout::FramebufferLayout* shown = nullptr;
    if (Layout::IsTopScreenWide()) {
        if (main_layout.top_screen_enabled) {
            shown = &main_layout;
        } else if (secondary_window && secondaryWindowEnabled &&
                   secondary_window->GetFramebufferLayout().top_screen_enabled) {
            shown = &secondary_window->GetFramebufferLayout();
        }
    }
    if (shown) {
        const f32 width = static_cast<f32>(shown->top_screen.GetWidth());
        const f32 height = static_cast<f32>(shown->top_screen.GetHeight());
        if (width >= 1.f && height >= 1.f) {
            const f32 shape = shown->is_rotated ? width / height : height / width;
            const f32 wanted = (5.f / 3.f) / shape;
            // (Not for a difference a rounded layout makes)
            if (std::abs(wanted - 1.f) >= 0.004f) {
                wide_factor = std::clamp(wanted, 0.5f, 1.5f);
            }
        }
    }
    rasterizer.SetWideFactor(wide_factor);
    if (wide_factor != last_wide_factor) {
        last_wide_factor = wide_factor;
        const Layout::FramebufferLayout& where = shown ? *shown : main_layout;
        LOG_INFO(Render_Vulkan,
                 "Top screen at {}x{} of a {} of {}x{}, bottom screen at {}x{}: the picture is "
                 "drawn for that shape with the factor {:.4f}",
                 where.top_screen.GetWidth(), where.top_screen.GetHeight(),
                 &where == &main_layout ? "main window" : "second display", where.width,
                 where.height, where.bottom_screen.GetWidth(), where.bottom_screen.GetHeight(),
                 wide_factor);
    }
}

void RendererVulkan::Compose() {
    system.perf_stats->StartSwap();
    screenRendered = false;
#ifndef ANDROID
    if (Settings::values.layout_option.GetValue() == Settings::LayoutOption::SeparateWindows) {
        ASSERT(secondary_window);
        secondaryWindowEnabled = true;
    } else {
        secondaryWindowEnabled = false;
    }
#endif

#ifdef ANDROID
    // The secondary window always exists on Android. Without a second screen it sits on a
    // hidden display, where drawing and presenting every frame would be wasted work.
    const bool use_secondary_window =
        secondary_window &&
        (secondary_window->IsVisible() ||
         (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyHiddenDisplay));
    secondaryWindowEnabled = use_secondary_window;
#endif

    const Layout::FramebufferLayout& layout = render_window.GetFramebufferLayout();
    UpdateWideFactor(layout);
    // The image of the last VBlank again would be composed and queued a second time for
    // nothing, and in the queue it gets in the way of the images that differ: it delays the
    // next one or takes the place of the one before.
    const bool present_image = ShouldPresentImage();
    if (present_image) {
        PrepareRendertarget();
        RenderScreenshot();
        if (main_present_window.DrawsIntoSwapchain()) {
            // What was drawn for the game is not to wait for an image of the display's
            renderpass_cache.EndRendering();
            scheduler.Flush();
        }
        isSecondaryWindow = false;
        RenderToWindow(main_present_window, layout, false);
    }
#ifndef ANDROID
    if (Settings::values.layout_option.GetValue() == Settings::LayoutOption::SeparateWindows) {
        ASSERT(secondary_window);
        if (present_image) {
            const auto& secondary_layout = secondary_window->GetFramebufferLayout();
            if (!secondary_present_window_ptr) {
                secondary_present_window_ptr = std::make_unique<PresentWindow>(
                    *secondary_window, instance, scheduler, IsLowRefreshRate());
            }
            isSecondaryWindow = true;
            RenderToWindow(*secondary_present_window_ptr, secondary_layout, false);
        }
        secondary_window->PollEvents();
    }
#endif

#ifdef ANDROID
    if (use_secondary_window) {
        if (present_image) {
            const auto& secondary_layout = secondary_window->GetFramebufferLayout();
            if (!secondary_present_window_ptr) {
                secondary_present_window_ptr = std::make_unique<PresentWindow>(
                    *secondary_window, instance, scheduler, IsLowRefreshRate());
            }
            isSecondaryWindow = true;
            RenderToWindow(*secondary_present_window_ptr, secondary_layout, false);
        }
        secondary_window->PollEvents();
    }
#endif
    if (!present_image) {
        // Nothing to show, but the GPU can start on what was drawn so far
        scheduler.Flush();
    } else if (!screenRendered) {
        scheduler.Finish();
    }

    // Where the window can tell when images reach the display, it keeps VBlanks in step with
    // the refreshes of that display
    system.frame_limiter.SetPacingScale(main_present_window.SpeedScale());

    system.perf_stats->EndSwap();
    rasterizer.TickFrame();
    ReportGpuTime();
}

namespace {

/// What the system says of the GPU's clock, where an app may read it
std::string GpuClockText() {
    const auto read = [](const char* path) -> std::string {
        std::FILE* const file = std::fopen(path, "r");
        if (!file) {
            return {};
        }
        char line[64]{};
        const bool got = std::fgets(line, sizeof(line), file) != nullptr;
        std::fclose(file);
        std::string text = got ? line : "";
        while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) {
            text.pop_back();
        }
        return text;
    };
    std::string now = read("/sys/class/kgsl/kgsl-3d0/gpuclk");
    if (now.empty()) {
        now = read("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq");
    }
    if (now.empty()) {
        return {};
    }
    const std::string most = read("/sys/class/kgsl/kgsl-3d0/max_gpuclk");
    return fmt::format(" (GPU clock {} of {} Hz)", now, most.empty() ? "?" : most);
}

} // Anonymous namespace

void RendererVulkan::ReportGpuTime() {
    const auto add = [](GpuSpanMeter::Sum& to, const GpuSpanMeter::Sum& sum) {
        to.time += sum.time;
        to.spans += sum.spans;
    };
    const auto draws = draw_meter.Take();
    add(reported_draws[0], draws[0]);
    add(reported_draws[1], draws[1]);
    add(reported_copies[0], main_present_window.TakeCopyTime());
    if (secondary_present_window_ptr) {
        add(reported_copies[1], secondary_present_window_ptr->TakeCopyTime());
    }

    for (const GpuPassMeter::Sum& sum : renderpass_cache.TakePassTimes()) {
        const auto same = std::ranges::find_if(reported_passes, [&](const auto& one) {
            return one.target.framebuffer == sum.target.framebuffer &&
                   one.target.loads == sum.target.loads && one.target.width == sum.target.width &&
                   one.target.height == sum.target.height;
        });
        if (same == reported_passes.end()) {
            reported_passes.push_back(sum);
            continue;
        }
        same->time += sum.time;
        same->passes += sum.passes;
        same->draws += sum.draws;
    }

    static constexpr u32 REPORT_VBLANKS = 300;
    const auto now = std::chrono::steady_clock::now();
    if (report_start == std::chrono::steady_clock::time_point{}) {
        report_start = now;
    }
    if (++report_vblanks < REPORT_VBLANKS) {
        return;
    }
    const auto ms = [](std::chrono::nanoseconds time) {
        return std::chrono::duration<double, std::milli>(time).count();
    };
    const auto per_span = [&](const GpuSpanMeter::Sum& sum) {
        return sum.spans == 0 ? 0.0 : ms(sum.time) / sum.spans;
    };
    const double wall = ms(now - report_start);
    const auto window_text = [&](const PresentWindow& window, const GpuSpanMeter::Sum& draw,
                                 const GpuSpanMeter::Sum& copy) {
        const vk::Extent2D extent = window.Extent();
        std::string text = fmt::format("{} frames of {}x{} drawn in {:.2f} ms each", draw.spans,
                                       extent.width, extent.height, per_span(draw));
        if (copy.spans != 0) {
            text += fmt::format(" and copied in {:.2f} ms", per_span(copy));
        }
        return text;
    };
    std::string second;
    if (secondary_present_window_ptr && secondaryWindowEnabled) {
        second = "; second window: " + window_text(*secondary_present_window_ptr,
                                                   reported_draws[1], reported_copies[1]);
    }
    LOG_INFO(Render_Vulkan,
             "Host GPU, the last {} VBlanks in {:.1f} s: at work {:.0f}% of the time, {:.2f} ms a "
             "VBlank, without the copies; main window: {}{}; scale {}x, upscaling {} at {}%",
             REPORT_VBLANKS, wall / 1000.0, 100.0 * ms(reported_busy) / wall,
             ms(reported_busy) / REPORT_VBLANKS,
             window_text(main_present_window, reported_draws[0], reported_copies[0]), second,
             GetResolutionScaleFactor().Factor(),
             static_cast<u32>(Settings::values.screen_upscaling.GetValue()),
             Settings::values.screen_upscaling_sharpness.GetValue());
    std::vector<GpuPassMeter::Sum> passes = std::move(reported_passes);
    reported_passes.clear();
    std::ranges::sort(passes, [](const auto& a, const auto& b) { return a.time > b.time; });
    if (!passes.empty()) {
        std::chrono::nanoseconds whole{};
        u32 count = 0;
        std::string text;
        for (const GpuPassMeter::Sum& sum : passes) {
            whole += sum.time;
            count += sum.passes;
        }
        for (std::size_t i = 0; i < std::min<std::size_t>(passes.size(), 9); i++) {
            const GpuPassMeter::Sum& sum = passes[i];
            text += fmt::format("{} {:X} {}x{} formats {}/{}{}{}{}: {:.2f} passes of {:.0f} draws, "
                                "{:.2f} ms",
                                i == 0 ? "" : ";", sum.target.framebuffer & 0xFFFFF,
                                sum.target.width, sum.target.height, sum.target.color,
                                sum.target.depth, (sum.target.loads & 1) != 0 ? " loads colour" : "",
                                (sum.target.loads & 2) != 0 ? " loads depth" : "",
                                (sum.target.loads & 4) != 0 ? " stores depth" : "",
                                static_cast<double>(sum.passes) / REPORT_VBLANKS,
                                sum.passes != 0 ? static_cast<double>(sum.draws) / sum.passes : 0.0,
                                ms(sum.time) / REPORT_VBLANKS);
        }
        LOG_INFO(Render_Vulkan,
                 "Render passes, a VBlank: {:.1f} in {:.2f} ms{}, into {} targets:{}",
                 static_cast<double>(count) / REPORT_VBLANKS, ms(whole) / REPORT_VBLANKS,
                 GpuClockText(), passes.size(), text);
    }
    report_vblanks = 0;
    report_start = now;
    reported_busy = {};
    reported_draws = {};
    reported_copies = {};
}

bool RendererVulkan::AlwaysLimitsFramesAheadOfGpu() const {
#ifdef HAVE_LIBRETRO
    return false;
#else
    // The emulation no longer waits for the render thread at every VBlank, which used to hold
    // it back (and held the GPU up with it). It may be one game frame ahead of what the GPU
    // has finished: the GPU then always has the next frame to go on with, and what is on
    // screen is no older than it has to be.
    return (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyFrameHandoff) == 0;
#endif
}

std::optional<std::chrono::nanoseconds> RendererVulkan::TakeGpuTime(bool wanted) {
    // (Always measured, for the log)
    const auto busy = scheduler.TakeGpuTime(true);
    reported_busy += busy.value_or(std::chrono::nanoseconds{});
    return wanted ? busy : std::nullopt;
}

void RendererVulkan::LimitFramesAheadOfGpu() {
    if (previous_game_frame_tick != 0) {
        scheduler.Wait(previous_game_frame_tick);
    }
    previous_game_frame_tick = scheduler.CurrentTick() - 1;
}

void RendererVulkan::RenderScreenshot() {
    if (!settings.screenshot_requested.exchange(false)) {
        return;
    }

    if (!TryRenderScreenshotWithHostMemory()) {
        RenderScreenshotWithStagingCopy();
    }

    settings.screenshot_complete_callback(false);
}

void RendererVulkan::RenderScreenshotWithStagingCopy() {
    const vk::Device device = instance.GetDevice();

    const Layout::FramebufferLayout layout{settings.screenshot_framebuffer_layout};
    const u32 width = layout.width;
    const u32 height = layout.height;

    const vk::BufferCreateInfo staging_buffer_info = {
        .size = width * height * 4,
        .usage = vk::BufferUsageFlagBits::eTransferDst,
    };

    const VmaAllocationCreateInfo alloc_create_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkBuffer unsafe_buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info;
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(staging_buffer_info);

    VkResult result = vmaCreateBuffer(instance.GetAllocator(), &unsafe_buffer_info,
                                      &alloc_create_info, &unsafe_buffer, &allocation, &alloc_info);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }

    vk::Buffer staging_buffer{unsafe_buffer};

    Frame frame{};
    main_present_window.RecreateFrame(&frame, width, height);

    DrawScreens(&frame, layout, false);

    scheduler.Record(
        [width, height, source_image = frame.image, staging_buffer](vk::CommandBuffer cmdbuf) {
            const vk::ImageMemoryBarrier read_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = source_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            const vk::ImageMemoryBarrier write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = source_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            static constexpr vk::MemoryBarrier memory_write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            };

            const vk::BufferImageCopy image_copy = {
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = {0, 0, 0},
                .imageExtent = {width, height, 1},
            };

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);
            cmdbuf.copyImageToBuffer(source_image, vk::ImageLayout::eTransferSrcOptimal,
                                     staging_buffer, image_copy);
            cmdbuf.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
                vk::DependencyFlagBits::eByRegion, memory_write_barrier, {}, write_barrier);
        });

    // Ensure the copy is fully completed before saving the screenshot
    scheduler.Finish();

    // Copy backing image data to the QImage screenshot buffer
    std::memcpy(settings.screenshot_bits, alloc_info.pMappedData, staging_buffer_info.size);

    // QImage::Format_RGB32 expects BGRA byte order. If the swapchain format is RGBA,
    // swap R and B channels so the screenshot colors are correct.
    if (main_present_window.GetSurfaceFormat() == vk::Format::eR8G8B8A8Unorm) {
        u8* pixels = static_cast<u8*>(settings.screenshot_bits);
        for (u32 i = 0; i < width * height; i++) {
            std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]);
        }
    }

    // Destroy allocated resources
    vmaDestroyBuffer(instance.GetAllocator(), staging_buffer, allocation);
    vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
    device.destroyFramebuffer(frame.framebuffer);
    device.destroyImageView(frame.image_view);
}

bool RendererVulkan::TryRenderScreenshotWithHostMemory() {
    // If the host-memory import alignment matches the allocation granularity of the platform, then
    // the entire span of memory can be trivially imported
    const bool trivial_import =
        instance.IsExternalMemoryHostSupported() &&
        instance.GetMinImportedHostPointerAlignment() == Common::GetPageSize();
    if (!trivial_import) {
        return false;
    }

    const vk::Device device = instance.GetDevice();

    const Layout::FramebufferLayout layout{settings.screenshot_framebuffer_layout};
    const u32 width = layout.width;
    const u32 height = layout.height;

    // For a span of memory [x, x + s], import [AlignDown(x, alignment), AlignUp(x + s, alignment)]
    // and maintain an offset to the start of the data
    const u64 import_alignment = instance.GetMinImportedHostPointerAlignment();
    const uintptr_t address = reinterpret_cast<uintptr_t>(settings.screenshot_bits);
    void* aligned_pointer = reinterpret_cast<void*>(Common::AlignDown(address, import_alignment));
    const u64 offset = address % import_alignment;
    const u64 aligned_size = Common::AlignUp(offset + width * height * 4ull, import_alignment);

    // Buffer<->Image mapping for the imported imported buffer
    const vk::BufferImageCopy buffer_image_copy = {
        .bufferOffset = offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1},
    };

    const vk::MemoryHostPointerPropertiesEXT import_properties =
        device.getMemoryHostPointerPropertiesEXT(
            vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, aligned_pointer);

    if (!import_properties.memoryTypeBits) {
        // Could not import memory
        return false;
    }

    const std::optional<u32> memory_type_index = FindMemoryType(
        instance.GetPhysicalDevice().getMemoryProperties(),
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        import_properties.memoryTypeBits);

    if (!memory_type_index.has_value()) {
        // Could not find memory type index
        return false;
    }

    const vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT>
        allocation_chain = {
            vk::MemoryAllocateInfo{
                .allocationSize = aligned_size,
                .memoryTypeIndex = memory_type_index.value(),
            },
            vk::ImportMemoryHostPointerInfoEXT{
                .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                .pHostPointer = aligned_pointer,
            },
        };

    // Import host memory
    const vk::UniqueDeviceMemory imported_memory =
        device.allocateMemoryUnique(allocation_chain.get());

    const vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_info =
        {
            vk::BufferCreateInfo{
                .size = aligned_size,
                .usage = vk::BufferUsageFlagBits::eTransferDst,
                .sharingMode = vk::SharingMode::eExclusive,
            },
            vk::ExternalMemoryBufferCreateInfo{
                .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
            },
        };

    // Bind imported memory to buffer
    const vk::UniqueBuffer imported_buffer = device.createBufferUnique(buffer_info.get());
    device.bindBufferMemory(imported_buffer.get(), imported_memory.get(), 0);

    Frame frame{};
    main_present_window.RecreateFrame(&frame, width, height);

    DrawScreens(&frame, layout, false);

    scheduler.Record([buffer_image_copy, source_image = frame.image,
                      imported_buffer = imported_buffer.get()](vk::CommandBuffer cmdbuf) {
        const vk::ImageMemoryBarrier read_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = source_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        const vk::ImageMemoryBarrier write_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = source_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        static constexpr vk::MemoryBarrier memory_write_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);
        cmdbuf.copyImageToBuffer(source_image, vk::ImageLayout::eTransferSrcOptimal,
                                 imported_buffer, buffer_image_copy);
        cmdbuf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
            vk::DependencyFlagBits::eByRegion, memory_write_barrier, {}, write_barrier);
    });

    // Ensure the copy is fully completed before saving the screenshot
    scheduler.Finish();

    // QImage::Format_RGB32 expects BGRA byte order. If the swapchain format is RGBA,
    // swap R and B channels so the screenshot colors are correct.
    if (main_present_window.GetSurfaceFormat() == vk::Format::eR8G8B8A8Unorm) {
        u8* pixels = static_cast<u8*>(settings.screenshot_bits);
        for (u32 i = 0; i < width * height; i++) {
            std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]);
        }
    }

    // Image data has been copied directly to host memory
    device.destroyFramebuffer(frame.framebuffer);
    device.destroyImageView(frame.image_view);

    return true;
}

void RendererVulkan::NotifySurfaceChanged(bool is_second_window) {
    if (is_second_window) {
        if (secondary_present_window_ptr) {
            secondary_present_window_ptr->NotifySurfaceChanged();
        }
    } else {
        main_present_window.NotifySurfaceChanged();
    }
}

} // namespace Vulkan
