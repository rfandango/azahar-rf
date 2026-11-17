// Copyright 2024-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <limits>
#include <boost/container/static_vector.hpp>
#include "common/assert.h"
#include "common/trace_markers.h"
#include "video_core/frame_state_dump.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

namespace Vulkan {

constexpr u32 MinDrawsToFlush = 20;

using VideoCore::PixelFormat;
using VideoCore::SurfaceType;

RenderManager::RenderManager(const Instance& instance, Scheduler& scheduler)
    : instance{instance}, scheduler{scheduler} {}

RenderManager::~RenderManager() = default;

void RenderManager::BeginRendering(const Framebuffer* framebuffer, Common::Rectangle<u32> draw_rect,
                                   bool overwrites_color, bool overwrites_depth) {
    const bool legacy =
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyAttachments) != 0;
    vk::Rect2D render_area = {
        .offset{
            .x = static_cast<s32>(draw_rect.left),
            .y = static_cast<s32>(draw_rect.bottom),
        },
        .extent{
            .width = draw_rect.GetWidth(),
            .height = draw_rect.GetHeight(),
        },
    };
    // Games draw into one part of a target and then into another: a quad that clears a few
    // pixels more than the scene after it covers, for one. Each draw is confined by its
    // scissor, so the pass that is open goes on as long as its area holds the draw. Ending it
    // has the GPU store everything and load it again. When it does have to be ended for a
    // wider area, the new one takes the old area along, for the draws that go back to it.
    const vk::Rect2D drawn_to = render_area;
    if (pass.render_pass && !pass.do_clear && pass.framebuffer == framebuffer->Handle() &&
        !legacy) {
        const vk::Rect2D& open = pass.render_area;
        const s32 left = std::min(open.offset.x, render_area.offset.x);
        const s32 bottom = std::min(open.offset.y, render_area.offset.y);
        const s32 right =
            std::max(open.offset.x + static_cast<s32>(open.extent.width),
                     render_area.offset.x + static_cast<s32>(render_area.extent.width));
        const s32 top =
            std::max(open.offset.y + static_cast<s32>(open.extent.height),
                     render_area.offset.y + static_cast<s32>(render_area.extent.height));
        render_area = vk::Rect2D{
            .offset{.x = left, .y = bottom},
            .extent{.width = static_cast<u32>(right - left),
                    .height = static_cast<u32>(top - bottom)},
        };
        if (render_area == open) {
            // Whatever it loaded when it began, the pass that is open is the one to go on with
            num_draws++;
            return;
        }
    }

    // A pass first loads what its area holds, from memory into the GPU's tile buffer: a copy
    // of every pixel, which is for nothing if the first draw overwrites them all. That is how
    // games clear a buffer and how they run a full-screen effect.
    vk::RenderPass render_pass = framebuffer->RenderPass();
    if (!legacy && render_area == drawn_to && framebuffer->Samples() == 1) {
        const VideoCore::PixelFormat color = framebuffer->Format(SurfaceType::Color);
        const VideoCore::PixelFormat depth = framebuffer->Format(SurfaceType::Depth);
        const bool discard_color = overwrites_color && color != PixelFormat::Invalid;
        const bool discard_depth = overwrites_depth && depth != PixelFormat::Invalid;
        if (discard_color || discard_depth) {
            CITRA_FRAME_DUMP("  vk renderpass does not load:{}{}", discard_color ? " colour" : "",
                             discard_depth ? " depth" : "");
            render_pass = GetDiscardingRenderpass(color, depth, discard_color, discard_depth);
        }
    }
    const RenderPass new_pass = {
        .framebuffer = framebuffer->Handle(),
        .render_pass = render_pass,
        .render_area = render_area,
        .clears = {},
        .do_clear = false,
    };
    images = framebuffer->Images();
    aspects = framebuffer->Aspects();
    shadow_rendering = framebuffer->shadow_rendering;
    BeginRendering(new_pass);
}

void RenderManager::BeginRendering(const RenderPass& new_pass) {
    if (pass == new_pass) [[likely]] {
        num_draws++;
        return;
    }

    EndRendering();
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "what came before this render pass");
    CITRA_FRAME_DUMP("vk renderpass begin fb={:X} area {},{} {}x{} clear={}",
                     reinterpret_cast<uintptr_t>(static_cast<VkFramebuffer>(new_pass.framebuffer)),
                     new_pass.render_area.offset.x, new_pass.render_area.offset.y,
                     new_pass.render_area.extent.width, new_pass.render_area.extent.height,
                     new_pass.do_clear);
    scheduler.Record([info = new_pass](vk::CommandBuffer cmdbuf) {
        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = info.render_pass,
            .framebuffer = info.framebuffer,
            .renderArea = info.render_area,
            .clearValueCount = info.do_clear ? 2u : 0u,
            .pClearValues = info.clears.data(),
        };
        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
    });

    pass = new_pass;
}

void RenderManager::EndRendering() {
    if (!pass.render_pass) {
        return;
    }

    CITRA_FRAME_DUMP("vk renderpass end");
    scheduler.Record([images = images, aspects = aspects,
                      shadow_rendering = shadow_rendering](vk::CommandBuffer cmdbuf) {
        u32 num_barriers = 0;
        vk::PipelineStageFlags pipeline_flags{};
        vk::AccessFlags src_access_flags{};
        std::array<vk::ImageMemoryBarrier, 4> barriers;
        for (u32 i = 0; i < images.size(); i++) {
            if (!images[i]) {
                continue;
            }
            const bool is_color = static_cast<bool>(aspects[i] & vk::ImageAspectFlagBits::eColor);
            if (is_color) {
                pipeline_flags |= shadow_rendering
                                      ? vk::PipelineStageFlagBits::eFragmentShader
                                      : vk::PipelineStageFlagBits::eColorAttachmentOutput;
                src_access_flags = shadow_rendering ? vk::AccessFlagBits::eShaderWrite
                                                    : vk::AccessFlagBits::eColorAttachmentWrite;
            } else {
                pipeline_flags |= vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                  vk::PipelineStageFlagBits::eLateFragmentTests;
                src_access_flags = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            }
            barriers[num_barriers++] = vk::ImageMemoryBarrier{
                .srcAccessMask = src_access_flags,
                .dstAccessMask =
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = images[i],
                .subresourceRange{
                    .aspectMask = aspects[i],
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
        }
        cmdbuf.endRenderPass();
        if (num_barriers == 0) {
            return;
        }
        cmdbuf.pipelineBarrier(pipeline_flags,
                               vk::PipelineStageFlagBits::eFragmentShader |
                                   vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr,
                               num_barriers, barriers.data());
    });

    // Reset state.
    pass.render_pass = VK_NULL_HANDLE;
    images = {};
    aspects = {};
    shadow_rendering = false;
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "render pass");

    // The Mali guide recommends flushing at the end of each major renderpass
    // Testing has shown this has a significant effect on rendering performance
    if (num_draws > MinDrawsToFlush && instance.ShouldFlush()) {
        scheduler.Flush();
        num_draws = 0;
    }
}

vk::RenderPass RenderManager::GetRenderpass(VideoCore::PixelFormat color,
                                            VideoCore::PixelFormat depth, bool is_clear,
                                            u8 sample_count) {
    std::scoped_lock lock{cache_mutex};

    const u32 color_index =
        color == VideoCore::PixelFormat::Invalid ? NumColorFormats : static_cast<u32>(color);
    const u32 depth_index =
        depth == VideoCore::PixelFormat::Invalid
            ? NumDepthFormats
            : (static_cast<u32>(depth - VideoCore::PixelFormat::NumColorFormat));

    ASSERT_MSG(color_index <= NumColorFormats && depth_index <= NumDepthFormats,
               "Invalid color index {} and/or depth_index {}", color_index, depth_index);

    ASSERT_MSG(sample_count && std::has_single_bit(sample_count) && sample_count <= MaxSamples,
               "Invalid sample count {}", static_cast<u32>(sample_count));

    const u32 samples_index = static_cast<u32>(std::bit_width(sample_count) - 1);

    vk::UniqueRenderPass& renderpass =
        cached_renderpasses[color_index][depth_index][samples_index][is_clear];
    if (!renderpass) {
        const vk::Format color_format = instance.GetTraits(color).native;
        const vk::Format depth_format = instance.GetTraits(depth).native;
        const vk::AttachmentLoadOp load_op =
            is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;

        renderpass = (sample_count > 1)
                         ? CreateRenderPassMSAA(color_format, depth_format, load_op,
                                                static_cast<vk::SampleCountFlagBits>(sample_count))
                         : CreateRenderPass(color_format, depth_format, load_op, load_op);
    }

    return *renderpass;
}

vk::RenderPass RenderManager::GetDiscardingRenderpass(VideoCore::PixelFormat color,
                                                      VideoCore::PixelFormat depth,
                                                      bool discard_color, bool discard_depth) {
    ASSERT(discard_color || discard_depth);
    std::scoped_lock lock{cache_mutex};

    const u32 color_index =
        color == VideoCore::PixelFormat::Invalid ? NumColorFormats : static_cast<u32>(color);
    const u32 depth_index =
        depth == VideoCore::PixelFormat::Invalid
            ? NumDepthFormats
            : (static_cast<u32>(depth - VideoCore::PixelFormat::NumColorFormat));
    ASSERT_MSG(color_index <= NumColorFormats && depth_index <= NumDepthFormats,
               "Invalid color index {} and/or depth_index {}", color_index, depth_index);

    const u32 which = (discard_color ? 1u : 0u) + (discard_depth ? 2u : 0u) - 1u;
    vk::UniqueRenderPass& renderpass = discarding_renderpasses[color_index][depth_index][which];
    if (!renderpass) {
        const auto load_op = [](bool discard) {
            return discard ? vk::AttachmentLoadOp::eDontCare : vk::AttachmentLoadOp::eLoad;
        };
        // Compatible with the pass that loads both, so framebuffers and pipelines made for
        // that one go with this one as well
        renderpass =
            CreateRenderPass(instance.GetTraits(color).native, instance.GetTraits(depth).native,
                             load_op(discard_color), load_op(discard_depth));
    }
    return *renderpass;
}

vk::UniqueRenderPass RenderManager::CreateRenderPass(vk::Format color, vk::Format depth,
                                                     vk::AttachmentLoadOp color_load_op,
                                                     vk::AttachmentLoadOp depth_load_op) const {

    boost::container::static_vector<vk::AttachmentDescription, 2> attachments{};
    bool use_color = false;
    vk::AttachmentReference color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments.emplace_back(vk::AttachmentDescription{
            .format = color,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = color_load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        });

        color_attachment_ref = vk::AttachmentReference{
            .attachment = static_cast<u32>(attachments.size() - 1),
            .layout = vk::ImageLayout::eGeneral,
        };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        attachments.emplace_back(vk::AttachmentDescription{
            .format = depth,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = depth_load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = depth_load_op,
            .stencilStoreOp = vk::AttachmentStoreOp::eStore,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        });

        depth_attachment_ref = vk::AttachmentReference{
            .attachment = static_cast<u32>(attachments.size() - 1),
            .layout = vk::ImageLayout::eGeneral,
        };

        use_depth = true;
    }

    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = use_color ? 1u : 0u,
        .pColorAttachments = &color_attachment_ref,
        .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr,
    };

    const vk::RenderPassCreateInfo renderpass_info = {
        .attachmentCount = static_cast<u32>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = nullptr,
    };

    return instance.GetDevice().createRenderPassUnique(renderpass_info);
}

vk::UniqueRenderPass RenderManager::CreateRenderPassMSAA(
    vk::Format color, vk::Format depth, vk::AttachmentLoadOp load_op,
    vk::SampleCountFlagBits sample_count) const {
    boost::container::static_vector<vk::AttachmentDescription2, 4> attachments{};

    vk::AttachmentReference2 color_resolve_attachment = {.attachment = VK_ATTACHMENT_UNUSED};
    vk::AttachmentReference2 depth_resolve_attachment = {.attachment = VK_ATTACHMENT_UNUSED};

    bool use_color = false;
    vk::AttachmentReference2 color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference2 depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments.emplace_back(vk::AttachmentDescription2{
            .format = color,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        });

        color_attachment_ref = vk::AttachmentReference2{
            .attachment = static_cast<u32>(attachments.size() - 1),
            .layout = vk::ImageLayout::eGeneral,
            .aspectMask = vk::ImageAspectFlagBits::eColor,
        };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        attachments.emplace_back(vk::AttachmentDescription2{
            .format = depth,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = load_op,
            .stencilStoreOp = vk::AttachmentStoreOp::eStore,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        });

        depth_attachment_ref = vk::AttachmentReference2{
            .attachment = static_cast<u32>(attachments.size() - 1),
            .layout = vk::ImageLayout::eGeneral,
            .aspectMask = vk::ImageAspectFlagBits::eDepth,
        };

        use_depth = true;
    }

    // In the case of MSAA, each attachment gets an additional MSAA attachment that now becomes the
    // main attachment and the original attachments now get resolved into
    if (sample_count > vk::SampleCountFlagBits::e1) {
        if (color != vk::Format::eUndefined) {
            attachments.emplace_back(vk::AttachmentDescription2{
                .format = color,
                .samples = sample_count,
                .loadOp = load_op,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                .initialLayout = vk::ImageLayout::eGeneral,
                .finalLayout = vk::ImageLayout::eGeneral,
            });

            color_resolve_attachment = color_attachment_ref;

            color_attachment_ref = vk::AttachmentReference2{
                .attachment = static_cast<u32>(attachments.size() - 1),
                .layout = vk::ImageLayout::eGeneral,
            };
        }

        if (depth != vk::Format::eUndefined) {
            attachments.emplace_back(vk::AttachmentDescription2{
                .format = depth,
                .samples = sample_count,
                .loadOp = load_op,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .stencilLoadOp = load_op,
                .stencilStoreOp = vk::AttachmentStoreOp::eStore,
                .initialLayout = vk::ImageLayout::eGeneral,
                .finalLayout = vk::ImageLayout::eGeneral,
            });

            depth_resolve_attachment = depth_attachment_ref;

            depth_attachment_ref = vk::AttachmentReference2{
                .attachment = static_cast<u32>(attachments.size() - 1),
                .layout = vk::ImageLayout::eGeneral,
            };
        }
    }

    const vk::StructureChain<vk::SubpassDescription2, vk::SubpassDescriptionDepthStencilResolve>
        subpass = {
            vk::SubpassDescription2{
                .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                .inputAttachmentCount = 0,
                .pInputAttachments = nullptr,
                .colorAttachmentCount = use_color ? 1u : 0u,
                .pColorAttachments = &color_attachment_ref,
                .pResolveAttachments = &color_resolve_attachment,
                .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr,
            },
            vk::SubpassDescriptionDepthStencilResolve{
                .depthResolveMode = vk::ResolveModeFlagBits::eSampleZero,
                .stencilResolveMode = vk::ResolveModeFlagBits::eSampleZero,
                .pDepthStencilResolveAttachment = &depth_resolve_attachment},
        };

    const vk::RenderPassCreateInfo2 renderpass_info = {
        .attachmentCount = static_cast<u32>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass.get(),
        .dependencyCount = 0,
        .pDependencies = nullptr,
    };

    return instance.GetDevice().createRenderPass2Unique(renderpass_info);
}

} // namespace Vulkan
