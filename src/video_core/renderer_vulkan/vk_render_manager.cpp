// Copyright 2024-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <limits>
#include <string>
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
    : instance{instance}, scheduler{scheduler}, pass_meter{instance} {}

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
    if (pass.render_pass && pass.framebuffer == framebuffer->Handle() && !legacy) {
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
            pass_draws++;
            return;
        }
    }

    // Clears that were held back for this pass. It can do them as it begins if they are all
    // for attachments of its framebuffer and have one area, which holds the draw: the pass
    // takes that area and starts those attachments from the clear value.
    const std::array<vk::Image, 4> attached = framebuffer->Images();
    std::array<bool, 2> cleared{};
    std::array<vk::ClearValue, 2> clear_values{};
    if (!deferred_clears.empty()) {
        const vk::Rect2D area = deferred_clears.front().area;
        const vk::Extent2D extent = framebuffer->Extent();
        const auto right = [](const vk::Rect2D& rect) {
            return rect.offset.x + static_cast<s32>(rect.extent.width);
        };
        const auto top = [](const vk::Rect2D& rect) {
            return rect.offset.y + static_cast<s32>(rect.extent.height);
        };
        bool usable = !legacy && framebuffer->Samples() == 1 && framebuffer->color_level == 0 &&
                      !framebuffer->shadow_rendering && area.offset.x <= render_area.offset.x &&
                      area.offset.y <= render_area.offset.y && right(area) >= right(render_area) &&
                      top(area) >= top(render_area) &&
                      right(area) <= static_cast<s32>(extent.width) &&
                      top(area) <= static_cast<s32>(extent.height);
        for (const DeferredClear& clear : deferred_clears) {
            usable = usable && clear.area == area &&
                     (clear.image == attached[0] || clear.image == attached[1]);
        }
        if (usable) {
            for (const DeferredClear& clear : deferred_clears) {
                const u32 index = clear.image == attached[0] ? 0 : 1;
                cleared[index] = true;
                clear_values[index] = clear.value;
            }
            deferred_clears.clear();
            render_area = area;
        } else {
            CarryOutClears();
        }
    }

    // Otherwise a pass first loads what its area holds, from memory into the GPU's tile
    // buffer: a copy of every pixel, which is for nothing if the first draw overwrites them
    // all. That is how games clear a buffer and how they run a full-screen effect.
    vk::RenderPass render_pass = framebuffer->RenderPass();
    const u32 experiments2 = Common::TraceMarkers::Experiments2();
    const bool never_load = (experiments2 & Common::TraceMarkers::NoAttachmentLoads) != 0;
    const bool never_store_depth = (experiments2 & Common::TraceMarkers::NoDepthStores) != 0;
    const VideoCore::PixelFormat color = framebuffer->Format(SurfaceType::Color);
    const VideoCore::PixelFormat depth = framebuffer->Format(SurfaceType::Depth);
    next_target.color = static_cast<u32>(color);
    next_target.depth = static_cast<u32>(depth);
    next_target.loads = 7;
    if (!legacy && framebuffer->Samples() == 1) {
        const bool at_start = render_area == drawn_to;
        const auto load_op = [&](u32 index, VideoCore::PixelFormat format, bool overwritten) {
            if (format == PixelFormat::Invalid) {
                return vk::AttachmentLoadOp::eLoad;
            }
            if (cleared[index]) {
                return vk::AttachmentLoadOp::eClear;
            }
            return (overwritten && at_start) || never_load ? vk::AttachmentLoadOp::eDontCare
                                                           : vk::AttachmentLoadOp::eLoad;
        };
        const vk::AttachmentLoadOp color_load = load_op(0, color, overwrites_color);
        const vk::AttachmentLoadOp depth_load = load_op(1, depth, overwrites_depth);
        const bool store_depth = !never_store_depth || depth == PixelFormat::Invalid;
        next_target.loads = (color_load == vk::AttachmentLoadOp::eLoad ? 1u : 0u) |
                            (depth_load == vk::AttachmentLoadOp::eLoad ? 2u : 0u) |
                            (store_depth ? 4u : 0u);
        if (color_load != vk::AttachmentLoadOp::eLoad ||
            depth_load != vk::AttachmentLoadOp::eLoad || !store_depth) {
            const auto is = [&](vk::AttachmentLoadOp op, const char* colour_text,
                                const char* depth_text) {
                return std::string{color_load == op ? colour_text : ""} +
                       (depth_load == op ? depth_text : "");
            };
            CITRA_FRAME_DUMP("  vk renderpass does not load:{} | clears:{}",
                             is(vk::AttachmentLoadOp::eDontCare, " colour", " depth"),
                             is(vk::AttachmentLoadOp::eClear, " colour", " depth"));
            render_pass =
                GetRenderpassBeginningWith(color, depth, color_load, depth_load, store_depth);
        }
    }
    RenderPass new_pass = {
        .framebuffer = framebuffer->Handle(),
        .render_pass = render_pass,
        .render_area = render_area,
        .clears = {},
        .do_clear = cleared[0] || cleared[1],
    };
    // Clear values go by attachment, and colour is the first one if there is colour
    u32 attachment = 0;
    if (color != PixelFormat::Invalid) {
        new_pass.clears[attachment++] = clear_values[0];
    }
    if (depth != PixelFormat::Invalid) {
        new_pass.clears[attachment++] = clear_values[1];
    }
    images = attached;
    aspects = framebuffer->Aspects();
    shadow_rendering = framebuffer->shadow_rendering;
    BeginRendering(new_pass);
}

void RenderManager::BeginRendering(const RenderPass& new_pass) {
    if (pass == new_pass) [[likely]] {
        num_draws++;
        pass_draws++;
        return;
    }

    EndRendering();
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "what came before this render pass");
    CITRA_FRAME_DUMP("vk renderpass begin fb={:X} area {},{} {}x{} clear={}",
                     reinterpret_cast<uintptr_t>(static_cast<VkFramebuffer>(new_pass.framebuffer)),
                     new_pass.render_area.offset.x, new_pass.render_area.offset.y,
                     new_pass.render_area.extent.width, new_pass.render_area.extent.height,
                     new_pass.do_clear);
    scheduler.Record([this, info = new_pass](vk::CommandBuffer cmdbuf) {
        pass_meter.Begin(cmdbuf);
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
    pass_draws = 1;
    pass_target = next_target;
    next_target = {};
    pass_target.framebuffer =
        reinterpret_cast<uintptr_t>(static_cast<VkFramebuffer>(new_pass.framebuffer));
    pass_target.width = new_pass.render_area.extent.width;
    pass_target.height = new_pass.render_area.extent.height;
}

void RenderManager::EndRendering() {
    EndPass();
    CarryOutClears();
}

void RenderManager::SetClearDeferral(bool on) {
    defer_clears = on && (Common::TraceMarkers::Experiments2() &
                          Common::TraceMarkers::LegacyClearBeforePass) == 0;
    if (!defer_clears) {
        CarryOutClears();
    }
}

void RenderManager::DeferClear(DeferredClear&& clear) {
    EndPass();
    // One clear an image, and no more than the two attachments a pass has
    const bool again = std::ranges::any_of(
        deferred_clears, [&clear](const DeferredClear& held) { return held.image == clear.image; });
    if (again || deferred_clears.size() == deferred_clears.capacity()) {
        CarryOutClears();
    }
    deferred_clears.push_back(std::move(clear));
}

void RenderManager::CarryOutClears() {
    if (deferred_clears.empty()) {
        return;
    }
    // Taken out first: recording may come back here
    boost::container::static_vector<DeferredClear, 2> clears;
    for (DeferredClear& clear : deferred_clears) {
        clears.push_back(std::move(clear));
    }
    deferred_clears.clear();
    for (const DeferredClear& clear : clears) {
        clear.carry_out();
    }
}

void RenderManager::EndPass() {
    if (!pass.render_pass) {
        return;
    }

    CITRA_FRAME_DUMP("vk renderpass end");
    scheduler.Record([this, images = images, aspects = aspects,
                      shadow_rendering = shadow_rendering, target = pass_target,
                      draws = pass_draws](vk::CommandBuffer cmdbuf) {
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
        pass_meter.End(cmdbuf, target, draws);
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
                         : CreateRenderPass(color_format, depth_format, load_op, load_op,
                                            vk::AttachmentStoreOp::eStore);
    }

    return *renderpass;
}

vk::RenderPass RenderManager::GetRenderpassBeginningWith(VideoCore::PixelFormat color,
                                                         VideoCore::PixelFormat depth,
                                                         vk::AttachmentLoadOp color_load,
                                                         vk::AttachmentLoadOp depth_load,
                                                         bool store_depth) {
    static_assert(static_cast<u32>(vk::AttachmentLoadOp::eLoad) == 0 &&
                  static_cast<u32>(vk::AttachmentLoadOp::eClear) == 1 &&
                  static_cast<u32>(vk::AttachmentLoadOp::eDontCare) == 2);
    std::scoped_lock lock{cache_mutex};

    const u32 color_index =
        color == VideoCore::PixelFormat::Invalid ? NumColorFormats : static_cast<u32>(color);
    const u32 depth_index =
        depth == VideoCore::PixelFormat::Invalid
            ? NumDepthFormats
            : (static_cast<u32>(depth - VideoCore::PixelFormat::NumColorFormat));
    ASSERT_MSG(color_index <= NumColorFormats && depth_index <= NumDepthFormats,
               "Invalid color index {} and/or depth_index {}", color_index, depth_index);

    vk::UniqueRenderPass& renderpass =
        beginning_renderpasses[color_index][depth_index][store_depth ? 1 : 0]
                              [static_cast<u32>(color_load)][static_cast<u32>(depth_load)];
    if (!renderpass) {
        // Compatible with the pass that loads both, so framebuffers and pipelines made for
        // that one go with this one as well
        renderpass = CreateRenderPass(instance.GetTraits(color).native,
                                      instance.GetTraits(depth).native, color_load, depth_load,
                                      store_depth ? vk::AttachmentStoreOp::eStore
                                                  : vk::AttachmentStoreOp::eDontCare);
    }
    return *renderpass;
}

vk::UniqueRenderPass RenderManager::CreateRenderPass(vk::Format color, vk::Format depth,
                                                     vk::AttachmentLoadOp color_load_op,
                                                     vk::AttachmentLoadOp depth_load_op,
                                                     vk::AttachmentStoreOp depth_store_op) const {

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
            .storeOp = depth_store_op,
            .stencilLoadOp = depth_load_op,
            .stencilStoreOp = depth_store_op,
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
