// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include "common/hitch_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

#include <algorithm>
#include <limits>
#include <span>
#include <string>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <vulkan/vulkan.hpp>
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/renderer_vulkan/vk_blit_helper.h"
#include "video_core/renderer_vulkan/vk_descriptor_update_queue.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"

#include "common/literals.h"
#include "common/microprofile.h"
#include "common/scope_exit.h"
#include "common/trace_markers.h"
#include "video_core/custom_textures/material.h"
#include "video_core/frame_state_dump.h"
#include "video_core/rasterizer_cache/texture_codec.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_format_traits.hpp>

MICROPROFILE_DEFINE(Vulkan_ImageAlloc, "Vulkan", "Texture Allocation", MP_RGB(192, 52, 235));

namespace Vulkan {

namespace {

using VideoCore::MapType;
using VideoCore::PixelFormat;
using VideoCore::SurfaceType;
using VideoCore::TextureType;
using namespace Common::Literals;

struct RecordParams {
    vk::ImageAspectFlags aspect;
    vk::Filter filter;
    vk::PipelineStageFlags pipeline_flags;
    vk::AccessFlags src_access;
    vk::AccessFlags dst_access;
    vk::Image src_image;
    vk::Image dst_image;
};

// Defined with the blits that wait, see TextureRuntime::BlitTextures
bool IsOneToOne(const VideoCore::TextureBlit& blit);
Common::Rectangle<u32> SourceOf(const VideoCore::TextureBlit& blit,
                                const Common::Rectangle<u32>& rect);

vk::Filter MakeFilter(VideoCore::PixelFormat pixel_format) {
    switch (pixel_format) {
    case VideoCore::PixelFormat::D16:
    case VideoCore::PixelFormat::D24:
    case VideoCore::PixelFormat::D24S8:
        return vk::Filter::eNearest;
    default:
        return vk::Filter::eLinear;
    }
}

[[nodiscard]] vk::ClearValue MakeClearValue(VideoCore::ClearValue clear) {
    static_assert(sizeof(VideoCore::ClearValue) == sizeof(vk::ClearValue));

    vk::ClearValue value{};
    std::memcpy(&value, &clear, sizeof(vk::ClearValue));
    return value;
}

[[nodiscard]] vk::ClearColorValue MakeClearColorValue(Common::Vec4f color) {
    return vk::ClearColorValue{
        .float32 = std::array{color[0], color[1], color[2], color[3]},
    };
}

[[nodiscard]] vk::ClearDepthStencilValue MakeClearDepthStencilValue(VideoCore::ClearValue clear) {
    return vk::ClearDepthStencilValue{
        .depth = clear.depth,
        .stencil = clear.stencil,
    };
}

u32 UnpackDepthStencil(const VideoCore::StagingData& data, vk::Format dest) {
    u32 depth_offset = 0;
    u32 stencil_offset = 4 * data.size / 5;
    const auto& mapped = data.mapped;

    switch (dest) {
    case vk::Format::eD24UnormS8Uint: {
        for (; stencil_offset < data.size; depth_offset += 4) {
            u8* ptr = mapped.data() + depth_offset;
            const u32 d24s8 = VideoCore::LoadFromBytes<u32>(ptr);
            const u32 d24 = d24s8 >> 8;
            mapped[stencil_offset] = d24s8 & 0xFF;
            std::memcpy(ptr, &d24, 4);
            stencil_offset++;
        }
        break;
    }
    case vk::Format::eD32SfloatS8Uint: {
        for (; stencil_offset < data.size; depth_offset += 4) {
            u8* ptr = mapped.data() + depth_offset;
            const u32 d24s8 = VideoCore::LoadFromBytes<u32>(ptr);
            const float d32 = (d24s8 >> 8) / 16777215.f;
            mapped[stencil_offset] = d24s8 & 0xFF;
            std::memcpy(ptr, &d32, 4);
            stencil_offset++;
        }
        break;
    }
    default:
        LOG_ERROR(Render_Vulkan, "Unimplemented convertion for depth format {}",
                  vk::to_string(dest));
        UNREACHABLE();
    }

    ASSERT(depth_offset == 4 * data.size / 5);
    return depth_offset;
}

void MakeInitBarriers(vk::ImageAspectFlags aspect, u32 num_images,
                      std::span<const vk::Image> images,
                      std::span<vk::ImageMemoryBarrier> out_barriers) {
    for (u32 i = 0; i < num_images; i++) {
        out_barriers[i] = vk::ImageMemoryBarrier{
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eNone,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = images[i],
            .subresourceRange{
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
    }
}

void MakeInitBarriers(vk::ImageAspectFlags aspect, std::span<const vk::Image> images,
                      std::span<vk::ImageMemoryBarrier> out_barriers) {
    MakeInitBarriers(aspect, images.size(), images, out_barriers);
}

/// Whether a new image of this format is written once in full when it is made, see InitImages.
/// Not one in a block-compressed format, which cannot be cleared and is uploaded whole.
bool WritesNewImages(vk::Format format) {
    return !vk::isCompressed(format) &&
           (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::UnwrittenImages) == 0;
}

/// Brings new images into the layout they are used in (at most four), and writes the ones in
/// `written` once in full.
///
/// An image nothing was written to is not harmless to draw into or to load. Where the GPU
/// keeps side data for an image (an Adreno's framebuffer compression has flags for every block
/// of pixels, which it consults when it writes into a block and not only when it reads one), a
/// new image has there whatever its memory held before, and once a game has run a while the
/// memory of a new image is memory the allocator has handed out before. A render pass that
/// begins on such an image without a clear (its first draw covers the viewport, or it loads
/// what is there) has the GPU take those bytes for flags. With a Turnip that does not
/// initialise them itself, an Adreno 650 locked up so that the whole device reset: Shinobi 3D
/// at the scene change after its opening video, four times out of four (2026-11-23); the
/// image was the second image of a colour target, made for a full-screen effect that samples
/// what it draws over. Clearing all of an image writes every block and its flags.
void InitImages(vk::CommandBuffer cmdbuf, vk::ImageAspectFlags aspect,
                std::span<const vk::Image> images, std::span<const vk::Image> written) {
    std::array<vk::ImageMemoryBarrier, 4> barriers;
    MakeInitBarriers(aspect, images, barriers);
    if (!written.empty()) {
        for (std::size_t i = 0; i < images.size(); i++) {
            barriers[i].dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        }
    }
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                           written.empty() ? vk::PipelineStageFlagBits::eTopOfPipe
                                           : vk::PipelineStageFlagBits::eTransfer,
                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr,
                           static_cast<u32>(images.size()), barriers.data());
    const vk::ImageSubresourceRange range = {
        .aspectMask = aspect,
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
    for (const vk::Image image : written) {
        if (aspect & vk::ImageAspectFlagBits::eColor) {
            cmdbuf.clearColorImage(image, vk::ImageLayout::eGeneral, vk::ClearColorValue{}, range);
        } else {
            cmdbuf.clearDepthStencilImage(image, vk::ImageLayout::eGeneral,
                                          vk::ClearDepthStencilValue{}, range);
        }
    }
}

vk::ImageSubresourceRange MakeSubresourceRange(vk::ImageAspectFlags aspect, u32 level = 0,
                                               u32 levels = 1, u32 layer = 0) {
    return vk::ImageSubresourceRange{
        .aspectMask = aspect,
        .baseMipLevel = level,
        .levelCount = levels,
        .baseArrayLayer = layer,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
}

constexpr u64 UPLOAD_BUFFER_SIZE = 512_MiB;
constexpr u64 DOWNLOAD_BUFFER_SIZE = 16_MiB;

} // Anonymous namespace

void Handle::Create(u32 width, u32 height, u32 levels, TextureType type, vk::Format format,
                    vk::SampleCountFlagBits samples, vk::ImageUsageFlags usage,
                    vk::ImageCreateFlags flags, vk::ImageAspectFlags aspect, bool need_format_list,
                    std::string_view debug_name, bool stencil_sampled) {

    Destroy();

    const bool is_cube_map = type == TextureType::CubeMap && instance.IsLayeredRenderingSupported();
    if (!is_cube_map) {
        flags &= ~vk::ImageCreateFlagBits::eCubeCompatible;
    }

    this->width = width;
    this->height = height;
    this->levels = levels;
    this->layers = is_cube_map ? 6 : 1;

    const std::array format_list = {
        vk::Format::eR8G8B8A8Unorm,
        vk::Format::eR32Uint,
    };
    const vk::ImageFormatListCreateInfo image_format_list = {
        .viewFormatCount = static_cast<u32>(format_list.size()),
        .pViewFormats = format_list.data(),
    };

    // What the stencil is used for, where that is less than what the depth is used for
    const vk::ImageStencilUsageCreateInfo stencil_usage = {
        .stencilUsage =
            usage & ~(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eInputAttachment),
    };
    const bool separate_stencil = !stencil_sampled && !need_format_list &&
                                  instance.IsSeparateStencilUsageSupported() &&
                                  static_cast<bool>(aspect & vk::ImageAspectFlagBits::eStencil) &&
                                  stencil_usage.stencilUsage != usage;
    const void* const next =
        need_format_list ? static_cast<const void*>(&image_format_list)
                         : (separate_stencil ? static_cast<const void*>(&stencil_usage) : nullptr);

    const vk::ImageCreateInfo image_info = {
        .pNext = next,
        .flags = flags,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = levels,
        .arrayLayers = layers,
        .samples = samples,
        .usage = usage,
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
                                     &unsafe_image, &allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating image with error {}", result);
        UNREACHABLE();
    }

    image = vk::Image{unsafe_image};

    const vk::ImageViewCreateInfo view_info = {
        .image = image,
        .viewType = is_cube_map ? vk::ImageViewType::eCube : vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    image_views[ViewType::Sample] = instance.GetDevice().createImageView(view_info);
    if (levels == 1) {
        image_views[ViewType::Mip0] = image_views[ViewType::Sample];
    }

    if (!debug_name.empty() && instance.HasDebuggingToolAttached()) {
        SetObjectName(instance.GetDevice(), image, debug_name);
        SetObjectName(instance.GetDevice(), image_views[ViewType::Sample], "{} View({})",
                      debug_name, vk::to_string(aspect));
    }
}

void Handle::Destroy() {
    const auto device = instance.GetDevice();

    // Image views
    if (auto view = image_views[ViewType::Sample]) {
        device.destroyImageView(view);
    }
    if (auto view = image_views[ViewType::Mip0]; view && view != image_views[ViewType::Sample]) {
        device.destroyImageView(view);
    }
    if (auto view = image_views[ViewType::Storage]) {
        device.destroyImageView(view);
    }
    if (auto view = image_views[ViewType::Depth]) {
        device.destroyImageView(view);
    }
    if (auto view = image_views[ViewType::Stencil]) {
        device.destroyImageView(view);
    }

    image_views = {};

    if (framebuffer) {
        device.destroyFramebuffer(framebuffer);
        framebuffer = VK_NULL_HANDLE;
    }

    if (allocation) {
        vmaDestroyImage(instance.GetAllocator(), image, allocation);
    }

    image = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
}

TextureRuntime::TextureRuntime(const Instance& instance, Scheduler& scheduler,
                               RenderManager& renderpass_cache, DescriptorUpdateQueue& update_queue,
                               u32 num_swapchain_images_)
    : instance{instance}, scheduler{scheduler}, renderpass_cache{renderpass_cache},
      blit_helper{instance, scheduler, renderpass_cache, update_queue},
      upload_buffer{instance, scheduler, vk::BufferUsageFlagBits::eTransferSrc, UPLOAD_BUFFER_SIZE,
                    BufferType::Upload},
      download_buffer{instance, scheduler,
                      vk::BufferUsageFlagBits::eTransferDst |
                          vk::BufferUsageFlagBits::eStorageBuffer,
                      DOWNLOAD_BUFFER_SIZE, BufferType::Download},
      num_swapchain_images{num_swapchain_images_} {}

TextureRuntime::~TextureRuntime() = default;

VideoCore::StagingData TextureRuntime::FindStaging(u32 size, bool upload) {
    StreamBuffer& buffer = upload ? upload_buffer : download_buffer;
    const auto [data, offset, invalidate] = buffer.Map(size, 16);
    return VideoCore::StagingData{
        .size = size,
        .offset = offset,
        .mapped = std::span{data, size},
    };
}

u64 TextureRuntime::GetResourceTick() {
    return scheduler.GetMasterSemaphore()->CurrentTick();
}

u64 TextureRuntime::GetResourceFreeTick() {
    // Ensure we are getting the latest GpuTick value to reduce garbage-collection latency and
    // allows the deletion of resources immediately when they are done being used.
    scheduler.GetMasterSemaphore()->Refresh();
    return scheduler.GetMasterSemaphore()->KnownGpuTick();
}

void TextureRuntime::Finish() {
    scheduler.Flush();
}

void TextureRuntime::Retire(Handle&& handle) {
    retired.emplace_back(std::move(handle), GetResourceTick());
}

void TextureRuntime::FreeRetired() {
    if (retired.empty()) {
        return;
    }
    // As for the surfaces the cache has sentenced: one used in a tick the GPU has not finished
    // stays. Destroyed before it is dropped from the list, because moving a Handle onto
    // another does not release what the other holds.
    const u64 free_tick = GetResourceFreeTick();
    for (auto& [handle, last_used_tick] : retired) {
        if (free_tick > last_used_tick) {
            handle.Destroy();
        }
    }
    std::erase_if(retired, [free_tick](const auto& entry) { return free_tick > entry.second; });
}

void TextureRuntime::NoteStorageUse(PAddr addr, u32 width, u32 height) {
    // A game has a handful of these at a time. What is noted and no longer true costs one
    // surface its framebuffer compression until it leaves the list.
    static constexpr std::size_t MAX_STORAGE_USES = 64;
    const StorageUse use{addr, width, height};
    if (std::ranges::find(storage_uses, use) != storage_uses.end()) {
        return;
    }
    if (storage_uses.size() == MAX_STORAGE_USES) {
        storage_uses.erase(storage_uses.begin());
    }
    storage_uses.push_back(use);
}

bool TextureRuntime::HadStorageUse(PAddr addr, u32 width, u32 height) const {
    return std::ranges::find(storage_uses, StorageUse{addr, width, height}) != storage_uses.end();
}

void TextureRuntime::NoteStencilUse(PAddr addr, u32 width, u32 height) {
    static constexpr std::size_t MAX_STENCIL_USES = 64;
    const StorageUse use{addr, width, height};
    if (std::ranges::find(stencil_uses, use) != stencil_uses.end()) {
        return;
    }
    if (stencil_uses.size() == MAX_STENCIL_USES) {
        stencil_uses.erase(stencil_uses.begin());
    }
    stencil_uses.push_back(use);
}

bool TextureRuntime::HadStencilUse(PAddr addr, u32 width, u32 height) const {
    return std::ranges::find(stencil_uses, StorageUse{addr, width, height}) != stencil_uses.end();
}

bool TextureRuntime::Reinterpret(Surface& source, Surface& dest,
                                 const VideoCore::TextureCopy& copy) {
    const PixelFormat src_format = source.pixel_format;
    const PixelFormat dst_format = dest.pixel_format;
    ASSERT_MSG(src_format != dst_format, "Reinterpretation with the same format is invalid");
    dest.NoteAllWritten();

    if (!source.traits.needs_conversion && !dest.traits.needs_conversion &&
        source.type == dest.type) {
        CopyTextures(source, dest, copy);
        return true;
    }

    if (src_format == PixelFormat::D24S8 && dst_format == PixelFormat::RGBA8) {
        // The conversion writes the destination as a storage image.
        dest.MakeStorageCapable();
        if (!dest.HasStorageUsage()) {
            return false;
        }
        return blit_helper.ConvertDS24S8ToRGBA8(source, dest, copy);
    }

    LOG_WARNING(Render_Vulkan, "Unimplemented reinterpretation {}({}) -> {}({})",
                VideoCore::PixelFormatAsString(src_format), vk::to_string(source.traits.native),
                VideoCore::PixelFormatAsString(dst_format), vk::to_string(dest.traits.native));
    return false;
}

bool TextureRuntime::ClearTexture(Surface& surface, const VideoCore::TextureClear& clear) {
    if (HasPendingBlits() && clear.texture_level == 0 && clear.texture_layer == 0 &&
        clear.texture_rect == surface.GetScaledRect()) {
        DropBlitInto(surface.SerialIfAny());
    }
    if (clear.texture_level == 0 && clear.texture_layer == 0) {
        surface.NoteWritten(clear.texture_rect);
    } else {
        surface.NoteAllWritten();
    }
    Common::Hitches::NotePixelsMoved(Common::Hitches::PixelMove::Clear,
                                     u64{clear.texture_rect.GetWidth()} *
                                         clear.texture_rect.GetHeight());
    // A clear that comes up while a draw gets its target ready is held back: the render
    // pass of the draw can do it as it begins
    const bool defer = renderpass_cache.DefersClears() && surface.GetSampleCount() == 1 &&
                       clear.texture_level == 0 && clear.texture_layer == 0;
    if (!defer) {
        renderpass_cache.EndRendering();
    }
    const auto carry_out_or_defer = [&](std::function<void()>&& carry_out) {
        if (!defer) {
            carry_out();
            return;
        }
        renderpass_cache.DeferClear(DeferredClear{
            .image = surface.Image(),
            .area{
                .offset{
                    .x = static_cast<s32>(clear.texture_rect.left),
                    .y = static_cast<s32>(clear.texture_rect.bottom),
                },
                .extent{
                    .width = clear.texture_rect.GetWidth(),
                    .height = clear.texture_rect.GetHeight(),
                },
            },
            .value = MakeClearValue(clear.value),
            .carry_out = std::move(carry_out),
        });
    };

    const RecordParams params = {
        .aspect = surface.Aspect(),
        .pipeline_flags = surface.PipelineStageFlags(),
        .src_access = surface.AccessFlags(),
        .src_image = surface.Image(),
    };

    // MSAA images should always use a render-pass to clear both the MSAA texture and the regular
    // texture at the same time

    if (clear.texture_rect == surface.GetScaledRect() && (surface.GetSampleCount() == 1)) {
        carry_out_or_defer([this, params, clear] {
            scheduler.Record([params, clear](vk::CommandBuffer cmdbuf) {
                const vk::ImageSubresourceRange range = {
                    .aspectMask = params.aspect,
                    .baseMipLevel = clear.texture_level,
                    .levelCount = 1,
                    .baseArrayLayer = clear.texture_layer,
                    .layerCount = 1,
                };

                const vk::ImageMemoryBarrier pre_barrier = {
                    .srcAccessMask = params.src_access,
                    .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eTransferDstOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = params.src_image,
                    .subresourceRange = range,
                };

                const vk::ImageMemoryBarrier post_barrier = {
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = params.src_access,
                    .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = params.src_image,
                    .subresourceRange = range,
                };

                cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                                       vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

                const bool is_color =
                    static_cast<bool>(params.aspect & vk::ImageAspectFlagBits::eColor);
                if (is_color) {
                    cmdbuf.clearColorImage(params.src_image, vk::ImageLayout::eTransferDstOptimal,
                                           MakeClearColorValue(clear.value.color), range);
                } else {
                    cmdbuf.clearDepthStencilImage(params.src_image,
                                                  vk::ImageLayout::eTransferDstOptimal,
                                                  MakeClearDepthStencilValue(clear.value), range);
                }

                cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                                       vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
            });
        });
        return true;
    }

    carry_out_or_defer(ClearWithRenderpass(surface, clear));
    return true;
}

std::function<void()> TextureRuntime::ClearWithRenderpass(Surface& surface,
                                                          const VideoCore::TextureClear& clear) {
    const bool is_color = surface.type != VideoCore::SurfaceType::Depth &&
                          surface.type != VideoCore::SurfaceType::DepthStencil;

    const auto color_format = is_color ? surface.pixel_format : PixelFormat::Invalid;
    const auto depth_format = is_color ? PixelFormat::Invalid : surface.pixel_format;
    const auto render_pass =
        renderpass_cache.GetRenderpass(color_format, depth_format, true, surface.GetSampleCount());

    const RecordParams params = {
        .aspect = surface.Aspect(),
        .pipeline_flags = surface.PipelineStageFlags(),
        .src_access = surface.AccessFlags(),
        .src_image = surface.Image(),
    };

    // Ensure we get the MSAA framebuffer if we are are doing an MSAA texture
    const vk::Framebuffer framebuffer =
        surface.Framebuffer((surface.GetSampleCount() > 1) ? Type::MultiSampled : Type::Current);

    return [this, params, is_color, clear, render_pass, framebuffer] {
        scheduler.Record(
            [params, is_color, clear, render_pass, framebuffer](vk::CommandBuffer cmdbuf) {
                const vk::AccessFlags access_flag =
                    is_color ? vk::AccessFlagBits::eColorAttachmentRead |
                                   vk::AccessFlagBits::eColorAttachmentWrite
                             : vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                   vk::AccessFlagBits::eDepthStencilAttachmentWrite;

                const vk::PipelineStageFlags pipeline_flags =
                    is_color ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                             : vk::PipelineStageFlagBits::eEarlyFragmentTests;

                const vk::ImageMemoryBarrier pre_barrier = {
                    .srcAccessMask = params.src_access,
                    .dstAccessMask = access_flag,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = params.src_image,
                    .subresourceRange = MakeSubresourceRange(params.aspect, clear.texture_level, 1,
                                                             clear.texture_layer),
                };

                const vk::ImageMemoryBarrier post_barrier = {
                    .srcAccessMask = access_flag,
                    .dstAccessMask = params.src_access,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = params.src_image,
                    .subresourceRange = MakeSubresourceRange(params.aspect, clear.texture_level),
                };

                const vk::Rect2D render_area = {
                    .offset{
                        .x = static_cast<s32>(clear.texture_rect.left),
                        .y = static_cast<s32>(clear.texture_rect.bottom),
                    },
                    .extent{
                        .width = clear.texture_rect.GetWidth(),
                        .height = clear.texture_rect.GetHeight(),
                    },
                };

                const auto clear_value = MakeClearValue(clear.value);
                std::array<vk::ClearValue, 2> clear_values = {clear_value, clear_value};

                const vk::RenderPassBeginInfo renderpass_begin_info = {
                    .renderPass = render_pass,
                    .framebuffer = framebuffer,
                    .renderArea = render_area,
                    .clearValueCount = static_cast<u32>(clear_values.size()),
                    .pClearValues = clear_values.data(),
                };

                cmdbuf.pipelineBarrier(params.pipeline_flags, pipeline_flags,
                                       vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

                cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
                cmdbuf.endRenderPass();

                cmdbuf.pipelineBarrier(pipeline_flags, params.pipeline_flags,
                                       vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
            });
    };
}

bool TextureRuntime::CopyTextures(Surface& source, Surface& dest,
                                  std::span<const VideoCore::TextureCopy> copies) {
    if (HasPendingBlits() && &source != &dest) {
        const PendingBlit* into = FindBlitInto(source.SerialIfAny());
        const bool level_zero = std::ranges::all_of(copies, [](const auto& copy) {
            return copy.src_level == 0 && copy.dst_level == 0 && copy.src_layer == 0 &&
                   copy.dst_layer == 0;
        });
        if (into && level_zero && IsOneToOne(into->blit) &&
            into->src_serial != dest.SerialIfAny() && dest.sample_count == 1 &&
            dest.Aspect() == vk::ImageAspectFlagBits::eColor && dest.traits.blit_support) {
            // (a copy of it: the list may change below)
            const PendingBlit from = *into;
            dest.NoteAllWritten();
            renderpass_cache.EndRendering();
            for (const VideoCore::TextureCopy& copy : copies) {
                PendingBlit piece = from;
                piece.dst_serial = 0;
                piece.dst_image = dest.RawImage();
                piece.dst_access = dest.AccessFlags();
                piece.stages = from.stages | dest.PipelineStageFlags();
                const Common::Rectangle<u32> read{
                    copy.src_offset.x, copy.src_offset.y + copy.extent.height,
                    copy.src_offset.x + copy.extent.width, copy.src_offset.y};
                piece.blit = VideoCore::TextureBlit{
                    .src_level = 0,
                    .dst_level = 0,
                    .src_layer = 0,
                    .dst_layer = 0,
                    .src_rect = SourceOf(from.blit, read),
                    .dst_rect = Common::Rectangle<u32>{copy.dst_offset.x,
                                                       copy.dst_offset.y + copy.extent.height,
                                                       copy.dst_offset.x + copy.extent.width,
                                                       copy.dst_offset.y},
                };
                CITRA_FRAME_DUMP("  vk copy of {}x{} reads through a blit that waits",
                                 copy.extent.width, copy.extent.height);
                RecordPendingBlit(piece);
            }
            return true;
        }
    }
    dest.NoteAllWritten();
    for (const VideoCore::TextureCopy& copy : copies) {
        Common::Hitches::NotePixelsMoved(Common::Hitches::PixelMove::Copy,
                                         u64{copy.extent.width} * copy.extent.height);
    }
    renderpass_cache.EndRendering();

    const RecordParams params = {
        .aspect = source.Aspect(),
        .filter = MakeFilter(source.pixel_format),
        .pipeline_flags = source.PipelineStageFlags() | dest.PipelineStageFlags(),
        .src_access = source.AccessFlags(),
        .dst_access = dest.AccessFlags(),
        .src_image = source.Image(),
        .dst_image = dest.Image(),
    };

    boost::container::small_vector<vk::ImageCopy, 2> vk_copies;
    std::ranges::transform(copies, std::back_inserter(vk_copies), [&](const auto& copy) {
        return vk::ImageCopy{
            .srcSubresource{
                .aspectMask = params.aspect,
                .mipLevel = copy.src_level,
                .baseArrayLayer = copy.src_layer,
                .layerCount = 1,
            },
            .srcOffset = {static_cast<s32>(copy.src_offset.x), static_cast<s32>(copy.src_offset.y),
                          0},
            .dstSubresource{
                .aspectMask = params.aspect,
                .mipLevel = copy.dst_level,
                .baseArrayLayer = copy.dst_layer,
                .layerCount = 1,
            },
            .dstOffset = {static_cast<s32>(copy.dst_offset.x), static_cast<s32>(copy.dst_offset.y),
                          0},
            .extent = {copy.extent.width, copy.extent.height, 1},
        };
    });

    scheduler.Record([params, copies = std::move(vk_copies)](vk::CommandBuffer cmdbuf) {
        const bool self_copy = params.src_image == params.dst_image;
        const vk::ImageLayout new_src_layout =
            self_copy ? vk::ImageLayout::eGeneral : vk::ImageLayout::eTransferSrcOptimal;
        const vk::ImageLayout new_dst_layout =
            self_copy ? vk::ImageLayout::eGeneral : vk::ImageLayout::eTransferDstOptimal;

        const std::array pre_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = params.src_access,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = new_src_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, VK_REMAINING_MIP_LEVELS),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = params.dst_access,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = new_dst_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, VK_REMAINING_MIP_LEVELS),
            },
        };
        const std::array post_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eNone,
                .oldLayout = new_src_layout,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, VK_REMAINING_MIP_LEVELS),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = params.dst_access,
                .oldLayout = new_dst_layout,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, VK_REMAINING_MIP_LEVELS),
            },
        };

        cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);

        cmdbuf.copyImage(params.src_image, new_src_layout, params.dst_image, new_dst_layout,
                         copies);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barriers);
    });

    return true;
}

bool TextureRuntime::CanRelayoutTextures(const Surface& source,
                                         const Surface& dest) const noexcept {
    // The conversion writes the destination as a storage image.
    return dest.pixel_format == PixelFormat::RGBA8 &&
           instance.GetTraits(PixelFormat::RGBA8).storage_support &&
           (source.pixel_format == PixelFormat::RGBA8 || source.pixel_format == PixelFormat::RGB8 ||
            source.pixel_format == PixelFormat::D24S8) &&
           source.sample_count == 1 && dest.sample_count == 1 && !source.IsCustom() &&
           !dest.IsCustom();
}

bool TextureRuntime::RelayoutTextures(Surface& source, Surface& dest,
                                      const VideoCore::TextureRelayout& relayout) {
    // The conversion writes the destination as a storage image.
    dest.MakeStorageCapable();
    dest.NoteAllWritten();
    Common::Hitches::NotePixelsMoved(Common::Hitches::PixelMove::Relayout,
                                     u64{relayout.dst_rect.GetWidth()} *
                                         relayout.dst_rect.GetHeight());
    return blit_helper.Relayout(source, dest, relayout);
}

namespace {

/// Records a blit between two images that are in the general layout
void RecordBlit(Scheduler& scheduler, const RecordParams& params,
                const VideoCore::TextureBlit& blit) {
    scheduler.Record([params, blit](vk::CommandBuffer cmdbuf) {
        const std::array source_offsets = {
            vk::Offset3D{static_cast<s32>(blit.src_rect.left),
                         static_cast<s32>(blit.src_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.src_rect.right), static_cast<s32>(blit.src_rect.top),
                         1},
        };

        const std::array dest_offsets = {
            vk::Offset3D{static_cast<s32>(blit.dst_rect.left),
                         static_cast<s32>(blit.dst_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.dst_rect.right), static_cast<s32>(blit.dst_rect.top),
                         1},
        };

        const vk::ImageBlit blit_area = {
            .srcSubresource{
                .aspectMask = params.aspect,
                .mipLevel = blit.src_level,
                .baseArrayLayer = blit.src_layer,
                .layerCount = 1,
            },
            .srcOffsets = source_offsets,
            .dstSubresource{
                .aspectMask = params.aspect,
                .mipLevel = blit.dst_level,
                .baseArrayLayer = blit.dst_layer,
                .layerCount = 1,
            },
            .dstOffsets = dest_offsets,
        };

        const std::array read_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = params.src_access,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, blit.src_level),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = params.dst_access,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, blit.dst_level),
            },
        };
        const std::array write_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                .dstAccessMask = params.src_access,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, blit.src_level),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = params.dst_access,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, blit.dst_level),
            },
        };

        cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, read_barriers);

        cmdbuf.blitImage(params.src_image, vk::ImageLayout::eTransferSrcOptimal, params.dst_image,
                         vk::ImageLayout::eTransferDstOptimal, blit_area, params.filter);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                               vk::DependencyFlagBits::eByRegion, {}, {}, write_barriers);
    });
}

/// Whether a pixel of the blit's destination is a pixel of its source, turned over or not
bool IsOneToOne(const VideoCore::TextureBlit& blit) {
    const auto span = [](u32 a, u32 b) { return a > b ? a - b : b - a; };
    return blit.src_level == 0 && blit.dst_level == 0 && blit.src_layer == 0 &&
           blit.dst_layer == 0 &&
           span(blit.src_rect.left, blit.src_rect.right) ==
               span(blit.dst_rect.left, blit.dst_rect.right) &&
           span(blit.src_rect.top, blit.src_rect.bottom) ==
               span(blit.dst_rect.top, blit.dst_rect.bottom);
}

/// Where in the source of a one-to-one blit a rectangle of its destination comes from. Left
/// and right, or top and bottom, come out the other way round where the blit turns over.
Common::Rectangle<u32> SourceOf(const VideoCore::TextureBlit& blit,
                                const Common::Rectangle<u32>& rect) {
    const auto map = [](u32 value, u32 dst_from, u32 dst_to, u32 src_from, u32 src_to) {
        const s64 direction = (dst_to >= dst_from) == (src_to >= src_from) ? 1 : -1;
        return static_cast<u32>(s64{src_from} + direction * (s64{value} - s64{dst_from}));
    };
    const auto& dst = blit.dst_rect;
    const auto& src = blit.src_rect;
    return Common::Rectangle<u32>{
        map(rect.left, dst.left, dst.right, src.left, src.right),
        map(rect.top, dst.bottom, dst.top, src.bottom, src.top),
        map(rect.right, dst.left, dst.right, src.left, src.right),
        map(rect.bottom, dst.bottom, dst.top, src.bottom, src.top),
    };
}

u64 PixelsOf(const VideoCore::TextureBlit& blit) {
    return u64{blit.dst_rect.GetWidth()} * blit.dst_rect.GetHeight();
}

} // Anonymous namespace

const TextureRuntime::PendingBlit* TextureRuntime::FindBlitInto(u64 serial) const {
    if (serial == 0) {
        return nullptr;
    }
    for (const PendingBlit& pending : pending_blits) {
        if (pending.dst_serial == serial) {
            return &pending;
        }
    }
    return nullptr;
}

void TextureRuntime::RecordPendingBlit(const PendingBlit& pending) {
    const RecordParams params = {
        .aspect = pending.aspect,
        .filter = vk::Filter::eNearest,
        .pipeline_flags = pending.stages,
        .src_access = pending.src_access,
        .dst_access = pending.dst_access,
        .src_image = pending.src_image,
        .dst_image = pending.dst_image,
    };
    RecordBlit(scheduler, params, pending.blit);
    Common::Hitches::NotePixelsMoved(Common::Hitches::PixelMove::Blit, PixelsOf(pending.blit));
    NoteLateUse(pending.src_serial);
}

void TextureRuntime::NoteLateUse(u64 serial) {
    const u64 tick = GetResourceTick();
    for (auto& [surface, last] : late_uses) {
        if (surface == serial) {
            last = tick;
            return;
        }
    }
    if (late_uses.size() >= 256) {
        const u64 free_tick = GetResourceFreeTick();
        std::erase_if(late_uses, [free_tick](const auto& use) { return free_tick > use.second; });
    }
    late_uses.emplace_back(serial, tick);
}

void TextureRuntime::CarryOutBlitsInto(u64 serial) {
    for (std::size_t i = 0; i < pending_blits.size(); i++) {
        if (pending_blits[i].dst_serial != serial) {
            continue;
        }
        // (taken off the list first: what follows may ask about it again)
        const PendingBlit pending = pending_blits[i];
        pending_blits.erase(pending_blits.begin() + i);
        CITRA_FRAME_DUMP("  vk the blit that waited is carried out: its destination is used");
        renderpass_cache.EndRendering();
        RecordPendingBlit(pending);
        return;
    }
}

void TextureRuntime::CarryOutBlitsFrom(u64 serial) {
    for (std::size_t i = 0; i < pending_blits.size();) {
        if (pending_blits[i].src_serial != serial) {
            i++;
            continue;
        }
        const PendingBlit pending = pending_blits[i];
        pending_blits.erase(pending_blits.begin() + i);
        CITRA_FRAME_DUMP("  vk the blit that waited is carried out: its source changes");
        renderpass_cache.EndRendering();
        RecordPendingBlit(pending);
    }
}

void TextureRuntime::DropBlitInto(u64 serial) {
    if (serial == 0) {
        return;
    }
    for (std::size_t i = 0; i < pending_blits.size(); i++) {
        if (pending_blits[i].dst_serial == serial) {
            CITRA_FRAME_DUMP("  vk the blit that waited is given up: nothing read what it "
                             "would have made");
            Common::Hitches::NotePixelsMoved(Common::Hitches::PixelMove::Spared,
                                             PixelsOf(pending_blits[i].blit));
            pending_blits.erase(pending_blits.begin() + i);
            return;
        }
    }
}

void TextureRuntime::NoteSurfaceRemoved(Surface& surface) {
    if (HasPendingBlits()) {
        DropBlitInto(surface.SerialIfAny());
    }
}

bool TextureRuntime::MayTakeOverNow(const Surface& surface) const {
    if ((Common::TraceMarkers::Experiments4() & Common::TraceMarkers::LegacySurfaceReuse) != 0 ||
        surface.type == VideoCore::SurfaceType::Texture || surface.IsCustom()) {
        return false;
    }
    const u64 serial = surface.SerialIfAny();
    return serial == 0 ||
           std::ranges::none_of(pending_blits, [serial](const PendingBlit& pending) {
               return pending.src_serial == serial;
           });
}

bool TextureRuntime::StillNeeded(Surface& surface) {
    const u64 serial = surface.SerialIfAny();
    if (serial == 0) {
        return false;
    }
    for (const PendingBlit& pending : pending_blits) {
        if (pending.src_serial == serial) {
            return true;
        }
    }
    const u64 free_tick = GetResourceFreeTick();
    for (std::size_t i = 0; i < late_uses.size(); i++) {
        if (late_uses[i].first != serial) {
            continue;
        }
        if (free_tick <= late_uses[i].second) {
            return true;
        }
        late_uses.erase(late_uses.begin() + i);
        return false;
    }
    return false;
}

bool TextureRuntime::BlitTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureBlit& blit) {
    const bool is_depth_stencil = source.type == VideoCore::SurfaceType::DepthStencil;
    const bool whole = blit.dst_level == 0 && blit.dst_layer == 0 &&
                       std::min(blit.dst_rect.left, blit.dst_rect.right) == 0 &&
                       std::max(blit.dst_rect.left, blit.dst_rect.right) == dest.GetScaledWidth() &&
                       std::min(blit.dst_rect.top, blit.dst_rect.bottom) == 0 &&
                       std::max(blit.dst_rect.top, blit.dst_rect.bottom) == dest.GetScaledHeight();
    if (whole && HasPendingBlits()) {
        // What waited to be put there will not be seen
        DropBlitInto(dest.SerialIfAny());
    }
    dest.NoteAllWritten();

    const bool plain =
        !is_depth_stencil && &source != &dest && source.sample_count == 1 &&
        dest.sample_count == 1 && blit.src_level == 0 && blit.dst_level == 0 &&
        blit.src_layer == 0 && blit.dst_layer == 0 &&
        source.Aspect() == vk::ImageAspectFlagBits::eColor &&
        dest.Aspect() == vk::ImageAspectFlagBits::eColor &&
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::LegacyTransfers) == 0;
    if (plain && (whole || HasPendingBlits())) {
        PendingBlit entry{
            .dst_serial = dest.Serial(),
            .src_serial = source.Serial(),
            .src_image = {},
            .dst_image = dest.RawImage(),
            .aspect = source.Aspect(),
            .stages = source.PipelineStageFlags() | dest.PipelineStageFlags(),
            .src_access = source.AccessFlags(),
            .dst_access = dest.AccessFlags(),
            .blit = blit,
        };
        // The source waits for a blit itself: this one reads where that one reads
        bool through = false;
        if (const PendingBlit* into = FindBlitInto(source.SerialIfAny())) {
            if (IsOneToOne(into->blit) && into->src_serial != entry.dst_serial) {
                entry.src_serial = into->src_serial;
                entry.src_image = into->src_image;
                entry.src_access = into->src_access;
                entry.stages = into->stages | dest.PipelineStageFlags();
                entry.blit.src_rect = SourceOf(into->blit, blit.src_rect);
                through = true;
            } else {
                CarryOutBlitsInto(source.SerialIfAny());
            }
        }
        if (!through) {
            entry.src_image = source.RawImage();
        }
        if (whole) {
            CITRA_FRAME_DUMP("  vk blit of {}x{} waits until what it makes is used{}",
                             blit.dst_rect.GetWidth(), blit.dst_rect.GetHeight(),
                             through ? " (it reads through another that waits)" : "");
            pending_blits.push_back(entry);
            return true;
        }
        if (through) {
            CITRA_FRAME_DUMP("  vk blit reads through one that waits");
            renderpass_cache.EndRendering();
            RecordPendingBlit(entry);
            return true;
        }
    }
    Common::Hitches::NotePixelsMoved(Common::Hitches::PixelMove::Blit, PixelsOf(blit));
    const auto& depth_traits = instance.GetTraits(source.pixel_format);
    if (is_depth_stencil && !depth_traits.blit_support) {
        return blit_helper.BlitDepthStencil(source, dest, blit);
    }

    renderpass_cache.EndRendering();

    // Must resolve images first
    // Todo(wunk): Add a "dirty" flag for msaa resolves to avoid redundant image resolves
    if (source.sample_count > 1) {
        blit_helper.ResolveTexture(source);
    }
    if (dest.sample_count > 1) {
        blit_helper.ResolveTexture(dest);
    }

    const RecordParams params = {
        .aspect = source.Aspect(),
        .filter = vk::Filter::eNearest,
        .pipeline_flags = source.PipelineStageFlags() | dest.PipelineStageFlags(),
        .src_access = source.AccessFlags(),
        .dst_access = dest.AccessFlags(),
        .src_image = source.Image(),
        .dst_image = dest.Image(),
    };

    RecordBlit(scheduler, params, blit);

    return true;
}

void TextureRuntime::GenerateMipmaps(Surface& surface) {
    surface.NoteAllWritten();
    if (VideoCore::IsCustomFormatCompressed(surface.custom_format)) {
        LOG_ERROR(Render_Vulkan, "Generating mipmaps for compressed formats unsupported!");
        return;
    }

    renderpass_cache.EndRendering();

    auto [width, height] = surface.RealExtent();
    const u32 levels = surface.levels;
    for (u32 i = 1; i < levels; i++) {
        const Common::Rectangle<u32> src_rect{0, height, width, 0};
        width = width > 1 ? width >> 1 : 1;
        height = height > 1 ? height >> 1 : 1;
        const Common::Rectangle<u32> dst_rect{0, height, width, 0};

        const VideoCore::TextureBlit blit = {
            .src_level = i - 1,
            .dst_level = i,
            .src_rect = src_rect,
            .dst_rect = dst_rect,
        };
        BlitTextures(surface, surface, blit);
    }
}

bool TextureRuntime::NeedsConversion(const Surface& surface) const {
    const FormatTraits& traits = surface.traits;
    return traits.needs_conversion &&
           // DepthStencil formats are handled elsewhere due to de-interleaving.
           traits.aspect != (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);
}

Surface::Surface(TextureRuntime& runtime_, const VideoCore::SurfaceParams& params,
                 const VideoCore::SurfaceFlagBits& initial_flag_bits)
    : SurfaceBase{params, initial_flag_bits}, runtime{runtime_}, instance{runtime_.GetInstance()},
      scheduler{runtime_.GetScheduler()}, traits{instance.GetTraits(pixel_format)},
      handles{Handle(instance), Handle(instance), Handle(instance), Handle(instance),
              Handle(instance)} {
    if (pixel_format == VideoCore::PixelFormat::Invalid || !traits.transfer_support) {
        return;
    }

    if (True(flags & VideoCore::SurfaceFlagBits::ShadowSource) &&
        traits.native != vk::Format::eR8G8B8A8Unorm) {
        // If the surface is a shadow source, it needs conversion
        // to be forced as it always has to be RGBA8
        traits = instance.GetTraits(VideoCore::PixelFormat::RGBA8);
        traits.needs_conversion = true;
    }

    // A shadow map is read through an R32 view and written as a storage image. An image
    // created for that gets no framebuffer compression on some GPUs (an Adreno 6xx leaves it
    // off for an image with storage usage, and for one with a view format of another kind),
    // and compression is worth a third of the frame where memory traffic is the limit. So only
    // a surface known to be a shadow source starts out that way, and one of guest pixels that
    // needed it before; any other gets such images when it turns out to need them, see
    // MakeStorageCapable.
    const bool is_mutable =
        traits.native == vk::Format::eR8G8B8A8Unorm &&
        (True(flags & VideoCore::SurfaceFlagBits::ShadowSource) ||
         runtime.HadStorageUse(addr, width, height) ||
         (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyStorageUsage) != 0);

    const vk::Format format = traits.native;

    ASSERT_MSG(format != vk::Format::eUndefined && levels >= 1,
               "Image allocation parameters are invalid");

    boost::container::static_vector<vk::Image, 3> raw_images;

    vk::ImageCreateFlags flags{};
    if (texture_type == VideoCore::TextureType::CubeMap) {
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }
    if (is_mutable) {
        flags |= vk::ImageCreateFlagBits::eMutableFormat;
    }

    // Ensure color formats have the color attachment bit set for framebuffers
    auto usage = traits.usage;
    const bool is_color =
        (traits.aspect & vk::ImageAspectFlagBits::eColor) != vk::ImageAspectFlags{};
    if (is_color) {
        usage |= vk::ImageUsageFlagBits::eColorAttachment;
    }
    if (is_mutable && traits.storage_support) {
        usage |= vk::ImageUsageFlagBits::eStorage;
    }

    const bool need_format_list = is_mutable && instance.IsImageFormatListSupported();
    image_usage = usage;
    image_flags = flags;
    image_format_list = need_format_list;
    has_image_params = true;
    // The stencil of a depth and stencil surface is declared as sampled only for guest pixels
    // whose stencil was sampled before, see MakeStencilSampleable
    image_stencil_sampled = !static_cast<bool>(traits.aspect & vk::ImageAspectFlagBits::eStencil) ||
                            !instance.IsSeparateStencilUsageSupported() ||
                            runtime.HadStencilUse(addr, width, height);
    handles[Type::Base].Create(width, height, levels, texture_type, format,
                               vk::SampleCountFlagBits::e1, usage, flags, traits.aspect,
                               need_format_list, DebugName(false), image_stencil_sampled);
    raw_images.emplace_back(handles[Type::Base].image);

    // Upscaled image
    if (!res_scale.IsNative()) {
        handles[Type::Scaled].Create(GetScaledWidth(), GetScaledHeight(), levels, texture_type,
                                     format, vk::SampleCountFlagBits::e1, usage, flags,
                                     traits.aspect, need_format_list, DebugName(true),
                                     image_stencil_sampled);
        raw_images.emplace_back(handles[Type::Scaled].image);
    }

    // Upscaled+MSAA image
    if (sample_count > 1) {
        handles[Type::MultiSampled].Create(
            GetScaledWidth(), GetScaledHeight(), levels, texture_type, format,
            vk::SampleCountFlagBits(sample_count), traits.usage, flags, traits.aspect,
            need_format_list, DebugName(true, false, sample_count));
        raw_images.emplace_back(handles[Type::MultiSampled].image);
    }

    current = !res_scale.IsNative() ? Type::Scaled : Type::Base;

    runtime.renderpass_cache.EndRendering();
    scheduler.Record([raw_images, aspect = traits.aspect,
                      write = WritesNewImages(format)](vk::CommandBuffer cmdbuf) {
        InitImages(cmdbuf, aspect, raw_images,
                   write ? std::span<const vk::Image>{raw_images} : std::span<const vk::Image>{});
    });
}

Surface::Surface(TextureRuntime& runtime_, const VideoCore::SurfaceBase& surface,
                 const VideoCore::Material* mat)
    : SurfaceBase{surface}, runtime{runtime_}, instance{runtime_.GetInstance()},
      scheduler{runtime_.GetScheduler()}, traits{instance.GetTraits(mat->format)},
      handles{Handle(instance), Handle(instance), Handle(instance), Handle(instance),
              Handle(instance)} {
    if (!traits.transfer_support) {
        return;
    }

    const bool has_normal = mat && mat->Map(MapType::Normal);
    const vk::Format format = traits.native;

    boost::container::static_vector<vk::Image, 4> raw_images;

    vk::ImageCreateFlags flags{};
    if (texture_type == VideoCore::TextureType::CubeMap) {
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }

    const std::string debug_name = DebugName(false, true);
    handles[Type::Base].Create(mat->width, mat->height, levels, texture_type, format,
                               vk::SampleCountFlagBits::e1, traits.usage, flags, traits.aspect,
                               false, debug_name);
    raw_images.emplace_back(handles[Type::Base].image);

    if (!res_scale.IsNative()) {
        handles[Type::Scaled].Create(mat->width, mat->height, levels, texture_type,
                                     vk::Format::eR8G8B8A8Unorm, vk::SampleCountFlagBits::e1,
                                     traits.usage, flags, traits.aspect, false, debug_name);
        raw_images.emplace_back(handles[Type::Scaled].image);
    }
    if (sample_count > 1) {
        handles[Type::MultiSampled].Create(
            GetScaledWidth(), GetScaledHeight(), levels, texture_type, format,
            vk::SampleCountFlagBits(sample_count), traits.usage, flags, traits.aspect, false,
            DebugName(!res_scale.IsNative(), true, sample_count));
        raw_images.emplace_back(handles[Type::MultiSampled].image);
    }
    if (has_normal) {
        handles[Type::Custom].Create(mat->width, mat->height, levels, texture_type, format,
                                     vk::SampleCountFlagBits::e1, traits.usage, flags,
                                     traits.aspect, false, debug_name);
        raw_images.emplace_back(handles[Type::Custom].image);
    }

    current = !res_scale.IsNative() ? Type::Scaled : Type::Base;

    // The image the pack's texture is scaled into is always one that can be cleared; the
    // others are in the pack's format
    boost::container::static_vector<vk::Image, 4> written_images;
    if (WritesNewImages(format)) {
        written_images = raw_images;
    } else if (!res_scale.IsNative() && WritesNewImages(vk::Format::eR8G8B8A8Unorm)) {
        written_images.emplace_back(handles[Type::Scaled].image);
    }

    runtime.renderpass_cache.EndRendering();
    scheduler.Record(
        [raw_images, written_images, aspect = traits.aspect](vk::CommandBuffer cmdbuf) {
            InitImages(cmdbuf, aspect, raw_images, written_images);
        });

    custom_format = mat->format;
    material = mat;
}

void Surface::Upload(const VideoCore::BufferTextureCopy& upload,
                     const VideoCore::StagingData& staging) {
    NoteAllWritten();
    runtime.renderpass_cache.EndRendering();

    const RecordParams params = {
        .aspect = Aspect(),
        .pipeline_flags = PipelineStageFlags(),
        .src_access = AccessFlags(),
        .src_image = Image(Type::Base),
    };

    scheduler.Record([buffer = runtime.upload_buffer.Handle(), format = traits.native, params,
                      staging, upload](vk::CommandBuffer cmdbuf) {
        boost::container::static_vector<vk::BufferImageCopy, 2> buffer_image_copies;

        const auto rect = upload.texture_rect;
        buffer_image_copies.emplace_back(vk::BufferImageCopy{
            .bufferOffset = upload.buffer_offset,
            .bufferRowLength = rect.GetWidth(),
            .bufferImageHeight = rect.GetHeight(),
            .imageSubresource{
                .aspectMask = params.aspect,
                .mipLevel = upload.texture_level,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
            .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1},
        });

        if (params.aspect & vk::ImageAspectFlagBits::eStencil) {
            buffer_image_copies[0].imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;

            vk::BufferImageCopy& stencil_copy =
                buffer_image_copies.emplace_back(buffer_image_copies[0]);
            stencil_copy = buffer_image_copies[0];
            stencil_copy.bufferOffset += UnpackDepthStencil(staging, format);
            stencil_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;
        }

        const vk::ImageMemoryBarrier read_barrier = {
            .srcAccessMask = params.src_access,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = params.src_image,
            .subresourceRange = MakeSubresourceRange(params.aspect, upload.texture_level),
        };
        const vk::ImageMemoryBarrier write_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = params.src_access,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = params.src_image,
            .subresourceRange = MakeSubresourceRange(params.aspect, upload.texture_level),
        };

        cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);

        cmdbuf.copyBufferToImage(buffer, params.src_image, vk::ImageLayout::eTransferDstOptimal,
                                 buffer_image_copies);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                               vk::DependencyFlagBits::eByRegion, {}, {}, write_barrier);
    });

    runtime.upload_buffer.Commit(staging.size);

    if (!res_scale.IsNative()) {
        ASSERT_MSG(handles[Type::Scaled], "Scaled allocation missing during upload");

        const VideoCore::TextureBlit blit = {
            .src_level = upload.texture_level,
            .dst_level = upload.texture_level,
            .src_rect = upload.texture_rect,
            .dst_rect = Scaling().Rect(upload.texture_rect),
        };
        // The texture filters assume the same scale in both directions.
        if ((type != SurfaceType::Color && type != SurfaceType::Texture) || half_scale_x ||
            half_scale_y || !runtime.blit_helper.Filter(*this, blit)) {
            BlitScale(blit, true);
        }
    }
}

void Surface::UploadCustom(const VideoCore::Material* material, u32 level) {
    NoteAllWritten();
    const u32 width = material->width;
    const u32 height = material->height;
    const auto color = material->textures[0];
    const Common::Rectangle rect{0U, height, width, 0U};

    const auto upload = [&](Type type, VideoCore::CustomTexture* texture) {
        const u32 custom_size = static_cast<u32>(texture->data.size());
        const RecordParams params = {
            .aspect = vk::ImageAspectFlagBits::eColor,
            .pipeline_flags = PipelineStageFlags(),
            .src_access = AccessFlags(),
            .src_image = Image(type),
        };

        const auto [data, offset, invalidate] = runtime.upload_buffer.Map(custom_size, 0);
        std::memcpy(data, texture->data.data(), custom_size);
        runtime.upload_buffer.Commit(custom_size);

        scheduler.Record([buffer = runtime.upload_buffer.Handle(), level, params, rect,
                          offset = offset](vk::CommandBuffer cmdbuf) {
            const vk::BufferImageCopy buffer_image_copy = {
                .bufferOffset = offset,
                .bufferRowLength = 0,
                .bufferImageHeight = rect.GetHeight(),
                .imageSubresource{
                    .aspectMask = params.aspect,
                    .mipLevel = level,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
                .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1},
            };

            const vk::ImageMemoryBarrier read_barrier = {
                .srcAccessMask = params.src_access,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, level),
            };
            const vk::ImageMemoryBarrier write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = params.src_access,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, level),
            };

            cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);

            cmdbuf.copyBufferToImage(buffer, params.src_image, vk::ImageLayout::eTransferDstOptimal,
                                     buffer_image_copy);

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, write_barrier);
        });
    };

    upload(Type::Base, color);
    if (auto* texture = material->textures[u32(MapType::Normal)]) {
        upload(Type::Custom, texture);
    }
}

void Surface::Download(const VideoCore::BufferTextureCopy& download,
                       const VideoCore::StagingData& staging, bool wait) {
    SCOPE_EXIT({
        if (wait) {
            scheduler.Finish();
        }
        runtime.download_buffer.Commit(staging.size);
    });

    runtime.renderpass_cache.EndRendering();

    if (pixel_format == PixelFormat::D24S8) {
        runtime.blit_helper.DepthToBuffer(*this, runtime.download_buffer.Handle(), download);
        return;
    }

    if (!res_scale.IsNative()) {
        const VideoCore::TextureBlit blit = {
            .src_level = download.texture_level,
            .dst_level = download.texture_level,
            .src_rect = Scaling().Rect(download.texture_rect),
            .dst_rect = download.texture_rect,
        };

        BlitScale(blit, false);
    }

    const RecordParams params = {
        .aspect = Aspect(),
        .pipeline_flags = PipelineStageFlags(),
        .src_access = AccessFlags(),
        .src_image = Image(Type::Base),
    };

    scheduler.Record(
        [buffer = runtime.download_buffer.Handle(), params, download](vk::CommandBuffer cmdbuf) {
            const auto rect = download.texture_rect;
            const vk::BufferImageCopy buffer_image_copy = {
                .bufferOffset = download.buffer_offset,
                .bufferRowLength = rect.GetWidth(),
                .bufferImageHeight = rect.GetHeight(),
                .imageSubresource{
                    .aspectMask = params.aspect,
                    .mipLevel = download.texture_level,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
                .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1},
            };

            const vk::ImageMemoryBarrier read_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, download.texture_level),
            };
            const vk::ImageMemoryBarrier image_write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, download.texture_level),
            };
            const vk::MemoryBarrier memory_write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            };

            cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);

            cmdbuf.copyImageToBuffer(params.src_image, vk::ImageLayout::eTransferSrcOptimal, buffer,
                                     buffer_image_copy);

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                                   vk::DependencyFlagBits::eByRegion, memory_write_barrier, {},
                                   image_write_barrier);
        });
}

void Surface::ScaleUp(Common::ResolutionScale new_scale, u8 new_sample_count) {
    NoteAllWritten();
    // Of the kind the surface's images are, see the constructor
    const bool is_mutable =
        has_image_params &&
        static_cast<bool>(image_flags & vk::ImageCreateFlagBits::eMutableFormat);
    const vk::ImageUsageFlags scaled_usage = has_image_params ? image_usage : traits.usage;
    const bool format_list = has_image_params && image_format_list;

    vk::ImageCreateFlags flags{};
    if (texture_type == VideoCore::TextureType::CubeMap) {
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }
    if (is_mutable) {
        flags |= vk::ImageCreateFlagBits::eMutableFormat;
    }

    const bool res_scale_modified = res_scale != new_scale;
    if (res_scale_modified && !new_scale.IsNative()) {
        res_scale = new_scale;

        // The surface moves to another image. What was recorded so far can be for the one it
        // was in, and for the second one that goes with it.
        RetireForNewImages(Type::Scaled);
        RetireForNewImages(Type::Copy);
        handles[Type::Scaled].Create(GetScaledWidth(), GetScaledHeight(), levels, texture_type,
                                     traits.native, vk::SampleCountFlagBits::e1, scaled_usage,
                                     flags, traits.aspect, format_list, DebugName(true),
                                     image_stencil_sampled);
        current = Type::Scaled;
        image_generation++;

        runtime.renderpass_cache.EndRendering();
        scheduler.Record([raw_images = std::array{Image()}, aspect = traits.aspect,
                          write = WritesNewImages(traits.native)](vk::CommandBuffer cmdbuf) {
            InitImages(cmdbuf, aspect, raw_images,
                       write ? std::span<const vk::Image>{raw_images}
                             : std::span<const vk::Image>{});
        });

        for (u32 level = 0; level < levels; level++) {
            const VideoCore::TextureBlit blit = {
                .src_level = level,
                .dst_level = level,
                .src_rect = GetRect(level),
                .dst_rect = GetScaledRect(level),
            };
            BlitScale(blit, true);
        }
    }

    if ((res_scale_modified || sample_count != new_sample_count) && new_sample_count > 1) {
        sample_count = new_sample_count;
        RetireForNewImages(Type::MultiSampled);
        image_generation++;
        handles[Type::MultiSampled].Create(
            GetScaledWidth(), GetScaledHeight(), levels, texture_type, traits.native,
            vk::SampleCountFlagBits(sample_count), traits.usage, flags, traits.aspect, false,
            DebugName(!res_scale.IsNative(), false, sample_count));

        // The multi-sampled image is just a transient image that is almost always immediately
        // resolved into the current image, and should not be representative of the entire surface!
        //
        // current = Type::MultiSampled;

        runtime.renderpass_cache.EndRendering();
        scheduler.Record([raw_images = std::array{Image(Type::MultiSampled)},
                          aspect = traits.aspect,
                          write = WritesNewImages(traits.native)](vk::CommandBuffer cmdbuf) {
            InitImages(cmdbuf, aspect, raw_images,
                       write ? std::span<const vk::Image>{raw_images}
                             : std::span<const vk::Image>{});
        });
    }
}

u32 Surface::GetInternalBytesPerPixel() const {
    // Request 5 bytes for D24S8 as well because we can use the
    // extra space when deinterleaving the data during upload
    if (traits.native == vk::Format::eD24UnormS8Uint) {
        return 5;
    }

    return vk::blockSize(traits.native);
}

vk::AccessFlags Surface::AccessFlags() const noexcept {
    const bool is_color = static_cast<bool>(Aspect() & vk::ImageAspectFlagBits::eColor);
    const vk::AccessFlags attachment_flags =
        is_color
            ? vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite
            : vk::AccessFlagBits::eDepthStencilAttachmentRead |
                  vk::AccessFlagBits::eDepthStencilAttachmentWrite;

    return vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead |
           vk::AccessFlagBits::eTransferWrite |
           (is_framebuffer ? attachment_flags : vk::AccessFlagBits::eNone) |
           (is_storage ? vk::AccessFlagBits::eShaderWrite : vk::AccessFlagBits::eNone);
}

vk::PipelineStageFlags Surface::PipelineStageFlags() const noexcept {
    const bool is_color = static_cast<bool>(Aspect() & vk::ImageAspectFlagBits::eColor);
    const vk::PipelineStageFlags attachment_flags =
        is_color ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                 : vk::PipelineStageFlagBits::eEarlyFragmentTests |
                       vk::PipelineStageFlagBits::eLateFragmentTests;

    return vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eFragmentShader |
           (is_framebuffer ? attachment_flags : vk::PipelineStageFlagBits::eNone) |
           (is_storage ? vk::PipelineStageFlagBits::eComputeShader
                       : vk::PipelineStageFlagBits::eNone);
}

bool Surface::MakeStorageCapable() {
    if (HasStorageUsage() || !has_image_params || !traits.storage_support ||
        traits.native != vk::Format::eR8G8B8A8Unorm) {
        return false;
    }
    runtime.renderpass_cache.EndRendering();
    NoteAllWritten();

    image_usage |= vk::ImageUsageFlagBits::eStorage;
    image_flags |= vk::ImageCreateFlagBits::eMutableFormat;
    image_format_list = instance.IsImageFormatListSupported();

    RemakeImages();
    runtime.NoteStorageUse(addr, width, height);
    LOG_INFO(Render_Vulkan, "Surface {}x{} @{:08X} was given images a shadow map can use",
             GetScaledWidth(), GetScaledHeight(), addr);
    return true;
}

bool Surface::MakeStencilSampleable() {
    if (image_stencil_sampled || !has_image_params) {
        return false;
    }
    runtime.renderpass_cache.EndRendering();
    NoteAllWritten();

    image_stencil_sampled = true;

    RemakeImages();
    runtime.NoteStencilUse(addr, width, height);
    LOG_INFO(Render_Vulkan, "Surface {}x{} @{:08X} was given images whose stencil can be sampled",
             GetScaledWidth(), GetScaledHeight(), addr);
    return true;
}

void Surface::RetireForNewImages(Type type) {
    // The framebuffer of the multisampled image is for the one it resolves into as well
    Handle& multisampled = handles[Type::MultiSampled];
    if (multisampled.framebuffer) {
        Handle old{instance};
        old.framebuffer = std::exchange(multisampled.framebuffer, VK_NULL_HANDLE);
        runtime.Retire(std::move(old));
    }
    if (handles[type]) {
        runtime.Retire(std::move(handles[type]));
    }
}

void Surface::RemakeImages() {
    // The second image holds nothing that lasts: it is made again when it is needed
    RetireForNewImages(Type::Copy);
    for (const Type type : {Type::Base, Type::Scaled}) {
        Handle& handle = handles[type];
        if (!handle) {
            continue;
        }
        Handle fresh{instance};
        fresh.Create(handle.width, handle.height, handle.levels, texture_type, traits.native,
                     vk::SampleCountFlagBits::e1, image_usage, image_flags, traits.aspect,
                     image_format_list, DebugName(type == Type::Scaled), image_stencil_sampled);

        const RecordParams params = {
            .aspect = Aspect(),
            .pipeline_flags = PipelineStageFlags(),
            .src_access = AccessFlags(),
            .src_image = handle.image,
            .dst_image = fresh.image,
        };
        scheduler.Record([params, levels = handle.levels, layers = handle.layers,
                          width = handle.width, height = handle.height](vk::CommandBuffer cmdbuf) {
            const vk::ImageSubresourceRange range = {
                .aspectMask = params.aspect,
                .baseMipLevel = 0,
                .levelCount = levels,
                .baseArrayLayer = 0,
                .layerCount = layers,
            };
            const std::array pre_barriers = {
                vk::ImageMemoryBarrier{
                    .srcAccessMask = params.src_access,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = params.src_image,
                    .subresourceRange = range,
                },
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eNone,
                    .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eTransferDstOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = params.dst_image,
                    .subresourceRange = range,
                },
            };
            const vk::ImageMemoryBarrier post_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = params.src_access,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = range,
            };
            boost::container::small_vector<vk::ImageCopy, 3> image_copies;
            for (u32 level = 0; level < levels; level++) {
                const vk::ImageSubresourceLayers subresource = {
                    .aspectMask = params.aspect,
                    .mipLevel = level,
                    .baseArrayLayer = 0,
                    .layerCount = layers,
                };
                image_copies.push_back(vk::ImageCopy{
                    .srcSubresource = subresource,
                    .srcOffset = {0, 0, 0},
                    .dstSubresource = subresource,
                    .dstOffset = {0, 0, 0},
                    .extent = {std::max(width >> level, 1u), std::max(height >> level, 1u), 1},
                });
            }
            cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);
            cmdbuf.copyImage(params.src_image, vk::ImageLayout::eTransferSrcOptimal,
                             params.dst_image, vk::ImageLayout::eTransferDstOptimal, image_copies);
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
        });

        // Commands recorded so far use the old image, and so may views handed out for it
        RetireForNewImages(type);
        handle = std::move(fresh);
    }
    image_generation++;
}

bool Surface::EnsureCopyHandle() {
    auto& copy_handle = handles[Type::Copy];
    if (copy_handle) {
        return false;
    }
    if (has_image_params) {
        // Of the same kind as the image the surface is in, so that the two can change places
        copy_handle.Create(GetScaledWidth(), GetScaledHeight(), levels, texture_type, traits.native,
                           vk::SampleCountFlagBits::e1, image_usage, image_flags, traits.aspect,
                           image_format_list, {}, image_stencil_sampled);
    } else {
        vk::ImageCreateFlags flags{};
        if (texture_type == VideoCore::TextureType::CubeMap) {
            flags |= vk::ImageCreateFlagBits::eCubeCompatible;
        }
        copy_handle.Create(GetScaledWidth(), GetScaledHeight(), levels, texture_type, traits.native,
                           vk::SampleCountFlagBits::e1, traits.usage, flags, traits.aspect, false);
    }
    // Nothing is in it yet
    copy_differs_all = true;
    if (!WritesNewImages(traits.native)) {
        return true;
    }
    // Written once in full before anything else is done with it, see InitImages: a full-screen
    // effect drawn in place makes this image the colour target of a render pass that neither
    // loads nor clears. It is in the general layout after that.
    runtime.renderpass_cache.EndRendering();
    scheduler.Record([images = std::array{copy_handle.image}, aspect = traits.aspect](
                         vk::CommandBuffer cmdbuf) { InitImages(cmdbuf, aspect, images, images); });
    return false;
}

bool Surface::CanSwapForOverwrite() const noexcept {
    return has_image_params && levels == 1 && sample_count == 1 &&
           texture_type == VideoCore::TextureType::Texture2D &&
           traits.aspect == vk::ImageAspectFlagBits::eColor &&
           current == (!res_scale.IsNative() ? Type::Scaled : Type::Base);
}

vk::ImageView Surface::SwapForOverwrite(const Common::Rectangle<u32>& rect) noexcept {
    if (serial.value != 0 && runtime.HasPendingBlits()) [[unlikely]] {
        // The surface's images change places: the blits that wait hold them by name
        runtime.CarryOutBlitsFrom(serial.value);
        runtime.CarryOutBlitsInto(serial.value);
    }
    const vk::ImageLayout copy_layout =
        EnsureCopyHandle() ? vk::ImageLayout::eUndefined : vk::ImageLayout::eGeneral;
    auto& copy_handle = handles[Type::Copy];

    runtime.renderpass_cache.EndRendering();

    // What is outside the rectangle, in up to four pieces: below, above, left and right
    const u32 width = GetScaledWidth();
    const u32 height = GetScaledHeight();
    const u32 left = std::min(std::min(rect.left, rect.right), width);
    const u32 right = std::min(std::max(rect.left, rect.right), width);
    const u32 bottom = std::min(std::min(rect.bottom, rect.top), height);
    const u32 top = std::min(std::max(rect.bottom, rect.top), height);
    std::array<vk::ImageCopy, 4> regions{};
    u32 num_regions = 0;
    const auto keep = [&regions, &num_regions](u32 x0, u32 y0, u32 x1, u32 y1) {
        if (x0 >= x1 || y0 >= y1) {
            return;
        }
        const vk::ImageSubresourceLayers layers = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        regions[num_regions++] = vk::ImageCopy{
            .srcSubresource = layers,
            .srcOffset = {static_cast<s32>(x0), static_cast<s32>(y0), 0},
            .dstSubresource = layers,
            .dstOffset = {static_cast<s32>(x0), static_cast<s32>(y0), 0},
            .extent = {x1 - x0, y1 - y0, 1},
        };
    };
    // Only where the two images may differ: what was written to the first since the two last
    // changed places (or the second was brought up to date). The rest is the same in both.
    u32 from_x = 0;
    u32 from_y = 0;
    u32 to_x = width;
    u32 to_y = height;
    if (!copy_differs_all) {
        from_x = std::min(differs_x0, width);
        to_x = std::min(differs_x1, width);
        from_y = std::min(differs_y0, height);
        to_y = std::min(differs_y1, height);
    }
    keep(from_x, from_y, to_x, std::min(bottom, to_y));
    keep(from_x, std::max(top, from_y), to_x, to_y);
    keep(from_x, std::max(bottom, from_y), std::min(left, to_x), std::min(top, to_y));
    keep(std::max(right, from_x), std::max(bottom, from_y), to_x, std::min(top, to_y));
#ifdef CITRA_ENABLE_TRACE_MARKERS
    if (VideoCore::FrameStateDump::IsActive()) {
        u64 copied = 0;
        for (u32 i = 0; i < num_regions; i++) {
            copied += u64(regions[i].extent.width) * regions[i].extent.height;
        }
        CITRA_FRAME_DUMP("  vk of the {} pixels outside the rectangle {} are copied",
                         u64(width) * height - u64(right - left) * (top - bottom), copied);
    }
#endif
    {
        u64 moved = 0;
        for (u32 i = 0; i < num_regions; i++) {
            moved += u64{regions[i].extent.width} * regions[i].extent.height;
        }
        Common::Hitches::NotePixelsMoved(Common::Hitches::PixelMove::SecondImage, moved);
    }
    // From here on the two differ within the rectangle at the most: the draw fills it in the
    // one, the other keeps what was there
    copy_differs_all = false;
    differs_x0 = left;
    differs_x1 = right;
    differs_y0 = bottom;
    differs_y1 = top;

    const RecordParams params = {
        .aspect = Aspect(),
        .pipeline_flags = PipelineStageFlags(),
        .src_access = AccessFlags(),
        .src_image = Image(),
        .dst_image = copy_handle.image,
    };
    scheduler.Record([params, copy_layout, regions, num_regions](vk::CommandBuffer cmdbuf) {
        const std::array pre_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, 1),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderRead,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = copy_layout,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, 1),
            },
        };
        // From here on the image the surface was in is sampled and the other one drawn to
        const std::array post_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, 1),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, 1),
            },
        };
        cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);
        if (num_regions != 0) {
            cmdbuf.copyImage(params.src_image, vk::ImageLayout::eTransferSrcOptimal,
                             params.dst_image, vk::ImageLayout::eTransferDstOptimal, num_regions,
                             regions.data());
        }
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barriers);
    });
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "copy of what is outside the rectangle");

    std::swap(handles[current], copy_handle);
    swapped = !swapped;
    return ImageView(ViewType::Sample, Type::Copy);
}

vk::ImageView Surface::CopyImageView() noexcept {
    if (serial.value != 0 && runtime.HasPendingBlits()) [[unlikely]] {
        runtime.CarryOutBlitsFrom(serial.value);
        runtime.CarryOutBlitsInto(serial.value);
    }
    auto& copy_handle = handles[Type::Copy];
    const vk::ImageLayout copy_layout =
        EnsureCopyHandle() ? vk::ImageLayout::eUndefined : vk::ImageLayout::eGeneral;

    // Only where the second image may differ from the first: mostly what one draw wrote
    // since the last time
    const bool partial = !copy_differs_all && levels == 1;
    vk::Offset3D partial_offset{};
    vk::Extent3D partial_extent{};
    if (partial) {
        const u32 x0 = std::min(differs_x0, GetScaledWidth());
        const u32 x1 = std::min(differs_x1, GetScaledWidth());
        const u32 y0 = std::min(differs_y0, GetScaledHeight());
        const u32 y1 = std::min(differs_y1, GetScaledHeight());
        if (x0 >= x1 || y0 >= y1) {
            // Nothing was written since: the copy is as the surface is
            return copy_handle.image_views[ViewType::Sample];
        }
        partial_offset = vk::Offset3D{static_cast<s32>(x0), static_cast<s32>(y0), 0};
        partial_extent = vk::Extent3D{x1 - x0, y1 - y0, 1};
    }
    copy_differs_all = false;
    differs_x0 = 0;
    differs_x1 = 0;
    differs_y0 = 0;
    differs_y1 = 0;
    Common::Hitches::NotePixelsMoved(Common::Hitches::PixelMove::SecondImage,
                                     partial ? u64{partial_extent.width} * partial_extent.height
                                             : u64{GetScaledWidth()} * GetScaledHeight());

    CITRA_FRAME_DUMP("  vk full-surface copy {}x{} levels {} @{:08X} (ends the render pass){}",
                     GetScaledWidth(), GetScaledHeight(), levels, addr,
                     partial
                         ? fmt::format(", of it only {}x{} at {},{}", partial_extent.width,
                                       partial_extent.height, partial_offset.x, partial_offset.y)
                         : std::string{});
    runtime.renderpass_cache.EndRendering();

    const RecordParams params = {
        .aspect = Aspect(),
        .pipeline_flags = PipelineStageFlags(),
        .src_access = AccessFlags(),
        .src_image = Image(),
        .dst_image = copy_handle.image,
    };

    scheduler.Record([params, copy_layout, levels = this->levels, width = GetScaledWidth(),
                      height = GetScaledHeight(), partial, partial_offset,
                      partial_extent](vk::CommandBuffer cmdbuf) {
        std::array pre_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, levels),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderRead,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = copy_layout,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, levels),
            },
        };
        std::array post_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, levels),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, levels),
            },
        };

        boost::container::small_vector<vk::ImageCopy, 3> image_copies;
        for (u32 level = 0; level < levels; level++) {
            image_copies.push_back(vk::ImageCopy{
                .srcSubresource{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = level,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .srcOffset = {0, 0, 0},
                .dstSubresource{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = level,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .dstOffset = {0, 0, 0},
                .extent = {width >> level, height >> level, 1},
            });
        }
        if (partial) {
            image_copies[0].srcOffset = partial_offset;
            image_copies[0].dstOffset = partial_offset;
            image_copies[0].extent = partial_extent;
        }

        cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);

        cmdbuf.copyImage(params.src_image, vk::ImageLayout::eTransferSrcOptimal, params.dst_image,
                         vk::ImageLayout::eTransferDstOptimal, image_copies);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barriers);
    });
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "full-surface copy");

    return copy_handle.image_views[ViewType::Sample];
}

vk::ImageView Surface::ImageView(ViewType view_type, Type type) noexcept {
    if (serial.value != 0 && runtime.HasPendingBlits()) [[unlikely]] {
        runtime.CarryOutBlitsInto(serial.value);
    }
    auto& handle = handles[type == Type::Current ? current : type];
    if (auto image_view = handle.image_views[view_type]) {
        return image_view;
    }

    auto aspect = traits.aspect;

    if (view_type == ViewType::Storage) {
        ASSERT_MSG(traits.storage_support,
                   "Creating a storage-view for format({}) which doesn't have storage-support!",
                   vk::to_string(traits.native));
        // The surface's images change here if they are not of the kind that has such a view
        MakeStorageCapable();
        is_storage = true;
    }
    if (view_type == ViewType::Depth || view_type == ViewType::Stencil) {
        ASSERT(this->type == SurfaceType::DepthStencil);
        // The two are asked for together, by the conversions and copies done in a shader, which
        // sample the stencil: the images of the surface change here if theirs cannot be
        MakeStencilSampleable();
        aspect = view_type == ViewType::Depth ? vk::ImageAspectFlagBits::eDepth
                                              : vk::ImageAspectFlagBits::eStencil;
    }

    const vk::ImageViewCreateInfo view_info = {
        .image = handle.image,
        .viewType = vk::ImageViewType::e2D,
        .format = view_type == ViewType::Storage ? vk::Format::eR32Uint : traits.native,
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = (view_type == ViewType::Mip0 || view_type == ViewType::Storage)
                              ? 1u
                              : VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    handle.image_views[view_type] = instance.GetDevice().createImageView(view_info);
    return handle.image_views[view_type];
}

vk::Framebuffer Surface::Framebuffer(Type type) noexcept {
    if (serial.value != 0 && runtime.HasPendingBlits()) [[unlikely]] {
        runtime.CarryOutBlitsInto(serial.value);
    }
    type = (type == Type::Current) ? current : type;
    auto& handle = handles[type];
    if (handle.framebuffer) {
        return handle.framebuffer;
    }

    const bool is_depth =
        this->type == SurfaceType::Depth || this->type == SurfaceType::DepthStencil;
    const auto color_format = is_depth ? PixelFormat::Invalid : pixel_format;
    const auto depth_format = is_depth ? pixel_format : PixelFormat::Invalid;

    boost::container::small_vector<vk::ImageView, 2> image_views;
    if (sample_count > 1) {
        // Main surface + MSAA surface
        image_views.emplace_back(ImageView(ViewType::Mip0, current));
        image_views.emplace_back(ImageView(ViewType::Mip0, Type::MultiSampled));
    } else {
        image_views.emplace_back(ImageView(ViewType::Mip0, type));
    }

    const vk::FramebufferCreateInfo framebuffer_info = {
        .renderPass =
            runtime.renderpass_cache.GetRenderpass(color_format, depth_format, false, sample_count),
        .attachmentCount = static_cast<u32>(image_views.size()),
        .pAttachments = image_views.data(),
        .width = handle.width,
        .height = handle.height,
        .layers = handle.layers,
    };
    handle.framebuffer = instance.GetDevice().createFramebuffer(framebuffer_info);
    return handle.framebuffer;
}

void Surface::BlitScale(const VideoCore::TextureBlit& blit, bool up_scale) {
    const bool is_depth_stencil = pixel_format == PixelFormat::D24S8;
    if (is_depth_stencil && !traits.blit_support) {
        LOG_WARNING(Render_Vulkan, "Depth scale unsupported by hardware");
        return;
    }

    const auto src_type = up_scale ? Type::Base : Type::Scaled;
    const auto dst_type = up_scale ? Type::Scaled : Type::Base;
    if (dst_type == current) {
        NoteAllWritten();
    }

    scheduler.Record([src_image = Image(src_type), aspect = Aspect(), filter = vk::Filter::eNearest,
                      dst_image = Image(dst_type), blit](vk::CommandBuffer render_cmdbuf) {
        const std::array source_offsets = {
            vk::Offset3D{static_cast<s32>(blit.src_rect.left),
                         static_cast<s32>(blit.src_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.src_rect.right), static_cast<s32>(blit.src_rect.top),
                         1},
        };

        const std::array dest_offsets = {
            vk::Offset3D{static_cast<s32>(blit.dst_rect.left),
                         static_cast<s32>(blit.dst_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.dst_rect.right), static_cast<s32>(blit.dst_rect.top),
                         1},
        };

        const vk::ImageBlit blit_area = {
            .srcSubresource{
                .aspectMask = aspect,
                .mipLevel = blit.src_level,
                .baseArrayLayer = blit.src_layer,
                .layerCount = 1,
            },
            .srcOffsets = source_offsets,
            .dstSubresource{
                .aspectMask = aspect,
                .mipLevel = blit.dst_level,
                .baseArrayLayer = blit.dst_layer,
                .layerCount = 1,
            },
            .dstOffsets = dest_offsets,
        };

        const std::array read_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange = MakeSubresourceRange(aspect, blit.src_level),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderRead |
                                 vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                 vk::AccessFlagBits::eColorAttachmentRead |
                                 vk::AccessFlagBits::eTransferRead,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange = MakeSubresourceRange(aspect, blit.dst_level),
            },
        };
        const std::array write_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange = MakeSubresourceRange(aspect, blit.src_level),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange = MakeSubresourceRange(aspect, blit.dst_level),
            },
        };

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, read_barriers);

        render_cmdbuf.blitImage(src_image, vk::ImageLayout::eTransferSrcOptimal, dst_image,
                                vk::ImageLayout::eTransferDstOptimal, blit_area, filter);

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eAllCommands,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, write_barriers);
    });
}

Framebuffer::Framebuffer(TextureRuntime& runtime, const VideoCore::FramebufferParams& params,
                         Surface* color, Surface* depth)
    : VideoCore::FramebufferParams{params}, instance{runtime.GetInstance()},
      scale{color ? color->Scaling() : (depth ? depth->Scaling() : VideoCore::SurfaceScale{})},
      sample_count{color ? color->sample_count : (depth ? depth->sample_count : u8(1u))} {
    auto& renderpass_cache = runtime.GetRenderpassCache();
    if (shadow_rendering && !color) {
        return;
    }

    width = height = std::numeric_limits<u32>::max();

    u32 num_attachments{};
    std::array<vk::ImageView, 4> attachments;

    const auto prepare = [&](u32 index, Surface* surface) {
        const auto extent = surface->RealExtent();
        width = std::min(width, extent.width);
        height = std::min(height, extent.height);
        formats[index] = surface->pixel_format;
        images[index] = surface->Image();
        aspects[index] = surface->Aspect();
        image_views[index] = surface->FramebufferView();
    };

    if (shadow_rendering) {
        const auto extent = color->RealExtent();
        width = extent.width;
        height = extent.height;
        render_pass = renderpass_cache.GetRenderpass(PixelFormat::Invalid, PixelFormat::Invalid,
                                                     false, sample_count);
        color->MakeStorageCapable();
        images[0] = color->Image();
        image_views[0] = color->StorageView();
        aspects[0] = vk::ImageAspectFlagBits::eColor;
    } else {
        if (color) {
            prepare(0, color);
            attachments[num_attachments++] = image_views[0];
        }
        if (depth) {
            prepare(1, depth);
            attachments[num_attachments++] = image_views[1];
        }

        // MSAA attachments
        if (sample_count > 1) {
            if (color) {
                images[2] = color->Image(Type::MultiSampled);
                image_views[2] = color->ImageView(ViewType::Mip0, Type::MultiSampled);
                aspects[2] = color->Aspect();
                attachments[num_attachments++] = image_views[2];
            }
            if (depth) {
                images[3] = depth->Image(Type::MultiSampled);
                image_views[3] = depth->ImageView(ViewType::Mip0, Type::MultiSampled);
                aspects[3] = depth->Aspect();
                attachments[num_attachments++] = image_views[3];
            }
        }
        render_pass = renderpass_cache.GetRenderpass(formats[0], formats[1], false, sample_count);
    }

    const vk::FramebufferCreateInfo framebuffer_info = {
        .renderPass = render_pass,
        .attachmentCount = num_attachments,
        .pAttachments = attachments.data(),
        .width = width,
        .height = height,
        .layers = 1,
    };
    framebuffer = instance.GetDevice().createFramebuffer(framebuffer_info);
}

Framebuffer::~Framebuffer() {
    if (framebuffer) {
        instance.GetDevice().destroyFramebuffer(framebuffer);
    }
}

Sampler::Sampler(TextureRuntime& runtime, const VideoCore::SamplerParams& params) {
    using TextureConfig = VideoCore::SamplerParams::TextureConfig;

    const Instance& instance = runtime.GetInstance();
    const vk::PhysicalDeviceProperties properties = instance.GetPhysicalDevice().getProperties();
    const bool use_border_color =
        instance.IsCustomBorderColorSupported() && (params.wrap_s == TextureConfig::ClampToBorder ||
                                                    params.wrap_t == TextureConfig::ClampToBorder);

    const auto color = PicaToVK::ColorRGBA8(params.border_color);
    const vk::SamplerCustomBorderColorCreateInfoEXT border_color_info = {
        .customBorderColor = MakeClearColorValue(color),
        .format = vk::Format::eUndefined,
    };

    const vk::Filter mag_filter = PicaToVK::TextureFilterMode(params.mag_filter);
    const vk::Filter min_filter = PicaToVK::TextureFilterMode(params.min_filter);
    const vk::SamplerMipmapMode mipmap_mode = PicaToVK::TextureMipFilterMode(params.mip_filter);
    const vk::SamplerAddressMode wrap_u = PicaToVK::WrapMode(params.wrap_s);
    const vk::SamplerAddressMode wrap_v = PicaToVK::WrapMode(params.wrap_t);
    const float lod_min = static_cast<float>(params.lod_min);
    const float lod_max = static_cast<float>(params.lod_max);

    // Do not apply anisotropic filtering if nearest filtering is used at all, as drivers
    // are only recommended (not enforced) to follow the mag/min filter in such cases.
    // Adreno drivers are an example of this, as they force linear filtering when using
    // anisotropic filtering.
    const bool use_anisotropy = instance.IsAnisotropicFilteringSupported() &&
                                mag_filter == vk::Filter::eLinear &&
                                min_filter == vk::Filter::eLinear;

    const vk::SamplerCreateInfo sampler_info = {
        .pNext = use_border_color ? &border_color_info : nullptr,
        .magFilter = mag_filter,
        .minFilter = min_filter,
        .mipmapMode = mipmap_mode,
        .addressModeU = wrap_u,
        .addressModeV = wrap_v,
        .mipLodBias = 0,
        .anisotropyEnable = use_anisotropy,
        .maxAnisotropy = use_anisotropy ? properties.limits.maxSamplerAnisotropy : 1.0f,
        .compareEnable = false,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = lod_min,
        .maxLod = lod_max,
        .borderColor =
            use_border_color ? vk::BorderColor::eFloatCustomEXT : vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = false,
    };
    sampler = instance.GetDevice().createSamplerUnique(sampler_info);
}

Sampler::~Sampler() = default;

DebugScope::DebugScope(TextureRuntime& runtime, Common::Vec4f color, std::string_view label)
    : scheduler{runtime.GetScheduler()},
      has_debug_tool{runtime.GetInstance().HasDebuggingToolAttached()} {
    if (!has_debug_tool) {
        return;
    }
    scheduler.Record([color, label = std::string(label)](vk::CommandBuffer cmdbuf) {
        const vk::DebugUtilsLabelEXT debug_label = {
            .pLabelName = label.data(),
            .color = std::array{color[0], color[1], color[2], color[3]},
        };
        cmdbuf.beginDebugUtilsLabelEXT(debug_label);
    });
}

DebugScope::~DebugScope() {
    if (!has_debug_tool) {
        return;
    }
    scheduler.Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endDebugUtilsLabelEXT(); });
}

} // namespace Vulkan
