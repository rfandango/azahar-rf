// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <algorithm>
#include <functional>
#include <span>
#include <utility>
#include <vector>
#include "video_core/rasterizer_cache/framebuffer_base.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/renderer_vulkan/vk_blit_helper.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"

VK_DEFINE_HANDLE(VmaAllocation)

namespace VideoCore {
struct Material;
}

namespace Vulkan {

class Instance;
class RenderManager;
class Surface;
class DescriptorUpdateQueue;

enum Type {
    Current = -1,
    Base = 0,
    Scaled,
    MultiSampled,
    Custom,
    Copy,
    Num,
};

enum ViewType {
    Sample = 0,
    Mip0,
    Storage,
    Depth,
    Stencil,
    Max,
};

struct Handle {
    explicit Handle(const Instance& _instance) : instance(_instance) {}

    ~Handle() {
        Destroy();
    }

    Handle(const Handle& other) = delete;

    Handle(Handle&& other) noexcept
        : instance(other.instance), allocation(std::exchange(other.allocation, VK_NULL_HANDLE)),
          image(std::exchange(other.image, VK_NULL_HANDLE)),
          image_views(std::exchange(other.image_views, {})),
          framebuffer(std::exchange(other.framebuffer, VK_NULL_HANDLE)),
          width(std::exchange(other.width, 0)), height(std::exchange(other.height, 0)),
          levels(std::exchange(other.levels, 0)), layers(std::exchange(other.layers, 0)) {}

    Handle& operator=(const Handle& other) = delete;

    Handle& operator=(Handle&& other) noexcept {
        if (this == &other)
            return *this;

        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        image = std::exchange(other.image, VK_NULL_HANDLE);
        image_views = std::exchange(other.image_views, {});
        framebuffer = std::exchange(other.framebuffer, VK_NULL_HANDLE);
        width = std::exchange(other.width, 0);
        height = std::exchange(other.height, 0);
        levels = std::exchange(other.levels, 0);
        layers = std::exchange(other.layers, 0);

        return *this;
    }

    void Create(u32 width, u32 height, u32 levels, VideoCore::TextureType type, vk::Format format,
                vk::SampleCountFlagBits samples, vk::ImageUsageFlags usage,
                vk::ImageCreateFlags flags, vk::ImageAspectFlags aspect, bool need_format_list,
                std::string_view debug_name = {});

    void Destroy();

    operator bool() const {
        return allocation;
    }

    const Instance& instance;

    VmaAllocation allocation{VK_NULL_HANDLE};
    vk::Image image{VK_NULL_HANDLE};
    std::array<vk::ImageView, ViewType::Max> image_views{};
    vk::Framebuffer framebuffer{VK_NULL_HANDLE};
    u32 width{};
    u32 height{};
    u32 levels{};
    u32 layers{};
};

/**
 * Provides texture manipulation functions to the rasterizer cache
 * Separating this into a class makes it easier to abstract graphics API code
 */
class TextureRuntime {
    friend class Surface;

public:
    explicit TextureRuntime(const Instance& instance, Scheduler& scheduler,
                            RenderManager& renderpass_cache, DescriptorUpdateQueue& update_queue,
                            u32 num_swapchain_images);
    ~TextureRuntime();

    const Instance& GetInstance() const {
        return instance;
    }

    bool SupportsCustomFormat(VideoCore::CustomPixelFormat pixel_format) const {
        return instance.GetTraits(pixel_format).transfer_support;
    }

    Scheduler& GetScheduler() const {
        return scheduler;
    }

    RenderManager& GetRenderpassCache() {
        return renderpass_cache;
    }

    /// Gets an opaque tick-value used to indicate to the garbage collector when a surface was made.
    u64 GetResourceTick();

    /// Gets an opaque tick-value used to indicate to the garbage collector what surfaces can be
    /// safely deleted.
    u64 GetResourceFreeTick();

    /// Submits and waits for current GPU work.
    void Finish();

    /// Takes an image that was replaced while commands that use it may still be on their way
    /// to the GPU. It is destroyed by FreeRetired once the GPU is past them.
    void Retire(Handle&& handle);

    /// Destroys the retired images the GPU is done with. Called once a frame.
    void FreeRetired();

    /// Notes that the surface of these guest pixels had to be given storage usage
    /// (Surface::MakeStorageCapable). Games overwrite such a surface every frame (a shadow map
    /// is cleared with a memory fill, a depth buffer that is sampled as colour is rendered
    /// again), which makes the cache drop it and create it anew: a surface of pixels noted
    /// here is created with storage usage, so that it is not given new images every frame.
    void NoteStorageUse(PAddr addr, u32 width, u32 height);

    /// Whether a surface of these guest pixels had to be given storage usage before
    bool HadStorageUse(PAddr addr, u32 width, u32 height) const;

    /// Maps an internal staging buffer of the provided size for pixel uploads/downloads
    VideoCore::StagingData FindStaging(u32 size, bool upload);

    /// Attempts to reinterpret a rectangle of source to another rectangle of dest
    bool Reinterpret(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);

    /// Returns true if RelayoutTextures can convert from source to dest
    bool CanRelayoutTextures(const Surface& source, const Surface& dest) const noexcept;

    /// Fills part of dest from source, which holds the same guest bytes with another layout
    bool RelayoutTextures(Surface& source, Surface& dest,
                          const VideoCore::TextureRelayout& relayout);

    /// Fills the rectangle of the texture with the clear value provided
    bool ClearTexture(Surface& surface, const VideoCore::TextureClear& clear);

    /// Copies a rectangle of src_tex to another rectange of dst_rect
    bool CopyTextures(Surface& source, Surface& dest,
                      std::span<const VideoCore::TextureCopy> copies);

    bool CopyTextures(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy) {
        return CopyTextures(source, dest, std::array{copy});
    }

    /// Blits a rectangle of src_tex to another rectange of dst_rect
    bool BlitTextures(Surface& surface, Surface& dest, const VideoCore::TextureBlit& blit);

    /// Generates mipmaps for all the available levels of the texture
    void GenerateMipmaps(Surface& surface);

    /// Returns true if the provided pixel format needs convertion
    bool NeedsConversion(const Surface& surface) const;

    /// Returns true if surfaces may have half the resolution scale along an axis
    bool SupportsHalfScaleTargets() const noexcept {
        return true;
    }

    /// A blit that fills all of a surface is not recorded at once. Games move an image
    /// through several buffers before it is shown (Luigi's Mansion: Dark Moon five times a
    /// frame, each the size of half its picture), and most of what that makes is read only
    /// by the next copy, and only in part, or not at all. The blit waits: it is carried out
    /// when its destination's image is used or its source's image is about to change, and
    /// given up when the destination is overwritten whole or taken out of use first. A copy
    /// or blit that reads a destination that waits reads the source of the waiting blit.
    bool HasPendingBlits() const noexcept {
        return !pending_blits.empty();
    }
    /// The image of the surface with this number is about to be used: what waits to be put
    /// there is put there
    void CarryOutBlitsInto(u64 serial);
    /// The image of the surface with this number is about to change: what waits to be read
    /// from it is carried out
    void CarryOutBlitsFrom(u64 serial);
    /// All of the surface with this number is about to be overwritten, or it is gone
    void DropBlitInto(u64 serial);
    /// The cache took the surface out of use: nothing reads it any more
    void NoteSurfaceRemoved(Surface& surface);
    /// Whether the images of a surface the cache is done with are still read: by a blit that
    /// waits, or by one the GPU may not have finished
    bool StillNeeded(Surface& surface);
    u64 NewSurfaceSerial() noexcept {
        return ++last_surface_serial;
    }

private:
    /// Clears a partial texture rect using a clear rectangle
    /// What records the clear of a part of a surface, with a render pass of its own
    std::function<void()> ClearWithRenderpass(Surface& surface,
                                              const VideoCore::TextureClear& clear);

private:
    const Instance& instance;
    Scheduler& scheduler;
    RenderManager& renderpass_cache;
    BlitHelper blit_helper;
    StreamBuffer upload_buffer;
    StreamBuffer download_buffer;
    u32 num_swapchain_images;
    /// Replaced images and the tick they were last used in, see Retire
    std::vector<std::pair<Handle, u64>> retired;
    /// Guest surfaces that needed storage usage, oldest first, see NoteStorageUse
    struct StorageUse {
        PAddr addr;
        u32 width;
        u32 height;
        bool operator==(const StorageUse&) const = default;
    };
    std::vector<StorageUse> storage_uses;
    /// The blits that wait, see HasPendingBlits. Surfaces are told apart by a number of
    /// their own (the cache moves them about); the images are those the surfaces were in
    /// when the blit was asked for, which is why every change of them carries it out first.
    struct PendingBlit {
        u64 dst_serial;
        u64 src_serial;
        vk::Image src_image;
        vk::Image dst_image;
        vk::ImageAspectFlags aspect;
        vk::PipelineStageFlags stages;
        vk::AccessFlags src_access;
        vk::AccessFlags dst_access;
        VideoCore::TextureBlit blit;
    };
    std::vector<PendingBlit> pending_blits;
    /// Surfaces a blit read from late, and the tick it was recorded in
    std::vector<std::pair<u64, u64>> late_uses;
    u64 last_surface_serial{};
    const PendingBlit* FindBlitInto(u64 serial) const;
    void RecordPendingBlit(const PendingBlit& pending);
    void NoteLateUse(u64 serial);
};

class Surface : public VideoCore::SurfaceBase {
    friend class TextureRuntime;

public:
    explicit Surface(TextureRuntime& runtime, const VideoCore::SurfaceParams& params,
                     const VideoCore::SurfaceFlagBits& initial_flag_bits = {});
    explicit Surface(TextureRuntime& runtime, const VideoCore::SurfaceBase& surface,
                     const VideoCore::Material* materal);

    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    Surface(Surface&& o) noexcept = default;
    Surface& operator=(Surface&& o) noexcept = default;

    vk::ImageAspectFlags Aspect() const noexcept {
        return traits.aspect;
    }

    /// Returns the image at index, otherwise the base image
    vk::Image Image(Type type = Type::Current) const noexcept {
        if (serial.value != 0 && runtime.HasPendingBlits()) [[unlikely]] {
            runtime.CarryOutBlitsInto(serial.value);
        }
        return handles[type == Type::Current ? current : type].image;
    }

    /// The image the surface is in, without carrying out a blit that waits to be put there
    vk::Image RawImage() const noexcept {
        return handles[current].image;
    }

    /// A number of the surface's own, for what the runtime keeps about it (the cache moves
    /// surfaces about). Given out when first asked for.
    u64 Serial() noexcept {
        if (serial.value == 0) {
            serial.value = runtime.NewSurfaceSerial();
        }
        return serial.value;
    }
    /// That number, or zero if the surface was never asked for it
    u64 SerialIfAny() const noexcept {
        return serial.value;
    }

    /// Returns the image view at index, otherwise the base view
    vk::ImageView ImageView(ViewType view_type = ViewType::Sample,
                            Type type = Type::Current) noexcept;

    /// Returns a framebuffer handle for rendering to this surface
    vk::Framebuffer Framebuffer(Type type = Type::Current) noexcept;

    /// Returns width of the surface
    u32 GetWidth() const noexcept {
        return width;
    }

    /// Returns height of the surface
    u32 GetHeight() const noexcept {
        return height;
    }

    /// Returns resolution scale of the surface
    u32 GetResScale() const noexcept {
        return res_scale;
    }

    /// Returns a copy of the upscaled image handle, used for feedback loops.
    vk::ImageView CopyImageView() noexcept;

    /// Whether the surface can change places with its second image, see SwapForOverwrite
    bool CanSwapForOverwrite() const noexcept;

    /// For a draw that samples the surface it draws to and overwrites every pixel of the
    /// rectangle it is confined to: instead of copying the surface for the draw to sample, the
    /// surface moves to its second image. What is outside the rectangle is copied there, the
    /// rectangle is left for the draw to fill. Returns the view to sample, of the image the
    /// surface was in.
    vk::ImageView SwapForOverwrite(const Common::Rectangle<u32>& rect) noexcept;

    /// The surface's second image (see SwapForOverwrite and CopyImageView) is brought up to
    /// date only where it may differ from the image the surface is in. For that, everything
    /// that writes to that image has to say so: within which rectangle (in pixels of the
    /// image), or, where that is not known, all of it.
    void NoteWritten(const Common::Rectangle<u32>& rect) noexcept {
        if (serial.value != 0 && runtime.HasPendingBlits()) [[unlikely]] {
            // What waits to be read from the image is read before it changes; what waits
            // to be put there goes there first, the write covers only a part
            runtime.CarryOutBlitsFrom(serial.value);
            runtime.CarryOutBlitsInto(serial.value);
        }
        if (copy_differs_all) {
            return;
        }
        const u32 x0 = std::min(rect.left, rect.right);
        const u32 x1 = std::max(rect.left, rect.right);
        const u32 y0 = std::min(rect.bottom, rect.top);
        const u32 y1 = std::max(rect.bottom, rect.top);
        if (differs_x0 >= differs_x1 || differs_y0 >= differs_y1) {
            differs_x0 = x0;
            differs_x1 = x1;
            differs_y0 = y0;
            differs_y1 = y1;
        } else {
            differs_x0 = std::min(differs_x0, x0);
            differs_x1 = std::max(differs_x1, x1);
            differs_y0 = std::min(differs_y0, y0);
            differs_y1 = std::max(differs_y1, y1);
        }
    }
    void NoteAllWritten() noexcept {
        if (serial.value != 0 && runtime.HasPendingBlits()) [[unlikely]] {
            runtime.CarryOutBlitsFrom(serial.value);
            runtime.CarryOutBlitsInto(serial.value);
        }
        copy_differs_all = true;
    }
    /// Whether it makes a difference how small the rectangle given to NoteWritten is: not
    /// while the second image differs everywhere, or is not there
    bool TracksWritten() const noexcept {
        return !copy_differs_all;
    }

    /// Returns the framebuffer view of the surface image
    vk::ImageView FramebufferView() noexcept {
        is_framebuffer = true;
        return ImageView(ViewType::Mip0);
    }

    /// Returns the depth view of the surface image
    vk::ImageView DepthView() noexcept {
        return ImageView(ViewType::Depth);
    }

    /// Returns the stencil view of the surface image
    vk::ImageView StencilView() noexcept {
        return ImageView(ViewType::Stencil);
    }

    /// Whether the images of the surface can be bound as storage images and viewed as R32
    bool HasStorageUsage() const noexcept {
        return has_image_params &&
               static_cast<bool>(image_usage & vk::ImageUsageFlagBits::eStorage);
    }

    /// Gives the surface images that can be bound as storage images and viewed as R32, with
    /// what the ones it had held, if it has none such yet: what a shadow map and the output of
    /// a conversion done in a compute shader need, and what costs framebuffer compression on
    /// some GPUs, so that surfaces are not created with it. Ends the render pass. Views and
    /// framebuffers obtained before are for the old images. Returns whether images changed;
    /// a surface made from a texture pack stays as it is, see HasStorageUsage.
    bool MakeStorageCapable();

    /// Returns the R32 image view used for atomic load/store.
    vk::ImageView StorageView() noexcept {
        is_storage = true;
        return ImageView(ViewType::Storage);
    }

    /// Uploads pixel data in staging to a rectangle region of the surface texture
    void Upload(const VideoCore::BufferTextureCopy& upload, const VideoCore::StagingData& staging);

    /// Uploads the custom material to the surface allocation.
    void UploadCustom(const VideoCore::Material* material, u32 level);

    /// Downloads pixel data to staging from a rectangle region of the surface texture.
    /// Without `wait` the copy is only recorded: the data is there once the GPU has got past
    /// the tick it was recorded under.
    void Download(const VideoCore::BufferTextureCopy& download,
                  const VideoCore::StagingData& staging, bool wait = true);

    /// Scales up the surface to match the new resolution scale and sample-count.
    void ScaleUp(u32 new_scale, u8 new_sample_count);

    /// Returns the bpp of the internal surface format
    u32 GetInternalBytesPerPixel() const;

    /// Returns the access flags indicative of the surface
    vk::AccessFlags AccessFlags() const noexcept;

    /// Returns the pipeline stage flags indicative of the surface
    vk::PipelineStageFlags PipelineStageFlags() const noexcept;

private:
    /// Makes sure there is a second image, like the one the surface is in. Returns whether it
    /// was made just now, with nothing in it and no layout.
    bool EnsureCopyHandle();

    /// Performs blit between the scaled/unscaled images
    void BlitScale(const VideoCore::TextureBlit& blit, bool up_scale);

    /// Downloads scaled depth stencil data
    void DepthStencilDownload(const VideoCore::BufferTextureCopy& download,
                              const VideoCore::StagingData& staging);

public:
    TextureRuntime& runtime;
    const Instance& instance;
    Scheduler& scheduler;
    FormatTraits traits;
    std::array<Handle, Type::Num> handles;
    Type current{};
    vk::UniqueImageView depth_view;
    vk::UniqueImageView stencil_view;
    vk::UniqueImageView storage_view;
    bool is_framebuffer{};
    bool is_storage{};
    /// What the images of the surface were created with, for a second one of the same kind
    vk::ImageUsageFlags image_usage{};
    vk::ImageCreateFlags image_flags{};
    bool image_format_list{};
    bool has_image_params{};
    /// Where the second image may differ from the one the surface is in: all of it (also
    /// while there is no second image), or within a rectangle, which is empty while
    /// differs_x0 >= differs_x1. See NoteWritten.
    bool copy_differs_all{true};
    /// See Serial. Not taken along by a copy of the object that is left behind when the
    /// cache moves the surface.
    struct MovedSerial {
        u64 value{};
        MovedSerial() = default;
        MovedSerial(MovedSerial&& other) noexcept : value{std::exchange(other.value, 0)} {}
        MovedSerial& operator=(MovedSerial&& other) noexcept {
            value = std::exchange(other.value, 0);
            return *this;
        }
    };
    MovedSerial serial;
    u32 differs_x0{};
    u32 differs_y0{};
    u32 differs_x1{};
    u32 differs_y1{};
};

class Framebuffer : public VideoCore::FramebufferParams {
public:
    explicit Framebuffer(TextureRuntime& runtime, const VideoCore::FramebufferParams& params,
                         Surface* color, Surface* depth_stencil);
    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    Framebuffer(Framebuffer&& other) noexcept
        : VideoCore::FramebufferParams(std::move(other)), instance(other.instance),
          images(std::exchange(other.images, {})),
          image_views(std::exchange(other.image_views, {})),
          framebuffer(std::exchange(other.framebuffer, VK_NULL_HANDLE)),
          render_pass(std::exchange(other.render_pass, VK_NULL_HANDLE)),
          framebuffer_views(std::move(other.framebuffer_views)),
          aspects(std::exchange(other.aspects, {})),
          formats(std::exchange(
              other.formats, {VideoCore::PixelFormat::Invalid, VideoCore::PixelFormat::Invalid})),
          width(std::exchange(other.width, 0)), height(std::exchange(other.height, 0)),
          scale(std::exchange(other.scale, {})),
          sample_count(std::exchange(other.sample_count, 1)) {}

    Framebuffer& operator=(Framebuffer&& other) noexcept {
        VideoCore::FramebufferParams::operator=(std::move(other));
        images = std::exchange(other.images, {});
        image_views = std::exchange(other.image_views, {});
        framebuffer = std::exchange(other.framebuffer, VK_NULL_HANDLE);
        render_pass = std::exchange(other.render_pass, VK_NULL_HANDLE);
        framebuffer_views = std::move(other.framebuffer_views);
        aspects = std::exchange(other.aspects, {});
        formats = std::exchange(other.formats,
                                {VideoCore::PixelFormat::Invalid, VideoCore::PixelFormat::Invalid});
        width = std::exchange(other.width, 0);
        height = std::exchange(other.height, 0);
        scale = std::exchange(other.scale, {});
        sample_count = std::exchange(other.sample_count, 1);

        return *this;
    }

    VideoCore::PixelFormat Format(VideoCore::SurfaceType type) const noexcept {
        return formats[Index(type)];
    }

    [[nodiscard]] vk::ImageView ImageView(VideoCore::SurfaceType type) const noexcept {
        return image_views[Index(type)];
    }

    [[nodiscard]] vk::Framebuffer Handle() const noexcept {
        return framebuffer;
    }

    [[nodiscard]] std::array<vk::Image, 4> Images() const noexcept {
        return images;
    }

    [[nodiscard]] std::array<vk::ImageAspectFlags, 4> Aspects() const noexcept {
        return aspects;
    }

    /// The size its attachments have in common
    [[nodiscard]] vk::Extent2D Extent() const noexcept {
        return {width, height};
    }

    [[nodiscard]] vk::RenderPass RenderPass() const noexcept {
        return render_pass;
    }

    u32 Scale() const noexcept {
        return scale.res_scale;
    }

    /// Host pixels per guest pixel of the attachments, along each axis
    VideoCore::SurfaceScale Scaling() const noexcept {
        return scale;
    }

    u8 Samples() const noexcept {
        return sample_count;
    }

private:
    const Instance& instance;
    // Color, Depth, ColorMSAA, DepthMSAA
    std::array<vk::Image, 4> images{};
    std::array<vk::ImageView, 4> image_views{};
    vk::Framebuffer framebuffer{};
    vk::RenderPass render_pass{};
    std::vector<vk::UniqueImageView> framebuffer_views;
    std::array<vk::ImageAspectFlags, 4> aspects{};
    std::array<VideoCore::PixelFormat, 4> formats{
        VideoCore::PixelFormat::Invalid, VideoCore::PixelFormat::Invalid,
        VideoCore::PixelFormat::Invalid, VideoCore::PixelFormat::Invalid};
    u32 width{};
    u32 height{};
    VideoCore::SurfaceScale scale{};
    u8 sample_count{1};
};

class Sampler {
public:
    Sampler(TextureRuntime& runtime, const VideoCore::SamplerParams& params);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    Sampler(Sampler&& o) noexcept = default;
    Sampler& operator=(Sampler&& o) noexcept = default;

    [[nodiscard]] vk::Sampler Handle() const noexcept {
        return sampler.get();
    }

private:
    vk::UniqueSampler sampler;
};

class DebugScope {
public:
    template <typename... T>
    explicit DebugScope(TextureRuntime& runtime, Common::Vec4f color,
                        fmt::format_string<T...> format, T... args)
        : DebugScope{runtime, color, fmt::format(format, std::forward<T>(args)...)} {}
    explicit DebugScope(TextureRuntime& runtime, Common::Vec4f color, std::string_view label);
    ~DebugScope();

private:
    Scheduler& scheduler;
    bool has_debug_tool;
};

struct Traits {
    using Runtime = Vulkan::TextureRuntime;
    using Surface = Vulkan::Surface;
    using Sampler = Vulkan::Sampler;
    using Framebuffer = Vulkan::Framebuffer;
    using DebugScope = Vulkan::DebugScope;
};

using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace Vulkan
