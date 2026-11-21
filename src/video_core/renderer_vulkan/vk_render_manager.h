// Copyright 2024-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <bit>
#include <functional>
#include <mutex>
#include <boost/container/static_vector.hpp>

#include "common/math_util.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
enum class PixelFormat : u32;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Framebuffer;

struct RenderPass {
    vk::Framebuffer framebuffer;
    vk::RenderPass render_pass;
    vk::Rect2D render_area;
    std::array<vk::ClearValue, 2> clears;
    u32 do_clear;

    bool operator==(const RenderPass& other) const noexcept {
        return std::tie(framebuffer, render_pass, render_area, do_clear) ==
                   std::tie(other.framebuffer, other.render_pass, other.render_area,
                            other.do_clear) &&
               std::memcmp(&clears, &other.clears, sizeof(clears)) == 0;
    }
};

/// A clear of a rectangle of an image that is held back, for the render pass that follows
/// to do as it begins if the image is one of its attachments
struct DeferredClear {
    vk::Image image;
    vk::Rect2D area;
    vk::ClearValue value;
    /// Records the clear by itself, as it is done when it is not held back
    std::function<void()> carry_out;
};

class RenderManager {
    static constexpr u32 NumColorFormats = static_cast<u32>(VideoCore::PixelFormat::NumColorFormat);
    static constexpr u32 NumDepthFormats = static_cast<u32>(VideoCore::PixelFormat::NumDepthFormat);
    static constexpr size_t MaxSamples = 16;
    static_assert(std::has_single_bit(MaxSamples));

public:
    explicit RenderManager(const Instance& instance, Scheduler& scheduler);
    ~RenderManager();

    /// Begins a new renderpass with the provided framebuffer as render target, unless the one
    /// that is open can go on. The draw that follows is confined to the rectangle; if it
    /// overwrites all the colour or all the depth and stencil in it, a pass that begins here
    /// does not load those.
    void BeginRendering(const Framebuffer* framebuffer, Common::Rectangle<u32> draw_rect,
                        bool overwrites_color = false, bool overwrites_depth = false);

    /// Begins a new renderpass with the provided render state.
    void BeginRendering(const RenderPass& new_pass);

    /// Exits from any currently active renderpass instance, and carries out the clears that
    /// were held back. Everything that records commands of its own calls this first.
    void EndRendering();

    /// While this is on, clears given to DeferClear are held back. A render pass that begins
    /// on their images, with an area that can be theirs, clears as it begins: a GPU that
    /// renders in tiles then starts from the clear value, where it would have loaded from
    /// memory what a clear before the pass had put there. Whatever else is recorded first
    /// has them carried out before it, and so does turning this off.
    void SetClearDeferral(bool on);
    bool DefersClears() const noexcept {
        return defer_clears;
    }

    /// Ends the render pass that is open and holds the clear back
    void DeferClear(DeferredClear&& clear);

    /// Returns the renderpass associated with the color-depth format pair
    vk::RenderPass GetRenderpass(VideoCore::PixelFormat color, VideoCore::PixelFormat depth,
                                 bool is_clear, u8 sample_count = 1);

    /// Returns a renderpass for the format pair that begins by doing this to each of its
    /// attachments: loading it, clearing it, or neither, which leaves its content in the
    /// render area undefined until it is drawn over. It may leave depth and stencil
    /// unstored when it ends.
    vk::RenderPass GetRenderpassBeginningWith(VideoCore::PixelFormat color,
                                              VideoCore::PixelFormat depth,
                                              vk::AttachmentLoadOp color_load,
                                              vk::AttachmentLoadOp depth_load,
                                              bool store_depth = true);

private:
    /// Creates a renderpass configured appropriately and stores it in cached_renderpasses
    vk::UniqueRenderPass CreateRenderPass(vk::Format color, vk::Format depth,
                                          vk::AttachmentLoadOp color_load_op,
                                          vk::AttachmentLoadOp depth_load_op,
                                          vk::AttachmentStoreOp depth_store_op) const;

    /// Creates an MSAA renderpass configured appropriately and stores it in cached_renderpasses
    vk::UniqueRenderPass CreateRenderPassMSAA(vk::Format color, vk::Format depth,
                                              vk::AttachmentLoadOp load_op,
                                              vk::SampleCountFlagBits sample_count) const;

private:
    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueRenderPass cached_renderpasses[NumColorFormats + 1][NumDepthFormats + 1]
                                            [std::bit_width(MaxSamples)][2];
    /// Ends the render pass that is open, if any
    void EndPass();
    void CarryOutClears();

    /// Those that begin otherwise than by loading both attachments, by what they do to
    /// colour and to depth, and those of them that do not store depth
    vk::UniqueRenderPass beginning_renderpasses[NumColorFormats + 1][NumDepthFormats + 1][2][3][3];
    boost::container::static_vector<DeferredClear, 2> deferred_clears;
    bool defer_clears{};
    std::mutex cache_mutex;
    std::array<vk::Image, 4> images;
    std::array<vk::ImageAspectFlags, 4> aspects;
    bool shadow_rendering{};
    RenderPass pass{};
    u32 num_draws{};
};

} // namespace Vulkan
