// Copyright 2023-2025 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include "common/hash.h"
#include "common/math_util.h"
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/rasterizer_cache/slot_id.h"
#include "video_core/rasterizer_cache/surface_params.h"

namespace VideoCore {

class SurfaceBase;

/// Not in whole host pixels: where the scale is not a whole number, neither is the place of
/// the viewport the game asks for
struct ViewportInfo {
    f32 x;
    f32 y;
    f32 width;
    f32 height;
};

struct FramebufferParams {
    SurfaceId color_id;
    SurfaceId depth_id;
    u32 color_level;
    u32 depth_level;
    bool shadow_rendering;
    /// Which of its two images the colour surface is in, see SurfaceBase::swapped
    bool color_swapped;
    INSERT_PADDING_BYTES(2);
    /// The images of the two surfaces the framebuffer is for, see SurfaceBase::image_generation
    u32 color_generation;
    u32 depth_generation;

    bool operator==(const FramebufferParams& params) const noexcept {
        return std::memcmp(this, &params, sizeof(FramebufferParams)) == 0;
    }

    u64 Hash() const noexcept {
        return Common::ComputeHash64(this, sizeof(FramebufferParams));
    }

    u32 Index(VideoCore::SurfaceType type) const noexcept {
        switch (type) {
        case VideoCore::SurfaceType::Color:
            return 0;
        case VideoCore::SurfaceType::Depth:
        case VideoCore::SurfaceType::DepthStencil:
            return 1;
        default:
            LOG_CRITICAL(HW_GPU, "Unknown surface type in framebuffer");
            return 0;
        }
    }
};
static_assert(std::has_unique_object_representations_v<FramebufferParams>,
              "FramebufferParams is not suitable for hashing");

template <class T>
class RasterizerCache;

/**
 * @brief FramebufferHelper is a RAII wrapper over backend specific framebuffer handle that
 * provides the viewport/scissor/draw rectanges and performs automatic rasterizer cache invalidation
 * when out of scope.
 */
template <class T>
class FramebufferHelper {
public:
    explicit FramebufferHelper(RasterizerCache<T>* res_cache_, typename T::Framebuffer* fb_,
                               bool flip_rect, const Pica::RasterizerRegs& regs,
                               Common::Rectangle<u32> surfaces_rect)
        : res_cache{res_cache_}, fb{fb_} {
        const SurfaceScale scale = fb->Scaling();
        const u32 height = scale.GuestY(surfaces_rect.GetHeight());

        // Determine the draw rectangle (render area + scissor)
        Common::Rectangle viewport_rect = regs.GetViewportRect();
        if (flip_rect) {
            viewport_rect = viewport_rect.VerticalMirror(height);
        }

        const Common::Rectangle<s32> viewport_scaled{
            scale.X(viewport_rect.left),
            scale.Y(viewport_rect.top),
            scale.X(viewport_rect.right),
            scale.Y(viewport_rect.bottom),
        };

        draw_rect.left =
            std::clamp<s32>(static_cast<s32>(surfaces_rect.left) + viewport_scaled.left,
                            surfaces_rect.left, surfaces_rect.right);
        draw_rect.top =
            std::clamp<s32>(static_cast<s32>(surfaces_rect.bottom) + viewport_scaled.top,
                            surfaces_rect.bottom, surfaces_rect.top);
        draw_rect.right =
            std::clamp<s32>(static_cast<s32>(surfaces_rect.left) + viewport_scaled.right,
                            surfaces_rect.left, surfaces_rect.right);
        draw_rect.bottom =
            std::clamp<s32>(static_cast<s32>(surfaces_rect.bottom) + viewport_scaled.bottom,
                            surfaces_rect.bottom, surfaces_rect.top);

        // Update viewport. (Its size is what the game says, whichever way round its corners
        // end up.)
        viewport.x = static_cast<f32>(surfaces_rect.left) + scale.BorderX(viewport_rect.left);
        viewport.y = static_cast<f32>(surfaces_rect.bottom) + scale.BorderY(viewport_rect.bottom);
        viewport.width = scale.BorderX(viewport_rect.GetWidth());
        viewport.height = scale.BorderY(viewport_rect.GetHeight());

        // Scissor checks are window-, not viewport-relative, which means that if the cached texture
        // sub-rect changes, the scissor bounds also need to be updated.
        const u32 scissor_x1 = regs.scissor_test.x1;
        const u32 scissor_y1 = regs.scissor_test.y1;
        scissor_rect.left = static_cast<s32>(surfaces_rect.left + scale.X(scissor_x1));
        scissor_rect.bottom = static_cast<s32>(surfaces_rect.bottom + scale.Y(scissor_y1));

        // x2, y2 have +1 added to cover the entire pixel area, otherwise you might get cracks when
        // scaling or doing multisampling.
        const u32 scissor_x2 = regs.scissor_test.x2 + 1;
        const u32 scissor_y2 = regs.scissor_test.y2 + 1;
        scissor_rect.right = static_cast<s32>(surfaces_rect.left + scale.X(scissor_x2));
        scissor_rect.top = static_cast<s32>(surfaces_rect.bottom + scale.Y(scissor_y2));

        if (flip_rect) {
            scissor_rect = scissor_rect.VerticalMirror(height);
        }
    }

    ~FramebufferHelper() {
        const Common::Rectangle draw_rect_unscaled{fb->Scaling().GuestRect(draw_rect)};
        const auto invalidate = [&](SurfaceId surface_id, u32 level) {
            const auto& surface = res_cache->GetSurface(surface_id);
            const SurfaceInterval interval = surface.GetSubRectInterval(draw_rect_unscaled, level);
            const PAddr addr = boost::icl::first(interval);
            const u32 size = boost::icl::length(interval);
            res_cache->InvalidateRegion(addr, size, surface_id);
        };
        if (fb->color_id && color_written) {
            invalidate(fb->color_id, fb->color_level);
        }
        if (fb->depth_id && depth_written) {
            invalidate(fb->depth_id, fb->depth_level);
        }
    }

    /// Tells the helper which attachments the draw can modify. An attachment that stays bound
    /// only to keep the render pass going is not written and must not be marked as modified.
    void SetWrittenAttachments(bool color, bool depth) noexcept {
        color_written = color;
        depth_written = depth;
    }

    /// To be called after something that can have the surfaces change images, as the syncing
    /// of the textures of a draw does: the framebuffer to go on with
    typename T::Framebuffer* Refresh() {
        const FramebufferParams& held = *fb;
        if (res_cache->WithCurrentImages(held) != held) {
            fb = res_cache->RefreshFramebuffer(held);
        }
        return fb;
    }

    typename T::Framebuffer* Framebuffer() const noexcept {
        return fb;
    }

    Common::Rectangle<u32> DrawRect() const noexcept {
        return draw_rect;
    }

    Common::Rectangle<s32> Scissor() const noexcept {
        return scissor_rect;
    }

    ViewportInfo Viewport() const noexcept {
        return viewport;
    }

private:
    RasterizerCache<T>* res_cache;
    typename T::Framebuffer* fb;
    Common::Rectangle<s32> scissor_rect;
    Common::Rectangle<u32> draw_rect;
    bool color_written{true};
    bool depth_written{true};
    ViewportInfo viewport;
};

} // namespace VideoCore

namespace std {
template <>
struct hash<VideoCore::FramebufferParams> {
    std::size_t operator()(const VideoCore::FramebufferParams& params) const noexcept {
        return params.Hash();
    }
};
} // namespace std
