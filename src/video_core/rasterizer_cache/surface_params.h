// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <concepts>
#include <boost/icl/right_open_interval.hpp>
#include "common/math_util.h"
#include "video_core/custom_textures/custom_format.h"
#include "video_core/rasterizer_cache/pixel_format.h"

namespace VideoCore {

using SurfaceInterval = boost::icl::right_open_interval<PAddr>;

constexpr std::size_t MAX_PICA_LEVELS = 8;

/**
 * Number of host pixels per guest pixel along each axis of a surface: res_scale, or half of
 * that along an axis marked as half. Host coordinates round down, so that guest rectangles that
 * touch map to host rectangles that touch.
 */
struct SurfaceScale {
    u32 res_scale = 1;
    bool half_x = false;
    bool half_y = false;

    template <std::integral T>
    [[nodiscard]] constexpr T X(T value) const noexcept {
        return Apply(value, half_x);
    }

    template <std::integral T>
    [[nodiscard]] constexpr T Y(T value) const noexcept {
        return Apply(value, half_y);
    }

    [[nodiscard]] constexpr Common::Rectangle<u32> Rect(
        const Common::Rectangle<u32>& rect) const noexcept {
        return {X(rect.left), Y(rect.top), X(rect.right), Y(rect.bottom)};
    }

    /// Guest length that a host length covers, rounded up to whole guest pixels
    [[nodiscard]] constexpr u32 GuestX(u32 value) const noexcept {
        return GuestCeil(value, half_x);
    }

    [[nodiscard]] constexpr u32 GuestY(u32 value) const noexcept {
        return GuestCeil(value, half_y);
    }

    /// Smallest guest rectangle that contains a host rectangle
    [[nodiscard]] constexpr Common::Rectangle<u32> GuestRect(
        const Common::Rectangle<u32>& rect) const noexcept {
        const bool y_up = rect.top >= rect.bottom;
        return {
            GuestFloor(rect.left, half_x),
            y_up ? GuestCeil(rect.top, half_y) : GuestFloor(rect.top, half_y),
            GuestCeil(rect.right, half_x),
            y_up ? GuestFloor(rect.bottom, half_y) : GuestCeil(rect.bottom, half_y),
        };
    }

    constexpr bool operator==(const SurfaceScale&) const noexcept = default;

private:
    template <std::integral T>
    [[nodiscard]] constexpr T Apply(T value, bool half) const noexcept {
        const T scaled = value * static_cast<T>(res_scale);
        return half ? scaled / 2 : scaled;
    }

    [[nodiscard]] constexpr u32 GuestFloor(u32 value, bool half) const noexcept {
        return (half ? value * 2 : value) / res_scale;
    }

    [[nodiscard]] constexpr u32 GuestCeil(u32 value, bool half) const noexcept {
        return ((half ? value * 2 : value) + res_scale - 1) / res_scale;
    }
};

class SurfaceParams {
public:
    /// Returns true if other_surface matches exactly params
    bool ExactMatch(const SurfaceParams& other_surface) const;

    /// Returns true if sub_surface is a subrect of params
    bool CanSubRect(const SurfaceParams& sub_surface) const;

    /// Returns true if other_surface can be used for reinterpretion.
    bool CanReinterpret(const SurfaceParams& other_surface);

    /// Returns true if params can be used for texcopy
    bool CanTexCopy(const SurfaceParams& texcopy_params) const;

    /// Updates remaining members from the already set addr, width, height and pixel_format
    void UpdateParams();

    /// Returns the unscaled rectangle referenced by sub_surface
    Common::Rectangle<u32> GetSubRect(const SurfaceParams& sub_surface) const;

    /// Returns the scaled rectangle referenced by sub_surface
    Common::Rectangle<u32> GetScaledSubRect(const SurfaceParams& sub_surface) const;

    /// Returns the outer rectangle containing interval
    SurfaceParams FromInterval(SurfaceInterval interval) const;

    /// Returns the address interval referenced by unscaled_rect
    SurfaceInterval GetSubRectInterval(Common::Rectangle<u32> unscaled_rect, u32 level = 0) const;

    /// Return the address interval of the provided level
    SurfaceInterval LevelInterval(u32 level) const;

    /// Returns the level of the provided address
    u32 LevelOf(PAddr addr) const;

    /// Returns a string identifier of the params object
    std::string DebugName(bool scaled, bool custom = false, u8 sample_count = 1) const noexcept;

    bool operator==(const SurfaceParams& other) const noexcept;

    [[nodiscard]] SurfaceInterval GetInterval() const noexcept {
        return SurfaceInterval{addr, end};
    }

    [[nodiscard]] u32 GetFormatBpp() const noexcept {
        return VideoCore::GetFormatBpp(pixel_format);
    }

    [[nodiscard]] SurfaceScale Scaling() const noexcept {
        return {res_scale, half_scale_x, half_scale_y};
    }

    [[nodiscard]] u32 GetScaledWidth() const noexcept {
        return Scaling().X(width);
    }

    [[nodiscard]] u32 GetScaledHeight() const noexcept {
        return Scaling().Y(height);
    }

    [[nodiscard]] u8 GetSampleCount() const noexcept {
        return sample_count;
    }

    [[nodiscard]] Common::Rectangle<u32> GetRect(u32 level = 0) const noexcept {
        return {0, height >> level, width >> level, 0};
    }

    [[nodiscard]] Common::Rectangle<u32> GetScaledRect(u32 level = 0) const noexcept {
        return {0, GetScaledHeight() >> level, GetScaledWidth() >> level, 0};
    }

    [[nodiscard]] u32 PixelsInBytes(u32 size) const noexcept {
        return size * 8 / GetFormatBpp();
    }

    [[nodiscard]] u32 BytesInPixels(u32 pixels) const noexcept {
        return pixels * GetFormatBpp() / 8;
    }

private:
    /// Computes the offset of each mipmap level
    void CalculateMipLevelOffsets();

    /// Calculates total surface size taking mipmaps into account
    u32 CalculateSurfaceSize() const;

public:
    PAddr addr = 0;
    PAddr end = 0;
    u32 size = 0;

    u32 width = 0;
    u32 height = 0;
    u32 stride = 0;
    u32 levels = 1;
    u32 res_scale = 1;
    u8 sample_count = 1;
    /// The surface has half of res_scale along this axis. Set for render targets that the guest
    /// shrinks to half size in that direction when it displays them (its form of anti-aliasing),
    /// where the full scale would render samples that never reach the screen.
    bool half_scale_x = false;
    bool half_scale_y = false;

    bool is_tiled = false;
    TextureType texture_type = TextureType::Texture2D;
    PixelFormat pixel_format = PixelFormat::Invalid;
    CustomPixelFormat custom_format = CustomPixelFormat::Invalid;
    SurfaceType type = SurfaceType::Invalid;

    std::array<u32, MAX_PICA_LEVELS> mipmap_offsets{};
};

} // namespace VideoCore
