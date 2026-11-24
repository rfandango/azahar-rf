// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <concepts>
#include "common/math_util.h"
#include "common/resolution_scale.h"

namespace VideoCore {

/**
 * Number of host pixels per guest pixel along each axis of a surface: res_scale, or half of
 * that along an axis marked as half.
 *
 * That need not be a whole number. A host pixel belongs to the guest pixel its centre lies in,
 * which is also what the host goes by when it decides whether a triangle covers it: guest
 * rectangles that touch come out as host rectangles that touch, and a viewport of so many guest
 * pixels draws to the host pixels that belong to them, not one more and not one less. Host
 * pixels with their centre right on the border between two guest pixels belong to the second.
 */
struct SurfaceScale {
    Common::ResolutionScale res_scale;
    bool half_x = false;
    bool half_y = false;

    /// Host pixels per guest pixel in eighths, along each axis
    [[nodiscard]] constexpr u32 EighthsX() const noexcept {
        return Eighths(half_x);
    }

    [[nodiscard]] constexpr u32 EighthsY() const noexcept {
        return Eighths(half_y);
    }

    /// The first host pixel of a guest pixel
    template <std::integral T>
    [[nodiscard]] constexpr T X(T value) const noexcept {
        return FirstOf(value, half_x);
    }

    template <std::integral T>
    [[nodiscard]] constexpr T Y(T value) const noexcept {
        return FirstOf(value, half_y);
    }

    /// Where the border of a guest pixel lies, which may be inside a host pixel
    template <std::integral T>
    [[nodiscard]] constexpr float BorderX(T value) const noexcept {
        return Border(value, half_x);
    }

    template <std::integral T>
    [[nodiscard]] constexpr float BorderY(T value) const noexcept {
        return Border(value, half_y);
    }

    [[nodiscard]] constexpr Common::Rectangle<u32> Rect(
        const Common::Rectangle<u32>& rect) const noexcept {
        return {X(rect.left), Y(rect.top), X(rect.right), Y(rect.bottom)};
    }

    /// Number of guest pixels, counted from the first one, that so many host pixels belong to
    [[nodiscard]] constexpr u32 GuestX(u32 value) const noexcept {
        return GuestEnd(value, half_x);
    }

    [[nodiscard]] constexpr u32 GuestY(u32 value) const noexcept {
        return GuestEnd(value, half_y);
    }

    /// The guest pixels that the host pixels of a rectangle belong to
    [[nodiscard]] constexpr Common::Rectangle<u32> GuestRect(
        const Common::Rectangle<u32>& rect) const noexcept {
        const bool y_up = rect.top >= rect.bottom;
        return {
            GuestOf(rect.left, half_x),
            y_up ? GuestEnd(rect.top, half_y) : GuestOf(rect.top, half_y),
            GuestEnd(rect.right, half_x),
            y_up ? GuestOf(rect.bottom, half_y) : GuestEnd(rect.bottom, half_y),
        };
    }

    constexpr bool operator==(const SurfaceScale&) const noexcept = default;

private:
    static_assert(Common::ResolutionScale::STEPS == 4);

    [[nodiscard]] constexpr u32 Eighths(bool half) const noexcept {
        return half ? res_scale.Steps() : res_scale.Steps() * 2;
    }

    /// The first host pixel with its centre at or past the border at `value` guest pixels:
    /// the border less half a host pixel, rounded up
    template <std::integral T>
    [[nodiscard]] constexpr T FirstOf(T value, bool half) const noexcept {
        return (value * static_cast<T>(Eighths(half)) * 2 + 7) >> 4;
    }

    template <std::integral T>
    [[nodiscard]] constexpr float Border(T value, bool half) const noexcept {
        return static_cast<float>(value * static_cast<T>(Eighths(half))) / 8.f;
    }

    /// The guest pixel a host pixel belongs to: the one with its centre in it
    [[nodiscard]] constexpr u32 GuestOf(u32 value, bool half) const noexcept {
        return (value * 8 + 4) / Eighths(half);
    }

    /// One past the guest pixel that the host pixel before `value` belongs to
    [[nodiscard]] constexpr u32 GuestEnd(u32 value, bool half) const noexcept {
        return value == 0 ? 0 : GuestOf(value - 1, half) + 1;
    }
};

} // namespace VideoCore
