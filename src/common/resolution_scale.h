// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <algorithm>
#include <cmath>
#include <compare>
#include <fmt/format.h>
#include "common/common_types.h"

namespace Common {

/**
 * Host pixels per pixel of the console, in steps of a quarter. The console's images are made of
 * tiles of eight by eight pixels, so what is made of whole tiles takes up whole host pixels at
 * each of these scales.
 */
class ResolutionScale {
public:
    /// Steps of a whole factor
    static constexpr u32 STEPS = 4;

    /// The console's own resolution
    constexpr ResolutionScale() = default;

    [[nodiscard]] static constexpr ResolutionScale FromSteps(u32 steps) noexcept {
        ResolutionScale scale;
        scale.steps = steps;
        return scale;
    }

    [[nodiscard]] static constexpr ResolutionScale Whole(u32 factor) noexcept {
        return FromSteps(factor * STEPS);
    }

    /// The scale closest to a factor, the console's own resolution at the least
    [[nodiscard]] static ResolutionScale Closest(float factor) noexcept {
        return FromSteps(static_cast<u32>(
            std::max(std::lround(factor * static_cast<float>(STEPS)), static_cast<long>(STEPS))));
    }

    /// The smallest scale at which so many of the console's pixels take up so many host pixels,
    /// the console's own resolution at the least
    [[nodiscard]] static constexpr ResolutionScale Covering(u32 host_pixels, u32 pixels) noexcept {
        return FromSteps(std::max((host_pixels * STEPS + pixels - 1) / pixels, STEPS));
    }

    [[nodiscard]] constexpr u32 Steps() const noexcept {
        return steps;
    }

    [[nodiscard]] constexpr float Factor() const noexcept {
        return static_cast<float>(steps) / static_cast<float>(STEPS);
    }

    /// The whole factor that is the same or the next one above
    [[nodiscard]] constexpr u32 RoundedUp() const noexcept {
        return (steps + STEPS - 1) / STEPS;
    }

    [[nodiscard]] constexpr bool IsNative() const noexcept {
        return steps == STEPS;
    }

    constexpr auto operator<=>(const ResolutionScale&) const noexcept = default;

private:
    u32 steps = STEPS;
};

} // namespace Common

template <>
struct fmt::formatter<Common::ResolutionScale> : fmt::formatter<float> {
    auto format(const Common::ResolutionScale& scale, fmt::format_context& ctx) const {
        return fmt::formatter<float>::format(scale.Factor(), ctx);
    }
};
