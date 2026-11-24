// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <catch2/catch_test_macros.hpp>
#include "video_core/rasterizer_cache/surface_scale.h"

using Common::ResolutionScale;
using VideoCore::SurfaceScale;

namespace {

constexpr u32 LOWEST = ResolutionScale::STEPS;
constexpr u32 HIGHEST = 18 * ResolutionScale::STEPS;

/// Calls `func` with each scale a surface can have
void ForEachScale(auto&& func) {
    for (u32 steps = LOWEST; steps <= HIGHEST; steps++) {
        const auto res_scale = ResolutionScale::FromSteps(steps);
        func(SurfaceScale{res_scale, false, false});
        // Half of it only from twice the console's resolution on
        if (res_scale >= ResolutionScale::Whole(2)) {
            func(SurfaceScale{res_scale, true, true});
        }
    }
}

} // Anonymous namespace

TEST_CASE("SurfaceScale::X (a host pixel belongs to the guest pixel its centre is in)",
          "[video_core][surface_scale]") {
    ForEachScale([](const SurfaceScale& scale) {
        const double factor = scale.res_scale.Factor() / (scale.half_x ? 2.0 : 1.0);
        REQUIRE(scale.X(0u) == 0);
        for (u32 pixel = 0; pixel < 1024; pixel++) {
            const u32 first = scale.X(pixel);
            const u32 end = scale.X(pixel + 1);
            // Every guest pixel has host pixels of its own
            REQUIRE(first < end);
            REQUIRE(first + 0.5 >= pixel * factor);
            REQUIRE(first - 0.5 < pixel * factor);
            REQUIRE(scale.BorderX(pixel) == static_cast<float>(pixel * factor));
            REQUIRE(scale.GuestX(end) == pixel + 1);
            REQUIRE(scale.GuestRect({first, 1, end, 0}).left == pixel);
            REQUIRE(scale.GuestRect({first, 1, end, 0}).right == pixel + 1);
        }
    });
}

TEST_CASE("SurfaceScale::X (whole tiles take up whole host pixels)",
          "[video_core][surface_scale]") {
    ForEachScale([](const SurfaceScale& scale) {
        const u32 tile = scale.X(8u);
        for (u32 pixel = 0; pixel < 1024; pixel++) {
            for (const u32 tiles : {1, 3, 30, 64}) {
                REQUIRE(scale.X(pixel + 8 * tiles) == scale.X(pixel) + tile * tiles);
            }
        }
    });
}

TEST_CASE("SurfaceScale::X (before the first pixel)", "[video_core][surface_scale]") {
    ForEachScale([](const SurfaceScale& scale) {
        const s32 tile = scale.X(8);
        for (s32 pixel = -1024; pixel < 0; pixel++) {
            REQUIRE(scale.X(pixel) == scale.X(pixel + 1024) - 128 * tile);
        }
    });
}

TEST_CASE("SurfaceScale::Rect (whole factors)", "[video_core][surface_scale]") {
    for (u32 factor = 1; factor <= 18; factor++) {
        const SurfaceScale scale{ResolutionScale::Whole(factor)};
        for (u32 pixel = 0; pixel < 1024; pixel++) {
            REQUIRE(scale.X(pixel) == pixel * factor);
            REQUIRE(scale.GuestX(pixel * factor) == pixel);
        }
        for (u32 pixel = 0; pixel < 1024; pixel++) {
            REQUIRE(scale.GuestX(pixel) == (pixel + factor - 1) / factor);
            REQUIRE(scale.GuestRect({pixel, 0, pixel, 0}).left == pixel / factor);
        }
    }
}

TEST_CASE("ResolutionScale::Covering", "[common][resolution_scale]") {
    // The top screen in a window: as large as it takes to be no smaller than shown
    REQUIRE(ResolutionScale::Covering(1800, 400) == ResolutionScale::Closest(4.5f));
    REQUIRE(ResolutionScale::Covering(1416, 400) == ResolutionScale::Closest(3.75f));
    REQUIRE(ResolutionScale::Covering(1801, 400) == ResolutionScale::Closest(4.75f));
    REQUIRE(ResolutionScale::Covering(400, 400).IsNative());
    REQUIRE(ResolutionScale::Covering(100, 400).IsNative());
    REQUIRE(ResolutionScale::Covering(0, 400).IsNative());
}

TEST_CASE("ResolutionScale::Closest", "[common][resolution_scale]") {
    REQUIRE(ResolutionScale::Closest(5.f) == ResolutionScale::Whole(5));
    REQUIRE(ResolutionScale::Closest(4.6f).Factor() == 4.5f);
    REQUIRE(ResolutionScale::Closest(4.5f).RoundedUp() == 5);
    REQUIRE(ResolutionScale::Closest(5.f).RoundedUp() == 5);
    REQUIRE(ResolutionScale::Closest(0.3f).IsNative());
    REQUIRE(fmt::format("{} {}", ResolutionScale::Closest(4.75f), ResolutionScale::Whole(5)) ==
            "4.75 5");
}
