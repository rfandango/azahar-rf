// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <algorithm>
#include <array>
#include "common/common_types.h"
#include "video_core/pica/regs_external.h"

namespace VideoCore {

/// The buffers the top screen showed to the left eye lately, and those it showed to the right
/// eye that were not also the left eye's (a game goes through two or three of each in turn)
struct EyeBuffers {
    std::array<PAddr, 8> left{};
    std::array<PAddr, 8> right{};
    u32 left_next{};
    u32 right_next{};

    /// Called when the game swapped the buffers of the top screen
    void Remember(const Pica::FramebufferConfig& top_screen) {
        const auto remember = [](std::array<PAddr, 8>& shown, u32& next, PAddr buffer) {
            if (buffer != 0 && std::ranges::find(shown, buffer) == shown.end()) {
                shown[next++ % shown.size()] = buffer;
            }
        };
        remember(left, left_next, top_screen.address_left1);
        remember(left, left_next, top_screen.address_left2);
        for (const PAddr address : {top_screen.address_right1, top_screen.address_right2}) {
            if (std::ranges::find(left, address) == left.end()) {
                remember(right, right_next, address);
            }
        }
    }

    /// Whether memory holds where a buffer begins. (A buffer an eye is shown need not begin
    /// where a transfer to it writes: New Super Mario Bros. 2 has the screen read from sixteen
    /// rows into what its transfers fill)
    static bool HoldsOneOf(const std::array<PAddr, 8>& shown, PAddr start, PAddr end) {
        return start != 0 && std::ranges::any_of(shown, [start, end](PAddr address) {
                   return address != 0 && address >= start && address < end;
               });
    }
    bool IsLeft(PAddr start, PAddr end) const {
        return HoldsOneOf(left, start, end);
    }
    bool IsRight(PAddr start, PAddr end) const {
        return !IsLeft(start, end) && HoldsOneOf(right, start, end);
    }
};

} // namespace VideoCore
