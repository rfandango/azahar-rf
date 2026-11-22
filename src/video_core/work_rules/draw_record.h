// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "video_core/work_rules.h"

namespace VideoCore {

/// A draw as a rule keeps it to compare a later one against: the hash over its registers and
/// vertex shader, and the float uniforms as they were
struct DrawRecord {
    u64 state{};
    bool clear{};
    DrawFacts::FloatUniforms uniforms;

    DrawRecord() = default;
    DrawRecord(DrawFacts& facts, bool as_clear)
        : state{facts.StateHash(as_clear)}, clear{as_clear} {
        if (!as_clear) {
            std::memcpy(uniforms.data(), facts.Uniforms().data(), sizeof(uniforms));
        }
    }
};

/// How the float uniforms the shader of the draw reads differ from those on record
struct UniformDifference {
    /// How many of them differ
    u32 differing{};
    /// The largest difference, as a share of the largest value in the registers around the
    /// uniform (the rows of its matrix); infinite for one that is not a number
    float largest_shift{};
};

/// Compares only the uniforms the program reads: the others hold what earlier draws left
/// there. Gives up, with what it found so far, once more than give_up_after differ or one
/// differs by more than give_up_above.
inline UniformDifference CompareUniforms(const DrawRecord& record, DrawFacts& facts,
                                         u32 give_up_after, float give_up_above) {
    UniformDifference difference;
    const DrawFacts::FloatUniforms& now = facts.Uniforms();
    if (std::memcmp(now.data(), record.uniforms.data(), sizeof(record.uniforms)) == 0) {
        return difference;
    }
    const DrawFacts::UniformSet& read = facts.UniformsRead();
    for (u32 i = 0; i < now.size(); i++) {
        if ((read[i / 64] >> (i % 64) & 1) == 0 ||
            std::memcmp(&now[i], &record.uniforms[i], sizeof(now[i])) == 0) {
            continue;
        }
        difference.differing++;
        float scale = 1e-4f;
        for (u32 j = i >= 3 ? i - 3 : 0; j < std::min<u32>(i + 4, 96); j++) {
            const auto& row = record.uniforms[j];
            scale = std::max({scale, std::abs(row.x.ToFloat32()), std::abs(row.y.ToFloat32()),
                              std::abs(row.z.ToFloat32()), std::abs(row.w.ToFloat32())});
        }
        const float shift =
            std::max({std::abs(now[i].x.ToFloat32() - record.uniforms[i].x.ToFloat32()),
                      std::abs(now[i].y.ToFloat32() - record.uniforms[i].y.ToFloat32()),
                      std::abs(now[i].z.ToFloat32() - record.uniforms[i].z.ToFloat32()),
                      std::abs(now[i].w.ToFloat32() - record.uniforms[i].w.ToFloat32())}) /
            scale;
        // (Also taken as too large: a value that is not a number)
        if (!(shift <= difference.largest_shift)) {
            difference.largest_shift = shift <= std::numeric_limits<float>::max()
                                           ? shift
                                           : std::numeric_limits<float>::infinity();
        }
        if (difference.differing > give_up_after || !(shift <= give_up_above)) {
            break;
        }
    }
    return difference;
}

} // namespace VideoCore
