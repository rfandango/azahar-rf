// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <fmt/format.h>
#include "video_core/pica/regs_internal.h"
#include "video_core/work_rules.h"

namespace VideoCore {

/// A draw as a rule keeps it to compare a later one against: the hash over its registers and
/// vertex shader, and the float uniforms as they were
struct DrawRecord {
    u64 state{};
    bool clear{};
    DrawFacts::FloatUniforms uniforms;
    /// What its vertex shader put out (DrawFacts::OutputHash), where whoever keeps the
    /// record asked for it
    bool has_output{};
    u64 output{};

    DrawRecord() = default;
    DrawRecord(DrawFacts& facts, bool as_clear) {
        Set(facts, as_clear);
    }
    void Set(DrawFacts& facts, bool as_clear) {
        state = facts.StateHash(as_clear);
        clear = as_clear;
        has_output = false;
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
    /// The first that differs: which register, what it held on record and what it holds now
    u32 first{};
    std::array<float, 4> was{};
    std::array<float, 4> is{};
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
    const DrawFacts::UniformSet& zero_sign = facts.UniformsZeroSign();
    for (u32 i = 0; i < now.size(); i++) {
        if ((read[i / 64] >> (i % 64) & 1) == 0 ||
            std::memcmp(&now[i], &record.uniforms[i], sizeof(now[i])) == 0) {
            continue;
        }
        // Other bytes and the same numbers: a zero with the other sign (a game works out the
        // same matrix twice, for each eye, and a product that comes to nothing has the sign
        // of what it was multiplied by). No difference unless the shader takes a reciprocal.
        if ((zero_sign[i / 64] >> (i % 64) & 1) == 0 &&
            now[i].x.ToFloat32() == record.uniforms[i].x.ToFloat32() &&
            now[i].y.ToFloat32() == record.uniforms[i].y.ToFloat32() &&
            now[i].z.ToFloat32() == record.uniforms[i].z.ToFloat32() &&
            now[i].w.ToFloat32() == record.uniforms[i].w.ToFloat32()) {
            continue;
        }
        if (difference.differing++ == 0) {
            difference.first = i;
            difference.was = {record.uniforms[i].x.ToFloat32(), record.uniforms[i].y.ToFloat32(),
                              record.uniforms[i].z.ToFloat32(), record.uniforms[i].w.ToFloat32()};
            difference.is = {now[i].x.ToFloat32(), now[i].y.ToFloat32(), now[i].z.ToFloat32(),
                             now[i].w.ToFloat32()};
        }
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

/// For the log: the registers a draw goes by, and which of them another draw has differently
inline void KeepStateRegisters(DrawFacts& facts, std::vector<u32>& kept) {
    kept.clear();
    facts.ForStateRegisters(false, [&](u32 from, u32 to) {
        kept.insert(kept.end(), &facts.Regs().reg_array[from], &facts.Regs().reg_array[to]);
    });
}

inline std::string DifferingRegisters(DrawFacts& facts, const std::vector<u32>& kept) {
    std::string text{"registers or the vertex shader differ"};
    const Pica::RegsInternal& regs = facts.Regs();
    std::size_t place = 0;
    u32 found = 0;
    bool same_stretches = true;
    facts.ForStateRegisters(false, [&](u32 from, u32 to) {
        for (u32 id = from; id < to; id++, place++) {
            if (place >= kept.size()) {
                same_stretches = false;
                return;
            }
            if (kept[place] != regs.reg_array[id] && found < 4) {
                text += fmt::format("{} register {:03X} was {:08X} and is {:08X}",
                                    found == 0 ? ":" : ",", id, kept[place], regs.reg_array[id]);
                found++;
            }
        }
    });
    if (!same_stretches || place != kept.size()) {
        text += ": not the same registers are gone by (a texture unit or the lights on or off)";
    } else if (found == 0) {
        text += ": no register, so the vertex shader's program or its operand table";
    }
    return text;
}

} // namespace VideoCore
