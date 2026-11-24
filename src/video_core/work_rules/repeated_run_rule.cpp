// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <string>
#include <utility>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/trace_markers.h"
#include "video_core/frame_state_dump.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/work_rules/repeated_run_rule.h"

namespace VideoCore {

namespace {

constexpr u64 Mix(u64 hash, u64 value) {
    hash ^= value + 0x9E3779B97F4A7C15ULL + (hash << 12) + (hash >> 4);
    hash *= 0xBF58476D1CE4E5B9ULL;
    return hash ^ (hash >> 31);
}

/// A few registers that tell draws apart: the target, the viewport, the first texture, how
/// many vertices and where the program begins
u64 QuickHash(u64 hash, const Pica::RegsInternal& regs) {
    const auto& array = regs.reg_array;
    hash = Mix(hash, (static_cast<u64>(array[0x11D]) << 32) | array[0x11C]);
    hash = Mix(hash, (static_cast<u64>(array[0x041]) << 32) | array[0x043]);
    hash = Mix(hash, (static_cast<u64>(array[0x085]) << 32) | array[0x228]);
    return Mix(hash, (static_cast<u64>(array[0x2BA]) << 32) | array[0x101]);
}

bool Explains() {
    return (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::ExplainScenePasses) != 0;
}

} // Anonymous namespace

void RepeatedRunRule::Note(PAddr start, PAddr end) {
    const Memory memory{start, end};
    if (start >= end || std::ranges::find(touched, memory) != touched.end()) {
        return;
    }
    if (touched.size() < MAX_MEMORY) {
        touched.push_back(memory);
    } else {
        too_much = true;
    }
}

void RepeatedRunRule::NoteDraw(const Pica::RegsInternal& regs) {
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const std::array<u32, 4> targets{regs.reg_array[0x11C], regs.reg_array[0x11D],
                                     regs.reg_array[0x11E],
                                     (regs.reg_array[0x116] << 16) ^ regs.reg_array[0x117]};
    if (targets != last_targets) {
        last_targets = targets;
        const u32 pixels = framebuffer.GetWidth() * framebuffer.GetHeight();
        if (const PAddr colour = framebuffer.GetColorBufferPhysicalAddress(); colour != 0) {
            Note(colour, colour + pixels * Pica::FramebufferRegs::BytesPerColorPixel(
                                               framebuffer.color_format));
        }
        if (const PAddr depth = framebuffer.GetDepthBufferPhysicalAddress(); depth != 0) {
            Note(depth, depth + pixels * Pica::FramebufferRegs::BytesPerDepthPixel(
                                             framebuffer.depth_format));
        }
    }
    const auto& main_config = regs.texturing.main_config;
    const std::array<std::pair<bool, u32>, 3> units{
        {{main_config.texture0_enable != 0, regs.reg_array[0x085]},
         {main_config.texture1_enable != 0, regs.reg_array[0x095]},
         {main_config.texture2_enable != 0, regs.reg_array[0x09D]}}};
    for (const auto& [enabled, address_register] : units) {
        if (!enabled) {
            continue;
        }
        const PAddr address = address_register * 8;
        const u32 bit = (address_register * 0x9E3779B1u) >> 22;
        u64& bits = sampled_bits[bit / 64];
        const u64 mask = u64{1} << (bit % 64);
        if ((bits & mask) != 0 && std::ranges::find(sampled, address) != sampled.end()) {
            continue;
        }
        bits |= mask;
        if (sampled.size() < MAX_SAMPLED) {
            sampled.push_back(address);
        } else {
            too_much = true;
        }
    }
}

bool RepeatedRunRule::Touches(PAddr start, PAddr end) const {
    return too_much ||
           std::ranges::any_of(touched,
                               [start, end](const Memory& memory) {
                                   return memory.start < end && start < memory.end;
                               }) ||
           std::ranges::any_of(
               sampled, [start, end](PAddr address) { return address >= start && address < end; });
}

void RepeatedRunRule::Begin(const Fill& fill) {
    phase = Phase::First;
    fresh = false;
    first_fill = fill;
    quick = Mix(Mix(1, (static_cast<u64>(fill.start) << 32) | fill.end), fill.value);
    count = 1;
    recording = quick_alike;
    recorded = 0;
    touched.clear();
    sampled.clear();
    sampled_bits.fill(0);
    last_targets.fill(0);
    first_draw_recorded = false;
    too_much = false;
    if (recording) {
        first.resize(std::max<std::size_t>(first.size(), 1));
        first[0].is_fill = true;
        first[0].fill = fill;
        recorded = 1;
        Note(fill.start, fill.end);
    }
}

void RepeatedRunRule::BeginSecond() {
    phase = Phase::Second;
    second_quick = 1;
    at = 0;
    left_out_now = 0;
    repeats = recording;
    leaving_out =
        recording && frames_alike >= frames_needed &&
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::KeepRepeatedPass) == 0;
}

void RepeatedRunRule::End(const char* why) {
    if (phase == Phase::Second && left_out_now != 0) {
        LOG_INFO(HW_GPU,
                 "The image drawn twice from {:08X} is no longer: {} ({} pieces of the second run "
                 "were left out)",
                 first_fill.start, why, left_out_now);
        frames_needed = std::min(frames_needed * 4, MOST_FRAMES_ALIKE);
    }
    if (phase == Phase::Second) {
        frames_alike = 0;
    }
    phase = Phase::Idle;
}

bool RepeatedRunRule::Repeats(bool same, const std::string& difference) {
    if (!same && repeats) {
        if (explained < (Explains() ? 12u : 2u)) {
            explained++;
            LOG_INFO(HW_GPU,
                     "Second run from {:08X}: piece {} of {} does not repeat the first run: {}",
                     first_fill.start, at, recorded, difference);
        }
        repeats = false;
        leaving_out = false;
    }
    at++;
    if (same && leaving_out) {
        left_out_now++;
        return true;
    }
    return false;
}

Vote RepeatedRunRule::OnFill(const FillFacts& facts) {
    const Fill fill{facts.start, facts.end, facts.config.value_32bit, facts.config.control & 0x300};
    switch (phase) {
    case Phase::Idle:
        if (fresh) {
            Begin(fill);
        }
        return Vote::Carry;
    case Phase::First:
        if (++count > MAX_PIECES) {
            phase = Phase::Idle;
            return Vote::Carry;
        }
        quick = Mix(Mix(quick, (static_cast<u64>(fill.start) << 32) | fill.end), fill.value);
        if (recording) {
            first.resize(std::max<std::size_t>(first.size(), recorded + 1));
            Piece& piece = first[recorded++];
            piece.is_fill = true;
            piece.fill = fill;
            Note(fill.start, fill.end);
        }
        return Vote::Carry;
    case Phase::Closed:
        if (fill != first_fill) {
            // Another image: it may be the first of two
            Begin(fill);
            return Vote::Carry;
        }
        BeginSecond();
        [[fallthrough]];
    default: {
        second_quick =
            Mix(Mix(second_quick, (static_cast<u64>(fill.start) << 32) | fill.end), fill.value);
        const bool same = repeats && at < recorded && first[at].is_fill && first[at].fill == fill;
        return Repeats(same, "another fill, or none, was there") ? Vote::Redundant : Vote::Carry;
    }
    }
}

Vote RepeatedRunRule::OnDraw(DrawFacts& facts) {
    if (phase == Phase::Idle) [[likely]] {
        fresh = false;
        return Vote::Carry;
    }
    if (facts.IsImmediate()) {
        // A triangle that is not on record and cannot be compared
        End("a triangle given vertex by vertex was drawn");
        fresh = false;
        return Vote::Carry;
    }
    const Pica::RegsInternal& regs = facts.Regs();
    if (phase == Phase::Closed) {
        // Something is drawn before the same fill comes: no second run
        phase = Phase::Idle;
        fresh = false;
        return Vote::Carry;
    }
    if (phase == Phase::First) {
        if (++count > MAX_PIECES) {
            phase = Phase::Idle;
            return Vote::Carry;
        }
        quick = QuickHash(quick, regs);
        if (!recording) {
            return Vote::Carry;
        }
        first.resize(std::max<std::size_t>(first.size(), recorded + 1));
        Piece& piece = first[recorded];
        piece.is_fill = false;
        piece.draw.Set(facts, false);
        piece.registers.clear();
        if (Explains()) {
            KeepStateRegisters(facts, piece.registers);
        }
        if (!first_draw_recorded) {
            // What the first draw puts out is kept: its shader may read uniforms the game does
            // not set for it, and those hold what the draw before the run left
            first_draw_recorded = true;
            piece.draw.has_output = facts.OutputHash(piece.draw.output);
        }
        recorded++;
        NoteDraw(regs);
        return Vote::Carry;
    }

    // The second run
    second_quick = QuickHash(second_quick, regs);
    if (!repeats) {
        at++;
        return Vote::Carry;
    }
    const bool known = at < recorded && !first[at].is_fill;
    bool same = known && first[at].draw.state == facts.StateHash(false);
    std::string difference;
    if (same) {
        const DrawRecord& draw = first[at].draw;
        const UniformDifference uniforms = CompareUniforms(draw, facts, 96, MAX_SAME_SHIFT);
        bool alike = uniforms.differing == 0 || uniforms.largest_shift <= MAX_SAME_SHIFT;
        u64 output = 0;
        if (!alike && draw.has_output && facts.OutputHash(output)) {
            alike = output == draw.output;
        }
        if (!alike) {
            same = false;
            difference = fmt::format(
                "{} float uniforms its shader reads differ, the first is {}: {} {} {} {} in the "
                "first run, {} {} {} {} now, a shift of {:.3g} of its matrix",
                uniforms.differing, uniforms.first, uniforms.was[0], uniforms.was[1],
                uniforms.was[2], uniforms.was[3], uniforms.is[0], uniforms.is[1], uniforms.is[2],
                uniforms.is[3], uniforms.largest_shift);
        }
    } else {
        difference = !known ? std::string{"the first run had a fill there or no more"}
                            : fmt::format("{} ({} vertices)",
                                          first[at].registers.empty()
                                              ? std::string{"registers or the vertex shader differ"}
                                              : DifferingRegisters(facts, first[at].registers),
                                          static_cast<u32>(regs.pipeline.num_vertices));
    }
    return Repeats(same, difference) ? Vote::Redundant : Vote::Carry;
}

Vote RepeatedRunRule::OnTransfer(const TransferFacts& facts) {
    const bool reads_image = facts.source < first_fill.end && first_fill.start < facts.source_end;
    switch (phase) {
    case Phase::First:
        if (reads_image && eyes.IsLeft(facts.dest, facts.dest_end)) {
            CITRA_FRAME_DUMP("  a run of {} pieces from a fill made the left eye's image{}", count,
                             recording ? "" : " (not on record)");
            phase = Phase::Closed;
            went_to = {facts.dest, facts.dest_end};
        } else {
            phase = Phase::Idle;
        }
        break;
    case Phase::Closed:
        // Others may go by the same image. None may change what the run goes by.
        if (Touches(facts.dest, facts.dest_end)) {
            CITRA_FRAME_DUMP("  a run of {} pieces from a fill is not followed: this writes to "
                             "what it fills, draws to or samples",
                             count);
            phase = Phase::Idle;
        }
        break;
    case Phase::Second: {
        const bool to_right_eye =
            reads_image && facts.dest != went_to.start && eyes.IsRight(facts.dest, facts.dest_end);
        CITRA_FRAME_DUMP("  the second run from a fill: {} pieces against {}, {}, {} left out, "
                         "{}for a buffer of the right eye",
                         at, count,
                         !recording ? "not on record"
                         : repeats  ? "all of them repeats"
                                    : "not all repeats",
                         left_out_now, to_right_eye ? "" : "not ");
        if (!to_right_eye) {
            End("its second image did not go to the right eye's buffer");
            break;
        }
        if (at == count && second_quick == quick) {
            quick_alike = true;
            last_quick_alike = frame;
        }
        if (!recording) {
            phase = Phase::Idle;
            break;
        }
        if (!repeats || at != recorded) {
            End("its second run was not like the first");
            break;
        }
        last_alike = frame;
        frames_alike = std::min(frames_alike + 1, 1u << 20);
        if (frames_alike == frames_needed) {
            LOG_INFO(HW_GPU,
                     "The game makes the image of the top screen twice from {:08X}, once for each "
                     "eye, and the two are the same: {} fills and draws each, every one of the "
                     "second run the one at its place in the first. The second run is left out "
                     "from here on.",
                     first_fill.start, recorded);
        }
        phase = Phase::Idle;
        break;
    }
    default:
        break;
    }
    fresh = true;
    return Vote::Carry;
}

void RepeatedRunRule::OnFrameEnd(const FrameFacts& facts) {
    eyes.Remember(facts.top_screen);
    frame++;
    if (frames_alike != 0 && frame - last_alike > FRAMES_WITHOUT) {
        if (frames_alike >= frames_needed) {
            LOG_INFO(HW_GPU,
                     "The image drawn twice from {:08X} is no longer: {} frames went by "
                     "without it",
                     first_fill.start, FRAMES_WITHOUT);
        }
        frames_alike = 0;
    }
    if (quick_alike && frame - last_quick_alike > RECORD_FOR) {
        quick_alike = false;
        first.clear();
        first.shrink_to_fit();
        recorded = 0;
        if (recording) {
            recording = false;
            phase = Phase::Idle;
        }
    }
}

} // namespace VideoCore
