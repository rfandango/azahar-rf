// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <string>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/trace_markers.h"
#include "video_core/frame_state_dump.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/work_rules/repeated_eye_rule.h"

namespace VideoCore {

Vote RepeatedEyeRule::OnDraw(DrawFacts& facts) {
    if (facts.IsImmediate()) {
        // A triangle that is not on record and cannot be compared
        const PAddr color = facts.Regs().framebuffer.framebuffer.GetColorBufferPhysicalAddress();
        if (phase != Phase::Idle && color == target) {
            EndPass("a triangle given vertex by vertex was drawn to its target");
        }
        return Vote::Carry;
    }
    return LeavesOut(facts) ? Vote::Redundant : Vote::Carry;
}

void RepeatedEyeRule::EndPass(const char* why) {
    if (phase == Phase::Closed || phase == Phase::Second) {
        if (frames_alike >= FRAMES_ALIKE) {
            LOG_INFO(HW_GPU,
                     "The image drawn twice at {:08X} is no longer: {} ({} draws of the second "
                     "run were left out)",
                     target, why, left_out_now);
        }
        frames_alike = 0;
    }
    phase = Phase::Idle;
}

bool RepeatedEyeRule::LeavesOut(DrawFacts& facts) {
    const Pica::RegsInternal& regs = facts.Regs();
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const PAddr color = framebuffer.GetColorBufferPhysicalAddress();
    // The size of the viewport both ways and its corner
    const auto& array = regs.reg_array;
    const std::array<u32, 3> draw_viewport{array[0x41], array[0x43], array[0x68]};
    const bool to_target = color == target && draw_viewport == viewport;
    // Nothing is followed and this draw is no clear by its registers: nothing to do. (Whether
    // its triangles cover the viewport is asked only then, which may shade them.)
    const bool may_clear = facts.MayOverwriteColour() && facts.MayOverwriteDepth();
    if (phase == Phase::Idle && !may_clear) {
        return false;
    }
    const bool clears = may_clear && color != 0 && facts.CoversViewportExactly();

    if (phase == Phase::Closed) {
        if (clears && to_target) {
            phase = Phase::Second;
            at = 0;
            left_out_now = 0;
            repeats = true;
            leaving_out = frames_alike >= FRAMES_ALIKE && alike_target == target &&
                          (Common::TraceMarkers::Experiments3() &
                           Common::TraceMarkers::KeepRepeatedPass) == 0;
        } else {
            // Something else is drawn once the image was taken out: no second run of it. Not
            // a contradiction, a frame of one image goes this way.
            phase = Phase::Idle;
        }
    }

    if (phase == Phase::Second) {
        if (to_target) {
            const bool known = at < first.size();
            const bool same_state = repeats && known && first[at].state == facts.StateHash(false);
            u32 differing = 0;
            if (same_state) {
                differing = CompareUniforms(first[at], facts, 0, 0.f).differing;
            }
            const bool same = same_state && differing == 0;
            if (!same) {
                const u32 lines = (Common::TraceMarkers::Experiments3() &
                                   Common::TraceMarkers::ExplainScenePasses) != 0
                                      ? 12
                                      : 2;
                if (repeats && explained < lines) {
                    explained++;
                    LOG_INFO(HW_GPU,
                             "Second run at {:08X}: draw {} of {} ({} vertices) does not repeat "
                             "the first run: {}",
                             target, at, first.size(), static_cast<u32>(regs.pipeline.num_vertices),
                             !known ? std::string{"the first run had no more draws"}
                             : differing != 0
                                 ? std::string{"float uniforms its shader reads differ"}
                                 : std::string{"registers or the vertex shader differ"});
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
        EndPass("a draw to another target or viewport came within its second run");
    }

    const auto record = [&] {
        first.emplace_back(facts, false);
        for (const auto& texture : regs.texturing.GetTextures()) {
            if (!texture.enabled) {
                continue;
            }
            const PAddr address = texture.config.GetPhysicalAddress();
            if (std::find(sampled.begin(), sampled.end(), address) != sampled.end()) {
                continue;
            }
            if (sampled.size() < MAX_SAMPLED) {
                sampled.push_back(address);
            } else {
                too_many_sampled = true;
            }
        }
    };
    if (clears) {
        // A run begins
        phase = Phase::First;
        target = color;
        target_end =
            color + framebuffer.GetWidth() * framebuffer.GetHeight() *
                        Pica::FramebufferRegs::BytesPerColorPixel(framebuffer.color_format);
        viewport = draw_viewport;
        first.clear();
        sampled.clear();
        too_many_sampled = false;
        record();
        return false;
    }
    if (phase == Phase::First) {
        if (to_target && first.size() < MAX_DRAWS) {
            record();
        } else {
            phase = Phase::Idle;
        }
    }
    return false;
}

Vote RepeatedEyeRule::OnTransfer(const TransferFacts& facts) {
    if (phase == Phase::Idle) {
        return Vote::Carry;
    }
    if (facts.dest < target_end && target < facts.dest_end) {
        EndPass("a transfer wrote to its target");
        return Vote::Carry;
    }
    // What the draws sample has to be the same for both runs
    bool feeds = too_many_sampled;
    for (const PAddr address : sampled) {
        feeds = feeds || (address >= facts.dest && address < facts.dest_end);
    }
    if (feeds) {
        CITRA_FRAME_DUMP("  a run of {} draws from a clear is not followed: {}", first.size(),
                         too_many_sampled ? "it samples too many textures"
                                          : "this writes to a texture of its draws");
        EndPass("a transfer wrote to a texture of its draws");
        return Vote::Carry;
    }
    if (!(facts.source < target_end && target < facts.source_end)) {
        return Vote::Carry;
    }
    // The image is taken out of the target. The two runs that are the same image are those
    // of the two eyes: the first goes to a buffer the top screen shows to the left eye, the
    // second to one it shows to the right eye.
    const PAddr dest = facts.dest;
    const bool left_eyes = dest != 0 && std::ranges::find(left_shown, dest) != left_shown.end();
    const bool right_eyes =
        dest != 0 && !left_eyes && std::ranges::find(right_shown, dest) != right_shown.end();
    switch (phase) {
    case Phase::First:
        if (left_eyes) {
            CITRA_FRAME_DUMP("  a run of {} draws from a clear made the left eye's image",
                             first.size());
            phase = Phase::Closed;
            first_went_to = dest;
        } else {
            CITRA_FRAME_DUMP("  a run of {} draws from a clear, but not for a buffer of the "
                             "left eye",
                             first.size());
            phase = Phase::Idle;
        }
        break;
    case Phase::Second: {
        const bool to_right_eye = dest != first_went_to && right_eyes;
        const bool whole = repeats && at == first.size();
        CITRA_FRAME_DUMP("  the second run: {} draws against {}, {}, {} left out, {}for a buffer "
                         "of the right eye",
                         at, first.size(), repeats ? "all of them repeats" : "not all repeats",
                         left_out_now, to_right_eye ? "" : "not ");
        if (whole && to_right_eye) {
            if (alike_target != target) {
                alike_target = target;
                frames_alike = 0;
            }
            completed = true;
            if (++frames_alike == FRAMES_ALIKE) {
                LOG_INFO(HW_GPU,
                         "The game draws the image of the top screen twice at {:08X}, once for "
                         "each eye, and the two are the same: {} draws from a clear each, every "
                         "one of the second run with the registers and the uniforms of the one "
                         "at its place in the first. The second run is left out from here on.",
                         target, first.size());
            }
            frames_alike = std::min(frames_alike, 1u << 20);
            phase = Phase::Idle;
        } else {
            EndPass(!to_right_eye ? "its second image did not go to the right eye's buffer"
                                  : "its second run was not like the first");
        }
        break;
    }
    default:
        // Closed: the same image is taken out once more
        break;
    }
    return Vote::Carry;
}

Vote RepeatedEyeRule::OnFill(const FillFacts& facts) {
    if (phase != Phase::Idle && facts.start < target_end && target < facts.end) {
        EndPass("a fill of its target came between");
    }
    return Vote::Carry;
}

void RepeatedEyeRule::OnFrameEnd(const FrameFacts& facts) {
    if (!completed) {
        if (frames_alike >= FRAMES_ALIKE) {
            LOG_INFO(HW_GPU,
                     "The image drawn twice at {:08X} is no longer: a frame went by without it",
                     alike_target);
        }
        frames_alike = 0;
    }
    completed = false;
    // Runs are not paired across frames
    phase = Phase::Idle;

    // Which buffers each eye is shown
    const auto remember = [](std::array<PAddr, 8>& shown, u32& next, PAddr buffer) {
        if (buffer != 0 && std::ranges::find(shown, buffer) == shown.end()) {
            shown[next++ % shown.size()] = buffer;
        }
    };
    remember(left_shown, left_next, facts.top_screen.address_left1);
    remember(left_shown, left_next, facts.top_screen.address_left2);
    for (const PAddr right : {facts.top_screen.address_right1, facts.top_screen.address_right2}) {
        if (std::ranges::find(left_shown, right) == left_shown.end()) {
            remember(right_shown, right_next, right);
        }
    }
}

} // namespace VideoCore
