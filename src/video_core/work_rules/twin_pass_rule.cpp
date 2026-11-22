// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <bit>
#include <string>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/work_rules/twin_pass_rule.h"

namespace VideoCore {

Vote TwinPassRule::OnDraw(DrawFacts& facts) {
    if (facts.IsImmediate()) {
        return Vote::Carry;
    }
    return LeavesOut(facts) ? Vote::Redundant : Vote::Carry;
}

Vote TwinPassRule::OnFill([[maybe_unused]] const FillFacts& facts) {
    EndRuns();
    return Vote::Carry;
}

Vote TwinPassRule::OnTransfer([[maybe_unused]] const TransferFacts& facts) {
    EndRuns();
    return Vote::Carry;
}

void TwinPassRule::EndRuns() {
    if (phase == Phase::Copied || phase == Phase::Second) {
        // Not the pattern: nothing but draws lies between the copy and the blending
        if (frames_alike >= FRAMES_ALIKE) {
            LOG_INFO(HW_GPU,
                     "The scene drawn twice at {:08X} is no longer: a fill or a transfer "
                     "came within its second pass ({} draws of it were left out)",
                     target, left_out_now);
        }
        frames_alike = 0;
    }
    phase = Phase::Idle;
    run_target = 0;
    run_draws = 0;
}

bool TwinPassRule::IsTwinOf(const DrawRecord& record, bool clears, DrawFacts& facts) {
    if (record.clear != clears || record.state != facts.StateHash(clears)) {
        return false;
    }
    if (clears) {
        return true;
    }
    const UniformDifference difference = CompareUniforms(record, facts, MAX_DIFFERING, MAX_SHIFT);
    if (difference.differing > MAX_DIFFERING || !(difference.largest_shift <= MAX_SHIFT)) {
        return false;
    }
    largest_shift = std::max(largest_shift, difference.largest_shift);
    most_differing = std::max(most_differing, difference.differing);
    return true;
}

void TwinPassRule::ExplainNoTwin(bool clears, DrawFacts& facts) {
    if (explained >= 12) {
        return;
    }
    explained++;
    const Pica::RegsInternal& regs = facts.Regs();
    std::string why;
    if (at >= first.size()) {
        why = fmt::format("the first pass has only {} draws on record", first.size());
    } else {
        const DrawRecord& record = first[at];
        if (record.clear != clears) {
            why = fmt::format("it {} all of its viewport, its counterpart {}",
                              clears ? "overwrites" : "does not overwrite",
                              record.clear ? "does" : "does not");
        } else if (record.state != facts.StateHash(clears)) {
            why = "registers differ:";
            u32 listed = 0;
            if (at < first_regs.size()) {
                const std::vector<u32>& before = first_regs[at];
                const auto& now = regs.reg_array;
                facts.ForStateRegisters(clears, [&](u32 from, u32 to) {
                    for (u32 i = from; i < to && i < before.size() && listed < 24; i++) {
                        if (before[i] != now[i]) {
                            why += fmt::format(" {:03X}: {:08X}->{:08X}", i, before[i], now[i]);
                            listed++;
                        }
                    }
                });
            }
            if (listed == 0) {
                why += " none of those that are compared now, so the vertex shader's program "
                       "or which registers are in use";
            }
        } else {
            const UniformDifference difference =
                CompareUniforms(record, facts, 96, std::numeric_limits<float>::infinity());
            const DrawFacts::UniformSet& read = facts.UniformsRead();
            const DrawFacts::FloatUniforms& now = facts.Uniforms();
            std::string which;
            u32 listed = 0;
            for (u32 i = 0; i < now.size() && listed < 6; i++) {
                if ((read[i / 64] >> (i % 64) & 1) == 0 ||
                    std::memcmp(&now[i], &record.uniforms[i], sizeof(now[i])) == 0) {
                    continue;
                }
                listed++;
                which +=
                    fmt::format(" c{} ({:.4f} {:.4f} {:.4f} {:.4f} -> {:.4f} {:.4f} {:.4f} {:.4f})",
                                i, record.uniforms[i].x.ToFloat32(),
                                record.uniforms[i].y.ToFloat32(), record.uniforms[i].z.ToFloat32(),
                                record.uniforms[i].w.ToFloat32(), now[i].x.ToFloat32(),
                                now[i].y.ToFloat32(), now[i].z.ToFloat32(), now[i].w.ToFloat32());
            }
            why = fmt::format("of the {} float uniforms its shader reads {} differ, by {:.4f} "
                              "of what is around them at the most:{}",
                              std::popcount(read[0]) + std::popcount(read[1]), difference.differing,
                              difference.largest_shift, which);
        }
    }
    LOG_INFO(HW_GPU, "Second pass at {:08X}: draw {} of {} ({} vertices) is no twin: {}", target,
             at, first_draws, static_cast<u32>(regs.pipeline.num_vertices), why);
}

bool TwinPassRule::LeavesOut(DrawFacts& facts) {
    using BlendFactor = Pica::FramebufferRegs::BlendFactor;
    const Pica::RegsInternal& regs = facts.Regs();
    const PAddr color = regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress();
    const auto& pipeline = regs.pipeline;
    const auto& texturing = regs.texturing;
    const auto& merger = regs.framebuffer.output_merger;
    // A quad that samples one texture: what copies an image, or blends one over another
    const bool quad =
        pipeline.num_vertices == 4 && pipeline.use_gs == Pica::PipelineRegs::UseGS::No;
    const PAddr sampled =
        quad && texturing.main_config.texture0_enable ? texturing.texture0.GetPhysicalAddress() : 0;
    const bool replaces =
        !merger.alphablend_enable || (merger.alpha_blending.factor_source_rgb == BlendFactor::One &&
                                      merger.alpha_blending.factor_dest_rgb == BlendFactor::Zero);
    // Whether the draw overwrites all of its viewport (a clear done with a quad). Asked only
    // where a draw is recorded or compared: it may shade the vertices.
    const auto clears = [&facts] { return facts.MayOverwriteColour() && facts.CoversViewport(); };

    if (phase == Phase::Copied) {
        if (color == target) {
            phase = Phase::Second;
            at = 0;
            left_out_now = 0;
            twins = compared;
            leaving_out = compared && frames_alike >= FRAMES_ALIKE &&
                          Settings::values.resolution_factor.GetValue() != 1 &&
                          (Common::TraceMarkers::Experiments3() &
                           Common::TraceMarkers::KeepSecondScenePass) == 0;
        } else {
            phase = Phase::Idle;
        }
    }

    if (phase == Phase::Second) {
        if (color == target && quad && sampled == keep && sampled != 0 && !replaces) {
            // The quad that blends the first image over the second
            const bool whole = at == first_draws;
            const u32 apart = at > first_draws ? at - first_draws : first_draws - at;
            if (apart <= first_draws / 8) {
                // The bare pattern: worth comparing draw by draw from the next frame on
                if (watched_target != target) {
                    LOG_INFO(HW_GPU,
                             "A scene that may be drawn twice: runs of {} and {} draws to {:08X} "
                             "with a copy to {:08X} between them and a blend of it after",
                             first_draws, at, target, keep);
                }
                watched_target = target;
            }
            if (compared) {
                const bool alike = twins && whole;
                if (alike && ++frames_alike == FRAMES_ALIKE) {
                    LOG_INFO(HW_GPU,
                             "The game draws its scene twice at {:08X} and blends the two images "
                             "(kept at {:08X}): {} draws each, every one of the second pass like "
                             "the one at its place in the first but for {} float uniforms at "
                             "the most, off by {:.4f} of what is around them at the most. The "
                             "second pass is left out from here on where the resolution is "
                             "above the console's.",
                             target, keep, first_draws, most_differing, largest_shift);
                } else if (!alike) {
                    if (frames_alike >= FRAMES_ALIKE) {
                        LOG_INFO(HW_GPU,
                                 "The scene drawn twice at {:08X}: the second pass was not like "
                                 "the first this time ({} draws against {}, {} left out)",
                                 target, at, first_draws, left_out_now);
                    }
                    frames_alike = 0;
                }
                frames_alike = std::min(frames_alike, 1u << 20);
            }
            const bool leave_out = leaving_out && whole && left_out_now == at;
            phase = Phase::Idle;
            run_target = color;
            run_draws = 1;
            return leave_out;
        }
        if (color == target) {
            // A draw of the second pass
            bool is_twin = false;
            const bool as_clear = twins ? clears() : false;
            if (twins && at < first.size()) {
                is_twin = IsTwinOf(first[at], as_clear, facts);
            }
            if (!is_twin) {
                if (twins && (Common::TraceMarkers::Experiments3() &
                              Common::TraceMarkers::ExplainScenePasses) != 0) {
                    ExplainNoTwin(as_clear, facts);
                }
                twins = false;
                leaving_out = false;
            }
            at++;
            if (is_twin && leaving_out) {
                left_out_now++;
                return true;
            }
            return false;
        }
        // The second run ended without the blending: not the pattern
        if (frames_alike >= FRAMES_ALIKE) {
            LOG_INFO(HW_GPU,
                     "The scene drawn twice at {:08X} is no longer: its second pass "
                     "ended without the blending ({} draws of it were left out)",
                     target, left_out_now);
        }
        frames_alike = 0;
        phase = Phase::Idle;
        run_target = 0;
    }

    if (color != run_target) {
        if (phase == Phase::First && quad && sampled == run_target && sampled != 0 && replaces &&
            run_draws >= MIN_DRAWS) {
            // The quad that copies what the run drew to somewhere else
            target = run_target;
            keep = color;
            first_draws = run_draws;
            compared = first.size() == run_draws;
            phase = Phase::Copied;
            run_target = color;
            run_draws = 1;
            return false;
        }
        phase = Phase::First;
        run_target = color;
        run_draws = 0;
        first.clear();
        first_regs.clear();
    }
    if (phase == Phase::First) {
        if (color == watched_target && first.size() == run_draws && run_draws < MAX_DRAWS) {
            first.emplace_back(facts, clears());
            if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::ExplainScenePasses) !=
                0) {
                first_regs.emplace_back(regs.reg_array.begin(), regs.reg_array.end());
            }
        }
        run_draws++;
    }
    return false;
}

} // namespace VideoCore
