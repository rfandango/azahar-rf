// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <bit>
#include <chrono>
#include <string>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/trace_markers.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/used_of_light.h"
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

TwinPassRule::FirstDraw::FirstDraw(DrawFacts& facts, bool clears) : draw{facts, clears} {
    if (!clears) {
        draw.state = facts.StateHashButLightVectors();
        num_lights = LightsOf(facts, lights);
    }
}

u32 TwinPassRule::LightsOf(DrawFacts& facts, std::array<std::array<float, 6>, 8>& lights) {
    const auto& lighting = facts.Regs().lighting;
    if (lighting.disable) {
        return 0;
    }
    const u32 count = lighting.max_light_index + 1;
    for (u32 index = 0; index < count; index++) {
        const auto& light = lighting.light[lighting.light_enable.GetNum(index)];
        const Pica::UsedOfLight used = Pica::UsedOf(lighting, index);
        lights[index] = {};
        if (used.vector) {
            lights[index][0] = Pica::f16::FromRaw(light.x).ToFloat32();
            lights[index][1] = Pica::f16::FromRaw(light.y).ToFloat32();
            lights[index][2] = Pica::f16::FromRaw(light.z).ToFloat32();
        }
        if (used.spot) {
            lights[index][3] = static_cast<float>(light.spot_x) / 2047.f;
            lights[index][4] = static_cast<float>(light.spot_y) / 2047.f;
            lights[index][5] = static_cast<float>(light.spot_z) / 2047.f;
        }
    }
    return count;
}

float TwinPassRule::LightShift(const FirstDraw& record, DrawFacts& facts) {
    std::array<std::array<float, 6>, 8> lights;
    const u32 count = LightsOf(facts, lights);
    if (count != record.num_lights) {
        return std::numeric_limits<float>::infinity();
    }
    float largest = 0.f;
    for (u32 index = 0; index < count; index++) {
        // Where it is, and where it points
        for (u32 part = 0; part < 6; part += 3) {
            float scale = 1e-4f;
            float apart = 0.f;
            for (u32 i = part; i < part + 3; i++) {
                scale = std::max(scale, std::abs(record.lights[index][i]));
                apart = std::max(apart, std::abs(lights[index][i] - record.lights[index][i]));
                if (!(lights[index][i] == lights[index][i]) ||
                    !(record.lights[index][i] == record.lights[index][i])) {
                    return std::numeric_limits<float>::infinity();
                }
            }
            largest = std::max(largest, apart / scale);
        }
    }
    return largest;
}

bool TwinPassRule::IsTwinOf(const FirstDraw& first_draw, bool clears, DrawFacts& facts) {
    const DrawRecord& record = first_draw.draw;
    if (record.clear != clears) {
        return false;
    }
    if (clears) {
        return record.state == facts.StateHash(true);
    }
    if (record.state != facts.StateHashButLightVectors() ||
        !(LightShift(first_draw, facts) <= MAX_SHIFT)) {
        return false;
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
    // One of them a second is enough
    const auto now_is = std::chrono::steady_clock::now();
    if (explained >= 300 || (explained != 0 && now_is - last_explained < std::chrono::seconds{1})) {
        return;
    }
    explained++;
    last_explained = now_is;
    const Pica::RegsInternal& regs = facts.Regs();
    std::string why;
    if (at >= first.size()) {
        why = fmt::format("the first pass has only {} draws on record", first.size());
    } else {
        const DrawRecord& record = first[at].draw;
        const u64 state = clears ? facts.StateHash(true) : facts.StateHashButLightVectors();
        if (record.clear != clears) {
            why = fmt::format("it {} all of its viewport, its counterpart {}",
                              clears ? "overwrites" : "does not overwrite",
                              record.clear ? "does" : "does not");
        } else if (record.state != state) {
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
                why += " none of those that are compared now, so the vertex shader's program, the "
                       "fog table or which registers are in use";
            }
        } else if (const float shift = LightShift(first[at], facts); !(shift <= MAX_SHIFT)) {
            why = fmt::format("its lights are off by {:.4f}", shift);
            std::array<std::array<float, 6>, 8> now_lights;
            const u32 count = LightsOf(facts, now_lights);
            const auto& lighting = regs.lighting;
            why += fmt::format(" (configuration {:08X} {:08X}, inputs {:08X}, {} lights, {} before)",
                               regs.reg_array[0x1C3], lighting.config1.raw,
                               regs.reg_array[0x1D1], count, first[at].num_lights);
            for (u32 i = 0; i < std::min(count, first[at].num_lights); i++) {
                const u32 num = lighting.light_enable.GetNum(i);
                const auto& light = lighting.light[num];
                const auto& was = first[at].lights[i];
                const auto& is = now_lights[i];
                why += fmt::format(
                    "; light {} {}{} colours {:08X} {:08X} {:08X} {:08X}: ({:.4f} {:.4f} {:.4f}) "
                    "spot ({:.4f} {:.4f} {:.4f}) -> ({:.4f} {:.4f} {:.4f}) spot ({:.4f} {:.4f} "
                    "{:.4f})",
                    num, light.config.directional ? "directional" : "with a place",
                    lighting.IsSpotAttenDisabled(num) ? "" : ", a spot",
                    regs.reg_array[0x140 + num * 0x10], regs.reg_array[0x141 + num * 0x10],
                    regs.reg_array[0x142 + num * 0x10], regs.reg_array[0x143 + num * 0x10], was[0],
                    was[1], was[2], was[3], was[4], was[5], is[0], is[1], is[2], is[3], is[4],
                    is[5]);
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
                             "second pass is left out from here on.",
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
                // (Why a pass that was left out is not any more is always said)
                if (twins && (frames_alike >= FRAMES_ALIKE ||
                              (Common::TraceMarkers::Experiments3() &
                               Common::TraceMarkers::ExplainScenePasses) != 0)) {
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
            if (first_regs.size() < KEPT_REGISTERS ||
                (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::ExplainScenePasses) !=
                    0) {
                first_regs.emplace_back(regs.reg_array.begin(), regs.reg_array.end());
            }
        }
        run_draws++;
    }
    return false;
}

} // namespace VideoCore
