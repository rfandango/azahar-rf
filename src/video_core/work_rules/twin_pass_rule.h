// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <chrono>
#include <vector>
#include "video_core/work_rules/draw_record.h"

namespace VideoCore {

/**
 * The second pass of a scene that is drawn twice a part of a pixel apart.
 *
 * Some games smooth edges by drawing their scene twice into the same target, a part of a
 * pixel apart, and blending the two images (the "MT Framework" games do while the 3D effect
 * is off; in Monster Hunter 4 Ultimate it is a quarter of a frame's GPU time). Above the
 * console's resolution the host's pixels are smaller than that shift, and the second image
 * only costs. At the console's it smooths the edges, for the price of the scene once more: left
 * out there too, so that a slow GPU has the scene to draw once at every resolution. So this
 * pattern is watched for: a run of draws to a target, a quad that copies
 * the target elsewhere, a second run to the target, and a quad that blends the copy over it.
 * Where it is seen the draws of the second run are compared with those of the first, place by
 * place. A twin goes by the same registers, the same vertex shader and the same float
 * uniforms but for a slight difference in a few of them (the shifted projection). After some
 * frames in which every draw of the second run was a twin, twins are voted out as they come,
 * and if all of them were, the blending as well: the target then holds the first image, which
 * is what blending two images that are all but the same comes to. A draw that is no twin is
 * drawn, and all after it.
 *
 * Votes Redundant. Off with KeepSecondScenePass; ExplainScenePasses has it say why a draw of
 * a second pass was no twin.
 */
class TwinPassRule final : public WorkRule {
public:
    const char* Name() const override {
        return "scene drawn twice";
    }
    const char* Why() const override {
        return "the twin of a draw of the scene's first pass";
    }
    Vote OnDraw(DrawFacts& facts) override;
    Vote OnFill(const FillFacts& facts) override;
    Vote OnTransfer(const TransferFacts& facts) override;

private:
    enum class Phase : u8 {
        Idle,
        /// A run of draws to one target is under way
        First,
        /// The quad that copies the target elsewhere came after it
        Copied,
        /// The second run to the target is under way
        Second,
    };
    static constexpr u32 MIN_DRAWS = 4;
    static constexpr u32 MAX_DRAWS = 1024;
    static constexpr u32 FRAMES_ALIKE = 4;
    static constexpr u32 MAX_DIFFERING = 12;
    /// How far a float uniform may be off, as a share of the largest value in the registers
    /// around it (the rows of its matrix)
    static constexpr float MAX_SHIFT = 0.02f;

    bool LeavesOut(DrawFacts& facts);
    /// A draw of the first pass, with where the lights that are on are and where they point, as
    /// far as its fragments go by that (nothing else of the draw's is in the camera's space, which
    /// is not quite the same)
    struct FirstDraw {
        DrawRecord draw;
        u32 num_lights{};
        std::array<std::array<float, 6>, 8> lights{};

        FirstDraw(DrawFacts& facts, bool clears);
    };
    static u32 LightsOf(DrawFacts& facts, std::array<std::array<float, 6>, 8>& lights);
    /// How far the lights are off, as CompareUniforms tells it of the uniforms
    static float LightShift(const FirstDraw& first, DrawFacts& facts);
    bool IsTwinOf(const FirstDraw& first, bool clears, DrawFacts& facts);
    void ExplainNoTwin(bool clears, DrawFacts& facts);
    /// A fill, a transfer or a copy is carried out: what runs of draws were under way are over
    void EndRuns();

    Phase phase{};
    /// The colour buffer of the run of draws under way, and how many it has
    PAddr run_target{};
    u32 run_draws{};
    /// Of the two passes; where the first image is kept; the draws of the first pass
    PAddr target{};
    PAddr keep{};
    u32 first_draws{};
    /// The place in the second pass
    u32 at{};
    /// The target the bare pattern was seen at: runs to it are recorded
    PAddr watched_target{};
    /// The first pass of this frame was recorded; every draw of the second so far was a
    /// twin; twins are left out as they come; how many were
    bool compared{};
    bool twins{};
    bool leaving_out{};
    u32 left_out_now{};
    /// Frames in a row in which the whole second pass was twins
    u32 frames_alike{};
    std::vector<FirstDraw> first;
    /// The registers of the first few of them (of all with ExplainScenePasses), and how often and
    /// when last it was said why a draw was no twin
    static constexpr std::size_t KEPT_REGISTERS = 4;
    std::vector<std::vector<u32>> first_regs;
    u32 explained{};
    std::chrono::steady_clock::time_point last_explained{};
    /// For the log: the most float uniforms that differed, and by how much
    u32 most_differing{};
    float largest_shift{};
};

} // namespace VideoCore
