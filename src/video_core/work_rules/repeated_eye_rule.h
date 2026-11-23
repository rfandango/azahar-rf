// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <vector>
#include "video_core/work_rules/draw_record.h"

namespace VideoCore {

/**
 * The right eye's run of draws where it repeats the left eye's.
 *
 * A game in stereoscopic mode draws the top screen twice into one target, each time from a
 * clear, and has each image taken out of it by a display transfer. With the depth of the 3D
 * effect at nothing the two are the same image: the second run of draws repeats the first,
 * register for register and uniform for uniform. What a run of draws makes of a target it
 * has just cleared depends on nothing that was there before, so the second run leaves the
 * target as it finds it, and need not be made. (The Legend of Zelda: A Link Between Worlds:
 * 43 of the 138 draws of a frame, more than a fifth of its GPU time. Its bottom screen is
 * then drawn into the same buffer, blended over what is there, which is why following the
 * results to the screen cannot tell that this image is never seen: see FrameLiveness.)
 *
 * So this is watched for, within one frame: a draw that clears colour, depth and stencil in
 * all of the viewport, a run of draws to that target and viewport, a transfer that takes the
 * image to a buffer the top screen showed the left eye, and then the same again, ending in a
 * transfer to a buffer it showed the right eye. After some frames in which every draw of the
 * second run was the one at its place in the first, such draws are voted out as they come. A
 * draw that is not is drawn, and all after it. What is not compared is what the game's memory
 * holds at the addresses the draws name: their vertices and textures are taken not to change
 * between the two runs of one frame.
 *
 * Not covered: runs that begin with a fill and not with a clear that is drawn (the fill of
 * the second run would have to be left out before the run is known to repeat), and eyes that
 * are drawn to two targets.
 *
 * Votes Redundant. Off with KeepRepeatedPass; ExplainScenePasses has it say more on why a
 * second run was no repetition.
 */
class RepeatedEyeRule final : public WorkRule {
public:
    const char* Name() const override {
        return "right eye repeats left";
    }
    const char* Why() const override {
        return "it repeats a draw of the run before, which made the same image for the other eye";
    }
    Vote OnDraw(DrawFacts& facts) override;
    Vote OnFill(const FillFacts& facts) override;
    Vote OnTransfer(const TransferFacts& facts) override;
    void OnFrameEnd(const FrameFacts& facts) override;

private:
    enum class Phase : u8 {
        Idle,
        /// A run of draws to a target, begun with a clear, is under way and on record
        First,
        /// A transfer that reads the target came after it
        Closed,
        /// A second run to the target, begun with the same clear, is under way
        Second,
    };
    static constexpr u32 MAX_DRAWS = 1024;
    static constexpr u32 FRAMES_ALIKE = 4;
    static constexpr std::size_t MAX_SAMPLED = 256;

    bool LeavesOut(DrawFacts& facts);
    /// The pattern is contradicted: what was learned about it is forgotten
    void EndPass(const char* why);

    Phase phase{};
    /// The colour buffer of the run and where it ends; the viewport its clear set
    PAddr target{};
    PAddr target_end{};
    std::array<u32, 3> viewport{};
    /// The draws of the first run, and the textures they sample
    std::vector<DrawRecord> first;
    std::vector<PAddr> sampled;
    bool too_many_sampled{};
    /// Where the first run's image was transferred to
    PAddr first_went_to{};
    /// The buffers the top screen showed to the left eye lately, and those it showed to the
    /// right eye that were not also the left eye's (a game goes through two or three of each
    /// in turn)
    std::array<PAddr, 8> left_shown{};
    std::array<PAddr, 8> right_shown{};
    u32 left_next{};
    u32 right_next{};
    /// The place in the second run; whether every draw of it so far repeated the first;
    /// whether those that do are left out; how many were
    u32 at{};
    bool repeats{};
    bool leaving_out{};
    u32 left_out_now{};
    /// Frames in a row in which the whole second run repeated the first, and for which target
    u32 frames_alike{};
    PAddr alike_target{};
    /// Whether this frame had a second run that repeated the first
    bool completed{};
    /// Lines written to say why a second run was no repetition
    u32 explained{};
    /// And to say where an image went that is no buffer of an eye
    u32 explained_eyes{};
};

} // namespace VideoCore
