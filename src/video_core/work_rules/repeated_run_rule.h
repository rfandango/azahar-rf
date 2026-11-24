// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <string>
#include <vector>
#include "video_core/work_rules/draw_record.h"
#include "video_core/work_rules/eye_buffers.h"

namespace VideoCore {

/**
 * The right eye's fills and draws where they repeat the left eye's, whatever they go to.
 *
 * RepeatedEyeRule follows a run of draws to one target from a clear that is drawn. A game may
 * as well clear with fills and draw an image by way of several buffers: Metal Gear Solid: Snake
 * Eater 3D fills colour and depth, draws the scene, makes smaller images of it in another buffer
 * and draws with those, for each eye, 218 draws and 6 fills in its opening scene. With the depth
 * of the 3D effect at nothing the second run is the first once more, register for register.
 *
 * A run is all that comes between a transfer and the next one: it begins with a fill that is
 * the first thing done after a transfer, and ends with the transfer of its image to a buffer
 * the top screen showed the left eye. What follows it, from a fill that is the same as its
 * first, to a transfer of the same memory to a buffer it showed the right eye, is compared with
 * it piece by piece. Is the one the other, in as many frames as it takes to be believed, such
 * fills and draws are voted out as they come; one that is not the one at its place is carried
 * out, and all after it.
 *
 * Why memory is afterwards as if the second run had been made: it would have made the same of
 * what the first made from. That holds of all it fills or clears. A buffer the run draws to and
 * does not clear holds what the run before left there, and what is drawn may go by that. But it
 * is the same buffer for both eyes, and with the 3D effect on, what one eye left in it is
 * nothing the other could use: games keep nothing in such a buffer from one run to the next,
 * they put things down in it and take them up again (those smaller images). What is not compared
 * is what the game's memory holds at the addresses the draws name, as in RepeatedEyeRule.
 *
 * Following every draw of every game this way would cost more than it gains in the games whose
 * eyes are no repetition, or that draw only one. So a run is first followed by a hash over a few
 * registers of each draw; only where the second run had the same is a run put on record.
 *
 * Votes Redundant. Off with KeepRepeatedPass; ExplainScenePasses has it say more on why a
 * second run was no repetition.
 */
class RepeatedRunRule final : public WorkRule {
public:
    const char* Name() const override {
        return "right eye's run repeats";
    }
    const char* Why() const override {
        return "it repeats a piece of the run before, which made the same image for the other "
               "eye";
    }
    Vote OnDraw(DrawFacts& facts) override;
    Vote OnFill(const FillFacts& facts) override;
    Vote OnTransfer(const TransferFacts& facts) override;
    void OnFrameEnd(const FrameFacts& facts) override;

private:
    enum class Phase : u8 {
        Idle,
        /// A run begun with a fill is under way
        First,
        /// Its image was transferred to a buffer of the left eye
        Closed,
        /// A second run begun with the same fill is under way
        Second,
    };
    /// A fill: where, with what, and how wide that is
    struct Fill {
        PAddr start{};
        PAddr end{};
        u32 value{};
        u32 control{};
        bool operator==(const Fill&) const = default;
    };
    struct Piece {
        bool is_fill{};
        Fill fill;
        DrawRecord draw;
        /// With ExplainScenePasses: the registers the draw went by
        std::vector<u32> registers;
    };
    struct Memory {
        PAddr start{};
        PAddr end{};
        bool operator==(const Memory&) const = default;
    };
    static constexpr u32 MAX_PIECES = 1024;
    static constexpr std::size_t MAX_MEMORY = 24;
    static constexpr std::size_t MAX_SAMPLED = 256;
    /// Frames in which the whole second run has to be the first until it is left out, at first
    /// and at most: four times as many each time that turned out wrong
    static constexpr u32 FRAMES_ALIKE = 4;
    static constexpr u32 MOST_FRAMES_ALIKE = 256;
    /// Frames without a repetition, after which what was learned is forgotten (a frame ends
    /// where the game's swap is taken up, which may be between the two runs)
    static constexpr u32 FRAMES_WITHOUT = 3;
    /// Frames for which runs are put on record after two were alike by their hash
    static constexpr u32 RECORD_FOR = 128;
    /// As in RepeatedEyeRule
    static constexpr float MAX_SAME_SHIFT = 1e-6f;

    void Begin(const Fill& fill);
    void BeginSecond();
    /// The next piece of the second run: whether it is to be left out
    bool Repeats(bool same, const std::string& difference);
    void Note(PAddr start, PAddr end);
    /// The memory a draw of the first run draws to and samples
    void NoteDraw(const Pica::RegsInternal& regs);
    bool Touches(PAddr start, PAddr end) const;
    /// The second run is none, or no longer
    void End(const char* why);

    Phase phase{};
    /// Nothing was drawn or filled since the last transfer
    bool fresh{true};
    /// The first run: its first fill; a hash over its pieces and how many there were
    Fill first_fill;
    u64 quick{};
    u32 count{};
    /// Whether it is put on record, and the record: the pieces, the memory they fill and draw
    /// to, the textures they sample
    bool recording{};
    std::vector<Piece> first;
    u32 recorded{};
    std::vector<Memory> touched;
    std::vector<PAddr> sampled;
    /// Bits for the addresses in `sampled`, to find most draws' textures without searching it
    std::array<u64, 16> sampled_bits{};
    /// The targets of the draw noted last
    std::array<u32, 4> last_targets{};
    bool first_draw_recorded{};
    bool too_much{};
    /// Where the first run's image was transferred to
    Memory went_to;
    /// The second run
    u64 second_quick{};
    u32 at{};
    bool repeats{};
    bool leaving_out{};
    u32 left_out_now{};

    EyeBuffers eyes;
    u32 frame{};
    /// Frames in a row with a second run that was the first; the last frame with one by the
    /// record and by the hash; how many it takes to be believed
    u32 frames_alike{};
    u32 last_alike{};
    u32 last_quick_alike{};
    bool quick_alike{};
    u32 frames_needed{FRAMES_ALIKE};
    /// Lines written to say why a second run was no repetition
    u32 explained{};
};

} // namespace VideoCore
