// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "common/common_types.h"
#include "common/vector_math.h"
#include "video_core/pica_types.h"

namespace Pica {
struct DisplayTransferConfig;
struct FramebufferConfig;
struct MemoryFillConfig;
struct RegsInternal;
} // namespace Pica

/**
 * Work a game sends to the GPU that need not be carried out, and the rules that find it.
 *
 * A game in stereoscopic mode renders the top screen twice, and with one eye on display one
 * of the two is for nothing; games also render things twice for other reasons. No one way of
 * telling covers all games: they keep the eyes in one buffer or in two, share a depth buffer
 * between them or not, clear with fills, with a quad or with a box around the scene, pass one
 * eye's image through the other's buffer, draw the bottom screen over what an eye left behind.
 * So there are several rules, each for the cases it can be sure of, and this is what they
 * have in common, made so that a rule can be added for a game none of them covers without
 * the others losing what they cover:
 *
 *  - A rule is a class of its own behind WorkRule, with its own state, and is added in one
 *    place (GPU's constructor). Nothing else knows the rules by name.
 *  - Every rule is told of every draw, fill, transfer and copy the game sends, whatever the
 *    other rules make of it, and votes. A piece of work is left out if any rule votes for
 *    that. So rules do not depend on the order they are asked in, and what a rule covers
 *    does not change when another is added or switched off.
 *  - A vote says why: Unseen (nobody gets to see the result; what the work would have
 *    written is stale from then on) or Redundant (the result is there already; memory is as
 *    if the work had been carried out). A rule keeps its own books as if work that another
 *    rule left out had been carried out, which holds for Redundant by what that means. The
 *    watch over stale memory, for the case that a guess of Unseen was wrong, is kept by the
 *    one rule that votes Unseen (FrameLiveness); a second rule of that kind has to report
 *    what it leaves stale to that watch, which is not provided for yet.
 *  - What rules want to know about a draw comes from DrawFacts. What costs something there
 *    (whether the triangles cover the viewport, which shades the vertices; a hash over the
 *    registers) is worked out when first asked for and kept for the rules that ask after,
 *    and a rule asks only once the cheap facts leave the question open. A rule that is not
 *    in the middle of something it follows is to return at once.
 *  - What each rule leaves out and what it costs is counted here in the same way for all
 *    and written to the log ("Work rules: ..."), so that coverage can be checked game by
 *    game after a change, and a rule that has become costly shows.
 *
 * Adding a rule: find out what the game does (a frame dump, or the frame in outline that
 * FrameLiveness writes when a transfer is left out but none of the draws before it), write
 * the class, add it in GPU's constructor, add the game to the table the coverage check goes
 * by, and run that check over all games in the table.
 *
 * Not one of these rules: the right-eye skip the emulator came with (RightEyeDisabler), which
 * goes by the order of command lists and transfers and drops whole lists. It is kept for
 * comparison only, behind a switch that turns FrameLiveness off when it turns that on.
 */
namespace VideoCore {

/// How the texture coordinates of a draw go with the rows of its target, for each set of
/// coordinates where that is a straight line: v = scale * row + offset, the row counted
/// from the bottom of the target as the viewport counts it
struct DrawTextureRows {
    std::array<bool, 3> known{};
    std::array<f32, 3> scale{};
    std::array<f32, 3> offset{};
};

/// What a rule makes of a piece of work
enum class Vote : u8 {
    /// No objection to it being carried out
    Carry,
    /// Nobody gets to see its result: to be left out; what it would have written is stale
    Unseen,
    /// Its result is there already: to be left out; memory is as if it had been carried out
    Redundant,
};

/// The draw that is about to be made. The registers are those of the draw.
class DrawFacts {
public:
    using FloatUniforms = std::array<Common::Vec4<Pica::f24>, 96>;
    using UniformSet = std::array<u64, 2>;

    virtual const Pica::RegsInternal& Regs() const = 0;
    /// A triangle given vertex by vertex, which is with the rasterizer already: it is drawn
    /// whatever the votes, and what is asked about triangles below is not known of it
    virtual bool IsImmediate() const = 0;

    /// By the registers alone: every fragment is written, and nothing of what a pixel held
    /// goes into its colour; into its depth and stencil
    virtual bool MayOverwriteColour() = 0;
    virtual bool MayOverwriteDepth() = 0;
    /// The same for the depth of each pixel alone, whatever becomes of its stencil (a
    /// backdrop drawn with the depth test at always resets depth this way)
    virtual bool MayOverwriteDepthValues() = 0;
    /// Its triangles cover the middle of every pixel of the console in its viewport; all of
    /// the viewport up to its edges. Shades the vertices of the draw to find out: to be
    /// asked after what the registers alone tell. Of a mesh (more than a few dozen vertices)
    /// the first is found out only where it would reset the depth buffer, and not for a
    /// while again once it turned out not to cover; the second is always false of one.
    virtual bool CoversViewport() = 0;
    virtual bool CoversViewportExactly() = 0;
    /// Its viewport is as wide as its target, so that what covers the viewport covers rows
    virtual bool ViewportSpansTarget() = 0;
    /// How its texture coordinates go with the rows of its target. False if in no known way.
    virtual bool TextureRows(DrawTextureRows& rows) = 0;

    /// What makes a draw the same draw as another: a hash over the registers it goes by and
    /// its vertex shader. For a clear only over what the fragments go by.
    virtual u64 StateHash(bool as_clear) = 0;
    /// The stretches of registers [from, to) that hash goes over
    virtual void ForStateRegisters(bool as_clear,
                                   const std::function<void(u32, u32)>& stretch) const = 0;
    /// The float uniforms of the vertex shader as they are now, and a bit for each of them
    /// that its program can read
    virtual const FloatUniforms& Uniforms() const = 0;
    virtual const UniformSet& UniformsRead() = 0;

protected:
    ~DrawFacts() = default;
};

struct FillFacts {
    PAddr start;
    PAddr end;
    const Pica::MemoryFillConfig& config;
};

/// A display transfer or a texture copy, with what it reads and writes as memory
struct TransferFacts {
    PAddr source;
    PAddr source_end;
    PAddr dest;
    PAddr dest_end;
    const Pica::DisplayTransferConfig& config;
};

/// The game swapped the buffers of the top screen
struct FrameFacts {
    const Pica::FramebufferConfig& top_screen;
};

class WorkRule {
public:
    virtual ~WorkRule() = default;

    /// What the rule is called in the log, and why it leaves a piece of work out, as it is
    /// put after "left out: " in a frame dump
    virtual const char* Name() const = 0;
    virtual const char* Why() const = 0;

    virtual Vote OnDraw(DrawFacts& facts) = 0;
    virtual Vote OnFill([[maybe_unused]] const FillFacts& facts) {
        return Vote::Carry;
    }
    virtual Vote OnTransfer([[maybe_unused]] const TransferFacts& facts) {
        return Vote::Carry;
    }
    virtual void OnFrameEnd([[maybe_unused]] const FrameFacts& facts) {}
};

/// The rules, asked one after the other
class WorkRules {
public:
    WorkRules();
    ~WorkRules();

    /// Returns the rule, which lives as long as this does
    template <typename Rule>
    Rule* Add(std::unique_ptr<Rule> rule) {
        Rule* added = rule.get();
        AddRule(std::move(rule));
        return added;
    }

    /// Whether the draw, fill, transfer or copy is to be left out. Each is to be asked about
    /// once, as it comes, also when it is carried out in any case.
    bool LeavesOutDraw(DrawFacts& facts);
    bool LeavesOutFill(const FillFacts& facts);
    bool LeavesOutTransfer(const TransferFacts& facts);
    void EndFrame(const FrameFacts& facts);

    /// Why the last piece of work asked about is left out (WorkRule::Why of the first rule
    /// that voted for it)
    const char* LastWhy() const {
        return last_why;
    }

    /// For a report made from another thread: the draws each rule left out since this was
    /// last asked, as "; 1.5 draws a vblank left out by 'name'" for each that left any out
    std::string TakeLeftOutText(u32 vblanks);

private:
    struct Counts {
        u32 draws{};
        u32 fills{};
        u32 transfers{};
    };
    struct Entry {
        std::unique_ptr<WorkRule> rule;
        /// Left out in the frame under way, and in the frames since the last report
        Counts frame;
        Counts reported;
        /// For TakeLeftOutText
        std::atomic<u64> draws_taken{};
        /// What its OnDraw took in the frames that were timed, and how many calls that was
        u64 timed_ns{};
        u64 timed_calls{};
    };

    void AddRule(std::unique_ptr<WorkRule> rule);
    template <typename Ask>
    Vote AskAll(u32 Counts::*kind, Ask&& ask);

    std::vector<std::unique_ptr<Entry>> entries;
    const char* last_why = "";
    /// Work the game sent and work left out by any rule, in the frames since the last report
    Counts seen;
    Counts left_out;
    u32 frames{};
    u32 frames_reported{};
    /// Whether the calls of the frame under way are timed: one frame in 64 is
    bool timed{};
    /// What reading the clock before and after a call takes by itself, which is taken off
    /// what is measured (on some devices more than an idle rule takes)
    u64 clock_ns{};
};

} // namespace VideoCore
