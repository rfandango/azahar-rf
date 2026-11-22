// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include "common/common_types.h"
#include "video_core/work_rules.h"

namespace VideoCore {

class GPU;

/// What a piece of memory is used as: plain bytes, or a surface with rows of a certain length
/// and pixels of a certain size. Games use the same memory for several surfaces in a frame,
/// and the pixels of one are not an image for another whose rows have another length.
namespace MemoryLayout {
enum Kind : u32 {
    Bytes = 0,   ///< Copies, fills and reads by the game
    Colour = 1,  ///< The colour buffer of a draw, or the source of a display transfer
    Depth = 2,   ///< The depth buffer of a draw
    Texture = 3, ///< What a draw samples
};
constexpr u32 Make(Kind kind, u32 width, u32 bits_per_pixel) {
    return (kind << 28) | (bits_per_pixel << 16) | width;
}
constexpr Kind KindOf(u32 layout) {
    return static_cast<Kind>(layout >> 28);
}
/// The length of a row and the size of a pixel
constexpr u32 PixelsOf(u32 layout) {
    return layout & 0x0FFFFFFF;
}
} // namespace MemoryLayout

/// A piece of guest memory and what it is used as
struct MemoryRange {
    PAddr start{};
    PAddr end{};
    u32 layout{};

    bool Empty() const {
        return start >= end;
    }
    /// Whether the two share memory
    bool Overlaps(const MemoryRange& other) const {
        return start < other.end && other.start < end;
    }
    /// Whether what is stored as one can be read as the other
    bool SameImage(const MemoryRange& other) const {
        using namespace MemoryLayout;
        if (layout == 0 || other.layout == 0) {
            return true;
        }
        const Kind kind = KindOf(layout);
        const Kind other_kind = KindOf(other.layout);
        if (kind == Texture || other_kind == Texture) {
            // A texture may be laid over colours with rows of another length and still be
            // of use, to average them for one. Over depth only where the rows line up.
            const Kind drawn_as = kind == Texture ? other_kind : kind;
            return drawn_as != Depth || PixelsOf(layout) == PixelsOf(other.layout);
        }
        return layout == other.layout;
    }
    bool operator==(const MemoryRange&) const = default;
};

/// Pieces of guest memory, kept sorted and merged per layout
class MemoryRanges {
public:
    bool Empty() const {
        return ranges.empty();
    }
    std::size_t Size() const {
        return ranges.size();
    }
    /// Whether any of them shares memory with the range and can be read as it
    bool Intersects(const MemoryRange& range) const;
    /// The range from the first to the last byte it has in common with those that can be
    /// read as it; empty if there is none
    MemoryRange Hull(const MemoryRange& range) const;
    void Add(MemoryRange range);
    /// Takes the memory of the range out, whatever it is laid out as
    void Remove(const MemoryRange& range);
    void Clear() {
        ranges.clear();
    }
    bool operator==(const MemoryRanges&) const = default;

private:
    boost::container::small_vector<MemoryRange, 16> ranges;
};

/**
 * Leaves out GPU work whose result nobody gets to see. One of the rules of WorkRules, the one
 * that votes Unseen, and the one that keeps the watch over what was left out wrongly.
 *
 * A game in stereoscopic mode renders the top screen twice. With one eye on display, half of
 * that is for nothing, as is anything else a game renders and then does not use. Which work
 * that is cannot be told from the buffers it goes to: games pass the image of one eye through
 * the buffer of the other while processing it. It can be told from where the results end up.
 *
 * Every draw, copy, transfer and fill is noted with the memory it reads and the memory it
 * writes. When a frame ends, the last two frames are gone through backwards, starting from
 * what was put on display or read by the game: a piece of work was needed if something needed
 * reads what it wrote before that is replaced. Frames repeat, so what comes after the last
 * frame is taken to be those frames again.
 *
 * Work that was not needed for a while is left out from then on. It is recognized by what it
 * reads and writes and by which copy, transfer or fill came before it, not by its position in
 * the frame, so that a scene with more or fewer objects in it remains the same frame. Copies,
 * transfers and fills that come up more than once are told apart by counting them from the
 * last time an image was put in a place for display, not from where the frame began: a frame
 * ends at the VBlank that takes up the swap, which is not always at the same point of what
 * the game does. The
 * memory it would have written counts as stale, and so does what other work makes from it.
 * Should anything stale get on display or to the game, the guess was wrong: everything is
 * rendered again, for longer each time.
 *
 * A game may also keep both eyes in one buffer, one above the other, and process the two
 * with one draw over all of it. So a draw is taken to write the rows its viewport and
 * scissor confine it to, and what is needed of it is followed by rows: a row it writes is
 * made from the same row of what its target held, and, where its texture coordinates go
 * with the rows in a straight line (a rectangle on the screen), from the rows of the
 * texture that line gives. The reach of a filter beyond those is not followed: where two
 * images meet in a buffer, a game's filter takes in a row of the other one, and that row
 * is not kept as it was for it.
 */
class FrameLiveness final : public WorkRule {
public:
    explicit FrameLiveness(GPU& gpu);
    ~FrameLiveness() override;

    const char* Name() const override {
        return "unseen work";
    }
    const char* Why() const override {
        return "nobody sees its result";
    }
    /// As a rule it is told of every piece of work the game sends, also of what another rule
    /// leaves out as done already: memory is then as if that had been carried out, which is
    /// how it is entered here.
    Vote OnDraw(DrawFacts& facts) override;
    Vote OnFill(const FillFacts& facts) override;
    Vote OnTransfer(const TransferFacts& facts) override;
    void OnFrameEnd(const FrameFacts& facts) override;

    /// Titles that are known not to cope with a missing right eye are left alone
    void SetAllowed(bool allowed_) {
        allowed = allowed_;
    }

    /// Whether a draw with these registers replaces all its target holds, if its triangles
    /// cover the viewport. Cheap, to be asked before the triangles are looked at.
    bool MayReplaceTarget(const Pica::RegsInternal& regs) const;

    /// Whether a draw with these registers samples something the GPU drew in a way that
    /// makes it worth finding out how its texture coordinates go with the rows of its
    /// target. Cheap, to be asked before the vertices are looked at.
    bool WantsTextureRows(const Pica::RegsInternal& regs) const;

    /// Called for each draw. Returns false for the ones to leave out, which are none if the
    /// draw cannot be left out any more.
    /// overwrites_colour, overwrites_depth: the draw writes every pixel of the rows it is
    /// confined to anew, the colour or depth and stencil of it, whatever those held: its
    /// triangles cover its viewport, the viewport is as wide as the target, and no test or
    /// blending lets through what was there (a clear done by drawing, also one that tests
    /// nothing but still writes depth, which replaces_target does not take in).
    bool ShouldAllowDraw(const Pica::RegsInternal& regs, bool replaces_target,
                         bool may_leave_out = true, const DrawTextureRows* texture_rows = nullptr,
                         bool overwrites_colour = false, bool overwrites_depth = false);

    /// Called for each texture copy and display transfer. Returns false for the ones to leave
    /// out; the game is told they are done as usual.
    bool ShouldAllowTransfer(const Pica::DisplayTransferConfig& config);

    /// Called for each memory fill. Returns false for the ones to leave out.
    bool ShouldAllowFill(const Pica::MemoryFillConfig& config);

    /// Guest memory the GPU may have written is about to be read from outside the GPU
    static void NoteGuestRead(PAddr start, u32 size);

    /// Called for each VBlank, before the screens are put together
    void NoteDisplay();

    /// Called when the game swaps the buffers of the top screen
    void EndFrame();

private:
    enum class Kind : u8 { Draw, Copy, Transfer, Fill, GuestRead, Count };

    /// How something a draw reads goes with the rows of its target
    struct Link {
        enum class How : u8 {
            Whole,  ///< Any of it may go into any row
            Rows,   ///< What the target held, colour or depth: row for row
            Mapped, ///< A texture whose rows go with those of the target in a straight line
        };
        How how{How::Whole};
        /// Rows: 0 for the colour buffer, 1 for the depth buffer
        u8 buffer{};
        /// Mapped: row of the texture = scale * row of the target + offset, both as rows
        /// of memory; the bytes a row of the texture takes and how many rows it has
        f32 scale{};
        f32 offset{};
        u32 row_bytes{};
        u32 rows{};
    };

    /// A piece of work
    struct Step {
        /// What makes it the same piece of work in another frame
        u64 key{};
        /// The memory it writes: the colour buffer of a draw or the destination, then the
        /// depth buffer of a draw
        std::array<MemoryRange, 2> writes{};
        /// The memory it reads that the GPU may have written, including what it writes to
        /// unless it replaces that
        boost::container::static_vector<MemoryRange, 12> reads;
        /// For a draw, one for each read: how it goes with the rows written
        boost::container::static_vector<Link, 12> links;
        /// For a draw: the rows of memory of its target it is confined to, in the eights
        /// memory holds them in. Both zero for other work.
        u32 first_row{};
        u32 last_row{};
        /// For a draw: where its colour and its depth buffer begin, what a row of each
        /// takes and what each is laid out as
        std::array<PAddr, 2> buffer_start{};
        std::array<u32, 2> row_bytes{};
        std::array<u32, 2> buffer_layout{};
        Kind kind{};
        /// Nothing of what the first place it writes held is left afterwards
        bool replaces{};
        /// A draw: nothing of what the rows of its depth buffer held is left afterwards, and
        /// nothing of it went into the result
        bool replaces_depth{};
        /// The image in the first place it writes is all new afterwards
        bool renews{};
        /// Its result is on display or goes to the game
        bool shown{};
    };

    struct Verdict {
        /// Looks at the last frames in a row that found this work not needed
        u16 unneeded_streak{};
        /// Frame it last appeared in
        u32 last_frame{};
        /// Bookkeeping of the look at the last frames that is under way
        u32 look{};
        bool needed_in_look{};
        bool updated_in_look{};
    };

    struct ShownPlace {
        MemoryRange range;
        u32 last_frame{};
    };

    void UpdateActive();
    void Forget();
    bool IsShown(const MemoryRange& range) const;
    /// Finishes a step, decides whether it is carried out and notes it
    bool Submit(Step& step);
    void AddRead(Step& step, const MemoryRange& range) const;
    void AddRead(Step& step, const MemoryRange& range, const Link& link) const;
    /// Rows of one of the buffers of a draw, as memory
    static MemoryRange RowsOf(const Step& step, u32 buffer, u32 first, u32 last);
    /// The rows of its target a draw has to produce for what is read after it. False if
    /// none.
    static bool NeededRows(const Step& step, const MemoryRanges& live, u32& first, u32& last);
    /// Adds what a draw reads to produce those rows
    static void AddReadsFor(const Step& step, u32 first, u32 last, MemoryRanges& live);
    /// The rows of its target a draw makes from stale memory, and one of the stale reads.
    /// False if none.
    bool StaleRows(const Step& step, u32& first, u32& last, const MemoryRange*& read) const;
    /// The key of a copy, transfer or fill, which tells apart the times it comes up since an
    /// image was last put in a place for display
    u64 NextOccurrence(u64 base);
    /// A copy or transfer put an image in a place for display: counting starts over
    void BeginStretch(u64 base);
    void GuestRead(const MemoryRange& range);
    void Mispredicted(const char* reader, const MemoryRange& range);
    void LookAtLastFrames();
    /// Writes to the log what the frame that just ended was made of, see EndFrame
    void LogOutline() const;
    /// Whether the game drew with the stencil test on lately (see OnDraw)
    bool StencilInUse() const {
        return stencil_seen && frame - stencil_last_frame <= STENCIL_FORGOTTEN_AFTER;
    }
    static constexpr u32 STENCIL_FORGOTTEN_AFTER = 600;
    bool stencil_seen = false;
    u32 stencil_last_frame = 0;
    /// Frames in a row with a transfer left out and every draw carried out, below zero
    /// while the outline was written lately; how often it was written
    s32 outline_streak = 0;
    u32 outlines_logged = 0;

    GPU& gpu;
    bool allowed = true;
    bool active = false;

    /// Counts frames, for ageing
    u32 frame = 0;
    /// Frames to render everything for after a wrong guess, and how many of those there were
    u32 distrust = 0;
    u32 wrong_guesses = 0;
    /// How often the time after a wrong guess was doubled, the time the last one was given,
    /// and the frames since it
    u32 distrust_doublings = 0;
    u32 last_distrust = 0;
    u64 frames_since_wrong_guess = 0;

    /// The steps of the frame under way and of the one before
    std::vector<Step> current;
    std::vector<Step> previous;
    bool too_many_steps = false;
    bool previous_too_many_steps = false;
    /// Whether the last step noted was carried out
    bool last_carried_out = false;
    std::vector<bool> needed;

    /// The last copy, transfer or fill, which the draws after it go with
    u64 anchor = 0;
    /// The copy or transfer that last put an image in a place for display
    u64 stretch = 0;
    bool stretch_began_in_frame = false;
    /// Whether copies, transfers and fills are counted from there, not from where the frame
    /// began
    bool counts_from_display = true;
    /// How often each copy, transfer and fill came up since then already
    boost::container::small_vector<std::pair<u64, u32>, 32> occurrences;

    std::unordered_map<u64, Verdict> verdicts;
    u32 look = 0;

    /// All memory the GPU was seen writing
    MemoryRanges written;
    /// Memory that does not hold what it should: what work left out would have written, and
    /// what was made from that
    MemoryRanges stale;
    /// Framebuffers that were on display lately
    boost::container::small_vector<ShownPlace, 8> shown_places;

    /// Steps of this frame and how many of them were left out, by kind
    std::array<u32, static_cast<std::size_t>(Kind::Count)> seen{};
    std::array<u32, static_cast<std::size_t>(Kind::Count)> left_out{};
    bool leaving_out = false;
    u32 frames_since_status = 0;
};

} // namespace VideoCore
