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

namespace Pica {
struct DisplayTransferConfig;
struct MemoryFillConfig;
struct RegsInternal;
} // namespace Pica

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
 * Leaves out GPU work whose result nobody gets to see.
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
 * the frame, so that a scene with more or fewer objects in it remains the same frame. The
 * memory it would have written counts as stale, and so does what other work makes from it.
 * Should anything stale get on display or to the game, the guess was wrong: everything is
 * rendered again, for longer each time.
 */
class FrameLiveness {
public:
    explicit FrameLiveness(GPU& gpu);
    ~FrameLiveness();

    /// Titles that are known not to cope with a missing right eye are left alone
    void SetAllowed(bool allowed_) {
        allowed = allowed_;
    }

    /// Whether a draw with these registers replaces all its target holds, if its triangles
    /// cover the viewport. Cheap, to be asked before the triangles are looked at.
    bool MayReplaceTarget(const Pica::RegsInternal& regs) const;

    /// Called for each draw. Returns false for the ones to leave out, which are none if the
    /// draw cannot be left out any more.
    bool ShouldAllowDraw(const Pica::RegsInternal& regs, bool replaces_target,
                         bool may_leave_out = true);

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
        Kind kind{};
        /// Nothing of what the first place it writes held is left afterwards
        bool replaces{};
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
    /// The key of a copy, transfer or fill, which tells apart the times it comes up in a frame
    u64 NextOccurrence(u64 base);
    void GuestRead(const MemoryRange& range);
    void Mispredicted(const char* reader, const MemoryRange& range);
    void LookAtLastFrames();

    GPU& gpu;
    bool allowed = true;
    bool active = false;

    /// Counts frames, for ageing
    u32 frame = 0;
    /// Frames to render everything for after a wrong guess, and how many of those there were
    u32 distrust = 0;
    u32 wrong_guesses = 0;

    /// The steps of the frame under way and of the one before
    std::vector<Step> current;
    std::vector<Step> previous;
    bool too_many_steps = false;
    bool previous_too_many_steps = false;
    /// Whether the last step noted was carried out
    bool last_carried_out = false;
    std::vector<bool> needed;

    /// The last copy, transfer or fill of this frame, which the draws after it go with
    u64 anchor = 0;
    /// How often each copy, transfer and fill came up in this frame already
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
