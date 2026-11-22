// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;

/**
 * The emulated system's memory as a buffer of the host GPU. The console has one memory for its
 * processor and its GPU; this gives the host the same: the driver takes the pages the emulated
 * memory is kept in as memory of its own (VK_EXT_external_memory_host), and what the game
 * wrote there the GPU reads there, with no copy in between.
 *
 * The GPU reads when it gets to a command, which is later than the game issued it. What the
 * game writes anew in the meantime the GPU would see as it is then: InPlaceData says what may
 * be read this way.
 */
class GuestMemory {
public:
    /// `pool` is what Memory::MemorySystem::GetHostPool() gave, which is empty if the emulated
    /// memory is not kept in one piece. It has to outlive this.
    GuestMemory(const Instance& instance, std::span<u8> pool);
    ~GuestMemory();

    GuestMemory(const GuestMemory&) = delete;
    GuestMemory& operator=(const GuestMemory&) = delete;

    /// False if there is no pool or the device did not take it: nothing else may be used then
    [[nodiscard]] bool IsValid() const {
        return static_cast<bool>(buffer);
    }

    [[nodiscard]] vk::Buffer Handle() const {
        return *buffer;
    }

    /// Whether the GPU sees what the processor has written as soon as it is written. Where
    /// it does not (a system on a chip whose memory bus does not keep the two coherent, an
    /// Adreno 610's), it sees what is in memory itself: what the processor's cache still
    /// holds of a page has to be written out before the GPU reads the page
    /// (Common::WriteOutDataCache, at the moment InPlaceData lets it be read in place).
    [[nodiscard]] bool IsCoherent() const {
        return coherent;
    }

    /// Where in the buffer the `size` bytes at `data` are, if all of them are emulated memory
    /// the buffer covers
    [[nodiscard]] std::optional<u32> Offset(const u8* data, std::size_t size) const {
        if (!buffer || data < pool.data() || size > pool.size()) {
            return std::nullopt;
        }
        const std::size_t offset = static_cast<std::size_t>(data - pool.data());
        if (offset > pool.size() - size) {
            return std::nullopt;
        }
        return static_cast<u32>(offset);
    }

private:
    std::span<u8> pool;
    vk::UniqueDeviceMemory memory;
    vk::UniqueBuffer buffer;
    bool coherent{true};
};

/**
 * Which vertex and index data a draw may take from GuestMemory, in place, and which it has to
 * copy as it is when the draw is issued.
 *
 * Nearly all of that data is models the game loads once (measured: 97.5% to 100% of the bytes
 * of a frame), the rest it writes anew every frame, and the host GPU reads when it gets to a
 * draw, by which time the game may have written the next frame's data: that flickers. So this
 * knows the emulated memory page by page, the way the cache of textures does:
 *
 * - A page a draw takes data from is watched: the emulation announces what it writes there
 *   (the rasterizer's InvalidateRegion), and while it is watched the data is still copied.
 *   Where the recompiled code reaches memory through the host's own address translation it
 *   still reads a watched page directly; elsewhere its reads go the long way as well.
 * - After CALM_FRAMES frames of use without a write the page is read in place. At that moment
 *   `settle` is called for it: where the GPU does not see through the processor's cache, that
 *   is when the cache's part of the page is written out. Once is enough, as from then on a
 *   write ends the reading in place before it happens.
 * - A write to a watched page ends that: its data is copied again, for a time that doubles
 *   with every such end, before it is tried anew. If a draw that is not through the GPU yet
 *   reads the page in place, the write has to wait for it first: OnWrite says for what.
 * - A page the game keeps reading is given up as well: while watched, each of its reads goes
 *   the long way.
 *
 * A stretch of data is read in place only if all of its pages are.
 */
class InPlaceData {
public:
    /// Has a stretch of emulated memory watched or no longer, see
    /// Memory::MemorySystem::RasterizerMarkRegionWatched
    using Mark = std::function<void(PAddr addr, u32 size, bool watch)>;
    /// Called for a page that is read in place from now on. May be empty.
    using Settle = std::function<void(PAddr addr, u32 size)>;

    InPlaceData(Mark mark, Settle settle, bool log);

    /// Whether a draw whose commands go to the GPU under `tick` may read the `size` bytes at
    /// `addr` in place
    [[nodiscard]] bool Use(PAddr addr, u32 size, u64 tick);

    /// The emulation is about to write the stretch. Returns the tick the GPU has to be done
    /// with before it may, zero if there is nothing to wait for. `gpu_tick` is how far the
    /// GPU is known to have come.
    [[nodiscard]] u64 OnWrite(PAddr addr, u32 size, u64 gpu_tick);

    /// The emulation reads the stretch the long way
    void OnRead(PAddr addr, u32 size);

    /// A write waited this long for the GPU
    void NoteWait(std::chrono::nanoseconds duration) {
        ++waits;
        waited += duration;
    }

    /// Once a frame
    void OnFrame();

    /// Gives up every page and starts over, when the rasterizer's caches are dropped
    void Reset();

    /// Counts up whenever a page stops being read in place. While it stands still, nothing
    /// that was read in place has been written: what was worked out from such data (the
    /// range of a draw's indices) still holds.
    [[nodiscard]] u64 Generation() const {
        return generation;
    }

private:
    enum class State : u8 {
        Unknown, ///< Not drawn from yet
        Trial,   ///< Watched; its data is copied until it has been left alone long enough
        InPlace, ///< Watched; its data is read where it is
        Copied,  ///< Written or read too often while watched: copied, tried anew later
    };

    struct Page {
        u64 last_use{};    ///< Tick of the last draw that reads it in place
        u32 frame{};       ///< Trial: the last frame it was drawn from
        u32 retry{};       ///< Copied: the frame from which it is tried anew
        u32 read_window{}; ///< The stretch of frames `reads` belongs to
        u16 calm{};        ///< Trial: frames it was drawn from since it is watched
        u16 reads{};
        State state{State::Unknown};
        u8 failures{}; ///< How often a trial or the use in place ended
    };

    /// The pages the stretch touches, first and last, as far as they are of a memory that
    /// is kept track of
    [[nodiscard]] std::optional<std::pair<u32, u32>> PagesOf(PAddr addr, u32 size) const;
    [[nodiscard]] static PAddr AddressOf(u32 page);
    void GiveUp(u32 index);

    Mark mark;
    Settle settle;
    bool log;
    std::vector<Page> pages;
    u32 frame{};
    u32 watched{};
    u64 generation{1};
    // Since the last line in the log
    u32 report_frames{};
    u32 begun{};
    u32 promoted{};
    u32 ended_by_write{};
    u32 ended_by_read{};
    u32 waits{};
    std::chrono::nanoseconds waited{};
};

/**
 * For measuring only (Common::TraceMarkers::MeasureVertexData): what a draw that takes its
 * vertex and index data in place, from GuestMemory, has to reckon with. It hashes every
 * stretch of data a draw uses, when the draw is issued, again when its commands go to the
 * GPU, and once more when the GPU is known to be done with them, and counts
 *
 * - how much of the data is as it was when it was last drawn with (models, as a rule), written
 *   anew (what the game makes every frame), or at a place not drawn from before;
 * - how much the game wrote anew before the commands were even sent: a draw in place is
 *   certain to get that wrong;
 * - how much it wrote anew before the GPU was seen to be done, looked at once a frame, so up to
 *   a frame after the GPU really was: an upper bound of what a draw in place gets wrong.
 */
class GuestDataWatch {
public:
    void NoteDraw() {
        ++draws;
    }

    /// A draw uses the `size` bytes at `data`. `tick` is that of the commands being recorded.
    void NoteUse(const u8* data, u32 size, bool indices, bool in_place, u64 tick);

    /// The commands recorded so far go to the GPU
    void OnSubmit();

    /// Once a frame. `gpu_tick` is how far the GPU is known to have come.
    void OnFrame(u64 gpu_tick);

private:
    struct Use {
        const u8* data;
        u32 size;
        u64 hash;
        u64 tick;
        bool rewritten;
    };

    struct Count {
        u64 uses{};
        u64 bytes{};

        void Add(u32 size) {
            ++uses;
            bytes += size;
        }
    };

    void Report();

    std::vector<Use> recorded; ///< By commands that have not been sent yet
    std::vector<Use> sent;     ///< By commands the GPU is not known to be done with
    /// The hash of each stretch of data when it was last drawn with, by its place and length
    std::unordered_map<u64, u64> last_hash;

    u32 frames{};
    u64 draws{};
    Count all;
    Count indices;
    Count in_place;
    Count unchanged;         ///< As when last drawn with
    Count changed;           ///< Written anew since it was last drawn with
    Count fresh;             ///< Not drawn with before
    Count rewritten_unsent;  ///< Written anew before its commands were sent
    Count rewritten_pending; ///< Written anew after that, before the GPU was seen to be done
};

} // namespace Vulkan
