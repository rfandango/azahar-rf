// Copyright 2014-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <csignal>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>
#include <boost/serialization/array.hpp>
#include <boost/serialization/binary_object.hpp>
#include "audio_core/dsp_interface.h"
#include "common/archives.h"
#include "common/assert.h"
#include "common/atomic_ops.h"
#include "common/common_types.h"
#include "common/hash.h"
#include "common/hitch_tracker.h"
#include "common/logging/log.h"
#include "common/memory_view.h"
#include "common/optional_helper.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/swap.h"
#include "common/trace_markers.h"
#include "core/arm/arm_interface.h"
#include "core/arm/exception_handler.h"
#include "core/core.h"
#ifdef ENABLE_GDBSTUB
#include "core/gdbstub/gdbstub.h"
#endif
#include "core/global.h"
#include "core/hle/kernel/process.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "core/memory.h"
#include "video_core/gpu.h"
#include "video_core/renderer_base.h"

SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::FCRAM>)
SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::VRAM>)
SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::DSP>)
SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::N3DS>)

#ifndef SIGTRAP
constexpr u32 SIGTRAP = 5;
#endif

#ifndef SIGSEGV
constexpr u32 SIGSEGV = 11;
#endif

namespace Memory {

namespace {

/// The emulated system's memory when host MMU access is in use: one shared piece that page
/// tables show in their views. Set for as long as the MemorySystem that owns it lives.
Common::SharedBacking* host_backing = nullptr;

/// Where the four memories of the emulated system are in it
constexpr std::size_t BACKING_FCRAM = 0;
constexpr std::size_t BACKING_VRAM = BACKING_FCRAM + FCRAM_N3DS_SIZE;
constexpr std::size_t BACKING_N3DS = BACKING_VRAM + VRAM_SIZE;
constexpr std::size_t BACKING_DSP = BACKING_N3DS + N3DS_EXTRA_RAM_SIZE;
constexpr std::size_t BACKING_SIZE = BACKING_DSP + DSP_RAM_SIZE;

/// A view takes in all 32 bits of an emulated address. One page more is reserved behind it and
/// never filled: an access that starts in the last bytes of the address space runs into it
/// and faults, instead of reaching whatever the host has there.
constexpr std::size_t HOST_VIEW_SIZE = std::size_t{1} << 32;
constexpr std::size_t HOST_VIEW_GUARD = CITRA_PAGE_SIZE;

/// Whether the recompiled code is to reach the emulated memory through views of it: the
/// host's own address translation in place of a lookup for every access (measured on an
/// Adreno 650 device: 8.5% to 11% fewer cycles of the emulation thread). The way it is done
/// where it has run, Android on arm64; elsewhere not until it has been tried there.
bool WantsHostMmu() {
#if defined(__ANDROID__) && defined(__aarch64__)
    return (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyMemoryAccess) == 0;
#else
    return false;
#endif
}

/// Whether the host GPU is to be offered the emulated memory as memory of its own, as on the
/// console, where processor and GPU have one memory. Like the views, the way it is done where
/// it has run.
bool WantsSharedGpuMemory() {
#if defined(__ANDROID__) && defined(__aarch64__)
    return (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyVertexCopies) == 0;
#else
    return false;
#endif
}

/// The emulated memory in one piece is what two things build on: views of it for the
/// recompiled code, and the host GPU taking it as memory of its own
std::unique_ptr<Common::SharedBacking> MakeHostBacking() {
    const bool for_gpu = WantsSharedGpuMemory();
    if (!WantsHostMmu() && !for_gpu) {
        return nullptr;
    }
    auto backing = std::make_unique<Common::SharedBacking>(BACKING_SIZE);
    if (!backing->IsValid()) {
        LOG_WARNING(HW_Memory, "The emulated memory was asked for in one shared piece, but "
                               "this host has no shared memory in pages of 4 KiB for it");
        return nullptr;
    }
    LOG_INFO(HW_Memory, "The emulated memory is one piece of shared memory{}{}",
             WantsHostMmu() ? ", shown again in a view of each address space" : "",
             for_gpu ? ", offered to the host GPU as it is" : "");
    return backing;
}

/// One of the emulated system's memories: a part of the shared backing if there is one, or
/// memory of its own
class RamBlock {
public:
    RamBlock(const Common::SharedBacking* backing, std::size_t offset, std::size_t size) {
        if (backing) {
            pointer = backing->Data() + offset;
        } else {
            own = std::make_unique<u8[]>(size);
            pointer = own.get();
        }
    }

    [[nodiscard]] u8* get() const {
        return pointer;
    }

private:
    std::unique_ptr<u8[]> own;
    u8* pointer = nullptr;
};

} // Anonymous namespace

PageTable::PageTable() {
    // Nothing is mapped until something is
    pointers.raw.fill(nullptr);
    attributes.fill(PageType::Unmapped);
}

PageTable::~PageTable() = default;

PageTable::Pointers::Pointers() {
    if (host_backing) {
        auto view = std::make_unique<Common::AddressSpaceView>(*host_backing,
                                                               HOST_VIEW_SIZE + HOST_VIEW_GUARD);
        if (view->IsValid()) {
            host_view = std::move(view);
            host_view_backing = host_backing;
        } else {
            LOG_ERROR(HW_Memory, "No room for a view of an address space: its code will look "
                                 "every address up");
        }
    }
}

PageTable::Pointers::~Pointers() = default;

u8* PageTable::Pointers::HostView() const {
    return host_view ? host_view->Base() : nullptr;
}

void PageTable::Pointers::BeginChanges() {
    changes_depth++;
}

void PageTable::Pointers::EndChanges() {
    if (changes_depth == 0 || --changes_depth != 0 || changed.empty()) {
        return;
    }
    // Runs of pages next to each other go to the host in one piece
    std::sort(changed.begin(), changed.end());
    changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
    std::size_t run_start = 0;
    for (std::size_t i = 1; i <= changed.size(); i++) {
        if (i == changed.size() || changed[i] != changed[i - 1] + 1) {
            ApplyToView(changed[run_start], i - run_start);
            run_start = i;
        }
    }
    changed.clear();
}

void PageTable::Pointers::NoteChange(std::size_t idx) {
    if (changes_depth != 0) {
        changed.push_back(static_cast<u32>(idx));
    } else {
        ApplyToView(idx, 1);
    }
}

void PageTable::Pointers::SetReadOnly(std::size_t idx, u8* memory) {
    if (!host_view) {
        return;
    }
    const auto it = read_only.find(static_cast<u32>(idx));
    if (memory == nullptr) {
        if (it == read_only.end()) {
            return;
        }
        read_only.erase(it);
    } else {
        if (it != read_only.end() && it->second == memory) {
            return;
        }
        read_only[static_cast<u32>(idx)] = memory;
    }
    NoteChange(idx);
}

void PageTable::Pointers::ApplyToView(std::size_t first, std::size_t count) {
    if (!host_view) {
        return;
    }
    // Only memory of the backing this view was made for can show in it. Everything else (a
    // page without a pointer: not mapped, cached by the renderer, watched; or memory that is
    // not the emulated system's own) is left empty, so that an access faults and the
    // recompiled code goes the long way.
    const Common::SharedBacking* const backing =
        host_backing == host_view_backing ? host_view_backing : nullptr;
    // A page with a pointer shows its memory, writable; one without, given to SetReadOnly,
    // shows it readable only
    struct Shown {
        std::size_t offset;
        bool writable;
    };
    const auto shown = [&](std::size_t idx) -> std::optional<Shown> {
        const u8* pointer = raw[idx];
        bool writable = true;
        if (!pointer) {
            if (read_only.empty()) {
                return std::nullopt;
            }
            const auto it = read_only.find(static_cast<u32>(idx));
            if (it == read_only.end()) {
                return std::nullopt;
            }
            pointer = it->second;
            writable = false;
        }
        if (!backing || !backing->Contains(pointer)) {
            return std::nullopt;
        }
        const std::size_t offset = static_cast<std::size_t>(pointer - backing->Data());
        if (offset % CITRA_PAGE_SIZE != 0) {
            return std::nullopt;
        }
        return Shown{offset, writable};
    };
    const std::size_t end = first + count;
    std::size_t idx = first;
    while (idx < end) {
        const auto start = shown(idx);
        std::size_t pages = 1;
        if (start) {
            while (idx + pages < end) {
                const auto next = shown(idx + pages);
                if (!next || next->writable != start->writable ||
                    next->offset != start->offset + pages * CITRA_PAGE_SIZE) {
                    break;
                }
                pages++;
            }
            host_view->Map(idx * CITRA_PAGE_SIZE, start->offset, pages * CITRA_PAGE_SIZE,
                           start->writable);
        } else {
            while (idx + pages < end && !shown(idx + pages)) {
                pages++;
            }
            host_view->Unmap(idx * CITRA_PAGE_SIZE, pages * CITRA_PAGE_SIZE);
        }
        idx += pages;
    }
}

void PageTable::Clear() {
    pointers.raw.fill(nullptr);
    pointers.refs.fill(MemoryRef());
    attributes.fill(PageType::Unmapped);
    pointers.changed.clear();
    pointers.read_only.clear();
    pointers.ApplyToView(0, PAGE_TABLE_NUM_ENTRIES);
}

class RasterizerCacheMarker {
public:
    void Mark(VAddr addr, bool cached) {
        bool* p = At(addr);
        if (p)
            *p = cached;
    }

    bool IsCached(VAddr addr) {
        bool* p = At(addr);
        if (p)
            return *p;
        return false;
    }

private:
    bool* At(VAddr addr) {
        if (addr >= VRAM_VADDR && addr < VRAM_VADDR_END) {
            return &vram[(addr - VRAM_VADDR) / CITRA_PAGE_SIZE];
        }
        if (addr >= LINEAR_HEAP_VADDR && addr < LINEAR_HEAP_VADDR_END) {
            return &linear_heap[(addr - LINEAR_HEAP_VADDR) / CITRA_PAGE_SIZE];
        }
        if (addr >= NEW_LINEAR_HEAP_VADDR && addr < NEW_LINEAR_HEAP_VADDR_END) {
            return &new_linear_heap[(addr - NEW_LINEAR_HEAP_VADDR) / CITRA_PAGE_SIZE];
        }
        if (addr >= PLUGIN_3GX_FB_VADDR && addr < PLUGIN_3GX_FB_VADDR_END) {
            return &plugin_fb[(addr - PLUGIN_3GX_FB_VADDR) / CITRA_PAGE_SIZE];
        }
        return nullptr;
    }

    std::array<bool, VRAM_SIZE / CITRA_PAGE_SIZE> vram{};
    std::array<bool, LINEAR_HEAP_SIZE / CITRA_PAGE_SIZE> linear_heap{};
    std::array<bool, NEW_LINEAR_HEAP_SIZE / CITRA_PAGE_SIZE> new_linear_heap{};
    std::array<bool, PLUGIN_3GX_FB_SIZE / CITRA_PAGE_SIZE> plugin_fb{};

    static_assert(sizeof(bool) == 1);
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar & vram;
        ar & linear_heap;
        ar & new_linear_heap;
        ar & plugin_fb;
    }
};

class MemorySystem::Impl {
public:
    /// With host MMU access in use, the shared memory the four below are parts of
    std::unique_ptr<Common::SharedBacking> backing = MakeHostBacking();
    // Visual Studio would try to allocate these on compile time
    // if they are std::array which would exceed the memory limit.
    RamBlock fcram{backing.get(), BACKING_FCRAM, Memory::FCRAM_N3DS_SIZE};
    RamBlock vram{backing.get(), BACKING_VRAM, Memory::VRAM_SIZE};
    RamBlock n3ds_extra_ram{backing.get(), BACKING_N3DS, Memory::N3DS_EXTRA_RAM_SIZE};
    RamBlock dsp_ram{backing.get(), BACKING_DSP, Memory::DSP_RAM_SIZE};

    Core::System& system;
    std::shared_ptr<PageTable> current_page_table = nullptr;
    RasterizerCacheMarker cache_marker;
    /// The pages that are watched for writes, see MemorySystem::RasterizerMarkRegionWatched.
    /// Not kept in a save state: the rasterizer starts over when one is made or loaded.
    RasterizerCacheMarker watch_marker;
    /// The cached pages that may be read without announcing it, see
    /// MemorySystem::RasterizerMarkRegionReadable. Not kept in a save state either.
    RasterizerCacheMarker readable_marker;
    /// The pages armed for a decoder that works behind the game's back, see
    /// MemorySystem::ArmGpuReferences. To the page tables an armed page is a watched one.
    RasterizerCacheMarker armed_marker;
    /// The pages shut for it: what the emulated GPU writes. Neither read nor written directly.
    RasterizerCacheMarker shut_marker;
    /// The marks of pages are changed by one thread at a time: the decoder's cache, and the
    /// emulation thread arming pages and mapping memory
    std::mutex mark_mutex;

    /// Whether a page that was found to be plain and without a pointer, which none is, is so all
    /// the same. The marks of a page are changed by the thread that decodes GPU commands, the
    /// type and the pointer one after the other, and another thread may come to see the second
    /// change before the first. They are to be believed while nobody changes them.
    bool IsPlainWithoutPointer(PageTable& page_table, VAddr vaddr) {
        std::scoped_lock lock{mark_mutex};
        const std::size_t page = vaddr >> CITRA_PAGE_BITS;
        return page_table.pointers[page] == nullptr &&
               page_table.attributes[page] == PageType::Memory;
    }
    std::atomic<bool> track_references{false};
    std::mutex reference_mutex;
    /// Pages noted since the last arming, by their place in surface_cells (under the mutex):
    /// read by the decoder for a command list, read by it for a transfer, written by it
    std::vector<u32> new_references;
    std::vector<u32> new_transfer_references;
    std::vector<u32> new_written;
    /// Decoder thread: the pages it has noted, bit 0 as read for a list, bit 1 as written,
    /// bit 4 as read for a transfer. And which of the two it is on.
    std::vector<u8> reference_noted;
    u8 gpu_reader{1};
    /// What the decoder last read a page as (a GpuReference), and the 64-byte cells of it
    /// that it read. Set by the decoder, looked at by the emulation thread.
    std::vector<u8> reference_kind;
    std::vector<u64> reference_cells;
    /// Emulation thread: bit 0 for a page known to be read by the decoder for a list and bit
    /// 4 for a transfer, bit 1 while it is armed; bit 2 for one known to be written by it and
    /// bit 3 while it is shut
    std::vector<u8> reference_state;
    /// Emulation thread: the known pages that are not armed, for either kind of command, and
    /// not shut
    std::vector<u32> to_arm;
    std::vector<u32> to_arm_transfer;
    std::vector<u32> to_shut;
    /// Emulation thread: the volatile pages (bit 5 of reference_state), in the order of their
    /// numbers. A page that is written while it is armed becomes one: it is copied at every
    /// hand-over instead of armed.
    struct VolatilePage {
        u32 page{};
        /// Looks in a row its contents were the same at, how many of those it takes for the
        /// page to be armed again, and the VBlank (late_vblank) of the next look
        u32 same{};
        u32 looks_needed{1};
        u64 next_look{};
        u64 checksum{};
        /// While it is still armed: the command that has to be done before it need not be
        u64 disarm_after{~u64{0}};
        /// The number of the last command that read a copy of it, as last looked at
        /// (snapshot_last_read), and the VBlank (late_vblank) at which that was seen to have
        /// moved on, or at which the page became volatile
        u32 read_turn{};
        u64 read_vblank{};
        /// A draw of the list about to be handed over reads it (WantCopyOf)
        bool wanted{};
        /// Copied for every list until this VBlank: the decoder read it without a copy
        u64 always_until{};
    };
    std::vector<VolatilePage> volatile_pages;
    /// For each page 1 if it is one of them, 2 once the decoder has said it read it without
    /// a copy (set here, looked at by the decoder), and how many there are
    std::unique_ptr<std::atomic<u8>[]> volatile_flag;
    std::atomic<u32> volatile_count{0};
    /// Decoder to emulation thread, under reference_mutex: volatile pages it read for a list
    /// that came without a copy of them
    std::vector<u32> unheld_pages;
    std::atomic<bool> has_unheld{false};
    std::atomic<u64> unheld_reads{};
    u64 narrowed_out{};
    /// VBlanks a page is copied for every list after that
    static constexpr u64 ALWAYS_COPIED_FOR = 600;
    /// Whether following the draws pays (NoteDrawsFollowed): over FOLLOW_WINDOW lists the
    /// draws followed and the pages left uncopied; a page not copied is taken to be worth
    /// DRAWS_A_PAGE draws followed (a copy of a page nobody has in its cache is one to two
    /// microseconds, a draw followed a quarter of one); and until which VBlank it is left be
    u64 follow_draws{};
    u64 follow_saved{};
    u32 follow_lists{};
    u64 follow_paused_until{};
    /// How long it is left be the next time it does not pay: a few seconds the first time
    /// (the lists right after a change of scene say little), twice as long each time after
    /// that up to half a minute, and from the beginning again once it has paid
    u64 follow_pause{FOLLOW_PAUSE_FIRST};
    static constexpr u32 FOLLOW_WINDOW = 64;
    static constexpr u64 DRAWS_A_PAGE = 4;
    static constexpr u64 FOLLOW_PAUSE_FIRST = 300;
    static constexpr u64 FOLLOW_PAUSE_MOST = 1800;

    /// The volatile pages the stretch lies on: all of them where the stretch is not within
    /// the memory they are counted in
    std::pair<std::size_t, std::size_t> VolatileIn(PAddr addr, u32 size) const {
        if (size == 0 || volatile_pages.empty()) {
            return {0, 0};
        }
        const auto first = CellPage(addr);
        const auto last = CellPage(addr + size - 1);
        if (!first || !last || *last < *first) {
            return {0, volatile_pages.size()};
        }
        const auto by_page = [](const VolatilePage& entry, std::size_t page) {
            return entry.page < page;
        };
        const auto from =
            std::lower_bound(volatile_pages.begin(), volatile_pages.end(), *first, by_page);
        const auto to = std::lower_bound(from, volatile_pages.end(), *last + 1, by_page);
        return {static_cast<std::size_t>(from - volatile_pages.begin()),
                static_cast<std::size_t>(to - volatile_pages.begin())};
    }

    /// Decoder thread, on a list: a read of the stretch. Says so if a volatile page of it
    /// came without a copy.
    void NoteUnheldRead(PAddr addr, u32 size) {
        const auto first = CellPage(addr);
        const auto last = CellPage(addr + size - 1);
        if (!first || !last || *last < *first) {
            return;
        }
        for (std::size_t page = *first; page <= *last; page++) {
            if (volatile_flag[page].load(std::memory_order_relaxed) == 0 ||
                (snapshot_pages != nullptr && snapshot_slot[page] != 0)) {
                continue;
            }
            unheld_reads.fetch_add(1, std::memory_order_relaxed);
            if (volatile_flag[page].exchange(2, std::memory_order_relaxed) == 1) {
                std::scoped_lock lock{reference_mutex};
                unheld_pages.push_back(static_cast<u32>(page));
                has_unheld.store(true, std::memory_order_relaxed);
            }
        }
    }
    /// Those of them a command gets copies of, while they are made
    std::vector<VolatilePage*> snapshot_wanted;
    u64 snapshot_bytes_copied{};
    /// Decoder thread: the copies that came with the command it is on. For each page its
    /// place among them plus one, or zero.
    const std::vector<u32>* snapshot_pages{};
    const u8* snapshot_bytes{};
    std::vector<u16> snapshot_slot;
    /// Which of them it read, and the sums of that for the figures
    std::vector<u8> snapshot_read;
    std::atomic<u64> snapshot_given_sum{};
    std::atomic<u64> snapshot_read_sum{};
    /// The number of the command with copies it is on, and for each page that of the last
    /// one that read its copy (0: none did). The emulation thread looks for the figures.
    std::atomic<u32> snapshot_turn{1};
    std::unique_ptr<std::atomic<u32>[]> snapshot_last_read;
    /// What it read across copied and other pages, put together
    std::deque<std::vector<u8>> stitched;

    static constexpr std::size_t MAX_VOLATILE_PAGES = 1024;
    /// When a volatile page is armed again. Most are written once and never again: a game
    /// that loads writes new data over what the decoder read from before, and a page that
    /// was copied for a quarter of a minute after that (the rule before this one) was 1.9 MB
    /// copied every VBlank after a change of rooms in Resident Evil: Revelations, up to 9 MB
    /// while it loaded, and 15% of the emulation thread. So a page is looked at every
    /// LOOK_EVERY VBlanks and armed as soon as one look finds it as the look before. One that
    /// is written again soon after takes twice as many looks the next time, up to MOST_LOOKS
    /// (a quarter of a minute); one that stood armed for CALM_AFTER VBlanks starts anew. What
    /// a game writes every now and then is so waited for a few times, then copied.
    static constexpr u64 LOOK_EVERY = 16;
    static constexpr u32 MOST_LOOKS = 64;
    static constexpr u64 CALM_AFTER = 900;
    /// For each page the looks it took the last time, and the VBlank it was armed again at
    /// (0: it was not)
    std::vector<u8> looks_taken;
    std::vector<u64> armed_again_at;
    /// Since the figures were last asked for: pages armed again, and pages written (so
    /// volatile again) that had been
    u32 armed_again_count{};
    u32 written_again_count{};

    /// A volatile page of which the decoder has read no copy for this many VBlanks (a quarter
    /// of a minute) is forgotten: neither copied nor armed any more, as if the decoder had
    /// never read from it, and learnt anew if it reads from it again. The decoder reads
    /// textures from a stretch of memory once, the game later uses that memory for something
    /// it writes all the time, and the page was copied at every hand-over from then on: its
    /// contents never stand still, which was the only way off the list. (Kid Icarus:
    /// Uprising, ten seconds after a state was loaded: 192 of 387 pages, half of 7.7 MB
    /// copied every VBlank.)
    static constexpr u64 FORGET_AFTER = 900;
    /// Emulation thread to decoder, under reference_mutex: pages forgotten, whose noting the
    /// decoder takes back (it notes a page once)
    std::vector<u32> forgotten_pages;
    std::atomic<bool> has_forgotten{false};
    /// Since the figures were last asked for
    u32 forgotten_count{};

    void AddVolatile(std::size_t page) {
        if ((reference_state[page] & 32) != 0 || volatile_pages.size() >= MAX_VOLATILE_PAGES) {
            return;
        }
        reference_state[page] |= 32;
        volatile_flag[page].store(1, std::memory_order_relaxed);
        volatile_count.fetch_add(1, std::memory_order_relaxed);
        VolatilePage entry{};
        entry.page = static_cast<u32>(page);
        entry.read_turn = snapshot_last_read[page].load(std::memory_order_relaxed);
        entry.read_vblank = late_vblank;
        // (Pages become volatile by the hundred, and are not all looked at at once)
        entry.next_look = late_vblank + (page & 7);
        if (armed_again_at[page] != 0) {
            written_again_count++;
            const bool calm = late_vblank - armed_again_at[page] > CALM_AFTER;
            entry.looks_needed = calm ? 1 : std::min<u32>(looks_taken[page] * 2u, MOST_LOOKS);
            armed_again_at[page] = 0;
        }
        looks_taken[page] = static_cast<u8>(entry.looks_needed);
        volatile_pages.insert(std::lower_bound(volatile_pages.begin(), volatile_pages.end(), entry,
                                               [](const VolatilePage& a, const VolatilePage& b) {
                                                   return a.page < b.page;
                                               }),
                              entry);
    }

    /// Decoder thread: stretches it has noted, each at its place in a small table. What is
    /// noted is never taken back, so one that is found here has nothing left to note.
    struct NotedStretch {
        u64 stretch{};
        u32 what{};
    };
    std::array<NotedStretch, 4096> noted_stretches{};

    /// Decoder thread: notes the pages of a stretch as read (bit 1 or 16) or written (bit 2)
    void NoteGpuPages(PAddr addr, u32 size, u8 bit, GpuReference kind) {
        if (!track_references.load(std::memory_order_relaxed) || size == 0) {
            return;
        }
        // A draw notes the same stretches frame after frame
        const u64 stretch = u64{addr} << 32 | size;
        const u32 what = u32{bit} << 8 | static_cast<u32>(kind);
        NotedStretch& noted = noted_stretches[(stretch * 0x9E3779B97F4A7C15ULL) >> 52];
        if (noted.stretch == stretch && noted.what == what) {
            return;
        }
        noted = {stretch, what};
        const u64 last = u64{addr} + size - 1;
        for (u64 at = addr & ~CITRA_PAGE_MASK; at <= last; at += CITRA_PAGE_SIZE) {
            const auto page = CellPage(static_cast<PAddr>(at));
            if (!page) {
                continue;
            }
            if (bit != 2) {
                const u32 first = at < addr ? addr & CITRA_PAGE_MASK : 0;
                const u32 end = at + CITRA_PAGE_SIZE - 1 > last
                                    ? static_cast<u32>(last) & CITRA_PAGE_MASK
                                    : CITRA_PAGE_MASK;
                reference_cells[*page] |= CellMask(first, end);
                reference_kind[*page] = static_cast<u8>(kind);
            }
            if ((reference_noted[*page] & bit) != 0) {
                continue;
            }
            reference_noted[*page] |= bit;
            std::scoped_lock lock{reference_mutex};
            (bit == 1   ? new_references
             : bit == 2 ? new_written
                        : new_transfer_references)
                .push_back(static_cast<u32>(*page));
        }
    }

    /// What an access that reaches the rasterizer is to: for telling apart what the game
    /// waits for the decoder for
    VideoCore::GPU::WaitReason AccessKind(VAddr vaddr, PAddr paddr, u32 size, FlushMode mode) {
        using Reason = VideoCore::GPU::WaitReason;
        const bool write = mode != FlushMode::Flush;
        Reason found = Reason::MemoryAccess;
        u32 done = 0;
        while (done < size) {
            const VAddr v = vaddr + done;
            const u32 in_page = (paddr + done) & CITRA_PAGE_MASK;
            const u32 take = std::min(size - done, CITRA_PAGE_SIZE - in_page);
            const auto page = CellPage(paddr + done);
            done += take;
            if (shut_marker.IsCached(v)) {
                return write ? Reason::WriteGpuWritten : Reason::ReadGpuWritten;
            }
            if (!write) {
                if (cache_marker.IsCached(v) && !readable_marker.IsCached(v)) {
                    return Reason::ReadGpuNewer;
                }
                if (watch_marker.IsCached(v)) {
                    found = Reason::ReadWatched;
                }
                continue;
            }
            const u64 mask = CellMask(in_page, in_page + take - 1);
            if (page && (surface_cells[*page] & mask) != 0) {
                return Reason::WriteSurface;
            }
            const bool armed = page && armed_marker.IsCached(v);
            if (armed && (reference_cells[*page] & mask) != 0) {
                switch (static_cast<GpuReference>(reference_kind[*page])) {
                case GpuReference::VertexData:
                    return Reason::WriteVertexData;
                case GpuReference::IndexData:
                    return Reason::WriteIndexData;
                case GpuReference::Texture:
                    return Reason::WriteTexture;
                case GpuReference::CalledList:
                    return Reason::WriteCalledList;
                default:
                    return Reason::WriteOtherReference;
                }
            }
            if (watch_marker.IsCached(v)) {
                found = Reason::WriteWatched;
            } else if (armed && found == Reason::MemoryAccess) {
                found = Reason::WriteBesideReference;
            }
        }
        return found;
    }

    /// Whether a read goes straight to a cached page's memory: nothing newer is held of it,
    /// and nothing that is still to be decoded writes it
    bool ReadsDirect(VAddr vaddr) {
        return readable_marker.IsCached(vaddr) && !shut_marker.IsCached(vaddr);
    }

    /// Stretches the rasterizer gives the game late, see MemorySystem::ExemptReads
    /// (emulation thread), and the bytes on their way (under the mutex)
    struct LateRange {
        PAddr addr{};
        u32 size{};
        u64 last_vblank{};
    };
    std::vector<LateRange> late_ranges;
    u64 late_vblank{};
    std::mutex late_mutex;
    std::vector<std::pair<PAddr, std::vector<u8>>> late_writes;
    std::atomic<bool> has_late_writes{false};
    /// VBlanks without a late write after which a stretch is read the usual way again
    static constexpr u64 LATE_RANGE_VBLANKS = 12;

    bool IsExemptRead(PAddr paddr, u32 size) const {
        for (const LateRange& range : late_ranges) {
            if (paddr >= range.addr && u64{paddr} + size <= u64{range.addr} + range.size &&
                late_vblank - range.last_vblank <= LATE_RANGE_VBLANKS) {
                return true;
            }
        }
        return false;
    }

    /// Whether the rasterizer has to be asked before `size` bytes are read. Not for pages
    /// that are only armed: what the decoder reads from them is what is in memory.
    bool ReadNeedsAnnouncing(VAddr vaddr, u32 size) {
        if (size == 0) {
            return true;
        }
        const u64 last = u64{vaddr} + size - 1;
        for (u64 at = vaddr & ~CITRA_PAGE_MASK; at <= last; at += CITRA_PAGE_SIZE) {
            const VAddr page = static_cast<VAddr>(at);
            if (cache_marker.IsCached(page) || watch_marker.IsCached(page) ||
                shut_marker.IsCached(page)) {
                return true;
            }
        }
        return false;
    }

    static PAddr CellPageAddress(std::size_t page) {
        const std::size_t fcram_pages = FCRAM_N3DS_SIZE >> CITRA_PAGE_BITS;
        if (page < fcram_pages) {
            return FCRAM_PADDR + static_cast<PAddr>(page << CITRA_PAGE_BITS);
        }
        return VRAM_PADDR + static_cast<PAddr>((page - fcram_pages) << CITRA_PAGE_BITS);
    }
    /// The cells of 64 bytes a cached surface covers, a bit each, a word a page: main memory's
    /// pages, then video memory's. See MemorySystem::RasterizerSetSurfaceCells. Empty until
    /// the first surface is cached.
    std::vector<u64> surface_cells;

    /// The word of surface_cells for the page the address is in, if that is main or video
    /// memory
    static std::optional<std::size_t> CellPage(PAddr paddr) {
        if (paddr >= FCRAM_PADDR && paddr < FCRAM_N3DS_PADDR_END) {
            return static_cast<std::size_t>((paddr - FCRAM_PADDR) >> CITRA_PAGE_BITS);
        }
        if (paddr >= VRAM_PADDR && paddr < VRAM_PADDR_END) {
            return static_cast<std::size_t>((FCRAM_N3DS_SIZE >> CITRA_PAGE_BITS) +
                                            ((paddr - VRAM_PADDR) >> CITRA_PAGE_BITS));
        }
        return std::nullopt;
    }

    /// The bits of the cells that the bytes `first` to `last` of a page lie in
    static u64 CellMask(u32 first, u32 last) {
        return (~u64{0} >> (63 - (last >> 6))) & (~u64{0} << (first >> 6));
    }

    /// Whether the rasterizer has to be told of a write of `size` bytes to a cached page: it
    /// touches what a surface covers, or a page that is watched for what a draw reads in
    /// place, or nothing is known of where it goes
    bool WriteNeedsAnnouncing(VAddr vaddr, PAddr paddr, u32 size) {
        if (surface_cells.empty() || size == 0 ||
            (Common::TraceMarkers::Experiments2() &
             Common::TraceMarkers::LegacyWriteAnnouncements) != 0) {
            return true;
        }
        u32 done = 0;
        while (done < size) {
            const auto page = CellPage(paddr + done);
            if (!page || watch_marker.IsCached(vaddr + done) ||
                armed_marker.IsCached(vaddr + done) || shut_marker.IsCached(vaddr + done)) {
                return true;
            }
            const u32 in_page = (paddr + done) & CITRA_PAGE_MASK;
            const u32 take = std::min(size - done, CITRA_PAGE_SIZE - in_page);
            if ((surface_cells[*page] & CellMask(in_page, in_page + take - 1)) != 0) {
                return true;
            }
            done += take;
        }
        return false;
    }

    /// Whether a read of `size` bytes on cached pages touches what a surface covers (of which
    /// the host GPU may hold something newer), or a page that is watched or that queued
    /// commands write, or nothing is known of where it goes. A surface seldom begins and ends
    /// with a page, and what shares its first and last page is the game's own data: Kid
    /// Icarus: Uprising keeps what it reads 150 times a frame in front of its framebuffer,
    /// and each of those reads waited for the decoding thread and went to the rasterizer to
    /// be told there was nothing to fetch (6% of the emulation thread).
    ///
    /// Only on pages the rasterizer has said it cannot let the game read freely
    /// (holds_newer_pages): a read that is announced is also what lets the rasterizer look at
    /// the page and, with nothing newer anywhere on it, mark it readable, after which reads
    /// of it come here no more. Left unannounced on any page, the reads of pages that could
    /// be made readable all came by the slow way for good (Xenoblade Chronicles: 9% of the
    /// emulation thread). And one in 256 is announced all the same, so that a page whose
    /// surface has since been written back is found out.
    bool ReadTouchesSurface(VAddr vaddr, PAddr paddr, u32 size) {
        if (surface_cells.empty() || holds_newer_pages.empty() || size == 0 ||
            (Common::TraceMarkers::Experiments3() &
             Common::TraceMarkers::LegacyReadAnnouncements) != 0) {
            return true;
        }
        u32 done = 0;
        while (done < size) {
            const auto page = CellPage(paddr + done);
            if (!page || holds_newer_pages[*page] == 0 || watch_marker.IsCached(vaddr + done) ||
                shut_marker.IsCached(vaddr + done)) {
                return true;
            }
            const u32 in_page = (paddr + done) & CITRA_PAGE_MASK;
            const u32 take = std::min(size - done, CITRA_PAGE_SIZE - in_page);
            if ((surface_cells[*page] & CellMask(in_page, in_page + take - 1)) != 0) {
                return true;
            }
            done += take;
        }
        return (++unannounced_reads & 255) == 0;
    }
    /// For each page (as surface_cells): set by the rasterizer when it looked at the page for
    /// a read and found a surface on it holding something newer than memory; cleared when the
    /// page is marked readable. Emulation thread.
    std::vector<u8> holds_newer_pages;
    u32 unannounced_reads{};

    std::vector<std::shared_ptr<PageTable>> page_table_list;

    std::shared_ptr<BackingMem> fcram_mem;
    std::shared_ptr<BackingMem> vram_mem;
    std::shared_ptr<BackingMem> n3ds_extra_ram_mem;
    std::shared_ptr<BackingMem> dsp_mem;

    PAddr plugin_fb_address{};

    Impl(Core::System& system_);
    ~Impl();

    const u8* GetPtr(Region r) const {
        switch (r) {
        case Region::VRAM:
            return vram.get();
        case Region::DSP:
            return dsp_ram.get();
        case Region::FCRAM:
            return fcram.get();
        case Region::N3DS:
            return n3ds_extra_ram.get();
        default:
            UNREACHABLE();
        }
    }

    u8* GetPtr(Region r) {
        switch (r) {
        case Region::VRAM:
            return vram.get();
        case Region::DSP:
            return dsp_ram.get();
        case Region::FCRAM:
            return fcram.get();
        case Region::N3DS:
            return n3ds_extra_ram.get();
        default:
            UNREACHABLE();
        }
    }

    u32 GetSize(Region r) const {
        switch (r) {
        case Region::VRAM:
            return VRAM_SIZE;
        case Region::DSP:
            return DSP_RAM_SIZE;
        case Region::FCRAM:
            return FCRAM_N3DS_SIZE;
        case Region::N3DS:
            return N3DS_EXTRA_RAM_SIZE;
        default:
            UNREACHABLE();
        }
    }

    u32 GetPC() const noexcept {
        return system.GetRunningCore().GetPC();
    }

    template <bool UNSAFE>
    void ReadBlockImpl(const Kernel::Process& process, const VAddr src_addr, void* dest_buffer,
                       const std::size_t size) {
        auto& page_table = *process.vm_manager.page_table;

        std::size_t remaining_size = size;
        std::size_t page_index = src_addr >> CITRA_PAGE_BITS;
        std::size_t page_offset = src_addr & CITRA_PAGE_MASK;

        while (remaining_size > 0) {
            const std::size_t copy_amount = std::min(CITRA_PAGE_SIZE - page_offset, remaining_size);
            const VAddr current_vaddr =
                static_cast<VAddr>((page_index << CITRA_PAGE_BITS) + page_offset);

            switch (page_table.attributes[page_index]) {
            case PageType::Unmapped: {
                LOG_ERROR(
                    HW_Memory,
                    "unmapped ReadBlock @ 0x{:08X} (start address = 0x{:08X}, size = {}) at PC "
                    "0x{:08X}",
                    current_vaddr, src_addr, size, GetPC());
                std::memset(dest_buffer, 0, copy_amount);
                break;
            }
            case PageType::Memory: {
                const u8* page = page_table.pointers[page_index];
                if (!page) [[unlikely]] {
                    // Marked for the rasterizer since the look at its type, by the thread
                    // that decodes GPU commands
                    RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                                 FlushMode::Flush);
                    page = GetPointerForRasterizerCache(current_vaddr & ~CITRA_PAGE_MASK).GetPtr();
                }
                std::memcpy(dest_buffer, page + page_offset, copy_amount);
                break;
            }
            case PageType::MemoryWatchpoint: {
                auto it = page_table.watchpoint_pages_map.find(page_index);
                ASSERT_MSG(it != page_table.watchpoint_pages_map.end(),
                           "Missing memory for watchpoint page");

                const u8* src_ptr = it->second.memory.GetPtr() + page_offset;
                std::memcpy(dest_buffer, src_ptr, copy_amount);
                break;
            }
            case PageType::RasterizerCachedMemory:
            case PageType::RasterizerCachedMemoryWatchpoint: {
                if constexpr (!UNSAFE) {
                    if (!ReadsDirect(current_vaddr)) {
                        RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                                     FlushMode::Flush);
                    }
                }
                std::memcpy(dest_buffer, RasterizerCachePointer(current_vaddr), copy_amount);
                break;
            }
            default:
                UNREACHABLE();
            }

            page_index++;
            page_offset = 0;
            dest_buffer = static_cast<u8*>(dest_buffer) + copy_amount;
            remaining_size -= copy_amount;
        }
    }

    template <bool UNSAFE>
    void WriteBlockImpl(const Kernel::Process& process, const VAddr dest_addr,
                        const void* src_buffer, const std::size_t size) {
        auto& page_table = *process.vm_manager.page_table;
        std::size_t remaining_size = size;
        std::size_t page_index = dest_addr >> CITRA_PAGE_BITS;
        std::size_t page_offset = dest_addr & CITRA_PAGE_MASK;

        while (remaining_size > 0) {
            const std::size_t copy_amount = std::min(CITRA_PAGE_SIZE - page_offset, remaining_size);
            const VAddr current_vaddr =
                static_cast<VAddr>((page_index << CITRA_PAGE_BITS) + page_offset);

            switch (page_table.attributes[page_index]) {
            case PageType::Unmapped: {
                LOG_ERROR(
                    HW_Memory,
                    "unmapped WriteBlock @ 0x{:08X} (start address = 0x{:08X}, size = {}) at PC "
                    "0x{:08X}",
                    current_vaddr, dest_addr, size, GetPC());
                break;
            }
            case PageType::Memory: {
                u8* page = page_table.pointers[page_index];
                if (!page) [[unlikely]] {
                    // Marked for the rasterizer since the look at its type, by the thread
                    // that decodes GPU commands
                    RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                                 FlushMode::Invalidate);
                    page = GetPointerForRasterizerCache(current_vaddr & ~CITRA_PAGE_MASK).GetPtr();
                }
                std::memcpy(page + page_offset, src_buffer, copy_amount);
                break;
            }
            case PageType::MemoryWatchpoint: {
                auto it = page_table.watchpoint_pages_map.find(page_index);
                ASSERT_MSG(it != page_table.watchpoint_pages_map.end(),
                           "Missing memory for watchpoint page");

                u8* dest_ptr = it->second.memory.GetPtr() + page_offset;
                std::memcpy(dest_ptr, src_buffer, copy_amount);
                break;
            }
            case PageType::RasterizerCachedMemory:
            case PageType::RasterizerCachedMemoryWatchpoint: {
                if constexpr (!UNSAFE) {
                    RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                                 FlushMode::Invalidate);
                }
                std::memcpy(RasterizerCachePointer(current_vaddr), src_buffer, copy_amount);
                break;
            }
            default:
                UNREACHABLE();
            }

            page_index++;
            page_offset = 0;
            src_buffer = static_cast<const u8*>(src_buffer) + copy_amount;
            remaining_size -= copy_amount;
        }
    }

    /// As GetPointerForRasterizerCache, the bare pointer: no count of the memory's users is
    /// taken, which is two atomic operations on a way a game can go ten thousand times a
    /// frame
    u8* RasterizerCachePointer(VAddr addr) const {
        if (addr >= LINEAR_HEAP_VADDR && addr < LINEAR_HEAP_VADDR_END) [[likely]] {
            return fcram_mem->GetPtr() + (addr - LINEAR_HEAP_VADDR);
        }
        if (addr >= NEW_LINEAR_HEAP_VADDR && addr < NEW_LINEAR_HEAP_VADDR_END) {
            return fcram_mem->GetPtr() + (addr - NEW_LINEAR_HEAP_VADDR);
        }
        if (addr >= VRAM_VADDR && addr < VRAM_VADDR_END) {
            return vram_mem->GetPtr() + (addr - VRAM_VADDR);
        }
        if (addr >= PLUGIN_3GX_FB_VADDR && addr < PLUGIN_3GX_FB_VADDR_END && plugin_fb_address) {
            return fcram_mem->GetPtr() +
                   (addr - PLUGIN_3GX_FB_VADDR + plugin_fb_address - FCRAM_PADDR);
        }
        UNREACHABLE();
        return nullptr;
    }

    MemoryRef GetPointerForRasterizerCache(VAddr addr) const {
        if (addr >= LINEAR_HEAP_VADDR && addr < LINEAR_HEAP_VADDR_END) {
            return {fcram_mem, addr - LINEAR_HEAP_VADDR};
        }
        if (addr >= NEW_LINEAR_HEAP_VADDR && addr < NEW_LINEAR_HEAP_VADDR_END) {
            return {fcram_mem, addr - NEW_LINEAR_HEAP_VADDR};
        }
        if (addr >= VRAM_VADDR && addr < VRAM_VADDR_END) {
            return {vram_mem, addr - VRAM_VADDR};
        }
        if (addr >= PLUGIN_3GX_FB_VADDR && addr < PLUGIN_3GX_FB_VADDR_END && plugin_fb_address) {
            return {fcram_mem, addr - PLUGIN_3GX_FB_VADDR + plugin_fb_address - FCRAM_PADDR};
        }

        UNREACHABLE();
        return MemoryRef{};
    }

    void RasterizerFlushVirtualRegion(VAddr start, u32 size, FlushMode mode) {
        const VAddr end = start + size;

        auto CheckRegion = [&](VAddr region_start, VAddr region_end, PAddr paddr_region_start) {
            if (start >= region_end || end <= region_start) {
                // No overlap with region
                return;
            }

            auto& renderer = system.GPU().Renderer();
            VAddr overlap_start = std::max(start, region_start);
            VAddr overlap_end = std::min(end, region_end);
            PAddr physical_start = paddr_region_start + (overlap_start - region_start);
            u32 overlap_size = overlap_end - overlap_start;

            // A write that touches nothing a surface covers is nobody's business
            if (mode == FlushMode::Invalidate &&
                !WriteNeedsAnnouncing(overlap_start, physical_start, overlap_size)) {
                return;
            }
            // Nor is a read of pages that are only armed, or not marked at all
            if (mode == FlushMode::Flush && !ReadNeedsAnnouncing(overlap_start, overlap_size)) {
                return;
            }
            // Nor one of what merely shares a page with a surface
            if (mode == FlushMode::Flush &&
                !ReadTouchesSurface(overlap_start, physical_start, overlap_size)) {
                return;
            }
            // Nor one of what the rasterizer gives the game late: memory as it is
            if (mode == FlushMode::Flush && !late_ranges.empty() &&
                IsExemptRead(physical_start, overlap_size)) {
                return;
            }

            // The rasterizer is about to be used from here: not while the thread that decodes
            // commands is at it
            const VideoCore::GPU::WaitReason kind =
                AccessKind(overlap_start, physical_start, overlap_size, mode);
            if (kind == VideoCore::GPU::WaitReason::WriteBesideReference) {
                // On a page the decoder reads from, but not what it reads of it: nothing the
                // decoder or the rasterizer has to know of. While the decoder has work the
                // page stays as it is, for what it does read of it; otherwise the page is
                // free until commands are handed over again.
                system.GPU().NoteAccess(kind);
                if (!system.GPU().HasPendingWork()) {
                    system.Memory().Disarm(physical_start, overlap_size, true);
                } else {
                    system.Memory().WantVolatile(physical_start, overlap_size);
                }
                return;
            }
            system.GPU().WaitIdle(kind);

            auto* rasterizer = renderer.Rasterizer();
            Common::Hitches::NoteGpuCall(mode == FlushMode::Flush
                                             ? Common::Hitches::GpuCall::MemoryRead
                                             : Common::Hitches::GpuCall::MemoryWrite);
            switch (mode) {
            // The decoder has been waited for: what is read or written here is free to be
            // until commands are handed over again
            case FlushMode::Flush:
                rasterizer->FlushRegion(physical_start, overlap_size);
                system.Memory().Disarm(physical_start, overlap_size, false);
                break;
            case FlushMode::Invalidate:
                rasterizer->InvalidateRegion(physical_start, overlap_size);
                system.Memory().Disarm(physical_start, overlap_size, true);
                break;
            case FlushMode::FlushAndInvalidate:
                rasterizer->FlushAndInvalidateRegion(physical_start, overlap_size);
                system.Memory().Disarm(physical_start, overlap_size, true);
                break;
            }
        };

        CheckRegion(LINEAR_HEAP_VADDR, LINEAR_HEAP_VADDR_END, FCRAM_PADDR);
        CheckRegion(NEW_LINEAR_HEAP_VADDR, NEW_LINEAR_HEAP_VADDR_END, FCRAM_PADDR);
        CheckRegion(VRAM_VADDR, VRAM_VADDR_END, VRAM_PADDR);
        if (plugin_fb_address) {
            CheckRegion(PLUGIN_3GX_FB_VADDR, PLUGIN_3GX_FB_VADDR_END, plugin_fb_address);
        }
    }

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        bool save_n3ds_ram = Settings::values.is_new_3ds.GetValue();
        ar & save_n3ds_ram;
        ar& boost::serialization::make_binary_object(vram.get(), Memory::VRAM_SIZE);
        ar& boost::serialization::make_binary_object(
            fcram.get(), save_n3ds_ram ? Memory::FCRAM_N3DS_SIZE : Memory::FCRAM_SIZE);
        ar& boost::serialization::make_binary_object(
            n3ds_extra_ram.get(), save_n3ds_ram ? Memory::N3DS_EXTRA_RAM_SIZE : 0);
        ar& boost::serialization::make_binary_object(dsp_ram.get(), Memory::DSP_RAM_SIZE);
        ar & cache_marker;
        ar & page_table_list;
        // dsp is set from Core::System at startup
        ar & current_page_table;
        ar & fcram_mem;
        ar & vram_mem;
        ar & n3ds_extra_ram_mem;
        ar & dsp_mem;
        ar & plugin_fb_address;
    }
};

// We use this rather than BufferMem because we don't want new objects to be allocated when
// deserializing. This avoids unnecessary memory thrashing.
template <Region R>
class MemorySystem::BackingMemImpl : public BackingMem {
public:
    BackingMemImpl() : impl(*Core::Global<Core::System>().Memory().impl) {}
    explicit BackingMemImpl(MemorySystem::Impl& impl_) : impl(impl_) {}
    u8* GetPtr() override {
        return impl.GetPtr(R);
    }
    const u8* GetPtr() const override {
        return impl.GetPtr(R);
    }
    std::size_t GetSize() const override {
        return impl.GetSize(R);
    }

private:
    MemorySystem::Impl& impl;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar& boost::serialization::base_object<BackingMem>(*this);
    }
    friend class boost::serialization::access;
};

MemorySystem::Impl::Impl(Core::System& system_)
    : system{system_}, fcram_mem(std::make_shared<BackingMemImpl<Region::FCRAM>>(*this)),
      vram_mem(std::make_shared<BackingMemImpl<Region::VRAM>>(*this)),
      n3ds_extra_ram_mem(std::make_shared<BackingMemImpl<Region::N3DS>>(*this)),
      dsp_mem(std::make_shared<BackingMemImpl<Region::DSP>>(*this)) {
    // Looked at on one thread while another sets them: never grown later
    surface_cells.resize((FCRAM_N3DS_SIZE + VRAM_SIZE) >> CITRA_PAGE_BITS);
    reference_noted.resize(surface_cells.size());
    reference_kind.resize(surface_cells.size());
    reference_cells.resize(surface_cells.size());
    reference_state.resize(surface_cells.size());
    looks_taken.resize(surface_cells.size());
    armed_again_at.resize(surface_cells.size());
    snapshot_slot.resize(surface_cells.size());
    snapshot_last_read = std::make_unique<std::atomic<u32>[]>(surface_cells.size());
    volatile_flag = std::make_unique<std::atomic<u8>[]>(surface_cells.size());
    // The page tables made from here on get their views of it
    if (WantsHostMmu()) {
        host_backing = backing.get();
    }
}

MemorySystem::Impl::~Impl() {
    if (backing && host_backing == backing.get()) {
        host_backing = nullptr;
    }
}

MemorySystem::MemorySystem(Core::System& system) : impl(std::make_unique<Impl>(system)) {}
MemorySystem::~MemorySystem() = default;

template <class Archive>
void MemorySystem::serialize(Archive& ar, const unsigned int file_version) {
    ar&* impl.get();
}

SERIALIZE_IMPL(MemorySystem)

void MemorySystem::SetCurrentPageTable(const std::shared_ptr<PageTable>& page_table) {
    if (impl->current_page_table != page_table) {
        impl->current_page_table = page_table;
    }
}

std::shared_ptr<PageTable> MemorySystem::GetCurrentPageTable() const {
    return impl->current_page_table;
}

bool MemorySystem::UsesHostMmu() const {
    return impl->backing != nullptr && host_backing == impl->backing.get();
}

std::span<u8> MemorySystem::GetHostPool() const {
    if (!impl->backing) {
        return {};
    }
    return {impl->backing->Data(), impl->backing->Size()};
}

void MemorySystem::RasterizerFlushVirtualRegion(VAddr start, u32 size, FlushMode mode) {
    impl->RasterizerFlushVirtualRegion(start, size, mode);
}

PAddr& Memory::MemorySystem::Plugin3GXFramebufferAddress() {
    return impl->plugin_fb_address;
}

void MemorySystem::RegisterWatchpoint(const Kernel::Process& process, VAddr addr, u32 size) {
    auto& page_table = *process.vm_manager.page_table;

    VAddr current = addr;
    VAddr end = addr + size;

    while (current < end) {
        const VAddr page_base = (current & ~CITRA_PAGE_MASK);
        const VAddr page_index = page_base >> CITRA_PAGE_BITS;

        auto it = page_table.watchpoint_pages_map.find(page_index);
        if (it != page_table.watchpoint_pages_map.end()) {
            // Nothing to do, only increment count.
            it->second.watchpoint_count++;
        } else {
            MemoryRef mem;
            PageType& type = page_table.attributes[page_index];

            switch (type) {
            case PageType::Memory:
                mem = page_table.pointers.Ref(page_index);
                type = PageType::MemoryWatchpoint;
                page_table.pointers[page_index] = nullptr;
                break;
            case PageType::RasterizerCachedMemory:
                mem = GetPointerForRasterizerCache(page_base);
                type = PageType::RasterizerCachedMemoryWatchpoint;
                break;
            default:
                LOG_ERROR(HW_Memory, "Cannot get pointer to register watchpoint for page 0x{:08X}",
                          page_base);
                continue;
            }

            page_table.watchpoint_pages_map.insert(
                {page_index,
                 PageTable::WatchpointPageInfo{.watchpoint_count = 1, .memory = std::move(mem)}});
        }

        current = page_base + CITRA_PAGE_SIZE;
    }
}

void MemorySystem::UnregisterWatchpoint(const Kernel::Process& process, VAddr addr, u32 size) {
    auto& page_table = *process.vm_manager.page_table;

    VAddr current = addr;
    VAddr end = addr + size;

    while (current < end) {
        const VAddr page_base = (current & ~CITRA_PAGE_MASK);
        const VAddr page_index = page_base >> CITRA_PAGE_BITS;

        auto it = page_table.watchpoint_pages_map.find(page_index);
        if (it != page_table.watchpoint_pages_map.end()) {
            if (--it->second.watchpoint_count == 0) {

                PageType& type = page_table.attributes[page_index];

                switch (type) {
                case PageType::MemoryWatchpoint:
                    type = PageType::Memory;
                    page_table.pointers[page_index] = it->second.memory;
                    break;
                case PageType::RasterizerCachedMemoryWatchpoint:
                    type = PageType::RasterizerCachedMemory;
                    break;
                default:
                    LOG_ERROR(HW_Memory, "Invalid watchpoint page type for page 0x{:08X}: {}",
                              page_base, static_cast<u8>(type));
                }

                page_table.watchpoint_pages_map.erase(page_index);
            }
        } else {
            LOG_ERROR(HW_Memory, "No watchpoint found on page 0x{:08X}", page_base);
        }

        current = page_base + CITRA_PAGE_SIZE;
    }
}

void MemorySystem::MapPages(PageTable& page_table, u32 base, u32 size, MemoryRef memory,
                            PageType type) {
    LOG_DEBUG(HW_Memory, "Mapping {} onto {:08X}-{:08X}", (void*)memory.GetPtr(),
              base * CITRA_PAGE_SIZE, (base + size) * CITRA_PAGE_SIZE);

    if (impl->system.IsPoweredOn()) {
        // The page tables are about to change: not while the thread that decodes commands
        // marks pages in them
        impl->system.GPU().WaitIdle(VideoCore::GPU::WaitReason::MapPages);
        RasterizerFlushVirtualRegion(base << CITRA_PAGE_BITS, size * CITRA_PAGE_SIZE,
                                     FlushMode::FlushAndInvalidate);
    }

    // The host's view of the address space follows in runs of pages, not page by page
    std::scoped_lock mark_lock{impl->mark_mutex};
    page_table.pointers.BeginChanges();
    u32 end = base + size;
    while (base != end) {
        ASSERT_MSG(base < PAGE_TABLE_NUM_ENTRIES, "out of range mapping at {:08X}", base);

        page_table.attributes[base] = type;
        page_table.pointers[base] = memory;

        // If the memory to map is already rasterizer-cached, mark the page
        const bool cached = impl->cache_marker.IsCached(base * CITRA_PAGE_SIZE);
        const bool shut = impl->shut_marker.IsCached(base * CITRA_PAGE_SIZE);
        const bool watched = impl->watch_marker.IsCached(base * CITRA_PAGE_SIZE) ||
                             impl->armed_marker.IsCached(base * CITRA_PAGE_SIZE);
        if (type == PageType::Memory && (cached || watched || shut)) {
            page_table.attributes[base] = PageType::RasterizerCachedMemory;
            page_table.pointers[base] = nullptr;
            const bool read_through =
                !shut &&
                (cached ? impl->readable_marker.IsCached(base * CITRA_PAGE_SIZE) : watched);
            page_table.pointers.SetReadOnly(base, read_through ? memory.GetPtr() : nullptr);
        } else {
            page_table.pointers.SetReadOnly(base, nullptr);
        }

        base += 1;
        if (memory != nullptr && memory.GetSize() > CITRA_PAGE_SIZE)
            memory += CITRA_PAGE_SIZE;
    }
    page_table.pointers.EndChanges();
}

void MemorySystem::MapMemoryRegion(PageTable& page_table, VAddr base, u32 size, MemoryRef target) {
    ASSERT_MSG((size & CITRA_PAGE_MASK) == 0, "non-page aligned size: {:08X}", size);
    ASSERT_MSG((base & CITRA_PAGE_MASK) == 0, "non-page aligned base: {:08X}", base);
    MapPages(page_table, base / CITRA_PAGE_SIZE, size / CITRA_PAGE_SIZE, target, PageType::Memory);
}

void MemorySystem::UnmapRegion(PageTable& page_table, VAddr base, u32 size) {
    ASSERT_MSG((size & CITRA_PAGE_MASK) == 0, "non-page aligned size: {:08X}", size);
    ASSERT_MSG((base & CITRA_PAGE_MASK) == 0, "non-page aligned base: {:08X}", base);
    MapPages(page_table, base / CITRA_PAGE_SIZE, size / CITRA_PAGE_SIZE, nullptr,
             PageType::Unmapped);
}

MemoryRef MemorySystem::GetPointerForRasterizerCache(VAddr addr) const {
    return impl->GetPointerForRasterizerCache(addr);
}

void MemorySystem::RegisterPageTable(std::shared_ptr<PageTable> page_table) {
    // The list is gone through by whoever marks pages, on another thread too
    std::scoped_lock mark_lock{impl->mark_mutex};
    impl->page_table_list.push_back(page_table);
}

void MemorySystem::UnregisterPageTable(std::shared_ptr<PageTable> page_table) {
    std::scoped_lock mark_lock{impl->mark_mutex};
    auto it = std::find(impl->page_table_list.begin(), impl->page_table_list.end(), page_table);
    if (it != impl->page_table_list.end()) {
        impl->page_table_list.erase(it);
    }
}

template <typename T>
void MemorySystem::UnmappedAccess(const VAddr vaddr, const T value, bool read) {
    const std::string mode = (read ? "Read" : "Write");
    const std::string value_str = read ? std::string("") : fmt::format(" 0x{:08X}", value);
    const std::string message = fmt::format("unmapped {}{}{} @ 0x{:08X} at PC 0x{:08X}", mode,
                                            sizeof(T) * 8, value_str, vaddr, impl->GetPC());
#ifdef ENABLE_GDBSTUB
    if (GDBStub::IsConnected()) {
        GDBStub::Break(SIGSEGV);
    }
#endif
    LOG_ERROR(HW_Memory, "{}", message);
    if (Settings::values.enable_exception_handler) {
        Core::LogException(impl->system, read ? Core::ExceptionType::UnmappedRead
                                              : Core::ExceptionType::UnmappedWrite);
    }
}

template <typename T>
T MemorySystem::Read(const std::shared_ptr<PageTable>& page_table, const VAddr vaddr) {
    constexpr bool is_optional = is_optional_type<T>;
    using ReadType = optional_inner_or_type<T>;

    constexpr size_t read_size = sizeof(ReadType);

    const u8* page_pointer = page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        // NOTE: Avoid adding any extra logic to this fast-path block
        ReadType value;
        std::memcpy(&value, &page_pointer[vaddr & CITRA_PAGE_MASK], read_size);
        return value;
    }

    // Custom Luma3ds mapping (bit 31 set) Bypasses page tables for FCRAM/MMIO
    constexpr VAddr LUMA_ALIAS_BIT = 0x80000000u;

    if (vaddr & LUMA_ALIAS_BIT) [[unlikely]] {
        const PAddr paddr = vaddr & ~LUMA_ALIAS_BIT;

        // FCRAM (0x2xxxxxxx)
        if ((paddr & 0xF0000000) == Memory::FCRAM_PADDR) {
            ReadType value;
            std::memcpy(&value, GetFCRAMPointer(paddr - Memory::FCRAM_PADDR), read_size);
            return value;
        }

        // MMIO (0x1xxxxxxx, >= IO_AREA_PADDR) - Strictly 32-bit
        if ((paddr & 0xF0000000) == 0x10000000 && paddr >= Memory::IO_AREA_PADDR) [[unlikely]] {
            return static_cast<ReadType>(impl->system.GPU().ReadReg(
                static_cast<VAddr>(paddr) - Memory::IO_AREA_PADDR + 0x1EC00000));
        }
        // Fallthrough: Standard page table lookup
    }

    PageType type = page_table->attributes[vaddr >> CITRA_PAGE_BITS];
    switch (type) {
    case PageType::Unmapped: {

        UnmappedAccess<ReadType>(vaddr, 0, true);

        if constexpr (is_optional) {
            return std::nullopt;
        } else {
            return T{};
        }
    }
    case PageType::Memory:
        if (!impl->IsPlainWithoutPointer(*page_table, vaddr)) {
            return Read<T>(page_table, vaddr);
        }
        ASSERT_MSG(false, "Mapped memory page without a pointer @ {:08X}", vaddr);
        break;
    case PageType::MemoryWatchpoint: {
        auto it = page_table->watchpoint_pages_map.find(vaddr >> CITRA_PAGE_BITS);
        ASSERT_MSG(it != page_table->watchpoint_pages_map.end(),
                   "Missing memory for watchpoint page");

        ReadType value;
        std::memcpy(&value, it->second.memory.GetPtr() + (vaddr & CITRA_PAGE_MASK), read_size);

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, read_size, GDBStub::BreakpointType::Read)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        return value;
    }
    [[likely]] case PageType::RasterizerCachedMemory: {
        // Not for a page the rasterizer holds nothing newer of
        // (nor for one that is only armed: asked here, not after the whole way to the check)
        if ((!impl->ReadsDirect(vaddr) || !impl->ReadsDirect(vaddr + read_size - 1)) &&
            impl->ReadNeedsAnnouncing(vaddr, read_size)) {
            RasterizerFlushVirtualRegion(vaddr, read_size, FlushMode::Flush);
        }

        ReadType value;
        std::memcpy(&value, impl->RasterizerCachePointer(vaddr), read_size);
        return value;
    }
    case PageType::RasterizerCachedMemoryWatchpoint: {
        RasterizerFlushVirtualRegion(vaddr, read_size, FlushMode::Flush);

        ReadType value;
        std::memcpy(&value, impl->RasterizerCachePointer(vaddr), read_size);

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, read_size, GDBStub::BreakpointType::Read)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        return value;
    }
    default:
        UNREACHABLE();
    }

    if constexpr (is_optional) {
        return std::nullopt;
    } else {
        return T{};
    }
}

template <typename T>
void MemorySystem::Write(const std::shared_ptr<PageTable>& page_table, const VAddr vaddr,
                         const T data) {
    u8* page_pointer = page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        // NOTE: Avoid adding any extra logic to this fast-path block
        std::memcpy(&page_pointer[vaddr & CITRA_PAGE_MASK], &data, sizeof(T));
        return;
    }

    // Custom Luma3ds mapping (bit 31 set) Bypasses page tables for FCRAM/MMIO
    constexpr VAddr LUMA_ALIAS_BIT = 0x80000000u;

    if (vaddr & LUMA_ALIAS_BIT) [[unlikely]] {
        const PAddr paddr = vaddr & ~LUMA_ALIAS_BIT;

        // FCRAM (0x2xxxxxxx)
        if ((paddr & 0xF0000000) == Memory::FCRAM_PADDR) {
            std::memcpy(GetFCRAMPointer(paddr - Memory::FCRAM_PADDR), &data, sizeof(T));
            return;
        }

        // MMIO (0x1xxxxxxx, >= IO_AREA_PADDR) - Strictly 32-bit
        if ((paddr & 0xF0000000) == 0x10000000 && paddr >= Memory::IO_AREA_PADDR) [[unlikely]] {
            ASSERT(sizeof(data) == sizeof(u32));
            impl->system.GPU().WriteReg(static_cast<VAddr>(paddr) - Memory::IO_AREA_PADDR +
                                            0x1EC00000,
                                        static_cast<u32>(data));
            return;
        }
        // Fallthrough: Standard page table lookup
    }

    PageType type = page_table->attributes[vaddr >> CITRA_PAGE_BITS];
    switch (type) {
    case PageType::Unmapped:
        (void)UnmappedAccess<T>(vaddr, data, false);
        return;
    case PageType::Memory:
        if (!impl->IsPlainWithoutPointer(*page_table, vaddr)) {
            Write<T>(page_table, vaddr, data);
            return;
        }
        ASSERT_MSG(false, "Mapped memory page without a pointer @ {:08X}", vaddr);
        break;
    case PageType::MemoryWatchpoint: {
        auto it = page_table->watchpoint_pages_map.find(vaddr >> CITRA_PAGE_BITS);
        ASSERT_MSG(it != page_table->watchpoint_pages_map.end(),
                   "Missing memory for watchpoint page");

        std::memcpy(it->second.memory.GetPtr() + (vaddr & CITRA_PAGE_MASK), &data, sizeof(T));

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, sizeof(T), GDBStub::BreakpointType::Write)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        break;
    }
    [[likely]] case PageType::RasterizerCachedMemory: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Invalidate);
        std::memcpy(impl->RasterizerCachePointer(vaddr), &data, sizeof(T));
        break;
    }
    case PageType::RasterizerCachedMemoryWatchpoint: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Invalidate);
        std::memcpy(impl->RasterizerCachePointer(vaddr), &data, sizeof(T));

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, sizeof(T), GDBStub::BreakpointType::Write)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        break;
    }
    default:
        UNREACHABLE();
    }
}

template <typename T>
bool MemorySystem::WriteExclusive(const VAddr vaddr, const T data, const T expected) {
    u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];

    if (page_pointer) {
        const auto volatile_pointer =
            reinterpret_cast<volatile T*>(&page_pointer[vaddr & CITRA_PAGE_MASK]);
        return Common::AtomicCompareAndSwap(volatile_pointer, data, expected);
    }

    PageType type = impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS];
    switch (type) {
    case PageType::Unmapped:
        (void)UnmappedAccess<T>(vaddr, data, false);
        return true;
    case PageType::Memory:
        if (!impl->IsPlainWithoutPointer(*impl->current_page_table, vaddr)) {
            return WriteExclusive<T>(vaddr, data, expected);
        }
        ASSERT_MSG(false, "Mapped memory page without a pointer @ {:08X}", vaddr);
        return true;
    case PageType::MemoryWatchpoint: {
        auto it = impl->current_page_table->watchpoint_pages_map.find(vaddr >> CITRA_PAGE_BITS);
        ASSERT_MSG(it != impl->current_page_table->watchpoint_pages_map.end(),
                   "Missing memory for watchpoint page");

        const auto volatile_pointer =
            reinterpret_cast<volatile T*>(it->second.memory.GetPtr() + (vaddr & CITRA_PAGE_MASK));

        bool ret = Common::AtomicCompareAndSwap(volatile_pointer, data, expected);

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, sizeof(T), GDBStub::BreakpointType::Write)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        return ret;
    }
    [[likely]] case PageType::RasterizerCachedMemory: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Invalidate);
        const auto volatile_pointer =
            reinterpret_cast<volatile T*>(impl->RasterizerCachePointer(vaddr));
        return Common::AtomicCompareAndSwap(volatile_pointer, data, expected);
    }
    case PageType::RasterizerCachedMemoryWatchpoint: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Invalidate);
        const auto volatile_pointer =
            reinterpret_cast<volatile T*>(impl->RasterizerCachePointer(vaddr));

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, sizeof(T), GDBStub::BreakpointType::Write)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        return Common::AtomicCompareAndSwap(volatile_pointer, data, expected);
    }
    default:
        UNREACHABLE();
    }
    return true;
}

bool MemorySystem::IsValidVirtualAddress(const Kernel::Process& process, const VAddr vaddr) {
    auto& page_table = *process.vm_manager.page_table;

    auto page_pointer = page_table.pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        return true;
    }

    if (page_table.attributes[vaddr >> CITRA_PAGE_BITS] != PageType::Unmapped) {
        return true;
    }

    return false;
}

bool MemorySystem::IsValidPhysicalAddress(const PAddr paddr) {
    return GetPhysicalRef(paddr);
}

u8* MemorySystem::GetPointer(const VAddr vaddr) {
    u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        return page_pointer + (vaddr & CITRA_PAGE_MASK);
    }

    if (impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
            PageType::RasterizerCachedMemory ||
        impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
            PageType::RasterizerCachedMemoryWatchpoint) {
        return GetPointerForRasterizerCache(vaddr);
    }

    // Its marks were changed between the two looks at it, by the thread that decodes GPU
    // commands
    if (u8* const again = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS]) {
        return again + (vaddr & CITRA_PAGE_MASK);
    }

    LOG_ERROR(HW_Memory, "unknown GetPointer @ 0x{:08x} at PC 0x{:08X}", vaddr, impl->GetPC());
    return nullptr;
}

const u8* MemorySystem::GetPointer(const VAddr vaddr) const {
    const u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        return page_pointer + (vaddr & CITRA_PAGE_MASK);
    }

    if (impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
            PageType::RasterizerCachedMemory ||
        impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
            PageType::RasterizerCachedMemoryWatchpoint) {
        return GetPointerForRasterizerCache(vaddr);
    }

    if (const u8* const again = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS]) {
        return again + (vaddr & CITRA_PAGE_MASK);
    }

    LOG_ERROR(HW_Memory, "unknown GetPointer @ 0x{:08x}", vaddr);
    return nullptr;
}

std::string MemorySystem::ReadCString(VAddr vaddr, std::size_t max_length) {
    std::string string;
    string.reserve(max_length);
    for (std::size_t i = 0; i < max_length; ++i) {
        char c = Read8(vaddr);
        if (c == '\0') {
            break;
        }

        string.push_back(c);
        ++vaddr;
    }

    string.shrink_to_fit();
    return string;
}

MemorySystem::PhysMemRegionInfo MemorySystem::GetPhysMemRegionInfo(PAddr address) {
    // Nothing is remembered from one call to the next. This is called by more than one thread,
    // and the answer of the call before, kept in a member and handed out again when the address
    // fits, could be changed by another thread between the check and the handing out: main
    // memory's start with an address in video memory, a pointer far outside both.
    //
    // Note: the region end check is inclusive because the user can pass in an address that
    // represents an open right bound
    const auto within = [address](PAddr start, u32 size) {
        return address >= start && address <= start + size;
    };
    if (within(FCRAM_PADDR, FCRAM_N3DS_SIZE)) {
        return {&impl->fcram_mem, FCRAM_PADDR, FCRAM_N3DS_SIZE};
    }
    if (within(VRAM_PADDR, VRAM_SIZE)) {
        return {&impl->vram_mem, VRAM_PADDR, VRAM_SIZE};
    }
    if (within(DSP_RAM_PADDR, DSP_RAM_SIZE)) {
        return {&impl->dsp_mem, DSP_RAM_PADDR, DSP_RAM_SIZE};
    }
    if (within(N3DS_EXTRA_RAM_PADDR, N3DS_EXTRA_RAM_SIZE)) {
        return {&impl->n3ds_extra_ram_mem, N3DS_EXTRA_RAM_PADDR, N3DS_EXTRA_RAM_SIZE};
    }
    LOG_ERROR(HW_Memory, "Unknown GetPhysMemRegionInfo @ {:#08X} at PC {:#08X}", address,
              impl->GetPC());
    return {};
}

u32 MemorySystem::PhysicalBytesFrom(PAddr address) const {
    const auto left = [address](PAddr start, u32 size) -> u32 {
        return address >= start && address - start < size ? size - (address - start) : 0;
    };
    if (const u32 bytes = left(FCRAM_PADDR, FCRAM_N3DS_SIZE)) {
        return bytes;
    }
    if (const u32 bytes = left(VRAM_PADDR, VRAM_SIZE)) {
        return bytes;
    }
    if (const u32 bytes = left(DSP_RAM_PADDR, DSP_RAM_SIZE)) {
        return bytes;
    }
    return left(N3DS_EXTRA_RAM_PADDR, N3DS_EXTRA_RAM_SIZE);
}

u8* MemorySystem::GetPhysicalPointer(PAddr address) {
    auto target_mem = GetPhysMemRegionInfo(address);

    if (!target_mem.valid()) [[unlikely]] {
        return {nullptr};
    }

    u32 offset_into_region = address - target_mem.region_start;
    return target_mem.backing_mem->get()->GetPtr() + offset_into_region;
}

MemoryRef MemorySystem::GetPhysicalRef(PAddr address) {
    const auto& target_mem = GetPhysMemRegionInfo(address);

    if (!target_mem.valid()) [[unlikely]] {
        return {nullptr};
    }

    u32 offset_into_region = address - target_mem.region_start;
    return {*target_mem.backing_mem, offset_into_region};
}

std::vector<VAddr> MemorySystem::PhysicalToVirtualAddressForRasterizer(PAddr addr) {
    if (addr >= VRAM_PADDR && addr < VRAM_PADDR_END) {
        return {addr - VRAM_PADDR + VRAM_VADDR};
    }
    // NOTE: Order matters here.
    PAddr plg_fb_addr = Plugin3GXFramebufferAddress();
    if (plg_fb_addr && addr >= plg_fb_addr && addr < plg_fb_addr + PLUGIN_3GX_FB_SIZE) {
        return {addr - plg_fb_addr + PLUGIN_3GX_FB_VADDR};
    }
    if (addr >= FCRAM_PADDR && addr < FCRAM_PADDR_END) {
        return {addr - FCRAM_PADDR + LINEAR_HEAP_VADDR, addr - FCRAM_PADDR + NEW_LINEAR_HEAP_VADDR};
    }
    if (addr >= FCRAM_PADDR_END && addr < FCRAM_N3DS_PADDR_END) {
        return {addr - FCRAM_PADDR + NEW_LINEAR_HEAP_VADDR};
    }
    // While the physical <-> virtual mapping is 1:1 for the regions supported by the cache,
    // some games (like Pokemon Super Mystery Dungeon) will try to use textures that go beyond
    // the end address of VRAM, causing the Virtual->Physical translation to fail when flushing
    // parts of the texture.
    LOG_ERROR(HW_Memory,
              "Trying to use invalid physical address for rasterizer: {:08X} at PC 0x{:08X}", addr,
              impl->GetPC());
    return {};
}

void MemorySystem::RasterizerMarkRegionCached(PAddr start, u32 size, bool cached) {
    MarkRegionForRasterizer(start, size, cached, 0);
}

void MemorySystem::RasterizerMarkRegionWatched(PAddr start, u32 size, bool watched) {
    MarkRegionForRasterizer(start, size, watched, 1);
}

void MemorySystem::MarkRegionForRasterizer(PAddr start, u32 size, bool set, int mark) {
    if (start == 0) {
        return;
    }
    // `mark`: 0 cached, 1 watched, 2 armed, 3 shut
    std::scoped_lock mark_lock{impl->mark_mutex};

    u32 num_pages = ((start + size - 1) >> CITRA_PAGE_BITS) - (start >> CITRA_PAGE_BITS) + 1;
    PAddr paddr = start;

    // The host's views of the address spaces follow in runs of pages, not page by page
    for (auto& page_table : impl->page_table_list) {
        page_table->pointers.BeginChanges();
    }
    SCOPE_EXIT({
        for (auto& page_table : impl->page_table_list) {
            page_table->pointers.EndChanges();
        }
    });

    for (unsigned i = 0; i < num_pages; ++i, paddr += CITRA_PAGE_SIZE) {
        for (VAddr vaddr : PhysicalToVirtualAddressForRasterizer(paddr)) {
            (mark == 0   ? impl->cache_marker
             : mark == 1 ? impl->watch_marker
             : mark == 2 ? impl->armed_marker
                         : impl->shut_marker)
                .Mark(vaddr, set);
            if (mark == 0 && !set) {
                // No longer cached: nothing is left to be readable in spite of
                impl->readable_marker.Mark(vaddr, false);
            }
            // A page is taken out of the direct way by either mark and put back when it has
            // neither
            const bool cached = impl->cache_marker.IsCached(vaddr);
            const bool watched = impl->watch_marker.IsCached(vaddr) ||
                                 impl->armed_marker.IsCached(vaddr) ||
                                 impl->shut_marker.IsCached(vaddr);
            const std::size_t idx = vaddr >> CITRA_PAGE_BITS;
            for (auto& page_table : impl->page_table_list) {
                PageType& page_type = page_table->attributes[idx];

                if (cached || watched) {
                    switch (page_type) {
                    case PageType::Unmapped:
                        // It is not necessary for a process to have this region mapped into its
                        // address space, for example, a system module need not have a VRAM mapping.
                        break;
                    case PageType::Memory:
                    case PageType::MemoryWatchpoint:
                        page_type = (page_type == PageType::Memory)
                                        ? PageType::RasterizerCachedMemory
                                        : PageType::RasterizerCachedMemoryWatchpoint;
                        page_table->pointers[idx] = nullptr;
                        break;
                    case PageType::RasterizerCachedMemory:
                    case PageType::RasterizerCachedMemoryWatchpoint:
                        // By the other of the two marks already
                        break;
                    default:
                        UNREACHABLE();
                    }
                } else {
                    switch (page_type) {
                    case PageType::Unmapped:
                    case PageType::Memory:
                    case PageType::MemoryWatchpoint:
                        break;
                    case PageType::RasterizerCachedMemory:
                    case PageType::RasterizerCachedMemoryWatchpoint: {
                        // The pointer first, then the type: another thread that reads the two
                        // while this goes on must not find a plain page without a pointer
                        const PageType plain = (page_type == PageType::RasterizerCachedMemory)
                                                   ? PageType::Memory
                                                   : PageType::MemoryWatchpoint;
                        if (plain == PageType::Memory) {
                            page_table->pointers[idx] =
                                GetPointerForRasterizerCache(vaddr & ~CITRA_PAGE_MASK);
                        }
                        page_type = plain;
                        break;
                    }
                    default:
                        UNREACHABLE();
                    }
                }

                ApplyReadThrough(*page_table, vaddr);
            }
        }
    }
}

void MemorySystem::ApplyReadThrough(PageTable& page_table, VAddr vaddr) {
    // What reads a page that is only watched, or cached with nothing newer held of it, need
    // not be told of: recompiled code that reaches memory through a view reads it there
    const std::size_t idx = vaddr >> CITRA_PAGE_BITS;
    const bool cached = impl->cache_marker.IsCached(vaddr);
    const bool read_through =
        page_table.attributes[idx] == PageType::RasterizerCachedMemory &&
        !impl->shut_marker.IsCached(vaddr) &&
        (cached ? impl->readable_marker.IsCached(vaddr)
                : impl->watch_marker.IsCached(vaddr) || impl->armed_marker.IsCached(vaddr));
    page_table.pointers.SetReadOnly(
        idx,
        read_through ? GetPointerForRasterizerCache(vaddr & ~CITRA_PAGE_MASK).GetPtr() : nullptr);
}

void MemorySystem::RasterizerSetSurfaceCells(PAddr start, u32 size, bool covered) {
    if (size == 0) {
        return;
    }
    std::vector<u64>& cells = impl->surface_cells;
    if (cells.empty()) {
        cells.resize((FCRAM_N3DS_SIZE + VRAM_SIZE) >> CITRA_PAGE_BITS);
    }
    u32 done = 0;
    while (done < size) {
        const u32 in_page = (start + done) & CITRA_PAGE_MASK;
        const u32 take = std::min(size - done, CITRA_PAGE_SIZE - in_page);
        if (const auto page = Impl::CellPage(start + done)) {
            const u64 mask = Impl::CellMask(in_page, in_page + take - 1);
            if (covered) {
                cells[*page] |= mask;
            } else {
                cells[*page] &= ~mask;
            }
        }
        done += take;
    }
}

void MemorySystem::RasterizerClearSurfaceCells() {
    std::fill(impl->surface_cells.begin(), impl->surface_cells.end(), u64{0});
    std::fill(impl->holds_newer_pages.begin(), impl->holds_newer_pages.end(), u8{0});
}

void MemorySystem::RasterizerNotePageHoldsNewer(PAddr start) {
    if (impl->holds_newer_pages.empty()) {
        impl->holds_newer_pages.resize((FCRAM_N3DS_SIZE + VRAM_SIZE) >> CITRA_PAGE_BITS);
    }
    if (const auto page = Impl::CellPage(start)) {
        impl->holds_newer_pages[*page] = 1;
    }
}

void MemorySystem::TrackGpuReferences(bool track) {
    impl->track_references.store(track);
}

void MemorySystem::NoteGpuReference(PAddr addr, u32 size, GpuReference kind) {
    impl->NoteGpuPages(addr, size, impl->gpu_reader, kind);
}

void MemorySystem::NoteGpuWritten(PAddr addr, u32 size) {
    impl->NoteGpuPages(addr, size, 2, GpuReference::Other);
}

void MemorySystem::SetGpuReader(bool transfer) {
    impl->gpu_reader = transfer ? 16 : 1;
    if (impl->has_forgotten.load(std::memory_order_relaxed)) [[unlikely]] {
        // Pages the emulation thread has forgotten (see FORGET_AFTER): noted anew the next
        // time something is read from them
        std::vector<u32> pages;
        {
            std::scoped_lock lock{impl->reference_mutex};
            pages.swap(impl->forgotten_pages);
            impl->has_forgotten.store(false, std::memory_order_relaxed);
        }
        for (const u32 page : pages) {
            impl->reference_noted[page] &= static_cast<u8>(~(1 | 16));
        }
        // (A stretch found in this table is taken as noted)
        impl->noted_stretches.fill({});
    }
}

void MemorySystem::ArmGpuReferences(bool for_list, bool for_transfer, u64 queued_before, u64 done,
                                    PAddr source_addr, u32 source_size,
                                    std::span<const std::pair<PAddr, u32>> written, bool narrow,
                                    std::vector<u32>& snapshot_pages,
                                    std::vector<u8>& snapshot_bytes) {
    if (impl->has_unheld.load(std::memory_order_relaxed)) [[unlikely]] {
        // Pages the decoder read without a copy: what the draws of their list told was not
        // all that was read of them. Copied for every list for a while.
        std::vector<u32> unheld;
        {
            std::scoped_lock lock{impl->reference_mutex};
            unheld.swap(impl->unheld_pages);
            impl->has_unheld.store(false, std::memory_order_relaxed);
        }
        for (const u32 page : unheld) {
            const auto entry = std::lower_bound(
                impl->volatile_pages.begin(), impl->volatile_pages.end(), page,
                [](const Impl::VolatilePage& one, u32 number) { return one.page < number; });
            if (entry != impl->volatile_pages.end() && entry->page == page) {
                entry->always_until = impl->late_vblank + Impl::ALWAYS_COPIED_FOR;
            }
        }
    }
    std::vector<u32> fresh_list;
    std::vector<u32> fresh_transfer;
    std::vector<u32> fresh_written;
    {
        std::scoped_lock lock{impl->reference_mutex};
        fresh_list.swap(impl->new_references);
        fresh_transfer.swap(impl->new_transfer_references);
        fresh_written.swap(impl->new_written);
    }
    if (for_transfer && source_size != 0) {
        // Known from the command, without the decoder having been there: learnt and armed
        // (or copied, if the game keeps writing them) below, for this very command
        const u64 last = u64{source_addr} + source_size - 1;
        for (u64 at = source_addr & ~CITRA_PAGE_MASK; at <= last; at += CITRA_PAGE_SIZE) {
            // (Those known already are armed, copied, or wait to be armed below)
            const auto page = Impl::CellPage(static_cast<PAddr>(at));
            if (page && (impl->reference_state[*page] & 16) == 0) {
                fresh_transfer.push_back(static_cast<u32>(*page));
            }
        }
    }
    for (const auto& [addr, size] : written) {
        if (size == 0) {
            continue;
        }
        const u64 last = u64{addr} + size - 1;
        for (u64 at = addr & ~CITRA_PAGE_MASK; at <= last; at += CITRA_PAGE_SIZE) {
            const auto page = Impl::CellPage(static_cast<PAddr>(at));
            if (page && (impl->reference_state[*page] & 4) == 0) {
                fresh_written.push_back(static_cast<u32>(*page));
            }
        }
    }
    std::vector<u8>& state = impl->reference_state;
    const auto learn = [&](const std::vector<u32>& fresh, std::vector<u32>& waiting, u8 known) {
        for (const u32 page : fresh) {
            if ((state[page] & known) == 0) {
                state[page] |= known;
                waiting.push_back(page);
            }
        }
    };
    learn(fresh_list, impl->to_arm, 1);
    learn(fresh_transfer, impl->to_arm_transfer, 16);
    learn(fresh_written, impl->to_shut, 4);

    // The volatile pages: copied for this command, and no longer armed once the commands
    // that were handed over without a copy of them are done
    std::vector<Impl::VolatilePage>& volatile_pages = impl->volatile_pages;
    bool demoted = false;
    std::vector<u32> forgotten;
    // Whether the decoder still reads the page (see FORGET_AFTER), and whether the game still
    // writes it. Looked at now and then only.
    const auto look = [&](Impl::VolatilePage& entry, const u8* bytes) {
        if (impl->late_vblank < entry.next_look) {
            return;
        }
        entry.next_look = impl->late_vblank + Impl::LOOK_EVERY;
        u8& bits = state[entry.page];
        const u32 read_turn = impl->snapshot_last_read[entry.page].load(std::memory_order_relaxed);
        if (read_turn != entry.read_turn) {
            entry.read_turn = read_turn;
            entry.read_vblank = impl->late_vblank;
        } else if ((bits & 2) == 0 && impl->late_vblank - entry.read_vblank > Impl::FORGET_AFTER) {
            bits &= static_cast<u8>(~(1 | 16 | 32));
            forgotten.push_back(entry.page);
            demoted = true;
            return;
        }
        const u64 checksum = Common::ComputeHash64(bytes, CITRA_PAGE_SIZE);
        if (checksum != entry.checksum) {
            entry.checksum = checksum;
            entry.same = 0;
        } else if (++entry.same >= entry.looks_needed && (bits & 2) == 0) {
            // Left alone: armed again, below
            bits &= static_cast<u8>(~32);
            if ((bits & 1) != 0) {
                impl->to_arm.push_back(entry.page);
            }
            if ((bits & 16) != 0) {
                impl->to_arm_transfer.push_back(entry.page);
            }
            impl->armed_again_at[entry.page] = impl->late_vblank;
            impl->armed_again_count++;
            demoted = true;
        }
    };
    for (Impl::VolatilePage& entry : volatile_pages) {
        u8& bits = state[entry.page];
        if ((bits & 2) != 0) {
            if (entry.disarm_after == ~u64{0}) {
                entry.disarm_after = queued_before;
            }
            if (done >= entry.disarm_after) {
                MarkRegionForRasterizer(Impl::CellPageAddress(entry.page), CITRA_PAGE_SIZE, false,
                                        2);
                bits &= static_cast<u8>(~2);
            }
        }
        if (!((for_list && (bits & 1) != 0) || (for_transfer && (bits & 16) != 0))) {
            continue;
        }
        if (narrow && !entry.wanted && impl->late_vblank >= entry.always_until) {
            const auto kind = static_cast<GpuReference>(impl->reference_kind[entry.page]);
            if (kind == GpuReference::VertexData || kind == GpuReference::IndexData) {
                // No draw of the list reads it: no copy. (From here on a read of it without
                // one is news again.) Whether the decoder reads it at all any more is looked
                // at as for the pages that are copied.
                impl->narrowed_out++;
                impl->follow_saved++;
                if (impl->volatile_flag[entry.page].load(std::memory_order_relaxed) == 2) {
                    impl->volatile_flag[entry.page].store(1, std::memory_order_relaxed);
                }
                look(entry, GetPhysicalPointer(Impl::CellPageAddress(entry.page)));
                continue;
            }
        }
        impl->snapshot_wanted.push_back(&entry);
    }
    if (!impl->snapshot_wanted.empty()) {
        // One buffer for the command, which comes with the room an earlier command needed and
        // is left as it comes: what is used of it is written here
        const std::size_t count = impl->snapshot_wanted.size();
        if (snapshot_bytes.size() < count * CITRA_PAGE_SIZE) {
            snapshot_bytes.resize(count * CITRA_PAGE_SIZE);
        }
        snapshot_pages.reserve(count);
        u8* copy = snapshot_bytes.data();
        for (Impl::VolatilePage* const entry : impl->snapshot_wanted) {
            std::memcpy(copy, GetPhysicalPointer(Impl::CellPageAddress(entry->page)),
                        CITRA_PAGE_SIZE);
            snapshot_pages.push_back(entry->page);
            look(*entry, copy);
            copy += CITRA_PAGE_SIZE;
        }
        impl->snapshot_bytes_copied += count * CITRA_PAGE_SIZE;
        impl->snapshot_wanted.clear();
    }
    if (demoted) {
        std::erase_if(volatile_pages, [&](const Impl::VolatilePage& entry) {
            if ((state[entry.page] & 32) != 0) {
                return false;
            }
            impl->volatile_flag[entry.page].store(0, std::memory_order_relaxed);
            impl->volatile_count.fetch_sub(1, std::memory_order_relaxed);
            return true;
        });
    }
    if (!forgotten.empty()) {
        for (const u32 page : forgotten) {
            impl->snapshot_last_read[page].store(0, std::memory_order_relaxed);
        }
        impl->forgotten_count += static_cast<u32>(forgotten.size());
        std::scoped_lock lock{impl->reference_mutex};
        impl->forgotten_pages.insert(impl->forgotten_pages.end(), forgotten.begin(),
                                     forgotten.end());
        impl->has_forgotten.store(true, std::memory_order_relaxed);
    }

    std::vector<u32> pages;
    const auto arm = [&](std::vector<u32>& waiting, u8 now, int mark) {
        if (waiting.empty()) {
            return;
        }
        // A page read for both kinds of command waits in both lists, and a volatile one is
        // copied instead
        pages.clear();
        for (const u32 page : waiting) {
            if ((state[page] & now) == 0 && (mark != 2 || (state[page] & 32) == 0)) {
                state[page] |= now;
                pages.push_back(page);
            }
        }
        waiting.clear();
        // Pages next to each other in one go
        std::sort(pages.begin(), pages.end());
        std::size_t run_start = 0;
        for (std::size_t i = 1; i <= pages.size(); i++) {
            if (i == pages.size() || pages[i] != pages[i - 1] + 1 ||
                pages[i] == (FCRAM_N3DS_SIZE >> CITRA_PAGE_BITS)) {
                MarkRegionForRasterizer(Impl::CellPageAddress(pages[run_start]),
                                        static_cast<u32>((i - run_start) << CITRA_PAGE_BITS), true,
                                        mark);
                run_start = i;
            }
        }
    };
    if (for_list) {
        arm(impl->to_arm, 2, 2);
    }
    if (for_transfer) {
        arm(impl->to_arm_transfer, 2, 2);
    }
    arm(impl->to_shut, 8, 3);
}

void MemorySystem::Disarm(PAddr addr, u32 size, bool written) {
    std::vector<u8>& state = impl->reference_state;
    if (size == 0 || !impl->track_references.load(std::memory_order_relaxed)) {
        return;
    }
    const u64 last = u64{addr} + size - 1;
    for (u64 at = addr & ~CITRA_PAGE_MASK; at <= last; at += CITRA_PAGE_SIZE) {
        const auto page = Impl::CellPage(static_cast<PAddr>(at));
        if (!page || (state[*page] & (2 | 8)) == 0) {
            continue;
        }
        u8& bits = state[*page];
        // What the decoder reads may be read by the game as well. A page that is written is
        // one the game keeps writing, as a rule: copied at hand-overs from here on, not armed
        // (the decoder has been waited for, nothing is left that reads it where it is)
        if (written && (bits & 2) != 0) {
            MarkRegionForRasterizer(static_cast<PAddr>(at), CITRA_PAGE_SIZE, false, 2);
            bits &= static_cast<u8>(~2);
            impl->AddVolatile(*page);
            if ((bits & 32) == 0) {
                // No room for more of them: armed again at the next hand-over
                if ((bits & 1) != 0) {
                    impl->to_arm.push_back(static_cast<u32>(*page));
                }
                if ((bits & 16) != 0) {
                    impl->to_arm_transfer.push_back(static_cast<u32>(*page));
                }
            }
        }
        if ((bits & 8) != 0) {
            MarkRegionForRasterizer(static_cast<PAddr>(at), CITRA_PAGE_SIZE, false, 3);
            bits &= static_cast<u8>(~8);
            impl->to_shut.push_back(static_cast<u32>(*page));
        }
    }
}

void MemorySystem::ExemptReads(PAddr addr, u32 size) {
    for (Impl::LateRange& range : impl->late_ranges) {
        if (range.addr == addr && range.size == size) {
            range.last_vblank = impl->late_vblank;
            return;
        }
    }
    if (impl->late_ranges.size() >= 16) {
        impl->late_ranges.erase(impl->late_ranges.begin());
    }
    impl->late_ranges.push_back({addr, size, impl->late_vblank});
}

void MemorySystem::PostLateWrite(PAddr addr, std::vector<u8>&& bytes) {
    std::scoped_lock lock{impl->late_mutex};
    impl->late_writes.emplace_back(addr, std::move(bytes));
    impl->has_late_writes.store(true);
}

void MemorySystem::ApplyLateWrites() {
    impl->late_vblank++;
    if (!impl->has_late_writes.load(std::memory_order_relaxed)) {
        return;
    }
    std::vector<std::pair<PAddr, std::vector<u8>>> writes;
    {
        std::scoped_lock lock{impl->late_mutex};
        writes.swap(impl->late_writes);
        impl->has_late_writes.store(false);
    }
    for (const auto& [addr, bytes] : writes) {
        // Only where the game is being given it late: not over what it has since been given
        // the usual way, or has written itself
        for (Impl::LateRange& range : impl->late_ranges) {
            if (addr >= range.addr && u64{addr} + bytes.size() <= u64{range.addr} + range.size &&
                impl->late_vblank - range.last_vblank <= Impl::LATE_RANGE_VBLANKS + 1) {
                if (u8* const target = GetPhysicalPointer(addr)) {
                    std::memcpy(target, bytes.data(), bytes.size());
                }
                range.last_vblank = impl->late_vblank;
                break;
            }
        }
    }
}

void MemorySystem::DisarmAll() {
    std::vector<u8>& state = impl->reference_state;
    for (std::size_t page = 0; page < state.size(); page++) {
        u8& bits = state[page];
        if ((bits & (2 | 8)) == 0) {
            continue;
        }
        const PAddr addr = Impl::CellPageAddress(page);
        if ((bits & 2) != 0) {
            MarkRegionForRasterizer(addr, CITRA_PAGE_SIZE, false, 2);
            bits &= static_cast<u8>(~2);
            if ((bits & 32) == 0) {
                if ((bits & 1) != 0) {
                    impl->to_arm.push_back(static_cast<u32>(page));
                }
                if ((bits & 16) != 0) {
                    impl->to_arm_transfer.push_back(static_cast<u32>(page));
                }
            }
        }
        if ((bits & 8) != 0) {
            MarkRegionForRasterizer(addr, CITRA_PAGE_SIZE, false, 3);
            bits &= static_cast<u8>(~8);
            impl->to_shut.push_back(static_cast<u32>(page));
        }
    }
    for (Impl::VolatilePage& entry : impl->volatile_pages) {
        entry.disarm_after = ~u64{0};
    }
}

void MemorySystem::WantVolatile(PAddr addr, u32 size) {
    if (size == 0) {
        return;
    }
    const u64 last = u64{addr} + size - 1;
    for (u64 at = addr & ~CITRA_PAGE_MASK; at <= last; at += CITRA_PAGE_SIZE) {
        const auto page = Impl::CellPage(static_cast<PAddr>(at));
        if (page && (impl->reference_state[*page] & 2) != 0) {
            impl->AddVolatile(*page);
        }
    }
}

bool MemorySystem::BeginWantedCopies() {
    bool any = false;
    for (Impl::VolatilePage& entry : impl->volatile_pages) {
        entry.wanted = false;
        const auto kind = static_cast<GpuReference>(impl->reference_kind[entry.page]);
        any |= kind == GpuReference::VertexData || kind == GpuReference::IndexData;
    }
    return any && impl->late_vblank >= impl->follow_paused_until &&
           (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::LegacyPageCopies) == 0;
}

void MemorySystem::NoteDrawsFollowed(u32 draws) {
    impl->follow_draws += draws;
    if (++impl->follow_lists < Impl::FOLLOW_WINDOW) {
        return;
    }
    if (impl->follow_saved * Impl::DRAWS_A_PAGE < impl->follow_draws) {
        impl->follow_paused_until = impl->late_vblank + impl->follow_pause;
        impl->follow_pause = std::min(impl->follow_pause * 2, Impl::FOLLOW_PAUSE_MOST);
    } else {
        impl->follow_pause = Impl::FOLLOW_PAUSE_FIRST;
    }
    impl->follow_draws = 0;
    impl->follow_saved = 0;
    impl->follow_lists = 0;
}

bool MemorySystem::HasVolatileIn(PAddr addr, u32 size) const {
    const auto [from, to] = impl->VolatileIn(addr, size);
    return to > from;
}

void MemorySystem::WantCopyOf(PAddr addr, u32 size) {
    const auto [from, to] = impl->VolatileIn(addr, size);
    for (std::size_t index = from; index < to; index++) {
        impl->volatile_pages[index].wanted = true;
    }
}

std::pair<u64, u64> MemorySystem::TakeNarrowFigures() {
    return {std::exchange(impl->narrowed_out, 0),
            impl->unheld_reads.exchange(0, std::memory_order_relaxed)};
}

void MemorySystem::UseGpuSnapshot(const std::vector<u32>& pages, const u8* bytes) {
    impl->snapshot_pages = &pages;
    impl->snapshot_bytes = bytes;
    impl->snapshot_read.assign(pages.size(), 0);
    for (std::size_t i = 0; i < pages.size(); i++) {
        impl->snapshot_slot[pages[i]] = static_cast<u16>(i + 1);
    }
}

void MemorySystem::DropGpuSnapshot() {
    if (!impl->snapshot_pages) {
        return;
    }
    for (const u32 page : *impl->snapshot_pages) {
        impl->snapshot_slot[page] = 0;
    }
    u64 read = 0;
    const u32 turn = impl->snapshot_turn.fetch_add(1, std::memory_order_relaxed);
    for (std::size_t i = 0; i < impl->snapshot_read.size(); i++) {
        if (impl->snapshot_read[i] != 0) {
            read++;
            impl->snapshot_last_read[(*impl->snapshot_pages)[i]].store(turn,
                                                                       std::memory_order_relaxed);
        }
    }
    impl->snapshot_given_sum.fetch_add(impl->snapshot_pages->size(), std::memory_order_relaxed);
    impl->snapshot_read_sum.fetch_add(read, std::memory_order_relaxed);
    impl->snapshot_pages = nullptr;
    impl->snapshot_bytes = nullptr;
    impl->stitched.clear();
}

const u8* MemorySystem::GpuRead(PAddr addr, u32 size) {
    const u8* const direct = GetPhysicalPointer(addr);
    // A page the game keeps writing, read for a list that came without a copy of it: the
    // list's draws did not tell of this read (see BeginWantedCopies). The emulation thread
    // is told, and copies the page for every list for a while.
    if (impl->gpu_reader == 1 && size != 0 && direct != nullptr &&
        impl->volatile_count.load(std::memory_order_relaxed) != 0) {
        impl->NoteUnheldRead(addr, size);
    }
    if (!impl->snapshot_pages || size == 0 || !direct) [[likely]] {
        return direct;
    }
    const auto first = Impl::CellPage(addr);
    const auto last = Impl::CellPage(addr + size - 1);
    if (!first || !last || *last < *first) {
        return direct;
    }
    const std::vector<u16>& slot = impl->snapshot_slot;
    bool any = false;
    bool all = true;
    for (std::size_t page = *first; page <= *last; page++) {
        if (slot[page] != 0) {
            any = true;
            impl->snapshot_read[slot[page] - 1u] = 1;
        } else {
            all = false;
        }
    }
    if (!any) {
        return direct;
    }
    const u32 in_page = addr & CITRA_PAGE_MASK;
    if (all) {
        // Pages next to each other are next to each other among the copies
        return impl->snapshot_bytes + std::size_t{slot[*first] - 1u} * CITRA_PAGE_SIZE + in_page;
    }
    std::vector<u8>& whole = impl->stitched.emplace_back(size);
    u32 done = 0;
    for (std::size_t page = *first; page <= *last; page++) {
        const u32 from = page == *first ? in_page : 0;
        const u32 take = std::min(size - done, CITRA_PAGE_SIZE - from);
        const u8* const source =
            slot[page] != 0
                ? impl->snapshot_bytes + std::size_t{slot[page] - 1u} * CITRA_PAGE_SIZE + from
                : direct + done;
        std::memcpy(whole.data() + done, source, take);
        done += take;
    }
    return whole.data();
}

std::pair<u64, u64> MemorySystem::TakeSnapshotReadFigures() {
    return {impl->snapshot_given_sum.exchange(0, std::memory_order_relaxed),
            impl->snapshot_read_sum.exchange(0, std::memory_order_relaxed)};
}

MemorySystem::VolatileAges MemorySystem::VolatileAgesNow() {
    VolatileAges ages{};
    ages.forgotten = std::exchange(impl->forgotten_count, 0);
    ages.armed_again = std::exchange(impl->armed_again_count, 0);
    ages.written_again = std::exchange(impl->written_again_count, 0);
    if (!impl->snapshot_last_read || impl->volatile_pages.empty()) {
        return ages;
    }
    const u32 turn = impl->snapshot_turn.load(std::memory_order_relaxed);
    u32 before = ~u32{0};
    for (const Impl::VolatilePage& entry : impl->volatile_pages) {
        const u32 last = impl->snapshot_last_read[entry.page].load(std::memory_order_relaxed);
        const u32 ago = turn - last;
        ages.read[last == 0 ? 3 : ago <= 256 ? 0 : ago <= 2048 ? 1 : 2]++;
        const std::size_t kind = std::min<std::size_t>(impl->reference_kind[entry.page], 4);
        ages.kinds[kind]++;
        ages.kinds_read[kind] += last != 0 && ago <= 256 ? 1 : 0;
        if (before == ~u32{0} || entry.page != before + 1) {
            ages.runs++;
        }
        before = entry.page;
    }
    ages.first = Impl::CellPageAddress(impl->volatile_pages.front().page);
    ages.last = Impl::CellPageAddress(impl->volatile_pages.back().page);
    return ages;
}

std::pair<u32, u64> MemorySystem::TakeSnapshotFigures() {
    return {static_cast<u32>(impl->volatile_pages.size()),
            std::exchange(impl->snapshot_bytes_copied, 0)};
}

void MemorySystem::RasterizerMarkRegionReadable(PAddr start, u32 size, bool readable) {
    if (start == 0 || size == 0) {
        return;
    }
    std::scoped_lock mark_lock{impl->mark_mutex};
    const u32 num_pages = ((start + size - 1) >> CITRA_PAGE_BITS) - (start >> CITRA_PAGE_BITS) + 1;
    PAddr paddr = start & ~CITRA_PAGE_MASK;

    for (auto& page_table : impl->page_table_list) {
        page_table->pointers.BeginChanges();
    }
    SCOPE_EXIT({
        for (auto& page_table : impl->page_table_list) {
            page_table->pointers.EndChanges();
        }
    });

    for (u32 i = 0; i < num_pages; ++i, paddr += CITRA_PAGE_SIZE) {
        if (const auto page = Impl::CellPage(paddr);
            page && *page < impl->holds_newer_pages.size()) {
            // Looked at anew at the next read that is announced
            impl->holds_newer_pages[*page] = 0;
        }
        for (VAddr vaddr : PhysicalToVirtualAddressForRasterizer(paddr)) {
            // Only of a cached page can this be said
            impl->readable_marker.Mark(vaddr, readable && impl->cache_marker.IsCached(vaddr));
            for (auto& page_table : impl->page_table_list) {
                ApplyReadThrough(*page_table, vaddr);
            }
        }
    }
}

u8 MemorySystem::Read8(const VAddr addr) {
    return Read<u8>(impl->current_page_table, addr);
}

u8 MemorySystem::Read8(const Kernel::Process& process, VAddr addr) {
    return Read<u8>(process.vm_manager.page_table, addr);
}

u16 MemorySystem::Read16(const VAddr addr) {
    return Read<u16_le>(impl->current_page_table, addr);
}

u16 MemorySystem::Read16(const Kernel::Process& process, VAddr addr) {
    return Read<u16_le>(process.vm_manager.page_table, addr);
}

u32 MemorySystem::Read32(const VAddr addr) {
    return Read<u32_le>(impl->current_page_table, addr);
}

u32 MemorySystem::Read32(const Kernel::Process& process, VAddr addr) {
    return Read<u32_le>(process.vm_manager.page_table, addr);
}

u64 MemorySystem::Read64(const VAddr addr) {
    return Read<u64_le>(impl->current_page_table, addr);
}

u64 MemorySystem::Read64(const Kernel::Process& process, VAddr addr) {
    return Read<u64_le>(process.vm_manager.page_table, addr);
}

std::optional<u32> MemorySystem::Read32OrNullopt(VAddr addr) {
    return Read<std::optional<u32_le>>(impl->current_page_table, addr);
}

std::optional<u32> MemorySystem::Read32OrNullopt(const Kernel::Process& process, VAddr addr) {
    return Read<std::optional<u32_le>>(process.vm_manager.page_table, addr);
}

void MemorySystem::ReadBlock(const Kernel::Process& process, const VAddr src_addr,
                             void* dest_buffer, const std::size_t size) {
    return impl->ReadBlockImpl<false>(process, src_addr, dest_buffer, size);
}

void MemorySystem::ReadBlock(VAddr src_addr, void* dest_buffer, std::size_t size) {
    const auto& process = *impl->system.Kernel().GetCurrentProcess();
    return impl->ReadBlockImpl<false>(process, src_addr, dest_buffer, size);
}

void MemorySystem::Write8(const VAddr addr, const u8 data) {
    Write<u8>(impl->current_page_table, addr, data);
}

void MemorySystem::Write8(const Kernel::Process& process, const VAddr addr, const u8 data) {
    Write<u8>(process.vm_manager.page_table, addr, data);
}

void MemorySystem::Write16(const VAddr addr, const u16 data) {
    Write<u16_le>(impl->current_page_table, addr, data);
}

void MemorySystem::Write16(const Kernel::Process& process, const VAddr addr, const u16 data) {
    Write<u16_le>(process.vm_manager.page_table, addr, data);
}

void MemorySystem::Write32(const VAddr addr, const u32 data) {
    Write<u32_le>(impl->current_page_table, addr, data);
}

void MemorySystem::Write32(const Kernel::Process& process, const VAddr addr, const u32 data) {
    Write<u32_le>(process.vm_manager.page_table, addr, data);
}

void MemorySystem::Write64(const VAddr addr, const u64 data) {
    Write<u64_le>(impl->current_page_table, addr, data);
}

void MemorySystem::Write64(const Kernel::Process& process, const VAddr addr, const u64 data) {
    Write<u64_le>(process.vm_manager.page_table, addr, data);
}

bool MemorySystem::WriteExclusive8(const VAddr addr, const u8 data, const u8 expected) {
    return WriteExclusive<u8>(addr, data, expected);
}

bool MemorySystem::WriteExclusive16(const VAddr addr, const u16 data, const u16 expected) {
    return WriteExclusive<u16_le>(addr, data, expected);
}

bool MemorySystem::WriteExclusive32(const VAddr addr, const u32 data, const u32 expected) {
    return WriteExclusive<u32_le>(addr, data, expected);
}

bool MemorySystem::WriteExclusive64(const VAddr addr, const u64 data, const u64 expected) {
    return WriteExclusive<u64_le>(addr, data, expected);
}

void MemorySystem::WriteBlock(const Kernel::Process& process, const VAddr dest_addr,
                              const void* src_buffer, const std::size_t size) {
    return impl->WriteBlockImpl<false>(process, dest_addr, src_buffer, size);
}

void MemorySystem::WriteBlock(const VAddr dest_addr, const void* src_buffer,
                              const std::size_t size) {
    auto& process = *impl->system.Kernel().GetCurrentProcess();
    return impl->WriteBlockImpl<false>(process, dest_addr, src_buffer, size);
}

void MemorySystem::ZeroBlock(const Kernel::Process& process, const VAddr dest_addr,
                             const std::size_t size) {
    auto& page_table = *process.vm_manager.page_table;
    std::size_t remaining_size = size;
    std::size_t page_index = dest_addr >> CITRA_PAGE_BITS;
    std::size_t page_offset = dest_addr & CITRA_PAGE_MASK;

    while (remaining_size > 0) {
        const std::size_t copy_amount = std::min(CITRA_PAGE_SIZE - page_offset, remaining_size);
        const VAddr current_vaddr =
            static_cast<VAddr>((page_index << CITRA_PAGE_BITS) + page_offset);

        switch (page_table.attributes[page_index]) {
        case PageType::Unmapped: {
            LOG_ERROR(HW_Memory,
                      "unmapped ZeroBlock @ 0x{:08X} (start address = 0x{:08X}, size = {}) at PC "
                      "0x{:08X}",
                      current_vaddr, dest_addr, size, impl->GetPC());
            break;
        }
        case PageType::Memory: {
            u8* page = page_table.pointers[page_index];
            if (!page) [[unlikely]] {
                // Marked for the rasterizer since the look at its type, by the thread that
                // decodes GPU commands
                RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                             FlushMode::Invalidate);
                page = GetPointerForRasterizerCache(current_vaddr & ~CITRA_PAGE_MASK).GetPtr();
            }
            std::memset(page + page_offset, 0, copy_amount);
            break;
        }
        case PageType::MemoryWatchpoint: {
            auto it = page_table.watchpoint_pages_map.find(page_index);
            ASSERT_MSG(it != page_table.watchpoint_pages_map.end(),
                       "Missing memory for watchpoint page");

            u8* dest_ptr = it->second.memory.GetPtr() + page_offset;
            std::memset(dest_ptr, 0, copy_amount);
            break;
        }
        case PageType::RasterizerCachedMemory:
        case PageType::RasterizerCachedMemoryWatchpoint: {
            RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                         FlushMode::Invalidate);
            std::memset(GetPointerForRasterizerCache(current_vaddr), 0, copy_amount);
            break;
        }
        default:
            UNREACHABLE();
        }

        page_index++;
        page_offset = 0;
        remaining_size -= copy_amount;
    }
}

void MemorySystem::CopyBlock(const Kernel::Process& process, VAddr dest_addr, VAddr src_addr,
                             const std::size_t size) {
    CopyBlock(process, process, dest_addr, src_addr, size);
}

void MemorySystem::CopyBlock(const Kernel::Process& dest_process,
                             const Kernel::Process& src_process, VAddr dest_addr, VAddr src_addr,
                             std::size_t size) {
    auto& page_table = *src_process.vm_manager.page_table;
    std::size_t remaining_size = size;
    std::size_t page_index = src_addr >> CITRA_PAGE_BITS;
    std::size_t page_offset = src_addr & CITRA_PAGE_MASK;

    while (remaining_size > 0) {
        const std::size_t copy_amount = std::min(CITRA_PAGE_SIZE - page_offset, remaining_size);
        const VAddr current_vaddr =
            static_cast<VAddr>((page_index << CITRA_PAGE_BITS) + page_offset);

        switch (page_table.attributes[page_index]) {
        case PageType::Unmapped: {
            LOG_ERROR(HW_Memory,
                      "unmapped CopyBlock @ 0x{:08X} (start address = 0x{:08X}, size = {}) at PC "
                      "0x{:08X}",
                      current_vaddr, src_addr, size, impl->GetPC());
            ZeroBlock(dest_process, dest_addr, copy_amount);
            break;
        }
        case PageType::Memory: {
            const u8* page = page_table.pointers[page_index];
            if (!page) [[unlikely]] {
                // Marked for the rasterizer since the look at its type, by the thread that
                // decodes GPU commands
                RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                             FlushMode::Flush);
                page = GetPointerForRasterizerCache(current_vaddr & ~CITRA_PAGE_MASK).GetPtr();
            }
            WriteBlock(dest_process, dest_addr, page + page_offset, copy_amount);
            break;
        }
        case PageType::MemoryWatchpoint: {
            auto it = page_table.watchpoint_pages_map.find(page_index);
            ASSERT_MSG(it != page_table.watchpoint_pages_map.end(),
                       "Missing memory for watchpoint page");

            const u8* src_ptr = it->second.memory.GetPtr() + page_offset;
            WriteBlock(dest_process, dest_addr, src_ptr, copy_amount);
            break;
        }
        case PageType::RasterizerCachedMemory:
        case PageType::RasterizerCachedMemoryWatchpoint: {
            if (!impl->ReadsDirect(current_vaddr)) {
                RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                             FlushMode::Flush);
            }
            WriteBlock(dest_process, dest_addr, GetPointerForRasterizerCache(current_vaddr),
                       copy_amount);
            break;
        }
        default:
            UNREACHABLE();
        }

        page_index++;
        page_offset = 0;
        dest_addr += static_cast<VAddr>(copy_amount);
        src_addr += static_cast<VAddr>(copy_amount);
        remaining_size -= copy_amount;
    }
}

u32 MemorySystem::GetFCRAMOffset(const u8* pointer) const {
    ASSERT(pointer >= impl->fcram.get() && pointer <= impl->fcram.get() + Memory::FCRAM_N3DS_SIZE);
    return static_cast<u32>(pointer - impl->fcram.get());
}

u8* MemorySystem::GetFCRAMPointer(std::size_t offset) {
    ASSERT(offset <= Memory::FCRAM_N3DS_SIZE);
    return impl->fcram.get() + offset;
}

const u8* MemorySystem::GetFCRAMPointer(std::size_t offset) const {
    ASSERT(offset <= Memory::FCRAM_N3DS_SIZE);
    return impl->fcram.get() + offset;
}

MemoryRef MemorySystem::GetFCRAMRef(std::size_t offset) const {
    ASSERT(offset <= Memory::FCRAM_N3DS_SIZE);
    return MemoryRef(impl->fcram_mem, offset);
}

u8* MemorySystem::GetDspMemory(std::size_t offset) const {
    ASSERT(offset <= Memory::DSP_RAM_SIZE);
    return impl->dsp_ram.get() + offset;
}

} // namespace Memory
