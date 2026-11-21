// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <chrono>
#include <vector>
#include "common/alignment.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_guest_memory.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_memory_util.h"

namespace Vulkan {

namespace {

/// What the buffer over the emulated memory is used as. Vertex and index data for now; the
/// transfer and storage uses are for what is to follow: textures decoded from it and results
/// written back into it by the GPU.
constexpr vk::BufferUsageFlags POOL_USAGE =
    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eStorageBuffer;

/// How many frames the measuring sums up for a line in the log
constexpr u32 REPORT_FRAMES = 600;

/// The memories vertex and index data can be in, one after the other in InPlaceData's table
constexpr u32 FCRAM_PAGES = Memory::FCRAM_N3DS_SIZE >> Memory::CITRA_PAGE_BITS;
constexpr u32 VRAM_PAGES = Memory::VRAM_SIZE >> Memory::CITRA_PAGE_BITS;

/// Frames a page has to be drawn from without being written before it is read in place.
/// Half a second: what the game writes every frame, or every other, is seen at once.
constexpr u16 CALM_FRAMES = 30;

/// Frames a page is left alone after it was written or read too much, before it is tried
/// anew: two seconds the first time, twice as long each time after that, up to four minutes
constexpr u32 RETRY_FRAMES = 120;
constexpr u8 MAX_FAILURES = 8;

/// Reads the long way within READ_WINDOW frames after which a page is given up
constexpr u16 READ_LIMIT = 16;
constexpr u32 READ_WINDOW = 64;

/// More stretches of data than this are not remembered: a game that draws from ever new
/// places would make the table grow without end
constexpr std::size_t MAX_REMEMBERED = 1u << 20;

/// Hands the device `size` bytes at `pointer` as memory of its own
[[nodiscard]] vk::UniqueDeviceMemory TakeByAddress(vk::Device device, u32 memory_type,
                                                   void* pointer, u64 size) {
    const vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT>
        allocation = {
            vk::MemoryAllocateInfo{
                .allocationSize = size,
                .memoryTypeIndex = memory_type,
            },
            vk::ImportMemoryHostPointerInfoEXT{
                .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                .pHostPointer = pointer,
            },
        };
    return device.allocateMemoryUnique(allocation.get());
}

/// Whether the device takes that memory, for telling what it objects to after it refused
[[nodiscard]] bool Takes(vk::Device device, u32 memory_type, void* pointer, u64 size) {
    try {
        (void)TakeByAddress(device, memory_type, pointer, size);
        return true;
    } catch (const vk::SystemError&) {
        return false;
    }
}

} // Anonymous namespace

GuestMemory::GuestMemory(const Instance& instance, std::span<u8> pool_) {
    if (pool_.empty()) {
        return;
    }
    if (!instance.IsExternalMemoryHostSupported()) {
        LOG_WARNING(Render_Vulkan, "The emulated memory is not a buffer of the host GPU: the "
                                   "device takes no memory by its address");
        return;
    }
    const u64 alignment = instance.GetMinImportedHostPointerAlignment();
    if (alignment == 0 || reinterpret_cast<std::uintptr_t>(pool_.data()) % alignment != 0 ||
        pool_.size() % alignment != 0) {
        LOG_WARNING(Render_Vulkan,
                    "The emulated memory is not a buffer of the host GPU: the device takes "
                    "memory in pieces of {} bytes, which it does not start or end on",
                    alignment);
        return;
    }

    const vk::Device device = instance.GetDevice();
    const auto handle_type = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
    std::optional<u32> type_tried;
    try {
        const vk::MemoryHostPointerPropertiesEXT properties =
            device.getMemoryHostPointerPropertiesEXT(handle_type, pool_.data());
        const std::optional<u32> memory_type = FindMemoryType(
            instance.GetPhysicalDevice().getMemoryProperties(),
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            properties.memoryTypeBits);
        if (!memory_type) {
            LOG_WARNING(Render_Vulkan, "The emulated memory is not a buffer of the host GPU: "
                                       "the device has no memory type for it");
            return;
        }

        type_tried = memory_type;

        const auto start = std::chrono::steady_clock::now();
        vk::UniqueDeviceMemory imported =
            TakeByAddress(device, *memory_type, pool_.data(), pool_.size());

        const vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo>
            buffer_info = {
                vk::BufferCreateInfo{
                    .size = pool_.size(),
                    .usage = POOL_USAGE,
                    .sharingMode = vk::SharingMode::eExclusive,
                },
                vk::ExternalMemoryBufferCreateInfo{
                    .handleTypes = handle_type,
                },
            };
        vk::UniqueBuffer over_it = device.createBufferUnique(buffer_info.get());
        device.bindBufferMemory(*over_it, *imported, 0);

        const std::chrono::duration<double, std::milli> taken =
            std::chrono::steady_clock::now() - start;
        LOG_INFO(Render_Vulkan,
                 "The emulated memory, {} MiB, is a buffer of the host GPU as well (taken in "
                 "{:.1f} ms)",
                 pool_.size() >> 20, taken.count());

        pool = pool_;
        memory = std::move(imported);
        buffer = std::move(over_it);
    } catch (const vk::SystemError& error) {
        LOG_WARNING(Render_Vulkan,
                    "The emulated memory is not a buffer of the host GPU: the device refused "
                    "it ({})",
                    error.what());
        if (type_tried) {
            // What it objects to: the size, or the kind of memory
            constexpr u64 PROBE_SIZE = 1u << 20;
            std::vector<u8> own(PROBE_SIZE + alignment);
            u8* const own_start = reinterpret_cast<u8*>(
                Common::AlignUp(reinterpret_cast<std::uintptr_t>(own.data()), alignment));
            LOG_WARNING(Render_Vulkan,
                        "Of the emulated memory the device takes the first MiB: {}; a MiB of "
                        "the emulator's own memory: {}",
                        pool_.size() >= PROBE_SIZE &&
                            Takes(device, *type_tried, pool_.data(), PROBE_SIZE),
                        Takes(device, *type_tried, own_start, PROBE_SIZE));
        }
    }
}

GuestMemory::~GuestMemory() = default;

InPlaceData::InPlaceData(Mark mark_, bool log_) : mark{std::move(mark_)}, log{log_} {}

std::optional<std::pair<u32, u32>> InPlaceData::PagesOf(PAddr addr, u32 size) const {
    if (size == 0) {
        return std::nullopt;
    }
    const u64 end = static_cast<u64>(addr) + size;
    const auto within = [&](PAddr start, u32 length,
                            u32 first_page) -> std::optional<std::pair<u32, u32>> {
        const u64 from = std::max<u64>(addr, start);
        const u64 to = std::min<u64>(end, static_cast<u64>(start) + length);
        if (from >= to) {
            return std::nullopt;
        }
        return std::pair{first_page + static_cast<u32>((from - start) >> Memory::CITRA_PAGE_BITS),
                         first_page +
                             static_cast<u32>((to - 1 - start) >> Memory::CITRA_PAGE_BITS)};
    };
    if (const auto fcram = within(Memory::FCRAM_PADDR, Memory::FCRAM_N3DS_SIZE, 0)) {
        return fcram;
    }
    return within(Memory::VRAM_PADDR, Memory::VRAM_SIZE, FCRAM_PAGES);
}

PAddr InPlaceData::AddressOf(u32 page) {
    if (page < FCRAM_PAGES) {
        return Memory::FCRAM_PADDR + (page << Memory::CITRA_PAGE_BITS);
    }
    return Memory::VRAM_PADDR + ((page - FCRAM_PAGES) << Memory::CITRA_PAGE_BITS);
}

bool InPlaceData::Use(PAddr addr, u32 size, u64 tick) {
    // All of it has to be memory that is kept track of
    const bool in_fcram = addr >= Memory::FCRAM_PADDR && size <= Memory::FCRAM_N3DS_SIZE &&
                          addr - Memory::FCRAM_PADDR <= Memory::FCRAM_N3DS_SIZE - size;
    const bool in_vram = addr >= Memory::VRAM_PADDR && size <= Memory::VRAM_SIZE &&
                         addr - Memory::VRAM_PADDR <= Memory::VRAM_SIZE - size;
    const auto range = PagesOf(addr, size);
    if ((!in_fcram && !in_vram) || !range) {
        return false;
    }
    if (pages.empty()) {
        pages.resize(FCRAM_PAGES + VRAM_PAGES);
    }

    bool in_place = true;
    // Pages next to each other are handed over to be watched in one piece
    u32 run_start = 0;
    u32 run_length = 0;
    const auto watch_run = [&] {
        if (run_length != 0) {
            mark(AddressOf(run_start), run_length << Memory::CITRA_PAGE_BITS, true);
            run_length = 0;
        }
    };
    for (u32 index = range->first; index <= range->second; index++) {
        Page& page = pages[index];
        const bool begin =
            page.state == State::Unknown || (page.state == State::Copied && frame >= page.retry);
        if (begin) {
            if (run_length == 0) {
                run_start = index;
            }
            ++run_length;
            ++watched;
            ++begun;
            page.state = State::Trial;
            page.frame = frame;
            page.calm = 0;
            page.reads = 0;
            in_place = false;
            continue;
        }
        watch_run();
        if (page.state == State::Trial) {
            if (page.frame != frame) {
                page.frame = frame;
                if (++page.calm >= CALM_FRAMES) {
                    page.state = State::InPlace;
                    ++promoted;
                }
            }
        }
        if (page.state != State::InPlace) {
            in_place = false;
        }
    }
    watch_run();

    if (in_place) {
        for (u32 index = range->first; index <= range->second; index++) {
            pages[index].last_use = tick;
        }
    }
    return in_place;
}

void InPlaceData::GiveUp(u32 index) {
    Page& page = pages[index];
    mark(AddressOf(index), Memory::CITRA_PAGE_SIZE, false);
    --watched;
    ++generation;
    page.state = State::Copied;
    page.failures = static_cast<u8>(std::min<u32>(page.failures + 1, MAX_FAILURES));
    page.retry = frame + (RETRY_FRAMES << (page.failures - 1));
}

u64 InPlaceData::OnWrite(PAddr addr, u32 size, u64 gpu_tick) {
    if (watched == 0) {
        return 0;
    }
    const auto range = PagesOf(addr, size);
    if (!range) {
        return 0;
    }
    u64 wait_for = 0;
    for (u32 index = range->first; index <= range->second; index++) {
        Page& page = pages[index];
        if (page.state != State::Trial && page.state != State::InPlace) {
            continue;
        }
        if (page.state == State::InPlace && page.last_use > gpu_tick) {
            wait_for = std::max(wait_for, page.last_use);
        }
        GiveUp(index);
        ++ended_by_write;
    }
    return wait_for;
}

void InPlaceData::OnRead(PAddr addr, u32 size) {
    if (watched == 0) {
        return;
    }
    const auto range = PagesOf(addr, size);
    if (!range) {
        return;
    }
    const u32 window = frame / READ_WINDOW;
    for (u32 index = range->first; index <= range->second; index++) {
        Page& page = pages[index];
        if (page.state != State::Trial && page.state != State::InPlace) {
            continue;
        }
        if (page.read_window != window) {
            page.read_window = window;
            page.reads = 0;
        }
        if (++page.reads > READ_LIMIT) {
            GiveUp(index);
            ++ended_by_read;
        }
    }
}

void InPlaceData::OnFrame() {
    ++frame;
    if (!log || ++report_frames < REPORT_FRAMES) {
        return;
    }
    u32 in_place = 0;
    u32 on_trial = 0;
    u32 copied = 0;
    for (const Page& page : pages) {
        in_place += page.state == State::InPlace;
        on_trial += page.state == State::Trial;
        copied += page.state == State::Copied;
    }
    LOG_INFO(Render_Vulkan,
             "Data in place, {} frames: pages read in place {}, on trial {}, copied {}. Since "
             "the last line: {} trials begun, {} pages taken to in place; ended by a write {}, "
             "by reads {}; {} writes waited for the GPU, {:.1f} ms in all",
             report_frames, in_place, on_trial, copied, begun, promoted, ended_by_write,
             ended_by_read, waits, std::chrono::duration<double, std::milli>(waited).count());
    report_frames = 0;
    begun = promoted = ended_by_write = ended_by_read = waits = 0;
    waited = {};
}

void InPlaceData::Reset() {
    // In runs of pages next to each other
    u32 run_start = 0;
    u32 run_length = 0;
    for (u32 index = 0; index <= pages.size(); index++) {
        const bool is_watched =
            index < pages.size() &&
            (pages[index].state == State::Trial || pages[index].state == State::InPlace) &&
            index != FCRAM_PAGES;
        if (is_watched && run_length != 0) {
            ++run_length;
            continue;
        }
        if (run_length != 0) {
            mark(AddressOf(run_start), run_length << Memory::CITRA_PAGE_BITS, false);
            run_length = 0;
        }
        if (index < pages.size() &&
            (pages[index].state == State::Trial || pages[index].state == State::InPlace)) {
            run_start = index;
            run_length = 1;
        }
    }
    pages.clear();
    watched = 0;
    ++generation;
}

void GuestDataWatch::NoteUse(const u8* data, u32 size, bool is_indices, bool is_in_place,
                             u64 tick) {
    if (!data || size == 0) {
        return;
    }
    const u64 hash = Common::ComputeHash64(data, size);
    all.Add(size);
    if (is_indices) {
        indices.Add(size);
    }
    if (is_in_place) {
        in_place.Add(size);
    }

    const u64 key = reinterpret_cast<std::uintptr_t>(data) ^ (static_cast<u64>(size) << 44);
    if (const auto it = last_hash.find(key); it != last_hash.end()) {
        (it->second == hash ? unchanged : changed).Add(size);
        it->second = hash;
    } else {
        fresh.Add(size);
        if (last_hash.size() >= MAX_REMEMBERED) {
            last_hash.clear();
        }
        last_hash.emplace(key, hash);
    }

    recorded.push_back({data, size, hash, tick, false});
}

void GuestDataWatch::OnSubmit() {
    for (Use& use : recorded) {
        if (Common::ComputeHash64(use.data, use.size) != use.hash) {
            use.rewritten = true;
            rewritten_unsent.Add(use.size);
        }
    }
    sent.insert(sent.end(), recorded.begin(), recorded.end());
    recorded.clear();
}

void GuestDataWatch::OnFrame(u64 gpu_tick) {
    std::erase_if(sent, [&](const Use& use) {
        if (use.tick > gpu_tick) {
            return false;
        }
        if (!use.rewritten && Common::ComputeHash64(use.data, use.size) != use.hash) {
            rewritten_pending.Add(use.size);
        }
        return true;
    });
    if (++frames >= REPORT_FRAMES) {
        Report();
    }
}

void GuestDataWatch::Report() {
    const double per_frame = 1.0 / frames;
    const auto share = [&](const Count& count) {
        return all.bytes ? 100.0 * count.bytes / all.bytes : 0.0;
    };
    LOG_INFO(Render_Vulkan,
             "Vertex data, {} frames: {:.0f} draws a frame take {:.0f} KiB in {:.0f} pieces "
             "({:.0f} KiB of it indices), {:.1f}% in place. Of the bytes: as when last drawn "
             "with {:.1f}%, written anew {:.1f}%, not drawn with before {:.1f}%. Written anew "
             "before the commands were sent: {:.2f}% ({:.1f} pieces a frame); after that, "
             "before the GPU was seen to be done: {:.2f}% ({:.1f} pieces a frame)",
             frames, draws * per_frame, all.bytes * per_frame / 1024.0, all.uses * per_frame,
             indices.bytes * per_frame / 1024.0, share(in_place), share(unchanged), share(changed),
             share(fresh), share(rewritten_unsent), rewritten_unsent.uses * per_frame,
             share(rewritten_pending), rewritten_pending.uses * per_frame);
    frames = 0;
    draws = 0;
    all = indices = in_place = unchanged = changed = fresh = {};
    rewritten_unsent = rewritten_pending = {};
}

} // namespace Vulkan
