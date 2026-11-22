// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <cstring>
#include <string_view>
#include <thread>
#include <fmt/format.h>
#include "common/alignment.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "common/trace_markers.h"
#include "core/arm/dynarmic/jit_code_cache.h"

namespace Core {

namespace {

constexpr u32 FILE_MAGIC = 0x434A5A41; // "AZJC"
/// Changes with the layout of the file
constexpr u32 FILE_VERSION = 1;

/// Blocks are written to the file when this much has gathered, or after this long
constexpr std::size_t FLUSH_SIZE = 256 * 1024;
constexpr std::chrono::seconds FLUSH_INTERVAL{3};
/// A write that takes longer is worth a line in the log
constexpr std::chrono::milliseconds SLOW_WRITE{100};
/// Nothing more is kept beyond this size
constexpr std::size_t MAX_SIZE = 256 * 1024 * 1024;
/// Blocks kept per location: code of other programs or modules at the same address
constexpr std::size_t MAX_BLOCKS_PER_LOCATION = 4;

struct FileHeader {
    u32 magic;
    u32 version;
    u64 program_id;
    u64 build_key;
};
static_assert(sizeof(FileHeader) == 24);

struct RecordHeader {
    u64 location;
    u32 size;
    u32 reserved;
};
static_assert(sizeof(RecordHeader) == 16);

/// Records start at multiples of this
constexpr std::size_t RECORD_ALIGNMENT = 8;

/// Identifies the recompiler: the code of a block relies on details of the recompiler and of
/// the state it works on. The build hashes the sources of both and how the recompiler is set
/// up (CMakeModules/GenerateSCMRev.cmake), so what is kept outlives the builds that leave
/// them alone. Nothing of the host's processor goes into the code, and the game's own code is
/// checked block by block where the blocks are taken up again.
u64 BuildKey(bool host_mmu) {
    std::string recompiler{Common::g_recompiler_cache_version};
    if (host_mmu) {
        recompiler += "|host MMU access";
    }
    return Common::ComputeHash64(recompiler.data(), recompiler.size());
}

std::string CachePath(u64 program_id) {
    return fmt::format("{}jit/{:016X}.bin", FileUtil::GetUserPath(FileUtil::UserPath::CacheDir),
                       program_id);
}

} // Anonymous namespace

JitCodeCache::JitCodeCache(u64 program_id_, bool host_mmu_)
    : program_id{program_id_}, host_mmu{host_mmu_}, path{CachePath(program_id)} {
    Load();
    last_flush = last_log = std::chrono::steady_clock::now();
    if (rewrite_file && !data.empty()) {
        // A file whose end was cut off is written from scratch now, while the game is
        // starting: all of it is copied for the writer, which is not for the middle of a game
        Flush();
    }
}

JitCodeCache::~JitCodeCache() {
    Flush();
    writer.WaitForRequests();
    if (append_failed.load()) {
        // The last addition failed: from scratch, then
        Flush();
        writer.WaitForRequests();
    }
}

void JitCodeCache::Load() {
    FileUtil::IOFile file{path, "rb"};
    if (!file.IsOpen()) {
        LOG_INFO(Core_ARM11, "No recompiled code kept for title {:016X} yet", program_id);
        return;
    }

    FileHeader header{};
    const u64 file_size = file.GetSize();
    if (file_size < sizeof(header) || file_size - sizeof(header) > MAX_SIZE ||
        file.ReadBytes(&header, sizeof(header)) != sizeof(header) || header.magic != FILE_MAGIC ||
        header.version != FILE_VERSION || header.program_id != program_id ||
        header.build_key != BuildKey(host_mmu)) {
        LOG_INFO(Core_ARM11,
                 "The recompiled code kept for title {:016X} is from another recompiler, starting "
                 "over",
                 program_id);
        return;
    }

    data.resize(file_size - sizeof(header));
    if (file.ReadBytes(data.data(), data.size()) != data.size()) {
        LOG_ERROR(Core_ARM11, "Could not read the recompiled code of title {:016X}", program_id);
        data.clear();
        return;
    }

    // A file that ends in the middle of a record was being written when the emulator was
    // stopped. What came before is good; the file is written anew without the rest.
    const bool complete = ParseRecords();
    flushed_size = data.size();
    rewrite_file = !complete;
    if (complete) {
        // Opened for adding to it now, while the game is still starting: on the storage of a
        // phone this can take a tenth of a second, which the first new block of a session
        // would otherwise wait for in the middle of the game
        file.Close();
        append_file = FileUtil::IOFile{path, "ab"};
    }

    LOG_INFO(Core_ARM11, "Kept recompiled code of title {:016X}: {} locations, {} KB{}", program_id,
             entries.size(), data.size() / 1024, complete ? "" : " (the end was cut off)");
}

bool JitCodeCache::ParseRecords() {
    std::size_t offset = 0;
    while (offset < data.size()) {
        RecordHeader record;
        if (data.size() - offset < sizeof(record)) {
            break;
        }
        std::memcpy(&record, data.data() + offset, sizeof(record));
        const std::size_t block_offset = offset + sizeof(record);
        if (record.size == 0 || data.size() - block_offset < record.size) {
            break;
        }
        auto& blocks = entries[record.location];
        if (blocks.size() == MAX_BLOCKS_PER_LOCATION) {
            blocks.erase(blocks.begin());
        }
        blocks.push_back(Entry{block_offset, record.size});
        offset = Common::AlignUp(block_offset + record.size, RECORD_ALIGNMENT);
        if (offset > data.size()) {
            // The padding of the last record is missing, which is all that is
            data.resize(offset);
        }
    }
    if (offset >= data.size()) {
        return true;
    }
    data.resize(offset);
    return false;
}

std::span<const std::byte> JitCodeCache::Lookup(u64 location, std::size_t index) {
    const auto it = entries.find(location);
    if (it == entries.end() || index >= it->second.size()) {
        return {};
    }
    if (index == 0) {
        LogActivity(std::chrono::steady_clock::now());
        found++;
        last_found = true;
        last_found_location = location;
    }
    // The newest is the last one
    const Entry& entry = it->second[it->second.size() - 1 - index];
    return std::span{data}.subspan(entry.offset, entry.size);
}

void JitCodeCache::Store(u64 location, std::span<const std::byte> block) {
    const auto now = std::chrono::steady_clock::now();

    stored++;
    if (last_found && last_found_location == location) {
        // What was kept for the location did not fit the guest code that is there now
        replaced++;
    }
    last_found = false;
    LogActivity(now);

    if (full || block.empty()) {
        return;
    }
    if (data.size() + block.size() > MAX_SIZE) {
        LOG_WARNING(Core_ARM11,
                    "The recompiled code kept for title {:016X} reached {} MB, not "
                    "keeping more",
                    program_id, MAX_SIZE / (1024 * 1024));
        full = true;
        return;
    }

    const RecordHeader record{
        .location = location,
        .size = static_cast<u32>(block.size()),
        .reserved = 0,
    };
    const std::size_t record_offset = data.size();
    data.resize(Common::AlignUp(record_offset + sizeof(record) + block.size(), RECORD_ALIGNMENT));
    std::memcpy(data.data() + record_offset, &record, sizeof(record));
    std::memcpy(data.data() + record_offset + sizeof(record), block.data(), block.size());

    auto& blocks = entries[location];
    if (blocks.size() == MAX_BLOCKS_PER_LOCATION) {
        blocks.erase(blocks.begin());
    }
    blocks.push_back(Entry{record_offset + sizeof(record), record.size});

    if (data.size() - flushed_size >= FLUSH_SIZE || now - last_flush >= FLUSH_INTERVAL) {
        Flush();
    }
}

void JitCodeCache::Flush() {
    last_flush = std::chrono::steady_clock::now();
    if (write_gave_up.load(std::memory_order_relaxed)) {
        full = true;
        return;
    }
    if (append_failed.exchange(false)) {
        // What the file holds now is unknown: written from scratch
        rewrite_file = true;
    }
    if (flushed_size == data.size() && !rewrite_file) {
        return;
    }

    // A copy for the writer, as data moves when it grows: a quarter of a megabyte as a rule,
    // all of it only where the file is written from scratch
    const std::size_t from = rewrite_file ? 0 : flushed_size;
    std::vector<std::byte> records(data.begin() + static_cast<std::ptrdiff_t>(from), data.end());
    writer.QueueWork([this, records = std::move(records), whole_file = rewrite_file] {
        WriteOut(records, whole_file);
    });
    rewrite_file = false;
    flushed_size = data.size();
}

void JitCodeCache::WriteOut(std::span<const std::byte> records, bool whole_file) {
    if (write_gave_up.load(std::memory_order_relaxed) || (!whole_file && writer_needs_whole)) {
        // Nothing more is written, or the request that writes the file from scratch follows
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    if (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::SlowCodeStorage) {
        std::this_thread::sleep_for(std::chrono::seconds{5});
    }
    if (whole_file) {
        append_file.Close();
        if (!FileUtil::CreateFullPath(path)) {
            LOG_ERROR(Core_ARM11, "Could not create the directory of {}", path);
            write_gave_up = true;
            return;
        }
        // The file stays open: what comes later is added through the same handle
        append_file = FileUtil::IOFile{path, "wb"};
        const FileHeader header{
            .magic = FILE_MAGIC,
            .version = FILE_VERSION,
            .program_id = program_id,
            .build_key = BuildKey(host_mmu),
        };
        if (!append_file.IsOpen() ||
            append_file.WriteBytes(&header, sizeof(header)) != sizeof(header) ||
            append_file.WriteBytes(records.data(), records.size()) != records.size() ||
            !append_file.Flush()) {
            LOG_ERROR(Core_ARM11, "Could not write {}", path);
            append_file.Close();
            write_gave_up = true;
            return;
        }
        writer_needs_whole = false;
    } else {
        if (!append_file.IsOpen()) {
            append_file = FileUtil::IOFile{path, "ab"};
        }
        if (!append_file.IsOpen() ||
            append_file.WriteBytes(records.data(), records.size()) != records.size() ||
            !append_file.Flush()) {
            LOG_ERROR(Core_ARM11, "Could not add to {}", path);
            // What the file holds now is unknown
            append_file.Close();
            writer_needs_whole = true;
            append_failed = true;
            return;
        }
    }

    const auto took = std::chrono::steady_clock::now() - start;
    if (took >= SLOW_WRITE) {
        LOG_WARNING(Core_ARM11,
                    "The storage took {} ms to write {} KB of recompiled code (the game did not "
                    "wait for it)",
                    std::chrono::duration_cast<std::chrono::milliseconds>(took).count(),
                    records.size() / 1024);
    }
}

void JitCodeCache::LogActivity(std::chrono::steady_clock::time_point now) {
    if (now - last_log < std::chrono::seconds{5}) {
        return;
    }
    last_log = now;
    if (found == 0 && stored == 0) {
        return;
    }
    // "Reused" counts the locations something was kept for, less those where it did not fit
    LOG_INFO(Core_ARM11,
             "Recompiler in the last 5 s: {} blocks reused, {} translated ({} of them "
             "in place of kept code that did not fit), {} KB kept",
             found - replaced, stored, replaced, data.size() / 1024);
    found = 0;
    replaced = 0;
    stored = 0;
}

} // namespace Core
