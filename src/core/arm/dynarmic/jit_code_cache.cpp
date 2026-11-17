// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <cstring>
#include <string_view>
#include <fmt/format.h>
#include "common/alignment.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "core/arm/dynarmic/jit_code_cache.h"

namespace Core {

namespace {

constexpr u32 FILE_MAGIC = 0x434A5A41; // "AZJC"
/// Changes with the layout of the file
constexpr u32 FILE_VERSION = 1;

/// Blocks are written to the file when this much has gathered, or after this long
constexpr std::size_t FLUSH_SIZE = 256 * 1024;
constexpr std::chrono::seconds FLUSH_INTERVAL{3};
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

/// Identifies the build: the code of a block relies on details of the recompiler and of the
/// state it works on, which are only known to be the same within one build.
u64 BuildKey() {
    const std::string build =
        fmt::format("{}|{}|{}", Common::g_scm_rev, Common::g_build_date, Common::g_build_fullname);
    return Common::ComputeHash64(build.data(), build.size());
}

std::string CachePath(u64 program_id) {
    return fmt::format("{}jit/{:016X}.bin", FileUtil::GetUserPath(FileUtil::UserPath::CacheDir),
                       program_id);
}

} // Anonymous namespace

JitCodeCache::JitCodeCache(u64 program_id_) : program_id{program_id_}, path{CachePath(program_id)} {
    Load();
    last_flush = last_log = std::chrono::steady_clock::now();
}

JitCodeCache::~JitCodeCache() {
    Flush();
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
        header.build_key != BuildKey()) {
        LOG_INFO(Core_ARM11,
                 "The recompiled code kept for title {:016X} is from another build, starting over",
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
    if (flushed_size == data.size() && !rewrite_file) {
        return;
    }

    if (rewrite_file) {
        append_file.Close();
        if (!FileUtil::CreateFullPath(path)) {
            LOG_ERROR(Core_ARM11, "Could not create the directory of {}", path);
            full = true;
            return;
        }
        FileUtil::IOFile file{path, "wb"};
        const FileHeader header{
            .magic = FILE_MAGIC,
            .version = FILE_VERSION,
            .program_id = program_id,
            .build_key = BuildKey(),
        };
        if (!file.IsOpen() || file.WriteBytes(&header, sizeof(header)) != sizeof(header) ||
            file.WriteBytes(data.data(), data.size()) != data.size()) {
            LOG_ERROR(Core_ARM11, "Could not write {}", path);
            full = true;
            return;
        }
        rewrite_file = false;
    } else {
        if (!append_file.IsOpen()) {
            append_file = FileUtil::IOFile{path, "ab"};
        }
        const std::size_t size = data.size() - flushed_size;
        if (!append_file.IsOpen() ||
            append_file.WriteBytes(data.data() + flushed_size, size) != size ||
            !append_file.Flush()) {
            LOG_ERROR(Core_ARM11, "Could not add to {}", path);
            // What the file holds now is unknown: written anew next time
            append_file.Close();
            rewrite_file = true;
            return;
        }
    }
    flushed_size = data.size();
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
