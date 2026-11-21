// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <chrono>
#include <cstddef>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <boost/container/small_vector.hpp>
#include "common/common_types.h"
#include "common/file_util.h"

namespace Core {

/**
 * The code the recompiler produced for one title, kept so that it does not have to be produced
 * again: not after a save state was loaded, which starts the recompiler anew, and not in the
 * next session, for which the code is also written to a file as it comes.
 *
 * What a block of code looks like is the recompiler's business (see Dynarmic::CodeCache). It
 * checks a block against the guest code in memory before using it, so blocks of other code that
 * was at the same address are harmless, and several are kept per location.
 *
 * A file is only read back by a build with the recompiler that wrote it.
 */
class JitCodeCache {
public:
    /// `host_mmu`: whether the recompiler is set up for host MMU access, which makes for
    /// other code: what is kept of one kind is not taken for the other
    JitCodeCache(u64 program_id, bool host_mmu);
    ~JitCodeCache();

    JitCodeCache(const JitCodeCache&) = delete;
    JitCodeCache& operator=(const JitCodeCache&) = delete;

    u64 GetProgramId() const {
        return program_id;
    }

    /// Returns a block kept for the location, the newest for index 0, or an empty span. Valid
    /// until the next call to Store.
    std::span<const std::byte> Lookup(u64 location, std::size_t index);

    /// Keeps a block for the location.
    void Store(u64 location, std::span<const std::byte> block);

    /// Writes the blocks that are not in the file yet.
    void Flush();

private:
    struct Entry {
        std::size_t offset;
        u32 size;
    };

    void Load();
    bool ParseRecords();
    void LogActivity(std::chrono::steady_clock::time_point now);

    u64 program_id;
    bool host_mmu;
    std::string path;

    /// The records of all blocks, as they are in the file after its header
    std::vector<std::byte> data;
    std::unordered_map<u64, boost::container::small_vector<Entry, 1>> entries;

    /// How much of data the file holds, and whether the file has to be written from scratch
    std::size_t flushed_size{};
    bool rewrite_file{true};
    /// The file, open for adding to it from the moment it was read or written anew: opening
    /// it can take long on some storage
    FileUtil::IOFile append_file;
    bool full{};
    std::chrono::steady_clock::time_point last_flush{};

    // What happened since the last log line
    u64 last_found_location{};
    bool last_found{};
    u32 found{};
    u32 replaced{};
    u32 stored{};
    std::chrono::steady_clock::time_point last_log{};
};

} // namespace Core
