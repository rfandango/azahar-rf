// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <dynarmic/interface/A32/translated_ahead.h>
#include "common/common_types.h"

namespace Kernel {
class Process;
struct VirtualMemoryArea;
} // namespace Kernel

namespace Core {

/**
 * Has a title's code translated before it runs, on a thread of its own: what ran in earlier
 * sessions first, which a small file per title says, and then what that leads to.
 *
 * The thread works on a copy of the title's code, so it needs nothing of the emulated system, and
 * whatever it makes is compared with what is in memory where it is used. Nothing that goes wrong
 * here, an old file, a file of something else, code that has changed, can do more harm than
 * having been done in vain.
 *
 * The file lists locations and no code: what a title ran in hours is some ten thousand of them in
 * a hundred and fifty kilobytes, where the code itself was tens of megabytes, all of them read and
 * kept in memory, and thrown away with every change of the recompiler.
 */
class AheadTranslation {
public:
    explicit AheadTranslation(u64 program_id);
    /// Ends the thread, and puts what is new into the file
    ~AheadTranslation();

    AheadTranslation(const AheadTranslation&) = delete;
    AheadTranslation& operator=(const AheadTranslation&) = delete;

    u64 GetProgramId() const {
        return program_id;
    }

    Dynarmic::A32::TranslatedAhead& Translated() {
        return translated;
    }

    /// From the thread that runs the emulated system: a process is about to run. Takes the copy
    /// of its code.
    void Begin(const Kernel::Process& process);

    /// Likewise: code was translated there because it was asked for (see
    /// Dynarmic::A32::UserCallbacks::TranslatedOnDemand)
    void NoteOnDemand(u64 location, const Kernel::Process* process);

    /// From any thread: puts what is new into the file, soon
    void Flush();

private:
    /// A stretch of the title's memory
    struct Copy {
        u32 base;
        u32 end;
        bool read_only;
        std::vector<u32> words;
        /// Which of them have been translated
        std::vector<bool> covered;
        std::chrono::steady_clock::time_point taken;
    };
    /// Where code begins, with the word there
    struct Entry {
        u64 location;
        u32 first_word;
    };
    struct Request {
        /// A new copy, or
        std::shared_ptr<Copy> copy;
        /// where code was asked for
        u64 location = 0;
        u32 first_word = 0;
    };

    class Memory;

    void Load();
    bool LoadFrom(const std::string& from);
    /// Locations the recompiler stored code for before there was this list
    void ReadStoredCode();
    void AddEntry(u64 location, u32 first_word);
    std::shared_ptr<Copy> TakeCopy(const Kernel::VirtualMemoryArea& area) const;

    // The thread's
    void Work();
    void TakeRequest(Request&& request);
    /// Does one thing. False if there is nothing to do.
    bool DoOne();
    bool Translate(u64 location);
    void FollowUp(u64 location, u32 depth);
    void WriteOut(bool all);
    void LogActivity();

    u64 program_id;
    std::string path;

    Dynarmic::A32::TranslatedAhead translated;

    // Of the thread that runs the emulated system
    /// What was copied: where, and when
    struct Copied {
        u32 base, end;
        std::shared_ptr<const Copy> copy;
    };
    std::vector<Copied> copied;

    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Request> requests;
    bool flush_wanted = false;
    bool stop_wanted = false;

    // Of the working thread
    std::vector<std::shared_ptr<Copy>> copies;
    /// What the file has, in the order it was run; and which locations those are
    std::vector<Entry> entries;
    std::unordered_map<u64, boost::container::small_vector<u32, 1>> known_entries;
    /// How many of the entries are in the file
    std::size_t written = 0;
    bool rewrite_file = false;
    /// Where else code begins that may have run: only good for where no code was translated
    std::vector<Entry> guesses;
    /// What of them is still to be gone through for the copy that came last
    std::size_t next_entry = 0, next_guess = 0;
    /// What comes after something that ran, and how far after it
    struct Lead {
        u64 location;
        u32 depth;
    };
    std::deque<Lead> leads;
    std::unordered_set<u64> led_to;
    std::size_t made_bytes = 0, led_to_bytes = 0;
    std::chrono::steady_clock::time_point last_write, last_log;

    // Since the last line in the log
    std::size_t made = 0, made_instructions = 0;
    std::chrono::nanoseconds worked{};
    Dynarmic::A32::TranslatedAhead::Statistics last_statistics;

    /// The last member: it ends before what it works on
    std::thread thread;
};

} // namespace Core
