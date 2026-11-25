// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/arch.h"
#if CITRA_ARCH(arm64)

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <optional>
#include <fmt/format.h>
#include "common/file_util.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "common/trace_markers.h"
#include "core/arm/dynarmic/ahead_translation.h"
#include "core/arm/dynarmic/arm_tick_counts.h"
#include "core/hle/kernel/process.h"
#include "core/hle/kernel/vm_manager.h"

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace Core {

namespace {

constexpr u32 FILE_MAGIC = 0x4C4A5A41; // "AZJL"
constexpr u32 FILE_VERSION = 1;

struct FileHeader {
    u32 magic;
    u32 version;
    u64 program_id;
};
static_assert(sizeof(FileHeader) == 16);

/// Each record stands for itself: one that was not written whole, or has changed since, is passed
/// over and costs nothing but itself
struct Record {
    u64 location;
    u32 first_word;
    u32 check;
};
static_assert(sizeof(Record) == 16);

u32 CheckOf(u64 program_id, u64 location, u32 first_word) {
    const std::array<u64, 3> what{program_id, location, u64{first_word} | u64{FILE_VERSION} << 32};
    return static_cast<u32>(Common::ComputeHash64(what.data(), sizeof(what)));
}

/// More than any title has
constexpr std::size_t MOST_ENTRIES = 128 * 1024;
/// Words kept per location: of other programs or modules at the same address
constexpr std::size_t MOST_WORDS_PER_LOCATION = 4;

/// How far what is translated runs ahead of what ran
constexpr u32 MOST_LEADS = 3;
/// And how much code that may come to
constexpr std::size_t MOST_LED_TO_BYTES = 4 * 1024 * 1024;

constexpr std::chrono::seconds WRITE_INTERVAL{10};
constexpr std::chrono::seconds LOG_INTERVAL{5};
/// A stretch of memory that has changed is not copied again before this long
constexpr std::chrono::seconds COPY_INTERVAL{2};

/// What the title is known to run is wanted soon, mostly while it starts. What may come after
/// something new is not: everything else comes first.
void SetUrgent(bool urgent) {
#ifdef __linux__
    setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), urgent ? 0 : 10);
#else
    Common::SetCurrentThreadPriority(urgent ? Common::ThreadPriority::Normal
                                            : Common::ThreadPriority::Low);
#endif
}

/// Memory code can be in
bool HoldsCode(const Kernel::VirtualMemoryArea& area) {
    if (area.type != Kernel::VMAType::BackingMemory) {
        return false;
    }
    const auto permissions = static_cast<u8>(area.permissions);
    const bool executed = (permissions & static_cast<u8>(Kernel::VMAPermission::Execute)) != 0;
    const bool written = (permissions & static_cast<u8>(Kernel::VMAPermission::Write)) != 0;
    return executed || (area.meminfo_state == Kernel::MemoryState::Code && !written);
}

} // Anonymous namespace

class AheadTranslation::Memory final : public Dynarmic::A32::TranslatedAhead::Memory {
public:
    explicit Memory(const std::vector<std::shared_ptr<Copy>>& copies_) : copies{copies_} {}

    std::optional<std::uint32_t> Read(std::uint32_t vaddr) override {
        const Copy* const copy = Find(vaddr);
        return copy ? std::optional<u32>{copy->words[(vaddr - copy->base) / 4]} : std::nullopt;
    }

    bool IsReadOnly(std::uint32_t vaddr) override {
        const Copy* const copy = Find(vaddr);
        return copy && copy->read_only;
    }

    std::uint64_t GetTicksForCode(std::uint32_t, std::uint32_t instruction) override {
        if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::CountGuestInstructions) !=
            0) [[unlikely]] {
            return 1;
        }
        return Core::TicksForInstruction(false, instruction);
    }

    Copy* Find(u32 vaddr) {
        if (last && vaddr >= last->base && vaddr < last->end) {
            return last;
        }
        for (const auto& copy : copies) {
            if (vaddr >= copy->base && vaddr < copy->end) {
                last = copy.get();
                return last;
            }
        }
        return nullptr;
    }

private:
    const std::vector<std::shared_ptr<Copy>>& copies;
    Copy* last = nullptr;
};

AheadTranslation::AheadTranslation(u64 program_id_)
    : program_id{program_id_}, path{fmt::format("{}jit/{:016X}.list",
                                                FileUtil::GetUserPath(FileUtil::UserPath::CacheDir),
                                                program_id)} {
    last_write = last_log = std::chrono::steady_clock::now();
    thread = std::thread([this] { Work(); });
}

AheadTranslation::~AheadTranslation() {
    {
        std::scoped_lock lock{mutex};
        stop_wanted = true;
    }
    wake.notify_one();
    thread.join();
    for (Request& request : requests) {
        if (!request.copy) {
            AddEntry(request.location, request.first_word);
        }
    }
    WriteOut(true);
}

std::shared_ptr<AheadTranslation::Copy> AheadTranslation::TakeCopy(
    const Kernel::VirtualMemoryArea& area) const {
    auto copy = std::make_shared<Copy>();
    copy->base = area.base;
    copy->end = area.base + (area.size & ~u32{3});
    copy->read_only = area.meminfo_state == Kernel::MemoryState::Code &&
                      (static_cast<u8>(area.permissions) &
                       static_cast<u8>(Kernel::VMAPermission::Write)) == 0;
    copy->words.resize(area.size / 4);
    copy->covered.resize(area.size / 4);
    copy->taken = std::chrono::steady_clock::now();
    std::memcpy(copy->words.data(), area.backing_memory.GetPtr(), copy->words.size() * 4);
    return copy;
}

void AheadTranslation::Begin(const Kernel::Process& process) {
    for (const auto& [base, area] : process.vm_manager.vma_map) {
        if (!HoldsCode(area)) {
            continue;
        }
        auto copy = TakeCopy(area);
        std::erase_if(copied, [&](const Copied& other) {
            return other.base < copy->end && copy->base < other.end;
        });
        copied.push_back({copy->base, copy->end, copy});
        std::scoped_lock lock{mutex};
        requests.push_back({.copy = std::move(copy)});
    }
    wake.notify_one();
}

void AheadTranslation::NoteOnDemand(u64 location, const Kernel::Process* process) {
    const u32 pc = static_cast<u32>(location);
    if (!process) {
        return;
    }
    const auto found = process->vm_manager.FindVMA(pc);
    if (found == process->vm_manager.vma_map.end() || !HoldsCode(found->second)) {
        return;
    }
    const Kernel::VirtualMemoryArea& area = found->second;
    if (pc - area.base + 4 > area.size) {
        return;
    }
    u32 word;
    std::memcpy(&word, area.backing_memory.GetPtr() + (pc - area.base), sizeof(word));

    // Is it part of what was copied, and as it was then?
    const auto known = std::find_if(copied.begin(), copied.end(), [&](const Copied& other) {
        return pc >= other.base && pc < other.end;
    });
    std::shared_ptr<Copy> copy;
    if (known == copied.end()) {
        copy = TakeCopy(area);
    } else if (known->copy->words[(pc - known->base) / 4] != word &&
               std::chrono::steady_clock::now() - known->copy->taken >= COPY_INTERVAL) {
        copy = TakeCopy(area);
    }
    if (copy) {
        std::erase_if(copied, [&](const Copied& other) {
            return other.base < copy->end && copy->base < other.end;
        });
        copied.push_back({copy->base, copy->end, copy});
    }
    {
        std::scoped_lock lock{mutex};
        if (copy) {
            requests.push_back({.copy = std::move(copy)});
        }
        requests.push_back({.location = location, .first_word = word});
    }
    wake.notify_one();
}

void AheadTranslation::Flush() {
    {
        std::scoped_lock lock{mutex};
        flush_wanted = true;
    }
    wake.notify_one();
}

// The thread's
// ----------------------------------------------------------------------------------------------

void AheadTranslation::AddEntry(u64 location, u32 first_word) {
    if (entries.size() >= MOST_ENTRIES) {
        return;
    }
    auto& words = known_entries[location];
    if (std::find(words.begin(), words.end(), first_word) != words.end()) {
        return;
    }
    if (words.size() == MOST_WORDS_PER_LOCATION) {
        return;
    }
    words.push_back(first_word);
    entries.push_back({location, first_word});
}

void AheadTranslation::Load() {
    // One that was written beside the other when the emulator was stopped is as good
    if (!LoadFrom(path)) {
        rewrite_file = LoadFrom(path + ".new") || rewrite_file;
    }
    if (entries.empty()) {
        ReadStoredCode();
    }
}

bool AheadTranslation::LoadFrom(const std::string& from) {
    FileUtil::IOFile file{from, "rb"};
    if (!file.IsOpen()) {
        return false;
    }
    FileHeader header{};
    const u64 size = file.GetSize();
    if (size < sizeof(header) || file.ReadBytes(&header, sizeof(header)) != sizeof(header) ||
        header.magic != FILE_MAGIC || header.version != FILE_VERSION ||
        header.program_id != program_id) {
        LOG_WARNING(Core_ARM11, "{} is no list of this title's code", from);
        rewrite_file = true;
        return false;
    }
    std::vector<Record> records(std::min<u64>((size - sizeof(header)) / sizeof(Record),
                                              2 * MOST_ENTRIES));
    const std::size_t read = file.ReadBytes(records.data(), records.size() * sizeof(Record));
    records.resize(read / sizeof(Record));
    std::size_t bad = 0;
    for (const Record& record : records) {
        if (record.check != CheckOf(program_id, record.location, record.first_word)) {
            bad++;
            continue;
        }
        AddEntry(record.location, record.first_word);
    }
    written = entries.size();
    // A record that is no good, or cut off, or there twice: written from scratch
    rewrite_file = (size - sizeof(header)) % sizeof(Record) != 0 || written != records.size();
    LOG_INFO(Core_ARM11, "Code of title {:016X} that ran before: {} places{}", program_id, written,
             bad ? fmt::format(" ({} records were no good)", bad) : std::string{});
    return !entries.empty();
}

void AheadTranslation::ReadStoredCode() {
    // What the recompiler stored before there was this list, whichever recompiler that was: the
    // locations and the program's code are in it, and what the file looks like has not changed
    const std::string stored =
        fmt::format("{}jit/{:016X}.bin", FileUtil::GetUserPath(FileUtil::UserPath::CacheDir),
                    program_id);
    FileUtil::IOFile file{stored, "rb"};
    if (!file.IsOpen()) {
        return;
    }
    struct StoredHeader {
        u32 magic, version;
        u64 program_id, build_key;
    } header{};
    const u64 size = file.GetSize();
    if (size < sizeof(header) || file.ReadBytes(&header, sizeof(header)) != sizeof(header) ||
        header.magic != 0x434A5A41 || header.program_id != program_id) {
        return;
    }
    struct StoredRecord {
        u64 location;
        u32 size, reserved;
    };
    struct StoredBlock {
        u32 magic, start_pc, end_pc, num_guest_words, unused[6], num_constant_words;
    };
    static_assert(sizeof(StoredBlock) == 44);
    // Record by record, so that not the whole file is in memory
    u64 at = sizeof(header);
    std::unordered_set<u64> seen;
    while (at + sizeof(StoredRecord) <= size && guesses.size() < MOST_ENTRIES) {
        StoredRecord record;
        if (!file.Seek(static_cast<s64>(at), SEEK_SET) ||
            file.ReadBytes(&record, sizeof(record)) != sizeof(record) || record.size == 0 ||
            at + sizeof(record) + record.size > size) {
            break;
        }
        std::array<u8, 512> start;
        const std::size_t wanted = std::min<std::size_t>(record.size, start.size());
        if (file.ReadBytes(start.data(), wanted) != wanted) {
            break;
        }
        at = (at + sizeof(record) + record.size + 7) & ~u64{7};
        StoredBlock block;
        if (wanted < sizeof(block)) {
            continue;
        }
        std::memcpy(&block, start.data(), sizeof(block));
        const u64 words_at = sizeof(block) + u64{block.num_constant_words} * 4;
        const bool thumb = (record.location >> 32 & 1) != 0;
        if (block.num_guest_words == 0 || words_at + 4 > wanted || thumb ||
            block.start_pc != static_cast<u32>(record.location) || (block.start_pc & 3) != 0 ||
            !seen.insert(record.location).second) {
            continue;
        }
        u32 word;
        std::memcpy(&word, start.data() + words_at, sizeof(word));
        guesses.push_back({record.location, word});
    }
    LOG_INFO(Core_ARM11, "Title {:016X}: {} places in the recompiled code that was stored before",
             program_id, guesses.size());
}

void AheadTranslation::Work() {
    Common::SetCurrentThreadName("TranslateAhead");
    Load();

    bool urgent = true;
    while (true) {
        std::deque<Request> taken;
        bool flush = false;
        {
            std::scoped_lock lock{mutex};
            if (stop_wanted) {
                return;
            }
            taken.swap(requests);
            std::swap(flush, flush_wanted);
        }
        for (Request& request : taken) {
            TakeRequest(std::move(request));
        }
        WriteOut(flush);
        LogActivity();

        const bool known_left = next_entry < entries.size() || next_guess < guesses.size();
        if (known_left != urgent) {
            urgent = known_left;
            SetUrgent(urgent);
        }

        // Nothing can be translated before there is a recompiler to say how: soon, then
        bool did_something = false;
        if (!copies.empty() && translated.IsReady()) {
            const auto start = std::chrono::steady_clock::now();
            auto now = start;
            while (now - start < std::chrono::milliseconds{5}) {
                if (!DoOne()) {
                    break;
                }
                did_something = true;
                now = std::chrono::steady_clock::now();
            }
            worked += now - start;
        }
        if (did_something) {
            continue;
        }
        std::unique_lock lock{mutex};
        const bool waiting_for_recompiler = !copies.empty() && !translated.IsReady();
        wake.wait_for(lock,
                      waiting_for_recompiler ? std::chrono::milliseconds{20}
                                             : std::chrono::milliseconds{2000},
                      [this] { return !requests.empty() || flush_wanted || stop_wanted; });
    }
}

void AheadTranslation::TakeRequest(Request&& request) {
    if (request.copy) {
        std::erase_if(copies, [&](const auto& other) {
            return other->base < request.copy->end && request.copy->base < other->end;
        });
        copies.push_back(std::move(request.copy));
        // Everything that is known is gone through again: it may be in what is new
        next_entry = next_guess = 0;
        return;
    }
    AddEntry(request.location, request.first_word);
    // It is translated by now, and something new: what it leads to may be next
    FollowUp(request.location, 0);
}

bool AheadTranslation::DoOne() {
    Memory memory{copies};
    const auto is_there = [&](const Entry& entry) {
        const u32 pc = static_cast<u32>(entry.location);
        return memory.Read(pc) == entry.first_word;
    };

    while (next_entry < entries.size()) {
        const Entry entry = entries[next_entry++];
        if (is_there(entry)) {
            Translate(entry.location);
            return true;
        }
    }
    while (next_guess < guesses.size()) {
        const Entry entry = guesses[next_guess++];
        if (!is_there(entry)) {
            continue;
        }
        const u32 pc = static_cast<u32>(entry.location);
        const Copy* const copy = memory.Find(pc);
        if (copy->covered[(pc - copy->base) / 4]) {
            continue;
        }
        if (Translate(entry.location)) {
            AddEntry(entry.location, entry.first_word);
        }
        return true;
    }
    while (!leads.empty()) {
        const Lead lead = leads.front();
        leads.pop_front();
        if (led_to_bytes >= MOST_LED_TO_BYTES) {
            leads.clear();
            break;
        }
        const std::size_t bytes_before = made_bytes;
        Translate(lead.location);
        led_to_bytes += made_bytes - bytes_before;
        if (lead.depth < MOST_LEADS) {
            FollowUp(lead.location, lead.depth);
        }
        return true;
    }
    return false;
}

void AheadTranslation::FollowUp(u64 location, u32 depth) {
    Memory memory{copies};
    for (const u32 target : translated.LeadsTo(location)) {
        const u64 to = (location & ~u64{0xFFFFFFFF}) | target;
        if ((target & 3) == 0 && memory.Read(target) && led_to.insert(to).second) {
            leads.push_back({to, depth + 1});
        }
    }
}

bool AheadTranslation::Translate(u64 location) {
    Memory memory{copies};
    const auto result = translated.Translate(location, memory);
    if (!result.made) {
        return false;
    }
    made_bytes += result.host_bytes;
    made++;
    made_instructions += result.instructions;
    for (const auto& [first, end] : result.code) {
        if (Copy* const copy = memory.Find(first); copy && end <= copy->end) {
            std::fill(copy->covered.begin() + (first - copy->base) / 4,
                      copy->covered.begin() + (end - copy->base) / 4, true);
        }
    }
    return true;
}

void AheadTranslation::WriteOut(bool all) {
    const auto now = std::chrono::steady_clock::now();
    if ((written == entries.size() && !rewrite_file) ||
        (!all && now - last_write < WRITE_INTERVAL)) {
        return;
    }
    last_write = now;
    if (entries.empty()) {
        return;
    }
    if (!FileUtil::CreateFullPath(path)) {
        LOG_ERROR(Core_ARM11, "Could not create the directory of {}", path);
        return;
    }
    const auto record_of = [this](const Entry& entry) {
        return Record{entry.location, entry.first_word,
                      CheckOf(program_id, entry.location, entry.first_word)};
    };
    const bool whole = rewrite_file || written == 0;
    std::vector<Record> records;
    std::transform(entries.begin() + static_cast<std::ptrdiff_t>(whole ? 0 : written),
                   entries.end(), std::back_inserter(records), record_of);
    const std::size_t bytes = records.size() * sizeof(Record);
    if (whole) {
        // Beside the one that is there, which stays what it is until this one is complete
        const std::string beside = path + ".new";
        const FileHeader header{FILE_MAGIC, FILE_VERSION, program_id};
        {
            FileUtil::IOFile file{beside, "wb"};
            if (!file.IsOpen() || file.WriteBytes(&header, sizeof(header)) != sizeof(header) ||
                file.WriteBytes(records.data(), bytes) != bytes || !file.Flush()) {
                LOG_ERROR(Core_ARM11, "Could not write {}", beside);
                return;
            }
        }
        if (!FileUtil::Rename(beside, path)) {
            // (Not every kind of storage replaces what is there)
            FileUtil::Delete(path);
            if (!FileUtil::Rename(beside, path)) {
                LOG_ERROR(Core_ARM11, "Could not rename {}", beside);
                return;
            }
        }
    } else {
        FileUtil::IOFile file{path, "ab"};
        if (!file.IsOpen() || file.WriteBytes(records.data(), bytes) != bytes || !file.Flush()) {
            LOG_ERROR(Core_ARM11, "Could not add to {}", path);
            // What the file holds now is unknown
            rewrite_file = true;
            return;
        }
    }
    rewrite_file = false;
    written = entries.size();
}

void AheadTranslation::LogActivity() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_log < LOG_INTERVAL) {
        return;
    }
    last_log = now;
    const auto s = translated.GetStatistics();
    if (made == 0 && s.taken == last_statistics.taken && s.missed == last_statistics.missed &&
        s.faults == last_statistics.faults) {
        return;
    }
    const auto ms = [](u64 ns) { return static_cast<double>(ns) / 1e6; };
    LOG_INFO(Core_ARM11,
             "Code in the last 5 s: {} pieces ({} instructions) translated ahead in {:.0f} ms, {} "
             "taken from there and {} along with them, {} translated when they were needed ({} of them "
             "had changed): {:.1f} ms to translate, {:.1f} ms to put in place; {} accesses faulted, {} look first, {} found their page shut. "
             "Ahead: {} pieces, {} places, {} KB; {} places known",
             made, made_instructions, ms(static_cast<u64>(worked.count())),
             s.taken - last_statistics.taken, s.along - last_statistics.along,
             s.missed - last_statistics.missed,
             s.stale - last_statistics.stale, ms(s.translating_ns - last_statistics.translating_ns),
             ms(s.placing_ns - last_statistics.placing_ns), s.faults - last_statistics.faults,
             s.checked, s.not_open - last_statistics.not_open, s.regions, s.locations,
             s.host_bytes / 1024, entries.size());
    made = made_instructions = 0;
    worked = {};
    last_statistics = s;
}

} // namespace Core

#endif
