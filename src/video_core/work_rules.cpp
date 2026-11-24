// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/work_rules.h"

namespace VideoCore {

namespace {

/// Frames between two lines in the log on what the rules left out
constexpr u32 REPORT_FRAMES = 300;
/// One frame in this many has the calls of its draws timed
constexpr u32 TIMED_EVERY = 64;

} // Anonymous namespace

WorkRules::WorkRules() = default;
WorkRules::~WorkRules() = default;

void WorkRules::AddRule(std::unique_ptr<WorkRule> rule) {
    auto entry = std::make_unique<Entry>();
    entry->rule = std::move(rule);
    entries.push_back(std::move(entry));
}

template <typename Ask>
Vote WorkRules::AskAll(u32 Counts::*kind, Ask&& ask) {
    // Every rule is asked, also once one has voted: each follows what the game does by
    // what it is told here
    Vote result = Vote::Carry;
    bool done = false;
    last_why = "";
    for (auto& entry : entries) {
        const Vote vote = ask(*entry);
        if (vote == Vote::Carry) {
            continue;
        }
        done = done || vote == Vote::Redundant;
        entry->frame.*kind += 1;
        if (kind == &Counts::draws) {
            entry->draws_taken.fetch_add(1, std::memory_order_relaxed);
        }
        if (result == Vote::Carry) {
            result = vote;
            last_why = entry->rule->Why();
        }
    }
    if (done) {
        for (auto& entry : entries) {
            entry->rule->OnDone();
        }
    }
    seen.*kind += 1;
    if (result != Vote::Carry) {
        left_out.*kind += 1;
    }
    return result;
}

bool WorkRules::LeavesOutDraw(DrawFacts& facts) {
    using Clock = std::chrono::steady_clock;
    const Vote vote = AskAll(&Counts::draws, [&](Entry& entry) {
        if (!timed) [[likely]] {
            return entry.rule->OnDraw(facts);
        }
        const Clock::time_point before = Clock::now();
        const Vote answer = entry.rule->OnDraw(facts);
        const u64 took = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - before).count());
        entry.timed_ns += took > clock_ns ? took - clock_ns : 0;
        entry.timed_calls++;
        return answer;
    });
    return vote != Vote::Carry && !facts.IsImmediate();
}

bool WorkRules::LeavesOutFill(const FillFacts& facts) {
    return AskAll(&Counts::fills, [&](Entry& entry) { return entry.rule->OnFill(facts); }) !=
           Vote::Carry;
}

bool WorkRules::LeavesOutTransfer(const TransferFacts& facts) {
    return AskAll(&Counts::transfers,
                  [&](Entry& entry) { return entry.rule->OnTransfer(facts); }) != Vote::Carry;
}

void WorkRules::EndFrame(const FrameFacts& facts) {
    for (auto& entry : entries) {
        entry->rule->OnFrameEnd(facts);
        entry->reported.draws += entry->frame.draws;
        entry->reported.fills += entry->frame.fills;
        entry->reported.transfers += entry->frame.transfers;
        entry->frame = Counts{};
    }
    frames++;
    timed = frames % TIMED_EVERY == 0;
    if (timed) {
        // What two readings of the clock one after the other are apart: the least of a few
        using Clock = std::chrono::steady_clock;
        u64 least = ~u64{0};
        for (int i = 0; i < 8; i++) {
            const Clock::time_point before = Clock::now();
            const u64 apart = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - before)
                    .count());
            least = std::min(least, apart);
        }
        clock_ns = least;
    }

#ifdef CITRA_ENABLE_TRACE_MARKERS
    // What each rule left out and what asking it cost, the same way for all of them: this
    // is what coverage is checked by, game by game, after a rule was changed or added
    if (frames - frames_reported < REPORT_FRAMES) {
        return;
    }
    const double per = static_cast<double>(frames - frames_reported);
    std::string text = fmt::format(
        "Work rules, a frame: {:.0f} draws, {:.1f} fills, {:.1f} transfers and copies, of them "
        "left out {:.0f}, {:.1f}, {:.1f}",
        seen.draws / per, seen.fills / per, seen.transfers / per, left_out.draws / per,
        left_out.fills / per, left_out.transfers / per);
    for (auto& entry : entries) {
        text += fmt::format("; '{}' {:.0f} draws {:.1f} fills {:.1f} transfers",
                            entry->rule->Name(), entry->reported.draws / per,
                            entry->reported.fills / per, entry->reported.transfers / per);
        if (entry->timed_calls != 0) {
            text += fmt::format(" ({} ns a draw)", entry->timed_ns / entry->timed_calls);
        }
        entry->reported = Counts{};
        entry->timed_ns = 0;
        entry->timed_calls = 0;
    }
    LOG_INFO(HW_GPU, "{}", text);
    frames_reported = frames;
    seen = Counts{};
    left_out = Counts{};
#endif
}

std::string WorkRules::TakeLeftOutText(u32 vblanks) {
    std::string text;
    for (auto& entry : entries) {
        const u64 draws = entry->draws_taken.exchange(0, std::memory_order_relaxed);
        if (draws != 0 && vblanks != 0) {
            text += fmt::format("; {:.1f} draws a vblank left out by '{}'",
                                static_cast<double>(draws) / vblanks, entry->rule->Name());
        }
    }
    return text;
}

} // namespace VideoCore
