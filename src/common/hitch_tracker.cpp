// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include "common/hitch_tracker.h"
#include "common/logging/log.h"
#include "common/trace_markers.h"

#ifdef __linux__
#include <sched.h>
#endif

namespace Common::Hitches {

namespace Detail {
#ifdef CITRA_ENABLE_TRACE_MARKERS
std::atomic_bool enabled{true};
#else
std::atomic_bool enabled{false};
#endif
} // namespace Detail

namespace {

using Clock = std::chrono::steady_clock;
using Nanos = s64;

constexpr std::size_t NUM_CATEGORIES = static_cast<std::size_t>(Category::Count);
using CategoryTimes = std::array<Nanos, NUM_CATEGORIES>;

constexpr Nanos MS = 1'000'000;

/// A VBlank period or an image on screen that lasts longer is a pause, not a hitch
constexpr Nanos DISCONTINUITY = 500 * MS;
/// Frames looked at to tell what is usual, and how many of them are needed to judge
constexpr std::size_t USUAL_WINDOW = 64;
constexpr std::size_t MIN_HISTORY = 16;
/// Windows of the hitch counts and of the cause tally in the summary
constexpr std::chrono::seconds RECENT{10};
constexpr std::chrono::seconds CAUSE_WINDOW{60};
constexpr std::chrono::seconds SUMMARY_INTERVAL{5};
constexpr u32 MAX_LOGGED_PER_SECOND = 8;

// Causes: a category, or one of these
constexpr u8 CAUSE_GAME = NUM_CATEGORIES;         ///< More VBlanks of emulated time than usual
constexpr u8 CAUSE_PIPELINE = NUM_CATEGORIES + 1; ///< Render thread waited for a pipeline
// Handed over on time, and then:
constexpr u8 CAUSE_REPLACED = NUM_CATEGORIES + 2;   ///< The next image replaced it in the queue
constexpr u8 CAUSE_SHOWN_LATE = NUM_CATEGORIES + 3; ///< The display took it later than usual
constexpr u8 CAUSE_UNKNOWN = NUM_CATEGORIES + 4;
constexpr std::size_t NUM_CAUSES = NUM_CATEGORIES + 5;

constexpr bool IsPresentationCause(u8 cause) {
    return cause == CAUSE_REPLACED || cause == CAUSE_SHOWN_LATE;
}

constexpr std::array<const char*, NUM_CATEGORIES> CATEGORY_NAMES{
    "cpu",      "gpu emu",    "textures",      "shaders", "compose",   "readback", "gpu wait",
    "gpu late", "render thr", "present queue", "sleep",   "oversleep", "off cpu",
};

constexpr std::array<const char*, NUM_CAUSES> CAUSE_NAMES{
    "emulated cpu",   "gpu emulation", "texture upload", "shader compile",  "screen compose",
    "gpu readback",   "gpu wait",      "gpu behind",     "render thread",   "present queue",
    "sleep",          "oversleep",     "off cpu",        "game frame time", "pipeline compile",
    "image replaced", "shown late",    "unknown",
};

constexpr std::size_t Index(Category category) {
    return static_cast<std::size_t>(category);
}

/// Categories in which the thread is expected not to run
constexpr bool IsWait(std::size_t category) {
    switch (static_cast<Category>(category)) {
    case Category::GpuReadback:
    case Category::GpuWait:
    case Category::GpuBehind:
    case Category::RenderThread:
    case Category::PresentQueue:
    case Category::Limiter:
    case Category::LimiterOvershoot:
        return true;
    default:
        return false;
    }
}

Nanos ToNanos(Clock::duration duration) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

double ToMs(Nanos time) {
    return static_cast<double>(time) / static_cast<double>(MS);
}

double ToSeconds(Clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

/// CPU time the calling thread has used, or -1 where that is not known.
Nanos ThreadCpuTime() {
#if defined(__linux__) || defined(__APPLE__)
    timespec time{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) == 0) {
        return static_cast<Nanos>(time.tv_sec) * 1'000'000'000 + time.tv_nsec;
    }
#endif
    return -1;
}

template <typename T, std::size_t N>
class Ring {
public:
    void Push(const T& item) {
        items[next] = item;
        next = (next + 1) % N;
        count = std::min(count + 1, N);
    }

    std::size_t Size() const {
        return count;
    }

    /// Item 0 is the newest
    T& operator[](std::size_t age) {
        return items[(next + N - 1 - age) % N];
    }

    const T& operator[](std::size_t age) const {
        return items[(next + N - 1 - age) % N];
    }

    void Clear() {
        next = 0;
        count = 0;
    }

private:
    std::array<T, N> items{};
    std::size_t next{};
    std::size_t count{};
};

template <typename T, std::size_t N>
T Median(std::array<T, N>& values, std::size_t count) {
    const auto middle = values.begin() + count / 2;
    std::nth_element(values.begin(), middle, values.begin() + count);
    return *middle;
}

/// Cores of the emulated system whose load is told apart
constexpr std::size_t GUEST_CORES = 4;
/// Ticks of the emulated system's clock in a millisecond of its time
constexpr double GUEST_TICKS_PER_MS = 268111.856;

using GuestTicks = std::array<u64, GUEST_CORES>;

/// A thread of the game and the emulated time it ran for in a game frame
struct GuestThread {
    u32 id;
    u32 priority;
    u32 core;
    u64 ticks;
};

struct VBlankSample {
    Nanos wall;
    CategoryTimes times;
    /// Share of the period each emulated core was running a thread, in percent
    std::array<u8, GUEST_CORES> guest_load;
    /// Of the emulated time threads ran for: the share that ran after a thread had handed the
    /// GPU a command list and before that thread next waited, by that thread alone and by
    /// all threads, in percent
    u8 after_list_own;
    u8 after_list_all;
};

struct FrameSample {
    u32 id;
    Clock::time_point shown;
    Nanos interval;
    u32 vblanks;
    bool hitch;
    u8 cause;
};

struct HitchSample {
    Clock::time_point when;
    bool on_screen;
    u8 cause;
    Nanos duration;
    Nanos excess;     ///< Over the usual frame
    Nanos cause_time; ///< Share of the excess explained by the cause, if known
    u32 periods;      ///< VBlanks of an emulation hitch, refreshes of one on screen
};

struct PresentSample {
    u32 id;
    u32 game_frame;
    Clock::time_point submitted;
    Clock::time_point queued;
    u64 desired_time;    ///< The display time asked for, or zero
    bool first_of_frame; ///< No earlier image of the same game frame
    bool shown;          ///< Reported by the display, or given up on
};

struct DropSample {
    Clock::time_point when;
    bool first_of_frame;
};

struct ScreenSample {
    Clock::time_point when;
    u32 refreshes;
};

struct SharedState {
    std::mutex mutex;

    Ring<VBlankSample, 128> vblanks;
    /// VBlanks since the last summary: what is counted between two summaries goes by these
    u64 vblanks_since_summary{};
    Ring<FrameSample, 256> frames;
    Ring<HitchSample, 256> hitches;

    // Display
    bool display_timing{};
    Nanos refresh_period{};
    const char* present_mode{""};
    Pacing pacing{};
    HostGpu host_gpu{.load = -1.0f};
    Ring<PresentSample, 64> presents;
    Ring<ScreenSample, 1024> screen;
    Ring<DropSample, 256> drops;
    Ring<Nanos, 32> latencies;
    bool has_shown{};
    u32 shown_game_frame{};
    u64 shown_since{};
    /// The image reported last, to tell when the next one took its place before it was seen
    u64 last_display_time{};
    bool last_first_of_frame{};

    // Log rate limit
    Clock::time_point log_window;
    u32 logged{};
    Clock::time_point last_summary;
};

SharedState shared;
std::atomic<u32> game_frame_seq{0};

/// What the audio output reported lately: a ring written by the audio thread alone, which must
/// not wait for a lock. Each entry packs four counts of frames and comes with its time.
constexpr std::size_t AUDIO_RING_SIZE = 4096;
std::array<std::atomic<u64>, AUDIO_RING_SIZE> audio_reports;
/// The margin and the frames mixed ahead, sixteen bits each, and the speed in millionths
std::array<std::atomic<u64>, AUDIO_RING_SIZE> audio_rates;
std::array<std::atomic<Nanos>, AUDIO_RING_SIZE> audio_times;
std::atomic<u32> audio_next{0};
std::atomic<u32> audio_sample_rate{0};
/// Time other threads waited in AnyThread scopes since the last VBlank
std::atomic<Nanos> foreign_stall{0};
/// Whether the app has the emulation paused, and whether it let it go on since the last VBlank
std::atomic_bool app_paused{false};
std::atomic_bool app_resumed{false};

struct OpenScope {
    Category category;
    Clock::time_point start;
    Nanos children;
};

struct ThreadState {
    bool tracked{};
    bool absorbing{};
    u32 depth{};
    std::array<OpenScope, 12> stack;
    /// Time per category in the current VBlank period
    CategoryTimes times{};

    Clock::time_point vblank_start;
    Nanos vblank_cpu{};
    /// Emulated time each core of the emulated system ran threads for, and had none to run,
    /// in the current VBlank period
    GuestTicks guest_busy{};
    GuestTicks guest_idle{};
    /// The thread that last handed the GPU a command list, whether it has run on since
    /// without waiting, and the emulated time it and all threads ran for in that state in
    /// the current VBlank period
    u32 list_thread{};
    bool list_thread_runs_on{};
    u64 after_list_own{};
    u64 after_list_all{};

    // The game frame in progress
    GuestTicks frame_guest_busy{};
    GuestTicks frame_guest_idle{};
    std::array<GuestThread, 12> frame_guest_threads;
    u32 frame_guest_thread_count{};
    CategoryTimes frame_times{};
    Nanos frame_wall{};
    Nanos frame_longest_vblank{};
    Nanos frame_pipeline_stall{};
    u32 frame_vblanks{};
    u32 frame_cores{};
    u32 last_game_frame{};
    bool skip_frame{};

    /// Usual time per category of a game frame
    CategoryTimes baseline{};
    bool has_baseline{};
};

thread_local ThreadState thread_state;

bool MayLog(Clock::time_point now) {
    if (now - shared.log_window >= std::chrono::seconds{1}) {
        shared.log_window = now;
        shared.logged = 0;
    }
    return shared.logged++ < MAX_LOGGED_PER_SECOND;
}

std::string Breakdown(const CategoryTimes& times, double divisor = 1.0) {
    std::string text;
    for (std::size_t i = 0; i < NUM_CATEGORIES; i++) {
        const double ms = ToMs(times[i]) / divisor;
        if (ms < 0.25) {
            continue;
        }
        fmt::format_to(std::back_inserter(text), "{}{} {:.1f}", text.empty() ? "" : ", ",
                       CATEGORY_NAMES[i], ms);
    }
    return text;
}

struct UsualFrame {
    Nanos interval;
    u32 vblanks;
};

std::optional<UsualFrame> FindUsualFrame() {
    const std::size_t count = std::min(shared.frames.Size(), USUAL_WINDOW);
    if (count < MIN_HISTORY) {
        return std::nullopt;
    }
    std::array<Nanos, USUAL_WINDOW> intervals;
    std::array<u32, USUAL_WINDOW> vblanks;
    for (std::size_t i = 0; i < count; i++) {
        intervals[i] = shared.frames[i].interval;
        vblanks[i] = shared.frames[i].vblanks;
    }
    return UsualFrame{Median(intervals, count), Median(vblanks, count)};
}

Nanos FindUsualVBlank() {
    const std::size_t count = std::min(shared.vblanks.Size(), USUAL_WINDOW);
    if (count == 0) {
        return 0;
    }
    std::array<Nanos, USUAL_WINDOW> walls;
    for (std::size_t i = 0; i < count; i++) {
        walls[i] = shared.vblanks[i].wall;
    }
    return Median(walls, count);
}

const FrameSample* FindFrame(u32 id) {
    const std::size_t count = std::min(shared.frames.Size(), USUAL_WINDOW);
    for (std::size_t i = 0; i < count; i++) {
        if (shared.frames[i].id == id) {
            return &shared.frames[i];
        }
    }
    return nullptr;
}

/// Share of its time each emulated core was running a thread, as "97% 12% 0% 40%"
std::string GuestLoad(const GuestTicks& busy, const GuestTicks& idle) {
    std::string text;
    for (std::size_t i = 0; i < GUEST_CORES; i++) {
        const u64 total = busy[i] + idle[i];
        fmt::format_to(std::back_inserter(text), "{}{}%", i == 0 ? "" : " ",
                       total == 0 ? 0 : busy[i] * 100 / total);
    }
    return text;
}

/// The threads of the game that ran for the longest in the frame in progress, in emulated time
std::string GuestThreads(ThreadState& ts) {
    const auto threads = std::span{ts.frame_guest_threads}.first(ts.frame_guest_thread_count);
    std::ranges::sort(threads, std::greater{}, &GuestThread::ticks);
    std::string text;
    for (const GuestThread& thread : threads.first(std::min<std::size_t>(threads.size(), 4))) {
        fmt::format_to(std::back_inserter(text), "{}{} (priority {}, core {}) {:.1f} ms",
                       text.empty() ? "" : ", ", thread.id, thread.priority, thread.core,
                       static_cast<double>(thread.ticks) / GUEST_TICKS_PER_MS);
    }
    return text;
}

void ResetFrame(ThreadState& ts) {
    ts.frame_guest_busy = {};
    ts.frame_guest_idle = {};
    ts.frame_guest_thread_count = 0;
    ts.frame_times = {};
    ts.frame_wall = 0;
    ts.frame_longest_vblank = 0;
    ts.frame_pipeline_stall = 0;
    ts.frame_vblanks = 0;
    ts.frame_cores = 0;
}

/// Starts tracking on the calling thread, which becomes the emulation thread.
void BeginSession(ThreadState& ts, Clock::time_point now, Nanos cpu) {
    {
        std::scoped_lock lock{shared.mutex};
        shared.vblanks.Clear();
        shared.frames.Clear();
        shared.hitches.Clear();
        shared.presents.Clear();
        shared.screen.Clear();
        shared.drops.Clear();
        shared.latencies.Clear();
        shared.has_shown = false;
        shared.last_summary = now;
    }
    foreign_stall.store(0, std::memory_order_relaxed);

    ts.tracked = true;
    ts.times = {};
    ts.guest_busy = {};
    ts.guest_idle = {};
    ts.vblank_start = now;
    ts.vblank_cpu = cpu;
    ts.last_game_frame = game_frame_seq.load(std::memory_order_relaxed);
    ts.skip_frame = true;
    ts.has_baseline = false;
    ResetFrame(ts);
}

/// Accounts the time of the scopes that are still open, as if they ended and started again.
void FlushOpenScopes(ThreadState& ts, Clock::time_point now) {
    for (u32 i = ts.depth; i-- > 0;) {
        OpenScope& scope = ts.stack[i];
        const Nanos elapsed = ToNanos(now - scope.start);
        ts.times[Index(scope.category)] += elapsed - scope.children;
        if (i > 0) {
            ts.stack[i - 1].children += elapsed;
        }
        scope.start = now;
        scope.children = 0;
    }
}

/// Closes the game frame the emulation thread just handed over. Returns a line to log if it was
/// a hitch.
std::string FinishFrame(ThreadState& ts, u32 id, Clock::time_point now) {
    std::string log_line;
    if (ts.skip_frame) {
        ts.skip_frame = false;
        ResetFrame(ts);
        return log_line;
    }

    FrameSample frame{
        .id = id,
        .shown = now,
        .interval = ts.frame_wall,
        .vblanks = ts.frame_vblanks,
        .hitch = false,
        .cause = CAUSE_UNKNOWN,
    };

    const std::optional<UsualFrame> usual = FindUsualFrame();
    const Nanos usual_vblank = FindUsualVBlank();
    Nanos excess = 0;
    if (usual && ts.has_baseline) {
        excess = frame.interval - usual->interval;
        // Late by a good part of a VBlank period: an image is shown for a period more
        frame.hitch = excess > usual_vblank * 2 / 5;
    }

    if (frame.hitch) {
        // Part of the excess that is the game taking more VBlanks, in emulated time, and part
        // that is the host being late for them
        const Nanos game_part =
            frame.vblanks > usual->vblanks ? (frame.vblanks - usual->vblanks) * usual_vblank : 0;
        const Nanos lateness = frame.interval - frame.vblanks * usual_vblank;

        Nanos cause_time = 0;
        if (game_part > 0 && game_part >= lateness) {
            frame.cause = CAUSE_GAME;
            cause_time = game_part;
        } else {
            for (std::size_t i = 0; i < NUM_CATEGORIES; i++) {
                if (i == Index(Category::Limiter)) {
                    continue;
                }
                const Nanos grown = ts.frame_times[i] - ts.baseline[i];
                if (grown > cause_time) {
                    frame.cause = static_cast<u8>(i);
                    cause_time = grown;
                }
            }
            if (cause_time * 4 < excess) {
                frame.cause = CAUSE_UNKNOWN;
                cause_time = 0;
            }
            // A render thread that waits for a pipeline makes this thread wait for it, or for
            // the GPU, in turn
            const bool waited = frame.cause == CAUSE_UNKNOWN || IsWait(frame.cause);
            if (waited && ts.frame_pipeline_stall * 2 >= excess) {
                frame.cause = CAUSE_PIPELINE;
                cause_time = ts.frame_pipeline_stall;
            }
        }

        shared.hitches.Push(HitchSample{
            .when = now,
            .on_screen = false,
            .cause = frame.cause,
            .duration = frame.interval,
            .excess = excess,
            .cause_time = cause_time,
            .periods = frame.vblanks,
        });

        if (MayLog(now)) {
            log_line = fmt::format(
                "Hitch: frame {} took {:.1f} ms over {} vblanks (usual {:.1f} over {}): {} +{:.1f} "
                "ms | {} | longest vblank {:.1f}, pipeline stall {:.1f}, cores {:#x} | emulated "
                "cores busy {} | threads {}",
                id, ToMs(frame.interval), frame.vblanks, ToMs(usual->interval), usual->vblanks,
                CAUSE_NAMES[frame.cause], ToMs(cause_time), Breakdown(ts.frame_times),
                ToMs(ts.frame_longest_vblank), ToMs(ts.frame_pipeline_stall), ts.frame_cores,
                GuestLoad(ts.frame_guest_busy, ts.frame_guest_idle), GuestThreads(ts));
        }
    } else if (ts.has_baseline) {
        for (std::size_t i = 0; i < NUM_CATEGORIES; i++) {
            ts.baseline[i] += (ts.frame_times[i] - ts.baseline[i]) / 8;
        }
    } else {
        ts.baseline = ts.frame_times;
        ts.has_baseline = true;
    }

    shared.frames.Push(frame);
    CITRA_TRACE_SET_COUNTER("azahar.frame_interval_us", frame.interval / 1000);
    CITRA_TRACE_SET_COUNTER("azahar.frame_vblanks", frame.vblanks);
    CITRA_TRACE_SET_COUNTER("azahar.hitch_ms", frame.hitch ? excess / MS : 0);

    ResetFrame(ts);
    return log_line;
}

/// The image of a game frame left the screen after `duration`, replaced by `next_game_frame`.
/// Returns a line to log if it stayed for longer than usual.
std::string FinishScreenFrame(u32 game_frame, u32 next_game_frame, Nanos duration,
                              const PresentSample& next, Nanos latency, Nanos margin,
                              Clock::time_point now) {
    std::string log_line;
    if (shared.refresh_period <= 0 || duration >= DISCONTINUITY) {
        return log_line;
    }
    const u32 refreshes =
        static_cast<u32>((duration + shared.refresh_period / 2) / shared.refresh_period);

    // What is usual, before this one is added
    const std::size_t count = std::min(shared.screen.Size(), USUAL_WINDOW);
    u32 usual = 0;
    if (count >= MIN_HISTORY) {
        std::array<u32, USUAL_WINDOW> history;
        for (std::size_t i = 0; i < count; i++) {
            history[i] = shared.screen[i].refreshes;
        }
        usual = Median(history, count);
    }
    shared.screen.Push(ScreenSample{.when = now, .refreshes = refreshes});

    if (usual == 0 || refreshes <= usual) {
        return log_line;
    }

    // The next frame came late. Either the emulation delivered it late, or it handed it over
    // on time and the way to the screen took longer: its first image was replaced by a later
    // one before the display took it, or the display took it later than usual.
    const FrameSample* frame = FindFrame(next_game_frame);
    const bool emulation_late = frame && frame->hitch;
    const u8 cause = emulation_late        ? frame->cause
                     : next.first_of_frame ? CAUSE_SHOWN_LATE
                                           : CAUSE_REPLACED;
    shared.hitches.Push(HitchSample{
        .when = now,
        .on_screen = true,
        .cause = cause,
        .duration = duration,
        .excess = (refreshes - usual) * shared.refresh_period,
        .cause_time = 0,
        .periods = refreshes,
    });

    if (MayLog(now)) {
        log_line = fmt::format(
            "Hitch on screen: frame {} stayed {} refreshes ({:.1f} ms, usual {}), next frame {} "
            "{} | emulation interval {:.1f} ms, shown image handed over to queued {:.1f} ms, to "
            "display {:.1f} ms, ready {:.1f} ms early",
            game_frame, refreshes, ToMs(duration), usual, next_game_frame,
            emulation_late ? fmt::format("late from emulation ({})", CAUSE_NAMES[cause])
                           : fmt::format("on time from emulation, {}", CAUSE_NAMES[cause]),
            frame ? ToMs(frame->interval) : 0.0, ToMs(ToNanos(next.queued - next.submitted)),
            ToMs(latency), ToMs(margin));
    }
    return log_line;
}

/// Summary of recent hitches. The mutex of the shared state must be held.
void SummarizeAudio(std::string& text, Clock::time_point now);

std::string Summarize(Clock::time_point now) {
    if (shared.vblanks.Size() == 0) {
        return "hitches: waiting for frames";
    }
    std::string text;
    const auto out = std::back_inserter(text);

    // Counts, worst and causes
    u32 on_screen = 0;
    u32 emulation = 0;
    Nanos worst = 0;
    std::array<u32, NUM_CAUSES> causes{};
    for (std::size_t i = 0; i < shared.hitches.Size(); i++) {
        const HitchSample& hitch = shared.hitches[i];
        const auto age = now - hitch.when;
        if (age > CAUSE_WINDOW) {
            break;
        }
        // A hitch on screen that the emulation caused is counted once, as that emulation hitch
        if (!hitch.on_screen || IsPresentationCause(hitch.cause)) {
            causes[hitch.cause]++;
        }
        if (age <= RECENT) {
            (hitch.on_screen ? on_screen : emulation)++;
            worst = std::max(worst, hitch.excess);
        }
    }
    if (shared.display_timing) {
        fmt::format_to(out, "hitches 10s: {} on screen, {} in emulation", on_screen, emulation);
    } else {
        fmt::format_to(out, "hitches 10s: {} in emulation", emulation);
    }
    if (worst > 0) {
        fmt::format_to(out, ", worst +{:.0f} ms", ToMs(worst));
    }

    // Last hitch
    if (shared.hitches.Size() == 0) {
        text += "\nlast: none";
    } else {
        const HitchSample& hitch = shared.hitches[0];
        fmt::format_to(out, "\nlast {:.1f}s ago: {} {:.1f} ms (+{:.1f}) over {} {}, {}",
                       ToSeconds(now - hitch.when), hitch.on_screen ? "on screen" : "emulation",
                       ToMs(hitch.duration), ToMs(hitch.excess), hitch.periods,
                       hitch.on_screen ? "refreshes" : "vblanks", CAUSE_NAMES[hitch.cause]);
        if (hitch.cause_time > 0 && hitch.cause < NUM_CATEGORIES) {
            fmt::format_to(out, " +{:.1f} ms", ToMs(hitch.cause_time));
        }
    }

    // Causes, most frequent first
    text += "\ncauses 60s: ";
    std::array<std::size_t, NUM_CAUSES> order;
    for (std::size_t i = 0; i < NUM_CAUSES; i++) {
        order[i] = i;
    }
    std::ranges::stable_sort(order,
                             [&](std::size_t a, std::size_t b) { return causes[a] > causes[b]; });
    if (causes[order[0]] == 0) {
        text += "none";
    }
    for (std::size_t i = 0; i < NUM_CAUSES && causes[order[i]] != 0; i++) {
        fmt::format_to(out, "{}{} {}", i == 0 ? "" : ", ", CAUSE_NAMES[order[i]], causes[order[i]]);
    }

    // Screen
    if (!shared.display_timing) {
        text += "\nscreen: no timing from the display";
    } else {
        std::array<u32, 6> held{}; // Index 5 stands for 5 refreshes or more
        u32 frames = 0;
        u32 last_second = 0;
        for (std::size_t i = 0; i < shared.screen.Size(); i++) {
            const ScreenSample& sample = shared.screen[i];
            const auto age = now - sample.when;
            if (age > RECENT) {
                break;
            }
            held[std::min<u32>(sample.refreshes, 5)]++;
            frames++;
            if (age <= std::chrono::seconds{1}) {
                last_second++;
            }
        }
        u32 dropped = 0;
        u32 dropped_first = 0;
        for (std::size_t i = 0; i < shared.drops.Size(); i++) {
            if (now - shared.drops[i].when > RECENT) {
                break;
            }
            dropped++;
            if (shared.drops[i].first_of_frame) {
                dropped_first++;
            }
        }
        fmt::format_to(out, "\nscreen {:.1f} Hz {}: {} fps",
                       shared.refresh_period > 0 ? 1e9 / static_cast<double>(shared.refresh_period)
                                                 : 0.0,
                       shared.present_mode, last_second);
        if (frames != 0) {
            std::array<std::size_t, 6> held_order{0, 1, 2, 3, 4, 5};
            std::ranges::stable_sort(
                held_order, [&](std::size_t a, std::size_t b) { return held[a] > held[b]; });
            text += ", held";
            for (std::size_t i = 0; i < 4 && held[held_order[i]] != 0; i++) {
                fmt::format_to(out, "{} {}{}x {:.0f}%", i == 0 ? "" : ",", held_order[i],
                               held_order[i] == 5 ? "+" : "", 100.0 * held[held_order[i]] / frames);
            }
        }
        fmt::format_to(out, ", {} dropped ({} new)", dropped, dropped_first);
        if (shared.latencies.Size() != 0) {
            std::array<Nanos, 32> latencies;
            for (std::size_t i = 0; i < shared.latencies.Size(); i++) {
                latencies[i] = shared.latencies[i];
            }
            fmt::format_to(out, ", latency {:.0f} ms",
                           ToMs(Median(latencies, shared.latencies.Size())));
        }

        const Pacing& pacing = shared.pacing;
        if (pacing.active) {
            fmt::format_to(out,
                           "\npacing: {}, speed {:+.2f}%, lead {:.0f} ms (aim {:.0f}, {:.0f} of it "
                           "to be queued), ready {:.0f} ms early at the least, {} late, {} missed",
                           pacing.overloaded ? "cannot keep up, images shown as they come"
                           : pacing.locked   ? "in step with the display"
                                             : "by vblank number",
                           (pacing.speed - 1.0) * 100.0, pacing.lead_ms, pacing.target_ms,
                           pacing.work_ms, pacing.ready_ms, pacing.slips, pacing.misses);
        } else {
            text += "\npacing: off";
        }
    }

    // Emulation thread
    const std::size_t vblank_count = std::min(shared.vblanks.Size(), USUAL_WINDOW);
    // What is counted between two summaries is counted over all the VBlanks between them, not
    // only over those the window holds
    const double counted_vblanks =
        static_cast<double>(std::max<u64>(std::exchange(shared.vblanks_since_summary, 0), 1));
    CategoryTimes sums{};
    Nanos wall_sum = 0;
    Nanos wall_max = 0;
    for (std::size_t i = 0; i < vblank_count; i++) {
        const VBlankSample& sample = shared.vblanks[i];
        wall_sum += sample.wall;
        wall_max = std::max(wall_max, sample.wall);
        for (std::size_t j = 0; j < NUM_CATEGORIES; j++) {
            sums[j] += sample.times[j];
        }
    }
    // How unevenly images are handed over: the deviation of the periods from their mean
    const double wall_mean = ToMs(wall_sum) / static_cast<double>(vblank_count);
    double wall_variance = 0.0;
    for (std::size_t i = 0; i < vblank_count; i++) {
        const double deviation = ToMs(shared.vblanks[i].wall) - wall_mean;
        wall_variance += deviation * deviation;
    }
    const double wall_jitter = std::sqrt(wall_variance / static_cast<double>(vblank_count));
    u32 game_frames = 0;
    for (std::size_t i = 0; i < shared.frames.Size(); i++) {
        if (now - shared.frames[i].shown > std::chrono::seconds{1}) {
            break;
        }
        game_frames++;
    }
    fmt::format_to(out, "\nemu {} fps, vblank {:.1f} +-{:.1f} ms (max {:.1f}): {}", game_frames,
                   wall_mean, wall_jitter, ToMs(wall_max),
                   Breakdown(sums, static_cast<double>(vblank_count)));

    // The emulated system: how much of its own time each of its cores was busy for
    std::array<u32, GUEST_CORES> load_sums{};
    std::array<u8, GUEST_CORES> load_max{};
    std::array<u32, GUEST_CORES> load_full{};
    for (std::size_t i = 0; i < vblank_count; i++) {
        for (std::size_t core = 0; core < GUEST_CORES; core++) {
            const u8 load = shared.vblanks[i].guest_load[core];
            load_sums[core] += load;
            load_max[core] = std::max(load_max[core], load);
            load_full[core] += load >= 99 ? 1 : 0;
        }
    }
    text += "\nemulated cores busy:";
    for (std::size_t core = 0; core < GUEST_CORES; core++) {
        fmt::format_to(out, " {}%", load_sums[core] / vblank_count);
    }
    text += ", at most";
    for (std::size_t core = 0; core < GUEST_CORES; core++) {
        fmt::format_to(out, " {}%", load_max[core]);
    }
    fmt::format_to(out, ", all of a vblank in");
    for (std::size_t core = 0; core < GUEST_CORES; core++) {
        fmt::format_to(out, " {}", load_full[core]);
    }
    fmt::format_to(out, " of {}", vblank_count);
    // What of the game's work runs between a command list and its thread's next wait
    u32 after_list_own = 0;
    u32 after_list_all = 0;
    for (std::size_t i = 0; i < vblank_count; i++) {
        after_list_own += shared.vblanks[i].after_list_own;
        after_list_all += shared.vblanks[i].after_list_all;
    }
    fmt::format_to(out,
                   "\nafter a command list: its thread runs on for {}% of the game's work before "
                   "it waits ({}% with the other threads meanwhile)",
                   after_list_own / vblank_count, after_list_all / vblank_count);
    {
        std::array<double, static_cast<std::size_t>(GpuCall::Count)> calls;
        for (std::size_t i = 0; i < calls.size(); i++) {
            calls[i] =
                static_cast<double>(Detail::GpuCalls()[i].exchange(0, std::memory_order_relaxed)) /
                counted_vblanks;
        }
        const auto of = [&](GpuCall call) { return calls[static_cast<std::size_t>(call)]; };
        fmt::format_to(out,
                       "\ngame to gpu, a vblank: handed over {:.1f} command lists, {:.1f} fills, "
                       "{:.1f} transfers and copies, {:.1f} other; has to wait for {:.1f} "
                       "register reads, {:.1f} reads and {:.1f} writes of memory the gpu keeps",
                       of(GpuCall::CommandList), of(GpuCall::Fill), of(GpuCall::Transfer),
                       of(GpuCall::OtherCommand), of(GpuCall::RegisterRead),
                       of(GpuCall::MemoryRead), of(GpuCall::MemoryWrite));
        fmt::format_to(out,
                       "\nof those: {:.1f} reads of what the gpu has newer, {:.1f} writes into a "
                       "cached surface",
                       of(GpuCall::MemoryReadNewer), of(GpuCall::MemoryWriteSurface));
        {
            Detail::CommandListWords& called = Detail::CalledListWords();
            const u64 called_words = called.words.exchange(0, std::memory_order_relaxed);
            const u64 called_same = called.same.exchange(0, std::memory_order_relaxed);
            const u32 called_lists = called.lists.exchange(0, std::memory_order_relaxed);
            const u32 called_identical = called.identical.exchange(0, std::memory_order_relaxed);
            if (called_words != 0 && called_lists != 0) {
                fmt::format_to(out,
                               "\nlists called from them: {:.1f} KiB a vblank; {:.0f}% of their "
                               "words as last time at that address, {} of {} the same throughout",
                               static_cast<double>(called_words) * 4.0 / 1024.0 / counted_vblanks,
                               100.0 * static_cast<double>(called_same) /
                                   static_cast<double>(called_words),
                               called_identical, called_lists);
            }
        }
        {
            std::array<double, static_cast<std::size_t>(DrawSite::Count)> sites;
            for (std::size_t i = 0; i < sites.size(); i++) {
                sites[i] = static_cast<double>(
                    Detail::DrawSites()[i].exchange(0, std::memory_order_relaxed));
            }
            const auto of = [&](DrawSite what) { return sites[static_cast<std::size_t>(what)]; };
            const double draws = of(DrawSite::Draws);
            const double others = draws - of(DrawSite::Same) - of(DrawSite::Fresh);
            if (draws > 0) {
                const auto share = [&](DrawSite what) {
                    return others > 0 ? 100.0 * of(what) / others : 0.0;
                };
                fmt::format_to(
                    out,
                    "\ndraw sites: {:.0f} draws a vblank; {:.0f}% find every register as the "
                    "last time at their place, {:.0f}% come from a new place; the others differ "
                    "in rasterizer {:.0f}%, texturing {:.0f}%, framebuffer {:.0f}%, lighting "
                    "{:.0f}%, geometry {:.0f}%, geometry shader {:.0f}%, vertex shader {:.0f}%",
                    draws / counted_vblanks, 100.0 * of(DrawSite::Same) / draws,
                    100.0 * of(DrawSite::Fresh) / draws, share(DrawSite::Rasterizer),
                    share(DrawSite::Texturing), share(DrawSite::Framebuffer),
                    share(DrawSite::Lighting), share(DrawSite::Geometry),
                    share(DrawSite::GeometryShader), share(DrawSite::VertexShader));
            }
        }
        Detail::CommandListWords& list_words = Detail::ListWords();
        const u64 words = list_words.words.exchange(0, std::memory_order_relaxed);
        const u64 same = list_words.same.exchange(0, std::memory_order_relaxed);
        const u32 lists = list_words.lists.exchange(0, std::memory_order_relaxed);
        const u32 identical = list_words.identical.exchange(0, std::memory_order_relaxed);
        if (words != 0 && lists != 0) {
            fmt::format_to(out,
                           "\ncommand lists: {:.1f} KiB a vblank; {:.0f}% of their words are as "
                           "in the list last sent from the same address, {} of {} lists the "
                           "same throughout",
                           static_cast<double>(words) * 4.0 / 1024.0 / counted_vblanks,
                           100.0 * static_cast<double>(same) / static_cast<double>(words),
                           identical, lists);
        }
    }
    {
        std::array<double, static_cast<std::size_t>(PixelMove::Count)> megapixels;
        std::array<double, static_cast<std::size_t>(PixelMove::Count)> times;
        double all = 0;
        for (std::size_t i = 0; i < megapixels.size(); i++) {
            megapixels[i] = static_cast<double>(
                                Detail::PixelMoves()[i].exchange(0, std::memory_order_relaxed)) /
                            1e6 / counted_vblanks;
            times[i] = static_cast<double>(
                           Detail::PixelMoveCounts()[i].exchange(0, std::memory_order_relaxed)) /
                       counted_vblanks;
            if (static_cast<PixelMove>(i) != PixelMove::Spared) {
                all += megapixels[i];
            }
        }
        if (all > 0) {
            const auto of = [&](PixelMove kind) {
                return fmt::format("{:.2f} in {:.1f}", megapixels[static_cast<std::size_t>(kind)],
                                   times[static_cast<std::size_t>(kind)]);
            };
            fmt::format_to(out,
                           "\npixels moved, millions a vblank: {:.2f} (blits {}, plain copies {}, "
                           "other layout {}, second image {}, clears {}); blits given up "
                           "unused {}",
                           all, of(PixelMove::Blit), of(PixelMove::Copy), of(PixelMove::Relayout),
                           of(PixelMove::SecondImage), of(PixelMove::Clear), of(PixelMove::Spared));
        }
    }
    if (shared.host_gpu.load >= 0.0f) {
        fmt::format_to(out,
                       "\nhost gpu: at work {:.0f}% of the time ({:.1f} ms a vblank), waited "
                       "for {:.0f}%, scale {}x",
                       100.0 * shared.host_gpu.load, shared.host_gpu.load * wall_mean,
                       100.0 * shared.host_gpu.waited, shared.host_gpu.scale);
        if (shared.host_gpu.scale != shared.host_gpu.asked_scale) {
            fmt::format_to(out, " of the {}x asked for", shared.host_gpu.asked_scale);
        }
    }
    SummarizeAudio(text, now);
    return text;
}

/// The audio output over the last ten seconds: how much sound was waiting each time the
/// device asked for some, and how often there was not enough
void SummarizeAudio(std::string& text, Clock::time_point now) {
    const u32 next = audio_next.load(std::memory_order_acquire);
    const u32 rate = audio_sample_rate.load(std::memory_order_relaxed);
    if (next == 0 || rate == 0) {
        return;
    }
    const Nanos since = ToNanos(now.time_since_epoch()) - 10'000'000'000;

    std::array<u16, 2048> levels;
    std::size_t count = 0;
    u32 gaps = 0;
    bool newer_had_gap = false;
    u32 late_stretches = 0;
    bool newer_was_ahead = false;
    u64 ahead = 0;
    u64 missing = 0;
    u64 dropped = 0;
    u64 wanted = 0;
    u32 margin = 0;
    u32 slowest = std::numeric_limits<u32>::max();
    u32 fastest = 0;
    Nanos newest = 0;
    Nanos oldest = 0;
    Nanos longest_gap = 0;
    const u32 available = std::min<u32>(next, AUDIO_RING_SIZE - 1);
    for (u32 back = 1; back <= available && count < levels.size(); back++) {
        const std::size_t index = (next - back) % AUDIO_RING_SIZE;
        const Nanos time = audio_times[index].load(std::memory_order_relaxed);
        if (time < since) {
            break;
        }
        const u64 report = audio_reports[index].load(std::memory_order_relaxed);
        const u32 report_missing = static_cast<u32>((report >> 16) & 0xFFFF);
        levels[count] = static_cast<u16>((report >> 32) & 0xFFFF);
        wanted += report & 0xFFFF;
        missing += report_missing;
        // A gap is a stretch of requests that came up short, however many they were
        if (report_missing == 0 && newer_had_gap) {
            gaps++;
        }
        newer_had_gap = report_missing != 0;
        dropped += (report >> 48) & 0xFFFF;
        const u64 rates = audio_rates[index].load(std::memory_order_relaxed);
        if (count == 0) {
            margin = static_cast<u32>(rates >> 48);
        }
        // Like a gap, a stretch of requests for which frames had to be mixed ahead
        const u32 report_ahead = static_cast<u32>((rates >> 32) & 0xFFFF);
        ahead += report_ahead;
        if (report_ahead == 0 && newer_was_ahead) {
            late_stretches++;
        }
        newer_was_ahead = report_ahead != 0;
        slowest = std::min(slowest, static_cast<u32>(rates));
        fastest = std::max(fastest, static_cast<u32>(rates));
        if (count == 0) {
            newest = time;
        } else {
            longest_gap = std::max(longest_gap, oldest - time);
        }
        oldest = time;
        count++;
    }
    if (count == 0) {
        text += "\naudio: the device is not asking for sound";
        return;
    }
    const auto frames_to_ms = [rate](u64 frames) {
        return 1000.0 * static_cast<double>(frames) / static_cast<double>(rate);
    };
    const auto [lowest, highest] = std::minmax_element(levels.begin(), levels.begin() + count);
    const u16 low = *lowest;
    const u16 high = *highest;
    std::nth_element(levels.begin(), levels.begin() + count / 2, levels.begin() + count);
    if (newer_had_gap) {
        gaps++;
    }
    if (newer_was_ahead) {
        late_stretches++;
    }
    const double period = count > 1 ? ToMs(newest - oldest) / static_cast<double>(count - 1) : 0.0;
    const auto out = std::back_inserter(text);
    fmt::format_to(out, "\naudio: {:.0f}-{:.0f} ms waiting (usually {:.0f})", frames_to_ms(low),
                   frames_to_ms(high), frames_to_ms(levels[count / 2]));
    if (margin != 0) {
        fmt::format_to(out, ", margin {:.0f} ms, speed {:.1f}-{:.1f}%", frames_to_ms(margin),
                       slowest / 10000.0, fastest / 10000.0);
    }
    fmt::format_to(out,
                   ", {} gaps of {:.0f} ms in all, {:.0f} ms thrown away, asked for {:.1f} ms "
                   "every {:.1f} ms (longest {:.0f})",
                   gaps, frames_to_ms(missing), frames_to_ms(dropped),
                   frames_to_ms(wanted) / static_cast<double>(count), period, ToMs(longest_gap));
    if (ahead != 0) {
        fmt::format_to(out, ", {:.0f} ms mixed ahead of a late emulation {} times",
                       frames_to_ms(ahead), late_stretches);
    }
}

void LogLines(std::string_view text) {
    while (!text.empty()) {
        const std::size_t end = text.find('\n');
        LOG_INFO(Common, "Hitch summary: {}", text.substr(0, end));
        if (end == std::string_view::npos) {
            break;
        }
        text.remove_prefix(end + 1);
    }
}

} // Anonymous namespace

void SetOverlayEnabled([[maybe_unused]] bool enabled) {
#ifndef CITRA_ENABLE_TRACE_MARKERS
    Detail::enabled.store(enabled, std::memory_order_relaxed);
#endif
}

void Scope::Begin(Category category, Mode mode) {
    ThreadState& ts = thread_state;
    if (!ts.tracked) {
        if (mode == Mode::AnyThread) {
            foreign_start = Clock::now();
            state = State::Foreign;
        }
        return;
    }
    if (ts.absorbing || ts.depth == ts.stack.size()) {
        return;
    }
    ts.stack[ts.depth++] = OpenScope{category, Clock::now(), 0};
    if (mode == Mode::Absorb) {
        ts.absorbing = true;
        state = State::TrackedAbsorb;
    } else {
        state = State::Tracked;
    }
}

void Scope::End() {
    const Clock::time_point now = Clock::now();
    if (state == State::Foreign) {
        foreign_stall.fetch_add(ToNanos(now - foreign_start), std::memory_order_relaxed);
        return;
    }

    ThreadState& ts = thread_state;
    if (state == State::TrackedAbsorb) {
        ts.absorbing = false;
    }
    if (ts.depth == 0) {
        return;
    }
    const OpenScope& scope = ts.stack[--ts.depth];
    const Nanos elapsed = ToNanos(now - scope.start);
    ts.times[Index(scope.category)] += elapsed - scope.children;
    if (ts.depth != 0) {
        ts.stack[ts.depth - 1].children += elapsed;
    }
}

void MoveTime(Category from, Category to, std::chrono::nanoseconds time) {
    ThreadState& ts = thread_state;
    if (!ts.tracked) {
        return;
    }
    ts.times[Index(from)] -= time.count();
    ts.times[Index(to)] += time.count();
}

void Detail::NoteCommandListDone(u32 thread) {
    ThreadState& ts = thread_state;
    if (ts.tracked) {
        ts.list_thread = thread;
        ts.list_thread_runs_on = true;
    }
}

void Detail::NoteGuestThreadWaits(u32 thread) {
    ThreadState& ts = thread_state;
    if (ts.tracked && ts.list_thread_runs_on && thread == ts.list_thread) {
        ts.list_thread_runs_on = false;
    }
}

void Detail::NoteGuestSlice(u32 core, bool idle, u32 thread, u32 priority, u64 ticks) {
    ThreadState& ts = thread_state;
    if (!ts.tracked || core >= GUEST_CORES) {
        return;
    }
    if (idle) {
        ts.guest_idle[core] += ticks;
        return;
    }
    ts.guest_busy[core] += ticks;
    if (ts.list_thread_runs_on) {
        ts.after_list_all += ticks;
        if (thread == ts.list_thread) {
            ts.after_list_own += ticks;
        }
    }
    for (u32 i = 0; i < ts.frame_guest_thread_count; i++) {
        GuestThread& known = ts.frame_guest_threads[i];
        if (known.id == thread) {
            known.ticks += ticks;
            return;
        }
    }
    if (ts.frame_guest_thread_count < ts.frame_guest_threads.size()) {
        ts.frame_guest_threads[ts.frame_guest_thread_count++] = GuestThread{
            .id = thread,
            .priority = priority,
            .core = core,
            .ticks = ticks,
        };
    }
}

void NoteGameFrame() {
    game_frame_seq.fetch_add(1, std::memory_order_relaxed);
}

u32 CurrentGameFrame() {
    return game_frame_seq.load(std::memory_order_relaxed);
}

void NotePaused(bool paused) {
    app_paused.store(paused, std::memory_order_relaxed);
    if (!paused) {
        app_resumed.store(true, std::memory_order_relaxed);
    }
}

void EndVBlank() {
    ThreadState& ts = thread_state;
    if (!IsEnabled()) {
        ts.tracked = false;
        return;
    }

    const Clock::time_point now = Clock::now();
    const Nanos cpu = ThreadCpuTime();
    if (!ts.tracked) {
        BeginSession(ts, now, cpu);
        return;
    }
    FlushOpenScopes(ts, now);

    const Nanos wall = ToNanos(now - ts.vblank_start);
    const Nanos cpu_used = cpu - ts.vblank_cpu;
    ts.vblank_start = now;
    ts.vblank_cpu = cpu;
    CategoryTimes times = ts.times;
    ts.times = {};
    const GuestTicks guest_busy = std::exchange(ts.guest_busy, {});
    const GuestTicks guest_idle = std::exchange(ts.guest_idle, {});
    const Nanos pipeline_stall = foreign_stall.exchange(0, std::memory_order_relaxed);
    const u32 game_frame = game_frame_seq.load(std::memory_order_relaxed);

    const bool by_the_app = app_resumed.exchange(false, std::memory_order_relaxed) ||
                            app_paused.load(std::memory_order_relaxed);
    if (wall >= DISCONTINUITY) {
        // Paused, the app was in the background, a state was loaded, or something on the host
        // held the thread. Not a hitch to count, but said: a standstill nobody asked for has to
        // be found in the log, and how long the thread ran tells waiting from working.
        if (by_the_app) {
            LOG_INFO(Common, "Emulation stood still for {:.1f} s, paused by the app", wall / 1e9);
        } else if (cpu >= 0 && cpu_used >= wall / 2) {
            // Loading a state, as a rule
            LOG_INFO(Common,
                     "Emulation stood still for {:.1f} s without the app pausing it, its thread "
                     "at work for {:.0f} ms of that",
                     wall / 1e9, cpu_used / 1e6);
        } else {
            LOG_WARNING(Common,
                        "Emulation stood still for {:.1f} s without the app pausing it: its "
                        "thread ran for {:.0f} ms of that and waited for the rest",
                        wall / 1e9, cpu < 0 ? 0.0 : cpu_used / 1e6);
        }
        ResetFrame(ts);
        ts.last_game_frame = game_frame;
        ts.skip_frame = true;
        return;
    }

    // What the scopes do not cover is the emulated system, as far as the thread was running
    Nanos scoped = 0;
    Nanos waits = 0;
    for (std::size_t i = 0; i < NUM_CATEGORIES; i++) {
        scoped += times[i];
        if (IsWait(i)) {
            waits += times[i];
        }
    }
    const Nanos off_cpu = cpu < 0 ? 0 : std::clamp<Nanos>(wall - cpu_used - waits, 0, wall);
    times[Index(Category::OffCpu)] = off_cpu;
    times[Index(Category::Guest)] = std::max<Nanos>(wall - scoped - off_cpu, 0);

    for (std::size_t i = 0; i < NUM_CATEGORIES; i++) {
        ts.frame_times[i] += times[i];
    }
    std::array<u8, GUEST_CORES> guest_load{};
    u64 all_busy = 0;
    for (std::size_t i = 0; i < GUEST_CORES; i++) {
        const u64 total = guest_busy[i] + guest_idle[i];
        guest_load[i] = total == 0 ? 0 : static_cast<u8>(guest_busy[i] * 100 / total);
        all_busy += guest_busy[i];
        ts.frame_guest_busy[i] += guest_busy[i];
        ts.frame_guest_idle[i] += guest_idle[i];
    }
    const u64 after_list_own = std::exchange(ts.after_list_own, 0);
    const u64 after_list_all = std::exchange(ts.after_list_all, 0);
    const auto share_of_busy = [all_busy](u64 ticks) {
        return all_busy == 0 ? u8{0} : static_cast<u8>(std::min<u64>(ticks * 100 / all_busy, 100));
    };
    ts.frame_wall += wall;
    ts.frame_longest_vblank = std::max(ts.frame_longest_vblank, wall);
    ts.frame_pipeline_stall += pipeline_stall;
    ts.frame_vblanks++;
#ifdef __linux__
    if (const int core = sched_getcpu(); core >= 0) {
        ts.frame_cores |= 1U << (core & 31);
    }
#endif

    std::string hitch_line;
    std::string summary;
    {
        std::scoped_lock lock{shared.mutex};
        shared.vblanks_since_summary++;
        shared.vblanks.Push(VBlankSample{
            .wall = wall,
            .times = times,
            .guest_load = guest_load,
            .after_list_own = share_of_busy(after_list_own),
            .after_list_all = share_of_busy(after_list_all),
        });
        if (game_frame != ts.last_game_frame) {
            ts.last_game_frame = game_frame;
            hitch_line = FinishFrame(ts, game_frame, now);
        }
        if (now - shared.last_summary >= SUMMARY_INTERVAL) {
            shared.last_summary = now;
            summary = Summarize(now);
        }
    }
    if (!hitch_line.empty()) {
        LOG_INFO(Common, "{}", hitch_line);
    }
    if (!summary.empty()) {
        LogLines(summary);
    }
}

void SetDisplayTiming(bool available, std::chrono::nanoseconds refresh_period,
                      const char* present_mode) {
    std::scoped_lock lock{shared.mutex};
    shared.display_timing = available;
    shared.refresh_period = refresh_period.count();
    shared.present_mode = present_mode;
    shared.has_shown = false;
}

void SetPacing(const Pacing& pacing) {
    std::scoped_lock lock{shared.mutex};
    shared.pacing = pacing;
}

std::array<std::atomic<u64>, static_cast<std::size_t>(PixelMove::Count)>& Detail::PixelMoves() {
    static std::array<std::atomic<u64>, static_cast<std::size_t>(PixelMove::Count)> pixels{};
    return pixels;
}

std::array<std::atomic<u32>, static_cast<std::size_t>(PixelMove::Count)>&
Detail::PixelMoveCounts() {
    static std::array<std::atomic<u32>, static_cast<std::size_t>(PixelMove::Count)> counts{};
    return counts;
}

std::array<std::atomic<u32>, static_cast<std::size_t>(GpuCall::Count)>& Detail::GpuCalls() {
    static std::array<std::atomic<u32>, static_cast<std::size_t>(GpuCall::Count)> calls{};
    return calls;
}

Detail::CommandListWords& Detail::ListWords() {
    static CommandListWords counts;
    return counts;
}

std::array<std::atomic<u64>, static_cast<std::size_t>(DrawSite::Count)>& Detail::DrawSites() {
    static std::array<std::atomic<u64>, static_cast<std::size_t>(DrawSite::Count)> counts{};
    return counts;
}

Detail::CommandListWords& Detail::CalledListWords() {
    static CommandListWords counts;
    return counts;
}

void SetHostGpu(const HostGpu& gpu) {
    std::scoped_lock lock{shared.mutex};
    shared.host_gpu = gpu;
}

void NoteAudioOutput(const AudioOutput& output) {
    if (!IsEnabled()) {
        return;
    }
    const auto clamp = [](u32 frames) -> u64 { return std::min<u32>(frames, 0xFFFF); };
    const u32 next = audio_next.load(std::memory_order_relaxed);
    const std::size_t index = next % AUDIO_RING_SIZE;
    audio_reports[index].store(clamp(output.wanted) | (clamp(output.missing) << 16) |
                                   (clamp(output.buffered) << 32) | (clamp(output.dropped) << 48),
                               std::memory_order_relaxed);
    audio_rates[index].store((clamp(output.margin) << 48) | (clamp(output.ahead) << 32) |
                                 static_cast<u32>(std::clamp(output.rate, 0.0f, 4.0f) * 1e6f),
                             std::memory_order_relaxed);
    audio_times[index].store(ToNanos(Clock::now().time_since_epoch()), std::memory_order_relaxed);
    audio_sample_rate.store(output.sample_rate, std::memory_order_relaxed);
    audio_next.store(next + 1, std::memory_order_release);
}

void NotePresentQueued(u32 id, u32 game_frame, Clock::time_point submitted, u64 desired_time_ns) {
    if (!IsEnabled() || id == 0) {
        return;
    }
    std::scoped_lock lock{shared.mutex};
    const bool first_of_frame =
        shared.presents.Size() == 0 || shared.presents[0].game_frame != game_frame;
    shared.presents.Push(PresentSample{
        .id = id,
        .game_frame = game_frame,
        .submitted = submitted,
        .queued = Clock::now(),
        .desired_time = desired_time_ns,
        .first_of_frame = first_of_frame,
        .shown = false,
    });
}

void NotePresentShown(u32 present_id, u64 display_time_ns, u64 margin_ns) {
    if (!IsEnabled()) {
        return;
    }
    std::string log_line;
    std::string detail_line;
    std::string dropped_lines;
    {
        std::scoped_lock lock{shared.mutex};
        const Clock::time_point now = Clock::now();

        std::size_t index = 0;
        while (index < shared.presents.Size() && shared.presents[index].id != present_id) {
            index++;
        }
        if (index == shared.presents.Size()) {
            return;
        }
        PresentSample& present = shared.presents[index];
        present.shown = true;
        // Images queued earlier that the display never reported were replaced before it took
        // them
        u32 dropped = 0;
        for (std::size_t i = index + 1; i < shared.presents.Size(); i++) {
            if (shared.presents[i].shown) {
                break;
            }
            shared.presents[i].shown = true;
            shared.drops.Push(DropSample{
                .when = now,
                .first_of_frame = shared.presents[i].first_of_frame,
            });
            dropped++;
            if (TraceMarkers::Experiments() & TraceMarkers::LogPresents) {
                const PresentSample& lost = shared.presents[i];
                const Nanos handed = ToNanos(lost.submitted.time_since_epoch());
                dropped_lines += fmt::format(
                    "{}Present {} frame {} never shown: handed over at {:.1f}, queued +{:.1f}, "
                    "asked for +{:.1f}; the one shown in its place, {}, was asked for {:.1f} "
                    "after it",
                    dropped_lines.empty() ? "" : "\n", lost.id, lost.game_frame,
                    ToMs(handed % (100'000 * MS)), ToMs(ToNanos(lost.queued - lost.submitted)),
                    lost.desired_time != 0 ? ToMs(static_cast<Nanos>(lost.desired_time) - handed)
                                           : 0.0,
                    present.id,
                    lost.desired_time != 0 && present.desired_time != 0
                        ? ToMs(static_cast<Nanos>(present.desired_time) -
                               static_cast<Nanos>(lost.desired_time))
                        : 0.0);
            }
        }

        // Two images reported for the same refresh: the earlier one was never seen
        const bool same_refresh =
            shared.has_shown &&
            std::abs(static_cast<Nanos>(display_time_ns - shared.last_display_time)) <
                shared.refresh_period / 4;
        if (same_refresh) {
            shared.drops.Push(DropSample{
                .when = now,
                .first_of_frame = shared.last_first_of_frame,
            });
            dropped++;
        }
        shared.last_display_time = display_time_ns;
        shared.last_first_of_frame = present.first_of_frame;

        // The steady clock and the display's timestamps share a time base where this is used
        const Nanos latency =
            static_cast<Nanos>(display_time_ns) - ToNanos(present.submitted.time_since_epoch());
        if (latency > 0 && latency < DISCONTINUITY) {
            shared.latencies.Push(latency);
        }
        const Nanos margin = static_cast<Nanos>(margin_ns);

        if ((TraceMarkers::Experiments() & TraceMarkers::LogPresents) &&
            shared.refresh_period > 0) {
            // Where in the display's refresh cycle the emulation thread handed the image over
            const Nanos phase =
                shared.refresh_period -
                ((latency % shared.refresh_period) + shared.refresh_period) % shared.refresh_period;
            const double handed_over =
                ToMs(ToNanos(present.submitted.time_since_epoch()) % (100'000 * MS));
            detail_line = fmt::format(
                "Present {} frame {}{}: handed over at {:.1f}, {:.1f} ms into the refresh, queued "
                "+{:.1f}, shown +{:.1f}, ready {:.1f} early, {} dropped before it, asked for "
                "+{:.1f}",
                present.id, present.game_frame, present.first_of_frame ? " (new)" : "", handed_over,
                ToMs(phase), ToMs(ToNanos(present.queued - present.submitted)), ToMs(latency),
                ToMs(margin), dropped,
                present.desired_time != 0 ? ToMs(static_cast<Nanos>(present.desired_time) -
                                                 ToNanos(present.submitted.time_since_epoch()))
                                          : 0.0);
        }

        if (!shared.has_shown) {
            shared.has_shown = true;
            shared.shown_game_frame = present.game_frame;
            shared.shown_since = display_time_ns;
        } else if (same_refresh &&
                   display_time_ns <= shared.shown_since + shared.refresh_period / 4) {
            // It took the place of the image that started this stretch
            shared.shown_game_frame = present.game_frame;
        } else if (present.game_frame != shared.shown_game_frame &&
                   display_time_ns > shared.shown_since) {
            log_line = FinishScreenFrame(shared.shown_game_frame, present.game_frame,
                                         static_cast<Nanos>(display_time_ns - shared.shown_since),
                                         present, latency, margin, now);
            shared.shown_game_frame = present.game_frame;
            shared.shown_since = display_time_ns;
        }
    }
    if (!dropped_lines.empty()) {
        LOG_INFO(Common, "{}", dropped_lines);
    }
    if (!detail_line.empty()) {
        LOG_INFO(Common, "{}", detail_line);
    }
    if (!log_line.empty()) {
        LOG_INFO(Common, "{}", log_line);
    }
}

std::string OverlayText() {
    if (!IsEnabled()) {
        return {};
    }
    std::scoped_lock lock{shared.mutex};
    return Summarize(Clock::now());
}

} // namespace Common::Hitches
