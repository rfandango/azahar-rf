// Copyright 2023 Citra Emulator Project
// Copyright 2024 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "common/archives.h"
#include "common/microprofile.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/gsp/gsp_command.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/gsp/gsp_interrupt.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/gpu.h"
#include "video_core/gpu_debugger.h"
#include "video_core/gpu_impl.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/right_eye_disabler.h"
#include "video_core/video_core.h"

namespace VideoCore {
struct GPU::Impl {
    Core::Timing& timing;
    Core::System& system;
    Memory::MemorySystem& memory;
    std::shared_ptr<Pica::DebugContext> debug_context;
    Pica::PicaCore pica;
    GraphicsDebugger gpu_debugger;
    std::unique_ptr<RendererBase> renderer;
    RasterizerInterface* rasterizer;
    std::unique_ptr<SwRenderer::SwBlitter> sw_blitter;
    Core::TimingEventType* vblank_event;
    /// The emulated time of the next VBlank, for PaceCompletion; zero while none is known
    u64 next_vblank_ticks{};
    Service::GSP::InterruptHandler signal_interrupt;

    // The thread that decodes commands, see GPU::Submit. What is below the mutex is under it.
    struct QueuedCommand {
        Service::GSP::Command command{};
        u64 submit_ticks{};
        /// The game was told at hand-over that this is done: what the decoder raises for it
        /// is not raised again
        bool early{};
        std::vector<Service::GSP::InterruptId> early_ids;
        /// The delay each of them was told with
        std::vector<u64> early_delays;
        /// The bytes of a command list as they were at hand-over
        std::vector<u8> list_copy;
        /// The lists it jumps to, as they were then (Pica::PicaCore::TakeCalledLists), if
        /// they were looked for
        bool has_called_lists{};
        std::vector<Pica::PicaCore::CalledListCopy> called_lists;
        std::vector<u8> called_bytes;
        /// Not a command but a buffer swap, kept in its place among the commands
        bool is_swap{};
        u32 swap_screen{};
        Service::GSP::FrameBufferInfo swap_info{};
        /// Nor a swap: the composing of a VBlank's image, in its place as well
        bool is_present{};
        u64 present_vblank{};
        /// When it was put into the queue, on the host's clock
        std::chrono::steady_clock::time_point queued_at{};
        /// Its place among the commands of the game's frame
        u32 ordinal{};
        /// Its number among all that was handed over
        u64 seq{};
        /// Copies of the pages the game keeps writing, as they were at hand-over
        std::vector<u32> snapshot_pages;
        /// (As large as the largest set of them it ever held: only what snapshot_pages
        /// names is of this command)
        std::vector<u8> snapshot_bytes;
    };
    /// The buffers of commands that are done, for the commands to come. A command list and
    /// what goes with it is tens to hundreds of kilobytes, and memory of that size the
    /// allocator takes from the system and gives back each time: a hand-over then paid for
    /// the mapping, for a fault at the first touch of every page of the copy, and the decoder
    /// for the unmapping (Xenoblade Chronicles: 3% of the emulation thread's time and 4% of
    /// the decoder's, in the kernel). Taken at hand-over and returned by the decoder, under
    /// their own mutex.
    struct SpareBuffers {
        std::vector<std::vector<u8>> lists;
        std::vector<std::vector<Pica::PicaCore::CalledListCopy>> called_lists;
        std::vector<std::vector<u8>> called_bytes;
        std::vector<std::vector<u32>> snapshot_pages;
        std::vector<std::vector<u8>> snapshot_bytes;
    };
    std::mutex spare_mutex;
    SpareBuffers spare;
    /// More than are ever on their way at once: beyond it a buffer is freed as before
    static constexpr std::size_t MOST_SPARE = 64;
    template <typename T>
    static void TakeSpare(std::vector<std::vector<T>>& from, std::vector<T>& buffer) {
        if (!from.empty()) {
            buffer = std::move(from.back());
            from.pop_back();
        }
    }
    /// `emptied`: handed on with nothing in it, its room kept
    template <typename T>
    static void GiveSpare(std::vector<std::vector<T>>& to, std::vector<T>& buffer, bool emptied) {
        if (buffer.capacity() == 0 || to.size() >= MOST_SPARE) {
            return;
        }
        if (emptied) {
            buffer.clear();
        }
        to.push_back(std::move(buffer));
    }
    struct RaisedInterrupt {
        Service::GSP::InterruptId id;
        u64 delay;
        u64 submit_ticks;
    };
    /// The handler of the service, which may only be called on the emulation thread
    Service::GSP::InterruptHandler deliver_interrupt;
    std::thread decoder;
    std::atomic<std::thread::id> decoder_id{};
    std::atomic<std::thread::id> emulation_id{};
    std::atomic<bool> has_pending{false};
    std::atomic<bool> has_raised{false};
    std::mutex decoder_mutex;
    std::condition_variable decoder_wake;
    std::condition_variable decoder_idle;
    std::deque<QueuedCommand> queued;
    std::vector<RaisedInterrupt> raised;
    u32 pending{};
    u64 working_submit_ticks{};
    bool stop_decoder{};
    /// Images handed to the decoder to compose (emulation thread) and composed by it
    u64 presents_queued{};
    std::atomic<u64> presents_done{};

    /// The number of the last item handed over (emulation thread) and of the last one done
    u64 queued_seq{};
    std::atomic<u64> done_seq{};

    /// What the lists handed over leave in the registers a list's jumps go by (emulation
    /// thread), and what was taken along of lists jumped to since the last report
    Pica::PicaCore::ListScanState list_scan{};
    /// Whether list_scan's registers of where draws read their vertices are as the lists
    /// handed over leave them: looked up with nothing left to decode, and followed in every
    /// list since
    bool list_scan_draws_known{};
    u64 called_taken{};
    u64 called_taken_bytes{};
    /// Hand-overs where not all lists jumped to could be taken, and the time the taking took
    u64 called_not_all{};
    u64 called_take_ns{};
    /// The most lists and bytes a hand-over took along lately: room is made for as many at
    /// once, and the buffers of a command do not grow a dozen times
    std::size_t called_lists_room{};
    std::size_t called_bytes_room{};

    void Queue(QueuedCommand&& item) {
        item.seq = ++queued_seq;
        {
            std::scoped_lock lock{decoder_mutex};
            queued.push_back(std::move(item));
            ++pending;
            has_pending.store(true);
            has_queued.store(true);
        }
        decoder_wake.notify_one();
    }
    /// Design B: the game is told at hand-over that a command is done
    bool early_completion{};
    /// What each kind of command raised the last time it was decoded, with its delays: what
    /// the game is told at the next hand-over of that kind (under the mutex)
    std::array<std::vector<RaisedInterrupt>, 32> early_known;
    /// The same for the command at each place of the game's frame, which is the better guess
    /// where a frame is the same commands as the one before: a list is told the delay of
    /// the list that stood where it stands (under the mutex)
    struct KnownAt {
        u32 kind{~u32{0}};
        std::vector<RaisedInterrupt> raised;
    };
    std::array<KnownAt, 128> early_at;
    /// Emulation thread: commands handed over since the game last swapped the top screen
    u32 frame_ordinal{};
    // Decoder thread: the command it is on
    bool working_early{};
    std::vector<RaisedInterrupt> working_raised;
    /// Set while commands are queued: what the decoding thread looks at before it sleeps
    std::atomic<bool> has_queued{false};

    // For the line in the log: what the emulation thread waited for, and the decoder's work
    struct Waited {
        std::atomic<u32> times{};
        std::atomic<u64> nanoseconds{};
    };
    std::array<Waited, static_cast<std::size_t>(GPU::WaitReason::Count)> waited;
    /// How often each was come by, with something to wait for or not
    std::array<std::atomic<u32>, static_cast<std::size_t>(GPU::WaitReason::Count)> touched{};
    std::atomic<u64> decoder_nanoseconds{};
    /// For the line in the log: the delays told at hand-over, by how much they differed from
    /// what the command then raised (too long, too short), and commands told nothing early
    /// From a VBlank to the start of the composing of its image: summed, the longest, images
    std::atomic<u64> compose_lag_ns{};
    std::atomic<u64> compose_lag_max_ns{};
    std::atomic<u32> compose_lags{};
    std::atomic<u64> told_ns{};
    std::atomic<u64> told_long_ns{};
    std::atomic<u64> told_short_ns{};
    std::atomic<u32> told_late{};
    std::atomic<u32> decoder_commands{};
    u32 report_vblanks{};

    explicit Impl(Core::System& system, Frontend::EmuWindow& emu_window,
                  Frontend::EmuWindow* secondary_window)
        : timing{system.CoreTiming()}, system{system}, memory{system.Memory()},
          debug_context{Pica::g_debug_context}, pica{memory, debug_context},
          renderer{VideoCore::CreateRenderer(emu_window, secondary_window, pica, system)},
          rasterizer{renderer->Rasterizer()},
          sw_blitter{std::make_unique<SwRenderer::SwBlitter>(memory, rasterizer)} {}
    ~Impl() = default;
};
} // namespace VideoCore