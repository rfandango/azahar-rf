// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/archives.h"
#include "common/hacks/hack_manager.h"
#include "common/hitch_tracker.h"
#include "common/microprofile.h"
#include "common/thread.h"
#include "common/trace_markers.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "core/loader/loader.h"
#include "core/movie.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/frame_liveness.h"
#include "video_core/frame_state_dump.h"
#include "video_core/gpu.h"
#include "video_core/gpu_debugger.h"
#include "video_core/gpu_impl.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/right_eye_disabler.h"
#include "video_core/video_core.h"
#include "video_core/work_rules.h"
#include "video_core/work_rules/repeated_eye_rule.h"
#include "video_core/work_rules/twin_pass_rule.h"

namespace VideoCore {

namespace {

/// For the hitch summary: how much of a command list is what the list last sent from the same
/// address had in the same place. A game builds its lists anew every frame, in the two or
/// three buffers it takes turns with; where most of a list is what it was, most of decoding
/// it is decoding the same again. Emulation thread only.
void MeasureListRepeat(const u8* data, u32 size, PAddr address) {
    static std::unordered_map<PAddr, std::vector<u32>> last_lists;
    if (!data || size < 4) {
        return;
    }
    // A game that sends its lists from ever new places is not to fill the memory with them
    if (last_lists.size() > 64) {
        last_lists.clear();
    }
    std::vector<u32> list(size / 4);
    std::memcpy(list.data(), data, list.size() * 4);
    std::vector<u32>& previous = last_lists[address];
    const std::size_t common = std::min(list.size(), previous.size());
    u32 same = 0;
    for (std::size_t i = 0; i < common; i++) {
        same += list[i] == previous[i];
    }
    Common::Hitches::NoteCommandListWords(static_cast<u32>(list.size()), same,
                                          list.size() == previous.size() && same == list.size());
    previous = std::move(list);
}

} // Anonymous namespace

constexpr VAddr VADDR_LCD = 0x1ED02000;
constexpr VAddr VADDR_GPU = 0x1EF00000;

class DelayGenerator {
private:
    DelayGenerator() = default;

    // Average transfer speed based on measurements taken from real
    // hardware. 4 different modes have been taken into consideration:
    // RAM -> RAM, RAM -> VRAM, VRAM -> RAM and VRAM -> VRAM.
    // Furthermore, measurements are split into DMA transfers and tex
    // copies. For simplicity, we will assume fills are as fast as
    // texture copies.

    static constexpr double mibps_to_ns_per_byte(double mib_per_sec) {
        return 1'000'000'000.0 / (mib_per_sec * 1024.0 * 1024.0);
    }

    static constexpr std::array<std::array<double, 4>, 2> speed_mibps = {
        {{
             190.0, // DMA RAMTORAM
             310.0, // DMA RAMTOVRAM
             380.0, // DMA VRAMTORAM
             380.0, // DMA VRAMTOVRAM
         },
         {
             450.0,  // TEX RAMTORAM
             3100.0, // TEX RAMTOVRAM
             5400.0, // TEX VRAMTORAM
             5400.0, // TEX VRAMTOVRAM
         }}};

public:
    enum class CopyMode {
        RAMTORAM,
        RAMTOVRAM,
        VRAMTORAM,
        VRAMTOVRAM,
    };

    static CopyMode GetCopyMode(bool input_vram, bool output_vram) {
        if (!input_vram && !output_vram) {
            return CopyMode::RAMTORAM;
        } else if (!input_vram && output_vram) {
            return CopyMode::RAMTOVRAM;
        } else if (input_vram && !output_vram) {
            return CopyMode::VRAMTORAM;
        } else {
            return CopyMode::VRAMTOVRAM;
        }
    }

    static u64 CalculateDelayNanoseconds(CopyMode mode, bool is_textre, size_t size) {
        double base_ns_per_byte =
            mibps_to_ns_per_byte(speed_mibps[is_textre][static_cast<u32>(mode)]);

        return static_cast<u64>(size * base_ns_per_byte);
    }
};

MICROPROFILE_DEFINE(GPU_DisplayTransfer, "GPU", "DisplayTransfer", MP_RGB(100, 100, 255));
MICROPROFILE_DEFINE(GPU_CmdlistProcessing, "GPU", "Cmdlist Processing", MP_RGB(100, 255, 100));

/// Whether the commands a game sends are decoded on a thread of their own while the game goes
/// on. The way it is done where it has run, Android on arm64; elsewhere not until it has been
/// tried there.
static bool WantsDecoderThread() {
#if defined(__ANDROID__) && defined(__aarch64__)
    return (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyGpuDecoding) == 0;
#else
    return false;
#endif
}

GPU::GPU(Core::System& system, Frontend::EmuWindow& emu_window,
         Frontend::EmuWindow* secondary_window)
    : right_eye_disabler{std::make_unique<RightEyeDisabler>(*this)},
      work_rules{std::make_unique<WorkRules>()},
      frame_liveness{work_rules->Add(std::make_unique<FrameLiveness>(*this))},
      impl{std::make_unique<Impl>(system, emu_window, secondary_window)} {
    // The rules that find work that need not be done. This is the one place they are named:
    // a new rule is a class behind VideoCore::WorkRule and a line here. Their order does not
    // matter, each is told of everything and votes.
    work_rules->Add(std::make_unique<RepeatedEyeRule>());
    work_rules->Add(std::make_unique<TwinPassRule>());

    impl->vblank_event = impl->timing.RegisterEvent(
        "GPU::VBlankCallback",
        [this](uintptr_t user_data, s64 cycles_late) { VBlankCallback(user_data, cycles_late); });
    impl->timing.ScheduleEvent(FRAME_TICKS, impl->vblank_event);

    // Bind the rasterizer to the PICA GPU
    impl->pica.BindRasterizer(impl->rasterizer);
    impl->pica.SetWorkRules(work_rules.get());
    CITRA_FRAME_DUMP_GAME_START();

    if (WantsDecoderThread()) {
        LOG_INFO(HW_GPU, "Commands are decoded on a thread of their own");
        impl->early_completion =
            (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LateGpuCompletion) == 0 &&
            impl->renderer->ComposesApart();
        if (impl->early_completion) {
            LOG_INFO(HW_GPU, "The game is told a command is done when it hands it over");
            impl->memory.TrackGpuReferences(true);
        }
        impl->decoder = std::thread([this] { DecoderLoop(); });
    }
}

GPU::~GPU() {
    StopDecoder();
}

namespace {

/// How long either side of a hand-off looks for the other before it goes to sleep. Waking a
/// sleeping thread takes the kernel tens of microseconds and more, and a frame has dozens of
/// hand-offs.
constexpr std::chrono::microseconds HANDOFF_SPIN{150};

/// The kinds of commands that raise the same interrupts: what one raised the last time is
/// what the game is told at the next hand-over of its kind. A fill raises one for each of
/// its two halves that is used.
u32 EarlyKind(const Service::GSP::Command& command) {
    u32 shape = 0;
    if (command.id == Service::GSP::CommandId::MemoryFill) {
        shape = (command.memory_fill.start1 != 0 ? 1u : 0u) |
                (command.memory_fill.start2 != 0 ? 2u : 0u);
    }
    return (static_cast<u32>(command.id.Value()) & 7) * 4 + shape;
}

inline void SpinPause() {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

} // Anonymous namespace

void GPU::DecoderLoop() {
    Common::SetCurrentThreadName("GpuDecoder");
    impl->decoder_id.store(std::this_thread::get_id());
    std::unique_lock lock{impl->decoder_mutex};
    while (true) {
        if (impl->queued.empty() && !impl->stop_decoder && !impl->early_completion) {
            // The game waits for this command to be done, and the next one mostly comes
            // within the time it takes to go to sleep and be woken: looked for a little while
            // first. (Where the game is told "done" at hand-over nothing waits for the
            // decoder, and looking was a tenth of this thread's time for nothing.)
            lock.unlock();
            const auto until = std::chrono::steady_clock::now() + HANDOFF_SPIN;
            while (!impl->has_queued.load(std::memory_order_relaxed) &&
                   std::chrono::steady_clock::now() < until) {
                SpinPause();
            }
            lock.lock();
        }
        impl->decoder_wake.wait(lock,
                                [this] { return impl->stop_decoder || !impl->queued.empty(); });
        if (impl->queued.empty()) {
            // Asked to stop, with nothing left
            return;
        }
        Impl::QueuedCommand next = std::move(impl->queued.front());
        impl->queued.pop_front();
        if (impl->queued.empty()) {
            impl->has_queued.store(false);
        }
        impl->working_submit_ticks = next.submit_ticks;
        impl->working_early = next.early;
        impl->working_raised.clear();
        lock.unlock();
        const auto start = std::chrono::steady_clock::now();
        const bool is_command = !next.is_swap && !next.is_present;
        if (next.is_present) {
            const u64 lag = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(start - next.queued_at)
                    .count());
            impl->compose_lag_ns.fetch_add(lag, std::memory_order_relaxed);
            impl->compose_lags.fetch_add(1, std::memory_order_relaxed);
            if (lag > impl->compose_lag_max_ns.load(std::memory_order_relaxed)) {
                impl->compose_lag_max_ns.store(lag, std::memory_order_relaxed);
            }
            // The image of a VBlank, of what was handed over before it
            frame_liveness->NoteDisplay();
            CITRA_TRACE_SCOPE("VBlank::SwapBuffers");
            CITRA_HITCH_SCOPE(Compose);
            impl->memory.SetGpuReader(true);
            impl->renderer->ComposeFrame(next.present_vblank);
        } else if (next.is_swap) {
            SetBufferSwap(next.swap_screen, next.swap_info);
        } else {
            if (!next.list_copy.empty()) {
                impl->pica.UseCommandListCopy(next.list_copy.data());
                impl->pica.UseCalledLists(next.has_called_lists ? &next.called_lists : nullptr,
                                          next.called_bytes.data());
            }
            impl->memory.SetGpuReader(next.command.id != Service::GSP::CommandId::SubmitCmdList);
            if (!next.snapshot_pages.empty()) {
                impl->memory.UseGpuSnapshot(next.snapshot_pages, next.snapshot_bytes.data());
            }
            if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::SlowTransfers) != 0 &&
                (next.command.id == Service::GSP::CommandId::DisplayTransfer ||
                 next.command.id == Service::GSP::CommandId::TextureCopy ||
                 next.command.id == Service::GSP::CommandId::MemoryFill)) [[unlikely]] {
                std::this_thread::sleep_for(std::chrono::milliseconds{3});
            }
            Execute(next.command);
            impl->pica.UseCommandListCopy(nullptr);
            impl->pica.UseCalledLists(nullptr, nullptr);
            impl->memory.DropGpuSnapshot();
            if (impl->early_completion &&
                next.command.id == Service::GSP::CommandId::SubmitCmdList) {
                // The host GPU can start on this list's drawing now, while the rest of the
                // frame is decoded: kept until the image is composed, the whole frame's
                // drawing would only begin then, and the image be ready that much later
                impl->rasterizer->SubmitPendingWork();
            }
        }
        if (is_command) {
            impl->decoder_nanoseconds.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count(),
                std::memory_order_relaxed);
            impl->decoder_commands.fetch_add(1, std::memory_order_relaxed);
        }
        impl->done_seq.store(next.seq);
        if (next.list_copy.capacity() != 0 || next.called_lists.capacity() != 0 ||
            next.called_bytes.capacity() != 0 || next.snapshot_pages.capacity() != 0 ||
            next.snapshot_bytes.capacity() != 0) {
            // Its buffers, for the commands to come (see Impl::SpareBuffers)
            std::scoped_lock spare_lock{impl->spare_mutex};
            Impl::GiveSpare(impl->spare.lists, next.list_copy, true);
            Impl::GiveSpare(impl->spare.called_lists, next.called_lists, true);
            Impl::GiveSpare(impl->spare.called_bytes, next.called_bytes, true);
            Impl::GiveSpare(impl->spare.snapshot_pages, next.snapshot_pages, true);
            Impl::GiveSpare(impl->spare.snapshot_bytes, next.snapshot_bytes, false);
        }
        if (is_command && impl->early_completion) {
            if (!next.early && !impl->working_raised.empty()) {
                impl->told_late.fetch_add(1, std::memory_order_relaxed);
            }
            for (const Impl::RaisedInterrupt& interrupt : impl->working_raised) {
                for (std::size_t i = 0; i < next.early_ids.size(); i++) {
                    if (next.early_ids[i] != interrupt.id) {
                        continue;
                    }
                    const u64 told = next.early_delays[i];
                    impl->told_ns.fetch_add(told, std::memory_order_relaxed);
                    if (told > interrupt.delay) {
                        impl->told_long_ns.fetch_add(told - interrupt.delay,
                                                     std::memory_order_relaxed);
                    } else {
                        impl->told_short_ns.fetch_add(interrupt.delay - told,
                                                      std::memory_order_relaxed);
                    }
                    break;
                }
            }
        }
        lock.lock();
        if (is_command) {
            if (next.early) {
                // What it raised that the game was not told of at hand-over still has to
                // reach the game
                for (const Impl::RaisedInterrupt& interrupt : impl->working_raised) {
                    if (std::ranges::find(next.early_ids, interrupt.id) == next.early_ids.end()) {
                        impl->raised.push_back(interrupt);
                        impl->has_raised.store(true);
                    }
                }
            }
            const u32 kind = EarlyKind(next.command);
            impl->early_known[kind] = impl->working_raised;
            if (next.ordinal < impl->early_at.size()) {
                impl->early_at[next.ordinal].kind = kind;
                impl->early_at[next.ordinal].raised = impl->working_raised;
            }
        }
        if (next.is_present) {
            impl->presents_done.fetch_add(1);
            impl->decoder_idle.notify_all();
        }
        if (--impl->pending == 0) {
            impl->has_pending.store(false);
            impl->decoder_idle.notify_all();
        }
    }
}

void GPU::StopDecoder() {
    if (!impl->decoder.joinable()) {
        return;
    }
    {
        std::scoped_lock lock{impl->decoder_mutex};
        impl->stop_decoder = true;
    }
    impl->decoder_wake.notify_all();
    impl->decoder.join();
}

Result GPU::Submit(const Service::GSP::Command& command) {
    using Service::GSP::CommandId;
    // Only what the emulated GPU does by itself goes to the thread. A copy between two places
    // of a process's memory (DMA) works with the process as it is now.
    const bool uses_decoder = UsesDecoder();
    const bool to_decoder =
        uses_decoder &&
        (command.id == CommandId::SubmitCmdList || command.id == CommandId::MemoryFill ||
         command.id == CommandId::DisplayTransfer || command.id == CommandId::TextureCopy);
    if (!to_decoder) {
        // Where the decoder works behind the game's back, what is left touches nothing of the
        // emulated GPU itself: the memory system asks for the decoder page by page
        // A copy between two places of the game's memory is carried out here and now. What
        // was handed over before it has to be done first: a fill that is still waiting would
        // otherwise clear what this puts there (a game clears video memory, then copies its
        // vertex data into it).
        const bool ordered_copy =
            command.id == CommandId::RequestDma &&
            (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::UnheldCommandRanges) == 0;
        if (!impl->early_completion || !uses_decoder || ordered_copy) {
            WaitIdle(WaitReason::Command);
        }
        return Execute(command);
    }
    impl->emulation_id.store(std::this_thread::get_id());

    Impl::QueuedCommand item{};
    item.command = command;
    item.submit_ticks = static_cast<u64>(impl->timing.GetGlobalTicks());
    item.seq = impl->queued_seq + 1;
    bool may_tell_early = impl->early_completion;
    PAddr pace_target = 0;
    if (impl->early_completion) {
        // The game is about to be told this is done. What the command reads must not change
        // before the decoder has read it, and what it writes must not be looked at before it
        // is written: its own bytes are taken along, and the pages the decoder has read from
        // or written to before cannot be touched from here on without waiting for it.
        // What a transfer or copy reads is named by the command: held from here on. (It was
        // held only once the decoder had read from those pages, which for data a game puts
        // into video memory by a copy is one time too late: the game refilled its buffer the
        // moment it was told the copy was done, and the copy carried what came after.)
        // And so is what a fill, a transfer or a copy writes: closed to the game from here on,
        // so that what the game puts there afterwards comes after it.
        {
            // In buffers that commands before it used (see Impl::SpareBuffers)
            std::scoped_lock spare_lock{impl->spare_mutex};
            Impl::TakeSpare(impl->spare.snapshot_pages, item.snapshot_pages);
            Impl::TakeSpare(impl->spare.snapshot_bytes, item.snapshot_bytes);
            if (command.id == CommandId::SubmitCmdList) {
                Impl::TakeSpare(impl->spare.lists, item.list_copy);
                Impl::TakeSpare(impl->spare.called_lists, item.called_lists);
                Impl::TakeSpare(impl->spare.called_bytes, item.called_bytes);
            }
        }
        PAddr source_addr = 0;
        u32 source_size = 0;
        std::array<std::pair<PAddr, u32>, 2> written{};
        const bool is_transfer =
            command.id == CommandId::DisplayTransfer || command.id == CommandId::TextureCopy;
        if (command.id != CommandId::SubmitCmdList &&
            (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::UnheldCommandRanges) == 0)
            [[likely]] {
            std::optional<std::pair<PAddr, u32>> source{std::in_place, PAddr{0}, u32{0}};
            if (is_transfer) {
                source = SourceOf(command);
            }
            const auto targets = TargetsOf(command);
            if (source && targets) {
                source_addr = source->first;
                source_size = source->second;
                written = *targets;
            } else {
                // Not known before the registers are as the lists before it leave them: the
                // game goes on when it is done
                may_tell_early = false;
            }
        }
        impl->memory.ArmGpuReferences(command.id == CommandId::SubmitCmdList, is_transfer,
                                      impl->queued_seq, impl->done_seq.load(), source_addr,
                                      source_size, written, item.snapshot_pages,
                                      item.snapshot_bytes);
        if (command.id == CommandId::SubmitCmdList) {
            const auto& list = command.submit_gpu_cmdlist;
            const auto address = VirtualToPhysicalAddress(list.address, list.size);
            const u8* const bytes = address ? impl->memory.GetPhysicalPointer(*address) : nullptr;
            if (bytes && list.size != 0 &&
                impl->memory.GetPhysicalPointer(*address + list.size - 1) ==
                    bytes + list.size - 1) {
                // With what follows it in memory, as far as its last command may read
                const u32 list_size = list.size;
                const u32 readable =
                    std::max(list_size, std::min(impl->memory.PhysicalBytesFrom(*address),
                                                 list_size + Pica::PicaCore::LIST_SLACK));
                item.list_copy.assign(bytes, bytes + readable);
                // (Nothing but zeroes where the memory ends before that)
                item.list_copy.resize(list_size + Pica::PicaCore::LIST_SLACK);
                if ((Common::TraceMarkers::Experiments3() &
                     Common::TraceMarkers::LiveCalledLists) == 0) [[likely]] {
                    // And the lists it jumps to. Where they are is in registers that the
                    // lists before it may have written, which the decoder may not have got
                    // to yet: followed here from list to list, and looked up whenever
                    // nothing is left to decode.
                    if (impl->done_seq.load() == impl->queued_seq) {
                        impl->list_scan = impl->pica.ListScanStateNow();
                    }
                    item.has_called_lists = true;
                    const auto take_start = std::chrono::steady_clock::now();
                    item.called_lists.reserve(impl->called_lists_room);
                    item.called_bytes.reserve(impl->called_bytes_room);
                    if (!Pica::PicaCore::TakeCalledLists(impl->memory, item.list_copy.data(),
                                                         list_size, readable, impl->list_scan,
                                                         item.called_lists, item.called_bytes)) {
                        // Too many to take along: the game goes on when they are decoded
                        may_tell_early = false;
                        impl->called_not_all++;
                    }
                    impl->called_take_ns +=
                        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - take_start)
                                             .count());
                    impl->called_taken += item.called_lists.size();
                    impl->called_taken_bytes += item.called_bytes.size();
                    // (Shrinks slowly, so that one command with few does not undo it)
                    impl->called_lists_room =
                        std::max(item.called_lists.size(), impl->called_lists_room * 15 / 16);
                    impl->called_bytes_room =
                        std::max(item.called_bytes.size(), impl->called_bytes_room * 15 / 16);
                }
            } else {
                may_tell_early = false;
            }
        } else if (command.id != CommandId::MemoryFill) {
            pace_target = PaceTargetOf(command);
        }
    }
    std::vector<Impl::RaisedInterrupt> tell;
    {
        std::scoped_lock lock{impl->decoder_mutex};
        const u32 kind = EarlyKind(command);
        item.ordinal = impl->frame_ordinal++;
        if (may_tell_early) {
            const bool same_place =
                item.ordinal < impl->early_at.size() && impl->early_at[item.ordinal].kind == kind;
            tell = same_place ? impl->early_at[item.ordinal].raised : impl->early_known[kind];
            item.early = !tell.empty();
            for (const Impl::RaisedInterrupt& interrupt : tell) {
                item.early_ids.push_back(interrupt.id);
                item.early_delays.push_back(interrupt.delay);
            }
        }
        ++impl->queued_seq;
        impl->queued.push_back(std::move(item));
        ++impl->pending;
        impl->has_pending.store(true);
        impl->has_queued.store(true);
    }
    impl->decoder_wake.notify_one();

    if (pace_target != 0) {
        // The pacing of a game that renders ahead of the display, see MemoryTransfer: the
        // copy that completes a second image for the top screen within one VBlank period is
        // not reported as done until the VBlank
        const bool paced =
            (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyRenderPace) == 0;
        if (paced && !tell.empty() && render_pace.filled != 0 &&
            render_pace.filled != pace_target && render_pace.held == 0) {
            if (!render_pace.reported) {
                render_pace.reported = true;
                LOG_INFO(HW_GPU, "The game renders faster than the display shows: its images for "
                                 "the top screen are paced to the VBlank");
            }
            render_pace.held = pace_target;
            render_pace.held_delay = tell.front().delay;
            return ResultSuccess;
        }
        render_pace.filled = pace_target;
    }
    for (const Impl::RaisedInterrupt& interrupt : tell) {
        impl->deliver_interrupt(interrupt.id, interrupt.delay);
    }
    return ResultSuccess;
}

bool GPU::CompletesEarly() const {
    return impl->early_completion;
}

bool GPU::UsesDecoder() const {
    if (!impl->decoder.joinable()) {
        return false;
    }
    // A movie plays back right only with every interrupt where it was when it was recorded:
    // commands are then decoded where the game hands them over, as they always were
    const Core::Movie::PlayMode movie = impl->system.Movie().GetPlayMode();
    return movie != Core::Movie::PlayMode::Recording && movie != Core::Movie::PlayMode::Playing;
}

PAddr GPU::PaceTargetOf(const Service::GSP::Command& command) {
    // As Execute fills the registers of the transfer engine and MemoryTransfer reads them
    PAddr output_start = 0;
    u32 output_size = 0;
    u32 output_extent = 0;
    if (command.id == Service::GSP::CommandId::DisplayTransfer) {
        const auto& params = command.display_transfer;
        if ((params.flags & 8) != 0) {
            // A copy by way of the transfer command: its size is what was in the registers
            return 0;
        }
        output_start = VirtualToPhysicalAddress(params.out_buffer_address, 1).value_or(0) & ~7u;
        output_size = (params.out_buffer_size & 0xFFFF) * (params.out_buffer_size >> 16) *
                      BytesPerPixel(static_cast<Pica::PixelFormat>((params.flags >> 12) & 7));
        output_extent = output_size;
    } else {
        const auto& params = command.texture_copy;
        if ((params.flags & 8) == 0) {
            return 0;
        }
        output_start =
            VirtualToPhysicalAddress(params.out_buffer_address, params.size).value_or(0) & ~7u;
        output_size = params.size;
        output_extent = output_size;
        const u32 row = (params.out_width_gap & 0xFFFF) * 16;
        const u32 gap = (params.out_width_gap >> 16) * 16;
        if (row != 0 && gap != 0 && output_size > row) {
            output_extent = (output_size / row - 1) * (row + gap) + row;
        }
    }
    if (output_start == 0) {
        return 0;
    }
    const u32 buffer_size = render_pace.stride * impl->pica.regs.framebuffer_config[0].height;
    PAddr target = 0;
    for (const PAddr buffer : render_pace.shown) {
        if (buffer != 0 && output_start < buffer + buffer_size &&
            buffer < output_start + output_extent) {
            target = buffer;
        }
    }
    return target;
}

std::optional<std::pair<PAddr, u32>> GPU::SourceOf(const Service::GSP::Command& command) {
    // As Execute fills the registers of the transfer engine and MemoryTransfer reads them
    if (command.id == Service::GSP::CommandId::DisplayTransfer) {
        const auto& params = command.display_transfer;
        if ((params.flags & 8) != 0) {
            return std::nullopt;
        }
        const PAddr start = VirtualToPhysicalAddress(params.in_buffer_address, 1).value_or(0) & ~7u;
        // Rows as long as the source's are, as many as the taller of the two has
        const u32 width = params.in_buffer_size & 0xFFFF;
        const u32 rows = std::max(params.in_buffer_size >> 16, params.out_buffer_size >> 16);
        const u32 size =
            width * rows * BytesPerPixel(static_cast<Pica::PixelFormat>((params.flags >> 8) & 7));
        return std::make_pair(start, start != 0 ? size : 0);
    }
    const auto& params = command.texture_copy;
    if ((params.flags & 8) == 0) {
        return std::nullopt;
    }
    const PAddr start =
        VirtualToPhysicalAddress(params.in_buffer_address, params.size).value_or(0) & ~7u;
    u32 extent = params.size;
    const u32 row = (params.in_width_gap & 0xFFFF) * 16;
    const u32 gap = (params.in_width_gap >> 16) * 16;
    if (row != 0 && gap != 0 && params.size > row) {
        extent = (params.size / row - 1) * (row + gap) + row;
    }
    return std::make_pair(start, start != 0 ? extent : 0);
}

std::optional<std::array<std::pair<PAddr, u32>, 2>> GPU::TargetsOf(
    const Service::GSP::Command& command) {
    using Service::GSP::CommandId;
    std::array<std::pair<PAddr, u32>, 2> targets{};
    if (command.id == CommandId::MemoryFill) {
        const auto& params = command.memory_fill;
        const auto place = [this](u32 start, u32 end) -> std::pair<PAddr, u32> {
            if (start == 0 || start >= end) {
                return {};
            }
            const PAddr address = VirtualToPhysicalAddress(start, end - start).value_or(0) & ~7u;
            return {address, address != 0 ? end - start : 0};
        };
        targets[0] = place(params.start1, params.end1);
        targets[1] = place(params.start2, params.end2);
        return targets;
    }
    if (command.id == CommandId::DisplayTransfer) {
        const auto& params = command.display_transfer;
        if ((params.flags & 8) != 0) {
            return std::nullopt;
        }
        const PAddr start =
            VirtualToPhysicalAddress(params.out_buffer_address, 1).value_or(0) & ~7u;
        const u32 pixel = BytesPerPixel(static_cast<Pica::PixelFormat>((params.flags >> 12) & 7));
        const u32 out_width = params.out_buffer_size & 0xFFFF;
        const u32 out_height = params.out_buffer_size >> 16;
        const u32 in_width = params.in_buffer_size & 0xFFFF;
        u32 size = out_width * out_height * pixel;
        // Flipped and cropped at once, it is written further on (see the transfer itself)
        if ((params.flags & 1) != 0 && (params.flags & 4) != 0 && in_width > out_width &&
            out_height != 0) {
            size += (in_width - out_width) * (out_height - 1) * pixel;
        }
        targets[0] = {start, start != 0 ? size : 0};
        return targets;
    }
    if (command.id == CommandId::TextureCopy) {
        const auto& params = command.texture_copy;
        if ((params.flags & 8) == 0) {
            return std::nullopt;
        }
        const PAddr start =
            VirtualToPhysicalAddress(params.out_buffer_address, params.size).value_or(0) & ~7u;
        u32 extent = params.size;
        const u32 row = (params.out_width_gap & 0xFFFF) * 16;
        const u32 gap = (params.out_width_gap >> 16) * 16;
        if (row != 0 && gap != 0 && params.size > row) {
            extent = (params.size / row - 1) * (row + gap) + row;
        }
        targets[0] = {start, start != 0 ? extent : 0};
        return targets;
    }
    return targets;
}

void GPU::WaitForPresent() {
    const auto done = [this] { return impl->presents_done.load() >= impl->presents_queued; };
    if (done()) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto until = start + HANDOFF_SPIN;
    while (!done() && std::chrono::steady_clock::now() < until) {
        SpinPause();
    }
    if (!done()) {
        std::unique_lock lock{impl->decoder_mutex};
        impl->decoder_idle.wait(lock, done);
    }
    Impl::Waited& waited = impl->waited[static_cast<std::size_t>(WaitReason::VBlank)];
    waited.times.fetch_add(1, std::memory_order_relaxed);
    waited.nanoseconds.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count(),
                                 std::memory_order_relaxed);
}

bool GPU::HasPendingWork() const {
    return impl->has_pending.load(std::memory_order_relaxed);
}

void GPU::NoteAccess(WaitReason reason) {
    impl->touched[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void GPU::WaitIdle(WaitReason reason) {
    if (!impl->decoder.joinable() || std::this_thread::get_id() == impl->decoder_id.load()) {
        return;
    }
    impl->touched[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    if (impl->has_pending.load()) {
        const auto start = std::chrono::steady_clock::now();
        // What is left is mostly less than going to sleep and being woken takes
        const auto until = start + HANDOFF_SPIN;
        while (impl->has_pending.load(std::memory_order_relaxed) &&
               std::chrono::steady_clock::now() < until) {
            SpinPause();
        }
        if (impl->has_pending.load()) {
            std::unique_lock lock{impl->decoder_mutex};
            impl->decoder_idle.wait(lock, [this] { return impl->pending == 0; });
        }
        Impl::Waited& waited = impl->waited[static_cast<std::size_t>(reason)];
        waited.times.fetch_add(1, std::memory_order_relaxed);
        waited.nanoseconds.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - start)
                                         .count(),
                                     std::memory_order_relaxed);
    }
    DeliverInterrupts();
}

void GPU::ReportDecoderWaits() {
    static constexpr u32 REPORT_VBLANKS = 600;
    if (!impl->decoder.joinable() || ++impl->report_vblanks < REPORT_VBLANKS) {
        return;
    }
    impl->report_vblanks = 0;
    static constexpr std::array<const char*, static_cast<std::size_t>(WaitReason::Count)> NAMES = {
        "other",
        "a command executed here",
        "a register",
        "a buffer swap",
        "the vblank",
        "a memory access",
        "a mapping",
        "nothing to run",
        "read of what a surface has newer",
        "read of what the gpu writes",
        "read of what a draw reads in place",
        "write into a surface",
        "write to what a draw reads in place",
        "write to vertex data",
        "write to index data",
        "write to a texture",
        "write to a called list",
        "write to something else it read",
        "write beside what it read",
        "write to what the gpu writes"};
    // For each: how often it was come by, and of those how often and how long it waited
    std::string waits;
    double waited_ms = 0.0;
    for (std::size_t i = 0; i < impl->waited.size(); i++) {
        const u32 touched = impl->touched[i].exchange(0, std::memory_order_relaxed);
        const u32 times = impl->waited[i].times.exchange(0, std::memory_order_relaxed);
        const u64 ns = impl->waited[i].nanoseconds.exchange(0, std::memory_order_relaxed);
        if (times == 0 && touched < REPORT_VBLANKS / 10) {
            continue;
        }
        waited_ms += static_cast<double>(ns) / 1e6 / REPORT_VBLANKS;
        waits += fmt::format("{}{} {:.1f} ({:.1f} waited, {:.2f} ms)", waits.empty() ? "" : "; ",
                             NAMES[i], static_cast<double>(touched) / REPORT_VBLANKS,
                             static_cast<double>(times) / REPORT_VBLANKS,
                             static_cast<double>(ns) / 1e6 / REPORT_VBLANKS);
    }
    const auto [volatile_pages, copied] = impl->memory.TakeSnapshotFigures();
    waits += fmt::format("; {} pages copied at hand-over, {:.0f} KiB a vblank", volatile_pages,
                         static_cast<double>(copied) / 1024.0 / REPORT_VBLANKS);
    if (const auto ages = impl->memory.VolatileAgesNow();
        volatile_pages != 0 || ages.forgotten != 0) {
        waits +=
            fmt::format(" (from {:08X} to {:08X} in {} runs; a copy of them was last read "
                        "within 256 commands: {}, within 2048: {}, longer ago: {}, never: "
                        "{}; {} forgotten for not being read; read as vertex data {} "
                        "({} lately), index data {} ({}), texture {} ({}), called list {} "
                        "({}), other {} ({}))",
                        ages.first, ages.last, ages.runs, ages.read[0], ages.read[1], ages.read[2],
                        ages.read[3], ages.forgotten, ages.kinds[1], ages.kinds_read[1],
                        ages.kinds[2], ages.kinds_read[2], ages.kinds[3], ages.kinds_read[3],
                        ages.kinds[4], ages.kinds_read[4], ages.kinds[0], ages.kinds_read[0]);
    }
    const auto called = impl->pica.TakeCalledListFigures();
    const u64 called_taken = std::exchange(impl->called_taken, 0);
    const u64 called_taken_bytes = std::exchange(impl->called_taken_bytes, 0);
    const u64 called_not_all = std::exchange(impl->called_not_all, 0);
    const u64 called_take_ns = std::exchange(impl->called_take_ns, 0);
    waits += fmt::format("; following the lists' jumps at hand-over {:.3f} ms a vblank",
                         static_cast<double>(called_take_ns) / 1e6 / REPORT_VBLANKS);
    if (called_taken != 0 || called.past_the_copies != 0 || called.another_list != 0) {
        waits += fmt::format(", {:.1f} lists jumped to taken along ({:.0f} KiB, {} times not "
                             "all of them), {} of {} had been written to by the game before "
                             "they were decoded, read from memory for want of a copy {} after "
                             "the last copy and {} where another list was due",
                             static_cast<double>(called_taken) / REPORT_VBLANKS,
                             static_cast<double>(called_taken_bytes) / 1024.0 / REPORT_VBLANKS,
                             called_not_all, called.rewritten, called.from_copy,
                             called.past_the_copies, called.another_list);
    }
    waits += work_rules->TakeLeftOutText(REPORT_VBLANKS);
    const auto [copies_given, copies_read] = impl->memory.TakeSnapshotReadFigures();
    if (copies_given != 0) {
        waits +=
            fmt::format(", {:.1f}% of the copies read", 100.0 * static_cast<double>(copies_read) /
                                                            static_cast<double>(copies_given));
    }
    const auto per_vblank_ms = [](std::atomic<u64>& ns) {
        return static_cast<double>(ns.exchange(0, std::memory_order_relaxed)) / 1e6 /
               REPORT_VBLANKS;
    };
    const u32 composed = impl->compose_lags.exchange(0, std::memory_order_relaxed);
    const u64 lag_sum = impl->compose_lag_ns.exchange(0, std::memory_order_relaxed);
    const u64 lag_max = impl->compose_lag_max_ns.exchange(0, std::memory_order_relaxed);
    waits += fmt::format("; an image is composed {:.2f} ms after its vblank, {:.1f} at the most",
                         composed != 0 ? static_cast<double>(lag_sum) / 1e6 / composed : 0.0,
                         static_cast<double>(lag_max) / 1e6);
    const double told = per_vblank_ms(impl->told_ns);
    const double told_long = per_vblank_ms(impl->told_long_ns);
    const double told_short = per_vblank_ms(impl->told_short_ns);
    waits +=
        fmt::format("; delays told at hand-over {:.2f} ms a vblank, {:.2f} too long and "
                    "{:.2f} too short, {:.2f} commands a vblank told late",
                    told, told_long, told_short,
                    static_cast<double>(impl->told_late.exchange(0, std::memory_order_relaxed)) /
                        REPORT_VBLANKS);
    const double decoder_ms =
        static_cast<double>(impl->decoder_nanoseconds.exchange(0, std::memory_order_relaxed)) /
        1e6 / REPORT_VBLANKS;
    const double commands =
        static_cast<double>(impl->decoder_commands.exchange(0, std::memory_order_relaxed)) /
        REPORT_VBLANKS;
    LOG_INFO(HW_GPU,
             "Decoder thread, a vblank: {:.2f} ms of work in {:.1f} commands; the emulation "
             "thread waited {:.2f} ms for it: {}",
             decoder_ms, commands, waited_ms, waits);
}

void GPU::DeliverInterrupts() {
    if (!impl->has_raised.load(std::memory_order_relaxed) ||
        std::this_thread::get_id() != impl->emulation_id.load()) {
        return;
    }
    std::vector<Impl::RaisedInterrupt> raised;
    {
        std::scoped_lock lock{impl->decoder_mutex};
        raised.swap(impl->raised);
        impl->has_raised.store(false);
    }
    // The emulated GPU would have taken `delay` from when the command was sent. Of that,
    // what the game has run through in the meantime is gone already.
    const u64 now = static_cast<u64>(impl->timing.GetGlobalTicks());
    for (const Impl::RaisedInterrupt& interrupt : raised) {
        const u64 elapsed = now > interrupt.submit_ticks ? now - interrupt.submit_ticks : 0;
        impl->deliver_interrupt(interrupt.id,
                                interrupt.delay > elapsed ? interrupt.delay - elapsed : 0);
    }
}

constexpr Result ResultInvalidGpuBuffer(ErrorDescription::InvalidAddress, ErrorModule::GX,
                                        ErrorSummary::InvalidArgument,
                                        ErrorLevel::Usage); // 0xE0E02BF5
constexpr Result ResultUnknownGxCommand(ErrorDescription::NotImplemented, ErrorModule::GX,
                                        ErrorSummary::NotSupported,
                                        ErrorLevel::Usage); // 0xE0C02BF4

std::optional<PAddr> GPU::VirtualToPhysicalAddress(VAddr addr, u32 size) {
    // Address translation of the GSP module.
    // The linear heaps and the New 3DS extra memory must contain both the start,
    // and the end of the buffer, VRAM only the start. The ends of the
    // ranges are inclusive.
    const VAddr end = addr + size;
    const auto contains = [addr, end](VAddr range_start, VAddr range_end) {
        return addr >= range_start && addr <= range_end && end >= range_start && end <= range_end;
    };

    if (contains(Memory::LINEAR_HEAP_VADDR, Memory::LINEAR_HEAP_VADDR_END)) {
        return addr - Memory::LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    if (contains(Memory::NEW_LINEAR_HEAP_VADDR, Memory::NEW_LINEAR_HEAP_VADDR_END)) {
        return addr - Memory::NEW_LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    if (addr >= Memory::VRAM_VADDR && addr <= Memory::VRAM_VADDR_END) {
        return addr - Memory::VRAM_VADDR + Memory::VRAM_PADDR;
    }
    if (contains(Memory::N3DS_EXTRA_RAM_VADDR, Memory::N3DS_EXTRA_RAM_VADDR_END)) {
        return addr - Memory::N3DS_EXTRA_RAM_VADDR + Memory::N3DS_EXTRA_RAM_PADDR;
    }

    // TODO(PabloMK7): Find a better way to handle this.
    PAddr plg_fb_addr;
    if (contains(Memory::PLUGIN_3GX_FB_VADDR, Memory::PLUGIN_3GX_FB_VADDR_END) &&
        (plg_fb_addr = impl->system.Memory().Plugin3GXFramebufferAddress())) {
        return addr - Memory::PLUGIN_3GX_FB_VADDR + plg_fb_addr;
    }

    LOG_ERROR(HW_GPU, "Invalid GPU buffer address 0x{:08X} (size 0x{:X})", addr, size);
    return std::nullopt;
}

void GPU::SetInterruptHandler(Service::GSP::InterruptHandler handler) {
    impl->deliver_interrupt = handler;
    // The service's handler schedules an event and wakes threads of the game: emulation
    // thread only. What the decoding thread raises is kept for DeliverInterrupts.
    Service::GSP::InterruptHandler raise = [this](Service::GSP::InterruptId id, u64 delay) {
        if (std::this_thread::get_id() == impl->decoder_id.load()) {
            impl->working_raised.push_back({id, delay, impl->working_submit_ticks});
            if (!impl->working_early) {
                std::scoped_lock lock{impl->decoder_mutex};
                impl->raised.push_back({id, delay, impl->working_submit_ticks});
                impl->has_raised.store(true);
            }
            return;
        }
        impl->deliver_interrupt(id, delay);
    };
    impl->signal_interrupt = raise;
    impl->pica.SetInterruptHandler(raise);
}

void GPU::FlushRegion(PAddr addr, u32 size) {
    WaitIdle();
    impl->rasterizer->FlushRegion(addr, size);
}

void GPU::InvalidateRegion(PAddr addr, u32 size) {
    WaitIdle();
    impl->rasterizer->InvalidateRegion(addr, size);
}

void GPU::ClearAll(bool flush) {
    WaitIdle();
    // The state may be about to be written out, page tables and all
    impl->memory.DisarmAll();
    impl->rasterizer->ClearAll(flush);
}

Result GPU::Execute(const Service::GSP::Command& command) {
    using Service::GSP::CommandId;
    auto& regs = impl->pica.regs;
    Result result = ResultSuccess;

    switch (command.id) {
    case CommandId::RequestDma: {
        impl->system.Memory().RasterizerFlushVirtualRegion(
            command.dma_request.source_address, command.dma_request.size, Memory::FlushMode::Flush);
        impl->system.Memory().RasterizerFlushVirtualRegion(command.dma_request.dest_address,
                                                           command.dma_request.size,
                                                           Memory::FlushMode::Invalidate);

        // TODO(Subv): These memory accesses should not go through the application's memory mapping.
        // They should go through the GSP module's memory mapping.
        const auto process = impl->system.Kernel().GetCurrentProcess();
        impl->memory.CopyBlock(*process, command.dma_request.dest_address,
                               command.dma_request.source_address, command.dma_request.size);

        auto is_vram = [&](u32 addr) {
            return addr >= Memory::VRAM_VADDR && addr <= Memory::VRAM_VADDR_END;
        };

        u64 delay = DelayGenerator::CalculateDelayNanoseconds(
            DelayGenerator::GetCopyMode(is_vram(command.dma_request.source_address),
                                        is_vram(command.dma_request.dest_address)),
            false, command.dma_request.size);

        impl->signal_interrupt(Service::GSP::InterruptId::DMA, delay);
        break;
    }
    case CommandId::SubmitCmdList: {
        auto& params = command.submit_gpu_cmdlist;
        auto& cmdbuffer = regs.internal.pipeline.command_buffer;

        // Write to the command buffer GPU registers.
        // The GSP module uses address 0 when the buffer is not in GPU visible memory.
        const PAddr address = VirtualToPhysicalAddress(params.address, params.size).value_or(0);
        cmdbuffer.addr[0].Assign(address >> 3);
        cmdbuffer.size[0].Assign(params.size >> 3);
        cmdbuffer.trigger[0] = 1;
        if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::MeasureCommandLists) != 0)
            [[unlikely]] {
            MeasureListRepeat(impl->memory.GetPhysicalPointer(address), params.size, address);
        }

        // Trigger processing of the command list
        SubmitCmdList(0);
        break;
    }
    case CommandId::MemoryFill: {
        auto& params = command.memory_fill;
        auto& memfill = regs.memory_fill_config;

        const auto translate = [this](u32 start, u32 end) -> std::optional<PAddr> {
            if (start >= end) {
                return std::nullopt;
            }
            return VirtualToPhysicalAddress(start, end - start);
        };

        // Write to the memory fill GPU registers.
        // If both buffers are set GSP dispatches PSC0 only.
        const bool has_both_bufs = params.start1 != 0 && params.start2 != 0;
        if (params.start1 != 0) {
            const auto start = translate(params.start1, params.end1);
            if (!start) {
                result = ResultInvalidGpuBuffer;
                break;
            }
            memfill[0].address_start = *start >> 3;
            memfill[0].address_end = (*start + params.end1 - params.start1) >> 3;
            memfill[0].value_32bit = params.value1;
            memfill[0].control = params.control1;
            MemoryFill(0, has_both_bufs ? std::numeric_limits<u32>::max() : 0);
        }
        if (params.start2 != 0) {
            const auto start = translate(params.start2, params.end2);
            if (!start) {
                result = ResultInvalidGpuBuffer;
                break;
            }
            memfill[1].address_start = *start >> 3;
            memfill[1].address_end = (*start + params.end2 - params.start2) >> 3;
            memfill[1].value_32bit = params.value2;
            memfill[1].control = params.control2;
            MemoryFill(1, has_both_bufs ? 0 : 1);
        }
        break;
    }
    case CommandId::DisplayTransfer: {
        auto& params = command.display_transfer;
        auto& display_transfer = regs.display_transfer_config;

        // Write to the transfer engine GPU registers.
        // The GSP module uses address 0 when the buffer is not in GPU visible memory (only the
        // first byte is checked).
        display_transfer.input_address =
            VirtualToPhysicalAddress(params.in_buffer_address, 1).value_or(0) >> 3;
        display_transfer.output_address =
            VirtualToPhysicalAddress(params.out_buffer_address, 1).value_or(0) >> 3;
        display_transfer.input_size = params.in_buffer_size;
        display_transfer.output_size = params.out_buffer_size;
        display_transfer.flags = params.flags;
        display_transfer.trigger.Assign(1);

        // Trigger the display transfer.
        MemoryTransfer();
        break;
    }
    case CommandId::TextureCopy: {
        auto& params = command.texture_copy;
        auto& texture_copy = regs.display_transfer_config;

        // Write to the transfer engine GPU registers.
        // The GSP module uses address 0 when the buffer is not in GPU visible memory.
        texture_copy.input_address =
            VirtualToPhysicalAddress(params.in_buffer_address, params.size).value_or(0) >> 3;
        texture_copy.output_address =
            VirtualToPhysicalAddress(params.out_buffer_address, params.size).value_or(0) >> 3;
        texture_copy.texture_copy.size = params.size;
        texture_copy.texture_copy.input_size = params.in_width_gap;
        texture_copy.texture_copy.output_size = params.out_width_gap;
        texture_copy.flags = params.flags;
        texture_copy.trigger.Assign(1);

        // Trigger the texture copy.
        MemoryTransfer();
        break;
    }
    case CommandId::CacheFlush: {
        // Rasterizer flushing handled elsewhere in CPU read/write and other GPU handlers
        // Use command.cache_flush.regions to implement this handler
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Unknown command {:#08X}", command.id.Value());
        result = ResultUnknownGxCommand;
    }

    // Notify debugger that a GSP command was processed.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::GSPCommandProcessed, &command);
    }
    return result;
}

void GPU::SetBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info) {
    const bool from_queue = std::this_thread::get_id() == impl->decoder_id.load();
    if (impl->early_completion && !from_queue && UsesDecoder()) {
        // What the pacing rule goes by is kept where the game is told "done": here
        if (screen_id == 0) {
            if (const auto left = VirtualToPhysicalAddress(info.address_left, info.stride)) {
                if (std::ranges::find(render_pace.shown, *left) == render_pace.shown.end()) {
                    render_pace.shown[render_pace.shown_next++ % render_pace.shown.size()] = *left;
                }
                render_pace.stride = info.stride;
            }
            // The game's frame ends here, whenever the decoder gets to it
            impl->system.perf_stats->EndGameFrame();
            Common::Hitches::NoteGameFrame();
            impl->frame_ordinal = 0;
        }
        // The transfer that fills the buffer may still be on its way through the decoder:
        // the swap keeps its place behind it
        Impl::QueuedCommand item{};
        item.is_swap = true;
        item.swap_screen = screen_id;
        item.swap_info = info;
        item.submit_ticks = static_cast<u64>(impl->timing.GetGlobalTicks());
        impl->Queue(std::move(item));
        return;
    }
    WaitIdle(WaitReason::BufferSwap);
    // The GSP module does not apply a framebuffer outside GPU visible memory. The right address
    // is used only for the top screen in stereoscopic mode and when it is not zero.
    // Otherwise the left address is used for both eyes.
    const auto phys_address_left = VirtualToPhysicalAddress(info.address_left, info.stride);
    if (!phys_address_left) {
        return;
    }

    // Update framebuffer properties.
    auto& framebuffer = impl->pica.regs.framebuffer_config[screen_id];
    if (screen_id == 0) {
        const bool stereoscopic = (info.format & 0x70) == 0x20;
        const PAddr phys_address_right =
            (!stereoscopic || info.address_right == 0)
                ? *phys_address_left
                : VirtualToPhysicalAddress(info.address_right, info.stride).value_or(0);
        if (info.active_fb == 0) {
            framebuffer.address_right1 = phys_address_right;
        } else {
            framebuffer.address_right2 = phys_address_right;
        }
    }
    if (info.active_fb == 0) {
        framebuffer.address_left1 = *phys_address_left;
    } else {
        framebuffer.address_left2 = *phys_address_left;
    }
    if (screen_id == 0 && !from_queue &&
        std::ranges::find(render_pace.shown, *phys_address_left) == render_pace.shown.end()) {
        render_pace.shown[render_pace.shown_next++ % render_pace.shown.size()] = *phys_address_left;
    }

    framebuffer.stride = info.stride;
    framebuffer.format = (framebuffer.format & 0xFFFF0000) | (info.format & 0xFFFF);
    framebuffer.active_fb = info.shown_fb & 0xFF;

    // Notify debugger about the buffer swap.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::BufferSwapped, nullptr);
    }
    impl->renderer->NotifyScreenChanged();

    if (screen_id == 0) {
        MicroProfileFlip();
        // (Counted at hand-over where the swap came through the decoder's queue)
        if (!from_queue) {
            impl->system.perf_stats->EndGameFrame();
            Common::Hitches::NoteGameFrame();
        }
        // Steps once per guest frame (top screen buffer swap).
        CITRA_TRACE_SET_COUNTER("azahar.game_frame", impl->system.perf_stats->GetGameFrameCount());
        CITRA_FRAME_DUMP("bufferswap top (guest frame end) to {:08X}, {} bytes a row, {} rows",
                         *phys_address_left, info.stride, framebuffer.height.Value());
        right_eye_disabler->ReportEndFrame();
        work_rules->EndFrame(FrameFacts{impl->pica.regs.framebuffer_config[0]});
    }
}

void GPU::SetColorFill(const Pica::ColorFill& fill) {
    WaitIdle(WaitReason::BufferSwap);
    impl->pica.regs_lcd.color_fill_top = fill;
    impl->pica.regs_lcd.color_fill_bottom = fill;
    impl->renderer->NotifyScreenChanged();
}

u32 GPU::ReadReg(VAddr addr) {
    WaitIdle(WaitReason::Register);
    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        return impl->pica.regs_lcd[index];
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        return impl->pica.regs.reg_array[index];
    }
    default:
        UNREACHABLE_MSG("Read from unknown GPU address {:#08X}", addr);
    }
}

void GPU::WriteReg(VAddr addr, u32 data) {
    WaitIdle(WaitReason::Register);
    // Registers written this way include the ones that select and configure what is displayed
    impl->renderer->NotifyScreenChanged();

    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        impl->pica.regs_lcd[index] = data;
        break;
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);

        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        impl->pica.regs.reg_array[index] = data;

        // Handle registers that trigger GPU actions
        switch (index) {
        case GPU_REG_INDEX(memory_fill_config[0].trigger):
            MemoryFill(0, 0);
            break;
        case GPU_REG_INDEX(memory_fill_config[1].trigger):
            MemoryFill(1, 1);
            break;
        case GPU_REG_INDEX(display_transfer_config.trigger):
            MemoryTransfer();
            break;
        case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[0]):
            SubmitCmdList(0);
            break;
        case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[1]):
            SubmitCmdList(1);
            break;
        default:
            break;
        }
        break;
    }
    default:
        UNREACHABLE_MSG("Write to unknown GPU address {:#08X}", addr);
    }
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

Pica::PicaCore& GPU::PicaCore() {
    return impl->pica;
}

const Pica::PicaCore& GPU::PicaCore() const {
    return impl->pica;
}

GraphicsDebugger& GPU::Debugger() {
    return impl->gpu_debugger;
}

#ifdef ENABLE_FREECAM
void GPU::SetFreeCam(const std::array<f32, 16>& matrix, bool skip_2d) {
    WaitIdle();
    impl->rasterizer->SetFreeCam(matrix, skip_2d);
}
#endif

void GPU::ApplyPerProgramSettings(u64 program_ID) {
    WaitIdle();
    auto hack = Common::Hacks::hack_manager.GetHack(
        Common::Hacks::HackType::ACCURATE_MULTIPLICATION, program_ID);
    bool use_accurate_mul = Settings::values.shaders_accurate_mul.GetValue();
    if (hack) {
        switch (hack->mode) {
        case Common::Hacks::HackAllowMode::DISALLOW:
            use_accurate_mul = false;
            break;
        case Common::Hacks::HackAllowMode::FORCE:
            use_accurate_mul = true;
            break;
        case Common::Hacks::HackAllowMode::ALLOW:
        default:
            break;
        }
    }
    impl->rasterizer->SetAccurateMul(use_accurate_mul);
}

namespace {

/// Whether a range of guest memory lies in one of the buffers on display.
template <typename FramebufferConfigs>
bool IsOnDisplay(const FramebufferConfigs& configs, PAddr start, PAddr end) {
    for (const auto& framebuffer : configs) {
        const u32 size = framebuffer.stride * framebuffer.height;
        const bool first = framebuffer.active_fb == 0;
        const std::array<PAddr, 2> addresses{
            first ? framebuffer.address_left1 : framebuffer.address_left2,
            first ? framebuffer.address_right1 : framebuffer.address_right2,
        };
        for (const PAddr address : addresses) {
            if (address != 0 && start < address + size && address < end) {
                return true;
            }
        }
    }
    return false;
}

} // Anonymous namespace

void GPU::SubmitCmdList(u32 index) {
    // Check if a command list was triggered.
    auto& config = impl->pica.regs.internal.pipeline.command_buffer;
    if (!config.trigger[index]) {
        return;
    }

    MICROPROFILE_SCOPE(GPU_CmdlistProcessing);
    CITRA_TRACE_SCOPE("GPU::CommandList");
    CITRA_HITCH_SCOPE(GpuEmulation);
    CITRA_TRACE_COUNT(CommandLists);

    // Forward command list processing to the PICA core.
    const PAddr addr = config.GetPhysicalAddress(index);
    const u32 size = config.GetSize(index);

    // Do not process the list if the memory range is not sane.
    auto& memory = impl->system.Memory();
    const u8* head = memory.GetPhysicalPointer(addr);
    if (!head || (size != 0 && memory.GetPhysicalPointer(addr + size - 1) != head + size - 1)) {
        LOG_ERROR(HW_GPU, "Invalid command list address 0x{:08X} (size 0x{:X})", addr, size);
        impl->signal_interrupt(Service::GSP::InterruptId::P3D, 0);
        config.trigger[index] = 0;
        return;
    }

    CITRA_FRAME_DUMP("cmdlist @{:08X} size {}", addr, size);
    impl->pica.ProcessCmdList(addr, size,
                              !right_eye_disabler->ShouldAllowCmdQueueTrigger(addr, size));
    CITRA_FRAME_DUMP("cmdlist end");
    config.trigger[index] = 0;
}

void GPU::MemoryFill(u32 index, u32 intr_index) {
    // Check if a memory fill was triggered.
    auto& config = impl->pica.regs.memory_fill_config[index];
    if (!config.trigger) {
        return;
    }
    CITRA_TRACE_SCOPE("GPU::MemoryFill");
    CITRA_HITCH_SCOPE(GpuEmulation);

    if (IsOnDisplay(impl->pica.regs.framebuffer_config, config.GetStartAddress(),
                    config.GetEndAddress())) {
        impl->renderer->NotifyScreenChanged();
    }

    // Perform memory fill, unless a rule finds that it need not be done.
    CITRA_FRAME_DUMP("fill {}", config.DebugName());
    if (work_rules->LeavesOutFill(
            FillFacts{config.GetStartAddress(), config.GetEndAddress(), config})) {
        CITRA_FRAME_DUMP("  fill left out: {}", work_rules->LastWhy());
    } else if (!impl->rasterizer->AccelerateFill(config)) {
        CITRA_TRACE_SCOPE("GPU::MemoryFill (software fallback)");
        CITRA_FRAME_DUMP("  fill: SOFTWARE fallback");
        impl->memory.NoteGpuWritten(config.GetStartAddress(),
                                    config.GetEndAddress() - config.GetStartAddress());
        impl->sw_blitter->MemoryFill(config);
    }

    // Treat fill as texture transfer from VRAM
    u64 delay = DelayGenerator::CalculateDelayNanoseconds(
        DelayGenerator::GetCopyMode(true, config.IsVRAM()), true,
        config.GetEndAddress() - config.GetStartAddress());

    // It seems that it won't signal interrupt if "address_start" is zero.
    // TODO: hwtest this
    if (config.GetStartAddress() != 0) {
        if (intr_index == 0) {
            impl->signal_interrupt(Service::GSP::InterruptId::PSC0, delay);
        } else if (intr_index == 1) {
            impl->signal_interrupt(Service::GSP::InterruptId::PSC1, delay);
        }
    }

    // Reset "trigger" flag and set the "finish" flag
    // This was confirmed to happen on hardware even if "address_start" is zero.
    config.trigger.Assign(0);
    config.finished.Assign(1);
}

void GPU::MemoryTransfer() {
    // Check if a transfer was triggered.
    auto& config = impl->pica.regs.display_transfer_config;
    if (!config.trigger.Value()) {
        return;
    }

    MICROPROFILE_SCOPE(GPU_DisplayTransfer);
    CITRA_TRACE_SCOPE(config.is_texture_copy ? "GPU::TextureCopy" : "GPU::DisplayTransfer");
    CITRA_HITCH_SCOPE(GpuEmulation);

    // Notify debugger about the display transfer.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::IncomingDisplayTransfer, nullptr);
    }

    // A game without a second buffer transfers into the one on display, without a swap after
    const PAddr output_start = config.GetPhysicalOutputAddress();
    const u32 output_size = config.is_texture_copy ? config.texture_copy.size
                                                   : config.output_width * config.output_height *
                                                         BytesPerPixel(config.output_format);
    if (IsOnDisplay(impl->pica.regs.framebuffer_config, output_start, output_start + output_size)) {
        impl->renderer->NotifyScreenChanged();
    }
    // Whether a rule finds that it need not be done. What it reads and writes as memory: a
    // transfer reads as many rows as it puts out, each as long as the rows of its source are;
    // a copy reads and writes rows with gaps between them.
    bool left_out;
    {
        const PAddr input_start = config.GetPhysicalInputAddress();
        const u32 last_row_read = std::min<u32>(config.output_width, config.input_width);
        u32 input_size = config.output_height == 0
                             ? 0
                             : ((config.output_height - 1) * config.input_width + last_row_read) *
                                   BytesPerPixel(config.input_format);
        u32 written = output_size;
        if (config.is_texture_copy) {
            const auto extent = [&](u32 width, u32 gap) {
                const u32 row = width * 16;
                const u32 size = config.texture_copy.size;
                return row != 0 && size > row ? (size / row - 1) * (row + gap * 16) + row : size;
            };
            input_size = extent(config.texture_copy.input_width, config.texture_copy.input_gap);
            written = extent(config.texture_copy.output_width, config.texture_copy.output_gap);
        }
        left_out = work_rules->LeavesOutTransfer(TransferFacts{
            input_start, input_start + input_size, output_start, output_start + written, config});
    }

    u64 delay{};
    // Perform memory transfer
    if (config.is_texture_copy) {
        CITRA_TRACE_SCOPE_FMT(
            "texcopy {:08X}->{:08X} size {} in {}+{} out {}+{}", config.GetPhysicalInputAddress(),
            config.GetPhysicalOutputAddress(), config.texture_copy.size,
            config.texture_copy.input_width * 16, config.texture_copy.input_gap * 16,
            config.texture_copy.output_width * 16, config.texture_copy.output_gap * 16);
        CITRA_FRAME_DUMP("texcopy {:08X}->{:08X} size {} in {}+{} out {}+{}",
                         config.GetPhysicalInputAddress(), config.GetPhysicalOutputAddress(),
                         config.texture_copy.size, config.texture_copy.input_width * 16,
                         config.texture_copy.input_gap * 16, config.texture_copy.output_width * 16,
                         config.texture_copy.output_gap * 16);
        if (left_out) {
            CITRA_FRAME_DUMP("  texcopy left out: {}", work_rules->LastWhy());
        } else if (!impl->rasterizer->AccelerateTextureCopy(config)) {
            CITRA_TRACE_SCOPE("GPU::TextureCopy (software fallback)");
            CITRA_FRAME_DUMP("  texcopy: SOFTWARE fallback");
            impl->memory.NoteGpuWritten(output_start, output_size);
            impl->sw_blitter->TextureCopy(config);
        }
        CITRA_FRAME_DUMP("texcopy end");
        delay = DelayGenerator::CalculateDelayNanoseconds(
            DelayGenerator::GetCopyMode(config.IsInputVRAM(), config.IsOutputVRAM()), true,
            config.texture_copy.size);
    } else {
        if (left_out) {
            CITRA_FRAME_DUMP("transfer {} left out: {}", config.DebugName(), work_rules->LastWhy());
        } else if (right_eye_disabler->ShouldAllowDisplayTransfer(config.GetPhysicalInputAddress(),
                                                                  config.input_height)) {
            CITRA_FRAME_DUMP("transfer {} in {}x{} fmt {} out {}x{} fmt {} flags {:08X}",
                             config.DebugName(), config.input_width.Value(),
                             config.input_height.Value(),
                             static_cast<u32>(config.input_format.Value()),
                             config.output_width.Value(), config.output_height.Value(),
                             static_cast<u32>(config.output_format.Value()), config.flags);
            if (!impl->rasterizer->AccelerateDisplayTransfer(config)) {
                CITRA_TRACE_SCOPE("GPU::DisplayTransfer (software fallback)");
                CITRA_FRAME_DUMP("  transfer: SOFTWARE fallback");
                impl->memory.NoteGpuWritten(output_start, output_size);
                impl->sw_blitter->DisplayTransfer(config);
            }
        }
        delay = DelayGenerator::CalculateDelayNanoseconds(
            DelayGenerator::GetCopyMode(config.IsInputVRAM(), config.IsOutputVRAM()), true,
            config.input_width * config.input_height * BytesPerPixel(config.input_format));
    }

    // Complete transfer.
    config.trigger.Assign(0);

    if (impl->early_completion && std::this_thread::get_id() == impl->decoder_id.load()) {
        // The game was told at hand-over, and the pacing below was done there: see Submit
        impl->signal_interrupt(Service::GSP::InterruptId::PPF, delay);
        return;
    }

    // Some games do not wait for the display: they render into one of several buffers as fast
    // as the GPU finishes their work and show the newest at each VBlank. How fast that is here
    // comes from a rough model of the console's GPU, made twice as fast as the console, and a
    // game then finishes more images than there are VBlanks (Luigi's Mansion: Dark Moon, four
    // for every three). The extra ones are rendered in full by the host and never shown, and
    // what is shown advances unevenly. A real GPU would not have finished: so the copy that
    // completes a second image for the top screen within one VBlank period is not reported as
    // done until the VBlank. The game, which waits for that, renders one image a VBlank.
    const auto& top_screen = impl->pica.regs.framebuffer_config[0];
    const u32 buffer_size = top_screen.stride * top_screen.height;
    // What the copy covers. A game may show a buffer from some rows in (Luigi's Mansion copies
    // 416 rows and shows 400 of them), so the start of the copy need not be inside what is shown.
    u32 output_extent = output_size;
    if (config.is_texture_copy) {
        const u32 row = config.texture_copy.output_width * 16;
        const u32 gap = config.texture_copy.output_gap * 16;
        if (row != 0 && gap != 0 && output_size > row) {
            output_extent = (output_size / row - 1) * (row + gap) + row;
        }
    }
    PAddr target = 0;
    for (const PAddr buffer : render_pace.shown) {
        if (buffer != 0 && output_start < buffer + buffer_size &&
            buffer < output_start + output_extent) {
            target = buffer;
        }
    }
    const bool paced =
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyRenderPace) == 0;
    if (paced && target != 0 && render_pace.filled != 0 && render_pace.filled != target &&
        render_pace.held == 0) {
        CITRA_FRAME_DUMP("  the game is ahead of the display: this copy completes at the VBlank");
        if (!render_pace.reported) {
            render_pace.reported = true;
            LOG_INFO(HW_GPU, "The game renders faster than the display shows: its images for "
                             "the top screen are paced to the VBlank");
        }
        render_pace.held = target;
        render_pace.held_delay = delay;
        return;
    }
    if (target != 0) {
        CITRA_FRAME_DUMP("  an image for the top screen, in the buffer at {:08X}", target);
        render_pace.filled = target;
    }
    impl->signal_interrupt(Service::GSP::InterruptId::PPF, delay);
}

void GPU::FinishHeldTransfer() {
    WaitIdle();
    if (render_pace.held == 0) {
        return;
    }
    impl->signal_interrupt(Service::GSP::InterruptId::PPF, render_pace.held_delay);
    render_pace.filled = std::exchange(render_pace.held, 0);
}

void GPU::VBlankCallback(std::uintptr_t user_data, s64 cycles_late) {
    if (impl->early_completion && UsesDecoder() && impl->renderer->ComposesApart()) {
        // The decoder works behind the game: at a VBlank it may still be on what was handed
        // over since the VBlank before, and on no more than that. Its composing of the image
        // goes into the queue in its place, and the frame ends here for the emulation only.
        WaitForPresent();
        DeliverInterrupts();
        ReportDecoderWaits();
        impl->memory.ApplyLateWrites();
        impl->deliver_interrupt(Service::GSP::InterruptId::PDC0, 0);
        impl->deliver_interrupt(Service::GSP::InterruptId::PDC1, 0);

        // The image whose completion was held for this VBlank is the first of the new period
        if (render_pace.held != 0) {
            impl->deliver_interrupt(Service::GSP::InterruptId::PPF, render_pace.held_delay);
            render_pace.filled = std::exchange(render_pace.held, 0);
        } else {
            render_pace.filled = 0;
        }

        Impl::QueuedCommand item{};
        item.is_present = true;
        item.queued_at = std::chrono::steady_clock::now();
        item.present_vblank = impl->renderer->GetCurrentFrame();
        item.submit_ticks = static_cast<u64>(impl->timing.GetGlobalTicks());
        ++impl->presents_queued;
        impl->Queue(std::move(item));
        impl->renderer->EndEmulatedFrame();

        impl->timing.ScheduleEvent(FRAME_TICKS - cycles_late, impl->vblank_event);
        return;
    }
    WaitIdle(WaitReason::VBlank);
    ReportDecoderWaits();
    impl->memory.ApplyLateWrites();
    // Signal to GSP that GPU interrupt has occurred
    impl->signal_interrupt(Service::GSP::InterruptId::PDC0, 0);
    impl->signal_interrupt(Service::GSP::InterruptId::PDC1, 0);

    // The image whose completion was held for this VBlank is the first of the new period
    if (render_pace.held != 0) {
        FinishHeldTransfer();
    } else {
        render_pace.filled = 0;
    }

    // Present renderered frame.
    frame_liveness->NoteDisplay();
    {
        CITRA_TRACE_SCOPE("VBlank::SwapBuffers");
        CITRA_HITCH_SCOPE(Compose);
        impl->renderer->SwapBuffers();
    }

    // Reschedule recurrent event
    impl->timing.ScheduleEvent(FRAME_TICKS - cycles_late, impl->vblank_event);
}

void GPU::RecreateRenderer(Frontend::EmuWindow& emu_window, Frontend::EmuWindow* secondary_window) {
    WaitIdle();
    // Reset the renderer (this will destroy OpenGL resources)
    impl->renderer.reset();

    // Create a new renderer
    impl->renderer =
        VideoCore::CreateRenderer(emu_window, secondary_window, impl->pica, impl->system);
    impl->rasterizer = impl->renderer->Rasterizer();

    // Rebind the rasterizer to the PICA GPU
    impl->pica.BindRasterizer(impl->rasterizer);

    // Update the sw_blitter with the new rasterizer
    impl->sw_blitter = std::make_unique<SwRenderer::SwBlitter>(impl->memory, impl->rasterizer);

    // Re-apply per-game configuration and reload disk shader cache
    u64 program_id{};
    impl->system.GetAppLoader().ReadProgramId(program_id);
    ApplyPerProgramSettings(program_id);
    if (Settings::values.use_disk_shader_cache) {
        impl->renderer->Rasterizer()->LoadDefaultDiskResources(false, nullptr);
    }

    // Mark ALL GPU registers as dirty so current state gets uploaded to new renderer
    impl->pica.dirty_regs.SetAllDirty();

    // Also mark shader setups as dirty so uniforms get re-uploaded and
    // stale pointers to the old rasterizer's JIT cache are cleared.
    impl->pica.vs_setup.uniforms_dirty = true;
    impl->pica.vs_setup.cached_shader = nullptr;
    impl->pica.gs_setup.uniforms_dirty = true;
    impl->pica.gs_setup.cached_shader = nullptr;

    // Mark all cached LUT/table state in pica as dirty
    impl->pica.lighting.lut_dirty = impl->pica.lighting.LutAllDirty;
    impl->pica.fog.lut_dirty = true;
    impl->pica.proctex.table_dirty = impl->pica.proctex.TableAllDirty;
}

void GPU::ReleaseRenderer() {
    WaitIdle();
    // Just reset the renderer to release OpenGL resources
    // Don't null out rasterizer pointer as it will become dangling
    impl->renderer.reset();
    impl->sw_blitter.reset();
    LOG_INFO(HW_GPU, "Renderer released for context destroy");
}

template <class Archive>
void GPU::serialize(Archive& ar, const u32 file_version) {
    WaitIdle();
    ar & impl->pica;
}

SERIALIZE_IMPL(GPU)

} // namespace VideoCore
