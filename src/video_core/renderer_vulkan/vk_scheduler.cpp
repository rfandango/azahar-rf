// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <string>
#include <utility>
#include "common/hitch_tracker.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/thread.h"
#include "common/trace_markers.h"
#include "video_core/frame_state_dump.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#ifdef HAVE_LIBRETRO
#include "citra_libretro/libretro_vk.h"
#endif

MICROPROFILE_DEFINE(Vulkan_WaitForWorker, "Vulkan", "Wait for worker", MP_RGB(255, 192, 192));
MICROPROFILE_DEFINE(Vulkan_Submit, "Vulkan", "Submit Exectution", MP_RGB(255, 192, 255));

namespace Vulkan {

namespace {

std::unique_ptr<MasterSemaphore> MakeMasterSemaphore(const Instance& instance) {
#ifdef HAVE_LIBRETRO
    return CreateLibRetroMasterSemaphore(instance);
#else
    if (instance.IsTimelineSemaphoreSupported()) {
        return std::make_unique<MasterSemaphoreTimeline>(instance);
    } else {
        return std::make_unique<MasterSemaphoreFence>(instance);
    }
#endif
}

} // Anonymous namespace

GpuTimeMeter::GpuTimeMeter(const Instance& instance_) : instance{instance_} {
    const vk::PhysicalDevice physical_device = instance.GetPhysicalDevice();
    const auto families = physical_device.getQueueFamilyProperties();
    const u32 family = instance.GetGraphicsQueueFamilyIndex();
    const u32 valid_bits = family < families.size() ? families[family].timestampValidBits : 0;
    if (valid_bits == 0) {
        return;
    }
    valid_mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
    tick_ns = physical_device.getProperties().limits.timestampPeriod;
    const vk::QueryPoolCreateInfo pool_info = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = SLOTS * 2,
    };
    try {
        pool = instance.GetDevice().createQueryPoolUnique(pool_info);
    } catch (const vk::SystemError& err) {
        LOG_WARNING(Render_Vulkan, "No timestamp queries, GPU time is not measured: {}",
                    err.what());
    }
}

GpuTimeMeter::~GpuTimeMeter() = default;

void GpuTimeMeter::Begin(vk::CommandBuffer cmdbuf) {
    slot_open = false;
    if (!pool || !enabled.load(std::memory_order_relaxed)) {
        return;
    }
    {
        // Slots are handed out and collected in the same order, so with one to spare the
        // next one is not among those still waiting
        std::scoped_lock lock{mutex};
        if (pending.size() >= SLOTS - 1) {
            return;
        }
    }
    open_slot = next_slot;
    next_slot = (next_slot + 1) % SLOTS;
    slot_open = true;
    cmdbuf.resetQueryPool(*pool, open_slot * 2, 2);
    cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *pool, open_slot * 2);
}

void GpuTimeMeter::End(vk::CommandBuffer cmdbuf, u64 tick) {
    if (!slot_open) {
        return;
    }
    slot_open = false;
    // At the bottom of the pipe: when what was recorded before it is done, not when the GPU
    // has read as far as this
    cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *pool, open_slot * 2 + 1);
    std::scoped_lock lock{mutex};
    pending.push_back(Pending{.slot = open_slot, .tick = tick});
}

std::chrono::nanoseconds GpuTimeMeter::Collect(u64 finished_tick) {
    u64 busy = 0;
    while (true) {
        Pending entry;
        {
            std::scoped_lock lock{mutex};
            if (pending.empty() || pending.front().tick > finished_tick) {
                break;
            }
            entry = pending.front();
        }
        // The entry stays in the queue while its slot is read, which keeps the slot from
        // being handed out again
        std::array<u64, 2> stamps{};
        const vk::Result result = instance.GetDevice().getQueryPoolResults(
            *pool, entry.slot * 2, 2, sizeof(stamps), stamps.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64);
        {
            std::scoped_lock lock{mutex};
            pending.pop_front();
        }
        if (result != vk::Result::eSuccess) {
            continue;
        }
        const u64 begin = stamps[0] & valid_mask;
        const u64 end = stamps[1] & valid_mask;
        if (end < last_end) {
            // Submissions end in the order they were made, so the GPU's counter started
            // again: some devices stop it when the GPU is powered down between frames that
            // come far apart
            last_end = 0;
        }
        // The GPU may have read the start of a submission while it still worked on the one
        // before
        const u64 start = std::max(begin, last_end);
        if (end > start) {
            busy += end - start;
        }
        last_end = std::max(last_end, end);
    }
    return std::chrono::nanoseconds{static_cast<s64>(static_cast<double>(busy) * tick_ns)};
}

GpuSpanMeter::GpuSpanMeter(const Instance& instance_) : instance{instance_} {
    const vk::PhysicalDevice physical_device = instance.GetPhysicalDevice();
    const auto families = physical_device.getQueueFamilyProperties();
    const u32 family = instance.GetGraphicsQueueFamilyIndex();
    const u32 valid_bits = family < families.size() ? families[family].timestampValidBits : 0;
    if (valid_bits == 0) {
        return;
    }
    valid_mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
    tick_ns = physical_device.getProperties().limits.timestampPeriod;
    const vk::QueryPoolCreateInfo pool_info = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = SLOTS * 2,
    };
    try {
        pool = instance.GetDevice().createQueryPoolUnique(pool_info);
    } catch (const vk::SystemError&) {
        // Nothing is measured then
    }
}

GpuSpanMeter::~GpuSpanMeter() = default;

void GpuSpanMeter::Begin(vk::CommandBuffer cmdbuf) {
    is_open = false;
    if (!pool) {
        return;
    }
    {
        std::scoped_lock lock{mutex};
        if (pending.size() >= SLOTS - 1) {
            return;
        }
    }
    open_slot = next_slot;
    next_slot = (next_slot + 1) % SLOTS;
    is_open = true;
    cmdbuf.resetQueryPool(*pool, open_slot * 2, 2);
    // (At the bottom of the pipe: what came before is done then)
    cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *pool, open_slot * 2);
}

void GpuSpanMeter::End(vk::CommandBuffer cmdbuf, std::size_t kind) {
    if (!is_open) {
        return;
    }
    is_open = false;
    cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *pool, open_slot * 2 + 1);
    std::scoped_lock lock{mutex};
    pending.push_back(Pending{.slot = open_slot, .kind = kind});
}

std::array<GpuSpanMeter::Sum, GpuSpanMeter::KINDS> GpuSpanMeter::Take() {
    std::array<Sum, KINDS> sums{};
    while (pool) {
        Pending entry;
        {
            std::scoped_lock lock{mutex};
            if (pending.empty()) {
                break;
            }
            entry = pending.front();
        }
        std::array<u64, 4> data{}; // two times, each with whether it is there
        const vk::Result result = instance.GetDevice().getQueryPoolResults(
            *pool, entry.slot * 2, 2, sizeof(data), data.data(), sizeof(u64) * 2,
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if (result != vk::Result::eSuccess || data[1] == 0 || data[3] == 0 ||
            data[0] == last_begin[entry.slot]) {
            break;
        }
        last_begin[entry.slot] = data[0];
        {
            std::scoped_lock lock{mutex};
            pending.pop_front();
        }
        const u64 begin = data[0] & valid_mask;
        const u64 end = data[2] & valid_mask;
        if (end > begin) {
            sums[entry.kind].time += std::chrono::nanoseconds{
                static_cast<s64>(static_cast<double>(end - begin) * tick_ns)};
        }
        sums[entry.kind].spans++;
    }
    return sums;
}

GpuPassMeter::GpuPassMeter(const Instance& instance_) : instance{instance_} {
    const vk::PhysicalDevice physical_device = instance.GetPhysicalDevice();
    const auto families = physical_device.getQueueFamilyProperties();
    const u32 family = instance.GetGraphicsQueueFamilyIndex();
    const u32 valid_bits = family < families.size() ? families[family].timestampValidBits : 0;
    if (valid_bits == 0) {
        return;
    }
    valid_mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
    tick_ns = physical_device.getProperties().limits.timestampPeriod;
    const vk::QueryPoolCreateInfo pool_info = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = SLOTS * 2,
    };
    try {
        pool = instance.GetDevice().createQueryPoolUnique(pool_info);
    } catch (const vk::SystemError&) {
        // Nothing is measured then
    }
}

GpuPassMeter::~GpuPassMeter() = default;

void GpuPassMeter::Begin(vk::CommandBuffer cmdbuf) {
    is_open = false;
    if (!pool) {
        return;
    }
    {
        std::scoped_lock lock{mutex};
        if (pending.size() >= SLOTS - 1) {
            return;
        }
    }
    open_slot = next_slot;
    next_slot = (next_slot + 1) % SLOTS;
    is_open = true;
    cmdbuf.resetQueryPool(*pool, open_slot * 2, 2);
    cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *pool, open_slot * 2);
}

void GpuPassMeter::End(vk::CommandBuffer cmdbuf, const Target& target, u32 draws) {
    if (!is_open) {
        return;
    }
    is_open = false;
    cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *pool, open_slot * 2 + 1);
    std::scoped_lock lock{mutex};
    pending.push_back(Pending{.slot = open_slot, .target = target, .draws = draws});
}

std::vector<GpuPassMeter::Sum> GpuPassMeter::Take() {
    std::vector<Sum> sums;
    while (pool) {
        Pending entry;
        {
            std::scoped_lock lock{mutex};
            if (pending.empty()) {
                break;
            }
            entry = pending.front();
        }
        std::array<u64, 4> data{};
        const vk::Result result = instance.GetDevice().getQueryPoolResults(
            *pool, entry.slot * 2, 2, sizeof(data), data.data(), sizeof(u64) * 2,
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if (result != vk::Result::eSuccess || data[1] == 0 || data[3] == 0 ||
            data[0] == last_begin[entry.slot]) {
            break;
        }
        last_begin[entry.slot] = data[0];
        {
            std::scoped_lock lock{mutex};
            pending.pop_front();
        }
        const u64 begin = data[0] & valid_mask;
        const u64 end = data[2] & valid_mask;
        auto sum = std::ranges::find_if(sums, [&](const Sum& one) {
            return one.target.framebuffer == entry.target.framebuffer &&
                   one.target.loads == entry.target.loads &&
                   one.target.width == entry.target.width &&
                   one.target.height == entry.target.height;
        });
        if (sum == sums.end()) {
            sums.push_back({.target = entry.target});
            sum = sums.end() - 1;
        }
        if (end > begin) {
            sum->time += std::chrono::nanoseconds{
                static_cast<s64>(static_cast<double>(end - begin) * tick_ns)};
        }
        sum->passes++;
        sum->draws += entry.draws;
    }
    std::ranges::sort(sums, [](const Sum& a, const Sum& b) { return a.time > b.time; });
    return sums;
}

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Instance& instance)
    : master_semaphore{MakeMasterSemaphore(instance)},
      command_pool{instance, master_semaphore.get()}, gpu_time{instance}, use_worker_thread{true} {
    AllocateWorkerCommandBuffers();
    if (use_worker_thread) {
        AcquireNewChunk();
        worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
    }
}

Scheduler::~Scheduler() = default;

void Scheduler::Flush(vk::Semaphore signal, vk::Semaphore wait) {
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    SubmitExecution(signal, wait);
}

void Scheduler::Finish(vk::Semaphore signal, vk::Semaphore wait) {
    CITRA_TRACE_SCOPE("Vulkan::Finish");
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitExecution(signal, wait);
    Wait(presubmit_tick);
}

void Scheduler::WaitWorker() {
    if (!use_worker_thread) {
        return;
    }

    MICROPROFILE_SCOPE(Vulkan_WaitForWorker);
    CITRA_TRACE_SCOPE("Vulkan::WaitWorker");
    CITRA_HITCH_SCOPE(RenderThread);
    DispatchWork();

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    // This needs to be done in the same order as WorkerThread.
    std::scoped_lock el{execution_mutex};
}

void Scheduler::Wait(u64 tick) {
    if (tick >= master_semaphore->CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        Flush();
    }
    master_semaphore->Wait(tick);
}

#ifdef CITRA_ENABLE_TRACE_MARKERS
void Scheduler::TimeGpuWork(const std::string& what) {
    if (is_submitting) {
        // Called while a submission is put together: its time goes to whoever asked for it
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    Finish();
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    CITRA_FRAME_DUMP("  gpu {:6.2f} ms  {}", elapsed.count(), what);
}
#endif

std::optional<std::chrono::nanoseconds> Scheduler::TakeGpuTime(bool wanted) {
    gpu_time.SetEnabled(wanted);
    if (!gpu_time.IsAvailable()) {
        return std::nullopt;
    }
    // Also when nobody wants it any more: what is still under way has to leave the queue
    master_semaphore->Refresh();
    const std::chrono::nanoseconds busy = gpu_time.Collect(master_semaphore->KnownGpuTick());
    return wanted ? std::optional{busy} : std::nullopt;
}

void Scheduler::DispatchWork() {
    if (!use_worker_thread || chunk->Empty()) {
        return;
    }

    on_dispatch();

    {
        std::scoped_lock ql{queue_mutex};
        work_queue.push(std::move(chunk));
    }

    event_cv.notify_all();
    AcquireNewChunk();
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            Common::CondvarWait(event_cv, lk, stop_token, [&] { return TryPopQueue(work); });

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            std::exchange(lk, std::unique_lock{execution_mutex});

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            {
                CITRA_TRACE_SCOPE("Vulkan::ExecuteChunk");
                work->ExecuteAll(current_cmdbuf);
            }

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffers();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    current_cmdbuf.begin(begin_info);
    gpu_time.Begin(current_cmdbuf);
}

void Scheduler::SubmitExecution(vk::Semaphore signal_semaphore, vk::Semaphore wait_semaphore) {
    state = StateFlags::AllDirty;
    const u64 signal_value = master_semaphore->NextTick();

    is_submitting = true;
    on_submit();
    is_submitting = false;
    CITRA_TRACE_COUNT(Submits);
    CITRA_FRAME_DUMP("vk submit");

    Record([signal_semaphore, wait_semaphore, signal_value, this](vk::CommandBuffer cmdbuf) {
        MICROPROFILE_SCOPE(Vulkan_Submit);
        CITRA_TRACE_SCOPE("vkQueueSubmit");
        std::scoped_lock lock{submit_mutex};
        gpu_time.End(cmdbuf, signal_value);
        master_semaphore->SubmitWork(cmdbuf, wait_semaphore, signal_semaphore, signal_value);
    });

    master_semaphore->Refresh();

    if (!use_worker_thread) {
        AllocateWorkerCommandBuffers();
    } else {
        chunk->MarkSubmit();
        DispatchWork();
    }
    recorded_since_submit = false;
}

void Scheduler::AcquireNewChunk() {
    std::scoped_lock lock{reserve_mutex};
    if (chunk_reserve.empty()) {
        chunk = std::make_unique<CommandChunk>();
        return;
    }

    chunk = std::move(chunk_reserve.back());
    chunk_reserve.pop_back();
}

} // namespace Vulkan
