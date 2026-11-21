// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include "common/alignment.h"
#include "common/common_funcs.h"
#include "common/polyfill_thread.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace Vulkan {

enum class StateFlags {
    AllDirty = 0,
    Pipeline = 1 << 0,
    DescriptorSets = 1 << 1,
};
DECLARE_ENUM_FLAG_OPERATORS(StateFlags)

class Instance;

/// Measures how long the GPU works on what is sent to it. A timestamp is written at the start
/// and at the end of every submission, and once the GPU is done with them the stretches
/// between the two are added up, each from where the one before ended. The timestamps are
/// only compared within a submission and with the one before: the counter behind them need
/// not keep running while the GPU has nothing to do.
class GpuTimeMeter {
public:
    explicit GpuTimeMeter(const Instance& instance);
    ~GpuTimeMeter();

    /// Whether the device can write timestamps at all
    [[nodiscard]] bool IsAvailable() const noexcept {
        return static_cast<bool>(pool);
    }

    /// Measures the submissions begun from now on, or stops doing so. Any thread.
    void SetEnabled(bool enabled_) noexcept {
        enabled.store(enabled_, std::memory_order_relaxed);
    }

    /// At the start of a command buffer, on the thread that records it.
    void Begin(vk::CommandBuffer cmdbuf);

    /// At the end of that command buffer, which is submitted as `tick`.
    void End(vk::CommandBuffer cmdbuf, u64 tick);

    /// The time the GPU worked on the submissions up to `finished_tick` that were not taken
    /// in yet. The GPU must be done with them. One thread at a time.
    [[nodiscard]] std::chrono::nanoseconds Collect(u64 finished_tick);

private:
    /// Submissions that can be under way at a time. One that finds no room is not measured.
    static constexpr u32 SLOTS = 128;

    struct Pending {
        u32 slot;
        u64 tick;
    };

    const Instance& instance;
    vk::UniqueQueryPool pool;
    double tick_ns{};
    u64 valid_mask{};
    std::atomic_bool enabled{false};
    // Of the recording thread
    u32 next_slot{};
    u32 open_slot{};
    bool slot_open{};
    // Submissions recorded and not collected yet, oldest first
    std::mutex mutex;
    std::deque<Pending> pending;
    // Of the collecting thread: where the last submission ended, in the GPU's ticks
    u64 last_end{};
};

/// The scheduler abstracts command buffer and fence management with an interface that's able to do
/// OpenGL-like operations on Vulkan command buffers.
class Scheduler {
public:
    explicit Scheduler(const Instance& instance);
    ~Scheduler();

    /// Sends the current execution context to the GPU.
    void Flush(vk::Semaphore signal = nullptr, vk::Semaphore wait = nullptr);

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish(vk::Semaphore signal = nullptr, vk::Semaphore wait = nullptr);

    /// Waits for the worker thread to finish executing everything. After this function returns it's
    /// safe to touch worker resources.
    void WaitWorker();

    /// Waits for the given tick to trigger on the GPU.
    void Wait(u64 tick);

#ifdef CITRA_ENABLE_TRACE_MARKERS
    /// For frame dumps with GPU times: sends what was recorded so far to the GPU, waits until
    /// it is done and notes how long that took in the dump.
    void TimeGpuWork(const std::string& what);
#endif

    /// The time the GPU worked on what was sent to it since this was last called, as far as
    /// it is done with it. Nothing if the device cannot tell or `wanted` is false, which also
    /// stops the measuring. To be called regularly, by one thread.
    [[nodiscard]] std::optional<std::chrono::nanoseconds> TakeGpuTime(bool wanted);

    /// Sends currently recorded work to the worker thread.
    void DispatchWork();

    /// Records the command to the current chunk.
    template <typename T>
    void Record(T&& command) {
        if (chunk->Record(command)) {
            return;
        }
        DispatchWork();
        (void)chunk->Record(command);
    }

    /// Marks the provided state as non dirty
    void MarkStateNonDirty(StateFlags flag) noexcept {
        state |= flag;
    }

    /// Marks the provided state as dirty
    void MakeDirty(StateFlags flag) noexcept {
        state &= ~flag;
    }

    /// Returns true if the state is dirty
    [[nodiscard]] bool IsStateDirty(StateFlags flag) const noexcept {
        return False(state & flag);
    }

    /// Registers a callback to perform on queue submission.
    void RegisterOnSubmit(std::function<void()>&& func) {
        on_submit = std::move(func);
    }

    /// Registers a callback to perform on queue submission.
    void RegisterOnDispatch(std::function<void()>&& func) {
        on_dispatch = std::move(func);
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore->CurrentTick();
    }

    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return master_semaphore->IsFree(tick);
    }

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore* GetMasterSemaphore() noexcept {
        return master_semaphore.get();
    }

    std::mutex submit_mutex;
    /// Held while an image is handed to the display on a queue of its own
    std::mutex present_mutex;

private:
    class Command {
    public:
        virtual ~Command() = default;

        virtual void Execute(vk::CommandBuffer cmdbuf) const = 0;

        Command* GetNext() const {
            return next;
        }

        void SetNext(Command* next_) {
            next = next_;
        }

    private:
        Command* next = nullptr;
    };

    template <typename T>
    class TypedCommand final : public Command {
    public:
        explicit TypedCommand(T&& command_) : command{std::move(command_)} {}
        ~TypedCommand() override = default;

        TypedCommand(TypedCommand&&) = delete;
        TypedCommand& operator=(TypedCommand&&) = delete;

        void Execute(vk::CommandBuffer cmdbuf) const override {
            command(cmdbuf);
        }

    private:
        T command;
    };

    class CommandChunk final {
    public:
        void ExecuteAll(vk::CommandBuffer cmdbuf);

        template <typename T>
        bool Record(T& command) {
            using FuncType = TypedCommand<T>;
            static_assert(sizeof(FuncType) < sizeof(data), "Lambda is too large");

            recorded_counts++;
            command_offset = Common::AlignUp(command_offset, alignof(FuncType));
            if (command_offset > sizeof(data) - sizeof(FuncType)) {
                return false;
            }
            Command* const current_last = last;
            last = new (data.data() + command_offset) FuncType(std::move(command));

            if (current_last) {
                current_last->SetNext(last);
            } else {
                first = last;
            }
            command_offset += sizeof(FuncType);
            return true;
        }

        void MarkSubmit() {
            submit = true;
        }

        bool Empty() const {
            return recorded_counts == 0;
        }

        bool HasSubmit() const {
            return submit;
        }

    private:
        Command* first = nullptr;
        Command* last = nullptr;

        std::size_t recorded_counts = 0;
        std::size_t command_offset = 0;
        bool submit = false;
        alignas(std::max_align_t) std::array<u8, 0x8000> data{};
    };

private:
    void WorkerThread(std::stop_token stop_token);

    void AllocateWorkerCommandBuffers();

    void SubmitExecution(vk::Semaphore signal_semaphore, vk::Semaphore wait_semaphore);

    void AcquireNewChunk();

private:
    std::unique_ptr<MasterSemaphore> master_semaphore;
    CommandPool command_pool;
    GpuTimeMeter gpu_time;
    std::unique_ptr<CommandChunk> chunk;
    std::queue<std::unique_ptr<CommandChunk>> work_queue;
    std::vector<std::unique_ptr<CommandChunk>> chunk_reserve;
    vk::CommandBuffer current_cmdbuf;
    StateFlags state{};
    std::function<void()> on_submit;
    std::function<void()> on_dispatch;
    /// Set while a submission is being put together, when nothing else may be submitted
    bool is_submitting{};
    std::mutex execution_mutex;
    std::mutex reserve_mutex;
    std::mutex queue_mutex;
    std::condition_variable_any event_cv;
    std::jthread worker_thread;
    bool use_worker_thread;
};

} // namespace Vulkan
