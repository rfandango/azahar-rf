// Copyright 2014-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>

#include <fmt/format.h>

#ifdef _WIN32
#include <share.h>   // For _SH_DENYWR
#include <windows.h> // For OutputDebugStringW
#else
#define _SH_DENYWR 0
#endif

#ifdef CITRA_LINUX_GCC_BACKTRACE
#define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>
#undef BOOST_STACKTRACE_USE_BACKTRACE
#include <signal.h>
#endif

#include "common/bounded_threadsafe_queue.h"
#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/literals.h"
#include "common/logging/backend.h"
#include "common/logging/log.h"
#include "common/logging/log_entry.h"
#include "common/logging/text_formatter.h"
#include "common/polyfill_thread.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "common/thread.h"
#include "common/trace_markers.h"

namespace Common::Log {

namespace {

/**
 * Interface for logging backends.
 */
class Backend {
public:
    virtual ~Backend() = default;

    virtual void Write(const Entry& entry) = 0;

    virtual void EnableForStacktrace() = 0;

    virtual void Flush() = 0;

    virtual void Close() = 0;
};

#ifdef HAVE_LIBRETRO
/**
 * LibRetro backend
 */
class LibRetroBackend : public Backend {
public:
    explicit LibRetroBackend() {}
    explicit LibRetroBackend(retro_log_printf_t callback) : callback(callback) {}

    ~LibRetroBackend() override = default;

    void Write(const Entry& entry) override {
        if (callback == nullptr) {
            return;
        }
        retro_log_level log_level;

        switch (entry.log_level) {
        case Common::Log::Level::Trace:
            log_level = retro_log_level::RETRO_LOG_DEBUG;
            break;
        case Common::Log::Level::Debug:
            log_level = retro_log_level::RETRO_LOG_DEBUG;
            break;
        case Common::Log::Level::Info:
            log_level = retro_log_level::RETRO_LOG_INFO;
            break;
        case Common::Log::Level::Warning:
            log_level = retro_log_level::RETRO_LOG_WARN;
            break;
        case Common::Log::Level::Error:
            log_level = retro_log_level::RETRO_LOG_ERROR;
            break;
        case Common::Log::Level::Critical:
            log_level = retro_log_level::RETRO_LOG_ERROR;
            break;
        default:
            log_level = retro_log_level::RETRO_LOG_DUMMY;
        }

        auto str = FormatLogMessage(entry).append(1, '\n');
        callback(log_level, str.c_str());
    }

    void Flush() override {}

    void Close() override {}

    void EnableForStacktrace() override {}

private:
    retro_log_printf_t callback = nullptr;
};
#endif

/**
 * Backend that writes to stderr and with color
 */
class ColorConsoleBackend final : public Backend {
public:
    explicit ColorConsoleBackend() = default;

    ~ColorConsoleBackend() override = default;

    void Write(const Entry& entry) override {
        if (enabled.load(std::memory_order_relaxed)) {
            PrintColoredMessage(entry);
        }
    }

    void Flush() override {
        std::fflush(stderr);
    }

    void Close() override {
        enabled = false;
    }

    void EnableForStacktrace() override {
        enabled = true;
    }

    void SetEnabled(bool enabled_) {
        enabled = enabled_;
    }

private:
    std::atomic_bool enabled{false};
};

/**
 * Backend that writes to a file passed into the constructor
 */
class FileBackend final : public Backend {
public:
    explicit FileBackend(const std::string& filename) {
        auto old_filename = filename;
        boost::replace_all(old_filename, ".txt", ".old.txt");

        // Existence checks are done within the functions themselves.
        // We don't particularly care if these succeed or not.
        static_cast<void>(FileUtil::Delete(old_filename));
        static_cast<void>(FileUtil::Rename(filename, old_filename));

        // _SH_DENYWR allows read only access to the file for other programs.
        // It is #defined to 0 on other platforms
#ifdef CITRA_ENABLE_TRACE_MARKERS
        writer = std::make_shared<Writer>();
        writer->file = std::make_unique<FileUtil::IOFile>(filename, "w", _SH_DENYWR);
        writer_thread = std::thread([state = writer] { WriterLoop(*state); });
#else
        file = std::make_unique<FileUtil::IOFile>(filename, "w", _SH_DENYWR);
#endif
    }

#ifdef CITRA_ENABLE_TRACE_MARKERS
    ~FileBackend() override {
        StopWriter();
    }
#else
    ~FileBackend() override = default;
#endif

    void Write(const Entry& entry) override {
        if (!enabled) {
            return;
        }

        // A message can carry a zero byte (a fixed-size name of the guest's, printed whole),
        // which makes the file something other than text to the programs it is shared with
        std::string message = FormatLogMessage(entry);
        std::erase(message, '\0');
        message.append(1, '\n');

        using namespace Common::Literals;
        // Prevent logs from exceeding a set maximum size in the event that log entries are spammed.
        const auto write_limit = 100_MiB;
#ifdef CITRA_ENABLE_TRACE_MARKERS
        // The log of a profiling build is what people send in, and it is written where the
        // user can get at it: shared storage, which can hold a write up for seconds on end
        // (a Retroid Pocket 5 let a burst of lines through at 12 KiB a second for 20 s, each
        // block of 4 KiB kept waiting longer than the one before). Written from the thread
        // that takes the lines from the queue, that kept every line behind it from the system
        // log as well, and once the queue was full every thread that logs would have waited
        // with it, the emulation's among them. So the lines are handed to a thread of the
        // file's own, which waits for the storage by itself. The file is flushed every few
        // seconds, at once after an error, and when asked (see Flush): it is never far behind
        // what the storage lets through.
        bytes_written += message.size();
        {
            std::scoped_lock lock{writer->mutex};
            if (writer->pending.size() + message.size() > MOST_PENDING) {
                writer->dropped_lines++;
            } else {
                writer->pending += message;
                writer->appended += message.size();
            }
            writer->flush_wanted = writer->flush_wanted || entry.log_level >= Level::Error;
        }
        writer->wake.notify_one();
        if (bytes_written > write_limit) {
            // Stop writing after the write limit is exceeded.
            enabled = false;
        }
#else
        bytes_written += file->WriteString(message);
        const bool write_limit_exceeded = bytes_written > write_limit;
        if (entry.log_level >= Level::Error || write_limit_exceeded) {
            if (write_limit_exceeded) {
                // Stop writing after the write limit is exceeded.
                // Don't close the file so we can print a stacktrace if necessary
                enabled = false;
            }
            file->Flush();
        }
#endif
    }

    void Flush() override {
#ifdef CITRA_ENABLE_TRACE_MARKERS
        // Until what was handed over so far is in the file, but not for longer than the
        // caller can stand still: the storage may be holding the writes up
        std::unique_lock lock{writer->mutex};
        if (writer->stopped) {
            return;
        }
        const u64 target = writer->appended;
        writer->flush_wanted = true;
        writer->wake.notify_one();
        writer->drained.wait_for(
            lock, FLUSH_WAIT, [&] { return writer->flushed_through >= target || writer->stopped; });
#else
        file->Flush();
#endif
    }

    void Close() override {
#ifdef CITRA_ENABLE_TRACE_MARKERS
        StopWriter();
#else
        file->Close();
#endif
        enabled = false;
    }

    void EnableForStacktrace() override {
        enabled = true;
        bytes_written = 0;
    }

private:
#ifdef CITRA_ENABLE_TRACE_MARKERS
    /// What the backend and the thread that writes the file share. Owned by both, so that a
    /// writer the storage does not let go of can be left behind when the backend goes.
    struct Writer {
        std::unique_ptr<FileUtil::IOFile> file;
        std::mutex mutex;
        std::condition_variable wake;
        std::condition_variable drained;
        /// Lines not yet written, and how many were not kept because too much was waiting
        std::string pending;
        u64 dropped_lines = 0;
        /// Bytes handed over so far, and how many of them are in the file and flushed
        u64 appended = 0;
        u64 flushed_through = 0;
        bool flush_wanted = false;
        bool stop = false;
        bool stopped = false;
    };
    /// What may wait to be written; lines beyond it are counted and left out
    static constexpr std::size_t MOST_PENDING = 32 * 1024 * 1024;
    /// How long a flush or the end waits for the storage
    static constexpr std::chrono::milliseconds FLUSH_WAIT{2500};

    static void WriterLoop(Writer& state) {
        using Clock = std::chrono::steady_clock;
        Common::SetCurrentThreadName("LogFile");
        std::string chunk;
        Clock::time_point last_flush = Clock::now();
        Clock::time_point last_report{};
        bool unflushed = false;
        std::unique_lock lock{state.mutex};
        while (true) {
            state.wake.wait_for(lock, std::chrono::seconds{1}, [&] {
                return state.stop || !state.pending.empty() || state.flush_wanted;
            });
            const bool stopping = state.stop;
            chunk.clear();
            chunk.swap(state.pending);
            const u64 through = state.appended;
            const u64 dropped = std::exchange(state.dropped_lines, 0);
            const bool flush_now = state.flush_wanted || stopping;
            state.flush_wanted = false;
            lock.unlock();

            const Clock::time_point before = Clock::now();
            if (dropped != 0) {
                chunk += fmt::format("[the log: {} lines were not kept while the storage held "
                                     "its writes up]\n",
                                     dropped);
            }
            if (!chunk.empty()) {
                if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::SlowLogStorage) !=
                    0) [[unlikely]] {
                    std::this_thread::sleep_for(std::chrono::milliseconds{
                        std::min<std::size_t>(chunk.size() / 12, 20'000)});
                }
                state.file->WriteString(chunk);
                unflushed = true;
            }
            if (unflushed && (flush_now || before - last_flush >= std::chrono::seconds{5})) {
                state.file->Flush();
                last_flush = Clock::now();
                unflushed = false;
            }
            const Clock::time_point after = Clock::now();
            const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(after - before);

            lock.lock();
            if (!unflushed) {
                state.flushed_through = through;
                state.drained.notify_all();
            }
            const std::size_t waiting = state.pending.size();
            if (took >= std::chrono::milliseconds{500} &&
                after - last_report >= std::chrono::seconds{30} && !stopping) {
                // (Said outside of the lock: the line comes back here through the queue)
                last_report = after;
                lock.unlock();
                LOG_WARNING(Log,
                            "The storage took {} ms to write {} KiB of the log; {} KiB more "
                            "waited meanwhile",
                            took.count(), chunk.size() / 1024, waiting / 1024);
                lock.lock();
            }
            if (stopping && state.pending.empty()) {
                break;
            }
        }
        state.stopped = true;
        state.drained.notify_all();
    }

    void StopWriter() {
        if (!writer_thread.joinable()) {
            return;
        }
        bool done;
        {
            std::unique_lock lock{writer->mutex};
            writer->stop = true;
            writer->wake.notify_one();
            done = writer->drained.wait_for(lock, FLUSH_WAIT, [&] { return writer->stopped; });
        }
        if (done) {
            writer_thread.join();
            writer->file->Close();
        } else {
            // The storage holds it: it finishes by itself, or with the process
            writer_thread.detach();
        }
    }

    std::shared_ptr<Writer> writer;
    std::thread writer_thread;
#else
    std::unique_ptr<FileUtil::IOFile> file;
#endif
    bool enabled = true;
    std::size_t bytes_written = 0;
};

/**
 * Backend that writes to Visual Studio's output window
 */
class DebuggerBackend final : public Backend {
public:
    explicit DebuggerBackend() = default;

    ~DebuggerBackend() override = default;

    void Write(const Entry& entry) override {
#ifdef _WIN32
        ::OutputDebugStringW(UTF8ToUTF16W(FormatLogMessage(entry).append(1, '\n')).c_str());
#endif
    }

    void Flush() override {}

    void Close() override {}

    void EnableForStacktrace() override {}
};

#ifdef ANDROID
/**
 * Backend that writes to the Android logcat
 */
class LogcatBackend : public Backend {
public:
    explicit LogcatBackend() = default;

    ~LogcatBackend() override = default;

    void Write(const Entry& entry) override {
        PrintMessageToLogcat(entry);
    }

    void Flush() override {}

    void Close() override {}

    void EnableForStacktrace() override {}
};
#endif

bool initialization_in_progress_suppress_logging = true;
bool logging_initialized = false;

#ifdef CITRA_LINUX_GCC_BACKTRACE
[[noreturn]] void SleepForever() {
    while (true) {
        pause();
    }
}
#endif

/**
 * Static state as a singleton.
 */
class Impl {
public:
    static Impl& Instance() {
        if (!instance) {
            throw std::runtime_error("Using Logging instance before its initialization");
        }
        return *instance;
    }
#ifdef HAVE_LIBRETRO
    static void Initialize(retro_log_printf_t callback) {
        if (instance) {
            LOG_WARNING(Log, "Reinitializing logging backend");
            return;
        }
        initialization_in_progress_suppress_logging = true;
        Filter filter;
        filter.ParseFilterString(Settings::values.log_filter.GetValue());
        instance = std::unique_ptr<Impl, decltype(&Deleter)>(new Impl(callback, filter), Deleter);
        initialization_in_progress_suppress_logging = false;
        logging_initialized = true;
    }
#endif
    static void Initialize(std::string_view log_file) {
        if (instance) {
            LOG_WARNING(Log, "Reinitializing logging backend");
            return;
        }
        initialization_in_progress_suppress_logging = true;
        const auto& log_dir = FileUtil::GetUserPath(FileUtil::UserPath::LogDir);
        void(FileUtil::CreateFullPath(log_dir));
        Filter filter;
        filter.ParseFilterString(Settings::values.log_filter.GetValue());
        instance = std::unique_ptr<Impl, decltype(&Deleter)>(
            new Impl(fmt::format("{}{}", log_dir, log_file), filter), Deleter);
        initialization_in_progress_suppress_logging = false;
        logging_initialized = true;
    }

    static void Start() {
        instance->StartBackendThread();
    }

    static void Stop() {
        instance->StopBackendThread();
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    void SetGlobalFilter(const Filter& f) {
        filter = f;
    }

    /// What was logged so far goes to the file, as far as that can be had in a few seconds:
    /// first what is still in the queue to the backends, then what the file backend holds
    /// back. From any thread but the backend's own.
    void FlushFile() {
#ifndef HAVE_LIBRETRO
        const u64 target = pushed.load(std::memory_order_relaxed);
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::milliseconds{1500};
        while (popped.load(std::memory_order_relaxed) < target &&
               std::chrono::steady_clock::now() < give_up) {
            std::this_thread::sleep_for(std::chrono::milliseconds{2});
        }
        file_backend.Flush();
#endif
    }

    bool SetRegexFilter(const std::string& regex) {
        if (regex.empty()) {
            regex_filter = boost::regex();
            return true;
        }
        regex_filter = boost::regex(regex, boost::regex_constants::no_except);
        if (regex_filter.status() != 0) {
            regex_filter = boost::regex();
            return false;
        }
        return true;
    }

    const Filter& GetFilter() const {
        return filter;
    }

    void SetColorConsoleBackendEnabled(bool enabled) {
        color_console_backend.SetEnabled(enabled);
    }

    void PushEntry(Class log_class, Level log_level, const char* filename, unsigned int line_num,
                   const char* function, std::string message) {
        Entry new_entry = CreateEntry(log_class, log_level, filename, line_num, function,
                                      std::move(message), time_origin);
        if (!regex_filter.empty() &&
            !boost::regex_search(FormatLogMessage(new_entry), regex_filter)) {
            return;
        }
        if (Settings::values.instant_debug_log.GetValue()) {
            ForEachBackend([&new_entry](Backend& backend) {
                backend.Write(new_entry);
                backend.Flush();
            });
        } else {
            message_queue.EmplaceWait(new_entry);
            pushed.fetch_add(1, std::memory_order_relaxed);
        }
    }

    static Entry CreateEntry(Class log_class, Level log_level, const char* filename,
                             unsigned int line_nr, const char* function, std::string&& message,
                             const std::chrono::steady_clock::time_point& time_origin) {
        using std::chrono::duration_cast;
        using std::chrono::microseconds;
        using std::chrono::steady_clock;

        return {
            .timestamp = duration_cast<microseconds>(steady_clock::now() - time_origin),
            .log_class = log_class,
            .log_level = log_level,
            .filename = filename,
            .line_num = line_nr,
            .function = function,
            .message = std::move(message),
        };
    }

private:
#ifdef HAVE_LIBRETRO
    Impl(retro_log_printf_t callback, const Filter& filter_)
        : filter{filter_}, file_backend{""}, libretro_backend{callback} {}
#endif
    Impl(const std::string& file_backend_filename, const Filter& filter_)
        : filter{filter_}, file_backend{file_backend_filename} {
#ifdef CITRA_LINUX_GCC_BACKTRACE
        int waker_pipefd[2];
        int done_printing_pipefd[2];
        if (pipe2(waker_pipefd, O_CLOEXEC) || pipe2(done_printing_pipefd, O_CLOEXEC)) {
            abort();
        }
        backtrace_thread_waker_fd = waker_pipefd[1];
        backtrace_done_printing_fd = done_printing_pipefd[0];
        std::thread([this, wait_fd = waker_pipefd[0], done_fd = done_printing_pipefd[1]] {
            Common::SetCurrentThreadName("citra:Crash");
            for (u8 ignore = 0; read(wait_fd, &ignore, 1) != 1;)
                ;
            const int sig = received_signal;
            if (sig <= 0) {
                abort();
            }
            backend_thread.request_stop();
            backend_thread.join();
            const auto signal_entry = CreateEntry(
                Class::Log, Level::Critical, "?", 0, "?",
                fmt::vformat("Received signal {}", fmt::make_format_args(sig)), time_origin);
            ForEachBackend([&signal_entry](Backend& backend) {
                backend.EnableForStacktrace();
                backend.Write(signal_entry);
            });
            const auto backtrace =
                boost::stacktrace::stacktrace::from_dump(backtrace_storage.data(), 4096);
            for (const auto& frame : backtrace.as_vector()) {
                auto line = boost::stacktrace::detail::to_string(&frame, 1);
                if (line.empty()) {
                    abort();
                }
                line.pop_back(); // Remove newline
                const auto frame_entry = CreateEntry(Class::Log, Level::Critical, "?", 0, "?",
                                                     std::move(line), time_origin);
                ForEachBackend([&frame_entry](Backend& backend) { backend.Write(frame_entry); });
            }
            using namespace std::literals;
            const auto rip_entry =
                CreateEntry(Class::Log, Level::Critical, "?", 0, "?", "RIP"s, time_origin);
            ForEachBackend([&rip_entry](Backend& backend) {
                backend.Write(rip_entry);
                backend.Flush();
            });
            for (const u8 anything = 0; write(done_fd, &anything, 1) != 1;)
                ;
            // Abort on original thread to help debugging
            SleepForever();
        }).detach();
        signal(SIGSEGV, &HandleSignal);
        signal(SIGABRT, &HandleSignal);
#endif
    }

    ~Impl() {
#ifdef CITRA_LINUX_GCC_BACKTRACE
        if (int zero_or_ignore = 0;
            !received_signal.compare_exchange_strong(zero_or_ignore, SIGKILL)) {
            SleepForever();
        }
#endif
    }

    void StartBackendThread() {
        backend_thread = std::jthread([this](std::stop_token stop_token) {
            Common::SetCurrentThreadName("citra:Log");
            Entry entry;
            const auto write_logs = [this, &entry]() {
                ForEachBackend([&entry](Backend& backend) { backend.Write(entry); });
            };
            while (!stop_token.stop_requested()) {
                message_queue.PopWait(entry, stop_token);
                // Only write the log if something was actually popped (entry.filename != nullptr)
                // (for example, when the stop token is signaled).
                if (entry.filename != nullptr) {
                    write_logs();
                    popped.fetch_add(1, std::memory_order_relaxed);
                }
            }
            // Drain the logging queue. Only writes out up to MAX_LOGS_TO_WRITE to prevent a
            // case where a system is repeatedly spamming logs even on close.
#ifdef CITRA_ENABLE_TRACE_MARKERS
            // (All of it in a build whose log is what people send in)
            int max_logs_to_write = INT_MAX;
#else
            int max_logs_to_write = filter.IsDebug() ? INT_MAX : 100;
#endif
            while (max_logs_to_write-- && message_queue.TryPop(entry)) {
                write_logs();
                popped.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    void StopBackendThread() {
        backend_thread.request_stop();
        if (backend_thread.joinable()) {
            backend_thread.join();
        }

        ForEachBackend([](Backend& backend) {
            backend.Flush();
            backend.Close();
        });
    }

    void ForEachBackend(auto lambda) {
#ifdef HAVE_LIBRETRO
        lambda(static_cast<Backend&>(libretro_backend));
#else
        lambda(static_cast<Backend&>(debugger_backend));
        lambda(static_cast<Backend&>(color_console_backend));
        lambda(static_cast<Backend&>(file_backend));
#ifdef ANDROID
        lambda(static_cast<Backend&>(lc_backend));
#endif // ANDROID
#endif // HAVE_LIBRETRO
    }

    static void Deleter(Impl* ptr) {
        delete ptr;
    }

#ifdef CITRA_LINUX_GCC_BACKTRACE
    [[noreturn]] static void HandleSignal(int sig) {
        signal(SIGABRT, SIG_DFL);
        signal(SIGSEGV, SIG_DFL);
        if (sig <= 0) {
            abort();
        }
        instance->InstanceHandleSignal(sig);
    }

    [[noreturn]] void InstanceHandleSignal(int sig) {
        if (int zero_or_ignore = 0; !received_signal.compare_exchange_strong(zero_or_ignore, sig)) {
            if (received_signal == SIGKILL) {
                abort();
            }
            SleepForever();
        }
        // Don't restart like boost suggests. We want to append to the log file and not lose dynamic
        // symbols. This may segfault if it unwinds outside C/C++ code but we'll just have to fall
        // back to core dumps.
        boost::stacktrace::safe_dump_to(backtrace_storage.data(), 4096);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (const int anything = 0; write(backtrace_thread_waker_fd, &anything, 1) != 1;)
            ;
        for (u8 ignore = 0; read(backtrace_done_printing_fd, &ignore, 1) != 1;)
            ;
        abort();
    }
#endif

    static inline std::unique_ptr<Impl, decltype(&Deleter)> instance{nullptr, Deleter};

    Filter filter;
    boost::regex regex_filter;
    DebuggerBackend debugger_backend{};
    ColorConsoleBackend color_console_backend{};
    FileBackend file_backend;
#ifdef ANDROID
    LogcatBackend lc_backend{};
#endif
#ifdef HAVE_LIBRETRO
    LibRetroBackend libretro_backend;
#endif

    MPSCQueue<Entry> message_queue{};
    /// Entries put into the queue and taken out of it and written, for FlushFile
    std::atomic<u64> pushed{0};
    std::atomic<u64> popped{0};
    std::chrono::steady_clock::time_point time_origin{std::chrono::steady_clock::now()};
    std::jthread backend_thread;

#ifdef CITRA_LINUX_GCC_BACKTRACE
    std::atomic_int received_signal{0};
    std::array<u8, 4096> backtrace_storage{};
    int backtrace_thread_waker_fd;
    int backtrace_done_printing_fd;
#endif
};
} // namespace

#ifdef HAVE_LIBRETRO
void LibRetroStart(retro_log_printf_t callback) {
    Impl::Initialize(callback);
    Impl::Start();
}
#endif

void Initialize(std::string_view log_file) {
    Impl::Initialize(log_file.empty() ? LOG_FILE : log_file);
}

void Start() {
    Impl::Start();
}

void Stop() {
    if (logging_initialized) {
        Impl::Stop();
    }
}

void DisableLoggingInTests() {
    initialization_in_progress_suppress_logging = true;
}

void Flush() {
    if (logging_initialized) {
        Impl::Instance().FlushFile();
    }
}

void SetGlobalFilter(const Filter& filter) {
    Impl::Instance().SetGlobalFilter(filter);
}

bool SetRegexFilter(const std::string& regex) {
    return Impl::Instance().SetRegexFilter(regex);
}

void SetColorConsoleBackendEnabled(bool enabled) {
    Impl::Instance().SetColorConsoleBackendEnabled(enabled);
}

void FmtLogMessageImpl(Class log_class, Level log_level, const char* filename,
                       unsigned int line_num, const char* function, fmt::string_view format,
                       const fmt::format_args& args) {
    if (initialization_in_progress_suppress_logging && log_level < Level::Critical) [[unlikely]] {
        return;
    }

    if (logging_initialized) [[likely]] {
        if (!Impl::Instance().GetFilter().CheckMessage(log_class, log_level)) {
            return;
        }
        Impl::Instance().PushEntry(log_class, log_level, filename, line_num, function,
                                   fmt::vformat(format, args));
    } else {
        // In the rare case that logging occurs before initialization, write the
        // message to stderr to preserve useful debug information.
        Entry new_entry = Impl::CreateEntry(log_class, log_level, filename, line_num, function,
                                            fmt::vformat(format, args), {});
        PrintMessage(new_entry);
    }
}
} // namespace Common::Log
