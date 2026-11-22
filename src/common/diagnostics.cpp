// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include "common/diagnostics.h"

#ifdef ANDROID

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <dlfcn.h>
#include <fcntl.h>
#include <fmt/format.h>
#include <sys/mman.h>
#include <sys/system_properties.h>
#include <unistd.h>
#include "common/file_util.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "common/trace_markers.h"

namespace Common::Diagnostics {

namespace {

constexpr const char* CONTROL_FILE = "diag.txt";
constexpr const char* DRIVER_LOG = "driver_log.txt";
constexpr std::string_view FILES_PLACEHOLDER = "%FILES%";
constexpr std::string_view PROPERTY_PREFIX = "debug.azahar.";
constexpr std::string_view FRAME_DUMPS = "dump_frames_at";
constexpr const char* KGSL = "/sys/class/kgsl/kgsl-3d0/";

std::mutex mutex;
std::vector<std::pair<std::string, std::string>> driver_environment;
/// Whether the control file says when frames are dumped, which turns the automatic ones off
bool frame_dumps_set = false;

/// The comparison a profiling build runs by itself while a game is below its speed: stretches
/// as the build is, and stretches with one of its newer rules turned off or with the driver
/// told to render differently, each said in the log. Whoever reads the log then has, from one
/// session and nothing set by hand, what each rule is worth on that device in that scene.
enum class Phase : u32 {
    AsBuilt,
    OldDepthPath,
    ClockAtConsole,
    AllDirect,
    VertexDataCopied,
    Count,
};
constexpr std::array<const char*, static_cast<std::size_t>(Phase::Count)> PHASE_NAMES{
    "as built",
    "the depth buffer loaded as before",
    "the processor clock not below the console's",
    "the driver renders every pass directly",
    "vertex data copied for every draw",
};
constexpr std::chrono::seconds AS_BUILT_TIME{30};
constexpr std::chrono::seconds OTHER_PHASE_TIME{15};
/// As for the frame dumps: 120 VBlanks take 2.0 s at full speed, and five such stretches below
/// 95% of it running make a game that is slow
constexpr u32 SPEED_WINDOW_VBLANKS = 120;
constexpr std::chrono::milliseconds SLOW_WINDOW{2105};
constexpr u32 SLOW_WINDOWS = 5;

struct Comparison {
    bool off = false;
    Phase phase = Phase::AsBuilt;
    std::chrono::steady_clock::time_point phase_start{};
    std::chrono::steady_clock::time_point window_start{};
    u32 window_vblanks = 0;
    u32 slow_windows = 0;
    std::atomic<u32> bits{0};
};
Comparison comparison;

/// A page shared with the graphics driver, which our Turnip reads at every render pass: its
/// first word asks for all passes directly (1), all in tiles (2), or as the driver sees fit
volatile s32* control_page = nullptr;

/// Made at the first use: Property() is called while the library is being loaded, by what
/// reads a setting into a global
std::map<std::string, std::string, std::less<>>& Overrides() {
    static std::map<std::string, std::string, std::less<>> overrides;
    return overrides;
}
std::vector<u32> frame_dump_times;

constexpr std::size_t CONTROL_PAGE_SIZE = 4096;

/// The name of the app's package, which its folders are named by
std::string PackageName() {
    std::string name;
    if (std::FILE* file = std::fopen("/proc/self/cmdline", "r")) {
        char data[256]{};
        const std::size_t read = std::fread(data, 1, sizeof(data) - 1, file);
        std::fclose(file);
        name.assign(data, strnlen(data, read));
    }
    return name;
}

std::string_view Trim(std::string_view text) {
    const auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (!text.empty() && blank(text.front())) {
        text.remove_prefix(1);
    }
    while (!text.empty() && blank(text.back())) {
        text.remove_suffix(1);
    }
    return text;
}

/// The first line of a small file of the kernel, without its end; empty if it cannot be read
std::string ReadLine(const std::string& path) {
    std::string line;
    if (std::FILE* file = std::fopen(path.c_str(), "r")) {
        char data[512]{};
        if (std::fgets(data, sizeof(data), file)) {
            line = Trim(data);
        }
        std::fclose(file);
    }
    return line;
}

std::string SystemProperty(const char* name) {
    char value[PROP_VALUE_MAX]{};
    return __system_property_get(name, value) > 0 ? std::string{value} : std::string{};
}

/// A run of cores that share their highest clock
struct Cluster {
    int first;
    int last;
    long max_khz;
};

const std::vector<Cluster>& Clusters() {
    static const std::vector<Cluster> clusters = [] {
        std::vector<Cluster> found;
        const long count = sysconf(_SC_NPROCESSORS_CONF);
        for (int cpu = 0; cpu < count; cpu++) {
            const std::string value = ReadLine(
                fmt::format("/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq", cpu));
            const long khz = value.empty() ? 0 : std::atol(value.c_str());
            if (!found.empty() && found.back().max_khz == khz) {
                found.back().last = cpu;
            } else {
                found.push_back({cpu, cpu, khz});
            }
        }
        return found;
    }();
    return clusters;
}

/// The part numbers of the cores as /proc/cpuinfo has them, one for each core
std::vector<std::string> CoreParts() {
    std::vector<std::string> parts;
    if (std::FILE* file = std::fopen("/proc/cpuinfo", "r")) {
        char data[256]{};
        while (std::fgets(data, sizeof(data), file)) {
            const std::string_view line{data};
            if (line.starts_with("CPU part")) {
                const std::size_t colon = line.find(':');
                parts.emplace_back(colon == std::string_view::npos ? std::string_view{}
                                                                   : Trim(line.substr(colon + 1)));
            }
        }
        std::fclose(file);
    }
    return parts;
}

long MemInfoKb(std::string_view key) {
    long kb = 0;
    if (std::FILE* file = std::fopen("/proc/meminfo", "r")) {
        char data[256]{};
        while (std::fgets(data, sizeof(data), file)) {
            const std::string_view line{data};
            if (line.starts_with(key)) {
                kb = std::atol(data + key.size() + 1);
                break;
            }
        }
        std::fclose(file);
    }
    return kb;
}

/// The thermal state as the system grades it, 0 (none) to 6 (shutdown), or -1 where the system
/// has no such grade (before Android 11). The functions are looked up by name as the app also
/// runs on systems that do not have them.
int ThermalStatus() {
    using Acquire = void* (*)();
    using Status = int (*)(void*);
    static void* manager = nullptr;
    static Status status = nullptr;
    static bool looked = false;
    if (!looked) {
        looked = true;
        if (void* library = dlopen("libandroid.so", RTLD_NOW)) {
            const auto acquire =
                reinterpret_cast<Acquire>(dlsym(library, "AThermal_acquireManager"));
            status = reinterpret_cast<Status>(dlsym(library, "AThermal_getCurrentThermalStatus"));
            if (acquire && status) {
                manager = acquire();
            }
        }
    }
    return manager && status ? status(manager) : -1;
}

} // Anonymous namespace

std::string FilesDir() {
    static const std::string dir = [] {
        const std::string name = PackageName();
        if (name.empty()) {
            return std::string{};
        }
        std::string path = fmt::format("/sdcard/Android/data/{}/files", name);
        std::error_code ec;
        std::filesystem::create_directories(path, ec);
        return path;
    }();
    return dir;
}

void LoadControlFile() {
    const std::string path =
        FileUtil::GetUserPath(FileUtil::UserPath::UserDir) + std::string{CONTROL_FILE};
    std::string text;
    if (!FileUtil::Exists(path) || FileUtil::ReadFileToString(true, path, text) == 0) {
        LOG_INFO(Common, "Diagnostics: no {} in the user directory, nothing set from it",
                 CONTROL_FILE);
        return;
    }

    std::scoped_lock lock{mutex};
    auto& overrides = Overrides();
    overrides.clear();
    driver_environment.clear();
    frame_dump_times.clear();
    frame_dumps_set = false;
    std::string_view rest{text};
    while (!rest.empty()) {
        const std::size_t end = rest.find('\n');
        std::string_view line = Trim(rest.substr(0, end));
        rest.remove_prefix(end == std::string_view::npos ? rest.size() : end + 1);
        if (const std::size_t comment = line.find('#'); comment != std::string_view::npos) {
            line = Trim(line.substr(0, comment));
        }
        const std::size_t equals = line.find('=');
        if (line.empty() || equals == std::string_view::npos) {
            continue;
        }
        const std::string_view name = Trim(line.substr(0, equals));
        const std::string_view value = Trim(line.substr(equals + 1));
        if (name.empty()) {
            continue;
        }
        if (name == "comparison") {
            comparison.off = value == "off";
        } else if (name == FRAME_DUMPS) {
            frame_dumps_set = true;
            std::string_view times = value;
            while (!times.empty()) {
                const std::size_t comma = times.find(',');
                const std::string number{Trim(times.substr(0, comma))};
                if (!number.empty()) {
                    frame_dump_times.push_back(
                        static_cast<u32>(std::strtoul(number.c_str(), nullptr, 10)));
                }
                times.remove_prefix(comma == std::string_view::npos ? times.size() : comma + 1);
            }
            std::ranges::sort(frame_dump_times);
        } else if (name.starts_with(PROPERTY_PREFIX)) {
            overrides[std::string{name}] = std::string{value};
        } else {
            driver_environment.emplace_back(name, value);
        }
        LOG_INFO(Common, "Diagnostics: {} sets {} = {}", CONTROL_FILE, name, value);
    }
}

std::optional<std::string> Property(const char* name) {
    {
        std::scoped_lock lock{mutex};
        const auto& overrides = Overrides();
        if (const auto it = overrides.find(std::string_view{name}); it != overrides.end()) {
            return it->second;
        }
    }
    char value[PROP_VALUE_MAX]{};
    if (__system_property_get(name, value) > 0) {
        return std::string{value};
    }
    return std::nullopt;
}

namespace {

/// Puts what the driver writes to its log file into the emulator's log as it comes, so that
/// one file holds both, in the order things happened: the file a user shares.
void RelayDriverLog(std::string path) {
    Common::SetCurrentThreadName("DriverLogRelay");
    long offset = 0;
    std::string pending;
    std::vector<char> data(256 * 1024);
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds{1});
        std::FILE* file = std::fopen(path.c_str(), "r");
        if (!file) {
            continue;
        }
        std::fseek(file, 0, SEEK_END);
        const long size = std::ftell(file);
        if (size < offset) {
            // Started anew
            offset = 0;
            pending.clear();
        }
        std::size_t read = 0;
        if (size > offset && std::fseek(file, offset, SEEK_SET) == 0) {
            read = std::fread(
                data.data(), 1,
                std::min<std::size_t>(data.size(), static_cast<std::size_t>(size - offset)), file);
            offset += static_cast<long>(read);
        }
        std::fclose(file);

        pending.append(data.data(), read);
        std::size_t start = 0;
        for (std::size_t end; (end = pending.find('\n', start)) != std::string::npos;
             start = end + 1) {
            if (end > start) {
                LOG_INFO(Render_Vulkan, "Driver: {}",
                         std::string_view{pending}.substr(start, end - start));
            }
        }
        pending.erase(0, start);
    }
}

/// Makes the page the driver is asked through, in the app's own data folder (no shared
/// storage in between: the driver looks at it at every render pass), and tells the driver
/// where it is. Once; without it the comparison leaves the driver alone.
void MakeControlPage() {
    if (control_page) {
        return;
    }
    const std::string package = PackageName();
    if (package.empty()) {
        return;
    }
    const std::string dir = fmt::format("/data/data/{}/files", package);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string path = dir + "/tu_control";
    const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) {
        LOG_INFO(Common, "Diagnostics: no control page for the driver ({}: {})", path,
                 std::strerror(errno));
        return;
    }
    if (ftruncate(fd, CONTROL_PAGE_SIZE) == 0) {
        void* page = mmap(nullptr, CONTROL_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (page != MAP_FAILED) {
            control_page = static_cast<volatile s32*>(page);
            control_page[0] = 0;
            setenv("TU_AZAHAR_CONTROL", path.c_str(), 1);
        }
    }
    close(fd);
}

/// Called with the comparison's phase about to change
void EnterPhase(Phase phase, std::chrono::steady_clock::time_point now) {
    comparison.phase = phase;
    comparison.phase_start = now;
    u32 bits = 0;
    if (phase == Phase::OldDepthPath) {
        bits = Common::TraceMarkers::LoadOverwrittenDepth;
    } else if (phase == Phase::ClockAtConsole) {
        bits = Common::TraceMarkers::ConsoleClockFloor;
    } else if (phase == Phase::VertexDataCopied) {
        bits = Common::TraceMarkers::CopiedVertexData;
    }
    comparison.bits.store(bits, std::memory_order_relaxed);
    if (control_page) {
        control_page[0] = phase == Phase::AllDirect ? 1 : 0;
    }
    if (phase == Phase::AsBuilt) {
        // A game has to be slow anew for the next round
        comparison.slow_windows = 0;
    }
    LOG_INFO(Common, "Diagnostics: comparison: {} from here", PHASE_NAMES[static_cast<u32>(phase)]);
}

} // Anonymous namespace

void OnVBlank() {
    if (comparison.off) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (comparison.window_start == std::chrono::steady_clock::time_point{}) {
        comparison.window_start = now;
        comparison.phase_start = now;
    }
    if (++comparison.window_vblanks == SPEED_WINDOW_VBLANKS) {
        comparison.slow_windows =
            now - comparison.window_start > SLOW_WINDOW ? comparison.slow_windows + 1 : 0;
        comparison.window_vblanks = 0;
        comparison.window_start = now;
    }

    const auto in_phase = now - comparison.phase_start;
    if (comparison.phase == Phase::AsBuilt) {
        // The comparison is for a game that runs below its speed. One at its speed stays as
        // built: nothing is to be learnt from slowing it, and nobody is to be bothered.
        if (in_phase >= AS_BUILT_TIME && comparison.slow_windows >= SLOW_WINDOWS) {
            EnterPhase(Phase::OldDepthPath, now);
        }
    } else if (in_phase >= OTHER_PHASE_TIME) {
        const u32 next = (static_cast<u32>(comparison.phase) + 1) % static_cast<u32>(Phase::Count);
        EnterPhase(static_cast<Phase>(next), now);
    }
}

void OnGameStart() {
    comparison.window_start = {};
    comparison.window_vblanks = 0;
    comparison.slow_windows = 0;
    if (comparison.phase != Phase::AsBuilt) {
        EnterPhase(Phase::AsBuilt, std::chrono::steady_clock::now());
    }
}

u32 ComparisonBits() {
    return comparison.bits.load(std::memory_order_relaxed);
}

void PrepareDriverEnvironment() {
    const std::string files = FilesDir();
    if (!files.empty()) {
        // Mesa's own switch. A driver that is not Mesa's does not look at it.
        const std::string log = fmt::format("{}/{}", files, DRIVER_LOG);
        setenv("MESA_LOG_FILE", log.c_str(), 0);
        static std::once_flag relay;
        std::call_once(relay, [&log] {
            // What the last run left is not this run's
            std::remove(log.c_str());
            std::thread{RelayDriverLog, log}.detach();
        });
    }
    setenv("TU_AZAHAR_DIAG", "1", 0);
    MakeControlPage();

    std::scoped_lock lock{mutex};
    for (const auto& [name, value] : driver_environment) {
        std::string resolved = value;
        if (const std::size_t at = resolved.find(FILES_PLACEHOLDER); at != std::string::npos) {
            resolved.replace(at, FILES_PLACEHOLDER.size(), files);
        }
        setenv(name.c_str(), resolved.c_str(), 1);
        LOG_INFO(Common, "Diagnostics: driver environment {} = {}", name, resolved);
    }
}

const std::vector<u32>& FrameDumpTimes() {
    return frame_dump_times;
}

bool AutomaticFrameDumps() {
    return !frame_dumps_set;
}

void LogDeviceReport() {
    LOG_INFO(Common,
             "Device: {} {} ({}), system on a chip {} {} (platform {}, hardware {}), Android {} "
             "(API {}), build {}",
             SystemProperty("ro.product.manufacturer"), SystemProperty("ro.product.model"),
             SystemProperty("ro.product.device"), SystemProperty("ro.soc.manufacturer"),
             SystemProperty("ro.soc.model"), SystemProperty("ro.board.platform"),
             SystemProperty("ro.hardware"), SystemProperty("ro.build.version.release"),
             SystemProperty("ro.build.version.sdk"), SystemProperty("ro.build.display.id"));

    const std::vector<std::string> parts = CoreParts();
    std::string cores;
    for (const Cluster& cluster : Clusters()) {
        const std::string base =
            fmt::format("/sys/devices/system/cpu/cpu{}/cpufreq/", cluster.first);
        const std::string min = ReadLine(base + "cpuinfo_min_freq");
        fmt::format_to(std::back_inserter(cores), "{}cores {}-{} (part {}) {}-{} MHz, governor {}",
                       cores.empty() ? "" : "; ", cluster.first, cluster.last,
                       static_cast<std::size_t>(cluster.first) < parts.size() ? parts[cluster.first]
                                                                              : std::string{"?"},
                       min.empty() ? 0 : std::atol(min.c_str()) / 1000, cluster.max_khz / 1000,
                       ReadLine(base + "scaling_governor"));
    }
    LOG_INFO(Common, "Processor: {}; memory {} MB, {} MB of it available", cores,
             MemInfoKb("MemTotal") / 1024, MemInfoKb("MemAvailable") / 1024);

    const std::string kgsl{KGSL};
    const std::string model = ReadLine(kgsl + "gpu_model");
    if (model.empty()) {
        LOG_INFO(Common, "GPU: the kernel's figures ({}) are not readable by the app", KGSL);
    } else {
        LOG_INFO(Common,
                 "GPU as the kernel has it: {}, clock at most {} MHz (steps {}), governor {}, "
                 "load readable: {}",
                 model, std::atol(ReadLine(kgsl + "max_gpuclk").c_str()) / 1000000,
                 ReadLine(kgsl + "gpu_available_frequencies"), ReadLine(kgsl + "devfreq/governor"),
                 ReadLine(kgsl + "gpubusy").empty() ? "no" : "yes");
    }
    LOG_INFO(Common, "Thermal state now: {} (0 none to 6 shutdown, -1 not known)", ThermalStatus());
}

std::string SystemLine() {
    std::string text;
    const auto out = std::back_inserter(text);
    for (const Cluster& cluster : Clusters()) {
        const std::string now = ReadLine(
            fmt::format("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", cluster.first));
        fmt::format_to(out, "{}cores {}-{} at {} of {} MHz", text.empty() ? "system: " : ", ",
                       cluster.first, cluster.last, now.empty() ? 0 : std::atol(now.c_str()) / 1000,
                       cluster.max_khz / 1000);
    }
    fmt::format_to(out, "{}thermal state {}", text.empty() ? "system: " : ", ", ThermalStatus());

    // "busy total" in microseconds since the governor last looked
    const std::string kgsl{KGSL};
    const std::string busy = ReadLine(kgsl + "gpubusy");
    if (!busy.empty()) {
        long on = 0;
        long all = 0;
        std::sscanf(busy.c_str(), "%ld %ld", &on, &all);
        fmt::format_to(out, ", gpu {}% busy at {} MHz", all > 0 ? on * 100 / all : 0,
                       std::atol(ReadLine(kgsl + "gpuclk").c_str()) / 1000000);
        if (const std::string temperature = ReadLine(kgsl + "temp"); !temperature.empty()) {
            fmt::format_to(out, ", {:.0f} C", std::atol(temperature.c_str()) / 1000.0);
        }
    }
    return text;
}

void CollectFiles() {
    const std::string files = FilesDir();
    if (files.empty()) {
        return;
    }
    const std::string log_dir = FileUtil::GetUserPath(FileUtil::UserPath::LogDir);
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator{files, ec}) {
        const std::string name = entry.path().filename().string();
        if (name != DRIVER_LOG && !name.starts_with("gpu_trace") &&
            !name.starts_with("frame-dump-")) {
            continue;
        }
        std::FILE* source = std::fopen(entry.path().c_str(), "rb");
        if (!source) {
            continue;
        }
        FileUtil::IOFile target{log_dir + name, "wb"};
        std::size_t copied = 0;
        if (target.IsOpen()) {
            std::vector<char> data(1024 * 1024);
            std::size_t read = 0;
            while ((read = std::fread(data.data(), 1, data.size(), source)) > 0 &&
                   target.WriteBytes(data.data(), read) == read) {
                copied += read;
            }
        }
        std::fclose(source);
        LOG_INFO(Common, "Diagnostics: {} copied to the log folder, {} KB", name, copied / 1024);
    }
}

} // namespace Common::Diagnostics

#else

namespace Common::Diagnostics {

std::string FilesDir() {
    return {};
}

void LoadControlFile() {}

std::optional<std::string> Property(const char*) {
    return std::nullopt;
}

void PrepareDriverEnvironment() {}

void OnVBlank() {}

void OnGameStart() {}

u32 ComparisonBits() {
    return 0;
}

const std::vector<u32>& FrameDumpTimes() {
    static const std::vector<u32> none;
    return none;
}

bool AutomaticFrameDumps() {
    return false;
}

void LogDeviceReport() {}

std::string SystemLine() {
    return {};
}

void CollectFiles() {}

} // namespace Common::Diagnostics

#endif
