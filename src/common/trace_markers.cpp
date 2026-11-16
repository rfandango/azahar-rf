// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include "common/trace_markers.h"

#ifdef CITRA_ENABLE_TRACE_MARKERS

#include <array>
#include <atomic>
#include <cstdlib>
#include <android/trace.h>
#include <sys/system_properties.h>

namespace Common::TraceMarkers {

namespace {

constexpr const char* LEVEL_PROPERTY = "debug.azahar.trace";
constexpr const char* EXPERIMENT_PROPERTY = "debug.azahar.experiment";

std::atomic<std::uint32_t> experiments{0};

constexpr std::array<const char*, static_cast<std::size_t>(Counter::Count)> COUNTER_NAMES{
    "azahar.draws_hw",        "azahar.draws_sw",        "azahar.vertices_sw",
    "azahar.command_lists",   "azahar.shader_compiles", "azahar.shader_jit_compiles",
    "azahar.pipeline_builds", "azahar.surface_uploads", "azahar.surface_downloads",
    "azahar.surface_copies",  "azahar.submits",         "azahar.gpu_waits",
};

std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Counter::Count)> counters{};

Level ReadConfiguredLevel() {
    char value[PROP_VALUE_MAX]{};
    if (__system_property_get(LEVEL_PROPERTY, value) <= 0) {
        return Level::Frame;
    }
    const int level = std::atoi(value);
    if (level <= static_cast<int>(Level::Off)) {
        return Level::Off;
    }
    if (level >= static_cast<int>(Level::Detail)) {
        return Level::Detail;
    }
    return Level::Frame;
}

Level ConfiguredLevel() {
    static const Level level = ReadConfiguredLevel();
    return level;
}

} // Anonymous namespace

bool IsEnabled(Level level) {
    const Level configured = ConfiguredLevel();
    return configured != Level::Off && level <= configured && ATrace_isEnabled();
}

void BeginSection(const char* name) {
    ATrace_beginSection(name);
}

void EndSection() {
    ATrace_endSection();
}

void BeginAsyncSection(const char* name, std::int32_t cookie) {
    if (IsEnabled(Level::Frame)) {
        ATrace_beginAsyncSection(name, cookie);
    }
}

void EndAsyncSection(const char* name, std::int32_t cookie) {
    if (IsEnabled(Level::Frame)) {
        ATrace_endAsyncSection(name, cookie);
    }
}

void SetCounterIfEnabled(const char* name, std::int64_t value) {
    if (IsEnabled(Level::Frame)) {
        ATrace_setCounter(name, value);
    }
}

void Increment(Counter counter, std::uint64_t amount) {
    if (!IsEnabled(Level::Frame)) {
        return;
    }
    counters[static_cast<std::size_t>(counter)].fetch_add(amount, std::memory_order_relaxed);
}

std::uint32_t Experiments() {
    return experiments.load(std::memory_order_relaxed);
}

void EndFrame() {
    char value[PROP_VALUE_MAX]{};
    const std::uint32_t requested =
        __system_property_get(EXPERIMENT_PROPERTY, value) > 0
            ? static_cast<std::uint32_t>(std::strtoul(value, nullptr, 0))
            : 0;
    experiments.store(requested, std::memory_order_relaxed);

    if (!IsEnabled(Level::Frame)) {
        return;
    }
    for (std::size_t i = 0; i < counters.size(); i++) {
        const auto value = counters[i].exchange(0, std::memory_order_relaxed);
        ATrace_setCounter(COUNTER_NAMES[i], static_cast<std::int64_t>(value));
    }
}

} // namespace Common::TraceMarkers

#endif
