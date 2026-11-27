// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <chrono>

struct APerformanceHintSession;

// Represents an ADPF hint session (APerformanceHint, Android 13 and later).
class PerformanceHintSession {
public:
    explicit PerformanceHintSession(std::chrono::nanoseconds target);
    ~PerformanceHintSession();

    PerformanceHintSession(const PerformanceHintSession&) = delete;
    PerformanceHintSession& operator=(const PerformanceHintSession&) = delete;

    /// Reports the duration of the last frame's work, and the target duration for the next one.
    /// Where the system takes it (Android 15 and later), the time the GPU worked on the frame
    /// goes with it, so that the system can give the GPU the clock the frame needs as well.
    void ReportWork(std::chrono::nanoseconds actual, std::chrono::nanoseconds target,
                    std::chrono::nanoseconds gpu = {});

    /// Whether the system takes the GPU's time with the work (see ReportWork)
    bool ReportsGpuTime() const;

private:
    APerformanceHintSession* session{};
    std::chrono::nanoseconds current_target{};
    bool gpu_reported{};
};
