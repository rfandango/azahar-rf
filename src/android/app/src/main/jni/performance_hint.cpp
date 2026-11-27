// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <dlfcn.h>
#include <unistd.h>
#include "common/logging/log.h"
#include "jni/performance_hint.h"

struct APerformanceHintManager;
struct AWorkDuration;

namespace {

struct HintApi {
    APerformanceHintManager* (*get_manager)();
    APerformanceHintSession* (*create_session)(APerformanceHintManager*, const int32_t*, size_t,
                                               int64_t);
    int (*update_target)(APerformanceHintSession*, int64_t);
    int (*report_actual)(APerformanceHintSession*, int64_t);
    void (*close_session)(APerformanceHintSession*);

    // Android 15 (API 35): a frame's work with its parts, the GPU's among them
    AWorkDuration* (*duration_create)();
    void (*duration_release)(AWorkDuration*);
    void (*duration_set_start)(AWorkDuration*, int64_t);
    void (*duration_set_total)(AWorkDuration*, int64_t);
    void (*duration_set_cpu)(AWorkDuration*, int64_t);
    void (*duration_set_gpu)(AWorkDuration*, int64_t);
    int (*report_actual2)(APerformanceHintSession*, AWorkDuration*);

    bool HasGpuDurations() const {
        return duration_create && duration_release && duration_set_start && duration_set_total &&
               duration_set_cpu && duration_set_gpu && report_actual2;
    }

    bool Load() {
        void* lib = dlopen("libandroid.so", RTLD_NOW);
        if (!lib) {
            return false;
        }
        const auto find = [lib]<typename F>(F& function, const char* name) {
            function = reinterpret_cast<F>(dlsym(lib, name));
        };
        find(duration_create, "AWorkDuration_create");
        find(duration_release, "AWorkDuration_release");
        find(duration_set_start, "AWorkDuration_setWorkPeriodStartTimestampNanos");
        find(duration_set_total, "AWorkDuration_setActualTotalDurationNanos");
        find(duration_set_cpu, "AWorkDuration_setActualCpuDurationNanos");
        find(duration_set_gpu, "AWorkDuration_setActualGpuDurationNanos");
        find(report_actual2, "APerformanceHint_reportActualWorkDuration2");
        get_manager =
            reinterpret_cast<decltype(get_manager)>(dlsym(lib, "APerformanceHint_getManager"));
        create_session = reinterpret_cast<decltype(create_session)>(
            dlsym(lib, "APerformanceHint_createSession"));
        update_target = reinterpret_cast<decltype(update_target)>(
            dlsym(lib, "APerformanceHint_updateTargetWorkDuration"));
        report_actual = reinterpret_cast<decltype(report_actual)>(
            dlsym(lib, "APerformanceHint_reportActualWorkDuration"));
        close_session =
            reinterpret_cast<decltype(close_session)>(dlsym(lib, "APerformanceHint_closeSession"));
        return get_manager && create_session && update_target && report_actual && close_session;
    }
};

const HintApi* GetApi() {
    static HintApi api{};
    static const bool loaded = api.Load();
    return loaded ? &api : nullptr;
}

} // Anonymous namespace

PerformanceHintSession::PerformanceHintSession(std::chrono::nanoseconds target)
    : current_target{target} {
    const HintApi* api = GetApi();
    if (!api) {
        LOG_INFO(Frontend, "Performance hint sessions are not available");
        return;
    }
    APerformanceHintManager* manager = api->get_manager();
    const int32_t tid = gettid();
    if (manager) {
        session = api->create_session(manager, &tid, 1, target.count());
    }
    if (session) {
        LOG_INFO(Frontend, "Performance hint session created, target {} us{}", target.count() / 1000,
                 api->HasGpuDurations() ? ", the GPU's time reported with the work" : "");
    } else {
        LOG_WARNING(Frontend, "Could not create a performance hint session");
    }
}

PerformanceHintSession::~PerformanceHintSession() {
    if (session) {
        GetApi()->close_session(session);
    }
}

bool PerformanceHintSession::ReportsGpuTime() const {
    return session && GetApi()->HasGpuDurations();
}

void PerformanceHintSession::ReportWork(std::chrono::nanoseconds actual,
                                        std::chrono::nanoseconds target,
                                        std::chrono::nanoseconds gpu) {
    if (!session) {
        return;
    }
    const HintApi* api = GetApi();
    if (target != current_target) {
        api->update_target(session, target.count());
        current_target = target;
    }
    if (actual.count() <= 0) {
        return;
    }
    if (gpu.count() > 0 && api->HasGpuDurations()) {
        if (AWorkDuration* const duration = api->duration_create()) {
            // (The steady clock is the monotonic one the system wants the start in)
            const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            api->duration_set_start(duration, std::max<int64_t>(now - actual.count(), 1));
            api->duration_set_total(duration, actual.count());
            api->duration_set_cpu(duration, actual.count());
            api->duration_set_gpu(duration, gpu.count());
            const int result = api->report_actual2(session, duration);
            api->duration_release(duration);
            if (result == 0) {
                if (!gpu_reported) {
                    gpu_reported = true;
                    LOG_INFO(Frontend, "Performance hint session: the GPU's time is taken");
                }
                return;
            }
        }
    }
    api->report_actual(session, actual.count());
}
