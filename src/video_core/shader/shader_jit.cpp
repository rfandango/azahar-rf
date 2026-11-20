// Copyright 2016-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include "common/arch.h"
#if CITRA_ARCH(x86_64) || CITRA_ARCH(arm64)

#include <array>
#include <chrono>
#include "common/assert.h"
#include "common/hash.h"
#include "common/hitch_tracker.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/trace_markers.h"
#include "video_core/shader/shader.h"
#include "video_core/shader/shader_jit.h"
#if CITRA_ARCH(arm64)
#include "video_core/shader/shader_jit_a64_compiler.h"
#endif
#if CITRA_ARCH(x86_64)
#include "video_core/shader/shader_jit_x64_compiler.h"
#endif

namespace Pica::Shader {

namespace {

#ifdef CITRA_ENABLE_TRACE_MARKERS
/// Milliseconds since `start`, for the log of profiling builds
double MsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}
#endif

} // Anonymous namespace

JitEngine::JitEngine() : compiler{1, "PICA shader JIT"} {}
JitEngine::~JitEngine() = default;

void JitEngine::SetupBatch(ShaderSetup& setup, u32 entry_point) {
    ASSERT(entry_point < MAX_PROGRAM_CODE_LENGTH);
    setup.entry_point = entry_point;

    setup.DoProgramCodeFixup();
    const u64 code_hash = setup.GetProgramCodeHash();
    const u64 swizzle_hash = setup.GetSwizzleDataHash();

    const u64 cache_key = Common::HashCombine(code_hash, swizzle_hash);
    std::unique_lock lock{cache_mutex};
    auto iter = cache.find(cache_key);
    if (iter == cache.end()) {
        CITRA_TRACE_SCOPE("ShaderJit::Compile");
        CITRA_HITCH_SCOPE(ShaderCompile);
        CITRA_TRACE_COUNT(ShaderJitCompiles);
#ifdef CITRA_ENABLE_TRACE_MARKERS
        const auto start = std::chrono::steady_clock::now();
#endif
        auto shader = std::make_unique<JitShader>();
        shader->Compile(&setup.GetProgramCode(), &setup.GetSwizzleData());
        iter = cache.emplace(cache_key, std::move(shader)).first;
#ifdef CITRA_ENABLE_TRACE_MARKERS
        LOG_INFO(HW_GPU, "PICA shader compiled for the CPU by its caller: {:.1f} ms",
                 MsSince(start));
#endif
    } else if (!iter->second) {
        // On its way through the background compiler for a small batch that came before. Only
        // this thread adds entries, so the one found stays where it is meanwhile.
        CITRA_HITCH_SCOPE(ShaderCompile);
        compiled.wait(lock, [&] { return iter->second != nullptr; });
    }
    setup.cached_shader = iter->second.get();
}

bool JitEngine::SetupBatchIfReady(ShaderSetup& setup, u32 entry_point) {
    ASSERT(entry_point < MAX_PROGRAM_CODE_LENGTH);
    setup.entry_point = entry_point;

    setup.DoProgramCodeFixup();
    const u64 cache_key =
        Common::HashCombine(setup.GetProgramCodeHash(), setup.GetSwizzleDataHash());
    std::scoped_lock lock{cache_mutex};
    const auto [iter, is_new] = cache.try_emplace(cache_key);
    if (!is_new) {
        if (!iter->second) {
            return false;
        }
        setup.cached_shader = iter->second.get();
        return true;
    }

    // Compiling a program takes around ten milliseconds on a phone, most of a frame, and a
    // handful of vertices are interpreted in a fraction of that. So it is done in the
    // background, from copies: the game may have loaded other code by the time it is its turn.
    auto code = std::make_unique<std::array<u32, MAX_PROGRAM_CODE_LENGTH>>(setup.GetProgramCode());
    auto swizzle =
        std::make_unique<std::array<u32, MAX_SWIZZLE_DATA_LENGTH>>(setup.GetSwizzleData());
    compiler.QueueWork([this, cache_key, code = std::move(code), swizzle = std::move(swizzle)] {
        CITRA_TRACE_SCOPE("ShaderJit::Compile (background)");
#ifdef CITRA_ENABLE_TRACE_MARKERS
        const auto start = std::chrono::steady_clock::now();
#endif
        auto shader = std::make_unique<JitShader>();
        shader->Compile(code.get(), swizzle.get());
        {
            std::scoped_lock cache_lock{cache_mutex};
            cache.find(cache_key)->second = std::move(shader);
        }
        compiled.notify_all();
#ifdef CITRA_ENABLE_TRACE_MARKERS
        LOG_INFO(HW_GPU, "PICA shader compiled for the CPU in the background: {:.1f} ms",
                 MsSince(start));
#endif
    });
    return false;
}

MICROPROFILE_DECLARE(GPU_Shader);

void JitEngine::Run(const ShaderSetup& setup, ShaderUnit& state) const {
    ASSERT(setup.cached_shader != nullptr);

    MICROPROFILE_SCOPE(GPU_Shader);

    const JitShader* shader = static_cast<const JitShader*>(setup.cached_shader);
    shader->Run(setup, state, setup.entry_point);
}

} // namespace Pica::Shader

#endif // CITRA_ARCH(x86_64) || CITRA_ARCH(arm64)
