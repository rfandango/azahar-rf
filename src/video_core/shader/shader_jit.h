// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include "common/arch.h"
#if CITRA_ARCH(x86_64) || CITRA_ARCH(arm64)

#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "common/common_types.h"
#include "common/thread_worker.h"
#include "video_core/shader/shader.h"

namespace Pica::Shader {

class JitShader;

class JitEngine final : public ShaderEngine {
public:
    JitEngine();
    ~JitEngine() override;

    void SetupBatch(ShaderSetup& setup, u32 entry_point) override;
    bool SetupBatchIfReady(ShaderSetup& setup, u32 entry_point) override;
    void Run(const ShaderSetup& setup, ShaderUnit& state) const override;

private:
    /// The compiled shaders. One that is being compiled in the background is there already,
    /// empty.
    std::unordered_map<u64, std::unique_ptr<JitShader>> cache;
    std::mutex cache_mutex;
    std::condition_variable compiled;
    /// Compiles the shaders that small batches came across. Last, so that it is gone before
    /// what it works on.
    Common::ThreadWorker compiler;
};

} // namespace Pica::Shader

#endif // CITRA_ARCH(x86_64) || CITRA_ARCH(arm64)
