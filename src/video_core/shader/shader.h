// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <memory>
#include "common/common_types.h"

namespace Pica {

struct ShaderSetup;
struct ShaderUnit;

class ShaderEngine {
public:
    virtual ~ShaderEngine() = default;

    /**
     * Performs any shader unit setup that only needs to happen once per shader (as opposed to once
     * per vertex, which would happen within the `Run` function).
     */
    virtual void SetupBatch(ShaderSetup& setup, u32 entry_point) = 0;

    /**
     * As SetupBatch, for a batch too small to be worth holding the caller up for. Returns false
     * if the engine does not have the shader ready: it then gets it ready in the background,
     * and the caller runs this batch with another engine.
     */
    virtual bool SetupBatchIfReady(ShaderSetup& setup, u32 entry_point) {
        SetupBatch(setup, entry_point);
        return true;
    }

    /**
     * Runs the currently setup shader.
     *
     * @param setup Shader engine state, must be setup with SetupBatch on each shader change.
     * @param state Shader unit state, must be setup with input data before each shader invocation.
     */
    virtual void Run(const ShaderSetup& setup, ShaderUnit& state) const = 0;
};

std::unique_ptr<ShaderEngine> CreateEngine(bool use_jit);

} // namespace Pica
