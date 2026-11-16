// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <functional>
#include <string>
#include "video_core/pica/shader_setup.h"

namespace Pica::Shader::Generator::GLSL {

using RegGetter = std::function<std::string(u32)>;

/**
 * Translates a PICA shader program to GLSL.
 * @param known_bools bool uniforms that hold the given values whenever the result is used. Flow
 * control that tests them is resolved during translation, leaving only the code that runs.
 */
std::string DecompileProgram(const Pica::ProgramCode& program_code,
                             const Pica::SwizzleData& swizzle_data, u32 main_offset,
                             const RegGetter& inputreg_getter, const RegGetter& outputreg_getter,
                             bool sanitize_mul, Pica::ShaderSetup::BoolSpecialization known_bools);

} // namespace Pica::Shader::Generator::GLSL
