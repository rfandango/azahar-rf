// Copyright 2016-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace VideoCore {

/// The names of the post-processing shaders in the "shaders" folder of the user directory, or in
/// its "anaglyph" folder, sorted: the files ending in ".glsl", without that ending
std::vector<std::string> GetPostProcessingShaderList(bool anaglyph);

/// The text of the post-processing shader of that name as the file has it, without the header a
/// renderer puts before it. Empty where there is no such shader or it cannot be read.
std::string GetPostProcessingShaderSource(bool anaglyph, std::string_view shader_name);

} // namespace VideoCore
