// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <string>
#include <string_view>

namespace OpenGL {

// Returns the shader code for the shader named "shader_name"
// with the appropriate header prepended to it
// If anaglyph is true, it searches the shaders/anaglyph directory rather than
// the shaders directory
// If the shader cannot be loaded, an empty string is returned
// (The list of shaders is VideoCore::GetPostProcessingShaderList)
std::string GetPostProcessingShaderCode(bool anaglyph, std::string_view shader_name);

} // namespace OpenGL
