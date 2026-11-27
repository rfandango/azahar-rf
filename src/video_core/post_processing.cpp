// Copyright 2016-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <mutex>
#include <optional>
#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/string_util.h"
#include "video_core/post_processing.h"

namespace VideoCore {

namespace {

std::string ShaderDir(bool anaglyph) {
    std::string shader_dir = FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir);
    if (!FileUtil::IsDirectory(shader_dir)) {
        FileUtil::CreateDir(shader_dir);
    }
    if (anaglyph) {
        shader_dir += "anaglyph";
        if (!FileUtil::IsDirectory(shader_dir)) {
            FileUtil::CreateDir(shader_dir);
        }
    }
    return shader_dir;
}

/// The name of a shader file without its ending, for a file that ends in ".glsl" in any case
std::optional<std::string> ShaderName(const std::string& file_name) {
    const std::size_t dot_pos = file_name.rfind('.');
    if (dot_pos == std::string::npos || Common::ToLower(file_name.substr(dot_pos + 1)) != "glsl") {
        return std::nullopt;
    }
    return file_name.substr(0, dot_pos);
}

} // Anonymous namespace

std::vector<std::string> GetPostProcessingShaderList(bool anaglyph) {
    std::vector<std::string> shader_names;
    const auto callback = [&shader_names](u64*, const std::string& directory,
                                          const std::string& virtual_name) -> bool {
        if (!FileUtil::IsDirectory(directory + DIR_SEP + virtual_name)) {
            if (auto name = ShaderName(virtual_name)) {
                shader_names.push_back(std::move(*name));
            }
        }
        return true;
    };
    FileUtil::ForeachDirectoryEntry(nullptr, ShaderDir(anaglyph), callback);
    std::sort(shader_names.begin(), shader_names.end());
    return shader_names;
}

std::string GetPostProcessingShaderSource(bool anaglyph, std::string_view shader_name) {
    // The folder is looked through for the file, as the ending may be written in any case
    std::string shader_path;
    const auto callback = [shader_name, &shader_path](u64*, const std::string& directory,
                                                      const std::string& virtual_name) -> bool {
        const std::string physical_name = directory + DIR_SEP + virtual_name;
        const auto name = ShaderName(virtual_name);
        if (name && *name == shader_name && !FileUtil::IsDirectory(physical_name)) {
            shader_path = physical_name;
            return false;
        }
        return true;
    };
    FileUtil::ForeachDirectoryEntry(nullptr, ShaderDir(anaglyph), callback);
    if (shader_path.empty()) {
        return {};
    }
    // (As it is: in text mode, Windows would give fewer characters than the file's size)
    std::string text;
    const std::size_t read = FileUtil::ReadFileToString(false, shader_path, text);
    text.resize(read);
    return text;
}

namespace {
std::mutex preset_folder_mutex;
std::string preset_folder;
} // Anonymous namespace

void SetShaderPresetFolder(std::string folder) {
    if (!folder.empty() && folder.back() != '/' && folder.back() != '\\') {
        folder += DIR_SEP;
    }
    std::scoped_lock lock{preset_folder_mutex};
    preset_folder = std::move(folder);
}

std::string GetShaderPresetFolder() {
    std::scoped_lock lock{preset_folder_mutex};
    if (!preset_folder.empty()) {
        return preset_folder;
    }
    return FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir) + "slang" DIR_SEP;
}

} // namespace VideoCore
