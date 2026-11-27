// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <memory>
#include <string>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

VK_DEFINE_HANDLE(VmaAllocation)

namespace Vulkan {

class Instance;
class Scheduler;

/**
 * Runs RetroArch shader presets (.slangp, as libretro's shader packs have them) over the screens
 * with librashader, which is loaded at run time (librashader.so): without it nothing here does
 * anything. Each screen has a filter chain of its own, as the presets keep frames of their own.
 *
 * A preset reads a screen the way it is shown (upright where the console's own image of it lies
 * on its side): the renderer draws it so into Source() first.
 */
class ShaderPresetRunner {
public:
    /// The screens' pictures for the presets are drawn with the present pipelines, so they are
    /// of the format of the display's images
    explicit ShaderPresetRunner(const Instance& instance, Scheduler& scheduler,
                                vk::Format source_format);
    ~ShaderPresetRunner();

    /// Whether a preset is loaded
    [[nodiscard]] bool Active() const noexcept {
        return active;
    }

    /// Loads the preset at that path (a file system path) for both screens, in place of the one
    /// before; an empty path or one that fails leaves none. Waits for the GPU.
    void Load(const std::string& path);

    /// Where the picture of a screen (0 top, 1 bottom) the preset reads is drawn: a render pass
    /// whose image is ready to be read by the preset afterwards
    struct Target {
        vk::RenderPass render_pass;
        vk::Framebuffer framebuffer;
    };
    [[nodiscard]] Target Source(u32 screen, u32 width, u32 height);

    /// Records running the preset of a screen over its source into an image of width x height,
    /// which is in the general layout once that has run. Returns its view, or a null one where
    /// the preset cannot run. To be called outside a render pass, after the source is drawn.
    vk::ImageView Run(u32 screen, u32 width, u32 height);

    /// Counts a frame, for the presets that change with time
    void EndFrame() noexcept {
        frame_count++;
    }

private:
    struct Image {
        vk::Image image{};
        VmaAllocation allocation{};
        vk::ImageView view{};
        vk::Framebuffer framebuffer{};
        u32 width{};
        u32 height{};
    };
    struct Library;

    void Free();
    void Ensure(Image& image, u32 width, u32 height, vk::Format format,
                vk::ImageUsageFlags usage, bool with_framebuffer);
    void Destroy(Image& image);

    const Instance& instance;
    Scheduler& scheduler;
    const vk::Format source_format;
    vk::RenderPass source_pass{};
    std::unique_ptr<Library> library;
    bool active{};
    std::string loaded_path;
    /// For each screen: its filter chain, the picture the preset reads and the image it draws into
    std::array<void*, 2> chains{};
    std::array<Image, 2> sources{};
    std::array<Image, 2> outputs{};
    std::size_t frame_count{};
};

} // namespace Vulkan
