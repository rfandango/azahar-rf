// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_preset.h"

// (After Vulkan's headers as the renderer has them)
#include <vk_mem_alloc.h>

#define LIBRA_RUNTIME_VULKAN
#include "video_core/renderer_vulkan/librashader/librashader_ld.h"

namespace Vulkan {

struct ShaderPresetRunner::Library {
    libra_instance_t api;
};

namespace {

constexpr vk::Format OUTPUT_FORMAT = vk::Format::eR8G8B8A8Unorm;

/// The text of a librashader error, which is freed
std::string ErrorText(const libra_instance_t& api, libra_error_t error) {
    if (!error) {
        return {};
    }
    char* text = nullptr;
    api.error_write(error, &text);
    std::string out = text ? text : "no description";
    if (text) {
        api.error_free_string(&text);
    }
    api.error_free(&error);
    return out;
}

} // Anonymous namespace

ShaderPresetRunner::ShaderPresetRunner(const Instance& instance_, Scheduler& scheduler_,
                                       vk::Format source_format_)
    : instance{instance_}, scheduler{scheduler_}, source_format{source_format_} {
    // The screen's picture is drawn whole, and read by the preset's first pass
    const vk::AttachmentDescription attachment{
        .format = source_format,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    const vk::AttachmentReference reference{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &reference,
    };
    const std::array dependencies{
        // After the preset read the picture before
        vk::SubpassDependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eFragmentShader,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        },
        // Before the preset reads it
        vk::SubpassDependency{
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        },
    };
    source_pass = instance.GetDevice().createRenderPass(vk::RenderPassCreateInfo{
        .attachmentCount = 1,
        .pAttachments = &attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<u32>(dependencies.size()),
        .pDependencies = dependencies.data(),
    });
}

ShaderPresetRunner::~ShaderPresetRunner() {
    // (The renderer is through with the GPU by now)
    Free();
    instance.GetDevice().destroyRenderPass(source_pass);
}

void ShaderPresetRunner::Load(const std::string& path) {
    if (path == loaded_path) {
        return;
    }
    loaded_path = path;
    // What is on its way to the GPU may still run the chains before
    scheduler.Finish();
    Free();
    if (path.empty()) {
        return;
    }
    if (!library) {
        auto loaded = std::make_unique<Library>();
        loaded->api = librashader_load_instance();
        if (!loaded->api.instance_loaded) {
            LOG_ERROR(Render_Vulkan, "Shader presets: librashader.so could not be loaded");
            return;
        }
        library = std::move(loaded);
        LOG_INFO(Render_Vulkan, "Shader presets: librashader loaded, API version {}",
                 library->api.instance_api_version());
    }
    const libra_instance_t& api = library->api;

    // The preset is used up by making a chain: one for each screen
    std::array<libra_shader_preset_t, 2> presets{};
    for (auto& preset : presets) {
        const libra_error_t error = api.preset_create(path.c_str(), &preset);
        if (error) {
            LOG_ERROR(Render_Vulkan, "Shader preset {} could not be read: {}", path,
                      ErrorText(api, error));
            for (auto& made : presets) {
                if (made) {
                    api.preset_free(&made);
                }
            }
            return;
        }
    }
    const libra_device_vk_t device{
        .physical_device = instance.GetPhysicalDevice(),
        .instance = instance.GetInstance(),
        .device = instance.GetDevice(),
        .queue = instance.GetGraphicsQueue(),
        .entry = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
    };
    filter_chain_vk_opt_t options{};
    options.version = LIBRASHADER_CURRENT_VERSION;
    // (Frames whose work may still be on the GPU while the next ones are recorded)
    options.frames_in_flight = 6;
    options.use_dynamic_rendering = false;
    options.disable_cache = true;

    // The chains' first work is recorded into a command buffer of their own, which is run to
    // the end before they are used
    std::array<libra_error_t, 2> errors{};
    scheduler.Record([this, &api, &presets, &errors, device, options](vk::CommandBuffer cmdbuf) {
        for (std::size_t i = 0; i < presets.size(); i++) {
            libra_vk_filter_chain_t chain{};
            errors[i] = api.vk_filter_chain_create_deferred(&presets[i], device, cmdbuf, &options,
                                                            &chain);
            chains[i] = errors[i] ? nullptr : chain;
        }
    });
    scheduler.Finish();
    for (const libra_error_t error : errors) {
        if (error) {
            LOG_ERROR(Render_Vulkan, "Shader preset {} could not be made: {}", path,
                      ErrorText(api, error));
            Free();
            return;
        }
    }
    active = true;
    LOG_INFO(Render_Vulkan, "Shader preset {} loaded", path);
}

ShaderPresetRunner::Target ShaderPresetRunner::Source(u32 screen, u32 width, u32 height) {
    Image& source = sources[screen];
    Ensure(source, width, height, source_format,
           vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, true);
    return Target{source_pass, source.framebuffer};
}

vk::ImageView ShaderPresetRunner::Run(u32 screen, u32 width, u32 height) {
    if (!active || screen >= chains.size() || !chains[screen] || !sources[screen].image ||
        width == 0 || height == 0) {
        return {};
    }
    const Image& source = sources[screen];
    Image& output = outputs[screen];
    Ensure(output, width, height, OUTPUT_FORMAT,
           vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, false);

    frame_vk_opt_t options{};
    options.version = LIBRASHADER_CURRENT_VERSION;
    options.frame_direction = 1;
    options.total_subframes = 1;
    options.current_subframe = 1;
    options.frames_per_second = 60.0f;
    options.frametime_delta = 17;

    const libra_image_vk_t input{
        .handle = source.image,
        .format = static_cast<VkFormat>(source_format),
        .width = source.width,
        .height = source.height,
    };
    const libra_image_vk_t target{
        .handle = output.image,
        .format = static_cast<VkFormat>(OUTPUT_FORMAT),
        .width = width,
        .height = height,
    };
    // (The handle is kept as a plain pointer, librashader's own type is one too)
    auto* const chain = reinterpret_cast<libra_vk_filter_chain_t*>(&chains[screen]);
    scheduler.Record([api = &library->api, chain, input, target, frame = frame_count,
                      options](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        const vk::ImageMemoryBarrier before{
            .srcAccessMask = vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = target.handle,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                               vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {}, {},
                               before);
        const libra_error_t error = api->vk_filter_chain_frame(chain, cmdbuf, frame, input, target,
                                                               nullptr, nullptr, &options);
        if (error) {
            static bool told = false;
            if (!told) {
                told = true;
                LOG_ERROR(Render_Vulkan, "Shader preset: a frame failed: {}",
                          ErrorText(*api, error));
            }
        }
        // Read by the screens' drawing, as the game's own images are
        const vk::ImageMemoryBarrier after{
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = target.handle,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, after);
    });
    return output.view;
}

void ShaderPresetRunner::Free() {
    active = false;
    if (library) {
        for (void*& chain : chains) {
            if (chain) {
                auto handle = static_cast<libra_vk_filter_chain_t>(chain);
                library->api.vk_filter_chain_free(&handle);
                chain = nullptr;
            }
        }
    }
    for (Image& image : sources) {
        Destroy(image);
    }
    for (Image& image : outputs) {
        Destroy(image);
    }
}

void ShaderPresetRunner::Ensure(Image& image, u32 width, u32 height, vk::Format format,
                                vk::ImageUsageFlags usage, bool with_framebuffer) {
    if (image.image && image.width == width && image.height == height) {
        return;
    }
    if (image.image) {
        // (Only when the screens are shown at another size)
        scheduler.Finish();
        Destroy(image);
    }
    const VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = static_cast<VkFormat>(format),
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = static_cast<VkImageUsageFlags>(usage),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo alloc_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    VkImage raw{};
    const VkResult result = vmaCreateImage(instance.GetAllocator(), &image_info, &alloc_info,
                                           &raw, &image.allocation, nullptr);
    ASSERT_MSG(result == VK_SUCCESS, "Shader preset: image of {}x{} failed: {}", width, height,
               static_cast<int>(result));
    image.image = raw;
    const vk::Device device = instance.GetDevice();
    image.view = device.createImageView(vk::ImageViewCreateInfo{
        .image = image.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (with_framebuffer) {
        image.framebuffer = device.createFramebuffer(vk::FramebufferCreateInfo{
            .renderPass = source_pass,
            .attachmentCount = 1,
            .pAttachments = &image.view,
            .width = width,
            .height = height,
            .layers = 1,
        });
    }
    image.width = width;
    image.height = height;
}

void ShaderPresetRunner::Destroy(Image& image) {
    if (!image.image) {
        return;
    }
    const vk::Device device = instance.GetDevice();
    if (image.framebuffer) {
        device.destroyFramebuffer(image.framebuffer);
    }
    device.destroyImageView(image.view);
    vmaDestroyImage(instance.GetAllocator(), static_cast<VkImage>(image.image), image.allocation);
    image = {};
}

} // namespace Vulkan
