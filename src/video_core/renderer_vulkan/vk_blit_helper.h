// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <memory>
#include <optional>
#include <unordered_map>

#include "common/vector_math.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"

namespace VideoCore {
struct TextureBlit;
struct TextureCopy;
struct TextureRelayout;
struct BufferTextureCopy;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class RenderManager;
class Scheduler;
class Surface;
class DescriptorUpdateQueue;

class BlitHelper {
    friend class TextureRuntime;

public:
    explicit BlitHelper(const Instance& instance, Scheduler& scheduler,
                        RenderManager& renderpass_cache, DescriptorUpdateQueue& update_queue);
    ~BlitHelper();
    bool Filter(Surface& surface, const VideoCore::TextureBlit& blit);

    bool BlitDepthStencil(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit);

    bool ConvertDS24S8ToRGBA8(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);

    /// Fills part of dest from source, which holds the same guest bytes with another layout
    bool Relayout(Surface& source, Surface& dest, const VideoCore::TextureRelayout& relayout);

    bool DepthToBuffer(Surface& source, vk::Buffer buffer,
                       const VideoCore::BufferTextureCopy& copy);

    void ResolveTexture(Surface& surface);

private:
    /// Copies the stencil of the surface's current image into the stencil buffer, a byte a
    /// texel, for a shader to read through stencil_view instead of sampling it: an image whose
    /// stencil is sampled is not compressed on every GPU. Where the copy begins in the buffer
    /// and its row length, or none where it does not fit (the stencil is then sampled).
    std::optional<Common::Vec2i> CopyStencil(Surface& source);

    vk::Pipeline MakeComputePipeline(vk::ShaderModule shader, vk::PipelineLayout layout);
    vk::Pipeline MakeDepthStencilBlitPipeline(vk::ShaderModule fragment_shader);
    vk::Pipeline MakeFilterPipeline(
        vk::ShaderModule fragment_shader, vk::PipelineLayout layout,
        VideoCore::PixelFormat color_format = VideoCore::PixelFormat::RGBA8);

    void FilterAnime4K(Surface& surface, const VideoCore::TextureBlit& blit);
    void FilterBicubic(Surface& surface, const VideoCore::TextureBlit& blit);
    void FilterScaleForce(Surface& surface, const VideoCore::TextureBlit& blit);
    void FilterXbrz(Surface& surface, const VideoCore::TextureBlit& blit);
    void FilterMMPX(Surface& surface, const VideoCore::TextureBlit& blit);

    void FilterPass(Surface& surface, vk::Pipeline pipeline, vk::PipelineLayout layout,
                    const VideoCore::TextureBlit& blit);

    void FilterPassThreeTextures(Surface& surface, vk::Pipeline pipeline, vk::PipelineLayout layout,
                                 const VideoCore::TextureBlit& blit);

    void FilterPassYGradient(Surface& surface, vk::Pipeline pipeline, vk::PipelineLayout layout,
                             const VideoCore::TextureBlit& blit);

private:
    const Instance& instance;
    Scheduler& scheduler;
    RenderManager& renderpass_cache;
    DescriptorUpdateQueue& update_queue;

    vk::Device device;
    vk::RenderPass r32_renderpass;

    DescriptorHeap compute_provider;
    DescriptorHeap compute_buffer_provider;
    DescriptorHeap two_textures_provider;
    DescriptorHeap single_texture_provider;
    DescriptorHeap three_textures_provider;
    vk::PipelineLayout compute_pipeline_layout;
    vk::PipelineLayout relayout_pipeline_layout;
    vk::PipelineLayout compute_buffer_pipeline_layout;
    vk::PipelineLayout two_textures_pipeline_layout;
    vk::PipelineLayout single_texture_pipeline_layout;
    vk::PipelineLayout three_textures_pipeline_layout;

    vk::ShaderModule full_screen_vert;
    vk::ShaderModule d24s8_to_rgba8_comp;
    vk::ShaderModule d24s8_to_rgba8_ms_comp;
    vk::ShaderModule depth_to_buffer_comp;
    vk::ShaderModule relayout_color_comp;
    vk::ShaderModule relayout_depth_comp;
    vk::ShaderModule relayout_rgb_comp;
    vk::ShaderModule blit_depth_stencil_frag;
    vk::ShaderModule bicubic_frag;
    vk::ShaderModule scale_force_frag;
    vk::ShaderModule xbrz_frag;
    vk::ShaderModule mmpx_frag;
    vk::ShaderModule refine_frag;

    vk::Pipeline d24s8_to_rgba8_pipeline;
    vk::Pipeline d24s8_to_rgba8_ms_pipeline;
    vk::Pipeline depth_to_buffer_pipeline;
    vk::Pipeline relayout_color_pipeline;
    vk::Pipeline relayout_depth_pipeline;
    vk::Pipeline relayout_rgb_pipeline;
    vk::Pipeline depth_blit_pipeline;
    vk::Sampler linear_sampler;
    vk::Sampler nearest_sampler;

    /// Stencil copied for the shaders that read it (CopyStencil), and its view of bytes
    std::unique_ptr<StreamBuffer> stencil_buffer;
    vk::BufferView stencil_view;
    u64 stencil_capacity{};
    /// The shaders that read the stencil from that copy, and their pipelines
    vk::ShaderModule d24s8_to_rgba8_bytes_comp;
    vk::ShaderModule depth_to_buffer_bytes_comp;
    vk::ShaderModule relayout_depth_bytes_comp;
    vk::ShaderModule blit_depth_stencil_bytes_frag;
    vk::Pipeline d24s8_to_rgba8_bytes_pipeline;
    vk::Pipeline depth_to_buffer_bytes_pipeline;
    vk::Pipeline relayout_depth_bytes_pipeline;
    vk::Pipeline depth_blit_bytes_pipeline;

    /// Cache of texture filter pipelines (keyed by shader+layout+format hash)
    std::unordered_map<std::uint64_t, vk::Pipeline> filter_pipeline_cache;
};

} // namespace Vulkan
