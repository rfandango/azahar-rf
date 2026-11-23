// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <cmath>
#include "common/alignment.h"
#include "common/data_cache.h"
#include "common/hitch_tracker.h"
#include "common/literals.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "common/microprofile.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "core/core.h"
#include "core/loader/loader.h"
#include "core/memory.h"
#include "video_core/frame_state_dump.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_timing.h"
#include "video_core/shader/generator/pica_fs_config.h"
#include "video_core/texture/texture_decode.h"

namespace Vulkan {

namespace {

MICROPROFILE_DEFINE(Vulkan_VS, "Vulkan", "Vertex Shader Setup", MP_RGB(192, 128, 128));
MICROPROFILE_DEFINE(Vulkan_GS, "Vulkan", "Geometry Shader Setup", MP_RGB(128, 192, 128));
MICROPROFILE_DEFINE(Vulkan_Drawing, "Vulkan", "Drawing", MP_RGB(128, 128, 192));

using TriangleTopology = Pica::PipelineRegs::TriangleTopology;
using VideoCore::SurfaceType;

using namespace Common::Literals;
using namespace Pica::Shader::Generator;

constexpr u64 STREAM_BUFFER_SIZE = 64_MiB;
constexpr u64 UNIFORM_BUFFER_SIZE = 8_MiB;
constexpr u64 TEXTURE_BUFFER_SIZE = 2_MiB;

constexpr vk::BufferUsageFlags BUFFER_USAGE =
    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer;

struct DrawParams {
    u32 vertex_count;
    s32 vertex_offset;
    u32 binding_count;
    std::array<u32, 16> bindings;
    u16 in_guest_memory; ///< See RasterizerVulkan::bindings_in_guest_memory
    bool is_indexed;
};

/// How many states of draws are kept (RasterizerVulkan::DrawSite), each in the one slot its
/// hash comes to
constexpr u32 DRAW_SITE_BITS = 12;

/// The emulated memory for the host GPU to use where it is, if that is asked for
[[nodiscard]] std::span<u8> PoolForGpu(Memory::MemorySystem& memory) {
    if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyVertexCopies) != 0) {
        return {};
    }
    return memory.GetHostPool();
}

[[nodiscard]] u64 TextureBufferSize(const Instance& instance) {
    // Use the smallest texel size from the texel views
    // which corresponds to eR32G32Sfloat
    const u64 max_size = instance.MaxTexelBufferElements() * 8;
    return std::min(max_size, TEXTURE_BUFFER_SIZE);
}

} // Anonymous namespace

RasterizerVulkan::RasterizerVulkan(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                                   VideoCore::CustomTexManager& custom_tex_manager,
                                   VideoCore::RendererBase& renderer,
                                   Frontend::EmuWindow& emu_window, const Instance& instance,
                                   Scheduler& scheduler, RenderManager& renderpass_cache,
                                   DescriptorUpdateQueue& update_queue_, u32 image_count)
    : RasterizerAccelerated{memory, pica}, instance{instance}, scheduler{scheduler},
      renderpass_cache{renderpass_cache}, update_queue{update_queue_},
      pipeline_cache{instance, scheduler, renderpass_cache, update_queue},
      runtime{instance, scheduler, renderpass_cache, update_queue, image_count},
      res_cache{memory, custom_tex_manager, runtime, regs, renderer},
      stream_buffer{instance, scheduler, BUFFER_USAGE, STREAM_BUFFER_SIZE},
      uniform_buffer{instance, scheduler, vk::BufferUsageFlagBits::eUniformBuffer,
                     UNIFORM_BUFFER_SIZE},
      texture_buffer{instance, scheduler, vk::BufferUsageFlagBits::eUniformTexelBuffer,
                     TextureBufferSize(instance)},
      texture_lf_buffer{instance, scheduler, vk::BufferUsageFlagBits::eUniformTexelBuffer,
                        TextureBufferSize(instance)},
      guest_memory{instance, PoolForGpu(memory)},
      in_place_data{
          [this](PAddr addr, u32 size, bool watch) {
              this->memory.RasterizerMarkRegionWatched(addr, size, watch);
          },
          [this](PAddr addr, u32 size) {
              // A page is read in place from now on. Where the GPU does not see through the
              // processor's cache, what the cache holds of the page goes to memory first.
              if (guest_memory.IsCoherent()) {
                  return;
              }
              if (const u8* const data = this->memory.GetPhysicalPointer(addr)) {
                  Common::WriteOutDataCache(data, size);
              }
          },
          (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::MeasureVertexData) != 0},
      watch_data{(Common::TraceMarkers::Experiments2() & Common::TraceMarkers::MeasureVertexData) !=
                 0},
      async_shaders{Settings::values.async_shader_compilation.GetValue()} {

    vertex_buffers.fill(stream_buffer.Handle());
    // What the cache writes back into the emulated memory may be vertex or index data that a
    // draw still on its way through the GPU reads there: a game can bring such data to where
    // it draws from with a GPU copy, which lands in a surface first
    res_cache.SetGuestWriteCallback(
        [this](PAddr addr, u32 size) { WaitForDataInPlace(addr, size); });

    // Query uniform buffer alignment.
    uniform_buffer_alignment = instance.UniformMinAlignment();
    uniform_size_aligned_vs_pica =
        Common::AlignUp<u32>(sizeof(VSPicaUniformData), uniform_buffer_alignment);
    uniform_size_aligned_vs = Common::AlignUp<u32>(sizeof(VSUniformData), uniform_buffer_alignment);
    uniform_size_aligned_fs = Common::AlignUp<u32>(sizeof(FSUniformData), uniform_buffer_alignment);

    // Define vertex layout for software shaders
    MakeSoftwareVertexLayout();
    pipeline_info.state.vertex_layout = software_layout;

    const vk::Device device = instance.GetDevice();
    texture_lf_view = device.createBufferViewUnique({
        .buffer = texture_lf_buffer.Handle(),
        .format = vk::Format::eR32G32Sfloat,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    });
    texture_rg_view = device.createBufferViewUnique({
        .buffer = texture_buffer.Handle(),
        .format = vk::Format::eR32G32Sfloat,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    });
    texture_rgba_view = device.createBufferViewUnique({
        .buffer = texture_buffer.Handle(),
        .format = vk::Format::eR32G32B32A32Sfloat,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    });

    scheduler.RegisterOnSubmit([this, &renderpass_cache] {
        renderpass_cache.EndRendering();
        if (watch_data) {
            data_watch.OnSubmit();
        }
    });

    // Prepare the static buffer descriptor set.
    const auto buffer_set = pipeline_cache.Acquire(DescriptorHeapType::Buffer);
    update_queue.AddBuffer(buffer_set, 0, uniform_buffer.Handle(), 0, sizeof(VSPicaUniformData));
    update_queue.AddBuffer(buffer_set, 1, uniform_buffer.Handle(), 0, sizeof(VSUniformData));
    update_queue.AddBuffer(buffer_set, 2, uniform_buffer.Handle(), 0, sizeof(FSUniformData));
    update_queue.AddTexelBuffer(buffer_set, 3, *texture_lf_view);
    update_queue.AddTexelBuffer(buffer_set, 4, *texture_rg_view);
    update_queue.AddTexelBuffer(buffer_set, 5, *texture_rgba_view);

    const auto texture_set = pipeline_cache.Acquire(DescriptorHeapType::Texture);
    Surface& null_surface = res_cache.GetSurface(VideoCore::NULL_SURFACE_ID);
    Sampler& null_sampler = res_cache.GetSampler(VideoCore::NULL_SAMPLER_ID);

    const vk::ImageView null_surface_view =
        instance.IsNullDescriptorSupported() ? vk::ImageView{} : null_surface.ImageView();
    const vk::ImageView null_surface_storage_view =
        instance.IsNullDescriptorSupported() ? vk::ImageView{} : null_surface.StorageView();

    // Prepare texture and utility descriptor sets.
    for (u32 i = 0; i < 4; i++) {
        update_queue.AddImageSampler(texture_set, i, 0, null_surface_view, null_sampler.Handle());
    }

    const auto utility_set = pipeline_cache.Acquire(DescriptorHeapType::Utility);
    update_queue.AddStorageImage(utility_set, 0, null_surface_storage_view);
    update_queue.AddImageSampler(utility_set, 1, 0, null_surface_view, null_sampler.Handle());
    update_queue.Flush();
}

RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::TickFrame() {
    // The cache frees what the GPU is done with, going by how far its submissions got. That
    // is past anything the render thread still has to pass on, so there is nothing to wait
    // for here. With LibRetro the frontend owns the device and that progress is not known.
#ifdef HAVE_LIBRETRO
    scheduler.WaitWorker();
#else
    if (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyFrameHandoff) {
        scheduler.WaitWorker();
    }
#endif
    res_cache.TickFrame();
    runtime.FreeRetired();
    ++frame_count;
    in_place_data.OnFrame();
    if (watch_data) {
        MasterSemaphore* const semaphore = scheduler.GetMasterSemaphore();
        semaphore->Refresh();
        data_watch.OnFrame(semaphore->KnownGpuTick());
    }
}

void RasterizerVulkan::LoadDefaultDiskResources(
    const std::atomic_bool& stop_loading, const VideoCore::DiskResourceLoadCallback& callback) {

    u64 program_id;
    if (Core::System::GetInstance().GetAppLoader().ReadProgramId(program_id) !=
        Loader::ResultStatus::Success) {
        program_id = 0;
    }

    if (callback) {
        callback(VideoCore::LoadCallbackStage::Prepare, 0, 0, "");
    }

    pipeline_cache.SetAccurateMul(accurate_mul);
    if (pipeline_cache.LoadCacheCrashed(program_id)) {
        LOG_WARNING(Render_Vulkan,
                    "Last load of the shader cache of title_id={:016X} crashed, not loading it",
                    program_id);
    } else {
        pipeline_cache.SetProgramID(program_id);
        pipeline_cache.LoadCache(stop_loading, callback);
    }

    if (callback) {
        callback(VideoCore::LoadCallbackStage::Complete, 0, 0, "");
    }
}

void RasterizerVulkan::SyncDrawState() {
    SyncDrawUniforms();

    // SyncCullMode();
    pipeline_info.state.rasterization.cull_mode.Assign(regs.rasterizer.cull_mode);
    // If the framebuffer is flipped, request to also flip vulkan viewport
    const bool is_flipped = regs.framebuffer.framebuffer.IsFlipped();
    pipeline_info.state.rasterization.flip_viewport.Assign(is_flipped);
    // SyncBlendEnabled();
    pipeline_info.state.blending.blend_enable = regs.framebuffer.output_merger.alphablend_enable;
    // SyncBlendFuncs();
    pipeline_info.state.blending.color_blend_eq.Assign(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_rgb);
    pipeline_info.state.blending.alpha_blend_eq.Assign(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_a);
    pipeline_info.state.blending.src_color_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_source_rgb);
    pipeline_info.state.blending.dst_color_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_dest_rgb);
    pipeline_info.state.blending.src_alpha_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_source_a);
    pipeline_info.state.blending.dst_alpha_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_dest_a);
    // SyncBlendColor();
    const auto blend_color = PicaToVK::ColorRGBA8(regs.framebuffer.output_merger.blend_const.raw);
    if (fs_data.blend_color != blend_color) {
        fs_data.blend_color = blend_color;
        fs_data_dirty = true;
    }
    pipeline_info.dynamic_info.blend_color = regs.framebuffer.output_merger.blend_const.raw;
    // SyncLogicOp();
    // SyncColorWriteMask();
    pipeline_info.state.blending.logic_op = regs.framebuffer.output_merger.logic_op;

    const u32 color_mask = regs.framebuffer.framebuffer.allow_color_write != 0
                               ? (regs.framebuffer.output_merger.depth_color_mask >> 8) & 0xF
                               : 0;
    pipeline_info.state.blending.color_write_mask = color_mask;

    bool rgb_blend_emulation = false;
    bool alpha_blend_emulation = false;
    pipeline_cache.QueryBlendEmulation(regs, rgb_blend_emulation, alpha_blend_emulation);
    pipeline_info.state.blending.rgb_blend_emulation.Assign(rgb_blend_emulation);
    pipeline_info.state.blending.alpha_blend_emulation.Assign(alpha_blend_emulation);

    // SyncStencilTest();
    const auto& stencil_test = regs.framebuffer.output_merger.stencil_test;
    const bool test_enable = stencil_test.enable && regs.framebuffer.framebuffer.depth_format ==
                                                        Pica::FramebufferRegs::DepthFormat::D24S8;

    pipeline_info.state.depth_stencil.stencil_test_enable.Assign(test_enable);
    pipeline_info.state.depth_stencil.stencil_fail_op.Assign(stencil_test.action_stencil_fail);
    pipeline_info.state.depth_stencil.stencil_pass_op.Assign(stencil_test.action_depth_pass);
    pipeline_info.state.depth_stencil.stencil_depth_fail_op.Assign(stencil_test.action_depth_fail);
    pipeline_info.state.depth_stencil.stencil_compare_op.Assign(stencil_test.func);
    pipeline_info.dynamic_info.stencil_reference = stencil_test.reference_value;
    pipeline_info.dynamic_info.stencil_compare_mask = stencil_test.input_mask;
    // SyncStencilWriteMask();
    pipeline_info.dynamic_info.stencil_write_mask =
        (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0)
            ? static_cast<u32>(regs.framebuffer.output_merger.stencil_test.write_mask)
            : 0;
    // SyncDepthTest();
    const bool test_enabled = regs.framebuffer.output_merger.depth_test_enable == 1 ||
                              regs.framebuffer.output_merger.depth_write_enable == 1;
    const auto compare_op = regs.framebuffer.output_merger.depth_test_enable == 1
                                ? regs.framebuffer.output_merger.depth_test_func.Value()
                                : Pica::FramebufferRegs::CompareFunc::Always;

    pipeline_info.state.depth_stencil.depth_test_enable.Assign(test_enabled);
    pipeline_info.state.depth_stencil.depth_compare_op.Assign(compare_op);
    // SyncDepthWriteMask();
    const bool write_enable = (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0 &&
                               regs.framebuffer.output_merger.depth_write_enable);
    pipeline_info.state.depth_stencil.depth_write_enable.Assign(write_enable);
}

void RasterizerVulkan::SetupVertexArray() {
    const auto [vs_input_index_min, vs_input_index_max, vs_input_size] = vertex_info;
    auto [array_ptr, array_offset, invalidate] = stream_buffer.Map(vs_input_size, 16);

    /**
     * The Nintendo 3DS has 12 attribute loaders which are used to tell the GPU
     * how to interpret vertex data. The program firsts sets GPUREG_ATTR_BUF_BASE to the base
     * address containing the vertex array data. The data for each attribute loader (i) can be found
     * by adding GPUREG_ATTR_BUFi_OFFSET to the base address. Attribute loaders can be thought
     * as something analogous to Vulkan bindings. The user can store attributes in separate loaders
     * or interleave them in the same loader.
     **/
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;
    const PAddr base_address = vertex_attributes.GetPhysicalBaseAddress(); // GPUREG_ATTR_BUF_BASE
    const u32 stride_alignment = instance.GetMinVertexStrideAlignment();

    VertexLayout& layout = pipeline_info.state.vertex_layout;
    layout.binding_count = 0;
    layout.attribute_count = 16;
    enable_attributes.fill(false);
    bindings_in_guest_memory = 0;
    if (watch_data) {
        data_watch.NoteDraw();
    }

    u32 buffer_offset = 0;
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        if (loader.component_count == 0 || loader.byte_count == 0) {
            continue;
        }

        // Analyze the attribute loader by checking which attributes it provides
        u32 offset = 0;
        for (u32 comp = 0; comp < loader.component_count && comp < 12; comp++) {
            const u32 attribute_index = loader.GetComponent(comp);
            if (attribute_index >= 12) {
                // Attribute ids 12, to 15 signify 4, 8, 12 and 16-byte paddings respectively.
                offset = Common::AlignUp(offset, 4);
                offset += (attribute_index - 11) * 4;
                continue;
            }

            const u32 size = vertex_attributes.GetNumElements(attribute_index);
            if (size == 0) {
                continue;
            }

            offset =
                Common::AlignUp(offset, vertex_attributes.GetElementSizeInBytes(attribute_index));

            const u32 input_reg = regs.vs.GetRegisterForAttribute(attribute_index);
            const auto format = vertex_attributes.GetFormat(attribute_index);

            VertexAttribute& attribute = layout.attributes[input_reg];
            attribute.binding.Assign(layout.binding_count);
            attribute.location.Assign(input_reg);
            attribute.offset.Assign(offset);
            attribute.type.Assign(format);
            attribute.size.Assign(size);

            enable_attributes[input_reg] = true;
            offset += vertex_attributes.GetStride(attribute_index);
        }

        const PAddr data_addr =
            base_address + loader.data_offset + (vs_input_index_min * loader.byte_count);
        const u32 vertex_num = vs_input_index_max - vs_input_index_min + 1;
        u32 data_size = loader.byte_count * vertex_num;
        memory.NoteGpuReference(data_addr, data_size, Memory::GpuReference::VertexData);
        res_cache.FlushRegion(data_addr, data_size);

        // A plain pointer: a reference would count the memory's users up and down, with
        // the bus locked, for every stream of every draw
        // (From the copy that came with the command, for a page the game keeps writing:
        // that is outside the emulated memory, and so not read in place below)
        const u8* src_ptr = memory.GpuRead(data_addr, data_size);
        u8* dst_ptr = array_ptr + buffer_offset;

        // Create the binding associated with this loader
        VertexBinding& binding = layout.bindings[layout.binding_count];
        binding.binding.Assign(layout.binding_count);
        binding.fixed.Assign(0);
        // Will be adjusted on pipeline build, to keep the info transferable.
        binding.byte_count.Assign(loader.byte_count);

        // Align stride up if required by Vulkan implementation.
        const u32 aligned_stride =
            Common::AlignUp(static_cast<u32>(loader.byte_count), stride_alignment);

        // With the emulated memory a buffer of the host GPU, the data is used where the game
        // has it. Not if the device wants another stride than the game's, and not from an
        // address the copy below would not put it at either: one that is no multiple of four,
        // which a device may refuse for an attribute of that size.
        std::optional<u32> in_place;
        if (aligned_stride == loader.byte_count) {
            in_place = guest_memory.Offset(src_ptr, data_size);
            // And only what the game leaves alone: the GPU reads it later than now
            if (in_place && (*in_place % 4 != 0 || !MayReadInPlace(data_addr, data_size))) {
                in_place.reset();
            }
        }
        if (watch_data) {
            data_watch.NoteUse(src_ptr, data_size, false, in_place.has_value(),
                               scheduler.CurrentTick());
        }
        if (in_place) {
            bindings_in_guest_memory |= static_cast<u16>(1u << layout.binding_count);
            binding_offsets[layout.binding_count++] = *in_place;
            continue;
        }

        if (aligned_stride == loader.byte_count) {
            std::memcpy(dst_ptr, src_ptr, data_size);
        } else {
            for (std::size_t vertex = 0; vertex < vertex_num; vertex++) {
                std::memcpy(dst_ptr + vertex * aligned_stride, src_ptr + vertex * loader.byte_count,
                            loader.byte_count);
            }
        }

        // Keep track of the binding offsets so we can bind the vertex buffer later
        binding_offsets[layout.binding_count++] = static_cast<u32>(array_offset + buffer_offset);
        buffer_offset += Common::AlignUp(aligned_stride * vertex_num, 4);
    }

    stream_buffer.Commit(buffer_offset);

    // Assign the rest of the attributes to the last binding
    SetupFixedAttribs();
}

void RasterizerVulkan::SetupFixedAttribs() {
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;
    VertexLayout& layout = pipeline_info.state.vertex_layout;

    auto [fixed_ptr, fixed_offset, _] = stream_buffer.Map(16 * sizeof(Common::Vec4f), 0);
    binding_offsets[layout.binding_count] = static_cast<u32>(fixed_offset);

    // Reserve the last binding for fixed and default attributes
    // Place the default attrib at offset zero for easy access
    static const Common::Vec4f default_attrib{0.f, 0.f, 0.f, 1.f};
    std::memcpy(fixed_ptr, default_attrib.AsArray(), sizeof(Common::Vec4f));

    // Find all fixed attributes and assign them to the last binding
    u32 offset = sizeof(Common::Vec4f);
    for (std::size_t i = 0; i < 16; i++) {
        if (vertex_attributes.IsDefaultAttribute(i)) {
            const u32 reg = regs.vs.GetRegisterForAttribute(i);
            if (!enable_attributes[reg]) {
                const auto& attr = pica.input_default_attributes[i];
                const std::array data = {attr.x.ToFloat32(), attr.y.ToFloat32(), attr.z.ToFloat32(),
                                         attr.w.ToFloat32()};

                const u32 data_size = sizeof(float) * static_cast<u32>(data.size());
                std::memcpy(fixed_ptr + offset, data.data(), data_size);

                VertexAttribute& attribute = layout.attributes[reg];
                attribute.binding.Assign(layout.binding_count);
                attribute.location.Assign(reg);
                attribute.offset.Assign(offset);
                attribute.type.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
                attribute.size.Assign(4);

                offset += data_size;
                enable_attributes[reg] = true;
            }
        }
    }

    // Loop one more time to find unused attributes and assign them to the default one
    // If the attribute is just disabled, shove the default attribute to avoid
    // errors if the shader ever decides to use it.
    for (u32 i = 0; i < 16; i++) {
        if (!enable_attributes[i]) {
            VertexAttribute& attribute = layout.attributes[i];
            attribute.binding.Assign(layout.binding_count);
            attribute.location.Assign(i);
            attribute.offset.Assign(0);
            attribute.type.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
            attribute.size.Assign(4);
        }
    }

    // Define the fixed+default binding
    VertexBinding& binding = layout.bindings[layout.binding_count];
    binding.binding.Assign(layout.binding_count++);
    binding.fixed.Assign(1);
    binding.byte_count.Assign(offset);

    stream_buffer.Commit(offset);
}

void RasterizerVulkan::LookUpDrawSite() {
    site = nullptr;
    site_matches = false;
    if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyDrawResolve) != 0) {
        return;
    }
    if (draw_sites.empty()) {
        draw_sites.resize(std::size_t{1} << DRAW_SITE_BITS);
    }
    const u64 shaping = ShapingHash();
    const u64 code_hash = pica.vs_setup.GetProgramCodeHash();
    const u64 swizzle_hash = pica.vs_setup.GetSwizzleDataHash();
    const void* const shader_cache = pipeline_cache.DiskCacheIdentity();
    const u64 slot =
        (shaping ^ (code_hash * 0x9E3779B97F4A7C15ull) ^ (swizzle_hash * 0xC2B2AE3D27D4EB4Full)) *
        0xD6E8FEB86659FD93ull;
    DrawSite& entry = draw_sites[slot >> (64 - DRAW_SITE_BITS)];
    site = &entry;

    if (entry.filled && entry.shaping == shaping && entry.vs_code_hash == code_hash &&
        entry.vs_swizzle_hash == swizzle_hash && entry.shader_cache == shader_cache &&
        std::memcmp(&entry.user_config, &user_config, sizeof(user_config)) == 0) {
        site_matches = true;
        return;
    }
    entry = DrawSite{};
    entry.filled = true;
    entry.shaping = shaping;
    entry.vs_code_hash = code_hash;
    entry.vs_swizzle_hash = swizzle_hash;
    entry.shader_cache = shader_cache;
    entry.user_config = user_config;
}

bool RasterizerVulkan::SetupVertexShader() {
    MICROPROFILE_SCOPE(Vulkan_VS);
    if (site_matches && site->has_vs) {
        pipeline_cache.RestoreShader(ProgramType::VS, site->vs);
        return true;
    }
    const bool usable = pipeline_cache.UseProgrammableVertexShader(
        regs, pica.vs_setup, pipeline_info.state.vertex_layout);
    if (usable && site) {
        site->vs = pipeline_cache.ChosenFor(ProgramType::VS);
        site->has_vs = true;
    }
    return usable;
}

bool RasterizerVulkan::SetupGeometryShader() {
    MICROPROFILE_SCOPE(Vulkan_GS);

    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        LOG_ERROR(Render_Vulkan, "Accelerate draw doesn't support geometry shader");
        return false;
    }

    // Enable the quaternion fix-up geometry-shader only if we are actually doing per-fragment
    // lighting and care about proper quaternions. Otherwise just use standard vertex+fragment
    // shaders. We also don't need a geometry shader if the barycentric extension is supported,
    // but that will be decided later as the GS config needs to be cached anyways.
    if (regs.lighting.disable) {
        pipeline_cache.UseTrivialGeometryShader();
        return true;
    }

    if (site_matches && site->has_gs) {
        pipeline_cache.RestoreShader(ProgramType::GS, site->gs);
        return true;
    }
    const bool usable = pipeline_cache.UseFixedGeometryShader(regs);
    if (usable && site) {
        site->gs = pipeline_cache.ChosenFor(ProgramType::GS);
        site->has_gs = true;
    }
    return usable;
}

bool RasterizerVulkan::AccelerateDrawBatch(bool is_indexed) {
    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        if (regs.pipeline.gs_config.mode != Pica::PipelineRegs::GSMode::Point) {
            return false;
        }
        if (regs.pipeline.triangle_topology != Pica::PipelineRegs::TriangleTopology::Shader) {
            return false;
        }
    }

    pipeline_info.state.rasterization.topology.Assign(regs.pipeline.triangle_topology);
    if (regs.pipeline.triangle_topology == TriangleTopology::Fan &&
        !instance.IsTriangleFanSupported()) {
        LOG_DEBUG(Render_Vulkan,
                  "Skipping accelerated draw with unsupported triangle fan topology");
        return false;
    }

    // Vertex data setup might involve scheduler flushes so perform it
    // early to avoid invalidating our state in the middle of the draw.
    LookUpDrawSite();

    // The range of the indices is worked out by reading them all. A draw in a state seen
    // before, whose indices were read in place then and have not been written since (the
    // cache's own writing back included: that ends their being read in place), has the range
    // it had.
    bool scan = true;
    if (is_indexed && site_matches && site->has_index_range) {
        const PAddr index_addr = regs.pipeline.vertex_attributes.GetPhysicalBaseAddress() +
                                 regs.pipeline.index_array.offset;
        const u32 index_size =
            regs.pipeline.num_vertices * (regs.pipeline.index_array.format != 0 ? 2 : 1);
        res_cache.FlushRegion(index_addr, index_size);
        if (site->index_generation == in_place_data.Generation()) {
            vertex_info = MakeVertexArrayInfo(site->index_min, site->index_max,
                                              instance.GetMinVertexStrideAlignment());
            scan = false;
        }
    }
    if (scan) {
        reading_draw_data = true;
        vertex_info = AnalyzeVertexArray(is_indexed, instance.GetMinVertexStrideAlignment());
        reading_draw_data = false;
        if (site) {
            site->has_index_range = false;
            site->index_min = vertex_info.vs_input_index_min;
            site->index_max = vertex_info.vs_input_index_max;
        }
    }
    if (vertex_info.Invalid()) {
        // Do not draw anything if the vertex array is invalid.
        site = nullptr;
        site_matches = false;
        return true;
    }
    SetupVertexArray();

    if (!SetupVertexShader() || !SetupGeometryShader()) {
        site = nullptr;
        site_matches = false;
        return false;
    }

    const bool drawn = Draw(true, is_indexed);
    site = nullptr;
    site_matches = false;
    return drawn;
}

bool RasterizerVulkan::AccelerateDrawBatchInternal(bool is_indexed) {
    if (is_indexed) {
        SetupIndexArray();
    }

    // With shaders compiled in the background a draw whose pipeline is not built yet is left
    // out, to turn up a frame or two later. Small draws used to wait instead, all of them,
    // since among them are the passes over the whole screen; now only those wait that would
    // be missed.
    const u32 vertices = regs.pipeline.num_vertices;
    if (!pipeline_cache.BindPipeline(pipeline_info, !async_shaders, vertices)) {
        if (vertices > 6 || !SmallDrawMustShow(is_indexed)) {
            return true;
        }
        if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacySmallDrawWait) ==
            0) {
            // It has to be drawn now, but not with this pipeline: its vertex shader can take
            // the driver tens of milliseconds, and the render thread would stand still for
            // them. The caller shades the few vertices on the CPU when this fails and draws
            // them with the pass-through vertex shader, which is always there. What is left to
            // wait for is the fragment shader, a few milliseconds, if that is new as well. The
            // pipeline asked for here is built in the background for the frames to come.
            LOG_SHADER_TIMING("Small draw shown with its vertices shaded on the CPU");
            return false;
        }
        if (!pipeline_cache.BindPipeline(pipeline_info, true, vertices)) {
            return true;
        }
    }

    const DrawParams params = {
        .vertex_count = regs.pipeline.num_vertices,
        .vertex_offset = -static_cast<s32>(vertex_info.vs_input_index_min),
        .binding_count = pipeline_info.state.vertex_layout.binding_count,
        .bindings = binding_offsets,
        .in_guest_memory = bindings_in_guest_memory,
        .is_indexed = is_indexed,
    };

    scheduler.Record([this, params](vk::CommandBuffer cmdbuf) {
        std::array<vk::DeviceSize, 16> offsets;
        std::transform(params.bindings.begin(), params.bindings.end(), offsets.begin(),
                       [](u32 offset) { return static_cast<vk::DeviceSize>(offset); });
        if (params.in_guest_memory != 0) {
            std::array<vk::Buffer, 16> buffers = vertex_buffers;
            for (u32 i = 0; i < params.binding_count; i++) {
                if ((params.in_guest_memory >> i) & 1) {
                    buffers[i] = guest_memory.Handle();
                }
            }
            cmdbuf.bindVertexBuffers(0, params.binding_count, buffers.data(), offsets.data());
        } else {
            cmdbuf.bindVertexBuffers(0, params.binding_count, vertex_buffers.data(),
                                     offsets.data());
        }
        if (params.is_indexed) {
            cmdbuf.drawIndexed(params.vertex_count, 1, 0, params.vertex_offset, 0);
        } else {
            cmdbuf.draw(params.vertex_count, 1, 0, 0);
        }
    });

    return true;
}

bool RasterizerVulkan::SmallDrawMustShow(bool is_indexed) {
    if (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacySmallDrawWait) {
        return true;
    }
    // What is drawn once has to be drawn: nothing would bring it back
    if (!target_redrawn) {
        LOG_SHADER_TIMING("Small draw waits for its pipeline: its target is not drawn anew");
        return true;
    }
    // A quarter of the viewport and more is a pass over the screen, which what is drawn
    // after it builds on; less is a sprite, a glyph or a particle
    constexpr float PASS_SHARE = 0.25f;
    const float share = pica.SmallDrawViewportShare(is_indexed);
    if (share >= PASS_SHARE) {
        // Unless all it does is lay something over what is there: blended with the picture
        // below and leaving depth and stencil as they are, a fade, a tint or a glow comes a
        // frame late as unnoticed as a sprite does. What replaces the picture cannot.
        const auto& blending = pipeline_info.state.blending;
        const bool overlay =
            blending.blend_enable != 0 &&
            blending.dst_color_blend_factor != Pica::FramebufferRegs::BlendFactor::Zero &&
            !pipeline_info.IsDepthWriteEnabled();
        LOG_SHADER_TIMING("Small draw {} its pipeline: {:.0f}% of the viewport{}",
                          overlay ? "left out without" : "waits for", share * 100.f,
                          overlay ? ", laid over what is there" : "");
        return !overlay;
    }
    LOG_SHADER_TIMING("Small draw left out without its pipeline: {:.0f}% of the viewport",
                      share * 100.f);
    return false;
}

Common::Rectangle<u32> RasterizerVulkan::WrittenBy(bool accelerate, bool is_indexed,
                                                   const Common::Rectangle<u32>& draw_rect,
                                                   const VideoCore::ViewportInfo& viewport,
                                                   u32 res_scale) {
    // A game that scales an image down step by step inside one target (the levels of a glow,
    // side by side) draws a small quad at each step that samples what the step before drew.
    // For each of them the target's second image is brought up to date where the target was
    // written since the last time: with every draw counted as writing all of the rectangle it
    // is confined to that was the whole screen each time, 25 times a frame in Metal Gear
    // Solid: Snake Eater 3D, more pixels copied than drawn.
    if (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::WholeAreaWritten) {
        return draw_rect;
    }
    // The box around the triangles, in the coordinates of the viewport
    Pica::PicaCore::DrawBox box{1.f, -1.f, 1.f, -1.f};
    if (accelerate) {
        if (!pica.SmallDrawBox(is_indexed, box)) {
            return draw_rect;
        }
    } else {
        // Shaded on the CPU, the vertices are here, of one draw or of several drawn together
        constexpr std::size_t MOST_VERTICES = 384;
        if (vertex_batch.empty() || vertex_batch.size() > MOST_VERTICES) {
            return draw_rect;
        }
        for (const HardwareVertex& vertex : vertex_batch) {
            const f32 w = vertex.position.w;
            const f32 x = vertex.position.x / w;
            const f32 y = vertex.position.y / w;
            if (!(w > 0.f) || !std::isfinite(x) || !std::isfinite(y)) {
                return draw_rect;
            }
            box.left = std::min(box.left, std::clamp(x, -1.f, 1.f));
            box.right = std::max(box.right, std::clamp(x, -1.f, 1.f));
            box.bottom = std::min(box.bottom, std::clamp(y, -1.f, 1.f));
            box.top = std::max(box.top, std::clamp(y, -1.f, 1.f));
        }
    }
    if (!(box.right >= box.left && box.top >= box.bottom)) {
        return draw_rect;
    }
    // The viewport of the host puts -1 at its x and y and 1 at the far side, as the console's
    // does. The host shades the vertices itself and may end up a fraction of a pixel away from
    // where they were found here: a pixel of the console on every side covers that.
    const auto along = [](f32 at, s32 start, s32 length) {
        return static_cast<f32>(start) + (at + 1.f) * 0.5f * static_cast<f32>(length);
    };
    const s32 spare = static_cast<s32>(res_scale) + 1;
    const s32 x0 =
        static_cast<s32>(std::floor(along(box.left, viewport.x, viewport.width))) - spare;
    const s32 x1 =
        static_cast<s32>(std::ceil(along(box.right, viewport.x, viewport.width))) + spare;
    const s32 y0 =
        static_cast<s32>(std::floor(along(box.bottom, viewport.y, viewport.height))) - spare;
    const s32 y1 = static_cast<s32>(std::ceil(along(box.top, viewport.y, viewport.height))) + spare;

    const s32 rect_x0 = static_cast<s32>(std::min(draw_rect.left, draw_rect.right));
    const s32 rect_x1 = static_cast<s32>(std::max(draw_rect.left, draw_rect.right));
    const s32 rect_y0 = static_cast<s32>(std::min(draw_rect.bottom, draw_rect.top));
    const s32 rect_y1 = static_cast<s32>(std::max(draw_rect.bottom, draw_rect.top));
    const u32 left = static_cast<u32>(std::clamp(x0, rect_x0, rect_x1));
    const u32 right = static_cast<u32>(std::clamp(x1, rect_x0, rect_x1));
    const u32 bottom = static_cast<u32>(std::clamp(y0, rect_y0, rect_y1));
    const u32 top = static_cast<u32>(std::clamp(y1, rect_y0, rect_y1));
    if (right - left < draw_rect.GetWidth() || top - bottom < draw_rect.GetHeight()) {
        CITRA_FRAME_DUMP("  vk draw writes only {}x{} at {},{} of its rectangle", right - left,
                         top - bottom, left, bottom);
    }
    return Common::Rectangle<u32>{left, top, right, bottom};
}

void RasterizerVulkan::SetupIndexArray() {
    const bool index_u8 = regs.pipeline.index_array.format == 0;
    const bool native_u8 = index_u8 && instance.IsIndexTypeUint8Supported();
    const u32 index_buffer_size = regs.pipeline.num_vertices * (native_u8 ? 1 : 2);
    const vk::IndexType index_type = native_u8 ? vk::IndexType::eUint8EXT : vk::IndexType::eUint16;

    const PAddr index_addr =
        regs.pipeline.vertex_attributes.GetPhysicalBaseAddress() + regs.pipeline.index_array.offset;
    const u8* index_data =
        memory.GpuRead(index_addr, regs.pipeline.num_vertices * (index_u8 ? 1 : 2));
    memory.NoteGpuReference(index_addr, regs.pipeline.num_vertices * (index_u8 ? 1 : 2),
                            Memory::GpuReference::IndexData);

    // The indices as the game has them, if the device can read them there
    std::optional<u32> in_place;
    if (!index_u8 || native_u8) {
        in_place = guest_memory.Offset(index_data, index_buffer_size);
        if (in_place &&
            ((!index_u8 && *in_place % 2 != 0) || !MayReadInPlace(index_addr, index_buffer_size))) {
            in_place.reset();
        }
    }
    if (watch_data) {
        data_watch.NoteUse(index_data, regs.pipeline.num_vertices * (index_u8 ? 1 : 2), true,
                           in_place.has_value(), scheduler.CurrentTick());
    }
    // Read where they are, the indices cannot change without the watch hearing of it: from
    // here on the range that was worked out for this draw holds until it does
    if (site && !site->has_index_range) {
        site->has_index_range = in_place.has_value();
        site->index_generation = in_place_data.Generation();
    }
    if (in_place) {
        scheduler.Record([buffer = guest_memory.Handle(), offset = *in_place,
                          index_type](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindIndexBuffer(buffer, offset, index_type);
        });
        return;
    }

    auto [index_ptr, index_offset, _] = stream_buffer.Map(index_buffer_size, 2);

    if (index_u8 && !native_u8) {
        u16* index_ptr_u16 = reinterpret_cast<u16*>(index_ptr);
        for (u32 i = 0; i < regs.pipeline.num_vertices; i++) {
            index_ptr_u16[i] = index_data[i];
        }
    } else {
        std::memcpy(index_ptr, index_data, index_buffer_size);
    }

    stream_buffer.Commit(index_buffer_size);

    scheduler.Record(
        [this, index_offset = index_offset, index_type = index_type](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindIndexBuffer(stream_buffer.Handle(), index_offset, index_type);
        });
}

void RasterizerVulkan::DrawTriangles() {
    if (vertex_batch.empty()) {
        return;
    }

    pipeline_info.state.rasterization.topology.Assign(Pica::PipelineRegs::TriangleTopology::List);
    pipeline_info.state.vertex_layout = software_layout;

    pipeline_cache.UseTrivialVertexShader();
    pipeline_cache.UseTrivialGeometryShader();

    Draw(false, false);
}

void RasterizerVulkan::NoteDrawOverwrites(bool color, bool depth) {
    next_draw_overwrites_color = color;
    next_draw_overwrites_depth = depth;
}

bool RasterizerVulkan::Draw(bool accelerate, bool is_indexed) {
    MICROPROFILE_SCOPE(Vulkan_Drawing);
    SyncDrawState();

    const bool shadow_rendering = regs.framebuffer.IsShadowRendering();
    const bool has_stencil = regs.framebuffer.HasStencil();

    const bool write_color_fb = shadow_rendering || pipeline_info.GetFinalColorWriteMask(instance);
    const bool write_depth_fb = pipeline_info.IsDepthWriteEnabled();
    const bool needs_color_fb =
        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress() != 0 && write_color_fb;
    const bool needs_depth_fb =
        !shadow_rendering && regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress() != 0 &&
        (write_depth_fb || regs.framebuffer.output_merger.depth_test_enable != 0 ||
         (has_stencil && pipeline_info.state.depth_stencil.stencil_test_enable));

    // Games alternate between draws that only touch depth and draws that touch colour (depth
    // pre-passes, occluders for glows, 2D layers). Selecting a different attachment set for each
    // would end the render pass every time, which on tile-based GPUs means storing and reloading
    // the whole target. So an attachment stays bound for as long as consecutive draws render to
    // the same guest target; the write masks and depth state already keep it untouched.
    const auto& fb_regs = regs.framebuffer.framebuffer;
    const BoundTarget target{
        .color_addr = fb_regs.GetColorBufferPhysicalAddress(),
        .depth_addr = fb_regs.GetDepthBufferPhysicalAddress(),
        .width = fb_regs.GetWidth(),
        .height = fb_regs.GetHeight(),
        .color_format = static_cast<u32>(fb_regs.color_format.Value()),
        .depth_format = static_cast<u32>(fb_regs.depth_format.Value()),
    };
    const bool same_target =
        bound_target.valid && !shadow_rendering && bound_target.color_addr == target.color_addr &&
        bound_target.depth_addr == target.depth_addr && bound_target.width == target.width &&
        bound_target.height == target.height && bound_target.color_format == target.color_format &&
        bound_target.depth_format == target.depth_format &&
        !(Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyAttachments);
    const bool using_color_fb = needs_color_fb || (same_target && bound_target.color);
    const bool using_depth_fb = needs_depth_fb || (same_target && bound_target.depth);
    if (!same_target || target_use_frame != frame_count) {
        TargetUse& use = target_use[target.color_addr != 0 ? target.color_addr : target.depth_addr];
        if (use.last != frame_count) {
            use.before = use.last;
            use.last = frame_count;
        }
        // Drawn to a few frames ago too: a game at 20 frames a second draws every third
        target_redrawn = use.before != 0 && frame_count - use.before <= 4;
        target_use_frame = frame_count;
    }
    bound_target = target;
    bound_target.color = using_color_fb;
    bound_target.depth = using_depth_fb;
    bound_target.valid = !shadow_rendering;

    // Getting the target ready brings to it what the game filled its memory with. That is
    // held back until the render pass begins, which can then clear instead of load.
    renderpass_cache.SetClearDeferral(true);
    // A draw that replaces depth and stencil in all of its viewport needs nothing of what the
    // depth buffer held there, which can be costly to bring (RasterizerCache::ValidateSurface)
    const bool depth_overwritten =
        next_draw_overwrites_depth && write_depth_fb && !shadow_rendering &&
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::LoadOverwrittenDepth) == 0;
    auto fb_helper =
        res_cache.GetFramebufferSurfaces(using_color_fb, using_depth_fb, depth_overwritten);
    fb_helper.SetWrittenAttachments(needs_color_fb, needs_depth_fb);
    const Framebuffer* framebuffer = fb_helper.Framebuffer();
    if (!framebuffer->Handle()) {
        renderpass_cache.SetClearDeferral(false);
        return true;
    }

    const auto draw_rect = fb_helper.DrawRect();
    if (draw_rect.GetArea() == 0) {
        renderpass_cache.SetClearDeferral(false);
        return true;
    }

    pipeline_info.state.attachments.color = framebuffer->Format(SurfaceType::Color);
    pipeline_info.state.attachments.depth = framebuffer->Format(SurfaceType::Depth);
    pipeline_info.state.attachments.sample_count = framebuffer->Samples();

    // Update scissor uniforms
    const auto [scissor_x1, scissor_y2, scissor_x2, scissor_y1] = fb_helper.Scissor();
    if (fs_data.scissor_x1 != scissor_x1 || fs_data.scissor_x2 != scissor_x2 ||
        fs_data.scissor_y1 != scissor_y1 || fs_data.scissor_y2 != scissor_y2) {

        fs_data.scissor_x1 = scissor_x1;
        fs_data.scissor_x2 = scissor_x2;
        fs_data.scissor_y1 = scissor_y1;
        fs_data.scissor_y2 = scissor_y2;
        fs_data_dirty = true;
    }

    // Sync and bind the texture surfaces
    const bool overwrites_color = next_draw_overwrites_color && write_color_fb && !shadow_rendering;
    const bool overwrites_depth = next_draw_overwrites_depth && write_depth_fb && !shadow_rendering;
    if (SyncTextureUnits(framebuffer, draw_rect, overwrites_color)) {
        framebuffer = fb_helper.Refresh();
    }
    SyncUtilityTextures(framebuffer);

    // Sync and bind the shader. With Z-buffering the viewport applies the PICA depth range, so
    // the shader does not write depth and the host GPU can reject hidden fragments early.
    const auto depth_range =
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyDepthWrite)
            ? std::nullopt
            : Pica::Shader::GetViewportDepthRange(regs.rasterizer);
    // A scissor that keeps the inside of a rectangle is done with the scissor of the draw
    const bool host_scissor =
        regs.rasterizer.scissor_test.mode == Pica::RasterizerRegs::ScissorMode::Include &&
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyScissor) == 0;
    // A draw that writes no colour (a stencil mask, depth-only geometry) still has the scene's
    // lighting and combiner state set. Unless the alpha test can drop fragments by it, or the
    // shader writes a shadow map itself, what that state computes goes nowhere: such a draw
    // gets a shader without it. A GPU that cannot leave an unused fragment shader out runs it
    // for every pixel otherwise (an Adreno 6xx does: 1.9 ms for one mask of 1.2 million pixels
    // in Luigi's Mansion: Dark Moon at 5x).
    const auto& alpha_test = regs.framebuffer.output_merger.alpha_test;
    const bool color_unused =
        !write_color_fb && !shadow_rendering &&
        (alpha_test.enable == 0 || alpha_test.func == Pica::FramebufferRegs::CompareFunc::Always) &&
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyColourlessShader) == 0;
    if (site_matches && site->has_fs && site->fs_depth_from_viewport == depth_range.has_value() &&
        site->fs_host_scissor == host_scissor && site->fs_color_unused == color_unused) {
        pipeline_cache.RestoreShader(ProgramType::FS, site->fs);
    } else {
        pipeline_cache.UseFragmentShader(regs, user_config, depth_range.has_value(), host_scissor,
                                         color_unused);
        if (site) {
            site->fs = pipeline_cache.ChosenFor(ProgramType::FS);
            site->has_fs = true;
            site->fs_depth_from_viewport = depth_range.has_value();
            site->fs_host_scissor = host_scissor;
            site->fs_color_unused = color_unused;
        }
    }
    pipeline_info.dynamic_info.depth_min = depth_range ? depth_range->min_depth : 0.f;
    pipeline_info.dynamic_info.depth_max = depth_range ? depth_range->max_depth : 1.f;

    // Sync the LUTs within the texture buffer
    SyncAndUploadLUTs();
    SyncAndUploadLUTsLF();
    UploadUniforms(accelerate);

    // Begin rendering
    CITRA_FRAME_DUMP("  vk draw rect {},{} {}x{}", draw_rect.left, draw_rect.bottom,
                     draw_rect.GetWidth(), draw_rect.GetHeight());
#ifdef CITRA_ENABLE_TRACE_MARKERS
    if (VideoCore::FrameStateDump::IsActive() && !regs.lighting.disable) {
        // What the lighting tables of this draw hold: the range of their values, and whether
        // they are flat (one value, no slope), in which case a lookup tells nothing
        const auto describe = [&](std::string_view name, u32 index) {
            const auto& lut = pica.lighting.luts[index];
            f32 low = 1.f, high = 0.f;
            bool flat = true;
            for (const auto& entry : lut) {
                low = std::min(low, entry.ToFloat());
                high = std::max(high, entry.ToFloat());
                flat = flat && entry.DiffToFloat() == 0.f;
            }
            flat = flat && low == high;
            return fmt::format(" {}={:.3f}..{:.3f}{}", name, low, high, flat ? " flat" : "");
        };
        std::string text = "  luts:";
        text += describe("D0", 0) + describe("D1", 1) + describe("FR", 3) + describe("RB", 4) +
                describe("RG", 5) + describe("RR", 6);
        for (u32 i = 0; i <= regs.lighting.max_light_index; i++) {
            const u32 num = regs.lighting.light_enable.GetNum(i);
            text += describe(fmt::format("SP{}", num), 8 + num);
            text += describe(fmt::format("DA{}", num), 16 + num);
        }
        CITRA_FRAME_DUMP("{}", text);
    }
#endif
    if (framebuffer->color_id) {
        // Where the colour target is written: its second image is kept up with it only there
        Surface& color_target = res_cache.GetSurface(framebuffer->color_id);
        if (shadow_rendering) {
            color_target.NoteAllWritten();
        } else if (color_target.TracksWritten()) {
            color_target.NoteWritten(WrittenBy(accelerate, is_indexed, draw_rect,
                                               fb_helper.Viewport(), color_target.GetResScale()));
        } else {
            color_target.NoteWritten(draw_rect);
        }
    }
    // The draw is confined to that rectangle. If it overwrites all of it and a render pass
    // begins with it, there is nothing in the rectangle for the pass to load.
    renderpass_cache.BeginRendering(framebuffer, draw_rect, overwrites_color, overwrites_depth);
    renderpass_cache.SetClearDeferral(false);

    // Configure viewport and scissor
    const auto viewport = fb_helper.Viewport();
    pipeline_info.dynamic_info.viewport = Common::Rectangle<s32>{
        viewport.x,
        viewport.y,
        viewport.x + viewport.width,
        viewport.y + viewport.height,
    };
    pipeline_info.dynamic_info.scissor = draw_rect;
    if (host_scissor) {
        // The same pixels the shader's test would keep: those from x1, y1 up to x2, y2
        const s32 left = std::max(static_cast<s32>(draw_rect.left), scissor_x1);
        const s32 right = std::max(left, std::min(static_cast<s32>(draw_rect.right), scissor_x2));
        const s32 bottom = std::max(static_cast<s32>(draw_rect.bottom), scissor_y1);
        const s32 top = std::max(bottom, std::min(static_cast<s32>(draw_rect.top), scissor_y2));
        pipeline_info.dynamic_info.scissor =
            Common::Rectangle<u32>{static_cast<u32>(left), static_cast<u32>(top),
                                   static_cast<u32>(right), static_cast<u32>(bottom)};
    }

    // Draw the vertex batch
    bool succeeded = true;
    if (accelerate) {
        succeeded = AccelerateDrawBatchInternal(is_indexed);
    } else {
        pipeline_cache.BindPipeline(pipeline_info, true);

        const u32 vertex_count = static_cast<u32>(vertex_batch.size());
        const u32 vertex_size = vertex_count * sizeof(HardwareVertex);
        const auto [buffer, offset, _] = stream_buffer.Map(vertex_size, sizeof(HardwareVertex));

        std::memcpy(buffer, vertex_batch.data(), vertex_size);
        stream_buffer.Commit(vertex_size);

        scheduler.Record([this, offset = offset, vertex_count](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindVertexBuffers(0, stream_buffer.Handle(), offset);
            cmdbuf.draw(vertex_count, 1, 0, 0);
        });
    }

    vertex_batch.clear();
    CITRA_FRAME_DUMP_GPU_TIME_OF_DRAW(scheduler);
    return succeeded;
}

bool RasterizerVulkan::SyncTextureUnits(const Framebuffer* framebuffer,
                                        const Common::Rectangle<u32>& draw_rect,
                                        bool overwrites_color) {
    using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;
    // The view the colour surface is drawn to through, before and after it changed images
    const vk::ImageView color_view = framebuffer->ImageView(SurfaceType::Color);
    vk::ImageView swapped_color_view{};
    vk::ImageView swapped_sample_view{};
    const bool may_swap =
        overwrites_color && framebuffer->Samples() == 1 && framebuffer->color_level == 0 &&
        !pipeline_info.state.blending.rgb_blend_emulation &&
        !pipeline_info.state.blending.alpha_blend_emulation &&
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyAttachments) == 0;

    const auto pica_textures = regs.texturing.GetTextures();

    // What each binding of the set is to hold is gathered first. Most draws ask for what the
    // draw before them had, and then that draw's set is used again instead of a new one being
    // taken and written; the writing alone was 2% of the emulation thread in one game. Not
    // with a cube in unit 0, which fills its bindings itself and needs the set for that.
    const bool cube =
        pica_textures[0].enabled && (pica_textures[0].config.type == TextureType::ShadowCube ||
                                     pica_textures[0].config.type == TextureType::TextureCube);
    std::array<TextureBinding, 4> wanted{};
    bool reusable = !cube;
    vk::DescriptorSet texture_set{};
    if (cube) {
        texture_set = pipeline_cache.Acquire(DescriptorHeapType::Texture);
    }
    const auto bind = [&](u32 binding, vk::ImageView view, vk::Sampler sampler) {
        if (cube) {
            update_queue.AddImageSampler(texture_set, binding, 0, view, sampler);
        } else {
            wanted[binding] = TextureBinding{view, sampler};
        }
    };

    for (u32 texture_index = 0; texture_index < pica_textures.size(); ++texture_index) {
        const auto& texture = pica_textures[texture_index];

        // If the texture unit is disabled bind a null surface to it
        if (!texture.enabled) {
            switch (texture.config.type.Value()) {
            case TextureType::TextureCube:
            case TextureType::ShadowCube: {
                const Sampler& null_sampler = res_cache.GetSampler(VideoCore::NULL_SURFACE_CUBE_ID);
                if (instance.IsNullDescriptorSupported()) {
                    bind(texture_index, vk::ImageView{}, null_sampler.Handle());
                } else {
                    Surface& null_surface = res_cache.GetSurface(VideoCore::NULL_SURFACE_CUBE_ID);
                    bind(texture_index, null_surface.ImageView(), null_sampler.Handle());
                }
                break;
            }
            default: {
                const Sampler& null_sampler = res_cache.GetSampler(VideoCore::NULL_SURFACE_ID);
                if (instance.IsNullDescriptorSupported()) {
                    bind(texture_index, vk::ImageView{}, null_sampler.Handle());
                } else {
                    Surface& null_surface = res_cache.GetSurface(VideoCore::NULL_SURFACE_ID);
                    bind(texture_index, null_surface.ImageView(), null_sampler.Handle());
                }
                break;
            }
            }
            continue;
        }

        // Handle special tex0 configurations
        if (texture_index == 0) {
            switch (texture.config.type.Value()) {
            case TextureType::Shadow2D: {
                Surface& surface = res_cache.GetTextureSurface(texture);
                // The shader looks the shadow map up by texel, and is told its size
                const auto shadow_extent = surface.RealExtent();
                const Common::Vec4f shadow_size{static_cast<f32>(shadow_extent.width),
                                                static_cast<f32>(shadow_extent.height),
                                                1.f / static_cast<f32>(shadow_extent.width),
                                                1.f / static_cast<f32>(shadow_extent.height)};
                if (fs_data.tex_size[texture_index] != shadow_size) {
                    fs_data.tex_size[texture_index] = shadow_size;
                    fs_data_dirty = true;
                }
                // The shader gathers the texels, which hold depths as integers: a sampler
                // that filters has no business with those
                Sampler& sampler = res_cache.GetSampler(texture.config, true);
                surface.flags |= VideoCore::SurfaceFlagBits::ShadowSource;
                bind(texture_index, surface.StorageView(), sampler.Handle());
                continue;
            }
            case TextureType::ShadowCube: {
                BindShadowCube(texture, texture_set);
                continue;
            }
            case TextureType::TextureCube: {
                BindTextureCube(texture, texture_set);
                continue;
            }
            default:
                break;
            }
        }

        // Bind the texture provided by the rasterizer cache
        Surface& surface = res_cache.GetTextureSurface(texture);
        Sampler& sampler = res_cache.GetSampler(texture.config);
        // The fragment shader works the level of detail out from the size of the host texture,
        // which it is told instead of asking the texture for it at every fragment
        const auto extent = surface.RealExtent();
        const Common::Vec4f tex_size{static_cast<f32>(extent.width),
                                     static_cast<f32>(extent.height), 0.f, 0.f};
        if (fs_data.tex_size[texture_index] != tex_size) {
            fs_data.tex_size[texture_index] = tex_size;
            fs_data_dirty = true;
        }
        const vk::ImageView surface_view = surface.FramebufferView();
        vk::ImageView texture_view;
        if (swapped_color_view && surface_view == swapped_color_view) {
            // Another unit samples the colour target as well
            texture_view = swapped_sample_view;
        } else if (surface_view != color_view) {
            texture_view = surface.ImageView();
        } else if (may_swap && surface.CanSwapForOverwrite()) {
            // The draw samples what it draws to and leaves nothing of it in the rectangle it
            // is confined to: a full-screen effect done in place. Instead of copying all of
            // the target for it to sample, it draws to the target's second image.
            CITRA_FRAME_DUMP("  vk tex{} is the colour target, all overwritten: the target moves "
                             "to its other image",
                             texture_index);
            reusable = false;
            swapped_sample_view = surface.SwapForOverwrite(draw_rect);
            swapped_color_view = surface.FramebufferView();
            texture_view = swapped_sample_view;
        } else {
            CITRA_FRAME_DUMP("  vk tex{} is the colour target: sampling a copy", texture_index);
            reusable = false;
            texture_view = surface.CopyImageView();
        }
        bind(texture_index, texture_view, sampler.Handle());
    }

    if ((pipeline_info.state.blending.rgb_blend_emulation ||
         pipeline_info.state.blending.alpha_blend_emulation) &&
        framebuffer->color_id) {
        const Sampler& null_sampler = res_cache.GetSampler(VideoCore::NULL_SAMPLER_ID);
        Surface& color_surface = res_cache.GetSurface(framebuffer->color_id);
        reusable = false;
        CITRA_FRAME_DUMP("  vk blend emulation: shader reads a copy of the colour target");
        bind(3, color_surface.CopyImageView(), null_sampler.Handle());
    }
    if (cube) {
        last_texture_set.valid = false;
    } else {
        const u64 tick = scheduler.CurrentTick();
        const bool same =
            reusable && last_texture_set.valid && last_texture_set.tick == tick &&
            last_texture_set.bindings == wanted &&
            (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyTextureSets) == 0;
        if (!same) {
            texture_set = pipeline_cache.Acquire(DescriptorHeapType::Texture);
            for (u32 binding = 0; binding < 3; binding++) {
                update_queue.AddImageSampler(texture_set, binding, 0, wanted[binding].view,
                                             wanted[binding].sampler);
            }
            if (wanted[3].sampler) {
                update_queue.AddImageSampler(texture_set, 3, 0, wanted[3].view, wanted[3].sampler);
            }
            last_texture_set = LastTextureSet{wanted, tick, reusable};
        }
    }
    return static_cast<bool>(swapped_color_view);
}

void RasterizerVulkan::SyncUtilityTextures(const Framebuffer* framebuffer) {
    const bool shadow_writing = regs.framebuffer.IsShadowRendering();
    bool shadow_reading = regs.lighting.config0.enable_shadow;
    // Ensure the shadow-texture slot is actually enabled
    if (shadow_reading) {
        const u32 shadow_texture_unit = regs.lighting.config0.shadow_selector.Value();
        const auto shadow_texture = regs.texturing.GetTextures()[shadow_texture_unit];
        shadow_reading &= shadow_texture.enabled;
    }

    const auto utility_set = pipeline_cache.Acquire(DescriptorHeapType::Utility);

    // Reading and writing are mutually exclusive
    assert(!(shadow_writing && shadow_reading));

    if (shadow_writing) {
        update_queue.AddStorageImage(utility_set, 0, framebuffer->ImageView(SurfaceType::Color));
    } else if (shadow_reading) {
        const u32 shadow_texture_unit = regs.lighting.config0.shadow_selector.Value();
        const auto shadow_texture = regs.texturing.GetTextures()[shadow_texture_unit];
        Surface& shadow_surface = res_cache.GetTextureSurface(shadow_texture);
        update_queue.AddStorageImage(utility_set, 0, shadow_surface.StorageView());
    } else {
        if (instance.IsNullDescriptorSupported()) {
            update_queue.AddStorageImage(utility_set, 0, vk::ImageView{});
        } else {
            Surface& null_surface = res_cache.GetSurface(VideoCore::NULL_SURFACE_ID);
            update_queue.AddStorageImage(utility_set, 0, null_surface.StorageView());
        }
    }
}

void RasterizerVulkan::BindShadowCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                                      vk::DescriptorSet texture_set) {
    using CubeFace = Pica::TexturingRegs::CubeFace;
    auto info = Pica::Texture::TextureInfo::FromPicaRegister(texture.config, texture.format);
    constexpr std::array faces = {
        CubeFace::PositiveX, CubeFace::NegativeX, CubeFace::PositiveY,
        CubeFace::NegativeY, CubeFace::PositiveZ, CubeFace::NegativeZ,
    };

    Sampler& sampler = res_cache.GetSampler(texture.config);

    for (CubeFace face : faces) {
        const u32 binding = static_cast<u32>(face);
        info.physical_address = regs.texturing.GetCubePhysicalAddress(face);

        const VideoCore::SurfaceId surface_id = res_cache.GetTextureSurface(info);
        Surface& surface = res_cache.GetSurface(surface_id);
        surface.flags |= VideoCore::SurfaceFlagBits::ShadowSource;
        update_queue.AddImageSampler(texture_set, 0, binding, surface.StorageView(),
                                     sampler.Handle());
    }
}

void RasterizerVulkan::BindTextureCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                                       vk::DescriptorSet texture_set) {
    using CubeFace = Pica::TexturingRegs::CubeFace;
    const VideoCore::TextureCubeConfig config = {
        .px = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveX),
        .nx = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeX),
        .py = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveY),
        .ny = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeY),
        .pz = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveZ),
        .nz = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeZ),
        .width = texture.config.width,
        .levels = texture.config.lod.max_level + 1,
        .format = texture.format,
    };

    Surface& surface = res_cache.GetTextureCube(config);
    Sampler& sampler = res_cache.GetSampler(texture.config);
    update_queue.AddImageSampler(texture_set, 0, 0, surface.ImageView(), sampler.Handle());
}

// Anything that can modify surfaces from outside a draw ends the run of consecutive draws that
// bound_target tracks, so the next draw selects exactly the attachments it needs. Flushes only
// read surfaces back (and run for every indexed draw's index buffer), so they do not.

void RasterizerVulkan::FlushAll() {
    res_cache.FlushAll();
}

void RasterizerVulkan::SubmitPendingWork() {
    scheduler.Flush();
}

void RasterizerVulkan::FlushRegion(PAddr addr, u32 size) {
    if (!reading_draw_data) {
        in_place_data.OnRead(addr, size);
        if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::MeasureCommandLists) !=
                0 &&
            res_cache.HoldsNewer(addr, size)) [[unlikely]] {
            Common::Hitches::NoteGpuCall(Common::Hitches::GpuCall::MemoryReadNewer);
        }
        if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyCachedPageReads) ==
            0) {
            res_cache.NoteCpuRead(addr, size);
        }
        res_cache.NoteReadbackForGame(addr, size);
        CITRA_FRAME_DUMP(
            "game reads {:08X}+{}: the GPU holds newer there {}, on its page {}", addr, size,
            res_cache.HoldsNewer(addr, size),
            res_cache.HoldsNewer(addr & ~Memory::CITRA_PAGE_MASK, Memory::CITRA_PAGE_SIZE));
    }
    res_cache.FlushRegion(addr, size);
}

bool RasterizerVulkan::MayReadInPlace(PAddr addr, u32 size) {
    if (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::CopiedVertexData)
        [[unlikely]] {
        return false;
    }
    return in_place_data.Use(addr, size, scheduler.CurrentTick());
}

void RasterizerVulkan::WaitForDataInPlace(PAddr addr, u32 size) {
    const u64 tick =
        in_place_data.OnWrite(addr, size, scheduler.GetMasterSemaphore()->KnownGpuTick());
    if (tick == 0) {
        return;
    }
    // A draw the GPU is not through with reads what is about to be written, where it is
    const auto start = std::chrono::steady_clock::now();
    scheduler.Wait(tick);
    in_place_data.NoteWait(std::chrono::steady_clock::now() - start);
}

void RasterizerVulkan::InvalidateRegion(PAddr addr, u32 size) {
    bound_target.valid = false;
    if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::MeasureCommandLists) != 0 &&
        res_cache.TouchesSurface(addr, size)) [[unlikely]] {
        Common::Hitches::NoteGpuCall(Common::Hitches::GpuCall::MemoryWriteSurface);
    }
    WaitForDataInPlace(addr, size);
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerVulkan::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    bound_target.valid = false;
    WaitForDataInPlace(addr, size);
    res_cache.FlushRegion(addr, size);
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerVulkan::ClearAll(bool flush) {
    bound_target.valid = false;
    res_cache.ClearAll(flush);
    in_place_data.Reset();
}

bool RasterizerVulkan::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    bound_target.valid = false;
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "what came before this transfer");
    const bool done = res_cache.AccelerateDisplayTransfer(config);
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "transfer");
    return done;
}

bool RasterizerVulkan::AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) {
    bound_target.valid = false;
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "what came before this texture copy");
    const bool done = res_cache.AccelerateTextureCopy(config);
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "texture copy");
    return done;
}

bool RasterizerVulkan::AccelerateFill(const Pica::MemoryFillConfig& config) {
    bound_target.valid = false;
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "what came before this fill");
    const bool done = res_cache.AccelerateFill(config);
    CITRA_FRAME_DUMP_GPU_TIME(scheduler, "fill");
    return done;
}

bool RasterizerVulkan::AccelerateDisplay(const Pica::FramebufferConfig& config,
                                         PAddr framebuffer_addr, u32 pixel_stride,
                                         ScreenInfo& screen_info) {
    if (framebuffer_addr == 0) [[unlikely]] {
        return false;
    }

    VideoCore::SurfaceParams src_params;
    src_params.addr = framebuffer_addr;
    src_params.width = std::min(config.width.Value(), pixel_stride);
    src_params.height = config.height;
    src_params.stride = pixel_stride;
    src_params.is_tiled = false;
    src_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.color_format);
    src_params.UpdateParams();

    const auto [src_surface_id, src_rect] =
        res_cache.GetSurfaceSubRect(src_params, VideoCore::ScaleMatch::Ignore, true);

    if (!src_surface_id) {
        return false;
    }

    Surface& src_surface = res_cache.GetSurface(src_surface_id);
    const u32 scaled_width = src_surface.GetScaledWidth();
    const u32 scaled_height = src_surface.GetScaledHeight();

    screen_info.texcoords = Common::Rectangle<f32>(
        (float)src_rect.bottom / (float)scaled_height, (float)src_rect.left / (float)scaled_width,
        (float)src_rect.top / (float)scaled_height, (float)src_rect.right / (float)scaled_width);

    screen_info.image_view = src_surface.ImageView();

    return true;
}

void RasterizerVulkan::MakeSoftwareVertexLayout() {
    constexpr std::array sizes = {4, 4, 2, 2, 2, 1, 4, 3};

    software_layout = VertexLayout{
        .binding_count = 1,
        .attribute_count = 8,
    };

    for (u32 i = 0; i < software_layout.binding_count; i++) {
        VertexBinding& binding = software_layout.bindings[i];
        binding.binding.Assign(i);
        binding.fixed.Assign(0);
        binding.byte_count.Assign(sizeof(HardwareVertex));
    }

    u32 offset = 0;
    for (u32 i = 0; i < 8; i++) {
        VertexAttribute& attribute = software_layout.attributes[i];
        attribute.binding.Assign(0);
        attribute.location.Assign(i);
        attribute.offset.Assign(offset);
        attribute.type.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
        attribute.size.Assign(sizes[i]);
        offset += sizes[i] * sizeof(float);
    }
}

void RasterizerVulkan::SyncAndUploadLUTsLF() {
    constexpr std::size_t max_size =
        sizeof(Common::Vec2f) * 256 * Pica::LightingRegs::NumLightingSampler +
        sizeof(Common::Vec2f) * 128; // fog

    if (!pica.lighting.lut_dirty && !pica.fog.lut_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto [buffer, offset, invalidate] = texture_lf_buffer.Map(max_size, sizeof(Common::Vec4f));

    if (invalidate) {
        pica.lighting.lut_dirty = pica.lighting.LutAllDirty;
        pica.fog.lut_dirty = true;
    }

    // Sync the lighting luts
    while (pica.lighting.lut_dirty) {
        u32 index = std::countr_zero(pica.lighting.lut_dirty);
        pica.lighting.lut_dirty &= ~(1 << index);

        Common::Vec2f* new_data = reinterpret_cast<Common::Vec2f*>(buffer + bytes_used);
        const auto& source_lut = pica.lighting.luts[index];
        for (u32 i = 0; i < source_lut.size(); i++) {
            new_data[i] = {source_lut[i].ToFloat(), source_lut[i].DiffToFloat()};
        }
        fs_data.lighting_lut_offset[index / 4][index % 4] =
            static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
        fs_data_dirty = true;
        bytes_used += source_lut.size() * sizeof(Common::Vec2f);
    }

    // Sync the fog lut
    if (pica.fog.lut_dirty) {
        Common::Vec2f* new_data = reinterpret_cast<Common::Vec2f*>(buffer + bytes_used);
        for (u32 i = 0; i < pica.fog.lut.size(); i++) {
            new_data[i] = {pica.fog.lut[i].ToFloat(), pica.fog.lut[i].DiffToFloat()};
        }
        fs_data.fog_lut_offset = static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
        fs_data_dirty = true;
        bytes_used += pica.fog.lut.size() * sizeof(Common::Vec2f);
        pica.fog.lut_dirty = false;
    }

    texture_lf_buffer.Commit(static_cast<u32>(bytes_used));
}

void RasterizerVulkan::SyncAndUploadLUTs() {
    const auto& proctex = pica.proctex;
    constexpr std::size_t max_size =
        sizeof(Common::Vec2f) * 128 * 3 + // proctex: noise + color + alpha
        sizeof(Common::Vec4f) * 256 +     // proctex
        sizeof(Common::Vec4f) * 256;      // proctex diff

    if (!pica.proctex.table_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto [buffer, offset, invalidate] = texture_buffer.Map(max_size, sizeof(Common::Vec4f));

    if (invalidate) {
        pica.proctex.table_dirty = pica.proctex.TableAllDirty;
    }

    // helper function for SyncProcTexNoiseLUT/ColorMap/AlphaMap
    const auto sync_proctex_value_lut =
        [&](const std::array<Pica::PicaCore::ProcTex::ValueEntry, 128>& lut, int& lut_offset) {
            Common::Vec2f* new_data = reinterpret_cast<Common::Vec2f*>(buffer + bytes_used);
            for (u32 i = 0; i < lut.size(); i++) {
                new_data[i] = {lut[i].ToFloat(), lut[i].DiffToFloat()};
            }
            lut_offset = static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
            fs_data_dirty = true;
            bytes_used += lut.size() * sizeof(Common::Vec2f);
        };

    // Sync the proctex noise lut
    if (pica.proctex.noise_lut_dirty) {
        sync_proctex_value_lut(proctex.noise_table, fs_data.proctex_noise_lut_offset);
    }

    // Sync the proctex color map
    if (pica.proctex.color_map_dirty) {
        sync_proctex_value_lut(proctex.color_map_table, fs_data.proctex_color_map_offset);
    }

    // Sync the proctex alpha map
    if (pica.proctex.alpha_map_dirty) {
        sync_proctex_value_lut(proctex.alpha_map_table, fs_data.proctex_alpha_map_offset);
    }

    // Sync the proctex lut
    if (pica.proctex.color_lut_dirty) {
        Common::Vec4f* new_data = reinterpret_cast<Common::Vec4f*>(buffer + bytes_used);
        for (u32 i = 0; i < proctex.color_table.size(); i++) {
            new_data[i] = proctex.color_table[i].ToVector() / 255.0f;
        }
        fs_data.proctex_lut_offset =
            static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
        fs_data_dirty = true;
        bytes_used += proctex.color_table.size() * sizeof(Common::Vec4f);
    }

    // Sync the proctex difference lut
    if (pica.proctex.color_diff_lut_dirty) {
        Common::Vec4f* new_data = reinterpret_cast<Common::Vec4f*>(buffer + bytes_used);
        for (u32 i = 0; i < proctex.color_diff_table.size(); i++) {
            new_data[i] = proctex.color_diff_table[i].ToVector() / 255.0f;
        }
        fs_data.proctex_diff_lut_offset =
            static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
        fs_data_dirty = true;
        bytes_used += proctex.color_diff_table.size() * sizeof(Common::Vec4f);
    }

    pica.proctex.table_dirty = 0;

    texture_buffer.Commit(static_cast<u32>(bytes_used));
}

void RasterizerVulkan::UploadUniforms(bool accelerate_draw) {
    const bool sync_vs_pica = accelerate_draw && pica.vs_setup.uniforms_dirty;
    if (!sync_vs_pica && !vs_data_dirty && !fs_data_dirty) {
        return;
    }

    const u32 uniform_size =
        uniform_size_aligned_vs_pica + uniform_size_aligned_vs + uniform_size_aligned_fs;
    auto [uniforms, offset, invalidate] =
        uniform_buffer.Map(uniform_size, uniform_buffer_alignment);

    u32 used_bytes = 0;

    if (vs_data_dirty || invalidate) {
        std::memcpy(uniforms + used_bytes, &vs_data, sizeof(vs_data));
        pipeline_cache.UpdateRange(1, offset + used_bytes);
        vs_data_dirty = false;
        used_bytes += uniform_size_aligned_vs;
    }

    if (fs_data_dirty || invalidate) {
        std::memcpy(uniforms + used_bytes, &fs_data, sizeof(fs_data));
        pipeline_cache.UpdateRange(2, offset + used_bytes);
        fs_data_dirty = false;
        used_bytes += uniform_size_aligned_fs;
    }

    if (sync_vs_pica || invalidate) {
        VSPicaUniformData vs_uniforms;
        vs_uniforms.SetFromRegs(pica.vs_setup);
        std::memcpy(uniforms + used_bytes, &vs_uniforms, sizeof(vs_uniforms));
        pipeline_cache.UpdateRange(0, offset + used_bytes);
        pica.vs_setup.uniforms_dirty = false;
        used_bytes += uniform_size_aligned_vs_pica;
    }

    uniform_buffer.Commit(used_bytes);
}

bool RasterizerVulkan::DiskResourcesLoadCrashed(u64 title_id) {
    return pipeline_cache.LoadCacheCrashed(title_id);
}

void RasterizerVulkan::SwitchDiskResources(u64 title_id) {
    std::atomic_bool stop_loading = false;

    if (switch_disk_resources_callback) {
        switch_disk_resources_callback(VideoCore::LoadCallbackStage::Prepare, 0, 0, "");
    }

    pipeline_cache.SetAccurateMul(accurate_mul);
    pipeline_cache.SwitchCache(title_id, stop_loading, switch_disk_resources_callback);

    if (switch_disk_resources_callback) {
        switch_disk_resources_callback(VideoCore::LoadCallbackStage::Complete, 0, 0, "");
    }
}

} // namespace Vulkan
