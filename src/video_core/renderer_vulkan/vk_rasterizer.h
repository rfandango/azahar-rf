// Copyright 2022-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <unordered_map>
#include "video_core/rasterizer_accelerated.h"
#include "video_core/renderer_vulkan/vk_descriptor_update_queue.h"
#include "video_core/renderer_vulkan/vk_guest_memory.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {
class CustomTexManager;
class RendererBase;
} // namespace VideoCore

namespace Pica {
struct DisplayTransferConfig;
struct MemoryFillConfig;
struct FramebufferConfig;
} // namespace Pica

namespace Vulkan {

struct ScreenInfo;

class Instance;
class Scheduler;
class RenderManager;

class RasterizerVulkan : public VideoCore::RasterizerAccelerated {
public:
    explicit RasterizerVulkan(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                              VideoCore::CustomTexManager& custom_tex_manager,
                              VideoCore::RendererBase& renderer, Frontend::EmuWindow& emu_window,
                              const Instance& instance, Scheduler& scheduler,
                              RenderManager& renderpass_cache, DescriptorUpdateQueue& update_queue,
                              u32 image_count);
    ~RasterizerVulkan() override;

    void TickFrame();
    void LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback) override;

    void DrawTriangles() override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;
    void SubmitPendingWork() override;
    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateFill(const Pica::MemoryFillConfig& config) override;
    bool AccelerateDisplay(const Pica::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, ScreenInfo& screen_info);
    bool AccelerateDrawBatch(bool is_indexed) override;
    void NoteDrawOverwrites(bool color, bool depth) override;

    /// Switches the disk resources to the specified title
    void SwitchDiskResources(u64 title_id) override;

    bool DiskResourcesLoadCrashed(u64 title_id) override;

private:
    /// Syncs pipeline state from PICA registers
    void SyncDrawState();

    /// Syncs and uploads the lighting, fog and proctex LUTs
    void SyncAndUploadLUTs();
    void SyncAndUploadLUTsLF();

    /// Syncs all enabled PICA texture units
    /// Returns whether the colour surface of the framebuffer moved to its other image, for a
    /// draw that samples it and overwrites all of the rectangle it is confined to
    bool SyncTextureUnits(const Framebuffer* framebuffer, const Common::Rectangle<u32>& draw_rect,
                          bool overwrites_color);

    /// Syncs all utility textures in the fragment shader.
    void SyncUtilityTextures(const Framebuffer* framebuffer);

    /// Binds the PICA shadow cube required for shadow mapping
    void BindShadowCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                        vk::DescriptorSet texture_set);

    /// Binds a texture cube to texture unit 0
    void BindTextureCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                         vk::DescriptorSet texture_set);

    /// Upload the uniform blocks to the uniform buffer object
    void UploadUniforms(bool accelerate_draw);

    /// Generic draw function for DrawTriangles and AccelerateDrawBatch
    bool Draw(bool accelerate, bool is_indexed);

    /// Internal implementation for AccelerateDrawBatch
    bool AccelerateDrawBatchInternal(bool is_indexed);

    /// Setup index array for AccelerateDrawBatch
    void SetupIndexArray();

    /// Setup vertex array for AccelerateDrawBatch
    void SetupVertexArray();

    /// Setup the fixed attribute emulation in vulkan
    void SetupFixedAttribs();

    /// The emulation is about to write the stretch: vertex and index data there is no longer
    /// read in place, and a draw that still is to read it there gets through the GPU first
    void WaitForDataInPlace(PAddr addr, u32 size);

    /// Whether the draw being set up may read the stretch of emulated memory where it is
    bool MayReadInPlace(PAddr addr, u32 size);

    /// Finds what is kept for the place the draw being set up is issued from, see DrawSite
    void LookUpDrawSite();

    /// Setup vertex shader for AccelerateDrawBatch
    bool SetupVertexShader();

    /// Setup geometry shader for AccelerateDrawBatch
    bool SetupGeometryShader();

    /// Creates the vertex layout struct used for software shader pipelines
    void MakeSoftwareVertexLayout();

private:
    const Instance& instance;
    Scheduler& scheduler;
    RenderManager& renderpass_cache;
    DescriptorUpdateQueue& update_queue;
    PipelineCache pipeline_cache;
    TextureRuntime runtime;
    RasterizerCache res_cache;

    VertexLayout software_layout;
    std::array<u32, 16> binding_offsets{};
    /// One bit a vertex binding: set if its data is taken from the emulated memory, where the
    /// game has it, and the offset is one into guest_memory; from stream_buffer otherwise
    u16 bindings_in_guest_memory{};
    std::array<bool, 16> enable_attributes{};
    std::array<vk::Buffer, 16> vertex_buffers;
    VertexArrayInfo vertex_info;
    PipelineInfo pipeline_info{};

    StreamBuffer stream_buffer;     ///< Vertex+Index buffer
    StreamBuffer uniform_buffer;    ///< Uniform buffer
    StreamBuffer texture_buffer;    ///< Texture buffer
    StreamBuffer texture_lf_buffer; ///< Texture Light-Fog buffer
    /// The emulated memory as a buffer, if the device takes it: vertex and index data are
    /// then used where they are instead of being copied to stream_buffer
    GuestMemory guest_memory;
    /// Which of that data may be used where it is: what the game leaves alone
    InPlaceData in_place_data;
    /// Set while a draw looks at its own vertex and index data, which is no read by the game
    bool reading_draw_data{false};

    /// What a draw resolved to, kept by the state it was issued in: the hash of all registers
    /// that shape a draw (ShapingHash) and the vertex shader's code. Games send, and call,
    /// command lists that are the same from frame to frame (measured: 99% of the draws of
    /// one game find every register as it was the time before), so most draws are issued in
    /// a state seen before and can take what was resolved then: the choice of shaders, and
    /// the range of their indices if those have been read in place, unwritten, since.
    struct DrawSite {
        bool filled{};
        u64 shaping{};
        u64 vs_code_hash{};
        u64 vs_swizzle_hash{};
        const void* shader_cache{};
        Pica::Shader::UserConfig user_config{};
        PipelineCache::ChosenShader vs;
        PipelineCache::ChosenShader gs;
        PipelineCache::ChosenShader fs;
        bool has_vs{};
        bool has_gs{};
        bool has_fs{};
        bool fs_depth_from_viewport{};
        bool fs_host_scissor{};
        bool fs_color_unused{};
        /// The lowest and highest index of the draw, good while has_index_range is set and
        /// InPlaceData::Generation() is index_generation
        bool has_index_range{};
        u32 index_min{};
        u32 index_max{};
        u64 index_generation{};
    };
    std::vector<DrawSite> draw_sites;
    /// Of the draw being set up; null if none is kept for it
    DrawSite* site{};
    /// Whether what `site` holds is good for this draw
    bool site_matches{};

    /// What a binding of the texture descriptor set holds
    struct TextureBinding {
        vk::ImageView view{};
        vk::Sampler sampler{};
        bool operator==(const TextureBinding&) const = default;
    };
    /// The texture set of the draw before: what it holds, and the tick of the commands it was
    /// taken for. A draw that asks for the same within the same commands uses it again.
    struct LastTextureSet {
        std::array<TextureBinding, 4> bindings{};
        u64 tick{};
        bool valid{};
    };
    LastTextureSet last_texture_set;
    /// Measuring only, see Common::TraceMarkers::MeasureVertexData
    GuestDataWatch data_watch;
    bool watch_data{false};
    vk::UniqueBufferView texture_lf_view;
    vk::UniqueBufferView texture_rg_view;
    vk::UniqueBufferView texture_rgba_view;
    vk::DeviceSize uniform_buffer_alignment;
    u32 uniform_size_aligned_vs_pica;
    u32 uniform_size_aligned_vs;
    u32 uniform_size_aligned_fs;
    bool async_shaders{false};

    /// Guest render target and attachments of the previous draw, used to keep attachments bound
    /// across consecutive draws that toggle colour or depth use (see Draw).
    struct BoundTarget {
        PAddr color_addr{};
        PAddr depth_addr{};
        u32 width{};
        u32 height{};
        u32 color_format{};
        u32 depth_format{};
        bool color{};
        bool depth{};
        bool valid{};
    };
    BoundTarget bound_target{};

    /// Which targets the game draws anew all the time: one of those can go a frame without
    /// a small draw whose pipeline is still being built, one that is drawn to once cannot.
    /// The frames are counted by TickFrame; kept for each colour buffer are the last two it
    /// was drawn to in.
    struct TargetUse {
        u64 last{};
        u64 before{};
    };
    std::unordered_map<PAddr, TargetUse> target_use;
    u64 frame_count{1};
    u64 target_use_frame{};
    bool target_redrawn{};

    /// Whether a draw of two triangles at the most has to wait for its pipeline: leaving
    /// it out until the pipeline is built would show, or last
    bool SmallDrawMustShow(bool is_indexed);
    /// The rectangle of its target, in the target's pixels, that the draw about to be made
    /// can write to: the box around its triangles where they are few, with a console pixel to
    /// spare, and the rectangle the draw is confined to otherwise
    Common::Rectangle<u32> WrittenBy(bool accelerate, bool is_indexed,
                                     const Common::Rectangle<u32>& draw_rect,
                                     const VideoCore::ViewportInfo& viewport, u32 res_scale);
    /// What the draw that comes next overwrites of the area it is confined to
    bool next_draw_overwrites_color{};
    bool next_draw_overwrites_depth{};
};

} // namespace Vulkan
