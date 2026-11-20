// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "common/hash.h"
#include "common/thread_worker.h"
#include "video_core/pica/regs_pipeline.h"
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_common.h"

#define LAYOUT_HASH static_cast<u64>(sizeof(T)), static_cast<u64>(alignof(T))
#define FIELD_HASH(x) static_cast<u64>(offsetof(T, x)), static_cast<u64>(sizeof(x))

namespace Common {

struct AsyncHandle {
public:
    AsyncHandle(bool is_done_ = false) : is_done{is_done_} {}

    [[nodiscard]] bool IsDone() noexcept {
        return is_done.load(std::memory_order::relaxed);
    }

    void WaitDone() noexcept {
        std::unique_lock lock{mutex};
        condvar.wait(lock, [this] { return is_done.load(std::memory_order::relaxed); });
    }

    void MarkDone(bool done = true) noexcept {
        std::scoped_lock lock{mutex};
        is_done = done;
        condvar.notify_all();
    }

private:
    std::condition_variable condvar;
    std::mutex mutex;
    std::atomic_bool is_done{false};
};

} // namespace Common

namespace Vulkan {

class Instance;
class RenderManager;

constexpr u32 MAX_SHADER_STAGES = 3;
constexpr u32 MAX_VERTEX_ATTRIBUTES = 16;
constexpr u32 MAX_VERTEX_BINDINGS = 13;

/**
 * The pipeline state is tightly packed with bitfields to reduce
 * the overhead of hashing as much as possible
 */
union RasterizationState {
    u8 value;
    BitField<0, 2, Pica::PipelineRegs::TriangleTopology> topology;
    BitField<4, 2, Pica::RasterizerRegs::CullMode> cull_mode;
    BitField<6, 1, u8> flip_viewport;

    static consteval u64 StructHash() {
        constexpr u64 STRUCT_VERSION = 0;

        using T = RasterizationState;
        return Common::HashCombine(STRUCT_VERSION,

                                   // layout
                                   LAYOUT_HASH,

                                   // fields
                                   FIELD_HASH(topology), FIELD_HASH(cull_mode),
                                   FIELD_HASH(flip_viewport));
    }
};
static_assert(std::is_trivial_v<RasterizationState>);

union DepthStencilState {
    u32 value;
    BitField<0, 1, u32> depth_test_enable;
    BitField<1, 1, u32> depth_write_enable;
    BitField<2, 1, u32> stencil_test_enable;
    BitField<3, 3, Pica::FramebufferRegs::CompareFunc> depth_compare_op;
    BitField<6, 3, Pica::FramebufferRegs::StencilAction> stencil_fail_op;
    BitField<9, 3, Pica::FramebufferRegs::StencilAction> stencil_pass_op;
    BitField<12, 3, Pica::FramebufferRegs::StencilAction> stencil_depth_fail_op;
    BitField<15, 3, Pica::FramebufferRegs::CompareFunc> stencil_compare_op;

    static consteval u64 StructHash() {
        constexpr u64 STRUCT_VERSION = 0;

        using T = DepthStencilState;
        return Common::HashCombine(STRUCT_VERSION,

                                   // layout
                                   LAYOUT_HASH,

                                   // fields
                                   FIELD_HASH(depth_test_enable), FIELD_HASH(depth_write_enable),
                                   FIELD_HASH(stencil_test_enable), FIELD_HASH(depth_compare_op),
                                   FIELD_HASH(stencil_fail_op), FIELD_HASH(stencil_pass_op),
                                   FIELD_HASH(stencil_depth_fail_op),
                                   FIELD_HASH(stencil_compare_op));
    }
};
static_assert(std::is_trivial_v<DepthStencilState>);

struct BlendingState {
    u16 blend_enable;
    u16 color_write_mask;
    Pica::FramebufferRegs::LogicOp logic_op;
    union {
        u32 value;
        BitField<0, 4, Pica::FramebufferRegs::BlendFactor> src_color_blend_factor;
        BitField<4, 4, Pica::FramebufferRegs::BlendFactor> dst_color_blend_factor;
        BitField<8, 3, Pica::FramebufferRegs::BlendEquation> color_blend_eq;
        BitField<11, 4, Pica::FramebufferRegs::BlendFactor> src_alpha_blend_factor;
        BitField<15, 4, Pica::FramebufferRegs::BlendFactor> dst_alpha_blend_factor;
        BitField<19, 3, Pica::FramebufferRegs::BlendEquation> alpha_blend_eq;
        BitField<22, 1, u32> rgb_blend_emulation;
        BitField<23, 1, u32> alpha_blend_emulation;
    };

    static consteval u64 StructHash() {
        constexpr u64 STRUCT_VERSION = 0;

        using T = BlendingState;
        return Common::HashCombine(
            STRUCT_VERSION,

            // layout
            LAYOUT_HASH,

            // fields
            FIELD_HASH(blend_enable), FIELD_HASH(color_write_mask), FIELD_HASH(logic_op),
            FIELD_HASH(src_color_blend_factor), FIELD_HASH(dst_color_blend_factor),
            FIELD_HASH(color_blend_eq), FIELD_HASH(src_alpha_blend_factor),
            FIELD_HASH(dst_alpha_blend_factor), FIELD_HASH(alpha_blend_eq),
            FIELD_HASH(rgb_blend_emulation), FIELD_HASH(alpha_blend_emulation));
    }
};
static_assert(std::is_trivial_v<BlendingState>);

union VertexBinding {
    u16 value;
    BitField<0, 4, u16> binding;
    BitField<4, 1, u16> fixed;
    BitField<5, 11, u16> byte_count;

    static consteval u64 StructHash() {
        constexpr u64 STRUCT_VERSION = 0;

        using T = VertexBinding;
        return Common::HashCombine(STRUCT_VERSION,

                                   // layout
                                   LAYOUT_HASH,

                                   // fields
                                   FIELD_HASH(binding), FIELD_HASH(fixed), FIELD_HASH(byte_count));
    }
};
static_assert(std::is_trivial_v<VertexBinding>);

union VertexAttribute {
    u32 value;
    BitField<0, 4, u32> binding;
    BitField<4, 4, u32> location;
    BitField<8, 3, Pica::PipelineRegs::VertexAttributeFormat> type;
    BitField<11, 3, u32> size;
    BitField<14, 11, u32> offset;

    static consteval u64 StructHash() {
        constexpr u64 STRUCT_VERSION = 0;

        using T = VertexAttribute;
        return Common::HashCombine(STRUCT_VERSION,

                                   // layout
                                   LAYOUT_HASH,

                                   // fields
                                   FIELD_HASH(binding), FIELD_HASH(location), FIELD_HASH(type),
                                   FIELD_HASH(size), FIELD_HASH(offset));
    }
};
static_assert(std::is_trivial_v<VertexAttribute>);

struct VertexLayout {
    u8 binding_count;
    u8 attribute_count;
    std::array<VertexBinding, MAX_VERTEX_BINDINGS> bindings;
    std::array<VertexAttribute, MAX_VERTEX_ATTRIBUTES> attributes;

    static consteval u64 StructHash() {
        constexpr u64 STRUCT_VERSION = 0;

        using T = VertexLayout;
        return Common::HashCombine(STRUCT_VERSION,

                                   // layout
                                   LAYOUT_HASH,

                                   // fields
                                   FIELD_HASH(binding_count), FIELD_HASH(attribute_count),
                                   FIELD_HASH(bindings), FIELD_HASH(attributes),

                                   // nested layout
                                   VertexBinding::StructHash(), VertexAttribute::StructHash());
    }
};
static_assert(std::is_trivial_v<VertexLayout>);

struct AttachmentInfo {
    VideoCore::PixelFormat color;
    VideoCore::PixelFormat depth;
    u8 sample_count;

    static consteval u64 StructHash() {
        constexpr u64 STRUCT_VERSION = 0;

        using T = AttachmentInfo;
        return Common::HashCombine(STRUCT_VERSION,

                                   // layout
                                   LAYOUT_HASH,

                                   // fields
                                   FIELD_HASH(color), FIELD_HASH(depth), FIELD_HASH(sample_count));
    }
};
static_assert(std::is_trivial_v<AttachmentInfo>);

struct StaticPipelineInfo {
    std::array<u64, MAX_SHADER_STAGES> shader_ids;

    BlendingState blending;
    AttachmentInfo attachments;
    VertexLayout vertex_layout;

    RasterizationState rasterization;
    DepthStencilState depth_stencil;

    [[nodiscard]] u64 OptimizedHash(const Instance& instance) const;

    static consteval u64 StructHash() {
        constexpr u64 STRUCT_VERSION = 0;

        using T = StaticPipelineInfo;
        return Common::HashCombine(
            STRUCT_VERSION,

            // layout
            LAYOUT_HASH,

            // fields
            FIELD_HASH(shader_ids), FIELD_HASH(blending), FIELD_HASH(attachments),
            FIELD_HASH(vertex_layout), FIELD_HASH(rasterization), FIELD_HASH(depth_stencil),

            // nested layout
            BlendingState::StructHash(), AttachmentInfo::StructHash(), VertexLayout::StructHash(),
            RasterizationState::StructHash(), DepthStencilState::StructHash());
    }
};
static_assert(std::is_trivial_v<StaticPipelineInfo>);

struct DynamicPipelineInfo {
    u32 blend_color = 0;
    u8 stencil_reference;
    u8 stencil_compare_mask;
    u8 stencil_write_mask;

    Common::Rectangle<u32> scissor;
    Common::Rectangle<s32> viewport;
    /// Viewport depth bounds; carry the PICA depth range when the shader does not write depth.
    float depth_min = 0.f;
    float depth_max = 1.f;

    bool operator==(const DynamicPipelineInfo& other) const noexcept {
        return std::memcmp(this, &other, sizeof(DynamicPipelineInfo)) == 0;
    }
};

/**
 * Information about a graphics pipeline
 */
struct PipelineInfo : Common::HashableStruct<StaticPipelineInfo> {
    DynamicPipelineInfo dynamic_info;

    [[nodiscard]] bool IsDepthWriteEnabled() const noexcept {
        const bool has_stencil = state.attachments.depth == VideoCore::PixelFormat::D24S8;
        const bool depth_write =
            state.depth_stencil.depth_test_enable && state.depth_stencil.depth_write_enable;
        const bool stencil_write = has_stencil && state.depth_stencil.stencil_test_enable &&
                                   dynamic_info.stencil_write_mask != 0;

        return depth_write || stencil_write;
    }

    [[nodiscard]] u16 GetFinalColorWriteMask(const Instance& instance);
};

struct Shader : public Common::AsyncHandle {
    explicit Shader(const Instance& instance);
    explicit Shader(const Instance& instance, vk::ShaderStageFlagBits stage, std::string code);
    ~Shader();

    [[nodiscard]] vk::ShaderModule Handle() const noexcept {
        return module;
    }

    vk::ShaderModule module;
    vk::Device device;
    std::string program;
};

/// A part of a pipeline that the driver builds and keeps on its own where it can
/// (VK_EXT_graphics_pipeline_library): the vertex layout, the vertex stage, the fragment
/// stage or the blending. A part is shared by all the pipelines that are made with it, so a
/// vertex shader is compiled once and not again with every fragment shader it meets.
struct PipelinePart {
    /// Held by the thread that builds the part; whoever else needs it waits here
    std::mutex build_mutex;
    std::atomic_bool is_built{false};
    vk::UniquePipeline pipeline;
};

/// The parts built so far, by a hash of what each is made from
class PipelineParts {
public:
    /// Returns the part for the key; nobody has built it yet if it was not asked for before
    PipelinePart& Get(u64 key) {
        std::scoped_lock lock{mutex};
        auto& part = parts[key];
        if (!part) {
            part = std::make_unique<PipelinePart>();
        }
        return *part;
    }

    /// A pipeline put together from parts is drawn with at once, and built in one piece
    /// here right after, with its stages fitted to each other, to take its place
    Common::ThreadWorker* optimizer{};
    /// How many were built that way since the driver's cache was last stored, and when the
    /// last of them was done (milliseconds of the steady clock)
    std::atomic<u32> optimized_unsaved{};
    std::atomic<s64> last_optimized_ms{};

private:
    std::mutex mutex;
    std::unordered_map<u64, std::unique_ptr<PipelinePart>> parts;
};

class GraphicsPipeline : public Common::AsyncHandle {
public:
    /// With `parts` the pipeline is put together from parts shared with other pipelines
    explicit GraphicsPipeline(const Instance& instance, RenderManager& renderpass_cache,
                              const PipelineInfo& info, vk::PipelineCache pipeline_cache,
                              vk::PipelineLayout layout, std::array<Shader*, 3> stages,
                              Common::ThreadWorker* worker, PipelineParts* parts = nullptr,
                              Common::ThreadWorker* urgent_worker = nullptr);
    ~GraphicsPipeline();

    /// `for_draw` is false when the pipeline is built ahead of any draw, from the disk cache
    bool TryBuild(bool wait_built, bool for_draw = true);

    /// With `optimize` the pipeline, already put together from parts, is built in one piece
    bool Build(bool fail_on_compile_required = false, bool for_waiting_draw = false,
               bool optimize = false);

    /// Returns once the pipeline is built, for a draw that cannot go on without it. A build
    /// that is still waiting for a worker is done on the calling thread right away: the workers
    /// can have many other pipelines to get through first. True if the calling thread built it.
    bool BuildNowOrWait();

    [[nodiscard]] vk::Pipeline Handle() const noexcept {
        return vk::Pipeline{current.load(std::memory_order_acquire)};
    }

private:
    const Instance& instance;
    RenderManager& renderpass_cache;
    Common::ThreadWorker* worker;
    /// Builds what a draw waits for, apart from the workers that have the pipelines of
    /// draws that can go without on their lists
    Common::ThreadWorker* urgent_worker;
    PipelineParts* parts;

    vk::UniquePipeline pipeline;
    /// Built in one piece after `pipeline` was put together from parts. That one stays, as
    /// commands recorded with it may still be on their way.
    vk::UniquePipeline optimized;
    /// The one to draw with
    std::atomic<VkPipeline> current{};
    vk::PipelineLayout pipeline_layout;
    vk::PipelineCache pipeline_cache;

    PipelineInfo info;
    std::array<Shader*, 3> stages;
    bool is_pending{};
    /// Set by whoever builds the pipeline after it was queued: a worker, or a thread waiting
    std::atomic_flag build_claimed;

    /// For the log of what it took to get the pipeline, in profiling builds: when the first
    /// draw asked for it, when its build was queued, and the draws left out in between
    std::chrono::steady_clock::time_point first_asked{};
    std::chrono::steady_clock::time_point queued{};
    std::atomic<u32> draws_left_out{};
};

} // namespace Vulkan

#undef LAYOUT_HASH
#undef FIELD_HASH
