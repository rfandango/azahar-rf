// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <boost/container/static_vector.hpp>

#include "common/alignment.h"
#include "common/hash.h"
#include "common/hitch_tracker.h"
#include "common/microprofile.h"
#include "common/trace_markers.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_shader_timing.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

MICROPROFILE_DEFINE(Vulkan_Pipeline, "Vulkan", "Pipeline Building", MP_RGB(0, 192, 32));

vk::ShaderStageFlagBits MakeShaderStage(std::size_t index) {
    switch (index) {
    case 0:
        return vk::ShaderStageFlagBits::eVertex;
    case 1:
        return vk::ShaderStageFlagBits::eFragment;
    case 2:
        return vk::ShaderStageFlagBits::eGeometry;
    default:
        LOG_CRITICAL(Render_Vulkan, "Invalid shader stage index!");
        UNREACHABLE();
    }
    return vk::ShaderStageFlagBits::eVertex;
}

u64 StaticPipelineInfo::OptimizedHash(const Instance& instance) const {
    u64 info_hash = Common::HashCombine(
        shader_ids[0], shader_ids[1], shader_ids[2], Common::ComputeStructHash64(vertex_layout),
        Common::ComputeStructHash64(attachments), Common::ComputeStructHash64(blending));

    if (!instance.IsExtendedDynamicStateSupported()) {
        info_hash = Common::HashCombine(info_hash, Common::ComputeStructHash64(rasterization),
                                        Common::ComputeStructHash64(depth_stencil));
    }

    return info_hash;
}

u16 PipelineInfo::GetFinalColorWriteMask(const Instance& instance) {
    u16 color_write_mask = state.blending.color_write_mask;
    const bool is_logic_op_emulated =
        instance.NeedsLogicOpEmulation() && !state.blending.blend_enable;
    const bool is_logic_op_noop = state.blending.logic_op == Pica::FramebufferRegs::LogicOp::NoOp;
    if (is_logic_op_emulated && is_logic_op_noop) {
        // Color output is disabled by logic operation. We use color write mask to skip
        // color but allow depth write.
        color_write_mask = 0;
    }
    return color_write_mask;
}

Shader::Shader(const Instance& instance) : device{instance.GetDevice()} {}

Shader::Shader(const Instance& instance, vk::ShaderStageFlagBits stage, std::string code)
    : Shader{instance} {
    module = Compile(code, stage, instance.GetDevice());
    MarkDone();
}

Shader::~Shader() {
    if (device && module) {
        device.destroyShaderModule(module);
    }
}

GraphicsPipeline::GraphicsPipeline(const Instance& instance_, RenderManager& renderpass_cache_,
                                   const PipelineInfo& info_, vk::PipelineCache pipeline_cache_,
                                   vk::PipelineLayout layout_, std::array<Shader*, 3> stages_,
                                   Common::ThreadWorker* worker_, PipelineParts* parts_)
    : instance{instance_}, renderpass_cache{renderpass_cache_}, worker{worker_}, parts{parts_},
      pipeline_layout{layout_}, pipeline_cache{pipeline_cache_}, info{info_}, stages{stages_} {}

GraphicsPipeline::~GraphicsPipeline() = default;

bool GraphicsPipeline::TryBuild(bool wait_built, bool for_draw) {
    if (for_draw) {
        if (first_asked == ShaderTiming::Clock::time_point{}) {
            first_asked = ShaderTiming::Clock::now();
        }
        if (!wait_built) {
            draws_left_out.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // The pipeline is currently being compiled. We can either wait for it
    // or skip the draw.
    if (is_pending) {
        return wait_built;
    }

    // The shaders may still be on their way through the shader workers. The build is queued
    // all the same and waits for them there, which is a matter of milliseconds. It used to be
    // left for the next draw with this pipeline to queue: a frame later at the best, and at a
    // change of scene, when frames are long and everything is new, several hundred
    // milliseconds later (Monster Hunter 4 Ultimate after a save state: 474 ms median).
    const bool shaders_pending = std::any_of(
        stages.begin(), stages.end(), [](Shader* shader) { return shader && !shader->IsDone(); });
    if (!wait_built && shaders_pending &&
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyPipelineQueueing)) {
        return false;
    }

    // Ask the driver if it can give us the pipeline quickly.
    if (!shaders_pending && instance.IsPipelineCreationCacheControlSupported()) {
        [[maybe_unused]] const auto probe_start = ShaderTiming::Clock::now();
        const bool found = Build(true);
        if (for_draw) {
            LOG_SHADER_TIMING("Pipeline probe: {} in {:.2f} ms on the emulation thread, {:.1f} ms "
                              "after the first draw, {} draws left out before",
                              found ? "found" : "not found", ShaderTiming::MsSince(probe_start),
                              ShaderTiming::MsBetween(first_asked, probe_start),
                              draws_left_out.load(std::memory_order_relaxed) -
                                  (wait_built ? 0 : 1));
        }
        if (found) {
            return true;
        }
    }

    // Fallback to (a)synchronous compilation
    queued = ShaderTiming::Clock::now();
    worker->QueueWork([this] {
        if (!build_claimed.test_and_set()) {
            Build();
        }
    });
    is_pending = true;
    return wait_built;
}

bool GraphicsPipeline::BuildNowOrWait() {
    if (IsDone()) {
        return false;
    }
    if (!build_claimed.test_and_set()) {
        Build(false, true);
        return true;
    }
    WaitDone();
    return false;
}

bool GraphicsPipeline::Build(bool fail_on_compile_required, bool for_waiting_draw, bool optimize) {
    MICROPROFILE_SCOPE(Vulkan_Pipeline);
    CITRA_HITCH_SCOPE(ShaderCompile);
    // The fail_on_compile_required variant only asks the driver for an already-cached pipeline.
    CITRA_TRACE_SCOPE(fail_on_compile_required ? "Vulkan::BuildPipeline (cache probe)"
                                               : "Vulkan::BuildPipeline");
    if (!fail_on_compile_required) {
        CITRA_TRACE_COUNT(PipelineBuilds);
    }
    [[maybe_unused]] const auto build_start = ShaderTiming::Clock::now();
    [[maybe_unused]] double shader_wait_ms = 0.0;

    const u32 stride_alignment = instance.GetMinVertexStrideAlignment();
    std::array<vk::VertexInputBindingDescription, MAX_VERTEX_BINDINGS> bindings;
    for (u32 i = 0; i < info.state.vertex_layout.binding_count; i++) {
        const auto& binding = info.state.vertex_layout.bindings[i];
        bindings[i] = vk::VertexInputBindingDescription{
            .binding = binding.binding,
            .stride = Common::AlignUp(binding.byte_count.Value(), stride_alignment),
            .inputRate = binding.fixed.Value() ? vk::VertexInputRate::eInstance
                                               : vk::VertexInputRate::eVertex,
        };
    }

    std::array<vk::VertexInputAttributeDescription, MAX_VERTEX_ATTRIBUTES> attributes;
    for (u32 i = 0; i < info.state.vertex_layout.attribute_count; i++) {
        const auto& attr = info.state.vertex_layout.attributes[i];
        const FormatTraits& traits = instance.GetTraits(attr.type, attr.size);
        attributes[i] = vk::VertexInputAttributeDescription{
            .location = attr.location,
            .binding = attr.binding,
            .format = traits.native,
            .offset = attr.offset,
        };

        // At the end there's always the fixed binding which takes up
        // at least 16 bytes so we should always be able to alias.
        if (traits.needs_emulation) {
            const FormatTraits& comp_four_traits = instance.GetTraits(attr.type, 4);
            attributes[i].format = comp_four_traits.native;
        }
    }

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .vertexBindingDescriptionCount = info.state.vertex_layout.binding_count,
        .pVertexBindingDescriptions = bindings.data(),
        .vertexAttributeDescriptionCount = info.state.vertex_layout.attribute_count,
        .pVertexAttributeDescriptions = attributes.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = PicaToVK::PrimitiveTopology(info.state.rasterization.topology),
        .primitiveRestartEnable = false,
    };

    const vk::PipelineRasterizationStateCreateInfo raster_state = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .cullMode = PicaToVK::CullMode(info.state.rasterization.cull_mode,
                                       info.state.rasterization.flip_viewport),
        .frontFace = PicaToVK::FrontFace(info.state.rasterization.cull_mode),
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples = vk::SampleCountFlagBits(info.state.attachments.sample_count),
        .sampleShadingEnable = true,
        .minSampleShading = 1.0f,
    };

    const auto& blending = info.state.blending;
    auto src_color_factor = PicaToVK::BlendFunc(blending.src_color_blend_factor);
    auto dst_color_factor = PicaToVK::BlendFunc(blending.dst_color_blend_factor);
    auto color_op = PicaToVK::BlendEquation(blending.color_blend_eq);
    auto src_alpha_factor = PicaToVK::BlendFunc(blending.src_alpha_blend_factor);
    auto dst_alpha_factor = PicaToVK::BlendFunc(blending.dst_alpha_blend_factor);
    auto alpha_op = PicaToVK::BlendEquation(blending.alpha_blend_eq);

    // Blending with min/max equations is emulated in the fragment shader so
    // configure blending to not modify the incoming fragment color.
    if (blending.rgb_blend_emulation) {
        src_color_factor = vk::BlendFactor::eOne;
        dst_color_factor = vk::BlendFactor::eZero;
        color_op = vk::BlendOp::eAdd;
    }
    if (blending.alpha_blend_emulation) {
        src_alpha_factor = vk::BlendFactor::eOne;
        dst_alpha_factor = vk::BlendFactor::eZero;
        alpha_op = vk::BlendOp::eAdd;
    }

    // The PICA blends whenever logic operations are off, and games draw opaque geometry with
    // "source times one plus destination times zero". Passed on as blending it makes the GPU
    // fetch the destination of every fragment for nothing, and a driver gives up on what it
    // does for opaque geometry: Turnip stops updating its coarse depth buffer for the rest of
    // the render pass at the first draw that blends and writes depth, which is the first draw
    // of every scene. Copying the source is no blending, so it is not asked for.
    const bool blend_copies_source =
        color_op == vk::BlendOp::eAdd && alpha_op == vk::BlendOp::eAdd &&
        src_color_factor == vk::BlendFactor::eOne && dst_color_factor == vk::BlendFactor::eZero &&
        src_alpha_factor == vk::BlendFactor::eOne && dst_alpha_factor == vk::BlendFactor::eZero &&
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyOpaqueBlend) == 0;

    const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
        .blendEnable = blending.blend_enable && !blend_copies_source,
        .srcColorBlendFactor = src_color_factor,
        .dstColorBlendFactor = dst_color_factor,
        .colorBlendOp = color_op,
        .srcAlphaBlendFactor = src_alpha_factor,
        .dstAlphaBlendFactor = dst_alpha_factor,
        .alphaBlendOp = alpha_op,
        .colorWriteMask =
            static_cast<vk::ColorComponentFlags>(info.GetFinalColorWriteMask(instance)),
    };

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable = !info.state.blending.blend_enable && !instance.NeedsLogicOpEmulation(),
        .logicOp = PicaToVK::LogicOp(info.state.blending.logic_op),
        .attachmentCount = 1,
        .pAttachments = &colorblend_attachment,
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    const vk::Viewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = 1.0f,
        .height = 1.0f,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor = {
        .offset = {0, 0},
        .extent = {1, 1},
    };

    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    boost::container::static_vector<vk::DynamicState, 14> dynamic_states = {
        vk::DynamicState::eViewport,           vk::DynamicState::eScissor,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilReference,   vk::DynamicState::eBlendConstants,
    };

    if (instance.IsExtendedDynamicStateSupported()) {
        constexpr std::array extended = {
            vk::DynamicState::eCullModeEXT,        vk::DynamicState::eDepthCompareOpEXT,
            vk::DynamicState::eDepthTestEnableEXT, vk::DynamicState::eDepthWriteEnableEXT,
            vk::DynamicState::eFrontFaceEXT,       vk::DynamicState::ePrimitiveTopologyEXT,
            vk::DynamicState::eStencilOpEXT,       vk::DynamicState::eStencilTestEnableEXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended.begin(), extended.end());
    }

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::StencilOpState stencil_op_state = {
        .failOp = PicaToVK::StencilOp(info.state.depth_stencil.stencil_fail_op),
        .passOp = PicaToVK::StencilOp(info.state.depth_stencil.stencil_pass_op),
        .depthFailOp = PicaToVK::StencilOp(info.state.depth_stencil.stencil_depth_fail_op),
        .compareOp = PicaToVK::CompareFunc(info.state.depth_stencil.stencil_compare_op),
    };

    const vk::PipelineDepthStencilStateCreateInfo depth_info = {
        .depthTestEnable = static_cast<u32>(info.state.depth_stencil.depth_test_enable.Value()),
        .depthWriteEnable = static_cast<u32>(info.state.depth_stencil.depth_write_enable.Value()),
        .depthCompareOp = PicaToVK::CompareFunc(info.state.depth_stencil.depth_compare_op),
        .depthBoundsTestEnable = false,
        .stencilTestEnable = static_cast<u32>(info.state.depth_stencil.stencil_test_enable.Value()),
        .front = stencil_op_state,
        .back = stencil_op_state,
    };

    u32 shader_count = 0;
    std::array<vk::PipelineShaderStageCreateInfo, MAX_SHADER_STAGES> shader_stages;
    for (std::size_t i = 0; i < stages.size(); i++) {
        Shader* shader = stages[i];
        if (!shader) {
            continue;
        }

        if (!shader->IsDone()) {
            CITRA_TRACE_SCOPE("Vulkan::WaitShader");
            const auto wait_start = ShaderTiming::Clock::now();
            shader->WaitDone();
            shader_wait_ms += ShaderTiming::MsSince(wait_start);
        }
        shader_stages[shader_count++] = vk::PipelineShaderStageCreateInfo{
            .stage = MakeShaderStage(i),
            .module = shader->Handle(),
            .pName = "main",
        };
    }

    vk::GraphicsPipelineCreateInfo pipeline_info = {
        .stageCount = shader_count,
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_state,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth_info,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = pipeline_layout,
        .renderPass = renderpass_cache.GetRenderpass(info.state.attachments.color,
                                                     info.state.attachments.depth, false,
                                                     info.state.attachments.sample_count),
    };

    if (fail_on_compile_required) {
        pipeline_info.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT;
    }

    // What the driver itself says it spent, in all and on each stage
    vk::PipelineCreationFeedbackEXT whole_feedback{};
    std::array<vk::PipelineCreationFeedbackEXT, MAX_SHADER_STAGES> stage_feedbacks{};
    const vk::PipelineCreationFeedbackCreateInfoEXT feedback_info = {
        .pPipelineCreationFeedback = &whole_feedback,
        .pipelineStageCreationFeedbackCount = shader_count,
        .pPipelineStageCreationFeedbacks = stage_feedbacks.data(),
    };
    if (instance.IsPipelineCreationFeedbackSupported() && !fail_on_compile_required &&
        (!parts || optimize)) {
        pipeline_info.pNext = &feedback_info;
    }

    [[maybe_unused]] const auto create_start = ShaderTiming::Clock::now();
    // Where the driver builds the stages of a pipeline one by one and keeps each, a pipeline
    // that is new to it is put together from such parts: of a vertex and a fragment shader
    // usually only the latter has to be compiled then, a few milliseconds against twenty
    // and more for both. Its stages are not fitted to each other, though (the vertex
    // shader works out everything any fragment shader might read), so right after it is
    // built in one piece in the background, and that one is drawn with from then on. One
    // the driver has in its cache in one piece is taken from there.
    bool from_parts = false;
    bool found_whole = false;
    if (parts && !optimize && instance.IsPipelineCreationCacheControlSupported()) {
        vk::GraphicsPipelineCreateInfo whole_info = pipeline_info;
        whole_info.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT;
        auto whole = instance.GetDevice().createGraphicsPipelineUnique(pipeline_cache, whole_info);
        if (whole.result == vk::Result::eSuccess) {
            pipeline = std::move(whole.value);
            found_whole = true;
        }
    }
    if (found_whole) {
        // Nothing left to do
    } else if (parts && !optimize) {
        from_parts = true;
        // The four parts, each with the state that is its own and nothing else. What the
        // parts are told apart by: the shader modules themselves (several shader ids can
        // stand for one module), and of the rest only what is not set while drawing.
        const bool dynamic = instance.IsExtendedDynamicStateSupported();
        const u64 attachments_hash = Common::ComputeStructHash64(info.state.attachments);
        const u64 raster_hash = dynamic ? 0 : Common::ComputeStructHash64(info.state.rasterization);
        const u64 depth_hash = dynamic ? 0 : Common::ComputeStructHash64(info.state.depth_stencil);
        const auto module_id = [](const Shader* shader) {
            return static_cast<u64>(reinterpret_cast<std::uintptr_t>(shader));
        };
        using Subset = vk::GraphicsPipelineLibraryFlagBitsEXT;
        using States = boost::container::static_vector<vk::DynamicState, 8>;

        // Returns the part, built here unless it is there already; nothing if it would have
        // to be compiled, or is being compiled by another thread, and that is not wanted
        const auto get_part = [&](u64 key, Subset subset, vk::GraphicsPipelineCreateInfo part_info,
                                  const States& states, double& spent_ms) -> vk::Pipeline {
            PipelinePart& part = parts->Get(key);
            if (part.is_built.load(std::memory_order_acquire)) {
                return *part.pipeline;
            }
            const auto part_start = ShaderTiming::Clock::now();
            std::unique_lock lock{part.build_mutex, std::defer_lock};
            if (!fail_on_compile_required) {
                lock.lock();
            } else if (!lock.try_lock()) {
                return nullptr;
            }
            if (!part.is_built.load(std::memory_order_acquire)) {
                const vk::PipelineDynamicStateCreateInfo part_dynamic = {
                    .dynamicStateCount = static_cast<u32>(states.size()),
                    .pDynamicStates = states.data(),
                };
                const vk::GraphicsPipelineLibraryCreateInfoEXT library_info = {
                    .flags = subset,
                };
                part_info.pNext = &library_info;
                part_info.pDynamicState = &part_dynamic;
                part_info.flags = vk::PipelineCreateFlagBits::eLibraryKHR;
                if (fail_on_compile_required) {
                    part_info.flags |=
                        vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT;
                }
                auto part_result =
                    instance.GetDevice().createGraphicsPipelineUnique(pipeline_cache, part_info);
                if (part_result.result == vk::Result::eErrorPipelineCompileRequiredEXT) {
                    return nullptr;
                }
                ASSERT_MSG(part_result.result == vk::Result::eSuccess,
                           "Pipeline part creation failed!");
                part.pipeline = std::move(part_result.value);
                part.is_built.store(true, std::memory_order_release);
            }
            spent_ms += ShaderTiming::MsSince(part_start);
            return *part.pipeline;
        };

        double layout_ms = 0.0;
        double vertex_ms = 0.0;
        double fragment_ms = 0.0;
        std::array<vk::Pipeline, 4> libraries;

        States layout_states;
        States vertex_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        States fragment_states = {vk::DynamicState::eStencilCompareMask,
                                  vk::DynamicState::eStencilWriteMask,
                                  vk::DynamicState::eStencilReference};
        const States blend_states = {vk::DynamicState::eBlendConstants};
        if (dynamic) {
            layout_states.push_back(vk::DynamicState::ePrimitiveTopologyEXT);
            vertex_states.push_back(vk::DynamicState::eCullModeEXT);
            vertex_states.push_back(vk::DynamicState::eFrontFaceEXT);
            fragment_states.push_back(vk::DynamicState::eDepthCompareOpEXT);
            fragment_states.push_back(vk::DynamicState::eDepthTestEnableEXT);
            fragment_states.push_back(vk::DynamicState::eDepthWriteEnableEXT);
            fragment_states.push_back(vk::DynamicState::eStencilOpEXT);
            fragment_states.push_back(vk::DynamicState::eStencilTestEnableEXT);
        }

        // The stages of the two parts that have shaders
        std::array<vk::PipelineShaderStageCreateInfo, 2> vertex_part_stages;
        u32 vertex_part_stage_count = 0;
        const vk::PipelineShaderStageCreateInfo* fragment_part_stage = nullptr;
        for (u32 i = 0; i < shader_count; i++) {
            if (shader_stages[i].stage == vk::ShaderStageFlagBits::eFragment) {
                fragment_part_stage = &shader_stages[i];
            } else {
                vertex_part_stages[vertex_part_stage_count++] = shader_stages[i];
            }
        }

        libraries[0] =
            get_part(Common::HashCombine(u64{1}, raster_hash,
                                         Common::ComputeStructHash64(info.state.vertex_layout)),
                     Subset::eVertexInputInterface,
                     vk::GraphicsPipelineCreateInfo{
                         .pVertexInputState = &vertex_input_info,
                         .pInputAssemblyState = &input_assembly,
                     },
                     layout_states, layout_ms);
        libraries[1] = get_part(Common::HashCombine(u64{2}, raster_hash, attachments_hash,
                                                    module_id(stages[0]), module_id(stages[2])),
                                Subset::ePreRasterizationShaders,
                                vk::GraphicsPipelineCreateInfo{
                                    .stageCount = vertex_part_stage_count,
                                    .pStages = vertex_part_stages.data(),
                                    .pViewportState = &viewport_info,
                                    .pRasterizationState = &raster_state,
                                    .layout = pipeline_layout,
                                    .renderPass = pipeline_info.renderPass,
                                },
                                vertex_states, vertex_ms);
        libraries[2] = get_part(
            Common::HashCombine(u64{3}, depth_hash, attachments_hash, module_id(stages[1])),
            Subset::eFragmentShader,
            vk::GraphicsPipelineCreateInfo{
                .stageCount = fragment_part_stage ? 1u : 0u,
                .pStages = fragment_part_stage,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depth_info,
                .layout = pipeline_layout,
                .renderPass = pipeline_info.renderPass,
            },
            fragment_states, fragment_ms);
        libraries[3] =
            get_part(Common::HashCombine(
                         u64{4}, attachments_hash, Common::ComputeStructHash64(info.state.blending),
                         u64{colorblend_attachment.blendEnable},
                         u64{static_cast<u32>(colorblend_attachment.colorWriteMask)}),
                     Subset::eFragmentOutputInterface,
                     vk::GraphicsPipelineCreateInfo{
                         .pMultisampleState = &multisampling,
                         .pColorBlendState = &color_blending,
                         .renderPass = pipeline_info.renderPass,
                     },
                     blend_states, layout_ms);
        if (std::ranges::any_of(libraries, [](vk::Pipeline part) { return !part; })) {
            return false;
        }

        // Putting them together compiles nothing
        const vk::PipelineLibraryCreateInfoKHR library_list = {
            .libraryCount = static_cast<u32>(libraries.size()),
            .pLibraries = libraries.data(),
        };
        vk::GraphicsPipelineCreateInfo linked_info = {
            .pNext = &library_list,
            .layout = pipeline_layout,
            .renderPass = pipeline_info.renderPass,
        };
        if (fail_on_compile_required) {
            linked_info.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT;
        }
        auto result =
            instance.GetDevice().createGraphicsPipelineUnique(pipeline_cache, linked_info);
        if (result.result == vk::Result::eErrorPipelineCompileRequiredEXT) {
            return false;
        }
        ASSERT_MSG(result.result == vk::Result::eSuccess, "Putting a pipeline together failed!");
        pipeline = std::move(result.value);
        // In the places the log has for what the driver says of a pipeline built in one piece
        whole_feedback.duration = static_cast<u64>(ShaderTiming::MsSince(create_start) * 1e6);
        stage_feedbacks[0].duration = static_cast<u64>(vertex_ms * 1e6);
        stage_feedbacks[1].duration = static_cast<u64>(fragment_ms * 1e6);
    } else {
        auto result =
            instance.GetDevice().createGraphicsPipelineUnique(pipeline_cache, pipeline_info);
        if (result.result == vk::Result::eSuccess) {
            (optimize ? optimized : pipeline) = std::move(result.value);
        } else if (result.result == vk::Result::eErrorPipelineCompileRequiredEXT) {
            return false;
        } else {
            UNREACHABLE_MSG("Graphics pipeline creation failed!");
        }
    }
    current.store(optimize ? *optimized : *pipeline, std::memory_order_release);

    if (optimize) {
        LOG_SHADER_TIMING("Pipeline built in one piece after its parts: {:.1f} ms on core {}",
                          ShaderTiming::MsSince(create_start), ShaderTiming::CurrentCore());
        parts->last_optimized_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                                           ShaderTiming::Clock::now().time_since_epoch())
                                           .count(),
                                       std::memory_order_relaxed);
        parts->optimized_unsaved.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (!fail_on_compile_required) {
        // Pipelines of the disk cache are built before any draw asks for them
        [[maybe_unused]] const bool asked = first_asked != ShaderTiming::Clock::time_point{};
        LOG_SHADER_TIMING(
            "Pipeline built: driver {:.1f} ms, shaders awaited {:.1f} ms, {} on core {}, queued "
            "{:.1f} ms before, first draw {:.1f} ms before, {} draws left out, stages {}, driver "
            "says {:.1f} ms: {:.1f} {:.1f} {:.1f}{}",
            ShaderTiming::MsSince(create_start), shader_wait_ms,
            for_waiting_draw ? "by the waiting draw" : "by a worker", ShaderTiming::CurrentCore(),
            ShaderTiming::MsBetween(queued, build_start),
            asked ? ShaderTiming::MsBetween(first_asked, build_start) : -1.0,
            draws_left_out.load(std::memory_order_relaxed), shader_count,
            whole_feedback.duration / 1e6, stage_feedbacks[0].duration / 1e6,
            stage_feedbacks[1].duration / 1e6, stage_feedbacks[2].duration / 1e6,
            from_parts    ? " (from parts)"
            : found_whole ? " (found whole)"
                          : "");
    }

    MarkDone();
    if (from_parts && parts->optimizer &&
        !(Common::TraceMarkers::Experiments2() & Common::TraceMarkers::KeepPipelinesFromParts)) {
        parts->optimizer->QueueWork([this] { Build(false, false, true); });
    }
    return true;
}

} // namespace Vulkan
