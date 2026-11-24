// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <bit>
#include <cstring>
#include "common/alignment.h"
#include "common/math_util.h"
#include "common/trace_markers.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/rasterizer_accelerated.h"

namespace VideoCore {

DiskResourceLoadCallback RasterizerInterface::switch_disk_resources_callback{};

using Pica::f24;

static Common::Vec4f ColorRGBA8(const u32 color) {
    const auto rgba =
        Common::Vec4u{color >> 0 & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF, color >> 24 & 0xFF};
    return rgba / 255.0f;
}

/// A colour of the texture combiner: the float the combiner's own rounding to 1/255 gives for
/// each byte (the byte times 1/255, which is not always the float of the byte divided by
/// 255). A fragment shader can then take the colour for a rounded value and need not round
/// a result that is only this colour.
static Common::Vec4f CombinerColorRGBA8(const u32 color) {
    constexpr f32 STEP = 1.f / 255.f;
    return Common::Vec4f{
        static_cast<f32>(color >> 0 & 0xFF) * STEP,
        static_cast<f32>(color >> 8 & 0xFF) * STEP,
        static_cast<f32>(color >> 16 & 0xFF) * STEP,
        static_cast<f32>(color >> 24 & 0xFF) * STEP,
    };
}

static Common::Vec3f LightColor(const Pica::LightingRegs::LightColor& color) {
    return Common::Vec3u{color.r, color.g, color.b} / 255.0f;
}

RasterizerAccelerated::HardwareVertex::HardwareVertex(const Pica::OutputVertex& v,
                                                      bool flip_quaternion) {
    position[0] = v.pos.x.ToFloat32();
    position[1] = v.pos.y.ToFloat32();
    position[2] = v.pos.z.ToFloat32();
    position[3] = v.pos.w.ToFloat32();
    color[0] = v.color.x.ToFloat32();
    color[1] = v.color.y.ToFloat32();
    color[2] = v.color.z.ToFloat32();
    color[3] = v.color.w.ToFloat32();
    tex_coord0[0] = v.tc0.x.ToFloat32();
    tex_coord0[1] = v.tc0.y.ToFloat32();
    tex_coord1[0] = v.tc1.x.ToFloat32();
    tex_coord1[1] = v.tc1.y.ToFloat32();
    tex_coord2[0] = v.tc2.x.ToFloat32();
    tex_coord2[1] = v.tc2.y.ToFloat32();
    tex_coord0_w = v.tc0_w.ToFloat32();
    normquat[0] = v.quat.x.ToFloat32();
    normquat[1] = v.quat.y.ToFloat32();
    normquat[2] = v.quat.z.ToFloat32();
    normquat[3] = v.quat.w.ToFloat32();
    view[0] = v.view.x.ToFloat32();
    view[1] = v.view.y.ToFloat32();
    view[2] = v.view.z.ToFloat32();

    if (flip_quaternion) {
        normquat = -normquat;
    }
}

RasterizerAccelerated::RasterizerAccelerated(Memory::MemorySystem& memory_, Pica::PicaCore& pica_)
    : memory{memory_}, pica{pica_}, regs{pica.regs.internal} {}

/**
 * This is a helper function to resolve an issue when interpolating opposite quaternions. See below
 * for a detailed description of this issue (yuriks):
 *
 * For any rotation, there are two quaternions Q, and -Q, that represent the same rotation. If you
 * interpolate two quaternions that are opposite, instead of going from one rotation to another
 * using the shortest path, you'll go around the longest path. You can test if two quaternions are
 * opposite by checking if Dot(Q1, Q2) < 0. In that case, you can flip either of them, therefore
 * making Dot(Q1, -Q2) positive.
 *
 * This solution corrects this issue per-vertex before passing the quaternions to OpenGL. This is
 * correct for most cases but can still rotate around the long way sometimes. An implementation
 * which did `lerp(lerp(Q1, Q2), Q3)` (with proper weighting), applying the dot product check
 * between each step would work for those cases at the cost of being more complex to implement.
 *
 * Fortunately however, the 3DS hardware happens to also use this exact same logic to work around
 * these issues, making this basic implementation actually more accurate to the hardware.
 */
static bool AreQuaternionsOpposite(Common::Vec4<f24> qa, Common::Vec4<f24> qb) {
    Common::Vec4f a{qa.x.ToFloat32(), qa.y.ToFloat32(), qa.z.ToFloat32(), qa.w.ToFloat32()};
    Common::Vec4f b{qb.x.ToFloat32(), qb.y.ToFloat32(), qb.z.ToFloat32(), qb.w.ToFloat32()};

    return (Common::Dot(a, b) < 0.f);
}

void RasterizerAccelerated::AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                                        const Pica::OutputVertex& v2) {
    vertex_batch.emplace_back(v0, false);
    vertex_batch.emplace_back(v1, AreQuaternionsOpposite(v0.quat, v1.quat));
    vertex_batch.emplace_back(v2, AreQuaternionsOpposite(v0.quat, v2.quat));
}

RasterizerAccelerated::VertexArrayInfo RasterizerAccelerated::AnalyzeVertexArray(
    bool is_indexed, u32 stride_alignment) [[hot]] {
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;

    u32 vertex_min;
    u32 vertex_max;
    if (is_indexed) {
        const auto& index_info = regs.pipeline.index_array;
        const PAddr address = vertex_attributes.GetPhysicalBaseAddress() + index_info.offset;
        const u8* index_address_8 =
            memory.GpuRead(address, regs.pipeline.num_vertices * (index_info.format != 0 ? 2 : 1));
        if (index_address_8 == nullptr) {
            // Mario & Luigi: Superstar Saga sets an invalid base address
            // for the vertex attributes. Return early if that is the case.
            return {0, 0, 0};
        }
        const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
        const bool index_u16 = index_info.format != 0;

        vertex_min = 0xFFFF;
        vertex_max = 0;
        const u32 count = regs.pipeline.num_vertices;
        const u32 index_size = index_u16 ? 2 : 1;
        const u32 size = count * index_size;
        FlushRegion(address, size);

        if (index_u16) {
            const auto res = Common::FindMinMax({index_address_16, static_cast<size_t>(count)});
            vertex_min = static_cast<u32>(res.first);
            vertex_max = static_cast<u32>(res.second);
        } else {
            const auto res = Common::FindMinMax({index_address_8, static_cast<size_t>(count)});
            vertex_min = static_cast<u32>(res.first);
            vertex_max = static_cast<u32>(res.second);
        }
    } else {
        vertex_min = regs.pipeline.vertex_offset;
        vertex_max = regs.pipeline.vertex_offset + regs.pipeline.num_vertices - 1;
    }

    return MakeVertexArrayInfo(vertex_min, vertex_max, stride_alignment);
}

RasterizerAccelerated::VertexArrayInfo RasterizerAccelerated::MakeVertexArrayInfo(
    u32 vertex_min, u32 vertex_max, u32 stride_alignment) const {
    const u32 vertex_num = vertex_max - vertex_min + 1;
    u32 vs_input_size = 0;
    for (const auto& loader : regs.pipeline.vertex_attributes.attribute_loaders) {
        if (loader.component_count != 0) {
            const u32 aligned_stride =
                Common::AlignUp(static_cast<u32>(loader.byte_count), stride_alignment);
            vs_input_size += Common::AlignUp(aligned_stride * vertex_num, 4);
        }
    }

    return {vertex_min, vertex_max, vs_input_size};
}

namespace {

/// One bit a register: set for those that say how a draw is shaped. Left out are the
/// registers data only passes through on its way elsewhere (uniform values, shader code,
/// lookup tables, a fixed attribute), the values of the lights (colours, positions,
/// directions, attenuation: they reach the shader as uniforms), those that say where the
/// command list is, and the two a draw is started with. A register that is not certain to be
/// one of these counts as shaping.
const std::array<u64, 12>& ShapingRegisterBits() {
    static const std::array<u64, 12> shaping = [] {
        std::array<u64, 12> bits;
        bits.fill(~u64{0});
        const auto leave_out = [&bits](u32 first, u32 last) {
            for (u32 id = first; id <= last; id++) {
                bits[id >> 6] &= ~(u64{1} << (id & 63));
            }
        };
        leave_out(0x000, 0x03F); // Interrupts and the end of a list
        leave_out(0x0AF, 0x0B7); // Procedural texture tables
        leave_out(0x0E6, 0x0EF); // Fog table
        for (u32 light = 0; light < 8; light++) {
            const u32 base = 0x140 + 0x10 * light;
            leave_out(base + 0x0, base + 0x7); // Colours, position, spot direction
            leave_out(base + 0xA, base + 0xB); // Attenuation
        }
        leave_out(0x1C0, 0x1C0); // Global ambient
        leave_out(0x1C5, 0x1CF); // Lighting tables
        leave_out(0x22E, 0x22F); // The two a draw is started with
        leave_out(0x232, 0x235); // Fixed attribute
        leave_out(0x238, 0x23D); // Where the command list is
        leave_out(0x25F, 0x25F); // Primitive restart
        leave_out(0x290, 0x298); // Geometry shader uniforms
        leave_out(0x29B, 0x2A3); // Geometry shader code
        leave_out(0x2A5, 0x2AD); // Geometry shader operand descriptors
        leave_out(0x2C0, 0x2C8); // Vertex shader uniforms
        leave_out(0x2CB, 0x2D3); // Vertex shader code
        leave_out(0x2D5, 0x2DD); // Vertex shader operand descriptors
        return bits;
    }();
    return shaping;
}

/// What a register with a value puts into the hash. The hash is these of all shaping
/// registers, one exclusive-or'ed with the next, so that a register changing takes its old
/// one out and puts its new one in, and the order of the writes does not matter.
constexpr u64 MixRegister(u32 id, u32 value) {
    u64 x = (static_cast<u64>(id) << 32) | value;
    x *= 0x9E3779B97F4A7C15ull;
    x ^= x >> 32;
    x *= 0xD6E8FEB86659FD93ull;
    x ^= x >> 32;
    return x;
}

} // Anonymous namespace

void RasterizerAccelerated::FoldWrittenRegisters() {
    const std::array<u64, 12>& shaping = ShapingRegisterBits();
    for (std::size_t word = 0; word < shaping.size(); word++) {
        u64 written = pica.dirty_regs.qwords[word] & shaping[word];
        while (written != 0) {
            const u32 id = static_cast<u32>(word * 64 + std::countr_zero(written));
            written &= written - 1;
            const u32 value = regs.reg_array[id];
            if (value != hashed_registers[id]) {
                shaping_hash ^= MixRegister(id, hashed_registers[id]) ^ MixRegister(id, value);
                hashed_registers[id] = value;
            }
        }
    }
}

u64 RasterizerAccelerated::ShapingHash() {
    FoldWrittenRegisters();
    return shaping_hash;
}

#ifdef ENABLE_FREECAM
void RasterizerAccelerated::SetFreeCam(const std::array<f32, 16>& matrix, bool skip_2d) {
    static_assert(sizeof(vs_data.freecam_matrix) == sizeof(matrix));
    std::memcpy(vs_data.freecam_matrix.data(), matrix.data(), sizeof(matrix));
    vs_data.freecam_skip_2d = skip_2d ? 1u : 0u;
    vs_data_dirty = true;
}
#endif

void RasterizerAccelerated::SetWideFactor(f32 factor) {
    pica.SetWideFactor(factor);
}

void RasterizerAccelerated::SyncDrawUniforms() {
    auto& dirty = pica.dirty_regs;

    // How this draw is scaled for a top screen shown at another shape than the console's
    const auto& wide = pica.GetWideScales();
    if (const Common::Vec4f scales{wide.x, wide.y, wide.flat, 0.f}; scales != vs_data.wide) {
        vs_data.wide = scales;
        vs_data_dirty = true;
    }

    // The register that contains the flip bit also contains the framebuffer dimentions
    // that we don't depend on. So avoid the dirty table and check manually
    const bool is_flipped = regs.framebuffer.framebuffer.IsFlipped();
    const bool prev_flipped = std::exchange(vs_data.flip_viewport, is_flipped);
    vs_data_dirty |= is_flipped != prev_flipped;

    // Sync clip plane uniforms
    if (dirty.CheckClipping()) {
        const auto raw_clip_coef = regs.rasterizer.GetClipCoef();
        vs_data.enable_clip1 = regs.rasterizer.clip_enable != 0;
        vs_data.clip_coef = {raw_clip_coef.x.ToFloat32(), raw_clip_coef.y.ToFloat32(),
                             raw_clip_coef.z.ToFloat32(), raw_clip_coef.w.ToFloat32()};
        vs_data_dirty = true;
    }

    // Sync depth testing uniforms
    if (dirty.CheckDepth()) {
        fs_data.depth_scale = f24::FromRaw(regs.rasterizer.viewport_depth_range).ToFloat32();
        fs_data.depth_offset = f24::FromRaw(regs.rasterizer.viewport_depth_near_plane).ToFloat32();
        fs_data_dirty = true;
    }

    // Sync alpha testing and blending uniforms
    if (dirty.CheckBlend()) {
        fs_data.alphatest_ref = regs.framebuffer.output_merger.alpha_test.ref;
        fs_data.blend_color = ColorRGBA8(regs.framebuffer.output_merger.blend_const.raw);
        fs_data_dirty = true;
    }

    // Sync texture unit uniforms
    if (dirty.CheckTexUnits()) {
        const auto pica_textures = regs.texturing.GetTextures();
        for (u32 tex_index = 0; tex_index < 3; tex_index++) {
            const auto& config = pica_textures[tex_index].config;
            fs_data.tex_lod_bias[tex_index] = config.lod.bias / 256.0f;
            fs_data.tex_border_color[tex_index] = ColorRGBA8(config.border_color.raw);
        }
        fs_data_dirty = true;
    }

    // Shaders made the old way (a profiling switch) take these values as they used to be
    const bool legacy_shaders =
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyStageRounding) != 0;

    // Sync texenv uniforms
    if (dirty.CheckTexEnv()) {
        const auto tev_stages = regs.texturing.GetTevStages();
        for (std::size_t index = 0; index < tev_stages.size(); ++index) {
            fs_data.const_color[index] = legacy_shaders
                                             ? ColorRGBA8(tev_stages[index].const_color)
                                             : CombinerColorRGBA8(tev_stages[index].const_color);
        }
        fs_data.tev_combiner_buffer_color =
            legacy_shaders ? ColorRGBA8(regs.texturing.tev_combiner_buffer_color.raw)
                           : CombinerColorRGBA8(regs.texturing.tev_combiner_buffer_color.raw);
        fs_data_dirty = true;
    }

    // Sync global lighting uniforms
    if (dirty.CheckLightingAmbient()) {
        fs_data.lighting_global_ambient = LightColor(regs.lighting.global_ambient);
        fs_data_dirty = true;
    }

    // Sync light uniforms
    for (u32 light_index = 0; light_index < 8; light_index++) {
        if (!dirty.CheckLight(light_index)) {
            continue;
        }

        const auto& light = regs.lighting.light[light_index];
        fs_data.light_src[light_index].specular_0 = LightColor(light.specular_0);
        fs_data.light_src[light_index].specular_1 = LightColor(light.specular_1);
        fs_data.light_src[light_index].diffuse = LightColor(light.diffuse);
        fs_data.light_src[light_index].ambient = LightColor(light.ambient);
        Common::Vec3f position{
            Pica::f16::FromRaw(light.x).ToFloat32(),
            Pica::f16::FromRaw(light.y).ToFloat32(),
            Pica::f16::FromRaw(light.z).ToFloat32(),
        };
        f32 position_length = 0.f;
        if (light.config.directional && !legacy_shaders) {
            // The same for every fragment: the shaders take the direction and the distance
            // as they are
            position_length = position.Length();
            if (position_length > 0.f) {
                position = position / position_length;
            }
        }
        fs_data.light_src[light_index].position = position;
        fs_data.light_src[light_index].position_length = position_length;
        fs_data.light_src[light_index].spot_direction = {
            light.spot_x / 2047.0f, light.spot_y / 2047.0f, light.spot_z / 2047.0f};
        fs_data.light_src[light_index].dist_atten_bias =
            Pica::f20::FromRaw(light.dist_atten_bias).ToFloat32();
        fs_data.light_src[light_index].dist_atten_scale =
            Pica::f20::FromRaw(light.dist_atten_scale).ToFloat32();
        fs_data_dirty = true;
    }

    // Sync fog uniforms
    if (dirty.CheckFogColor()) {
        fs_data.fog_color = {
            regs.texturing.fog_color.r.Value() / 255.0f,
            regs.texturing.fog_color.g.Value() / 255.0f,
            regs.texturing.fog_color.b.Value() / 255.0f,
        };
        fs_data_dirty = true;
    }

    // Sync proctex uniforms
    if (dirty.CheckProctex()) {
        fs_data.proctex_noise_f = {
            Pica::f16::FromRaw(regs.texturing.proctex_noise_frequency.u).ToFloat32(),
            Pica::f16::FromRaw(regs.texturing.proctex_noise_frequency.v).ToFloat32(),
        };
        fs_data.proctex_noise_a = {
            regs.texturing.proctex_noise_u.amplitude / 4095.0f,
            regs.texturing.proctex_noise_v.amplitude / 4095.0f,
        };
        fs_data.proctex_noise_p = {
            Pica::f16::FromRaw(regs.texturing.proctex_noise_u.phase).ToFloat32(),
            Pica::f16::FromRaw(regs.texturing.proctex_noise_v.phase).ToFloat32(),
        };
        fs_data.proctex_bias = Pica::f16::FromRaw(regs.texturing.proctex.bias_low |
                                                  (regs.texturing.proctex_lut.bias_high << 8))
                                   .ToFloat32();
        fs_data_dirty = true;
    }

    // Sync shadow uniforms
    if (dirty.CheckShadow()) {
        const auto& shadow = regs.framebuffer.shadow;
        fs_data.shadow_bias_constant = Pica::f16::FromRaw(shadow.constant).ToFloat32();
        fs_data.shadow_bias_linear = Pica::f16::FromRaw(shadow.linear).ToFloat32();
        fs_data.shadow_texture_bias = regs.texturing.shadow.bias << 1;
        fs_data_dirty = true;
    }

    // What was written goes into the hash of the shaping registers before it is forgotten
    FoldWrittenRegisters();

    // We have synched all uniforms, reset dirty state.
    pica.dirty_regs.Reset();
}

} // namespace VideoCore
