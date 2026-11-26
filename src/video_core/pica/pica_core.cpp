// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <nihstro/shader_bytecode.h>
#include "common/arch.h"
#include "common/archives.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "common/hitch_tracker.h"
#include "common/math_util.h"
#include "common/microprofile.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/frame_state_dump.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/used_of_light.h"
#include "video_core/pica/vertex_loader.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/shader/debug_data.h"
#include "video_core/shader/shader.h"
#include "video_core/shader/shader_interpreter.h"
#include "video_core/work_rules.h"

namespace Pica {

namespace {

/// For the hitch summary: how much of a list that a command list calls into is what was at
/// that address the last time it was called. Emulation thread only.
void MeasureCalledList(const u8* data, u32 size, PAddr address) {
    static std::unordered_map<PAddr, std::vector<u32>> last_lists;
    if (!data || size < 4) {
        return;
    }
    // A game that calls into ever new places is not to fill the memory with them
    if (last_lists.size() > 4096) {
        last_lists.clear();
    }
    std::vector<u32> list(size / 4);
    std::memcpy(list.data(), data, list.size() * 4);
    std::vector<u32>& previous = last_lists[address];
    const std::size_t common = std::min(list.size(), previous.size());
    u32 same = 0;
    for (std::size_t i = 0; i < common; i++) {
        same += list[i] == previous[i];
    }
    Common::Hitches::NoteCalledListWords(static_cast<u32>(list.size()), same,
                                         list.size() == previous.size() && same == list.size());
    previous = std::move(list);
}

/// For the hitch summary: whether a draw finds the registers as they were the last time a draw
/// was issued from the same place of the same command list, and if not, where they differ.
/// Left out of the comparison are the registers that data only passes through on its way
/// elsewhere (uniform values, shader code, lookup tables, fixed attributes), the ones that say
/// where the command list is, and the two a draw is started with. Emulation thread only.
void MeasureDrawSite(PAddr list, u32 index, const std::array<u32, 0x300>& now) {
    using Common::Hitches::DrawSite;
    static std::unordered_map<u64, std::array<u32, 0x300>> last_registers;
    static const std::array<bool, 0x300> passes_through = [] {
        std::array<bool, 0x300> flags{};
        const auto mark = [&flags](u32 first, u32 last) {
            for (u32 id = first; id <= last; id++) {
                flags[id] = true;
            }
        };
        mark(0x000, 0x03F); // Interrupts and the end of a list
        mark(0x0AF, 0x0B7); // Procedural texture tables
        mark(0x0E6, 0x0EF); // Fog table
        mark(0x1C5, 0x1CF); // Lighting tables
        mark(0x22E, 0x22F); // The two a draw is started with
        mark(0x232, 0x235); // Fixed attribute
        mark(0x238, 0x23D); // Where the command list is
        mark(0x25F, 0x25F); // Primitive restart
        mark(0x290, 0x298); // Geometry shader uniforms
        mark(0x29B, 0x2A3); // Geometry shader code
        mark(0x2A5, 0x2AD); // Geometry shader operand descriptors
        mark(0x2C0, 0x2C8); // Vertex shader uniforms
        mark(0x2CB, 0x2D3); // Vertex shader code
        mark(0x2D5, 0x2DD); // Vertex shader operand descriptors
        return flags;
    }();

    if (last_registers.size() > 20000) {
        last_registers.clear();
    }
    Common::Hitches::NoteDrawSite(DrawSite::Draws);
    const auto [it, fresh] = last_registers.try_emplace((static_cast<u64>(list) << 32) | index);
    if (fresh) {
        Common::Hitches::NoteDrawSite(DrawSite::Fresh);
    } else {
        const std::array<u32, 0x300>& before = it->second;
        std::array<bool, 0x300 / 0x40> differs{};
        bool any = false;
        for (u32 id = 0; id < 0x300; id++) {
            if (now[id] != before[id] && !passes_through[id]) {
                differs[id / 0x40] = true;
                any = true;
            }
        }
        if (!any) {
            Common::Hitches::NoteDrawSite(DrawSite::Same);
        } else {
            if (differs[1]) {
                Common::Hitches::NoteDrawSite(DrawSite::Rasterizer);
            }
            if (differs[2] || differs[3]) {
                Common::Hitches::NoteDrawSite(DrawSite::Texturing);
            }
            if (differs[4]) {
                Common::Hitches::NoteDrawSite(DrawSite::Framebuffer);
            }
            if (differs[5] || differs[6] || differs[7]) {
                Common::Hitches::NoteDrawSite(DrawSite::Lighting);
            }
            if (differs[8] || differs[9]) {
                Common::Hitches::NoteDrawSite(DrawSite::Geometry);
            }
            if (differs[10]) {
                Common::Hitches::NoteDrawSite(DrawSite::GeometryShader);
            }
            if (differs[11]) {
                Common::Hitches::NoteDrawSite(DrawSite::VertexShader);
            }
        }
    }
    it->second = now;
}

} // Anonymous namespace

MICROPROFILE_DEFINE(GPU_Drawing, "GPU", "Drawing", MP_RGB(50, 50, 240));

using namespace DebugUtils;

// Class representing implementation details of each internal register
// The set/get pattern is used instead of a bitfield union to allow
// constexpr evaluation.
class RegImplInfo {
private:
    using NeedsSpecialHandlingBF = BitField<0, 1, u16>;
    using SupportsBatchBF = BitField<1, 1, u16>;
    using RegsUntilSpecialBF = BitField<2, 14, u16>;

    u16 raw{};

public:
    constexpr bool NeedsSpecialHandling() const {
        return NeedsSpecialHandlingBF::ExtractValue(raw) != 0;
    }

    constexpr void SetNeedsSpecialHandling() {
        raw = (raw & ~NeedsSpecialHandlingBF::mask) | NeedsSpecialHandlingBF::FormatValue(1);
    }

    constexpr bool SupportsBatch() const {
        return SupportsBatchBF::ExtractValue(raw) != 0;
    }

    constexpr void SetSupportsBatch() {
        raw = (raw & ~SupportsBatchBF::mask) | SupportsBatchBF::FormatValue(1);
    }

    constexpr u16 RegsUntilSpecial() const {
        return RegsUntilSpecialBF::ExtractValue(raw);
    }

    constexpr void SetRegsUntilSpecial(u16 value) {
        raw = (raw & ~RegsUntilSpecialBF::mask) | RegsUntilSpecialBF::FormatValue(value);
    }
};

union CommandHeader {
    u32 hex;
    BitField<0, 16, u32> cmd_id;
    BitField<16, 4, u32> parameter_mask;
    BitField<20, 8, u32> extra_data_length;
    BitField<31, 1, u32> group_commands;
};
static_assert(sizeof(CommandHeader) == sizeof(u32), "CommandHeader has incorrect size!");

PicaCore::PicaCore(Memory::MemorySystem& memory_, std::shared_ptr<DebugContext> debug_context_)
    : memory{memory_}, debug_context{std::move(debug_context_)},
      geometry_pipeline{regs.internal, gs_unit, gs_setup},
      shader_engine{CreateEngine(Settings::values.use_shader_jit.GetValue())},
      few_vertices_engine{CreateEngine(false)} {
    InitializeRegs();
    dirty_regs.SetAllDirty();

    const auto submit_vertex = [this](const AttributeBuffer& buffer) {
        const auto add_triangle = [this](const OutputVertex& v0, const OutputVertex& v1,
                                         const OutputVertex& v2) {
            rasterizer->AddTriangle(v0, v1, v2);
        };
        const auto vertex = OutputVertex(regs.internal.rasterizer, buffer);
        primitive_assembler.SubmitVertex(vertex, add_triangle);
    };

    gs_unit.SetVertexHandlers(submit_vertex, [this]() { primitive_assembler.SetWinding(); });
    geometry_pipeline.SetVertexHandler(submit_vertex);

    primitive_assembler.Reconfigure(PipelineRegs::TriangleTopology::List);
}

PicaCore::~PicaCore() = default;

void PicaCore::InitializeRegs() {
    // Values initialized by GSP
    regs.internal.irq_autostop = 1;
    regs.internal.irq_mask = 0xFFFFFFF0;
    // Older versions of libctru didn't initialize this, initialize it here to avoid endless black
    // screen. Not needed on actual hardware due to previous software already having set it up
    regs.internal.irq_compare = 0x12345678;

    auto& framebuffer_top = regs.framebuffer_config[0];
    auto& framebuffer_sub = regs.framebuffer_config[1];

    // Set framebuffer defaults from nn::gx::Initialize
    framebuffer_top.address_left1 = 0x181E6000;
    framebuffer_top.address_left2 = 0x1822C800;
    framebuffer_top.address_right1 = 0x18273000;
    framebuffer_top.address_right2 = 0x182B9800;
    framebuffer_sub.address_left1 = 0x1848F000;
    framebuffer_sub.address_left2 = 0x184C7800;

    framebuffer_top.width.Assign(240);
    framebuffer_top.height.Assign(400);
    framebuffer_top.stride = 3 * 240;
    framebuffer_top.color_format.Assign(PixelFormat::RGB8);
    framebuffer_top.active_fb = 0;

    framebuffer_sub.width.Assign(240);
    framebuffer_sub.height.Assign(320);
    framebuffer_sub.stride = 3 * 240;
    framebuffer_sub.color_format.Assign(PixelFormat::RGB8);
    framebuffer_sub.active_fb = 0;

    // Tales of Abyss expects this register to have the following default values.
    auto& gs = regs.internal.gs;
    gs.max_input_attribute_index.Assign(1);
    gs.shader_mode.Assign(ShaderRegs::ShaderMode::VS);
}

void PicaCore::BindRasterizer(VideoCore::RasterizerInterface* rasterizer) {
    this->rasterizer = rasterizer;
}

void PicaCore::SetInterruptHandler(Service::GSP::InterruptHandler& signal_interrupt) {
    this->signal_interrupt = signal_interrupt;
}

static consteval std::array<RegImplInfo, RegsInternal::NUM_REGS> BuildRegImplFlagsLUT() {
    std::array<RegImplInfo, RegsInternal::NUM_REGS> table{};

    // Marks the register as needing special handling.
    const auto mark_special = [&table](u32 index) { table[index].SetNeedsSpecialHandling(); };

    // Marks the register as supporting batch writes.
    const auto mark_batch = [&table](u32 index) { table[index].SetSupportsBatch(); };

    // Single registers
    mark_special(PICA_REG_INDEX(irq_request));
    mark_special(PICA_REG_INDEX(pipeline.triangle_topology));
    mark_special(PICA_REG_INDEX(pipeline.restart_primitive));
    mark_special(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index));
    mark_special(PICA_REG_INDEX(pipeline.trigger_draw));
    mark_special(PICA_REG_INDEX(pipeline.trigger_draw_indexed));
    mark_special(PICA_REG_INDEX(gs.bool_uniforms));
    mark_special(PICA_REG_INDEX(vs.output_mask));
    mark_special(PICA_REG_INDEX(vs.bool_uniforms));

    // Array based registers
    for (u32 i = 0; i < 2; ++i) {
        mark_special(PICA_REG_INDEX(pipeline.command_buffer.trigger[0]) + i);
    }
    for (u32 i = 0; i < 3; ++i) {
        mark_special(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]) + i);
        mark_batch(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]) + i);
    }
    for (u32 i = 0; i < 4; ++i) {
        mark_special(PICA_REG_INDEX(vs.int_uniforms[0]) + i);
        mark_special(PICA_REG_INDEX(gs.int_uniforms[0]) + i);
    }
    for (u32 i = 0; i < 8; ++i) {
        mark_special(PICA_REG_INDEX(gs.uniform_setup.set_value[0]) + i);
        mark_batch(PICA_REG_INDEX(gs.uniform_setup.set_value[0]) + i);
        mark_special(PICA_REG_INDEX(vs.uniform_setup.set_value[0]) + i);
        mark_batch(PICA_REG_INDEX(vs.uniform_setup.set_value[0]) + i);
        mark_special(PICA_REG_INDEX(gs.program.set_word[0]) + i);
        mark_batch(PICA_REG_INDEX(gs.program.set_word[0]) + i);
        mark_special(PICA_REG_INDEX(vs.program.set_word[0]) + i);
        mark_batch(PICA_REG_INDEX(vs.program.set_word[0]) + i);
        mark_special(PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]) + i);
        mark_batch(PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]) + i);
        mark_special(PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]) + i);
        mark_batch(PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]) + i);
        mark_special(PICA_REG_INDEX(lighting.lut_data[0]) + i);
        mark_batch(PICA_REG_INDEX(lighting.lut_data[0]) + i);
        mark_special(PICA_REG_INDEX(texturing.fog_lut_data[0]) + i);
        mark_batch(PICA_REG_INDEX(texturing.fog_lut_data[0]) + i);
        mark_special(PICA_REG_INDEX(texturing.proctex_lut_data[0]) + i);
        mark_batch(PICA_REG_INDEX(texturing.proctex_lut_data[0]) + i);
    }

    // Build distances to next special register.
    u16 regs_since_special = std::numeric_limits<u16>::max();
    for (size_t i = RegsInternal::NUM_REGS; i-- > 0;) {
        if (table[i].NeedsSpecialHandling()) {
            regs_since_special = 0;
        }
        table[i].SetRegsUntilSpecial(regs_since_special);
        if (regs_since_special != std::numeric_limits<u16>::max()) {
            regs_since_special++;
        }
    }

    return table;
}

static constexpr std::array<RegImplInfo, RegsInternal::NUM_REGS> reg_impl_flags_lut =
    BuildRegImplFlagsLUT();

// Expand a 4-bit mask to 4-byte mask, e.g. 0b0101 -> 0x00FF00FF
static constexpr std::array<u32, 16> ExpandBitsToBytes = {
    0x00000000, 0x000000ff, 0x0000ff00, 0x0000ffff, 0x00ff0000, 0x00ff00ff, 0x00ffff00, 0x00ffffff,
    0xff000000, 0xff0000ff, 0xff00ff00, 0xff00ffff, 0xffff0000, 0xffff00ff, 0xffffff00, 0xffffffff,
};

/**
 * This is the main loop for processing GPU command lists. On Azahar, it's the most
 * CPU expensive function (excluding the inner Draw calls) due to applications submitting
 * 10-50 command lists per frame, each with hundreds of commands in them. For this reason,
 * it is important that this function is well optimized to reduce the load on the CPU.
 *
 * Each command in the list has the following properties:
 *  - Commands come in [value (32 bit), header (32 bit)] pairs, most of the time.
 *  - The register ID that the 32 bit value should be written to is stored in the header.
 *  - The mask of bits that should be written comes in the header
 *    (to be able to write individual bytes of the 4-byte register)
 *  - Commands can have an extra length N, which means that N extra words follow
 *    after the header word.
 *    - If group_command is set in the header, N sequential registers are
 *      written to starting from the ID + 1 specified in the header.
 *    - If group_command is not set, the same register is written
 *      to with the N extra words. This is used for things like shader uploads
 *      which has a single register ID.
 *
 * Regarding implementation details, we store all register values in an array,
 * as well as a dirty array to indicate which registers have changed since
 * the last draw. Some registers need special handling, as they are "trigger"
 * registers that start the draw, or store the data in the shader units.
 *
 * To be able to determine if a register is special, we use a lookup table
 * generated by BuildRegImplFlagsLUT(). This allows determining if a register
 * is special or not in O(1). This LUT also determines if a register has
 * support for batch handling (extra_data_length != 0 && group_command == 0)
 * and the amount of commands away from the next special register, useful for
 * sequential writes (extra_data_length != 0 && group_command == 1).
 *
 * As much as possible, we want to target the following optimizations:
 *  - We should prevent branches and jumps to functions if they are not needed.
 *  - We should clearly separate special command handling from normal commands
 *    that are much cheaper to handle.
 *  - Commands with extra length should be processed in batch if possible.
 *  - Vectorization should be used as much as possible.
 *
 * On the other hand, if PICA debugging is enabled we should avoid optimizations
 * that would make debugging more complicated.
 */
void PicaCore::ProcessCmdList(PAddr list, u32 size, bool ignore_list) [[hot]] {
    if (ignore_list) {
        list_copy = nullptr;
        signal_interrupt(Service::GSP::InterruptId::P3D, delay_generator.CalculateAndResetDelay());
        return;
    }

    const u8* head =
        list_copy ? std::exchange(list_copy, nullptr) : memory.GetPhysicalPointer(list);
    cmd_list.Reset(list, head, size);

    bool stop_requested = false;
    bool skip_fast_path = false;
    while (cmd_list.current_index < cmd_list.length) {
        if (stop_requested) [[unlikely]] {
            break;
        }
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        // Early path that processes commands in batches of 4. If any of the commands
        // needs special handling or has extra length it stops and falls back to the
        // slower path. This pattern allows the compiler to auto-vectorize the function
        // if the current ISA allows it (that's why we process in batches of 4
        // as most SIMD operations work with 128 bit registers). MSVC is not able to
        // auto-vectorize this part with SSE4.2, due to the LUT read, instead it just
        // unrolls the loop. Other ISAs and/or compilers may be able to do it,
        // that's why it was decided to keep the structure like this.
        if (!debug_context) [[likely]] {
            if (!skip_fast_path) {
                constexpr u32 batch_size = 4;
                u32 index = cmd_list.current_index;
                u32 ids[batch_size], values[batch_size], masks[batch_size];
                u32 run = 0;

                while (run < batch_size && index + 1 < cmd_list.length) {
                    const u32 value = cmd_list.head[index];
                    const CommandHeader header{cmd_list.head[index + 1]};

                    // If extra handling is needed stop and fallback to slower path.
                    if (header.extra_data_length != 0 || header.cmd_id >= RegsInternal::NUM_REGS ||
                        reg_impl_flags_lut[header.cmd_id].NeedsSpecialHandling()) {
                        skip_fast_path = true;
                        break;
                    }

                    ids[run] = header.cmd_id;
                    values[run] = value;
                    masks[run] = header.parameter_mask;
                    ++run;
                    index += 2;
                }

                // Process the commands that we have read so far (up to 4).
                if (run > 0) {
                    delay_generator.AddCommands(run);
                    for (u32 i = 0; i < run; ++i) {
                        const u32 id = ids[i];
                        const u32 write_mask = ExpandBitsToBytes[masks[i]];
                        const u32 old_value = regs.internal.reg_array[id];
                        const u32 new_value = (old_value & ~write_mask) | (values[i] & write_mask);
                        // Games set their materials anew before every draw, mostly to what
                        // they are: that is not a change
                        if (new_value == old_value) {
                            continue;
                        }
                        if (sw_batch_open && id < FIRST_VERTEX_REG) [[unlikely]] {
                            FlushSoftwareBatch();
                        }
                        regs.internal.reg_array[id] = new_value;
                        dirty_regs.Set(id);
                    }
                    cmd_list.current_index = index;

                    // Continue from the while loop in case we reached the end of the list.
                    continue;
                }
            }
        }
        // Slow path, command needs special handling.

        skip_fast_path = false;

        // Read the header and the value to write.
        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        // Write to the requested PICA register.
        WriteInternalReg(header.cmd_id, value, header.parameter_mask, stop_requested);

        // Write any extra paramters as well.
        const u32 count = header.extra_data_length;
        if (count == 0)
            continue;

        if (debug_context) [[unlikely]] {
            // Fallback to per word register writes if debugging is
            // enabled.
            for (u32 i = 0; i < count; ++i) {
                if (stop_requested) [[unlikely]] {
                    break;
                }
                const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
                const u32 extra_value = cmd_list.head[cmd_list.current_index++];
                WriteInternalReg(cmd, extra_value, header.parameter_mask, stop_requested);
            }
        } else {
            // Handle commands with extra length.
            const u32* extra = &cmd_list.head[cmd_list.current_index];
            cmd_list.current_index += count;

            if (!header.group_commands) {
                // Same register written count times in a row (program/swizzle upload, LUT, etc.).
                WriteInternalRegBatch(header.cmd_id, extra, count, header.parameter_mask,
                                      stop_requested);
            } else {
                // Sequential registers written from header.cmd_id+1 to header.cmd_id+count.
                WriteInternalRegSequential(header.cmd_id + 1, extra, count, header.parameter_mask,
                                           stop_requested);
            }
        }
    }
    FlushSoftwareBatch();
}

template <typename Func>
void PicaCore::ForTwinRegisters(bool clears, Func&& func, bool light_vectors) const {
    const auto& texturing = regs.internal.texturing;
    // What the fragments go by. Culling, viewport, depth range and scissor, and the output
    // map as far as it is used
    func(0x040, 0x050);
    func(0x050, 0x050 + std::min<u32>(regs.internal.rasterizer.vs_output_total, 7));
    func(0x057, 0x070);
    // The texture units that are on (what an idle one holds was left there by earlier draws
    // and differs between the two passes), and the combiner
    func(0x080, 0x081);
    if (texturing.main_config.texture0_enable) {
        func(0x081, 0x08F);
    }
    if (texturing.main_config.texture1_enable) {
        func(0x091, 0x097);
    }
    if (texturing.main_config.texture2_enable) {
        func(0x099, 0x09F);
    }
    if (texturing.main_config.texture3_enable) {
        func(0x0A8, 0x0AE);
    }
    // (The fog's colour and the gas where the draw has fog, and what is left there by other draws
    // where it has none. Not the fog table's port, which holds what was last put through it: that
    // was before the draw or after the one before it, as it comes, and what a draw goes by is the
    // table, see TwinState)
    func(0x0C0, PICA_REG_INDEX(texturing.fog_color));
    if (texturing.fog_mode != TexturingRegs::FogMode::None) {
        func(PICA_REG_INDEX(texturing.fog_color), PICA_REG_INDEX(texturing.fog_lut_offset));
    }
    func(PICA_REG_INDEX(texturing.fog_lut_data[0]) + 8, 0x100);
    // Blending, the tests and the masks; the target
    func(0x100, 0x108);
    func(0x116, 0x11F);
    // Whether the fragments are lit at all
    func(PICA_REG_INDEX(lighting.disable), PICA_REG_INDEX(lighting.disable) + 1);
    if (clears) {
        // A clear done with a quad: which program of the vertex shader's memory it runs does
        // not matter, and what else that memory holds changes between the two
        return;
    }
    // The lights; where the vertices and indices are and how many; the stages after the
    // vertex shader; the vertex shader's configuration
    if (const auto& lighting = regs.internal.lighting; !lighting.disable) {
        // The lights that are on, as with the texture units (Metal Gear Solid: Snake Eater 3D
        // draws most of a scene with the first of two, and the second holds what the frame
        // before or this frame last set)
        for (u32 index = 0; index <= lighting.max_light_index; index++) {
            // (Its colours say whether the rest of it is used)
            const u32 light = 0x140 + lighting.light_enable.GetNum(index) * 0x10;
            const UsedOfLight used = UsedOf(lighting, index);
            func(light, light + 0x04);
            if (used.vector) {
                if (light_vectors) {
                    func(light + 0x04, light + 0x06);
                }
                func(light + 0x09, light + 0x0A);
            }
            if (used.spot && light_vectors) {
                func(light + 0x06, light + 0x08);
            }
            if (used.distance) {
                func(light + 0x0A, light + 0x0C);
            }
        }
        func(0x1C0, 0x1C5);
        func(0x1D0, 0x1D3);
        func(0x1D9, 0x1DA);
    }
    func(0x200, 0x22B);
    func(0x242, 0x243);
    func(0x244, 0x245);
    func(0x24A, 0x24B);
    func(0x251, 0x254);
    func(0x25E, 0x25F);
    func(0x2B0, 0x2BE);
    if (regs.internal.pipeline.use_gs != PipelineRegs::UseGS::No) {
        func(0x280, 0x28E);
    }
}

u64 PicaCore::TwinState(bool clears, bool light_vectors) {
    const auto& array = regs.internal.reg_array;
    u64 state = clears ? 1 : light_vectors ? 2 : 3;
    ForTwinRegisters(
        clears,
        [&](u32 from, u32 to) {
            state = Common::HashCombine(
                state, Common::ComputeHash64(&array[from], (to - from) * sizeof(u32)));
        },
        light_vectors);
    if (regs.internal.texturing.fog_mode != TexturingRegs::FogMode::None) {
        state = Common::HashCombine(state, Common::ComputeHash64(fog.lut.data(), sizeof(fog.lut)));
    }
    if (clears) {
        return state;
    }
    state = Common::HashCombine(state, RunnableHash(vs_setup, regs.internal.vs.main_offset));
    if (regs.internal.pipeline.use_gs != PipelineRegs::UseGS::No) {
        state = Common::HashCombine(state, RunnableHash(gs_setup, regs.internal.gs.main_offset));
    }
    return state;
}

u64 PicaCore::RunnableHash(ShaderSetup& setup, u32 main) {
    const u64 key =
        Common::HashCombine(setup.GetProgramCodeHash(), setup.GetSwizzleDataHash(), u64{main});
    auto& [known, hash] = runnable_hashes[(key * 0x9E3779B97F4A7C15ull) >> 56];
    if (known == key) {
        return hash;
    }
    const auto& code = setup.GetProgramCode();
    const auto& swizzles = setup.GetSwizzleData();
    const u32 size = static_cast<u32>(code.size());
    // Everything that can be reached, whichever way the branches go
    std::vector<bool> seen(size);
    std::vector<std::pair<u32, u32>> stretches{{main, size}};
    while (!stretches.empty()) {
        auto [at, end] = stretches.back();
        stretches.pop_back();
        for (; at < end && at < size; at++) {
            const u32 word = code[at];
            const u32 opcode = word >> 26;
            const bool was_seen = seen[at];
            seen[at] = true;
            if (opcode == 0x22) {
                // END
                break;
            }
            if (was_seen) {
                continue;
            }
            const u32 dest = (word >> 10) & 0xFFF;
            const u32 count = word & 0xFF;
            switch (opcode) {
            case 0x24: // CALL
            case 0x25: // CALLC
            case 0x26: // CALLU
                stretches.emplace_back(dest, dest + count);
                break;
            case 0x27: // IFU
            case 0x28: // IFC
                // The part for the other case, and what follows both, which the part gone
                // through here may not come to
                stretches.emplace_back(dest, dest + count);
                stretches.emplace_back(dest + count, end);
                break;
            case 0x29: // LOOP
                stretches.emplace_back(dest + 1, end);
                break;
            case 0x2C: // JMPC
            case 0x2D: // JMPU
                stretches.emplace_back(dest, size);
                break;
            default:
                break;
            }
        }
    }
    u64 result = 0;
    for (u32 at = 0; at < size; at++) {
        if (!seen[at]) {
            continue;
        }
        const u32 word = code[at];
        const u32 opcode = word >> 26;
        result = Common::HashCombine(result, (u64{at} << 32) | word);
        if (opcode < 0x20 || opcode > 0x2D) {
            // Not one that only controls the flow: it has operands
            const u32 pattern = opcode >= 0x30 ? word & 0x1F : word & 0x7F;
            result = Common::HashCombine(result, u64{swizzles[pattern]});
        }
    }
    known = key;
    hash = result;
    return result;
}

const PicaCore::UniformSet& PicaCore::FloatUniformsRead() {
    const u32 main = regs.internal.vs.main_offset;
    // The shader's bool uniforms decide which of its branches run: a game has one program
    // serve many kinds of draw, and switches parts of it on and off with them. The part that
    // is off may read uniforms the game does not set for this kind of draw at all, which hold
    // what another draw left there (New Super Mario Bros. 2: the first draws of each eye's run
    // would read uniform 14 on a branch they do not take, and it is not the same before the
    // two runs). So what is read is found for the bool uniforms as they are.
    u32 bools = 0;
    for (u32 i = 0; i < vs_setup.uniforms.b.size(); i++) {
        bools |= (vs_setup.uniforms.b[i] ? 1u : 0u) << i;
    }
    const u64 key =
        Common::HashCombine(RunnableHash(vs_setup, main), u64{main} | (u64{bools} << 32));
    for (const auto& [known, set] : uniforms_read) {
        if (known == key) {
            return set;
        }
    }
    const auto& code = vs_setup.GetProgramCode();
    const u32 size = static_cast<u32>(code.size());
    // Goes through what the program can run from its entry. `by_bools`: a branch on a bool
    // uniform is followed the way it goes; otherwise both ways, as a branch on what the
    // program worked out always is. False if that did not come to an end in good time.
    const auto walk = [&](bool by_bools, UniformSet& set) {
        set = {};
        std::vector<bool> seen(size);
        u64 steps = 0;
        // Stretches of the program still to go through: from, up to
        std::vector<std::pair<u32, u32>> stretches{{main, size}};
        while (!stretches.empty()) {
            auto [at, end] = stretches.back();
            stretches.pop_back();
            for (; at < end && at < size; at++) {
                if (++steps > u64{size} * 64) {
                    return false;
                }
                const u32 word = code[at];
                const u32 opcode = word >> 26;
                if (opcode == 0x22) {
                    // END
                    break;
                }
                const bool was_seen = seen[at];
                seen[at] = true;
                if (opcode >= 0x20 && opcode <= 0x2D) {
                    // Flow. What a call or a jump leads to is gone through as well; what an
                    // IF or a LOOP covers follows it and is gone through anyway, but for an
                    // IF on a bool uniform, of which only one part runs.
                    const u32 dest = (word >> 10) & 0xFFF;
                    const u32 count = word & 0xFF;
                    const bool on = ((bools >> ((word >> 22) & 0xF)) & 1) != 0;
                    bool leaves = false;
                    switch (opcode) {
                    case 0x24: // CALL
                    case 0x25: // CALLC
                        if (!was_seen) {
                            stretches.emplace_back(dest, dest + count);
                        }
                        break;
                    case 0x26: // CALLU
                        if ((on || !by_bools) && !was_seen) {
                            stretches.emplace_back(dest, dest + count);
                        }
                        break;
                    case 0x27: // IFU: from here to dest if it is on, the count from dest if not
                        if (by_bools && dest > at) {
                            if (on) {
                                stretches.emplace_back(dest + count, end);
                                end = dest;
                            } else {
                                at = dest - 1;
                            }
                        }
                        break;
                    case 0x2C: // JMPC
                        if (!was_seen) {
                            stretches.emplace_back(dest, size);
                        }
                        break;
                    case 0x2D: { // JMPU: taken if the bool is what the count's lowest bit asks
                        const bool taken = on == ((count & 1) == 0);
                        if ((taken || !by_bools) && !was_seen) {
                            stretches.emplace_back(dest, size);
                        }
                        leaves = by_bools && taken;
                        break;
                    }
                    default:
                        break;
                    }
                    if (leaves) {
                        break;
                    }
                    continue;
                }
                if (was_seen) {
                    continue;
                }
                // The operand that can be a float uniform, and whether it is indexed
                u32 reg = 0;
                u32 indexed = 0;
                if (opcode >= 0x38) {
                    // MAD
                    reg = (word >> 10) & 0x7F;
                    indexed = (word >> 22) & 3;
                } else if (opcode >= 0x30) {
                    // MADI
                    reg = (word >> 5) & 0x7F;
                    indexed = (word >> 22) & 3;
                } else if (opcode >= 0x18 && opcode <= 0x1B) {
                    // The instructions with the operands the other way round
                    reg = (word >> 7) & 0x7F;
                    indexed = (word >> 19) & 3;
                } else {
                    reg = (word >> 12) & 0x7F;
                    indexed = (word >> 19) & 3;
                }
                // An indexed read picks from a table (the bones of a skinned model). How far
                // into it is not known here, and what lies beyond a model's own bones is what
                // other models left there: tables are not compared.
                if (reg < 0x20 || indexed != 0) {
                    continue;
                }
                const u32 i = reg - 0x20;
                set[i / 64] |= u64{1} << (i % 64);
            }
        }
        return true;
    };
    UniformSet set{};
    if (!walk(true, set)) {
        walk(false, set);
    }
    if (uniforms_read.size() >= 256) {
        uniforms_read.clear();
    }
    return uniforms_read.emplace_back(key, set).second;
}

namespace {

/// What a value in a shader register depends on: which float uniforms, and whether the
/// vertex's own data
struct Taint {
    static constexpr u64 VERTEX = u64{1} << 63;
    u64 low{};
    /// Uniforms 64 to 95 in the low bits, the vertex in the top one
    u64 high{};

    static Taint Uniform(u32 index) {
        Taint taint;
        (index < 64 ? taint.low : taint.high) = u64{1} << (index % 64);
        return taint;
    }
    static Taint Vertex() {
        Taint taint;
        taint.high = VERTEX;
        return taint;
    }
    bool OfVertex() const {
        return (high & VERTEX) != 0;
    }
    Taint& operator|=(const Taint& other) {
        low |= other.low;
        high |= other.high;
        return *this;
    }
    friend Taint operator|(Taint a, const Taint& b) {
        return a |= b;
    }
};
using Taint4 = std::array<Taint, 4>;

/// What every register of a vertex shader depends on at one point of its run
struct TaintState {
    std::array<Taint4, 16> temporary{};
    std::array<Taint4, 16> output{};
    std::array<Taint, 2> condition{};
    std::array<Taint, 2> address{};

    /// Either this or the other: a register depends on what it depends on in one of them
    void Join(const TaintState& other) {
        for (std::size_t reg = 0; reg < temporary.size(); reg++) {
            for (std::size_t i = 0; i < 4; i++) {
                temporary[reg][i] |= other.temporary[reg][i];
                output[reg][i] |= other.output[reg][i];
            }
        }
        for (std::size_t i = 0; i < 2; i++) {
            condition[i] |= other.condition[i];
            address[i] |= other.address[i];
        }
    }
};

} // Anonymous namespace

const PicaCore::UniformsThatMatter* PicaCore::FloatUniformsThatMatter() {
    using nihstro::Instruction;
    using nihstro::OpCode;
    using nihstro::RegisterType;
    using nihstro::SourceRegister;
    using nihstro::SwizzlePattern;
    using Record = Shader::DebugDataRecord;

    const u32 main = regs.internal.vs.main_offset;
    const auto& uniforms = vs_setup.uniforms;
    u32 bools = 0;
    for (u32 i = 0; i < uniforms.b.size(); i++) {
        bools |= (uniforms.b[i] ? 1u : 0u) << i;
    }
    // The program, where it begins, and what else decides its way besides float uniforms
    // (and what decides which of its outputs the fragments are made of, see the end)
    const u64 key = Common::HashCombine(
        RunnableHash(vs_setup, main), u64{main} | (u64{bools} << 32),
        Common::ComputeHash64(&uniforms.i, sizeof(uniforms.i)), u64{regs.internal.vs.output_mask},
        Common::ComputeHash64(&regs.internal.reg_array[0x4F], 8 * sizeof(u32)),
        u64{regs.internal.reg_array[0x80]} | (u64{regs.internal.lighting.disable} << 32));
    const auto values_of = [&](const UniformSet& set) {
        u64 hash = 0;
        for (u32 i = 0; i < uniforms.f.size(); i++) {
            if ((set[i / 64] >> (i % 64) & 1) != 0) {
                hash = Common::HashCombine(
                    hash, Common::ComputeHash64(&uniforms.f[i], sizeof(uniforms.f[i])));
            }
        }
        return hash;
    };
    // A program has a result for each set of values of the uniforms its way depends on (a
    // few: what a game has differ from model to model). Most results of one program go by
    // the same uniforms, so their values are hashed once.
    {
        const UniformSet* hashed = nullptr;
        u64 hash = 0;
        for (const UniformsThatMatter& entry : uniforms_that_matter) {
            if (entry.key != key) {
                continue;
            }
            if (hashed == nullptr || *hashed != entry.control) {
                hashed = &entry.control;
                hash = values_of(entry.control);
            }
            if (entry.control_values == hash) {
                return entry.known ? &entry : nullptr;
            }
        }
    }

    // One run of the program, on a vertex of nothing but zeroes, with every instruction it
    // came by noted. What each register depends on is then followed along that run.
    //
    // A branch that depends only on uniforms goes the same way for every vertex, and for
    // other values of the other uniforms: those it depends on are kept with what they held
    // and count among those that matter. A branch that depends on the vertex goes the other
    // way for other vertices: the side this run did not take is gone through as well, if it
    // is plain arithmetic, and where the two sides meet a register depends on what it
    // depends on in either. Anything else that depends on the vertex, and nothing is told.
    //
    // A product with a uniform that holds zero is zero whatever the other factor is (on the
    // console also for an infinite one; not for one that is not a number, which a shader
    // gets from nothing a game gives it), and so depends on that uniform being zero and on
    // nothing else. That is how a game switches a part of its shader off: New Super Mario
    // Bros. 2 moves every vertex sideways for its eye by a term that ends in a product with
    // the depth of the 3D effect, and with that at nothing the uniform the term begins with
    // is not even set for the left eye.
    //
    // A table read at a place the vertex picks (the bones of a skinned model) is not
    // followed into, as in FloatUniformsRead.
    few_vertices_engine->SetupBatch(vs_setup, main);
    AttributeBuffer nothing;
    nothing.fill(Common::Vec4<f24>::AssignToAll(f24::Zero()));
    const auto trace = static_cast<const Shader::InterpreterEngine&>(*few_vertices_engine)
                           .ProduceDebugInfo(vs_setup, nothing, regs.internal.vs);
    const auto& code = vs_setup.GetProgramCode();
    const auto& swizzles = vs_setup.GetSwizzleData();

    TaintState state;
    std::array<s32, 2> address_value{};
    Common::Vec4<u8> loop{};
    // The uniforms that pin the way this run went and what it found (the branches that depend
    // on uniforms alone, the factors that are zero, where tables are read): with other values
    // in them another run has to be followed. And those a branch depends on together with
    // the vertex: both of its sides are followed, so they pin nothing, but which side a
    // vertex takes depends on them.
    Taint control;
    Taint influence;
    // What a reciprocal is taken of: there the sign of a zero is the sign of an infinity
    Taint reciprocal;
    bool known = true;
    // While the side of a branch that the run did not take is gone through: an address
    // register that is set there holds a value this run does not have. That is no matter
    // where the vertex picks it (what is read through it is a table that is not followed
    // into); anything else read through it cannot be told.
    bool other_side = false;
    std::array<bool, 2> address_unknown{};

    struct Source {
        Taint4 taints{};
        /// A float uniform read at its own place, and which
        bool uniform{};
        u32 index{};
    };
    const auto is_zero = [&](const Source& from, u32 component) {
        return from.uniform && uniforms.f[from.index][component].ToFloat32() == 0.f;
    };
    // What one arithmetic instruction makes of what the registers depend on
    const auto apply = [&](const Instruction& instr, TaintState& on) {
        const auto source = [&](const SourceRegister& reg, u32 address_index) {
            Source from;
            const u32 index = static_cast<u32>(reg.GetIndex());
            switch (reg.GetRegisterType()) {
            case RegisterType::Input:
                from.taints.fill(Taint::Vertex());
                break;
            case RegisterType::Temporary:
                from.taints = on.temporary[index % on.temporary.size()];
                break;
            case RegisterType::FloatUniform: {
                Taint whole;
                if (address_index == 0) {
                    if (index < 96) {
                        whole = Taint::Uniform(index);
                        from.uniform = true;
                        from.index = index;
                    }
                } else if (address_index == 3) {
                    // By the loop's counter: one of the registers the loop goes through
                    for (u32 turn = 0; turn <= loop.x; turn++) {
                        s32 offset = static_cast<s32>(loop.y) + static_cast<s32>(turn * loop.z);
                        if (offset > 127) {
                            offset = 0;
                        }
                        const u32 at = (index + static_cast<u32>(offset)) & 0x7F;
                        if (at < 96) {
                            whole |= Taint::Uniform(at);
                        }
                    }
                } else if (!on.address[address_index - 1].OfVertex()) {
                    if (address_unknown[address_index - 1]) {
                        known = false;
                    }
                    s32 offset = address_value[address_index - 1];
                    if (offset < -128 || offset > 127) {
                        offset = 0;
                    }
                    const u32 at = (index + static_cast<u32>(offset)) & 0x7F;
                    if (at < 96) {
                        whole = Taint::Uniform(at);
                    }
                    // (Where it reads depends on what the address was worked out from)
                    whole |= on.address[address_index - 1];
                    control |= on.address[address_index - 1];
                }
                from.taints.fill(whole);
                break;
            }
            default:
                break;
            }
            return from;
        };
        const auto destination = [&](const nihstro::DestRegister& reg) -> Taint4* {
            if (reg < 0x10) {
                return &on.output[reg.GetIndex() % on.output.size()];
            }
            if (reg < 0x20) {
                return &on.temporary[reg.GetIndex() % on.temporary.size()];
            }
            return nullptr;
        };
        const OpCode opcode = instr.opcode.Value();
        if (opcode.GetInfo().type == OpCode::Type::MultiplyAdd) {
            const SwizzlePattern swizzle = {swizzles[instr.mad.operand_desc_id]};
            const bool inverted = opcode.EffectiveOpCode() == OpCode::Id::MADI;
            const Source s1 = source(instr.mad.GetSrc1(inverted), 0);
            const Source s2 = source(instr.mad.GetSrc2(inverted),
                                     inverted ? 0 : instr.mad.address_register_index);
            const Source s3 = source(instr.mad.GetSrc3(inverted),
                                     inverted ? instr.mad.address_register_index : 0);
            const std::array<u32, 4> pick1 = {static_cast<u32>(swizzle.src1_selector_0.Value()),
                                              static_cast<u32>(swizzle.src1_selector_1.Value()),
                                              static_cast<u32>(swizzle.src1_selector_2.Value()),
                                              static_cast<u32>(swizzle.src1_selector_3.Value())};
            const std::array<u32, 4> pick2 = {static_cast<u32>(swizzle.src2_selector_0.Value()),
                                              static_cast<u32>(swizzle.src2_selector_1.Value()),
                                              static_cast<u32>(swizzle.src2_selector_2.Value()),
                                              static_cast<u32>(swizzle.src2_selector_3.Value())};
            const std::array<u32, 4> pick3 = {static_cast<u32>(swizzle.src3_selector_0.Value()),
                                              static_cast<u32>(swizzle.src3_selector_1.Value()),
                                              static_cast<u32>(swizzle.src3_selector_2.Value()),
                                              static_cast<u32>(swizzle.src3_selector_3.Value())};
            if (Taint4* const into = destination(instr.mad.dest.Value())) {
                for (u32 i = 0; i < 4; i++) {
                    if (!swizzle.DestComponentEnabled(i)) {
                        continue;
                    }
                    Taint product = s1.taints[pick1[i]] | s2.taints[pick2[i]];
                    for (const auto& [factor, component] :
                         {std::pair{&s1, pick1[i]}, std::pair{&s2, pick2[i]}}) {
                        if (is_zero(*factor, component)) {
                            product = Taint::Uniform(factor->index);
                            control |= product;
                        }
                    }
                    (*into)[i] = product | s3.taints[pick3[i]];
                }
            }
            return;
        }
        const SwizzlePattern swizzle = {swizzles[instr.common.operand_desc_id]};
        const bool inverted = (opcode.GetInfo().subtype & OpCode::Info::SrcInversed) != 0;
        const Source s1 = source(instr.common.GetSrc1(inverted),
                                 inverted ? 0 : instr.common.address_register_index);
        const Source s2 = source(instr.common.GetSrc2(inverted),
                                 inverted ? instr.common.address_register_index : 0);
        const std::array<u32, 4> pick1 = {static_cast<u32>(swizzle.src1_selector_0.Value()),
                                          static_cast<u32>(swizzle.src1_selector_1.Value()),
                                          static_cast<u32>(swizzle.src1_selector_2.Value()),
                                          static_cast<u32>(swizzle.src1_selector_3.Value())};
        const std::array<u32, 4> pick2 = {static_cast<u32>(swizzle.src2_selector_0.Value()),
                                          static_cast<u32>(swizzle.src2_selector_1.Value()),
                                          static_cast<u32>(swizzle.src2_selector_2.Value()),
                                          static_cast<u32>(swizzle.src2_selector_3.Value())};
        Taint4 a{};
        Taint4 b{};
        for (u32 i = 0; i < 4; i++) {
            a[i] = s1.taints[pick1[i]];
            b[i] = s2.taints[pick2[i]];
        }
        Taint4 result{};
        switch (opcode.EffectiveOpCode()) {
        case OpCode::Id::CMP:
            on.condition[0] = a[0] | b[0];
            on.condition[1] = a[1] | b[1];
            return;
        case OpCode::Id::MOVA:
            for (u32 i = 0; i < 2; i++) {
                if (swizzle.DestComponentEnabled(i)) {
                    on.address[i] = a[i];
                    address_unknown[i] = other_side;
                }
            }
            return;
        case OpCode::Id::MUL:
            for (u32 i = 0; i < 4; i++) {
                result[i] = a[i] | b[i];
                for (const auto& [factor, component] :
                     {std::pair{&s1, pick1[i]}, std::pair{&s2, pick2[i]}}) {
                    if (is_zero(*factor, component)) {
                        result[i] = Taint::Uniform(factor->index);
                        control |= result[i];
                    }
                }
            }
            break;
        case OpCode::Id::ADD:
        case OpCode::Id::MAX:
        case OpCode::Id::MIN:
        case OpCode::Id::SGE:
        case OpCode::Id::SGEI:
        case OpCode::Id::SLT:
        case OpCode::Id::SLTI:
            for (u32 i = 0; i < 4; i++) {
                result[i] = a[i] | b[i];
            }
            break;
        case OpCode::Id::MOV:
        case OpCode::Id::FLR:
            result = a;
            break;
        case OpCode::Id::RCP:
        case OpCode::Id::RSQ:
            reciprocal |= a[0];
            result.fill(a[0]);
            break;
        case OpCode::Id::EX2:
        case OpCode::Id::LG2:
            result.fill(a[0]);
            break;
        case OpCode::Id::DP3:
            result.fill(a[0] | a[1] | a[2] | b[0] | b[1] | b[2]);
            break;
        default:
            // DP4, DPH and whatever else: by all of both
            result.fill(a[0] | a[1] | a[2] | a[3] | b[0] | b[1] | b[2] | b[3]);
            break;
        }
        if (Taint4* const into = destination(instr.common.dest.Value())) {
            for (u32 i = 0; i < 4; i++) {
                if (swizzle.DestComponentEnabled(i)) {
                    (*into)[i] = result[i];
                }
            }
        }
    };
    const auto is_arithmetic = [](const Instruction& instr) {
        const auto type = instr.opcode.Value().GetInfo().type;
        return type == OpCode::Type::Arithmetic || type == OpCode::Type::MultiplyAdd;
    };

    // Where the two sides of a branch that depends on the vertex meet again, and what the
    // registers depend on if the side this run did not take is taken
    struct Meeting {
        u32 at;
        TaintState other;
    };
    std::vector<Meeting> meetings;

    for (const Record& record : trace.records) {
        if (!known) {
            break;
        }
        if ((record.mask & Record::CUR_INSTR) == 0 || record.instruction_offset >= code.size()) {
            continue;
        }
        const u32 pc = record.instruction_offset;
        std::erase_if(meetings, [&](const Meeting& meeting) {
            if (meeting.at != pc) {
                return false;
            }
            state.Join(meeting.other);
            return true;
        });
        Instruction instr{};
        instr.hex = code[pc];
        const OpCode opcode = instr.opcode.Value();
        if (is_arithmetic(instr)) {
            apply(instr, state);
            if (opcode.EffectiveOpCode() == OpCode::Id::MOVA &&
                (record.mask & Record::ADDR_REG_OUT) != 0) {
                address_value = {record.address_registers[0], record.address_registers[1]};
            }
            continue;
        }
        switch (opcode.EffectiveOpCode()) {
        case OpCode::Id::IFC:
        case OpCode::Id::JMPC:
        case OpCode::Id::CALLC:
        case OpCode::Id::BREAKC: {
            using Op = Instruction::FlowControlType::Op;
            const auto& flow = instr.flow_control;
            const Taint by = flow.op == Op::JustX   ? state.condition[0]
                             : flow.op == Op::JustY ? state.condition[1]
                                                    : state.condition[0] | state.condition[1];
            if (!by.OfVertex()) {
                control |= by;
                break;
            }
            influence |= by;
            if ((record.mask & Record::COND_CMP_IN) == 0) {
                known = false;
                break;
            }
            const bool x = flow.refx.Value() == record.cond_cmp[0];
            const bool y = flow.refy.Value() == record.cond_cmp[1];
            const bool taken = flow.op == Op::Or      ? x || y
                               : flow.op == Op::And   ? x && y
                               : flow.op == Op::JustX ? x
                                                      : y;
            const u32 dest = flow.dest_offset;
            const u32 count = flow.num_instructions;
            // The side that was not run, from and up to, and where the run goes on after
            u32 from = 0;
            u32 to = 0;
            u32 meet = 0;
            switch (opcode.EffectiveOpCode()) {
            case OpCode::Id::IFC:
                meet = dest + count;
                from = taken ? dest : pc + 1;
                to = taken ? dest + count : dest;
                break;
            case OpCode::Id::JMPC:
                known = dest > pc;
                meet = dest;
                from = pc + 1;
                to = taken ? dest : pc + 1;
                break;
            case OpCode::Id::CALLC:
                meet = pc + 1;
                from = dest;
                to = taken ? dest : dest + count;
                break;
            default:
                known = false;
                break;
            }
            if (!known || to > code.size()) {
                known = false;
                break;
            }
            Meeting meeting{meet, state};
            other_side = true;
            for (u32 at = from; at < to && known; at++) {
                Instruction there{};
                there.hex = code[at];
                if (there.opcode.Value().EffectiveOpCode() == OpCode::Id::NOP) {
                    continue;
                }
                if (!is_arithmetic(there)) {
                    known = false;
                    break;
                }
                apply(there, meeting.other);
            }
            other_side = false;
            // (Where the sides meet the registers may hold either side's addresses: one that
            // was set on the other side stays unknown for what follows, unless the vertex
            // picks it)
            meetings.push_back(std::move(meeting));
            break;
        }
        case OpCode::Id::LOOP:
            if ((record.mask & Record::LOOP_INT_IN) != 0) {
                loop = record.loop_int;
            }
            break;
        default:
            break;
        }
    }
    // (The program ended before two sides met)
    for (const Meeting& meeting : meetings) {
        state.Join(meeting.other);
    }

    // What matters of what the program puts out is what the fragments are made of. A program
    // made for many kinds of draw puts out texture coordinates for three textures and what
    // lighting needs, from uniforms the game sets only for a draw that has those textures or
    // is lit: the output map says what each component is, and the registers which of them a
    // fragment of this draw is given.
    const auto& config = regs.internal.texturing.main_config;
    const bool lit = !regs.internal.lighting.disable;
    const bool procedural = config.texture3_enable != 0;
    const std::array<bool, 3> coordinates = {
        config.texture0_enable != 0 || (procedural && config.texture3_coordinates == 0),
        config.texture1_enable != 0 ||
            (config.texture2_enable != 0 && config.texture2_use_coord1 != 0) ||
            (procedural && config.texture3_coordinates == 1),
        (config.texture2_enable != 0 && config.texture2_use_coord1 == 0) ||
            (procedural && config.texture3_coordinates == 2),
    };
    const auto fragments_use = [&](u32 meaning) {
        using Semantic = RasterizerRegs::VSOutputAttributes::Semantic;
        switch (meaning) {
        case Semantic::QUATERNION_X:
        case Semantic::QUATERNION_Y:
        case Semantic::QUATERNION_Z:
        case Semantic::QUATERNION_W:
        case Semantic::VIEW_X:
        case Semantic::VIEW_Y:
        case Semantic::VIEW_Z:
            return lit;
        case Semantic::TEXCOORD0_U:
        case Semantic::TEXCOORD0_V:
        case Semantic::TEXCOORD0_W:
            return coordinates[0];
        case Semantic::TEXCOORD1_U:
        case Semantic::TEXCOORD1_V:
            return coordinates[1];
        case Semantic::TEXCOORD2_U:
        case Semantic::TEXCOORD2_V:
            return coordinates[2];
        default:
            // Position and colour, and whatever is not known here; a component that is
            // mapped to nothing goes nowhere
            return meaning <= Semantic::TEXCOORD2_V;
        }
    };
    Taint matter = control | influence;
    const u32 outputs_on = regs.internal.vs.output_mask;
    const u32 mapped = std::min<u32>(regs.internal.rasterizer.vs_output_total, 7);
    u32 slot = 0;
    for (u32 reg = 0; reg < state.output.size(); reg++) {
        if ((outputs_on >> reg & 1) == 0) {
            continue;
        }
        // (The outputs that are switched on are taken in their order by the map's entries)
        if (slot < mapped) {
            const auto& map = regs.internal.rasterizer.vs_output_attributes[slot];
            const std::array<u32, 4> meanings = {map.map_x, map.map_y, map.map_z, map.map_w};
            for (u32 i = 0; i < 4; i++) {
                if (fragments_use(meanings[i])) {
                    matter |= state.output[reg][i];
                }
            }
        }
        slot++;
    }
    if (uniforms_that_matter.size() >= 512) {
        uniforms_that_matter.clear();
    }
    UniformsThatMatter& entry = uniforms_that_matter.emplace_back();
    entry.key = key;
    entry.control = {control.low, control.high & ~Taint::VERTEX};
    entry.control_values = values_of(entry.control);
    entry.known = known;
    entry.matter = {matter.low, matter.high & ~Taint::VERTEX};
    entry.zero_sign = {reciprocal.low, reciprocal.high & ~Taint::VERTEX};

    // What was found, for whoever asks why two draws did not compare alike
    static std::atomic<u32> told{};
    if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::ExplainScenePasses) != 0 &&
        told.fetch_add(1, std::memory_order_relaxed) < 40) {
        const auto list = [](Taint taint) {
            std::string text;
            for (u32 i = 0; i < 96; i++) {
                if (((i < 64 ? taint.low : taint.high) >> (i % 64) & 1) != 0) {
                    text += fmt::format("{}{}", text.empty() ? "" : " ", i);
                }
            }
            return (text.empty() ? std::string{"none"} : text) +
                   (taint.OfVertex() ? " and the vertex" : "");
        };
        std::string by_output;
        for (u32 reg = 0; reg < state.output.size(); reg++) {
            if ((outputs_on >> reg & 1) == 0) {
                continue;
            }
            Taint all;
            for (const Taint& taint : state.output[reg]) {
                all |= taint;
            }
            by_output += fmt::format("; o{} by {}", reg, list(all));
        }
        // The run itself, once for each program and set of bool uniforms: place, instruction
        // and operand pattern of each instruction it came by, to be read with a disassembler
        static std::array<u64, 8> runs_told{};
        static std::atomic<u32> runs{};
        const u64 run_key =
            Common::HashCombine(vs_setup.GetProgramCodeHash(), u64{main}, u64{bools});
        if (std::ranges::find(runs_told, run_key) == runs_told.end() && runs.load() < 8) {
            runs_told[runs.fetch_add(1) % runs_told.size()] = run_key;
            std::string words;
            for (const Record& record : trace.records) {
                if ((record.mask & Record::CUR_INSTR) == 0 ||
                    record.instruction_offset >= code.size()) {
                    continue;
                }
                const u32 word = code[record.instruction_offset];
                const u32 pattern = (word >> 26) >= 0x30 ? word & 0x1F : word & 0x7F;
                words += fmt::format(" {:03X}:{:08X}:{:08X}", record.instruction_offset, word,
                                     swizzles[pattern]);
            }
            LOG_INFO(HW_GPU, "Vertex shader {:016X} from {:X}, bool uniforms {:04X}, its run:{}",
                     vs_setup.GetProgramCodeHash(), main, bools, words);
        }
        const UniformSet& read = FloatUniformsRead();
        LOG_INFO(HW_GPU,
                 "Vertex shader {:016X} from {:X}, bool uniforms {:04X}, traced over {} "
                 "instructions: {}; its way and what it multiplies by nothing go by uniforms "
                 "{}{}; in all {} of the {} uniforms it reads matter",
                 vs_setup.GetProgramCodeHash(), main, bools, trace.records.size(),
                 known ? "followed to its end"
                       : "not followed, it depends on the vertex where "
                         "that cannot be gone into: nothing told",
                 list(control), by_output,
                 std::popcount(entry.matter[0]) + std::popcount(entry.matter[1]),
                 std::popcount(read[0]) + std::popcount(read[1]));
    }
    return known ? &entry : nullptr;
}

const PicaCore::UniformSet& PicaCore::FloatUniformsCompared() {
    if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::AllUniformsRead) == 0) {
        if (const UniformsThatMatter* const found = FloatUniformsThatMatter()) {
            return found->matter;
        }
    }
    return FloatUniformsRead();
}

const PicaCore::UniformSet& PicaCore::FloatUniformsZeroSign() {
    static constexpr UniformSet ALL{~u64{0}, ~u64{0}};
    if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::AllUniformsRead) == 0) {
        if (const UniformsThatMatter* const found = FloatUniformsThatMatter()) {
            return found->zero_sign;
        }
    }
    return ALL;
}

namespace {

/// The clip axis along which a perspective projection among the float uniforms squeezes the
/// picture for a wider screen, or -1 with none there. Looked for: four uniforms in a row, as
/// the rows of a matrix, of which the last is (0, 0, 1 or -1, 0) and the first two hold one
/// scale term each, on x and y, in the console's 5:3. The smaller term is the one for the
/// picture's horizontal; its row is the clip axis. (The top screen is turned by a quarter, so
/// that is mostly y; a target a game draws upright has it on x.) Projections for shadows and
/// the like are not 5:3 and are left alone by that.
s8 FindProjectionAxis(const Uniforms& uniforms, const std::array<f32, 4>& ratios) {
    for (std::size_t base = 0; base + 3 < uniforms.f.size(); base++) {
        const auto& last = uniforms.f[base + 3];
        if (last.x.ToFloat32() != 0.f || last.y.ToFloat32() != 0.f ||
            std::abs(std::abs(last.z.ToFloat32()) - 1.f) > 1e-3f || last.w.ToFloat32() != 0.f) {
            continue;
        }
        const f32 first_x = uniforms.f[base].x.ToFloat32();
        const f32 first_y = uniforms.f[base].y.ToFloat32();
        const f32 second_x = uniforms.f[base + 1].x.ToFloat32();
        const f32 second_y = uniforms.f[base + 1].y.ToFloat32();
        const bool turned = first_x == 0.f && first_y != 0.f && second_x != 0.f && second_y == 0.f;
        const bool upright = first_x != 0.f && first_y == 0.f && second_x == 0.f && second_y != 0.f;
        if (!turned && !upright) {
            continue;
        }
        const f32 one = std::abs(turned ? first_y : first_x);
        const f32 other = std::abs(turned ? second_x : second_y);
        const f32 ratio = std::max(one, other) / std::min(one, other);
        const bool known = std::ranges::any_of(ratios, [ratio](f32 wanted) {
            return wanted > 0.f && std::abs(ratio - wanted) <= 0.006f * wanted;
        });
        if (!known) {
            continue;
        }
        return one < other ? s8{0} : s8{1};
    }
    return -1;
}

/// What the host's vertex stage does to a vertex's clip coordinates (ApplyWide in the
/// generated shaders), for the tests made here of where a draw's vertices end up
void WidenShaded(const PicaCore::WideScales& scales, OutputVertex& vertex) {
    if (scales == PicaCore::WideScales{}) {
        return;
    }
    if (std::abs(vertex.pos.w.ToFloat32() - 1.f) < 0.0001f) {
        vertex.pos.y = f24::FromFloat32(vertex.pos.y.ToFloat32() * scales.flat);
    } else {
        vertex.pos.x = f24::FromFloat32(vertex.pos.x.ToFloat32() * scales.x);
        vertex.pos.y = f24::FromFloat32(vertex.pos.y.ToFloat32() * scales.y);
    }
}

} // Anonymous namespace

PicaCore::WideScales PicaCore::DecideWideScales(bool is_indexed) {
    WideScales scales;
    const f32 factor = wide_factor.load(std::memory_order_relaxed);
    draws_wide = factor != 1.f && wide_enabled;
    if (!draws_wide) {
        return scales;
    }

    if (wide_epoch_seen != vs_setup.uniforms_epoch) {
        wide_epoch_seen = vs_setup.uniforms_epoch;
        wide_axis = FindProjectionAxis(vs_setup.uniforms, wide_ratios);
    }
    if (wide_axis == 0) {
        scales.x = factor;
    } else if (wide_axis == 1) {
        scales.y = factor;
    }

    const auto& rasterizer = regs.internal.rasterizer;
    const bool whole_top_screen = f24::FromRaw(rasterizer.viewport_size_x).ToFloat32() == 120.f &&
                                  f24::FromRaw(rasterizer.viewport_size_y).ToFloat32() == 200.f &&
                                  rasterizer.viewport_corner.x == 0 &&
                                  rasterizer.viewport_corner.y == 0;
    if (wide_keeps_flat_shape && whole_top_screen) {
        scales.flat = factor;
        // A small draw may be the kind that fills the screen from side to side: a fade, a
        // flash, a film. (Also asked of one while the game's projection is among the
        // uniforms: games leave it there while they draw flat.)
        if (regs.internal.pipeline.num_vertices <= 6) {
            DrawBox box;
            const WideScales before = std::exchange(wide_scales, WideScales{});
            const bool boxed = SmallDrawBox(is_indexed, box);
            wide_scales = before;
            if (boxed && box.bottom <= -0.98f && box.top >= 0.98f) {
                scales.flat = 1.f;
                wide_counts[3]++;
            }
        }
    }
#ifdef CITRA_ENABLE_TRACE_MARKERS
    wide_counts[1] += wide_axis >= 0;
    wide_counts[2] += scales.flat != 1.f;
    if (++wide_counts[0] == 16384) {
        LOG_INFO(HW_GPU,
                 "Top screen of another shape (factor {:.4f}): of the last {} draws {} carried "
                 "the game's projection and have their vertices with perspective squeezed, {} "
                 "onto the whole top screen have those without squeezed about the middle, {} "
                 "small ones are left to fill the screen from side to side",
                 factor, wide_counts[0], wide_counts[1], wide_counts[2], wide_counts[3]);
        wide_counts = {};
    }
#endif
    return scales;
}

void PicaCore::LoadWidePatch(u64 program_id) {
    wide_enabled = true;
    wide_keeps_flat_shape = true;
    wide_epoch_seen = 0;
    wide_ratios = {5.f / 3.f, 0.f, 0.f, 0.f};
    // Games whose projection is not in the console's plain 5:3
    struct KnownGame {
        u64 program_id;
        f32 ratio;
    };
    static constexpr std::array<KnownGame, 1> KNOWN_GAMES{{
        // Luigi's Mansion: Dark Moon (USA) draws its scene in two halves, one above the
        // other, each at twice the height
        {0x0004000000055F00ull, 10.f / 3.f},
    }};
    for (const KnownGame& game : KNOWN_GAMES) {
        if (game.program_id == program_id) {
            wide_ratios[1] = game.ratio;
        }
    }
    const std::string path =
        fmt::format("{}mods/{:016X}/widescreen.txt",
                    FileUtil::GetUserPath(FileUtil::UserPath::LoadDir), program_id);
    std::string text;
    if (FileUtil::ReadFileToString(true, path, text) == 0) {
        return;
    }
    // Lines of "key = value"; what follows a '#' is a comment
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t end = text.find('\n', at);
        if (end == std::string::npos) {
            end = text.size();
        }
        std::string line = text.substr(at, end - at);
        at = end + 1;
        if (const std::size_t comment = line.find('#'); comment != std::string::npos) {
            line.erase(comment);
        }
        std::string compact;
        for (const char c : line) {
            if (c != ' ' && c != '\t' && c != '\r') {
                compact += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
        }
        if (compact == "widen=off") {
            wide_enabled = false;
        } else if (compact == "hud=stretch") {
            wide_keeps_flat_shape = false;
        } else if (compact.starts_with("ratio=")) {
            const f32 ratio = std::strtof(compact.c_str() + 6, nullptr);
            // After the built-in ones, as many as there is room for
            for (f32& entry : wide_ratios) {
                if (ratio >= 1.f && entry == 0.f) {
                    entry = ratio;
                    break;
                }
            }
        }
    }
    LOG_INFO(HW_GPU, "{}: the scene is {}, what is drawn flat {}", path,
             wide_enabled ? "drawn for the shape the top screen is shown at" : "left as it is",
             wide_enabled && wide_keeps_flat_shape ? "keeps its proportions" : "is stretched");
}

void PicaCore::FlushSoftwareBatch() {
    if (!sw_batch_open) {
        return;
    }
    sw_batch_open = false;
    sw_batch_draws = 0;
    rasterizer->DrawTriangles();
    rasterizer->NoteDrawOverwrites(false, false);
}

static bool any_byte_match(u32 a, u32 b) {
    return ((a & 0xFF) == (b & 0xFF)) || (((a >> 8) & 0xFF) == ((b >> 8) & 0xFF)) ||
           (((a >> 16) & 0xFF) == ((b >> 16) & 0xFF)) || (((a >> 24) & 0xFF) == ((b >> 24) & 0xFF));
}

// Handle registers which our backend support batch writes.
void PicaCore::HandleSpecialRegBatch(u32 id, const u32* values, u32 count) {
    switch (id) {
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[1]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[2]): {
        for (u32 i = 0; i < count; i++) {
            SubmitImmediate(values[i]);
        }
        break;
    }
    case PICA_REG_INDEX(gs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[7]): {
        gs_setup.WriteUniformFloatRegRange(regs.internal.gs, values, count);
        break;
    }
    case PICA_REG_INDEX(gs.program.set_word[0]):
    case PICA_REG_INDEX(gs.program.set_word[1]):
    case PICA_REG_INDEX(gs.program.set_word[2]):
    case PICA_REG_INDEX(gs.program.set_word[3]):
    case PICA_REG_INDEX(gs.program.set_word[4]):
    case PICA_REG_INDEX(gs.program.set_word[5]):
    case PICA_REG_INDEX(gs.program.set_word[6]):
    case PICA_REG_INDEX(gs.program.set_word[7]): {
        u32& offset = regs.internal.gs.program.offset;
        if (offset + count > 4096) {
            LOG_ERROR(HW_GPU, "Invalid GS program offset {} count {}", offset, count);
        } else {
            gs_setup.UpdateProgramCodeRange(offset, values, count);
            offset += count;
        }
        break;
    }
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.gs.swizzle_patterns.offset;
        if (offset + count > gs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid GS swizzle pattern offset {} count {}", offset, count);
        } else {
            gs_setup.UpdateSwizzleDataRange(offset, values, count);
            offset += count;
        }
        break;
    }
    case PICA_REG_INDEX(vs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[7]): {
        const auto range = vs_setup.WriteUniformFloatRegRange(regs.internal.vs, values, count);
        if (range && !regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            for (u32 i = 0; i < range->count; ++i) {
                const u32 idx = range->first_index + i;
                gs_setup.uniforms.f[idx] = vs_setup.uniforms.f[idx];
            }
        }
        break;
    }
    case PICA_REG_INDEX(vs.program.set_word[0]):
    case PICA_REG_INDEX(vs.program.set_word[1]):
    case PICA_REG_INDEX(vs.program.set_word[2]):
    case PICA_REG_INDEX(vs.program.set_word[3]):
    case PICA_REG_INDEX(vs.program.set_word[4]):
    case PICA_REG_INDEX(vs.program.set_word[5]):
    case PICA_REG_INDEX(vs.program.set_word[6]):
    case PICA_REG_INDEX(vs.program.set_word[7]): {
        u32& offset = regs.internal.vs.program.offset;
        if (offset + count > 512) {
            LOG_ERROR(HW_GPU, "Invalid VS program offset {} count {}", offset, count);
        } else {
            vs_setup.UpdateProgramCodeRange(offset, values, count);
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
                regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
                gs_setup.UpdateProgramCodeRange(offset, values, count);
            }
            offset += count;
        }
        break;
    }
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.vs.swizzle_patterns.offset;
        if (offset + count > vs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid VS swizzle pattern offset {} count {}", offset, count);
        } else {
            vs_setup.UpdateSwizzleDataRange(offset, values, count);
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
                regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
                gs_setup.UpdateSwizzleDataRange(offset, values, count);
            }
            offset += count;
        }
        break;
    }
    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        auto& lut_config = regs.internal.lighting.lut_config;

        for (u32 i = 0; i < count; i++) {
            const u32 prev =
                std::exchange(lighting
                                  .luts[lut_config.type][(lut_config.index + i) %
                                                         lighting.luts[lut_config.type].size()]
                                  .raw,
                              values[i]);
            lighting.lut_dirty |= (prev != values[i]) << lut_config.type;
        }
        lut_config.index.Assign(lut_config.index + count);
        break;
    }
    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]): {
        for (u32 i = 0; i < count; i++) {
            const u32 prev = std::exchange(
                fog.lut[(regs.internal.texturing.fog_lut_offset + i) % 128].raw, values[i]);
            fog.lut_dirty |= prev != values[i];
        }
        regs.internal.texturing.fog_lut_offset.Assign(regs.internal.texturing.fog_lut_offset +
                                                      count);
        break;
    }
    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]): {
        auto& index = regs.internal.texturing.proctex_lut_config.index;
        const auto lut_table = regs.internal.texturing.proctex_lut_config.ref_table.Value();

        for (u32 i = 0; i < count; i++) {

            const auto sync_lut = [&](auto& proctex_table) {
                const u32 prev =
                    std::exchange(proctex_table[(index + i) % proctex_table.size()].raw, values[i]);
                proctex.table_dirty |= (prev != values[i]) << u32(lut_table);
            };

            switch (lut_table) {
            case TexturingRegs::ProcTexLutTable::Noise:
                sync_lut(proctex.noise_table);
                break;
            case TexturingRegs::ProcTexLutTable::ColorMap:
                sync_lut(proctex.color_map_table);
                break;
            case TexturingRegs::ProcTexLutTable::AlphaMap:
                sync_lut(proctex.alpha_map_table);
                break;
            case TexturingRegs::ProcTexLutTable::Color:
                sync_lut(proctex.color_table);
                break;
            case TexturingRegs::ProcTexLutTable::ColorDiff:
                sync_lut(proctex.color_diff_table);
                break;
            }
        }
        index.Assign(index + count);
        break;
    }
    }
}

// Handle special registers. This function should also include the
// registers that support batch processing can be submitted
// individually.
void PicaCore::HandleSpecialReg(u32 id, u32 value, bool& stop_requested) {
    switch (id) {
    // Trigger IRQ
    case PICA_REG_INDEX(irq_request):
        // TODO(PabloMK7): This logic is not fully accurate, but close enough:
        // https://problemkaputt.de/gbatek-3ds-gpu-internal-registers-finalize-interrupt-registers.htm
        if (any_byte_match(regs.internal.reg_array[id], regs.internal.irq_compare)) [[likely]] {
            signal_interrupt(Service::GSP::InterruptId::P3D,
                             delay_generator.CalculateAndResetDelay());
            if (regs.internal.irq_autostop) [[likely]] {
                stop_requested = true;
            }
        }
        break;

    case PICA_REG_INDEX(pipeline.triangle_topology):
        primitive_assembler.Reconfigure(regs.internal.pipeline.triangle_topology);
        break;

    case PICA_REG_INDEX(pipeline.restart_primitive):
        primitive_assembler.Reset();
        break;

    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index):
        immediate.Reset();
        break;

    // Load default vertex input attributes
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[1]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[2]):
        SubmitImmediate(value);
        break;

    case PICA_REG_INDEX(pipeline.gpu_mode):
        // This register likely just enables vertex processing and doesn't need any special handling
        break;

    case PICA_REG_INDEX(pipeline.command_buffer.trigger[0]):
    case PICA_REG_INDEX(pipeline.command_buffer.trigger[1]): {
        const u32 index = static_cast<u32>(id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
        const PAddr addr = regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
        u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
        // The list as it was when the one that jumps to it was handed over, if it was taken
        // along then: the game may have been told since that all of this is done, and be
        // writing its next lists to the same place
        const u8* head = size != 0 ? TakenCalledList(addr, size) : nullptr;
        if (head == nullptr && size != 0) {
            if (memory.PhysicalBytesFrom(addr) >= size) {
                head = memory.GpuRead(addr, size);
                memory.NoteGpuReference(addr, size, Memory::GpuReference::CalledList);
            }
            if (head == nullptr) {
                static std::atomic<u32> told{};
                if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
                    LOG_ERROR(HW_GPU,
                              "A command list jumps to {:08X} ({} bytes), which is not in the "
                              "emulated memory: the list ends there",
                              addr, size);
                }
            }
        }
        if (head == nullptr) {
            // Nothing to carry out. (What the command that jumped has left of parameters is
            // read from the beginning of the list jumped to: nothing but zeroes here.)
            static constexpr std::array<u32, LIST_SLACK / sizeof(u32)> NOTHING{};
            head = reinterpret_cast<const u8*>(NOTHING.data());
            size = 0;
        }
        if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::MeasureCommandLists) != 0)
            [[unlikely]] {
            MeasureCalledList(head, size, addr);
        }
        cmd_list.Reset(addr, head, size);
        break;
    }

    // It seems like these trigger vertex rendering
    case PICA_REG_INDEX(pipeline.trigger_draw):
    case PICA_REG_INDEX(pipeline.trigger_draw_indexed): {
        const bool is_indexed = (id == PICA_REG_INDEX(pipeline.trigger_draw_indexed));
        DrawArrays(is_indexed);
        break;
    }

    case PICA_REG_INDEX(gs.bool_uniforms):
        gs_setup.WriteUniformBoolReg(regs.internal.gs.bool_uniforms.Value());
        break;

    case PICA_REG_INDEX(gs.int_uniforms[0]):
    case PICA_REG_INDEX(gs.int_uniforms[1]):
    case PICA_REG_INDEX(gs.int_uniforms[2]):
    case PICA_REG_INDEX(gs.int_uniforms[3]): {
        const u32 index = (id - PICA_REG_INDEX(gs.int_uniforms[0]));
        gs_setup.WriteUniformIntReg(index, regs.internal.gs.GetIntUniform(index));
        break;
    }

    case PICA_REG_INDEX(gs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[7]): {
        gs_setup.WriteUniformFloatReg(regs.internal.gs, value);
        break;
    }

    case PICA_REG_INDEX(gs.program.set_word[0]):
    case PICA_REG_INDEX(gs.program.set_word[1]):
    case PICA_REG_INDEX(gs.program.set_word[2]):
    case PICA_REG_INDEX(gs.program.set_word[3]):
    case PICA_REG_INDEX(gs.program.set_word[4]):
    case PICA_REG_INDEX(gs.program.set_word[5]):
    case PICA_REG_INDEX(gs.program.set_word[6]):
    case PICA_REG_INDEX(gs.program.set_word[7]): {
        u32& offset = regs.internal.gs.program.offset;
        if (offset >= 4096) {
            LOG_ERROR(HW_GPU, "Invalid GS program offset {}", offset);
        } else {
            gs_setup.UpdateProgramCode(offset, value);
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.gs.swizzle_patterns.offset;
        if (offset >= gs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid GS swizzle pattern offset {}", offset);
        } else {
            gs_setup.UpdateSwizzleData(offset, value);
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(vs.output_mask):
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            regs.internal.gs.output_mask.Assign(value);
        }
        break;

    case PICA_REG_INDEX(vs.bool_uniforms):
        vs_setup.WriteUniformBoolReg(regs.internal.vs.bool_uniforms.Value());
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            gs_setup.WriteUniformBoolReg(regs.internal.vs.bool_uniforms.Value());
        }
        break;

    case PICA_REG_INDEX(vs.int_uniforms[0]):
    case PICA_REG_INDEX(vs.int_uniforms[1]):
    case PICA_REG_INDEX(vs.int_uniforms[2]):
    case PICA_REG_INDEX(vs.int_uniforms[3]): {
        const u32 index = (id - PICA_REG_INDEX(vs.int_uniforms[0]));
        vs_setup.WriteUniformIntReg(index, regs.internal.vs.GetIntUniform(index));
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            gs_setup.WriteUniformIntReg(index, regs.internal.vs.GetIntUniform(index));
        }
        break;
    }

    case PICA_REG_INDEX(vs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[7]): {
        const auto index = vs_setup.WriteUniformFloatReg(regs.internal.vs, value);
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No && index) {
            gs_setup.uniforms.f[index.value()] = vs_setup.uniforms.f[index.value()];
        }
        break;
    }

    case PICA_REG_INDEX(vs.program.set_word[0]):
    case PICA_REG_INDEX(vs.program.set_word[1]):
    case PICA_REG_INDEX(vs.program.set_word[2]):
    case PICA_REG_INDEX(vs.program.set_word[3]):
    case PICA_REG_INDEX(vs.program.set_word[4]):
    case PICA_REG_INDEX(vs.program.set_word[5]):
    case PICA_REG_INDEX(vs.program.set_word[6]):
    case PICA_REG_INDEX(vs.program.set_word[7]): {
        u32& offset = regs.internal.vs.program.offset;
        if (offset >= 512) {
            LOG_ERROR(HW_GPU, "Invalid VS program offset {}", offset);
        } else {
            vs_setup.UpdateProgramCode(offset, value);
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
                regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
                gs_setup.UpdateProgramCode(offset, value);
            }
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.vs.swizzle_patterns.offset;
        if (offset >= vs_setup.GetSwizzleData().size()) {
            LOG_ERROR(HW_GPU, "Invalid VS swizzle pattern offset {}", offset);
        } else {
            vs_setup.UpdateSwizzleData(offset, value);
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
                regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
                gs_setup.UpdateSwizzleData(offset, value);
            }
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        auto& lut_config = regs.internal.lighting.lut_config;

        const u32 prev = std::exchange(lighting.luts[lut_config.type][lut_config.index].raw, value);
        lighting.lut_dirty |= (prev != value) << lut_config.type;
        lut_config.index.Assign(lut_config.index + 1);
        break;
    }

    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]): {
        const u32 prev =
            std::exchange(fog.lut[regs.internal.texturing.fog_lut_offset % 128].raw, value);
        fog.lut_dirty |= prev != value;
        regs.internal.texturing.fog_lut_offset.Assign(regs.internal.texturing.fog_lut_offset + 1);
        break;
    }

    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]): {
        auto& index = regs.internal.texturing.proctex_lut_config.index;
        const auto lut_table = regs.internal.texturing.proctex_lut_config.ref_table.Value();

        const auto sync_lut = [&](auto& proctex_table) {
            const u32 prev = std::exchange(proctex_table[index % proctex_table.size()].raw, value);
            proctex.table_dirty |= (prev != value) << u32(lut_table);
        };

        switch (lut_table) {
        case TexturingRegs::ProcTexLutTable::Noise:
            sync_lut(proctex.noise_table);
            break;
        case TexturingRegs::ProcTexLutTable::ColorMap:
            sync_lut(proctex.color_map_table);
            break;
        case TexturingRegs::ProcTexLutTable::AlphaMap:
            sync_lut(proctex.alpha_map_table);
            break;
        case TexturingRegs::ProcTexLutTable::Color:
            sync_lut(proctex.color_table);
            break;
        case TexturingRegs::ProcTexLutTable::ColorDiff:
            sync_lut(proctex.color_diff_table);
            break;
        }
        index.Assign(index + 1);
        break;
    }
    default:
        break;
    }
}

// Handle batch register writes
void PicaCore::WriteInternalRegBatch(u32 id, const u32* values, u32 count, u32 mask,
                                     bool& stop_requested) {
    if (id >= RegsInternal::NUM_REGS) [[unlikely]] {
        // Writes to OOB registers are no-op.
        LOG_DEBUG(HW_GPU,
                  "Commandlist tried to write to invalid register 0x{:03X} repeated 0x{:04X} times"
                  "(mask: {:X})",
                  id, count, mask);
        return;
    }

    if (sw_batch_open && id < FIRST_VERTEX_REG) [[unlikely]] {
        FlushSoftwareBatch();
    }
    delay_generator.AddCommands(count);
    const u32 write_mask = ExpandBitsToBytes[mask];
    // Only write the last value to the register array.
    // Batch handlers should take this in mind.
    regs.internal.reg_array[id] =
        (regs.internal.reg_array[id] & ~write_mask) | (values[count - 1] & write_mask);
    dirty_regs.Set(id);

    if (reg_impl_flags_lut[id].SupportsBatch()) {
        // If the register supports batch then call the handler.
        HandleSpecialRegBatch(id, values, count);
    } else if (reg_impl_flags_lut[id].NeedsSpecialHandling()) [[unlikely]] {
        // Unlikely as all special regs that make sense to use batch mode already
        // support batch handling.
        for (u32 i = 0; i < count && !stop_requested; ++i) {
            HandleSpecialReg(id, values[i], stop_requested);
        }
    }
}

// Handle sequential register writes.
void PicaCore::WriteInternalRegSequential(u32 id, const u32* __restrict values, u32 count, u32 mask,
                                          bool& stop_requested) {
    if (id + count > RegsInternal::NUM_REGS) [[unlikely]] {
        // Writes to OOB registers are no-op.
        LOG_DEBUG(
            HW_GPU,
            "Commandlist tried to write to invalid register range 0x{:03X}-0x{:03X} (mask: {:X})",
            id, id + count, mask);
        if (id >= RegsInternal::NUM_REGS) {
            return;
        } else {
            count = RegsInternal::NUM_REGS - id;
        }
    }
    const u32 write_mask = ExpandBitsToBytes[mask];
    u32* __restrict dst = &regs.internal.reg_array[id];
    u32 offset = 0;

    // This code is structured so that it uses the LUT to get the distance from the
    // register ID to the next special register. Then copies the range of normal registers
    // until it reaches the special register which is handled individually, and so on.
    while (offset < count) {
        if (stop_requested) [[unlikely]] {
            break;
        }
        const u32 reg = id + offset;
        // Distance to the next special register, capped by how many writes
        // are actually left in this write. If this register is special this is 0.
        const u32 batch_count =
            std::min<u32>(reg_impl_flags_lut[reg].RegsUntilSpecial(), count - offset);
        if (batch_count > 0) {
            delay_generator.AddCommands(batch_count);

            // Allows the compiler to auto-vectorize thanks to the __restrict keywords.
            // Verified in MSVC that vectorization is happening.
            u32* __restrict batch_dst = dst + offset;
            const u32* __restrict batch_src = values + offset;
            // Only if any of them changes (see the single writes above)
            u32 difference = 0;
            for (u32 i = 0; i < batch_count; ++i) {
                difference |= (batch_dst[i] ^ batch_src[i]) & write_mask;
            }
            if (difference != 0) {
                if (sw_batch_open && reg < FIRST_VERTEX_REG) [[unlikely]] {
                    FlushSoftwareBatch();
                }
                for (u32 i = 0; i < batch_count; ++i) {
                    batch_dst[i] = (batch_dst[i] & ~write_mask) | (batch_src[i] & write_mask);
                }
                dirty_regs.SetRange(reg, batch_count);
            }
            offset += batch_count;
        }
        // Whatever register stopped the batch (if we didn't reach the end)
        // needs individual handling, then resume batching after it.
        if (offset < count) {
            WriteInternalReg(id + offset, values[offset], mask, stop_requested);
            ++offset;
        }
    }
}

// Handle individual command write.
void PicaCore::WriteInternalReg(u32 id, u32 value, u32 mask, bool& stop_requested) {
    if (id >= RegsInternal::NUM_REGS) [[unlikely]] {
        // Writes to OOB registers are no-op.
        LOG_DEBUG(
            HW_GPU,
            "Commandlist tried to write to invalid register 0x{:03X} (value: {:08X}, mask: {:X})",
            id, value, mask);
        return;
    }

    delay_generator.AddCommands(1);

    // TODO: Figure out how register masking acts on e.g. vs.uniform_setup.set_value
    const u32 old_value = regs.internal.reg_array[id];
    const u32 write_mask = ExpandBitsToBytes[mask];
    const u32 new_value = (old_value & ~write_mask) | (value & write_mask);
    const bool special = reg_impl_flags_lut[id].NeedsSpecialHandling();
    if (!special && new_value == old_value && !debug_context) [[likely]] {
        // A plain register set to what it is: nothing changes
        return;
    }
    if (sw_batch_open && id < FIRST_VERTEX_REG) [[unlikely]] {
        FlushSoftwareBatch();
    }
    regs.internal.reg_array[id] = new_value;

    if (debug_context) [[unlikely]] {
        // Track register write.
        DebugUtils::OnPicaRegWrite(id, mask, regs.internal.reg_array[id]);
        // Track events.
        debug_context->OnEvent(DebugContext::Event::PicaCommandLoaded, &id);
    }

    if (special) {
        HandleSpecialReg(id, value, stop_requested);
    }

    dirty_regs.Set(id);

    if (debug_context) [[unlikely]] {
        debug_context->OnEvent(DebugContext::Event::PicaCommandProcessed, &id);
    }
}

void PicaCore::SubmitImmediate(u32 value) {
    // Push to word to the queue. This returns true when a full attribute is formed.
    if (!immediate.queue.Push(value)) {
        return;
    }

    constexpr std::size_t IMMEDIATE_MODE_INDEX = 0xF;

    auto& setup = regs.internal.pipeline.vs_default_attributes_setup;
    if (setup.index > IMMEDIATE_MODE_INDEX) {
        LOG_ERROR(HW_GPU, "Invalid VS default attribute index {}", setup.index);
        return;
    }

    // Retrieve the attribute and place it in the default attribute buffer.
    const auto attribute = immediate.queue.Get();
    if (setup.index < IMMEDIATE_MODE_INDEX) {
        input_default_attributes[setup.index] = attribute;
        setup.index++;
        return;
    }

    // When index is 0xF the attribute is used for immediate mode drawing.
    immediate.input_vertex[immediate.current_attribute] = attribute;
    if (immediate.current_attribute < regs.internal.pipeline.max_input_attrib_index) {
        immediate.current_attribute++;
        return;
    }

    // We formed a vertex, flush.
    DrawImmediate();
}

void PicaCore::DrawImmediate() {
    CITRA_TRACE_COUNT(SwVertices);

    // Vertices given one at a time are left as they are for a top screen of another shape
    if (wide_scales != WideScales{}) {
        FlushSoftwareBatch();
        wide_scales = WideScales{};
    }

    // Compile the vertex shader.
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    // Track vertex in the debug recorder.
    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                               std::addressof(immediate.input_vertex));
    }

    ShaderUnit shader_unit;
    AttributeBuffer output{};

    // Invoke the vertex shader for the vertex.
    shader_unit.LoadInput(regs.internal.vs, immediate.input_vertex);
    shader_engine->Run(vs_setup, shader_unit);
    shader_unit.WriteOutput(regs.internal.vs, output);

    // Reconfigure geometry pipeline if needed.
    if (immediate.reset_geometry_pipeline) {
        geometry_pipeline.Reconfigure();
        immediate.reset_geometry_pipeline = false;
    }

    // Send to geometry pipeline.
    ASSERT(!geometry_pipeline.NeedIndexInput());
    geometry_pipeline.Setup(shader_engine.get());
    geometry_pipeline.SubmitVertex(output);

    // Flush the immediate triangle. Its vertices are with the rasterizer already, so it is
    // carried out in any case.
    TellRulesOfImmediateDraw();
    sw_batch_open = false;
    sw_batch_draws = 0;
    rasterizer->DrawTriangles();
    immediate.current_attribute = 0;

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr);
    }
}

namespace {

/// What a draw overwrites if its triangles cover the viewport: every pixel of the colour
/// buffer in the area it is confined to, every pixel of depth and stencil there
struct Overwrites {
    bool color{};
    bool depth{};
    /// Every pixel's depth there, whatever becomes of its stencil
    bool depth_values{};
};

Overwrites MayOverwriteArea(const RegsInternal& regs) {
    using Func = FramebufferRegs::CompareFunc;
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& merger = regs.framebuffer.output_merger;
    const auto& rasterizer = regs.rasterizer;

    // Few triangles, and no fragment of them is dropped. Whether they cover the viewport and
    // face the way that is drawn is looked at with their vertices, in CoversViewport.
    if (regs.pipeline.num_vertices > PicaCore::MAX_COVER_VERTICES_LARGE ||
        merger.alpha_test.enable || rasterizer.clip_enable ||
        merger.fragment_operation_mode != FramebufferRegs::FragmentOperationMode::Default) {
        return {};
    }
    if (merger.depth_test_enable && merger.depth_test_func != Func::Always) {
        return {};
    }
    const bool has_stencil = regs.framebuffer.HasStencil();
    const bool tests_stencil = has_stencil && merger.stencil_test.enable;
    if (tests_stencil && merger.stencil_test.func != Func::Always) {
        return {};
    }
    // The scissor leaves all of the viewport
    const auto& scissor = rasterizer.scissor_test;
    if (scissor.mode == RasterizerRegs::ScissorMode::Exclude) {
        return {};
    }
    if (scissor.mode == RasterizerRegs::ScissorMode::Include) {
        const auto viewport = rasterizer.GetViewportRect();
        if (static_cast<s32>(scissor.x1) > viewport.left ||
            static_cast<s32>(scissor.y1) > viewport.bottom ||
            static_cast<s32>(scissor.x2) + 1 < viewport.right ||
            static_cast<s32>(scissor.y2) + 1 < viewport.top) {
            return {};
        }
    }

    Overwrites result;
    // All of each pixel is written, and nothing of what it held goes into the result
    if (framebuffer.allow_color_write != 0 && merger.red_enable && merger.green_enable &&
        merger.blue_enable && merger.alpha_enable) {
        if (merger.alphablend_enable) {
            const auto& blending = merger.alpha_blending;
            result.color = blending.factor_source_rgb == FramebufferRegs::BlendFactor::One &&
                           blending.factor_source_a == FramebufferRegs::BlendFactor::One &&
                           blending.factor_dest_rgb == FramebufferRegs::BlendFactor::Zero &&
                           blending.factor_dest_a == FramebufferRegs::BlendFactor::Zero &&
                           blending.blend_equation_rgb == FramebufferRegs::BlendEquation::Add &&
                           blending.blend_equation_a == FramebufferRegs::BlendEquation::Add;
        } else {
            result.color = merger.logic_op == FramebufferRegs::LogicOp::Copy;
        }
    }
    // Depth and stencil are one buffer, so both have to be all new
    const bool may_write = framebuffer.allow_depth_stencil_write != 0;
    bool new_stencil = true;
    if (has_stencil) {
        using Action = FramebufferRegs::StencilAction;
        const Action action = merger.stencil_test.action_depth_pass;
        new_stencil = tests_stencil && may_write && merger.stencil_test.write_mask == 0xFF &&
                      (action == Action::Replace || action == Action::Zero);
    }
    result.depth = may_write && merger.depth_write_enable && new_stencil;
    result.depth_values = may_write && merger.depth_write_enable;
    return result;
}

} // Anonymous namespace

/// The draw about to be made, as the rules that leave work out get to ask about it. What
/// costs something is worked out when first asked for and kept.
class PicaCore::Facts final : public VideoCore::DrawFacts {
public:
    Facts(PicaCore& pica_, bool is_indexed_, bool immediate_)
        : pica{pica_}, is_indexed{is_indexed_}, immediate{immediate_} {}

    const RegsInternal& Regs() const override {
        return pica.regs.internal;
    }
    bool IsImmediate() const override {
        return immediate;
    }
    bool MayOverwriteColour() override {
        return May().color;
    }
    bool MayOverwriteDepth() override {
        return May().depth;
    }
    bool MayOverwriteDepthValues() override {
        return May().depth_values;
    }
    bool CoversViewport() override {
        if (!IsLarge()) {
            Cover();
            return covers;
        }
        // A mesh: looked at only where it would reset the depth buffer (a backdrop drawn
        // with the depth test at always), which is what makes it worth shading hundreds
        // of vertices here
        if (!large_known) {
            large_covers = !immediate && May().depth_values &&
                           pica.LargeMeshCoversPixels(is_indexed, StateHash(false));
            large_known = true;
        }
        return large_covers;
    }
    bool CoversViewportExactly() override {
        if (IsLarge()) {
            return false;
        }
        Cover();
        return covers && exactly;
    }
    bool CoverTestIsCostly() const override {
        return !immediate && IsLarge();
    }
    bool ViewportSpansTarget() override {
        const auto& rasterizer = pica.regs.internal.rasterizer;
        return rasterizer.viewport_corner.x <= 0 &&
               static_cast<f32>(rasterizer.viewport_corner.x) +
                       f24::FromRaw(rasterizer.viewport_size_x).ToFloat32() * 2.f >=
                   static_cast<f32>(pica.regs.internal.framebuffer.framebuffer.GetWidth());
    }
    bool TextureRows(VideoCore::DrawTextureRows& rows) override {
        return !immediate && pica.TextureRowsOfDraw(is_indexed, rows);
    }
    u64 StateHash(bool as_clear) override {
        const std::size_t which = as_clear ? 1 : 0;
        if (!hash_known[which]) {
            hash[which] = pica.TwinState(as_clear);
            hash_known[which] = true;
        }
        return hash[which];
    }
    u64 StateHashButLightVectors() override {
        if (!hash_but_known) {
            hash_but = pica.TwinState(false, false);
            hash_but_known = true;
        }
        return hash_but;
    }
    void ForStateRegisters(bool as_clear,
                           const std::function<void(u32, u32)>& stretch) const override {
        pica.ForTwinRegisters(as_clear, stretch);
    }
    const FloatUniforms& Uniforms() const override {
        return pica.vs_setup.uniforms.f;
    }
    const UniformSet& UniformsRead() override {
        return pica.FloatUniformsCompared();
    }
    const UniformSet& UniformsZeroSign() override {
        return pica.FloatUniformsZeroSign();
    }
    bool OutputHash(u64& hash) override {
        if (!output_known) {
            output_ok = !immediate && pica.SmallDrawOutputHash(is_indexed, output_hash);
            output_known = true;
        }
        hash = output_hash;
        return output_ok;
    }

private:
    const Overwrites& May() {
        if (!may_known) {
            // (Nothing is known of the triangles of an immediate draw)
            may = immediate ? Overwrites{} : MayOverwriteArea(pica.regs.internal);
            may_known = true;
        }
        return may;
    }
    bool IsLarge() const {
        return pica.regs.internal.pipeline.num_vertices > MAX_COVER_VERTICES;
    }
    void Cover() {
        if (!cover_known) {
            covers = !immediate && pica.CoversViewport(is_indexed, &exactly);
            cover_known = true;
        }
    }

    PicaCore& pica;
    bool is_indexed;
    bool immediate;
    bool may_known{};
    bool cover_known{};
    bool covers{};
    bool exactly{};
    bool large_known{};
    bool large_covers{};
    bool output_known{};
    bool output_ok{};
    u64 output_hash{};
    Overwrites may;
    std::array<bool, 2> hash_known{};
    u64 hash_but{};
    bool hash_but_known{};
    std::array<u64, 2> hash{};
};

void PicaCore::TellRulesOfImmediateDraw() {
    if (work_rules != nullptr) {
        Facts facts{*this, false, true};
        work_rules->LeavesOutDraw(facts);
    }
}

void PicaCore::DrawArrays(bool is_indexed) {
    MICROPROFILE_SCOPE(GPU_Drawing);
    CITRA_TRACE_SCOPE_DETAIL("PICA::DrawArrays");
    if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::MeasureCommandLists) != 0)
        [[unlikely]] {
        MeasureDrawSite(cmd_list.addr, cmd_list.current_index, regs.internal.reg_array);
    }

    // The scales for a top screen shown at another shape than the console's. Draws shaded here
    // that wait to be drawn together go out with the scales they were given.
    if (const WideScales scales = DecideWideScales(is_indexed); scales != wide_scales) {
        FlushSoftwareBatch();
        wide_scales = scales;
    }

    // A draw that need not be made is left out: one whose result nobody sees, one that only
    // makes over again what is there (VideoCore::WorkRules). The registers were written as
    // usual.
    Facts facts{*this, is_indexed, false};
    if (work_rules != nullptr && work_rules->LeavesOutDraw(facts)) {
        CITRA_FRAME_DUMP("draw left out: {}", work_rules->LastWhy());
        return;
    }

    // Track vertex in the debug recorder.
    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::IncomingPrimitiveBatch, nullptr);
    }

#ifdef CITRA_ENABLE_TRACE_MARKERS
    // Measurement aids: drop whole classes of draws to see what they cost or hide.
    if (const u32 experiments = Common::TraceMarkers::Experiments(); experiments != 0)
        [[unlikely]] {
        const auto& output_merger = regs.internal.framebuffer.output_merger;
        const bool lit = !regs.internal.lighting.disable;
        const bool flat = !lit && !output_merger.depth_test_enable;
        const bool effect = !lit && !flat && !output_merger.depth_write_enable;
        const bool solid = !lit && !flat && !effect;
        if ((lit && (experiments & Common::TraceMarkers::SkipLitDraws)) ||
            (flat && (experiments & Common::TraceMarkers::SkipFlatDraws)) ||
            (effect && (experiments & Common::TraceMarkers::SkipEffectDraws)) ||
            (solid && (experiments & Common::TraceMarkers::SkipUnlitSolidDraws))) {
            return;
        }
    }
#endif

    const bool accelerate_draw = [this] {
        // Geometry shaders cannot be accelerated due to register preservation.
        if (regs.internal.pipeline.use_gs == PipelineRegs::UseGS::Yes) {
            return false;
        }

        // TODO (wwylele): for Strip/Fan topology, if the primitive assember is not restarted
        // after this draw call, the buffered vertex from this draw should "leak" to the next
        // draw, in which case we should buffer the vertex into the software primitive assember,
        // or disable accelerate draw completely. However, there is not game found yet that does
        // this, so this is left unimplemented for now. Revisit this when an issue is found in
        // games.

        bool accelerate_draw =
            Settings::values.use_hw_shader && primitive_assembler.IsEmpty() &&
            (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::CpuVertexShading) == 0;
        const auto topology = primitive_assembler.GetTopology();
        if (topology == PipelineRegs::TriangleTopology::Shader ||
            topology == PipelineRegs::TriangleTopology::List) {
            accelerate_draw = accelerate_draw && (regs.internal.pipeline.num_vertices % 3) == 0;
        }
        return accelerate_draw;
    }();

    // Add vertices to the delay generator.
    delay_generator.AddVertices(regs.internal.pipeline.num_vertices,
                                regs.internal.pipeline.triangle_topology);

#ifdef CITRA_ENABLE_TRACE_MARKERS
    if (VideoCore::FrameStateDump::IsActive()) [[unlikely]] {
        VideoCore::FrameStateDump::Draw(regs.internal, accelerate_draw, is_indexed,
                                        vs_setup.GetProgramCodeHash());
        DumpSmallDraw(is_indexed);
        DumpGeometryProgram();
    }
#endif

    // A pixel of the target is several of the host's, which are all overwritten only by
    // triangles that reach the edges of the viewport, not just the middle of its last pixels
    const bool may_overwrite_color = facts.MayOverwriteColour();
    const bool may_overwrite_depth = facts.MayOverwriteDepth();
    const bool overwrites =
        (may_overwrite_color || may_overwrite_depth) && facts.CoversViewportExactly();
    const bool tells_overwrite = overwrites;
    // Draws shaded on the CPU that are waiting to be drawn together go first, before a draw
    // that is not added to them: one the host shades, or one the rasterizer is told about
    if (sw_batch_open && (accelerate_draw || tells_overwrite)) {
        FlushSoftwareBatch();
    }
    rasterizer->NoteDrawOverwrites(overwrites && may_overwrite_color,
                                   overwrites && may_overwrite_depth);

    // Attempt to use hardware vertex shaders if possible.
    if (accelerate_draw && rasterizer->AccelerateDrawBatch(is_indexed)) {
        rasterizer->NoteDrawOverwrites(false, false);
        CITRA_TRACE_COUNT(HwDraws);
        return;
    }

    CITRA_FRAME_DUMP("  draw: vertex shading on the CPU");
    CITRA_TRACE_COUNT(SwDraws);
    CITRA_TRACE_COUNT_N(SwVertices, regs.internal.pipeline.num_vertices);

    // We cannot accelerate the draw, so load and execute the vertex shader for each vertex.
    {
        CITRA_TRACE_SCOPE_DETAIL("PICA::SoftwareVertexShading");
        LoadVertices(is_indexed);
    }

    // The triangles are with the rasterizer. Draws like this one come in runs (sprites, a
    // quad each): they are drawn together, see FlushSoftwareBatch, unless this one has to be
    // seen on its own.
    bool on_its_own = tells_overwrite || debug_context || ++sw_batch_draws >= MAX_BATCHED_DRAWS;
#ifdef CITRA_ENABLE_TRACE_MARKERS
    on_its_own |= VideoCore::FrameStateDump::IsActive();
#endif
    if (on_its_own) {
        sw_batch_open = false;
        sw_batch_draws = 0;
        rasterizer->DrawTriangles();
        rasterizer->NoteDrawOverwrites(false, false);
    } else {
        sw_batch_open = true;
    }

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr);
    }
}

ShaderEngine& PicaCore::EngineForFewVertices() {
    if (shader_engine->SetupBatchIfReady(vs_setup, regs.internal.vs.main_offset)) {
        return *shader_engine;
    }
    few_vertices_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);
    return *few_vertices_engine;
}

float PicaCore::SmallDrawViewportShare(bool is_indexed) {
    DrawBox box;
    if (!SmallDrawBox(is_indexed, box)) {
        return 1.f;
    }
    if (!(box.right > box.left && box.top > box.bottom)) {
        return 0.f;
    }
    return (box.right - box.left) * (box.top - box.bottom) / 4.f;
}

bool PicaCore::SmallDrawBox(bool is_indexed, DrawBox& box) {
    const auto& pipeline = regs.internal.pipeline;
    const u32 count = pipeline.num_vertices;
    if (pipeline.use_gs != PipelineRegs::UseGS::No || !primitive_assembler.IsEmpty() ||
        count == 0 || count > 6) {
        return false;
    }

    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GpuRead(
        base_address + index_info.offset, pipeline.num_vertices * (index_info.format != 0 ? 2 : 1));
    if (is_indexed && index_address_8 == nullptr) {
        return false;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    const VertexLoader loader(memory, pipeline);
    ShaderUnit shader_unit;
    ShaderEngine& engine = EngineForFewVertices();

    // The box around the vertices, cut to the viewport, which goes from -1 to 1 both ways
    f32 left = 1.f;
    f32 right = -1.f;
    f32 bottom = 1.f;
    f32 top = -1.f;
    for (u32 index = 0; index < count; ++index) {
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);
        AttributeBuffer input;
        loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);
        shader_unit.LoadInput(regs.internal.vs, input);
        engine.Run(vs_setup, shader_unit);
        AttributeBuffer output;
        shader_unit.WriteOutput(regs.internal.vs, output);
        OutputVertex shaded{regs.internal.rasterizer, output};
        WidenShaded(wide_scales, shaded);

        // A triangle with a vertex behind the eye can reach anywhere
        const f32 w = shaded.pos.w.ToFloat32();
        const f32 x = shaded.pos.x.ToFloat32() / w;
        const f32 y = shaded.pos.y.ToFloat32() / w;
        if (!(w > 0.f) || !std::isfinite(x) || !std::isfinite(y)) {
            return false;
        }
        left = std::min(left, std::clamp(x, -1.f, 1.f));
        right = std::max(right, std::clamp(x, -1.f, 1.f));
        bottom = std::min(bottom, std::clamp(y, -1.f, 1.f));
        top = std::max(top, std::clamp(y, -1.f, 1.f));
    }
    box = DrawBox{left, right, bottom, top};
    return true;
}

bool PicaCore::CoversViewport(bool is_indexed, bool* exactly) {
    using Topology = PipelineRegs::TriangleTopology;
    const auto& pipeline = regs.internal.pipeline;
    if (pipeline.use_gs != PipelineRegs::UseGS::No || !primitive_assembler.IsEmpty()) {
        return false;
    }
    const Topology topology = pipeline.triangle_topology;
    const u32 count = pipeline.num_vertices;
    const bool is_list = topology == Topology::List || topology == Topology::Shader;
    if (count != (is_list ? 6u : 4u)) {
        return MeshCoversViewport(is_indexed, exactly);
    }

    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GpuRead(
        base_address + index_info.offset, pipeline.num_vertices * (index_info.format != 0 ? 2 : 1));
    if (is_indexed && index_address_8 == nullptr) {
        return false;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    const VertexLoader loader(memory, pipeline);
    ShaderUnit shader_unit;
    ShaderEngine& engine = EngineForFewVertices();

    // A full-screen pass is often drawn a fraction of a pixel off the viewport. It replaces
    // what the target holds as long as its triangles cover the centre of every pixel, which
    // they do with each vertex less than half a pixel inside of its corner. A pixel is one
    // of the console's here: that is where the game means the triangles to end.
    const f32 width = f24::FromRaw(regs.internal.rasterizer.viewport_size_x).ToFloat32() * 2.f;
    const f32 height = f24::FromRaw(regs.internal.rasterizer.viewport_size_y).ToFloat32() * 2.f;
    if (!(width >= 1.f && height >= 1.f)) {
        return false;
    }
    constexpr f32 MAX_INSIDE = 0.45f;
    constexpr f32 MAX_OUTSIDE = 1.f;
    constexpr f32 DEPTH_TOLERANCE = 1e-4f;
    // How far inside still counts as at the corner itself
    constexpr f32 MAX_INSIDE_EXACTLY = 0.01f;
    bool at_corners = true;

    // Where each vertex ends up, in pixels from the corner it is at (positive to the outside),
    // and that corner: bit 0 for the right, bit 1 for the top
    std::array<Common::Vec2<f32>, 6> offsets{};
    std::array<Common::Vec2<f32>, 6> positions{};
    std::array<u32, 6> corners{};
    for (u32 index = 0; index < count; ++index) {
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);
        AttributeBuffer input;
        loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);
        shader_unit.LoadInput(regs.internal.vs, input);
        engine.Run(vs_setup, shader_unit);
        AttributeBuffer output;
        shader_unit.WriteOutput(regs.internal.vs, output);
        OutputVertex shaded{regs.internal.rasterizer, output};
        WidenShaded(wide_scales, shaded);

        // What follows is the quick way, for two triangles with their corners at those of
        // the viewport, which is how nearly every full-screen pass is drawn. Two triangles
        // that are not that may cover the viewport all the same (New Super Mario Bros. 2
        // clears with a quad many times its size): the way for any small mesh finds out,
        // which cuts triangles at the depth range and at the viewport as the GPU does.
        const f32 w = shaded.pos.w.ToFloat32();
        if (!(w > 0.f)) {
            return MeshCoversViewport(is_indexed, exactly);
        }
        // A vertex in front of or behind the depth range has its triangle cut
        const f32 z = shaded.pos.z.ToFloat32() / w;
        if (!(z <= DEPTH_TOLERANCE && z >= -1.f - DEPTH_TOLERANCE)) {
            return MeshCoversViewport(is_indexed, exactly);
        }
        const f32 x = shaded.pos.x.ToFloat32() / w;
        const f32 y = shaded.pos.y.ToFloat32() / w;
        const Common::Vec2<f32> offset{(std::abs(x) - 1.f) * width / 2.f,
                                       (std::abs(y) - 1.f) * height / 2.f};
        if (!(offset.x >= -MAX_INSIDE && offset.y >= -MAX_INSIDE)) {
            // Inside the viewport: with a corner of theirs in it two triangles leave a part of
            // it uncovered (but for one of them covering it alone, which no game was seen to
            // draw). This is where every sprite ends, without more being asked of it.
            CITRA_FRAME_DUMP("  draw: vertex {} at {},{} is {},{} pixels off a corner of the "
                             "viewport",
                             index, x, y, offset.x, offset.y);
            return false;
        }
        if (!(offset.x <= MAX_OUTSIDE && offset.y <= MAX_OUTSIDE)) {
            CITRA_FRAME_DUMP("  draw: vertex {} at {},{} is {},{} pixels outside of the "
                             "viewport: looked at as a mesh",
                             index, x, y, offset.x, offset.y);
            return MeshCoversViewport(is_indexed, exactly);
        }
        offsets[index] = offset;
        positions[index] = {x, y};
        corners[index] = (x > 0.f ? 1u : 0u) | (y > 0.f ? 2u : 0u);
        at_corners =
            at_corners && offset.x >= -MAX_INSIDE_EXACTLY && offset.y >= -MAX_INSIDE_EXACTLY;
    }

    const std::array<u32, 3> first{0, 1, 2};
    const std::array<u32, 3> second = is_list                       ? std::array<u32, 3>{3, 4, 5}
                                      : topology == Topology::Strip ? std::array<u32, 3>{1, 2, 3}
                                                                    : std::array<u32, 3>{0, 2, 3};
    // The corner a triangle lacks, if its vertices are at three different ones
    const auto lacking = [&corners](const std::array<u32, 3>& triangle) -> s32 {
        const u32 present = (1u << corners[triangle[0]]) | (1u << corners[triangle[1]]) |
                            (1u << corners[triangle[2]]);
        return std::popcount(present) == 3 ? std::countr_zero(~present & 0xFu) : -1;
    };
    // Two halves of the viewport along a diagonal lack corners opposite each other
    const s32 first_lacks = lacking(first);
    const s32 second_lacks = lacking(second);
    if (first_lacks < 0 || second_lacks < 0 || (first_lacks ^ second_lacks) != 3) {
        return MeshCoversViewport(is_indexed, exactly);
    }
    // No gap along that diagonal: the two triangles end in the same places
    for (const u32 a : first) {
        for (const u32 b : second) {
            if (corners[a] == corners[b] && offsets[a] != offsets[b]) {
                return MeshCoversViewport(is_indexed, exactly);
            }
        }
    }
    // A triangle that faces away from what is kept is not drawn at all. Which way it faces is
    // what the software rasterizer goes by: the sign of its area on the screen, with the
    // vertices in the order the triangle is put together in (a strip takes those of every
    // other triangle the other way round).
    using CullMode = RasterizerRegs::CullMode;
    const CullMode cull_mode = regs.internal.rasterizer.cull_mode;
    if (cull_mode == CullMode::KeepClockWise || cull_mode == CullMode::KeepCounterClockWise) {
        const std::array<u32, 3> second_as_drawn =
            topology == Topology::Strip ? std::array<u32, 3>{2, 1, 3} : second;
        const auto area = [&positions](const std::array<u32, 3>& triangle) {
            const auto ab = positions[triangle[1]] - positions[triangle[0]];
            const auto ac = positions[triangle[2]] - positions[triangle[0]];
            return ab.x * ac.y - ab.y * ac.x;
        };
        const f32 kept = cull_mode == CullMode::KeepCounterClockWise ? 1.f : -1.f;
        if (!(area(first) * kept > 0.f && area(second_as_drawn) * kept > 0.f)) {
            CITRA_FRAME_DUMP("  draw: covers the viewport but faces away, cull mode {}",
                             static_cast<u32>(cull_mode));
            return false;
        }
    }
    if (exactly != nullptr) {
        *exactly = at_corners;
    }
    return true;
}

namespace {

using Point = Common::Vec2<f32>;
/// A convex polygon
using Polygon = boost::container::small_vector<Point, 10>;

f32 Cross(const Point& a, const Point& b, const Point& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

/// The part of the polygon to the left of the line from a to b, or to its right
Polygon SideOf(const Polygon& polygon, const Point& a, const Point& b, bool left) {
    Polygon part;
    const auto side = [&](const Point& point) {
        const f32 cross = Cross(a, b, point);
        return left ? cross : -cross;
    };
    for (std::size_t i = 0; i < polygon.size(); i++) {
        const Point& from = polygon[i];
        const Point& to = polygon[(i + 1) % polygon.size()];
        const f32 from_side = side(from);
        const f32 to_side = side(to);
        if (from_side >= 0.f) {
            part.push_back(from);
        }
        if ((from_side > 0.f && to_side < 0.f) || (from_side < 0.f && to_side > 0.f)) {
            const f32 along = from_side / (from_side - to_side);
            part.push_back(
                Point{from.x + (to.x - from.x) * along, from.y + (to.y - from.y) * along});
        }
    }
    return part;
}

f32 AreaOf(const Polygon& polygon) {
    f32 twice = 0.f;
    for (std::size_t i = 0; i < polygon.size(); i++) {
        const Point& from = polygon[i];
        const Point& to = polygon[(i + 1) % polygon.size()];
        twice += from.x * to.y - to.x * from.y;
    }
    return std::abs(twice) / 2.f;
}

/// Pieces smaller than this, in pixels of the console, are what rounding leaves along the
/// edges two triangles share
constexpr f32 SLIVER = 1e-3f;

f32 SignedAreaOf(const Polygon& polygon) {
    f32 twice = 0.f;
    for (std::size_t i = 0; i < polygon.size(); i++) {
        const Point& from = polygon[i];
        const Point& to = polygon[(i + 1) % polygon.size()];
        twice += from.x * to.y - to.x * from.y;
    }
    return twice / 2.f;
}

/// Takes the convex shape away from the pieces
void TakeAway(std::vector<Polygon>& pieces, Polygon shape) {
    // Counter-clockwise, so that its inside is to the left of each of its edges
    if (SignedAreaOf(shape) < 0.f) {
        std::reverse(shape.begin(), shape.end());
    }
    std::vector<Polygon> left_over;
    for (Polygon& piece : pieces) {
        Polygon inside = std::move(piece);
        for (std::size_t i = 0; i < shape.size() && inside.size() >= 3; i++) {
            const Point& from = shape[i];
            const Point& to = shape[(i + 1) % shape.size()];
            // (Cutting leaves corners that fall together: no edge between those)
            if (std::abs(to.x - from.x) + std::abs(to.y - from.y) < 1e-6f) {
                continue;
            }
            Polygon outside = SideOf(inside, from, to, false);
            if (outside.size() >= 3 && AreaOf(outside) > SLIVER) {
                left_over.push_back(std::move(outside));
            }
            inside = SideOf(inside, from, to, true);
        }
    }
    pieces = std::move(left_over);
}

/// A vertex as the vertex shader puts it out, and a convex polygon of such
using ClipPoint = Common::Vec4<f32>;
using ClipPolygon = boost::container::small_vector<ClipPoint, 8>;

/// The part of the polygon where the value is not negative
template <typename Value>
ClipPolygon KeepWhere(const ClipPolygon& polygon, Value&& value) {
    ClipPolygon part;
    for (std::size_t i = 0; i < polygon.size(); i++) {
        const ClipPoint& from = polygon[i];
        const ClipPoint& to = polygon[(i + 1) % polygon.size()];
        const f32 from_value = value(from);
        const f32 to_value = value(to);
        if (from_value >= 0.f) {
            part.push_back(from);
        }
        if ((from_value > 0.f && to_value < 0.f) || (from_value < 0.f && to_value > 0.f)) {
            const f32 along = from_value / (from_value - to_value);
            part.push_back(
                ClipPoint{from.x + (to.x - from.x) * along, from.y + (to.y - from.y) * along,
                          from.z + (to.z - from.z) * along, from.w + (to.w - from.w) * along});
        }
    }
    return part;
}

} // Anonymous namespace

template <typename Sink>
bool PicaCore::ForDrawnTriangles(bool is_indexed, u32 most_vertices, f32& width, f32& height,
                                 Sink&& sink) {
    using Topology = PipelineRegs::TriangleTopology;
    const auto& pipeline = regs.internal.pipeline;
    const Topology topology = pipeline.triangle_topology;
    const u32 count = pipeline.num_vertices;
    const bool is_list = topology == Topology::List || topology == Topology::Shader;
    if (pipeline.use_gs != PipelineRegs::UseGS::No || !primitive_assembler.IsEmpty() || count < 3 ||
        count > most_vertices || (is_list && count % 3 != 0)) {
        return false;
    }

    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GpuRead(
        base_address + index_info.offset, pipeline.num_vertices * (index_info.format != 0 ? 2 : 1));
    if (is_indexed && index_address_8 == nullptr) {
        return false;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    const VertexLoader loader(memory, pipeline);
    ShaderUnit shader_unit;
    // A mesh is shaded with the vertex shader made into host code, which is made for it here
    // once if the host has not been shading on the CPU with it: hundreds of vertices with
    // every frame are too many for the interpreter that a few vertices are given to
    const bool mesh = count > MAX_COVER_VERTICES;
    if (mesh) {
        shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);
    }
    ShaderEngine& engine = mesh ? *shader_engine : EngineForFewVertices();

    // In pixels of the console, from the corner of the viewport
    width = f24::FromRaw(regs.internal.rasterizer.viewport_size_x).ToFloat32() * 2.f;
    height = f24::FromRaw(regs.internal.rasterizer.viewport_size_y).ToFloat32() * 2.f;
    if (!(width >= 1.f && height >= 1.f)) {
        return false;
    }
    constexpr f32 DEPTH_TOLERANCE = 1e-4f;

    boost::container::small_vector<ClipPoint, MAX_COVER_VERTICES> at(count);
    for (u32 index = 0; index < count; ++index) {
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);
        AttributeBuffer input;
        loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);
        shader_unit.LoadInput(regs.internal.vs, input);
        engine.Run(vs_setup, shader_unit);
        AttributeBuffer output;
        shader_unit.WriteOutput(regs.internal.vs, output);
        OutputVertex shaded{regs.internal.rasterizer, output};
        WidenShaded(wide_scales, shaded);
        at[index] = ClipPoint{shaded.pos.x.ToFloat32(), shaded.pos.y.ToFloat32(),
                              shaded.pos.z.ToFloat32(), shaded.pos.w.ToFloat32()};
        if (!std::isfinite(at[index].x) || !std::isfinite(at[index].y) ||
            !std::isfinite(at[index].z) || !std::isfinite(at[index].w)) {
            return false;
        }
    }

    using CullMode = RasterizerRegs::CullMode;
    const CullMode cull_mode = regs.internal.rasterizer.cull_mode;
    const bool culls =
        cull_mode == CullMode::KeepClockWise || cull_mode == CullMode::KeepCounterClockWise;
    const f32 kept = cull_mode == CullMode::KeepCounterClockWise ? 1.f : -1.f;
    const u32 triangles = is_list ? count / 3 : count - 2;
    for (u32 triangle = 0; triangle < triangles; triangle++) {
        std::array<u32, 3> corner;
        if (is_list) {
            corner = {triangle * 3, triangle * 3 + 1, triangle * 3 + 2};
        } else if (topology == Topology::Strip) {
            // A strip takes the vertices of every other triangle the other way round
            corner = triangle % 2 == 0 ? std::array<u32, 3>{triangle, triangle + 1, triangle + 2}
                                       : std::array<u32, 3>{triangle + 1, triangle, triangle + 2};
        } else {
            corner = {0, triangle + 1, triangle + 2};
        }
        // A triangle that is wholly beyond one side of the viewport covers nothing of it, and
        // neither does what the cuts below leave of it: no more is done with it. Of a backdrop
        // around the camera most triangles are such. (That this changes no pixel was checked
        // against the path without it on two million triangles: cover_test.cpp, kept with the
        // notes of 2026-11-22.)
        {
            const ClipPoint& a = at[corner[0]];
            const ClipPoint& b = at[corner[1]];
            const ClipPoint& c = at[corner[2]];
            if ((a.x > a.w && b.x > b.w && c.x > c.w) || (a.x < -a.w && b.x < -b.w && c.x < -c.w) ||
                (a.y > a.w && b.y > b.w && c.y > c.w) || (a.y < -a.w && b.y < -b.w && c.y < -c.w)) {
                continue;
            }
        }
        // What is in front of the depth range, behind it or behind the eye is cut off, as the
        // console and the host do (a box around the scene, as a game may clear with, has faces
        // that reach behind the range). As far outside of it as CoversViewport lets a vertex
        // be still counts as inside.
        ClipPolygon drawn{at[corner[0]], at[corner[1]], at[corner[2]]};
        drawn = KeepWhere(drawn, [](const ClipPoint& p) { return p.w - 1e-6f; });
        drawn = KeepWhere(drawn, [](const ClipPoint& p) { return p.w * DEPTH_TOLERANCE - p.z; });
        drawn = KeepWhere(drawn,
                          [](const ClipPoint& p) { return p.z + p.w * (1.f + DEPTH_TOLERANCE); });
        if (drawn.size() < 3) {
            continue;
        }
        Polygon shape;
        // (A corner too far out to say where its edges pass to a part of a pixel: such a
        // triangle is not counted on)
        constexpr f32 FAR_OUT = 1e5f;
        for (const ClipPoint& p : drawn) {
            const Point point{(p.x / p.w + 1.f) * width / 2.f, (p.y / p.w + 1.f) * height / 2.f};
            if (!(std::abs(point.x) < FAR_OUT && std::abs(point.y) < FAR_OUT)) {
                shape.clear();
                break;
            }
            shape.push_back(point);
        }
        if (shape.size() < 3) {
            continue;
        }
        // Which way it faces, as in CoversViewport
        const f32 area = SignedAreaOf(shape);
        if (culls && !(area * kept > 0.f)) {
            continue;
        }
        if (!(std::abs(area) > SLIVER)) {
            continue;
        }
        if (!sink(std::move(shape))) {
            break;
        }
    }
    return true;
}

bool PicaCore::MeshCoversViewport(bool is_indexed, bool* exactly) {
    // As in CoversViewport: every pixel is covered whose middle is, so the triangles may end
    // up to nearly half a pixel inside of the viewport
    constexpr f32 MAX_INSIDE = 0.45f;
    constexpr f32 MAX_INSIDE_EXACTLY = 0.01f;
    // What may be left uncovered in all, as rounding leaves it
    constexpr f32 LEFT_BY_ROUNDING = 1e-2f;
    constexpr std::size_t MOST_PIECES = 256;

    // What of the viewport is not covered yet: all of it but a hair at its edges, from which
    // what is drawn of each triangle is taken away
    f32 width = 0.f;
    f32 height = 0.f;
    std::vector<Polygon> pieces;
    bool begun = false;
    bool too_many = false;
    u32 triangles = 0;
    const bool shaded =
        ForDrawnTriangles(is_indexed, MAX_COVER_VERTICES, width, height, [&](Polygon&& shape) {
            if (!begun) {
                begun = true;
                pieces.push_back(
                    Polygon{Point{MAX_INSIDE_EXACTLY, MAX_INSIDE_EXACTLY},
                            Point{width - MAX_INSIDE_EXACTLY, MAX_INSIDE_EXACTLY},
                            Point{width - MAX_INSIDE_EXACTLY, height - MAX_INSIDE_EXACTLY},
                            Point{MAX_INSIDE_EXACTLY, height - MAX_INSIDE_EXACTLY}});
            }
            triangles++;
            TakeAway(pieces, std::move(shape));
            too_many = pieces.size() > MOST_PIECES;
            return !too_many && !pieces.empty();
        });
    if (!shaded || !begun || too_many) {
        return false;
    }
    f32 left = 0.f;
    for (const Polygon& piece : pieces) {
        left += AreaOf(piece);
    }
    if (left <= LEFT_BY_ROUNDING) {
        CITRA_FRAME_DUMP("  draw: its {} triangles cover the viewport", triangles);
        if (exactly != nullptr) {
            *exactly = true;
        }
        return true;
    }
    // Not up to the edges: whether at least to the middles of the pixels there
    const Point low{MAX_INSIDE, MAX_INSIDE};
    const Point high{width - MAX_INSIDE, height - MAX_INSIDE};
    left = 0.f;
    for (Polygon& piece : pieces) {
        piece = SideOf(piece, low, Point{high.x, low.y}, true);
        piece = SideOf(piece, Point{high.x, low.y}, high, true);
        piece = SideOf(piece, high, Point{low.x, high.y}, true);
        piece = SideOf(piece, Point{low.x, high.y}, low, true);
        if (piece.size() >= 3) {
            left += AreaOf(piece);
        }
    }
    CITRA_FRAME_DUMP(
        "  draw: its {} triangles leave {:.2f} square pixels of the viewport uncovered", triangles,
        left);
    if (left > LEFT_BY_ROUNDING) {
        return false;
    }
    if (exactly != nullptr) {
        *exactly = false;
    }
    return true;
}

bool PicaCore::LargeMeshCoversPixels(bool is_indexed, u64 state) {
    // A mesh that was found not to cover its viewport a few times running is not looked at
    // for a while: looking shades all its vertices, and one that covers nothing would cost
    // that with every frame. (One that does cover is looked at every time: that it did the
    // frame before says nothing of this one.)
    constexpr u32 MISSES_BEFORE_REST = 3;
    constexpr u32 REST = 512;
    CoverMiss* known = nullptr;
    for (CoverMiss& miss : cover_misses) {
        if (miss.state == state) {
            known = &miss;
        }
    }
    if (known != nullptr && known->rest != 0) {
        known->rest--;
        return false;
    }

    // A pixel of the console for a bit, set where the middle of the pixel is within a
    // triangle that is drawn, or a hair outside of it: where two triangles meet, the middle
    // of a pixel on their edge belongs to one of them.
    constexpr f32 HAIR = 1e-3f;
    constexpr u32 MOST_ACROSS = 1024;
    f32 width = 0.f;
    f32 height = 0.f;
    u32 columns = 0;
    u32 rows = 0;
    u32 words = 0;
    bool usable = true;
    u32 triangles = 0;
    const bool shaded = ForDrawnTriangles(
        is_indexed, MAX_COVER_VERTICES_LARGE, width, height, [&](Polygon&& shape) {
            if (rows == 0) {
                columns = static_cast<u32>(std::lround(width));
                rows = static_cast<u32>(std::lround(height));
                if (columns == 0 || rows == 0 || columns > MOST_ACROSS || rows > MOST_ACROSS ||
                    std::abs(width - static_cast<f32>(columns)) > HAIR ||
                    std::abs(height - static_cast<f32>(rows)) > HAIR) {
                    usable = false;
                    return false;
                }
                words = (columns + 63) / 64;
                cover_bits.assign(std::size_t{rows} * words, 0);
            }
            triangles++;
            // Its edges, each from its lower end to its upper one, with how far it goes
            // sideways for each row it goes up: a row asks every edge where it is at the
            // row's height, which is then a multiplication and no division
            struct Edge {
                f32 low;
                f32 high;
                f32 x_low;
                f32 x_high;
                f32 slope;
            };
            std::array<Edge, 12> edges;
            if (shape.size() > edges.size()) {
                // (Not counted on)
                return true;
            }
            f32 lowest = shape[0].y;
            f32 highest = shape[0].y;
            for (std::size_t i = 0; i < shape.size(); i++) {
                const Point& from = shape[i];
                const Point& to = shape[(i + 1) % shape.size()];
                const Point& lower = from.y <= to.y ? from : to;
                const Point& upper = from.y <= to.y ? to : from;
                edges[i] =
                    Edge{lower.y, upper.y, lower.x, upper.x,
                         upper.y != lower.y ? (upper.x - lower.x) / (upper.y - lower.y) : 0.f};
                lowest = std::min(lowest, from.y);
                highest = std::max(highest, from.y);
            }
            const std::size_t edge_count = shape.size();
            const s32 first_row =
                std::max<s32>(0, static_cast<s32>(std::ceil(lowest - 0.5f - HAIR)));
            const s32 last_row = std::min<s32>(static_cast<s32>(rows) - 1,
                                               static_cast<s32>(std::floor(highest - 0.5f + HAIR)));
            for (s32 row = first_row; row <= last_row; row++) {
                // Where the line through the middles of this row's pixels is inside the shape
                const f32 y = static_cast<f32>(row) + 0.5f;
                f32 left = std::numeric_limits<f32>::max();
                f32 right = std::numeric_limits<f32>::lowest();
                for (std::size_t i = 0; i < edge_count; i++) {
                    const Edge& edge = edges[i];
                    if (!(edge.low <= y && edge.high >= y)) {
                        continue;
                    }
                    if (edge.low == edge.high) {
                        left = std::min({left, edge.x_low, edge.x_high});
                        right = std::max({right, edge.x_low, edge.x_high});
                    } else {
                        const f32 x = edge.x_low + (y - edge.low) * edge.slope;
                        left = std::min(left, x);
                        right = std::max(right, x);
                    }
                }
                if (!(right >= left)) {
                    continue;
                }
                const s32 first = std::max<s32>(0, static_cast<s32>(std::ceil(left - 0.5f - HAIR)));
                const s32 last = std::min<s32>(static_cast<s32>(columns) - 1,
                                               static_cast<s32>(std::floor(right - 0.5f + HAIR)));
                u64* const line = &cover_bits[static_cast<std::size_t>(row) * words];
                for (s32 word = first / 64; first <= last && word <= last / 64; word++) {
                    const s32 from_bit = std::max(first, word * 64) - word * 64;
                    const s32 to_bit = std::min(last, word * 64 + 63) - word * 64;
                    const u64 upto = to_bit == 63 ? ~u64{0} : (u64{1} << (to_bit + 1)) - 1;
                    line[word] |= upto & ~((u64{1} << from_bit) - 1);
                }
            }
            return true;
        });

    bool covers = shaded && usable && rows != 0;
    for (u32 row = 0; covers && row < rows; row++) {
        for (u32 word = 0; word < words; word++) {
            const u32 bits_here = std::min<u32>(64, columns - word * 64);
            const u64 all = bits_here == 64 ? ~u64{0} : (u64{1} << bits_here) - 1;
            if ((cover_bits[std::size_t{row} * words + word] & all) != all) {
                covers = false;
                break;
            }
        }
    }
    CITRA_FRAME_DUMP("  draw: a mesh of {} triangles drawn {} every pixel of its viewport",
                     triangles, covers ? "covers" : "does not cover");

    if (covers) {
        if (known != nullptr) {
            known->misses = 0;
        }
        return true;
    }
    if (known == nullptr) {
        known = &cover_misses[cover_miss_next++ % cover_misses.size()];
        *known = CoverMiss{.state = state};
    }
    if (++known->misses >= MISSES_BEFORE_REST) {
        known->misses = 0;
        known->rest = REST;
    }
    return false;
}

bool PicaCore::SmallDrawOutputHash(bool is_indexed, u64& hash) {
    const auto& pipeline = regs.internal.pipeline;
    const u32 count = pipeline.num_vertices;
    if (pipeline.use_gs != PipelineRegs::UseGS::No || !primitive_assembler.IsEmpty() ||
        count == 0 || count > MAX_COVER_VERTICES) {
        return false;
    }
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GpuRead(
        base_address + index_info.offset, pipeline.num_vertices * (index_info.format != 0 ? 2 : 1));
    if (is_indexed && index_address_8 == nullptr) {
        return false;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    const VertexLoader loader(memory, pipeline);
    ShaderUnit shader_unit;
    ShaderEngine& engine = EngineForFewVertices();
    // The output registers that are switched on, which is what the rasterizer is given of a
    // vertex, in their order
    const std::size_t outputs = std::min<std::size_t>(
        std::popcount(static_cast<u32>(regs.internal.vs.output_mask)), AttributeBuffer{}.size());
    u64 result = Common::HashCombine(u64{count}, u64{outputs});
    for (u32 index = 0; index < count; ++index) {
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);
        AttributeBuffer input;
        loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);
        shader_unit.LoadInput(regs.internal.vs, input);
        engine.Run(vs_setup, shader_unit);
        AttributeBuffer output;
        shader_unit.WriteOutput(regs.internal.vs, output);
        result = Common::HashCombine(
            result, Common::ComputeHash64(output.data(), outputs * sizeof(output[0])));
    }
    hash = result;
    return true;
}

bool PicaCore::TextureRowsOfDraw(bool is_indexed, VideoCore::DrawTextureRows& rows) {
    const auto& pipeline = regs.internal.pipeline;
    const u32 count = pipeline.num_vertices;
    if (pipeline.use_gs != PipelineRegs::UseGS::No || !primitive_assembler.IsEmpty() || count < 3 ||
        count > 6) {
        return false;
    }
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GpuRead(
        base_address + index_info.offset, pipeline.num_vertices * (index_info.format != 0 ? 2 : 1));
    if (is_indexed && index_address_8 == nullptr) {
        return false;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    const VertexLoader loader(memory, pipeline);
    ShaderUnit shader_unit;
    ShaderEngine& engine = EngineForFewVertices();

    const auto& rasterizer = regs.internal.rasterizer;
    const f32 height = f24::FromRaw(rasterizer.viewport_size_y).ToFloat32() * 2.f;
    // The row each vertex ends up in and its three texture coordinates along the rows
    std::array<f32, 6> row{};
    std::array<std::array<f32, 3>, 6> coordinate{};
    f32 first_w = 0.f;
    u32 lowest = 0;
    u32 highest = 0;
    for (u32 index = 0; index < count; ++index) {
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);
        AttributeBuffer input;
        loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);
        shader_unit.LoadInput(regs.internal.vs, input);
        engine.Run(vs_setup, shader_unit);
        AttributeBuffer output;
        shader_unit.WriteOutput(regs.internal.vs, output);
        const OutputVertex shaded{rasterizer, output};

        // With all vertices at one distance, what goes in a straight line with the rows at
        // the vertices does so at every pixel between them
        const f32 w = shaded.pos.w.ToFloat32();
        if (!(w > 0.f)) {
            return false;
        }
        if (index == 0) {
            first_w = w;
        } else if (std::abs(w - first_w) > 1e-5f * first_w) {
            return false;
        }
        row[index] = (shaded.pos.y.ToFloat32() / w + 1.f) * height / 2.f +
                     static_cast<f32>(rasterizer.viewport_corner.y);
        if (!std::isfinite(row[index])) {
            return false;
        }
        coordinate[index] = {shaded.tc0.y.ToFloat32(), shaded.tc1.y.ToFloat32(),
                             shaded.tc2.y.ToFloat32()};
        if (row[index] < row[lowest]) {
            lowest = index;
        }
        if (row[index] > row[highest]) {
            highest = index;
        }
    }
    const f32 span = row[highest] - row[lowest];
    if (!(span >= 1.f)) {
        return false;
    }

    // The line through the lowest and the highest vertex has to pass through all the
    // others, to within a small part of a texel of the largest texture there is
    constexpr f32 TOLERANCE = 1.f / 16384.f;
    bool any = false;
    for (u32 set = 0; set < 3; ++set) {
        const f32 scale = (coordinate[highest][set] - coordinate[lowest][set]) / span;
        const f32 offset = coordinate[lowest][set] - scale * row[lowest];
        bool on_the_line = std::isfinite(scale) && std::isfinite(offset);
        for (u32 index = 0; index < count && on_the_line; ++index) {
            on_the_line =
                std::abs(scale * row[index] + offset - coordinate[index][set]) <= TOLERANCE;
        }
        rows.known[set] = on_the_line;
        rows.scale[set] = scale;
        rows.offset[set] = offset;
        any = any || on_the_line;
    }
    return any;
}

#ifdef CITRA_ENABLE_TRACE_MARKERS
void PicaCore::DumpGeometryProgram() {
    const auto& pipeline = regs.internal.pipeline;
    if (pipeline.use_gs != PipelineRegs::UseGS::Yes) {
        return;
    }
    const auto& gs = regs.internal.gs;
    const auto& vs = regs.internal.vs;
    // Once for each program as it is configured
    static std::array<u64, 16> listed{};
    static u32 next_listed = 0;
    u64 key = gs_setup.GetProgramCodeHash() ^ (gs_setup.GetSwizzleDataHash() * 3);
    key ^= u64{gs.main_offset} << 48 ^ u64 { gs.input_buffer_config } << 16 ^ gs.output_mask;
    key ^= u64{pipeline.gs_config.mode.Value() == PipelineRegs::GSMode::Point ? 1u : 2u} << 40;
    key ^= u64{pipeline.vs_outmap_total_minus_1_a} << 44;
    if (std::ranges::find(listed, key) != listed.end()) {
        return;
    }
    listed[next_listed++ % listed.size()] = key;

    std::string inputs;
    for (u32 attr = 0; attr <= gs.max_input_attribute_index; attr++) {
        inputs += fmt::format("{}v{}", attr == 0 ? "" : ",", gs.GetRegisterForAttribute(attr));
    }
    std::string vs_inputs;
    for (u32 attr = 0; attr <= vs.max_input_attribute_index; attr++) {
        vs_inputs += fmt::format("{}v{}", attr == 0 ? "" : ",", vs.GetRegisterForAttribute(attr));
    }
    std::string ints;
    for (const auto& values : gs.int_uniforms) {
        ints += fmt::format(" ({},{},{})", values.x.Value(), values.y.Value(), values.z.Value());
    }
    CITRA_FRAME_DUMP(
        "  gs program {:016X} swizzle {:016X} main={} mode={} vs_outputs={} vs_output_mask={:04X} "
        "gs_inputs={} [{}] gs_output_mask={:04X} exclusive={} variable={} fixed_vertices={} "
        "stride={} start={} main_vertices={} input_to_uniform={} bools={:04X} ints={} | vs "
        "{:016X} main={} inputs [{}] | outmap total {}",
        gs_setup.GetProgramCodeHash(), gs_setup.GetSwizzleDataHash(), gs.main_offset.Value(),
        static_cast<u32>(pipeline.gs_config.mode.Value()),
        pipeline.vs_outmap_total_minus_1_a.Value() + 1, vs.output_mask.Value(),
        gs.max_input_attribute_index.Value() + 1, inputs, gs.output_mask.Value(),
        pipeline.gs_unit_exclusive_configuration.Value(), pipeline.variable_primitive.Value(),
        pipeline.gs_config.fixed_vertex_num_minus_1.Value() + 1,
        pipeline.gs_config.stride_minus_1.Value() + 1, pipeline.gs_config.start_index.Value(),
        pipeline.variable_vertex_main_num_minus_1 + 1, gs.input_to_uniform.Value(),
        gs.bool_uniforms.Value(), ints, vs_setup.GetProgramCodeHash(), vs.main_offset.Value(),
        vs_inputs, regs.internal.rasterizer.vs_output_total.Value());
    for (u32 i = 0; i < regs.internal.rasterizer.vs_output_total && i < 7; i++) {
        const auto& map = regs.internal.rasterizer.vs_output_attributes[i];
        CITRA_FRAME_DUMP("  gs outmap {}: {} {} {} {}", i, static_cast<u32>(map.map_x.Value()),
                         static_cast<u32>(map.map_y.Value()), static_cast<u32>(map.map_z.Value()),
                         static_cast<u32>(map.map_w.Value()));
    }
    const auto words = [](std::string_view name, std::span<const u32> data) {
        std::size_t last = data.size();
        while (last > 0 && data[last - 1] == 0) {
            last--;
        }
        for (std::size_t at = 0; at < last; at += 8) {
            std::string line;
            for (std::size_t i = at; i < std::min(at + 8, last); i++) {
                line += fmt::format(" {:08X}", data[i]);
            }
            CITRA_FRAME_DUMP("  gs {} {:04X}:{}", name, at, line);
        }
    };
    words("code", gs_setup.GetProgramCode());
    words("operands", gs_setup.GetSwizzleData());
    // The float uniforms it has now, which tell what its constants are
    for (u32 i = 0; i < 96; i++) {
        const auto& value = gs_setup.uniforms.f[i];
        if (value.x.ToFloat32() != 0.f || value.y.ToFloat32() != 0.f ||
            value.z.ToFloat32() != 0.f || value.w.ToFloat32() != 0.f) {
            CITRA_FRAME_DUMP("  gs c{}: {} {} {} {}", i, value.x.ToFloat32(), value.y.ToFloat32(),
                             value.z.ToFloat32(), value.w.ToFloat32());
        }
    }
}

void PicaCore::DumpMeshDraw(bool is_indexed) {
    const auto& pipeline = regs.internal.pipeline;
    const u32 count = pipeline.num_vertices;
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const bool index_u16 = index_info.format != 0;
    const u8* index_address_8 =
        memory.GpuRead(base_address + index_info.offset, count * (index_u16 ? 2 : 1));
    if (is_indexed && index_address_8 == nullptr) {
        VideoCore::FrameStateDump::Line("  draw mesh: its indices are not in memory");
        return;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);

    const VertexLoader loader(memory, pipeline);
    ShaderUnit shader_unit;
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    const auto& rasterizer = regs.internal.rasterizer;
    const f32 width = f24::FromRaw(rasterizer.viewport_size_x).ToFloat32() * 2.f;
    const f32 height = f24::FromRaw(rasterizer.viewport_size_y).ToFloat32() * 2.f;
    f32 lowest[4] = {1e30f, 1e30f, 1e30f, 1e30f};
    f32 highest[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    u32 behind = 0;
    u32 broken = 0;
    u32 lowest_vertex = 0xFFFFFFFF;
    u32 highest_vertex = 0;
    std::string first;
    // Inputs that are no ordinary numbers: the console multiplies an infinity by zero to zero,
    // a host GPU to "not a number", so these are where a mesh can differ between the two
    std::array<u32, 12> infinite_inputs{};
    std::array<u32, 12> nan_inputs{};
    std::array<f32, 12> largest_input{};
    std::string odd_vertex;
    for (u32 index = 0; index < count; ++index) {
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);
        lowest_vertex = std::min(lowest_vertex, vertex);
        highest_vertex = std::max(highest_vertex, vertex);
        AttributeBuffer input;
        loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);
        for (u32 attribute = 0; attribute < 12; attribute++) {
            bool odd = false;
            for (u32 component = 0; component < 4; component++) {
                const f32 value = input[attribute][component].ToFloat32();
                if (value != value) {
                    nan_inputs[attribute]++;
                    odd = true;
                } else if (std::abs(value) > 3.0e38f) {
                    infinite_inputs[attribute]++;
                    odd = true;
                } else {
                    largest_input[attribute] = std::max(largest_input[attribute], std::abs(value));
                }
            }
            if (odd && odd_vertex.empty()) {
                odd_vertex =
                    fmt::format(" the first: vertex #{} a{}=({},{},{},{})", vertex, attribute,
                                input[attribute].x.ToFloat32(), input[attribute].y.ToFloat32(),
                                input[attribute].z.ToFloat32(), input[attribute].w.ToFloat32());
            }
        }
        shader_unit.LoadInput(regs.internal.vs, input);
        shader_engine->Run(vs_setup, shader_unit);
        AttributeBuffer output;
        shader_unit.WriteOutput(regs.internal.vs, output);
        const OutputVertex shaded{rasterizer, output};
        const f32 w = shaded.pos.w.ToFloat32();
        const f32 values[4] = {
            (shaded.pos.x.ToFloat32() / w + 1.f) * width / 2.f +
                static_cast<f32>(rasterizer.viewport_corner.x),
            (shaded.pos.y.ToFloat32() / w + 1.f) * height / 2.f +
                static_cast<f32>(rasterizer.viewport_corner.y),
            shaded.pos.z.ToFloat32() / w,
            w,
        };
        if (!(values[0] == values[0]) || !(values[1] == values[1]) || !(values[2] == values[2]) ||
            !(w == w)) {
            broken++;
        } else {
            behind += w <= 0.f;
            for (u32 i = 0; i < 4; i++) {
                lowest[i] = std::min(lowest[i], values[i]);
                highest[i] = std::max(highest[i], values[i]);
            }
        }
        if (index < 3) {
            first += fmt::format(" #{} in", vertex);
            for (u32 attribute = 0; attribute < 8; attribute++) {
                first +=
                    fmt::format("({:.3f},{:.3f},{:.3f},{:.3f})", input[attribute].x.ToFloat32(),
                                input[attribute].y.ToFloat32(), input[attribute].z.ToFloat32(),
                                input[attribute].w.ToFloat32());
            }
            first += fmt::format(
                " out pos({:.3f},{:.3f},{:.3f},{:.3f}) colour({:.3f},{:.3f},"
                "{:.3f},{:.3f}) tc0({:.4f},{:.4f})",
                shaded.pos.x.ToFloat32(), shaded.pos.y.ToFloat32(), shaded.pos.z.ToFloat32(), w,
                shaded.color.x.ToFloat32(), shaded.color.y.ToFloat32(), shaded.color.z.ToFloat32(),
                shaded.color.w.ToFloat32(), shaded.tc0.x.ToFloat32(), shaded.tc0.y.ToFloat32());
        }
    }
    std::string loaders;
    for (const auto& attribute_loader : pipeline.vertex_attributes.attribute_loaders) {
        if (attribute_loader.component_count != 0 && attribute_loader.byte_count != 0) {
            loaders += fmt::format(" +{:X}/{}x{}", attribute_loader.data_offset.Value(),
                                   attribute_loader.byte_count.Value(),
                                   attribute_loader.component_count.Value());
        }
    }
    // The attributes as the game declares them: b, ub, s or f and how many, or "fixed" for
    // one that has the same value for every vertex
    std::string attributes;
    for (u32 n = 0; n < pipeline.vertex_attributes.GetNumTotalAttributes(); n++) {
        if (pipeline.vertex_attributes.IsDefaultAttribute(n)) {
            attributes += fmt::format(" a{}=fixed", n);
            continue;
        }
        static constexpr std::array<const char*, 4> names{"b", "ub", "s", "f"};
        attributes += fmt::format(
            " a{}={}{}", n, names[static_cast<u32>(pipeline.vertex_attributes.GetFormat(n)) & 3],
            pipeline.vertex_attributes.GetNumElements(n));
    }
    // And the register of the shader each goes to, for as many as the shader takes
    attributes +=
        fmt::format("; the shader takes {}:", regs.internal.vs.max_input_attribute_index + 1);
    for (u32 n = 0; n <= regs.internal.vs.max_input_attribute_index; n++) {
        attributes += fmt::format(" a{}>v{}", n, regs.internal.vs.GetRegisterForAttribute(n));
    }
    VideoCore::FrameStateDump::Line(fmt::format(
        "  draw mesh, shaded here: x {:.1f} to {:.1f}, y {:.1f} to {:.1f} (pixels of the target), "
        "depth {:.4f} to {:.4f}, w {:.3f} to {:.3f}; {} of {} vertices behind the viewer, {} not "
        "numbers; vertices {} to {}; base {:08X} indices at {:08X} ({} bytes each) loaders{}; "
        "attributes{}",
        lowest[0], highest[0], lowest[1], highest[1], lowest[2], highest[2], lowest[3], highest[3],
        behind, count, broken, lowest_vertex, highest_vertex, base_address,
        base_address + index_info.offset, index_u16 ? 2 : 1, loaders, attributes));
    VideoCore::FrameStateDump::Line("  draw mesh, its first vertices:" + first);
    std::string odd_inputs = "  draw mesh, inputs that are no ordinary numbers:";
    bool any_odd = false;
    for (u32 attribute = 0; attribute < 12; attribute++) {
        if (infinite_inputs[attribute] != 0 || nan_inputs[attribute] != 0) {
            odd_inputs += fmt::format(" a{}: {} infinite, {} not numbers;", attribute,
                                      infinite_inputs[attribute], nan_inputs[attribute]);
            any_odd = true;
        }
    }
    for (u32 i = 0; i < 96; i++) {
        const auto& value = vs_setup.uniforms.f[i];
        for (u32 component = 0; component < 4; component++) {
            const f32 number = value[component].ToFloat32();
            if (number != number || std::abs(number) > 3.0e38f) {
                odd_inputs +=
                    fmt::format(" c{}=({},{},{},{});", i, value.x.ToFloat32(), value.y.ToFloat32(),
                                value.z.ToFloat32(), value.w.ToFloat32());
                any_odd = true;
                break;
            }
        }
    }
    if (!any_odd) {
        odd_inputs += " none";
    }
    odd_inputs += odd_vertex;
    odd_inputs += "; largest ordinary value of each attribute:";
    for (u32 attribute = 0; attribute < pipeline.vertex_attributes.GetNumTotalAttributes();
         attribute++) {
        odd_inputs += fmt::format(" a{} {:.4g}", attribute, largest_input[attribute]);
    }
    VideoCore::FrameStateDump::Line(std::move(odd_inputs));
    std::string uniforms = "  draw uniforms: bools";
    for (u32 i = 0; i < 16; i++) {
        uniforms += vs_setup.uniforms.b[i] ? '1' : '0';
    }
    for (u32 i = 0; i < 4; i++) {
        const auto& value = vs_setup.uniforms.i[i];
        uniforms += fmt::format(" i{}=({},{},{},{})", i, value.x, value.y, value.z, value.w);
    }
    u32 shown = 0;
    for (u32 i = 0; i < 96; i++) {
        const auto& value = vs_setup.uniforms.f[i];
        // All of the last ones, where games keep constants, and the others while there is room
        if (i < 88 && shown >= 64) {
            continue;
        }
        if (value.x.ToFloat32() != 0.f || value.y.ToFloat32() != 0.f ||
            value.z.ToFloat32() != 0.f || value.w.ToFloat32() != 0.f) {
            uniforms += fmt::format(" c{}=({:.3f},{:.3f},{:.3f},{:.3f})", i, value.x.ToFloat32(),
                                    value.y.ToFloat32(), value.z.ToFloat32(), value.w.ToFloat32());
            shown++;
        }
    }
    VideoCore::FrameStateDump::Line(std::move(uniforms));
}

void PicaCore::DumpSmallDraw(bool is_indexed) {
    const auto& pipeline = regs.internal.pipeline;
    const u32 count = pipeline.num_vertices;
    if (pipeline.use_gs != PipelineRegs::UseGS::No || count == 0) {
        return;
    }
    if (count > MAX_COVER_VERTICES) {
        DumpMeshDraw(is_indexed);
        return;
    }
    // Of a mesh only where its vertices end up
    const bool whole = count <= 6;
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GpuRead(
        base_address + index_info.offset, pipeline.num_vertices * (index_info.format != 0 ? 2 : 1));
    if (is_indexed && index_address_8 == nullptr) {
        return;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    const VertexLoader loader(memory, pipeline);
    ShaderUnit shader_unit;
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    const auto& rasterizer = regs.internal.rasterizer;
    const f32 width = f24::FromRaw(rasterizer.viewport_size_x).ToFloat32() * 2.f;
    const f32 height = f24::FromRaw(rasterizer.viewport_size_y).ToFloat32() * 2.f;
    std::string text =
        whole ? "  draw vertices (pixel in the target; texture coordinates 0, 1, 2; colour; depth):"
              : "  draw vertices (pixel in the target; depth):";
    for (u32 index = 0; index < count; ++index) {
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);
        AttributeBuffer input;
        loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);
        shader_unit.LoadInput(regs.internal.vs, input);
        shader_engine->Run(vs_setup, shader_unit);
        AttributeBuffer output;
        shader_unit.WriteOutput(regs.internal.vs, output);
        const OutputVertex shaded{rasterizer, output};
        const f32 w = shaded.pos.w.ToFloat32();
        const f32 x = (shaded.pos.x.ToFloat32() / w + 1.f) * width / 2.f +
                      static_cast<f32>(rasterizer.viewport_corner.x);
        const f32 y = (shaded.pos.y.ToFloat32() / w + 1.f) * height / 2.f +
                      static_cast<f32>(rasterizer.viewport_corner.y);
        if (!whole) {
            text += fmt::format(" [{:.2f},{:.2f}; {:.4f}]", x, y, shaded.pos.z.ToFloat32() / w);
            continue;
        }
        text += fmt::format(
            " [{:.2f},{:.2f}; {:.4f},{:.4f}; {:.4f},{:.4f}; {:.4f},{:.4f}; "
            "{:.3f},{:.3f},{:.3f},{:.3f}; {:.6f}]",
            x, y, shaded.tc0.x.ToFloat32(), shaded.tc0.y.ToFloat32(), shaded.tc1.x.ToFloat32(),
            shaded.tc1.y.ToFloat32(), shaded.tc2.x.ToFloat32(), shaded.tc2.y.ToFloat32(),
            shaded.color.x.ToFloat32(), shaded.color.y.ToFloat32(), shaded.color.z.ToFloat32(),
            shaded.color.w.ToFloat32(), shaded.pos.z.ToFloat32() / w);
    }
    VideoCore::FrameStateDump::Line(std::move(text));
    if (!whole) {
        return;
    }
    // What went into the shader: the vertices as loaded (which of them, and their first
    // attributes) and the float uniforms that are not zero. Tells a draw whose data is wrong
    // from one whose constants are.
    std::string inputs =
        fmt::format("  draw inputs: base {:08X} indices at {:08X} ({} bytes each) loaders",
                    base_address, base_address + index_info.offset, index_u16 ? 2 : 1);
    for (const auto& attribute_loader : pipeline.vertex_attributes.attribute_loaders) {
        if (attribute_loader.component_count != 0 && attribute_loader.byte_count != 0) {
            inputs += fmt::format(" +{:X}/{}", attribute_loader.data_offset.Value(),
                                  attribute_loader.byte_count.Value());
        }
    }
    inputs += " |";
    for (u32 index = 0; index < count; ++index) {
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);
        AttributeBuffer input;
        loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);
        inputs += fmt::format(" #{}", vertex);
        for (u32 attribute = 0; attribute < 3; attribute++) {
            inputs += fmt::format("({:.3f},{:.3f},{:.3f},{:.3f})", input[attribute].x.ToFloat32(),
                                  input[attribute].y.ToFloat32(), input[attribute].z.ToFloat32(),
                                  input[attribute].w.ToFloat32());
        }
    }
    VideoCore::FrameStateDump::Line(std::move(inputs));
    std::string uniforms = "  draw uniforms:";
    u32 shown = 0;
    for (u32 i = 0; i < 96 && shown < 48; i++) {
        const auto& value = vs_setup.uniforms.f[i];
        if (value.x.ToFloat32() != 0.f || value.y.ToFloat32() != 0.f ||
            value.z.ToFloat32() != 0.f || value.w.ToFloat32() != 0.f) {
            uniforms += fmt::format(" c{}=({:.3f},{:.3f},{:.3f},{:.3f})", i, value.x.ToFloat32(),
                                    value.y.ToFloat32(), value.z.ToFloat32(), value.w.ToFloat32());
            shown++;
        }
    }
    VideoCore::FrameStateDump::Line(std::move(uniforms));
}
#endif

void PicaCore::LoadVertices(bool is_indexed) {
    // Read and validate vertex information from the loaders
    const auto& pipeline = regs.internal.pipeline;
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto loader = VertexLoader(memory, pipeline);
    regs.internal.rasterizer.ValidateSemantics();

    // Locate index buffer.
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GpuRead(
        base_address + index_info.offset, pipeline.num_vertices * (index_info.format != 0 ? 2 : 1));
    if (index_address_8 == nullptr) {
        // Mario & Luigi: Superstar Saga sets an invalid base address
        // for the vertex attributes. Return early if that is the case.
        return;
    }
    const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
    const bool index_u16 = index_info.format != 0;

    // What is read here is read when the list is decoded, which may be after the game was
    // told it is done
    if (pipeline.num_vertices != 0) {
        u32 lowest = pipeline.vertex_offset;
        u32 highest = pipeline.vertex_offset + pipeline.num_vertices - 1;
        if (is_indexed) {
            memory.NoteGpuReference(base_address + index_info.offset,
                                    pipeline.num_vertices * (index_u16 ? 2 : 1),
                                    Memory::GpuReference::IndexData);
            lowest = 0xFFFF;
            highest = 0;
            for (u32 index = 0; index < pipeline.num_vertices; ++index) {
                const u32 vertex = index_u16 ? index_address_16[index] : index_address_8[index];
                lowest = std::min(lowest, vertex);
                highest = std::max(highest, vertex);
            }
        }
        for (const auto& attribute_loader : pipeline.vertex_attributes.attribute_loaders) {
            if (attribute_loader.component_count != 0 && attribute_loader.byte_count != 0) {
                memory.NoteGpuReference(base_address + attribute_loader.data_offset +
                                            lowest * attribute_loader.byte_count,
                                        (highest - lowest + 1) * attribute_loader.byte_count,
                                        Memory::GpuReference::VertexData);
            }
        }
    }

    // Simple circular-replacement vertex cache
    const std::size_t VERTEX_CACHE_SIZE = 64;
    std::array<bool, VERTEX_CACHE_SIZE> vertex_cache_valid{};
    std::array<u16, VERTEX_CACHE_SIZE> vertex_cache_ids;
    std::array<AttributeBuffer, VERTEX_CACHE_SIZE> vertex_cache;
    u32 vertex_cache_pos = 0;

    // Compile the vertex shader for this batch.
    ShaderUnit shader_unit;
    AttributeBuffer vs_output;
    // A batch of a few hundred vertices is interpreted in less time than the recompiler
    // takes for a shader it has not seen, so it does not wait for it
    constexpr u32 FEW_VERTICES = 512;
    ShaderEngine* engine = shader_engine.get();
    if (pipeline.num_vertices <= FEW_VERTICES && pipeline.use_gs == PipelineRegs::UseGS::No) {
        engine = &EngineForFewVertices();
    } else {
        shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);
    }

    // Setup geometry pipeline in case we are using a geometry shader.
    geometry_pipeline.Reconfigure();
    geometry_pipeline.Setup(shader_engine.get());
    ASSERT(!geometry_pipeline.NeedIndexInput() || is_indexed);

    for (u32 index = 0; index < pipeline.num_vertices; ++index) {
        // Indexed rendering doesn't use the start offset
        const u32 vertex = is_indexed
                               ? (index_u16 ? index_address_16[index] : index_address_8[index])
                               : (index + pipeline.vertex_offset);

        bool vertex_cache_hit = false;
        if (is_indexed) {
            if (geometry_pipeline.NeedIndexInput()) {
                geometry_pipeline.SubmitIndex(vertex);
                continue;
            }

            for (u32 i = 0; i < VERTEX_CACHE_SIZE; ++i) {
                if (vertex_cache_valid[i] && vertex == vertex_cache_ids[i]) {
                    vs_output = vertex_cache[i];
                    vertex_cache_hit = true;
                    break;
                }
            }
        }

        if (!vertex_cache_hit) {
            // Initialize data for the current vertex
            AttributeBuffer input;
            loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);

            // Record vertex processing to the debugger.
            if (debug_context) {
                debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                                       std::addressof(input));
            }

            // Invoke the vertex shader for this vertex.
            shader_unit.LoadInput(regs.internal.vs, input);
            engine->Run(vs_setup, shader_unit);
            shader_unit.WriteOutput(regs.internal.vs, vs_output);

            // Cache the vertex when doing indexed rendering.
            if (is_indexed) {
                vertex_cache[vertex_cache_pos] = vs_output;
                vertex_cache_valid[vertex_cache_pos] = true;
                vertex_cache_ids[vertex_cache_pos] = vertex;
                vertex_cache_pos = (vertex_cache_pos + 1) % VERTEX_CACHE_SIZE;
            }
        }

        // Send to geometry pipeline
        geometry_pipeline.SubmitVertex(vs_output);
    }
}

namespace {

/// The first of the two registers that start a jump, among the places of SCAN_SLOTS
constexpr s8 SCAN_JUMP = static_cast<s8>(PicaCore::ListScanState::Count);
/// And the first of the registers a draw's vertices and indices are found by, which are the
/// first of the pipeline's (ListScanState::vertex), with the register that is
constexpr s8 SCAN_VERTEX = SCAN_JUMP + 2;
constexpr u32 SCAN_VERTEX_FIRST = PICA_REG_INDEX(pipeline);
static_assert(PICA_REG_INDEX(pipeline.trigger_draw_indexed) <
              SCAN_VERTEX_FIRST + PicaCore::ListScanState::VERTEX_REGS);
static_assert(PICA_REG_INDEX(pipeline.command_buffer.size[0]) >=
              SCAN_VERTEX_FIRST + PicaCore::ListScanState::VERTEX_REGS);

/// For each register what TakeCalledLists has to do with it: -1 nothing, a place in
/// ListScanState::regs for one it follows, SCAN_JUMP and SCAN_JUMP + 1 for the two that start
/// a jump
constexpr std::array<s8, RegsInternal::NUM_REGS> SCAN_SLOTS = [] {
    using State = PicaCore::ListScanState;
    std::array<s8, RegsInternal::NUM_REGS> slots{};
    for (s8& slot : slots) {
        slot = -1;
    }
    slots[PICA_REG_INDEX(irq_request)] = State::IrqRequest;
    slots[PICA_REG_INDEX(irq_compare)] = State::IrqCompare;
    slots[PICA_REG_INDEX(irq_autostop)] = State::IrqAutostop;
    slots[PICA_REG_INDEX(pipeline.command_buffer.size[0])] = State::Size0;
    slots[PICA_REG_INDEX(pipeline.command_buffer.size[0]) + 1] = State::Size1;
    slots[PICA_REG_INDEX(pipeline.command_buffer.addr[0])] = State::Addr0;
    slots[PICA_REG_INDEX(pipeline.command_buffer.addr[0]) + 1] = State::Addr1;
    slots[PICA_REG_INDEX(pipeline.command_buffer.trigger[0])] = SCAN_JUMP;
    slots[PICA_REG_INDEX(pipeline.command_buffer.trigger[0]) + 1] = SCAN_JUMP + 1;
    for (u32 i = 0; i < State::VERTEX_REGS; i++) {
        slots[SCAN_VERTEX_FIRST + i] = static_cast<s8>(SCAN_VERTEX + i);
    }
    return slots;
}();

/// Whether any of the registers `low` to `high` is one TakeCalledLists has to do with
constexpr bool RunHasScanSlot(u32 low, u32 high) {
    const auto has = [low, high](u32 reg) { return low <= reg && reg <= high; };
    return has(PICA_REG_INDEX(irq_request)) || has(PICA_REG_INDEX(irq_compare)) ||
           has(PICA_REG_INDEX(irq_autostop)) ||
           (low <= PICA_REG_INDEX(pipeline.command_buffer.trigger[0]) + 1 &&
            high >= PICA_REG_INDEX(pipeline.command_buffer.size[0]));
}

/// And whether any of them is one a draw's vertices and indices are found by
constexpr bool RunHasVertexSlot(u32 low, u32 high) {
    return low < SCAN_VERTEX_FIRST + PicaCore::ListScanState::VERTEX_REGS &&
           high >= SCAN_VERTEX_FIRST;
}

/// A draw is about to be found in a list that is handed over: tells the memory system what
/// vertex and index data it reads, which is what LoadVertices and the rasterizers'
/// AnalyzeVertexArray read for it (or more), so that the pages of it the game keeps writing
/// are copied for the list. `words` are the pipeline's first registers as the lists leave
/// them.
void WantDrawReads(Memory::MemorySystem& memory,
                   const std::array<u32, PicaCore::ListScanState::VERTEX_REGS>& words,
                   bool indexed) {
    // Read as the pipeline's own registers are (what lies behind the first of them is not
    // looked at)
    PipelineRegs pipeline;
    std::memcpy(static_cast<void*>(&pipeline), words.data(), sizeof(words));
    const u32 count = pipeline.num_vertices;
    if (count == 0) {
        return;
    }
    const auto& attributes = pipeline.vertex_attributes;
    const PAddr base = attributes.GetPhysicalBaseAddress();
    u32 lowest = pipeline.vertex_offset;
    u32 highest = pipeline.vertex_offset + count - 1;
    if (indexed) {
        const bool index_u16 = static_cast<u32>(pipeline.index_array.format.Value()) != 0;
        const PAddr index_addr = base + pipeline.index_array.offset;
        const u32 index_bytes = count * (index_u16 ? 2 : 1);
        memory.WantCopyOf(index_addr, index_bytes);
        // Which vertices it reads is in the indices. They are looked through only if a
        // volatile page lies where the vertices of some loader can be at all.
        bool any = false;
        for (const auto& loader : attributes.attribute_loaders) {
            any |= loader.component_count != 0 && loader.byte_count != 0 &&
                   memory.HasVolatileIn(base + loader.data_offset, 0x10000 * loader.byte_count);
        }
        if (!any) {
            return;
        }
        const u8* const indices = memory.GetPhysicalPointer(index_addr);
        if (indices == nullptr) {
            // (The decoder draws nothing then)
            return;
        }
        if (memory.PhysicalBytesFrom(index_addr) < index_bytes) {
            // Indices that run out of the memory they are in: any vertex
            lowest = 0;
            highest = 0xFFFF;
        } else if (index_u16) {
            const auto [least, most] = Common::FindMinMax(
                std::span<const u16>{reinterpret_cast<const u16*>(indices), count});
            lowest = least;
            highest = most;
        } else {
            const auto [least, most] = Common::FindMinMax(std::span<const u8>{indices, count});
            lowest = least;
            highest = most;
        }
    }
    for (const auto& loader : attributes.attribute_loaders) {
        if (loader.component_count != 0 && loader.byte_count != 0) {
            memory.WantCopyOf(base + loader.data_offset + lowest * loader.byte_count,
                              (highest - lowest + 1) * loader.byte_count);
        }
    }
}

} // Anonymous namespace

PicaCore::ListScanState PicaCore::ListScanStateNow() const {
    const auto& reg = regs.internal.reg_array;
    ListScanState state;
    for (u32 id = 0; id < RegsInternal::NUM_REGS; id++) {
        if (const s8 slot = SCAN_SLOTS[id]; slot >= 0 && slot < SCAN_JUMP) {
            state.regs[slot] = reg[id];
        }
    }
    for (u32 i = 0; i < ListScanState::VERTEX_REGS; i++) {
        state.vertex[i] = reg[SCAN_VERTEX_FIRST + i];
    }
    return state;
}

bool PicaCore::TakeCalledLists(Memory::MemorySystem& memory, const u8* bytes, u32 size,
                               u32 readable, ListScanState& state, bool follow_draws,
                               std::vector<CalledListCopy>& lists, std::vector<u8>& taken) {
    // More than any game's frame has (Kirby: Planet Robobot jumps some 3000 times a frame).
    // A list that jumps to itself would go on for ever.
    constexpr std::size_t MOST_LISTS = 32768;
    constexpr std::size_t MOST_BYTES = 16 << 20;

    // The list being followed: its words, how many of them are the list and how many may be
    // read. This walks as ProcessCmdList does, down to what that does with a command whose
    // parameters reach beyond its list, so that the two come by the same jumps.
    const u32* head = reinterpret_cast<const u32*>(bytes);
    u32 length = size / sizeof(u32);
    u32 words = std::max(readable, size) / sizeof(u32);
    u32 index = 0;
    bool stop = false;
    bool all = true;
    // Of the list jumped to that the walk is in (none while it is in the list handed over):
    // where it is in memory and in `taken`, how much of it is there to take and was taken
    bool in_taken = false;
    PAddr taken_addr = 0;
    std::size_t taken_offset = 0;
    u32 taken_in_memory = 0;
    u32 taken_bytes = 0;

    const auto handle = [&](u32 id, u32 value, u32 mask) {
        const s8 slot = SCAN_SLOTS[id];
        if (slot >= SCAN_VERTEX) {
            // Where a draw's vertices and indices are, and the draw itself
            const u32 write_mask = ExpandBitsToBytes[mask];
            u32& word = state.vertex[static_cast<u32>(slot - SCAN_VERTEX)];
            word = (word & ~write_mask) | (value & write_mask);
            if (id == PICA_REG_INDEX(pipeline.trigger_draw) ||
                id == PICA_REG_INDEX(pipeline.trigger_draw_indexed)) {
                state.draws_followed++;
                WantDrawReads(memory, state.vertex,
                              id == PICA_REG_INDEX(pipeline.trigger_draw_indexed));
            }
            return;
        }
        if (slot < SCAN_JUMP) {
            const u32 write_mask = ExpandBitsToBytes[mask];
            state.regs[slot] = (state.regs[slot] & ~write_mask) | (value & write_mask);
            // The interrupt a list ends with: the decoder stops there
            stop = static_cast<u32>(slot) == ListScanState::IrqRequest &&
                   any_byte_match(state.regs[ListScanState::IrqRequest],
                                  state.regs[ListScanState::IrqCompare]) &&
                   state.regs[ListScanState::IrqAutostop] != 0;
            return;
        }
        // A jump: the list goes on in another one
        const u32 which = static_cast<u32>(slot - SCAN_JUMP);
        const PAddr addr = 8 * (state.regs[ListScanState::Addr0 + which] & 0x0FFFFFFF);
        const u32 list_size = 8 * (state.regs[ListScanState::Size0 + which] & 0x000FFFFF);
        const u32 in_memory = memory.PhysicalBytesFrom(addr);
        if (list_size == 0 || in_memory < list_size) {
            // Nothing there to go on with: the decoder ends the list here as well
            stop = true;
            return;
        }
        if (lists.size() >= MOST_LISTS || taken.size() + list_size > MOST_BYTES) {
            all = false;
            stop = true;
            return;
        }
        // The list and the two words after it, which the decoder reads where a list has an
        // odd number of words. More of what follows only once a command reaches there.
        const u32 take = std::min(in_memory, list_size + 8);
        const std::size_t offset = (taken.size() + 7) & ~std::size_t{7};
        taken.resize(offset);
        const u8* const from = memory.GetPhysicalPointer(addr);
        taken.insert(taken.end(), from, from + take);
        lists.push_back(CalledListCopy{addr, list_size, static_cast<u32>(offset)});
        in_taken = true;
        taken_addr = addr;
        taken_offset = offset;
        taken_in_memory = in_memory;
        taken_bytes = take;
        // (`taken` may have moved: the walk is in it only from here to the next jump)
        head = reinterpret_cast<const u32*>(taken.data() + offset);
        length = list_size / sizeof(u32);
        words = take / sizeof(u32);
        index = 0;
    };

    while (index < length && !stop) {
        if (index % 2 != 0) {
            index++;
        }
        if (index + 1 >= words) {
            break;
        }
        const u32 value = head[index];
        const CommandHeader header{head[index + 1]};
        index += 2;
        const u32 id = header.cmd_id;
        const u32 count = header.extra_data_length;
        const s8 own_slot = id < RegsInternal::NUM_REGS ? SCAN_SLOTS[id] : s8{-1};
        const bool followed = own_slot >= 0 && (own_slot < SCAN_VERTEX || follow_draws);
        if (followed) [[unlikely]] {
            handle(id, value, header.parameter_mask);
            if (stop) {
                break;
            }
        }
        if (count == 0) {
            continue;
        }
        // The parameters that follow, read where the list is now: after a jump that is the
        // beginning of the list jumped to
        const u32 first = index;
        index += count;
        if (first + count > words && in_taken) [[unlikely]] {
            // It says it has more parameters than its list has room for. They are read from
            // what follows the list in memory, which is taken along now: the list is the last
            // in `taken`, so it is added on.
            const u32 want = std::min<u64>(taken_in_memory, u64{first + count} * sizeof(u32) + 8);
            if (want > taken_bytes) {
                const u8* const from = memory.GetPhysicalPointer(taken_addr);
                taken.insert(taken.end(), from + taken_bytes, from + want);
                taken_bytes = want;
                head = reinterpret_cast<const u32*>(taken.data() + taken_offset);
                words = want / sizeof(u32);
            }
        }
        const u32 have = first < words ? std::min(count, words - first) : 0;
        if (have == 0) {
            continue;
        }
        if (!header.group_commands) {
            // The same register again and again: it keeps the last, and what writing it sets
            // off comes to the same however often
            if (followed) [[unlikely]] {
                handle(id, head[first + have - 1], header.parameter_mask);
            }
            continue;
        }
        // The registers after it, one after the other. Those followed are picked out first:
        // a jump among them moves the walk, and the rest are still the old list's.
        if (!RunHasScanSlot(id + 1, id + have) &&
            !(follow_draws && RunHasVertexSlot(id + 1, id + have))) [[likely]] {
            continue;
        }
        // (No more of them than there are places in SCAN_SLOTS)
        std::array<std::pair<u32, u32>, 64> found;
        u32 found_count = 0;
        for (u32 i = 0; i < have && id + 1 + i < RegsInternal::NUM_REGS; i++) {
            const s8 slot = SCAN_SLOTS[id + 1 + i];
            if (slot >= 0 && (slot < SCAN_VERTEX || follow_draws) && found_count < found.size()) {
                found[found_count++] = {id + 1 + i, head[first + i]};
            }
        }
        for (u32 i = 0; i < found_count && !stop; i++) {
            handle(found[i].first, found[i].second, header.parameter_mask);
        }
    }
    // Whatever the decoder reads behind the last list is in the buffer: also where a list
    // ends with the memory it is in, and nothing could be taken of what follows
    if (!taken.empty()) {
        taken.resize(taken.size() + LIST_SLACK);
    }
    return all;
}

const u8* PicaCore::TakenCalledList(PAddr addr, u32 size) {
    if (called_lists == nullptr) {
        return nullptr;
    }
    // The copies are in the order of the jumps. (The one before once more: a jump that is
    // written several times over.)
    const auto is = [&](std::size_t at) {
        return at < called_lists->size() && (*called_lists)[at].addr == addr &&
               (*called_lists)[at].size == size;
    };
    const u8* copy = nullptr;
    if (is(called_next)) {
        copy = called_bytes + (*called_lists)[called_next].offset;
        called_next++;
    } else if (called_next != 0 && is(called_next - 1)) {
        copy = called_bytes + (*called_lists)[called_next - 1].offset;
    }
    if (copy == nullptr) {
        (called_next >= called_lists->size() ? called_past_copies : called_another)
            .fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    called_from_copy.fetch_add(1, std::memory_order_relaxed);
    // For the log: whether the game has written to the list since it handed it over
    if (const u8* const now = memory.GetPhysicalPointer(addr);
        now != nullptr && std::memcmp(copy, now, size) != 0) {
        called_rewritten.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> told{};
        if (told.fetch_add(1, std::memory_order_relaxed) < 6) {
            LOG_INFO(HW_GPU,
                     "A list a command list jumps to ({:08X}, {} bytes) was written to by the "
                     "game between its hand-over and its decoding; decoded as it was at "
                     "hand-over",
                     addr, size);
        }
    }
    return copy;
}

PicaCore::RenderPropertiesGuess PicaCore::GuessCmdRenderProperties(PAddr list, u32 size) {
    constexpr size_t max_iterations = 0x100;

    RenderPropertiesGuess find_info{};

    find_info.vp_height = regs.internal.rasterizer.viewport_size_y.Value();
    find_info.paddr = regs.internal.framebuffer.framebuffer.color_buffer_address.Value() * 8;

    // Initialize command list tracking.
    const u8* head = memory.GetPhysicalPointer(list);
    if (head == nullptr) {
        return find_info;
    }
    cmd_list.Reset(list, head, size);

    auto process_write = [this, &find_info](u32 cmd_id, u32 value) {
        switch (cmd_id) {
        case PICA_REG_INDEX(rasterizer.viewport_size_y):
            find_info.vp_height = value;
            find_info.vp_heigh_found = true;
            break;
        case PICA_REG_INDEX(framebuffer.framebuffer.color_buffer_address):
            find_info.paddr = value * 8;
            find_info.paddr_found = true;
            break;
        case PICA_REG_INDEX(pipeline.trigger_draw):
        case PICA_REG_INDEX(pipeline.trigger_draw_indexed):
            find_info.has_draw = true;
            break;
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[0]):
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[1]): {
            const u32 index =
                static_cast<u32>(cmd_id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
            const PAddr addr = regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
            const u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
            const u8* head = memory.GetPhysicalPointer(addr);
            if (head != nullptr) {
                cmd_list.Reset(addr, head, size);
            }
            break;
        }
        default:
            break;
        }
        return find_info.vp_heigh_found && find_info.paddr_found && find_info.has_draw;
    };

    size_t iterations = 0;
    while (cmd_list.current_index < cmd_list.length && iterations < max_iterations) {
        // Align read pointer to 8 bytes
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        // Read the header and the value to write.
        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        // Write to the requested PICA register.
        if (process_write(header.cmd_id, value))
            break;

        // Write any extra paramters as well.
        for (u32 i = 0; i < header.extra_data_length; ++i) {
            const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
            const u32 extra_value = cmd_list.head[cmd_list.current_index++];
            if (process_write(cmd, extra_value))
                break;
        }

        iterations++;
    }

    return find_info;
}

template <class Archive>
void PicaCore::CommandList::serialize(Archive& ar, const u32 file_version) {
    ar & addr;
    ar & length;
    ar & current_index;
    if (Archive::is_loading::value) {
        const u8* ptr = Core::System::GetInstance().Memory().GetPhysicalPointer(addr);
        head = reinterpret_cast<const u32*>(ptr);
    }
}

SERIALIZE_IMPL(PicaCore::CommandList)

} // namespace Pica
