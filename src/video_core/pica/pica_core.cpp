// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/arch.h"
#include "common/archives.h"
#include "common/hash.h"
#include "common/hitch_tracker.h"
#include "common/microprofile.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/frame_liveness.h"
#include "video_core/frame_state_dump.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/vertex_loader.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/shader/shader.h"

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

void PicaCore::NoteTargetCommand() {
    TwinPass& twin = twin_pass;
    if (twin.phase == TwinPass::Phase::Copied || twin.phase == TwinPass::Phase::Second) {
        // Not the pattern: nothing but draws lies between the copy and the blending
        if (twin.frames_alike >= TwinPass::FRAMES_ALIKE) {
            LOG_INFO(HW_GPU,
                     "The scene drawn twice at {:08X} is no longer: a fill or a transfer "
                     "came within its second pass ({} draws of it were left out)",
                     twin.target, twin.left_out_now);
        }
        twin.frames_alike = 0;
    }
    twin.phase = TwinPass::Phase::Idle;
    twin.run_target = 0;
    twin.run_draws = 0;
}

template <typename Func>
void PicaCore::ForTwinRegisters(bool clears, Func&& func) const {
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
    func(0x0C0, 0x100);
    // Blending, the tests and the masks; the target
    func(0x100, 0x108);
    func(0x116, 0x11F);
    if (clears) {
        // A clear done with a quad: which program of the vertex shader's memory it runs does
        // not matter, and what else that memory holds changes between the two
        return;
    }
    // The lights; where the vertices and indices are and how many; the stages after the
    // vertex shader; the vertex shader's configuration
    if (!regs.internal.lighting.disable) {
        func(0x140, 0x1C5);
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

u64 PicaCore::TwinState(bool clears) {
    const auto& array = regs.internal.reg_array;
    u64 state = clears ? 1 : 2;
    ForTwinRegisters(clears, [&](u32 from, u32 to) {
        state = Common::HashCombine(state,
                                    Common::ComputeHash64(&array[from], (to - from) * sizeof(u32)));
    });
    if (clears) {
        return state;
    }
    state = Common::HashCombine(state, vs_setup.GetProgramCodeHash());
    state = Common::HashCombine(state, vs_setup.GetSwizzleDataHash());
    if (regs.internal.pipeline.use_gs != PipelineRegs::UseGS::No) {
        state = Common::HashCombine(state, gs_setup.GetProgramCodeHash());
    }
    return state;
}

const PicaCore::UniformSet& PicaCore::FloatUniformsRead() {
    const u32 main = regs.internal.vs.main_offset;
    const u64 key = Common::HashCombine(vs_setup.GetProgramCodeHash(), u64{main});
    for (const auto& [known, set] : uniforms_read) {
        if (known == key) {
            return set;
        }
    }
    const auto& code = vs_setup.GetProgramCode();
    UniformSet set{};
    std::vector<bool> seen(code.size());
    // Stretches of the program still to go through: from, up to
    std::vector<std::pair<u32, u32>> stretches{{main, static_cast<u32>(code.size())}};
    while (!stretches.empty()) {
        auto [at, end] = stretches.back();
        stretches.pop_back();
        for (; at < end && at < code.size(); at++) {
            const u32 word = code[at];
            const u32 opcode = word >> 26;
            if (opcode == 0x22) {
                // END
                break;
            }
            if (seen[at]) {
                continue;
            }
            seen[at] = true;
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
            } else if (opcode >= 0x20 && opcode <= 0x2D) {
                // Flow: what a call or a jump leads to is gone through as well. What an IF or
                // a LOOP covers follows it and is gone through anyway.
                const u32 dest = (word >> 10) & 0xFFF;
                const u32 count = word & 0xFF;
                if (opcode >= 0x24 && opcode <= 0x26) {
                    stretches.emplace_back(dest, dest + count);
                } else if (opcode == 0x2C || opcode == 0x2D) {
                    stretches.emplace_back(dest, static_cast<u32>(code.size()));
                }
                continue;
            } else if (opcode >= 0x18 && opcode <= 0x1B) {
                // The instructions with the operands the other way round
                reg = (word >> 7) & 0x7F;
                indexed = (word >> 19) & 3;
            } else {
                reg = (word >> 12) & 0x7F;
                indexed = (word >> 19) & 3;
            }
            // An indexed read picks from a table (the bones of a skinned model). How far into
            // it is not known here, and what lies beyond a model's own bones is what other
            // models left there: tables are not compared.
            if (reg < 0x20 || indexed != 0) {
                continue;
            }
            const u32 i = reg - 0x20;
            set[i / 64] |= u64{1} << (i % 64);
        }
    }
    if (uniforms_read.size() >= 256) {
        uniforms_read.clear();
    }
    return uniforms_read.emplace_back(key, set).second;
}

bool PicaCore::IsTwinOf(const TwinPass::Draw& first, bool clears, u64 state) {
    if (first.clear != clears || first.state != state) {
        return false;
    }
    if (clears) {
        return true;
    }
    const auto& now = vs_setup.uniforms.f;
    if (std::memcmp(now.data(), first.uniforms.data(), sizeof(first.uniforms)) == 0) {
        return true;
    }
    // Only those the program reads: the others hold what earlier draws left there
    const UniformSet& read = FloatUniformsRead();
    u32 differing = 0;
    for (u32 i = 0; i < now.size(); i++) {
        if ((read[i / 64] >> (i % 64) & 1) == 0 ||
            std::memcmp(&now[i], &first.uniforms[i], sizeof(now[i])) == 0) {
            continue;
        }
        if (++differing > TwinPass::MAX_DIFFERING) {
            return false;
        }
        // Against the size of what is around it: the rows of the matrix it is one of
        float scale = 1e-4f;
        for (u32 j = i >= 3 ? i - 3 : 0; j < std::min<u32>(i + 4, 96); j++) {
            const auto& row = first.uniforms[j];
            scale = std::max({scale, std::abs(row.x.ToFloat32()), std::abs(row.y.ToFloat32()),
                              std::abs(row.z.ToFloat32()), std::abs(row.w.ToFloat32())});
        }
        const float shift =
            std::max({std::abs(now[i].x.ToFloat32() - first.uniforms[i].x.ToFloat32()),
                      std::abs(now[i].y.ToFloat32() - first.uniforms[i].y.ToFloat32()),
                      std::abs(now[i].z.ToFloat32() - first.uniforms[i].z.ToFloat32()),
                      std::abs(now[i].w.ToFloat32() - first.uniforms[i].w.ToFloat32())}) /
            scale;
        // (also false for a value that is not a number)
        if (!(shift <= TwinPass::MAX_SHIFT)) {
            return false;
        }
        twin_pass.largest_shift = std::max(twin_pass.largest_shift, shift);
    }
    twin_pass.most_differing = std::max(twin_pass.most_differing, differing);
    return true;
}

void PicaCore::ExplainNoTwin(bool clears) {
    TwinPass& twin = twin_pass;
    if (twin.explained >= 12) {
        return;
    }
    twin.explained++;
    std::string why;
    if (twin.at >= twin.first.size()) {
        why = fmt::format("the first pass has only {} draws on record", twin.first.size());
    } else {
        const TwinPass::Draw& first = twin.first[twin.at];
        if (first.clear != clears) {
            why = fmt::format("it {} all of its viewport, its counterpart {}",
                              clears ? "overwrites" : "does not overwrite",
                              first.clear ? "does" : "does not");
        } else if (first.state != TwinState(clears)) {
            why = "registers differ:";
            u32 listed = 0;
            if (twin.at < twin.first_regs.size()) {
                const auto& before = twin.first_regs[twin.at];
                const auto& now = regs.internal.reg_array;
                ForTwinRegisters(clears, [&](u32 from, u32 to) {
                    for (u32 i = from; i < to && listed < 24; i++) {
                        if (before[i] != now[i]) {
                            why += fmt::format(" {:03X}: {:08X}->{:08X}", i, before[i], now[i]);
                            listed++;
                        }
                    }
                });
            }
            if (listed == 0) {
                why += " none of those that are compared now, so the vertex shader's program "
                       "or which registers are in use";
            }
        } else {
            const auto& now = vs_setup.uniforms.f;
            const UniformSet& read = FloatUniformsRead();
            u32 differing = 0;
            float largest = 0.f;
            std::string which;
            for (u32 i = 0; i < now.size(); i++) {
                if ((read[i / 64] >> (i % 64) & 1) == 0 ||
                    std::memcmp(&now[i], &first.uniforms[i], sizeof(now[i])) == 0) {
                    continue;
                }
                differing++;
                float scale = 1e-4f;
                for (u32 j = i >= 3 ? i - 3 : 0; j < std::min<u32>(i + 4, 96); j++) {
                    const auto& row = first.uniforms[j];
                    scale =
                        std::max({scale, std::abs(row.x.ToFloat32()), std::abs(row.y.ToFloat32()),
                                  std::abs(row.z.ToFloat32()), std::abs(row.w.ToFloat32())});
                }
                const float shift =
                    std::max({std::abs(now[i].x.ToFloat32() - first.uniforms[i].x.ToFloat32()),
                              std::abs(now[i].y.ToFloat32() - first.uniforms[i].y.ToFloat32()),
                              std::abs(now[i].z.ToFloat32() - first.uniforms[i].z.ToFloat32()),
                              std::abs(now[i].w.ToFloat32() - first.uniforms[i].w.ToFloat32())}) /
                    scale;
                largest = std::max(largest, shift);
                if (differing <= 6) {
                    which += fmt::format(
                        " c{} ({:.4f} {:.4f} {:.4f} {:.4f} -> {:.4f} {:.4f} {:.4f} {:.4f})", i,
                        first.uniforms[i].x.ToFloat32(), first.uniforms[i].y.ToFloat32(),
                        first.uniforms[i].z.ToFloat32(), first.uniforms[i].w.ToFloat32(),
                        now[i].x.ToFloat32(), now[i].y.ToFloat32(), now[i].z.ToFloat32(),
                        now[i].w.ToFloat32());
                }
            }
            why = fmt::format("of the {} float uniforms its shader reads {} differ, by {:.4f} "
                              "of what is around them at the most:{}",
                              std::popcount(read[0]) + std::popcount(read[1]), differing, largest,
                              which);
        }
    }
    LOG_INFO(HW_GPU, "Second pass at {:08X}: draw {} of {} ({} vertices) is no twin: {}",
             twin.target, twin.at, twin.first_draws, regs.internal.pipeline.num_vertices, why);
}

bool PicaCore::IsLeftOutTwin(bool clears) {
    using Phase = TwinPass::Phase;
    using BlendFactor = FramebufferRegs::BlendFactor;
    TwinPass& twin = twin_pass;
    const PAddr color = regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress();
    const auto& pipeline = regs.internal.pipeline;
    const auto& texturing = regs.internal.texturing;
    const auto& merger = regs.internal.framebuffer.output_merger;
    // A quad that samples one texture: what copies an image, or blends one over another
    const bool quad = pipeline.num_vertices == 4 && pipeline.use_gs == PipelineRegs::UseGS::No;
    const PAddr sampled =
        quad && texturing.main_config.texture0_enable ? texturing.texture0.GetPhysicalAddress() : 0;
    const bool replaces =
        !merger.alphablend_enable || (merger.alpha_blending.factor_source_rgb == BlendFactor::One &&
                                      merger.alpha_blending.factor_dest_rgb == BlendFactor::Zero);

    if (twin.phase == Phase::Copied) {
        if (color == twin.target) {
            twin.phase = Phase::Second;
            twin.at = 0;
            twin.left_out_now = 0;
            twin.twins = twin.compared;
            twin.leaving_out = twin.compared && twin.frames_alike >= TwinPass::FRAMES_ALIKE &&
                               Settings::values.resolution_factor.GetValue() != 1 &&
                               (Common::TraceMarkers::Experiments3() &
                                Common::TraceMarkers::KeepSecondScenePass) == 0;
        } else {
            twin.phase = Phase::Idle;
        }
    }

    if (twin.phase == Phase::Second) {
        if (color == twin.target && quad && sampled == twin.keep && sampled != 0 && !replaces) {
            // The quad that blends the first image over the second
            const bool whole = twin.at == twin.first_draws;
            const u32 apart = twin.at > twin.first_draws ? twin.at - twin.first_draws
                                                         : twin.first_draws - twin.at;
            if (apart <= twin.first_draws / 8) {
                // The bare pattern: worth comparing draw by draw from the next frame on
                if (twin.watched_target != twin.target) {
                    LOG_INFO(HW_GPU,
                             "A scene that may be drawn twice: runs of {} and {} draws to {:08X} "
                             "with a copy to {:08X} between them and a blend of it after",
                             twin.first_draws, twin.at, twin.target, twin.keep);
                }
                twin.watched_target = twin.target;
            }
            if (twin.compared) {
                const bool alike = twin.twins && whole;
                if (alike && ++twin.frames_alike == TwinPass::FRAMES_ALIKE) {
                    LOG_INFO(HW_GPU,
                             "The game draws its scene twice at {:08X} and blends the two images "
                             "(kept at {:08X}): {} draws each, every one of the second pass like "
                             "the one at its place in the first but for {} float uniforms at "
                             "the most, off by {:.4f} of what is around them at the most. The "
                             "second pass is left out from here on where the resolution is "
                             "above the console's.",
                             twin.target, twin.keep, twin.first_draws, twin.most_differing,
                             twin.largest_shift);
                } else if (!alike) {
                    if (twin.frames_alike >= TwinPass::FRAMES_ALIKE) {
                        LOG_INFO(HW_GPU,
                                 "The scene drawn twice at {:08X}: the second pass was not like "
                                 "the first this time ({} draws against {}, {} left out)",
                                 twin.target, twin.at, twin.first_draws, twin.left_out_now);
                    }
                    twin.frames_alike = 0;
                }
                twin.frames_alike = std::min(twin.frames_alike, 1u << 20);
            }
            const bool leave_out = twin.leaving_out && whole && twin.left_out_now == twin.at;
            twin.phase = Phase::Idle;
            twin.run_target = color;
            twin.run_draws = 1;
            if (leave_out) {
                twin.left_out.fetch_add(1, std::memory_order_relaxed);
            }
            return leave_out;
        }
        if (color == twin.target) {
            // A draw of the second pass
            bool is_twin = false;
            if (twin.twins && twin.at < twin.first.size()) {
                is_twin = IsTwinOf(twin.first[twin.at], clears, TwinState(clears));
            }
            if (!is_twin) {
                if (twin.twins && (Common::TraceMarkers::Experiments3() &
                                   Common::TraceMarkers::ExplainScenePasses) != 0) {
                    ExplainNoTwin(clears);
                }
                twin.twins = false;
                twin.leaving_out = false;
            }
            twin.at++;
            if (is_twin && twin.leaving_out) {
                twin.left_out_now++;
                twin.left_out.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            return false;
        }
        // The second run ended without the blending: not the pattern
        if (twin.frames_alike >= TwinPass::FRAMES_ALIKE) {
            LOG_INFO(HW_GPU,
                     "The scene drawn twice at {:08X} is no longer: its second pass "
                     "ended without the blending ({} draws of it were left out)",
                     twin.target, twin.left_out_now);
        }
        twin.frames_alike = 0;
        twin.phase = Phase::Idle;
        twin.run_target = 0;
    }

    if (color != twin.run_target) {
        if (twin.phase == Phase::First && quad && sampled == twin.run_target && sampled != 0 &&
            replaces && twin.run_draws >= TwinPass::MIN_DRAWS) {
            // The quad that copies what the run drew to somewhere else
            twin.target = twin.run_target;
            twin.keep = color;
            twin.first_draws = twin.run_draws;
            twin.compared = twin.first.size() == twin.run_draws;
            twin.phase = Phase::Copied;
            twin.run_target = color;
            twin.run_draws = 1;
            return false;
        }
        twin.phase = Phase::First;
        twin.run_target = color;
        twin.run_draws = 0;
        twin.first.clear();
        twin.first_regs.clear();
    }
    if (twin.phase == Phase::First) {
        if (color == twin.watched_target && twin.first.size() == twin.run_draws &&
            twin.run_draws < TwinPass::MAX_DRAWS) {
            TwinPass::Draw& draw = twin.first.emplace_back();
            draw.state = TwinState(clears);
            draw.clear = clears;
            if (!clears) {
                std::memcpy(draw.uniforms.data(), vs_setup.uniforms.f.data(),
                            sizeof(draw.uniforms));
            }
            if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::ExplainScenePasses) !=
                0) {
                twin.first_regs.push_back(regs.internal.reg_array);
            }
        }
        twin.run_draws++;
    }
    return false;
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
        const u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
        const u8* head = memory.GpuRead(addr, size);
        memory.NoteGpuReference(addr, size, Memory::GpuReference::CalledList);
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
    if (draw_filter) {
        draw_filter->ShouldAllowDraw(regs.internal, false, false);
    }
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
};

Overwrites MayOverwriteArea(const RegsInternal& regs) {
    using Func = FramebufferRegs::CompareFunc;
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& merger = regs.framebuffer.output_merger;
    const auto& rasterizer = regs.rasterizer;

    // Two triangles at the most, and no fragment of them is dropped. Whether they face the way
    // that is drawn is looked at with their vertices, in CoversViewport.
    if (regs.pipeline.num_vertices > 6 || merger.alpha_test.enable || rasterizer.clip_enable ||
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
    return result;
}

} // Anonymous namespace

void PicaCore::DrawArrays(bool is_indexed) {
    MICROPROFILE_SCOPE(GPU_Drawing);
    CITRA_TRACE_SCOPE_DETAIL("PICA::DrawArrays");
    if ((Common::TraceMarkers::Experiments2() & Common::TraceMarkers::MeasureCommandLists) != 0)
        [[unlikely]] {
        MeasureDrawSite(cmd_list.addr, cmd_list.current_index, regs.internal.reg_array);
    }

    // Whether the triangles of a draw cover the viewport is of use twice over: a draw that
    // replaces all its target holds ends the life of what was drawn there before, and one that
    // overwrites the area it is confined to lets the renderer begin without what that held.
    const bool may_replace_target = draw_filter && draw_filter->MayReplaceTarget(regs.internal);
    const Overwrites may_overwrite = MayOverwriteArea(regs.internal);
    bool covers_viewport = false;
    bool covers_exactly = false;
    if (may_replace_target || may_overwrite.color || may_overwrite.depth) {
        covers_viewport = CoversViewport(is_indexed, &covers_exactly);
    }

    // A draw that only repeats one already made, a part of a pixel apart, is left out
    if (IsLeftOutTwin(covers_viewport && may_overwrite.color)) {
        CITRA_FRAME_DUMP("draw left out: the twin of a draw of the scene's first pass");
        return;
    }
    // The quad that keeps the first image of a scene drawn twice is always drawn. While the
    // second pass and the blending are left out nothing reads that image, and it would be
    // taken for work nobody sees; the frame in which the second pass is not like the first
    // after all blends it in, and must not get an old one.
    const bool keeps_first_image =
        twin_pass.phase == TwinPass::Phase::Copied && twin_pass.target == twin_pass.watched_target;

    // A draw whose result nobody will see is left out. The registers were written as usual.
    if (draw_filter) {
        VideoCore::DrawTextureRows texture_rows;
        const bool rows_known = draw_filter->WantsTextureRows(regs.internal) &&
                                TextureRowsOfDraw(is_indexed, texture_rows);
        if (!draw_filter->ShouldAllowDraw(regs.internal, may_replace_target && covers_viewport,
                                          !keeps_first_image,
                                          rows_known ? &texture_rows : nullptr)) {
            CITRA_FRAME_DUMP("draw left out: nobody sees its result");
            return;
        }
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

        bool accelerate_draw = Settings::values.use_hw_shader && primitive_assembler.IsEmpty();
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
    const bool overwrites = covers_viewport && covers_exactly;
    const bool tells_overwrite = overwrites && (may_overwrite.color || may_overwrite.depth);
    // Draws shaded on the CPU that are waiting to be drawn together go first, before a draw
    // that is not added to them: one the host shades, or one the rasterizer is told about
    if (sw_batch_open && (accelerate_draw || tells_overwrite)) {
        FlushSoftwareBatch();
    }
    rasterizer->NoteDrawOverwrites(overwrites && may_overwrite.color,
                                   overwrites && may_overwrite.depth);

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
    const auto& pipeline = regs.internal.pipeline;
    const u32 count = pipeline.num_vertices;
    if (pipeline.use_gs != PipelineRegs::UseGS::No || !primitive_assembler.IsEmpty() ||
        count == 0 || count > 6) {
        return 1.f;
    }

    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto& index_info = pipeline.index_array;
    const u8* index_address_8 = memory.GpuRead(
        base_address + index_info.offset, pipeline.num_vertices * (index_info.format != 0 ? 2 : 1));
    if (is_indexed && index_address_8 == nullptr) {
        return 1.f;
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
        const OutputVertex shaded{regs.internal.rasterizer, output};

        // A triangle with a vertex behind the eye can reach anywhere
        const f32 w = shaded.pos.w.ToFloat32();
        const f32 x = shaded.pos.x.ToFloat32() / w;
        const f32 y = shaded.pos.y.ToFloat32() / w;
        if (!(w > 0.f) || !std::isfinite(x) || !std::isfinite(y)) {
            return 1.f;
        }
        left = std::min(left, std::clamp(x, -1.f, 1.f));
        right = std::max(right, std::clamp(x, -1.f, 1.f));
        bottom = std::min(bottom, std::clamp(y, -1.f, 1.f));
        top = std::max(top, std::clamp(y, -1.f, 1.f));
    }
    if (!(right > left && top > bottom)) {
        return 0.f;
    }
    return (right - left) * (top - bottom) / 4.f;
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
        const OutputVertex shaded{regs.internal.rasterizer, output};

        const f32 w = shaded.pos.w.ToFloat32();
        if (!(w > 0.f)) {
            return false;
        }
        // A vertex in front of or behind the depth range has its triangle cut
        const f32 z = shaded.pos.z.ToFloat32() / w;
        if (!(z <= DEPTH_TOLERANCE && z >= -1.f - DEPTH_TOLERANCE)) {
            return false;
        }
        const f32 x = shaded.pos.x.ToFloat32() / w;
        const f32 y = shaded.pos.y.ToFloat32() / w;
        const Common::Vec2<f32> offset{(std::abs(x) - 1.f) * width / 2.f,
                                       (std::abs(y) - 1.f) * height / 2.f};
        if (!(offset.x >= -MAX_INSIDE && offset.x <= MAX_OUTSIDE && offset.y >= -MAX_INSIDE &&
              offset.y <= MAX_OUTSIDE)) {
            CITRA_FRAME_DUMP("  draw: vertex {} at {},{} is {},{} pixels off a corner of the "
                             "viewport",
                             index, x, y, offset.x, offset.y);
            return false;
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
        return false;
    }
    // No gap along that diagonal: the two triangles end in the same places
    for (const u32 a : first) {
        for (const u32 b : second) {
            if (corners[a] == corners[b] && offsets[a] != offsets[b]) {
                return false;
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

void PicaCore::DumpSmallDraw(bool is_indexed) {
    const auto& pipeline = regs.internal.pipeline;
    const u32 count = pipeline.num_vertices;
    if (pipeline.use_gs != PipelineRegs::UseGS::No || count == 0 || count > 6) {
        return;
    }
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
        "  draw vertices (pixel in the target; texture coordinates 0, 1, 2; colour; depth):";
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
        text += fmt::format(
            " [{:.2f},{:.2f}; {:.4f},{:.4f}; {:.4f},{:.4f}; {:.4f},{:.4f}; "
            "{:.3f},{:.3f},{:.3f},{:.3f}; {:.6f}]",
            x, y, shaded.tc0.x.ToFloat32(), shaded.tc0.y.ToFloat32(), shaded.tc1.x.ToFloat32(),
            shaded.tc1.y.ToFloat32(), shaded.tc2.x.ToFloat32(), shaded.tc2.y.ToFloat32(),
            shaded.color.x.ToFloat32(), shaded.color.y.ToFloat32(), shaded.color.z.ToFloat32(),
            shaded.color.w.ToFloat32(), shaded.pos.z.ToFloat32() / w);
    }
    VideoCore::FrameStateDump::Line(std::move(text));
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
