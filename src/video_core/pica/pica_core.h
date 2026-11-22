// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <utility>
#include <vector>
#include <boost/container/small_vector.hpp>
#include "common/common_types.h"
#include "common/trace_markers.h"
#include "core/hle/service/gsp/gsp_interrupt.h"
#include "video_core/pica/dirty_regs.h"
#include "video_core/pica/geometry_pipeline.h"
#include "video_core/pica/packed_attribute.h"
#include "video_core/pica/primitive_assembly.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/pica/shader_unit.h"

namespace Memory {
class MemorySystem;
}

namespace VideoCore {
struct DrawTextureRows;
class RasterizerInterface;
class WorkRules;
} // namespace VideoCore

namespace Pica {

class DebugContext;
class ShaderEngine;

class DelayGenerator {
private:
    // A GPU is a very complex system, the timings resulting from
    // a 3D draw depend on many factors, including triangle counts,
    // texture sizes and format, shader complexity, cache
    // and memory layout, etc. At this point in time, we don't
    // have enough information nor implemented hw emulation
    // capabilities to achieve a proper timing estimate.
    //
    // Instead, we will try to measure how complex a scene is based
    // on the amount of geometry that is drawn, the amount of GPU
    // commands and the shader complexity. We will ignore all
    // the other factors for now.

    // Using Mario Kart 7 as the reference, it is understood that on
    // average the console can handle around 20k triangles per frame.
    // This game uses standard GPU features, with no fancy stuff,
    // so we can consider it an average. To prevent hurting performance,
    // we will also assume the GPU is twice as powerful. Afterall we only
    // want timing accuracy to fix bugs at this point.
    // This average already takes into account shader complexity averages.
    static constexpr float nanoseconds_per_triangle = 800.f / 2;

    // Of the total amount of submitted triangles, many of them will be culled.
    // This heavily depends on the specific scene, so we will assume 35% of the
    // triangles being culled. Furthermore, the culled triangles will take way less
    // processing time as they will skip most of the pipeline processing, so we
    // can assume that a culled triangle will only take about 20% of the time.
    static constexpr float culled_triangle_threshold = 0.35f; // 35%
    static constexpr float culled_triangle_time_cost = 0.20f; // 20%

    // We will assume that each command will take around 6 cycles @ 268MHz
    // There are no real measurements to support this claim, but it sounds
    // reasonable. TODO: Measure on real HW.
    static constexpr float nanoseconds_per_command = 22.4f;

public:
    inline void AddCommands(size_t commands) {
        command_count += commands;
    }

    inline void AddVertices(size_t vertices, PipelineRegs::TriangleTopology topology) {
        size_t triangles{};
        if (topology == PipelineRegs::TriangleTopology::Fan ||
            topology == PipelineRegs::TriangleTopology::Strip) {
            triangles = (vertices >= 3) ? (vertices - 2) : 1;
        } else {
            // Geometry shaders produce more vertices per given vertex,
            // but they are not that relevant for timing emulation.
            triangles = vertices / 3;
        }

        triangle_count += triangles;
    }

    u64 CalculateAndResetDelay() {
        float result = command_count * nanoseconds_per_command;

        result += (1.f - culled_triangle_threshold) * triangle_count * nanoseconds_per_triangle;
        result += culled_triangle_threshold * triangle_count *
                  (nanoseconds_per_triangle * culled_triangle_time_cost);
        if (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::FasterConsoleGpu)
            [[unlikely]] {
            result /= 4.f;
        }

        triangle_count = 0;
        command_count = 0;

        return static_cast<u64>(result);
    }

private:
    size_t triangle_count{};
    size_t command_count{};

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const u32 file_version) {
        ar & triangle_count;
        ar & command_count;
    }
};

class PicaCore {
public:
    explicit PicaCore(Memory::MemorySystem& memory, std::shared_ptr<DebugContext> debug_context_);
    ~PicaCore();

    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer);

    void SetInterruptHandler(Service::GSP::InterruptHandler& signal_interrupt);

    void ProcessCmdList(PAddr list, u32 size, bool ignore_list);

    /// Where the command being worked on is: the address of its list and the position in it.
    /// The same from one frame to the next for a draw of a list that is sent or called again
    /// from where it was, which most are.
    /// Has the next ProcessCmdList read the list from these bytes, a copy made when the list
    /// was handed over, instead of from the emulated memory, which the game may have
    /// written since. Lists called from it are read where they are.
    void UseCommandListCopy(const u8* copy) {
        list_copy = copy;
    }

    [[nodiscard]] u64 CommandSite() const {
        return (static_cast<u64>(cmd_list.addr) << 32) | cmd_list.current_index;
    }

    /// The share of the viewport that the box around the vertices of the draw about to be
    /// carried out takes up, for a draw of two triangles at the most, whose vertices are
    /// shaded here to find out. One, all of it, where that cannot be told.
    float SmallDrawViewportShare(bool is_indexed);

    /// The box around the triangles of a draw, in the coordinates of the viewport, which goes
    /// from -1 to 1 both ways, and cut to it
    struct DrawBox {
        float left;
        float right;
        float bottom;
        float top;
    };

    /// The box around the vertices of the draw about to be carried out, for a draw of two
    /// triangles at the most, whose vertices are shaded here to find out. False where that
    /// cannot be told; a box without area for a draw that is outside of the viewport.
    bool SmallDrawBox(bool is_indexed, DrawBox& box);

private:
    void InitializeRegs();

    void HandleSpecialRegBatch(u32 id, const u32* values, u32 count);

    void HandleSpecialReg(u32 id, u32 value, bool& stop_requested);

    void WriteInternalRegBatch(u32 id, const u32* values, u32 count, u32 mask,
                               bool& stop_requested);

    void WriteInternalRegSequential(u32 id, const u32* __restrict values, u32 count, u32 mask,
                                    bool& stop_requested);

    void WriteInternalReg(u32 id, u32 value, u32 mask, bool& stop_requested);

    void SubmitImmediate(u32 data);

    void DrawImmediate();

    void DrawArrays(bool is_indexed);

    /// Draws shaded on the CPU that follow each other (sprites, a quad each) are drawn by the
    /// host together: their triangles stay with the rasterizer until something the drawing
    /// itself goes by is about to change. That is any register below the vertex pipeline's
    /// (targets, textures, blending, lighting; the pipeline's, the vertex shader's and the
    /// geometry shader's only decide what the triangles are), a draw that is not added to
    /// them, or the end of the list.
    void FlushSoftwareBatch();
    static constexpr u32 FIRST_VERTEX_REG = 0x200;
    static constexpr u32 MAX_BATCHED_DRAWS = 128;
    bool sw_batch_open{};
    u32 sw_batch_draws{};

    /// What rules that leave work out want to know of the draw about to be made, see
    /// VideoCore::DrawFacts
    class Facts;
    friend class Facts;
    /// A triangle given vertex by vertex is drawn: the rules are told, none can leave it out
    void TellRulesOfImmediateDraw();

    /// What makes a draw the same draw as another: a hash over the registers it goes by
    /// and its vertex shader; for a clear done by drawing only over what the fragments go by
    u64 TwinState(bool clears);
    /// Calls back with each stretch of registers [from, to) that a draw goes by as the
    /// registers are now
    template <typename Func>
    void ForTwinRegisters(bool clears, Func&& func) const;
    /// The float uniforms the vertex shader's program can read from its entry point, a bit
    /// for each (96 of them). Worked out once for each program.
    using UniformSet = std::array<u64, 2>;
    const UniformSet& FloatUniformsRead();
    std::vector<std::pair<u64, UniformSet>> uniforms_read;

    void LoadVertices(bool is_indexed);

    /// Sets up and returns the engine to shade a few vertices of the vertex shader in use
    /// with, without holding the emulation up: the configured one if it has the shader
    /// ready, the interpreter until then
    ShaderEngine& EngineForFewVertices();

    /// Whether the triangles of the draw about to be carried out cover all of the viewport.
    /// Only looks at draws of MAX_COVER_VERTICES vertices at the most, which are shaded here
    /// to find out.
    bool CoversViewport(bool is_indexed, bool* exactly = nullptr);
    /// The same for a draw that is not two triangles: a clear done with a mesh
    bool MeshCoversViewport(bool is_indexed, bool* exactly);
    /// Whether a mesh of up to MAX_COVER_VERTICES_LARGE vertices covers the middle of every
    /// pixel of the console in its viewport (not whether it reaches the viewport's edges).
    /// state: StateHash of the draw, by which one that does not cover is left alone for a
    /// while.
    bool LargeMeshCoversPixels(bool is_indexed, u64 state);
    /// Shades the vertices of the draw about to be made and hands over what is drawn of
    /// each of its triangles: the part within the depth range, as a polygon in pixels of
    /// the console from the corner of the viewport (whose size is put in width and height),
    /// if it faces the way that is kept. The sink returns false to stop. False if the
    /// vertices are not to be had, or are more than most_vertices.
    template <typename Sink>
    bool ForDrawnTriangles(bool is_indexed, u32 most_vertices, f32& width, f32& height,
                           Sink&& sink);
    /// A bit for each pixel of the viewport, for LargeMeshCoversPixels
    std::vector<u64> cover_bits;
    /// Meshes that did not cover their viewport lately, and for how many more askings each
    /// is left alone
    struct CoverMiss {
        u64 state{};
        u32 misses{};
        u32 rest{};
    };
    std::array<CoverMiss, 8> cover_misses{};
    u32 cover_miss_next{};

    /// How the texture coordinates of the draw about to be carried out go with the rows of
    /// its target, for each set of coordinates where that is a straight line. Only looks at
    /// draws of two triangles at the most, whose vertices are shaded here to find out.
    /// False if it is none.
    bool TextureRowsOfDraw(bool is_indexed, VideoCore::DrawTextureRows& rows);

#ifdef CITRA_ENABLE_TRACE_MARKERS
    /// Frame dumps: where the vertices of a draw of two triangles at the most end up in the
    /// target, and which texture coordinates they have
    void DumpSmallDraw(bool is_indexed);
    /// Frame dumps: the geometry shader program of a draw that has one, with its
    /// configuration, once for each
    void DumpGeometryProgram();
#endif

public:
    /// The most vertices of a draw that is looked at for whether it covers its viewport
    static constexpr u32 MAX_COVER_VERTICES = 36;
    /// The same for a mesh that would reset the depth buffer, which is looked at pixel by
    /// pixel
    static constexpr u32 MAX_COVER_VERTICES_LARGE = 1024;

    union Regs {
        static constexpr std::size_t NUM_REGS = 0x732;

        struct {
            u32 hardware_id;
            INSERT_PADDING_WORDS(0x3);
            MemoryFillConfig memory_fill_config[2];
            u32 vram_bank_control;
            u32 gpu_busy;
            INSERT_PADDING_WORDS(0x22);
            u32 backlight_control;
            INSERT_PADDING_WORDS(0xCF);
            FramebufferConfig framebuffer_config[2];
            INSERT_PADDING_WORDS(0x180);
            DisplayTransferConfig display_transfer_config;
            INSERT_PADDING_WORDS(0xF5);
            RegsInternal internal;
        };
        std::array<u32, NUM_REGS> reg_array;
    };
    static_assert(sizeof(Regs) == Regs::NUM_REGS * sizeof(u32));

    struct CommandList {
        PAddr addr;
        const u32* head;
        u32 current_index;
        u32 length;

        void Reset(PAddr addr, const u8* head, u32 size) {
            this->addr = addr;
            this->head = reinterpret_cast<const u32*>(head);
            this->length = size / sizeof(u32);
            current_index = 0;
        }

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const u32 file_version);
    };

    struct ImmediateModeState {
        AttributeBuffer input_vertex{};
        u32 current_attribute{};
        bool reset_geometry_pipeline{true};
        PackedAttribute queue;

        void Reset() {
            current_attribute = 0;
            reset_geometry_pipeline = true;
            queue.Reset();
        }

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const u32 file_version) {
            ar & input_vertex;
            ar & current_attribute;
            ar & reset_geometry_pipeline;
            ar & queue;
        }
    };

    struct ProcTex {
        static constexpr u8 TableAllDirty = 0xFF;

        union ValueEntry {
            u32 raw;

            // LUT value, encoded as 12-bit fixed point, with 12 fraction bits
            BitField<0, 12, u32> value; // 0.0.12 fixed point

            // Difference between two entry values. Used for efficient interpolation.
            // 0.0.12 fixed point with two's complement. The range is [-0.5, 0.5).
            // Note: the type of this is different from the one of lighting LUT
            BitField<12, 12, s32> difference;

            f32 ToFloat() const {
                return static_cast<f32>(value) / 4095.f;
            }

            f32 DiffToFloat() const {
                return static_cast<f32>(difference) / 4095.f;
            }
        };

        union ColorEntry {
            u32 raw;
            BitField<0, 8, u32> r;
            BitField<8, 8, u32> g;
            BitField<16, 8, u32> b;
            BitField<24, 8, u32> a;

            Common::Vec4<u8> ToVector() const {
                return {static_cast<u8>(r), static_cast<u8>(g), static_cast<u8>(b),
                        static_cast<u8>(a)};
            }
        };

        union ColorDifferenceEntry {
            u32 raw;
            BitField<0, 8, s32> r; // half of the difference between two ColorEntry
            BitField<8, 8, s32> g;
            BitField<16, 8, s32> b;
            BitField<24, 8, s32> a;

            Common::Vec4<s32> ToVector() const {
                return Common::Vec4<s32>{r, g, b, a} * 2;
            }
        };

        std::array<ValueEntry, 128> noise_table;
        std::array<ValueEntry, 128> color_map_table;
        std::array<ValueEntry, 128> alpha_map_table;
        std::array<ColorEntry, 256> color_table;
        std::array<ColorDifferenceEntry, 256> color_diff_table;
        union {
            u8 table_dirty = TableAllDirty;
            BitField<0, 1, u8> noise_lut_dirty;
            BitField<2, 1, u8> color_map_dirty;
            BitField<3, 1, u8> alpha_map_dirty;
            BitField<4, 1, u8> color_lut_dirty;
            BitField<5, 1, u8> color_diff_lut_dirty;
        };

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const u32 file_version) {
            ar& boost::serialization::make_binary_object(this, sizeof(ProcTex));
            if (Archive::is_loading::value) {
                table_dirty = TableAllDirty;
            }
        }
    };

    struct Lighting {
        static constexpr u32 LutAllDirty = 0xFFFFFF;

        union LutEntry {
            // Used for raw access
            u32 raw;

            // LUT value, encoded as 12-bit fixed point, with 12 fraction bits
            BitField<0, 12, u32> value; // 0.0.12 fixed point

            // Used for efficient interpolation.
            BitField<12, 11, u32> difference; // 0.0.11 fixed point
            BitField<23, 1, u32> neg_difference;

            f32 ToFloat() const {
                return static_cast<f32>(value) / 4095.f;
            }

            f32 DiffToFloat() const {
                const f32 diff = static_cast<f32>(difference) / 2047.f;
                return neg_difference ? -diff : diff;
            }

            template <class Archive>
            void serialize(Archive& ar, const u32 file_version) {
                ar & raw;
            }
        };

        std::array<std::array<LutEntry, 256>, 24> luts;
        u32 lut_dirty = LutAllDirty;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const u32 file_version) {
            ar& boost::serialization::make_binary_object(this, sizeof(Lighting));
            if (Archive::is_loading::value) {
                lut_dirty = LutAllDirty;
            }
        }
    };

    struct Fog {
        union LutEntry {
            // Used for raw access
            u32 raw;

            BitField<0, 13, s32> difference; // 1.1.11 fixed point
            BitField<13, 11, u32> value;     // 0.0.11 fixed point

            f32 ToFloat() const {
                return static_cast<f32>(value) / 2047.0f;
            }

            f32 DiffToFloat() const {
                return static_cast<f32>(difference) / 2047.0f;
            }
        };

        std::array<LutEntry, 128> lut;
        bool lut_dirty = true;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const u32 file_version) {
            ar& boost::serialization::make_binary_object(this, sizeof(Fog));
            if (Archive::is_loading::value) {
                lut_dirty = true;
            }
        }
    };

    RegsLcd regs_lcd{};
    Regs regs{};
    DirtyRegs dirty_regs{};
    GeometryShaderUnit gs_unit;
    ShaderSetup vs_setup;
    ShaderSetup gs_setup;
    ProcTex proctex{};
    Lighting lighting{};
    Fog fog{};
    AttributeBuffer input_default_attributes{};
    ImmediateModeState immediate{};

    DelayGenerator delay_generator{};

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const u32 file_version) {
        ar & regs_lcd;
        ar & regs.reg_array;
        ar & gs_unit;
        ar & vs_setup;
        ar & gs_setup;
        ar & proctex;
        ar & lighting;
        ar & fog;
        ar & input_default_attributes;
        ar & immediate;
        ar & delay_generator;
        ar & geometry_pipeline;
        ar & primitive_assembler;
        ar & cmd_list;
        if (Archive::is_loading::value) {
            dirty_regs.SetAllDirty();
        }
    }

public:
    struct RenderPropertiesGuess {
        u32 vp_height;
        PAddr paddr;
        bool vp_heigh_found = false;
        bool paddr_found = false;
        // Whether the scanned list issues any geometry.
        bool has_draw = false;
    };

    RenderPropertiesGuess GuessCmdRenderProperties(PAddr list, u32 size);

    /// Sets who is asked about each draw before it is carried out: the rules that find work
    /// that need not be done, see VideoCore::WorkRules
    void SetWorkRules(VideoCore::WorkRules* rules) {
        work_rules = rules;
    }

private:
    Memory::MemorySystem& memory;
    VideoCore::RasterizerInterface* rasterizer;
    std::shared_ptr<DebugContext> debug_context;
    Service::GSP::InterruptHandler signal_interrupt;
    GeometryPipeline geometry_pipeline;
    PrimitiveAssembler primitive_assembler;
    CommandList cmd_list;
    const u8* list_copy{};
    VideoCore::WorkRules* work_rules{};
    std::unique_ptr<ShaderEngine> shader_engine;
    /// Shades a few vertices while `shader_engine` does not have their shader ready
    std::unique_ptr<ShaderEngine> few_vertices_engine;
};

#define GPU_REG_INDEX(field_name) (offsetof(Pica::PicaCore::Regs, field_name) / sizeof(u32))

} // namespace Pica
