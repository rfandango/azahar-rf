// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <cmath>
#include <limits>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "video_core/frame_liveness.h"
#include "video_core/gpu.h"
#include "video_core/gpu_impl.h"
#include "video_core/pica_types.h"

namespace VideoCore {

namespace {

/// Looks at the last frames in a row that have to find a piece of work not needed before it
/// is left out
constexpr u16 REQUIRED_UNNEEDED = 30;
/// Frames to render everything for after a wrong guess, doubled with each one that follows
/// closely on the one before (see Mispredicted). What a wrong guess costs on the screen is one
/// image made from what its buffer held before, mostly the frame before; what this time costs
/// is the right eye drawn for all of it. On an Adreno 610 Kid Icarus: Uprising ran at 95% of
/// its speed with the right eye left out and at 75% with it drawn, and with 600 frames here
/// that doubled every time, four wrong guesses in ten minutes (the game's start and three
/// changes of scene) had it drawn for 180 of 500 seconds.
constexpr u32 DISTRUST_FRAMES = 150;
constexpr u32 MAX_DISTRUST_DOUBLINGS = 5;
/// A frame with more steps than this is not looked at
constexpr std::size_t MAX_STEPS = 4096;
/// Frames after which work and framebuffers that did not come up again are forgotten
constexpr u32 FORGET_AFTER = 600;
/// Times the last frames are gone through until what is needed no longer changes
constexpr int MAX_ROUNDS = 8;
constexpr std::size_t MAX_SHOWN_PLACES = 16;
/// Times the same copy, transfer or fill is told apart between two images put up for display
constexpr u32 MAX_OCCURRENCES = 255;
/// Rows are kept in memory eight at a time, in tiles
constexpr u32 TILE_ROWS = 8;
/// Pixels of the smaller of the two screens
constexpr u32 SCREEN_PIXELS = 240 * 320;
/// Frames in which work is left out, or none is, by which they have to outnumber the others
/// for that to be written to the log: a game may go from one to the other within frames
constexpr u32 CHANGE_HELD_FOR = 60;
/// What a row worked out from texture coordinates may be off by, in rows
constexpr f32 ROW_SLACK = 1.f / 32.f;

/// The one tracker there is, for the reads that reach it from the surface cache
FrameLiveness* instance = nullptr;

constexpr u64 Mix(u64 hash, u64 value) {
    hash ^= value + 0x9E3779B97F4A7C15ULL + (hash << 12) + (hash >> 4);
    hash *= 0xBF58476D1CE4E5B9ULL;
    return hash ^ (hash >> 31);
}

/// Bits a pixel of a texture takes
u32 TextureBitsPerPixel(Pica::TexturingRegs::TextureFormat format) {
    using Format = Pica::TexturingRegs::TextureFormat;
    switch (format) {
    case Format::RGB8:
        return 24;
    case Format::RGB5A1:
    case Format::RGB565:
    case Format::RGBA4:
    case Format::IA8:
    case Format::RG8:
        return 16;
    case Format::I8:
    case Format::A8:
    case Format::IA4:
    case Format::ETC1A4:
        return 8;
    case Format::I4:
    case Format::A4:
    case Format::ETC1:
        return 4;
    default:
        // RGBA8, and the most there can be for a format that is not known
        return 32;
    }
}

} // Anonymous namespace

bool MemoryRanges::Intersects(const MemoryRange& range) const {
    return std::ranges::any_of(ranges, [&range](const MemoryRange& other) {
        return other.Overlaps(range) && other.SameImage(range);
    });
}

MemoryRange MemoryRanges::Hull(const MemoryRange& range) const {
    MemoryRange hull{};
    for (const MemoryRange& other : ranges) {
        if (!other.Overlaps(range) || !other.SameImage(range)) {
            continue;
        }
        const PAddr start = std::max(other.start, range.start);
        const PAddr end = std::min(other.end, range.end);
        if (hull.Empty()) {
            hull = {start, end, range.layout};
        } else {
            hull.start = std::min(hull.start, start);
            hull.end = std::max(hull.end, end);
        }
    }
    return hull;
}

void MemoryRanges::Add(MemoryRange range) {
    if (range.Empty()) {
        return;
    }
    // Sorted by layout, then by where they start. Everything of its layout that it overlaps or
    // touches becomes one with it.
    const auto first = std::ranges::find_if(ranges, [&range](const MemoryRange& other) {
        return other.layout > range.layout ||
               (other.layout == range.layout && other.end >= range.start);
    });
    auto last = first;
    while (last != ranges.end() && last->layout == range.layout && last->start <= range.end) {
        range.start = std::min(range.start, last->start);
        range.end = std::max(range.end, last->end);
        ++last;
    }
    if (first == last) {
        ranges.insert(first, range);
    } else {
        *first = range;
        ranges.erase(first + 1, last);
    }
}

void MemoryRanges::Remove(const MemoryRange& range) {
    if (range.Empty()) {
        return;
    }
    for (auto it = ranges.begin(); it != ranges.end();) {
        if (!it->Overlaps(range)) {
            ++it;
            continue;
        }
        const MemoryRange old = *it;
        if (old.start < range.start && old.end > range.end) {
            // Cut out of the middle
            it->end = range.start;
            it = ranges.insert(it + 1, MemoryRange{range.end, old.end, old.layout});
            ++it;
        } else if (old.start < range.start) {
            it->end = range.start;
            ++it;
        } else if (old.end > range.end) {
            it->start = range.end;
            ++it;
        } else {
            it = ranges.erase(it);
        }
    }
}

FrameLiveness::FrameLiveness(GPU& gpu_) : gpu{gpu_} {
    instance = this;
}

FrameLiveness::~FrameLiveness() {
    if (instance == this) {
        instance = nullptr;
    }
}

void FrameLiveness::UpdateActive() {
    // Only with one eye on display is there an image nobody sees for certain. The setting is
    // the one that used to leave out the right eye by the order things are rendered in.
    const bool stereo_shown =
        Settings::values.render_3d.GetValue() != Settings::StereoRenderOption::Off &&
        Settings::values.render_3d_which_display.GetValue() != Settings::StereoWhichDisplay::None;
    const bool wanted =
        allowed && Settings::values.disable_right_eye_render.GetValue() && !stereo_shown &&
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyRightEyeSkip) == 0;
    if (wanted != active) {
        active = wanted;
        Forget();
    }
}

void FrameLiveness::Forget() {
    current.clear();
    previous.clear();
    too_many_steps = false;
    anchor = 0;
    stretch = 0;
    stretch_began_in_frame = false;
    occurrences.clear();
    verdicts.clear();
    written.Clear();
    written_coarse.fill(0);
    stale.Clear();
    shown_places.clear();
    unseen_places.clear();
    column_uses = {};
    last_column_draw = {};
    events++;
    draw_memo = DrawMemo{};
    for (KeptLook& kept : kept_looks) {
        kept.valid = false;
    }
    previous_hash = 0;
    seen.fill(0);
    left_out.fill(0);
    kept_for_later.fill(0);
    leaving_out = false;
    leaving_out_changing = 0;
    distrust = 0;
}

bool FrameLiveness::IsShown(const MemoryRange& range) const {
    return !range.Empty() && std::ranges::any_of(shown_places, [&range](const ShownPlace& place) {
        return place.range.Overlaps(range);
    });
}

bool FrameLiveness::IsImage(Kind kind, const MemoryRange& range) const {
    if (range.Empty() || kind == Kind::Draw) {
        return false;
    }
    // A game need not say where the image of the other eye goes while it is not asked for
    if (kind == Kind::Transfer && std::ranges::any_of(shown_places, [&range](const auto& place) {
            return place.range.end - place.range.start == range.end - range.start;
        })) {
        return true;
    }
    return std::ranges::any_of(
        unseen_places, [&range](const ShownPlace& place) { return place.range.Overlaps(range); });
}

void FrameLiveness::AddRead(Step& step, const MemoryRange& range) const {
    AddRead(step, range, Link{});
}

void FrameLiveness::AddRead(Step& step, const MemoryRange& range, const Link& link) const {
    // Memory the GPU never wrote holds what the game put there, which is always as it should
    if (range.Empty() || !WrittenIntersects(range)) {
        return;
    }
    ASSERT(step.reads.size() < step.reads.capacity());
    step.reads.push_back(range);
    step.links.push_back(link);
}

MemoryRange FrameLiveness::RowsOf(const Step& step, u32 buffer, u32 first, u32 last) {
    return {step.buffer_start[buffer] + first * step.row_bytes[buffer],
            step.buffer_start[buffer] + last * step.row_bytes[buffer], step.buffer_layout[buffer]};
}

bool FrameLiveness::NeededRows(const Step& step, const MemoryRanges& live, u32& first, u32& last) {
    if (step.shown) {
        first = step.first_row;
        last = step.last_row;
        return true;
    }
    first = std::numeric_limits<u32>::max();
    last = 0;
    for (u32 buffer = 0; buffer < 2; buffer++) {
        const MemoryRange hull = live.Hull(step.writes[buffer]);
        if (hull.Empty()) {
            continue;
        }
        const u32 bytes = step.row_bytes[buffer];
        first = std::min(first, (hull.start - step.buffer_start[buffer]) / bytes);
        last = std::max(last, (hull.end - step.buffer_start[buffer] + bytes - 1) / bytes);
    }
    if (last <= first) {
        return false;
    }
    first = std::max(Common::AlignDown(first, TILE_ROWS), step.first_row);
    last = std::min(Common::AlignUp(last, TILE_ROWS), step.last_row);
    return last > first;
}

void FrameLiveness::AddReadsFor(const Step& step, u32 first, u32 last, MemoryRanges& live) {
    for (std::size_t index = 0; index < step.reads.size(); index++) {
        const MemoryRange& read = step.reads[index];
        const Link link = index < step.links.size() ? step.links[index] : Link{};
        switch (link.how) {
        case Link::How::Rows:
            live.Add(RowsOf(step, link.buffer, first, last));
            break;
        case Link::How::Mapped: {
            // From the edge of the first row to the edge of the last: what a filter takes
            // in beyond that is not followed
            const f32 one_end = link.scale * static_cast<f32>(first) + link.offset;
            const f32 other_end = link.scale * static_cast<f32>(last) + link.offset;
            const f32 rows = static_cast<f32>(link.rows);
            const f32 low = std::clamp(std::min(one_end, other_end) + ROW_SLACK, 0.f, rows);
            const f32 high = std::clamp(std::max(one_end, other_end) - ROW_SLACK, 0.f, rows);
            const u32 low_row = Common::AlignDown(static_cast<u32>(std::floor(low)), TILE_ROWS);
            const u32 high_row =
                std::min(link.rows, Common::AlignUp(static_cast<u32>(std::ceil(high)), TILE_ROWS));
            if (high_row > low_row) {
                live.Add(MemoryRange{read.start + low_row * link.row_bytes,
                                     std::min(read.end, read.start + high_row * link.row_bytes),
                                     read.layout});
            }
            break;
        }
        default:
            live.Add(read);
            break;
        }
    }
}

bool FrameLiveness::StaleRows(const Step& step, u32& first, u32& last,
                              const MemoryRange*& stale_read) const {
    first = std::numeric_limits<u32>::max();
    last = 0;
    for (std::size_t index = 0; index < step.reads.size(); index++) {
        const MemoryRange& read = step.reads[index];
        const MemoryRange hull = stale.Hull(read);
        if (hull.Empty()) {
            continue;
        }
        const Link link = index < step.links.size() ? step.links[index] : Link{};
        u32 from = step.first_row;
        u32 to = step.last_row;
        if (link.how == Link::How::Rows) {
            const u32 bytes = step.row_bytes[link.buffer];
            const PAddr start = step.buffer_start[link.buffer];
            from = Common::AlignDown((hull.start - start) / bytes, TILE_ROWS);
            to = Common::AlignUp((hull.end - start + bytes - 1) / bytes, TILE_ROWS);
        } else if (link.how == Link::How::Mapped) {
            // The rows of the target that take from the stale rows of the texture
            const f32 texture_from = static_cast<f32>((hull.start - read.start) / link.row_bytes);
            const f32 texture_to =
                static_cast<f32>((hull.end - read.start + link.row_bytes - 1) / link.row_bytes);
            const f32 one_end = (texture_from - link.offset) / link.scale;
            const f32 other_end = (texture_to - link.offset) / link.scale;
            const f32 most = static_cast<f32>(step.last_row);
            const f32 low = std::clamp(std::min(one_end, other_end) + ROW_SLACK, 0.f, most);
            const f32 high = std::clamp(std::max(one_end, other_end) - ROW_SLACK, 0.f, most);
            from = Common::AlignDown(static_cast<u32>(std::floor(low)), TILE_ROWS);
            to = Common::AlignUp(static_cast<u32>(std::ceil(high)), TILE_ROWS);
        }
        from = std::max(from, step.first_row);
        to = std::min(to, step.last_row);
        if (to <= from) {
            continue;
        }
        first = std::min(first, from);
        last = std::max(last, to);
        stale_read = &read;
    }
    return last > first;
}

u64 FrameLiveness::NextOccurrence(u64 base) {
    const auto name = [this, base](u32 count) {
        return counts_from_display ? Mix(Mix(base, count), stretch) : Mix(base, count);
    };
    for (auto& [key, count] : occurrences) {
        if (key == base) {
            // Past this many they are no longer told apart, and count as needed if one of them is
            count = std::min(count + 1, MAX_OCCURRENCES);
            return name(count);
        }
    }
    occurrences.emplace_back(base, 0);
    return name(0);
}

void FrameLiveness::BeginStretch(u64 base) {
    if (!counts_from_display) {
        return;
    }
    stretch = base;
    stretch_began_in_frame = true;
    occurrences.clear();
}

bool FrameLiveness::MayReplaceTarget(const Pica::RegsInternal& regs) const {
    if (!active) {
        return false;
    }
    using Pica::FramebufferRegs;
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& merger = regs.framebuffer.output_merger;

    // Two triangles at the most. Whether they face the way that is drawn is looked at with
    // their vertices, by the caller.
    if (regs.pipeline.num_vertices > 6) {
        return false;
    }
    // No pixel is kept from being written
    if (merger.alpha_test.enable || merger.depth_test_enable || merger.stencil_test.enable ||
        merger.fragment_operation_mode != FramebufferRegs::FragmentOperationMode::Default) {
        return false;
    }
    // All of each pixel is written, and nothing of what it held goes into the result
    if (framebuffer.allow_color_write == 0 || !merger.red_enable || !merger.green_enable ||
        !merger.blue_enable || !merger.alpha_enable) {
        return false;
    }
    if (merger.alphablend_enable) {
        const auto& blending = merger.alpha_blending;
        if (blending.factor_source_rgb != FramebufferRegs::BlendFactor::One ||
            blending.factor_source_a != FramebufferRegs::BlendFactor::One ||
            blending.factor_dest_rgb != FramebufferRegs::BlendFactor::Zero ||
            blending.factor_dest_a != FramebufferRegs::BlendFactor::Zero ||
            blending.blend_equation_rgb != FramebufferRegs::BlendEquation::Add ||
            blending.blend_equation_a != FramebufferRegs::BlendEquation::Add) {
            return false;
        }
    } else if (merger.logic_op != FramebufferRegs::LogicOp::Copy) {
        return false;
    }
    // The viewport is all of the target
    const auto& rasterizer = regs.rasterizer;
    if (rasterizer.scissor_test.mode != Pica::RasterizerRegs::ScissorMode::Disabled ||
        rasterizer.viewport_corner.x > 0 || rasterizer.viewport_corner.y > 0) {
        return false;
    }
    const f32 width = Pica::f24::FromRaw(rasterizer.viewport_size_x).ToFloat32() * 2.f;
    const f32 height = Pica::f24::FromRaw(rasterizer.viewport_size_y).ToFloat32() * 2.f;
    return static_cast<f32>(rasterizer.viewport_corner.x) + width >=
               static_cast<f32>(framebuffer.GetWidth()) &&
           static_cast<f32>(rasterizer.viewport_corner.y) + height >=
               static_cast<f32>(framebuffer.GetHeight());
}

bool FrameLiveness::WantsTextureRows(const Pica::RegsInternal& regs) const {
    using Pica::TexturingRegs;
    if (!active || regs.pipeline.num_vertices > 6 ||
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyWholeTargetWork)) {
        return false;
    }
    const auto textures = regs.texturing.GetTextures();
    for (std::size_t unit = 0; unit < textures.size(); unit++) {
        const auto& texture = textures[unit];
        if (!texture.enabled || texture.config.lod.max_level != 0 ||
            (unit == 0 && texture.config.type.Value() != TexturingRegs::TextureConfig::Texture2D)) {
            continue;
        }
        const PAddr address = texture.config.GetPhysicalAddress();
        const u32 size =
            texture.config.width * texture.config.height * TextureBitsPerPixel(texture.format) / 8;
        if (WrittenIntersects(MemoryRange{address, address + size})) {
            return true;
        }
    }
    return false;
}

void FrameLiveness::NoteWritten(const MemoryRange& range) {
    if (range.Empty()) {
        return;
    }
    written.Add(range);
    const u32 last = (range.end - 1) >> COARSE_SHIFT;
    for (u32 piece = range.start >> COARSE_SHIFT; piece <= last; piece++) {
        written_coarse[piece >> 6] |= u64{1} << (piece & 63);
    }
}

bool FrameLiveness::WrittenNear(PAddr start, PAddr end) const {
    if (start >= end) {
        return false;
    }
    const u32 last = (end - 1) >> COARSE_SHIFT;
    for (u32 piece = start >> COARSE_SHIFT; piece <= last; piece++) {
        if ((written_coarse[piece >> 6] >> (piece & 63)) & 1) {
            return true;
        }
    }
    return false;
}

bool FrameLiveness::SamplesWritten(const Pica::RegsInternal& regs) const {
    // The same textures, places and sizes as ShouldAllowDraw enters as reads
    using Pica::TexturingRegs;
    const auto& texturing = regs.texturing;
    const auto samples = [&](bool enabled, const TexturingRegs::TextureConfig& config,
                             TexturingRegs::TextureFormat format, bool first_unit) {
        if (!enabled) {
            return false;
        }
        const auto type =
            first_unit ? config.type.Value() : TexturingRegs::TextureConfig::Texture2D;
        if (type == TexturingRegs::TextureConfig::Disabled) {
            return false;
        }
        u32 size = config.width * config.height * TextureBitsPerPixel(format) / 8;
        if (config.lod.max_level != 0) {
            size += size / 3 + 1;
        }
        if (type == TexturingRegs::TextureConfig::TextureCube ||
            type == TexturingRegs::TextureConfig::ShadowCube) {
            for (u32 face = 0; face < 6; face++) {
                const PAddr address =
                    texturing.GetCubePhysicalAddress(static_cast<TexturingRegs::CubeFace>(face));
                if (WrittenIntersects(MemoryRange{address, address + size})) {
                    return true;
                }
            }
            return false;
        }
        const PAddr address = config.GetPhysicalAddress();
        return WrittenIntersects(MemoryRange{address, address + size});
    };
    return samples(texturing.main_config.texture0_enable != 0, texturing.texture0,
                   texturing.texture0_format, true) ||
           samples(texturing.main_config.texture1_enable != 0, texturing.texture1,
                   texturing.texture1_format, false) ||
           samples(texturing.main_config.texture2_enable != 0, texturing.texture2,
                   texturing.texture2_format, false);
}

FrameLiveness::DrawSignature FrameLiveness::SignatureOf(const Pica::RegsInternal& regs,
                                                        bool replaces_target, bool may_leave_out,
                                                        bool overwrites_colour,
                                                        bool overwrites_depth, bool by_rows) {
    // Everything ShouldAllowDraw reads that is not a texture. What is read there and missing
    // here would make two different draws one: the comparison that is made now and then (see
    // there) is what would show it.
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& merger = regs.framebuffer.output_merger;
    const auto& rasterizer = regs.rasterizer;
    DrawSignature signature;
    signature.colour = framebuffer.GetColorBufferPhysicalAddress();
    signature.depth = framebuffer.GetDepthBufferPhysicalAddress();
    signature.width = framebuffer.GetWidth();
    signature.height = framebuffer.GetHeight();
    signature.formats = (static_cast<u32>(framebuffer.color_format.Value()) << 8) |
                        static_cast<u32>(framebuffer.depth_format.Value());
    signature.viewport_corner_y = rasterizer.viewport_corner.y;
    signature.viewport_size_y = rasterizer.viewport_size_y;
    signature.scissor = (static_cast<u32>(rasterizer.scissor_test.mode.Value()) << 20) |
                        (rasterizer.scissor_test.y1 << 10) | rasterizer.scissor_test.y2;
    signature.flags = (framebuffer.allow_color_write != 0 ? 1u : 0u) |
                      (framebuffer.allow_depth_stencil_write != 0 ? 2u : 0u) |
                      (regs.framebuffer.IsShadowRendering() ? 4u : 0u) |
                      (merger.stencil_test.enable ? 8u : 0u) | (merger.red_enable ? 16u : 0u) |
                      (merger.green_enable ? 32u : 0u) | (merger.blue_enable ? 64u : 0u) |
                      (merger.alpha_enable ? 128u : 0u) | (merger.depth_write_enable ? 256u : 0u) |
                      (merger.depth_test_enable ? 512u : 0u) | (replaces_target ? 1024u : 0u) |
                      (may_leave_out ? 2048u : 0u) | (overwrites_colour ? 4096u : 0u) |
                      (overwrites_depth ? 8192u : 0u) | (by_rows ? 16384u : 0u);
    return signature;
}

void FrameLiveness::OnDone() {
    if (!active || !just_left_out || too_many_steps || current.empty()) {
        return;
    }
    just_left_out = false;
    stale = stale_before;
    last_carried_out = true;
    CarryOut(current.back());
}

Vote FrameLiveness::OnDraw(DrawFacts& facts) {
    just_left_out = false;
    if (!active) {
        return Vote::Carry;
    }
    const Pica::RegsInternal& regs = facts.Regs();
    const bool shared_depth = NoteDepthUse(regs);
    NoteColumns(regs);
    if (facts.IsImmediate()) {
        // Its vertices are with the rasterizer already: carried out in any case
        ShouldAllowDraw(regs, false, false);
        return Vote::Carry;
    }
    // What the registers alone tell is asked first: whether the triangles cover the viewport
    // has their vertices shaded. Of a mesh that costs a few tenths of a millisecond, and has
    // made a difference only where two colour buffers take turns with one depth buffer, the
    // two eyes of a game, and a backdrop resets the depth before each: asked there only.
    // (Kirby: Planet Robobot draws its flat layers as meshes of a few hundred vertices with
    // the depth test off, twice a frame, for one eye.)
    const bool may_ask = !facts.CoverTestIsCostly() || shared_depth;
    const auto covers = [&] { return may_ask && facts.CoversViewport(); };
    const bool replaces_target = MayReplaceTarget(regs) && covers();
    DrawTextureRows texture_rows;
    const bool rows_known = WantsTextureRows(regs) && facts.TextureRows(texture_rows);
    // A draw that covers its viewport writes every pixel of the rows it is confined to if
    // the viewport is as wide as the target, or as what is in use of it
    const bool overwrites_colour =
        facts.MayOverwriteColour() &&
        (facts.ViewportSpansTarget() || ViewportSpansColumnsInUse(regs)) && covers();
    // Depth and stencil are one buffer. A draw that renews the depth of every pixel but
    // leaves the stencil renews the buffer as far as anything goes that work left out could
    // have changed, as long as the game draws nothing with the stencil test on: only such
    // draws write stencil. (Star Fox 64 3D resets the depth its two eyes share with a
    // backdrop before each eye, in a buffer whose stencil bits it never uses.)
    const bool renews_depth =
        facts.MayOverwriteDepth() || (facts.MayOverwriteDepthValues() && !StencilInUse());
    const bool overwrites_depth = renews_depth && facts.ViewportSpansTarget() && covers();
    return ShouldAllowDraw(regs, replaces_target, true, rows_known ? &texture_rows : nullptr,
                           overwrites_colour, overwrites_depth)
               ? Vote::Carry
               : Vote::Unseen;
}

void FrameLiveness::NoteColumns(const Pica::RegsInternal& regs) {
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& rasterizer = regs.rasterizer;
    const PAddr colour = framebuffer.GetColorBufferPhysicalAddress();
    f32 right = static_cast<f32>(rasterizer.viewport_corner.x) +
                Pica::f24::FromRaw(rasterizer.viewport_size_x).ToFloat32() * 2.f;
    if (rasterizer.scissor_test.mode == Pica::RasterizerRegs::ScissorMode::Include) {
        right = std::min(right, static_cast<f32>(rasterizer.scissor_test.x2) + 1.f);
    }
    const u32 width = framebuffer.GetWidth();
    const u32 columns =
        right > 0.f ? std::min(static_cast<u32>(std::min(std::ceil(right), 65535.f)), width) : 0;
    const ColumnDraw draw{
        .colour = framebuffer.allow_color_write != 0 ? colour : 0,
        .depth = framebuffer.allow_depth_stencil_write != 0
                     ? framebuffer.GetDepthBufferPhysicalAddress()
                     : 0,
        .size = (width << 16) | framebuffer.GetHeight(),
        .formats = (static_cast<u32>(framebuffer.color_format.Value()) << 8) |
                   static_cast<u32>(framebuffer.depth_format.Value()),
        .right = columns,
        .frame = frame,
    };
    if (draw == last_column_draw) {
        return;
    }
    last_column_draw = draw;

    const u32 pixels = width * framebuffer.GetHeight();
    if (draw.depth != 0) {
        NoteWholeWrite({draw.depth, draw.depth + pixels * Pica::FramebufferRegs::BytesPerDepthPixel(
                                                              framebuffer.depth_format)});
    }
    if (draw.colour == 0) {
        return;
    }
    const u32 bytes = Pica::FramebufferRegs::BytesPerColorPixel(framebuffer.color_format);
    const MemoryRange buffer{colour, colour + pixels * bytes,
                             MemoryLayout::Make(MemoryLayout::Colour, width, bytes * 8)};
    ColumnUse* use = nullptr;
    for (ColumnUse& known : column_uses) {
        if (known.buffer.start == buffer.start && known.buffer.layout == buffer.layout) {
            use = &known;
        }
    }
    // The same memory as a surface with other rows: its columns are not those of this one
    NoteWholeWrite(buffer, use != nullptr ? colour : 0);
    const u32 stretch = frame / COLUMN_STRETCH;
    if (use == nullptr) {
        use = &*std::ranges::min_element(column_uses, {}, &ColumnUse::last_frame);
        *use = ColumnUse{.buffer = buffer, .stretch = stretch, .whole_frame = frame};
    }
    if (use->stretch != stretch) {
        use->before = use->stretch + 1 == stretch ? use->now : 0;
        use->now = 0;
        use->stretch = stretch;
    }
    use->buffer.end = std::max(use->buffer.end, buffer.end);
    use->now = std::max(use->now, columns);
    use->last_frame = frame;
}

void FrameLiveness::NoteWholeWrite(const MemoryRange& range, PAddr but) {
    for (ColumnUse& use : column_uses) {
        if (use.buffer.Overlaps(range) &&
            (but == 0 || use.buffer.start != but || use.buffer.layout != range.layout)) {
            use.whole_frame = frame;
        }
    }
}

bool FrameLiveness::ViewportSpansColumnsInUse(const Pica::RegsInternal& regs) const {
    // Known for a while: what reached further before that is no part of what goes on now
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& rasterizer = regs.rasterizer;
    const PAddr colour = framebuffer.GetColorBufferPhysicalAddress();
    const u32 bytes = Pica::FramebufferRegs::BytesPerColorPixel(framebuffer.color_format);
    const u32 layout = MemoryLayout::Make(MemoryLayout::Colour, framebuffer.GetWidth(), bytes * 8);
    const auto use = std::ranges::find_if(column_uses, [&](const ColumnUse& known) {
        return known.buffer.start == colour && known.buffer.layout == layout;
    });
    return use != column_uses.end() && use->before != 0 &&
           frame - use->whole_frame > COLUMN_STRETCH && rasterizer.viewport_corner.x <= 0 &&
           static_cast<f32>(rasterizer.viewport_corner.x) +
                   Pica::f24::FromRaw(rasterizer.viewport_size_x).ToFloat32() * 2.f >=
               static_cast<f32>(std::max(use->now, use->before));
}

bool FrameLiveness::NoteDepthUse(const Pica::RegsInternal& regs) {
    // How long after the last change of partner a depth buffer counts as shared
    constexpr u32 SHARED_FOR = 8;
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const PAddr depth = framebuffer.GetDepthBufferPhysicalAddress();
    const PAddr colour = framebuffer.GetColorBufferPhysicalAddress();
    if (depth == 0 || colour == 0) {
        return false;
    }
    DepthUse* use = nullptr;
    for (DepthUse& known : depth_uses) {
        if (known.depth == depth) {
            use = &known;
        }
    }
    if (use == nullptr) {
        use = &depth_uses[depth_use_next++ % depth_uses.size()];
        *use = DepthUse{.depth = depth, .colour = colour};
        return false;
    }
    if (use->colour != colour) {
        use->colour = colour;
        use->changed_frame = frame;
        use->shared = true;
    }
    return use->shared && frame - use->changed_frame <= SHARED_FOR;
}

Vote FrameLiveness::OnFill(const FillFacts& facts) {
    return ShouldAllowFill(facts.config) ? Vote::Carry : Vote::Unseen;
}

Vote FrameLiveness::OnTransfer(const TransferFacts& facts) {
    return ShouldAllowTransfer(facts.config) ? Vote::Carry : Vote::Unseen;
}

void FrameLiveness::OnFrameEnd([[maybe_unused]] const FrameFacts& facts) {
    EndFrame();
}

bool FrameLiveness::ShouldAllowDraw(const Pica::RegsInternal& regs, bool replaces_target,
                                    bool may_leave_out, const DrawTextureRows* texture_rows,
                                    bool overwrites_colour, bool overwrites_depth) {
    if (!active) {
        return true;
    }
    using Pica::FramebufferRegs;
    using Pica::TexturingRegs;
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& merger = regs.framebuffer.output_merger;
    const u32 width = framebuffer.GetWidth();
    const u32 height = framebuffer.GetHeight();
    const bool shadow = regs.framebuffer.IsShadowRendering();
    const bool stencil = regs.framebuffer.HasStencil() && merger.stencil_test.enable;
    const bool by_rows =
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyWholeTargetWork) == 0;

    // The draw before once more, as far as anything here goes (see DrawMemo): the same answer
    const DrawSignature signature = SignatureOf(regs, replaces_target, may_leave_out,
                                                overwrites_colour, overwrites_depth, by_rows);
    const bool plain = !SamplesWritten(regs);
    draws_asked++;
    bool compare = false;
    if (plain && draw_memo.valid && draw_memo.events == events &&
        draw_memo.signature == signature) {
        // One in 64 of them goes the long way all the same and is compared with the memo
        compare = (++memo_checks & 63) == 0;
        if (!compare) {
            draws_repeated++;
            seen[static_cast<std::size_t>(Kind::Draw)]++;
            if (!draw_memo.carried_out) {
                left_out[static_cast<std::size_t>(Kind::Draw)]++;
            } else if (draw_memo.kept_for_later) {
                kept_for_later[static_cast<std::size_t>(Kind::Draw)]++;
            }
            last_carried_out = draw_memo.carried_out;
            return draw_memo.carried_out;
        }
    }
    const DrawMemo memo_before = draw_memo;
#ifdef CITRA_ENABLE_TRACE_MARKERS
    if (!compare) {
        if (!plain) {
            memo_sampling++;
        } else if (!draw_memo.valid) {
            memo_after_sampling++;
        } else if (draw_memo.events != events) {
            memo_after_other++;
        } else {
            memo_other_signature++;
        }
    }
#endif

    Step step;
    step.kind = Kind::Draw;
    // What cannot be left out counts as needed, and with it all that it reads
    step.shown = !may_leave_out;

    // The rows the viewport and the scissor confine the draw to. They are counted from the
    // bottom, and the bottom row is the last one in memory.
    step.first_row = 0;
    step.last_row = height;
    // Whether those are the rows of the viewport exactly: only then does a draw that
    // overwrites its viewport overwrite all of the rows it is taken to write
    bool rows_exact = false;
    if (by_rows) {
        const auto& rasterizer = regs.rasterizer;
        f32 bottom = static_cast<f32>(rasterizer.viewport_corner.y);
        f32 top = bottom + Pica::f24::FromRaw(rasterizer.viewport_size_y).ToFloat32() * 2.f;
        const f32 viewport_bottom = bottom;
        const f32 viewport_top = top;
        if (rasterizer.scissor_test.mode == Pica::RasterizerRegs::ScissorMode::Include) {
            bottom = std::max(bottom, static_cast<f32>(rasterizer.scissor_test.y1));
            top = std::min(top, static_cast<f32>(rasterizer.scissor_test.y2) + 1.f);
        }
        const bool whole_viewport = bottom == viewport_bottom && top == viewport_top;
        bottom = std::clamp(std::floor(bottom), 0.f, static_cast<f32>(height));
        top = std::clamp(std::ceil(top), 0.f, static_cast<f32>(height));
        if (top > bottom) {
            const u32 first = height - static_cast<u32>(top);
            const u32 last = height - static_cast<u32>(bottom);
            step.first_row = Common::AlignDown(first, TILE_ROWS);
            step.last_row = std::min(height, Common::AlignUp(last, TILE_ROWS));
            rows_exact = whole_viewport && step.first_row == first && step.last_row == last &&
                         viewport_bottom >= 0.f && viewport_top <= static_cast<f32>(height) &&
                         viewport_bottom == std::floor(viewport_bottom) &&
                         viewport_top == std::floor(viewport_top);
        }
    }
    overwrites_colour = overwrites_colour && rows_exact && !shadow;
    overwrites_depth = overwrites_depth && rows_exact && !shadow;

    const PAddr colour = framebuffer.GetColorBufferPhysicalAddress();
    const bool writes_colour = colour != 0 && framebuffer.allow_color_write != 0 &&
                               (shadow || merger.red_enable || merger.green_enable ||
                                merger.blue_enable || merger.alpha_enable);
    MemoryRange colour_range;
    if (writes_colour) {
        const u32 bytes_per_pixel = FramebufferRegs::BytesPerColorPixel(framebuffer.color_format);
        step.buffer_start[0] = colour;
        step.row_bytes[0] = width * bytes_per_pixel;
        step.buffer_layout[0] =
            MemoryLayout::Make(MemoryLayout::Colour, width, bytes_per_pixel * 8);
        colour_range = RowsOf(step, 0, step.first_row, step.last_row);
        step.writes[0] = colour_range;
        step.replaces = replaces_target || overwrites_colour;
        step.renews = step.replaces;
        NoteWritten(MemoryRange{colour_range.start, colour_range.end});
    }

    const PAddr depth = framebuffer.GetDepthBufferPhysicalAddress();
    const bool writes_depth =
        (merger.depth_write_enable && framebuffer.allow_depth_stencil_write != 0) || stencil;
    const bool uses_depth = depth != 0 && !shadow && (merger.depth_test_enable || writes_depth);
    MemoryRange depth_range;
    if (uses_depth) {
        const u32 bytes_per_pixel = FramebufferRegs::BytesPerDepthPixel(framebuffer.depth_format);
        step.buffer_start[1] = depth;
        step.row_bytes[1] = width * bytes_per_pixel;
        step.buffer_layout[1] = MemoryLayout::Make(MemoryLayout::Depth, width, bytes_per_pixel * 8);
        depth_range = RowsOf(step, 1, step.first_row, step.last_row);
        if (writes_depth) {
            step.writes[1] = depth_range;
            NoteWritten(MemoryRange{depth_range.start, depth_range.end});
            // A clear done by drawing: games whose two eyes share a depth buffer clear it
            // this way before each (Star Fox 64 3D), and what the other eye left in it is
            // then no part of this one
            step.replaces_depth = overwrites_depth;
        }
    }

    // What the target holds goes into the result, unless all of it is replaced. That is what
    // was drawn there as a surface with rows of this length: the same memory is another
    // image as anything else, and no game shows that.
    if (writes_colour && !step.replaces) {
        AddRead(step, colour_range, Link{.how = Link::How::Rows, .buffer = 0});
    }
    if (!step.replaces_depth) {
        AddRead(step, depth_range, Link{.how = Link::How::Rows, .buffer = 1});
    }
    if (step.writes[0].Empty() && step.writes[1].Empty()) {
        // Nothing to go by rows with
        step.first_row = 0;
        step.last_row = 0;
    }

    const auto textures = regs.texturing.GetTextures();
    for (std::size_t unit = 0; unit < textures.size(); unit++) {
        const auto& texture = textures[unit];
        if (!texture.enabled) {
            continue;
        }
        const auto type =
            unit == 0 ? texture.config.type.Value() : TexturingRegs::TextureConfig::Texture2D;
        if (type == TexturingRegs::TextureConfig::Disabled) {
            continue;
        }
        const u32 bits_per_pixel = TextureBitsPerPixel(texture.format);
        const u32 layout =
            MemoryLayout::Make(MemoryLayout::Texture, texture.config.width, bits_per_pixel);
        u32 size = texture.config.width * texture.config.height * bits_per_pixel / 8;
        if (texture.config.lod.max_level != 0) {
            // The smaller levels follow the first in memory
            size += size / 3 + 1;
        }
        if (type == TexturingRegs::TextureConfig::TextureCube ||
            type == TexturingRegs::TextureConfig::ShadowCube) {
            for (u32 face = 0; face < 6; face++) {
                const PAddr address = regs.texturing.GetCubePhysicalAddress(
                    static_cast<TexturingRegs::CubeFace>(face));
                AddRead(step, MemoryRange{address, address + size, layout});
            }
        } else {
            const PAddr address = texture.config.GetPhysicalAddress();
            // Which rows of the texture a row of the target takes from, where the
            // coordinates of this unit go with the rows in a straight line. As rows of
            // memory, which run from the top down in both.
            Link link;
            const std::size_t set =
                unit == 2 && regs.texturing.main_config.texture2_use_coord1 ? 1 : unit;
            const u32 row_bytes = texture.config.width * bits_per_pixel / 8;
            if (texture_rows != nullptr && texture_rows->known[set] && step.last_row != 0 &&
                texture.config.lod.max_level == 0 && row_bytes != 0 &&
                type == TexturingRegs::TextureConfig::Texture2D) {
                const f32 rows = static_cast<f32>(texture.config.height);
                const f32 scale = texture_rows->scale[set] * rows;
                const f32 offset =
                    rows - scale * static_cast<f32>(height) - texture_rows->offset[set] * rows;
                // Not one that repeats or stops at its edge within the draw, and not one
                // row for all of the target
                const f32 one_end = scale * static_cast<f32>(step.first_row) + offset;
                const f32 other_end = scale * static_cast<f32>(step.last_row) + offset;
                if (std::abs(scale) > 1.f / 64.f && std::min(one_end, other_end) >= -1.f &&
                    std::max(one_end, other_end) <= rows + 1.f) {
                    link = Link{.how = Link::How::Mapped,
                                .scale = scale,
                                .offset = offset,
                                .row_bytes = row_bytes,
                                .rows = texture.config.height};
                }
            }
            AddRead(step, MemoryRange{address, address + size, layout}, link);
        }
    }

    u64 key = Mix(static_cast<u64>(Kind::Draw), anchor);
    key = Mix(key, (static_cast<u64>(colour_range.start) << 32) | colour_range.end);
    key = Mix(key, colour_range.layout);
    key = Mix(key, (static_cast<u64>(depth_range.start) << 1) | (writes_depth ? 1 : 0));
    // (With the stencil test on it is another kind of work: it writes what a draw without
    // it does not, and what was learned of the one says nothing of the other)
    key = Mix(key, (step.replaces ? 1 : 0) | (may_leave_out ? 0 : 2) |
                       (step.replaces_depth ? 4 : 0) | (stencil ? 8 : 0));
    if (stencil) {
        stencil_seen = true;
        stencil_last_frame = frame;
    }
    for (const MemoryRange& read : step.reads) {
        key = Mix(key, read.start);
    }
    for (const Link& link : step.links) {
        if (link.how == Link::How::Mapped) {
            // The same draw with other texture coordinates takes from other rows
            key = Mix(key, static_cast<u64>(std::llround(link.scale * 256.f)));
            key = Mix(key, static_cast<u64>(std::llround(link.offset * 4.f)));
        }
    }
    step.key = key;
    const u32 kept_before = kept_for_later[static_cast<std::size_t>(Kind::Draw)];
    const bool carried_out = Submit(step);
    if (compare && (key != memo_before.key || carried_out != memo_before.carried_out)) {
        if (memo_differences++ == 0) {
            LOG_ERROR(HW_GPU,
                      "Unseen work: a draw taken for the one before it is another one (key "
                      "{:016X} against {:016X}, carried out {} against {})",
                      key, memo_before.key, carried_out, memo_before.carried_out);
        }
    }
    draw_memo.valid = plain;
    draw_memo.signature = signature;
    draw_memo.events = events;
    draw_memo.key = key;
    draw_memo.carried_out = carried_out;
    draw_memo.kept_for_later = kept_for_later[static_cast<std::size_t>(Kind::Draw)] != kept_before;
    return carried_out;
}

bool FrameLiveness::ShouldAllowTransfer(const Pica::DisplayTransferConfig& config) {
    just_left_out = false;
    if (!active) {
        return true;
    }
    const PAddr input = config.GetPhysicalInputAddress();
    const PAddr output = config.GetPhysicalOutputAddress();

    Step step;
    MemoryRange source;
    MemoryRange destination;
    if (config.is_texture_copy) {
        // Lines of bytes with gaps between them, on either side
        const u32 size = Common::AlignDown(config.texture_copy.size, 16);
        const auto span = [size](u32 line, u32 gap) -> u32 {
            if (gap == 0 || line == 0) {
                return size;
            }
            const u32 lines = (size + line - 1) / line;
            return lines * (line + gap) - gap;
        };
        source = {input, input + span(config.texture_copy.input_width * 16,
                                      config.texture_copy.input_gap * 16)};
        destination = {output, output + span(config.texture_copy.output_width * 16,
                                             config.texture_copy.output_gap * 16)};
        step.kind = Kind::Copy;
        step.replaces = config.texture_copy.output_gap == 0;
    } else {
        u32 width = config.output_width;
        u32 height = config.output_height;
        if (config.scaling != Pica::DisplayTransferConfig::NoScale) {
            width /= 2;
        }
        if (config.scaling == Pica::DisplayTransferConfig::ScaleXY) {
            height /= 2;
        }
        // A transfer reads rows of pixels, which is an image only as what was drawn with
        // rows of that length. It reads as many rows as it puts out (as many as its output
        // has before it is halved), not as many as the game says the source has: a game
        // takes a screen out of the upper part of a buffer by starting some rows in, and the
        // rows it says the source has then reach as far beyond the end of the buffer, into
        // what lies behind it. Taken as read, that made work on the buffer behind it needed
        // that was not, and where that work had been left out, it made a wrong guess of what
        // was none (Kid Icarus: Uprising in play, three times in ten minutes, each with all
        // work carried out for longer).
        const u32 bytes_per_pixel = Pica::BytesPerPixel(config.input_format);
        const u32 rows_read = config.output_height;
        const u32 last_row_read = std::min<u32>(config.output_width, config.input_width);
        const u32 pixels_read =
            rows_read == 0 ? 0 : (rows_read - 1) * config.input_width + last_row_read;
        source = {
            input, input + pixels_read * bytes_per_pixel,
            MemoryLayout::Make(MemoryLayout::Colour, config.input_width, bytes_per_pixel * 8)};
        destination = {output, output + width * height * Pica::BytesPerPixel(config.output_format)};
        step.kind = Kind::Transfer;
        step.replaces = true;
    }
    if (destination.Empty()) {
        return true;
    }
    step.writes[0] = destination;
    NoteWritten(destination);
    NoteWholeWrite(destination);
    AddRead(step, source);

    u64 base = Mix(static_cast<u64>(step.kind), (static_cast<u64>(input) << 32) | output);
    base = Mix(base, (static_cast<u64>(destination.end) << 1) | (step.replaces ? 1 : 0));
    step.key = NextOccurrence(base);
    anchor = step.key;

    // A copy with gaps leaves alone what is between the lines of an image, which is no part
    // of the image
    step.renews = true;
    const bool carried_out = Submit(step);
    if (step.shown) {
        // An image, or a part of one, is where it gets displayed from: what follows is counted
        // from here
        BeginStretch(base);
    }
    return carried_out;
}

bool FrameLiveness::ShouldAllowFill(const Pica::MemoryFillConfig& config) {
    just_left_out = false;
    if (!active) {
        return true;
    }
    Step step;
    step.kind = Kind::Fill;
    step.writes[0] = {config.GetStartAddress(), config.GetEndAddress()};
    step.replaces = true;
    step.renews = true;
    if (step.writes[0].Empty()) {
        return true;
    }
    NoteWritten(step.writes[0]);
    NoteWholeWrite(step.writes[0]);

    const u64 base = Mix(static_cast<u64>(Kind::Fill),
                         (static_cast<u64>(step.writes[0].start) << 32) | step.writes[0].end);
    step.key = NextOccurrence(base);
    anchor = step.key;
    return Submit(step);
}

bool FrameLiveness::Submit(Step& step) {
    events++;
    const auto kind = static_cast<std::size_t>(step.kind);
    step.shown = step.shown || IsShown(step.writes[0]);
    step.is_image = step.shown || IsImage(step.kind, step.writes[0]);

    bool leave_out = false;
    if (!step.shown) {
        const auto it = verdicts.find(step.key);
        if (it != verdicts.end()) {
            leave_out = distrust == 0 && it->second.unneeded_streak >= REQUIRED_UNNEEDED;
            if (it->second.kept_for_later) {
                kept_for_later[kind]++;
            }
        }
    }

    seen[kind]++;
    if (leave_out) {
        left_out[kind]++;
        stale_before = stale;
        just_left_out = true;
        for (const MemoryRange& write : step.writes) {
            stale.Add(write);
        }
    } else {
        CarryOut(step);
    }
    last_carried_out = !leave_out;

    // The same step again changes nothing about what is needed
    if (!current.empty() && current.back().key == step.key && current.back().shown == step.shown) {
        return !leave_out;
    }
    if (current.size() < MAX_STEPS) {
        current.push_back(step);
    } else {
        too_many_steps = true;
    }
    return !leave_out;
}

void FrameLiveness::CarryOut(const Step& step) {
    if (stale.Empty()) {
        return;
    }
    static constexpr std::array<const char*, static_cast<std::size_t>(Kind::Count)> names{
        "a draw for display", "a texture copy for display", "a display transfer to a screen",
        "a memory fill", "the game"};
    const auto kind = static_cast<std::size_t>(step.kind);
    const auto renew = [this, &step] {
        if (step.renews) {
            stale.Remove(step.writes[0]);
        }
        if (step.last_row != 0 && step.replaces_depth) {
            stale.Remove(step.writes[1]);
        }
    };
    if (step.last_row != 0) {
        // A draw: only the rows made from stale memory come out stale
        u32 first_stale = 0;
        u32 last_stale = 0;
        const MemoryRange* stale_for_rows = nullptr;
        if (!StaleRows(step, first_stale, last_stale, stale_for_rows)) {
            renew();
        } else if (step.shown) {
            Mispredicted(names[kind], *stale_for_rows);
        } else {
            renew();
            for (u32 buffer = 0; buffer < 2; buffer++) {
                if (!step.writes[buffer].Empty()) {
                    stale.Add(RowsOf(step, buffer, first_stale, last_stale));
                }
            }
        }
        return;
    }
    const auto stale_read = std::ranges::find_if(
        step.reads, [this](const MemoryRange& read) { return stale.Intersects(read); });
    if (stale_read == step.reads.end()) {
        renew();
    } else if (step.shown) {
        Mispredicted(names[kind], *stale_read);
    } else {
        // What is made from stale memory is stale as well. New work for an image nobody
        // sees does this until it is found not to be needed either.
        for (const MemoryRange& write : step.writes) {
            stale.Add(write);
        }
    }
}

void FrameLiveness::NoteGuestRead(PAddr start, u32 size) {
    if (instance == nullptr || !instance->active || size == 0) {
        return;
    }
    const u64 end = std::min<u64>(static_cast<u64>(start) + size, std::numeric_limits<u32>::max());
    instance->GuestRead(MemoryRange{start, static_cast<PAddr>(end)});
}

void FrameLiveness::GuestRead(const MemoryRange& range) {
    if (!stale.Empty() && stale.Intersects(range)) {
        Mispredicted("the game", range);
    }
    if (!WrittenIntersects(range)) {
        return;
    }
    if (!current.empty()) {
        const Step& last = current.back();
        // The same read again
        if (last.kind == Kind::GuestRead && last.reads[0] == range) {
            return;
        }
        // The step under way fetching what it reads, as a texture that is brought in from
        // guest memory: that read is noted with the step
        if (last_carried_out && last.kind != Kind::GuestRead &&
            std::ranges::any_of(
                last.reads, [&range](const MemoryRange& read) { return read.Overlaps(range); })) {
            return;
        }
    }
    Step step;
    step.kind = Kind::GuestRead;
    step.key = Mix(static_cast<u64>(Kind::GuestRead), 0);
    step.reads.push_back(range);
    step.shown = true;
    step.is_image = true;
    events++;
    seen[static_cast<std::size_t>(Kind::GuestRead)]++;
    if (current.size() < MAX_STEPS) {
        current.push_back(step);
    } else {
        too_many_steps = true;
    }
}

void FrameLiveness::NoteDisplay() {
    if (!active) {
        return;
    }
    events++;
    const auto& screens = gpu.impl->pica.regs.framebuffer_config;
    const bool right_eye =
        Settings::values.mono_render_option.GetValue() == Settings::MonoRenderOption::RightEye;
    for (u32 screen = 0; screen < 2; screen++) {
        const auto& framebuffer = screens[screen];
        const bool right = screen == 0 && right_eye && framebuffer.address_right1 != 0 &&
                           framebuffer.address_right2 != 0;
        const PAddr address =
            framebuffer.active_fb == 0
                ? (right ? framebuffer.address_right1 : framebuffer.address_left1)
                : (right ? framebuffer.address_right2 : framebuffer.address_left2);
        if (address == 0) {
            continue;
        }
        const MemoryRange place{address, address + framebuffer.stride * framebuffer.height};
        if (place.Empty()) {
            continue;
        }
        if (!stale.Empty() && stale.Intersects(place)) {
            Mispredicted("the display", place);
        }
        // Games swap between a few framebuffers: what goes into any of them is for display
        NotePlace(shown_places, place);
        if (screen == 0) {
            // The image of the other eye goes elsewhere, if the game has one
            const PAddr other =
                framebuffer.active_fb == 0
                    ? (right ? framebuffer.address_left1 : framebuffer.address_right1)
                    : (right ? framebuffer.address_left2 : framebuffer.address_right2);
            if (other != 0 && other != address) {
                NotePlace(unseen_places,
                          MemoryRange{other, other + framebuffer.stride * framebuffer.height});
            }
        }
    }
}

void FrameLiveness::NotePlace(Places& places, const MemoryRange& place) const {
    const auto known = std::ranges::find_if(
        places, [&place](const ShownPlace& other) { return other.range == place; });
    if (known != places.end()) {
        known->last_frame = frame;
    } else if (places.size() < MAX_SHOWN_PLACES) {
        places.push_back(ShownPlace{place, frame});
    } else {
        *std::ranges::min_element(places, {}, &ShownPlace::last_frame) = ShownPlace{place, frame};
    }
}

void FrameLiveness::Mispredicted(const char* reader, const MemoryRange& range) {
    // Wrong guesses that follow each other closely are a game doing now this, now that, and
    // each is met with twice the time of the one before. One that comes long after the last
    // (eight times the time that one was given) is a new situation, and starts from the
    // beginning: a game played for an hour would otherwise end up with all its work carried
    // out for minutes on end after every change of scene that goes wrong.
    if (frames_since_wrong_guess >= u64{last_distrust} * 8) {
        distrust_doublings = 0;
    }
    distrust = DISTRUST_FRAMES << std::min(distrust_doublings, MAX_DISTRUST_DOUBLINGS);
    distrust_doublings++;
    last_distrust = distrust;
    frames_since_wrong_guess = 0;
    wrong_guesses++;
    LOG_WARNING(HW_GPU,
                "Work that was left out turned out to be needed: {} reads {:08X}-{:08X}. "
                "Rendering everything for {} frames",
                reader, range.start, range.end, distrust);
    stale.Clear();
    events++;
    for (auto& [key, verdict] : verdicts) {
        verdict.unneeded_streak = 0;
    }
}

bool FrameLiveness::GoBack(bool images, std::vector<bool>& found) const {
    // A step goes into what is asked for if it is that itself, or if it writes what a step
    // after it that does so reads. What is read after the last step is what turned out to be
    // read before the first, as the frames after these look like these.
    const std::size_t total = previous.size() + current.size();
    found.assign(total, false);
    MemoryRanges live;
    for (int round = 0; round < MAX_ROUNDS; round++) {
        const MemoryRanges at_end = live;
        for (std::size_t index = total; index-- > 0;) {
            const Step& step = StepAt(index);
            if (step.last_row != 0) {
                // A draw: what it reads depends on which of its rows are needed
                u32 first = 0;
                u32 last = 0;
                found[index] = NeededRows(step, live, first, last);
                if (!found[index]) {
                    continue;
                }
                if (step.replaces) {
                    live.Remove(step.writes[0]);
                }
                if (step.replaces_depth) {
                    live.Remove(step.writes[1]);
                }
                AddReadsFor(step, first, last, live);
                continue;
            }
            found[index] = (images ? step.is_image : step.shown) ||
                           live.Intersects(step.writes[0]) || live.Intersects(step.writes[1]);
            if (!found[index]) {
                continue;
            }
            if (step.replaces) {
                live.Remove(step.writes[0]);
            }
            for (const MemoryRange& read : step.reads) {
                live.Add(read);
            }
        }
        if (live == at_end) {
            return true;
        }
    }
    return false;
}

bool FrameLiveness::IsFilledAgain(std::size_t index) const {
    const MemoryRange& filled = StepAt(index).writes[0];
    const std::size_t total = previous.size() + current.size();
    // What follows the last step is what the frame began with
    for (std::size_t after = 1; after <= current.size(); after++) {
        const std::size_t next = index + after;
        const Step& step = StepAt(next < total ? next : next - current.size());
        if (step.kind != Kind::Fill) {
            return false;
        }
        if (step.writes[0].start <= filled.start && step.writes[0].end >= filled.end) {
            return true;
        }
    }
    return false;
}

bool FrameLiveness::IsCopy(const Step& step) {
    if (step.kind != Kind::Draw) {
        return step.kind == Kind::Copy || step.kind == Kind::Transfer;
    }
    const u32 row_pixels = step.buffer_layout[0] & 0xFFFF;
    return !step.writes[0].Empty() &&
           (step.last_row - step.first_row) * row_pixels >= SCREEN_PIXELS &&
           std::ranges::any_of(step.links,
                               [](const Link& link) { return link.how != Link::How::Rows; });
}

void FrameLiveness::KeepForLater() {
    // What is made from an image nobody sees goes with it, though it goes into no image itself:
    // with the image left out there is nothing to make it from. Twice through the frames, for
    // what is made from the image of the frame before.
    const std::size_t total = needed.size();
    for_later.assign(total, false);
    MemoryRanges unseen;
    for (int round = 0; round < 2; round++) {
        for (std::size_t index = 0; index < total; index++) {
            const Step& step = StepAt(index);
            if (!needed[index] &&
                (in_image[index] ||
                 std::ranges::any_of(step.reads, [&unseen](const MemoryRange& read) {
                     return unseen.Intersects(read);
                 }))) {
                for_later[index] = false;
                for (const MemoryRange& write : step.writes) {
                    unseen.Add(write);
                }
                continue;
            }
            for_later[index] = !needed[index];
            if (step.renews) {
                unseen.Remove(step.writes[0]);
            }
            if (step.replaces_depth) {
                unseen.Remove(step.writes[1]);
            }
        }
    }
    // A copy stays only if what stays of the rest reads what it makes
    MemoryRanges read_later;
    for (std::size_t index = 0; index < total; index++) {
        const Step& step = StepAt(index);
        if (!for_later[index] || IsCopy(step)) {
            continue;
        }
        if (step.kind == Kind::Fill && IsFilledAgain(index)) {
            for_later[index] = false;
            continue;
        }
        for (const MemoryRange& read : step.reads) {
            read_later.Add(read);
        }
    }
    for (bool again = true; again;) {
        again = false;
        for (std::size_t index = 0; index < total; index++) {
            const Step& step = StepAt(index);
            if (!for_later[index] || needed[index] || !IsCopy(step) ||
                !read_later.Intersects(step.writes[0])) {
                continue;
            }
            needed[index] = true;
            again = true;
            for (const MemoryRange& read : step.reads) {
                read_later.Add(read);
            }
        }
    }
    for (std::size_t index = 0; index < total; index++) {
        if (for_later[index] && IsCopy(StepAt(index))) {
            for_later[index] = needed[index];
        } else if (for_later[index]) {
            needed[index] = true;
        }
    }
}

void FrameLiveness::LookAtLastFrames() {
    const auto start_over = [this] {
        for (auto& [key, verdict] : verdicts) {
            verdict.unneeded_streak = 0;
        }
    };
    if (too_many_steps || previous_too_many_steps) {
        start_over();
        return;
    }
    const std::size_t total = previous.size() + current.size();
    if (total == 0) {
        return;
    }
    // Frames repeat. What a look finds comes from the steps of the two frames alone, so two
    // frames made of the steps of two frames looked at before give what that look gave, for
    // each kind of work: entered again as it was found then.
    const u64 current_hash = HashOf(current);
    for (const KeptLook& kept : kept_looks) {
        if (!kept.valid || kept.after_hash != current_hash || kept.before_hash != previous_hash ||
            kept.before != previous || kept.after != current) {
            continue;
        }
        looks_repeated++;
        looks_in_a_row = 0;
        needed = kept.needed;
        for (const KeptLook::Outcome& outcome : kept.outcome) {
            Verdict& verdict = verdicts[outcome.key];
            verdict.last_frame = frame;
            verdict.kept_for_later = outcome.kept_for_later;
            if (outcome.needed) {
                verdict.unneeded_streak = 0;
            } else if (verdict.unneeded_streak < std::numeric_limits<u16>::max()) {
                verdict.unneeded_streak++;
            }
        }
        return;
    }
    looks_made++;
    looks_in_a_row = std::min<u32>(looks_in_a_row + 1, 1u << 20);
    const bool keep = previous.size() <= MOST_KEPT_STEPS && current.size() <= MOST_KEPT_STEPS &&
                      (looks_in_a_row <= 32 || looks_in_a_row % 8 == 0);
#ifdef CITRA_ENABLE_TRACE_MARKERS
    if (current != previous) {
        NoteDifference();
    }
#endif
    // A step is needed if it goes into what is on display or read by the game
    if (!GoBack(false, needed) || !GoBack(true, in_image)) {
        start_over();
        return;
    }
    KeepForLater();

    // A kind of work is not needed if none of its steps was
    look++;
    for (std::size_t index = 0; index < total; index++) {
        Verdict& verdict = verdicts[StepAt(index).key];
        if (verdict.look != look) {
            verdict.look = look;
            verdict.needed_in_look = false;
            verdict.kept_in_look = false;
            verdict.updated_in_look = false;
        }
        verdict.needed_in_look |= needed[index];
        verdict.kept_in_look |= for_later[index];
    }
    for (std::size_t index = 0; index < total; index++) {
        Verdict& verdict = verdicts[StepAt(index).key];
        if (verdict.updated_in_look) {
            continue;
        }
        verdict.updated_in_look = true;
        verdict.last_frame = frame;
        verdict.kept_for_later = verdict.kept_in_look;
        if (verdict.needed_in_look) {
            verdict.unneeded_streak = 0;
        } else if (verdict.unneeded_streak < std::numeric_limits<u16>::max()) {
            verdict.unneeded_streak++;
        }
    }
    if (!keep) {
        return;
    }
    KeptLook& kept = kept_looks[kept_next++ % KEPT_LOOKS];
    kept.before = previous;
    kept.after = current;
    kept.before_hash = previous_hash;
    kept.after_hash = current_hash;
    kept.needed = needed;
    kept.outcome.clear();
    // (Each kind of work once: the look number marks the ones entered)
    look++;
    for (std::size_t index = 0; index < total; index++) {
        const u64 key = StepAt(index).key;
        Verdict& verdict = verdicts[key];
        if (verdict.look != look) {
            verdict.look = look;
            kept.outcome.push_back({key, verdict.needed_in_look, verdict.kept_in_look});
        }
    }
    kept.valid = true;
}

u64 FrameLiveness::HashOf(const std::vector<Step>& steps) {
    u64 hash = steps.size();
    for (const Step& step : steps) {
        hash = Mix(hash, step.key ^ (step.shown ? 1 : 0));
        hash = Mix(hash, (static_cast<u64>(step.writes[0].start) << 32) | step.writes[0].end);
    }
    return hash;
}

#ifdef CITRA_ENABLE_TRACE_MARKERS
void FrameLiveness::NoteDifference() {
    difference_steps = {static_cast<u32>(previous.size()), static_cast<u32>(current.size())};
    difference_index = 0;
    difference_kind = Kind::Count;
    difference_what = "the number of steps";
    const std::size_t both = std::min(previous.size(), current.size());
    for (std::size_t index = 0; index < both; index++) {
        const Step& a = previous[index];
        const Step& b = current[index];
        if (a == b) {
            continue;
        }
        difference_index = static_cast<u32>(index);
        difference_kind = b.kind;
        difference_what = a.kind != b.kind ? "the kind of work"
                          : a.key != b.key
                              ? (a.writes == b.writes && a.reads == b.reads ? "the key alone"
                                 : a.writes != b.writes ? "the key and what is written"
                                                        : "the key and what is read")
                          : a.shown != b.shown   ? "whether it is shown"
                          : a.reads != b.reads   ? "what it reads"
                          : a.links != b.links   ? "how its reads go with its rows"
                          : a.writes != b.writes ? "what it writes"
                          : (a.first_row != b.first_row || a.last_row != b.last_row) ? "its rows"
                                                                                     : "its flags";
        return;
    }
}

void FrameLiveness::LogOutline() const {
    const std::size_t offset = previous.size();
    if (needed.size() != offset + current.size()) {
        return;
    }
    static constexpr std::array<const char*, static_cast<std::size_t>(Kind::Count)> names{
        "draws", "copy", "transfer", "fill", "read by the game"};
    const auto index = [](Kind kind) { return static_cast<std::size_t>(kind); };
    LOG_INFO(HW_GPU,
             "Unseen work, a frame in outline: a transfer is left out but none of the {} draws "
             "({} fills, {} transfers, {} copies)",
             seen[index(Kind::Draw)], seen[index(Kind::Fill)], seen[index(Kind::Transfer)],
             seen[index(Kind::Copy)]);
    u32 lines = 0;
    for (std::size_t i = 0; i < current.size() && lines < 28; lines++) {
        const Step& step = current[i];
        if (step.kind == Kind::Draw) {
            // Draws one after the other to the same buffers, as one line
            std::size_t end = i;
            u32 needed_ones = 0;
            u32 replacing = 0;
            u32 replacing_depth = 0;
            while (end < current.size() && current[end].kind == Kind::Draw &&
                   current[end].buffer_start == step.buffer_start) {
                needed_ones += needed[offset + end] ? 1 : 0;
                replacing += current[end].replaces ? 1 : 0;
                replacing_depth += current[end].replaces_depth ? 1 : 0;
                end++;
            }
            LOG_INFO(HW_GPU,
                     "Unseen work, outline: {} kinds of draw to colour {:08X}, depth {:08X}: {} "
                     "needed, {} replace the colour the target held and {} its depth, the first "
                     "writes {:08X}-{:08X}",
                     end - i, step.buffer_start[0], step.buffer_start[1], needed_ones, replacing,
                     replacing_depth, step.writes[0].start, step.writes[0].end);
            i = end;
            continue;
        }
        const auto verdict = verdicts.find(step.key);
        const bool out =
            verdict != verdicts.end() && verdict->second.unneeded_streak >= REQUIRED_UNNEEDED;
        const MemoryRange read = step.reads.empty() ? MemoryRange{} : step.reads[0];
        LOG_INFO(HW_GPU, "Unseen work, outline: {} from {:08X}-{:08X} to {:08X}-{:08X}: {}{}",
                 names[index(step.kind)], read.start, read.end, step.writes[0].start,
                 step.writes[0].end,
                 step.shown           ? "shown"
                 : needed[offset + i] ? "needed"
                                      : "not needed",
                 out ? ", left out" : "");
        i++;
    }
}
#endif

void FrameLiveness::EndFrame() {
    UpdateActive();
    if (!active) {
        return;
    }
    frame++;
    events++;
    LookAtLastFrames();

    constexpr auto index = [](Kind kind) { return static_cast<std::size_t>(kind); };
    const u32 total_left_out = left_out[index(Kind::Draw)] + left_out[index(Kind::Copy)] +
                               left_out[index(Kind::Transfer)] + left_out[index(Kind::Fill)];
    if ((total_left_out != 0) == leaving_out) {
        leaving_out_changing -= std::min(leaving_out_changing, 1u);
    } else if (++leaving_out_changing >= CHANGE_HELD_FOR) {
        leaving_out_changing = 0;
        leaving_out = total_left_out != 0;
        if (leaving_out) {
            LOG_INFO(HW_GPU,
                     "Leaving out work nobody sees: {} of {} draws, {} of {} texture copies, "
                     "{} of {} display transfers and {} of {} memory fills in a frame",
                     left_out[index(Kind::Draw)], seen[index(Kind::Draw)],
                     left_out[index(Kind::Copy)], seen[index(Kind::Copy)],
                     left_out[index(Kind::Transfer)], seen[index(Kind::Transfer)],
                     left_out[index(Kind::Fill)], seen[index(Kind::Fill)]);
        } else {
            LOG_INFO(HW_GPU, "No work is left out any more");
        }
    }
#ifdef CITRA_ENABLE_TRACE_MARKERS
    if (++frames_since_status >= 300) {
        frames_since_status = 0;
        LOG_INFO(HW_GPU,
                 "Unseen work: left out {} of {} draws, {} of {} copies, {} of {} transfers, "
                 "{} of {} fills; not left out though nothing reads it: {} draws, {} copies, {} "
                 "fills; {} reads by the game, {} kinds of work known, {} stale "
                 "places, {} frames of distrust left, {} wrong guesses; in the last 300 frames "
                 "{} of {} draws were the draw before again ({} compared, {} of them another "
                 "one after all), {} of {} looks at the last frames were the look before again",
                 left_out[index(Kind::Draw)], seen[index(Kind::Draw)], left_out[index(Kind::Copy)],
                 seen[index(Kind::Copy)], left_out[index(Kind::Transfer)],
                 seen[index(Kind::Transfer)], left_out[index(Kind::Fill)], seen[index(Kind::Fill)],
                 kept_for_later[index(Kind::Draw)], kept_for_later[index(Kind::Copy)],
                 kept_for_later[index(Kind::Fill)], seen[index(Kind::GuestRead)], verdicts.size(),
                 stale.Size(), distrust, wrong_guesses, draws_repeated, draws_asked,
                 memo_checks / 64, memo_differences, looks_repeated, looks_repeated + looks_made);
        static constexpr std::array<const char*, static_cast<std::size_t>(Kind::Count) + 1> kinds{
            "a draw", "a copy", "a transfer", "a fill", "a read by the game", "no step"};
        LOG_INFO(HW_GPU,
                 "Unseen work, draws not taken for the one before: {} sample what the GPU "
                 "wrote, {} follow one that does, {} follow other work, {} are other draws. "
                 "The last frame unlike the one before it: {} steps after {}, unlike from "
                 "step {} ({}) in {}",
                 memo_sampling, memo_after_sampling, memo_after_other, memo_other_signature,
                 difference_steps[1], difference_steps[0], difference_index,
                 kinds[static_cast<std::size_t>(difference_kind)], difference_what);
        memo_sampling = 0;
        memo_after_sampling = 0;
        memo_after_other = 0;
        memo_other_signature = 0;
        draws_repeated = 0;
        draws_asked = 0;
        looks_repeated = 0;
        looks_made = 0;
    }
    // A game whose image for one eye goes nowhere, the transfer of it is left out, and yet
    // every draw is carried out: something after it reads what those draws leave behind.
    // What a frame of it is made of is written down once it has been so for a while, and once
    // more much later (play, after a menu), for a log that is sent in to tell which case it is.
    const bool draws_kept = distrust == 0 && left_out[index(Kind::Transfer)] != 0 &&
                            left_out[index(Kind::Draw)] == 0 && seen[index(Kind::Draw)] >= 20;
    outline_streak = draws_kept ? outline_streak + 1 : std::min<s32>(outline_streak, 0);
    if (outline_streak >= 120 && outlines_logged < 2) {
        outlines_logged++;
        outline_streak = -3600;
        LogOutline();
    }
#endif

    if (distrust != 0) {
        distrust--;
    }
    frames_since_wrong_guess = std::min<u64>(frames_since_wrong_guess + 1, u64{1} << 40);
    if (frame % 256 == 0) {
        std::erase_if(verdicts, [this](const auto& entry) {
            return frame - entry.second.last_frame > FORGET_AFTER;
        });
        for (Places* places : {&shown_places, &unseen_places}) {
            places->erase(std::remove_if(places->begin(), places->end(),
                                         [this](const ShownPlace& place) {
                                             return frame - place.last_frame > FORGET_AFTER;
                                         }),
                          places->end());
        }
    }

    previous_hash = HashOf(current);
    previous.swap(current);
    current.clear();
    previous_too_many_steps = too_many_steps;
    too_many_steps = false;
    last_carried_out = false;
    // Where a frame ends in what a game does is not fixed: a swap is taken up at the VBlank,
    // and a game that is late for one, or that renders without waiting for them, is somewhere
    // else in its frame each time. Counting from here would name the same work differently
    // from one frame to the next (and the left eye as the right one was named the frame
    // before). Only a game that puts nothing in a place for display is counted from here.
    const bool counted_from_display = counts_from_display;
    counts_from_display =
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyWorkNaming) == 0;
    if (!stretch_began_in_frame || counted_from_display != counts_from_display) {
        anchor = 0;
        stretch = 0;
        occurrences.clear();
    }
    stretch_began_in_frame = false;
    seen.fill(0);
    left_out.fill(0);
    kept_for_later.fill(0);
}

} // namespace VideoCore
