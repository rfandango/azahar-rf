// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
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
/// Frames to render everything for after the first wrong guess, doubled with each one after
constexpr u32 DISTRUST_FRAMES = 600;
constexpr u32 MAX_DISTRUST_DOUBLINGS = 5;
/// A frame with more steps than this is not looked at
constexpr std::size_t MAX_STEPS = 4096;
/// Frames after which work and framebuffers that did not come up again are forgotten
constexpr u32 FORGET_AFTER = 600;
/// Times the last frames are gone through until what is needed no longer changes
constexpr int MAX_ROUNDS = 8;
constexpr std::size_t MAX_SHOWN_PLACES = 16;

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
    occurrences.clear();
    verdicts.clear();
    written.Clear();
    stale.Clear();
    shown_places.clear();
    seen.fill(0);
    left_out.fill(0);
    leaving_out = false;
    distrust = 0;
}

bool FrameLiveness::IsShown(const MemoryRange& range) const {
    return !range.Empty() && std::ranges::any_of(shown_places, [&range](const ShownPlace& place) {
        return place.range.Overlaps(range);
    });
}

void FrameLiveness::AddRead(Step& step, const MemoryRange& range) const {
    // Memory the GPU never wrote holds what the game put there, which is always as it should
    if (range.Empty() || !written.Intersects(range)) {
        return;
    }
    ASSERT(step.reads.size() < step.reads.capacity());
    step.reads.push_back(range);
}

u64 FrameLiveness::NextOccurrence(u64 base) {
    for (auto& [key, count] : occurrences) {
        if (key == base) {
            return Mix(base, ++count);
        }
    }
    occurrences.emplace_back(base, 0);
    return Mix(base, 0);
}

bool FrameLiveness::MayReplaceTarget(const Pica::RegsInternal& regs) const {
    if (!active) {
        return false;
    }
    using Pica::FramebufferRegs;
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& merger = regs.framebuffer.output_merger;

    // Two triangles at the most
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

bool FrameLiveness::ShouldAllowDraw(const Pica::RegsInternal& regs, bool replaces_target,
                                    bool may_leave_out) {
    if (!active) {
        return true;
    }
    using Pica::FramebufferRegs;
    using Pica::TexturingRegs;
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& merger = regs.framebuffer.output_merger;
    const u32 pixels = framebuffer.GetWidth() * framebuffer.GetHeight();
    const bool shadow = regs.framebuffer.IsShadowRendering();
    const bool stencil = regs.framebuffer.HasStencil() && merger.stencil_test.enable;

    Step step;
    step.kind = Kind::Draw;
    // What cannot be left out counts as needed, and with it all that it reads
    step.shown = !may_leave_out;

    const PAddr colour = framebuffer.GetColorBufferPhysicalAddress();
    const bool writes_colour = colour != 0 && framebuffer.allow_color_write != 0 &&
                               (shadow || merger.red_enable || merger.green_enable ||
                                merger.blue_enable || merger.alpha_enable);
    MemoryRange colour_range;
    if (writes_colour) {
        const u32 bytes_per_pixel = FramebufferRegs::BytesPerColorPixel(framebuffer.color_format);
        colour_range = {
            colour, colour + pixels * bytes_per_pixel,
            MemoryLayout::Make(MemoryLayout::Colour, framebuffer.GetWidth(), bytes_per_pixel * 8)};
        step.writes[0] = colour_range;
        step.replaces = replaces_target;
        step.renews = replaces_target;
        written.Add(MemoryRange{colour_range.start, colour_range.end});
    }

    const PAddr depth = framebuffer.GetDepthBufferPhysicalAddress();
    const bool writes_depth =
        (merger.depth_write_enable && framebuffer.allow_depth_stencil_write != 0) || stencil;
    const bool uses_depth = depth != 0 && !shadow && (merger.depth_test_enable || writes_depth);
    MemoryRange depth_range;
    if (uses_depth) {
        const u32 bytes_per_pixel = FramebufferRegs::BytesPerDepthPixel(framebuffer.depth_format);
        depth_range = {
            depth, depth + pixels * bytes_per_pixel,
            MemoryLayout::Make(MemoryLayout::Depth, framebuffer.GetWidth(), bytes_per_pixel * 8)};
        if (writes_depth) {
            step.writes[1] = depth_range;
            written.Add(MemoryRange{depth_range.start, depth_range.end});
        }
    }

    // What the target holds goes into the result, unless all of it is replaced. That is what
    // was drawn there as a surface with rows of this length: the same memory is another
    // image as anything else, and no game shows that.
    if (writes_colour && !step.replaces) {
        AddRead(step, colour_range);
    }
    AddRead(step, depth_range);

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
            AddRead(step, MemoryRange{address, address + size, layout});
        }
    }

    u64 key = Mix(static_cast<u64>(Kind::Draw), anchor);
    key = Mix(key, (static_cast<u64>(colour_range.start) << 32) | colour_range.end);
    key = Mix(key, colour_range.layout);
    key = Mix(key, (static_cast<u64>(depth_range.start) << 1) | (writes_depth ? 1 : 0));
    key = Mix(key, (step.replaces ? 1 : 0) | (may_leave_out ? 0 : 2));
    for (const MemoryRange& read : step.reads) {
        key = Mix(key, read.start);
    }
    step.key = key;
    return Submit(step);
}

bool FrameLiveness::ShouldAllowTransfer(const Pica::DisplayTransferConfig& config) {
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
        // rows of that length
        const u32 bytes_per_pixel = Pica::BytesPerPixel(config.input_format);
        source = {
            input, input + config.input_width * config.input_height * bytes_per_pixel,
            MemoryLayout::Make(MemoryLayout::Colour, config.input_width, bytes_per_pixel * 8)};
        destination = {output, output + width * height * Pica::BytesPerPixel(config.output_format)};
        step.kind = Kind::Transfer;
        step.replaces = true;
    }
    if (destination.Empty()) {
        return true;
    }
    step.writes[0] = destination;
    written.Add(destination);
    AddRead(step, source);

    u64 base = Mix(static_cast<u64>(step.kind), (static_cast<u64>(input) << 32) | output);
    base = Mix(base, (static_cast<u64>(destination.end) << 1) | (step.replaces ? 1 : 0));
    step.key = NextOccurrence(base);
    anchor = step.key;

    // A copy with gaps leaves alone what is between the lines of an image, which is no part
    // of the image
    step.renews = true;
    return Submit(step);
}

bool FrameLiveness::ShouldAllowFill(const Pica::MemoryFillConfig& config) {
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
    written.Add(step.writes[0]);

    const u64 base = Mix(static_cast<u64>(Kind::Fill),
                         (static_cast<u64>(step.writes[0].start) << 32) | step.writes[0].end);
    step.key = NextOccurrence(base);
    anchor = step.key;
    return Submit(step);
}

bool FrameLiveness::Submit(Step& step) {
    const auto kind = static_cast<std::size_t>(step.kind);
    step.shown = step.shown || IsShown(step.writes[0]);

    bool leave_out = false;
    if (distrust == 0 && !step.shown) {
        const auto it = verdicts.find(step.key);
        leave_out = it != verdicts.end() && it->second.unneeded_streak >= REQUIRED_UNNEEDED;
    }

    seen[kind]++;
    if (leave_out) {
        left_out[kind]++;
        for (const MemoryRange& write : step.writes) {
            stale.Add(write);
        }
    } else if (!stale.Empty()) {
        static constexpr std::array<const char*, static_cast<std::size_t>(Kind::Count)> names{
            "a draw for display", "a texture copy for display", "a display transfer to a screen",
            "a memory fill", "the game"};
        const auto stale_read = std::ranges::find_if(
            step.reads, [this](const MemoryRange& read) { return stale.Intersects(read); });
        if (stale_read == step.reads.end()) {
            if (step.renews) {
                stale.Remove(step.writes[0]);
            }
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
    if (!written.Intersects(range)) {
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
        const auto known = std::ranges::find_if(
            shown_places, [&place](const ShownPlace& other) { return other.range == place; });
        if (known != shown_places.end()) {
            known->last_frame = frame;
        } else if (shown_places.size() < MAX_SHOWN_PLACES) {
            shown_places.push_back(ShownPlace{place, frame});
        } else {
            *std::ranges::min_element(shown_places, {}, &ShownPlace::last_frame) =
                ShownPlace{place, frame};
        }
    }
}

void FrameLiveness::Mispredicted(const char* reader, const MemoryRange& range) {
    distrust = DISTRUST_FRAMES << std::min(wrong_guesses, MAX_DISTRUST_DOUBLINGS);
    wrong_guesses++;
    LOG_WARNING(HW_GPU,
                "Work that was left out turned out to be needed: {} reads {:08X}-{:08X}. "
                "Rendering everything for {} frames",
                reader, range.start, range.end, distrust);
    stale.Clear();
    for (auto& [key, verdict] : verdicts) {
        verdict.unneeded_streak = 0;
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
    const auto step_at = [this](std::size_t index) -> const Step& {
        return index < previous.size() ? previous[index] : current[index - previous.size()];
    };

    // Backwards from the end: a step is needed if it is for display or the game, or if it
    // writes what a needed step after it reads. What is needed after the last step is what
    // turned out to be needed before the first, as the frames after these look like these.
    needed.assign(total, false);
    MemoryRanges live;
    bool settled = false;
    for (int round = 0; round < MAX_ROUNDS && !settled; round++) {
        const MemoryRanges at_end = live;
        for (std::size_t index = total; index-- > 0;) {
            const Step& step = step_at(index);
            const bool is_needed =
                step.shown || live.Intersects(step.writes[0]) || live.Intersects(step.writes[1]);
            needed[index] = is_needed;
            if (!is_needed) {
                continue;
            }
            if (step.replaces) {
                live.Remove(step.writes[0]);
            }
            for (const MemoryRange& read : step.reads) {
                live.Add(read);
            }
        }
        settled = live == at_end;
    }
    if (!settled) {
        start_over();
        return;
    }

    // A kind of work is not needed if none of its steps was
    look++;
    for (std::size_t index = 0; index < total; index++) {
        Verdict& verdict = verdicts[step_at(index).key];
        if (verdict.look != look) {
            verdict.look = look;
            verdict.needed_in_look = false;
            verdict.updated_in_look = false;
        }
        verdict.needed_in_look |= needed[index];
    }
    for (std::size_t index = 0; index < total; index++) {
        Verdict& verdict = verdicts[step_at(index).key];
        if (verdict.updated_in_look) {
            continue;
        }
        verdict.updated_in_look = true;
        verdict.last_frame = frame;
        if (verdict.needed_in_look) {
            verdict.unneeded_streak = 0;
        } else if (verdict.unneeded_streak < std::numeric_limits<u16>::max()) {
            verdict.unneeded_streak++;
        }
    }
}

void FrameLiveness::EndFrame() {
    UpdateActive();
    if (!active) {
        return;
    }
    frame++;
    LookAtLastFrames();

    constexpr auto index = [](Kind kind) { return static_cast<std::size_t>(kind); };
    const u32 total_left_out = left_out[index(Kind::Draw)] + left_out[index(Kind::Copy)] +
                               left_out[index(Kind::Transfer)] + left_out[index(Kind::Fill)];
    if ((total_left_out != 0) != leaving_out) {
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
                 "{} of {} fills; {} reads by the game, {} kinds of work known, {} stale "
                 "places, {} frames of distrust left, {} wrong guesses",
                 left_out[index(Kind::Draw)], seen[index(Kind::Draw)], left_out[index(Kind::Copy)],
                 seen[index(Kind::Copy)], left_out[index(Kind::Transfer)],
                 seen[index(Kind::Transfer)], left_out[index(Kind::Fill)], seen[index(Kind::Fill)],
                 seen[index(Kind::GuestRead)], verdicts.size(), stale.Size(), distrust,
                 wrong_guesses);
    }
#endif

    if (distrust != 0) {
        distrust--;
    }
    if (frame % 256 == 0) {
        std::erase_if(verdicts, [this](const auto& entry) {
            return frame - entry.second.last_frame > FORGET_AFTER;
        });
        shown_places.erase(std::remove_if(shown_places.begin(), shown_places.end(),
                                          [this](const ShownPlace& place) {
                                              return frame - place.last_frame > FORGET_AFTER;
                                          }),
                           shown_places.end());
    }

    previous.swap(current);
    current.clear();
    previous_too_many_steps = too_many_steps;
    too_many_steps = false;
    last_carried_out = false;
    anchor = 0;
    occurrences.clear();
    seen.fill(0);
    left_out.fill(0);
}

} // namespace VideoCore
