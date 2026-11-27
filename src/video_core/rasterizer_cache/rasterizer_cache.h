// Copyright 2015-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <type_traits>
#include <boost/container/small_vector.hpp>
#include <boost/range/iterator_range.hpp>
#include "common/alignment.h"
#include "common/hitch_tracker.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/trace_markers.h"
#include "core/memory.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/frame_liveness.h"
#include "video_core/frame_state_dump.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/renderer_base.h"
#include "video_core/texture/texture_decode.h"

namespace VideoCore {

MICROPROFILE_DECLARE(RasterizerCache_CopySurface);
MICROPROFILE_DECLARE(RasterizerCache_UploadSurface);
MICROPROFILE_DECLARE(RasterizerCache_DownloadSurface);
MICROPROFILE_DECLARE(RasterizerCache_Invalidation);

constexpr auto RangeFromInterval(const auto& map, const auto& interval) {
    return boost::make_iterator_range(map.equal_range(interval));
}

/// Whether two rectangles of host pixels can be copied one to the other. Those of the same guest
/// pixels in two surfaces of one scale mostly can; where the scale is not a whole number and the
/// rectangle not one of whole tiles, its place decides whether it has a host pixel more or less.
inline bool HasSameSize(const Common::Rectangle<u32>& a, const Common::Rectangle<u32>& b) {
    return a.GetWidth() == b.GetWidth() && a.GetHeight() == b.GetHeight();
}

template <class T>
RasterizerCache<T>::RasterizerCache(Memory::MemorySystem& memory_,
                                    CustomTexManager& custom_tex_manager_, Runtime& runtime_,
                                    Pica::RegsInternal& regs_, RendererBase& renderer_)
    : memory{memory_}, custom_tex_manager{custom_tex_manager_}, runtime{runtime_}, regs{regs_},
      renderer{renderer_}, resolution_scale_factor{renderer.GetResolutionScaleFactor()},
      sample_count{renderer.GetSampleCount()}, filter{Settings::values.texture_filter.GetValue()},
      dump_textures{Settings::values.dump_textures.GetValue()},
      use_custom_textures{Settings::values.custom_textures.GetValue()} {
    using TextureConfig = Pica::TexturingRegs::TextureConfig;

    // Create null handles for all cached resources
    void(slot_surfaces.insert(runtime, SurfaceParams{
                                           .width = 1,
                                           .height = 1,
                                           .stride = 1,
                                           .texture_type = VideoCore::TextureType::Texture2D,
                                           .pixel_format = VideoCore::PixelFormat::RGBA8,
                                           .type = VideoCore::SurfaceType::Color,
                                       }));
    void(slot_surfaces.insert(runtime, SurfaceParams{
                                           .width = 1,
                                           .height = 1,
                                           .stride = 1,
                                           .texture_type = TextureType::CubeMap,
                                           .pixel_format = PixelFormat::RGBA8,
                                           .type = SurfaceType::Color,
                                       }));
    void(slot_samplers.insert(runtime, SamplerParams{
                                           .mag_filter = TextureConfig::TextureFilter::Linear,
                                           .min_filter = TextureConfig::TextureFilter::Linear,
                                           .mip_filter = TextureConfig::TextureFilter::Linear,
                                           .wrap_s = TextureConfig::WrapMode::ClampToBorder,
                                           .wrap_t = TextureConfig::WrapMode::ClampToBorder,
                                       }));

    auto& null_surface = slot_surfaces[NULL_SURFACE_ID];
    runtime.ClearTexture(null_surface, {
                                           .texture_level = 0,
                                           .texture_layer = 0,
                                           .texture_rect = null_surface.GetScaledRect(),
                                           .value =
                                               {
                                                   .color = {0.f, 0.f, 0.f, 0.f},
                                               },
                                       });

    auto& null_surface_cube = slot_surfaces[NULL_SURFACE_CUBE_ID];
    // Clear all cubemap faces
    for (u32 cube_layer_index = 0; cube_layer_index < 6; ++cube_layer_index) {
        runtime.ClearTexture(null_surface_cube,
                             {
                                 .texture_level = 0,
                                 .texture_layer = cube_layer_index,
                                 .texture_rect = null_surface_cube.GetScaledRect(),
                                 .value =
                                     {
                                         .color = {0.f, 0.f, 0.f, 0.f},
                                     },
                             });
    }
}

template <class T>
RasterizerCache<T>::~RasterizerCache() {
    ClearAll(false);
}

template <class T>
void RasterizerCache<T>::TickFrame() {
    // What was looked up holds for a frame at the most: a texture whose replacement was
    // still being loaded is looked at again
    surface_set_epoch++;
    late_frame++;
    ServeLateReadbacks();
    custom_tex_manager.TickFrame();
    RunGarbageCollector();
#ifdef CITRA_ENABLE_TRACE_MARKERS
    if (late_frame % 300 == 0) {
        LOG_INFO(HW_GPU,
                 "Surfaces in 300 VBlanks: {} taken over, {} of which the GPU was not through "
                 "with, {} targets made new ({} of them while one like it was given up), {} given "
                 "up and kept",
                 taken_over + taken_over_early, taken_over_early, made_new, made_new_waiting,
                 sentenced.size());
        taken_over = taken_over_early = made_new = made_new_waiting = 0;
    }
#endif

    const auto new_filter = Settings::values.texture_filter.GetValue();
    if (filter != new_filter) [[unlikely]] {
        filter = new_filter;
        UnregisterAll();
    }

    const Common::ResolutionScale scale_factor = renderer.GetResolutionScaleFactor();
    const u32 samples = renderer.GetSampleCount();
    const bool resolution_scale_changed = resolution_scale_factor != scale_factor;
    const bool sample_count_changed = sample_count != samples;
    const bool use_custom_texture_changed =
        Settings::values.custom_textures.GetValue() != use_custom_textures;

    if (resolution_scale_changed || use_custom_texture_changed || sample_count_changed) {
        resolution_scale_factor = scale_factor;
        sample_count = renderer.GetSampleCount();
        use_custom_textures = Settings::values.custom_textures.GetValue();
        if (use_custom_textures) {
            custom_tex_manager.FindCustomTextures();
        }
        UnregisterAll();
    }
}

template <class T>
void RasterizerCache<T>::Sentence(SurfaceId surface_id) {
    sentenced.push_back(Sentenced{
        .surface_id = surface_id,
        .last_used_tick = runtime.GetResourceTick(),
        .frame = late_frame,
    });
}

template <class T>
void RasterizerCache<T>::RunGarbageCollector(bool at_once) {
    /// Frames (TickFrame, one a VBlank) a sentenced surface is kept for CreateSurface to take
    /// over: a game at 15 fps asks for a target again four VBlanks after it gave it up
    static constexpr u64 SENTENCED_FRAMES = 8;
    const u64 resource_free_tick = runtime.GetResourceFreeTick();
    for (auto it = sentenced.begin(); it != sentenced.end();) {
        const SurfaceId surface_id = it->surface_id;
        const u64 resource_last_used_tick = it->last_used_tick;
        // Tick-values less than or equal to the current resource-tick are possibly still in-use.
        // Tick-values larger than the current resource-tick are not being utilized anymore and can
        // be safely deleted.
        if (resource_free_tick <= resource_last_used_tick ||
            (!at_once && late_frame - it->frame < SENTENCED_FRAMES) ||
            runtime.StillNeeded(slot_surfaces[surface_id])) {
            // Resource is still in-use, skip deletion
            it++;
            continue;
        }
        RemoveFramebuffers(surface_id);
        slot_surfaces.erase(surface_id);
        it = sentenced.erase(it);
    }
}

template <class T>
void RasterizerCache<T>::RemoveFramebuffers(SurfaceId surface_id) {
    for (auto it = framebuffers.begin(); it != framebuffers.end();) {
        const auto& params = it->first;
        if (params.color_id == surface_id || params.depth_id == surface_id) {
            slot_framebuffers.erase(it->second);
            it = framebuffers.erase(it);
        } else {
            it++;
        }
    }
}

template <class T>
void RasterizerCache<T>::RemoveTextureCubeFace(SurfaceId surface_id) {
    if (False(slot_surfaces[surface_id].flags & SurfaceFlagBits::Tracked)) {
        return;
    }

    for (auto it = texture_cube_cache.begin(); it != texture_cube_cache.end();) {
        TextureCube& cube = it->second;
        for (SurfaceId& face_id : cube.face_ids) {
            if (face_id == surface_id) {
                face_id = SurfaceId{};
            }
        }
        if (std::none_of(cube.face_ids.begin(), cube.face_ids.end(),
                         [](SurfaceId id) { return id; })) {
            Sentence(cube.surface_id);
            it = texture_cube_cache.erase(it);
        } else {
            it++;
        }
    }
}

template <class T>
bool RasterizerCache<T>::AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) {
    const DebugScope scope{runtime, Common::Vec4f{0.f, 0.f, 1.f, 1.f},
                           "RasterizerCache::AccelerateTextureCopy ({})", config.DebugName()};

    // Texture copy size is aligned to 16 byte units
    const u32 copy_size = Common::AlignDown(config.texture_copy.size, 16);
    if (copy_size == 0) {
        return false;
    }

    u32 input_gap = config.texture_copy.input_gap * 16;
    u32 input_width = config.texture_copy.input_width * 16;
    if (input_width == 0 && input_gap != 0) {
        return false;
    }
    if (input_gap == 0 || input_width >= copy_size) {
        input_width = copy_size;
        input_gap = 0;
    }
    if (copy_size % input_width != 0) {
        return false;
    }

    u32 output_gap = config.texture_copy.output_gap * 16;
    u32 output_width = config.texture_copy.output_width * 16;
    if (output_width == 0 && output_gap != 0) {
        return false;
    }
    if (output_gap == 0 || output_width >= copy_size) {
        output_width = copy_size;
        output_gap = 0;
    }
    if (copy_size % output_width != 0) {
        return false;
    }

    SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.stride = input_width + input_gap; // stride in bytes
    src_params.width = input_width;              // width in bytes
    src_params.height = copy_size / input_width;
    src_params.size = ((src_params.height - 1) * src_params.stride) + src_params.width;
    src_params.end = src_params.addr + src_params.size;

    const auto [src_surface_id, src_subrect] = GetTexCopySurface(src_params);
    if (!src_surface_id) {
        return false;
    }

    const SurfaceParams src_info = slot_surfaces[src_surface_id];
    const Common::Rectangle<u32> src_rect = src_info.Scaling().Rect(src_subrect);
    NoteFullSizeUse(src_info);
    const u32 copy_width = src_subrect.GetWidth();
    const u32 copy_height = src_subrect.GetHeight();
    if (output_gap != 0 &&
        (output_width != src_info.BytesInPixels(copy_width) * (src_info.is_tiled ? 8 : 1) ||
         output_gap % src_info.BytesInPixels(src_info.is_tiled ? 64 : 1) != 0)) {
        return false;
    }

    SurfaceParams dst_params = src_info;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = copy_width;
    dst_params.stride =
        dst_params.width + src_info.PixelsInBytes(src_info.is_tiled ? output_gap / 8 : output_gap);
    dst_params.height = copy_height;
    dst_params.res_scale = src_info.res_scale;
    dst_params.UpdateParams();

    auto [dst_surface_id, dst_rect] = GetSurfaceSubRect(dst_params, ScaleMatch::Upscale, false);
    if (!dst_surface_id) {
        return false;
    }
    if (output_gap != 0) {
        // Images are put together out of columns. The column that comes first after a save state
        // was loaded, or after everything was dropped, makes an image that begins with that
        // column, in the middle of the rows of the image it is a part of. Both hold the column
        // from then on: it goes into the one that begins first and has the others as well, or
        // the columns would be in two images that cannot be copied to each other other than row
        // by row, and what is shown would be read back from the GPU to be loaded again.
        const u32 row_bytes = dst_params.BytesInPixels(dst_params.stride) *
                              (dst_params.is_tiled ? 8 : 1);
        SurfaceId first_id = dst_surface_id;
        ForEachSurfaceInRegion(
            dst_params.addr, dst_params.size, [&](SurfaceId surface_id, Surface& surface) {
                const Surface& first = slot_surfaces[first_id];
                if (surface.addr < first.addr && surface.levels == 1 &&
                    surface.type != SurfaceType::Texture && surface.stride == first.stride &&
                    surface.res_scale == first.res_scale &&
                    surface.half_scale_x == first.half_scale_x &&
                    surface.half_scale_y == first.half_scale_y &&
                    surface.sample_count == first.sample_count &&
                    (first.addr - surface.addr) % row_bytes != 0 &&
                    surface.CanSubRect(dst_params)) {
                    first_id = surface_id;
                }
            });
        if (first_id != dst_surface_id) {
            dst_surface_id = first_id;
            dst_rect = slot_surfaces[first_id].GetScaledSubRect(dst_params);
        }
    }

    // A copy with an output gap writes rows of the destination and leaves the bytes between
    // them untouched. Handing the whole range over to the destination surface afterwards keeps
    // the bookkeeping simple, but then the surface must hold the gap contents too. If that
    // would mean reading another surface back from the host GPU, the gaps stay with their
    // current owner instead and only the rows written change hands. Games use such copies to
    // assemble an image out of columns, where the old gap contents are never looked at.
    const u32 output_stride = output_width + output_gap;
    const u32 output_rows = copy_size / output_width;
    bool rows_only = false;
    if (output_gap != 0) {
        if (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyTexCopyGap) {
            ValidateSurface(dst_surface_id, dst_params.addr, dst_params.size);
        } else {
            const Surface& gap_surface = slot_surfaces[dst_surface_id];
            bool gaps_valid = true;
            for (u32 row = 0; row + 1 < output_rows && gaps_valid; row++) {
                const PAddr gap_addr = dst_params.addr + row * output_stride + output_width;
                gaps_valid = gap_surface.IsRegionValid({gap_addr, gap_addr + output_gap});
            }
            rows_only = !gaps_valid &&
                        !ValidateSurface(dst_surface_id, dst_params.addr, dst_params.size, false);
        }
    }

    Surface& src_surface = slot_surfaces[src_surface_id];
    Surface& dst_surface = slot_surfaces[dst_surface_id];

    if (dst_surface.type == SurfaceType::Texture ||
        !CheckFormatsBlittable(src_surface.pixel_format, dst_surface.pixel_format)) {
        return false;
    }

    if (src_surface.Scaling() == dst_surface.Scaling() && !HasSameSize(src_rect, dst_rect)) {
        const TextureBlit texture_blit = {
            .src_level = src_surface.LevelOf(src_params.addr),
            .dst_level = dst_surface.LevelOf(dst_params.addr),
            .src_rect = src_rect,
            .dst_rect = dst_rect,
        };
        runtime.BlitTextures(src_surface, dst_surface, texture_blit);
    } else if (src_rect.GetWidth() != dst_rect.GetWidth()) {
        LOG_ERROR(
            HW_GPU,
            "Surface source and destination width mismatch, skipping... src_width={}, dst_width={}",
            src_rect.GetWidth(), dst_rect.GetHeight());
        return false;
    } else {
        const TextureCopy texture_copy = {
            .src_level = src_surface.LevelOf(src_params.addr),
            .dst_level = dst_surface.LevelOf(dst_params.addr),
            .src_offset = {src_rect.left, src_rect.bottom},
            .dst_offset = {dst_rect.left, dst_rect.bottom},
            .extent = {src_rect.GetWidth(), src_rect.GetHeight()},
        };
        runtime.CopyTextures(src_surface, dst_surface, texture_copy);
    }

    if (rows_only) {
        CITRA_FRAME_DUMP("  texcopy: {} rows of {} bytes change owner, gaps left as they are",
                         output_rows, output_width);
        InvalidateRows(dst_params.addr, output_width, output_stride, output_rows, dst_surface_id);
    } else {
        InvalidateRegion(dst_params.addr, dst_params.size, dst_surface_id);
    }
    return true;
}

template <class T>
bool RasterizerCache<T>::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    const DebugScope scope{runtime, Common::Vec4f{0.f, 0.f, 1.f, 1.f},
                           "RasterizerCache::AccelerateDisplayTransfer ({})", config.DebugName()};

    SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.width = config.output_width;
    src_params.stride = config.input_width;
    src_params.height = config.output_height;
    src_params.is_tiled = !config.input_linear;
    src_params.pixel_format = PixelFormatFromGPUPixelFormat(config.input_format);
    src_params.UpdateParams();

    NoteDisplayTransferSource(src_params, config.scaling != config.NoScale,
                              config.scaling == config.ScaleXY);

    SurfaceParams dst_params;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = config.scaling != config.NoScale ? config.output_width.Value() / 2
                                                        : config.output_width.Value();
    dst_params.height = config.scaling == config.ScaleXY ? config.output_height.Value() / 2
                                                         : config.output_height.Value();
    dst_params.is_tiled = config.input_linear != config.dont_swizzle;
    dst_params.pixel_format = PixelFormatFromGPUPixelFormat(config.output_format);
    dst_params.sample_count = sample_count;
    dst_params.UpdateParams();

    // Using flip_vertically alongside crop_input_lines produces skewed output on hardware.
    // We have to emulate this because some games rely on this behaviour to render correctly.
    if (config.flip_vertically && config.crop_input_lines) {
        dst_params.addr += (config.input_width - config.output_width) * (config.output_height - 1) *
                           Pica::BytesPerPixel(config.output_format);
    }

    auto [src_surface_id, src_rect] = GetSurfaceSubRect(src_params, ScaleMatch::Ignore, true);
    if (!src_surface_id) {
        return false;
    }

    dst_params.res_scale = slot_surfaces[src_surface_id].res_scale;
    dst_params.sample_count = slot_surfaces[src_surface_id].sample_count;

    const auto [dst_surface_id, dst_rect] =
        GetSurfaceSubRect(dst_params, ScaleMatch::Upscale, false);
    if (!dst_surface_id) {
        return false;
    }

    Surface& src_surface = slot_surfaces[src_surface_id];
    Surface& dst_surface = slot_surfaces[dst_surface_id];

    if (src_surface.is_tiled != dst_surface.is_tiled) {
        std::swap(src_rect.top, src_rect.bottom);
    }
    if (config.flip_vertically) {
        std::swap(src_rect.top, src_rect.bottom);
    }

    if (!CheckFormatsBlittable(src_surface.pixel_format, dst_surface.pixel_format)) {
        return false;
    }

    const TextureBlit texture_blit = {
        .src_level = src_surface.LevelOf(src_params.addr),
        .dst_level = dst_surface.LevelOf(dst_params.addr),
        .src_rect = src_rect,
        .dst_rect = dst_rect,
    };
    runtime.BlitTextures(src_surface, dst_surface, texture_blit);

    InvalidateRegion(dst_params.addr, dst_params.size, dst_surface_id);
    return true;
}

template <class T>
bool RasterizerCache<T>::AccelerateFill(const Pica::MemoryFillConfig& config) {
    const DebugScope scope{runtime, Common::Vec4f{1.f, 0.f, 1.f, 1.f},
                           "RasterizerCache::AccelerateFill ({})", config.DebugName()};

    SurfaceParams params;
    params.addr = config.GetStartAddress();
    params.end = config.GetEndAddress();
    params.size = params.end - params.addr;
    params.type = SurfaceType::Fill;
    params.res_scale = Common::ResolutionScale::FromSteps(std::numeric_limits<u16>::max());

    SurfaceId fill_surface_id = slot_surfaces.insert(runtime, params);
    Surface& fill_surface = slot_surfaces[fill_surface_id];

    std::memcpy(&fill_surface.fill_data[0], &config.value_32bit, sizeof(u32));
    if (config.fill_32bit) {
        fill_surface.fill_size = 4;
    } else if (config.fill_24bit) {
        fill_surface.fill_size = 3;
    } else {
        fill_surface.fill_size = 2;
    }

    RegisterSurface(fill_surface_id);
    InvalidateRegion(fill_surface.addr, fill_surface.size, fill_surface_id);
    return true;
}

template <class T>
typename T::Surface& RasterizerCache<T>::GetSurface(SurfaceId surface_id) {
    return slot_surfaces[surface_id];
}

template <class T>
typename T::Sampler& RasterizerCache<T>::GetSampler(SamplerId sampler_id) {
    return slot_samplers[sampler_id];
}

template <class T>
typename T::Sampler& RasterizerCache<T>::GetSampler(
    const Pica::TexturingRegs::TextureConfig& config, bool nearest) {
    using TextureFilter = Pica::TexturingRegs::TextureConfig::TextureFilter;

    const auto get_filter = [nearest](TextureFilter filter) {
        if (nearest) {
            return TextureFilter::Nearest;
        }
        switch (Settings::values.texture_sampling.GetValue()) {
        case Settings::TextureSampling::GameControlled:
            return filter;
        case Settings::TextureSampling::NearestNeighbor:
            return TextureFilter::Nearest;
        case Settings::TextureSampling::Linear:
            return TextureFilter::Linear;
        default:
            return filter;
        }
    };

    const SamplerParams params = {
        .mag_filter = get_filter(config.mag_filter),
        .min_filter = get_filter(config.min_filter),
        .mip_filter = get_filter(config.mip_filter),
        .wrap_s = config.wrap_s,
        .wrap_t = config.wrap_t,
        .border_color = config.border_color.raw,
        .lod_min = config.lod.min_level,
        .lod_max = config.lod.max_level,
        .lod_bias = config.lod.bias,
    };

    for (const auto& [asked, id] : recent_samplers) {
        if (id && asked == params) {
            return slot_samplers[id];
        }
    }

    auto [it, is_new] = samplers.try_emplace(params);
    if (is_new) {
        it->second = slot_samplers.insert(runtime, params);
    }
    recent_samplers[recent_sampler_next++ % recent_samplers.size()] = {params, it->second};

    return slot_samplers[it->second];
}

template <class T>
void RasterizerCache<T>::CopySurface(Surface& src_surface, Surface& dst_surface,
                                     SurfaceInterval copy_interval) {
    MICROPROFILE_SCOPE(RasterizerCache_CopySurface);
    CITRA_TRACE_SCOPE_DETAIL("Surface::Copy");
    CITRA_TRACE_COUNT(SurfaceCopies);
    CITRA_FRAME_DUMP(
        "cache copy {} {}x{} s{} t{} @{:08X} -> {} {}x{} s{} t{} @{:08X} [{:08X}-{:08X}]",
        PixelFormatAsString(src_surface.pixel_format), src_surface.width, src_surface.height,
        src_surface.res_scale, static_cast<u32>(src_surface.type), src_surface.addr,
        PixelFormatAsString(dst_surface.pixel_format), dst_surface.width, dst_surface.height,
        dst_surface.res_scale, static_cast<u32>(dst_surface.type), dst_surface.addr,
        copy_interval.lower(), copy_interval.upper());
    const PAddr copy_addr = copy_interval.lower();
    const SurfaceParams subrect_params = dst_surface.FromInterval(copy_interval);
    ASSERT(subrect_params.GetInterval() == copy_interval);

    if (src_surface.type == SurfaceType::Fill) {
        const TextureClear clear = {
            .texture_level = dst_surface.LevelOf(copy_addr),
            .texture_rect = dst_surface.GetScaledSubRect(subrect_params),
            .value = src_surface.MakeClearValue(copy_addr, dst_surface.pixel_format),
        };
        runtime.ClearTexture(dst_surface, clear);
        return;
    }

    // Another surface takes its contents from this one. Unless that is the same guest surface
    // at a different scale, the contents are used for something other than being shrunk.
    if (src_surface.addr != dst_surface.addr || src_surface.width != dst_surface.width ||
        src_surface.height != dst_surface.height ||
        src_surface.pixel_format != dst_surface.pixel_format) {
        NoteFullSizeUse(src_surface);
    }

    const Common::ResolutionScale src_scale = src_surface.res_scale;
    const Common::ResolutionScale dst_scale = dst_surface.res_scale;
    const u32 src_sample_count = src_surface.sample_count;
    const u32 dst_sample_count = dst_surface.sample_count;
    // A surface takes on the scale of what it is filled from, except a shadow map that is
    // kept small on purpose: that one is filled by shrinking
    if (((src_scale > dst_scale) || (src_sample_count > dst_sample_count)) &&
        !IsNativeScaleShadowMap(dst_surface)) {
        dst_surface.ScaleUp(src_scale, src_sample_count);
    }

    const auto src_rect = src_surface.GetScaledSubRect(subrect_params);
    const auto dst_rect = dst_surface.GetScaledSubRect(subrect_params);
    // Between surfaces of one scale that is a copy, unless the guest pixels lie where they come
    // to a host pixel more in one of the two than in the other (see HasSameSize)
    if (src_scale == dst_scale && src_surface.half_scale_x == dst_surface.half_scale_x &&
        src_surface.half_scale_y == dst_surface.half_scale_y && HasSameSize(src_rect, dst_rect)) {
        const TextureCopy copy = {
            .src_level = src_surface.LevelOf(copy_addr),
            .dst_level = dst_surface.LevelOf(copy_addr),
            .src_offset = {src_rect.left, src_rect.bottom},
            .dst_offset = {dst_rect.left, dst_rect.bottom},
            .extent = {src_rect.GetWidth(), src_rect.GetHeight()},
        };
        runtime.CopyTextures(src_surface, dst_surface, copy);
    } else {
        const TextureBlit blit = {
            .src_level = src_surface.LevelOf(copy_addr),
            .dst_level = dst_surface.LevelOf(copy_addr),
            .src_rect = src_rect,
            .dst_rect = dst_rect,
        };
        runtime.BlitTextures(src_surface, dst_surface, blit);
    }
}

template <class T>
SurfaceId RasterizerCache<T>::GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                                         bool load_if_create,
                                         const SurfaceFlagBits& create_initial_flags) {
    if (params.addr == 0 || params.height * params.width == 0) {
        return {};
    }
    // Use GetSurfaceSubRect instead
    ASSERT(params.width == params.stride);
    ASSERT(!params.is_tiled || (params.width % 8 == 0 && params.height % 8 == 0));

    // Check for an exact match in existing surfaces
    SurfaceId surface_id = FindMatch<MatchFlags::Exact>(params, match_res_scale);

    if (!surface_id) {
        surface_id = CreateSurface(params, create_initial_flags);
        RegisterSurface(surface_id);
    }

    if (load_if_create) {
        ValidateSurface(surface_id, params.addr, params.size);
    }

    return surface_id;
}

template <class T>
typename RasterizerCache<T>::SurfaceRect_Tuple RasterizerCache<T>::GetSurfaceSubRect(
    const SurfaceParams& params, ScaleMatch match_res_scale, bool load_if_create,
    const SurfaceFlagBits& create_initial_flags) {
    if (params.addr == 0 || params.height * params.width == 0) {
        return std::make_pair(SurfaceId{}, Common::Rectangle<u32>{});
    }

    // Attempt to find encompassing surface
    SurfaceId surface_id = FindMatch<MatchFlags::SubRect>(params, match_res_scale);

    // Check if FindMatch failed because of res scaling. If that's the case create a new surface
    // with the dimensions of the lower res_scale surface to suggest it should not be used again.
    if (!surface_id && match_res_scale != ScaleMatch::Ignore) {
        surface_id = FindMatch<MatchFlags::SubRect>(params, ScaleMatch::Ignore);
        if (surface_id) {
            SurfaceParams new_params = slot_surfaces[surface_id];
            new_params.res_scale = params.res_scale;
            new_params.half_scale_x = params.half_scale_x;
            new_params.half_scale_y = params.half_scale_y;
            new_params.sample_count = params.sample_count;

            surface_id = CreateSurface(new_params, create_initial_flags);
            RegisterSurface(surface_id);
        }
    }

    SurfaceParams aligned_params = params;
    if (params.is_tiled) {
        aligned_params.height = Common::AlignUp(params.height, 8);
        aligned_params.width = Common::AlignUp(params.width, 8);
        aligned_params.stride = Common::AlignUp(params.stride, 8);
        aligned_params.UpdateParams();
    }

    // No subrect found - create and return a new surface
    if (!surface_id) {
        SurfaceParams new_params = aligned_params;
        // Can't have gaps in a surface
        new_params.width = aligned_params.stride;
        new_params.UpdateParams();
        // GetSurface will create the new surface and possibly adjust res_scale if necessary
        surface_id = GetSurface(new_params, match_res_scale, load_if_create, create_initial_flags);
    } else if (load_if_create) {
        ValidateSurface(surface_id, aligned_params.addr, aligned_params.size);
    }

    return std::make_pair(surface_id, slot_surfaces[surface_id].GetScaledSubRect(params));
}

template <class T>
typename T::Surface& RasterizerCache<T>::GetTextureSurface(
    const Pica::TexturingRegs::FullTextureConfig& config) {
    const auto info = Pica::Texture::TextureInfo::FromPicaRegister(config.config, config.format);
    const u32 max_level = MipLevels(info.width, info.height, config.config.lod.max_level) - 1;

    // The same texture asked for again with the cache as it was: what was found then
    const bool remember =
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyTextureLookup) == 0;
    TextureLookup& known =
        texture_lookups[((info.physical_address >> 6) ^ (info.physical_address >> 15) ^ info.width ^
                         (max_level << 3)) %
                        texture_lookups.size()];
    if (remember && known.surface_id && known.address == info.physical_address &&
        known.width == info.width && known.height == info.height &&
        known.format == static_cast<u32>(info.format) && known.max_level == max_level &&
        known.shadow_source == info.is_shadow_source && known.set_epoch == surface_set_epoch &&
        known.validity_epoch == SurfaceBase::validity_epoch) {
        return slot_surfaces[known.surface_id];
    }

    const SurfaceId surface_id = GetTextureSurface(info, max_level);
    // Not a surface made for this one use (a size that is no multiple of eight), nor none
    const u32 min_width = info.width >> max_level;
    const u32 min_height = info.height >> max_level;
    if (remember && surface_id != NULL_SURFACE_ID && min_width % 8 == 0 && min_height % 8 == 0) {
        known = TextureLookup{
            .address = info.physical_address,
            .width = info.width,
            .height = info.height,
            .format = static_cast<u32>(info.format),
            .max_level = max_level,
            .shadow_source = info.is_shadow_source,
            .set_epoch = surface_set_epoch,
            .validity_epoch = SurfaceBase::validity_epoch,
            .surface_id = surface_id,
        };
    } else {
        known.surface_id = {};
    }
    return slot_surfaces[surface_id];
}

template <class T>
SurfaceId RasterizerCache<T>::GetTextureSurface(const Pica::Texture::TextureInfo& info,
                                                u32 max_level) {
    if (info.physical_address == 0) [[unlikely]] {
        // Can occur when texture addr is null or its memory is unmapped/invalid
        // HACK: In this case, the correct behaviour for the PICA is to use the last
        // rendered colour. But because this would be impractical to implement, the
        // next best alternative is to use a clear texture, essentially skipping
        // the geometry in question.
        // For example: a bug in Pokemon X/Y causes NULL-texture squares to be drawn
        // on the male character's face, which in the OpenGL default appear black.
        return NULL_SURFACE_ID;
    }

    SurfaceParams params;
    params.addr = info.physical_address;
    params.width = info.width;
    params.height = info.height;
    params.levels = max_level + 1;
    params.is_tiled = true;
    params.pixel_format = PixelFormatFromTextureFormat(info.format);
    params.res_scale = filter != Settings::TextureFilter::NoFilter ? resolution_scale_factor
                                                                   : Common::ResolutionScale{};
    SurfaceFlagBits initial_flags{};
    if (info.is_shadow_source) {
        initial_flags |= SurfaceFlagBits::ShadowSource;
    }
    params.UpdateParams();
    NoteFullSizeUse(params);

    const u32 min_width = info.width >> max_level;
    const u32 min_height = info.height >> max_level;
    if (min_width % 8 != 0 || min_height % 8 != 0) {
        if (min_width % 4 != 0 || min_height % 4 != 0) {
            LOG_CRITICAL(HW_GPU, "Texture size ({}x{}) is not multiple of 4", min_width,
                         min_height);
            return NULL_SURFACE_ID;
        }
        const auto [src_surface_id, rect] =
            GetSurfaceSubRect(params, ScaleMatch::Ignore, true, initial_flags);

        // CreateSurface can grow slot_surfaces, so take Surface& after it returns
        params.res_scale = slot_surfaces[src_surface_id].res_scale;
        SurfaceId tmp_surface_id = CreateSurface(params, initial_flags);
        Surface& src_surface = slot_surfaces[src_surface_id];
        Surface& tmp_surface = slot_surfaces[tmp_surface_id];
        Sentence(tmp_surface_id);

        const TextureBlit blit = {
            .src_level = src_surface.LevelOf(params.addr),
            .dst_level = 0,
            .src_rect = rect,
            .dst_rect = tmp_surface.GetScaledRect(),
        };
        runtime.BlitTextures(src_surface, tmp_surface, blit);
        return tmp_surface_id;
    }
    if (info.width != (min_width << max_level) || info.height != (min_height << max_level)) {
        LOG_CRITICAL(HW_GPU, "Texture size ({}x{}) does not support required mipmap level ({})",
                     params.width, params.height, max_level);
        return NULL_SURFACE_ID;
    }

    SurfaceId surface_id = GetSurface(params, ScaleMatch::Ignore, true, initial_flags);
    return surface_id ? surface_id : NULL_SURFACE_ID;
}

template <class T>
typename T::Surface& RasterizerCache<T>::GetTextureCube(const TextureCubeConfig& config) {
    if (config.width == 0) [[unlikely]] {
        return slot_surfaces[NULL_SURFACE_CUBE_ID];
    }

    auto [it, new_surface] = texture_cube_cache.try_emplace(config);
    TextureCube& cube = it->second;

    const std::array addresses = {config.px, config.nx, config.py, config.ny, config.pz, config.nz};

    if (new_surface) {
        Pica::Texture::TextureInfo info = {
            .width = config.width,
            .height = config.width,
            .format = config.format,
        };
        info.SetDefaultStride();

        Common::ResolutionScale res_scale;
        for (u32 i = 0; i < addresses.size(); i++) {
            if (!addresses[i]) {
                continue;
            }

            SurfaceId& face_id = cube.face_ids[i];
            if (!face_id) {
                info.physical_address = addresses[i];
                face_id = GetTextureSurface(info, config.levels - 1);
                Surface& surface = slot_surfaces[face_id];
                ASSERT_MSG(
                    surface.levels >= config.levels,
                    "Texture cube face levels are not enough to validate the levels requested");
                surface.flags |= SurfaceFlagBits::Tracked;
            }
            Surface& surface = slot_surfaces[face_id];
            res_scale = std::max(surface.res_scale, res_scale);
        }

        SurfaceParams cube_params = {
            .addr = config.px,
            .width = config.width,
            .height = config.width,
            .stride = config.width,
            .levels = config.levels,
            .res_scale = res_scale,
            .texture_type = TextureType::CubeMap,
            .pixel_format = PixelFormatFromTextureFormat(config.format),
            .type = SurfaceType::Texture,
        };
        cube_params.UpdateParams();
        cube.surface_id = CreateSurface(cube_params);
    }

    Surface& cube_surface = slot_surfaces[cube.surface_id];
    for (u32 i = 0; i < addresses.size(); i++) {
        const SurfaceId& face_id = cube.face_ids[i];
        if (!addresses[i] || !face_id) {
            continue;
        }
        Surface& surface = slot_surfaces[face_id];
        if (cube.ticks[i] == surface.modification_tick) {
            continue;
        }
        cube.ticks[i] = surface.modification_tick;
        boost::container::small_vector<TextureCopy, 8> upload_copies;
        for (u32 level = 0; level < config.levels; level++) {
            const u32 width_lod = surface.GetScaledWidth() >> level;
            upload_copies.push_back({
                .src_level = level,
                .dst_level = level,
                .src_layer = 0,
                .dst_layer = i,
                .src_offset = {0, 0},
                .dst_offset = {0, 0},
                .extent = {width_lod, width_lod},
            });
        }
        runtime.CopyTextures(surface, cube_surface, upload_copies);
    }

    return slot_surfaces[cube.surface_id];
}

template <class T>
FramebufferHelper<T> RasterizerCache<T>::GetFramebufferSurfaces(bool using_color_fb,
                                                                bool using_depth_fb,
                                                                bool depth_overwritten) {
    const auto& config = regs.framebuffer.framebuffer;

    const s32 framebuffer_width = config.GetWidth();
    const s32 framebuffer_height = config.GetHeight();
    const auto viewport_rect = regs.rasterizer.GetViewportRect();
    const Common::Rectangle<u32> viewport_clamped = {
        static_cast<u32>(std::clamp(viewport_rect.left, 0, framebuffer_width)),
        static_cast<u32>(std::clamp(viewport_rect.top, 0, framebuffer_height)),
        static_cast<u32>(std::clamp(viewport_rect.right, 0, framebuffer_width)),
        static_cast<u32>(std::clamp(viewport_rect.bottom, 0, framebuffer_height)),
    };

    // The target of the draw before, asked for again with the cache as it was: what was
    // found then is what would be found now
    const bool shadow_rendering = regs.framebuffer.IsShadowRendering();
    const LastTarget asked = {
        .valid = true,
        .color_addr = config.GetColorBufferPhysicalAddress(),
        .depth_addr = config.GetDepthBufferPhysicalAddress(),
        .color_format = static_cast<u32>(config.color_format.Value()),
        .depth_format = static_cast<u32>(config.depth_format.Value()),
        .width = static_cast<u32>(framebuffer_width),
        .height = static_cast<u32>(framebuffer_height),
        .viewport = viewport_rect,
        .using_color = using_color_fb,
        .using_depth = using_depth_fb,
        .shadow_rendering = shadow_rendering,
        .res_scale = resolution_scale_factor,
        .sample_count = sample_count,
        .validity_epoch = SurfaceBase::validity_epoch,
    };
    if (last_target.valid && last_target.color_addr == asked.color_addr &&
        last_target.depth_addr == asked.depth_addr &&
        last_target.color_format == asked.color_format &&
        last_target.depth_format == asked.depth_format && last_target.width == asked.width &&
        last_target.height == asked.height && last_target.viewport.left == asked.viewport.left &&
        last_target.viewport.top == asked.viewport.top &&
        last_target.viewport.right == asked.viewport.right &&
        last_target.viewport.bottom == asked.viewport.bottom &&
        last_target.using_color == asked.using_color &&
        last_target.using_depth == asked.using_depth &&
        last_target.shadow_rendering == asked.shadow_rendering &&
        last_target.res_scale == asked.res_scale &&
        last_target.sample_count == asked.sample_count &&
        last_target.validity_epoch == asked.validity_epoch &&
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyTargetLookup) == 0) {
        const FramebufferParams found = {
            .color_id = last_target.color_id,
            .depth_id = last_target.depth_id,
            .color_level = last_target.color_level,
            .depth_level = last_target.depth_level,
            .shadow_rendering = shadow_rendering,
        };
        return FramebufferHelper<T>{this, RefreshFramebuffer(found), config.IsFlipped(),
                                    regs.rasterizer, last_target.fb_rect};
    }
    last_target.valid = false;

    SurfaceParams color_params;
    color_params.is_tiled = true;
    color_params.res_scale = resolution_scale_factor;
    color_params.sample_count = sample_count;
    color_params.width = config.GetWidth();
    color_params.height = config.GetHeight();
    SurfaceParams depth_params = color_params;

    color_params.addr = config.GetColorBufferPhysicalAddress();
    color_params.pixel_format = PixelFormatFromColorFormat(config.color_format);

    // A shadow map is looked up, not looked at: every lit pixel reads four of its texels and
    // compares depths. At the resolution scale it takes that scale squared as much memory for
    // the same lookups, 25 times at 5x, far more than the GPU keeps at hand, so that each of
    // them goes to main memory. It stays the size the game gave it, as on the console. The
    // game clears it with an ordinary draw before it renders into it, which is why the
    // buffers are remembered.
    if (using_color_fb && !resolution_scale_factor.IsNative()) {
        static constexpr std::size_t MAX_SHADOW_MAP_TARGETS = 8;
        auto target =
            std::ranges::find(shadow_map_targets, color_params.addr, &ShadowMapTarget::addr);
        if (target == shadow_map_targets.end() && regs.framebuffer.IsShadowRendering()) {
            if (shadow_map_targets.size() == MAX_SHADOW_MAP_TARGETS) {
                shadow_map_targets.erase(shadow_map_targets.begin());
            }
            shadow_map_targets.push_back(ShadowMapTarget{color_params.addr, false});
            target = shadow_map_targets.end() - 1;
        }
        if (target != shadow_map_targets.end()) {
            // Colour and depth have to be of one size. Should the buffer be drawn to with a
            // depth buffer, it keeps the resolution scale from then on.
            target->with_depth = target->with_depth || using_depth_fb;
            if (!target->with_depth && sample_count == 1 &&
                (Common::TraceMarkers::Experiments() &
                 Common::TraceMarkers::LegacyShadowMapScale) == 0) {
                color_params.res_scale = Common::ResolutionScale{};
            }
        }
    }
    color_params.UpdateParams();

    depth_params.addr = config.GetDepthBufferPhysicalAddress();
    depth_params.pixel_format = PixelFormatFromDepthFormat(config.depth_format);
    depth_params.UpdateParams();

    ApplyDownscaledTargetScale(color_params, depth_params);

    auto color_vp_interval = color_params.GetSubRectInterval(viewport_clamped);
    auto depth_vp_interval = depth_params.GetSubRectInterval(viewport_clamped);

    Common::Rectangle<u32> color_rect{};
    SurfaceId color_id{};
    u32 color_level{};
    if (using_color_fb)
        std::tie(color_id, color_rect) = GetSurfaceSubRect(color_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> depth_rect{};
    SurfaceId depth_id{};
    u32 depth_level{};
    if (using_depth_fb)
        std::tie(depth_id, depth_rect) = GetSurfaceSubRect(depth_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> fb_rect{};
    if (color_id && depth_id) {
        fb_rect = color_rect;
        // Color and Depth surfaces must have the same dimensions and offsets
        if (color_rect.bottom != depth_rect.bottom || color_rect.top != depth_rect.top ||
            color_rect.left != depth_rect.left || color_rect.right != depth_rect.right) {
            color_id = GetSurface(color_params, ScaleMatch::Exact, false);
            depth_id = GetSurface(depth_params, ScaleMatch::Exact, false);
            fb_rect = slot_surfaces[color_id].GetScaledRect();
        }
    } else if (color_id) {
        fb_rect = color_rect;
    } else if (depth_id) {
        fb_rect = depth_rect;
    }

    if (color_id) {
        // Validation can grow slot_surfaces, reference lifetime should expire after validation
        Surface& color_surface = slot_surfaces[color_id];
        color_level = color_surface.LevelOf(color_params.addr);
        color_surface.flags |= SurfaceFlagBits::RenderTarget;
        ValidateSurface(color_id, boost::icl::first(color_vp_interval),
                        boost::icl::length(color_vp_interval));
    }
    if (depth_id) {
        // Validation can grow slot_surfaces, reference lifetime should expire after validation
        Surface& depth_surface = slot_surfaces[depth_id];
        depth_level = depth_surface.LevelOf(depth_params.addr);
        depth_surface.flags |= SurfaceFlagBits::RenderTarget;
        ValidateSurface(depth_id, boost::icl::first(depth_vp_interval),
                        boost::icl::length(depth_vp_interval), true, !depth_overwritten);
    }

    const FramebufferParams fb_params = {
        .color_id = color_id,
        .depth_id = depth_id,
        .color_level = color_level,
        .depth_level = depth_level,
        .shadow_rendering = regs.framebuffer.IsShadowRendering(),
    };

    // For the next draw that asks for the same. The cache as it is now, after what the
    // looking up itself made and validated, is what has to stand still until then.
    last_target = asked;
    last_target.validity_epoch = SurfaceBase::validity_epoch;
    last_target.color_id = color_id;
    last_target.depth_id = depth_id;
    last_target.color_level = color_level;
    last_target.depth_level = depth_level;
    last_target.fb_rect = fb_rect;

    return FramebufferHelper<T>{this, RefreshFramebuffer(fb_params),
                                regs.framebuffer.framebuffer.IsFlipped(), regs.rasterizer, fb_rect};
}

template <class T>
FramebufferParams RasterizerCache<T>::WithCurrentImages(FramebufferParams fb_params) const {
    if (fb_params.color_id) {
        const Surface& color_surface = slot_surfaces[fb_params.color_id];
        fb_params.color_swapped = color_surface.swapped;
        fb_params.color_generation = color_surface.image_generation;
    }
    if (fb_params.depth_id) {
        fb_params.depth_generation = slot_surfaces[fb_params.depth_id].image_generation;
    }
    return fb_params;
}

template <class T>
typename RasterizerCache<T>::Framebuffer* RasterizerCache<T>::RefreshFramebuffer(
    FramebufferParams fb_params) {
    fb_params = WithCurrentImages(fb_params);

    auto [it, new_framebuffer] = framebuffers.try_emplace(fb_params);
    if (new_framebuffer) {
        // Validation can grow slot_surfaces, so get fresh references here
        Surface* color_surface = fb_params.color_id ? &slot_surfaces[fb_params.color_id] : nullptr;
        Surface* depth_surface = fb_params.depth_id ? &slot_surfaces[fb_params.depth_id] : nullptr;
        it->second = slot_framebuffers.insert(runtime, fb_params, color_surface, depth_surface);
    }
    return &slot_framebuffers[it->second];
}

template <class T>
typename RasterizerCache<T>::DownscaledTarget* RasterizerCache<T>::FindDownscaledTarget(
    const SurfaceParams& params) {
    const auto it = std::ranges::find_if(downscaled_targets, [&](const DownscaledTarget& target) {
        return target.addr == params.addr && target.width == params.width &&
               target.height == params.height && target.pixel_format == params.pixel_format;
    });
    return it != downscaled_targets.end() ? &*it : nullptr;
}

template <class T>
void RasterizerCache<T>::NoteDisplayTransferSource(const SurfaceParams& src_params, bool half_x,
                                                   bool half_y) {
    DownscaledTarget* target = FindDownscaledTarget(src_params);
    if (!half_x && !half_y) {
        if (target) {
            target->used_at_full_size = true;
        }
        return;
    }
    if (!target) {
        // Games have a handful of render targets at most. Should one keep making up new ones,
        // forget the oldest.
        static constexpr std::size_t MAX_DOWNSCALED_TARGETS = 8;
        last_target.valid = false;
        surface_set_epoch++;
        if (downscaled_targets.size() == MAX_DOWNSCALED_TARGETS) {
            downscaled_targets.erase(downscaled_targets.begin());
        }
        target = &downscaled_targets.emplace_back(DownscaledTarget{
            .addr = src_params.addr,
            .width = src_params.width,
            .height = src_params.height,
            .pixel_format = src_params.pixel_format,
        });
    }
    bool& shrunk = half_y ? target->shrunk_xy : target->shrunk_x;
    if (!shrunk) {
        // Which surface a target is looked up as goes by this
        shrunk = true;
        last_target.valid = false;
        surface_set_epoch++;
    }
}

template <class T>
void RasterizerCache<T>::NoteFullSizeUse(const SurfaceParams& params) {
    if (downscaled_targets.empty()) {
        return;
    }
    if (DownscaledTarget* target = FindDownscaledTarget(params)) {
        if (!target->used_at_full_size) {
            target->used_at_full_size = true;
            last_target.valid = false;
            surface_set_epoch++;
        }
    }
}

template <class T>
void RasterizerCache<T>::ApplyDownscaledTargetScale(SurfaceParams& color_params,
                                                    SurfaceParams& depth_params) {
    // A display transfer with scaling halves its source in one or both directions. The host
    // shrinks with a nearest-neighbour blit, so at the full resolution scale half (or three
    // quarters) of the pixels rendered to such a target are thrown away. Rendering at half the
    // scale in the directions that get halved puts the same number of samples on screen for a
    // fraction of the work.
    if (downscaled_targets.empty()) {
        return;
    }
    DownscaledTarget* target = FindDownscaledTarget(color_params);
    if (!target) {
        return;
    }
    if (target->shrunk_x || target->shrunk_xy || target->used_at_full_size) {
        // First draw since the target was last used, which shows what this draw will be used
        // for. Anything that showed the contents at full size (a plain display transfer, a
        // texture, a copy) means the target is not only an oversized version of what gets
        // displayed, and it keeps the full scale.
        target->half_x = !target->used_at_full_size;
        target->half_y = !target->used_at_full_size && !target->shrunk_x;
        target->shrunk_x = target->shrunk_xy = target->used_at_full_size = false;
    }

    // With no upscaling, or too little of it to halve, the target keeps its size.
    if (resolution_scale_factor < Common::ResolutionScale::Whole(2) || sample_count != 1 ||
        !runtime.SupportsHalfScaleTargets() ||
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyFullScaleTargets)) {
        return;
    }
    color_params.half_scale_x = depth_params.half_scale_x = target->half_x;
    color_params.half_scale_y = depth_params.half_scale_y = target->half_y;
}

template <class T>
typename RasterizerCache<T>::SurfaceRect_Tuple RasterizerCache<T>::GetTexCopySurface(
    const SurfaceParams& params) {
    Common::Rectangle<u32> rect{};

    SurfaceId match_id = FindMatch<MatchFlags::TexCopy>(params, ScaleMatch::Ignore);

    if (match_id) {
        ValidateSurface(match_id, params.addr, params.size);

        SurfaceParams match_subrect;
        Surface& match_surface = slot_surfaces[match_id];
        if (params.width != params.stride) {
            const u32 tiled_size = match_surface.is_tiled ? 8 : 1;
            match_subrect = params;
            match_subrect.width = match_surface.PixelsInBytes(params.width) / tiled_size;
            match_subrect.stride = match_surface.PixelsInBytes(params.stride) / tiled_size;
            match_subrect.height *= tiled_size;
        } else {
            match_subrect = match_surface.FromInterval(params.GetInterval());
            ASSERT(match_subrect.GetInterval() == params.GetInterval());
        }

        rect = match_surface.GetSubRect(match_subrect);
    }

    return std::make_pair(match_id, rect);
}

template <class T>
template <typename Func>
void RasterizerCache<T>::ForEachSurfaceInRegion(PAddr addr, std::size_t size, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, SurfaceId, Surface&>::type;
    static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<SurfaceId, 8> surfaces;
    ForEachPage(addr, size, [this, &surfaces, addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            if constexpr (BOOL_BREAK) {
                return false;
            } else {
                return;
            }
        }
        for (const SurfaceId surface_id : it->second) {
            Surface& surface = slot_surfaces[surface_id];
            if (True(surface.flags & SurfaceFlagBits::Picked)) {
                continue;
            }
            if (!surface.Overlaps(addr, size)) {
                continue;
            }

            surface.flags |= SurfaceFlagBits::Picked;
            surfaces.push_back(surface_id);
            if constexpr (BOOL_BREAK) {
                if (func(surface_id, surface)) {
                    return true;
                }
            } else {
                func(surface_id, surface);
            }
        }
        if constexpr (BOOL_BREAK) {
            return false;
        }
    });
    for (const SurfaceId surface_id : surfaces) {
        slot_surfaces[surface_id].flags &= ~SurfaceFlagBits::Picked;
    }
}

template <class T>
template <MatchFlags find_flags>
SurfaceId RasterizerCache<T>::FindMatch(const SurfaceParams& params, ScaleMatch match_scale_type,
                                        std::optional<SurfaceInterval> validate_interval) {
    SurfaceId match_id{};
    bool match_valid = false;
    auto match_scale = Common::ResolutionScale::FromSteps(0);
    u8 match_sample_count = 0;
    SurfaceInterval match_interval{};

    ForEachSurfaceInRegion(params.addr, params.size, [&](SurfaceId surface_id, Surface& surface) {
        // A surface with half the scale along an axis only stands in for a request that asks
        // for that as well.
        const bool half_scale_matched = match_scale_type == ScaleMatch::Exact
                                            ? (params.half_scale_x == surface.half_scale_x &&
                                               params.half_scale_y == surface.half_scale_y)
                                            : ((params.half_scale_x || !surface.half_scale_x) &&
                                               (params.half_scale_y || !surface.half_scale_y));
        const bool res_scale_matched =
            half_scale_matched &&
            (match_scale_type == ScaleMatch::Exact ? (params.res_scale == surface.res_scale)
                                                   : (params.res_scale <= surface.res_scale));
        const bool sample_count_matched = match_scale_type == ScaleMatch::Exact
                                              ? (params.sample_count == surface.sample_count)
                                              : (params.sample_count <= surface.sample_count);
        const bool is_valid =
            True(find_flags & MatchFlags::Copy)
                ? true
                : surface.IsRegionValid(validate_interval.value_or(params.GetInterval()));

        auto IsMatch_Helper = [&](auto check_type, auto match_fn) {
            if (False(find_flags & check_type))
                return;

            bool matched;
            SurfaceInterval surface_interval;
            std::tie(matched, surface_interval) = match_fn();
            if (!matched)
                return;

            if (!res_scale_matched && match_scale_type != ScaleMatch::Ignore &&
                surface.type != SurfaceType::Fill)
                return;

            if (!sample_count_matched && match_scale_type != ScaleMatch::Ignore &&
                surface.type != SurfaceType::Fill)
                return;

            // Found a match, update only if this is better than the previous one
            auto UpdateMatch = [&] {
                match_id = surface_id;
                match_valid = is_valid;
                match_scale = surface.res_scale;
                match_sample_count = surface.sample_count;
                match_interval = surface_interval;
            };

            if (surface.res_scale > match_scale) {
                UpdateMatch();
                return;
            } else if (surface.res_scale < match_scale) {
                return;
            }

            if (surface.sample_count > match_sample_count) {
                UpdateMatch();
                return;
            } else if (surface.sample_count < match_sample_count) {
                return;
            }

            if (is_valid && !match_valid) {
                UpdateMatch();
                return;
            } else if (is_valid != match_valid) {
                return;
            }

            if (boost::icl::length(surface_interval) > boost::icl::length(match_interval)) {
                UpdateMatch();
            }
        };
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Exact>{}, [&] {
            return std::make_pair(surface.ExactMatch(params), surface.GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::SubRect>{}, [&] {
            return std::make_pair(surface.CanSubRect(params), surface.GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Copy>{}, [&] {
            ASSERT(validate_interval);
            const SurfaceInterval copy_interval =
                surface.GetCopyableInterval(params.FromInterval(*validate_interval));
            const bool matched = boost::icl::length(copy_interval & *validate_interval) != 0 &&
                                 surface.CanCopy(params, copy_interval);
            return std::make_pair(matched, copy_interval);
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Reinterpret>{}, [&] {
            ASSERT(validate_interval);
            const SurfaceInterval copy_interval =
                surface.GetCopyableInterval(params.FromInterval(*validate_interval));
            const bool matched = boost::icl::length(copy_interval & *validate_interval) != 0 &&
                                 surface.CanReinterpret(params);
            return std::make_pair(matched, copy_interval);
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::TexCopy>{}, [&] {
            return std::make_pair(surface.CanTexCopy(params), surface.GetInterval());
        });
    });
    return match_id;
}

template <class T>
bool RasterizerCache<T>::ValidateSurface(SurfaceId surface_id, PAddr addr, u32 size,
                                         bool allow_download, bool needs_contents) {
    if (size == 0) [[unlikely]] {
        return true;
    }

    const SurfaceInterval validate_interval(addr, addr + size);
    SurfaceRegions validate_regions;
    u32 level = 0;
    SurfaceInterval level_interval{};
    {
        // Keep this reference out of the loop. A custom upload below can grow slot_surfaces
        Surface& surface = slot_surfaces[surface_id];
        if (surface.type == SurfaceType::Fill) {
            ASSERT_MSG(surface.IsRegionValid(validate_interval),
                       "Attempted to validate a non-valid fill surface");
            return true;
        }

        validate_regions = surface.invalid_regions & validate_interval;
        if (validate_regions.empty()) {
            return true;
        }
        if (!needs_contents) {
            // What the range holds is replaced before anything reads it: a draw that covers its
            // viewport and writes depth and stencil anew in every pixel. So nothing is brought
            // to it, not from another surface and not from guest memory, which would first have
            // every surface that holds newer bytes of the range read back from the GPU, each
            // time waiting for the GPU to be done. A game that renders small colour targets
            // into the memory of its depth buffer between frames paid five such readbacks a
            // frame, two thirds of the frame's time on a slow GPU, for depth values that no
            // pixel ever saw. The draw then makes this surface the owner of the range, and the
            // others lose it without being read.
            //
            // The range is whole rows. Of a viewport narrower than the surface, the pixels
            // beside it keep what the image had there, where the long way round would have put
            // what the range's bytes mean in this format. Nothing reads those either until a
            // draw with a wider viewport comes, which finds old depth values there.
            //
            // That holds for what the image has been given before. A part of it that nothing
            // was ever written to is another matter: it is not old depth values but memory as
            // the image was made with, and the GPU must not be handed that to load or to draw
            // into beside. (New Super Mario Bros. 2 draws its map into a target of 256x416
            // through a viewport of 240x400, from a clear that is drawn; the depth image, new
            // with the scene, went to its first render pass with nothing in it, and the whole
            // device reset, on an Adreno 650 with Turnip, three times out of three.) Such a
            // part is cleared, which a render pass does as it begins and costs next to
            // nothing; where it is not whole rows the range goes the long way after all.
            bool whole_rows = true;
            boost::container::small_vector<SurfaceInterval, 4> never_given;
            for (const auto& piece : SurfaceRegions{validate_regions - surface.given_regions}) {
                whole_rows = whole_rows && surface.FromInterval(piece).GetInterval() == piece;
                never_given.push_back(piece);
            }
            if (whole_rows) {
                for (const SurfaceInterval& piece : never_given) {
                    const SurfaceParams rows = surface.FromInterval(piece);
                    const bool is_depth = surface.type == SurfaceType::Depth ||
                                          surface.type == SurfaceType::DepthStencil;
                    TextureClear clear{
                        .texture_level = surface.LevelOf(boost::icl::first(piece)),
                        .texture_rect = surface.GetScaledSubRect(rows),
                    };
                    if (is_depth) {
                        clear.value.depth = 1.f;
                        clear.value.stencil = 0;
                    } else {
                        clear.value.color = Common::Vec4f{0.f, 0.f, 0.f, 0.f};
                    }
                    runtime.ClearTexture(surface, clear);
                }
                CITRA_FRAME_DUMP("cache takes {} @{:08X} +{:X} without what it held: all of it "
                                 "is about to be overwritten{}",
                                 PixelFormatAsString(surface.pixel_format), addr, size,
                                 never_given.empty() ? ""
                                                     : "; what the image never held anything "
                                                       "for is cleared");
                surface.MarkValid(validate_interval);
                return true;
            }
        }
        level = surface.LevelOf(addr);
        level_interval = surface.LevelInterval(level);
    }

    auto notify_validated = [&](SurfaceInterval interval) {
        slot_surfaces[surface_id].MarkValid(interval);
        validate_regions.erase(interval);
    };

    const DebugScope scope{runtime, Common::Vec4f{0.f, 1.f, 0.f, 1.f},
                           "RasterizerCache::ValidateSurface (from {:#x} to {:#x})", addr,
                           addr + size};

    while (!validate_regions.empty()) {
        // A synchronous custom upload can grow slot_surfaces and invalidate Surface&
        Surface& surface = slot_surfaces[surface_id];

        // Take an invalid interval from the validation regions and clamp it
        // to the current level interval. If the interval is empty
        // then we have validated the entire level so move to the next.
        const auto interval = *validate_regions.begin() & level_interval;
        if (boost::icl::is_empty(interval)) {
            level_interval = surface.LevelInterval(++level);
            continue;
        }

        // Look for a valid surface to copy from.
        const SurfaceParams params = surface.FromInterval(interval);
        const SurfaceId copy_surface_id =
            FindMatch<MatchFlags::Copy>(params, ScaleMatch::Ignore, interval);
        if (copy_surface_id && copy_surface_id != surface_id) {
            Surface& copy_surface = slot_surfaces[copy_surface_id];
            const SurfaceInterval copy_interval = copy_surface.GetCopyableInterval(params);
            CopySurface(copy_surface, surface, copy_interval);
            notify_validated(copy_interval);
            continue;
        }

        // Try to find surface in cache with different format
        // that can can be reinterpreted to the requested format.
        if (ValidateByReinterpretation(surface, params, interval)) {
            CITRA_FRAME_DUMP("cache reinterpret -> {} {}x{} t{} @{:08X}",
                             PixelFormatAsString(surface.pixel_format), params.width, params.height,
                             static_cast<u32>(surface.type), params.addr);
            notify_validated(interval);
            continue;
        }

        // Games reuse memory for surfaces of another shape, so the newest contents of this
        // region can sit in surfaces that lay the same bytes out differently. Convert those on
        // the host GPU. Not for a caller that can do without the old contents (allow_download
        // unset, a texture copy with gaps): the conversion goes through the whole range, most
        // of which that caller is about to write.
        if (allow_download && ValidateByRelayout(surface_id, params.GetInterval())) {
            notify_validated(params.GetInterval());
            continue;
        }

        // Last resort: take the data from guest memory. If another surface holds the newest
        // data for this region, the flush downloads it from the GPU first, which has to wait
        // for the GPU to finish everything queued so far.
        if (!allow_download) {
            for (const auto& [region, owner_id] :
                 RangeFromInterval(dirty_regions, params.GetInterval())) {
                if (slot_surfaces[owner_id].type != SurfaceType::Fill) {
                    return false;
                }
            }
        }
        {
            CITRA_TRACE_SCOPE_FMT("Surface::LoadFromGuestMemory {} {}x{} type {} @{:08X}",
                                  PixelFormatAsString(surface.pixel_format), params.width,
                                  params.height, static_cast<u32>(surface.type), params.addr);
            CITRA_FRAME_DUMP("cache load-from-guest-memory {} {}x{} s{} t{} @{:08X} tiled {} "
                             "stride {} levels {} (surface {}x{} @{:08X})",
                             PixelFormatAsString(surface.pixel_format), params.width, params.height,
                             surface.res_scale, static_cast<u32>(surface.type), params.addr,
                             surface.is_tiled, surface.stride, surface.levels, surface.width,
                             surface.height, surface.addr);
            // (The draw or copy this is for has been noted with what it reads)
            WriteBack(params.addr, params.size);
            if (!use_custom_textures || !UploadCustomSurface(surface_id, interval)) {
                UploadSurface(slot_surfaces[surface_id], interval);
            }
        }
        notify_validated(params.GetInterval());
    }

    // Filtered mipmaps often look really bad. We can achieve better quality by
    // generating them from the base level.
    Surface& surface = slot_surfaces[surface_id];
    if (!surface.res_scale.IsNative() && level != 0) {
        runtime.GenerateMipmaps(surface);
    }
    return true;
}

template <class T>
void RasterizerCache<T>::UploadSurface(Surface& surface, SurfaceInterval interval) {
    MICROPROFILE_SCOPE(RasterizerCache_UploadSurface);
    CITRA_TRACE_SCOPE("Surface::Upload");
    CITRA_HITCH_SCOPE(TextureUpload);
    CITRA_TRACE_COUNT(SurfaceUploads);

    const SurfaceParams load_info = surface.FromInterval(interval);
    ASSERT(load_info.addr >= surface.addr && load_info.end <= surface.end);
    CITRA_FRAME_DUMP("cache upload {} {}x{} s{} t{} @{:08X}",
                     PixelFormatAsString(surface.pixel_format), load_info.width, load_info.height,
                     surface.res_scale, static_cast<u32>(surface.type), load_info.addr);

    // An ETC1 texture kept in its blocks takes half a byte a pixel (TranscodeETC1)
    const auto staging = runtime.FindStaging(
        surface.keeps_etc_blocks
            ? load_info.width * load_info.height / 2
            : load_info.width * load_info.height * surface.GetInternalBytesPerPixel(),
        true);

    MemoryRef source_ptr = memory.GetPhysicalRef(load_info.addr);
    if (!source_ptr) [[unlikely]] {
        return;
    }
    // Read when the command is decoded, which may be after the game was told it is done
    memory.NoteGpuReference(load_info.addr, load_info.end - load_info.addr,
                            Memory::GpuReference::Texture);

    const std::span<u8> upload_data{
        const_cast<u8*>(memory.GpuRead(load_info.addr, load_info.end - load_info.addr)),
        load_info.end - load_info.addr};
    if (surface.keeps_etc_blocks) {
        TranscodeETC1(load_info, upload_data, staging.mapped);
    } else {
        DecodeTexture(load_info, load_info.addr, load_info.end, upload_data, staging.mapped,
                      runtime.NeedsConversion(surface));
    }

    const bool should_dump = False(surface.flags & SurfaceFlagBits::Custom) &&
                             False(surface.flags & SurfaceFlagBits::RenderTarget);
    if (dump_textures && should_dump) {
        const u64 hash = ComputeHash(load_info, upload_data);
        const u32 level = surface.LevelOf(load_info.addr);
        custom_tex_manager.DumpTexture(load_info, level, upload_data, hash);
    }

    const BufferTextureCopy upload = {
        .buffer_offset = staging.offset,
        .buffer_size = staging.size,
        .texture_rect = surface.GetSubRect(load_info),
        .texture_level = surface.LevelOf(load_info.addr),
    };
    surface.Upload(upload, staging);
}

template <class T>
u64 RasterizerCache<T>::ComputeHash(const SurfaceParams& load_info, std::span<u8> upload_data) {
    if (!custom_tex_manager.UseNewHash()) {
        const u32 width = load_info.width;
        const u32 height = load_info.height;
        const u32 bpp = GetFormatBytesPerPixel(load_info.pixel_format);
        auto decoded = std::vector<u8>(width * height * bpp);
        DecodeTexture(load_info, load_info.addr, load_info.end, upload_data, decoded, false);
        return Common::ComputeHash64<Common::HashAlgo64::CityHash>(decoded.data(), decoded.size());
    } else {
        return Common::ComputeHash64<Common::HashAlgo64::CityHash>(upload_data.data(),
                                                                   upload_data.size());
    }
}

template <class T>
bool RasterizerCache<T>::UploadCustomSurface(SurfaceId surface_id, SurfaceInterval interval) {
    MICROPROFILE_SCOPE(RasterizerCache_UploadSurface);

    Surface& surface = slot_surfaces[surface_id];
    const SurfaceParams load_info = surface.FromInterval(interval);
    ASSERT(load_info.addr >= surface.addr && load_info.end <= surface.end);

    MemoryRef source_ptr = memory.GetPhysicalRef(load_info.addr);
    if (!source_ptr) [[unlikely]] {
        return false;
    }
    memory.NoteGpuReference(load_info.addr, load_info.end - load_info.addr,
                            Memory::GpuReference::Texture);

    const std::span<u8> upload_data{
        const_cast<u8*>(memory.GpuRead(load_info.addr, load_info.end - load_info.addr)),
        load_info.end - load_info.addr};
    const u64 hash = ComputeHash(load_info, upload_data);

    const u32 level = surface.LevelOf(load_info.addr);
    Material* material = custom_tex_manager.GetMaterial(hash);

    if (!material) {
        return surface.IsCustom();
    }
    if (level != 0 && custom_tex_manager.SkipMipmaps()) {
        return true;
    }

    // Dump skips a surface that already has this flag, so set it before that upload
    surface.flags |= SurfaceFlagBits::Custom;

    // Decode finishes on a later frame, so the callback checks the slot again before uploading
    const SurfaceParams expected_surface = surface;

    const auto upload = [this, level, surface_id, material, expected_surface, load_info,
                         hash]() -> bool {
        if (!slot_surfaces.contains(surface_id)) {
            return false;
        }

        // A live slot may now hold another texture, or this one may have been unregistered.
        // custom_format is left out of the compare because this upload is what sets it
        Surface& target = slot_surfaces[surface_id];
        const bool same_guest =
            target.addr == expected_surface.addr && target.end == expected_surface.end &&
            target.width == expected_surface.width && target.height == expected_surface.height &&
            target.stride == expected_surface.stride && target.levels == expected_surface.levels &&
            target.is_tiled == expected_surface.is_tiled &&
            target.texture_type == expected_surface.texture_type &&
            target.pixel_format == expected_surface.pixel_format;
        if (False(target.flags & SurfaceFlagBits::Registered) || !same_guest ||
            !runtime.SupportsCustomFormat(material->format)) {
            return false;
        }

        // Guest pixels can change without a new surface. Drop the material when they no longer
        // match the hash captured at queue time
        MemoryRef source_ptr = memory.GetPhysicalRef(load_info.addr);
        if (!source_ptr) {
            return false;
        }
        const auto upload_data = source_ptr.GetWriteBytes(load_info.end - load_info.addr);
        if (ComputeHash(load_info, upload_data) != hash) {
            return false;
        }

        // Each mip is its own material and only the base material may allocate the image. A mip
        // that arrives first would size that image to the mip
        if (level != 0) {
            if (!target.IsCustom()) {
                return false;
            }
            u32 mip_width = target.material->width >> level;
            u32 mip_height = target.material->height >> level;
            if (mip_width == 0) {
                mip_width = 1;
            }
            if (mip_height == 0) {
                mip_height = 1;
            }
            if (target.material != material &&
                (material->format != target.custom_format || material->width != mip_width ||
                 material->height != mip_height)) {
                return false;
            }
        }

        // Queuing set this flag on the old surface. Set it again for a surface that was
        // recreated with the same guest parameters before the image is replaced.
        target.flags |= SurfaceFlagBits::Custom;
        // A new base material replaces the image in this slot. The previous image is sentenced
        // and the page table keeps surface_id
        if (!target.IsCustom() || (level == 0 && target.material != material)) {
            const SurfaceBase old_surface{target};
            const SurfaceId old_id =
                slot_surfaces.swap_and_insert(surface_id, runtime, old_surface, material);
            Surface& replaced = slot_surfaces[old_id];
            replaced.flags &= ~SurfaceFlagBits::Registered;
            // The sentenced copy inherited the flag set above. A guest image is not custom
            if (!replaced.IsCustom()) {
                replaced.flags &= ~SurfaceFlagBits::Custom;
            }
            Sentence(old_id);
            // What was made for the image that goes is found under the id that stays
            slot_surfaces[surface_id].image_generation++;
        }

        // swap_and_insert can grow slot_surfaces, so the reference above may be dangling
        Surface& custom_surface = slot_surfaces[surface_id];
        custom_surface.UploadCustom(material, level);
        if (custom_tex_manager.SkipMipmaps()) {
            runtime.GenerateMipmaps(custom_surface);
        }
        return true;
    };
    return custom_tex_manager.Decode(material, std::move(upload));
}

template <class T>
void RasterizerCache<T>::DownloadSurface(Surface& surface, SurfaceInterval interval) {
    MICROPROFILE_SCOPE(RasterizerCache_DownloadSurface);
    // GPU -> guest memory readback; typically forces the CPU to wait for the GPU.
    CITRA_TRACE_SCOPE("Surface::Download");
    CITRA_HITCH_SCOPE_ABSORB(GpuReadback);
    CITRA_TRACE_COUNT(SurfaceDownloads);

    const SurfaceParams flush_info = surface.FromInterval(interval);
    const u32 flush_start = boost::icl::first(interval);
    const u32 flush_end = boost::icl::last_next(interval);
    ASSERT(flush_start >= surface.addr && flush_end <= surface.end);

    // Identifies what is being read back.
    CITRA_TRACE_SCOPE_FMT("download {} {}x{} scale {} type {} @{:08X}",
                          PixelFormatAsString(surface.pixel_format), flush_info.width,
                          flush_info.height, surface.res_scale, static_cast<u32>(surface.type),
                          flush_info.addr);
    CITRA_FRAME_DUMP("cache DOWNLOAD (waits for GPU) {} {}x{} s{} t{} @{:08X} tiled {} stride {} "
                     "levels {} (surface {}x{} @{:08X})",
                     PixelFormatAsString(surface.pixel_format), flush_info.width, flush_info.height,
                     surface.res_scale, static_cast<u32>(surface.type), flush_info.addr,
                     surface.is_tiled, surface.stride, surface.levels, surface.width,
                     surface.height, surface.addr);

    const auto staging = runtime.FindStaging(
        flush_info.width * flush_info.height * surface.GetInternalBytesPerPixel(), false);

    const BufferTextureCopy download = {
        .buffer_offset = staging.offset,
        .buffer_size = staging.size,
        .texture_rect = surface.GetSubRect(flush_info),
        .texture_level = surface.LevelOf(flush_start),
    };
    surface.Download(download, staging);
    runtime.InvalidateStaging(staging);

    MemoryRef dest_ptr = memory.GetPhysicalRef(flush_start);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const auto download_dest = dest_ptr.GetWriteBytes(flush_end - flush_start);
    EncodeTexture(flush_info, flush_start, flush_end, staging.mapped, download_dest,
                  runtime.NeedsConversion(surface));
}

template <class T>
void RasterizerCache<T>::DownloadFillSurface(Surface& surface, SurfaceInterval interval) {
    const u32 flush_start = boost::icl::first(interval);
    const u32 flush_end = boost::icl::last_next(interval);
    ASSERT(flush_start >= surface.addr && flush_end <= surface.end);

    MemoryRef dest_ptr = memory.GetPhysicalRef(surface.addr);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const u32 start_offset = flush_start - surface.addr;
    const u32 end_offset =
        std::clamp(flush_end - surface.addr, 0u, static_cast<u32>(dest_ptr.GetSize()));
    const u32 coarse_start_offset = start_offset - (start_offset % surface.fill_size);
    const u32 backup_bytes = start_offset % surface.fill_size;

    std::array<u8, 4> backup_data;
    if (backup_bytes) {
        std::memcpy(backup_data.data(), &dest_ptr[coarse_start_offset], backup_bytes);
    }

    for (u32 offset = coarse_start_offset; offset < end_offset; offset += surface.fill_size) {
        std::memcpy(&dest_ptr[offset], &surface.fill_data[0],
                    std::min(surface.fill_size, end_offset - offset));
    }

    if (backup_bytes) {
        std::memcpy(&dest_ptr[coarse_start_offset], &backup_data[0], backup_bytes);
    }
}

template <class T>
bool RasterizerCache<T>::ValidateByRelayout(SurfaceId surface_id, const SurfaceInterval& interval) {
    if (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyLayoutReadback) {
        return false;
    }

    Surface& surface = slot_surfaces[surface_id];
    if (surface.levels != 1 || surface.stride != surface.width ||
        surface.texture_type != TextureType::Texture2D || surface.half_scale_x ||
        surface.half_scale_y) {
        return false;
    }
    const u32 pixel_size = surface.BytesInPixels(1);
    const auto pixel_aligned = [&](PAddr addr) { return (addr - surface.addr) % pixel_size == 0; };

    // Everything the surface is missing in the interval must come from surfaces that can be
    // converted. Nothing may have to come from guest memory, as an upload works in rectangles
    // and would overwrite what is converted.
    const SurfaceRegions wanted = surface.invalid_regions & interval;
    SurfaceRegions missing = wanted;
    boost::container::small_vector<std::pair<SurfaceInterval, SurfaceId>, 4> sources;
    for (const auto& [region, owner_id] : RangeFromInterval(dirty_regions, interval)) {
        const SurfaceInterval part = region & interval;
        // What the surface already has needs no source, whoever holds the newest copy.
        if (owner_id == surface_id || (wanted & part).empty()) {
            continue;
        }
        const Surface& owner = slot_surfaces[owner_id];
        // A fill that could not be taken as a clear: its value is no pixel of this surface, or it
        // ends inside a row of it. It has no image to convert and no format, so no pixel size
        // either. Its bytes are written to guest memory, without a wait for the GPU.
        if (owner.type == SurfaceType::Fill) {
            CITRA_FRAME_DUMP("cache no relayout to {} {}x{} @{:08X}: [{:08X}-{:08X}] is a fill of "
                             "{} bytes",
                             PixelFormatAsString(surface.pixel_format), surface.width,
                             surface.height, surface.addr, part.lower(), part.upper(),
                             owner.fill_size);
            return false;
        }
        // A source with pixels of another size is gone through by the byte, wherever it starts
        const bool same_pixel_size = owner.BytesInPixels(1) == pixel_size;
        if (owner.levels != 1 || owner.stride != owner.width ||
            owner.texture_type != TextureType::Texture2D ||
            (same_pixel_size && !pixel_aligned(owner.addr)) || !pixel_aligned(part.lower()) ||
            !pixel_aligned(part.upper()) || !runtime.CanRelayoutTextures(owner, surface)) {
            return false;
        }
        missing -= part;
        sources.emplace_back(part, owner_id);
    }
    if (sources.empty() || !missing.empty()) {
        return false;
    }

    for (const auto& [part, owner_id] : sources) {
        Surface& owner = slot_surfaces[owner_id];
        CITRA_FRAME_DUMP("cache relayout {} {}x{} t{} @{:08X} -> {} {}x{} t{} @{:08X} "
                         "[{:08X}-{:08X}]",
                         PixelFormatAsString(owner.pixel_format), owner.width, owner.height,
                         static_cast<u32>(owner.type), owner.addr,
                         PixelFormatAsString(surface.pixel_format), surface.width, surface.height,
                         static_cast<u32>(surface.type), surface.addr, part.lower(), part.upper());
        const s32 byte_delta = static_cast<s32>(surface.addr) - static_cast<s32>(owner.addr);
        const TextureRelayout relayout = {
            .dst_rect = surface.GetScaledSubRect(surface.FromInterval(part)),
            .pixel_delta = owner.BytesInPixels(1) == pixel_size
                               ? byte_delta / static_cast<s32>(pixel_size)
                               : byte_delta,
            .first_pixel = (part.lower() - surface.addr) / pixel_size,
            .end_pixel = (part.upper() - surface.addr) / pixel_size,
        };
        runtime.RelayoutTextures(owner, surface, relayout);
    }
    return true;
}

template <class T>
bool RasterizerCache<T>::IsNativeScaleShadowMap(const SurfaceParams& surface) const {
    if (!surface.res_scale.IsNative() || resolution_scale_factor.IsNative() ||
        (Common::TraceMarkers::Experiments() & Common::TraceMarkers::LegacyShadowMapScale)) {
        return false;
    }
    const auto target = std::ranges::find(shadow_map_targets, surface.addr, &ShadowMapTarget::addr);
    return target != shadow_map_targets.end() && !target->with_depth;
}

template <class T>
bool RasterizerCache<T>::ValidateByReinterpretation(Surface& surface, SurfaceParams params,
                                                    const SurfaceInterval& interval) {
    SurfaceId reinterpret_id =
        FindMatch<MatchFlags::Reinterpret>(params, ScaleMatch::Ignore, interval);
    if (reinterpret_id) {
        Surface& src_surface = slot_surfaces[reinterpret_id];
        const SurfaceInterval copy_interval = src_surface.GetCopyableInterval(params);
        if (boost::icl::is_empty(copy_interval & interval)) {
            return false;
        }
        const Common::ResolutionScale res_scale = src_surface.res_scale;
        const u8 sample_count = src_surface.sample_count;
        if (res_scale > surface.res_scale && IsNativeScaleShadowMap(surface)) {
            return false;
        }
        if ((res_scale > surface.res_scale) || (sample_count > surface.sample_count)) {
            surface.ScaleUp(res_scale, sample_count);
        }
        if (src_surface.half_scale_x != surface.half_scale_x ||
            src_surface.half_scale_y != surface.half_scale_y) {
            return false;
        }
        const PAddr addr = boost::icl::lower(interval);
        const SurfaceParams copy_params = surface.FromInterval(copy_interval);
        const auto src_rect = src_surface.GetScaledSubRect(copy_params);
        const auto dst_rect = surface.GetScaledSubRect(copy_params);
        const TextureCopy reinterpret = {
            .src_level = src_surface.LevelOf(addr),
            .dst_level = surface.LevelOf(addr),
            .src_offset = {src_rect.left, src_rect.bottom},
            .dst_offset = {dst_rect.left, dst_rect.bottom},
            .extent = {std::min(src_rect.GetWidth(), dst_rect.GetWidth()),
                       std::min(src_rect.GetHeight(), dst_rect.GetHeight())},
        };
        return runtime.Reinterpret(src_surface, surface, reinterpret);
    }

    bool has_invalid = false;
    const PAddr addr = boost::icl::lower(interval);
    const u32 size = boost::icl::length(interval);

    ForEachSurfaceInRegion(addr, size, [&](auto, auto& cached_surface) {
        if (cached_surface.pixel_format == PixelFormat::Invalid) {
            has_invalid = true;
            return true;
        }
        return false;
    });

    if (has_invalid) {
        return false;
    }

    if (!boost::icl::contains(dirty_regions, interval)) {
        return false;
    }

    const auto it = dirty_regions.find(interval);
    if (it == dirty_regions.end() || !it->second) {
        return false;
    }

    return slot_surfaces[it->second].stride != surface.stride;
}

template <class T>
void RasterizerCache<T>::ClearAll(bool flush) {
    const auto flush_interval = PageMap::interval_type::right_open(0x0, 0xFFFFFFFF);
    // Force flush all surfaces from the cache
    if (flush) {
        WriteBack(0x0, 0xFFFFFFFF);
    }
    // Unmark all of the marked pages
    for (auto& pair : RangeFromInterval(cached_pages, flush_interval)) {
        const auto interval = pair.first & flush_interval;

        const PAddr interval_start_addr = boost::icl::first(interval) << Memory::CITRA_PAGE_BITS;
        const PAddr interval_end_addr = boost::icl::last_next(interval) << Memory::CITRA_PAGE_BITS;
        const u32 interval_size = interval_end_addr - interval_start_addr;

        memory.RasterizerMarkRegionCached(interval_start_addr, interval_size, false);
    }

    // Remove the whole cache without really looking at it.
    last_target.valid = false;
    surface_set_epoch++;
    memory.RasterizerClearSurfaceCells();
    readable_pages.clear();
    readable_count = 0;
    cached_pages -= flush_interval;
    dirty_regions.clear();
    page_table.clear();
}

template <class T>
void RasterizerCache<T>::FlushRegion(PAddr addr, u32 size) {
    FrameLiveness::NoteGuestRead(addr, size);
    WriteBack(addr, size);
}

template <class T>
void RasterizerCache<T>::WriteBack(PAddr addr, u32 size, SurfaceId flush_surface_id) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval flush_interval(addr, addr + size);
    SurfaceRegions flushed_intervals;

    for (const auto& [region, surface_id] : RangeFromInterval(dirty_regions, flush_interval)) {
        if (flush_surface_id && surface_id != flush_surface_id) {
            continue;
        }

        // Small sizes imply that this most likely comes from the cpu, flush the entire region
        // the point is to avoid thousands of small writes every frame if the cpu decides to
        // access that region, anything higher than 8 you're guaranteed it comes from a service
        const auto interval = size <= 8 ? region : region & flush_interval;
        Surface& surface = slot_surfaces[surface_id];
        ASSERT_MSG(surface.IsRegionValid(interval), "Region owner has invalid regions");

        const DebugScope scope{runtime, Common::Vec4f{0.f, 0.f, 0.f, 1.f},
                               "RasterizerCache::FlushRegion (from {:#x} to {:#x})",
                               interval.lower(), interval.upper()};

        SCOPE_EXIT({ flushed_intervals += interval; });
        // The emulated memory is about to change under whoever reads it in place
        if (on_guest_write) {
            on_guest_write(interval.lower(), interval.upper() - interval.lower());
        }
        if (surface.type == SurfaceType::Fill) {
            DownloadFillSurface(surface, interval);
            continue;
        }

        // Download each requested level of the surface.
        const u32 start_level = surface.LevelOf(interval.lower());
        const u32 end_level = surface.LevelOf(interval.upper());
        for (u32 level = start_level; level <= end_level; level++) {
            const auto download_interval = interval & surface.LevelInterval(level);
            if (boost::icl::is_empty(download_interval)) {
                continue;
            }
            DownloadSurface(surface, download_interval);
        }
    }

    // Reset dirty regions
    dirty_regions -= flushed_intervals;
}

template <class T>
void RasterizerCache<T>::FlushAll() {
    // (Putting everything into guest memory, as before a save state, is nobody reading it)
    WriteBack(0, 0xFFFFFFFF);
}

template <class T>
void RasterizerCache<T>::InvalidateRegion(PAddr addr, u32 size, SurfaceId region_owner_id) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval invalid_interval(addr, addr + size);

    if (region_owner_id) {
        Surface& region_owner = slot_surfaces[region_owner_id];
        ASSERT(region_owner.type != SurfaceType::Texture);
        ASSERT(addr >= region_owner.addr && addr + size <= region_owner.end);
        ASSERT(region_owner.width == region_owner.stride);
        // A draw into what the draw before drew into finds it valid already
        if (region_owner.IsRegionValid(invalid_interval)) {
            region_owner.modification_tick++;
        } else {
            region_owner.MarkValid(invalid_interval);
        }
    }

    boost::container::small_vector<SurfaceId, 4> remove_surfaces;
    ForEachSurfaceInRegion(addr, size, [&](SurfaceId surface_id, Surface& surface) {
        if (surface_id == region_owner_id) {
            return;
        }
        // If the CPU is invalidating this region we want to remove it
        // to (likely) mark the memory pages as uncached
        if (!region_owner_id && size <= 8) {
            WriteBack(surface.addr, surface.size, surface_id);
            remove_surfaces.push_back(surface_id);
            return;
        }
        const auto interval = surface.GetInterval() & invalid_interval;
        if (boost::icl::contains(surface.invalid_regions, interval)) {
            // As it was: invalid there already, and not gone, so not invalid throughout
            return;
        }
        surface.MarkInvalid(interval);
        if (!surface.IsFullyInvalid()) {
            return;
        }
        remove_surfaces.push_back(surface_id);
    });

    if (region_owner_id) {
        DropReadable(addr, size);
        memory.NoteGpuWritten(addr, size);
        // A draw into what the draw before drew into finds the surface noted as the one
        // that has the newest of the stretch: setting it again takes the stretch out of its
        // piece and puts it back, to end up with the same piece
        const auto held = dirty_regions.find(invalid_interval);
        if (held == dirty_regions.end() || held->second != region_owner_id ||
            !boost::icl::contains(held->first, invalid_interval)) {
            dirty_regions.set({invalid_interval, region_owner_id});
        }
    } else {
        dirty_regions.erase(invalid_interval);
    }

    for (const SurfaceId surface_id : remove_surfaces) {
        UnregisterSurface(surface_id);
    }
}

template <class T>
void RasterizerCache<T>::InvalidateRows(PAddr addr, u32 row_size, u32 row_stride, u32 rows,
                                        SurfaceId region_owner_id) {
    if (row_size == 0 || rows == 0) [[unlikely]] {
        return;
    }

    const u32 size = (rows - 1) * row_stride + row_size;
    DropReadable(addr, size);
    memory.NoteGpuWritten(addr, size);
    Surface& region_owner = slot_surfaces[region_owner_id];
    ASSERT(region_owner.type != SurfaceType::Texture);
    ASSERT(addr >= region_owner.addr && addr + size <= region_owner.end);
    ASSERT(region_owner.width == region_owner.stride);

    // A copy like this comes every frame, to the same rows. What it has to note is noted row
    // by row, since what lies between the rows stays as it is: but not by looking each row up
    // from the top of a tree and cutting it out of the piece it lies in (three searches and a
    // split for each row, in each of three sets: with the 416 rows Luigi's Mansion 2 copies
    // every frame that was a tenth of the decoding thread's time). The rows come in the order
    // of their addresses, and so do the pieces of a set: both are walked side by side, and a
    // row goes in next to the piece before it. A piece that holds several rows is taken out
    // once and put back around them. Only a row that lies partly in a piece is done the
    // general way. The sets end up as they did before.
    const auto row_interval = [&](u32 row) {
        const PAddr row_addr = addr + row * row_stride;
        return SurfaceInterval(row_addr, row_addr + row_size);
    };
    const SurfaceInterval span(addr, addr + size);
    /// For a frame dump: rows put in where there was nothing, rows cut out of a piece, rows
    /// that were as they have to be, rows done the general way
    struct RowWalk {
        u32 fresh{};
        u32 cut{};
        u32 as_is{};
        u32 searched{};
    };

    // Who has the newest of the rows: this surface
    RowWalk newest{};
    {
        auto piece = dirty_regions.lower_bound(span);
        u32 row = 0;
        while (row < rows) {
            const SurfaceInterval interval = row_interval(row);
            while (piece != dirty_regions.end() && piece->first.upper() <= interval.lower()) {
                ++piece;
            }
            if (piece == dirty_regions.end() || piece->first.lower() >= interval.upper()) {
                // Nobody's: in it goes, before the piece that follows
                piece = dirty_regions.insert(piece, {interval, region_owner_id});
                newest.fresh++;
                row++;
                continue;
            }
            const PAddr piece_lower = piece->first.lower();
            const PAddr piece_upper = piece->first.upper();
            if (piece_lower > interval.lower() || piece_upper < interval.upper()) {
                dirty_regions.set({interval, region_owner_id});
                piece = dirty_regions.lower_bound(interval);
                newest.searched++;
                row++;
                continue;
            }
            const SurfaceId other = piece->second;
            if (other == region_owner_id) {
                // This surface's already, and so are the rows after it within the piece
                do {
                    newest.as_is++;
                    row++;
                } while (row < rows && row_interval(row).upper() <= piece_upper);
                continue;
            }
            // In a piece of another surface, like the rows after it as far as the piece goes
            auto at = std::next(piece);
            dirty_regions.erase(piece);
            PAddr restored_to = piece_lower;
            while (row < rows) {
                const SurfaceInterval inside = row_interval(row);
                if (inside.upper() > piece_upper) {
                    break;
                }
                if (restored_to < inside.lower()) {
                    at = dirty_regions.insert(
                        at, {SurfaceInterval(restored_to, inside.lower()), other});
                }
                at = dirty_regions.insert(at, {inside, region_owner_id});
                restored_to = inside.upper();
                newest.cut++;
                row++;
            }
            if (restored_to < piece_upper) {
                at = dirty_regions.insert(at, {SurfaceInterval(restored_to, piece_upper), other});
            }
            piece = at;
        }
    }

    // The rows are valid in this surface
    RowWalk valid{};
    {
        SurfaceRegions& invalid = region_owner.invalid_regions;
        auto piece = invalid.lower_bound(span);
        u32 row = 0;
        while (row < rows) {
            const SurfaceInterval interval = row_interval(row);
            while (piece != invalid.end() && piece->upper() <= interval.lower()) {
                ++piece;
            }
            if (piece == invalid.end() || piece->lower() >= interval.upper()) {
                valid.as_is++;
                row++;
                continue;
            }
            const PAddr piece_lower = piece->lower();
            const PAddr piece_upper = piece->upper();
            if (piece_lower > interval.lower() || piece_upper < interval.upper()) {
                invalid.erase(interval);
                piece = invalid.lower_bound(interval);
                valid.searched++;
                row++;
                continue;
            }
            auto at = std::next(piece);
            invalid.erase(piece);
            PAddr restored_to = piece_lower;
            while (row < rows) {
                const SurfaceInterval inside = row_interval(row);
                if (inside.upper() > piece_upper) {
                    break;
                }
                if (restored_to < inside.lower()) {
                    at = invalid.insert(at, SurfaceInterval(restored_to, inside.lower()));
                }
                restored_to = inside.upper();
                valid.cut++;
                row++;
            }
            if (restored_to < piece_upper) {
                at = invalid.insert(at, SurfaceInterval(restored_to, piece_upper));
            }
            piece = at;
        }
        if (valid.cut + valid.searched != 0) {
            // (What was valid before was given before)
            if (!boost::icl::contains(region_owner.given_regions, span)) {
                for (u32 given = 0; given < rows; given++) {
                    region_owner.given_regions.insert(row_interval(given));
                }
            }
            SurfaceBase::validity_epoch++;
        }
    }
    // What the rows hold is new in any case
    region_owner.modification_tick++;
    CITRA_FRAME_DUMP("  texcopy rows: the newest of {} rows is this surface's: {} put in where "
                     "nobody had it, {} cut out of another surface's piece, {} were, {} the "
                     "general way; valid in it: {} cut out, {} were, {} the general way",
                     rows, newest.fresh, newest.cut, newest.as_is, newest.searched, valid.cut,
                     valid.as_is, valid.searched);

    boost::container::small_vector<SurfaceId, 4> remove_surfaces;
    ForEachSurfaceInRegion(addr, size, [&](SurfaceId surface_id, Surface& surface) {
        if (surface_id == region_owner_id) {
            return;
        }
        const SurfaceInterval whole = surface.GetInterval();
        const SurfaceInterval within = span & whole;
        if (boost::icl::is_empty(within)) {
            return;
        }
        // The rows are not valid in any other surface
        RowWalk theirs{};
        SurfaceRegions& invalid = surface.invalid_regions;
        auto piece = invalid.lower_bound(within);
        for (u32 row = 0; row < rows; row++) {
            const SurfaceInterval part = row_interval(row) & whole;
            if (boost::icl::is_empty(part)) {
                continue;
            }
            while (piece != invalid.end() && piece->upper() <= part.lower()) {
                ++piece;
            }
            if (piece == invalid.end() || piece->lower() >= part.upper()) {
                piece = invalid.insert(piece, part);
                theirs.fresh++;
            } else if (piece->lower() <= part.lower() && piece->upper() >= part.upper()) {
                theirs.as_is++;
            } else {
                invalid.insert(part);
                piece = invalid.lower_bound(part);
                theirs.searched++;
            }
        }
        CITRA_FRAME_DUMP("  texcopy rows: surface at {:08X} ({} bytes): {} rows made invalid in "
                         "it, {} were, {} the general way",
                         surface.addr, surface.size, theirs.fresh, theirs.as_is, theirs.searched);
        const bool changed = theirs.fresh + theirs.searched != 0;
        if (changed) {
            SurfaceBase::validity_epoch++;
        }
        if (!changed) {
            return;
        }
        surface.modification_tick++;
        if (surface.IsFullyInvalid()) {
            remove_surfaces.push_back(surface_id);
        }
    });

    for (const SurfaceId surface_id : remove_surfaces) {
        UnregisterSurface(surface_id);
    }
}

template <class T>
SurfaceId RasterizerCache<T>::CreateSurface(const SurfaceParams& params,
                                            const SurfaceFlagBits& initial_flags) {
    last_target.valid = false;
    surface_set_epoch++;
    const SurfaceId surface_id = [&] {
        const u64 resource_free_tick = runtime.GetResourceFreeTick();
        const auto matches = [&](const auto& entry) {
            const Surface& candidate = slot_surfaces[entry.surface_id];
            return
                // The surface parameters match the incoming texture
                (candidate == params)
                // Its resolution scale is equal or less than the incoming texture
                && (candidate.res_scale <= params.res_scale) &&
                (candidate.half_scale_x == params.half_scale_x) &&
                (candidate.half_scale_y == params.half_scale_y)
                // Its sample count is equal or less than the incoming texture
                && (candidate.sample_count <= params.sample_count);
        };
        // Try to find a matching texture in the deletion queue that the texture-runtime is
        // completely done with
        auto it = std::find_if(sentenced.begin(), sentenced.end(), [&](const auto& entry) {
            return resource_free_tick > entry.last_used_tick && matches(entry);
        });
        // Otherwise the one that was given up first, and has exactly the images that are needed,
        // although the GPU is not through with it. A game that fills its targets' memory gives
        // each of them up every frame, and a GPU that is kept busy is through with nothing of
        // that for a frame or two: at a high resolution a new image for every target of those
        // frames was hundreds of megabytes.
        if (it == sentenced.end()) {
            it = std::find_if(sentenced.begin(), sentenced.end(), [&](const auto& entry) {
                const Surface& candidate = slot_surfaces[entry.surface_id];
                return matches(entry) && candidate.res_scale == params.res_scale &&
                       candidate.sample_count == params.sample_count &&
                       runtime.MayTakeOverNow(candidate);
            });
            if (it != sentenced.end()) {
                taken_over_early++;
            }
        } else {
            taken_over++;
        }
        // If a matching texture was found in the deletion queue, recycle it.
        if (it != sentenced.end()) {
            const SurfaceId surface_id = it->surface_id;
            sentenced.erase(it);
            return surface_id;
        }
        if (params.type != SurfaceType::Texture) {
            made_new++;
            if (std::ranges::any_of(sentenced, matches)) {
                made_new_waiting++;
            }
        }
        return slot_surfaces.insert(runtime, params, initial_flags);
    }();
    Surface& surface = slot_surfaces[surface_id];
    if ((params.res_scale > surface.res_scale) || (params.sample_count > surface.sample_count)) {
        // Texture is going to be upscaled, remove any previous framebuffer references
        RemoveFramebuffers(surface_id);
        surface.ScaleUp(params.res_scale, params.sample_count);
    }
    surface.MarkInvalid(surface.GetInterval());
    return surface_id;
}

template <class T>
void RasterizerCache<T>::RegisterSurface(SurfaceId surface_id) {
    last_target.valid = false;
    surface_set_epoch++;
    Surface& surface = slot_surfaces[surface_id];
    ASSERT_MSG(False(surface.flags & SurfaceFlagBits::Registered),
               "Trying to register an already registered surface");

    surface.flags |= SurfaceFlagBits::Registered;
    UpdatePagesCachedCount(surface.addr, surface.size, 1);
    memory.RasterizerSetSurfaceCells(surface.addr, surface.size, true);
    ForEachPage(surface.addr, surface.size,
                [this, surface_id](u64 page) { page_table[page].push_back(surface_id); });
}

template <class T>
void RasterizerCache<T>::UnregisterSurface(SurfaceId surface_id) {
    last_target.valid = false;
    surface_set_epoch++;
    Surface& surface = slot_surfaces[surface_id];
    ASSERT_MSG(True(surface.flags & SurfaceFlagBits::Registered),
               "Trying to unregister an already unregistered surface");
    if (surface.type != SurfaceType::Fill) {
        runtime.NoteSurfaceRemoved(surface);
    }

    surface.flags &= ~SurfaceFlagBits::Registered;
    UpdatePagesCachedCount(surface.addr, surface.size, -1);
    ForEachPage(surface.addr, surface.size, [this, surface_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == page_table.end()) {
            ASSERT_MSG(false, "Unregistering unregistered page=0x{:x}", page << CITRA_PAGEBITS);
            return;
        }
        std::vector<SurfaceId>& surfaces = page_it.value();
        const auto vector_it = std::find(surfaces.begin(), surfaces.end(), surface_id);
        if (vector_it == surfaces.end()) {
            ASSERT_MSG(false, "Unregistering unregistered surface in page=0x{:x}",
                       page << CITRA_PAGEBITS);
            return;
        }
        surfaces.erase(vector_it);
    });

    // What it covered is free for the game to write without a word, but for what the
    // surfaces that are left still cover of it
    memory.RasterizerSetSurfaceCells(surface.addr, surface.size, false);
    ForEachSurfaceInRegion(surface.addr, surface.size, [this](SurfaceId, Surface& other) {
        memory.RasterizerSetSurfaceCells(other.addr, other.size, true);
    });

    if (surface.type != SurfaceType::Fill) {
        RemoveTextureCubeFace(surface_id);
        Sentence(surface_id);
        return;
    }

    slot_surfaces.erase(surface_id);
}

template <class T>
void RasterizerCache<T>::UnregisterAll() {
    FlushAll();
    for (auto& [page, surfaces] : page_table) {
        while (!surfaces.empty()) {
            UnregisterSurface(surfaces.back());
        }
    }
    runtime.Finish();
    RunGarbageCollector(true);
}

template <class T>
void RasterizerCache<T>::NoteReadbackForGame(PAddr addr, u32 size) {
    /// Pixels of a target that may be given late, and frames in a row it has to be read in
    static constexpr u32 LATE_PIXELS = 256;
    static constexpr u32 LATE_STREAK = 8;
    static constexpr u64 LATE_FRAME_GAP = 4;
    if (size == 0 || dirty_regions.empty() ||
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacySmallReadbacks) != 0) {
        return;
    }
    const SurfaceInterval interval(addr, addr + size);
    for (const auto& [region, surface_id] : RangeFromInterval(dirty_regions, interval)) {
        const Surface& surface = slot_surfaces[surface_id];
        if (surface.type != SurfaceType::Color || surface.levels != 1 ||
            surface.width * surface.height > LATE_PIXELS) {
            continue;
        }
        auto entry = std::ranges::find_if(late_readbacks, [&](const LateReadback& candidate) {
            return candidate.addr == surface.addr && candidate.size == surface.size;
        });
        if (entry == late_readbacks.end()) {
            if (late_readbacks.size() >= 16) {
                continue;
            }
            late_readbacks.push_back({.addr = surface.addr, .size = surface.size});
            entry = late_readbacks.end() - 1;
        }
        if (late_frame != entry->last_read_frame) {
            entry->streak =
                late_frame - entry->last_read_frame <= LATE_FRAME_GAP ? entry->streak + 1 : 1;
            entry->last_read_frame = late_frame;
        }
        if (entry->streak < LATE_STREAK) {
            continue;
        }
        if (!entry->late) {
            entry->late = true;
            LOG_INFO(HW_GPU,
                     "The game reads a {}x{} image back every frame (at {:08X}): it gets it a "
                     "frame late from here on, without the wait for the GPU",
                     surface.width, surface.height, surface.addr);
        }
        // This read still waits, and leaves memory as the surface is; the next ones do not
        memory.ExemptReads(surface.addr, surface.size);
    }
}

template <class T>
void RasterizerCache<T>::ServeLateReadbacks() {
    for (LateReadback& entry : late_readbacks) {
        if (!entry.late) {
            continue;
        }
        if (entry.pending && runtime.GetResourceFreeTick() >= entry.pending_tick) {
            runtime.InvalidateStaging(entry.staging);
            std::vector<u8> bytes(entry.info.end - entry.info.addr);
            EncodeTexture(entry.info, entry.info.addr, entry.info.end, entry.staging.mapped, bytes,
                          entry.convert);
            memory.PostLateWrite(entry.info.addr, std::move(bytes));
            entry.pending = false;
        }
        if (entry.pending) {
            continue;
        }
        // What the host GPU holds newer of it now, if anything
        const SurfaceInterval whole(entry.addr, entry.addr + entry.size);
        for (const auto& [region, surface_id] : RangeFromInterval(dirty_regions, whole)) {
            Surface& surface = slot_surfaces[surface_id];
            if (surface.addr != entry.addr || surface.size != entry.size ||
                surface.type != SurfaceType::Color || !surface.IsRegionValid(whole)) {
                continue;
            }
            const SurfaceParams info = surface.FromInterval(whole);
            const auto staging = runtime.FindStaging(
                info.width * info.height * surface.GetInternalBytesPerPixel(), false);
            const BufferTextureCopy download = {
                .buffer_offset = staging.offset,
                .buffer_size = staging.size,
                .texture_rect = surface.GetSubRect(info),
                .texture_level = surface.LevelOf(entry.addr),
            };
            entry.pending_tick = runtime.GetResourceTick();
            surface.Download(download, staging, false);
            entry.staging = staging;
            entry.info = info;
            entry.convert = runtime.NeedsConversion(surface);
            entry.pending = true;
            // On its way to the GPU now, not with the next frame's work
            runtime.Finish();
            break;
        }
    }
}

template <class T>
void RasterizerCache<T>::NoteCpuRead(PAddr addr, u32 size) {
    if (size == 0) {
        return;
    }
    const u32 first = addr >> Memory::CITRA_PAGE_BITS;
    const u32 last = (addr + size - 1) >> Memory::CITRA_PAGE_BITS;
    for (u32 page = first; page <= last; page++) {
        const std::size_t word = page / 64;
        const u64 bit = u64{1} << (page % 64);
        if (word < readable_pages.size() && (readable_pages[word] & bit) != 0) {
            continue;
        }
        // A page a surface has marked, with nothing on it that the host GPU has newer
        const PAddr start = page << Memory::CITRA_PAGE_BITS;
        if (cached_pages.find(page) == cached_pages.end()) {
            continue;
        }
        if (HoldsNewer(start, Memory::CITRA_PAGE_SIZE)) {
            // Not to be read freely. Reads of what merely shares the page with the surface
            // need not come here again (see Memory::MemorySystem::Impl::ReadTouchesSurface)
            memory.RasterizerNotePageHoldsNewer(start);
            continue;
        }
        if (word >= readable_pages.size()) {
            readable_pages.resize(word + 1);
        }
        readable_pages[word] |= bit;
        ++readable_count;
        memory.RasterizerMarkRegionReadable(start, Memory::CITRA_PAGE_SIZE, true);
    }
}

template <class T>
void RasterizerCache<T>::DropReadable(PAddr addr, u32 size, bool tell_memory) {
    if (readable_count == 0 || size == 0) {
        return;
    }
    const u64 first = addr >> Memory::CITRA_PAGE_BITS;
    const u64 last = std::min<u64>((static_cast<u64>(addr) + size - 1) >> Memory::CITRA_PAGE_BITS,
                                   readable_pages.size() * 64 - 1);
    for (u64 page = first; page <= last; page++) {
        const std::size_t word = page / 64;
        if (readable_pages[word] == 0) {
            // Nothing in these 64 pages
            page = (word + 1) * 64 - 1;
            continue;
        }
        const u64 bit = u64{1} << (page % 64);
        if ((readable_pages[word] & bit) == 0) {
            continue;
        }
        readable_pages[word] &= ~bit;
        --readable_count;
        if (tell_memory) {
            memory.RasterizerMarkRegionReadable(static_cast<PAddr>(page << Memory::CITRA_PAGE_BITS),
                                                Memory::CITRA_PAGE_SIZE, false);
        }
    }
}

template <class T>
void RasterizerCache<T>::UpdatePagesCachedCount(PAddr addr, u32 size, int delta) {
    const u32 num_pages =
        ((addr + size - 1) >> Memory::CITRA_PAGE_BITS) - (addr >> Memory::CITRA_PAGE_BITS) + 1;
    const u32 page_start = addr >> Memory::CITRA_PAGE_BITS;
    const u32 page_end = page_start + num_pages;

    // Interval maps will erase segments if count reaches 0, so if delta is negative we have to
    // subtract after iterating
    const auto pages_interval = PageMap::interval_type::right_open(page_start, page_end);
    if (delta > 0) {
        cached_pages.add({pages_interval, delta});
    }

    for (const auto& pair : RangeFromInterval(cached_pages, pages_interval)) {
        const auto interval = pair.first & pages_interval;
        const int count = pair.second;

        const PAddr interval_start_addr = boost::icl::first(interval) << Memory::CITRA_PAGE_BITS;
        const PAddr interval_end_addr = boost::icl::last_next(interval) << Memory::CITRA_PAGE_BITS;
        const u32 interval_size = interval_end_addr - interval_start_addr;

        if (delta > 0 && count == delta) {
            memory.RasterizerMarkRegionCached(interval_start_addr, interval_size, true);
        } else if (delta < 0 && count == -delta) {
            DropReadable(interval_start_addr, interval_size, false);
            memory.RasterizerMarkRegionCached(interval_start_addr, interval_size, false);
        } else {
            ASSERT(count >= 0);
        }
    }

    if (delta < 0) {
        cached_pages.add({pages_interval, delta});
    }
}

} // namespace VideoCore
