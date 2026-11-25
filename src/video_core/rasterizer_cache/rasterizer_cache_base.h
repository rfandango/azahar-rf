// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <functional>
#include <list>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include <boost/icl/interval_map.hpp>
#include <tsl/robin_map.h>

#include "video_core/rasterizer_cache/framebuffer_base.h"
#include "video_core/rasterizer_cache/node_pool.h"
#include "video_core/rasterizer_cache/sampler_params.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/texture_cube.h"
#include "video_core/rasterizer_cache/utils.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
struct RegsInternal;
struct DisplayTransferConfig;
struct MemoryFillConfig;
} // namespace Pica

namespace Pica::Texture {
struct TextureInfo;
}

namespace Settings {
enum class TextureFilter : u32;
}

namespace VideoCore {

enum class ScaleMatch {
    Exact,   ///< Only accept same res scale
    Upscale, ///< Only allow higher scale than params
    Ignore   ///< Accept every scaled res
};

enum class MatchFlags {
    Exact = 1 << 0,       ///< Surface perfectly matches params
    SubRect = 1 << 1,     ///< Surface encompasses params
    Copy = 1 << 2,        ///< Surface that can be used as a copy source
    TexCopy = 1 << 3,     ///< Surface that will match a display transfer "texture copy" parameters
    Reinterpret = 1 << 4, ///< Surface might have different pixel format.
};

DECLARE_ENUM_FLAG_OPERATORS(MatchFlags);

class CustomTexManager;
class RendererBase;
enum class SurfaceFlagBits : u32;

template <class T>
class RasterizerCache {
    /// Address shift for caching surfaces into a hash table
    static constexpr u64 CITRA_PAGEBITS = 18;

    using Runtime = typename T::Runtime;
    using Sampler = typename T::Sampler;
    using Surface = typename T::Surface;
    using Framebuffer = typename T::Framebuffer;
    using DebugScope = typename T::DebugScope;

    using SurfaceMap =
        boost::icl::interval_map<PAddr, SurfaceId, boost::icl::partial_absorber, std::less,
                                 boost::icl::inplace_plus, boost::icl::inter_section,
                                 SurfaceInterval, NodePoolAllocator>;

    using SurfaceRect_Tuple = std::pair<SurfaceId, Common::Rectangle<u32>>;
    using PageMap = boost::icl::interval_map<u32, int>;

public:
    explicit RasterizerCache(Memory::MemorySystem& memory, CustomTexManager& custom_tex_manager,
                             Runtime& runtime, Pica::RegsInternal& regs, RendererBase& renderer);
    ~RasterizerCache();

    /// Notify the cache that a new frame has been queued
    void TickFrame();

    /// Perform hardware accelerated texture copy according to the provided configuration
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config);

    /// Perform hardware accelerated display transfer according to the provided configuration
    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config);

    /// Perform hardware accelerated memory fill according to the provided configuration
    bool AccelerateFill(const Pica::MemoryFillConfig& config);

    /// Returns a reference to the surface object assigned to surface_id
    Surface& GetSurface(SurfaceId surface_id);

    /// Returns a reference to the sampler object matching the provided configuration
    /// With `nearest` the sampler does not filter, whatever the texture and the settings ask
    /// for: for textures whose texels are not colours to blend
    Sampler& GetSampler(const Pica::TexturingRegs::TextureConfig& config, bool nearest = false);
    Sampler& GetSampler(SamplerId sampler_id);

    /// Copy one surface's region to another
    void CopySurface(Surface& src_surface, Surface& dst_surface, SurfaceInterval copy_interval);

    /// Load a texture from 3DS memory to OpenGL and cache it (if not already cached)
    SurfaceId GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                         bool load_if_create, const SurfaceFlagBits& create_initial_flags = {});

    /// Attempt to find a subrect (resolution scaled) of a surface, otherwise loads a texture from
    /// 3DS memory to OpenGL and caches it (if not already cached)
    SurfaceRect_Tuple GetSurfaceSubRect(const SurfaceParams& params, ScaleMatch match_res_scale,
                                        bool load_if_create,
                                        const SurfaceFlagBits& create_initial_flags = {});

    /// Get a surface based on the texture configuration
    Surface& GetTextureSurface(const Pica::TexturingRegs::FullTextureConfig& config);
    SurfaceId GetTextureSurface(const Pica::Texture::TextureInfo& info, u32 max_level = 0);

    /// Get a texture cube based on the texture configuration
    Surface& GetTextureCube(const TextureCubeConfig& config);

    /// Get the color and depth surfaces based on the framebuffer configuration. With
    /// depth_overwritten the draw this is for replaces depth and stencil in every pixel of its
    /// viewport, so the depth surface is made ready without what it held there being brought
    /// (see ValidateSurface).
    FramebufferHelper<T> GetFramebufferSurfaces(bool using_color_fb, bool using_depth_fb,
                                                bool depth_overwritten = false);

    /// These parameters with the images the surfaces they name are in now
    FramebufferParams WithCurrentImages(FramebufferParams params) const;

    /// The framebuffer for the same surfaces as these parameters name, with the images the
    /// surfaces are in now
    Framebuffer* RefreshFramebuffer(FramebufferParams params);

    /// Get a surface that matches a "texture copy" display transfer config
    SurfaceRect_Tuple GetTexCopySurface(const SurfaceParams& params);

    /// Write any cached resources overlapping the region back to memory (if dirty), for it to be
    /// read from there: by the game, or by what reads guest memory for a draw
    void FlushRegion(PAddr addr, u32 size);

    /// The same without anybody reading it: for the cache itself, which brings memory into a
    /// surface that way, or gets a surface out of the way
    void WriteBack(PAddr addr, u32 size, SurfaceId flush_surface = {});

    /// Mark region as being invalidated by region_owner (nullptr if 3DS memory)
    void InvalidateRegion(PAddr addr, u32 size, SurfaceId region_owner = {});

    /// Like InvalidateRegion, for rows of row_size bytes that start every row_stride bytes. The
    /// bytes between the rows keep their validity and owner.
    void InvalidateRows(PAddr addr, u32 row_size, u32 row_stride, u32 rows, SurfaceId region_owner);

    /// Flush all cached resources tracked by this cache manager
    void FlushAll();

    /// Clear all cached resources tracked by this cache manager
    void ClearAll(bool flush);

    /// Has `callback` called before the cache writes into the emulated memory itself, with the
    /// stretch it is about to write: what a surface holds, put back where the game can read
    /// it. For whoever lets the host GPU read emulated memory in place: a draw that is not
    /// through the GPU yet must not find it changed.
    void SetGuestWriteCallback(std::function<void(PAddr, u32)> callback) {
        on_guest_write = std::move(callback);
    }

    /// Whether the host GPU holds anything newer than the emulated memory in the stretch
    [[nodiscard]] bool HoldsNewer(PAddr addr, u32 size) const {
        return dirty_regions.find(SurfaceInterval{addr, addr + size}) != dirty_regions.end();
    }

    /// The game read the stretch, which came here because its pages are cached. Pages of it
    /// the host GPU holds nothing newer of are made readable from now on, see
    /// Memory::MemorySystem::RasterizerMarkRegionReadable: most of what a game reads on cached
    /// pages only shares the page with a texture (measured: all of 37,000 reads a frame in
    /// one game).
    void NoteCpuRead(PAddr addr, u32 size);

    /// The game is about to read the stretch, which may make the cache read a surface back
    /// from the host GPU and wait for it. A small render target that is read back frame
    /// after frame (an image of the screen shrunk to 8x8, to tell how bright it is) is given
    /// to the game late from then on, see ServeLateReadbacks. Before FlushRegion.
    void NoteReadbackForGame(PAddr addr, u32 size);

    /// Once a frame: copies the small targets the game reads every frame out of the host GPU
    /// without waiting, and hands over what the GPU has finished of earlier copies
    /// (Memory::MemorySystem::PostLateWrite). The game's reads of such a stretch do not wait
    /// (Memory::MemorySystem::ExemptReads): it reads what was there a frame or two ago.
    void ServeLateReadbacks();

    /// Whether a cached surface covers any of the stretch
    [[nodiscard]] bool TouchesSurface(PAddr addr, u32 size) {
        bool any = false;
        ForEachSurfaceInRegion(addr, size, [&any](SurfaceId, Surface&) { any = true; });
        return any;
    }

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    void ForEachPage(PAddr addr, std::size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> CITRA_PAGEBITS;
        for (u64 page = addr >> CITRA_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    /// Iterates over all the surfaces in a region calling func
    template <typename Func>
    void ForEachSurfaceInRegion(PAddr addr, std::size_t size, Func&& func);

    /// Get the best surface match (and its match type) for the given flags
    template <MatchFlags find_flags>
    SurfaceId FindMatch(const SurfaceParams& params, ScaleMatch match_scale_type,
                        std::optional<SurfaceInterval> validate_interval = std::nullopt);

    /// Unregisters sentenced surfaces that have surpassed the destruction threshold: all the
    /// GPU is done with if at_once, else those of them sentenced a few frames ago.
    void RunGarbageCollector(bool at_once = false);

    /// Puts a surface that is no longer in the cache on the list of those to be destroyed
    void Sentence(SurfaceId surface_id);

    /// Removes any framebuffers that reference the provided surface_id.
    void RemoveFramebuffers(SurfaceId surface_id);

    /// Removes any references of the provided surface id from cached texture cubes.
    void RemoveTextureCubeFace(SurfaceId surface_id);

    /// Computes the hash of the provided texture data.
    u64 ComputeHash(const SurfaceParams& load_info, std::span<u8> upload_data);

    /// Update surface's texture for given region when necessary. With allow_download unset the
    /// validation stops, returning false, as soon as it would have to read another surface back
    /// from the host GPU; whatever was validated up to that point stays valid. With
    /// needs_contents unset the caller overwrites all of the range before anything reads it:
    /// the range counts as valid from then on and nothing is brought to it.
    bool ValidateSurface(SurfaceId surface, PAddr addr, u32 size, bool allow_download = true,
                         bool needs_contents = true);

    /// Copies pixel data in interval from the guest VRAM to the host GPU surface
    void UploadSurface(Surface& surface, SurfaceInterval interval);

    /// Uploads a custom texture identified with hash to the target surface
    bool UploadCustomSurface(SurfaceId surface_id, SurfaceInterval interval);

    /// Copies pixel data in interval from the host GPU surface to the guest VRAM
    void DownloadSurface(Surface& surface, SurfaceInterval interval);

    /// Downloads a fill surface to guest VRAM
    void DownloadFillSurface(Surface& surface, SurfaceInterval interval);

    /// Attempt to find a reinterpretable surface in the cache and use it to copy for validation
    bool ValidateByReinterpretation(Surface& surface, SurfaceParams params,
                                    const SurfaceInterval& interval);

    /// Attempt to fill the interval from the surfaces that hold its newest contents with another
    /// layout, by converting on the host GPU
    bool ValidateByRelayout(SurfaceId surface_id, const SurfaceInterval& interval);

    /// Create a new surface
    SurfaceId CreateSurface(const SurfaceParams& params, const SurfaceFlagBits& initial_flags = {});

    /// Register surface into the cache
    void RegisterSurface(SurfaceId surface);

    /// Remove surface from the cache
    void UnregisterSurface(SurfaceId surface);

    /// Unregisters all surfaces from the cache
    void UnregisterAll();

    /// Increase/decrease the number of surface in pages touching the specified region
    void UpdatePagesCachedCount(PAddr addr, u32 size, int delta);

    /// Takes back that the pages of the stretch are readable (NoteCpuRead): the host GPU is
    /// about to hold something newer there, or they stop being cached. With `tell_memory`
    /// unset only the note here is dropped, for when the memory system drops its own.
    void DropReadable(PAddr addr, u32 size, bool tell_memory = true);

private:
    Memory::MemorySystem& memory;
    CustomTexManager& custom_tex_manager;
    Runtime& runtime;
    Pica::RegsInternal& regs;
    RendererBase& renderer;
    std::unordered_map<TextureCubeConfig, TextureCube> texture_cube_cache;
    tsl::robin_pg_map<u64, std::vector<SurfaceId>, Common::IdentityHash<u64>> page_table;
    std::unordered_map<FramebufferParams, FramebufferId> framebuffers;
    std::unordered_map<SamplerParams, SamplerId> samplers;
    /// The samplers asked for last, with what they were asked for: a draw asks for one for
    /// each of its textures, nearly always one of the last few, and finding it in the map
    /// above means hashing what it was asked with every time (2.5% of the decoding thread in
    /// a game of 426 draws a frame). Samplers are never taken out of the map.
    std::array<std::pair<SamplerParams, SamplerId>, 4> recent_samplers{};
    u32 recent_sampler_next{};
    /// A surface taken out of the cache. It stays until the GPU is done with what used it and
    /// a few frames longer (RunGarbageCollector), and in that time a surface asked for with
    /// the same parameters takes it over, images and all (CreateSurface). A game that draws
    /// both screens into the same memory has its targets taken out and asked for again every
    /// frame: Shinobi 3D made 200 new images a second that way.
    struct Sentenced {
        SurfaceId surface_id;
        u64 last_used_tick;
        u64 frame;
    };
    std::list<Sentenced> sentenced;
    Common::SlotVector<Surface> slot_surfaces;
    Common::SlotVector<Sampler> slot_samplers;
    Common::SlotVector<Framebuffer> slot_framebuffers;
    SurfaceMap dirty_regions;
    PageMap cached_pages;
    std::function<void(PAddr, u32)> on_guest_write;

    /// What GetFramebufferSurfaces was asked for and found the last time. A game draws many
    /// times into one target, and finding the target was 4% of the emulation thread in one
    /// game; the next draw that asks for the same takes this, if no surface came or went and
    /// nothing changed of what is valid of any (SurfaceBase::validity_epoch).
    struct LastTarget {
        bool valid{};
        // Asked for
        PAddr color_addr{};
        PAddr depth_addr{};
        u32 color_format{};
        u32 depth_format{};
        u32 width{};
        u32 height{};
        Common::Rectangle<s32> viewport{};
        bool using_color{};
        bool using_depth{};
        bool shadow_rendering{};
        Common::ResolutionScale res_scale;
        u32 sample_count{};
        u64 validity_epoch{};
        // Found
        SurfaceId color_id{};
        SurfaceId depth_id{};
        u32 color_level{};
        u32 depth_level{};
        Common::Rectangle<u32> fb_rect{};
    };
    LastTarget last_target;

    /// Counts the surfaces that come and go, what else decides which surface a lookup finds,
    /// and the frames. A lookup of before holds while this and SurfaceBase::validity_epoch
    /// stand still.
    u64 surface_set_epoch{};

    /// The textures looked up this frame, by where they are: most draws of a frame use
    /// textures a draw before them used, and the lookup was 2% to 3% of the emulation thread
    struct TextureLookup {
        PAddr address{};
        u32 width{};
        u32 height{};
        u32 format{};
        u32 max_level{};
        bool shadow_source{};
        u64 set_epoch{};
        u64 validity_epoch{};
        SurfaceId surface_id{};
    };
    std::array<TextureLookup, 64> texture_lookups{};
    /// One bit a page of the emulated memory, by its physical address: readable, see NoteCpuRead
    std::vector<u64> readable_pages;
    u32 readable_count{};
    /// A small render target and how the game reads it back, see NoteReadbackForGame
    struct LateReadback {
        PAddr addr{};
        u32 size{};
        /// The frame it was last read back in for the game, and frames in a row it was
        u64 last_read_frame{};
        u32 streak{};
        /// Given to the game late from here on
        bool late{};
        /// A copy is on its way: the tick it was recorded under, where it lands, what of
        bool pending{};
        u64 pending_tick{};
        StagingData staging{};
        SurfaceParams info{};
        bool convert{};
    };
    std::vector<LateReadback> late_readbacks;
    u64 late_frame{};
    /// For the log, see CreateSurface
    u32 taken_over{};
    u32 taken_over_early{};
    u32 made_new{};
    u32 made_new_waiting{};
    Common::ResolutionScale resolution_scale_factor;
    u8 sample_count;
    FramebufferParams fb_params;

    /// A render target that the guest has shrunk to half size with a display transfer
    struct DownscaledTarget {
        PAddr addr;
        u32 width;
        u32 height;
        PixelFormat pixel_format;
        // The scale the target is rendered at, until its uses say otherwise
        bool half_x;
        bool half_y;
        // What has been done with the target since it was last drawn to
        bool shrunk_x;
        bool shrunk_xy;
        bool used_at_full_size;
    };
    DownscaledTarget* FindDownscaledTarget(const SurfaceParams& params);
    /// Remembers how the source of a display transfer was scaled
    void NoteDisplayTransferSource(const SurfaceParams& src_params, bool half_x, bool half_y);
    /// Remembers that the contents of a surface were used in a way that shows them at full size
    void NoteFullSizeUse(const SurfaceParams& params);
    /// Gives a render target, and the depth buffer used with it, the reduced scale it can have
    void ApplyDownscaledTargetScale(SurfaceParams& color_params, SurfaceParams& depth_params);

    std::vector<DownscaledTarget> downscaled_targets;

    /// A colour buffer that a shadow map was rendered into
    struct ShadowMapTarget {
        PAddr addr;
        /// Drawn to together with a depth buffer, which is at the resolution scale
        bool with_depth;
    };
    std::vector<ShadowMapTarget> shadow_map_targets;
    /// Whether the surface is a shadow map that is kept at the size the game gave it
    bool IsNativeScaleShadowMap(const SurfaceParams& surface) const;
    Settings::TextureFilter filter;
    bool dump_textures;
    bool use_custom_textures;
};

} // namespace VideoCore
