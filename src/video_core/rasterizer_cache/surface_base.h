// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <boost/icl/interval_set.hpp>
#include "video_core/rasterizer_cache/node_pool.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/utils.h"

namespace VideoCore {

using SurfaceRegions =
    boost::icl::interval_set<PAddr, std::less, SurfaceInterval, NodePoolAllocator>;

struct Material;

enum class SurfaceFlagBits : u32 {
    Registered = 1 << 0,   ///< Surface is registed in the rasterizer cache.
    Picked = 1 << 1,       ///< Surface has been picked when searching for a match.
    Tracked = 1 << 2,      ///< Surface is part of a texture cube and should be tracked.
    Custom = 1 << 3,       ///< Surface texture has been replaced with a custom texture.
    ShadowSource = 1 << 4, ///< Surface is used as a shadow source.
    RenderTarget = 1 << 5, ///< Surface was a render target.
};
DECLARE_ENUM_FLAG_OPERATORS(SurfaceFlagBits);

class SurfaceBase : public SurfaceParams {
public:
    SurfaceBase(const SurfaceParams& params, const SurfaceFlagBits& initial_flag_bits);
    ~SurfaceBase();

    /// Returns true when this surface can be used to fill the fill_interval of dest_surface
    bool CanFill(const SurfaceParams& dest_surface, SurfaceInterval fill_interval) const;

    /// Returns true when surface can validate copy_interval of dest_surface
    bool CanCopy(const SurfaceParams& dest_surface, SurfaceInterval copy_interval) const;

    /// Returns the region of the biggest valid rectange within interval
    SurfaceInterval GetCopyableInterval(const SurfaceParams& params) const;

    /// Returns the clear value used to validate another surface from this fill surface
    ClearValue MakeClearValue(PAddr copy_addr, PixelFormat dst_format);

    /// Returns the internal surface extent.
    Extent RealExtent(bool scaled = true) const;

    /// Returns true if the surface contains a custom material with a normal map.
    bool HasNormalMap() const noexcept;

    bool Overlaps(PAddr overlap_addr, std::size_t overlap_size) const noexcept {
        const PAddr overlap_end = overlap_addr + static_cast<PAddr>(overlap_size);
        return addr < overlap_end && overlap_addr < end;
    }

    u64 ModificationTick() const noexcept {
        return modification_tick;
    }

    bool IsCustom() const noexcept {
        return True(flags & SurfaceFlagBits::Custom) && custom_format != CustomPixelFormat::Invalid;
    }

    bool IsRegionValid(SurfaceInterval interval) const {
        return invalid_regions.find(interval) == invalid_regions.end();
    }

    void MarkValid(SurfaceInterval interval) {
        invalid_regions.erase(interval);
        given_regions.insert(interval);
        modification_tick++;
        validity_epoch++;
    }

    void MarkInvalid(SurfaceInterval interval) {
        invalid_regions.insert(interval);
        modification_tick++;
        validity_epoch++;
    }

    void MarkValid(const SurfaceRegions& regions) {
        invalid_regions -= regions;
        given_regions += regions;
        modification_tick++;
        validity_epoch++;
    }

    void MarkInvalid(const SurfaceRegions& regions) {
        invalid_regions += regions;
        modification_tick++;
        validity_epoch++;
    }

    /// Counts every change of what is valid of any surface. What was found out about surfaces
    /// holds for as long as it stands still (and no surface comes or goes). Whoever changes
    /// invalid_regions by hand counts it up as well.
    static inline u64 validity_epoch = 0;

    bool IsFullyInvalid() const {
        auto interval = GetInterval();
        return *invalid_regions.equal_range(interval).first == interval;
    }

private:
    /// Returns the fill buffer value starting from copy_addr
    std::array<u8, 4> MakeFillBuffer(PAddr copy_addr);

public:
    SurfaceFlagBits flags{};
    /// Renderers that keep a second image of a surface, to draw into while the first is
    /// sampled, flip this when the two change places: what was made for one image of the
    /// surface, a framebuffer for one, is not for the other.
    bool swapped{};
    /// Counts the times the surface was given other images than the ones it was in: what was
    /// made for the old ones, which are destroyed as soon as the GPU is done with them, is not
    /// for these. A framebuffer for one, and that goes for either of its surfaces.
    u32 image_generation{};
    /// The hitch summary in which a draw last sampled the surface (Common::Hitches::
    /// NoteTextureSampled), to count each texture once a summary
    u32 sampled_in_summary = ~0u;
    /// An ETC1 texture whose image holds the guest's blocks, for the GPU to decode when it
    /// samples them, and not the pixels they decode to: an eighth of the memory, and of what is
    /// read for every sample. Its rows are stored from the top, the other way up to those of
    /// every other surface (a block cannot be turned over exactly: the second colour of one in
    /// differential mode is not always the first's plus a difference that fits), so its uploads
    /// and copies count rows from the other end and the fragment shader turns the coordinate
    /// over. The uploads are of the blocks (TranscodeETC1). Set by the backend that can.
    bool keeps_etc_blocks{};
    const Material* material = nullptr;
    SurfaceRegions invalid_regions;
    /// What of the surface was valid at some time: the part its image on the host has been
    /// given contents for, by a load, a copy, a clear or a draw. The rest of the image is as
    /// it was made, with nothing ever written to it, and must not be handed to the GPU to
    /// read or to draw into like that (see RasterizerCache::ValidateSurface). Whoever takes
    /// from invalid_regions by hand adds here as well.
    SurfaceRegions given_regions;
    u32 fill_size = 0;
    std::array<u8, 4> fill_data{};
    u64 modification_tick = 1;
};

} // namespace VideoCore
