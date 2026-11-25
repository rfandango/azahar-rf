// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <cstring>
#include <vector>
#include "common/alignment.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/texture_codec.h"
#include "video_core/rasterizer_cache/utils.h"

namespace VideoCore {

u32 MipLevels(u32 width, u32 height, u32 max_level) {
    u32 levels = 1;
    while (width > 8 && height > 8) {
        levels++;
        width >>= 1;
        height >>= 1;
    }

    return std::min(levels, max_level + 1);
}

void EncodeTexture(const SurfaceParams& surface_info, PAddr start_addr, PAddr end_addr,
                   std::span<u8> source, std::span<u8> dest, bool convert) {
    const PixelFormat format = surface_info.pixel_format;
    const u32 func_index = static_cast<u32>(format);

    if (surface_info.is_tiled) {
        const MortonFunc SwizzleImpl =
            (convert ? SWIZZLE_TABLE_CONVERTED : SWIZZLE_TABLE)[func_index];
        if (SwizzleImpl) {
            SwizzleImpl(surface_info.width, surface_info.height, start_addr - surface_info.addr,
                        end_addr - surface_info.addr, source, dest);
            return;
        }
    } else {
        const LinearFunc LinearEncodeImpl =
            (convert ? LINEAR_ENCODE_TABLE_CONVERTED : LINEAR_ENCODE_TABLE)[func_index];
        if (LinearEncodeImpl) {
            // The image handed over begins with the row that start_addr lies in, not with
            // start_addr itself. (Luigi's Mansion 2 puts its picture together out of columns, and
            // after a save state was loaded one that began in the middle of a row was an image of
            // its own: what was read back from it came to lie that far beside its place in every
            // row, and the picture was shown wrapped round.)
            const u32 bytes_per_pixel = surface_info.GetFormatBpp() / 8;
            const u32 host_bytes_per_pixel = convert ? 4 : bytes_per_pixel;
            const u32 offset = start_addr - surface_info.addr;
            const u32 first_pixel = offset / bytes_per_pixel;
            const u32 lead = offset % bytes_per_pixel;
            const std::size_t source_offset = std::size_t{first_pixel} * host_bytes_per_pixel;
            if (source_offset >= source.size()) {
                return;
            }
            source = source.subspan(source_offset);
            if (lead == 0 && dest.size() % bytes_per_pixel == 0) {
                LinearEncodeImpl(source, dest);
                return;
            }
            // Begins or ends inside a pixel
            std::vector<u8> pixels(Common::AlignUp(lead + dest.size(), bytes_per_pixel));
            LinearEncodeImpl(source, pixels);
            std::memcpy(dest.data(), pixels.data() + lead, dest.size());
            return;
        }
    }

    LOG_ERROR(HW_GPU, "Unimplemented texture encode function for pixel format = {}, tiled = {}",
              func_index, surface_info.is_tiled);
    UNIMPLEMENTED();
}

void DecodeTexture(const SurfaceParams& surface_info, PAddr start_addr, PAddr end_addr,
                   std::span<u8> source, std::span<u8> dest, bool convert) {
    const PixelFormat format = surface_info.pixel_format;
    const u32 func_index = static_cast<u32>(format);

    if (surface_info.is_tiled) {
        const MortonFunc UnswizzleImpl =
            (convert ? UNSWIZZLE_TABLE_CONVERTED : UNSWIZZLE_TABLE)[func_index];
        if (UnswizzleImpl) {
            UnswizzleImpl(surface_info.width, surface_info.height, start_addr - surface_info.addr,
                          end_addr - surface_info.addr, dest, source);
            return;
        }
    } else {
        const LinearFunc LinearDecodeImpl =
            (convert ? LINEAR_DECODE_TABLE_CONVERTED : LINEAR_DECODE_TABLE)[func_index];
        if (LinearDecodeImpl) {
            LinearDecodeImpl(source, dest);
            return;
        }
    }

    LOG_ERROR(HW_GPU, "Unimplemented texture decode function for pixel format = {}, tiled = {}",
              func_index, surface_info.is_tiled);
    UNIMPLEMENTED();
}

} // namespace VideoCore
