// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <cstring>
#include <vector>
#include "common/alignment.h"
#include "common/swap.h"
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

void TranscodeETC1(const SurfaceParams& surface_info, std::span<const u8> source,
                   std::span<u8> dest) {
    // The guest's tiles of 8x8 pixels come row by row from the top of the picture, and each is
    // four blocks of 4x4: top left, top right, bottom left, bottom right. A block read as a
    // little-endian number is the ETC1 block, whose bytes are in the other order, the most
    // significant first. The blocks go out row by row from the top as well.
    constexpr u32 TILE_SIZE = 32;
    const u32 tiles_x = surface_info.width / 8;
    const u32 blocks_x = surface_info.width / 4;
    if (tiles_x == 0) {
        return;
    }
    const u32 tiles = std::min<u32>(static_cast<u32>(source.size() / TILE_SIZE),
                                    tiles_x * (surface_info.height / 8));
    ASSERT(dest.size() >= std::size_t{tiles} * TILE_SIZE);
    for (u32 tile = 0; tile < tiles; tile++) {
        const u32 tile_x = tile % tiles_x;
        const u32 tile_y = tile / tiles_x;
        for (u32 i = 0; i < 4; i++) {
            u64 block;
            std::memcpy(&block, source.data() + tile * TILE_SIZE + i * 8, sizeof(block));
            block = Common::swap64(block);
            const u32 block_x = 2 * tile_x + (i & 1);
            const u32 block_y = 2 * tile_y + (i >> 1);
            std::memcpy(dest.data() + (std::size_t{block_y} * blocks_x + block_x) * 8, &block,
                        sizeof(block));
        }
    }
}

} // namespace VideoCore
