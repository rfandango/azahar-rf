// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <cstddef>
#include <cstdint>

namespace Common {

/**
 * Writes what the processor's data cache holds of `size` bytes at `start` out to memory.
 *
 * For memory a device reads without seeing through the processor's cache: a GPU that was handed
 * the emulator's own pages on a system on a chip whose memory bus does not keep the two
 * coherent (an Adreno 610's does not). Whatever the processor wrote there last has to be in
 * memory itself before the device looks.
 *
 * Returns false where it is not known how to do that, and nothing was done.
 */
inline bool WriteOutDataCache([[maybe_unused]] const void* start,
                              [[maybe_unused]] std::size_t size) {
#if defined(__aarch64__) && defined(__linux__)
    // The cache type register says how long a line of the data cache is, in words, as a
    // power of two. Both it and the cleaning of a line by its address are open to programs
    // on Linux.
    std::uint64_t cache_type;
    __asm__ volatile("mrs %0, ctr_el0" : "=r"(cache_type));
    const std::uintptr_t line = std::uintptr_t{4} << ((cache_type >> 16) & 0xF);
    const std::uintptr_t end = reinterpret_cast<std::uintptr_t>(start) + size;
    for (std::uintptr_t address = reinterpret_cast<std::uintptr_t>(start) & ~(line - 1);
         address < end; address += line) {
        __asm__ volatile("dc cvac, %0" : : "r"(address) : "memory");
    }
    __asm__ volatile("dsb sy" : : : "memory");
    return true;
#else
    return false;
#endif
}

} // namespace Common
