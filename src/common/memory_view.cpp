// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include "common/memory_view.h"

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/mman.h>
#include <unistd.h>
#define CITRA_HAS_MEMORY_VIEWS 1
#endif

namespace Common {

#ifdef CITRA_HAS_MEMORY_VIEWS

namespace {

constexpr long EMULATED_PAGE_SIZE = 4096;

} // Anonymous namespace

SharedBacking::SharedBacking(std::size_t size_) {
    // The emulated system's pages are mapped one by one, which a host with larger pages
    // cannot do
    if (sysconf(_SC_PAGESIZE) != EMULATED_PAGE_SIZE || size_ == 0 ||
        size_ % EMULATED_PAGE_SIZE != 0) {
        return;
    }
    // Shared, so that it can be mapped a second time, and of nothing that has a descriptor
    void* const memory =
        mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        return;
    }
    base = static_cast<u8*>(memory);
    size = size_;
}

SharedBacking::~SharedBacking() {
    if (base != nullptr) {
        munmap(base, size);
    }
}

AddressSpaceView::AddressSpaceView(const SharedBacking& backing_, std::size_t size_,
                                   std::size_t guard, std::size_t alignment)
    : backing{backing_} {
    if (!backing.IsValid() || size_ == 0 || size_ % EMULATED_PAGE_SIZE != 0 ||
        guard % EMULATED_PAGE_SIZE != 0 || alignment % EMULATED_PAGE_SIZE != 0) {
        return;
    }
    const std::size_t wanted = size_ + 2 * guard;
    const std::size_t asked_for = wanted + (alignment - EMULATED_PAGE_SIZE);
    void* const memory =
        mmap(nullptr, asked_for, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (memory == MAP_FAILED) {
        return;
    }
    // What is too much on either side goes back
    u8* const start = static_cast<u8*>(memory);
    const std::uintptr_t aligned =
        (reinterpret_cast<std::uintptr_t>(start) + guard + alignment - 1) & ~(alignment - 1);
    base = reinterpret_cast<u8*>(aligned);
    size = size_;
    access.resize(size / EMULATED_PAGE_SIZE);
    reserved = base - guard;
    reserved_size = wanted;
    if (reserved != start) {
        munmap(start, static_cast<std::size_t>(reserved - start));
    }
    if (reserved + wanted != start + asked_for) {
        munmap(reserved + wanted, static_cast<std::size_t>(start + asked_for - reserved - wanted));
    }
}

AddressSpaceView::~AddressSpaceView() {
    if (base != nullptr) {
        munmap(reserved, reserved_size);
    }
}

bool AddressSpaceView::Map(std::size_t offset, std::size_t backing_offset, std::size_t length,
                           bool writable) {
    if (base == nullptr || length == 0 || offset > size || length > size - offset ||
        backing_offset > backing.size || length > backing.size - backing_offset) {
        return false;
    }
    // Asked to move a mapping of no length, the host makes another mapping of the same shared
    // pages and leaves the first where it is. It replaces what was at the new place.
    void* const memory = mremap(backing.base + backing_offset, 0, length,
                                MREMAP_MAYMOVE | MREMAP_FIXED, base + offset);
    if (memory == MAP_FAILED) {
        // What was there may be gone in part: leave nothing
        Unmap(offset, length);
        return false;
    }
    // The new mapping is like the one it was made from, writable
    if (!writable && mprotect(base + offset, length, PROT_READ) != 0) {
        Unmap(offset, length);
        return false;
    }
    std::memset(access.data() + offset / EMULATED_PAGE_SIZE,
                writable ? AccessRead | AccessWrite : AccessRead, length / EMULATED_PAGE_SIZE);
    return true;
}

void AddressSpaceView::Unmap(std::size_t offset, std::size_t length) {
    if (base == nullptr || length == 0 || offset > size || length > size - offset) {
        return;
    }
    std::memset(access.data() + offset / EMULATED_PAGE_SIZE, 0, length / EMULATED_PAGE_SIZE);
    // A mapping of nothing in place of what was there: the stretch stays reserved
    mmap(base + offset, length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED,
         -1, 0);
}

#else // No memory views on this host

SharedBacking::SharedBacking(std::size_t) {}

SharedBacking::~SharedBacking() = default;

AddressSpaceView::AddressSpaceView(const SharedBacking& backing_, std::size_t, std::size_t,
                                   std::size_t)
    : backing{backing_} {}

AddressSpaceView::~AddressSpaceView() = default;

bool AddressSpaceView::Map(std::size_t, std::size_t, std::size_t, bool) {
    return false;
}

void AddressSpaceView::Unmap(std::size_t, std::size_t) {}

#endif

} // namespace Common
