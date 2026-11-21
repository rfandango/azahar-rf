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

AddressSpaceView::AddressSpaceView(const SharedBacking& backing_, std::size_t size_)
    : backing{backing_} {
    if (!backing.IsValid() || size_ == 0 || size_ % EMULATED_PAGE_SIZE != 0) {
        return;
    }
    void* const memory =
        mmap(nullptr, size_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (memory == MAP_FAILED) {
        return;
    }
    base = static_cast<u8*>(memory);
    size = size_;
}

AddressSpaceView::~AddressSpaceView() {
    if (base != nullptr) {
        munmap(base, size);
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
    return true;
}

void AddressSpaceView::Unmap(std::size_t offset, std::size_t length) {
    if (base == nullptr || length == 0 || offset > size || length > size - offset) {
        return;
    }
    // A mapping of nothing in place of what was there: the stretch stays reserved
    mmap(base + offset, length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED,
         -1, 0);
}

#else // No memory views on this host

SharedBacking::SharedBacking(std::size_t) {}

SharedBacking::~SharedBacking() = default;

AddressSpaceView::AddressSpaceView(const SharedBacking& backing_, std::size_t)
    : backing{backing_} {}

AddressSpaceView::~AddressSpaceView() = default;

bool AddressSpaceView::Map(std::size_t, std::size_t, std::size_t, bool) {
    return false;
}

void AddressSpaceView::Unmap(std::size_t, std::size_t) {}

#endif

} // namespace Common
