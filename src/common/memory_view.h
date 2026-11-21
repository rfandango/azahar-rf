// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <cstddef>
#include "common/common_types.h"

namespace Common {

/**
 * Memory that can be shown at more than one place of the host's address space: shared memory,
 * of which any part can be mapped again anywhere, see AddressSpaceView. It starts and ends on a
 * page of the host.
 *
 * It has no name and no file descriptor. That is on purpose: a host GPU driver that is handed
 * this memory by its address looks at what it is a mapping of, and the Adreno kernel driver
 * takes any mapping whose file has a descriptor open for one of its own kind of buffers and
 * refuses it when it is not.
 *
 * Only where the host offers this in pages of 4 KiB, the size of the emulated system's pages
 * (Linux and Android for now). Elsewhere IsValid() is false and nothing else may be used.
 */
class SharedBacking {
public:
    /// `size` is a multiple of 4 KiB. The memory starts out as zeroes.
    explicit SharedBacking(std::size_t size);
    ~SharedBacking();

    SharedBacking(const SharedBacking&) = delete;
    SharedBacking& operator=(const SharedBacking&) = delete;

    [[nodiscard]] bool IsValid() const {
        return base != nullptr;
    }

    [[nodiscard]] u8* Data() const {
        return base;
    }

    [[nodiscard]] std::size_t Size() const {
        return size;
    }

    /// Whether the pointer lies in this memory
    [[nodiscard]] bool Contains(const u8* pointer) const {
        return base != nullptr && pointer >= base && pointer < base + size;
    }

private:
    friend class AddressSpaceView;

    u8* base = nullptr;
    std::size_t size = 0;
};

/**
 * A stretch of the host's address space that is reserved and, to begin with, empty: an access
 * anywhere in it faults. Parts of a SharedBacking are shown in it page by page, which lays an
 * emulated address space out in the host's own, so that the host's memory management unit does
 * the translation of an emulated address instead of code.
 */
class AddressSpaceView {
public:
    /// `size` is a multiple of 4 KiB. The backing has to outlive the view.
    AddressSpaceView(const SharedBacking& backing, std::size_t size);
    ~AddressSpaceView();

    AddressSpaceView(const AddressSpaceView&) = delete;
    AddressSpaceView& operator=(const AddressSpaceView&) = delete;

    [[nodiscard]] bool IsValid() const {
        return base != nullptr;
    }

    [[nodiscard]] u8* Base() const {
        return base;
    }

    /// Shows `length` bytes of the backing, from `backing_offset` on, at `offset` of the view,
    /// readable and writable, or readable only: a write there then faults like any access to
    /// an empty part. All three are multiples of 4 KiB. Returns false if the host refused,
    /// after which that part of the view is empty.
    bool Map(std::size_t offset, std::size_t backing_offset, std::size_t length,
             bool writable = true);

    /// Leaves `length` bytes at `offset` of the view empty again.
    void Unmap(std::size_t offset, std::size_t length);

private:
    const SharedBacking& backing;
    u8* base = nullptr;
    std::size_t size = 0;
};

} // namespace Common
