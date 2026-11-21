// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <cstddef>
#include <new>

namespace VideoCore {

/**
 * An allocator for containers that take and give back one node at a time, all day: the
 * interval sets and maps the rasterizer cache keeps its books in. A node that is given back is
 * kept in a list of its size, and the next one asked for comes from there; the heap is only
 * asked when the list is empty. Nothing is ever returned to the heap, the lists hold as many
 * nodes as were in use at the most.
 *
 * With the general allocator every row of a texture copy that changes owner was an allocation
 * and a release (measured in Luigi's Mansion: 3.6% of the emulation thread in those two alone).
 *
 * The lists are one a thread and a node size, so nothing is locked. A node released by another
 * thread than the one that took it goes to that thread's list.
 */
template <class T>
class NodePoolAllocator {
public:
    using value_type = T;

    NodePoolAllocator() noexcept = default;
    template <class U>
    NodePoolAllocator(const NodePoolAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count) {
        if (count == 1 && Poolable()) {
            Node*& free_list = FreeList();
            if (Node* const node = free_list) {
                free_list = node->next;
                return reinterpret_cast<T*>(node);
            }
        }
        return static_cast<T*>(::operator new(count * sizeof(T)));
    }

    void deallocate(T* pointer, std::size_t count) noexcept {
        if (count == 1 && Poolable()) {
            Node* const node = reinterpret_cast<Node*>(pointer);
            Node*& free_list = FreeList();
            node->next = free_list;
            free_list = node;
            return;
        }
        ::operator delete(pointer);
    }

    template <class U>
    bool operator==(const NodePoolAllocator<U>&) const noexcept {
        return true;
    }
    template <class U>
    bool operator!=(const NodePoolAllocator<U>&) const noexcept {
        return false;
    }

private:
    struct Node {
        Node* next;
    };

    /// A released node holds the link to the next one, and comes from operator new, which
    /// aligns for the fundamental types and no more
    static constexpr bool Poolable() {
        return sizeof(T) >= sizeof(Node) && alignof(T) <= alignof(std::max_align_t);
    }

    /// The released nodes of this size, of this thread
    static Node*& FreeList() {
        static thread_local Node* free_list = nullptr;
        return free_list;
    }
};

} // namespace VideoCore
