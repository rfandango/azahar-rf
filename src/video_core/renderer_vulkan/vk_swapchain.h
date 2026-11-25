// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <chrono>
#include <mutex>
#include <optional>
#include <span>
#include <vector>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

class Swapchain {
public:
    explicit Swapchain(const Instance& instance, u32 width, u32 height, vk::SurfaceKHR surface,
                       bool low_refresh_rate);
    ~Swapchain();

    /// Creates (or recreates) the swapchain with a given size.
    void Create(u32 width, u32 height, vk::SurfaceKHR surface, bool low_refresh_rate);

    /// Acquires the next image in the swapchain.
    bool AcquireNextImage();

    /// Acquires an image, which is free to be drawn into once the semaphore is signalled, and says
    /// which. Nothing if none came free in that time, or if the swapchain has to be made anew
    /// (NeedsRecreation).
    std::optional<u32> Acquire(vk::Semaphore semaphore, std::chrono::nanoseconds timeout);

    /// Presents that image once `ready` is signalled, otherwise as the function below
    void Present(u32 image, vk::Semaphore ready, u32 present_id, u64 desired_time);

    /// Whether it was made for another output resolution than the settings ask for by now
    bool IsOutputOutdated() const;

    /// Whether the swapchain cannot be used any more and has to be made anew
    bool NeedsRecreation() const {
        return needs_recreation;
    }

    /// Presents the current image and move to the next one. A present_id other than zero asks
    /// the display, where it can tell, when the image reached it: see TakePresentTimings. A
    /// desired_time other than zero is the time the image should be displayed at (steady clock,
    /// nanoseconds), instead of as soon as possible.
    void Present(u32 present_id = 0, u64 desired_time = 0);

    /// Returns when the display showed images presented with an id, for those it has not
    /// reported yet. Valid until the next call.
    std::span<const vk::PastPresentationTimingGOOGLE> TakePresentTimings();

    /// Whether every image is displayed, in the order presented
    bool IsFifo() const {
        return present_mode == vk::PresentModeKHR::eFifo;
    }

    /// Refresh period of the display, zero if unknown
    std::chrono::nanoseconds GetRefreshPeriod() const {
        return refresh_period;
    }

    /// Name of the present mode in use
    const char* GetPresentModeName() const {
        switch (present_mode) {
        case vk::PresentModeKHR::eImmediate:
            return "immediate";
        case vk::PresentModeKHR::eMailbox:
            return "mailbox";
        case vk::PresentModeKHR::eFifo:
            return "fifo";
        case vk::PresentModeKHR::eFifoRelaxed:
            return "fifo relaxed";
        default:
            return "other";
        }
    }

    vk::SurfaceKHR GetSurface() const {
        return surface;
    }

    vk::Image Image() const {
        return images[image_index];
    }

    std::span<const vk::Image> Images() const {
        return images;
    }

    vk::SurfaceFormatKHR GetSurfaceFormat() const {
        return surface_format;
    }

    vk::SwapchainKHR GetHandle() const {
        return swapchain;
    }

    u32 GetWidth() const {
        return width;
    }

    u32 GetHeight() const {
        return height;
    }

    u32 GetImageCount() const {
        return image_count;
    }

    vk::Extent2D GetExtent() const {
        return extent;
    }

    /// By how many degrees what is drawn into the images has to be turned (clockwise as seen
    /// on the image, 0, 90, 180 or 270) for it to be upright on the display: the images are
    /// in the orientation of the display's panel, so that the compositor need not turn them.
    u32 GetPreRotation() const {
        return pre_rotation;
    }

    [[nodiscard]] vk::Semaphore GetImageAcquiredSemaphore() const {
        return image_acquired[frame_index];
    }

    [[nodiscard]] vk::Semaphore GetPresentReadySemaphore() const {
        return present_ready[image_index];
    }

private:
    /// Selects the best available swapchain image format
    void FindPresentFormat();

    /// Sets the best available present mode
    void SetPresentMode();

    /// Sets the surface properties according to device capabilities
    void SetSurfaceProperties();

    /// Destroys current swapchain resources
    void Destroy();

    /// Performs creation of image views and framebuffers from the swapchain images
    void SetupImages();

    /// Creates the image acquired and present ready semaphores
    void RefreshSemaphores();

private:
    const Instance& instance;
    vk::SwapchainKHR swapchain{VK_NULL_HANDLE};
    vk::SurfaceKHR surface{};
    vk::SurfaceFormatKHR surface_format;
    vk::PresentModeKHR present_mode;
    std::chrono::nanoseconds refresh_period{};
    std::array<vk::PastPresentationTimingGOOGLE, 16> past_timings;
    vk::Extent2D extent;
    vk::SurfaceTransformFlagBitsKHR transform;
    u32 pre_rotation{};
    u32 output_percent = 100;
    vk::CompositeAlphaFlagBitsKHR composite_alpha;
    std::vector<vk::Image> images;
    std::vector<vk::Semaphore> image_acquired;
    std::vector<vk::Semaphore> present_ready;
    u32 width = 0;
    u32 height = 0;
    u32 image_count = 0;
    u32 image_index = 0;
    u32 frame_index = 0;
    bool needs_recreation = true;
    bool low_refresh_rate;
};

} // namespace Vulkan
