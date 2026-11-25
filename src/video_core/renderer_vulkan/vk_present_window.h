// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include "common/polyfill_thread.h"
#include "video_core/frame_pacer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

VK_DEFINE_HANDLE(VmaAllocation)

namespace Frontend {
class EmuWindow;
}

namespace Vulkan {

class Instance;
class Swapchain;
class Scheduler;
class RenderManager;

/// What a frame is drawn into: an image apart that is then copied into one of the swapchain, or that
/// one itself (PresentWindow::DrawsIntoSwapchain)
struct Frame {
    u32 width;
    u32 height;
    VmaAllocation allocation;
    vk::Framebuffer framebuffer;
    vk::Image image;
    vk::ImageView image_view;
    vk::Semaphore render_ready;
    vk::Fence present_done;
    vk::CommandBuffer cmdbuf;
    /// Game frame the image belongs to and when the emulation thread handed it over
    u32 game_frame{};
    std::chrono::steady_clock::time_point submitted{};
    /// Identifies the image in what the display reports, zero for none
    u32 present_id{};
    /// When the image should be displayed (steady clock, nanoseconds), zero for at once
    u64 desired_time{};
    /// The render passes to draw it with: one that clears all of it first and one that does not
    vk::RenderPass renderpass{};
    vk::RenderPass renderpass_uncleared{};

    // Of an image of the swapchain:
    u32 image_index{};
    /// Signalled when the display is done with it, to be waited for by what draws into it. Null for
    /// an image apart.
    vk::Semaphore image_acquired{};
    u32 acquire_slot{};
    /// The submission that does that
    u64 tick{};
    /// By how many degrees what is drawn into it is turned
    u32 rotation{};
};

class PresentWindow final {
public:
    explicit PresentWindow(Frontend::EmuWindow& emu_window, const Instance& instance,
                           Scheduler& scheduler, bool low_refresh_rate);
    ~PresentWindow();

    /// Waits for all queued frames to finish presenting.
    void WaitPresent();

    /// Returns the last used render frame. Null where there is nothing to draw into, because the
    /// window's surface is gone and no new one has come yet.
    Frame* GetRenderFrame();

    /// Whether frames are images of the swapchain. They are as large as those are and turned
    /// already, and the one who draws has to wait for Frame::image_acquired on the GPU.
    bool DrawsIntoSwapchain() const noexcept {
        return draws_into_swapchain;
    }

    /// The time the GPU took to copy frames into images of the swapchain, since this was last called
    GpuSpanMeter::Sum TakeCopyTime() {
        return copy_meter.Take()[0];
    }

    /// Recreates the render frame to match provided parameters.
    void RecreateFrame(Frame* frame, u32 width, u32 height);

    /// Queues the provided frame, the image of the given VBlank, for presentation.
    void Present(Frame* frame, u64 vblank = 0);

    /// Emulation speed, relative to the nominal one, that keeps VBlanks in step with the
    /// refreshes of the display. 1 where that is not known or not wanted.
    double SpeedScale() const {
        return pacer.SpeedScale();
    }

    /// This is called to notify the rendering backend of a surface change
    void NotifySurfaceChanged();

    [[nodiscard]] vk::RenderPass Renderpass() const noexcept {
        return present_renderpass;
    }

    u32 ImageCount() const noexcept {
        return swapchain.GetImageCount();
    }

    vk::Extent2D Extent() const noexcept {
        return swapchain.GetExtent();
    }

    vk::Format GetSurfaceFormat() const noexcept {
        return swapchain.GetSurfaceFormat().format;
    }

private:
    void PresentThread(std::stop_token token);

    void CopyToSwapchain(Frame* frame);

    // With frames that are images of the swapchain:
    void DirectPresentThread(std::stop_token token);
    /// Hands the frame to the display
    void Show(Frame* frame);
    /// Gets one more image to be drawn into, if one is free or comes free in that time. False if
    /// none was.
    bool AcquireAhead(std::chrono::nanoseconds patience, std::stop_token token);
    /// Makes the swapchain anew, for a new surface once there is one or for the same. False if the
    /// thread is to stop.
    bool RemakeImages(bool new_surface, std::stop_token token);
    void MakeImageFrames();
    void DestroyImageFrames();
    /// Whether something changed that the swapchain was made for
    bool IsOutdated();

    /// Tells the frame pacer and the hitch tracker about the display behind the swapchain.
    void UpdateDisplay();

public:
    /// By how many degrees the frames for this window are to be drawn turned, see
    /// Swapchain::GetPreRotation. Any thread.
    u32 PreRotation() const noexcept {
        return pre_rotation.load(std::memory_order_relaxed);
    }

private:
    std::atomic<u32> pre_rotation{};

    vk::RenderPass CreateRenderpass(vk::AttachmentLoadOp load_op, vk::ImageLayout final_layout);

private:
    Frontend::EmuWindow& emu_window;
    const Instance& instance;
    Scheduler& scheduler;
    bool low_refresh_rate;
    vk::SurfaceKHR surface;
    vk::SurfaceKHR next_surface{};
    Swapchain swapchain;
    vk::CommandPool command_pool;
    vk::Queue graphics_queue;
    vk::RenderPass present_renderpass;
    vk::RenderPass present_renderpass_uncleared;
    vk::RenderPass direct_renderpass;
    vk::RenderPass direct_renderpass_uncleared;
    std::vector<Frame> swap_chain;
    std::queue<Frame*> free_queue;
    std::queue<Frame*> present_queue;
    std::condition_variable free_cv;
    std::condition_variable_any recreate_surface_cv;
    std::condition_variable_any frame_cv;
    std::mutex swapchain_mutex;
    std::mutex recreate_surface_mutex;
    std::mutex queue_mutex;
    std::mutex free_mutex;
    std::jthread present_thread;
    bool vsync_enabled{};
    bool blit_supported;
    bool use_present_thread{true};
    /// Whether this is the window whose images are paced and followed by the hitch tracker
    bool is_main_window{};
    VideoCore::FramePacer pacer;
    /// Whether the swapchain shows every image and tells when, which pacing needs
    std::atomic_bool can_pace{};
    u32 next_present_id{};
    /// State of the LegacyPresentMode experiment the swapchain was made for
    bool legacy_present_mode{};
    void* last_render_surface{};
    GpuSpanMeter copy_meter;

    bool draws_into_swapchain{};
    /// Images at hand to be drawn into, being drawn into or on their way to the display, at the most:
    /// handing one over takes until the display has taken the one before, and what is to be drawn
    /// next is not to wait for that. The display keeps as many as the swapchain must have.
    static constexpr u32 IMAGES_AHEAD = 3;
    // Under queue_mutex: whether a frame was asked for yet, whether one is being waited for, images
    // acquired and not presented, those of them that are being drawn into, whether one is being
    // presented, and whether there is no surface to get any from
    bool wants_images{};
    bool waits_for_image{};
    u32 images_held{};
    u32 images_drawn_into{};
    bool is_showing{};
    bool surface_lost{};
    // Of the present thread
    std::vector<vk::Semaphore> acquire_semaphores;
    std::vector<u64> acquire_ticks;
    u32 next_acquire_slot{};
};

} // namespace Vulkan
