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
};

class PresentWindow final {
public:
    explicit PresentWindow(Frontend::EmuWindow& emu_window, const Instance& instance,
                           Scheduler& scheduler, bool low_refresh_rate);
    ~PresentWindow();

    /// Waits for all queued frames to finish presenting.
    void WaitPresent();

    /// Returns the last used render frame.
    Frame* GetRenderFrame();

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

    vk::Format GetSurfaceFormat() const noexcept {
        return swapchain.GetSurfaceFormat().format;
    }

private:
    void PresentThread(std::stop_token token);

    void CopyToSwapchain(Frame* frame);

    /// Tells the frame pacer and the hitch tracker about the display behind the swapchain.
    void UpdateDisplay();

    vk::RenderPass CreateRenderpass();

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
    std::vector<Frame> swap_chain;
    std::queue<Frame*> free_queue;
    std::queue<Frame*> present_queue;
    std::condition_variable free_cv;
    std::condition_variable recreate_surface_cv;
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
};

} // namespace Vulkan
