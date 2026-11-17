// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

// Decides when each image the emulation hands over should reach the screen.
//
// The emulation thread hands an image over when the work of a VBlank is done, which is not at
// a regular time: the work of one VBlank can take a few milliseconds and that of the next one
// thirty. Shown as they come, such images stay on screen for uneven numbers of refreshes. Here
// each image gets a refresh of its own instead, from the number of the VBlank it belongs to:
// images of VBlanks that are n apart are shown n refreshes apart, as long as they are handed
// over early enough for theirs.
//
// For that the emulation has to produce one VBlank per display refresh (or per whole number
// of them). When the two rates are close, the pacer asks for a slightly different emulation
// speed (0.28% faster for a 60 Hz display) and keeps correcting it, so that images are handed
// over a steady time ahead of the refresh they are meant for. When the rates are not close the
// images are still spaced by their VBlank numbers, at the emulation's own rate.
//
// It needs a presentation backend that tells when images reached the display and that accepts
// a display time for each image. Without one, Schedule returns no time and the speed stays 1.

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include "common/common_types.h"

namespace VideoCore {

class FramePacer {
public:
    using Clock = std::chrono::steady_clock;

    struct Status {
        bool active;       ///< Images are being given display times
        bool locked;       ///< Emulated VBlanks are kept in step with the display's refreshes
        bool overloaded;   ///< Images keep coming too late: they are shown as they come
        double speed;      ///< Emulation speed asked for, relative to the nominal one
        double period_ms;  ///< Measured refresh period of the display
        double lead_ms;    ///< How long before its refresh the latest of the recent images came
        double target_ms;  ///< The lead aimed for
        double minimum_ms; ///< The lead below which an image misses its refresh
        u32 slips;         ///< Images that came too late for their refresh
        u32 misses;        ///< Images that were on time and still shown late
    };

    /// The refresh period of the display, and whether it reports display times and takes them.
    void SetDisplay(std::chrono::nanoseconds refresh_period, bool can_schedule);

    /// An image reached the display at the given time (steady clock, nanoseconds).
    void NoteDisplayed(u32 present_id, u64 display_time_ns);

    /// The image `present_id` of VBlank number `vblank` is being handed over. Returns the time it
    /// should be displayed at (steady clock, nanoseconds), or 0 for as soon as possible.
    u64 Schedule(u32 present_id, u64 vblank, Clock::time_point now);

    /// Stops pacing until the next image, as after a pause or while the speed is not the
    /// normal one.
    void Reset();

    /// Emulation speed that keeps VBlanks in step with the display, relative to the nominal
    /// speed. 1 when they are not being kept in step.
    double SpeedScale() const {
        return speed_scale.load(std::memory_order_relaxed);
    }

    Status GetStatus() const;

private:
    struct Scheduled {
        u32 present_id;
        s64 target;
        s64 lead;
    };
    struct HandedOver {
        u32 present_id;
        s64 time;
    };

    s64 BoundaryAtOrAfter(s64 time) const;
    s64 NearestBoundary(s64 time) const;
    s64 LowestRecentLead() const;
    s64 LowestRecentDelay() const;
    void ResetLocked();

    mutable std::mutex mutex;

    // Display
    bool can_schedule{};
    double period{};      ///< Refresh period in nanoseconds, measured once reports come in
    s64 nominal_period{}; ///< As the display announces it
    s64 last_display{};   ///< Time of the last reported refresh
    s64 period_anchor{};  ///< An earlier refresh the period is measured from
    Clock::time_point last_report{};

    // Schedule
    bool has_reference{};
    u64 reference_vblank{};
    s64 reference_target{};
    bool locked{};
    s64 minimum_lead{};
    /// Kept between the minimum lead and the latest recent image. Grows when images come too
    /// late, shrinks slowly while they do not.
    s64 margin{};
    std::array<s64, 32> recent_leads{};
    u32 lead_samples{};
    u32 slips{};
    u32 misses{};
    /// Share of the recent images that came too late for their refresh. When that is many of
    /// them, the emulation or the GPU cannot keep up and no amount of lead helps: giving the
    /// images times would only add to the delay.
    double late_rate{};
    bool overloaded{};
    /// Images shown at their refresh since the minimum lead was last changed
    u32 on_time_streak{};
    s64 previous_scheduled_display{};
    std::array<Scheduled, 16> scheduled{};
    std::size_t next_scheduled{};

    /// When the last images were handed over, and how long after that the display showed
    /// them: what an image that has no slot to keep is given its time by
    std::array<HandedOver, 16> handed_over_at{};
    std::array<s64, 8> recent_delays{};
    u32 delay_samples{};
    /// The time the last image was given
    s64 last_time_given{};

    std::atomic<double> speed_scale{1.0};
};

} // namespace VideoCore
