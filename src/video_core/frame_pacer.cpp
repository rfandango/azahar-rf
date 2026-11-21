// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <cmath>
#include "common/trace_markers.h"
#include "core/core_timing.h"
#include "video_core/frame_pacer.h"

namespace VideoCore {

namespace {

using Nanos = s64;

constexpr Nanos MS = 1'000'000;
constexpr Nanos SECOND = 1000 * MS;

/// Time an emulated VBlank takes at the nominal speed, in nanoseconds
constexpr double VBLANK_PERIOD = 1e9 / SCREEN_REFRESH_RATE;

/// How far the emulation's rate may be from the display's (or a whole fraction of it) for the
/// emulation to be kept in step with it
constexpr double MAX_RATE_DIFFERENCE = 0.01;

/// Largest change of speed to hold the lead, and the change per nanosecond the lead is off:
/// 0.1% for 5 ms, which takes those 5 ms back in about five seconds
constexpr double MAX_CORRECTION = 0.003;
constexpr double CORRECTION_GAIN = 0.001 / (5.0 * MS);

/// Without a report from the display for this long there is nothing to schedule against
constexpr std::chrono::milliseconds STALE_AFTER{250};

/// Images of VBlanks further apart than this are not paced relative to each other
constexpr u64 MAX_VBLANK_GAP = 120;

/// Shares of late images at which pacing is given up and taken up again
constexpr double OVERLOAD_ENTER = 0.15;
constexpr double OVERLOAD_LEAVE = 0.02;

/// Images shown on time in a row after which the minimum lead is tried a little lower
constexpr u32 LOWER_LEAD_AFTER = 120;

/// Where the display reports how early each image was ready: how early the least ready of the
/// recent images is to be, and how far below what the display needed for them a lead may be
/// before the image is taken for late (images reported 2 to 5 ms late were still shown)
constexpr Nanos READY_TARGET = 3 * MS;
constexpr Nanos SLIP_SLACK = 2 * MS;
constexpr u32 REPORTS_TO_GO_BY = 8;

/// The largest change of speed while the lead is more than a refresh off, as after a load
constexpr double MAX_CATCH_UP = 0.010;

Nanos ToNanos(FramePacer::Clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

} // Anonymous namespace

void FramePacer::SetDisplay(std::chrono::nanoseconds refresh_period, bool can_schedule_) {
    std::scoped_lock lock{mutex};
    can_schedule = can_schedule_ && refresh_period.count() > 0;
    nominal_period = refresh_period.count();
    period = static_cast<double>(nominal_period);
    last_display = 0;
    period_anchor = 0;
    // A first guess: the display takes an image a little under a refresh before it shows it.
    // Images that are shown late in spite of it raise the guess.
    minimum_lead = nominal_period * 3 / 4 + 4 * MS;
    work_allowance = 0;
    margin = nominal_period / 4;
    ResetLocked();
}

void FramePacer::Reset() {
    std::scoped_lock lock{mutex};
    ResetLocked();
}

void FramePacer::ResetLocked() {
    has_reference = false;
    locked = false;
    lead_samples = 0;
    late_rate = 0;
    overloaded = false;
    speed_scale.store(1.0, std::memory_order_relaxed);
}

s64 FramePacer::BoundaryAtOrAfter(s64 time) const {
    const double refreshes = std::ceil(static_cast<double>(time - last_display) / period);
    return last_display + std::llround(refreshes * period);
}

s64 FramePacer::NearestBoundary(s64 time) const {
    const double refreshes = std::round(static_cast<double>(time - last_display) / period);
    return last_display + std::llround(refreshes * period);
}

s64 FramePacer::LowestRecentLead() const {
    const std::size_t count = std::min<std::size_t>(lead_samples, recent_leads.size());
    if (count == 0) {
        return 0;
    }
    return *std::min_element(recent_leads.begin(), recent_leads.begin() + count);
}

s64 FramePacer::LowestRecentDelay() const {
    const std::size_t count = std::min<std::size_t>(delay_samples, recent_delays.size());
    if (count == 0) {
        return minimum_lead + nominal_period;
    }
    return *std::min_element(recent_delays.begin(), recent_delays.begin() + count);
}

void FramePacer::NoteQueued(u32 present_id, Clock::time_point now) {
    std::scoped_lock lock{mutex};
    if (present_id == 0) {
        return;
    }
    const HandedOver& image = handed_over_at[present_id % handed_over_at.size()];
    const Nanos time = ToNanos(now);
    if (image.present_id != present_id || time <= image.time) {
        return;
    }
    // Not more than a few refreshes: beyond that the GPU cannot keep up, which is not for a
    // lead to make up
    const Nanos work = std::min(time - image.time, static_cast<Nanos>(nominal_period * 2));
    if (work > work_allowance) {
        work_allowance = work;
    } else {
        // Down slowly, over a few seconds: a scene's heavy frames come back
        work_allowance -= (work_allowance - work) / 256;
    }
    for (Scheduled& scheduled_image : scheduled) {
        if (scheduled_image.present_id == present_id) {
            scheduled_image.work = work;
            break;
        }
    }
}

void FramePacer::NoteDisplayed(u32 present_id, u64 display_time_ns, s64 margin_ns) {
    std::scoped_lock lock{mutex};
    const Nanos time = static_cast<Nanos>(display_time_ns);
    last_report = Clock::now();

    if (present_id != 0) {
        const HandedOver& image = handed_over_at[present_id % handed_over_at.size()];
        if (image.present_id == present_id && time > image.time) {
            recent_delays[delay_samples % recent_delays.size()] = time - image.time;
            delay_samples++;
        }
    }

    // The refresh period, from reports far enough apart for their precision not to matter
    if (period_anchor == 0 || time <= last_display || time - period_anchor > 4 * SECOND) {
        period_anchor = time;
    } else if (time - period_anchor >= SECOND) {
        const double elapsed = static_cast<double>(time - period_anchor);
        const double refreshes = std::round(elapsed / period);
        if (refreshes > 0) {
            const double measured = elapsed / refreshes;
            if (std::abs(measured / static_cast<double>(nominal_period) - 1.0) <
                MAX_RATE_DIFFERENCE) {
                period += (measured - period) * 0.5;
            }
        }
        period_anchor = time;
    }
    last_display = std::max(last_display, time);

    // An image given a time that the display never reported, while a later one is reported
    // now, was skipped: it was queued too late for its refresh and the next one took its
    // place. The display needs images earlier than estimated, as below.
    if (present_id != 0 && !overloaded) {
        for (Scheduled& image : scheduled) {
            // (ids count up and wrap; a skipped one is a few behind)
            const u32 behind = present_id - image.present_id;
            if (image.present_id == 0 || behind == 0 || behind > scheduled.size()) {
                continue;
            }
            // (one that had lead to spare was skipped for another reason)
            const Nanos slack = image.lead - image.work;
            if (slack < minimum_lead + static_cast<Nanos>(period / 4)) {
                misses++;
                on_time_streak = 0;
                const Nanos raised = slack + static_cast<Nanos>(period / 8);
                minimum_lead =
                    std::min(std::max(minimum_lead, raised), static_cast<Nanos>(period * 5 / 2));
            }
            image.present_id = 0;
        }
    }

    // An image that was handed over in time by the current estimate and still shown a refresh
    // late: the display needs images earlier than estimated
    for (Scheduled& image : scheduled) {
        if (present_id == 0 || image.present_id != present_id) {
            continue;
        }
        const Nanos half_period = static_cast<Nanos>(period / 2);
        const bool late = time - image.target > half_period;
        const bool held_up = previous_scheduled_display > image.target - half_period;
        const Nanos lowest_minimum = nominal_period * 3 / 4 + 4 * MS;
        // What it had left once it was through the GPU
        const Nanos slack = image.lead - image.work;
        if (overloaded) {
            // Nothing to learn from while images are not given times
        } else if (late && !held_up && slack < minimum_lead + half_period &&
                   (margin_ns == NO_MARGIN || margin_ns < 0)) {
            // (where the display says the image was ready in time, its lateness is not for
            // want of lead)
            misses++;
            on_time_streak = 0;
            const Nanos raised = slack + static_cast<Nanos>(period / 8);
            minimum_lead =
                std::min(std::max(minimum_lead, raised), static_cast<Nanos>(period * 5 / 2));
        } else if (!late && margin_ns == NO_MARGIN && ++on_time_streak >= LOWER_LEAD_AFTER) {
            // What made the display need its images earlier may be over
            on_time_streak = 0;
            minimum_lead = std::max(minimum_lead - static_cast<Nanos>(period / 32), lowest_minimum);
        }
        if (margin_ns != NO_MARGIN && !overloaded) {
            // The display says how early the image was ready. With when it was handed over
            // and how long it took to be queued, that is what the display needed for it once
            // it was queued. The largest of the recent ones is what the lead goes by: up at
            // once, down over a few seconds.
            const Nanos handed_over = image.target - image.lead;
            const Nanos needed = (time - handed_over) - image.work - static_cast<Nanos>(margin_ns);
            const Nanos lowest = static_cast<Nanos>(period / 8);
            const Nanos highest = static_cast<Nanos>(period * 5 / 2);
            if (needed > minimum_lead) {
                minimum_lead = std::min(needed, highest);
            } else {
                minimum_lead = std::max(minimum_lead - (minimum_lead - needed) / 256, lowest);
            }
            recent_ready[margin_reports % recent_ready.size()] = static_cast<Nanos>(margin_ns);
            margin_reports++;
        }
        previous_scheduled_display = time;
        image.present_id = 0;
        break;
    }
}

u64 FramePacer::Schedule(u32 present_id, u64 vblank, Clock::time_point now) {
    const u32 experiments = Common::TraceMarkers::Experiments();

    std::scoped_lock lock{mutex};
    if (!can_schedule || period <= 0 || last_display == 0 || now - last_report > STALE_AFTER) {
        ResetLocked();
        return 0;
    }
    const Nanos handed_over = ToNanos(now);
    handed_over_at[present_id % handed_over_at.size()] = HandedOver{present_id, handed_over};

    // One emulated VBlank per refresh, or per whole number of refreshes on a faster display
    const double refreshes_per_vblank = std::max(1.0, std::round(VBLANK_PERIOD / period));
    const double locked_step = refreshes_per_vblank * period;
    locked = std::abs(VBLANK_PERIOD / locked_step - 1.0) < MAX_RATE_DIFFERENCE &&
             !(experiments & Common::TraceMarkers::NoDisplayRateLock);
    const double step = locked ? locked_step : VBLANK_PERIOD;

    // Images are not handed over evenly. The one that comes latest of the recent ones is to
    // have the minimum lead and a margin. The margin shrinks slowly while that holds.
    const Nanos margin_floor = static_cast<Nanos>(period / 4);
    const Nanos margin_ceiling = static_cast<Nanos>(period / 2);
    margin = std::max(margin - (margin - margin_floor) / 512, margin_floor);
    // What an image needs from hand-over to its refresh: its time with the GPU, and what the
    // display needs after that
    // Where the display reports how early images were ready, the minimum is measured and the
    // aim is that much plus a little; elsewhere it is an estimate with a margin on top.
    const bool by_reports = margin_reports >= REPORTS_TO_GO_BY;
    const Nanos needed_lead = minimum_lead + work_allowance - (by_reports ? SLIP_SLACK : 0);
    const Nanos wanted_lead = minimum_lead + work_allowance + (by_reports ? READY_TARGET : margin);
    bool came_late = false;

    bool restart =
        !has_reference || vblank <= reference_vblank || vblank - reference_vblank > MAX_VBLANK_GAP;
    Nanos target = 0;
    if (!restart) {
        target =
            reference_target + std::llround(static_cast<double>(vblank - reference_vblank) * step);
        // Far too early: the emulation ran ahead, as after a change of speed
        restart = target - handed_over > needed_lead + static_cast<Nanos>(4 * period);
    }
    if (restart) {
        target = handed_over + wanted_lead;
        if (locked) {
            target = BoundaryAtOrAfter(target);
        }
        lead_samples = 0;
    } else {
        if (locked) {
            target = NearestBoundary(target);
        }
        if (target - handed_over < needed_lead) {
            // Too late for its refresh. It gets the first one it can make, and the images after
            // it keep their distance to it; the speed then brings the lead back down slowly.
            slips++;
            came_late = true;
            if (!overloaded) {
                margin = std::min(margin + margin_floor / 2, margin_ceiling);
            }
            const Nanos slot = target;
            target = handed_over + needed_lead;
            if (locked) {
                target = BoundaryAtOrAfter(target);
            }
            // The earlier images would have had that much more lead as well
            for (Nanos& recent : recent_leads) {
                recent += target - slot;
            }
        }
    }
    has_reference = true;
    reference_vblank = vblank;
    reference_target = target;

    const Nanos lead = target - handed_over;
    recent_leads[lead_samples % recent_leads.size()] = lead;
    lead_samples++;

    // When images keep coming late, something cannot keep up: the emulation, or the GPU, which
    // makes the display wait for each image. Times would only hold the images back further.
    late_rate += ((came_late ? 1.0 : 0.0) - late_rate) / 32.0;
    if (!overloaded && late_rate > OVERLOAD_ENTER) {
        overloaded = true;
        margin = margin_floor;
    } else if (overloaded && late_rate < OVERLOAD_LEAVE) {
        overloaded = false;
    }

    if (!overloaded) {
        // Until it is known how long this one took, the allowance stands for it
        scheduled[next_scheduled] = Scheduled{present_id, target, lead, work_allowance};
        next_scheduled = (next_scheduled + 1) % scheduled.size();
    }

    if (locked) {
        // Handing over earlier than needed: slow down, so that the next images come later
        const double error = static_cast<double>(LowestRecentLead() - wanted_lead);
        // (more of it while the lead is far off, as after a load: that is images queued ahead,
        // which only a slower emulation works off)
        const double limit = std::abs(error) > period ? MAX_CATCH_UP : MAX_CORRECTION;
        const double correction =
            overloaded ? 0.0 : std::clamp(-error * CORRECTION_GAIN, -limit, limit);
        speed_scale.store(VBLANK_PERIOD / locked_step * (1.0 + correction),
                          std::memory_order_relaxed);
    } else {
        speed_scale.store(1.0, std::memory_order_relaxed);
    }

    if (experiments & Common::TraceMarkers::NoPresentTimes) {
        return 0;
    }
    const Nanos half_period = static_cast<Nanos>(period / 2);
    Nanos time;
    if (overloaded) {
        // There is no slot to keep, so the image is to be shown as soon as it can be. It is
        // given a time all the same: the display goes on using the last time it was given, and
        // an image whose time is long past is skipped as soon as another waits behind it, which
        // with a busy GPU is always. The time is when the image will be on display anyway,
        // going by the last few, and half a refresh less in case it can be sooner by now.
        time = handed_over + LowestRecentDelay() - half_period;
    } else {
        // The display shows an image at the first refresh after the time asked for: ask for
        // the middle of the refresh period before, which leaves half a period of error either
        // way
        time = target - half_period;
    }
    // No two images for the same refresh
    time = std::max(time, last_time_given + static_cast<Nanos>(period));
    last_time_given = time;
    return static_cast<u64>(time);
}

FramePacer::Status FramePacer::GetStatus() const {
    std::scoped_lock lock{mutex};
    const bool by_reports = margin_reports >= REPORTS_TO_GO_BY;
    const std::size_t ready_count = std::min<std::size_t>(margin_reports, recent_ready.size());
    const Nanos least_ready =
        ready_count != 0
            ? *std::min_element(recent_ready.begin(), recent_ready.begin() + ready_count)
            : 0;
    return Status{
        .active = has_reference,
        .locked = locked,
        .overloaded = overloaded,
        .speed = speed_scale.load(std::memory_order_relaxed),
        .period_ms = period / static_cast<double>(MS),
        .lead_ms = static_cast<double>(LowestRecentLead()) / static_cast<double>(MS),
        .target_ms = static_cast<double>(minimum_lead + work_allowance +
                                         (by_reports ? READY_TARGET : margin)) /
                     static_cast<double>(MS),
        .minimum_ms = static_cast<double>(minimum_lead + work_allowance) / static_cast<double>(MS),
        .work_ms = static_cast<double>(work_allowance) / static_cast<double>(MS),
        .ready_ms = static_cast<double>(least_ready) / static_cast<double>(MS),
        .slips = slips,
        .misses = misses,
    };
}

} // namespace VideoCore
