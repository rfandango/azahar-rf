// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <optional>
#include <string>
#include <vector>
#include "common/common_types.h"

/**
 * What a profiling build needs to find out on a device nobody can attach a debugger to why a
 * game runs slowly there. The person with the device plays for a minute and copies the `log`
 * folder of the user directory; everything is in it:
 *
 *  - the emulator's own log, with a description of the device at its start and a line on the
 *    state of the system (processor clocks, thermal state, GPU load) beside every summary of
 *    the hitch tracker;
 *  - the log of the graphics driver (`driver_log.txt`), where the driver is Mesa's: it is told
 *    to write one, and our build of Turnip puts its diagnostic log there (TU_AZAHAR_DIAG);
 *  - what the driver or the emulator wrote besides: traces of the GPU's work (`gpu_trace*`),
 *    frames dumped with their GPU times (`frame-dump-*`).
 *
 * `diag.txt` in the user directory sets what `adb shell setprop` would otherwise, one setting
 * a line, `#` starts a comment:
 *
 *     debug.azahar.experiment3=16          a debug.azahar.* setting, as the system property
 *     TU_DEBUG=perf                        anything else: the driver's environment
 *     MESA_GPU_TRACEFILE=%FILES%/gpu_trace.txt     %FILES% is the folder the driver can write to
 *     dump_frames_at=30,90                 seconds into a game at which a frame is dumped
 *
 * The file is read when the app starts, as the driver's environment has to be there before the
 * driver is loaded.
 */
namespace Common::Diagnostics {

/// The folder of the app that plain file calls reach (/sdcard/Android/data/<package>/files),
/// which is where a driver can write. Empty where there is none.
std::string FilesDir();

/// Reads `diag.txt` from the user directory. Call once the user directory is known.
void LoadControlFile();

/// A `debug.azahar.*` setting: from the control file if it is there, else the system property
std::optional<std::string> Property(const char* name);

/// Before the graphics driver is loaded: tells it where to write its log, turns on the
/// diagnostic log of our Turnip, and adds what the control file has for it
void PrepareDriverEnvironment();

/// Seconds into a game at which a frame is dumped with its GPU times, in rising order
const std::vector<u32>& FrameDumpTimes();

/// Whether frames are dumped on the emulator's own judgement: when a game has been running
/// below its speed for a while (VideoCore::FrameStateDump). So unless the control file says
/// when to dump (`dump_frames_at=` with nothing after it for never).
bool AutomaticFrameDumps();

/// Called at every emulated VBlank. Runs the build's own comparison: while a game is below
/// its speed it runs in turns as built (30 s) and, 15 s each, with the depth buffer loaded as
/// before, with the processor clock held at the console's, with the driver asked to render
/// every pass directly, and with vertex data copied for every draw. Each turn is said in the
/// log, so the summaries beside it show what
/// each is worth in that scene on that device. A game at its speed is left as built.
/// `comparison=off` in the control file turns it off.
void OnVBlank();

/// A game starts: the comparison begins again as built
void OnGameStart();

/// The bits of `debug.azahar.experiment3` the comparison turns on for the turn it is in
u32 ComparisonBits();

/// Describes the device in the log: model, system on a chip, the processor's cores and their
/// clocks, memory, what the kernel says of the GPU
void LogDeviceReport();

/// The state of the system now, for the log: the clock of each group of cores, the thermal
/// state, the GPU's load and clock where the app may read them. Empty where nothing is known.
std::string SystemLine();

/// Copies what the driver and the emulator wrote into FilesDir() to the log folder of the user
/// directory
void CollectFiles();

} // namespace Common::Diagnostics
