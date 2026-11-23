// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string>

/**
 * Whether the driver that a Vulkan library opened with a replacement driver uses is that driver,
 * and works on this device: a driver that cannot be loaded is silently replaced by the system's.
 * Makes a device with it. Only for Mesa's Turnip, which is what the app may come with.
 * @param vulkan_library What adrenotools_open_libvulkan gave.
 * @param description Out: the device and driver found, or what failed.
 */
bool ProbeTurnipDriver(void* vulkan_library, std::string& description);
