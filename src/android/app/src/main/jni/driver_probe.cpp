// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/arch.h"
#include "jni/driver_probe.h"

#if defined(ENABLE_VULKAN) && CITRA_ARCH(arm64)

#include <vector>
#include <dlfcn.h>
#include <fmt/format.h>
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan_core.h>
#include "common/common_types.h"
#include "common/scope_exit.h"

bool ProbeTurnipDriver(void* vulkan_library, std::string& description) {
    const auto get_proc =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(vulkan_library, "vkGetInstanceProcAddr"));
    if (!get_proc) {
        description = "has no entry point";
        return false;
    }
    const auto create_instance =
        reinterpret_cast<PFN_vkCreateInstance>(get_proc(nullptr, "vkCreateInstance"));
    if (!create_instance) {
        description = "cannot make an instance";
        return false;
    }

    // Under the names the renderer goes by (VideoCore's CreateInstance): what Turnip is told of an
    // application it may take once for the process, from whoever comes first, such as that it
    // renders directly for this one on a6xx.
    const VkApplicationInfo application = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Citra",
        .pEngineName = "Citra Vulkan",
        .apiVersion = VK_API_VERSION_1_1,
    };
    const VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application,
    };
    VkInstance instance{};
    if (const VkResult result = create_instance(&instance_info, nullptr, &instance);
        result != VK_SUCCESS) {
        description = fmt::format("cannot make an instance ({})", static_cast<int>(result));
        return false;
    }
#define INSTANCE_FUNCTION(name) reinterpret_cast<PFN_##name>(get_proc(instance, #name))
    const auto destroy_instance = INSTANCE_FUNCTION(vkDestroyInstance);
    const auto enumerate_devices = INSTANCE_FUNCTION(vkEnumeratePhysicalDevices);
    const auto get_properties = INSTANCE_FUNCTION(vkGetPhysicalDeviceProperties2);
    const auto get_queue_families = INSTANCE_FUNCTION(vkGetPhysicalDeviceQueueFamilyProperties);
    const auto create_device = INSTANCE_FUNCTION(vkCreateDevice);
    const auto destroy_device = INSTANCE_FUNCTION(vkDestroyDevice);
#undef INSTANCE_FUNCTION
    SCOPE_EXIT({
        if (destroy_instance) {
            destroy_instance(instance, nullptr);
        }
    });
    if (!enumerate_devices || !get_properties || !get_queue_families || !create_device ||
        !destroy_device) {
        description = "lacks functions";
        return false;
    }

    // A chip the driver does not know, or a kernel interface it cannot use, leaves no device
    u32 device_count = 1;
    VkPhysicalDevice physical_device{};
    const VkResult enumerated = enumerate_devices(instance, &device_count, &physical_device);
    if ((enumerated != VK_SUCCESS && enumerated != VK_INCOMPLETE) || device_count == 0) {
        description = fmt::format("finds no GPU it can drive ({})", static_cast<int>(enumerated));
        return false;
    }
    VkPhysicalDeviceDriverProperties driver = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 properties = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &driver,
    };
    get_properties(physical_device, &properties);
    if (driver.driverID != VK_DRIVER_ID_MESA_TURNIP) {
        description = fmt::format("was not taken: {} answers", properties.properties.deviceName);
        return false;
    }

    u32 family_count = 0;
    get_queue_families(physical_device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    get_queue_families(physical_device, &family_count, families.data());
    u32 family = 0;
    while (family < family_count && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
        ++family;
    }
    if (family == family_count) {
        description = "has no graphics queue";
        return false;
    }
    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };
    VkDevice device{};
    if (const VkResult result = create_device(physical_device, &device_info, nullptr, &device);
        result != VK_SUCCESS) {
        description = fmt::format("cannot make a device ({})", static_cast<int>(result));
        return false;
    }
    destroy_device(device, nullptr);
    description = fmt::format("{}, {} {}", properties.properties.deviceName, driver.driverName,
                              driver.driverInfo);
    return true;
}

#else

bool ProbeTurnipDriver(void*, std::string& description) {
    description = "not for this kind of device";
    return false;
}

#endif
