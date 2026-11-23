add_library(citra-android SHARED
    android_common/android_common.cpp
    android_common/android_common.h
    applets/mii_selector.cpp
    applets/mii_selector.h
    applets/swkbd.cpp
    applets/swkbd.h
    input_manager.cpp
    input_manager.h
    camera/ndk_camera.cpp
    camera/ndk_camera.h
    camera/still_image_camera.cpp
    camera/still_image_camera.h
    cheats/cheat.cpp
    cheats/cheat.h
    cheats/cheat_engine.cpp
    config.cpp
    config.h
    default_ini.h
    emu_window/emu_window.cpp
    emu_window/emu_window.h
    game_info.cpp
    id_cache.cpp
    id_cache.h
    jni_setting_keys.cpp
    multiplayer.cpp
    multiplayer.h
    native.cpp
    ndk_motion.cpp
    ndk_motion.h
    performance_hint.cpp
    performance_hint.h
    system_save_game.cpp
    native_log.cpp
)

target_link_libraries(citra-android PRIVATE audio_core citra_common citra_core input_common network)
target_link_libraries(citra-android PRIVATE android camera2ndk inih jnigraphics log mediandk yuv)

# A call from one of the library's functions to another goes straight there. Left to itself the
# linker sends every such call across two source files by way of a table that another library
# could put its own function into (which none does): 12,476 of this library's 12,925 entries of
# that table were its own functions, each call four instructions and an indirect jump longer,
# and the stubs alone took 2% of the GPU decoding thread's time and 0.8% of the emulation
# thread's. What the library offers to others (the functions Java calls) is offered as before.
target_link_options(citra-android PRIVATE -Wl,-Bsymbolic-functions)

if (ENABLE_OPENGL)
    target_sources(citra-android PRIVATE
        emu_window/emu_window_gl.cpp
        emu_window/emu_window_gl.h
    )
    target_link_libraries(citra-android PRIVATE EGL glad)
endif()
if (ENABLE_VULKAN)
    target_sources(citra-android PRIVATE
        emu_window/emu_window_vk.cpp
        emu_window/emu_window_vk.h
    )
    if ("arm64" IN_LIST ARCHITECTURE)
        target_link_libraries(citra-android PRIVATE adrenotools)
    endif()
endif()

set(CPACK_PACKAGE_EXECUTABLES ${CPACK_PACKAGE_EXECUTABLES} citra-android)
