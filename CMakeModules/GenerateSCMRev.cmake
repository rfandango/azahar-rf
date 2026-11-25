list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")

include(GenerateBuildInfo)
generate_build_info()

# The variable SRC_DIR must be passed into the script (since it uses the current build directory for all values of CMAKE_*_DIR)
set(VIDEO_CORE "${CMAKE_SOURCE_DIR}/src/video_core")
set(HASH_FILES
    "${VIDEO_CORE}/renderer_opengl/gl_shader_disk_cache.cpp"
    "${VIDEO_CORE}/renderer_opengl/gl_shader_disk_cache.h"
    "${VIDEO_CORE}/renderer_opengl/gl_shader_util.cpp"
    "${VIDEO_CORE}/renderer_opengl/gl_shader_util.h"
    "${VIDEO_CORE}/renderer_vulkan/vk_shader_util.cpp"
    "${VIDEO_CORE}/renderer_vulkan/vk_shader_util.h"
    "${VIDEO_CORE}/renderer_vulkan/vk_shader_disk_cache.cpp"
    "${VIDEO_CORE}/renderer_vulkan/vk_shader_disk_cache.h"
    "${VIDEO_CORE}/renderer_vulkan/vk_pipeline_cache.cpp"
    "${VIDEO_CORE}/renderer_vulkan/vk_pipeline_cache.h"
    "${VIDEO_CORE}/shader/generator/glsl_fs_shader_gen.cpp"
    "${VIDEO_CORE}/shader/generator/glsl_fs_shader_gen.h"
    "${VIDEO_CORE}/shader/generator/glsl_shader_decompiler.cpp"
    "${VIDEO_CORE}/shader/generator/glsl_shader_decompiler.h"
    "${VIDEO_CORE}/shader/generator/glsl_shader_gen.cpp"
    "${VIDEO_CORE}/shader/generator/glsl_shader_gen.h"
    "${VIDEO_CORE}/shader/generator/pica_fs_config.cpp"
    "${VIDEO_CORE}/shader/generator/pica_fs_config.h"
    "${VIDEO_CORE}/shader/generator/profile.h"
    "${VIDEO_CORE}/shader/generator/shader_gen.cpp"
    "${VIDEO_CORE}/shader/generator/shader_gen.h"
    "${VIDEO_CORE}/shader/generator/shader_uniforms.cpp"
    "${VIDEO_CORE}/shader/generator/shader_uniforms.h"
    "${VIDEO_CORE}/shader/generator/spv_fs_shader_gen.cpp"
    "${VIDEO_CORE}/shader/generator/spv_fs_shader_gen.h"
    "${VIDEO_CORE}/shader/shader.cpp"
    "${VIDEO_CORE}/shader/shader.h"
    "${VIDEO_CORE}/pica/regs_framebuffer.h"
    "${VIDEO_CORE}/pica/regs_lighting.h"
    "${VIDEO_CORE}/pica/regs_pipeline.h"
    "${VIDEO_CORE}/pica/regs_rasterizer.h"
    "${VIDEO_CORE}/pica/regs_shader.h"
    "${VIDEO_CORE}/pica/regs_texturing.h"
    "${VIDEO_CORE}/pica/regs_internal.cpp"
    "${VIDEO_CORE}/pica/regs_internal.h"
)
set(COMBINED "")
foreach (F IN LISTS HASH_FILES)
    file(READ ${F} TMP)
    set(COMBINED "${COMBINED}${TMP}")
endforeach()
# The free camera option changes the shaders, but not the sources, so hash as well.
if (ENABLE_FREECAM)
    set(COMBINED "${COMBINED}ENABLE_FREECAM")
endif()
string(MD5 SHADER_CACHE_VERSION "${COMBINED}")

# The recompiler keeps the machine code it makes between runs (core/arm/dynarmic/jit_code_cache).
# That code is only right for the recompiler that made it, so what is kept is tied to a hash
# of the recompiler's sources and of how it is set up: it then outlives the builds that leave
# them alone, and no others.
file(GLOB_RECURSE RECOMPILER_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/externals/dynarmic/src/dynarmic/*"
    "${CMAKE_SOURCE_DIR}/src/core/arm/dynarmic/*"
)
list(SORT RECOMPILER_FILES)
set(COMBINED "")
# The store itself decides nothing about the code that is kept, and its file has a version of
# its own (FILE_VERSION): a change to how the file is written must not throw away what every
# title has gathered. Its two sources stand in the hash with the sums they had when they were
# taken out of it, which left the hash as it was.
set(RECOMPILER_STORE_SUM_jit_code_cache.cpp "e33e6f25e1cc32e814d73b657b2ad3bf")
set(RECOMPILER_STORE_SUM_jit_code_cache.h "014b5cd2fb3d16c3fc87a2e5dad67118")
# Nor does what translates ahead, of which nothing is kept.
list(FILTER RECOMPILER_FILES EXCLUDE REGEX "/(ahead_translation|a32_translated_ahead|translated_ahead)\\.(h|cpp)$")
foreach (F IN LISTS RECOMPILER_FILES)
    file(RELATIVE_PATH NAME "${CMAKE_SOURCE_DIR}" "${F}")
    get_filename_component(BASENAME "${F}" NAME)
    if (NAME MATCHES "^src/core/arm/dynarmic/" AND DEFINED RECOMPILER_STORE_SUM_${BASENAME})
        set(TMP "${RECOMPILER_STORE_SUM_${BASENAME}}")
    else()
        file(MD5 "${F}" TMP)
    endif()
    string(APPEND COMBINED "${NAME}=${TMP}\n")
endforeach()
get_cmake_property(RECOMPILER_OPTIONS CACHE_VARIABLES)
list(SORT RECOMPILER_OPTIONS)
foreach (OPTION IN LISTS RECOMPILER_OPTIONS)
    if (OPTION MATCHES "^DYNARMIC_")
        string(APPEND COMBINED "${OPTION}=${${OPTION}}\n")
    endif()
endforeach()
string(MD5 RECOMPILER_CACHE_VERSION "${COMBINED}")
# The hash is made when the build is configured: a change to any of the files has to bring
# that about
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${RECOMPILER_FILES})

configure_file("${CMAKE_SOURCE_DIR}/src/common/scm_rev.cpp.in" "scm_rev.cpp" @ONLY)
