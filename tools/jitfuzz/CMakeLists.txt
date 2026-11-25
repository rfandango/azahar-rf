# jitfuzz: the recompiler tested against itself, see jitfuzz.cpp. Built on its own (not part of
# the emulator's build), for the machine the recompiler's backend is for:
#
#   cmake -S tools/jitfuzz -B <build dir> -G Ninja -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake \
#       -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30
cmake_minimum_required(VERSION 3.16)
project(jitfuzz C CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AZAHAR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

set(DYNARMIC_TESTS OFF CACHE BOOL "")
set(DYNARMIC_FRONTENDS "A32" CACHE STRING "")
set(DYNARMIC_USE_BUNDLED_EXTERNALS ON CACHE BOOL "")
set(DYNARMIC_USE_PRECOMPILED_HEADERS OFF CACHE BOOL "")
set(DYNARMIC_WARNINGS_AS_ERRORS OFF CACHE BOOL "")
set(BOOST_ROOT "${AZAHAR_ROOT}/externals/boost" CACHE STRING "")
set(Boost_INCLUDE_DIR "${AZAHAR_ROOT}/externals/boost" CACHE STRING "")
set(Boost_NO_SYSTEM_PATHS ON CACHE BOOL "")
# The toolchain file of the NDK looks for packages in its own tree only
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

add_subdirectory("${AZAHAR_ROOT}/externals/dynarmic" dynarmic EXCLUDE_FROM_ALL)

add_executable(jitfuzz
    jitfuzz.cpp
    "${AZAHAR_ROOT}/externals/dynarmic/tests/fuzz_util.cpp"
)
target_include_directories(jitfuzz PRIVATE
    "${AZAHAR_ROOT}/externals/dynarmic/src"
    "${AZAHAR_ROOT}/externals/dynarmic/tests"
    "${AZAHAR_ROOT}/externals/boost"
)
target_link_libraries(jitfuzz PRIVATE dynarmic fmt::fmt merry::mcl)
target_link_options(jitfuzz PRIVATE -static-libstdc++)

# What the direct translator makes of a program's code
add_executable(directdump directdump.cpp)
target_include_directories(directdump PRIVATE
    "${AZAHAR_ROOT}/externals/dynarmic/src"
    "${AZAHAR_ROOT}/externals/dynarmic/externals/oaknut/include"
    "${AZAHAR_ROOT}/externals/dynarmic/externals/robin-map/include"
    "${AZAHAR_ROOT}/externals/boost"
)
target_link_libraries(directdump PRIVATE dynarmic fmt::fmt merry::mcl)
target_link_options(directdump PRIVATE -static-libstdc++)

# How long the direct translator takes
add_executable(directbench directbench.cpp)
target_include_directories(directbench PRIVATE
    "${AZAHAR_ROOT}/externals/dynarmic/src"
    "${AZAHAR_ROOT}/externals/dynarmic/externals/oaknut/include"
    "${AZAHAR_ROOT}/externals/dynarmic/externals/robin-map/include"
    "${AZAHAR_ROOT}/externals/boost"
)
target_link_libraries(directbench PRIVATE dynarmic fmt::fmt merry::mcl)
target_link_options(directbench PRIVATE -static-libstdc++)
