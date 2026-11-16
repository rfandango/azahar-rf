#!/usr/bin/env bash
# Shared settings for the Android profiling scripts. Source this file; don't run it.
#
# Environment overrides:
#   AZAHAR_PACKAGE   package to profile        (default: org.azahar_emu.azahar.profile)
#   AZAHAR_VARIANT   Gradle variant            (default: vanillaProfile)
#   ANDROID_HOME     Android SDK               (default: sdk.dir from src/android/local.properties)
#   ANDROID_NDK_HOME NDK, used for simpleperf  (default: the ndkVersion from build.gradle.kts)
#   ADB              adb binary                (default: adb on PATH, then the SDK's)
#   ANDROID_SERIAL   device to use when several are connected (standard adb variable)

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TOOLS_DIR/../.." && pwd)"
ANDROID_DIR="$REPO_ROOT/src/android"

PACKAGE="${AZAHAR_PACKAGE:-org.azahar_emu.azahar.profile}"
VARIANT="${AZAHAR_VARIANT:-vanillaProfile}"
NATIVE_LIB="libcitra-android.so"
EMU_THREAD="NativeEmulation"

info() { echo -e "\033[1;34m==>\033[0m $*" >&2; }
warn() { echo -e "\033[1;33mwarning:\033[0m $*" >&2; }
die() {
    echo -e "\033[1;31merror:\033[0m $*" >&2
    exit 1
}

find_sdk() {
    if [ -n "${ANDROID_HOME:-}" ]; then
        echo "$ANDROID_HOME"
    elif [ -n "${ANDROID_SDK_ROOT:-}" ]; then
        echo "$ANDROID_SDK_ROOT"
    elif [ -f "$ANDROID_DIR/local.properties" ]; then
        sed -n 's/^sdk\.dir=//p' "$ANDROID_DIR/local.properties" | tail -1
    fi
}

find_ndk() {
    if [ -n "${ANDROID_NDK_HOME:-}" ]; then
        echo "$ANDROID_NDK_HOME"
        return
    fi
    local sdk version
    sdk="$(find_sdk)"
    [ -n "$sdk" ] || return 0
    version="$(sed -n 's/^ *ndkVersion *= *"\([^"]*\)".*/\1/p' "$ANDROID_DIR/app/build.gradle.kts" | head -1)"
    if [ -n "$version" ] && [ -d "$sdk/ndk/$version" ]; then
        echo "$sdk/ndk/$version"
    elif [ -d "$sdk/ndk" ]; then
        # Any NDK works for simpleperf; prefer the newest.
        find "$sdk/ndk" -mindepth 1 -maxdepth 1 | sort -V | tail -1
    fi
}

find_adb() {
    if [ -n "${ADB:-}" ]; then
        echo "$ADB"
    elif command -v adb > /dev/null 2>&1; then
        command -v adb
    else
        local sdk
        sdk="$(find_sdk)"
        [ -x "$sdk/platform-tools/adb" ] && echo "$sdk/platform-tools/adb"
    fi
}

ADB="$(find_adb || true)"
[ -n "$ADB" ] || die "adb not found; put it on PATH or set ADB / ANDROID_HOME"
# simpleperf's scripts expect adb on PATH.
export PATH="$(dirname "$ADB"):$PATH"

adb_shell() { "$ADB" shell "$@"; }

require_device() {
    "$ADB" get-state > /dev/null 2>&1 || die "no device connected (adb get-state failed)"
}

app_pid() { { adb_shell pidof "$PACKAGE" 2> /dev/null || true; } | tr -d '\r' | awk '{print $1}'; }

require_running_app() {
    require_device
    local pid
    pid="$(app_pid)"
    [ -n "$pid" ] || die "$PACKAGE is not running; start a game in it first"
    echo "$pid"
}

# Directory holding the unstripped native libraries of the last build of $VARIANT.
find_unstripped_libs() {
    local lib
    lib="$( (find "$ANDROID_DIR/app/build/intermediates/merged_native_libs/$VARIANT" \
        -path '*arm64-v8a*' -name "$NATIVE_LIB" 2> /dev/null || true) | head -1)"
    [ -n "$lib" ] && dirname "$lib"
}

new_output_dir() {
    local kind="$1" dir
    dir="${OUTPUT_DIR:-$REPO_ROOT/build/android-profiles/$kind-$(date +%Y%m%d-%H%M%S)}"
    mkdir -p "$dir"
    echo "$dir"
}

# Records device state that commonly explains performance differences between runs.
write_device_info() {
    local out="$1/device-info.txt"
    {
        echo "date: $(date -Is)"
        echo "package: $PACKAGE ($("$ADB" shell dumpsys package "$PACKAGE" | sed -n 's/ *versionName=//p' | head -1 | tr -d '\r'))"
        for p in ro.product.model ro.soc.model ro.build.fingerprint debug.azahar.trace; do
            echo "$p: $(adb_shell getprop "$p" | tr -d '\r')"
        done
        echo "thermal status: $(adb_shell dumpsys thermalservice 2> /dev/null | sed -n 's/^Thermal Status: //p' | tr -d '\r')"
        echo "battery: $(adb_shell dumpsys battery | grep -E 'level|temperature|AC powered|USB powered' | tr -d '\r' | xargs)"
        echo "--- cpufreq (cur/max kHz, governor)"
        adb_shell 'for c in /sys/devices/system/cpu/cpu[0-9]*; do
            echo "${c##*/}: $(cat $c/cpufreq/scaling_cur_freq 2>/dev/null)/$(cat $c/cpufreq/scaling_max_freq 2>/dev/null) $(cat $c/cpufreq/scaling_governor 2>/dev/null)"
        done' | tr -d '\r'
        echo "--- gpu"
        adb_shell 'cat /sys/class/kgsl/kgsl-3d0/gpu_model /sys/class/kgsl/kgsl-3d0/gpuclk /sys/class/kgsl/kgsl-3d0/max_gpuclk 2>/dev/null' | tr -d '\r'
    } > "$out" 2>&1 || true
}
