#!/usr/bin/env bash
# Builds the arm64 profiling APK and installs it next to the release build.
#
# Usage: build-install.sh [--no-build] [--no-install]

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

build=1
install=1
for arg in "$@"; do
    case "$arg" in
    --no-build) build=0 ;;
    --no-install) install=0 ;;
    -h | --help)
        sed -n '2,4p' "$0"
        exit 0
        ;;
    *) die "unknown argument: $arg" ;;
    esac
done

task="assemble$(echo "${VARIANT:0:1}" | tr '[:lower:]' '[:upper:]')${VARIANT:1}"
flavor="${VARIANT%Profile}"

if [ "$build" = 1 ]; then
    info "Building $task (arm64-v8a only)"
    if command -v ccache > /dev/null 2>&1; then
        export NDK_CCACHE="${NDK_CCACHE:-$(command -v ccache)}"
    fi
    (cd "$ANDROID_DIR" && ./gradlew --console=plain "$task" -Pandroid.injected.build.abi=arm64-v8a)
fi

# Builds restricted to one ABI (android.injected.build.abi) leave the APK under intermediates/.
apk=""
for candidate in "$ANDROID_DIR"/app/build/{outputs,intermediates}/apk/"$flavor"/profile/*.apk; do
    if [ -f "$candidate" ] && { [ -z "$apk" ] || [ "$candidate" -nt "$apk" ]; }; then
        apk="$candidate"
    fi
done
[ -n "$apk" ] || die "no $VARIANT APK found under $ANDROID_DIR/app/build"
info "APK: $apk"

libs="$(find_unstripped_libs || true)"
if [ -n "$libs" ]; then
    info "Unstripped native libraries (for symbolization): $libs"
else
    warn "unstripped $NATIVE_LIB not found; simpleperf reports will lack symbols"
fi

if [ "$install" = 1 ]; then
    require_device
    info "Installing $PACKAGE"
    # -t: single-ABI builds are marked testOnly by the Android Gradle plugin.
    "$ADB" install -r -t "$apk"
    cat >&2 << EOF

The profiling build is a separate app ("Azahar Profile") with its own data. On first launch,
point it at the same user folder/games as the release build, and use the same settings as the
configuration you want to investigate.
EOF
fi
