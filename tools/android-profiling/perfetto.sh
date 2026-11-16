#!/usr/bin/env bash
# Records a Perfetto system trace while the profiling build runs a game: CPU scheduling and
# clocks, GPU clocks, thermal state, SurfaceFlinger's frame timeline and Azahar's own trace
# markers (see src/common/trace_markers.h).
#
# Usage: perfetto.sh [-d SECONDS] [-l LEVEL] [-g] [-o DIR]
#   -d  duration (default 10)
#   -l  Azahar marker level: 0 off, 1 per-frame work (default), 2 also per draw / SVC / HLE
#       service call. Stored in the debug.azahar.trace property, which the app reads once per
#       process: restart the app after changing it.
#   -g  also record GPU counters and render stages through Android GPU Inspector's producer
#       (needs AGI's APK on the device and a supported driver; see README.md)
#   -o  output directory (default build/android-profiles/perfetto-<timestamp>)
#
# Open the trace at https://ui.perfetto.dev, or summarize it with analyze-trace.py.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

duration=10
level=""
gpu=0
while getopts "d:l:go:h" opt; do
    case "$opt" in
    d) duration="$OPTARG" ;;
    l) level="$OPTARG" ;;
    g) gpu=1 ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    *)
        sed -n '2,15p' "$0"
        exit 0
        ;;
    esac
done

require_device
pid="$(app_pid)"

if [ -n "$level" ]; then
    current="$(adb_shell getprop debug.azahar.trace | tr -d '\r')"
    if [ "$current" != "$level" ]; then
        adb_shell setprop debug.azahar.trace "$level"
        if [ -n "$pid" ]; then
            warn "marker level changed to $level, but $PACKAGE is already running with the old level;"
            warn "restart the app and the game, then run this again"
            exit 1
        fi
    fi
fi

if [ "$gpu" = 1 ]; then
    # Enables the driver's render stage producer for apps started after this point.
    if [ "$(adb_shell getprop debug.graphics.gpu.profiler.perfetto | tr -d '\r')" != "1" ]; then
        adb_shell setprop debug.graphics.gpu.profiler.perfetto 1
        [ -n "$pid" ] && warn "GPU render stages need the app to be restarted to be captured"
    fi
    "$TOOLS_DIR/gpu-producer.sh" start || warn "GPU counters unavailable; continuing without them"
fi

[ -n "$pid" ] || warn "$PACKAGE is not running; the trace will only contain system activity until it starts"

out="$(new_output_dir perfetto)"
write_device_info "$out"
config="$out/config.pbtx"
sed -e "s|@DURATION_MS@|$((duration * 1000))|" -e "s|@PACKAGE@|$PACKAGE|" \
    "$TOOLS_DIR/perfetto.pbtx" > "$config"
if [ "$gpu" = 1 ]; then
    "$TOOLS_DIR/gpu-producer.sh" config >> "$config" || true
fi

remote="/data/misc/perfetto-traces/azahar-$(date +%s).perfetto-trace"
info "Tracing for ${duration}s"
"$ADB" shell perfetto --txt -c - -o "$remote" < "$config"
"$ADB" pull "$remote" "$out/trace.perfetto-trace" > /dev/null
adb_shell rm -f "$remote"

info "Trace: $out/trace.perfetto-trace (open at https://ui.perfetto.dev)"
if python3 -c 'import perfetto.trace_processor' > /dev/null 2>&1; then
    python3 "$TOOLS_DIR/analyze-trace.py" "$out/trace.perfetto-trace" --package "$PACKAGE" |
        tee "$out/summary.txt"
else
    info "For a text summary: pip install perfetto, then run analyze-trace.py on the trace"
fi
