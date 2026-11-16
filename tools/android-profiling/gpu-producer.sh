#!/usr/bin/env bash
# Manages the GPU counter producer from Android GPU Inspector (AGI), which publishes the GPU
# driver's hardware counters to Perfetto as the "gpu.counters" data source.
#
# Usage: gpu-producer.sh start|stop|list|config
#   start   launch AGI's producer on the device unless gpu.counters is already available
#   stop    stop it
#   list    print the counters the device offers ("*" = selected by default)
#   config  print Perfetto data sources for GPU counters and render stages
#
# The producer launcher (/data/local/tmp/agi_launch_producer) is pushed to the device by the
# AGI desktop app, together with AGI's APK: open AGI once with the device connected, or run a
# system profile from it, to set this up.
#
# By default, "config" selects the counters the driver marks as defaults plus a few useful
# extras. Override with AZAHAR_GPU_COUNTERS=id,id,... (ids from "list").

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LAUNCHER=/data/local/tmp/agi_launch_producer
COUNTER_PERIOD_NS=1000000

has_counters() { adb_shell perfetto --query 2> /dev/null | grep -q '^gpu\.counters'; }

# Prints "<id>\t<*| >\t<name>" for every counter in the gpu.counters descriptor.
list_counters() {
    "$ADB" exec-out perfetto --query-raw | python3 -c '
import sys

def varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7

def fields(buf):
    i = 0
    while i < len(buf):
        key, i = varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, i = varint(buf, i)
        elif wire == 1:
            val, i = buf[i:i + 8], i + 8
        elif wire == 2:
            size, i = varint(buf, i)
            val, i = buf[i:i + size], i + size
        elif wire == 5:
            val, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(wire)
        yield num, val

def first(buf, num):
    return next((v for n, v in fields(buf) if n == num), None)

# TracingServiceState.data_sources(2) -> ds_descriptor(1) -> name(1), gpu_counter_descriptor(5)
#   GpuCounterDescriptor.specs(1) -> counter_id(1), name(2), select_by_default(9)
state = sys.stdin.buffer.read()
for num, ds in fields(state):
    desc = first(ds, 1) if num == 2 else None
    if desc is None or first(desc, 1) != b"gpu.counters":
        continue
    counters = first(desc, 5)
    for n, spec in fields(counters or b""):
        if n != 1:
            continue
        mark = "*" if first(spec, 9) else " "
        print(f"{first(spec, 1)}\t{mark}\t{first(spec, 2).decode()}")
    break
'
}

case "${1:-}" in
start)
    require_device
    if has_counters; then
        info "gpu.counters is available"
        exit 0
    fi
    adb_shell "[ -x $LAUNCHER ]" ||
        die "$LAUNCHER not found; set up the device once with the AGI desktop app"
    info "Starting AGI's GPU counter producer"
    adb_shell "cd /data/local/tmp && (nohup $LAUNCHER > agi_launch_producer.log 2>&1 &)"
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        has_counters && {
            info "gpu.counters is available"
            exit 0
        }
        sleep 0.5
    done
    adb_shell cat /data/local/tmp/agi_launch_producer.log >&2 || true
    die "the producer did not register gpu.counters (unsupported GPU driver?)"
    ;;
stop)
    require_device
    adb_shell pkill -f agi_launch_producer || true
    ;;
list)
    require_device
    list_counters
    ;;
config)
    require_device
    if [ -n "${AZAHAR_GPU_COUNTERS:-}" ]; then
        ids="${AZAHAR_GPU_COUNTERS//,/ }"
    else
        # Driver defaults plus a few counters that help tell bottlenecks apart.
        ids="$(list_counters | awk -F '\t' '
            $2 == "*" || $3 ~ /Bus Busy|GPU Frequency|Texture Pipes Busy|Shaders Stalled/ {
                print $1
            }' | tr '\n' ' ')"
    fi
    echo
    echo "# GPU hardware counters (via AGI's producer)"
    echo "data_sources {"
    echo "  config {"
    echo "    name: \"gpu.counters\""
    echo "    gpu_counter_config {"
    echo "      counter_period_ns: $COUNTER_PERIOD_NS"
    for id in $ids; do
        echo "      counter_ids: $id"
    done
    echo "    }"
    echo "  }"
    echo "}"
    echo
    echo "# GPU render stages from the driver (needs debug.graphics.gpu.profiler.perfetto=1"
    echo "# before the app starts)"
    echo "data_sources {"
    echo "  config {"
    echo "    name: \"gpu.renderstages\""
    echo "  }"
    echo "}"
    ;;
*)
    sed -n '2,17p' "$0"
    exit 1
    ;;
esac
