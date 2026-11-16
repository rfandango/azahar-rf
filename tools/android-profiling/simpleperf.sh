#!/usr/bin/env bash
# Samples where CPU time goes in the running profiling build, using the NDK's simpleperf.
# Start the game first and get to the slow scene, then run this.
#
# Usage: simpleperf.sh [-d SECONDS] [-c fp|dwarf|none] [-f HZ] [-o DIR]
#   -d  recording duration (default 15)
#   -c  call graph mode (default fp; the profile build keeps frame pointers). "dwarf" unwinds
#       through code without frame pointers but costs far more; "none" records only leaf
#       functions.
#   -f  sampling frequency per thread (default 4000)
#   -o  output directory (default build/android-profiles/simpleperf-<timestamp>)
#
# Outputs: perf.data, report.html (interactive, with flame graphs), threads.txt (CPU time per
# thread), symbols.txt (hottest functions per thread) and emu-thread.txt (call graph of the
# emulation thread).
#
# Samples inside JIT-compiled code (the Dynarmic ARM JIT and the PICA shader JIT) have no
# symbols and appear as anonymous memory ("[anon:...]" / "//anon").

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

duration=15
callgraph=fp
freq=4000
while getopts "d:c:f:o:h" opt; do
    case "$opt" in
    d) duration="$OPTARG" ;;
    c) callgraph="$OPTARG" ;;
    f) freq="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    *)
        sed -n '2,13p' "$0"
        exit 0
        ;;
    esac
done

case "$callgraph" in
fp) cg_opts="--call-graph fp" ;;
dwarf) cg_opts="--call-graph dwarf" ;;
none) cg_opts="" ;;
*) die "unknown call graph mode: $callgraph" ;;
esac

command -v python3 > /dev/null || die "python3 is required for simpleperf's scripts"
ndk="$(find_ndk || true)"
[ -n "$ndk" ] && [ -d "$ndk/simpleperf" ] || die "NDK with simpleperf not found; set ANDROID_NDK_HOME"
sp="$ndk/simpleperf"
case "$(uname -s)" in
Linux) host_simpleperf="$sp/bin/linux/x86_64/simpleperf" ;;
Darwin) host_simpleperf="$sp/bin/darwin/x86_64/simpleperf" ;;
*) host_simpleperf="$sp/bin/windows/x86_64/simpleperf.exe" ;;
esac

pid="$(require_running_app)"
libs="$(find_unstripped_libs || true)"
[ -n "$libs" ] || warn "unstripped libraries not found (build with build-install.sh); symbols will be missing"

out="$(new_output_dir simpleperf)"
write_device_info "$out"
info "Recording $PACKAGE (pid $pid) for ${duration}s -> $out"

# cpu-clock samples on-CPU time in both user space and the kernel (driver ioctls, futexes, ...).
record_opts="-e cpu-clock -f $freq $cg_opts --duration $duration"
(
    cd "$out"
    # -nb: don't collect binaries now. Collecting with -lib would also push the ~0.5 GB
    # unstripped library to the device, which frame-pointer unwinding doesn't need.
    python3 "$sp/app_profiler.py" -p "$PACKAGE" -r "$record_opts" --ndk_path "$ndk" -nb \
        -o perf.data ||
        die "recording failed. On a device's first simpleperf run, simpleperf enables shell
profiling (security.perf_harden=0, reset on reboot), which can take a moment to apply: retry."
)

info "Collecting symbols"
(
    cd "$out"
    # Pulls the system libraries that were sampled from the device, and takes Azahar's
    # libraries from the unstripped build outputs (matched by build id).
    python3 "$sp/binary_cache_builder.py" -i perf.data --ndk_path "$ndk" \
        ${libs:+-lib "$libs"} > /dev/null 2>&1 || warn "binary_cache_builder.py failed"
)

info "Generating reports"
(
    cd "$out"
    report=("$host_simpleperf" report -i perf.data --symfs binary_cache)
    "${report[@]}" --sort comm --percent-limit 0.5 > threads.txt 2> /dev/null
    "${report[@]}" --sort comm,dso,symbol --percent-limit 0.2 > symbols.txt 2> /dev/null
    if [ -n "$cg_opts" ]; then
        "${report[@]}" --comms "$EMU_THREAD" --children --sort symbol -g caller \
            --percent-limit 1 > emu-thread.txt 2> /dev/null || true
    fi
    python3 "$sp/report_html.py" -i perf.data -o report.html --ndk_path "$ndk" --no_browser \
        > /dev/null 2>&1 || warn "report_html.py failed"
)

info "Done. CPU time per thread:"
sed -n '/^Overhead/,$p' "$out/threads.txt" | head -25 || true
echo
info "Reports in $out (open report.html in a browser)"
