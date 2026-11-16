#!/usr/bin/env python3
# Copyright 2026 Citra Emulator Project / Azahar Emulator Project
# Licensed under GPLv2 or any later version
# Refer to the misc/licenses/gplv2.txt file included.

"""Summarizes a Perfetto trace of Azahar recorded with perfetto.sh.

Prints frame pacing, where the emulation thread spends its time (running, waiting for a
CPU, blocked), CPU time per thread and core cluster, the heaviest trace marker sections,
Azahar's per-frame counters, and CPU/GPU clocks and temperatures.

Requires the Perfetto Python package (pip install perfetto), which downloads the
trace_processor binary on first use; pass --shell to use a local one instead.
"""

import argparse
import statistics
import sys

try:
    from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
except ImportError:
    sys.exit("The perfetto package is required: pip install perfetto")

EMU_THREAD = "NativeEmulation"
FRAME_SLICE = "EmuFrame"
VBLANK_MS = 1000.0 / 59.8261  # 3DS refresh rate

# Readable names for Linux thread states as reported by trace_processor.
STATE_NAMES = {
    "Running": "running",
    "R": "runnable (waiting for a CPU)",
    "R+": "runnable (preempted)",
    "S": "sleeping (waiting on a lock, condition or I/O)",
    "D": "uninterruptible (kernel / driver wait)",
    "DK": "uninterruptible (kernel / driver wait)",
}


def rows(tp, sql):
    return list(tp.query(sql))


def ms(ns):
    return ns / 1e6


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def section(title):
    print()
    print(title)
    print("-" * len(title))


def find_process(tp, package):
    found = rows(tp, f"""
        select upid, pid, name from process
        where name = '{package}' or name like '{package}:%'
        order by pid desc""")
    return found[0] if found else None


def report_frames(tp, upid, trace_ns):
    section("Frame pacing")
    frames = [r.dur for r in rows(tp, f"""
        select s.dur from slice s join process_track pt on s.track_id = pt.id
        where pt.upid = {upid} and s.name = '{FRAME_SLICE}' and s.dur > 0""")]
    if frames:
        work = [ms(d) for d in frames]
        over = sum(1 for w in work if w > VBLANK_MS)
        print(f"Emulated VBlank periods: {len(work)} ({len(work) / (trace_ns / 1e9):.1f}/s, "
              f"target {1000.0 / VBLANK_MS:.1f}/s)")
        print(f"Emulation work per VBlank (ms; excludes frame limiting and GPU throttling): "
              f"mean {statistics.mean(work):.2f}  p50 {percentile(work, 50):.2f}  "
              f"p90 {percentile(work, 90):.2f}  p99 {percentile(work, 99):.2f}  "
              f"max {max(work):.2f}")
        print(f"VBlanks whose work exceeded the {VBLANK_MS:.2f} ms budget: {over} "
              f"({100.0 * over / len(work):.1f}%)")
    else:
        print(f"No '{FRAME_SLICE}' slices: app tracing was off (debug.azahar.trace=0, "
              f"not a profiling build, or the app was not running).")

    game = rows(tp, f"""
        select min(c.value) as first, max(c.value) as last,
               min(c.ts) as t0, max(c.ts) as t1
        from counter c join process_counter_track t on c.track_id = t.id
        where t.upid = {upid} and t.name = 'azahar.game_frame'""")
    if game and game[0].last is not None and game[0].t1 > game[0].t0:
        g = game[0]
        fps = (g.last - g.first) / ((g.t1 - g.t0) / 1e9)
        print(f"Guest (game) frame rate: {fps:.1f} fps")


def report_emu_thread(tp, upid):
    section(f"Emulation thread ({EMU_THREAD}) state")
    # Earlier emulation sessions in the same process leave threads with the same name behind;
    # pick the one that actually ran.
    threads = rows(tp, f"""
        select t.utid, t.tid, coalesce(sum(s.dur), 0) as run
        from thread t left join sched s using (utid)
        where t.upid = {upid} and t.name = '{EMU_THREAD}'
        group by t.utid order by run desc""")
    if not threads:
        print("Emulation thread not found in the trace.")
        return None
    utid = threads[0].utid
    states = rows(tp, f"""
        select state, sum(dur) as dur from thread_state
        where utid = {utid} group by state order by dur desc""")
    total = sum(r.dur for r in states) or 1
    for r in states:
        name = STATE_NAMES.get(r.state, r.state)
        print(f"  {100.0 * r.dur / total:5.1f}%  {ms(r.dur):9.1f} ms  {name}")

    blocked = rows(tp, f"""
        select blocked_function as fn, sum(dur) as dur from thread_state
        where utid = {utid} and state in ('D', 'DK') and blocked_function is not null
        group by fn order by dur desc limit 5""")
    for r in blocked:
        print(f"      blocked in {r.fn}: {ms(r.dur):.1f} ms")
    return utid


def cpu_clusters(tp):
    """Maps cpu -> cluster label, using linux.system_info data when present."""
    try:
        cpus = rows(tp, "select cpu, cluster_id from cpu where cluster_id is not null")
    except Exception:
        cpus = []
    if cpus:
        return {r.cpu: f"cluster{r.cluster_id}" for r in cpus}
    # Fall back to grouping by the highest frequency seen for each CPU.
    freqs = rows(tp, """
        select t.cpu as cpu, max(c.value) as fmax from counter c
        join cpu_counter_track t on c.track_id = t.id
        where t.name = 'cpufreq' group by t.cpu""")
    return {r.cpu: f"{int(r.fmax) // 1000}MHz-max" for r in freqs}


def report_threads(tp, upid, trace_ns):
    section("CPU time per thread (app process)")
    clusters = cpu_clusters(tp)
    data = rows(tp, f"""
        select t.utid, t.tid, t.name, s.cpu, sum(s.dur) as dur
        from sched s join thread t using (utid)
        where t.upid = {upid} group by t.utid, s.cpu""")
    per_thread = {}
    for r in data:
        entry = per_thread.setdefault(r.utid, {"name": r.name or f"tid {r.tid}", "total": 0,
                                               "clusters": {}})
        entry["total"] += r.dur
        label = clusters.get(r.cpu, f"cpu{r.cpu}")
        entry["clusters"][label] = entry["clusters"].get(label, 0) + r.dur
    ranked = sorted(per_thread.values(), key=lambda e: e["total"], reverse=True)[:15]
    print("  core%   = share of one core over the trace;  cluster split in parentheses")
    for e in ranked:
        split = ", ".join(f"{k} {100.0 * v / e['total']:.0f}%"
                          for k, v in sorted(e["clusters"].items()))
        print(f"  {100.0 * e['total'] / trace_ns:6.1f}%  {e['name']:<20} ({split})")


def report_sections(tp, upid):
    section("Heaviest trace marker sections (all app threads)")
    data = rows(tp, f"""
        select t.name as thread, s.name as name, count(*) as cnt,
               sum(s.dur) as total, max(s.dur) as longest
        from slice s join thread_track tt on s.track_id = tt.id join thread t using (utid)
        where t.upid = {upid} and s.dur > 0
        group by t.name, s.name order by total desc limit 30""")
    if not data:
        print("No trace marker sections found.")
        return
    print(f"  {'total ms':>9} {'count':>7} {'max ms':>8}  thread / section")
    for r in data:
        print(f"  {ms(r.total):9.1f} {r.cnt:7d} {ms(r.longest):8.2f}  "
              f"{r.thread} / {r.name}")


def report_counters(tp, upid):
    section("Azahar counters (per emulated VBlank)")
    data = rows(tp, f"""
        select t.name as name, avg(c.value) as mean, max(c.value) as peak,
               sum(c.value) as total
        from counter c join process_counter_track t on c.track_id = t.id
        where t.upid = {upid} and t.name like 'azahar.%' and t.name != 'azahar.game_frame'
        group by t.name order by t.name""")
    if not data:
        print("No counters found.")
        return
    print(f"  {'mean':>9} {'max':>8} {'total':>9}  counter")
    for r in data:
        print(f"  {r.mean:9.1f} {r.peak:8.0f} {r.total:9.0f}  {r.name}")


def time_weighted(tp, track_table, track_filter="1"):
    """Time-weighted mean, min and max of every counter track in track_table."""
    return rows(tp, f"""
        with c as (
            select t.id as track_id, t.name as name, c.ts as ts, c.value as value,
                   lead(c.ts) over (partition by c.track_id order by c.ts) as next_ts
            from counter c join {track_table} t on c.track_id = t.id
            where {track_filter})
        select track_id, name,
               sum(value * (coalesce(next_ts, (select end_ts from trace_bounds)) - ts)) /
               max(sum(coalesce(next_ts, (select end_ts from trace_bounds)) - ts), 1) as mean,
               min(value) as low, max(value) as peak
        from c group by track_id order by name""")


def format_value(name, value):
    lowered = name.lower()
    if "temp" in lowered:
        return f"{value / 1000:.1f} C" if abs(value) >= 1000 else f"{value:.1f} C"
    if "freq" in lowered or "clocks" in lowered:
        # Sources report Hz or kHz; anything this large is Hz.
        return f"{value / 1e6:.0f} MHz" if value >= 1e7 else f"{value / 1e3:.0f} MHz"
    if "bytes" in lowered:
        return f"{value / 1e6:.0f} MB/s"
    if "/ second" in lowered or "/second" in lowered or "/ sec" in lowered:
        return f"{value / 1e6:.2f} M/s"
    return f"{value:.1f}"


def print_counter_rows(tp, data):
    """Prints mean, p95 and max per track. Sampled rate counters spike in single samples, so
    p95 is the more useful upper bound."""
    valid = [r for r in data if r.peak is not None and abs(r.peak) < 1e18]  # Drop invalid
    values = {}
    if valid:
        ids = ",".join(str(r.track_id) for r in valid)
        for v in rows(tp, f"select track_id, value from counter where track_id in ({ids})"):
            values.setdefault(v.track_id, []).append(v.value)
    for r in valid:
        p95 = percentile(values.get(r.track_id, []), 95)
        print(f"  {r.name:<44} mean {format_value(r.name, r.mean):>11}   "
              f"p95 {format_value(r.name, p95):>11}   max {format_value(r.name, r.peak):>11}")


def report_clocks(tp):
    section("CPU clocks (time-weighted)")
    cpu_tracks = rows(tp, "select id, cpu from cpu_counter_track where name = 'cpufreq'")
    cpu_of = {r.id: r.cpu for r in cpu_tracks}
    for r in time_weighted(tp, "cpu_counter_track", "t.name = 'cpufreq'"):
        cpu = cpu_of.get(r.track_id, "?")
        print(f"  cpu{cpu}: mean {r.mean / 1000:5.0f} MHz   min {r.low / 1000:5.0f} MHz   "
              f"max {r.peak / 1000:5.0f} MHz")

    section("GPU counters and clock (mean is time-weighted)")
    data = time_weighted(tp, "gpu_counter_track")
    if data:
        print_counter_rows(tp, data)
    else:
        print("  none recorded (use perfetto.sh -g for hardware counters)")

    section("Device clocks and temperatures (mean is time-weighted)")
    # GPU devfreq, CPU L3 clocks and thermal zones; other devfreq nodes are bus votes in
    # driver-specific units.
    data = time_weighted(tp, "counter_track", """
        t.id not in (select id from gpu_counter_track)
        and t.id not in (select id from cpu_counter_track)
        and t.id not in (select id from process_counter_track)
        and (t.name like '%kgsl%' or t.name like '%gpu%' or t.name like '%-l3-%'
             or t.name like '%temp%' or t.name like '%Temp%')
        and t.name not like '%gpubw%' and t.name not like '%busmon%'""")
    if data:
        print_counter_rows(tp, data)
    else:
        print("  none recorded")


def report_render_stages(tp, upid, trace_ns):
    section("GPU render stages (driver timeline, per emulated VBlank)")
    try:
        data = rows(tp, f"""
            select name, count(*) as cnt, sum(dur) as total, max(dur) as longest
            from gpu_slice where upid = {upid} or upid is null
            group by name order by total desc limit 12""")
    except Exception:
        data = []
    if not data:
        print("  none recorded (perfetto.sh -g, with the app started after it enabled them)")
        return
    vblanks = rows(tp, f"select count(*) as n from slice where name = '{FRAME_SLICE}'")[0].n
    vblanks = vblanks or 1
    print(f"  {'ms/VBlank':>9} {'count/VBlank':>12} {'max ms':>7}  stage   (stages can overlap)")
    for r in data:
        print(f"  {ms(r.total) / vblanks:9.2f} {r.cnt / vblanks:12.1f} {ms(r.longest):7.2f}  "
              f"{r.name}")
    passes = rows(tp, f"""
        select render_target, count(*) as cnt from gpu_slice
        where (upid = {upid} or upid is null) and render_target is not null and depth = 0
        group by render_target order by cnt desc limit 1""")
    if passes:
        print(f"  Busiest render target: {passes[0].cnt / vblanks:.1f} stages per VBlank "
              f"(many stages per target means render passes are split up)")


def report_jank(tp, upid):
    section("SurfaceFlinger frame timeline (app layer)")
    try:
        data = rows(tp, f"""
            select jank_type, count(*) as cnt from actual_frame_timeline_slice
            where upid = {upid} group by jank_type order by cnt desc""")
    except Exception:
        data = []
    if not data:
        print("  no frame timeline data")
    for r in data:
        print(f"  {r.cnt:6d}  {r.jank_type}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--package", default="org.azahar_emu.azahar.profile")
    parser.add_argument("--shell", help="path to a local trace_processor_shell binary")
    args = parser.parse_args()

    config = TraceProcessorConfig(bin_path=args.shell) if args.shell else None
    tp = TraceProcessor(trace=args.trace, config=config) if config else TraceProcessor(
        trace=args.trace)

    bounds = rows(tp, "select start_ts, end_ts from trace_bounds")[0]
    trace_ns = max(bounds.end_ts - bounds.start_ts, 1)
    print(f"Trace: {args.trace}  ({trace_ns / 1e9:.1f} s)")

    proc = find_process(tp, args.package)
    if proc is None:
        print(f"Process {args.package} not found in the trace.")
        report_clocks(tp)
        return
    print(f"Process: {proc.name} (pid {proc.pid})")

    report_frames(tp, proc.upid, trace_ns)
    report_emu_thread(tp, proc.upid)
    report_threads(tp, proc.upid, trace_ns)
    report_sections(tp, proc.upid)
    report_counters(tp, proc.upid)
    report_render_stages(tp, proc.upid, trace_ns)
    report_clocks(tp)
    report_jank(tp, proc.upid)
    tp.close()


if __name__ == "__main__":
    main()
