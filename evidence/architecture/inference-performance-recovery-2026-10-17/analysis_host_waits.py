"""Read-only G09 host accounting of existing diagnostic captures; no execution.

Run from any directory. Writes only the sibling host-waits-analysis.json.
No measurements are selected for qualification and no clock model is repaired.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OBS = ROOT / ".scratch/inference-gaps/g09-completion-observation-25"
SHAPES = ["independent-cpu-first", "independent-gpu-first", "two-microbatches", "serial-holdout"]
RECORD = re.compile(r"# G09_RECORD tag=(\S+) ns=(\d+) kind=(\d+) run=(\d+) token=(\d+) value=(\d+) thread=(\d+)")
HEADER = re.compile(r"# G09_OBSERVATION tag=(\S+) count=(\d+) capacity=65536 overflow=0 clock=QPC-ns")


def identity(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summary(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), "minimum": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "maximum": max(values)}


def r_squared(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    xx, yy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return (sum((a - mx) * (b - my) for a, b in zip(x, y)) ** 2 / (xx * yy)
            if xx and yy else None)


def union(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        assert hi >= lo
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    return sum(hi - lo for lo, hi in merged)


def parse_source(path):
    blocks = []
    for line in path.read_text(encoding="utf8").splitlines():
        if line.startswith("# G09_OBSERVATION tag="):
            m = HEADER.fullmatch(line)
            assert m, line
            blocks.append({"tag": m[1], "declared_count": int(m[2]), "records": []})
        elif line.startswith("# G09_RECORD "):
            m = RECORD.fullmatch(line)
            assert m and blocks and m[1] == blocks[-1]["tag"], line
            blocks[-1]["records"].append(dict(zip(
                ["ns", "kind", "run", "token", "value", "thread"], map(int, m.groups()[1:]))))
    assert len(blocks) == 8
    rows, all_parks = [], []
    for bi, block in enumerate(blocks):
        records = block["records"]
        assert len(records) == block["declared_count"]
        records.sort(key=lambda v: v["ns"])
        entries = [v for v in records if v["kind"] == 1]
        exits = [v for v in records if v["kind"] == 3]
        assert len(entries) == len(exits) == 22
        for ordinal, (entry, exit_) in enumerate(zip(entries, exits)):
            assert entry["run"] == exit_["run"] and entry["thread"] == exit_["thread"]
            assert entry["ns"] < exit_["ns"]
            pending, intervals, polls = {}, defaultdict(list), []
            subset = [v for v in records if entry["ns"] <= v["ns"] <= exit_["ns"]]
            for v in subset:
                if v["kind"] in (11, 20, 30, 40):
                    key = (v["kind"], v["run"], v["token"])
                    assert key not in pending
                    pending[key] = v
                elif v["kind"] in (12, 21, 31, 41):
                    kind = v["kind"] - 1
                    begin = pending.pop((kind, v["run"], v["token"]))
                    intervals[kind].append((begin["ns"], v["ns"]))
                    if kind == 11:
                        polls.append({"start_qpc_ns": begin["ns"], "end_qpc_ns": v["ns"],
                                      "run": v["run"], "token": v["token"],
                                      "entry_thread": begin["thread"], "exit_thread": v["thread"],
                                      "result": v["value"], "duration_ns": v["ns"] - begin["ns"]})
                    elif kind in (20, 30):
                        all_parks.append({"block": bi, "ordinal": ordinal, "kind": kind,
                                          "duration_ns": v["ns"] - begin["ns"],
                                          "entry_thread": begin["thread"], "exit_thread": v["thread"]})
            assert not pending
            duration = exit_["ns"] - entry["ns"]
            park = union(intervals[20] + intervals[30])
            before = next(v for v in records if v["kind"] == 2 and v["ns"] >= entry["ns"])
            after = next(v for v in records if v["kind"] == 2 and v["ns"] >= exit_["ns"])
            assert before["run"] == after["run"] == entry["run"]
            rows.append({"block": bi, "tag": block["tag"], "shape": SHAPES[bi // 2],
                         "image": bi % 2, "ordinal": ordinal,
                         "phase": "measurement" if 9 <= ordinal <= 20 else "excluded-oracle-warmup",
                         "start_qpc_ns": entry["ns"], "end_qpc_ns": exit_["ns"],
                         "entry_marker_payload": before["value"], "exit_marker_payload": after["value"],
                         "replay_ns": duration, "park_union_ns": park,
                         "native_frontier_park_union_ns": union(intervals[30]),
                         "progress_park_union_ns": union(intervals[20]),
                         "finished_wait_union_ns": union(intervals[40]),
                         "park_count": len(intervals[20]) + len(intervals[30]),
                         "native_frontier_park_count": len(intervals[30]),
                         "progress_park_count": len(intervals[20]),
                         "poll_count": len(polls), "poll_total_ns": sum(v["duration_ns"] for v in polls),
                         "pending_poll_count": sum(v["result"] == 0 for v in polls),
                         "complete_poll_count": sum(v["result"] == 1 for v in polls),
                         "error_poll_count": sum(v["result"] == 2 for v in polls),
                         "outside_park_elapsed_ns": duration - park, "polls": polls})
    return rows, all_parks


def shape_analysis(rows, parks):
    out = []
    keys = ["replay_ns", "park_union_ns", "park_count", "poll_count", "poll_total_ns", "outside_park_elapsed_ns"]
    for shape in SHAPES:
        measured = [v for v in rows if v["shape"] == shape and v["phase"] == "measurement"]
        assert len(measured) == 24
        images = [[v for v in measured if v["image"] == i] for i in (0, 1)]
        chronological = sorted(measured, key=lambda v: v["start_qpc_ns"])
        groups = []
        for gi in range(6):
            group = chronological[4 * gi:4 * gi + 4]
            arms = [[v for v in group if v["image"] == i] for i in (0, 1)]
            assert all(len(a) == 2 for a in arms)
            order = "".join("AB"[v["image"]] for v in group)
            assert order in ("ABBA", "BAAB")
            groups.append({"original_four_call_group": gi, "order": order,
                           "image0_minus_image1_mean": {
                               k: statistics.mean(v[k] for v in arms[0]) - statistics.mean(v[k] for v in arms[1])
                               for k in keys}})
        index = SHAPES.index(shape)
        ps = [v["duration_ns"] for v in parks if v["block"] // 2 == index and 9 <= v["ordinal"] <= 20]
        deltas = [g["image0_minus_image1_mean"] for g in groups]
        polls = [p for v in measured for p in v["polls"]]
        out.append({"shape": shape,
                    "images": [{"image": i, **{k: summary([v[k] for v in arm]) for k in keys}}
                               for i, arm in enumerate(images)],
                    "all24_replay_r_squared": {
                        "park_count": r_squared([v["park_count"] for v in measured], [v["replay_ns"] for v in measured]),
                        "poll_count": r_squared([v["poll_count"] for v in measured], [v["replay_ns"] for v in measured]),
                        "park_union_ns": r_squared([v["park_union_ns"] for v in measured], [v["replay_ns"] for v in measured])},
                    "original_six_ABBA_group_differences": groups,
                    "six_group_difference_r_squared": {
                        "park_count": r_squared([v["park_count"] for v in deltas], [v["replay_ns"] for v in deltas]),
                        "park_union_ns": r_squared([v["park_union_ns"] for v in deltas], [v["replay_ns"] for v in deltas])},
                    "park_interval_ns": summary(ps), "requested_interval_ns": 1000000,
                    "poll_body_duration_ns_by_result": {
                        str(result): summary([p["duration_ns"] for p in polls if p["result"] == result])
                        for result in (0, 1, 2)},
                    "poll_bodies_above100us_descriptive_count": sum(p["duration_ns"] > 100000 for p in polls),
                    "poll_bodies_above1ms_descriptive_count": sum(p["duration_ns"] > 1000000 for p in polls),
                    "park_duration_floor_100us_histogram": dict(sorted(Counter(v // 100000 for v in ps).items())),
                    "exact_integer_ms_duration_count": sum(v % 1000000 == 0 for v in ps),
                    "clock_quantization_verdict": "No strict integer-ms duration lattice observed. Durations include useful native work/condition changes, timer wake latency and scheduling; this is not an OS timer-resolution verdict."})
    return out


def api_accounting(rows, path):
    c = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    tables = [v[0] for v in c.execute("select name from sqlite_master where type='table'")]
    nvtx = {}
    for v in c.execute("select n.start,n.uint64Value,coalesce(n.text,s.value) label from NVTX_EVENTS n left join StringIds s on s.id=n.textId where coalesce(n.text,s.value) like '%:clock-align'"):
        key = (v["label"].removesuffix(":clock-align"), v["uint64Value"])
        assert key not in nvtx
        nvtx[key] = v["start"]
    apis = [dict(v) for v in c.execute("select a.*,s.value name from VULKAN_API a join StringIds s on s.id=a.nameId")]
    summaries = []
    for bi in range(8):
        selected, per_replay = defaultdict(list), []
        for row in [v for v in rows if v["block"] == bi and v["phase"] == "measurement"]:
            lo = nvtx[(row["tag"], row["entry_marker_payload"])]
            hi = nvtx[(row["tag"], row["exit_marker_payload"])]
            calls = [v for v in apis if lo <= v["start"] and v["end"] <= hi]
            assert not [v for v in apis if v["name"] == "vkQueueSubmit" and v["start"] < hi and v["end"] > lo and v not in calls]
            submit_ns = sum(v["end"] - v["start"] for v in calls if v["name"] == "vkQueueSubmit")
            per_replay.append({"ordinal": row["ordinal"], "trace_bookends_ns": [lo, hi],
                               "queue_submit_count": sum(v["name"] == "vkQueueSubmit" for v in calls),
                               "queue_submit_api_total_ns": submit_ns,
                               "all_recorded_api_total_ns": sum(v["end"] - v["start"] for v in calls)})
            for v in calls:
                selected[v["name"]].append(v["end"] - v["start"])
        summaries.append({"block": bi, "shape": SHAPES[bi // 2], "image": bi % 2,
                          "api_duration_ns_by_name": {k: summary(v) for k, v in selected.items()},
                          "queue_submit_api_total_ns_per_replay": summary([v["queue_submit_api_total_ns"] for v in per_replay]),
                          "all_recorded_api_total_ns_per_replay": summary([v["all_recorded_api_total_ns"] for v in per_replay]),
                          "replays": per_replay})
    c.close()
    return {"existing_tables": tables, "blocks": summaries,
            "limits": ["API durations use trace clock and source-identified NVTX bookends only; no QPC/trace offset is fitted.",
                       "API host-call duration is not GPU queue residence or execution time.",
                       "Absent API names are missing tracing coverage, not proof the calls did not occur.",
                       "No scheduler context-switch/running-state, sampled GPU power/clock or external-process GPU timelines are present."]}


def main():
    individual = OBS / "run-individual-1"
    direct = ROOT / ".scratch/inference-gaps/g09-av-debug-1/run"
    inputs = [individual / "embedded-stderr.log", individual / "trace.sqlite", individual / "source-analysis.json",
              individual / "gpu-attribution.json", direct / "normalized-markers.log", direct / "profile.log",
              ROOT / "evidence/architecture/inference-g09-completion-observation/individual-raw.zip",
              ROOT / "evidence/architecture/inference-g09-completion-observation/individual-source.zip",
              ROOT / "evidence/architecture/inference-g09-completion-observation/direct-debug-raw.zip",
              ROOT / "evidence/architecture/inference-g09-completion-observation/individual-receipt.json",
              ROOT / "evidence/architecture/inference-g09-completion-observation/direct-debug-receipt.json",
              OBS / "source/invocation.cc", OBS / "source/facade.cc", OBS / "source/recorder.h"]
    data = {}
    for name, source in [("individual", individual / "embedded-stderr.log"),
                         ("direct_debug_without_nsight", direct / "normalized-markers.log")]:
        rows, parks = parse_source(source)
        data[name] = {"source": source.relative_to(ROOT).as_posix(), "source_records": sum(1 for l in source.read_text(encoding="utf8").splitlines() if l.startswith("# G09_RECORD ")),
                      "replays": [{k: v for k, v in row.items() if k != "polls"} for row in rows],
                      "park_intervals": parks, "shape_analysis": shape_analysis(rows, parks),
                      "measurement_poll_bodies_above100us": [
                          {"block": row["block"], "ordinal": row["ordinal"], **poll}
                          for row in rows if row["phase"] == "measurement"
                          for poll in row["polls"] if poll["duration_ns"] > 100000]}
        if name == "individual":
            data[name]["api_accounting"] = api_accounting(rows, individual / "trace.sqlite")
    result = {
        "schema": "hyle.g09.offline-host-waits-accounting/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "product_candidate": "3524413272cb52d06af5e64d634edbfef09b9864",
        "analyzer": identity(Path(__file__)), "inputs": [identity(p) for p in inputs],
        "actions": {"native_gpu_remote_benchmark_runs": 0, "product_registry_changes": 0,
                    "sqlite_access": "mode=ro", "writes": ["this analyzer and sibling result only"]},
        "method": ["Independently parse retained source records, exact eight nonoverflow blocks and22replays per block.",
                   "Select original ordinals9..20 only; preserve before/after oracles and8warmups as excluded rows.",
                   "Pair enter/leave by kind/run/token. Union overlapping park intervals; never add caller finished waits as independent overhead.",
                   "Report original chronological ABBA/BAAB four-call groups, six per shape; differences are group means of image0 minus image1.",
                   "R-squared is descriptive one-variable linear association with intercept, no causal verdict, significance claim or formal statistic replacement."],
        "datasets": data,
        "conclusions": [
            "In both existing runs the independent-shape image differences closely track park-union differences; count alone explains less. This localizes much observed variability to waiting intervals, not to slow poll-call bodies or recorded QueueSubmit host calls.",
            "The same extended and variable parks occur in the direct ProcDump run without Nsight, so Nsight injection is not necessary for that phenomenon. These two different diagnostic conditions are not a controlled performance comparison.",
            "Two-microbatch runs contain additional long complete-poll bodies (up to millisecond scale). Do not attribute their total difference to frontier park: this shape uses progress park and overlapping slots.",
            "Park duration includes useful native work and host scheduling. Current data does not separate GPU completion time, thread readiness, timer wakeup and thread dispatch; it cannot certify removable delay.",
            "Original native readiness remains UNKNOWN, original A/A causes remain unresolved, valid formal A/B remains0. No product fix or new run is authorized by this offline result."],
        "minimal_future_diagnostic_fields": [
            "Keep original timed-poll control and mark treatment explicitly; preserve shape, image, original ABBA group, inputs, ownership, scalar observation, retirement and drain.",
            "At wait choice record run/slot/stage/completion identity, pending count, ready-native/ready-CPU work, and why blocking cannot strand independent runnable work.",
            "Record same-QPC entry/exit, requested deadline, native wait result/error/cancellation, and retired-versus-only-complete status. A successful Completion.wait includes its retirement tails, not a bare GPU end timestamp.",
            "For control wait record timeout-versus-notified result and before/after predicate; current kind30/31 records omit that distinction.",
            "Record polling count and body intervals, actual worker thread/CPU at each boundary, runtime-mutex entry/acquire boundaries around the known long-poll path, and accepted upload/download/submit boundaries.",
            "If independently distinguishing GPU readiness from ready-thread dispatch is required, capture an attributable completion signal and scheduling evidence in a separately reviewed finite diagnostic; do not infer either from park-return timestamps."]}
    target = HERE / "host-waits-analysis.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output": target.relative_to(ROOT).as_posix(),
                      "datasets": {k: {"source_records": v["source_records"], "replays": len(v["replays"])} for k, v in data.items()},
                      "valid_formal_ab": 0}))


if __name__ == "__main__":
    main()
