#!/usr/bin/env python3
"""Compare two single-request Nsight exports at their real prefill endpoints.

Each profile directory must contain trace.sqlite, target-stdout.log and
target-stderr.log from an instrumented matched consumer. Hyle must emit
hyle.prefill.ledger_endpoint.observed and hyle.generation.prepare; the peer
must emit peer.prefill.endpoint.observed. This tool refuses older ambiguous
ranges instead of guessing where prefill ended. It does not run a benchmark.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import sys


@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    outer_end: int


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prefill_interval(events, arm):
    """Select the first-token publication observation, never the later outer end."""
    require(arm in ("hyle", "peer"), "Unknown profile arm")
    whole = [(a, b) for a, b, name in events if name == arm + ".prefill" and b is not None]
    require(len(whole) == 1, f"{arm}: expected one completed prefill range")
    start, outer_end = whole[0]
    require(start < outer_end, f"{arm}: invalid prefill range")
    marker = ("hyle.prefill.ledger_endpoint.observed" if arm == "hyle"
              else "peer.prefill.endpoint.observed")
    ends = [a for a, _, name in events if name == marker and start <= a <= outer_end]
    require(len(ends) == 1, f"{arm}: real prefill endpoint missing or ambiguous")
    end = ends[0]
    require(start < end, f"{arm}: invalid prefill endpoint")
    generation = [a for a, _, name in events
                  if name == arm + ".generation.prepare" and start <= a <= outer_end]
    require(not any(a < end for a in generation), f"{arm}: decode preparation precedes prefill endpoint")
    # New captures explicitly stop at the ledger. For the earlier corrected
    # capture, accept a later outer end only if its decode marker proves which
    # side of the ledger the next stage occupied.
    if arm == "hyle" and end < outer_end:
        closed = [a for a, _, name in events if name == "hyle.prefill.capture_closed_at_ledger"
                  and end <= a <= outer_end]
        require(len(closed) == 1 or any(a >= end for a in generation),
                "hyle: prefill capture closure or decode boundary was not observed")
    return Interval(start, end, outer_end)


def classify(name):
    name = name.lower()
    if any(x in name for x in ("flash_attn", "batchprefillwith", "mergestateskernel", "merge_states")):
        return "attention"
    if any(x in name for x in ("gemm", "gemvx", "mul_mat_vec", "mul_mat_f", "splitkreduce")):
        return "gemm"
    if "gated_delta_net" in name or "ssm_conv" in name:
        return "gdn_conv"
    # Without graph-operation provenance a generic conversion cannot safely be
    # charged to GDN, attention, or copies; expose it as its own category.
    if "convert_unary" in name:
        return "conversion"
    if any(x in name for x in ("concat", "cpy_", "copy_rows", "gather", "k_get_rows", "k_set_rows",
                                 "append_rows", "append_choice_rows", "append_device_lengths", "::publish(")):
        return "copy"
    return "other"


def workload(profile, arm):
    raw = json.loads((profile / "target-stdout.log").read_text(encoding="utf-8-sig"))
    plan = (profile / "plan.txt").read_text(encoding="utf-8-sig")
    versions = re.findall(r"^version=(.+)$", plan, re.MULTILINE)
    require(len(versions) == 1 and versions[0].strip(), f"{arm}: profile source version missing or ambiguous")
    result = raw["sessions"][0]["result"] if arm == "hyle" else raw
    require(len(result["turns"]) == 1, f"{arm}: only single-turn prefill is supported")
    turn = result["turns"][0]
    ids = turn["input_ids"]
    require(ids and turn["context_before"] == 0, f"{arm}: require an empty starting context and actual input")
    require(len(result["generated_ids"]) >= 2 and result["decode_seconds"] > 0,
            f"{arm}: complete post-prefill generation was not observed")
    if arm == "hyle":
        stderr = (profile / "target-stderr.log").read_text(encoding="utf-8-sig")
        caps = re.findall(r"\bcold_plan\b[^\n]*\bconfigured_capacity=(\d+)", stderr)
        require(len(caps) == 1, "hyle: configured capacity missing or ambiguous")
        capacity = int(caps[0])
        require(raw.get("startup_prefill_warmup_calls", 0) > 0 and raw.get("startup_prefill_warmup_retired") is True,
                "hyle: actual startup prefill warmup is missing")
        require(raw.get("future_shape_hints") == 0 and raw.get("cold_planning") == "configured_capacity",
                "hyle: the cold plan must not use future prompt shapes")
        prefill_chunk = raw.get("configured_prefill_rows")
    else:
        capacity = raw.get("target_effective_context")
        require(isinstance(capacity, int) and capacity > 0, "peer: effective target n_ctx missing")
        require(raw.get("requested_context") == capacity, "peer: effective target n_ctx differs from request")
        warmup = raw.get("startup_warmup") or {}
        require(raw.get("startup_warmup_seconds", 0) > 0 and warmup.get("enabled") is True
                and warmup.get("target_empty_pos_max") == -1,
                "peer: completed startup warmup or empty target state is missing")
        if raw["mtp"]:
            require(warmup.get("draft_empty_pos_max") == -1, "peer: draft state after warmup is not empty")
            require(raw.get("draft_effective_context") == capacity, "peer: effective draft n_ctx differs")
        require(raw.get("requested_batch_capacity") == raw.get("effective_batch_capacity"),
                "peer: effective batch capacity differs")
        prefill_chunk = raw.get("effective_microbatch_capacity")
        require(prefill_chunk == raw.get("requested_microbatch_capacity"),
                "peer: effective microbatch capacity differs")
    require(isinstance(prefill_chunk, int) and prefill_chunk > 0, f"{arm}: prefill chunk capacity missing")
    return {"input_ids": ids, "capacity": capacity, "mtp": bool(raw["mtp"]),
            "quota": turn["requested_steps"], "output_ids": result["generated_ids"],
            "prefill_chunk": prefill_chunk, "version": versions[0].strip()}


def match_workloads(hyle, peer):
    for key in ("input_ids", "capacity", "mtp", "quota"):
        require(hyle[key] == peer[key], f"Workload mismatch: {key}")
    if "prefill_chunk" in hyle and "prefill_chunk" in peer:
        chunk_h, chunk_p = hyle["prefill_chunk"], peer["prefill_chunk"]
        require(chunk_h == chunk_p or len(hyle["input_ids"]) <= min(chunk_h, chunk_p),
                "Workload mismatch: prefill chunk capacity")


def activity(db, interval):
    tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
    kernel = ("CUPTI_ACTIVITY_KIND_KERNEL" if "CUPTI_ACTIVITY_KIND_KERNEL" in tables
              else "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
    require(kernel in tables and "StringIds" in tables, "Missing CUDA kernel trace")
    pids = list(db.execute(f"select distinct globalPid,deviceId from {kernel} "
                           "where start>=? and end<=?", (interval.start, interval.end)))
    require(len(pids) == 1, "Prefill must contain CUDA activity from exactly one process and one device")
    pid, device = pids[0]
    require(pid is not None and device is not None, "CUDA process or device identity missing")
    sums, counts, intervals = defaultdict(int), defaultdict(int), []
    first_kernel = None
    kernel_count = 0
    for table in (kernel, "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table not in tables:
            continue
        crossing = db.execute(f"select count(*) from {table} where globalPid=? and deviceId=? and "
                              "((start<? and end>?) or (start<? and end>?))",
                              (pid, device, interval.start, interval.start, interval.end, interval.end)).fetchone()[0]
        require(crossing == 0, f"{table}: CUDA activity crosses a prefill boundary")
        if table == kernel:
            query = (f"select k.start,k.end,s.value from {table} k join StringIds s on k.demangledName=s.id "
                     "where k.globalPid=? and k.deviceId=? and k.start>=? and k.end<=?")
        else:
            query = (f"select start,end,'' from {table} where globalPid=? and deviceId=? "
                     "and start>=? and end<=?")
        for a, b, name in db.execute(query, (pid, device, interval.start, interval.end)):
            require(a <= b, "Invalid CUDA activity timestamps")
            category = classify(name) if table == kernel else "copy"
            sums[category] += b - a
            counts[category] += 1
            intervals.append((a, b))
            if table == kernel:
                kernel_count += 1
                first_kernel = a if first_kernel is None else min(first_kernel, a)
    require(first_kernel is not None, "No prefill kernels")
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    busy = sum(end - begin for begin, end in merged)
    return {"window": interval.end - interval.start, "sums": sums, "counts": counts,
            "idle": interval.end - interval.start - busy, "first_kernel": first_kernel - interval.start,
            "kernels": kernel_count, "pid": pid, "busy_intervals": merged}


def host_preparation(events, interval, busy):
    """Intersect complete host preparation ranges with this process's CUDA activity."""
    names = ("hyle.host.cold_batch", "hyle.host.layout",
             "hyle.host.batch_prepare", "hyle.host.replay_prepare")
    for start, end, name in events:
        if name not in names:
            continue
        if end is None:
            require(not interval.start <= start <= interval.end,
                    "hyle: host preparation range has no end")
        elif start < interval.end and end > interval.start:
            require(interval.start <= start <= end <= interval.end,
                    "hyle: host preparation crosses a prefill boundary")
    spans = {name: sorted((a, b) for a, b, event in events
                          if event == name and b is not None and interval.start <= a <= b <= interval.end)
             for name in names}
    if not any(spans.values()):
        return None  # Profiles collected before host ranges were added.
    count = len(spans[names[0]])
    require(count and all(len(spans[name]) == count for name in names),
            "hyle: host preparation ranges are incomplete")
    for i in range(count):
        cold, layout, batch, replay = (spans[name][i] for name in names)
        require(cold[0] <= layout[0] <= layout[1] <= batch[0] <= batch[1] <=
                replay[0] <= replay[1] <= cold[1], "hyle: host preparation stages are not nested")
        if i:
            require(spans[names[0]][i - 1][1] <= cold[0],
                    "hyle: overlapping host preparations need a different attribution")
    ends = [end for _, end in busy]
    result = {}
    for name in names:
        elapsed = overlap = 0
        for start, end in spans[name]:
            elapsed += end - start
            index = bisect_right(ends, start)
            while index < len(busy) and busy[index][0] < end:
                overlap += max(0, min(end, busy[index][1]) - max(start, busy[index][0]))
                index += 1
        require(overlap <= elapsed, "hyle: host/CUDA overlap exceeds host preparation time")
        result[name] = {"count": count, "elapsed": elapsed, "busy": overlap, "idle": elapsed - overlap}
    return result


def load_profile(profile, arm):
    profile = profile.resolve()
    info = workload(profile, arm)
    db = sqlite3.connect((profile / "trace.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        events = list(db.execute("select n.start,n.end,coalesce(n.text,s.value) "
                                 "from NVTX_EVENTS n left join StringIds s on n.textId=s.id"))
        interval = prefill_interval(events, arm)
        sample = activity(db, interval)
        if arm == "hyle":
            sample["host_preparation"] = host_preparation(events, interval, sample["busy_intervals"])
    finally:
        db.close()
    return info, interval, sample


def report(h, p):
    hwork, hi, hs = h
    pwork, pi, ps = p
    match_workloads(hwork, pwork)
    rows = (("GEMM", "gemm"), ("注意力", "attention"), ("GDN / conv（直接命名）", "gdn_conv"),
            ("拷贝 / gather", "copy"), ("通用类型转换", "conversion"), ("其它 kernel", "other"))
    lines = [f"输入 {len(hwork['input_ids'])} token，capacity {hwork['capacity']}，"
             f"MTP {'on' if hwork['mtp'] else 'off'}，生成配额 {hwork['quota']}。",
             f"Hyle: {hwork['version']}；参考: {pwork['version']}。",
             "同 prompt 的单进程 profiler 诊断；性能验收须另用无 profiler 的交错多进程计时。", "",
             "| 项目 | Hyle ms | 参考 ms | 差 ms |", "|---|---:|---:|---:|"]
    for title, key in rows:
        a, b = hs["sums"][key] / 1e6, ps["sums"][key] / 1e6
        lines.append(f"| {title} | {a:.3f} | {b:.3f} | {a-b:+.3f} |")
    for title, key in (("所选进程 CUDA 活动空档", "idle"), ("范围开始到首个 kernel", "first_kernel"),
                       ("选定 prefill 范围", "window")):
        a, b = hs[key] / 1e6, ps[key] / 1e6
        lines.append(f"| {title} | {a:.3f} | {b:.3f} | {a-b:+.3f} |")
    host = hs.get("host_preparation")
    if host:
        lines += ["", f"Hyle 主机准备：{host['hyle.host.cold_batch']['count']} 次；"
                      "按同一进程 CUDA 活动时间线取交集。", "",
                  "| 主机阶段 | 总用时 ms | 与 CUDA 活动重叠 ms | 与 CUDA 空档重叠 ms |",
                  "|---|---:|---:|---:|"]
        for title, name in (("完整准备", "hyle.host.cold_batch"), ("布局 / 状态", "hyle.host.layout"),
                            ("批次绑定", "hyle.host.batch_prepare"), ("回放准备", "hyle.host.replay_prepare")):
            row = host[name]
            lines.append(f"| {title} | {row['elapsed']/1e6:.3f} | {row['busy']/1e6:.3f} | {row['idle']/1e6:.3f} |")
        lines += ["", "时间重叠仅表明同时发生，不能单凭它证明 GPU 等待的原因。"]
    lines += ["", "类型转换单列：无法仅凭 kernel 名称判断它属于 GDN、注意力或其他图算子。",
              "空档是被追踪进程的 kernel、memcpy、memset 活动并集之外的时间；不代表整卡空闲。",
              "首 kernel 前时间与空档和拷贝可能重叠；各类 GPU 时间也可能并发，不能直接相加推算请求时间。",
              f"kernel 数：Hyle {hs['kernels']}，参考 {ps['kernels']}。",
              f"Hyle 使用首 token 发布记录的观测点，略晚于原始记录时刻；外围范围还延长 {(hi.outer_end-hi.end)/1e6:.3f} ms，已排除。",
              f"首 token ID：{'相同' if hwork['output_ids'][0] == pwork['output_ids'][0] else '不同'}。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyle", type=Path, required=True, help="Hyle profile directory")
    parser.add_argument("--reference", type=Path, required=True, help="llama.cpp profile directory")
    args = parser.parse_args()
    try:
        print(report(load_profile(args.hyle, "hyle"), load_profile(args.reference, "peer")), end="")
    except (OSError, ValueError, KeyError, sqlite3.Error, json.JSONDecodeError) as error:
        print(f"inference-nsys-compare: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
