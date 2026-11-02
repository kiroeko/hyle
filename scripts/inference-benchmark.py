#!/usr/bin/env python3
"""Run the recurring Qwen3.5 matched Hyle/llama.cpp inference comparison.

Both consumers, the model files and tokenized schedules must already exist.
Each timing sample is a fresh process. Memory is sampled in separate fresh
processes, so observing memory cannot contaminate the timing measurements.
The reference must implement the matched consumer protocol used by
inference-compare.py; it is not the unmodified llama-cli or llama-bench.
The shared comparison parser currently uses Qwen3.5 EOS and report contracts.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COMPARE_PATH = ROOT / "scripts/inference-compare.py"
spec = importlib.util.spec_from_file_location("hyle_inference_compare", COMPARE_PATH)
compare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compare
spec.loader.exec_module(compare)


@dataclass(frozen=True)
class Case:
    name: str
    capacity: int
    scenario: object


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Ordinary reference GGUF; required when ordinary mode is selected")
    parser.add_argument("--mtp-model", type=Path, help="Reference GGUF with MTP; required when MTP mode is selected")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference-inputs", type=Path, required=True)
    parser.add_argument("--hyle-version", required=True)
    parser.add_argument("--reference-version", required=True)
    parser.add_argument("--hyle-runtime", type=Path)
    parser.add_argument("--reference-runtime", type=Path)
    parser.add_argument("--long-workload", type=Path,
                        help="Directory containing <length>/schedule.txt and native-tokenized IDs")
    parser.add_argument("--multi-schedule", type=Path,
                        help="Real, distinct multi-turn prompts in the existing schedule format")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384, 32736])
    parser.add_argument("--long-capacity", type=int, default=32768)
    parser.add_argument("--multi-capacities", type=int, nargs="+", default=[2048, 32768])
    parser.add_argument("--only", nargs="+", help="Run only named cases shown by --dry-run")
    parser.add_argument("--modes", choices=("off", "on", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3, help="Fresh timing processes per arm, at least 3")
    parser.add_argument("--memory-repeats", type=int, default=1,
                        help="Separate memory processes per arm; 0 skips memory, 3 supports memory acceptance")
    parser.add_argument("--memory-interval", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True, help="New directory below .scratch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze", action="store_true", help="Rebuild the table from an existing complete run; no model process")
    args = parser.parse_args()
    require(args.repeats >= 3 and args.memory_repeats >= 0, "Need >=3 timing and nonnegative independent memory processes per arm")
    require(math.isfinite(args.timeout) and args.timeout > 0 and math.isfinite(args.memory_interval)
            and args.memory_interval > 0, "Timeout and memory sampling interval must be positive")
    require(all(x > 0 for x in args.lengths + args.multi_capacities) and args.long_capacity > 0,
            "Lengths and capacities must be positive")
    require(len(set(args.lengths)) == len(args.lengths) and len(set(args.multi_capacities)) == len(args.multi_capacities),
            "Duplicate lengths or multi-turn capacities")
    args.mode_list = ("off", "on") if args.modes == "both" else (args.modes,)
    require(not args.only or len(set(args.only)) == len(args.only), "Duplicate case names")
    for name in ("hyle", "reference", "model", "mtp_model", "inputs", "reference_inputs",
                 "hyle_runtime", "reference_runtime", "long_workload", "multi_schedule", "output"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    scratch = (ROOT / ".scratch").resolve()
    require(args.output.is_relative_to(scratch) and args.output != scratch, "Output must be a new directory under .scratch")
    require(not (args.dry_run and args.analyze), "Choose dry-run or analyze, not both")
    if args.analyze:
        require(args.output.is_dir() and (args.output / "comparison.md").is_file(),
                "Analyze requires a previous run with a complete table")
        incomplete = args.output / "INCOMPLETE.txt"
        if incomplete.exists():
            require(incomplete.read_text(encoding="utf-8-sig").startswith("UnicodeEncodeError:"),
                    "Cannot recover an incomplete measurement; preserve the raw result")
    else:
        require(not args.output.exists(), "Preserve previous results; choose a new output directory")
    for path in (args.hyle, args.reference):
        require(path.is_file(), f"Required local file is missing: {path}")
    for mode in args.mode_list:
        model = args.mtp_model if mode == "on" else args.model
        require(model is not None and model.is_file(), f"Required local {mode} model is missing: {model}")
    require(args.inputs.is_dir() and args.reference_inputs.is_dir(), "Inputs must already exist; no download is attempted")
    require(os.name == "nt", "Independent GPU and process memory sampling currently requires Windows")
    return args


def cases_for(args):
    cases = []
    allowed = {f"long-{length}-c{args.long_capacity}" for length in args.lengths}
    allowed.update(f"multi-c{capacity}" for capacity in args.multi_capacities)
    require(not args.only or set(args.only) <= allowed, "Unknown --only case; use --dry-run to list cases")
    for length in args.lengths:
        name = f"long-{length}-c{args.long_capacity}"
        if args.only and name not in args.only:
            continue
        require(args.long_workload is not None and args.long_workload.is_dir(),
                "A local --long-workload is required for long cases")
        schedule = args.long_workload / str(length) / "schedule.txt"
        scenario = compare.read_schedule(schedule)
        require(len(scenario.prompts) == 1 and len(scenario.prompts[0]) == length,
                f"Long schedule is not one {length}-token prompt: {schedule}")
        require(length + scenario.quotas[0] <= args.long_capacity,
                f"Long schedule exceeds capacity: {schedule}")
        cases.append(Case(name, args.long_capacity, scenario))
    selected_multi = [capacity for capacity in args.multi_capacities
                      if not args.only or f"multi-c{capacity}" in args.only]
    if not selected_multi:
        return cases
    require(args.multi_schedule is not None and args.multi_schedule.is_file(),
            "A local --multi-schedule is required for multi-turn cases")
    multi = compare.read_schedule(args.multi_schedule)
    require(len(multi.prompts) >= 2 and len({tuple(x) for x in multi.prompts}) == len(multi.prompts),
            "Multi-turn schedule must contain distinct prompts")
    for capacity in selected_multi:
        require(sum(map(len, multi.prompts)) + sum(multi.quotas) <= capacity,
                f"Multi-turn schedule exceeds capacity {capacity}")
        cases.append(Case(f"multi-c{capacity}", capacity, multi))
    return cases


def request_args(args, case, mode):
    return SimpleNamespace(inputs=args.inputs, reference_inputs=args.reference_inputs,
                           capacity=case.capacity, mtp=mode, timeout=args.timeout,
                           model=args.mtp_model if mode == "on" else args.model)


def qualify(raw, stderr, observation, arm, case, mode):
    require(raw.get("mtp") == (mode == "on"), "Producer mode differs")
    require(len(observation["turns"]) == len(case.scenario.prompts), "Producer turn count differs")
    if arm.hyle:
        require(raw.get("future_shape_hints") == 0 and raw.get("cold_planning") == "configured_capacity",
                "Hyle exposed future schedule to cold planning")
        require(raw.get("backend") == "cuda" and raw.get("sampling") == "greedy" and
                raw.get("prefix_cache") is False, "Hyle actual backend/sampling/prefix policy differs")
        require(raw.get("live_committed_versions") == 0, "Hyle retained committed versions")
        require(re.search(rf"\bconfigured_capacity={case.capacity}\b", stderr) is not None and
                re.search(r"\bvisible_turns=1\b", stderr) is not None,
                "Hyle cold planning capacity/first-turn boundary differs")
    else:
        require(raw.get("target_effective_context") == case.capacity, "Reference actual target context differs")
        require(raw.get("effective_batch_capacity") == 2048 and raw.get("effective_microbatch_capacity") == 512,
                "Reference batch/ubatch differs from the fixed 2048/512 protocol")
        warm = raw.get("startup_warmup") or {}
        require(warm.get("enabled") and warm.get("target_empty_pos_max") == -1,
                "Reference did not warm up and restore target state")
        if mode == "on":
            require(raw.get("draft_effective_context") == case.capacity and warm.get("draft_empty_pos_max") == -1,
                    "Reference draft capacity/warmup differs")


def observe_memory(folder, arm, case, mode, args, memory_probe):
    """An independent process; its elapsed time is never a timing sample."""
    folder.mkdir()
    command = compare.command(arm, case.scenario, request_args(args, case, mode))
    (folder / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    environment = compare.environment(arm)
    probe = memory_probe.MemoryProbe(interval_s=args.memory_interval)
    process = None
    stamp = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        with (folder / "stdout.log").open("wb") as out, (folder / "stderr.log").open("wb") as err:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                probe.start(process.pid)
                code = process.wait(timeout=args.timeout)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                probe.stop()
        result = probe.result()
    finally:
        probe.close()
    elapsed = time.perf_counter() - start
    require(code == 0, f"Memory producer failed: {folder}")
    require(result["samples"] and result["thread_error"] is None, f"Memory observer failed: {folder}")
    stamps = [row["timestamp_ns"] for row in result["samples"]]
    gap_ms = max((b - a for a, b in zip(stamps, stamps[1:])), default=0) / 1e6
    fields = ("timestamp_ns", "elapsed_ns", "working_set_bytes", "os_peak_working_set_bytes",
              "private_commit_bytes", "os_peak_commit_bytes", "gpu_dedicated_bytes", "gpu_shared_bytes", "gpu_total_bytes")
    with (folder / "samples.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["samples"])
    peaks = result["peaks"]
    summary = [f"exit={code} independent_memory_process=True timing_comparison=False start_utc={stamp}",
               f"sample_count={len(stamps)} max_gap_ms={gap_ms:.3f} gpu_setup_error={result['gpu_setup_error']!r}"]
    summary += [f"{key}={value}" for key, value in peaks.items()]
    (folder / "result.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    stdout = (folder / "stdout.log").read_text(encoding="utf-8-sig")
    stderr = (folder / "stderr.log").read_text(encoding="utf-8", errors="replace")
    raw = json.loads(stdout)
    observation = compare.parse_report(stdout, stderr, arm, case.scenario, case.capacity, elapsed, mode == "on")
    qualify(raw, stderr, observation, arm, case, mode)
    return {"peaks": peaks, "observation": observation, "samples": len(stamps), "max_gap_ms": gap_ms,
            "gpu_setup_error": result["gpu_setup_error"]}


def med(values):
    require(bool(values) and all(v is not None and math.isfinite(v) for v in values), "Missing measurement")
    return statistics.median(values)


def speed_cell(left, right):
    if any(value is None for value in left + right):
        return "不可用"
    h, r = med(left), med(right)
    overlap = max(min(left), min(right)) <= min(max(left), max(right))
    return f"{h:.1f} / {r:.1f} ({(h / r - 1) * 100:+.1f}%){'†' if overlap else ''}"


def memory_cell(left, right, key):
    h = [row["peaks"].get(key) for row in left]
    r = [row["peaks"].get(key) for row in right]
    if any(x is None for x in h + r):
        return "不可用"
    h, r = med(h) / 2**20, med(r) / 2**20
    return f"{h:.0f} / {r:.0f} ({h-r:+.0f})"


def summarize(cases, measurements, args):
    table = ["| 场景 / 轮次 | 输入 token | 历史 token | 准备 s（普通；MTP） | prefill 普通 tok/s | prefill MTP tok/s | decode 普通 tok/s | decode MTP tok/s | 显存 MiB（普通；MTP） | 内存 MiB（普通；MTP） | ID（普通；MTP） |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    details = []
    for case in cases:
        groups = {mode: {arm: measurements[(case.name, mode, arm)] for arm in ("hyle", "reference")}
                  for mode in args.mode_list}
        for mode, arms in groups.items():
            for name, group in arms.items():
                require(len(group["timing"]) == args.repeats and len(group["memory"]) == args.memory_repeats,
                        f"Incomplete samples: {case.name}/{mode}/{name}")
                sequences = [x["generated_ids"] for x in group["timing"]]
                require(all(s == sequences[0] for s in sequences), f"Greedy IDs unstable: {case.name}/{mode}/{name}")
                require(all(x["observation"]["generated_ids"] == sequences[0] for x in group["memory"]),
                        f"Memory process produced different IDs: {case.name}/{mode}/{name}")
                for key in ("startup_to_ready_seconds", "startup_to_endpoint_seconds", "parent_process_wall_seconds"):
                    details.append(f"{case.name} {mode} {name} {key}: "
                                   + compare.distribution([r["metrics"].get(key) for r in group["timing"]]))
                for memory in group["memory"]:
                    details.append(f"{case.name} {mode} {name} memory samples={memory['samples']} "
                                   f"max_gap_ms={memory['max_gap_ms']:.3f} gpu_setup_error={memory['gpu_setup_error']!r} "
                                   f"peaks={memory['peaks']}")
                for index in range(len(case.scenario.prompts)):
                    turns = [r["turns"][index] for r in group["timing"]]
                    for key in ("outputs", "input_rows", "prefill_seconds", "decode_seconds",
                                "prefill_tokens_per_second", "decode_tokens_per_second"):
                        vals = [t[key] for t in turns]
                        details.append(f"{case.name} {mode} {name} turn={index+1} {key}: {compare.distribution(vals)}")
                    for key in ("prefill_captures", "prefill_capture_associated_cold_prep_seconds",
                                "decode_captures", "decode_capture_associated_cold_prep_seconds"):
                        details.append(f"{case.name} {mode} {name} turn={index+1} {key}: "
                                       + compare.distribution([t["capture"][key] for t in turns]))
                    if mode == "on":
                        keys = sorted(set().union(*(t.get("producer_work", {}) for t in turns)))
                        if not keys:
                            details.append(f"{case.name} {mode} {name} turn={index+1} MTP work: not exported")
                        for key in keys:
                            details.append(f"{case.name} {mode} {name} turn={index+1} producer_{key}: "
                                           + compare.distribution([t.get("producer_work", {}).get(key) for t in turns]))
            left, right = arms["hyle"]["timing"][0], arms["reference"]["timing"][0]
            details.append(f"{case.name} {mode} Hyle/reference IDs: {compare.id_comparison(left['generated_ids'], right['generated_ids'])}")
        def ready(mode):
            if mode not in groups:
                return "未测"
            return "{:.3f} / {:.3f}".format(*[med([r["metrics"]["startup_to_ready_seconds"] for r in groups[mode][arm]["timing"]])
                                               for arm in ("hyle", "reference")])
        def memory(mode, key):
            if mode not in groups or not args.memory_repeats:
                return "未测"
            return memory_cell(groups[mode]["hyle"]["memory"], groups[mode]["reference"]["memory"], key)
        for i, prompt in enumerate(case.scenario.prompts):
            def speed(mode, key):
                if mode not in groups:
                    return "未测"
                return speed_cell(*[[r["turns"][i][key] for r in groups[mode][arm]["timing"]]
                                    for arm in ("hyle", "reference")])
            name = (f"单轮 C{case.capacity}" if len(case.scenario.prompts) == 1
                    else f"多轮 C{case.capacity} 第{i+1}轮")
            history_pairs = {mode: [sum(t["target_rows"] for t in groups[mode][arm]["timing"][0]["turns"][:i])
                                    for arm in ("hyle", "reference")] for mode in groups}
            history = [value for pair in history_pairs.values() for value in pair]
            history_cell = str(history[0]) if len(set(history)) == 1 else "; ".join(
                "/".join(map(str, history_pairs[mode])) if mode in history_pairs else "未测" for mode in ("off", "on"))
            prepared = ready("off") + "; " + ready("on") if i == 0 else "同上"
            vram = memory("off", "sampled_peak_gpu_dedicated_bytes") + "; " + memory("on", "sampled_peak_gpu_dedicated_bytes") if i == 0 else "同上"
            ram = memory("off", "observed_os_peak_working_set_bytes") + "; " + memory("on", "observed_os_peak_working_set_bytes") if i == 0 else "同上"
            def id_cell(mode):
                if mode not in groups:
                    return "未测"
                left, right = [groups[mode][arm]["timing"][0]["turns"][i]["generated_ids"]
                               for arm in ("hyle", "reference")]
                if left == right:
                    return "相同"
                first = next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
                return f"差@{first+1} ({len(left)}/{len(right)})"
            table.append(f"| {name} | {len(prompt)} | {history_cell} | {prepared} | "
                         f"{speed('off','prefill_tokens_per_second')} | {speed('on','prefill_tokens_per_second')} | "
                         f"{speed('off','decode_tokens_per_second')} | {speed('on','decode_tokens_per_second')} | {vram} | {ram} | "
                         f"{id_cell('off')}; {id_cell('on')} |")
    header = ["# Hyle / llama.cpp 推理对照", "",
              f"测量开始（UTC）：{args.measurement_started}；表格更新：{datetime.now(timezone.utc).isoformat()}。",
              f"Hyle：`{args.hyle_version}`；参考：`{args.reference_version}`。GPU / 驱动：{args.hardware}。",
              f"速度、准备：每臂 {args.repeats} 个独立进程的中位数，Hyle / 参考（相对差）；" +
              (f"显存、内存：每臂 {args.memory_repeats} 个独立采样进程，Hyle / 参考（Hyle − 参考），单位 MiB。"
               if args.memory_repeats else "本次未采样显存和内存。"), ""]
    notes = ["", "† 表示两组计时范围重叠，不能据此断定领先或退化；未标记也不代表经过显著性检验。",
             "准备为启动到 ready，含模型加载和启动预热；prefill 从请求到首 token，decode 含该轮首次使用开销。",
             "输入是本轮用户 prompt；多轮 prefill 还包含上一轮待提交 token，速度按实际输入行数计算。",
             "历史 token 是本轮之前已处理的上下文；若四个进程组不同，按普通 Hyle/参考；MTP Hyle/参考列出。",
             "ID 是该轮自然输出对比；一旦分叉，后续轮次的实际上下文可能不同。",
             "显存为本进程专用 GPU 内存采样峰；内存为观测到的进程工作集操作系统峰值，包含共享/映射页，不等于独占物理内存。",
             "GPU 采样峰是下界，跨进程共享显存不可相加；多轮的准备和内存为整次进程峰值，不是该轮增量。",
             "每次为新进程，但不清除操作系统文件缓存；准备时间不等于首次从冷磁盘加载。",
             "各场景的两引擎在同一窗口交错运行；不同场景及普通/MTP 使用各自窗口，不能跨窗口推算差距。"]
    return "\n".join(header + table + notes) + "\n", "\n".join(details) + "\n"


def read_previous_run(args, cases, arms):
    """Re-validate and reformat raw logs; never launch a model or a sampler."""
    plan = (args.output / "plan.txt").read_text(encoding="utf-8-sig")
    old_table = (args.output / "comparison.md").read_text(encoding="utf-8-sig")
    require(f"`{args.hyle_version}`" in old_table and f"`{args.reference_version}`" in old_table,
            "Re-analysis version labels differ from the original run")
    require(f"timing_repeats={args.repeats} independent_memory_repeats={args.memory_repeats} " in plan,
            "Re-analysis sample counts differ from the original run")
    hardware = re.search(r"^GPU/driver=(.*)$", plan, re.M)
    require(hardware is not None, "Original hardware record is missing")
    args.hardware = hardware.group(1)
    start = re.search(r"^UTC=(\S+)", plan, re.M)
    require(start is not None, "Original measurement start is missing")
    args.measurement_started = start.group(1)
    data = {}
    for ci, case in enumerate(cases):
        for mi, mode in enumerate(args.mode_list):
            parent = args.output / case.name / mode
            for arm in arms:
                data[(case.name, mode, arm.name)] = {"timing": [], "memory": []}
            for category, count in (("timing", args.repeats), ("memory", args.memory_repeats)):
                expected = compare.interleaving(arms, count, ci + mi)
                for ordinal, arm in enumerate(expected, 1):
                    folder = parent / category / f"{ordinal:02}-{arm.name}"
                    command = subprocess.list2cmdline(compare.command(arm, case.scenario, request_args(args, case, mode)))
                    require((folder / "command.txt").read_text(encoding="utf-8-sig").strip() == command,
                            f"Recorded command differs: {folder}")
                    result = (folder / "result.txt").read_text(encoding="utf-8-sig")
                    require(result.startswith("exit=0 "), f"Producer failed: {folder}")
                    stdout = (folder / "stdout.log").read_text(encoding="utf-8-sig")
                    stderr = (folder / "stderr.log").read_text(encoding="utf-8", errors="replace")
                    raw = json.loads(stdout)
                    if category == "timing":
                        wall = re.search(r"\bwall_seconds=([0-9.]+)", result)
                        require(wall is not None, f"Original parent wall is missing: {folder}")
                        observation = compare.parse_report(stdout, stderr, arm, case.scenario,
                                                           case.capacity, float(wall.group(1)), mode == "on")
                        qualify(raw, stderr, observation, arm, case, mode)
                        data[(case.name, mode, arm.name)]["timing"].append(observation)
                        continue
                    require("independent_memory_process=True" in result and "timing_comparison=False" in result,
                            f"Memory was not independently observed: {folder}")
                    detail = raw["sessions"][0]["result"] if arm.hyle else raw
                    observation = {"generated_ids": detail["generated_ids"], "turns": detail["turns"]}
                    qualify(raw, stderr, observation, arm, case, mode)
                    output_at = 0
                    context = 0
                    for index, (turn, prompt, quota) in enumerate(zip(detail["turns"], case.scenario.prompts, case.scenario.quotas)):
                        begin, end = turn["output_begin"], turn["output_end"]
                        generated = detail["generated_ids"]
                        expected_input = ([generated[begin-1]] if index else []) + prompt
                        require(begin == output_at and 1 <= end-begin <= quota and turn["input_ids"] == expected_input
                                and turn["context_before"] == context and turn["requested_steps"] == quota,
                                f"Memory process request ledger differs: {folder}")
                        context += len(expected_input) + end-begin-1
                        require(turn["context_after"] == context, f"Memory process context differs: {folder}")
                        output_at = end
                    require(output_at == len(detail["generated_ids"]), f"Memory output ledger differs: {folder}")
                    peaks = {}
                    for key in ("sampled_max_working_set_bytes", "observed_os_peak_working_set_bytes",
                                "sampled_max_private_commit_bytes", "observed_os_peak_commit_bytes",
                                "sampled_peak_gpu_dedicated_bytes", "sampled_peak_gpu_shared_bytes", "sampled_peak_gpu_total_bytes"):
                        found = re.search(rf"^{key}=(\S+)$", result, re.M)
                        require(found is not None, f"Memory peak field missing: {folder}/{key}")
                        peaks[key] = None if found.group(1) == "None" else int(found.group(1))
                    sample = re.search(r"\bsample_count=(\d+) max_gap_ms=([0-9.]+) gpu_setup_error=(.*)$", result, re.M)
                    require(sample is not None and int(sample.group(1)) > 0, f"Memory sample metadata missing: {folder}")
                    data[(case.name, mode, arm.name)]["memory"].append(
                        {"peaks": peaks, "observation": observation, "samples": int(sample.group(1)),
                         "max_gap_ms": float(sample.group(2)), "gpu_setup_error": ast.literal_eval(sample.group(3))})
    return data


def main():
    args = arguments()
    cases = cases_for(args)
    arms = [compare.Arm("hyle", args.hyle, args.hyle_version, args.hyle_runtime, True),
            compare.Arm("reference", args.reference, args.reference_version, args.reference_runtime, False)]
    for arm in arms:
        compare.environment(arm)
    if args.analyze:
        data = read_previous_run(args, cases, arms)
        table, details = summarize(cases, data, args)
        (args.output / "comparison.md").write_text(table, encoding="utf-8", newline="\n")
        (args.output / "details.txt").write_text(details, encoding="utf-8", newline="\n")
        if (args.output / "INCOMPLETE.txt").exists():
            (args.output / "RECOVERED.txt").write_text(
                "Original console encoding failure retained in INCOMPLETE.txt. "
                "All expected commands, successful process exits, output ledgers, independent memory samples, "
                "versions and per-arm IDs were revalidated offline; no model process was rerun.\n", encoding="utf-8")
        print(table, flush=True)
        return 0
    smi = shutil.which("nvidia-smi")
    args.hardware = "不可用"
    if smi:
        try:
            info = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total",
                                   "--format=csv,noheader"], capture_output=True, text=True, timeout=5, check=True)
            args.hardware = "; ".join(info.stdout.strip().splitlines()) or "不可用"
        except (OSError, subprocess.SubprocessError):
            pass
    args.measurement_started = datetime.now(timezone.utc).isoformat()
    plan = [f"UTC={args.measurement_started} platform={platform.platform()}",
            f"GPU/driver={args.hardware}",
            f"hyle_version={args.hyle_version} reference_version={args.reference_version}",
            f"timing_repeats={args.repeats} independent_memory_repeats={args.memory_repeats} interval={args.memory_interval}",
            "No build/download/extra warmup/profiler/retry. GPU concurrency=1."]
    for case in cases:
        plan.append(f"{case.name} capacity={case.capacity} prompts={[len(x) for x in case.scenario.prompts]} quotas={case.scenario.quotas}")
        for mode in args.mode_list:
            for arm in arms:
                plan.append(f"{mode}/{arm.name} " + subprocess.list2cmdline(compare.command(arm, case.scenario, request_args(args, case, mode))))
    print("\n".join(plan), flush=True)
    if args.dry_run:
        return 0
    import inference_memory_probe
    args.output.mkdir(parents=True)
    (args.output / "plan.txt").write_text("\n".join(plan) + "\n", encoding="utf-8")
    data = {}
    try:
        for ci, case in enumerate(cases):
            for mi, mode in enumerate(args.mode_list):
                for arm in arms:
                    data[(case.name, mode, arm.name)] = {"timing": [], "memory": []}
                parent = args.output / case.name / mode
                (parent / "timing").mkdir(parents=True)
                (parent / "memory").mkdir()
                for ordinal, arm in enumerate(compare.interleaving(arms, args.repeats, ci + mi), 1):
                    folder = parent / "timing" / f"{ordinal:02}-{arm.name}"
                    print(f"START timing {case.name}/{mode}/{folder.name}", flush=True)
                    record = compare.execute(folder, arm, case.scenario, request_args(args, case, mode))
                    raw = json.loads((folder / "stdout.log").read_text(encoding="utf-8-sig"))
                    stderr = (folder / "stderr.log").read_text(encoding="utf-8", errors="replace")
                    qualify(raw, stderr, record["observation"], arm, case, mode)
                    data[(case.name, mode, arm.name)]["timing"].append(record["observation"])
                for ordinal, arm in enumerate(compare.interleaving(arms, args.memory_repeats, ci + mi), 1):
                    folder = parent / "memory" / f"{ordinal:02}-{arm.name}"
                    print(f"START memory {case.name}/{mode}/{folder.name}", flush=True)
                    data[(case.name, mode, arm.name)]["memory"].append(
                        observe_memory(folder, arm, case, mode, args, inference_memory_probe))
            partial, partial_details = summarize(cases[:ci+1], data, args)
            (args.output / "partial-comparison.md").write_text(partial, encoding="utf-8", newline="\n")
            (args.output / "partial-details.txt").write_text(partial_details, encoding="utf-8", newline="\n")
        table, details = summarize(cases, data, args)
        (args.output / "comparison.md").write_text(table, encoding="utf-8", newline="\n")
        (args.output / "details.txt").write_text(details, encoding="utf-8", newline="\n")
        print(table, flush=True)
    except BaseException as error:
        (args.output / "INCOMPLETE.txt").write_text(f"{type(error).__name__}: {error}\nNo automatic retry.\n", encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    # Windows terminals can default to GBK; report labels may contain Unicode.
    # The text artifacts are UTF-8 independently of the display encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"inference-benchmark: {error}", file=sys.stderr)
        raise SystemExit(1)
