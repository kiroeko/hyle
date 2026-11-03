#!/usr/bin/env python3
"""M1-7a: serial, fresh-process comparisons with stock upstream llama.cpp.

No engine changes, weight downloads, profiling, or in-process memory observers.
prepare verifies local inputs/software; select screens both builds/FA choices;
standard runs pp512/tg128; multi runs cached real conversations; memory repeats
multi with a separate-process Windows sampler; analyze reads the TSV/logs.
Software retrieval/build commands and pinned versions belong in the run's text
notes. All paths must already exist; the script never downloads model files.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("phase", "arm", "mode", "repeat", "scenario", "depth", "turn", "prefill_ms",
          "prefill_tps", "decode_tps", "decode_ms", "outputs", "input_rows", "context_before",
          "startup_s", "parent_ready_s", "wall_s", "proposals", "accepted", "gpu_mib",
          "private_mib", "log_dir")
PP_DEPTHS = (0, 4096, 16384, 30720)
TG_DEPTHS = tuple(d + 512 for d in PP_DEPTHS)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


MATCHED = module("hyle_upstream_matched", ROOT / "scripts/inference-compare.py")


def local_path(value):
    path = Path(value).resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Missing local path: {path}")
    return path


def finite(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (value == 0 and not zero)):
        raise ValueError(f"Invalid {name}: {value!r}")
    return float(value)


def parse_bench(text, kind, depths, reps):
    records = json.loads(text.lstrip("\ufeff"))
    if not isinstance(records, list) or len(records) != len(depths):
        raise ValueError("llama-bench did not report all requested depths")
    result = []
    for raw, depth in zip(records, depths):
        wanted = (512, 0) if kind == "pp" else (0, 128)
        if (raw["n_prompt"], raw["n_gen"], raw["n_depth"]) != (*wanted, depth):
            raise ValueError("llama-bench row differs from requested pp/tg/depth")
        samples = raw["samples_ns"]
        if len(samples) != reps:
            raise ValueError("llama-bench repetition count differs")
        for sample in samples:
            finite(sample, "bench sample")
        # Preserve the stock benchmark's mean-throughput endpoint. Samples are
        # within-process repetitions, never counted as independent processes.
        tps = finite(raw["avg_ts"], "avg_ts")
        result.append(dict(depth=depth, prefill_tps=tps if kind == "pp" else None,
                           prefill_ms=512000 / tps if kind == "pp" else None,
                           decode_tps=tps if kind == "tg" else None,
                           input_rows=512 if kind == "pp" else 0,
                           outputs=128 if kind == "tg" else 0))
    return result


def server_turn(raw, requested_prompt, cached_history, quota, *, mtp=False):
    tokens = raw.get("tokens")
    if (not isinstance(tokens, list) or len(tokens) != quota or
            any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens)):
        raise ValueError("Server output IDs/quota missing or truncated")
    timings = raw["timings"]
    if timings["predicted_n"] != len(tokens):
        raise ValueError("Server predicted_n differs from returned tokens")
    rows = timings["prompt_n"]
    cached = timings["cache_n"]
    # Recurrent cache reuse must actually consume the prior conversation. A
    # full replay is not quietly relabelled as an incremental prompt prefill.
    if rows + cached != len(requested_prompt):
        raise ValueError("Server prompt/cache accounting differs from token array")
    expected_rows = len(requested_prompt) - cached_history + (1 if cached_history else 0)
    if rows != expected_rows:
        raise ValueError(f"Server recomputed {rows} prompt rows; expected {expected_rows}; inspect cache semantics")
    prefill = finite(timings["prompt_ms"], "prompt_ms")
    decode = finite(timings["predicted_ms"], "predicted_ms")
    proposals, accepted = timings.get("draft_n"), timings.get("draft_n_accepted")
    if mtp and (proposals is None or accepted is None or not 0 <= accepted <= proposals):
        raise ValueError("MTP server did not export valid proposal/accept counts")
    return dict(prefill_ms=prefill, prefill_tps=rows * 1000 / prefill,
                decode_ms=decode, decode_tps=(quota - 1) * 1000 / decode,
                outputs=quota, input_rows=rows, context_before=cached,
                proposals=proposals, accepted=accepted), tokens


def hyle_mtp_counts(raw):
    """Count verified proposals from the exported committed ledger, no guesses."""
    detail = raw["sessions"][0]["result"]
    result = {turn["turn"]: dict(proposals=0, accepted=0) for turn in detail["turns"]}
    for record in raw["sessions"][0]["ledger"]:
        if not record["generation"]:
            continue
        proposals, accepted = record["draft_limit"], record["consumed_drafts"]
        if not 0 <= accepted <= proposals <= 1:
            raise ValueError("Expected Hyle's default K1 proposal ledger")
        result[record["turn"]]["proposals"] += proposals
        result[record["turn"]]["accepted"] += accepted
    if (sum(t["proposals"] for t in result.values()) != raw["draft_rounds"] or
            sum(t["accepted"] for t in result.values()) != raw["consumed_drafts"]):
        raise ValueError("Hyle per-turn MTP ledger differs from complete counters")
    return result


def write_rows(path, rows):
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", extrasaction="raise")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for key in FIELDS:
            if key not in ("phase", "arm", "mode", "scenario", "log_dir"):
                row[key] = float(row[key]) if row[key] else None
    return rows


def env_for(args, arm):
    env = os.environ.copy()
    receipt = args.hyle_runtime if arm == "hyle" else args.reference_runtime if arm == "frozen" else None
    if receipt:
        env["PATH"] = json.loads(receipt.read_text(encoding="utf-8-sig"))["PATH"]
    if arm in ("official", "source"):
        folder = args.official_bin if arm == "official" else args.source_bin
        env["PATH"] = str(folder) + os.pathsep + env["PATH"]
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", GIT_TERMINAL_PROMPT="0")
    # Do not inherit tuning/CLI overrides into the default build comparison.
    for key in tuple(env):
        if key.startswith(("GGML_", "LLAMA_ARG_")):
            env.pop(key)
    return env


def exclusive(args, folder):
    """Fail on competing inference; require a quiet device before the run."""
    if os.name == "nt":
        query = ("Get-Process -ErrorAction SilentlyContinue | Where-Object { "
                 "$_.ProcessName -match '^(hyle.*|llama.*|peer_long|nsys|ncu)$' } | "
                 "Select-Object -ExpandProperty Id")
        running = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                                 capture_output=True, text=True, check=True)
        if running.stdout.strip():
            raise RuntimeError("Competing GPU test process exists: " + running.stdout.strip())
    deadline, quiet, observations = time.monotonic() + args.idle_timeout, 0, []
    while time.monotonic() < deadline:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        values = gpu.stdout.strip()
        observations.append(f"{datetime.now(timezone.utc).isoformat()} {values}")
        utilization = max(float(line.split(",")[0]) for line in values.splitlines())
        quiet = quiet + 1 if utilization <= args.idle_max_util else 0
        if quiet >= 3:
            (folder / "gpu-before.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")
            return
        time.sleep(1)
    (folder / "gpu-before.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")
    raise RuntimeError("GPU did not become idle; no timing process started")


def new_run(args, name):
    folder = args.output / name
    folder.mkdir()  # never overwrite/retry an existing timed result
    exclusive(args, folder)
    return folder


def raw_run(args, arm, command, folder, *, memory=False):
    (folder / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    probe = None
    if memory:
        from inference_memory_probe import MemoryProbe
        probe = MemoryProbe(interval_s=0.01)
    ready = []
    start = time.perf_counter()
    with (folder / "stdout.log").open("wb") as output, (folder / "stderr.log").open("wb") as error:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.PIPE, env=env_for(args, arm))
        def drain():
            for line in process.stderr:
                error.write(line)
                if b"startup_to_ready_seconds=" in line:
                    ready.append(time.perf_counter() - start)
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            if probe:
                probe.start(process.pid)
            code = process.wait(timeout=args.process_timeout)
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            reader.join()
            if probe:
                probe.stop()
                memory_report(folder, probe.result())
                probe.close()
    wall = time.perf_counter() - start
    (folder / "result.txt").write_text(f"exit={code} wall_s={wall:.9f}\n", encoding="utf-8")
    if code:
        raise RuntimeError(f"Native command failed ({code}); logs: {folder}")
    return wall, ready[0] if ready else None


def memory_report(folder, result):
    # Raw samples remain TSV, no new evidence JSON/manifest bundles.
    with (folder / "memory-samples.tsv").open("w", encoding="utf-8", newline="") as stream:
        keys = ("timestamp_ns", "working_set_bytes", "private_commit_bytes", "os_peak_commit_bytes",
                "gpu_dedicated_bytes", "gpu_shared_bytes", "sample_duration_ns")
        writer = csv.DictWriter(stream, keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["samples"])
    (folder / "memory.txt").write_text("\n".join(f"{key}={value}" for key, value in result["peaks"].items()) +
                                      f"\ngpu_setup_error={result['gpu_setup_error']}\nthread_error={result['thread_error']}\n" +
                                      "\n".join(result["limits"]) + "\n", encoding="utf-8")
    return result["peaks"]


def memory_peaks(folder):
    path = folder / "memory.txt"
    if not path.exists():
        return {}
    values = dict(re.findall(r"^(\w+)=(\d+)$", path.read_text(encoding="utf-8"), re.MULTILINE))
    return {"gpu_mib": int(values["sampled_peak_gpu_dedicated_bytes"]) / 2**20
            if "sampled_peak_gpu_dedicated_bytes" in values else None,
            "private_mib": int(values["observed_os_peak_commit_bytes"]) / 2**20
            if "observed_os_peak_commit_bytes" in values else None}


def bench(args, build, fa, kind, repeat, phase):
    folder = new_run(args, f"{phase}-{repeat}-{build}-{fa}-{kind}")
    binary = (args.official_bin if build == "official" else args.source_bin) / "llama-bench.exe"
    depths = PP_DEPTHS if kind == "pp" else TG_DEPTHS
    command = [str(binary), "-m", str(args.model), "-ngl", "99", "-r", str(args.bench_reps),
               "-p", "512" if kind == "pp" else "0", "-n", "0" if kind == "pp" else "128",
               "-d", ",".join(map(str, depths)), "-fa", "1" if fa == "on" else "0", "-o", "json"]
    wall, _ = raw_run(args, build, command, folder)
    rows = parse_bench((folder / "stdout.log").read_text(encoding="utf-8-sig"), kind, depths, args.bench_reps)
    for row in rows:
        row.update(phase=phase, arm=f"{build}-fa-{fa}", mode="off", repeat=repeat,
                   scenario=kind, wall_s=wall, log_dir=str(folder))
    write_rows(args.output / "records.tsv", rows)
    print(folder.name, [(r["depth"], round(r.get("prefill_tps") or r["decode_tps"], 2)) for r in rows], flush=True)


def standard_schedule(args, depth):
    folder = args.output / "workload" / str(depth)
    folder.mkdir(parents=True, exist_ok=True)
    tokens = MATCHED.read_ids(args.long_ids)
    if len(tokens) < depth + 512:
        raise ValueError("Long token tape too short")
    prompt = tokens[depth:depth + 512]
    lines = []
    if depth:
        (folder / "prefix.ids").write_text(" ".join(map(str, tokens[:depth])) + "\n", encoding="utf-8")
        lines += ["prompt prefix.ids", "generate 1"]
        # Previous turn's pending output is processed once. It is one of the
        # next 512 compute rows; sending 512 new IDs would incorrectly time 513.
        prompt = prompt[1:]
    (folder / "prompt.ids").write_text(" ".join(map(str, prompt)) + "\n", encoding="utf-8")
    lines += ["prompt prompt.ids", "generate 128"]
    path = folder / "schedule.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MATCHED.read_schedule(path)


def native(args, arm, scenario, mode, repeat, phase, *, depth=None, memory=False):
    name = f"{phase}-{repeat}-{arm}-{mode}-{depth if depth is not None else 'eight'}"
    folder = new_run(args, name)
    if arm == "hyle":
        command = [str(args.hyle), "--inputs", str(args.inputs), "--capacity", str(args.capacity),
                   "--backend", "cuda", "--mtp", mode, "--schedule", str(scenario.schedule)]
        version, exe, runtime = args.hyle_version, args.hyle, args.hyle_runtime
    else:
        command = [str(args.reference), "--model", str(args.mtp_model if mode == "on" else args.model),
                   "--inputs", str(args.reference_inputs), "--capacity", str(args.capacity),
                   "--mtp", mode, "--memory-snapshots", "off", "--schedule", str(scenario.schedule)]
        version, exe, runtime = args.reference_version, args.reference, args.reference_runtime
    wall, ready = raw_run(args, arm, command, folder, memory=memory)
    stderr = (folder / "stderr.log").read_text(encoding="utf-8-sig")
    if arm == "hyle":
        capacity = re.search(r"\bconfigured_capacity=(\d+)", stderr)
        if not capacity or int(capacity[1]) != args.capacity or "future_shape_hints=0" not in stderr:
            raise ValueError("Hyle cold-plan capacity/future-input boundary was not confirmed")
    report = MATCHED.parse_report((folder / "stdout.log").read_text(encoding="utf-8-sig"),
                                  stderr,
                                  MATCHED.Arm(arm, exe, version, runtime, arm == "hyle"),
                                  scenario, args.capacity, wall, mode == "on")
    selected = report["turns"][-1:] if depth is not None else report["turns"]
    producer_raw = json.loads((folder / "stdout.log").read_text(encoding="utf-8-sig"))
    counts = hyle_mtp_counts(producer_raw) if arm == "hyle" and mode == "on" else {}
    rows = []
    for turn in selected:
        if depth is not None:
            raw = json.loads((folder / "stdout.log").read_text(encoding="utf-8-sig"))
            actual = raw["sessions"][0]["result"]["turns"][-1]
            if turn["input_rows"] != 512 or actual["context_before"] != depth:
                raise ValueError("Hyle pp512 is not exactly 512 processed rows at requested depth")
        work = turn.get("producer_work", {})
        counted = counts.get(turn["turn"], {})
        rows.append(dict(phase=phase, arm=arm, mode=mode, repeat=repeat,
                         scenario="standard" if depth is not None else "eight", depth=depth,
                         turn=turn["turn"], prefill_ms=turn["prefill_seconds"] * 1000,
                         prefill_tps=turn["prefill_tokens_per_second"],
                         decode_tps=turn["decode_tokens_per_second"], decode_ms=turn["decode_seconds"] * 1000,
                         outputs=turn["outputs"], input_rows=turn["input_rows"],
                         context_before=depth if depth is not None else None,
                         startup_s=report["metrics"]["startup_to_ready_seconds"], parent_ready_s=ready,
                         wall_s=wall, proposals=counted.get("proposals", work.get("proposals", work.get("draft_rounds"))),
                         accepted=counted.get("accepted", work.get("accepted_drafts", work.get("consumed_drafts"))),
                         log_dir=str(folder), **memory_peaks(folder)))
    write_rows(args.output / "records.tsv", rows)
    print(name, "decode", [round(r["decode_tps"], 2) if r["decode_tps"] else None for r in rows], flush=True)


def request(port, route, body=None, timeout=300):
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    call = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=data,
                                  headers={"Content-Type": "application/json"})
    # Local traffic must never be sent through a user/system HTTP proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(call, timeout=timeout) as response:
        return json.load(response)


def upstream_server(args, scenario, mode, repeat, phase, *, memory=False):
    folder = new_run(args, f"{phase}-{repeat}-upstream-{mode}-eight")
    with socket.socket() as check:
        if check.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} already occupied")
    binary = (args.official_bin if args.selected == "official" else args.source_bin) / "llama-server.exe"
    command = [str(binary), "-m", str(args.mtp_model if mode == "on" else args.model), "-np", "1",
               "-c", str(args.capacity), "-ngl", "99", "-fa", args.fa,
               "--host", "127.0.0.1", "--port", str(args.port), "--metrics"]
    if mode == "on":
        command += ["--spec-type", "draft-mtp"]
    (folder / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    probe = None
    if memory:
        from inference_memory_probe import MemoryProbe
        probe = MemoryProbe(interval_s=0.01)
    rows = []
    with (folder / "stdout.log").open("wb") as output, (folder / "stderr.log").open("wb") as error:
        start = time.perf_counter()
        process = subprocess.Popen(command, stdout=output, stderr=error, env=env_for(args, args.selected))
        try:
            if probe:
                probe.start(process.pid)
            deadline = time.monotonic() + args.process_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited before ready: {folder}")
                try:
                    if request(args.port, "/health", timeout=0.25).get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server ready timeout")
                time.sleep(0.01)
            ready = time.perf_counter() - start
            history = []
            for number, (prompt, quota) in enumerate(zip(scenario.prompts, scenario.quotas), 1):
                old_count = len(history)
                history.extend(prompt)
                payload = dict(prompt=history, n_predict=quota, temperature=0, cache_prompt=True,
                               return_tokens=True, stream=False, seed=0)
                raw = request(args.port, "/completion", payload, args.process_timeout)
                (folder / f"round{number}.response.log").write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
                row, generated = server_turn(raw, history, old_count, quota, mtp=mode == "on")
                row.update(phase=phase, arm="upstream", mode=mode, repeat=repeat, scenario="eight",
                           turn=number, startup_s=ready, parent_ready_s=ready, log_dir=str(folder))
                rows.append(row)
                history.extend(generated)
            wall = time.perf_counter() - start
            for row in rows:
                row["wall_s"] = wall
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if probe:
                probe.stop()
                memory_report(folder, probe.result())
                probe.close()
    stderr = (folder / "stderr.log").read_text(encoding="utf-8-sig", errors="replace")
    capacities = [int(value) for value in re.findall(r"n_ctx(?:_per_seq|_slot)?\s*=\s*(\d+)", stderr)]
    if not capacities or any(value != args.capacity for value in capacities):
        raise ValueError("Server effective n_ctx was not confirmed from logs")
    for row in rows:
        row.update(memory_peaks(folder))
    (folder / "result.txt").write_text(f"ready_s={ready:.9f} wall_s={wall:.9f} context_values={capacities}\n", encoding="utf-8")
    write_rows(args.output / "records.tsv", rows)
    print(folder.name, "decode", [round(r["decode_tps"], 2) for r in rows], flush=True)


def cell(left, right):
    if not left or not right:
        return "未测"
    l, r = statistics.median(left), statistics.median(right)
    overlap = max(min(left), min(right)) <= min(max(left), max(right))
    return f"{l:.3f} / {r:.3f} = {100*l/r:.1f}%" + ("†" if overlap else "")


def normalized_rows(args):
    rows = read_rows(args.output / "records.tsv")
    # Older logs already export the ledger even if their normalized TSV omitted
    # the per-turn fields. Offline extraction never reruns or changes timings.
    for row in rows:
        if row["arm"] == "hyle" and row["mode"] == "on" and row["proposals"] is None:
            raw = json.loads((Path(row["log_dir"]) / "stdout.log").read_text(encoding="utf-8-sig"))
            row.update(hyle_mtp_counts(raw)[int(row["turn"])])
    return rows


def validate(args):
    rows = normalized_rows(args)
    repeats = set(range(1, args.repeats + 1))
    checked = []
    def group(values, label):
        if (len(values) != args.repeats or {r["repeat"] for r in values} != repeats or
                len({r["log_dir"] for r in values}) != args.repeats):
            raise ValueError(f"Missing/duplicate independent processes for {label}")
        checked.extend(values)
    for depth in PP_DEPTHS:
        hyle = [r for r in rows if r["phase"] == "A" and r["arm"] == "hyle" and r["depth"] == depth]
        group(hyle, f"Hyle d={depth}")
        if any(r["input_rows"] != 512 or r["outputs"] != 128 or r["context_before"] != depth for r in hyle):
            raise ValueError("Standard Hyle row/output/depth mismatch")
        for kind, actual in (("pp", depth), ("tg", depth+512)):
            group([r for r in rows if r["phase"] == "A" and r["arm"] == f"{args.selected}-fa-{args.fa}"
                   and r["scenario"] == kind and r["depth"] == actual], f"upstream {kind} d={actual}")
    for phase in ("BC", "D"):
        for arm in args.arms:
            for mode in args.modes:
                for turn in range(1,9):
                    values = [r for r in rows if r["phase"] == phase and r["arm"] == arm
                              and r["mode"] == mode and r["turn"] == turn]
                    group(values, f"{phase}/{arm}/{mode}/turn{turn}")
                    if any(r["outputs"] != 64 or not r["decode_tps"] for r in values):
                        raise ValueError("Eight-turn output quota or endpoint missing")
                    if mode == "on" and any(r["accepted"] is None or r["proposals"] is None or
                                             not 0 <= r["accepted"] <= r["proposals"] for r in values):
                        raise ValueError("MTP per-turn acceptance counters missing")
                    if phase == "D" and any(r["gpu_mib"] is None or r["gpu_mib"] <= 0 for r in values):
                        raise ValueError("Independent GPU memory samples missing")
    folders = {r["log_dir"] for r in checked}
    for name in sorted(folders):
        folder = Path(name)
        stderr = (folder/"stderr.log").read_text(encoding="utf-8-sig", errors="replace")
        sample = next(r for r in checked if r["log_dir"] == name)
        if sample["arm"] == "hyle":
            if f"configured_capacity={args.capacity}" not in stderr or "future_shape_hints=0" not in stderr:
                raise ValueError("Hyle cold-plan capacity/future boundary missing")
        elif sample["arm"] == "upstream":
            capacities = re.findall(r"n_ctx(?:_per_seq|_slot)?\s*=\s*(\d+)", stderr)
            if not capacities or any(int(n) != args.capacity for n in capacities):
                raise ValueError("Server effective context differs")
        if sample["phase"] == "D":
            memory = (folder/"memory.txt").read_text(encoding="utf-8")
            if "gpu_setup_error=None" not in memory or "thread_error=None" not in memory:
                raise ValueError("Independent memory sampler error")
    standard_count = sum(Path(n).name.startswith("A-") for n in folders)
    result = f"PASS: {len(folders)} independent processes; A={standard_count}, BC/D complete; every requested cell n={args.repeats}.\n"
    (args.output/"validation.txt").write_text(result, encoding="utf-8")
    print(result.strip())


def analyze(args):
    rows = normalized_rows(args)
    lines = ["# M1-7a 与上游正面对照", "", f"生成时间：{datetime.now(timezone.utc).isoformat()}",
             f"Hyle {args.hyle_version}; upstream {args.upstream_version}; frozen {args.reference_version}.",
             f"参考选择：{args.selected}, FA={args.fa}; capacity={args.capacity}。", "",
             "数值为 Hyle / 上游 = Hyle÷上游。速度越高越好；毫秒、启动和内存越低越好。",
             "†表示独立进程范围重叠；不是显著性检验。缺项不以旧窗口填补。", "",
             "| 场景 | 指标 | Hyle / 上游 = 比例 | Hyle范围 | 上游范围 | 冻结版中位数 | 独立进程数 |", "|---|---|---:|---:|---:|---:|---:|"]
    def compare(label, field, left, right, third=()):
        lv = [r[field] for r in left if r.get(field) is not None]
        rv = [r[field] for r in right if r.get(field) is not None]
        fv = [r[field] for r in third if r.get(field) is not None]
        range_text = lambda v: f"{min(v):.3f}–{max(v):.3f}" if v else "未测"
        frozen = f"{statistics.median(fv):.3f} (n={len(fv)})" if fv else "—"
        lines.append(f"| {label} | {field} | {cell(lv,rv)} | {range_text(lv)} | {range_text(rv)} | {frozen} | {len(lv)}/{len(rv)} |")
    for depth in PP_DEPTHS:
        hyle = [r for r in rows if r["phase"] == "A" and r["arm"] == "hyle" and r["depth"] == depth]
        for kind, field, actual_depth in (("pp512", "prefill_tps", depth), ("tg128", "decode_tps", depth + 512)):
            upstream = [r for r in rows if r["phase"] == "A" and r["arm"] == f"{args.selected}-fa-{args.fa}"
                        and r["depth"] == actual_depth and r["scenario"] == ("pp" if kind == "pp512" else "tg")]
            compare(f"{kind} d={actual_depth}", field, hyle, upstream)
    for mode in ("off", "on"):
        combined = {}
        for arm in ("hyle", "upstream", "frozen"):
            values = []
            group = [r for r in rows if r["phase"] == "BC" and r["mode"] == mode and r["arm"] == arm]
            for repeat in sorted({r["repeat"] for r in group}):
                process = [r for r in group if r["repeat"] == repeat]
                if len(process) != 8:
                    continue
                milliseconds = sum(r["decode_ms"] for r in process)
                values.append(dict(prefill_ms=sum(r["prefill_ms"] for r in process),
                                   decode_tps=sum(r["outputs"]-1 for r in process) * 1000 / milliseconds))
            combined[arm] = values
        for field in ("prefill_ms", "decode_tps"):
            compare(f"8轮 MTP={mode} 合并", field, combined["hyle"], combined["upstream"], combined["frozen"])
        for turn in range(1, 9):
            group = [r for r in rows if r["phase"] == "BC" and r["mode"] == mode and r["turn"] == turn]
            left, right = [r for r in group if r["arm"] == "hyle"], [r for r in group if r["arm"] == "upstream"]
            input_rows = next((int(r["input_rows"]) for r in left+right if r["input_rows"] is not None), "?")
            for field in ("prefill_ms", "decode_tps"):
                compare(f"8轮 MTP={mode} 第{turn}轮/{input_rows}行", field, left, right, [r for r in group if r["arm"] == "frozen"])
        # Each process contributes once to startup/memory, never once per turn.
        for phase, field in (("BC", "parent_ready_s"), ("D", "gpu_mib"), ("D", "private_mib")):
            group = [r for r in rows if r["phase"] == phase and r["mode"] == mode and r["turn"] == 1]
            compare(f"MTP={mode}", field, [r for r in group if r["arm"] == "hyle"],
                    [r for r in group if r["arm"] == "upstream"], [r for r in group if r["arm"] == "frozen"])
    lines += ["", "A: llama-bench tg128为随机token、无采样的128步；Hyle为128输出中的127个decode步，含greedy采样。",
              "A: 非零深度的Hyle先处理d行并产生一个待提交token；第二轮511个新ID加该token，共512计算行。",
              "llama-bench请求context=d+p+n，内部可按对齐取整；Hyle固定C32768。B/C/D两边固定C32768并核对server日志。",
              "B/C每个引擎使用自身输出续写同一组新prompt；分叉后历史内容可能不同。",
              "启动比较统一用parent_ready_s，含进程启动、模型加载、默认warmup及外部观察；原始startup_s保留Hyle内部计时。",
              "D为独立采样进程；专用显存采样峰是下界，private_mib是私有提交峰，不是RSS。", "",
              "## MTP 接受 / 提案", "", "Hyle默认K1，上游默认最多3个草稿。Hyle从导出的提交ledger逐轮汇总，并核对完整计数；不靠输出长度反推。", "",
              "| 轮 | Hyle | 上游 | 冻结版 |", "|---:|---|---|---|"]
    for turn in range(1,9):
        values = []
        for arm in ("hyle", "upstream", "frozen"):
            group = [r for r in rows if r["phase"] == "BC" and r["arm"] == arm and r["mode"] == "on" and r["turn"] == turn]
            pairs = [f"{r['accepted']:g}/{r['proposals']:g}" for r in group
                     if r["accepted"] is not None and r["proposals"] is not None]
            values.append(f"{pairs[0]} ×{len(pairs)}" if pairs and len(set(pairs)) == 1 else "; ".join(pairs) or "未测")
        lines.append(f"| {turn} | {' | '.join(values)} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "summary.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "select", "standard", "multi", "memory", "validate", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    for name in ("hyle", "inputs", "model", "mtp-model", "reference", "reference-inputs", "long-ids", "schedule",
                 "official-bin", "source-bin", "hyle-runtime", "reference-runtime"):
        parser.add_argument("--"+name, type=local_path)
    parser.add_argument("--hyle-version", default="unrecorded")
    parser.add_argument("--upstream-version", default="unrecorded")
    parser.add_argument("--reference-version", default="unrecorded")
    parser.add_argument("--selected", choices=("official", "source"), default="official")
    parser.add_argument("--fa", choices=("on", "off"), default="on")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bench-reps", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument("--modes", nargs="+", choices=("off", "on"), default=["off", "on"])
    parser.add_argument("--arms", nargs="+", choices=("hyle", "upstream", "frozen"), default=["hyle", "upstream", "frozen"])
    parser.add_argument("--port", type=int, default=2048)
    parser.add_argument("--idle-max-util", type=int, default=10)
    parser.add_argument("--idle-timeout", type=int, default=120)
    parser.add_argument("--process-timeout", type=int, default=600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.repeats < 1 or args.bench_reps < 1:
        parser.error("Positive independent process/repetition counts required")
    required = []
    if args.phase == "prepare":
        required = ["official_bin", "source_bin"]
    elif args.phase == "select":
        required = ["official_bin", "source_bin", "model"]
    elif args.phase == "standard":
        required = ["hyle", "inputs", "model", "long_ids", args.selected + "_bin"]
        if args.capacity < PP_DEPTHS[-1] + 512 + 127:
            parser.error("capacity cannot hold the longest standard prompt plus generation")
    elif args.phase in ("multi", "memory"):
        required = ["schedule"]
        if "hyle" in args.arms:
            required += ["hyle", "inputs"]
        if "upstream" in args.arms:
            required += [args.selected + "_bin", "model"]
        if "frozen" in args.arms:
            required += ["reference", "reference_inputs", "model"]
        if "on" in args.modes and any(arm != "hyle" for arm in args.arms):
            required += ["mtp_model"]
    for name in required:
        if getattr(args, name) is None:
            parser.error("Missing --" + name.replace("_", "-"))
    if args.phase == "analyze":
        analyze(args)
        return
    if args.phase == "validate":
        validate(args)
        return
    (args.output / f"invocation-{args.phase}-{time.time_ns()}.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    if args.phase == "prepare":
        for arm in ("official", "source"):
            folder = args.official_bin if arm == "official" else args.source_bin
            for tool in ("llama-bench.exe", "llama-server.exe"):
                result = subprocess.run([str(folder / tool), "--version"], env=env_for(args, arm),
                                        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
                (args.output / f"{arm}-{tool}-version.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        print("Verified local executables; no model loaded/downloaded")
    elif args.phase == "select":
        for build, fa in (("official","on"),("source","off"),("source","on"),("official","off")):
            for kind in ("pp", "tg"):
                bench(args, build, fa, kind, 1, "select")
        rows = [r for r in read_rows(args.output / "records.tsv") if r["phase"] == "select"]
        scores = {}
        for arm in sorted({r["arm"] for r in rows}):
            values = [r["prefill_tps"] or r["decode_tps"] for r in rows if r["arm"] == arm]
            scores[arm] = math.exp(statistics.mean(map(math.log, values)))
        winner = max(scores, key=scores.get)
        (args.output / "selection.txt").write_text("metric=geometric mean of 4 pp512 and 4 tg128 TPS\n" +
              "\n".join(f"{k}={v:.9f}" for k,v in scores.items()) + f"\nselected={winner}\n", encoding="utf-8")
        print("selected", winner, scores)
    elif args.phase == "standard":
        jobs = [("bench", "pp"), ("hyle",0), ("bench","tg"), ("hyle",4096), ("hyle",16384), ("hyle",30720)]
        for repeat in range(1, args.repeats + 1):
            for kind, value in jobs[repeat-1:] + jobs[:repeat-1]:
                if kind == "bench":
                    bench(args,args.selected,args.fa,value,repeat,"A")
                else:
                    native(args,"hyle",standard_schedule(args,value),"off",repeat,"A",depth=value)
    else:
        scenario = MATCHED.read_schedule(args.schedule)
        if len(scenario.prompts) != 8 or scenario.quotas != [64]*8:
            raise ValueError("M1-7a requires eight different prompts, 64 outputs each")
        if len({tuple(p) for p in scenario.prompts}) != 8:
            raise ValueError("Repeated prompts are not the real conversation workload")
        phase = "D" if args.phase == "memory" else "BC"
        jobs = [(arm,mode) for mode in args.modes for arm in args.arms]
        for repeat in range(1, args.repeats + 1):
            offset = (repeat-1) % len(jobs)
            for arm,mode in jobs[offset:] + jobs[:offset]:
                if arm == "upstream":
                    upstream_server(args,scenario,mode,repeat,phase,memory=phase == "D")
                else:
                    native(args,arm,scenario,mode,repeat,phase,memory=phase == "D")


if __name__ == "__main__":
    main()
