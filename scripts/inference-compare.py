#!/usr/bin/env python3
r"""Compare ordinary or K1 MTP Qwen3.5 inference in fresh, interleaved processes.

The reference must implement the matched-input native consumer CLI (--model,
--inputs, --steps/--schedule, --capacity, --mtp, --memory-snapshots); it is not
llama-cli. Inputs and weights must already exist locally. No build, download,
extra warmup run, profiler, memory sampler, or automatic retry is started.
MTP is off by default; --mtp on requires a reference GGUF containing its MTP weights.

Example (add --dry-run to inspect without starting model processes):
  python scripts/inference-compare.py --hyle BUILD/bin/hyle.exe \
    --reference CACHE/peer.exe --model CACHE/model.gguf \
    --inputs CACHE/hyle-inputs --reference-inputs CACHE/peer-inputs \
    --schedule CACHE/schedule.txt --hyle-version COMMIT --reference-version PIN \
    --output .scratch/ordinary-comparison

Schedules contain alternating ``prompt IDSFILE`` / ``generate COUNT`` lines.
Their round count and quotas are read from the file. --steps controls only the
single-round scenario. Each scenario uses independent processes; all rounds in
a schedule share one loaded model/session. Raw logs and summaries are text only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EOS = {248044, 248046}  # The Qwen3.5 matched-input consumer protocol.


@dataclass(frozen=True)
class Arm:
    name: str
    executable: Path
    version: str
    runtime: Path | None
    hyle: bool


@dataclass(frozen=True)
class Scenario:
    name: str
    prompts: list[list[int]]
    quotas: list[int]
    schedule: Path | None = None


def read_ids(path: Path) -> list[int]:
    values = [int(value) for value in path.read_text(encoding="utf-8-sig").split()]
    if not values or any(value < 0 for value in values):
        raise ValueError(f"Empty or negative token IDs: {path}")
    return values


def read_schedule(path: Path) -> Scenario:
    prompts, quotas = [], []
    pending = None
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Invalid schedule line {path}:{number}")
        kind, value = fields
        if kind == "prompt" and pending is None:
            ids_path = (path.parent / value.strip('"')).resolve()
            pending = read_ids(ids_path)
        elif kind == "generate" and pending is not None:
            quota = int(value)
            if quota < 1:
                raise ValueError(f"Nonpositive schedule quota {path}:{number}")
            prompts.append(pending)
            quotas.append(quota)
            pending = None
        else:
            raise ValueError(f"Expected alternating prompt/generate: {path}:{number}")
    if pending is not None or not prompts:
        raise ValueError(f"Incomplete or empty schedule: {path}")
    return Scenario("schedule", prompts, quotas, path)


def interleaving(arms: list[Arm], repeats: int, offset: int = 0) -> list[Arm]:
    """Rotate the leading arm each independent round; never run concurrently."""
    return [arms[(index + repeat + offset) % len(arms)]
            for repeat in range(repeats) for index in range(len(arms))]


def command(arm: Arm, scenario: Scenario, args: argparse.Namespace) -> list[str]:
    if arm.hyle:
        result = [str(arm.executable), "--inputs", str(args.inputs),
                  "--capacity", str(args.capacity), "--max-sessions", "1",
                  "--active-sequences", "1", "--prefix-cache", "off",
                  "--progress", "direct", "--replay", "auto", "--mtp", args.mtp,
                  "--backend", "cuda", "--sampling", "greedy", "--decode-overlap", "auto"]
        if args.mtp == "on":
            result += ["--mtp-max-drafts", "1"]
    else:
        result = [str(arm.executable), "--model", str(args.model),
                  "--inputs", str(args.reference_inputs), "--capacity", str(args.capacity),
                  "--mtp", args.mtp, "--memory-snapshots", "off"]
    if scenario.schedule:
        result += ["--schedule", str(scenario.schedule)]
    else:
        result += ["--steps", str(scenario.quotas[0])]
    return result


def environment(arm: Arm) -> dict[str, str]:
    result = os.environ.copy()
    if arm.runtime:
        receipt = json.loads(arm.runtime.read_text(encoding="utf-8-sig"))
        if not isinstance(receipt.get("PATH"), str) or not receipt["PATH"]:
            raise ValueError(f"Runtime receipt has no PATH: {arm.runtime}")
        result["PATH"] = receipt["PATH"]
    result.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    return result


def positive(report: dict, key: str, *, zero: bool = False) -> float:
    value = report.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(f"Missing/invalid complete endpoint {key}={value!r}")
    return float(value)


def split_startup_decode_warmup(stderr: str) -> tuple[dict, str]:
    fields = {
        "seconds": ("startup_decode_warmup_seconds", float),
        "graph_calls": ("startup_decode_warmup_graph_calls", int),
        "target_rows": ("startup_decode_warmup_target_rows", int),
        "capture_recorded": ("startup_decode_warmup_captures", int),
        "capture_associated_cold_prep_seconds":
            ("startup_decode_warmup_capture_associated_cold_prep_seconds", float),
    }
    metrics = {name: convert(0) for name, convert in fields.values()}
    active, seen = False, False
    episode_lines = []
    for line in stderr.splitlines():
        if re.match(r"^decode_warmup_begin(?:\s|$)", line):
            if active or seen:
                raise ValueError("Duplicate startup decode warmup begin marker")
            active, seen = True, True
        elif re.match(r"^decode_warmup_end(?:\s|$)", line):
            if not active:
                raise ValueError("Startup decode warmup end without begin marker")
            observed = dict(re.findall(r"(\w+)=([^\s]+)", line))
            for key, (name, convert) in fields.items():
                if key not in observed:
                    raise ValueError(f"Missing startup decode warmup field: {key}")
                value = convert(observed[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid startup decode warmup field: {key}")
                metrics[name] = value
            active = False
        elif not active:
            episode_lines.append(line)
    if active:
        raise ValueError("Startup decode warmup begin without end marker")
    return metrics, "\n".join(episode_lines)


def capture_rounds(stderr: str, count: int) -> list[dict]:
    # Startup preparation can emit ordinary phase/prepared/decode records. None
    # of those records belongs to the first real request or advances its turn.
    _, stderr = split_startup_decode_warmup(stderr)
    result = [dict(prepared_lines=0, capture_recorded=0, capture_reused=0,
                   capture_associated_cold_prep_seconds=0.0,
                   prefill_prepared_lines=0, prefill_captures=0, prefill_capture_reused=0,
                   prefill_capture_associated_cold_prep_seconds=0.0,
                   decode_prepared_lines=0, decode_captures=0, decode_capture_reused=0,
                   decode_capture_associated_cold_prep_seconds=0.0,
                   warmup_complete=0, warmup_reset=0)
              for _ in range(count)]
    turn, stage = -1, None
    for line in stderr.splitlines():
        phase = re.search(r"\bmemory_phase=(\S+)", line)
        if phase:
            name = phase[1]
            explicit = re.fullmatch(r"round(\d+)_(prefill|decode)_(?:begin|end)", name)
            if explicit:
                turn = int(explicit[1]) - 1
                stage = explicit[2]
            else:
                generation = re.search(r"_(prefill|decode)_(begin|end)$", name)
                if generation:
                    stage = generation[1]
                    if generation.groups() == ("prefill", "begin"):
                        turn += 1
        if not 0 <= turn < count:
            continue
        current = result[turn]
        if line.startswith("prepared "):
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            if "capture_recorded" in fields and "capture_reused" in fields:
                recorded, reused = int(fields["capture_recorded"]), int(fields["capture_reused"])
                cold = float(fields["cold_batch_seconds"])
                if min(recorded, reused, cold) < 0 or not math.isfinite(cold):
                    raise ValueError("Invalid capture diagnostic")
                current["prepared_lines"] += 1
                current["capture_recorded"] += recorded
                current["capture_reused"] += reused
                if recorded:
                    current["capture_associated_cold_prep_seconds"] += cold
                if stage in ("prefill", "decode"):
                    current[f"{stage}_prepared_lines"] += 1
                    current[f"{stage}_captures"] += recorded
                    current[f"{stage}_capture_reused"] += reused
                    if recorded:
                        current[f"{stage}_capture_associated_cold_prep_seconds"] += cold
        if "CUDA graph warmup complete" in line:
            current["warmup_complete"] += 1
        if "CUDA graph warmup reset" in line:
            current["warmup_reset"] += 1
    for current in result:
        if not current["prepared_lines"]:
            for key in ("capture_recorded", "capture_reused", "capture_associated_cold_prep_seconds"):
                current[key] = None
        for stage in ("prefill", "decode"):
            if not current[f"{stage}_prepared_lines"]:
                for key in (f"{stage}_captures", f"{stage}_capture_reused",
                            f"{stage}_capture_associated_cold_prep_seconds"):
                    current[key] = None
    return result


def parse_report(stdout: str, stderr: str, arm: Arm, scenario: Scenario,
                 capacity: int, wall: float, expected_mtp: bool = False) -> dict:
    # Hyle emits one complete final object; the native reference emits a pretty
    # printed object. Preserve the producer's stdout and never rewrite its JSON.
    raw = json.loads(stdout.lstrip("\ufeff"))
    if raw.get("mtp") is not expected_mtp:
        raise ValueError(f"Expected producer report with mtp={expected_mtp}")
    if arm.hyle:
        if len(raw.get("sessions", [])) != 1:
            raise ValueError("Expected one resident Hyle session")
        detail = raw["sessions"][0]["result"]
        if not expected_mtp:
            for field in ("mtp_tensors", "draft_program_graph_calls", "draft_vocabulary_rows"):
                if raw.get(field, 0) != 0:
                    raise ValueError(f"Unexpected ordinary draft work: {field}")
        elif raw.get("mtp_max_drafts") != 1:
            raise ValueError("Expected the requested Hyle K1 MTP producer")
    else:
        detail = raw
    turns, generated = detail["turns"], detail["generated_ids"]
    if len(turns) != len(scenario.prompts) or not generated:
        raise ValueError("Actual turn/output count differs from requested schedule")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in generated):
        raise ValueError("Invalid generated IDs")
    for field, expected in (("generated_tokens", len(generated)),
                            ("generated_decode_tokens", len(generated) - len(turns)),
                            ("turn_count", len(turns)), ("session_count", 1)):
        if field in raw and raw[field] != expected:
            raise ValueError(f"Complete {field} differs from turn ledger")
    for field in ("capacity", "requested_context", "context_capacity"):
        if field in raw and raw[field] != capacity:
            raise ValueError(f"Actual {field} differs from requested capacity")
    metrics = {key: positive(raw, key) for key in
               ("prefill_seconds", "session_seconds", "startup_to_ready_seconds",
                "startup_to_endpoint_seconds", "first_token_elapsed_seconds")}
    metrics["decode_seconds"] = positive(raw, "decode_seconds", zero=True)
    if len(generated) > len(turns) and metrics["decode_seconds"] == 0:
        raise ValueError("Generated decode outputs require positive complete decode_seconds")
    if not math.isclose(metrics["prefill_seconds"] + metrics["decode_seconds"],
                        metrics["session_seconds"], rel_tol=1e-5, abs_tol=2e-6):
        raise ValueError("Complete prefill+decode does not partition session")
    if not (metrics["startup_to_ready_seconds"] <= metrics["first_token_elapsed_seconds"]
            <= metrics["startup_to_endpoint_seconds"] + 2e-6):
        raise ValueError("Ready/first-token/last-token endpoints are out of order")
    load = raw.get("weights_load_seconds")
    if load is None:
        found = re.search(r"^weights_load_seconds=([^\s]+)", stderr, re.MULTILINE)
        load = float(found[1]) if found else None
    if load is not None and (not math.isfinite(load) or not 0 <= load <= metrics["startup_to_ready_seconds"]):
        raise ValueError("Invalid weights-load endpoint")
    metrics["weights_load_seconds"] = load
    metrics["parent_process_wall_seconds"] = wall
    startup_warmup, episode_stderr = split_startup_decode_warmup(stderr)
    metrics.update(startup_warmup)
    captures = capture_rounds(episode_stderr, len(turns))
    observed, context, output_at = [], 0, 0
    for index, (turn, prompt, quota, capture) in enumerate(zip(turns, scenario.prompts, scenario.quotas, captures)):
        begin, end = turn["output_begin"], turn["output_end"]
        values = generated[begin:end]
        if (turn["turn"] != index + 1 or begin != output_at or not 1 <= len(values) <= quota
                or turn["requested_steps"] != quota):
            raise ValueError(f"Turn {index + 1} output/quota ledger differs")
        if "generated_ids" in turn and turn["generated_ids"] != values:
            raise ValueError(f"Turn {index + 1} generated IDs differ from complete output")
        if any(value in EOS for value in values[:-1]) or (len(values) < quota and values[-1] not in EOS):
            raise ValueError(f"Turn {index + 1} unexplained truncation or generation after EOS")
        inputs = ([generated[begin - 1]] if index else []) + prompt
        if turn["input_ids"] != inputs or turn["context_before"] != context:
            raise ValueError(f"Turn {index + 1} actual input/history differs")
        if bool(turn["pending_flushed"]) != bool(index):
            raise ValueError(f"Turn {index + 1} pending-flush ledger differs")
        prefill = positive(turn, "prefill_seconds")
        decode = positive(turn, "decode_seconds", zero=len(values) == 1)
        target_rows = len(inputs) + len(values) - 1
        context += target_rows
        if turn["context_after"] != context or context > capacity:
            raise ValueError(f"Turn {index + 1} context/target-row ledger differs")
        if not expected_mtp and "target_decode_rows" in turn and turn["target_decode_rows"] != len(values) - 1:
            raise ValueError(f"Turn {index + 1} ordinary target decode-row ledger differs")
        observed.append(dict(turn=index + 1, outputs=len(values), generated_ids=values, input_rows=len(inputs),
                             target_rows=target_rows, prefill_seconds=prefill, decode_seconds=decode,
                             prefill_tokens_per_second=len(inputs) / prefill,
                             decode_tokens_per_second=(len(values) - 1) / decode if decode else None,
                             capture=capture))
        if expected_mtp:
            observed[-1]["producer_work"] = producer_work(turn)
        output_at = end
    if output_at != len(generated):
        raise ValueError("Generated IDs outside actual turn output ranges")
    for field in (("committed_target_rows", "processed_tokens") if expected_mtp else
                  ("actual_target_rows", "committed_target_rows", "processed_tokens")):
        if field in raw and raw[field] != context:
            raise ValueError(f"Complete {field} differs from turn ledger")
    if "prefill_target_rows" in raw and raw["prefill_target_rows"] != sum(turn["input_rows"] for turn in observed):
        raise ValueError("Complete prefill_target_rows differs from turn ledger")
    if expected_mtp:
        work = producer_work(raw)
        for field in ("actual_target_rows", "actual_draft_rows"):
            if field not in work:
                raise ValueError(f"MTP producer did not report {field}")
        if work["actual_target_rows"] < context:
            raise ValueError("Actual MTP target rows are below the committed turn context")
        metrics.update({f"producer_{key}": value for key, value in work.items()})
        # Extra execution includes rejected proposals and discarded successors;
        # it is independent of the number of rejected draft proposals.
        metrics["derived_extra_target_rows_actual_minus_committed"] = work["actual_target_rows"] - context
        proposals = work.get("proposals", work.get("draft_rounds"))
        if proposals is None and all("proposals" in turn["producer_work"] for turn in observed):
            proposals = sum(turn["producer_work"]["proposals"] for turn in observed)
            metrics["proposals_sum_of_producer_turns"] = proposals
        accepted = work.get("accepted_drafts", work.get("consumed_drafts"))
        if proposals is not None and accepted is not None:
            if accepted > proposals:
                raise ValueError("Accepted MTP drafts exceed producer proposal count")
            metrics["derived_rejected_drafts_proposals_minus_accepted"] = proposals - accepted
    decode_outputs = len(generated) - len(turns)
    metrics.update(full_decode_tokens_per_second=decode_outputs / metrics["decode_seconds"]
                   if metrics["decode_seconds"] else None,
                   process_output_tokens_per_second=len(generated) / wall,
                   prefill_tokens_per_second=sum(turn["input_rows"] for turn in observed) / metrics["prefill_seconds"],
                   new_user_prefill_tokens_per_second=sum(map(len, scenario.prompts)) / metrics["prefill_seconds"],
                   session_output_tokens_per_second=len(generated) / metrics["session_seconds"],
                   cold_output_tokens_per_second=len(generated) / metrics["startup_to_endpoint_seconds"])
    return dict(metrics=metrics, turns=observed, generated_ids=generated, mtp=expected_mtp,
                target_rows=context, source_id=raw.get("source_id"),
                actual_draft_rows_scope=raw.get("actual_draft_rows_scope"))


def producer_work(report: dict) -> dict[str, int]:
    """Preserve only exported work counters; never substitute logical context."""
    fields = ("actual_target_rows", "committed_target_rows", "prefill_target_rows", "verified_target_rows",
              "target_decode_rows", "actual_draft_rows", "prefill_draft_rows", "draft_rows", "draft_decode_rows",
              "draft_rounds", "proposals", "agreements", "accepted_drafts", "consumed_drafts", "rejected_drafts",
              "extra_target_rows", "target_compute_calls", "draft_compute_calls",
              "overlap_successor_submissions", "overlap_successor_discards")
    result = {}
    for key in fields:
        if key in report:
            value = report[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid producer work counter {key}={value!r}")
            result[key] = value
    return result


def distribution(values: list) -> str:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return "unknown"
    return (f"n={len(numbers)} median={statistics.median(numbers):.9g} "
            f"range=[{min(numbers):.9g},{max(numbers):.9g}] samples={numbers}")


def id_comparison(left: list[int], right: list[int]) -> str:
    common = next((index for index, (a, b) in enumerate(zip(left, right)) if a != b),
                  min(len(left), len(right)))
    return (f"equal={left == right} lengths={len(left)}/{len(right)} "
            f"common_prefix={common} first_difference_1based="
            f"{common + 1 if left != right else 'none'}")


def summarize(records: list[dict], arms: list[Arm], scenarios: list[Scenario],
              mtp: bool = False) -> tuple[str, bool]:
    lines = [f"{'MTP K1' if mtp else 'Ordinary'} inference: full native endpoints; independent interleaved processes.",
             "Decode TPS uses full continuous decode intervals; explicit startup warmup remains in startup/cold TPS and parent wall.",
             "No decode_warmup markers means this binary/invocation did not execute that explicit startup stage; its fields are zero.",
             "Capture-associated cold prep sums the cold_batch_seconds of capture_recorded steps.",
             "These are preparation spans, not pure capture time or removable wall time; prefill/decode are separate.",
             "Reference capture duration/count unknown unless producer exports them; warmup log counts are separate.",
             "Process output TPS uses parent wall, including launch, report publication and teardown; cold TPS uses native startup-to-endpoint.",
             "Natural cross-engine IDs may differ; equal prompt/quota does not establish equal follow-up work."]
    if mtp:
        lines.append("Turn target_rows are committed logical rows; producer_ counters preserve actual exported work. "
                     "Derived extra target rows include discarded successors; rejected drafts use producer proposal/round count minus accepted/consumed drafts.")
    valid = True
    baseline = "old_hyle" if any(arm.name == "old_hyle" for arm in arms) else "reference"
    for scenario in scenarios:
        representatives = {}
        for arm in arms:
            group = [row for row in records if row["scenario"] == scenario.name and row["arm"] == arm.name]
            if not group:
                continue
            first = group[0]["observation"]
            stable = all(row["observation"]["generated_ids"] == first["generated_ids"] for row in group)
            valid = valid and stable
            representatives[arm.name] = first
            lines.append(f"\n{scenario.name} {arm.name}: processes={len(group)} IDs_within_arm_stable={stable}")
            for index in range(len(scenario.prompts)):
                turns = [row["observation"]["turns"][index] for row in group]
                prefix = f"  turn{index + 1}"
                lines.append(f"{prefix} IDs_within_arm_stable="
                             f"{all(turn['generated_ids'] == turns[0]['generated_ids'] for turn in turns)}")
                for key in ("outputs", "input_rows", "target_rows", "prefill_seconds", "decode_seconds",
                            "prefill_tokens_per_second", "decode_tokens_per_second"):
                    lines.append(f"{prefix} {key}: {distribution([turn[key] for turn in turns])}")
                if mtp:
                    for key in sorted(set().union(*(turn["producer_work"] for turn in turns))):
                        lines.append(f"{prefix} producer_{key}: "
                                     f"{distribution([turn['producer_work'].get(key) for turn in turns])}")
                for key in ("prefill_captures", "prefill_capture_associated_cold_prep_seconds",
                            "decode_captures", "decode_capture_associated_cold_prep_seconds",
                            "capture_recorded", "capture_reused", "capture_associated_cold_prep_seconds",
                            "warmup_complete", "warmup_reset"):
                    lines.append(f"{prefix} {key}: {distribution([turn['capture'][key] for turn in turns])}")
            for key in first["metrics"]:
                lines.append(f"  {key}: {distribution([row['observation']['metrics'][key] for row in group])}")
            if mtp:
                lines.append(f"  producer_actual_draft_rows_scope={first['actual_draft_rows_scope'] or 'not exported'}")
        if baseline in representatives:
            for name, first in representatives.items():
                if name != baseline:
                    anchor = representatives[baseline]
                    lines.append(f"{scenario.name} {name} IDs_vs_{baseline}: "
                                 + id_comparison(first["generated_ids"], anchor["generated_ids"]))
                    for left, right in zip(first["turns"], anchor["turns"]):
                        lines.append(f"{scenario.name} {name} turn{left['turn']} IDs_vs_{baseline}: "
                                     + id_comparison(left["generated_ids"], right["generated_ids"]))
                    if name == "hyle" and baseline == "old_hyle":
                        valid = valid and first["generated_ids"] == anchor["generated_ids"]
    return "\n".join(lines) + "\n", valid


def execute(folder: Path, arm: Arm, scenario: Scenario, args: argparse.Namespace) -> dict:
    folder.mkdir()
    argv = command(arm, scenario, args)
    runtime_environment = environment(arm)
    (folder / "command.txt").write_text(subprocess.list2cmdline(argv) + "\n", encoding="utf-8")
    status = "launch_failed"
    stamp = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        with (folder / "stdout.log").open("wb") as out, (folder / "stderr.log").open("wb") as err:
            start = time.perf_counter()
            result = subprocess.run(argv, cwd=ROOT, env=runtime_environment, stdout=out, stderr=err,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=args.timeout)
        status = f"exit={result.returncode}"
    except subprocess.TimeoutExpired:
        status = f"timeout_seconds={args.timeout}; child killed and waited; no retry"
        raise
    finally:
        wall = time.perf_counter() - start
        (folder / "result.txt").write_text(f"{status} wall_seconds={wall:.9f} start_utc={stamp}\n", encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Consumer exit {result.returncode}; raw attempt retained: {folder}")
    observation = parse_report((folder / "stdout.log").read_text(encoding="utf-8-sig"),
                               (folder / "stderr.log").read_text(encoding="utf-8", errors="replace"),
                               arm, scenario, args.capacity, wall, args.mtp == "on")
    return dict(arm=arm.name, scenario=scenario.name, observation=observation)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hyle", type=Path, required=True, help="Current Hyle executable")
    parser.add_argument("--reference", type=Path, required=True, help="Matched-input native reference consumer executable")
    parser.add_argument("--model", type=Path, required=True, help="Existing reference GGUF; Hyle weights are selected by --inputs")
    parser.add_argument("--mtp", choices=("off", "on"), default="off",
                        help="Greedy K1 MTP; --model must contain MTP weights when on (default: off)")
    parser.add_argument("--inputs", type=Path, required=True, help="Existing Hyle inputs directory")
    parser.add_argument("--reference-inputs", type=Path, help="Reference prompt directory; defaults to --inputs")
    parser.add_argument("--schedule", type=Path, help="Existing prompt/generate schedule, any positive round count")
    parser.add_argument("--scenario", choices=("single", "schedule", "both"), default="both")
    parser.add_argument("--steps", type=int, default=64, help="Single-round output budget only (default: 64)")
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3, help="Independent processes per arm/scenario; at least 3")
    parser.add_argument("--old-hyle", type=Path, help="Optional old Hyle executable as same-window correctness/performance anchor")
    parser.add_argument("--hyle-version", required=True, help="Explicit source/build identity; never inferred from workspace HEAD")
    parser.add_argument("--reference-version", required=True, help="Explicit reference source/build/patch identity")
    parser.add_argument("--old-version", help="Required with --old-hyle")
    parser.add_argument("--hyle-runtime", type=Path, help="Existing runtime receipt: only its PATH is applied")
    parser.add_argument("--reference-runtime", type=Path, help="Existing runtime receipt: only its PATH is applied")
    parser.add_argument("--old-runtime", type=Path, help="Old Hyle runtime receipt; defaults to --hyle-runtime")
    parser.add_argument("--output", type=Path, required=True, help="New directory below repository .scratch; existing attempts are preserved")
    parser.add_argument("--timeout", type=float, default=900, help="Seconds per process; failure is never retried")
    parser.add_argument("--dry-run", action="store_true", help="Read inputs and print plan; no directory creation or model process")
    args = parser.parse_args()
    if args.repeats < 3 or args.steps < 1 or args.capacity < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Need repeats>=3, positive steps/capacity/timeout")
    if bool(args.old_hyle) != bool(args.old_version) or (args.old_runtime and not args.old_hyle):
        parser.error("--old-hyle and --old-version must be supplied together; --old-runtime requires them")
    if args.scenario != "single" and not args.schedule:
        parser.error("--schedule is required for --scenario schedule/both")
    args.reference_inputs = args.reference_inputs or args.inputs
    for key in ("hyle", "reference", "model", "inputs", "reference_inputs", "schedule", "old_hyle",
                "hyle_runtime", "reference_runtime", "old_runtime", "output"):
        value = getattr(args, key)
        if value:
            setattr(args, key, value.resolve())
    return args


def main() -> int:
    args = arguments()
    if not args.output.is_relative_to((ROOT / ".scratch").resolve()):
        raise ValueError("Output must remain below the repository .scratch directory")
    if args.output.exists():
        raise ValueError(f"Preserve existing attempt; choose a new --output: {args.output}")
    arms = [Arm("hyle", args.hyle, args.hyle_version, args.hyle_runtime, True),
            Arm("reference", args.reference, args.reference_version, args.reference_runtime, False)]
    if args.old_hyle:
        arms.append(Arm("old_hyle", args.old_hyle, args.old_version, args.old_runtime or args.hyle_runtime, True))
    for path in (args.model, *(arm.executable for arm in arms)):
        if not path.is_file():
            raise ValueError(f"Missing cached file; no build/download attempted: {path}")
    prompt = read_ids(args.inputs / "prompt.ids")
    if read_ids(args.reference_inputs / "prompt.ids") != prompt:
        raise ValueError("Hyle and reference initial prompt IDs differ")
    scenarios = []
    if args.scenario != "schedule":
        scenarios.append(Scenario("single", [prompt], [args.steps]))
    if args.scenario != "single":
        scenarios.append(read_schedule(args.schedule))
    for scenario in scenarios:
        if sum(map(len, scenario.prompts)) + sum(scenario.quotas) - 1 > args.capacity:
            raise ValueError(f"{scenario.name}: complete requested inputs/outputs exceed capacity")
    plan = [f"UTC={datetime.now(timezone.utc).isoformat()} platform={platform.platform()} Python={platform.python_version()}",
            f"mode={'MTP K1' if args.mtp == 'on' else 'ordinary'}; no download/build/retry/extra warmup/profile/memory observer. GPU concurrency=1.",
            f"capacity={args.capacity} repeats={args.repeats} root={ROOT}"]
    for arm in arms:
        environment(arm)  # Validate each receipt before launching any process.
        info = arm.executable.stat()
        plan.append(f"{arm.name} version={arm.version} executable={arm.executable} bytes={info.st_size} mtime_ns={info.st_mtime_ns} runtime_PATH_receipt={arm.runtime or 'inherited'}")
    for offset, scenario in enumerate(scenarios):
        plan.append(f"{scenario.name}: rounds={len(scenario.prompts)} prompt_rows={list(map(len, scenario.prompts))} quotas={scenario.quotas} schedule={scenario.schedule}")
        plan.append("order=" + ",".join(arm.name for arm in interleaving(arms, args.repeats, offset)))
        for arm in arms:
            plan.append(subprocess.list2cmdline(command(arm, scenario, args)))
    plan_text = "\n".join(plan) + "\n"
    print(plan_text, flush=True)
    if args.dry_run:
        return 0
    args.output.mkdir(parents=True)
    (args.output / "plan.txt").write_text(plan_text, encoding="utf-8")
    records = []
    try:
        for offset, scenario in enumerate(scenarios):
            for ordinal, arm in enumerate(interleaving(arms, args.repeats, offset), 1):
                folder = args.output / f"{scenario.name}-{ordinal:02d}-{arm.name}"
                print(f"{folder.name}: START", flush=True)
                record = execute(folder, arm, scenario, args)
                records.append(record)
                observation = record["observation"]
                print(f"{folder.name}: exit0 target_rows={observation['target_rows']} "
                      f"decode_seconds={[turn['decode_seconds'] for turn in observation['turns']]}", flush=True)
    except BaseException as error:
        text, _ = summarize(records, arms, scenarios, args.mtp == "on")
        (args.output / "endpoints.txt").write_text("INCOMPLETE WINDOW\n" + text, encoding="utf-8")
        (args.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\nNo retry. All attempts retained.\n", encoding="utf-8")
        raise
    text, valid = summarize(records, arms, scenarios, args.mtp == "on")
    (args.output / "endpoints.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    if not valid:
        print("FAILED: within-arm IDs unstable or current/old Hyle IDs differ; raw samples retained.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"inference-compare: {error}", file=sys.stderr)
        raise SystemExit(1)
