"""Offline accounting for the four frozen R29 guarded-native-wait diagnostics.

This program never launches workloads or waits for files. Run once after all
order-{0..3}.log results are ready. Every order is retained; no best run is selected.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCOPE = ROOT / ".scratch/inference-gaps/g09-native-wait-29"
SHAPES = ["independent-cpu-first", "independent-gpu-first", "two-microbatches", "serial-holdout"]
TREATMENT, CONTROL = "native-wait-1", "candidate-1"
HEAD = re.compile(r"# G09_OBSERVATION tag=(\S+) count=(\d+) capacity=65536 overflow=0 clock=QPC-ns")
RECORD = re.compile(r"# G09_RECORD tag=(\S+) ns=(\d+) kind=(\d+) run=(\d+) token=(\d+) value=(\d+) thread=(\d+)")
KINDS = {1, 2, 3, 11, 12, 20, 21, 30, 31, 40, 41, 50, 51, 52}


def identity(path):
    path = Path(path)
    try:
        name = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        name = str(path.resolve())
    return {"path": name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summary(v):
    if not v:
        return {"count": 0}
    med, mean = statistics.median(v), statistics.mean(v)
    return {"count": len(v), "minimum": min(v), "median": med, "mean": mean, "maximum": max(v),
            "median_absolute_deviation": statistics.median(abs(x - med) for x in v),
            "coefficient_of_variation": statistics.pstdev(v) / mean if mean else None}


def union(iv):
    out = []
    for lo, hi in sorted(iv):
        assert hi >= lo
        if out and lo <= out[-1][1]:
            out[-1][1] = max(hi, out[-1][1])
        else:
            out.append([lo, hi])
    return sum(hi - lo for lo, hi in out)


def parse_log(path, order):
    # Only decode/normalize line endings. No numeric fields are rewritten.
    text = path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    assert "\x00" not in text, "Unexpected NUL/mixed encoding; preserve raw file and audit its decoder"
    blocks = []
    for line in text.splitlines():
        if line.startswith("# G09_OBSERVATION tag="):
            match = HEAD.fullmatch(line)
            assert match, line
            blocks.append({"tag": match[1], "count": int(match[2]), "records": []})
        elif line.startswith("# G09_RECORD "):
            match = RECORD.fullmatch(line)
            assert match and blocks and match[1] == blocks[-1]["tag"], line
            record = dict(zip(["ns", "kind", "run", "token", "value", "thread"], map(int, match.groups()[1:])))
            assert record["kind"] in KINDS
            blocks[-1]["records"].append(record)
    assert len(blocks) == 8
    for shape in range(4):
        assert {b["tag"] for b in blocks[shape * 2:shape * 2 + 2]} == {TREATMENT, CONTROL}

    rows, native_waits, intervals_out, block_facts = [], [], [], []
    for bi, block in enumerate(blocks):
        records = block["records"]
        assert len(records) == block["count"]
        records.sort(key=lambda v: v["ns"])
        entries = [v for v in records if v["kind"] == 1]
        exits = [v for v in records if v["kind"] == 3]
        assert len(entries) == len(exits) == 22
        counts = Counter(v["kind"] for v in records)
        if block["tag"] == CONTROL or bi // 2 >= 2:
            assert all(counts[k] == 0 for k in (50, 51, 52)), "Intervention leaked into control/negative-control domain"
        assert counts[50] == counts[51] == counts[52]
        block_facts.append({"block": bi, "shape": SHAPES[bi // 2], "tag": block["tag"],
                            "source_records": len(records), "replays": 22,
                            "kind_counts": dict(sorted(counts.items()))})
        for ordinal, (entry, exit_) in enumerate(zip(entries, exits)):
            assert entry["run"] == exit_["run"] and entry["thread"] == exit_["thread"]
            assert entry["ns"] < exit_["ns"]
            if ordinal < 21:
                assert exit_["ns"] < entries[ordinal + 1]["ns"]
            phase = "measurement" if 9 <= ordinal <= 20 else "excluded-oracle-warmup"
            records_in = [v for v in records if entry["ns"] <= v["ns"] <= exit_["ns"]]
            pending, snapshots, intervals = {}, {}, defaultdict(list)
            polls, wait_entries = [], []
            for v in records_in:
                if v["kind"] == 52:
                    key = (v["run"], v["token"])
                    assert key not in snapshots
                    snap = {"qpc_ns": v["ns"], "pending_count": v["value"] & 0xFFFFFFFF,
                            "cpu_ready": v["value"] >> 32, "thread": v["thread"]}
                    assert snap["pending_count"] == 1 and snap["cpu_ready"] in (0, 1)
                    # CPU-ready is allowed: native wait releases pipeline_mutex,
                    # leaving the separately scheduled CPU lane able to progress.
                    snapshots[key] = snap
                elif v["kind"] in (11, 20, 30, 40, 50):
                    key = (v["kind"], v["run"], v["token"])
                    assert key not in pending
                    pending[key] = v
                    if v["kind"] == 50:
                        snap = snapshots.pop((v["run"], v["token"]))
                        assert snap["qpc_ns"] <= v["ns"]
                        wait_entries.append((v, snap))
                elif v["kind"] in (12, 21, 31, 41, 51):
                    kind = v["kind"] - 1
                    begin = pending.pop((kind, v["run"], v["token"]))
                    duration = v["ns"] - begin["ns"]
                    intervals[kind].append((begin["ns"], v["ns"]))
                    if kind == 11:
                        polls.append({"duration_ns": duration, "result": v["value"]})
                    elif kind in (20, 30, 50):
                        intervals_out.append({"block": bi, "ordinal": ordinal, "phase": phase,
                                              "kind": kind, "start_qpc_ns": begin["ns"], "end_qpc_ns": v["ns"],
                                              "duration_ns": duration, "run": v["run"], "token": v["token"],
                                              "entry_thread": begin["thread"], "exit_thread": v["thread"]})
                    if kind == 50:
                        wait_entry, snap = wait_entries.pop(0)
                        assert wait_entry is begin
                        assert v["value"] in (0, 1)
                        native_waits.append({"block": bi, "shape": SHAPES[bi // 2], "tag": block["tag"],
                                             "ordinal": ordinal, "phase": phase, "run": v["run"], "token": v["token"],
                                             "job_id": begin["value"], "guard_snapshot": snap,
                                             "start_qpc_ns": begin["ns"], "end_qpc_ns": v["ns"],
                                             "duration_ns": duration, "status_error": bool(v["value"]),
                                             "entry_thread": begin["thread"], "exit_thread": v["thread"]})
            assert not pending and not snapshots and not wait_entries
            duration = exit_["ns"] - entry["ns"]
            observed_wait = union(intervals[20] + intervals[30] + intervals[50])
            rows.append({"order": order, "block": bi, "shape": SHAPES[bi // 2], "tag": block["tag"],
                         "ordinal": ordinal, "phase": phase,
                         "start_qpc_ns": entry["ns"], "end_qpc_ns": exit_["ns"], "replay_ns": duration,
                         "timer_park_union_ns": union(intervals[20] + intervals[30]),
                         "frontier_timer_park_union_ns": union(intervals[30]),
                         "progress_timer_park_union_ns": union(intervals[20]),
                         "native_wait_union_ns": union(intervals[50]),
                         "all_observed_wait_union_ns": observed_wait,
                         "outside_observed_wait_elapsed_ns": duration - observed_wait,
                         "finished_wait_union_ns": union(intervals[40]),
                         "timer_park_count": len(intervals[20]) + len(intervals[30]),
                         "frontier_timer_park_count": len(intervals[30]), "progress_timer_park_count": len(intervals[20]),
                         "native_wait_count": len(intervals[50]), "poll_count": len(polls),
                         "poll_total_ns": sum(v["duration_ns"] for v in polls),
                         "pending_poll_count": sum(v["result"] == 0 for v in polls),
                         "complete_poll_count": sum(v["result"] == 1 for v in polls),
                         "error_poll_count": sum(v["result"] == 2 for v in polls)})

    shapes = []
    keys = ["replay_ns", "timer_park_union_ns", "native_wait_union_ns", "all_observed_wait_union_ns",
            "outside_observed_wait_elapsed_ns", "timer_park_count", "native_wait_count", "poll_count", "poll_total_ns"]
    for si, shape in enumerate(SHAPES):
        measured = [v for v in rows if v["shape"] == shape and v["phase"] == "measurement"]
        arms = {tag: [v for v in measured if v["tag"] == tag] for tag in (TREATMENT, CONTROL)}
        assert all(len(v) == 12 for v in arms.values())
        chronological, groups = sorted(measured, key=lambda v: v["start_qpc_ns"]), []
        for gi in range(6):
            group = chronological[4 * gi:4 * gi + 4]
            word = "".join("T" if v["tag"] == TREATMENT else "C" for v in group)
            assert word in ("TCCT", "CTTC")
            pair = {tag: [v for v in group if v["tag"] == tag] for tag in (TREATMENT, CONTROL)}
            diffs = {k: statistics.mean(v[k] for v in pair[CONTROL]) - statistics.mean(v[k] for v in pair[TREATMENT]) for k in keys}
            c_ns = statistics.mean(v["replay_ns"] for v in pair[CONTROL])
            t_ns = statistics.mean(v["replay_ns"] for v in pair[TREATMENT])
            groups.append({"group": gi, "order": word, "control_minus_treatment": diffs,
                           "treatment_divided_by_control_mean_replay_time": t_ns / c_ns,
                           "control_minus_treatment_replay_fraction": (c_ns - t_ns) / c_ns})
        selected_waits = [v for v in native_waits if v["shape"] == shape and v["phase"] == "measurement"]
        shapes.append({"shape": shape, "domain": "intervention" if si < 2 else "negative_control",
                       "arms": {tag: {k: summary([v[k] for v in arm]) for k in keys} for tag, arm in arms.items()},
                       "control_minus_treatment_mean": {k: statistics.mean(v[k] for v in arms[CONTROL]) - statistics.mean(v[k] for v in arms[TREATMENT]) for k in keys},
                       "native_waits": {"count": len(selected_waits),
                                        "duration_ns": summary([v["duration_ns"] for v in selected_waits]),
                                        "errors": sum(v["status_error"] for v in selected_waits),
                                        "cpu_ready_at_guard_count": sum(v["guard_snapshot"]["cpu_ready"] for v in selected_waits)},
                       "treatment_observed_in_measurement": bool(selected_waits),
                       "original_six_four_call_groups": groups,
                       "all_six_group_fraction_summary": summary([v["control_minus_treatment_replay_fraction"] for v in groups])})

    normal = len(re.findall(r"^# G09_OBSERVATION_PROCESS_EXIT code=0$", text, re.M)) == 1
    run_markers = re.findall(r"^# HYLE_G09_RUN .*", text, re.M)
    policy = re.findall(r"^# G09_OBSERVATION_ACTUAL_POLICY .*", text, re.M)
    instrument = re.findall(r"^# HYLE_G09_INSTRUMENT_RUN .*", text, re.M)
    assert len(run_markers) == len(policy) == len(instrument) == 1
    fields = lambda value: dict(re.findall(r"(\w+)=([^\s]*)", value))
    run_fields, policy_fields, instrument_fields = fields(run_markers[0]), fields(policy[0]), fields(instrument[0])
    expected_execution = "peer-first" if order & 1 else "candidate-first"
    expected_allocation = "peer-first" if order // 2 else "candidate-first"
    assert run_fields["execution_order"] == instrument_fields["execution_order"] == expected_execution
    assert run_fields["allocation_and_load_order"] == instrument_fields["allocation_order"] == expected_allocation
    assert policy_fields["nonce"] == run_fields["run_nonce"] == instrument_fields["run_nonce"]
    assert policy_fields["mask"] == "0x14" and policy_fields["priority"] == "32768"
    assert instrument_fields["caller_cpu"] == "2" and instrument_fields["affinity_ok"] == "1"
    assert instrument_fields["affinity_topology"] == "native-runtime-pair"
    assert run_fields["aa_route"] == "" and run_fields["verify_only"] == "0"
    assert run_fields["actual_paired_rounds"] == "12" and run_fields["exact_iterations"] == "1"
    measurement_markers = re.findall(r"^# G09_MEASUREMENT .*actual_paired_rounds=12.*$", text, re.M)
    assert len(measurement_markers) == 4
    modules = re.findall(r"^# G09_MODULE phase=(\S+) path=(.*)$", text, re.M)
    assert {phase for phase, _ in modules} == set(SHAPES)
    assert not any(re.search(r"graphics-hook|ToolsInjection64|nsight", path, re.I) for _, path in modules)
    oracle_before = len(re.findall(r"^# G09_ORACLE .* before=PASS .*", text, re.M))
    oracle_after = len(re.findall(r"^# G09_ORACLE .* after=PASS .*", text, re.M))
    retired = re.findall(r"^# G09_DEFERRED_DESTROY .*status=0$", text, re.M)
    return {"order": order, "input": identity(path), "source_records": sum(b["count"] for b in blocks),
            "run_markers": run_markers, "actual_policy_markers": policy,
            "verified_order": {"execution": expected_execution, "allocation": expected_allocation},
            "verified_nonce": policy_fields["nonce"], "verified_pid": int(policy_fields["pid"]),
            "module_phases": sorted({phase for phase, _ in modules}), "no_logged_nsight_or_obs": True,
            "normal_main0_marker": normal, "oracle_before_count": oracle_before,
            "oracle_after_count": oracle_after, "successful_destroy_markers": len(retired),
            "markers_complete": normal and oracle_before == oracle_after == 4 and len(retired) == 8 and len(run_markers) == len(policy) == 1,
            "native_wait_status_errors": sum(v["status_error"] for v in native_waits),
            "block_facts": block_facts, "shapes": shapes,
            "replays": rows, "native_waits": native_waits, "wait_intervals": intervals_out,
            "limits": ["Main/oracle/destroy markers are log evidence; actual process exit-code and frozen artifact verification belong to the launcher receipt.",
                       "All times are the recorder's QPC clock; no cross-clock mapping or independent GPU readiness is claimed.",
                       "Completion.wait includes native retirement tails. Finished waits overlap work and are never added to other wait costs.",
                       "Shape0/1 intervention is guarded to one accepted pending native prefix after ready-native admissions are serviced. CPU-ready may be true and is reported, not silently rejected."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=SCOPE / "run")
    parser.add_argument("--output", type=Path, default=HERE / "native-wait-analysis.json")
    args = parser.parse_args()
    receipt_path = args.run_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf8")) if receipt_path.is_file() else None
    orders = []
    for order in range(4):
        path = args.run_dir / f"order-{order}.log"
        if not path.is_file():
            orders.append({"order": order, "state": "MISSING_INPUT", "path": str(path)})
            continue
        try:
            value = parse_log(path, order)
            if receipt is not None:
                assert receipt["state"] == "COMPLETE_NONRANKING_CAUSAL_DIAGNOSTIC"
                processes = [v for v in receipt["processes"] if v["index"] == order]
                assert len(processes) == 1
                process = processes[0]
                assert process["exit_code"] == 0 and process["validation"] == "PASS"
                assert process["sha256"] == value["input"]["sha256"]
                assert process["nonce"] == value["verified_nonce"] and process["pid"] == value["verified_pid"]
                assert process["rival_first"] == order & 1 and process["allocation_first"] == order // 2
                value["launcher_receipt_match"] = True
                value["actual_launcher_exit_code"] = process["exit_code"]
            value["state"] = "ACCOUNTED" if value["markers_complete"] and not value["native_wait_status_errors"] else "INCOMPLETE_OR_FAILURE_MARKERS"
            orders.append(value)
        except Exception as exc:
            orders.append({"order": order, "state": "PARSER_REJECTED_INPUT", "input": identity(path),
                           "error": f"{type(exc).__name__}: {exc}"})
    accounted = all(v["state"] == "ACCOUNTED" for v in orders)
    exercised = accounted and all(s["treatment_observed_in_measurement"] for v in orders for s in v["shapes"][:2])
    comparisons = []
    if accounted:
        for si, shape in enumerate(SHAPES):
            comparisons.append({"shape": shape,
                                "all_four_orders": [{"order": v["order"],
                                                     "mean_replay_ns_control_minus_treatment": v["shapes"][si]["control_minus_treatment_mean"]["replay_ns"],
                                                     "mean_all_wait_ns_control_minus_treatment": v["shapes"][si]["control_minus_treatment_mean"]["all_observed_wait_union_ns"],
                                                     "mean_outside_wait_ns_control_minus_treatment": v["shapes"][si]["control_minus_treatment_mean"]["outside_observed_wait_elapsed_ns"],
                                                     "all_six_group_fraction_summary": v["shapes"][si]["all_six_group_fraction_summary"]} for v in orders],
                                "all24_group_difference_signs": dict(Counter(
                                    "less_elapsed_with_treatment" if group["control_minus_treatment"]["replay_ns"] > 0
                                    else "more_elapsed_with_treatment" if group["control_minus_treatment"]["replay_ns"] < 0 else "equal"
                                    for v in orders for group in v["shapes"][si]["original_six_four_call_groups"])),
                                "selection": "Allfour retained; no pooled qualification, best-order selection, threshold verdict or original P1/A/A rank."})
    known_inputs = [SCOPE / "launch-authorization.json", SCOPE / "source-manifest.json",
                    SCOPE / "build-reviewed-result.json", SCOPE / "source/invocation.cc",
                    SCOPE / "source/facade.cc", SCOPE / "source/driver.cc", SCOPE / "source/recorder.h",
                    HERE / "g09-native-wait-review.json"]
    if receipt_path.is_file():
        known_inputs.append(receipt_path)
    result = {"schema": "hyle.g09.guarded-native-wait-offline-accounting/v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "analyzer": identity(Path(__file__)),
              "state": "FOUR_ORDER_DIAGNOSTIC_ACCOUNTED" if accounted and exercised else "INCOMPLETE_OR_UNEXERCISED_DIAGNOSTIC",
              "actions": {"native_gpu_remote_benchmark_runs": 0, "product_registry_changes": 0,
                          "no_file_polling": True, "no_rank_or_qualification": True},
              "analyzer_preparse_correction": "First invocation used invalid codec spelling utf8-sig and rejected allfour inputs before parsing any record. Corrected to Python utf-8-sig; no record selection, numeric field or workload rerun changed.",
              "known_inputs": [identity(p) for p in known_inputs if p.is_file()],
              "frozen_semantics": {"treatment_tag": TREATMENT, "control_tag": CONTROL,
                                   "measurement_ordinals": [9, 20], "excluded_ordinals": [0, 1, 2, 3, 4, 5, 6, 7, 8, 21],
                                   "kind50_51": "same original completion.wait entry/exit, not raw GPU readiness",
                                   "kind52": "low32 pending==1; high32 CPU-ready boolean; ready CPU is permitted",
                                   "negative_controls": "two-microbatches/serial-holdout, both arms zero kinds50/51/52 throughout all22replays",
                                   "difference_direction": "control minus treatment; positive replay difference means less diagnostic elapsed time with treatment",
                                   "groups": "Original chronological four-call ABBA/BAAB groups; six per shape, every order reported"},
              "orders": orders, "comparisons_all_four_orders": comparisons,
              "limits": ["Small diagnostic sample; no original workload A/A, P1, rank, numerical mechanism correctness or complete pipeline qualification is issued.",
                         "A treatment change can identify this guarded waiting path as an influential mechanism without establishing every original A/A failure's cause.",
                         "Negative controls and all four orders must be assessed together. No favorable order or subset may be used to sign recovery.",
                         "The diagnostic does not separate GPU queue/power/other-process effects from host scheduling, or measure independent readiness-to-observation latency."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output": str(args.output), "state": result["state"],
                      "order_states": [{"order": v["order"], "state": v["state"]} for v in orders]}))


if __name__ == "__main__":
    main()
