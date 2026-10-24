"""Existing-engine regression for simultaneous generated-prefix Text suffixes.

This explicitly runs a supplied local image and prepared weights. It never
builds, downloads, changes a runtime policy, or retries a failed model process.
Use any current local image for `prepare`, then `check` against the same case:

  python tests/tooling/qwen_text_prefix_batch_test.py prepare --image IMAGE \
    --inputs PREPARED --case .scratch/text-prefix-case
  python tests/tooling/qwen_text_prefix_batch_test.py check --image IMAGE \
    --inputs PREPARED --case .scratch/text-prefix-case --output .scratch/text-prefix-run

An old image is optional: add --expect-missing-family to check the historical
failure. These are correctness runs, not TPS tests.
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pin(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def write_turn(folder, name, ids, outputs):
    (folder / (name + ".ids")).write_text(
        " ".join(map(str, ids)) + "\n", encoding="utf-8")
    (folder / (name + ".schedule")).write_text(
        f"prompt {name}.ids\ngenerate {outputs}\n", encoding="utf-8")


def run(image, inputs, manifest, output, capacity):
    output.mkdir(parents=True, exist_ok=False)
    command = [str(image.resolve()), "--inputs", str(inputs.resolve()),
               "--session-schedules", str(manifest.resolve()),
               "--backend", "cuda", "--mtp", "off", "--sampling", "greedy",
               "--capacity", str(capacity), "--max-sessions", "2",
               "--active-sequences", "2", "--cache-bytes", "536870912",
               "--prefix-cache", "auto", "--progress", "direct",
               "--replay", "commands", "--trace-level", "decisions"]
    save(output / "command.json", {"command": command, "image": pin(image),
                                  "inputs": pin(inputs / "weights.tsv")})
    started = time.monotonic()
    with (output / "stdout.log").open("wb") as stdout, \
            (output / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        save(output / "live.json", {"pid": child.pid})
        timed_out = False
        try:
            code = child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait()
    process = {"exit": code, "timeout": timed_out,
               "wall_seconds": time.monotonic() - started}
    save(output / "process.json", process)
    require(not timed_out, "model process timed out; no automatic retry")
    reports = []
    for line in (output / "stdout.log").read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            reports.append(json.loads(line))
    if reports:
        require(len(reports) == 1, "expected one aggregate natural request result")
        save(output / "consumer-result.json", reports[0])
    return process, reports[0] if reports else None


def prepare(args):
    case = args.case.resolve()
    case.mkdir(parents=True, exist_ok=False)
    prompt = list(map(int, (args.inputs / "prompt.ids").read_text().split()))
    require(len(prompt) > 5, "regression needs an existing prompt longer than five tokens")
    for name in ("A", "D"):
        write_turn(case, name, prompt, 3)
    calibration = case / "calibration.sessions"
    calibration.write_text("A A.schedule\nD D.schedule\n", encoding="utf-8")
    capacity = len(prompt) + 16
    process, report = run(args.image, args.inputs, calibration,
                          case / "calibration", capacity)
    require(process["exit"] == 0 and report is not None,
            "R2 calibration failed; preserve its first failure")
    require(report["mtp"] is False and report["session_count"] == 2 and
            report["active_sequences"] == 2, "calibration did not execute ordinary R2")
    sessions = {entry["session_id"]: entry["result"] for entry in report["sessions"]}
    a, d = sessions["A"]["generated_ids"], sessions["D"]["generated_ids"]
    require(len(a) == len(d) == 3 and a == d,
            "identical real R2 inputs did not produce the same three nonterminal outputs")
    key = prompt + a[:2]  # The third emitted ID is pending, outside consumed KV.
    suffix = prompt[:3]
    require(len(key) + len(suffix) <= capacity, "case exceeds its actual context")
    for name in ("B", "C"):
        write_turn(case, name, key + suffix, 2)
    (case / "regression.sessions").write_text(
        "A A.schedule\nD D.schedule\nB B.schedule\nC C.schedule\n", encoding="utf-8")
    save(case / "case.json", {"capacity": capacity, "original_prompt_rows": len(prompt),
         "expected_key_rows": len(key), "suffix_rows": len(suffix),
         "calibration_generated_ids": a, "key": key, "suffix": suffix,
         "cold_original_shapes": [1, len(prompt), len(key) + len(suffix),
                                  len(key) + len(suffix) - len(prompt)],
         "weights": pin(args.inputs / "weights.tsv")})


def check(args):
    case = args.case.resolve()
    facts = json.loads((case / "case.json").read_text())
    require(pin(args.inputs / "weights.tsv") == facts["weights"],
            "prepared role manifest changed after calibration")
    require(facts["suffix_rows"] not in facts["cold_original_shapes"],
            "new suffix is already a preplanned original Text shape")
    process, report = run(args.image, args.inputs, case / "regression.sessions",
                          args.output.resolve(), facts["capacity"])
    if args.expect_missing_family:
        stderr = (args.output / "stderr.log").read_text(encoding="utf-8")
        require(process["exit"] != 0 and report is None and
                "text finite family has not been prepared" in stderr,
                "old image did not reproduce the exact missing-family failure")
        save(args.output / "assertions.json", {"status": "EXPECTED_FAILURE",
             "failure": "missing actual Text R2 suffix family", "retry": False})
        return
    require(process["exit"] == 0 and report is not None, "fixed natural request run failed")
    require(report["mtp"] is False and report["text_tensors"] > 0 and
            report["mtp_tensors"] == report["draft_program_graph_calls"] ==
            report["prefix_decision_graph_calls"] == 0, "ordinary run used draft resources")
    require(report["session_count"] == report["retired_requests"] == 4 and
            report["peak_committed_versions"] <= 2 and report["peak_resident_requests"] <= 2,
            "request retirement or finite N2 budget changed")
    require(report["prefix_hits"] == 2 and report["prefix_full_hits"] == 0 and
            report["reused_tokens"] == 2 * facts["expected_key_rows"],
            "both later requests must actually reuse the generated consumed prefix")
    batches = [batch for batch in report["batches"] if not batch["generation"] and
               batch["tokens"] == facts["suffix_rows"]]
    require(len(batches) == 1 and batches[0]["requests"] == 2 and
            set(batches[0]["session_ids"]) == {"B", "C"},
            "new suffix was padded, serialized, or not executed as one real R2 batch")
    sessions = {entry["session_id"]: entry["result"] for entry in report["sessions"]}
    require(sessions["A"]["generated_ids"] == sessions["D"]["generated_ids"] ==
            facts["calibration_generated_ids"], "original work changed from calibration")
    require(sessions["B"]["generated_ids"] == sessions["C"]["generated_ids"] and
            len(sessions["B"]["generated_ids"]) == 2, "new peer continuations differ")
    require(report["cold_exact_admissions"] >= 2 and
            report["cache_live_requested_bytes"] <= report["cache_requested_byte_limit"],
            "new bounded exact families were not admitted within the finite cache budget")
    save(args.output / "assertions.json", {"status": "PASS", "actual_suffix_batch": batches[0],
         "partial_hits": 2, "key_rows": facts["expected_key_rows"],
         "generated_ids": {name: value["generated_ids"] for name, value in sessions.items()},
         "retired_requests": report["retired_requests"], "draft_resources": False,
         "cache_budget_passed": True, "throughput_claim": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check"))
    parser.add_argument("--image", type=pathlib.Path, required=True)
    parser.add_argument("--inputs", type=pathlib.Path, required=True)
    parser.add_argument("--case", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--expect-missing-family", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "check":
        parser.error("check requires --output") if arguments.output is None else None
    (prepare if arguments.mode == "prepare" else check)(arguments)
