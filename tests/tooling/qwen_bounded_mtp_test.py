"""Black-box CUDA K2/K3 cold-boundary check; uses only supplied local weights.

Run with --hyle EXE --inputs PREPARED_TINY_GA_DIR --output FRESH_SCRATCH_DIR.
The original tiny fixture must support C32 and token IDs1..8. Inherit the
engine's normal runtime environment. These runs check behavior, not speed.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess


def check(args):
    args.output.mkdir(parents=True)
    for width in (2, 3):
        inventories = []
        for name, later, quota in (("short", [4], 7), ("long", [4,5,6,7,8], 4)):
            folder = args.output / f"K{width}-{name}"
            folder.mkdir()
            (folder / "first.ids").write_text("1 2 3\n", encoding="utf-8")
            (folder / "later.ids").write_text(" ".join(map(str,later))+"\n", encoding="utf-8")
            schedule = folder / "schedule.txt"
            schedule.write_text(f"prompt first.ids\ngenerate 1\nprompt later.ids\ngenerate {quota}\n", encoding="utf-8")
            command = [str(args.hyle), "--inputs", str(args.inputs), "--backend", "cuda",
                       "--capacity", "32", "--mtp", "on", "--mtp-max-drafts", str(width),
                       "--diagnostic-first-turn-only", "1", "--schedule", str(schedule)]
            (folder / "command.txt").write_text(subprocess.list2cmdline(command)+"\n", encoding="utf-8")
            with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
                result = subprocess.run(command,stdout=stdout,stderr=stderr,timeout=180)
            assert result.returncode == 0, folder
            raw = json.loads((folder / "stdout.log").read_text(encoding="utf-8"))
            stderr = (folder / "stderr.log").read_text(encoding="utf-8")
            assert raw["cold_planning"] == "configured_capacity" and raw["future_shape_hints"] == 0
            assert raw["mtp_max_drafts"] == width and raw["live_committed_versions"] == 0
            assert raw["sampling"] == "greedy" and raw["draft_program_graph_calls"] > 0
            detail = raw["sessions"][0]["result"]
            assert len(detail["turns"]) == 2 and len(detail["generated_ids"]) == 1+quota
            assert detail["turns"][0]["output_end"] == 1
            assert detail["turns"][1]["new_user_tokens"] == len(later)
            assert detail["turns"][1]["pending_flushed"] == 1
            for turn in detail["turns"]:
                assert turn["prefill_shape_admission_seconds"] == 0
                assert turn["prefill_phase_creations"] == 0
            generations = [row for row in raw["sessions"][0]["ledger"] if row["generation"]]
            assert all(row["turn"] == 2 and row["sampling_draw_before"] == row["sampling_draw_after"] == 0
                       and row["accepted"] == 1+row["consumed_drafts"]
                       and len(row["verification_input_ids"]) == row["verified_rows"]
                       and 0 <= row["consumed_drafts"] <= row["verified_rows"]-1 <= row["draft_limit"] <= width
                       for row in generations)
            assert sum(row["accepted"] for row in generations) == quota-1
            cold = stderr.split("startup_to_ready_seconds=",1)[0]
            inventory = re.findall(r"^compile full text finite family .+$",cold,re.M)
            assert inventory, "no actual cold families were observed"
            inventories.append(sorted(inventory))
            print(f"PASS K{width} {name}: first quota1, late decode{quota-1}, no future hints/admission, exact work/RNG, retired",flush=True)
        assert inventories[0] == inventories[1], "future prompt/quotas changed the actual cold family inventory"
    print("PASS K2/K3 cold inventories ignore future lengths and quotas; recursive programs exist before late decode",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyle",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.hyle = args.hyle.resolve();args.inputs = args.inputs.resolve();args.output = args.output.resolve()
    assert args.hyle.is_file() and (args.inputs/"weights.tsv").is_file(), "only existing local binaries/weights are accepted"
    check(args)
