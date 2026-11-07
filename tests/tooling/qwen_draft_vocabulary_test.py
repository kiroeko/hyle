"""Black-box prepared draft-vocabulary policy; only supplied local tiny weights.

Use --inputs with the existing H64/V64 tiny GA fixture and its CUDA runtime.
The complete-ID asset must preserve greedy IDs and work/RNG accounting, while
automatic selection must ignore assets for ordinary and random generation.
"""
from pathlib import Path
import argparse
import json
import subprocess


def check(args):
    args.output.mkdir(parents=True)
    text = (args.inputs / "weights.tsv").read_text(encoding="utf-8")
    lines = text.splitlines()
    vocabulary = int(lines[2].split()[3])
    assert vocabulary == 64, "use the existing tiny V64 fixture, not a full model"
    for name, identity in (("asset", lines[1]), ("partial", lines[1]), ("missing", None), ("stale", "0" * 64)):
        folder = args.output / name
        folder.mkdir()
        (folder / "weights.tsv").write_text(text, encoding="utf-8")
        if identity is not None:
            ids = reversed(range(16 if name == "partial" else vocabulary))
            (folder / "mtp-draft-vocabulary.ids").write_text(
                f"hyle-draft-vocabulary-v2\n{identity}\n{vocabulary}\nrecommended_drafts none\nhead original\n" +
                " ".join(map(str, ids)) + "\n", encoding="utf-8")
    (args.output / "prompt.ids").write_text("1 2 3\n", encoding="utf-8")
    schedule = args.output / "schedule.txt"
    schedule.write_text("prompt prompt.ids\ngenerate 9\n", encoding="utf-8")

    def run(name, inputs="asset", extra=(), selected=0, success=True):
        folder = args.output / name
        folder.mkdir()
        command = [str(args.hyle), "--inputs", str(args.output / inputs), "--backend", "cuda",
                   "--capacity", "32", "--mtp", "on", "--schedule", str(schedule), *extra]
        (folder / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=180)
        if not success:
            assert result.returncode != 0 and "source identity differs" in (folder / "stderr.log").read_text()
            return
        assert result.returncode == 0, folder
        raw = json.loads((folder / "stdout.log").read_text())
        assert raw["draft_vocabulary_rows"] == selected and raw["live_committed_versions"] == 0
        if raw["draft_vocabulary_indexed_rows"]:
            assert selected and raw["active_sequences"] == 1
            assert raw["draft_vocabulary_weight_bytes"] == 0
        elif selected:
            assert raw["draft_vocabulary_weight_bytes"] > 0
        assert raw["draft_vocabulary_map_bytes"] == selected * 4
        detail = raw["sessions"][0]["result"]
        assert len(detail["generated_ids"]) == 9
        ledger = raw["sessions"][0]["ledger"]
        generation = [row for row in ledger if row["generation"]]
        assert sum(row["accepted"] for row in generation) == 8
        if raw["sampling"] == "greedy":
            assert all(row["sampling_draw_before"] == row["sampling_draw_after"] == 0 for row in ledger)
        assert raw["actual_target_rows"] == raw["prefill_target_rows"] + sum(row["verified_rows"] for row in generation)
        if raw["mtp_max_drafts"]:
            assert raw["actual_draft_rows"] == raw["prefill_draft_rows"] + sum(row["draft_rows"] for row in generation)
        return detail["generated_ids"]

    for width in (1, 2, 3):
        options = ["--mtp-max-drafts", str(width)]
        auto = run(f"K{width}-auto", extra=options, selected=vocabulary)
        full = run(f"K{width}-full", extra=[*options, "--mtp-draft-vocabulary", "full"])
        assert auto == full, "complete subset changed greedy outputs"
    assert run("partial-auto", inputs="partial", selected=16) == run(
        "partial-full", inputs="partial", extra=["--mtp-draft-vocabulary", "full"])
    assert run("missing-fallback", inputs="missing") == run("auto-full", extra=["--mtp-draft-vocabulary", "full"])
    run("stale-rejected", inputs="stale", success=False)
    run("stale-full", inputs="stale", extra=["--mtp-draft-vocabulary", "full"])
    # Planned shape admission keeps the tiny C32 fixture out of the ordinary
    # fixed-capacity startup recording used by the full C32768 performance run.
    assert run("ordinary", extra=["--mtp", "off", "--warmup-shapes", "planned"]) == run(
        "ordinary-full", extra=["--mtp", "off", "--warmup-shapes", "planned", "--mtp-draft-vocabulary", "full"])
    assert run("random", extra=["--sampling", "random", "--seed", "7"]) == run(
        "random-full", extra=["--sampling", "random", "--seed", "7", "--mtp-draft-vocabulary", "full"])
    print("PASS: auto/full/partial/missing/stale policies, K1/K2/K3 IDs, indexed storage reporting, ordinary/random fallback, work/RNG and retirement", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyle", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.hyle = args.hyle.resolve(); args.inputs = args.inputs.resolve(); args.output = args.output.resolve()
    assert args.hyle.is_file() and (args.inputs / "weights.tsv").is_file()
    check(args)
