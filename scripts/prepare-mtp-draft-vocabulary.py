#!/usr/bin/env python3
"""Calibrate a source-bound MTP vocabulary using only local model generations.

Run once during model preparation, outside engine startup. The fixed prompts
below are independent of tests/fixtures/qwen35-mtp-gating and its outputs.
Only generated IDs count toward frequency; unseen IDs tie by original ID.
No weights, tokenizer files or datasets are downloaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import subprocess


PROMPTS = (
    "Explain how ocean tides arise from gravity. Include spring and neap tides and explain why coastlines change their height.",
    "Describe how a seed becomes a flowering plant, including photosynthesis, roots, pollination and the movement of water.",
    "Give a worked example of Bayes' theorem using a medical screening test. Clearly distinguish sensitivity and posterior probability.",
    "Explain the lifecycle of a star like the Sun, from a cloud of gas to a white dwarf, for a curious teenager.",
    "Write a concise guide to measuring electrical resistance, voltage and current safely on a low voltage breadboard.",
    "Compare induction, deduction and falsification using concrete examples from scientific experiments.",
    "Write a C++20 example of a move-only RAII file descriptor wrapper. Explain ownership transfer and error handling.",
    "Write Python code that uses asyncio to fetch independent tasks with bounded concurrency, cancellation and timeouts.",
    "Explain database transaction isolation with a bank transfer example. Compare read committed and serializable isolation.",
    "Describe a reliable method to debug an intermittent memory leak in a long running service, including tools and measurements.",
    "Explain how HTTP caching works, including ETag, Cache-Control, conditional requests and shared caches.",
    "Write a Rust example that parses a configuration file and returns useful typed errors without panicking.",
    "用中文解释地震波为什么可以帮助科学家了解地球内部结构，包括纵波、横波和液态外核。",
    "请写一份家用阳台香草种植指南，包括薄荷、罗勒和迷迭香的日照、浇水和修剪方法。",
    "用中文介绍纸张从植物纤维到成品的制造过程，并解释再生纸的优点和限制。",
    "请讲解如何阅读一份企业现金流量表，区分经营、投资和筹资活动，并举具体例子。",
    "用中文说明数码照片的曝光、快门、光圈、感光度和景深之间的关系。",
    "请写一篇关于传统木工榫卯结构的科普短文，解释其受力方式和常见连接形式。",
    "日本語で、火山の噴火の仕組みと防災情報の読み方を説明してください。",
    "Expliquez en français comment fonctionne une pompe à chaleur et dans quelles conditions son rendement varie.",
    "Explica en español las diferencias entre interés simple y compuesto con un ejemplo de ahorro durante cinco años.",
    "Erkläre auf Deutsch, wie ein Fahrradgetriebe funktioniert und wann kleine oder große Gänge sinnvoll sind.",
    "اشرح باللغة العربية كيف تتكون السحب ولماذا تختلف أنواعها، مع أمثلة مبسطة.",
    "Explain how to plan a small community library: cataloguing, lending rules, accessibility and preserving old books.",
)


def recorded_counts(folder: Path, identity: str, vocabulary: int) -> tuple[Counter[int], int]:
    """Read source-bound ordinary generations; mixed corpora import train only."""
    records = folder / "prompts.tsv"
    prompts: list[tuple[Path, str | None]] = []
    if records.is_file():
        with records.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if not rows or any(set(row) != {"split", "index", "group", "prompt"} for row in rows):
            raise ValueError("recorded corpus requires split/index/group/prompt columns")
        seen = set()
        for row in rows:
            if row["split"] not in ("train", "validation"):
                raise ValueError("unknown recorded corpus split")
            if row["split"] != "train":
                continue
            if not row["index"].isdigit() or int(row["index"]) < 1 or not re.fullmatch(r"[A-Za-z0-9_]+", row["group"]):
                raise ValueError("invalid recorded training path")
            index = int(row["index"])
            if index in seen or not row["prompt"]:
                raise ValueError("duplicate or empty recorded training prompt")
            seen.add(index)
            prompts.append((folder / f'train-{index:02}-{row["group"]}', row["prompt"]))
    else:
        summary = (folder / "calibration.txt").read_text(encoding="utf-8")
        if f"source={identity}" not in summary.splitlines():
            raise ValueError("recorded calibration source identity differs")
        prompts = [(path, None) for path in sorted(folder.iterdir())
                   if path.is_dir() and re.fullmatch(r"prompt[0-9]+", path.name)]
    if not prompts:
        raise ValueError("recorded calibration has no training prompts")
    counts: Counter[int] = Counter()
    for path, prompt in prompts:
        raw = json.loads((path / "stdout.log").read_text(encoding="utf-8"))
        if raw.get("source_id") != identity or raw.get("sampling") != "greedy" or raw.get("mtp_max_drafts") != 0:
            raise ValueError("recorded training requires matching ordinary greedy source")
        if raw.get("live_committed_versions") != 0 or len(raw.get("sessions", [])) != 1:
            raise ValueError("recorded training did not retire its original session")
        if prompt is not None and (path / "prompt.txt").read_text(encoding="utf-8").strip() != prompt.strip():
            raise ValueError("recorded training prompt differs from its declared split")
        generated = raw["sessions"][0]["result"]["generated_ids"]
        if not generated or any(type(token) is not int or not 0 <= token < vocabulary for token in generated):
            raise ValueError("invalid recorded training IDs")
        if (path / "generated.ids").exists() and list(map(int, (path / "generated.ids").read_text().split())) != generated:
            raise ValueError("recorded ID text differs from its original process output")
        counts.update(generated)
    return counts, len(prompts)


def selected_ids(counts: Counter[int], vocabulary: int, rows: int) -> list[int]:
    if not counts or not 0 < rows <= vocabulary:
        raise ValueError("nonempty calibration and valid row count required")
    if any(type(token) is not int or not 0 <= token < vocabulary or type(count) is not int or count <= 0
           for token, count in counts.items()):
        raise ValueError("invalid calibration counts")
    return sorted(sorted(range(vocabulary), key=lambda token: (-counts[token], token))[:rows])


def write_vocabulary(output, identity: str, vocabulary: int, ids: list[int],
                     recommended_drafts: int | None) -> None:
    recommendation = 'none' if recommended_drafts is None else str(recommended_drafts)
    output.write(f'hyle-draft-vocabulary-v2\n{identity}\n{vocabulary}\n'
                 f'recommended_drafts {recommendation}\nhead original\n')
    output.write(' '.join(map(str, ids)) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyle", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--recommended-drafts", type=int,
                        help="optional measured automatic draft depth for this source")
    parser.add_argument("--calibration-run", type=Path, action="append", default=[],
                        help="reuse existing source-bound calibration; repeat to combine runs; train split only, no GPU")
    args = parser.parse_args()
    for name in ("hyle", "inputs", "tokenizer_dir", "work_dir", "output"):
        value = getattr(args, name)
        setattr(args, name, value.resolve() if value is not None else None)
    args.calibration_run = [path.resolve() for path in args.calibration_run]
    if args.recommended_drafts is not None and not 0 < args.recommended_drafts <= (1 << 64) - 1:
        parser.error("recommended drafts must be a positive 64-bit count")
    if len(set(args.calibration_run)) != len(args.calibration_run):
        parser.error("the same calibration run cannot be counted twice")
    if not args.calibration_run and (args.hyle is None or args.tokenizer_dir is None or
                                    not args.hyle.is_file() or not args.tokenizer_dir.is_dir()):
        parser.error("existing local executable and tokenizer directory required")
    lines = (args.inputs / "weights.tsv").read_text(encoding="utf-8").splitlines()
    identity = lines[1]
    if len(identity) != 64 or any(ch not in "0123456789abcdef" for ch in identity):
        parser.error("invalid prepared source identity")
    geometry = lines[2].split()
    vocabulary = int(geometry[3])
    if not 0 < args.rows <= vocabulary or not 0 < args.tokens <= 1024:
        parser.error("rows must fit model vocabulary; tokens must be in [1,1024]")
    if args.work_dir.exists() or args.output.exists():
        parser.error("work directory and output must be new; calibration is never overwritten")
    if args.calibration_run:
        counts: Counter[int] = Counter()
        prompts = 0
        for path in args.calibration_run:
            current, count = recorded_counts(path, identity, vocabulary)
            counts.update(current)
            prompts += count
        selected = selected_ids(counts, vocabulary, args.rows)
        args.work_dir.mkdir(parents=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            write_vocabulary(output, identity, vocabulary, selected, args.recommended_drafts)
        summary = (f"source={identity}\nprompts={prompts}; generated={sum(counts.values())}; distinct={len(counts)}\n"
                   f"rows={len(selected)}; generated-frequency descending; ties=original ID ascending\n"
                   "Offline reuse only; validation split ignored; no executable, tokenizer or GPU was invoked.\n")
        (args.work_dir / "calibration.txt").write_text(summary, encoding="utf-8")
        print(summary, flush=True)
        return
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, trust_remote_code=False)
    args.work_dir.mkdir(parents=True)
    counts: Counter[int] = Counter()
    for index, prompt in enumerate(PROMPTS, 1):
        folder = args.work_dir / f"prompt{index:02}"
        folder.mkdir()
        text = tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + args.tokens > 2048:
            raise ValueError("calibration prompt exceeds configured capacity")
        (folder / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        (folder / "prompt.ids").write_text(" ".join(map(str, ids)) + "\n", encoding="utf-8")
        schedule = folder / "schedule.txt"
        schedule.write_text(f"prompt prompt.ids\ngenerate {args.tokens}\n", encoding="utf-8")
        command = [str(args.hyle), "--inputs", str(args.inputs), "--backend", "cuda",
                   "--mtp", "off", "--capacity", "2048", "--schedule", str(schedule)]
        (folder / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=600)
        if result.returncode:
            raise RuntimeError(f"calibration failed: {folder}")
        raw = json.loads((folder / "stdout.log").read_text(encoding="utf-8"))
        generated = raw["sessions"][0]["result"]["generated_ids"]
        if not generated or any(not 0 <= token < vocabulary for token in generated):
            raise ValueError("invalid model-generated calibration IDs")
        counts.update(generated)
        (folder / "generated.ids").write_text(" ".join(map(str, generated)) + "\n", encoding="utf-8")
        (folder / "generated.txt").write_text(tokenizer.decode(generated), encoding="utf-8")
        print(f"calibration {index}/{len(PROMPTS)}: generated={len(generated)}, distinct={len(counts)}", flush=True)
    selected = selected_ids(counts, vocabulary, args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        write_vocabulary(output, identity, vocabulary, sorted(selected), args.recommended_drafts)
    summary = (f"source={identity}\nprompts={len(PROMPTS)}; generated={sum(counts.values())}; distinct={len(counts)}\n"
               f"rows={len(selected)}; generated-frequency descending; ties=original ID ascending\n"
               "Only generated output IDs were counted; no benchmark prompts or outputs were used.\n")
    (args.work_dir / "calibration.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
