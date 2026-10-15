"""Authenticate pristine peer bytes; emitted compiler dependency closure is checked at preflight."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PIN = "69ff11fc4954"
INPUTS = {'attention/prefill.cuh': 'f93cbea96e3a723254c15f5a52d6dd4c13e3892e88eb9b9fed1c0c0048c7d710', 'attention/variants.cuh': '90a228dd71b3ce8030d663e2b16f67f840fe4a3e99912f8b47774d3fd905a080', 'attention/default_prefill_params.cuh': '5b0545df7e134dc031e6a1179ec9c64a6d6b0e4d41b2cd98d3826165b3ed3615', 'attention/scheduler.cuh': 'a11a39aea393cdb380ad9439aac9170cd96849d7426ebbb920c5c8cdadfcc575', 'page.cuh': '98e0b40aa6a5007d767cc49b9a2421bf8df2fbea5045394982ea451892d03ace'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.original), "rev-parse", "HEAD"], text=True).strip()
    assert revision.startswith(PIN), revision
    assert not subprocess.check_output(["git", "-C", str(args.original), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    actual = {}
    for name, expected in INPUTS.items():
        actual[name] = hashlib.sha256((args.original / "include/flashinfer" / name).read_bytes()).hexdigest()
        assert actual[name] == expected, name
    headers = {str(p.relative_to(args.original)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((args.original / "include").rglob("*")) if p.is_file()}
    args.output.write_text(json.dumps({"revision": revision, "source": str(args.original.resolve()),
        "inputs": actual, "all_original_headers": headers,
        "adaptation": "preprocessor namespace token only; original arithmetic/header bytes unchanged",
        "status": "SOURCE_IDENTITY_ONLY_NOT_COMPILER_NUMERIC_OR_P1"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
