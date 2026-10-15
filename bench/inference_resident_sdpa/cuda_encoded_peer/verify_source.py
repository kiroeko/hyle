"""Authenticate pristine peer bytes; emitted compiler dependency closure is checked at preflight."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

PIN = "69ff11fc4954396d98326656dc85debd2223f637"
ARCHIVE_SHA256 = "4fc85abaa6210d6a1773e6011ff572aa0c04b7030644b74f75a2db2dbda472a9"
INPUTS = {'attention/prefill.cuh': 'f93cbea96e3a723254c15f5a52d6dd4c13e3892e88eb9b9fed1c0c0048c7d710', 'attention/variants.cuh': '90a228dd71b3ce8030d663e2b16f67f840fe4a3e99912f8b47774d3fd905a080', 'attention/default_prefill_params.cuh': '5b0545df7e134dc031e6a1179ec9c64a6d6b0e4d41b2cd98d3826165b3ed3615', 'attention/scheduler.cuh': 'a11a39aea393cdb380ad9439aac9170cd96849d7426ebbb920c5c8cdadfcc575', 'page.cuh': '98e0b40aa6a5007d767cc49b9a2421bf8df2fbea5045394982ea451892d03ace'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    archive = args.archive or args.original.parent.parent / "archives" / (args.original.name + ".tar.gz")
    archive_proof = None
    if archive.is_file():
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        assert digest == ARCHIVE_SHA256, "original archive hash"
        with tarfile.open(archive) as package:
            original_headers = {}
            for member in package.getmembers():
                parts = Path(member.name).parts
                if member.isfile() and len(parts) > 2 and parts[1] == "include":
                    original_headers[str(Path(*parts[1:]))] = hashlib.sha256(package.extractfile(member).read()).hexdigest()
        archive_proof = {"path": str(archive.resolve()), "sha256": digest}
        revision = PIN
    else:
        # Never let git walk up to an unrelated ancestor repository.
        assert (args.original / ".git").exists(), "a real peer checkout or authenticated original archive is required"
        revision = subprocess.check_output(["git", "-C", str(args.original), "rev-parse", "HEAD"], text=True).strip()
        assert revision == PIN, revision
        assert Path(subprocess.check_output(["git", "-C", str(args.original), "rev-parse", "--show-toplevel"], text=True).strip()).resolve() == args.original.resolve()
        assert not subprocess.check_output(["git", "-C", str(args.original), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    actual = {}
    for name, expected in INPUTS.items():
        actual[name] = hashlib.sha256((args.original / "include/flashinfer" / name).read_bytes()).hexdigest()
        assert actual[name] == expected, name
    headers = {str(p.relative_to(args.original)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((args.original / "include").rglob("*")) if p.is_file()}
    if archive_proof:
        assert headers == original_headers, "original complete include closure differs from authenticated archive"
    args.output.write_text(json.dumps({"revision": revision, "source": str(args.original.resolve()),
        "archive": archive_proof,
        "inputs": actual, "all_original_headers": headers,
        "adaptation": "preprocessor namespace token only; original arithmetic/header bytes unchanged",
        "status": "SOURCE_IDENTITY_ONLY_NOT_COMPILER_NUMERIC_OR_P1"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
