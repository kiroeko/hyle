"""Authenticate the peer pin and materialize its minimal output synchronization repair."""
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
    parser.add_argument("--prepared", type=Path, required=True)
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
    manifest = Path(__file__).resolve().parents[3] / "third_party/flashinfer_cache/patches.json"
    patches = json.loads(manifest.read_text())
    prefill = next(p for p in patches["files"] if p["path"] == "include/flashinfer/attention/prefill.cuh")
    needles = (
        "  } else {\n    if (get_warp_idx_kv<KTraits>(tid.z) == 0) {\n#pragma unroll\n      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {",
        "      uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<UPCAST_STRIDE_O>(\n          warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);",
    )
    selected = [next(e for e in prefill["edits"] if e["before"] == needle) for needle in needles]
    assert "__syncthreads();" in selected[0]["after"] and "__syncwarp();" in selected[1]["after"]
    data = (args.original / prefill["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == prefill["before_sha256"]
    for edit in selected:
        before, after = edit["before"].encode(), edit["after"].encode()
        assert data.count(before) == 1
        data = data.replace(before, after, 1)
    effective = dict(headers)
    effective[prefill["path"]] = hashlib.sha256(data).hexdigest()
    original_root, prepared_root = args.original.resolve(), args.prepared.resolve()
    assert original_root != prepared_root and original_root not in prepared_root.parents \
        and prepared_root not in original_root.parents, "prepared peer must be outside the pristine tree"
    for name, expected in effective.items():
        destination = args.prepared / name
        content = data if name == prefill["path"] else (args.original / name).read_bytes()
        if destination.exists():
            assert destination.read_bytes() == content, "prepared peer input changed: " + name
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected
    derivation = {"kind": "OUTPUT_STORAGE_CTA_AND_WARP_SYNCHRONIZATION_ONLY",
        "product_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "selected_edits_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        "changed_headers": [prefill["path"]], "barriers": 2,
        "original_prefill_sha256": prefill["before_sha256"],
        "effective_prefill_sha256": effective[prefill["path"]]}
    args.output.write_text(json.dumps({"revision": revision, "source": str(args.original.resolve()),
        "archive": archive_proof,
        "inputs": actual, "all_original_headers": headers,
        "prepared_source": str(args.prepared.resolve()), "effective_headers": effective,
        "derivation": derivation,
        "adaptation": "namespace token plus two output synchronization barriers; original arithmetic/tile/layout/planner unchanged",
        "status": "SOURCE_IDENTITY_ONLY_NOT_COMPILER_NUMERIC_OR_P1"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
