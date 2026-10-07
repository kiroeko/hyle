"""Compose locked CPU physical types, CUDA resources and original IQ1 math.

This owning helper derives one manifest for the existing source materializer;
it neither changes mathematical blocks nor writes to the shared source cache.
"""
import argparse
import hashlib
import json
from pathlib import Path


def derive(source, base_manifest, cuda_manifest, iq1_manifest, output):
    base = json.loads(base_manifest.read_text(encoding="utf-8"))
    cuda = json.loads(cuda_manifest.read_text(encoding="utf-8"))
    iq1 = json.loads(iq1_manifest.read_text(encoding="utf-8"))
    revision = "456172ec733a135778adcd32d00e576a58232e45"
    if any(item["upstream_revision"] != revision for item in (base, cuda, iq1)):
        raise ValueError("CUDA IQ1 core identity changed; re-audit the port")
    if iq1["borrowed_iq1_revision"] != "46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f":
        raise ValueError("original CUDA IQ1 identity changed")
    edits = {}
    for manifest in (base, cuda, iq1):
        for entry in manifest["files"]:
            name = entry["path"]
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe CUDA IQ1 path")
            edits.setdefault(name, []).extend(entry["edits"])
    files = []
    for name, changes in edits.items():
        before = (source / name).read_bytes().replace(b"\r\n", b"\n")
        after = before
        for change in changes:
            old, new = change["before"].encode(), change["after"].encode()
            if after.count(old) != 1:
                raise ValueError(f"non-unique original CUDA IQ1 context: {name}")
            after = after.replace(old, new, 1)
        files.append({"path": name,
                      "before_sha256": hashlib.sha256(before).hexdigest(),
                      "after_sha256": hashlib.sha256(after).hexdigest(),
                      "edits": changes})
    output.write_text(json.dumps({"upstream_revision": revision,
        "borrowed_iq1_revision": iq1["borrowed_iq1_revision"],
        "files": files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "base-manifest", "cuda-manifest", "iq1-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    derive(args.source, args.base_manifest, args.cuda_manifest, args.iq1_manifest, args.output)
