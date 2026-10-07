"""Derive the old-core resource + original CPU IQ1 manifest from locked inputs.

No mathematical bodies are rewritten or separately maintained. The caller
authenticates the source and both owning manifests before invoking this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path


def derive(source, resource_manifest, cpu_manifest, output):
    resource = json.loads(resource_manifest.read_text(encoding="utf-8"))
    cpu = json.loads(cpu_manifest.read_text(encoding="utf-8"))
    if resource["upstream_revision"] != "456172ec733a135778adcd32d00e576a58232e45":
        raise ValueError("mixed CPU core changed; re-audit the port")
    existing = {e["before"] for f in resource["files"] for e in f["edits"]}
    edits = {f["path"]: list(f["edits"]) for f in resource["files"]}
    added = 0
    for file in cpu["files"]:
        borrowed = [e for e in file["edits"] if e["before"] not in existing]
        if borrowed:
            edits.setdefault(file["path"], []).extend(borrowed)
            added += len(borrowed)
    if added != 19:
        raise ValueError("original IQ1 port composition changed; re-audit")
    files = []
    for name, changes in edits.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe patch path")
        before = (source / relative).read_bytes().replace(b"\r\n", b"\n")
        after = before
        for edit in changes:
            match = edit["before"].encode()
            if after.count(match) != 1:
                raise ValueError(f"non-unique original port context: {name}")
            after = after.replace(match, edit["after"].encode(), 1)
        files.append({"path": name,
                      "before_sha256": hashlib.sha256(before).hexdigest(),
                      "after_sha256": hashlib.sha256(after).hexdigest(),
                      "edits": changes})
    output.write_text(json.dumps({"upstream_revision": resource["upstream_revision"],
                                 "borrowed_iq1_revision": cpu["borrowed_iq1_revision"],
                                 "files": files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "resource-manifest", "cpu-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    derive(args.source, args.resource_manifest, args.cpu_manifest, args.output)
