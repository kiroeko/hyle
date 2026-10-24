"""Compose authenticated original core/resource/Metal edits for the Apple route."""
import argparse
import hashlib
import json
from pathlib import Path

def compose(source, manifests, output):
    changes = {}
    created = {}
    for manifest in manifests:
        for item in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
            name = item["path"]
            if name in created or ("content" in item and name in changes):
                raise ValueError(f"duplicate created Metal port path: {name}")
            if "content" in item:
                created[name] = item
            else:
                changes.setdefault(name, []).extend(item["edits"])
    files = []
    for name, edits in changes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe original Metal port path")
        before = (source / path).read_bytes().replace(b"\r\n", b"\n")
        after = before
        for edit in edits:
            old = edit["before"].encode()
            if after.count(old) != 1:
                raise ValueError(f"non-unique original Metal port context: {name}")
            after = after.replace(old, edit["after"].encode(), 1)
        files.append({"path": name, "before_sha256": hashlib.sha256(before).hexdigest(),
                      "after_sha256": hashlib.sha256(after).hexdigest(), "edits": edits})
    for name, item in created.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe original Metal port path")
        original = source / path
        if (item["before_sha256"] is not None or "edits" in item or
                original.exists() or original.is_symlink()):
            raise ValueError(f"created Metal port already exists or has edits: {name}")
        if hashlib.sha256(item["content"].encode()).hexdigest() != item["after_sha256"]:
            raise ValueError(f"created Metal port identity differs: {name}")
        files.append(item)
    output.write_text(json.dumps({"upstream_revision": "456172ec733a135778adcd32d00e576a58232e45",
                                 "files": files}, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compose(args.source, args.manifest, args.output)
