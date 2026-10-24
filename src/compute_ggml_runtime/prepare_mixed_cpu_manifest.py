"""Derive old-core resource, CPU IQ1 and unary FWHT edits from locked inputs.

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
    existing = {f["path"]: {(e["before"], e["after"]) for e in f["edits"]}
                for f in resource["files"]}
    edits = {f["path"]: list(f["edits"]) for f in resource["files"]}
    for file in cpu["files"]:
        borrowed = []
        for edit in file["edits"]:
            # The old core has the same F32 FWHT body without the newer F16
            # template overload. Adapt only its signature/call-site spelling;
            # the original arithmetic remains in each owning native source.
            if "ggml_compute_forward_fwht_impl<ggml_fp16_t>" in edit["before"]:
                continue
            ported = {}
            for key in ("before", "after"):
                ported[key] = edit[key].replace(
                    "ggml_compute_forward_fwht_impl<float>",
                    "ggml_compute_forward_fwht_f32").replace(
                    "ggml_compute_forward_fwht_impl(",
                    "ggml_compute_forward_fwht_f32(")
            shared = existing.setdefault(file["path"], set())
            identity = (ported["before"], ported["after"])
            if identity in shared:
                continue
            if any(before == ported["before"] for before, _ in shared):
                raise ValueError(f"conflicting shared CPU edit: {file['path']}")
            borrowed.append(ported)
            shared.add(identity)
        if borrowed:
            edits.setdefault(file["path"], []).extend(borrowed)
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
