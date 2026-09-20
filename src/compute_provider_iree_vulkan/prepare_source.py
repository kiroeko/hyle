"""Assemble authenticated upstream sources and this provider's exact overlay.

CMake owns downloading and source identity. This script only creates a private
build copy; it never edits the shared cache or invokes the model compiler.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reject_links(path):
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError(f"build-local path is a link: {part}")


def safe_child(root, name):
    if (not isinstance(name, str) or not name or ":" in name or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("invalid source manifest path")
    path = root / name
    reject_links(path)
    if not path.resolve().is_relative_to(root):
        raise ValueError("source manifest path escapes build copy")
    return path


def prepare(source, flatcc, headers, vendor, output, generated):
    roots = [(source.resolve(), ""), (flatcc.resolve(), "third_party/flatcc/"),
             (headers.resolve(), "third_party/vulkan_headers/")]
    reject_links(output.absolute())
    output = output.resolve()
    for root, _ in roots:
        if root == output or root in output.parents or output in root.parents:
            raise ValueError("upstream and build-local source must be separate")
    files = {}
    for root, prefix in roots:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            # Upstream documentation has a broken optional icon symlink. It is
            # not a runtime build input; never create Windows privileged links.
            if ".git" in relative.parts or (not prefix and relative.parts[0] == "docs"):
                continue
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError(f"unexpected runtime source symlink: {path}")
            if path.is_file():
                name = prefix + relative.as_posix()
                if name in files:
                    raise ValueError(f"overlapping source: {name}")
                files[name] = path.read_bytes()
    manifest = json.loads((vendor / "patches.json").read_text(encoding="utf-8"))
    for patch in manifest["files"]:
        name = patch["path"]
        data = files[name].replace(b"\r\n", b"\n")
        if sha(data) != patch["before_sha256"]:
            raise ValueError(f"upstream patch identity differs: {name}")
        for edit in patch["edits"]:
            before, after = edit["before"].encode(), edit["after"].encode()
            if data.count(before) != 1:
                raise ValueError(f"patch context is not unique: {name}")
            data = data.replace(before, after, 1)
        if sha(data) != patch["after_sha256"]:
            raise ValueError(f"patch result identity differs: {name}")
        files[name] = data
    # One authenticated, module-owned build copy keeps unchanged compiler
    # inputs at stable paths. Never overwrite unrelated edits in that copy.
    stamp = output.parent / "source-manifest.json"
    reject_links(stamp)
    previous = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    # Validate every previous name before even inspecting/removing old files.
    # A corrupted stamp must never turn cleanup into a traversal outside output.
    for name in previous.keys() | files.keys():
        safe_child(output, name)
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() or p.is_symlink()}
    if actual - (previous.keys() | files.keys()):
        raise ValueError("build-local source has unexpected files")
    for name in actual:
        path = safe_child(output, name)
        current = sha(path.read_bytes())
        desired = sha(files[name]) if name in files else None
        if current != previous.get(name) and current != desired:
            raise ValueError(f"build-local source was modified: {name}")
    for name in previous.keys() - files.keys():
        safe_child(output, name).unlink(missing_ok=True)
    for name, data in files.items():
        path = safe_child(output, name)
        if not path.exists() or path.read_bytes() != data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() or p.is_symlink()}
    if actual != files.keys():
        raise ValueError("build-local source has unexpected files")
    stamp.write_bytes((json.dumps({name: sha(data) for name, data in files.items()},
                                 sort_keys=True) + "\n").encode())
    generated.mkdir(parents=True, exist_ok=True)
    programs = json.loads((vendor / "programs/manifest.json").read_text(encoding="utf-8"))
    lines = ["// Embedded qualified upstream programs; generated, do not edit.",
             '#include "programs.h"', "namespace hyle::compute::iree_vulkan::detail {"]
    for program in programs["programs"]:
        recipe = (vendor / "programs" / program["recipe"]).read_bytes().replace(b"\r\n", b"\n")
        binary = base64.b64decode((vendor / "programs" / program["file"]).read_bytes())
        if sha(binary) != program["sha256"] or len(binary) != program["bytes"] or sha(recipe) != program["recipe_sha256"]:
            raise ValueError("offline program identity differs")
        name = program["name"]
        lines.append(f"alignas(64) static constexpr unsigned char {name}_bytes[] = {{")
        for start in range(0, len(binary), 24):
            lines.append(",".join(f"0x{b:02x}" for b in binary[start:start + 24]) + ",")
        lines += ["};", f"std::span<const unsigned char> {name}_program() noexcept {{ return {name}_bytes; }}"]
    lines.append("}")
    path = generated / "programs.cc"
    data = ("\n".join(lines) + "\n").encode()
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    print(f"IREE source: {len(files)} files, {len(manifest['files'])} exact patches, {len(programs['programs'])} offline programs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["source", "flatcc", "headers", "vendor", "output", "generated"]:
        parser.add_argument("--" + name, type=Path, required=True)
    prepare(**vars(parser.parse_args()))
