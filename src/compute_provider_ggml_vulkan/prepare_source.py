"""Materialize this provider's exact, locked edits in a build-local copy.

The owning CMake authenticates the upstream checkout/archive and this manifest
before calling us. This is not a downloader or an alternative source cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, output, manifest):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("patched source must be separate from the upstream source")
    patches = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    edited = {}
    for patch in patches:
        relative = Path(patch["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe patch path")
        # Git checkouts may have CRLF; archive identities are checked by CMake.
        data = (source / relative).read_bytes().replace(b"\r\n", b"\n")
        if digest(data) != patch["before_sha256"]:
            raise ValueError(f"upstream patch identity differs: {relative}")
        for edit in patch["edits"]:
            before, after = edit["before"].encode(), edit["after"].encode()
            if data.count(before) != 1:
                raise ValueError(f"patch context is not unique: {relative}")
            data = data.replace(before, after, 1)
        if digest(data) != patch["after_sha256"]:
            raise ValueError(f"patch result identity differs: {relative}")
        edited[relative.as_posix()] = data

    expected = set()
    # Do not carry Git metadata into the build tree or touch the shared cache.
    for entry in source.rglob("*"):
        relative = entry.relative_to(source)
        if ".git" in relative.parts:
            continue
        destination = output / relative
        for parent in destination.parents:
            if parent == output:
                break
            if parent.is_symlink():
                raise ValueError(f"build-local directory is a symlink: {parent}")
        if entry.is_symlink():
            expected.add(relative.as_posix())
            target = os.readlink(entry)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                if os.readlink(destination) != target:
                    raise ValueError(f"build-local symlink was modified: {destination}")
            elif destination.exists():
                raise ValueError(f"build-local symlink was replaced: {destination}")
            else:
                destination.symlink_to(target)
            continue
        if entry.is_dir():
            if destination.is_symlink():
                raise ValueError(f"build-local directory is a symlink: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
            continue
        expected.add(relative.as_posix())
        data = edited.get(relative.as_posix())
        if data is None:
            data = entry.read_bytes()
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != data:
                raise ValueError(f"build-local source was modified: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() or p.is_symlink()}
    if actual != expected:
        raise ValueError("build-local source has unexpected files")
    print(f"ggml Vulkan source: {len(expected)} files, {len(edited)} exact patches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.manifest)
