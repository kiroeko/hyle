"""Verify or copy the explicitly supplied local model. Never downloads assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inventory(path: Path) -> dict:
    inventory = json.loads(path.read_bytes())
    entries = inventory["files"]
    identity = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if identity != inventory["artifact_identity_sha256"]:
        raise ValueError("inventory identity mismatch")
    names = [entry["path"] for entry in entries]
    if len(set(names)) != len(names) or names != sorted(names):
        raise ValueError("inventory paths must be unique and sorted")
    for name in names:
        relative = Path(name)
        if relative.anchor or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError(f"unsafe inventory path: {name}")
    return inventory


def matches(directory: Path, entry: dict) -> bool:
    path = directory / entry["path"]
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == entry["size_bytes"]
        and sha256(path) == entry["sha256"]
    )


def verify(directory: Path, inventory: dict) -> dict:
    for entry in inventory["files"]:
        if not matches(directory, entry):
            raise ValueError(f"missing, changed or invalid model file: {directory / entry['path']}")
    return {
        "schema_version": 1,
        "kind": "local_model_input_receipt_not_compute_evidence",
        "source_id": inventory["artifact_identity_sha256"],
        "directory": str(directory.resolve()),
        "machine": socket.gethostname(),
        "platform": sys.platform,
        "files_verified": len(inventory["files"]),
        "bytes_verified": sum(entry["size_bytes"] for entry in inventory["files"]),
        "verification": "size_and_sha256_every_file",
        "result": "PASS",
    }


def copy_local(source: Path, destination: Path, inventory: dict) -> dict:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination must be disjoint directories")
    if destination.exists():
        return verify(destination, inventory)
    partial = destination.with_name(destination.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    for entry in inventory["files"]:
        if matches(partial, entry):
            continue
        target = partial / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / entry["path"], target)
    receipt = verify(partial, inventory)
    partial.rename(destination)
    receipt["directory"] = str(destination)
    receipt["verification"] = "size_and_sha256_every_file_before_atomic_directory_publish"
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "copy", "publish"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    inventory = read_inventory(args.inventory)
    if args.action == "verify":
        receipt = verify(args.source, inventory)
    elif args.destination is None:
        parser.error("copy/publish requires --destination")
    elif args.action == "copy":
        receipt = copy_local(args.source, args.destination, inventory)
    else:
        # Publish a fully received temporary directory; never replaces an existing cache.
        source = args.source.resolve(strict=True)
        destination = args.destination.resolve()
        if source != destination.with_name(destination.name + ".partial"):
            raise ValueError("publish source must be the destination's .partial sibling")
        receipt = verify(source, inventory)
        source.rename(destination)
        receipt["directory"] = str(destination)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
