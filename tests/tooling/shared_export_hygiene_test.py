#!/usr/bin/env python3
"""Focused contracts for the installed shared-export checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys
import tempfile


def load_checker():
    script = Path(__file__).parents[2] / "scripts" / "check-shared-export-hygiene.py"
    spec = importlib.util.spec_from_file_location(
        "hyle_shared_export_hygiene", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_pe(path: Path, names: tuple[str, ...]) -> None:
    data = bytearray(0x1000)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    coff = 0x84
    struct.pack_into("<H", data, coff, 0x8664)
    struct.pack_into("<H", data, coff + 2, 1)
    struct.pack_into("<H", data, coff + 16, 0xF0)
    optional = coff + 20
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<II", data, optional + 112, 0x1100, 0x300)
    section = optional + 0xF0
    data[section : section + 8] = b".rdata\0\0"
    struct.pack_into(
        "<IIII", data, section + 8, 0x800, 0x1000, 0x800, 0x200
    )
    directory = 0x300
    struct.pack_into("<I", data, directory + 24, len(names))
    struct.pack_into("<I", data, directory + 32, 0x1200)
    cursor = 0x500
    for index, name in enumerate(names):
        encoded = name.encode("ascii") + b"\0"
        name_rva = 0x1000 + cursor - 0x200
        struct.pack_into("<I", data, 0x400 + index * 4, name_rva)
        data[cursor : cursor + len(encoded)] = encoded
        cursor += len(encoded)
    path.write_bytes(data)


def main() -> None:
    checker = load_checker()
    with tempfile.TemporaryDirectory(prefix="hyle-export-test-") as raw:
        root = Path(raw)
        clean = root / "hyle_clean.dll"
        synthetic_pe(clean, ("?run@Scheduler@hyle@@QEAAXXZ",))
        assert checker.pe_exports(clean) == ("?run@Scheduler@hyle@@QEAAXXZ",)
        assert checker.main(["--root", str(root)]) == 0

        dirty = root / "hyle_dirty.dll"
        synthetic_pe(
            dirty,
            (
                "?ok@hyle@@YAXXZ",
                "jump_fcontext",
                "hyle_status_create",
                "mi_malloc",
                "dnnl_brgemm_execute",
            ),
        )
        result = checker.scan_binary(dirty)
        reasons = {
            name: checker.forbidden_reason(name)
            for name in result.exports
            if checker.forbidden_reason(name)
        }
        assert set(reasons) == {
            "jump_fcontext",
            "hyle_status_create",
            "mi_malloc",
            "dnnl_brgemm_execute",
        }
        assert checker.main(["--root", str(root)]) == 1

        empty = root / "headers-only"
        empty.mkdir()
        assert checker.main(["--root", str(empty), "--allow-empty"]) == 0
        assert checker.main(["--root", str(empty)]) == 2


if __name__ == "__main__":
    main()
