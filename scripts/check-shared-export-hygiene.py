#!/usr/bin/env python3
"""Reject accidental plain-C exports from installed HYLE shared libraries.

This is intentionally not a C++ ABI whitelist. HYLE's supported shared product
surface is the explicitly annotated, compiler-mangled C++ API. While Native FFI
is N/A, project-prefixed C entry points, vendored allocator/provider symbols and
raw fcontext procedures must remain implementation details.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from typing import Iterable


class ExportScanError(RuntimeError):
    pass


_RAW_FCONTEXT = frozenset(
    {"jump_fcontext", "make_fcontext", "ontop_fcontext"}
)
_MACH_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


@dataclass(frozen=True)
class ScanResult:
    path: Path
    exports: tuple[str, ...]


def _u16(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 2 > len(data):
        raise ExportScanError("truncated 16-bit field")
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ExportScanError("truncated 32-bit field")
    return struct.unpack_from("<I", data, offset)[0]


def _read_c_string(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        raise ExportScanError("export name points outside the image")
    end = data.find(b"\0", offset)
    if end < 0:
        raise ExportScanError("unterminated export name")
    try:
        return data[offset:end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ExportScanError("non-ASCII PE export name") from exc


def pe_exports(path: Path) -> tuple[str, ...]:
    """Read names from a PE export directory without requiring Visual Studio."""
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise ExportScanError(f"{path}: not a PE image")
    pe_offset = _u32(data, 0x3C)
    if pe_offset + 24 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ExportScanError(f"{path}: invalid PE header")

    coff = pe_offset + 4
    section_count = _u16(data, coff + 2)
    optional_size = _u16(data, coff + 16)
    optional = coff + 20
    magic = _u16(data, optional)
    if magic == 0x10B:
        data_directories = optional + 96
    elif magic == 0x20B:
        data_directories = optional + 112
    else:
        raise ExportScanError(
            f"{path}: unsupported PE optional-header magic 0x{magic:x}"
        )
    if data_directories + 8 > optional + optional_size:
        raise ExportScanError(f"{path}: PE image has no export data directory")
    export_rva = _u32(data, data_directories)
    export_size = _u32(data, data_directories + 4)
    if export_rva == 0 or export_size == 0:
        return ()

    sections: list[tuple[int, int, int, int]] = []
    section_table = optional + optional_size
    for index in range(section_count):
        entry = section_table + index * 40
        if entry + 40 > len(data):
            raise ExportScanError(f"{path}: truncated PE section table")
        virtual_size = _u32(data, entry + 8)
        virtual_address = _u32(data, entry + 12)
        raw_size = _u32(data, entry + 16)
        raw_offset = _u32(data, entry + 20)
        sections.append((virtual_address, virtual_size, raw_offset, raw_size))

    def rva_offset(rva: int, size: int = 1) -> int:
        if rva < section_table and rva + size <= len(data):
            return rva
        for virtual_address, virtual_size, raw_offset, raw_size in sections:
            span = max(virtual_size, raw_size)
            if virtual_address <= rva and rva + size <= virtual_address + span:
                offset = raw_offset + (rva - virtual_address)
                if offset + size <= len(data):
                    return offset
                break
        raise ExportScanError(
            f"{path}: RVA 0x{rva:x} is outside file-backed sections"
        )

    directory = rva_offset(export_rva, 40)
    name_count = _u32(data, directory + 24)
    names_rva = _u32(data, directory + 32)
    if name_count > len(data) // 4:
        raise ExportScanError(
            f"{path}: unreasonable PE export-name count {name_count}"
        )
    names_table = rva_offset(names_rva, name_count * 4)
    names = []
    for index in range(name_count):
        name_rva = _u32(data, names_table + index * 4)
        names.append(_read_c_string(data, rva_offset(name_rva)))
    return tuple(sorted(set(names)))


def _nm_exports(path: Path, image_format: str) -> tuple[str, ...]:
    failures = []
    for candidate in ("llvm-nm", "nm"):
        tool = shutil.which(candidate)
        if tool is None:
            continue
        args = [tool]
        if image_format == "elf":
            args.extend(["-D", "--defined-only", "--extern-only"])
        else:
            args.extend(["-gU"])
        args.append(str(path))
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode:
            failures.append(f"{candidate}: {result.stderr.strip()}")
            continue
        names = []
        for raw_line in result.stdout.splitlines():
            fields = raw_line.strip().split()
            if len(fields) < 2 or fields[-1].endswith(":"):
                continue
            name = fields[-1]
            if image_format == "mach" and name.startswith("_"):
                name = name[1:]
            names.append(name)
        return tuple(sorted(set(names)))
    detail = "; ".join(failures) if failures else "llvm-nm/nm not found"
    raise ExportScanError(
        f"{path}: cannot inspect {image_format} exports: {detail}"
    )


def scan_binary(path: Path) -> ScanResult:
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic[:2] == b"MZ":
        exports = pe_exports(path)
    elif magic == b"\x7fELF":
        exports = _nm_exports(path, "elf")
    elif magic in _MACH_MAGICS:
        exports = _nm_exports(path, "mach")
    else:
        raise ExportScanError(f"{path}: unsupported shared-library format")
    return ScanResult(path=path, exports=exports)


def policy_name(symbol: str) -> str:
    # Remove ELF symbol versions and 32-bit PE C/stdcall decoration. Compiler-
    # mangled C++ names (_Z..., ?...@...) deliberately remain distinguishable.
    name = symbol.split("@@", 1)[0]
    if re.fullmatch(r"_[A-Za-z][A-Za-z0-9_]*@[0-9]+", name):
        name = name[1:].rsplit("@", 1)[0]
    elif re.fullmatch(r"_[A-Za-z][A-Za-z0-9_]*", name):
        name = name[1:]
    return name


def forbidden_reason(symbol: str) -> str | None:
    name = policy_name(symbol)
    if name in _RAW_FCONTEXT:
        return "raw fcontext implementation seam"
    if name == "hyle_signal_handler":
        return "private POSIX signal trampoline"
    if name.startswith("hyle_"):
        return "plain HYLE C entry while Native FFI is N/A"
    if name.startswith("mi_"):
        return "vendored mimalloc entry"
    if name.startswith("dnnl_"):
        return "vendored oneDNN entry"
    return None


def discover_binaries(
    roots: Iterable[Path], explicit: Iterable[Path]
) -> tuple[Path, ...]:
    found: dict[Path, None] = {}
    for item in explicit:
        path = item.resolve()
        if not path.is_file():
            raise ExportScanError(f"shared library does not exist: {item}")
        found[path] = None
    for root_value in roots:
        root = root_value.resolve()
        if not root.is_dir():
            raise ExportScanError(f"scan root is not a directory: {root_value}")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            is_hyle = (
                (name.startswith("hyle_") and name.endswith(".dll"))
                or (name.startswith("libhyle_") and ".so" in name)
                or (name.startswith("libhyle_") and name.endswith(".dylib"))
            )
            if is_hyle:
                found[path.resolve()] = None
    return tuple(sorted(found, key=lambda value: str(value)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--binary", action="append", type=Path, default=[])
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="accept a header-only selected product with no shared artifact",
    )
    args = parser.parse_args(argv)
    if not args.root and not args.binary:
        parser.error("pass at least one --root or --binary")

    try:
        binaries = discover_binaries(args.root, args.binary)
        if not binaries and not args.allow_empty:
            raise ExportScanError("no installed HYLE shared libraries found")
        violations = []
        export_count = 0
        for binary in binaries:
            result = scan_binary(binary)
            export_count += len(result.exports)
            for symbol in result.exports:
                reason = forbidden_reason(symbol)
                if reason:
                    violations.append((binary, symbol, reason))
    except (OSError, ExportScanError) as exc:
        print(f"shared export hygiene check failed: {exc}", file=sys.stderr)
        return 2

    if violations:
        print("shared export hygiene check failed:", file=sys.stderr)
        for binary, symbol, reason in violations:
            print(f"  {binary}: {symbol}: {reason}", file=sys.stderr)
        return 1
    print(
        f"shared export hygiene passed: {len(binaries)} HYLE libraries, "
        f"{export_count} exports inspected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
