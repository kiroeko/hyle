#!/usr/bin/env python3
"""Audit one HYLE exact-SIMD build without executing the target ISA."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


CANONICAL = ("scalar", "sse4", "avx2", "avx512", "neon", "sve", "rvv")
X86_FORBIDDEN_SOURCES = {
    "scalar": ("dot_f16c.cc", "dot_avx512bf16.cc", "cvt_f16c.cc", "cvt_avx2.cc",
               "cvt_avx512bf16.cc", "transcendental_highway.cc",
               "transcendental_xsimd.cc", "transcendental_highway_avx512.cc",
               "transcendental_xsimd_avx512.cc", "vqsort_hyle_avx2.cc",
               "vqsort_hyle_avx512.cc", "bitpack_hyle_avx2.cc",
               "bitpack_hyle_avx512.cc"),
    "sse4": ("dot_f16c.cc", "dot_avx512bf16.cc", "cvt_f16c.cc", "cvt_avx2.cc",
             "cvt_avx512bf16.cc", "transcendental_highway.cc",
             "transcendental_xsimd.cc", "transcendental_highway_avx512.cc",
             "transcendental_xsimd_avx512.cc", "vqsort_hyle_avx2.cc",
             "vqsort_hyle_avx512.cc", "bitpack_hyle_avx2.cc",
             "bitpack_hyle_avx512.cc"),
    "avx2": ("dot_f16c.cc", "dot_avx512bf16.cc", "cvt_avx512bf16.cc",
             "transcendental_highway_avx512.cc", "transcendental_xsimd_avx512.cc",
             "vqsort_hyle_avx512.cc", "bitpack_hyle_avx512.cc"),
    # AVX2 math/dtype files are intentional per-op implementation dependencies of the canonical
    # AVX-512 product. Compiler-specific sort handling is refined in check_build_graph below.
    "avx512": ("dot_f16c.cc", "dot_avx512bf16.cc", "cvt_avx512bf16.cc",
               "vqsort_hyle_avx2.cc", "bitpack_hyle_avx2.cc"),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def uses_classic_msvc_frontend(build: Path) -> bool:
    """Return true for cl.exe, but not clang-cl (which also defines _MSC_VER)."""
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        return False
    text = cache.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^CMAKE_CXX_COMPILER:FILEPATH=(.+)$", text, re.M)
    if match is None:
        return False
    compiler = match.group(1).strip().replace("\\", "/").rsplit("/", 1)[-1]
    return compiler.lower() in {"cl", "cl.exe"}


def find_dumpbin() -> Path | None:
    found = shutil.which("dumpbin")
    if found:
        return Path(found)
    roots = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value) / "Microsoft Visual Studio")
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("**/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"))
    return max(candidates, default=None)


def check_manifest(build: Path, isa: str) -> None:
    manifest = build / "src/simd/generated/simd-dispatch/target_manifest.h"
    if not manifest.is_file():
        fail(f"missing generated target manifest: {manifest}")
    text = manifest.read_text(encoding="utf-8")
    compiled = {
        match.group(1).lower()
        for match in re.finditer(r"^#define HYLE_SIMD_COMPILED_([A-Z0-9]+) 1$", text, re.M)
    }
    if compiled != {isa}:
        fail(f"manifest compiled targets are {sorted(compiled)}, expected only {isa}")
    match = re.search(r"^#define HYLE_SIMD_EXACT_VARIANT (\w+)$", text, re.M)
    if match is None or match.group(1) != isa:
        fail("manifest exact binding does not match requested ISA")

    generated = manifest.parent
    variant_sources = [
        path for path in generated.rglob("*.cc")
        if path.name.rsplit("_", 1)[-1].removesuffix(".cc") != isa
    ]
    if variant_sources:
        fail("foreign generated variants: " + ", ".join(str(path) for path in variant_sources))


def check_build_graph(build: Path, isa: str) -> None:
    ninja = build / "build.ninja"
    if not ninja.is_file():
        return
    text = ninja.read_text(encoding="utf-8", errors="replace").replace("\\", "/")
    forbidden = list(X86_FORBIDDEN_SOURCES.get(isa, ()))
    if isa == "avx512" and uses_classic_msvc_frontend(build):
        # The overall exact artifact remains canonical AVX-512, but sort is an
        # explicit compiler-specific AVX2 implementation dependency. Conversely,
        # the known-broken AVX-512 vqsort TU must not enter this cl.exe artifact.
        forbidden.remove("vqsort_hyle_avx2.cc")
        forbidden.append("vqsort_hyle_avx512.cc")
    for source in forbidden:
        if f"src/simd/" in text and source in text:
            fail(f"forbidden non-target implementation is present in build graph: {source}")


def linked_simd_objects(build: Path) -> list[Path]:
    ninja = build / "build.ninja"
    if ninja.is_file():
        for line in ninja.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("build ") or "hyle_simd." not in line:
                continue
            if "CXX_SHARED_LIBRARY_LINKER__hyle_simd_" not in line:
                continue
            inputs = line.split(": ", 1)[1].split("|", 1)[0].split()[1:]
            objects = [
                build / token.replace("\\", os.sep).replace("/", os.sep)
                for token in inputs if token.endswith((".obj", ".o"))
            ]
            if objects:
                return objects
    object_root = build / "src/simd/CMakeFiles/hyle_simd.dir"
    return list(object_root.rglob("*.obj")) + list(object_root.rglob("*.o"))


def check_imports_and_instructions(binary: Path, build: Path, isa: str) -> None:
    if os.name != "nt" or isa not in {"scalar", "sse4", "avx2", "avx512"}:
        return
    dumpbin = find_dumpbin()
    if dumpbin is None:
        fail("dumpbin is required for the Windows exact-artifact audit")

    imports = subprocess.run(
        [str(dumpbin), "/nologo", "/imports", str(binary)],
        check=True, capture_output=True, text=True, errors="replace",
    ).stdout.lower()
    for forbidden in ("cpu_features", "best_isa", "supports@simd"):
        if forbidden in imports:
            fail(f"exact artifact imports runtime CPU probe symbol containing {forbidden!r}")

    # Audit HYLE's own SIMD objects. The final PE can also contain MSVC CRT helper objects whose
    # guarded internal multiversioning is outside HYLE's target manifest; treating those as HYLE
    # code would make even /arch:SSE2 fail spuriously.
    objects = linked_simd_objects(build)
    if not objects:
        fail("no linked hyle_simd object files found")
    disassembly = "\n".join(
        subprocess.run(
            [str(dumpbin), "/nologo", "/disasm:nobytes", str(obj)],
            check=True, capture_output=True, text=True, errors="replace",
        ).stdout.lower()
        for obj in objects
    )
    if isa in {"scalar", "sse4"}:
        if re.search(r"\b(?:ymm|zmm)[0-9]+\b", disassembly):
            fail(f"{isa} artifact contains YMM/ZMM instructions")
        if re.search(r"^\s+[0-9a-f`]+\s+v[a-z0-9]+(?:\s|$)", disassembly, re.M):
            fail(f"{isa} artifact contains a VEX/EVEX instruction")
    elif isa == "avx2":
        if re.search(r"\bzmm[0-9]+\b|\{k[0-7]\}|\b(?:kmov|kortest|ktest)", disassembly):
            fail("AVX2 artifact contains AVX-512 register/opmask instructions")
        if re.search(r"\bvp(?:compress|expand)|\bvpermi2|\bvpermt2", disassembly):
            fail("AVX2 artifact contains an AVX-512-only instruction")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--isa", required=True)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()

    build = args.build_dir.resolve()
    binary = args.binary.resolve()
    isa = args.isa.lower()
    if isa not in CANONICAL:
        parser.error(f"--isa must be one of: {', '.join(CANONICAL)}")
    if not binary.is_file():
        fail(f"missing exact artifact: {binary}")
    check_manifest(build, isa)
    check_build_graph(build, isa)
    check_imports_and_instructions(binary, build, isa)
    print(f"exact artifact OK: isa={isa} binary={binary}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"exact artifact FAILED: {error}", file=sys.stderr)
        sys.exit(1)
