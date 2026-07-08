#!/usr/bin/env python3
"""HYLE shader tool: turn GLSL/SPIR-V into a C header HYLE's Vulkan backend can load.

Two subcommands:

  embed   <in.spv> -o <out.h> --name NAME [--namespace NS]
      SPIR-V binary -> a C header defining an `inline constexpr hyle::compute::SpirvBlob`
      you #include into your project and pass to Kernel::load_spirv(). Pure Python, no
      dependencies -- works anywhere, including from a binary (headers+dll) HYLE release.

  compile <in.comp> -o <out.spv> [--stage STAGE] [--glslang PATH]
      GLSL -> SPIR-V by invoking a glslang executable (NOT reimplemented here). The
      compiler is found via --glslang, else $HYLE_GLSLANG, else `glslc`/`glslangValidator`
      on PATH. Use your Vulkan SDK's compiler; a binary HYLE release need not ship one.
      Add --embed <out.h> --name NAME to also run `embed` in one step (.comp -> .h).

Why a self-describing SpirvBlob (not a bare unsigned[])? The blob carries its own byte
size, so load_spirv() needs no separate length arg -- and `inline constexpr` means the
data is folded to ONE copy across every TU that includes the header (no per-TU bloat),
which is how --name/--namespace let many shaders coexist without symbol clashes.

The emitted header is stable/reproducible from source: regenerate rather than hand-edit.
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys


def _words_from_spv(path: str) -> list:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        sys.stderr.write(f"{path}: size {len(data)} not a multiple of 4 (not SPIR-V?)\n")
        sys.exit(1)
    if len(data) >= 4:
        magic = struct.unpack("<I", data[:4])[0]
        if magic != 0x07230203:  # SPIR-V magic (little-endian)
            sys.stderr.write(f"{path}: bad SPIR-V magic 0x{magic:08x} (expected 0x07230203)\n")
            sys.exit(1)
    return list(struct.unpack(f"<{len(data)//4}I", data))


def _emit_header(words: list, out_h: str, name: str, namespace: str, src: str) -> None:
    lines = [
        f"// Auto-generated from {os.path.basename(src)} by tools/hyle_shader.py. DO NOT EDIT.",
        "// Regenerate with hyle_shader.py; #include and pass the SpirvBlob to Kernel::load_spirv().",
        "#pragma once",
        '#include "hyle/compute/types.h"',
        "",
    ]
    indent = ""
    if namespace:
        lines.append(f"namespace {namespace} {{")
        indent = "  "
    lines.append(f"{indent}inline constexpr unsigned {name}_words[] = {{")
    for i in range(0, len(words), 8):
        chunk = ", ".join(f"0x{w:08x}" for w in words[i:i + 8])
        lines.append(f"{indent}  {chunk},")
    lines.append(f"{indent}}};")
    # byte_size from the array's sizeof so the two can never drift.
    lines.append(f"{indent}inline constexpr ::hyle::compute::SpirvBlob {name}{{")
    lines.append(f"{indent}    {name}_words, sizeof({name}_words)}};")
    if namespace:
        lines.append(f"}}  // namespace {namespace}")
    lines.append("")
    with open(out_h, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _find_glslang(explicit: str) -> str:
    if explicit:
        return explicit
    env = os.environ.get("HYLE_GLSLANG")
    if env:
        return env
    for cand in ("glslc", "glslangValidator", "glslang"):
        found = shutil.which(cand)
        if found:
            return found
    sys.stderr.write(
        "hyle_shader compile: no glslang found. Pass --glslang <path>, set $HYLE_GLSLANG,\n"
        "or put glslc/glslangValidator (Vulkan SDK) on PATH.\n")
    sys.exit(1)


def _compile(glslang: str, src: str, out_spv: str, stage: str) -> None:
    exe = os.path.basename(glslang).lower()
    if "glslc" in exe:  # shaderc's glslc: -fshader-stage, -o
        cmd = [glslang, src, "-o", out_spv]
        if stage:
            cmd[1:1] = [f"-fshader-stage={stage}"]
    else:  # glslang/glslangValidator: -V (Vulkan SPIR-V), -S <stage>, -o
        cmd = [glslang, "-V", src, "-o", out_spv]
        if stage:
            cmd[2:2] = ["-S", stage]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"glslang failed on {src}:\n{r.stdout}\n{r.stderr}\n")
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(prog="hyle_shader", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="SPIR-V binary -> C header (SpirvBlob)")
    pe.add_argument("input", help="input .spv")
    pe.add_argument("-o", "--output", required=True, help="output .h")
    pe.add_argument("--name", required=True, help="C identifier for the SpirvBlob")
    pe.add_argument("--namespace", default="", help="optional namespace to wrap the symbols")

    pc = sub.add_parser("compile", help="GLSL -> SPIR-V (via glslang), optionally then embed")
    pc.add_argument("input", help="input .comp/.vert/.frag ...")
    pc.add_argument("-o", "--output", required=True, help="output .spv")
    pc.add_argument("--stage", default="", help="shader stage (comp/vert/frag/...) if not inferred")
    pc.add_argument("--glslang", default="", help="path to glslc/glslangValidator")
    pc.add_argument("--embed", default="", help="also embed the .spv into this .h")
    pc.add_argument("--name", default="", help="SpirvBlob name (required with --embed)")
    pc.add_argument("--namespace", default="", help="namespace for --embed")

    args = ap.parse_args()
    if args.cmd == "embed":
        _emit_header(_words_from_spv(args.input), args.output, args.name,
                     args.namespace, args.input)
        return 0

    # compile
    _compile(_find_glslang(args.glslang), args.input, args.output, args.stage)
    if args.embed:
        if not args.name:
            sys.stderr.write("hyle_shader compile --embed requires --name\n")
            return 2
        _emit_header(_words_from_spv(args.output), args.embed, args.name,
                     args.namespace, args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
