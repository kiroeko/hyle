"""Compile original standard tensor artifact recipes with the pinned compiler.

This maintainer-only offline step never runs in CMake or a C++ application.
Keep the compiler's original schedule and add the existing IEEE execution mode
contract between its public stages. Record every bytecode and compiler input.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from generate_artifacts import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.name == "nt":
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    exports = generate()
    recipe = Path(__file__).with_name("artifacts.mlir").resolve()
    flags = [
        "--iree-hal-target-device=vulkan",
        "--iree-vulkan-target=vp_android_baseline_2022",
        "--iree-vulkan-dispatch-abi=descriptors",
        "--iree-vulkan-target-features=spirv:v1.3,cap:Shader,cap:Int16,"
        "cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,"
        "ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls",
        "--iree-execution-model=async-external",
    ]
    commands = []

    def run(arguments, log):
        command = [str(args.compiler.resolve()), *arguments]
        commands.append(command)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        (output / log).write_bytes(result.stdout)
        result.check_returncode()

    version = output / "compiler-version.txt"
    run(["--version"], version.name)
    if "6ef606b6fa3e12bd2c12d36a07463069eb71185f" not in version.read_text():
        raise ValueError("compiler/runtime source pin must match")
    configured = output / "configured.mlir"
    run([str(recipe), *flags, "--compile-to=executable-configurations", "-o", str(configured)], "configure.log")
    text = configured.read_text(encoding="utf-8")
    pattern = re.compile(r"attributes \{translation_info = #iree_codegen.translation_info<"
                         r"pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> workgroup_size = \[64, 1, 1\]>")
    if len(pattern.findall(text)) != len(exports):
        raise ValueError("original artifact schedule differs from the admitted 24 entries")
    annotation = ("attributes {spirv.entry_point_abi = #spirv.entry_point_abi<"
                  "workgroup_size = [64, 1, 1], target_width = 32>, translation_info")
    annotated = output / "annotated.mlir"
    annotated.write_text(pattern.sub(lambda m: m.group().replace(
        "attributes {translation_info", annotation), text), encoding="utf-8", newline="\n")
    binary = output / "artifacts.vmfb"
    run([str(annotated), *flags, "--compile-from=executable-configurations", "-o", str(binary),
         "--iree-hal-dump-executable-files-to=" + str(output / "spirv")], "compile.log")
    modules = sorted((output / "spirv").glob("*.spirv.mlir"))
    if len(modules) != len(exports) or any('"SignedZeroInfNanPreserve", 32' not in p.read_text() for p in modules):
        raise ValueError("required IEEE mode was not emitted for every artifact entry")
    receipt = {"commands": commands, "exports": exports, "files": {}}
    for path in [recipe, Path(__file__), Path(__file__).with_name("generate_artifacts.py"),
                 configured, annotated, binary, version, *modules]:
        data = path.read_bytes()
        if path.suffix in (".py", ".mlir", ".txt"):
            data = data.replace(b"\r\n", b"\n")
        receipt["files"][path.name] = hashlib.sha256(data).hexdigest()
    (output / "compilation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
