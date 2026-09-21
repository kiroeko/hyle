"""Offline recipe for the locked original IREE compiler; never run by CMake.

Keep original arith/linalg lowering and workgroup selection. The only annotation
added between the compiler's public stages declares the required IEEE special
value behavior, which Vulkan otherwise permits a driver to optimize away.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    recipe = Path(__file__).with_name("precision.mlir").resolve()
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
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / log).write_bytes(result.stdout)
        result.check_returncode()

    configured = output / "configured.mlir"
    run([str(recipe), *flags, "--compile-to=executable-configurations", "-o", str(configured)], "configure.log")
    text = configured.read_text(encoding="utf-8")
    # Fail closed if the locked compiler selects a different schedule. Do not
    # replace its arithmetic, tiling, descriptor, synchronization or VM code.
    selected = (
        "attributes {translation_info = #iree_codegen.translation_info<"
        "pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> workgroup_size = [64, 1, 1]>"
    )
    if text.count(selected) != 2:
        raise ValueError("locked precision program workgroup selection differs")
    annotation = (
        "attributes {spirv.entry_point_abi = #spirv.entry_point_abi<"
        "workgroup_size = [64, 1, 1], target_width = 32>, translation_info"
    )
    annotated = output / "annotated.mlir"
    annotated.write_text(text.replace(selected, selected.replace("attributes {translation_info", annotation)), encoding="utf-8", newline="\n")
    binary = output / "precision.vmfb"
    run([str(annotated), *flags, "--compile-from=executable-configurations", "-o", str(binary),
         "--iree-hal-dump-executable-files-to=" + str(output / "spirv")], "compile.log")
    modules = list((output / "spirv").glob("*.spirv.mlir"))
    if len(modules) != 2 or any('"SignedZeroInfNanPreserve", 32' not in p.read_text() for p in modules):
        raise ValueError("required IEEE execution mode was not emitted")
    receipt = {"commands": commands, "files": {}}
    for path in [recipe, Path(__file__), configured, annotated, binary, *modules]:
        receipt["files"][path.name] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n") if path.suffix in (".py", ".mlir") else path.read_bytes()).hexdigest()
    (output / "compilation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
