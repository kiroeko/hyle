"""Offline original official float/QAT source; never run by a C++ consumer."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from generate_official import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.name == 'nt': ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    recipe = Path(__file__).with_name('official.mlir').resolve();generate(recipe)
    flags = ['--iree-hal-target-device=vulkan', '--iree-vulkan-target=vp_android_baseline_2022',
        '--iree-vulkan-dispatch-abi=descriptors',
        '--iree-vulkan-target-features=spirv:v1.3,cap:Shader,cap:Int16,cap:Float16,cap:StorageBuffer16BitAccess,cap:SignedZeroInfNanPreserve,ext:SPV_KHR_16bit_storage,ext:SPV_KHR_float_controls',
        '--iree-execution-model=async-external']
    commands = []

    def run(arguments, log):
        command = [str(args.compiler.resolve()), *arguments];commands.append(command)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        (output / log).write_bytes(result.stdout);result.check_returncode()

    run(['--version'], 'compiler-version.txt')
    if '6ef606b6fa3e12bd2c12d36a07463069eb71185f' not in (output / 'compiler-version.txt').read_text():
        raise ValueError('original compiler/runtime pin must match')
    configured = output / 'configured.mlir'
    run([str(recipe), *flags, '--compile-to=executable-configurations', '-o', str(configured)], 'configure.log')
    text = configured.read_text()
    pattern = re.compile(r'attributes \{translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> workgroup_size = \[64, 1, 1\]>')
    if len(pattern.findall(text)) != 27: raise ValueError('27 original official schedules differ')
    annotation = 'attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1], target_width = 32>, translation_info'
    annotated = output / 'annotated.mlir'
    annotated.write_text(pattern.sub(lambda m:m.group().replace('attributes {translation_info', annotation), text), encoding='utf-8', newline='\n')
    binary = output / 'official.vmfb'
    run([str(annotated), *flags, '--compile-from=executable-configurations', '-o', str(binary),
         '--iree-hal-dump-executable-files-to=' + str(output / 'spirv')], 'compile.log')
    modules = sorted((output / 'spirv').glob('*.spirv.mlir'))
    control = [p for p in modules if '"SignedZeroInfNanPreserve", 32' not in p.read_text()]
    # Original global finite-bit reductions use one lane and integer tests;
    # they contain no floating arithmetic to which IEEE32 mode would apply.
    if len(modules) != 29 or len(control) != 2 or any(
        '"LocalSize", 1, 1, 1' not in p.read_text() or
        re.search(r'spirv\.(?:FAdd|FSub|FMul|FDiv|FConvert)', p.read_text())
        for p in control):
        raise ValueError('original official IEEE32 mode missing')
    exports = ['weight_decode_f32','weight_decode_bf16','fp8_roundtrip_64','fp8_roundtrip_128','fp4_roundtrip_32',
        'linear_normal_f32','linear_power_f32','linear_normal_bf16','linear_power_bf16',
        'grouped8_f32','grouped8_bf16','grouped4_bf16','encoded_block128']
    receipt = {'commands': commands, 'exports': exports, 'files': {}}
    for file in [recipe, Path(__file__), recipe.with_name('generate_official.py'),
                 recipe.with_name('generate_mlx.py'), configured, annotated, binary, output / 'compiler-version.txt', *modules]:
        raw = file.read_bytes()
        if file.suffix in ('.py','.mlir','.txt'): raw = raw.replace(b'\r\n', b'\n')
        receipt['files'][file.name] = hashlib.sha256(raw).hexdigest()
    (output / 'compilation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__': main()
