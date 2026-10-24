"""Offline original MLX tensor programs; no application/compiler dependency."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from generate_mlx import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.name == 'nt': ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    recipe = Path(__file__).with_name('mlx.mlir').resolve()
    exports = generate(recipe)
    if recipe.read_text().count('%bits_arg: i32') != sum(name.startswith('affine_') for name in exports):
        raise ValueError('every affine export requires runtime bits')
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
        raise ValueError('compiler/runtime source pin must match')
    configured = output / 'configured.mlir'
    run([str(recipe), *flags, '--compile-to=executable-configurations', '-o', str(configured)], 'configure.log')
    text = configured.read_text()
    pattern = re.compile(r'attributes \{translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> workgroup_size = \[64, 1, 1\]>')
    expected = {f'{fmt}_{kind}_{dtype}' for fmt in ('affine', 'mx4', 'mx8')
                for kind in ('linear', 'grouped') for dtype in ('f32', 'f16', 'bf16')}
    expected.update(f'affine_embedding_{dtype}' for dtype in ('f32', 'f16', 'bf16'))
    expected.update(f'affine_{kind}_{activation}_{scale}'
                    for kind in ('linear', 'grouped')
                    for activation in ('f32', 'f16', 'bf16')
                    for scale in ('f32', 'f16', 'bf16') if activation != scale)
    expected.update(f'affine_{kind}_as_stored_{activation}'
                    for kind in ('linear', 'grouped')
                    for activation in ('f32', 'f16', 'bf16'))
    expected.update(f'affine_{kind}_as_stored_{activation}_{scale}'
                    for kind in ('linear', 'grouped')
                    for activation in ('f32', 'f16', 'bf16')
                    for scale in ('f32', 'f16', 'bf16') if activation != scale)
    declarations = dict(re.findall(r'iree\.abi\.declaration = "async func @(\w+)\(([^"]*)\) ->', text))
    source_signatures = dict(re.findall(r'func\.func @(\w+)\(([^\n]*)\) ->', recipe.read_text()))
    if set(exports) != expected or set(declarations) != expected:
        raise ValueError('MLX same-type and finite mixed-source export set differs')
    types = lambda signature: re.findall(r': (tensor<[^>]+>|i32)', signature)
    if any(types(declarations[name]) != types(source_signatures[name]) for name in expected):
        raise ValueError('compiled MLX argument signature differs from its recipe')
    declaration_results = dict(re.findall(r'iree\.abi\.declaration = "async func @(\w+)\([^"\n]*\) -> \(%output0: (tensor<[^>]+>)\)"', text))
    source_results = dict(re.findall(r'func\.func @(\w+)\([^\n]*\) -> (tensor<[^>]+>)', recipe.read_text()))
    if declaration_results != source_results:
        raise ValueError('compiled MLX result signature differs from its recipe')
    schedules = len(pattern.findall(text))
    if not schedules or schedules != text.count('translation_info ='):
        raise ValueError('MLX has an unrecognized dispatch schedule')
    annotation = 'attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1], target_width = 32>, translation_info'
    annotated = output / 'annotated.mlir'
    annotated.write_text(pattern.sub(lambda m: m.group().replace('attributes {translation_info', annotation), text), encoding='utf-8', newline='\n')
    binary = output / 'mlx.vmfb'
    run([str(annotated), *flags, '--compile-from=executable-configurations', '-o', str(binary),
         '--iree-hal-dump-executable-files-to=' + str(output / 'spirv')], 'compile.log')
    modules = sorted((output / 'spirv').glob('*.spirv.mlir'))
    if len(modules) != schedules or any('"SignedZeroInfNanPreserve", 32' not in p.read_text() for p in modules):
        raise ValueError('original MLX IEEE32 execution mode missing')
    if any(name.encode() not in binary.read_bytes() for name in expected):
        raise ValueError('compiled MLX bytecode export name missing')
    receipt = {'commands': commands, 'exports': exports, 'deduplicated_schedules': schedules,
               'compiled_argument_signatures': {name: types(declarations[name]) for name in sorted(expected)},
               'compiled_result_signatures': declaration_results,
               'mixed_affine_sources': [dict(activation=a, scale_bias=s, output='f32')
                   for a in ('f32', 'f16', 'bf16') for s in ('f32', 'f16', 'bf16') if a != s],
               'affine_runtime_bits': [4, 5, 6, 8],
               'affine_scalar_order': ['group_size', 'bits'],
               'packing': 'continuous low-bit-first U32 stream, words=(K/32)*bits',
               'workspace': 'transpose: one selected K group; as-stored: one physical K output group and at most64 reduction-N rows, including tail; selected rows for embedding. Grouped input retains original physical activation span',
               'as_stored_affine': 'stored[N,K] unchanged; A reduction=N/output=K. Only K is quantization-group aligned. No full floating bank.',
               'grouped_activation_scalars': ['selection_slots', 'activation_slots', 'activation_token_stride_elements'],
               'files': {}}
    for file in [recipe, Path(__file__), recipe.with_name('generate_mlx.py'), configured,
                 annotated, binary, output / 'compiler-version.txt', *modules]:
        raw = file.read_bytes()
        if file.suffix in ('.py', '.mlir', '.txt'): raw = raw.replace(b'\r\n', b'\n')
        receipt['files'][file.name] = hashlib.sha256(raw).hexdigest()
    (output / 'compilation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
