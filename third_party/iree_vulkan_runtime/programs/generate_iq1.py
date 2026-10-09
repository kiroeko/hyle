"""Original Unsloth IQ1 GPU grids and 32-lane contribution program in tensor IR.

The scalar equations and lane-strided reduction are from registered mul_mv.metal,
not a new codec. Native IREE owns the loop, dispatches and temporary storage.
"""
import hashlib
import json
from pathlib import Path
import re


def sources():
    root = Path(__file__).resolve().parents[3]
    vendor = root / 'third_party/ggml_metal_iq1'
    patches = json.loads((vendor / 'patches.json').read_text(encoding='utf-8'))
    origin = json.loads((vendor / 'origins.json').read_text(encoding='utf-8'))
    grids = {}
    for file in patches['files']:
        if file['path'] != 'src/ggml-common.h':
            continue
        for edit in file['edits']:
            for fmt in ('xs', 'xxs', 'xxxs'):
                name = f'iq1_{fmt}_grid_gpu'
                match = re.search(r'GGML_TABLE_BEGIN\(uint32_t, ' + name + r', [^)]+\).*?GGML_TABLE_END\(\)', edit['after'], re.S)
                if not match:
                    continue
                fragment = match.group()
                expected = next(f['sha256'] for f in origin['fragments'] if f.get('name') == name)
                if hashlib.sha256(fragment.encode()).hexdigest() != expected:
                    raise ValueError(f'original grid identity differs: {name}')
                words = [int(v, 16) for v in re.findall(r'0x[0-9a-fA-F]+', fragment)]
                if len(words) != {'xs': 1024, 'xxs': 512, 'xxxs': 256}[fmt]:
                    raise ValueError('original grid dimensions differ')
                grids[fmt] = words
    if len(grids) != 3:
        raise ValueError('all three original grids required')
    return grids


def program(fmt, grid):
    size = {'xs': 46, 'xxs': 42, 'xxxs': 38}[fmt]
    gt = f'tensor<{len(grid)}xi32>'
    constants = {0, 1, 2, 3, 4, 7, 8, 15, 16, 32, 34, 42, 256, size}
    s = [f'func.func @iq1_{fmt}(%a: tensor<?x?xf32>, %w: tensor<?xi8>, %ids: tensor<?xi32>, %n_arg: i32) -> tensor<?x?xf32> {{']
    s += [f'%c{n} = arith.constant {n} : index' for n in sorted(constants)]
    s += ['%mask255 = arith.constant 255 : i32', '%mask15 = arith.constant 15 : i32',
          '%mask7 = arith.constant 7 : i32', '%mask8 = arith.constant 8 : i32',
          '%shift4 = arith.constant 4 : i32', '%shift8 = arith.constant 8 : i32',
          '%one = arith.constant 1 : i32', '%two = arith.constant 2 : i32',
          '%zero = arith.constant 0.0 : f32', '%plus = arith.constant -0.875 : f32',
          '%minus = arith.constant -1.125 : f32',
          '%r = tensor.dim %a, %c0 : tensor<?x?xf32>',
          '%k = tensor.dim %a, %c1 : tensor<?x?xf32>',
          '%n = arith.index_cast %n_arg : i32 to index',
          '%nb32 = arith.divui %k, %c32 : index',
          f'%grid = arith.constant dense<[{", ".join(str(x if x < 2**31 else x - 2**32) for x in grid)}]> : {gt}',
          '%empty = tensor.empty(%r, %n) : tensor<?x?x32xf32>',
          '%partial = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty : tensor<?x?x32xf32>) {',
          '  ^bb0(%unused: f32):', '%row = linalg.index 0 : index',
          '%col = linalg.index 1 : index', '%lane = linalg.index 2 : index',
          '%id = tensor.extract %ids[%row] : tensor<?xi32>',
          '%expert = arith.index_cast %id : i32 to index',
          '%en = arith.muli %expert, %n : index', '%wn = arith.addi %en, %col : index',
          '%blocks = arith.divui %k, %c256 : index', '%row_blocks = arith.muli %wn, %blocks : index',
          f'%row_base = arith.muli %row_blocks, %c{size} : index',
          '%sumf = scf.for %ib32 = %lane to %nb32 step %c32 iter_args(%acc = %zero) -> (f32) {',
          '%ibl = arith.divui %ib32, %c8 : index', '%ib = arith.remui %ib32, %c8 : index',
          f'%block_offset = arith.muli %ibl, %c{size} : index',
          '%base = arith.addi %row_base, %block_offset : index',
          '%a_base = arith.muli %ib32, %c32 : index',
          '%sumy = scf.for %j = %c0 to %c32 step %c1 iter_args(%sy = %zero) -> (f32) {',
          '%aj = arith.addi %a_base, %j : index',
          '%av = tensor.extract %a[%row, %aj] : tensor<?x?xf32>',
          '%next_y = arith.addf %sy, %av : f32', 'scf.yield %next_y : f32', '}',
          '%dlo = tensor.extract %w[%base] : tensor<?xi8>',
          '%dhi_offset = arith.addi %base, %c1 : index',
          '%dhi = tensor.extract %w[%dhi_offset] : tensor<?xi8>',
          '%dlo_ext = arith.extui %dlo : i8 to i32', '%dhi_ext = arith.extui %dhi : i8 to i32',
          '%dlo32 = arith.andi %dlo_ext, %mask255 : i32', '%dhi32 = arith.andi %dhi_ext, %mask255 : i32',
          '%dhi_shift = arith.shli %dhi32, %shift8 : i32',
          '%dbits32 = arith.ori %dlo32, %dhi_shift : i32',
          '%dbits = arith.trunci %dbits32 : i32 to i16',
          '%dh = arith.bitcast %dbits : i16 to f16', '%d = arith.extf %dh : f16 to f32']
    if fmt in ('xs', 'xxs'):
        s += ['%qh_begin = arith.addi %base, %c34 : index',
              '%qh_offset = arith.addi %qh_begin, %ib : index',
              '%qh_byte = tensor.extract %w[%qh_offset] : tensor<?xi8>',
              '%qh_ext = arith.extui %qh_byte : i8 to i32',
              '%qh = arith.andi %qh_ext, %mask255 : i32']
    if fmt == 'xxs':
        s += ['%scale_shift = arith.shrui %qh, %shift4 : i32',
              '%scale = arith.andi %scale_shift, %mask7 : i32',
              '%sign_shift = arith.constant 7 : i32',
              '%sign = arith.shrui %qh, %sign_shift : i32']
    else:
        sc_offset = 42 if fmt == 'xs' else 34
        s += [f'%sc_begin = arith.addi %base, %c{sc_offset} : index',
              '%sc_index = arith.divui %ib, %c2 : index', '%sc_offset = arith.addi %sc_begin, %sc_index : index',
              '%sc_byte = tensor.extract %w[%sc_offset] : tensor<?xi8>',
              '%sc_ext = arith.extui %sc_byte : i8 to i32',
              '%sc = arith.andi %sc_ext, %mask255 : i32',
              '%sc_lane = arith.remui %ib, %c2 : index', '%sc_shift_index = arith.muli %sc_lane, %c4 : index',
              '%sc_shift = arith.index_cast %sc_shift_index : index to i32',
              '%sc_shifted = arith.shrui %sc, %sc_shift : i32',
              '%scale = arith.andi %sc_shifted, %mask7 : i32',
              '%sign_shift = arith.constant 3 : i32',
              '%sign0 = arith.shrui %sc_shifted, %sign_shift : i32',
              '%sign = arith.andi %sign0, %one : i32']
    s += [
          '%izero = arith.constant 0 : i32', '%negative = arith.cmpi ne, %sign, %izero : i32',
          '%delta = arith.select %negative, %minus, %plus : f32',
          '%scale2 = arith.muli %scale, %two : i32', '%scale_odd = arith.addi %scale2, %one : i32',
          '%factor = arith.uitofp %scale_odd : i32 to f32',
          '%qs_start0 = arith.muli %ib, %c4 : index', '%qs_start = arith.addi %qs_start0, %c2 : index',
          '%qs_base = arith.addi %base, %qs_start : index']
    for group in range(4):
        s += [f'%qi{group} = arith.addi %qs_base, %c{group} : index',
              f'%qbyte{group} = tensor.extract %w[%qi{group}] : tensor<?xi8>',
              f'%qext{group} = arith.extui %qbyte{group} : i8 to i32',
              f'%qlo{group} = arith.andi %qext{group}, %mask255 : i32']
        if fmt != 'xxxs':
            shift = group * (2 if fmt == 'xs' else 1)
            mask = 3 if fmt == 'xs' else 1
            s += [f'%high_shift{group} = arith.constant {shift} : i32',
                  f'%high_mask{group} = arith.constant {mask} : i32',
                  f'%high0_{group} = arith.shrui %qh, %high_shift{group} : i32',
                  f'%high1_{group} = arith.andi %high0_{group}, %high_mask{group} : i32',
                  f'%high2_{group} = arith.shli %high1_{group}, %shift8 : i32',
                  f'%q{group} = arith.ori %qlo{group}, %high2_{group} : i32']
        else:
            s += [f'%q{group} = arith.andi %qlo{group}, %mask255 : i32']
        s += [f'%qindex{group} = arith.index_castui %q{group} : i32 to index',
              f'%word{group} = tensor.extract %grid[%qindex{group}] : {gt}']
    s += ['%dot = scf.for %j = %c0 to %c4 step %c1 iter_args(%sd = %zero) -> (f32) {',
          '%js = arith.index_cast %j : index to i32',
          '%byte_bits = arith.muli %js, %shift8 : i32']
    for group in range(4):
        s += [f'%byte{group} = arith.shrui %word{group}, %byte_bits : i32']
        for high in range(2):
            name = f'{group}_{high}'
            offset = group * 8 + high * 4
            s += [f'%offset{name} = arith.constant {offset} : index',
                  f'%aj{name} = arith.addi %a_base, %offset{name} : index',
                  f'%ak{name} = arith.addi %aj{name}, %j : index',
                  f'%av{name} = tensor.extract %a[%row, %ak{name}] : tensor<?x?xf32>']
            raw = f'%byte{group}'
            if high:
                s += [f'%nibble_high{name} = arith.shrui {raw}, %shift4 : i32']
                raw = f'%nibble_high{name}'
            s += [f'%nibble{name} = arith.andi {raw}, %mask15 : i32',
                  f'%value{name} = arith.uitofp %nibble{name} : i32 to f32',
                  f'%term{name} = arith.mulf %av{name}, %value{name} : f32']
    terms = [f'%term{g}_{h}' for g in range(4) for h in range(2)]
    result = terms[0]
    for i, term in enumerate(terms[1:]):
        s += [f'%j_sum{i} = arith.addf {result}, {term} : f32']; result = f'%j_sum{i}'
    s += [f'%next_dot = arith.addf %sd, {result} : f32', 'scf.yield %next_dot : f32', '}',
          '%delta_sum = arith.mulf %sumy, %delta : f32', '%adjusted = arith.addf %dot, %delta_sum : f32',
          '%scaled_d = arith.mulf %d, %adjusted : f32', '%scaled = arith.mulf %scaled_d, %factor : f32',
          '%next = arith.addf %acc, %scaled : f32', 'scf.yield %next : f32', '}',
          'linalg.yield %sumf : f32', '} -> tensor<?x?x32xf32>',
          '%bar = util.optimization_barrier %partial : tensor<?x?x32xf32>',
          '%tied = flow.tensor.tie_shape %bar : tensor<?x?x32xf32>{%r, %n}',
          '%out_empty = tensor.empty(%r, %n) : tensor<?x?xf32>',
          '%init = linalg.fill ins(%zero : f32) outs(%out_empty : tensor<?x?xf32>) -> tensor<?x?xf32>',
          '%out = linalg.reduce ins(%tied : tensor<?x?x32xf32>) outs(%init : tensor<?x?xf32>) dimensions = [2]',
          '  (%v: f32, %sum: f32) {', '%combined = arith.addf %v, %sum : f32',
          'linalg.yield %combined : f32', '}', 'return %out : tensor<?x?xf32>', '}']
    return '\n'.join(s)


def generate(output):
    grids = sources()
    output.write_text('module {\n' + '\n'.join(program(fmt, grid) for fmt, grid in grids.items()) + '\n}\n', encoding='utf-8', newline='\n')
    return [f'iq1_{fmt}' for fmt in grids]


if __name__ == '__main__':
    generate(Path(__file__).with_name('iq1.mlir'))
