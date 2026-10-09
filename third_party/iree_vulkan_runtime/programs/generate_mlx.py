"""Offline original MLX tensor group programs; see NOTICE.md for sources/licenses."""
from pathlib import Path

TYPES = {'f32': 'f32', 'f16': 'f16', 'bf16': 'i16'}


def wide(value, dtype, name):
    if dtype == 'f32':
        return [], value
    if dtype == 'bf16':
        return [f'%{name}_bf = arith.bitcast {value} : i16 to bf16',
                f'%{name} = arith.extf %{name}_bf : bf16 to f32'], f'%{name}'
    return [f'%{name} = arith.extf {value} : f16 to f32'], f'%{name}'


def narrow(value, dtype, name):
    if dtype == 'f32':
        return [], value
    result = [f'%{name}_float = arith.truncf {value} : f32 to {dtype}']
    if dtype == 'bf16':
        result.append(f'%{name} = arith.bitcast %{name}_float : bf16 to i16')
        return result, f'%{name}'
    return result, f'%{name}_float'


def group_program(fmt, dtype, grouped):
    t = TYPES[dtype]
    affine = fmt == 'affine'
    bits = 8 if fmt == 'mx8' else 4
    pack = 32 // bits
    a_type, w_type = f'tensor<?x?x{t}>', 'tensor<?x?x?xi32>'
    scale_type = f'tensor<?x?x?x{t if affine else "i8"}>'
    out_type, value_type = f'tensor<?x?x{t}>', 'tensor<?x?x?xf32>'
    args = [f'%a: {a_type}', f'%w: {w_type}', f'%s: {scale_type}']
    if affine: args.append(f'%b: {scale_type}')
    if grouped: args.append('%ids: tensor<?xi32>')
    args.append('%group_arg: i32')
    name = f'{fmt}_{"grouped" if grouped else "linear"}_{dtype}'
    lines = [f'func.func @{name}({", ".join(args)}) -> {out_type} {{',
             '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index',
             f'%pack = arith.constant {pack} : index', f'%bits = arith.constant {bits} : index',
             f'%mask = arith.constant {(1 << bits) - 1} : i32',
             '%group = arith.index_cast %group_arg : i32 to index',
             f'%rows = tensor.dim %a, %c0 : {a_type}',
             f'%k = tensor.dim %a, %c1 : {a_type}',
             f'%n = tensor.dim %w, %c1 : {w_type}',
             '%groups = arith.divui %k, %group : index',
             '%zero = arith.constant 0.0 : f32',
             '%empty_sum = tensor.empty(%rows, %n) : tensor<?x?xf32>',
             '%sum0 = linalg.fill ins(%zero : f32) outs(%empty_sum : tensor<?x?xf32>) -> tensor<?x?xf32>']
    if fmt == 'mx4':
        lines.append('%fp4 = arith.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]> : tensor<16xf32>')
    lines += ['%total = scf.for %block = %c0 to %groups step %c1 iter_args(%sum = %sum0) -> (tensor<?x?xf32>) {',
              '  %begin = arith.muli %block, %group : index',
              '  %empty_a = tensor.empty(%rows, %c1, %group) : tensor<?x?x?xf32>',
              '  %ax = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_a : tensor<?x?x?xf32>) {',
              '    ^bb0(%unused_a: f32):', '      %r = linalg.index 0 : index',
              '      %j = linalg.index 2 : index', '      %ac = arith.addi %begin, %j : index',
              f'      %av = tensor.extract %a[%r, %ac] : {a_type}']
    conversions, activation = wide('%av', dtype, 'activation')
    lines += ['      ' + line for line in conversions] + [f'      linalg.yield {activation} : f32',
              '  } -> tensor<?x?x?xf32>',
              '  %empty_w = tensor.empty(%rows, %n, %group) : tensor<?x?x?xf32>',
              '  %wx = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>], iterator_types = ["parallel","parallel","parallel"]} outs(%empty_w : tensor<?x?x?xf32>) {',
              '    ^bb0(%unused_w: f32):', '      %r = linalg.index 0 : index',
              '      %col = linalg.index 1 : index', '      %j = linalg.index 2 : index']
    if grouped:
        lines += ['      %id = tensor.extract %ids[%r] : tensor<?xi32>',
                  '      %expert = arith.index_cast %id : i32 to index']
    else: lines.append('      %expert = arith.constant 0 : index')
    lines += ['      %column = arith.addi %begin, %j : index',
              '      %wc = arith.divui %column, %pack : index',
              '      %lane = arith.remui %column, %pack : index',
              '      %shift_index = arith.muli %lane, %bits : index',
              '      %shift = arith.index_cast %shift_index : index to i32',
              f'      %word = tensor.extract %w[%expert, %col, %wc] : {w_type}',
              '      %shifted = arith.shrui %word, %shift : i32',
              '      %code = arith.andi %shifted, %mask : i32']
    if affine:
        lines += ['      %cv = arith.uitofp %code : i32 to f32',
                  f'      %sv = tensor.extract %s[%expert, %col, %block] : {scale_type}',
                  f'      %bv = tensor.extract %b[%expert, %col, %block] : {scale_type}']
        sc, s = wide('%sv', dtype, 'scale'); bc, b = wide('%bv', dtype, 'bias')
        lines += ['      ' + line for line in sc + bc]
        lines += [f'      %scaled = arith.mulf %cv, {s} : f32',
                  f'      %value = arith.addf %scaled, {b} : f32']
    elif fmt == 'mx4':
        lines += ['      %ci = arith.index_castui %code : i32 to index',
                  '      %value = tensor.extract %fp4[%ci] : tensor<16xf32>']
    else:
        lines += ['      %encoded = arith.trunci %code : i32 to i8',
                  '      %fp8 = arith.bitcast %encoded : i8 to f8E4M3FN',
                  '      %decoded_value = arith.extf %fp8 : f8E4M3FN to f32',
                  # E4M3FN reserves both signs of exponent15/mantissa7.
                  # The locked original lowering recognizes only +NaN.
                  '      %nan_mask = arith.constant 127 : i32',
                  '      %magnitude = arith.andi %code, %nan_mask : i32',
                  '      %is_nan = arith.cmpi eq, %magnitude, %nan_mask : i32',
                  '      %nan_bits = arith.constant 2143289344 : i32',
                  '      %nan = arith.bitcast %nan_bits : i32 to f32',
                  '      %value = arith.select %is_nan, %nan, %decoded_value : f32']
    lines += ['      linalg.yield %value : f32', '  } -> tensor<?x?x?xf32>',
              '  %a_bar = util.optimization_barrier %ax : tensor<?x?x?xf32>',
              '  %w_bar = util.optimization_barrier %wx : tensor<?x?x?xf32>',
              '  %ab = flow.tensor.tie_shape %a_bar : tensor<?x?x?xf32>{%rows, %c1, %group}',
              '  %wb = flow.tensor.tie_shape %w_bar : tensor<?x?x?xf32>{%rows, %n, %group}',
              '  %empty_dot = tensor.empty(%rows, %c1, %n) : tensor<?x?x?xf32>',
              '  %dot0 = linalg.fill ins(%zero : f32) outs(%empty_dot : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>',
              '  %dot = linalg.batch_matmul indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d2,d3)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>] ins(%ab, %wb : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%dot0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>']
    if not affine:
        st = f'tensor<?x?x{t}>'
        lines += [f'  %empty_scale = tensor.empty(%rows, %n) : {st}',
                  f'  %typed_scale = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}} outs(%empty_scale : {st}) {{',
                  f'    ^bb0(%unused_scale: {t}):', '      %r = linalg.index 0 : index',
                  '      %col = linalg.index 1 : index']
        if grouped:
            lines += ['      %id = tensor.extract %ids[%r] : tensor<?xi32>',
                      '      %expert = arith.index_cast %id : i32 to index']
        else: lines.append('      %expert = arith.constant 0 : index')
        lines += [f'      %sv = tensor.extract %s[%expert, %col, %block] : {scale_type}',
                  '      %u = arith.extui %sv : i8 to i32',
                  '      %z = arith.constant 0 : i32', '      %shift = arith.constant 23 : i32',
                  '      %subnormal = arith.constant 4194304 : i32',
                  '      %raw = arith.shli %u, %shift : i32',
                  '      %is_zero = arith.cmpi eq, %u, %z : i32',
                  '      %scale_bits = arith.select %is_zero, %subnormal, %raw : i32',
                  '      %factor = arith.bitcast %scale_bits : i32 to f32']
        conversions, factor = narrow('%factor', dtype, 'source_scale')
        lines += ['      ' + line for line in conversions] + [f'      linalg.yield {factor} : {t}', f'  }} -> {st}',
                  f'  %s_bar = util.optimization_barrier %typed_scale : {st}',
                  f'  %sb = flow.tensor.tie_shape %s_bar : {st}{{%rows, %n}}']
    lines += ['  %next = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]} outs(%empty_sum : tensor<?x?xf32>) {',
              '    ^bb0(%unused_sum: f32):', '      %r = linalg.index 0 : index',
              '      %col = linalg.index 1 : index',
              '      %d = tensor.extract %dot[%r, %c0, %col] : tensor<?x?x?xf32>',
              '      %old = tensor.extract %sum[%r, %col] : tensor<?x?xf32>']
    if affine: product = '%d'
    else:
        lines.append(f'      %sval = tensor.extract %sb[%r, %col] : tensor<?x?x{t}>')
        conversions, factor = wide('%sval', dtype, 'source_factor')
        lines += ['      ' + line for line in conversions] + [f'      %scaled_dot = arith.mulf %d, {factor} : f32']
        product = '%scaled_dot'
    lines += [f'      %value = arith.addf %old, {product} : f32', '      linalg.yield %value : f32',
              '  } -> tensor<?x?xf32>', '  scf.yield %next : tensor<?x?xf32>', '}',
              f'%out_empty = tensor.empty(%rows, %n) : {out_type}',
              f'%out = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}} ins(%total : tensor<?x?xf32>) outs(%out_empty : {out_type}) {{',
              f'  ^bb0(%v: f32, %unused_out: {t}):']
    conversions, output = narrow('%v', dtype, 'converted')
    lines += ['    ' + line for line in conversions] + [f'    linalg.yield {output} : {t}',
              f'}} -> {out_type}', f'return %out : {out_type}', '}']
    return '\n'.join(lines), name


def embedding(dtype):
    t = TYPES[dtype]
    st = f'tensor<?x?x{t}>'
    lines = [f'func.func @affine_embedding_{dtype}(%w: tensor<?x?xi32>, %s: {st}, %b: {st}, %ids: tensor<?xi32>, %group_arg: i32) -> {st} {{',
             '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index',
             '%c8 = arith.constant 8 : index', '%c4 = arith.constant 4 : index',
             '%mask = arith.constant 15 : i32',
             '%group = arith.index_cast %group_arg : i32 to index',
             '%rows = tensor.dim %ids, %c0 : tensor<?xi32>',
             '%packed_columns = tensor.dim %w, %c1 : tensor<?x?xi32>',
             '%k = arith.muli %packed_columns, %c8 : index',
             f'%empty = tensor.empty(%rows, %k) : {st}',
             f'%product = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}} outs(%empty : {st}) {{',
             f'  ^bb0(%unused: {t}):', '%r = linalg.index 0 : index', '%j = linalg.index 1 : index',
             '%id = tensor.extract %ids[%r] : tensor<?xi32>',
             '%row = arith.index_cast %id : i32 to index',
             '%packed_column = arith.divui %j, %c8 : index',
             '%lane = arith.remui %j, %c8 : index',
             '%shift_index = arith.muli %lane, %c4 : index',
             '%shift = arith.index_cast %shift_index : index to i32',
             '%word = tensor.extract %w[%row, %packed_column] : tensor<?x?xi32>',
             '%shifted = arith.shrui %word, %shift : i32',
             '%code = arith.andi %shifted, %mask : i32',
             '%value = arith.uitofp %code : i32 to f32',
             '%block = arith.divui %j, %group : index',
             f'%sv = tensor.extract %s[%row, %block] : {st}']
    conversions, scale = wide('%sv', dtype, 'scale')
    lines += conversions + [f'%scaled = arith.mulf %value, {scale} : f32']
    conversions, product = narrow('%scaled', dtype, 'product_value')
    lines += conversions + [f'linalg.yield {product} : {t}', f'}} -> {st}',
              f'%barrier = util.optimization_barrier %product : {st}',
              f'%typed = flow.tensor.tie_shape %barrier : {st}{{%rows, %k}}',
              f'%out = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}} ins(%typed : {st}) outs(%empty : {st}) {{',
              f'  ^bb0(%p: {t}, %unused_out: {t}):', '%r = linalg.index 0 : index', '%j = linalg.index 1 : index',
              '%id = tensor.extract %ids[%r] : tensor<?xi32>',
              '%row = arith.index_cast %id : i32 to index', '%block = arith.divui %j, %group : index',
              f'%bv = tensor.extract %b[%row, %block] : {st}']
    pc, p = wide('%p', dtype, 'p_wide');bc, b = wide('%bv', dtype, 'b_wide')
    lines += pc + bc + [f'%added = arith.addf {p}, {b} : f32']
    conversions, result = narrow('%added', dtype, 'out_value')
    lines += conversions + [f'linalg.yield {result} : {t}', f'}} -> {st}', f'return %out : {st}', '}']
    return '\n'.join(lines), f'affine_embedding_{dtype}'


def generate(destination):
    entries, names = [], []
    for fmt in ('affine', 'mx4', 'mx8'):
        for dtype in TYPES:
            for grouped in (False, True):
                text, name = group_program(fmt, dtype, grouped)
                entries.append(text);names.append(name)
    for dtype in TYPES:
        text, name = embedding(dtype)
        entries.append(text);names.append(name)
    destination.write_text('module {\n' + '\n'.join(entries) + '\n}\n', encoding='utf-8', newline='\n')
    return names


if __name__ == '__main__':
    generate(Path(__file__).with_name('mlx.mlir'))
