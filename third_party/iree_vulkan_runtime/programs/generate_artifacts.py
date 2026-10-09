"""Offline original NF4/AWQ/CT tensor recipes for the locked IREE compiler.

No custom codegen, tiling, kernels, or runtime compiler dependency. Sources and
licenses are recorded in NOTICE.md. BF16 storage follows the already qualified
IREE i16 bit carrier; source multiplication rounding precedes final conversion.
"""
from pathlib import Path


TYPES = {"f32": "f32", "f16": "f16", "bf16": "i16"}


def convert(value, source, destination, result):
    lines = []
    if source == "bf16":
        lines.append(f"%{result}_bf = arith.bitcast {value} : i16 to bf16")
        value = f"%{result}_bf"
    if source != "f32":
        lines.append(f"%{result}_wide = arith.extf {value} : {source} to f32")
        value = f"%{result}_wide"
    if destination != "f32":
        lines.append(f"%{result}_narrow = arith.truncf {value} : f32 to {destination}")
        value = f"%{result}_narrow"
    if destination == "bf16":
        lines.append(f"%{result} = arith.bitcast {value} : bf16 to i16")
    else:
        lines.append(f"%{result} = arith.addf {value}, %{destination}_zero : {destination}")
        # An identity add would change signed zero. Use the operand directly.
        lines.pop()
        return lines, value
    return lines, f"%{result}"


def entry(kind, scale, output):
    nf4 = kind in ("nf4", "nf4_nested")
    nested = kind == "nf4_nested"
    packed_type = "tensor<?x?xi8>" if nf4 else "tensor<?x?xi32>"
    scale_type = f"tensor<?x?x{TYPES[scale]}>"
    args = [f"%packed: {packed_type}"]
    if nf4:
        if nested:
            args += ["%scale_codes: tensor<?xi8>", "%outer_scales: tensor<?xf32>",
                     "%scale_book: tensor<256xf32>", "%offset: tensor<1xf32>"]
        else:
            args.append("%scales: tensor<?xf32>")
        args.append("%book: tensor<16xf32>")
    else:
        if kind == "awq":
            args.append("%zeros: tensor<?x?xi32>")
        args.append(f"%scales: {scale_type}")
    args.append("%group_arg: i32")
    name = f"{kind}_{output}" if nf4 else f"{kind}_{scale}_{output}"
    out_type = f"tensor<?x?x{TYPES[output]}>"
    lines = [f"  func.func @{name}({', '.join(args)}) -> {out_type} {{"]
    body = ["%c0 = arith.constant 0 : index", "%c1 = arith.constant 1 : index",
            "%c2 = arith.constant 2 : index", "%c4 = arith.constant 4 : index",
            "%c8 = arith.constant 8 : index", "%c256 = arith.constant 256 : index",
            "%mask = arith.constant 15 : i32", "%eight = arith.constant 8 : i32",
            "%group = arith.index_cast %group_arg : i32 to index",
            f"%d0 = tensor.dim %packed, %c0 : {packed_type}",
            f"%d1 = tensor.dim %packed, %c1 : {packed_type}"]
    if kind == "awq":
        body += ["%n = arith.muli %d1, %c8 : index"]
        n, k = "%n", "%d0"
    else:
        body += [f"%k = arith.muli %d1, {'%c2' if nf4 else '%c8'} : index"]
        n, k = "%d0", "%k"
    body += [f"%empty = tensor.empty({n}, {k}) : {out_type}",
             "%result = linalg.generic {",
             "  indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],",
             "  iterator_types = [\"parallel\", \"parallel\"]",
             f"}} outs(%empty : {out_type}) {{",
             f"  ^bb0(%unused: {TYPES[output]}):",
             "    %row = linalg.index 0 : index", "    %column = linalg.index 1 : index"]
    inner = []
    if nf4:
        inner += ["%packed_column = arith.divui %column, %c2 : index",
                  "%lane = arith.remui %column, %c2 : index",
                  f"%byte = tensor.extract %packed[%row, %packed_column] : {packed_type}",
                  "%word = arith.extui %byte : i8 to i32",
                  "%high = arith.cmpi eq, %lane, %c0 : index",
                  "%high_word = arith.shrui %word, %eight : i32"]
        # NF4 high nibble is bits 4..7, not an eight-bit shift.
        inner[-1] = "%four_bits = arith.constant 4 : i32"
        inner += ["%high_word = arith.shrui %word, %four_bits : i32",
                  "%selected = arith.select %high, %high_word, %word : i32",
                  "%code = arith.andi %selected, %mask : i32",
                  "%book_index = arith.index_castui %code : i32 to index",
                  "%value = tensor.extract %book[%book_index] : tensor<16xf32>",
                  f"%row_base = arith.muli %row, {k} : index",
                  "%flat = arith.addi %row_base, %column : index",
                  "%group_index = arith.divui %flat, %group : index"]
        if nested:
            inner += ["%outer_index = arith.divui %group_index, %c256 : index",
                      "%scale_code = tensor.extract %scale_codes[%group_index] : tensor<?xi8>",
                      "%scale_signed_index = arith.index_cast %scale_code : i8 to index",
                      "%byte_mask = arith.constant 255 : index",
                      "%scale_index = arith.andi %scale_signed_index, %byte_mask : index",
                      "%scale_value = tensor.extract %scale_book[%scale_index] : tensor<256xf32>",
                      "%outer = tensor.extract %outer_scales[%outer_index] : tensor<?xf32>",
                      "%bias = tensor.extract %offset[%c0] : tensor<1xf32>",
                      "%scaled = arith.mulf %scale_value, %outer : f32",
                      "%scale = arith.addf %scaled, %bias : f32"]
        else:
            inner.append("%scale = tensor.extract %scales[%group_index] : tensor<?xf32>")
        inner.append("%decoded = arith.mulf %value, %scale : f32")
        rounded = "%decoded"
    else:
        inner += ["%group_index = arith.divui %column, %group : index"]
        if kind == "awq":
            inner += ["%packed_column = arith.divui %row, %c8 : index",
                      "%lane = arith.remui %row, %c8 : index",
                      "%even = arith.divui %lane, %c2 : index",
                      "%odd = arith.remui %lane, %c2 : index",
                      "%odd_shift = arith.muli %odd, %c4 : index",
                      "%physical_lane = arith.addi %even, %odd_shift : index",
                      f"%word = tensor.extract %packed[%column, %packed_column] : {packed_type}",
                      "%zero_word = tensor.extract %zeros[%group_index, %packed_column] : tensor<?x?xi32>"]
            indices = "%group_index, %row"
        else:
            inner += ["%packed_column = arith.divui %column, %c8 : index",
                      "%physical_lane = arith.remui %column, %c8 : index",
                      f"%word = tensor.extract %packed[%row, %packed_column] : {packed_type}"]
            indices = "%row, %group_index"
        inner += ["%shift_index = arith.muli %physical_lane, %c4 : index",
                  "%shift = arith.index_cast %shift_index : index to i32",
                  "%shifted = arith.shrui %word, %shift : i32",
                  "%code = arith.andi %shifted, %mask : i32"]
        if kind == "awq":
            inner += ["%zero_shifted = arith.shrui %zero_word, %shift : i32",
                      "%zero_code = arith.andi %zero_shifted, %mask : i32",
                      "%signed = arith.subi %code, %zero_code : i32"]
        else:
            inner.append("%signed = arith.subi %code, %eight : i32")
        inner += ["%value = arith.sitofp %signed : i32 to f32",
                  f"%scale_bits = tensor.extract %scales[{indices}] : {scale_type}"]
        casts, scale_value = convert("%scale_bits", scale, "f32", "scale")
        inner += casts + [f"%decoded = arith.mulf %value, {scale_value} : f32"]
        # Signed nibble differences are exactly representable in every source
        # dtype. Materialize the original source multiplication rounding.
        casts, source_product = convert("%decoded", "f32", scale, "source_product")
        inner += casts
        casts, rounded = convert(source_product, scale, "f32", "source_rounded")
        inner += casts
    casts, result = convert(rounded, "f32", output, "output")
    inner += casts + [f"linalg.yield {result} : {TYPES[output]}"]
    body += ["    " + line for line in inner]
    body += [f"}} -> {out_type}", f"return %result : {out_type}"]
    lines += ["    " + line for line in body] + ["  }"]
    return "\n".join(lines), name


def generate():
    entries, names = [], []
    for kind in ("nf4", "nf4_nested", "awq", "ct"):
        for scale in (("f32",) if kind.startswith("nf4") else TYPES):
            for output in TYPES:
                # Cross-dtype half sources use the route's already qualified
                # native Cast after the original typed decode result.
                if not kind.startswith("nf4") and scale != "f32" and output != scale:
                    continue
                text, name = entry(kind, scale, output)
                entries.append(text)
                names.append(name)
    recipe = Path(__file__).with_name("artifacts.mlir")
    recipe.write_text("// Original mature tensor sequences; see NOTICE.md.\nmodule {\n" +
                      "\n".join(entries) + "\n}\n", encoding="utf-8", newline="\n")
    return names


if __name__ == "__main__":
    generate()
