"""Prepare only the AVX2 leaf's BF16 multirow call into original tinyBLAS.

The owning CMake already authenticates and prepares the selected ggml source.
Baseline CPU and other ISA programs continue to use that original source.
"""
import argparse
from pathlib import Path
import tempfile


_INCLUDE = """#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif
"""
_LEAF_INCLUDE = """
// HYLE AVX2 leaf only: original BF16 multirow kernel, not the global backend.
#if defined(hyle_ggml_cpu_avx2_EXPORTS) && defined(__AVX2__) && !defined(GGML_USE_LLAMAFILE)
#include "llamafile/sgemm.h"
#endif
"""
_BARRIER = """    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);
"""
_MULTIROW = """
    // The ordinary BF16 dot path already prepares a BF16 activation row when
    // src1 is F32. Reuse that exact original scratch after its existing barrier;
    // never widen or copy the weight bank, and leave single-row/other types alone.
#if defined(hyle_ggml_cpu_avx2_EXPORTS) && defined(__AVX2__) && !defined(GGML_USE_LLAMAFILE)
    if (!params->use_ref && src0->type == GGML_TYPE_BF16 &&
        vec_dot_type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32 &&
        ne11 > 1 && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
        ggml_is_contiguous(src0) &&
        ((src1->type == GGML_TYPE_BF16 && ggml_is_contiguous(src1)) ||
         src1->type == GGML_TYPE_F32)) {
        const bool converted = src1->type != vec_dot_type;
        const void * b = converted ? params->wdata : src1->data;
        const int64_t ldb = converted ? ne10 : (int64_t)(nb11 / sizeof(ggml_bf16_t));
        if (llamafile_sgemm(params, ne01, ne11, ne00,
                           src0->data, nb01 / sizeof(ggml_bf16_t), b, ldb,
                           dst->data, nb1 / sizeof(float),
                           GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32)) {
            return;
        }
    }
#endif
"""


def prepare(source, output):
    if source.resolve() == output.resolve() or output.is_symlink():
        raise ValueError("AVX2 generated C must be separate from its original")
    data = source.read_bytes().replace(b"\r\n", b"\n")
    for before, after in ((_INCLUDE, _INCLUDE + _LEAF_INCLUDE),
                          (_BARRIER, _BARRIER + _MULTIROW)):
        if data.count(before.encode()) != 1:
            raise ValueError("locked ggml CPU multirow context changed")
        data = data.replace(before.encode(), after.encode(), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_bytes() != data:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
