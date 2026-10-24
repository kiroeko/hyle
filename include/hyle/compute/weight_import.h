#pragma once

#include <span>

#include "hyle/compute/tensor.h"

namespace hyle::compute {

// Cold host import of original E4M3FN [N,K] weight bytes and unsigned E8M0
// or plain F32 macroblock scales [ceil(N/128),K/128]. Sources may have valid
// nonnegative byte strides. K must contain complete128-value blocks. Output is
// the exact
// float8_weight_description(K,N,kFloat32Block128) carrier: unchanged row-major
// E4 bytes followed by native F32 scale storage. This never expands the weight
// matrix to floating point and performs no activation quantization.
//
// All views must expose stable canonical host addresses (HostAccessible and
// HostAddressIdentity). Device transfers and unknown shadow mappings are not
// implicit. Output must not overlap either source, including through another
// resource/handle. Geometry, capacity and aliases are checked before writing.
// The caller owns allocation, source readiness, synchronization and publication.
//
// U8 scale bytes0..254 widen exactly;0 is2^-127. Byte255 is stored as F32 raw
// bits0x7f800001, matching the fixed PyTorch2.8 E8M0 conversion. No floating
// operation quiets that NaN during import. F32 scale bits are copied unchanged,
// including signed zero and NaNs, without an intermediate lower-precision cast.
// Singleton axes in an external scale layout must be explicitly removed by
// the caller's metadata view; this import does not infer producer arithmetic.
// Flush/unmap failures are returned;
// a failure after writing does not guarantee rollback or grant ready content.
HYLE_COMPUTE_API Status import_float8_block128_weight(
    ConstTensorView weight, ConstTensorView scales, TensorView output) noexcept;

// Cold numeric conversion of positive finite plain F32 scales to dense U8 E8M0
// with the same nonempty shape. Source byte strides may be nonnegative. The
// exponent uses guard/round/sticky nearest-even, including the implied F32
// mantissa bit; this is not ceil(log2(x)) or a raw-byte reinterpretation.
// E8 code0 represents2^-127. Invalid source values and rounding to code255
// fail before any output write. No sign-discarding conversion is implicit.
//
// Host address, alias, caller synchronization and flush/unmap contracts are
// the same as import_float8_block128_weight above. Only the scale grid is
// converted; packed weights and activation storage are unaffected.
HYLE_COMPUTE_API Status encode_e8m0_scales(
    ConstTensorView scales, TensorView output) noexcept;

// Cold concatenation of low-first INT4 zero rows, without changing any nibble
// value. Each source is I32[ceil(rows_per_source/8),groups]; output is dense
// I32[ceil(source_count*rows_per_source/8),groups]. Word(row/8,group) stores
// logical row's nibble at4*(row%8). Sources may have nonnegative byte strides.
// Source padding nibbles are ignored and output padding is zero. In particular
// independent experts with rows%8!=0 are bit-packed across expert boundaries,
// never concatenated as independently padded byte grids.
//
// All sources and output need canonical host mappings, without overlap with
// output. Geometry/capacity/aliases are checked before writes; readiness and
// publication remain caller-owned. Flush/unmap failures and rollback semantics
// are the same as the cold import functions above.
HYLE_COMPUTE_API Status pack_int4_zero_rows(
    std::span<const ConstTensorView> sources, std::int64_t rows_per_source,
    TensorView output) noexcept;

} // namespace hyle::compute
