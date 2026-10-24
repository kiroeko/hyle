#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <mlx/ops.h>
#include <mlx/fast.h>

#include "apple_checks.h"
#include "apple_ct_packed.h"
#include "apple_blockwise_w4a8.h"
#include "apple_static_fp8.h"
#include "apple_internal.h"
#include "apple_nvfp4.h"
#include "apple_row_cache.h"
#include "apple_row_cache_indexer.h"
#include "../compute_packed_shared/artifact_packed_linear.h"
#include "../compute_shader_shared/nvfp4_f32.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
constexpr auto kMode = Float8ScaleMode::kFloat32Block128;

Result<mx::array> encode_cache(const model_ops::CacheEncode &encode,
                             const mx::array &input, const TensorDesc &output,
                             mx::Stream stream) {
  if (encode.storage.data_type() == DataType::kFloat8E4M3FN)
    return lower_float8_cache_encode(encode.storage,input,output,stream);
  if (encode.storage != CacheStorage::q8_0().value())
    return std::unexpected(Status::Unimplemented("Apple cache encoder admits Q8_0 storage"));
  // Work only on the supplied new rows. Packing uses the same block32
  // absmax/127, unrounded inverse and ties-away rules as ggml's reference.
  const auto rows = static_cast<int>(output.dimensions()[0]);
  const auto row_bytes = static_cast<int>(output.dimensions()[1]);
  auto values = mx::reshape(mx::astype(input, mx::float32, stream), {-1, 32}, stream);
  auto finite = mx::less_equal(mx::abs(values, stream),
      mx::array(std::numeric_limits<float>::max()), stream);
  auto safe = mx::where(finite, values, mx::array(0.F), stream);
  auto scale = mx::divide(mx::max(mx::abs(safe, stream), -1, true, stream),
      mx::array(127.F), stream);
  auto valid_scale = mx::less_equal(scale, mx::array(65504.F), stream);
  scale = mx::where(valid_scale, scale, mx::array(0.F), stream);
  auto stored_scale = mx::astype(scale, mx::float16, stream);
  // Use bits so a nonzero F16 subnormal is not compared under half FTZ rules.
  auto nonzero = mx::not_equal(mx::view(stored_scale, mx::uint16, stream),
      mx::array(std::uint16_t{0}), stream);
  // Select the operands before division/multiply, including eager graph paths.
  auto denominator = mx::where(nonzero, scale, mx::array(1.F), stream);
  auto inverse = mx::divide(mx::array(1.F), denominator, stream);
  auto normalized = mx::multiply(mx::where(nonzero, safe, mx::array(0.F), stream), inverse, stream);
  auto magnitude = mx::abs(normalized, stream);
  auto base = mx::floor(magnitude, stream);
  // Adding .5 first can round nextafter(.5,0) to one in F32. Preserve the
  // fractional comparison required by ggml roundf, including negative ties.
  auto increment = mx::astype(mx::greater_equal(mx::subtract(magnitude, base, stream),
      mx::array(.5F), stream), mx::float32, stream);
  auto rounded = mx::multiply(mx::sign(normalized, stream), mx::add(base, increment, stream), stream);
  auto codes = mx::astype(mx::clip(rounded, mx::array(-127.F), mx::array(127.F), stream),
      mx::int8, stream);
  auto scales = mx::view(stored_scale, mx::uint8, stream);
  auto packed = mx::reshape(mx::concatenate({scales, mx::view(codes, mx::uint8, stream)},
      -1, stream), {rows, row_bytes}, stream);
  auto valid = mx::logical_and(mx::all(finite, false, stream),
      mx::all(valid_scale, false, stream), stream);
  auto status = mx::where(valid, mx::array(0), mx::array(kInvalidNumericInput), stream);
  return with_device_status(std::move(packed), std::move(status), stream);
}

mx::array fp4_roundtrip(const mx::array &input, mx::Stream s) {
  auto shape = input.shape();
  shape.back() /= 32;
  shape.push_back(32);
  auto bits = mx::reshape(mx::astype(mx::view(input, mx::uint16, s), mx::uint32, s), shape, s);
  auto magnitude = mx::bitwise_and(bits, mx::array(std::uint32_t{0x7fff}), s);
  // BF16 nonnegative finite bit order is numerical order. The floor is the
  // original DS4 6*2^-126, exactly represented by BF16 0x01c0.
  auto maximum = mx::maximum(mx::max(magnitude, -1, true, s),
      mx::array(std::uint32_t{0x01c0}), s);
  auto amax = mx::view(mx::left_shift(maximum, mx::array(std::uint32_t{16}), s), mx::float32, s);
  auto scale_bits = mx::view(mx::divide(amax, mx::array(6.0F), s), mx::uint32, s);
  auto exponent = mx::add(mx::right_shift(scale_bits, mx::array(std::uint32_t{23}), s),
      mx::astype(mx::not_equal(mx::bitwise_and(scale_bits, mx::array(std::uint32_t{0x7fffff}), s),
          mx::array(std::uint32_t{0}), s), mx::uint32, s), s);
  auto scale = mx::view(mx::left_shift(exponent, mx::array(std::uint32_t{23}), s), mx::float32, s);
  auto x = mx::view(mx::left_shift(magnitude, mx::array(std::uint32_t{16}), s), mx::float32, s);
  auto normalized = mx::divide(x, scale, s);
  // Preserve only the low-scale domain where flushing BF16 subnormal input
  // changes E2M1 rounding. Integer significands remain small and exact.
  auto source_exp = mx::right_shift(magnitude, mx::array(std::uint32_t{7}), s);
  auto significand = mx::bitwise_or(mx::bitwise_and(magnitude, mx::array(std::uint32_t{127}), s),
      mx::where(mx::equal(source_exp, mx::array(std::uint32_t{0}), s),
          mx::array(std::uint32_t{0}), mx::array(std::uint32_t{128}), s), s);
  auto shift = mx::subtract(mx::minimum(mx::maximum(source_exp, mx::array(std::uint32_t{1}), s),
      mx::array(std::uint32_t{4}), s), mx::array(std::uint32_t{1}), s);
  auto exact_low = mx::multiply(mx::astype(mx::left_shift(significand, shift, s), mx::float32, s),
      mx::where(mx::equal(exponent, mx::array(std::uint32_t{1}), s),
          mx::array(1.0F / 128), mx::array(1.0F / 256), s), s);
  normalized = mx::minimum(mx::where(mx::less_equal(exponent, mx::array(std::uint32_t{2}), s),
      exact_low, normalized, s), mx::array(6.0F), s);
  // Original DS4 dsv4_e2m1fn_dequant: eight magnitudes, nearest distance,
  // even-code tie. This is its tensor translation, not a new quantizer.
  constexpr std::array<float, 8> values{0, .5F, 1, 1.5F, 2, 3, 4, 6};
  auto best = mx::zeros(shape, mx::uint32, s);
  auto distance = normalized;
  for (std::uint32_t code = 1; code < values.size(); ++code) {
    auto difference = mx::abs(mx::subtract(normalized, mx::array(values[code]), s), s);
    auto choose = mx::less(difference, distance, s);
    if ((code & 1) == 0)
      choose = mx::logical_or(choose, mx::logical_and(mx::equal(difference, distance, s),
          mx::not_equal(mx::bitwise_and(best, mx::array(std::uint32_t{1}), s),
              mx::array(std::uint32_t{0}), s), s), s);
    best = mx::where(choose, mx::array(code), best, s);
    distance = mx::where(choose, difference, distance, s);
  }
  const mx::array table(values.data(), {8}, mx::float32);
  auto dequant = mx::multiply(mx::take(table, best, 0, s), scale, s);
  auto output = mx::view(dequant, mx::uint32, s);
  auto half_minimum = mx::logical_and(mx::equal(exponent, mx::array(std::uint32_t{1}), s),
      mx::equal(best, mx::array(std::uint32_t{1}), s), s);
  output = mx::where(half_minimum, mx::array(std::uint32_t{0x00400000}), output, s);
  output = mx::where(mx::equal(best, mx::array(std::uint32_t{0}), s), mx::array(std::uint32_t{0}), output, s);
  auto sign = mx::left_shift(mx::bitwise_and(bits, mx::array(std::uint32_t{0x8000}), s),
      mx::array(std::uint32_t{16}), s);
  // E2M1 times a power2 is exactly BF16-representable (or infinity). Keep
  // the original stored bits, including subnormals and zero sign, on GPU.
  output = mx::astype(mx::right_shift(mx::bitwise_or(output, sign, s),
      mx::array(std::uint32_t{16}), s), mx::uint16, s);
  return mx::view(mx::reshape(output, input.shape(), s), mx::bfloat16, s);
}

// These are original MLX conversion/arithmetic nodes, traced once as part of
// its native region. No HYLE conversion kernel or hot graph interpreter.
mx::array decode(const mx::array &codes, mx::Stream s) {
  auto value = mx::from_fp8(codes, mx::float32, s);
  // MLX's finite-only conversion maps the reserved E4M3FN NaN code to 480.
  // Retain the storage format's NaN using native selection, not that value.
  auto nan = mx::equal(mx::bitwise_and(codes, mx::array(std::uint8_t{127}), s),
                       mx::array(std::uint8_t{127}), s);
  return mx::where(nan, mx::array(std::numeric_limits<float>::quiet_NaN()),
                   value, s);
}

struct Quantized {
  mx::array codes;
  mx::array scales;
};

Quantized quantize_exact(const mx::array &input, int group, float floor,
                         model_ops::Float8ScalePolicy policy, mx::Stream s) {
  auto shape = input.shape();
  shape.back() /= group;
  shape.push_back(group);
  auto raw = mx::reshape(mx::view(input,
      input.dtype() == mx::bfloat16 ? mx::uint16 : mx::uint32, s), shape, s);
  const auto groups = input.size() / static_cast<std::size_t>(group);
  if (groups == 0) {
    auto scale_shape = shape;
    scale_shape.back() = 1;
    return {mx::zeros(shape, mx::uint8, s), mx::zeros(scale_shape, mx::float32, s)};
  }
  if (groups > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    throw std::overflow_error("Apple reciprocal FP8 quantizer exceeds native template capacity");
  const auto groups_x = groups < 65535 ? static_cast<int>(groups) : 65535;
  const auto width = static_cast<std::size_t>(groups_x);
  const auto groups_y = groups / width + (groups % width != 0);
  if (groups_y > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    throw std::overflow_error("Apple reciprocal FP8 quantizer grid exceeds native int axes");
  static const auto header = std::string(R"METAL(
    inline uint hyle_mul_hi(uint a, uint b) { return uint((ulong(a) * ulong(b)) >> 32); }
    inline uint hyle_clz(uint a) { return clz(a); }
  )METAL") + shader::kNvfp4F32;
  static const auto kernel = mx::fast::metal_kernel("hyle_fp8_group_exact",
      {"source", "floor"}, {"normalized", "scale"}, R"METAL(
    const ulong group_index = ulong(threadgroup_position_in_grid.y) * GROUPS_X +
        threadgroup_position_in_grid.x;
    if (group_index >= GROUPS) return;
    const uint lane = thread_index_in_simdgroup;
    uint maximum = 0u;
    for (uint item = lane; item < GROUP; item += 32) {
      const ulong index = group_index * GROUP + item;
      ulong remaining = index, offset = 0;
      for (int axis = RANK - 1; axis >= 0; --axis) {
        const ulong coordinate = remaining % ulong(source_shape[axis]);
        remaining /= ulong(source_shape[axis]);
        offset += coordinate * source_strides[axis];
      }
      const uint value = uint(source[offset]) << (BFLOAT ? 16u : 0u);
      maximum = max(maximum, value & 0x7fffffffu);
    }
    maximum = max(simd_max(maximum), floor[0]);
    uint inverse = 0u;
    if (lane == 0) {
      uint scale_bits = DIVIDE ? hyle_f32_div(maximum, 0x43e00000u)
                               : hyle_f32_mul(maximum, 0x3b124925u);
      if (POWER) scale_bits = ((scale_bits >> 23u) +
          uint((scale_bits & 0x7fffffu) != 0u)) << 23u;
      scale[group_index] = scale_bits;
      inverse = RECIPROCAL ? hyle_f32_div(0x3f800000u, scale_bits) : scale_bits;
    }
    inverse = simd_broadcast_first(inverse);
    for (uint item = lane; item < GROUP; item += 32) {
      const ulong index = group_index * GROUP + item;
      ulong remaining = index, offset = 0;
      for (int axis = RANK - 1; axis >= 0; --axis) {
        const ulong coordinate = remaining % ulong(source_shape[axis]);
        remaining /= ulong(source_shape[axis]);
        offset += coordinate * source_strides[axis];
      }
      const uint value = uint(source[offset]) << (BFLOAT ? 16u : 0u);
      const uint product = RECIPROCAL ? hyle_f32_mul(value, inverse)
                                      : hyle_f32_div(value, inverse);
      // The original codec receives a saturated finite F32 value, with
      // signed zero retained. Every preceding boundary is strict IEEE RNE.
      normalized[index] = (product & 0x80000000u) |
          min(product & 0x7fffffffu, 0x43e00000u);
    }
  )METAL", header, false, false, mx::CompileOptions{});
  auto scale_shape = shape;
  scale_shape.back() = 1;
  auto floor_bits = mx::reshape(mx::view(mx::array(floor), mx::uint32, s), {1}, s);
  auto outputs = kernel({raw, floor_bits}, {shape, scale_shape}, {mx::uint32, mx::uint32},
      {groups_x * 32, static_cast<int>(groups_y), 1}, {32, 1, 1},
      {{"GROUPS_X", groups_x}, {"GROUPS", static_cast<int>(groups)},
       {"GROUP", group}, {"RANK", static_cast<int>(raw.ndim())},
       {"BFLOAT", input.dtype() == mx::bfloat16},
       {"DIVIDE", policy == model_ops::Float8ScalePolicy::kDivide},
       {"POWER", policy == model_ops::Float8ScalePolicy::kPowerOfTwo},
       {"RECIPROCAL", policy == model_ops::Float8ScalePolicy::kMultiplyReciprocal}},
      std::nullopt, false, s);
  return {mx::to_fp8(mx::view(outputs[0], mx::float32, s), s),
          mx::view(outputs[1], mx::float32, s)};
}

Quantized quantize(const mx::array &input, int group, float floor,
                   model_ops::Float8ScalePolicy policy,
                   mx::Stream s) {
  if (policy == model_ops::Float8ScalePolicy::kMultiplyReciprocal)
    return quantize_exact(input, group, floor, policy, s);
  const bool power = policy == model_ops::Float8ScalePolicy::kPowerOfTwo;
  auto shape = input.shape();
  shape.back() /= group;
  shape.push_back(group);
  auto x = mx::reshape(mx::astype(input, mx::float32, s), shape, s);
  auto maximum = mx::maximum(mx::max(mx::abs(x, s), -1, true, s), mx::array(floor), s);
  auto scale = power ? mx::multiply(maximum, mx::array(1.0F / 448.0F), s)
                     : mx::divide(maximum, mx::array(448.0F), s);
  if (power) {
    // DeepSeek fast_log2_ceil/fast_pow2: raise a normal positive F32 scale
    // to the next power of two, preserving an already exact power.
    auto bits = mx::view(scale, mx::uint32, s);
    auto exponent = mx::right_shift(bits, mx::array(std::uint32_t{23}), s);
    auto fraction = mx::not_equal(
        mx::bitwise_and(bits, mx::array(std::uint32_t{0x7fffff}), s),
        mx::array(std::uint32_t{0}), s);
    scale = mx::view(
        mx::left_shift(mx::add(exponent, mx::astype(fraction, mx::uint32, s), s),
                        mx::array(std::uint32_t{23}), s),
        mx::float32, s);
  }
  auto normalized = mx::clip(mx::divide(x, scale, s), mx::array(-448.0F),
                             mx::array(448.0F), s);
  return {mx::to_fp8(normalized, s), scale};
}

Quantized weight(const mx::array &carrier, int count, int n, int k,
                  mx::Stream s, bool compact_scales = false) {
  const int blocks = k / 128;
  const auto payload = std::int64_t{count} * n * k;
  const auto bytes = payload + std::int64_t{count} *
      ((std::int64_t{n} + 127) / 128) * blocks * 4;
  // Large grouped banks use four-byte rows. Explicit multidimensional views
  // avoid flatten/as_strided, whose public flatten dimension is native int.
  const bool wide = carrier.ndim() == 2;
  auto code_bytes = wide
      ? mx::slice(carrier, {0, 0}, {static_cast<int>(payload / 4), 4}, s)
      : mx::slice(carrier, {0}, {static_cast<int>(payload)}, s);
  auto codes = mx::reshape(code_bytes, {count, n, blocks, 128}, s);
  auto scale_bytes = wide
      ? mx::slice(carrier, {static_cast<int>(payload / 4), 0},
                  {static_cast<int>(bytes / 4), 4}, s)
      : mx::slice(carrier, {static_cast<int>(payload)},
                  {static_cast<int>(bytes)}, s);
  auto scales = mx::view(scale_bytes, mx::float32, s);
  scales = mx::reshape(scales, {count, (n + 127) / 128, blocks}, s);
  // Grouped kernels read the canonical bank with safe expert IDs. Keep its
  // scales compact instead of repeating or copying weights for every slot.
  if (compact_scales) return {codes, mx::expand_dims(scales, -1, s)};
  scales = mx::slice(mx::repeat(scales, 128, 1, s), {0, 0, 0},
                      {count, n, blocks}, s);
  return {codes, mx::expand_dims(scales, -1, s)};
}

Quantized weight4(const mx::array &carrier, int count, int n, int k,
                  mx::Stream s) {
  const auto payload = std::int64_t{count} * n * (k / 2);
  const auto bytes = payload + std::int64_t{count} * n * (k / 32) * 4;
  const bool wide = carrier.ndim() == 2;
  auto code_bytes = wide
      ? mx::slice(carrier, {0, 0}, {static_cast<int>(payload / 4), 4}, s)
      : mx::slice(carrier, {0}, {static_cast<int>(payload)}, s);
  auto codes = mx::reshape(code_bytes, {count, n, k / 32, 16}, s);
  auto scale_bytes = wide
      ? mx::slice(carrier, {static_cast<int>(payload / 4), 0},
                  {static_cast<int>(bytes / 4), 4}, s)
      : mx::slice(carrier, {static_cast<int>(payload)},
                  {static_cast<int>(bytes)}, s);
  auto scales = mx::view(scale_bytes, mx::float32, s);
  scales = mx::reshape(scales, {count, n, k / 32, 1}, s);
  return {codes, scales};
}

mx::array packed_fp8_dot(const Quantized &a, const Quantized &b, mx::Stream s,
                         const mx::array *expert_ids = nullptr,
                         bool four = false) {
  // Original MLX SIMDgroup F32 matmul and exact packed decode, with the
  // original activation-scale, weight-scale, ordered F32 block accumulation.
  // Keep the small tile; larger M uses original
  // MLX small-device NT64x32x32 / four-SIMDgroup reuse. Only a bounded tile
  // is staged, never float weights or the complete [Kblocks,M,N] dots.
  // MLX owns pipeline/stream/completion and cold template compilation.
  static const auto kernel = mx::fast::metal_kernel("hyle_fp8_block128_f32",
      {"qa", "qw", "qs", "ws", "ids"}, {"out"}, R"METAL(
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint batch = threadgroup_position_in_grid.z;
    const ulong weight_batch = INDEXED ? ulong(ids[batch]) : ulong(batch);
    const uint row0 = threadgroup_position_in_grid.y * BM;
    const uint column0 = threadgroup_position_in_grid.x * BN;
    const uint rows = qa_shape[1], columns = qw_shape[1];
    threadgroup float left[BM * 32];
    threadgroup float right[32 * BN];
    float accumulated[BM * BN / 128] = {0.f};
    for (uint block = 0; block < uint(qa_shape[2]); ++block) {
      simdgroup_float8x8 partial[BM / 16][BN / 16];
      #pragma unroll
      for (uint mi = 0; mi < BM / 16; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < BN / 16; ++ni)
          partial[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.f);
      }
      for (uint chunk = 0; chunk < (FOUR ? 32 : 128); chunk += 32) {
        for (uint index = tid; index < BM * 32; index += 128) {
          const uint row = row0 + index / 32;
          const uint k = chunk + index % 32;
          left[index] = row < rows ? hyle_fp8_value(qa[
              batch * qa_strides[0] + row * qa_strides[1] +
              block * qa_strides[2] + k * qa_strides[3]]) : 0.f;
        }
        for (uint index = tid; index < 32 * BN; index += 128) {
          const uint column = column0 + index % BN;
          const uint wk = chunk + index / BN;
          if (column < columns) {
            const uchar code = qw[weight_batch * qw_strides[0] +
                column * qw_strides[1] + block * qw_strides[2] +
                (FOUR ? wk / 2 : wk) * qw_strides[3]];
            right[index] = FOUR
                ? hyle_fp4_value((code >> ((wk & 1) * 4)) & 15)
                : hyle_fp8_value(code);
          } else {
            right[index] = 0.f;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
          simdgroup_float8x8 lhs[BM / 16], rhs[BN / 16];
          #pragma unroll
          for (uint mi = 0; mi < BM / 16; ++mi)
            simdgroup_load(lhs[mi], left +
                ((sg / 2) * (BM / 2) + mi * 8) * 32 + k, 32);
          #pragma unroll
          for (uint ni = 0; ni < BN / 16; ++ni)
            simdgroup_load(rhs[ni], right +
                k * BN + (sg % 2) * (BN / 2) + ni * 8, BN);
          #pragma unroll
          for (uint mi = 0; mi < BM / 16; ++mi) {
            #pragma unroll
            for (uint ni = 0; ni < BN / 16; ++ni)
              simdgroup_multiply_accumulate(
                  partial[mi][ni], lhs[mi], rhs[ni], partial[mi][ni]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      // Reuse left only after every SIMDgroup has finished reading it.
      // BM*BN fits BM*32 for both admitted tiles. Reuse the activation
      // staging only after every SIMDgroup has finished reading it.
      #pragma unroll
      for (uint mi = 0; mi < BM / 16; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < BN / 16; ++ni)
          simdgroup_store(partial[mi][ni], left +
              ((sg / 2) * (BM / 2) + mi * 8) * BN +
              (sg % 2) * (BN / 2) + ni * 8, BN);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint part = 0; part < BM * BN / 128; ++part) {
        const uint index = tid + part * 128;
        const uint row = row0 + index / BN;
        const uint column = column0 + index % BN;
        if (row < rows && column < columns) {
          float value = left[index] * qs[batch * qs_strides[0] +
              row * qs_strides[1] + block * qs_strides[2]];
          // FP8 weights retain one scale per128 columns for both dense and
          // indexed dots; FP4 keeps its original per-column group32 scales.
          value = value * ws[weight_batch * ws_strides[0] +
              (FOUR ? column : column / 128) * ws_strides[1] +
              block * ws_strides[2]];
          accumulated[part] = accumulated[part] + value;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint part = 0; part < BM * BN / 128; ++part) {
      const uint index = tid + part * 128;
      const uint row = row0 + index / BN;
      const uint column = column0 + index % BN;
      if (row < rows && column < columns)
        out[(ulong(batch) * rows + row) * columns + column] = accumulated[part];
    }
  )METAL", R"METAL(
    #include <metal_simdgroup_matrix>
    #pragma clang fp contract(off)
    inline float hyle_fp4_value(uchar code) {
      // Exact E2M1 values, low nibble first; unary minus retains negative zero.
      const float values[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
      const float value = values[code & 7];
      return (code & 8) ? -value : value;
    }
    inline float hyle_fp8_value(uchar code) {
      const uint exponent = (code >> 3) & 15;
      const uint mantissa = code & 7;
      if (exponent == 15 && mantissa == 7)
        return as_type<float>(uint(0x7fc00000) | ((uint(code) & 128) << 24));
      // Every finite E4M3 normal maps exactly to these F32 fields. Avoid a
      // general ldexp for each repeatedly loaded packed tile element; keep
      // the original exact subnormal and signed-zero equation separately.
      if (exponent == 0) {
        const float value = float(mantissa) * 0.001953125f;
        return (code & 128) ? -value : value;
      }
      return as_type<float>(((uint(code) & 128) << 24) |
          ((exponent + 120) << 23) | (mantissa << 20));
    }
  )METAL", false, false, mx::CompileOptions{});
  const int batch = a.codes.shape(0), rows = a.codes.shape(1), columns = b.codes.shape(1);
  if (rows == 0 || columns == 0)
    return mx::zeros({batch, rows, columns}, mx::float32, s);
  if (columns > (std::numeric_limits<int>::max)() / 128 * 16)
    throw std::overflow_error("Apple FP8 native grid exceeds int capacity");
  const int block_rows = rows <= 16 ? 16 : 64;
  const int block_columns = rows <= 16 ? 16 : 32;
  const auto ids = expert_ids ? *expert_ids
                             : mx::reshape(mx::array(std::int32_t{0}), {1}, s);
  return kernel({a.codes, b.codes, a.scales, b.scales, ids}, {{batch, rows, columns}},
      {mx::float32}, {(1 + (columns - 1) / block_columns) * 128,
                     1 + (rows - 1) / block_rows, batch},
      {128, 1, 1}, {{"BM", block_rows}, {"BN", block_columns},
                    {"INDEXED", expert_ids != nullptr}, {"FOUR", four}},
      std::nullopt, false, s)[0];
}

} // namespace

bool is_mlx_float_operation(const Operation &operation,
                            std::span<const TensorDesc> inputs) noexcept {
  if (const auto packed = operation.custom_as<packed::internal::ArtifactPackedLinear>();
      packed && packed->format == model_ops::WeightArtifactFormat::kNf4DoubleQuant)
    return false;
  return operation.custom_as<model_ops::CacheEncode>() ||
         is_mlx_float8_cast(operation, inputs) ||
         is_mlx_exact_binary(operation, inputs) ||
         operation.custom_as<packed::internal::StaticFloat8Linear>() ||
         operation.custom_as<packed::internal::ArtifactPackedLinear>() ||
         operation.custom_as<model_ops::Mxfp4Linear>() ||
         operation.custom_as<model_ops::GroupedMxfp4Linear>() ||
         operation.custom_as<model_ops::BlockwiseW4A8Linear>() ||
         operation.custom_as<model_ops::GroupedBlockwiseW4A8Linear>() ||
         operation.custom_as<model_ops::Nvfp4Linear>() ||
         operation.custom_as<model_ops::GroupedNvfp4Linear>() ||
         operation.custom_as<model_ops::LocalNvfp4Linear>() ||
         operation.custom_as<model_ops::GroupedLocalNvfp4Linear>() ||
         operation.custom_as<model_ops::RowCacheIndexerScores>() ||
         operation.custom_as<compute::detail::RowCachePagedIndexerScores>() ||
         operation.custom_as<compute::detail::RowCacheMap>() ||
         operation.custom_as<model_ops::RowCacheEncode>() ||
         operation.custom_as<model_ops::RowCacheGatherDecode>() ||
         operation.custom_as<model_ops::RowCacheGatherDecodePadded>() ||
         operation.custom_as<compute::detail::RowCachePaddedMap>() ||
         operation.custom_as<model_ops::Float8BlockQuantize>() ||
         operation.custom_as<model_ops::Float8BlockLinear>() ||
         operation.custom_as<model_ops::Float8WeightDecode>() ||
         operation.custom_as<model_ops::GroupedFloat8Linear>() ||
         operation.custom_as<model_ops::GroupedFloat4Linear>() ||
         operation.custom_as<model_ops::BlockQuantizationRoundtrip>() ||
         (operation.get_if<MatMul>() && inputs.size() == 2 &&
          inputs[1].storage_format().id() == kFloat8WeightFloat32Block128);
}

Result<InputSpec> mlx_float_input_spec(const TensorDesc &description) noexcept {
  try {
    const auto id = description.storage_format().id();
    Result<TensorDesc> expected = std::unexpected(
        Status::Unimplemented("Apple native FP8 storage format is not admitted"));
    const auto dims = description.dimensions();
    if (id == kFloat8ActivationFloat32Block128 && dims.size() == 2)
      expected = float8_activation_description(dims[0], dims[1], kMode);
    else if (id == kFloat8WeightFloat32Block128 && dims.size() == 2)
      expected = float8_weight_description(dims[0], dims[1], kMode);
    else if (id == kFloat8GroupedWeightFloat32Block128 && dims.size() == 3)
      expected = float8_grouped_weight_description(dims[0], dims[1], dims[2]);
    else if (id == kFloat4GroupedWeightFloat32Group32 && dims.size() == 3)
      expected = float4_grouped_weight_description(dims[0], dims[1], dims[2]);
    if (!expected)
      return std::unexpected(std::move(expected).error());
    if (description != *expected)
      return std::unexpected(Status::InvalidArgument("Apple FP8 requires canonical bytes and layout"));
    HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
    if (bytes > static_cast<std::size_t>((std::numeric_limits<int>::max)())) {
      // Keep each expert and every axis in the original native int domain;
      // only the total grouped bank may exceed one axis. Four-byte alignment
      // is part of this canonical F32 block-scale format.
      const auto limit = (std::numeric_limits<int>::max)();
      if ((id != kFloat8GroupedWeightFloat32Block128 &&
           id != kFloat4GroupedWeightFloat32Group32) || dims.size() != 3 ||
          dims[0] > limit || dims[1] > limit || dims[2] > limit ||
          dims[1] > limit / dims[2] || bytes % 4 != 0 ||
          bytes / 4 > static_cast<std::size_t>(limit))
        return std::unexpected(Status::OutOfRange("Apple FP8 carrier exceeds native int shape"));
      return InputSpec{{static_cast<int>(bytes / 4), 4}, mx::uint8, description};
    }
    return InputSpec{{static_cast<int>(bytes)}, mx::uint8, description};
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

std::optional<std::size_t> mlx_row_bank_input(const Operation &operation) noexcept {
  if (operation.custom_as<model_ops::RowCacheGatherDecode>() ||
      operation.custom_as<model_ops::RowCacheGatherDecodePadded>()) return 0;
  if (operation.custom_as<model_ops::RowCacheIndexerScores>() ||
      operation.custom_as<compute::detail::RowCachePagedIndexerScores>()) return 1;
  return std::nullopt;
}

Result<std::optional<RowCacheAccessFacts>> analyze_mlx_row_access(
    const Operation &operation, std::span<const TensorDesc> inputs) noexcept {
  const auto map = operation.custom_as<compute::detail::RowCacheMap>();
  const auto padded_map = operation.custom_as<compute::detail::RowCachePaddedMap>();
  const auto decoder = operation.custom_as<model_ops::RowCacheGatherDecode>();
  const auto padded_decoder = operation.custom_as<model_ops::RowCacheGatherDecodePadded>();
  const auto scorer = operation.custom_as<compute::detail::RowCachePagedIndexerScores>();
  if (!map && !padded_map && !decoder && !padded_decoder && !scorer) return std::nullopt;
  if (decoder && decoder->bounds != IndexBoundsPolicy::kChecked) return std::nullopt;
  std::array<TensorDesc, 2> outputs;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs,
      std::span(outputs).first(padded_map ? 2u : 1u)));
  if (scorer) {
    const auto &mapping = scorer->mapping;
    if (mapping.metadata_bytes > INT32_MAX || !mapping.max_refs())
      return std::unexpected(Status::Unimplemented("Apple paged indexer metadata exceeds native I32 carrier"));
    return RowCacheAccessFacts{.kind = RowCacheAccessKind::kPagedIndexer,
        .projected_input = 1,
        .query_rows = static_cast<std::uint32_t>(inputs[0].dimensions()[0]),
        .max_requests = mapping.max_requests(), .max_refs = mapping.max_refs(),
        .page_size = mapping.page_size};
  }
  if (map || padded_map) {
    const auto &mapping = map ? *map : padded_map->mapping;
    if (mapping.metadata_bytes > INT32_MAX || !mapping.max_refs())
      return std::unexpected(Status::Unimplemented("Apple row metadata exceeds native I32 carrier"));
    return RowCacheAccessFacts{
        .kind = padded_map ? RowCacheAccessKind::kPaddedMap : RowCacheAccessKind::kMap,
        .query_rows = static_cast<std::uint32_t>(inputs[0].dimensions()[0]),
        .max_requests = mapping.max_requests(), .max_refs = mapping.max_refs(),
        .page_size = mapping.page_size};
  }
  // Only the rank-two typed page-map chain requests this permission. Other
  // expert decoder shapes keep the original numerical implementation.
  if (inputs[1].rank() != 2) return std::nullopt;
  return RowCacheAccessFacts{
      .kind = padded_decoder ? RowCacheAccessKind::kPaddedCheckedDecode
                             : RowCacheAccessKind::kCheckedDecode,
      .projected_input = 0,
      .query_rows = static_cast<std::uint32_t>(inputs[1].dimensions()[0])};
}

Result<std::vector<mx::array>> lower_mlx_float(
    const Operation &operation, std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream s, RowCacheAccessFacts *access) noexcept {
  if (access) *access = {};
  try {
    if (s.device.type != mx::Device::gpu || inputs.size() != descriptions.size())
      return std::unexpected(Status::InvalidArgument("Apple FP8 lowering requires GPU and matching ports"));
    if (is_mlx_float8_cast(operation, descriptions))
      return lower_float8_cast(operation, descriptions, inputs, s);
    if (is_mlx_exact_binary(operation, descriptions))
      return lower_exact_binary(operation, descriptions, inputs, s);
    if (const auto fp8 = operation.custom_as<packed::internal::StaticFloat8Linear>())
      return lower_static_fp8(*fp8, descriptions, inputs, s);
    if (const auto mx = operation.custom_as<model_ops::Mxfp4Linear>())
      return lower_mxfp4(*mx, descriptions, inputs, s);
    if (const auto mx = operation.custom_as<model_ops::GroupedMxfp4Linear>())
      return lower_grouped_mxfp4(*mx, descriptions, inputs, s);
    if (operation.custom_as<model_ops::BlockwiseW4A8Linear>())
      return lower_blockwise_w4a8(descriptions, inputs, s, false);
    if (operation.custom_as<model_ops::GroupedBlockwiseW4A8Linear>())
      return lower_blockwise_w4a8(descriptions, inputs, s, true);
    if (operation.custom_as<model_ops::Nvfp4Linear>())
      return lower_nvfp4(descriptions, inputs, s);
    if (operation.custom_as<model_ops::GroupedNvfp4Linear>())
      return lower_grouped_nvfp4(descriptions, inputs, s);
    if (const auto local = operation.custom_as<model_ops::LocalNvfp4Linear>())
      return lower_local_nvfp4(*local, descriptions, inputs, s);
    if (const auto local = operation.custom_as<model_ops::GroupedLocalNvfp4Linear>())
      return lower_grouped_local_nvfp4(*local, descriptions, inputs, s);
    if (const auto ct = operation.custom_as<packed::internal::ArtifactPackedLinear>())
      return lower_ct_packed(*ct, descriptions, inputs, s);
    if (const auto quantizer = operation.custom_as<model_ops::Float8BlockQuantize>()) {
      std::array<TensorDesc, 2> output;
      HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, descriptions, output));
      HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(descriptions[0]));
      if (inputs[0].dtype() != spec.dtype || inputs[0].shape() != spec.shape)
        return std::unexpected(Status::InvalidArgument("Apple FP8 quantizer input carrier differs"));
      auto finite = mx::less_equal(mx::abs(mx::astype(inputs[0], mx::float32, s), s),
          mx::array(std::numeric_limits<float>::max()), s);
      auto safe = mx::where(finite, inputs[0], mx::array(0.F, inputs[0].dtype()), s);
      // The same strict integer IEEE helpers used by reciprocal quantization
      // retain subnormals and all three explicit F32 rounding boundaries.
      auto q = quantize_exact(safe, 128, quantizer->absmax_floor, quantizer->scale_policy, s);
      auto status = mx::where(mx::all(finite, false, s), mx::array(0),
          mx::array(kInvalidNumericInput), s);
      auto codes = mx::reshape(q.codes, spec.shape, s);
      auto scales = mx::reshape(q.scales, {spec.shape[0], spec.shape[1] / 128}, s);
      // Either published output must retain the check, including a graph that
      // uses only the scale grid. Completion owns the check and both arrays.
      HYLE_ASSIGN_OR_RETURN(codes, with_device_status(std::move(codes), status, s));
      HYLE_ASSIGN_OR_RETURN(scales, with_device_status(std::move(scales), std::move(status), s));
      return std::vector{std::move(codes), std::move(scales)};
    }
    TensorDesc output;
    if(operation.custom_as<compute::detail::RowCachePaddedMap>()){
      std::array<TensorDesc,2> inferred;HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation,descriptions,inferred));output=inferred[0];
    }else{HYLE_ASSIGN_OR_RETURN(output,infer_single_output(operation,descriptions));}
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(descriptions[i]));
      if (inputs[i].dtype() != spec.dtype || inputs[i].shape() != spec.shape)
        return std::unexpected(Status::InvalidArgument("Apple FP8 carrier differs from logical input"));
      HYLE_ASSIGN_OR_RETURN(auto bytes, descriptions[i].required_bytes());
      bool checked_row_bank = false;
      if (bytes > static_cast<std::size_t>((std::numeric_limits<int>::max)()) &&
          mlx_row_bank_input(operation) == i) {
        HYLE_ASSIGN_OR_RETURN(auto fact, analyze_mlx_row_access(operation, descriptions));
        checked_row_bank = fact && fact->projected_input == i;
      }
      if (bytes > static_cast<std::size_t>((std::numeric_limits<int>::max)()) &&
          !checked_row_bank && !operation.custom_as<model_ops::RowCacheIndexerScores>() &&
          !(i == 0 && (operation.custom_as<model_ops::GroupedFloat8Linear>() ||
                      operation.custom_as<model_ops::GroupedFloat4Linear>()) &&
            spec.shape.size() == 2))
        return std::unexpected(Status::OutOfRange("Apple FP8 input exceeds native index range"));
    }
    if (operation.custom_as<model_ops::RowCacheIndexerScores>()) {
      HYLE_ASSIGN_OR_RETURN(auto result,row_cache_indexer_scores(inputs,s));
      return std::vector{std::move(result)};
    }
    const auto row_result = [&](std::vector<mx::array> result) -> Result<std::vector<mx::array>> {
      if (access) {
        HYLE_ASSIGN_OR_RETURN(auto fact, analyze_mlx_row_access(operation, descriptions));
        if (fact) *access = *fact;
      }
      return result;
    };
    if (const auto scorer = operation.custom_as<compute::detail::RowCachePagedIndexerScores>()) {
      HYLE_ASSIGN_OR_RETURN(auto result, row_cache_paged_indexer_scores(*scorer, inputs, s));
      return row_result(std::vector{std::move(result)});
    }
    if(const auto map=operation.custom_as<compute::detail::RowCachePaddedMap>()){
      HYLE_ASSIGN_OR_RETURN(auto result,map_row_cache_pages_padded(*map,inputs[0],inputs[1],inputs[2],s));
      return row_result(std::move(result));
    }
    if(const auto gather=operation.custom_as<model_ops::RowCacheGatherDecodePadded>()){
      HYLE_ASSIGN_OR_RETURN(auto result,gather_decode_row_cache_padded(gather->encoding,inputs[0],inputs[1],s));return row_result(std::vector{std::move(result)});
    }
    if (const auto map = operation.custom_as<compute::detail::RowCacheMap>()) {
      HYLE_ASSIGN_OR_RETURN(auto result, map_row_cache_pages(*map,inputs[0],inputs[1],s));return row_result(std::vector{std::move(result)});
    }
    if (const auto encode = operation.custom_as<model_ops::RowCacheEncode>()) {
      HYLE_ASSIGN_OR_RETURN(auto result, encode_row_cache(encode->encoding, inputs[0], s));
      return std::vector{std::move(result)};
    }
    if (const auto gather = operation.custom_as<model_ops::RowCacheGatherDecode>()) {
      HYLE_ASSIGN_OR_RETURN(auto result, gather_decode_row_cache(*gather, inputs[0], inputs[1], s));
      return row_result(std::vector{std::move(result)});
    }
    if (const auto encode = operation.custom_as<model_ops::CacheEncode>()) {
      HYLE_ASSIGN_OR_RETURN(auto result, encode_cache(*encode, inputs[0], output, s));
      return std::vector{std::move(result)};
    }
    const auto type = output.data_type() == DataType::kBFloat16 ? mx::bfloat16 : mx::float32;
    if (const auto roundtrip = operation.custom_as<model_ops::BlockQuantizationRoundtrip>()) {
      if (roundtrip->format == model_ops::BlockQuantizationFormat::kFloat4E2M1Block32)
        return std::vector{fp4_roundtrip(inputs[0], s)};
      const auto group = roundtrip->format == model_ops::BlockQuantizationFormat::kFloat8E4M3Block64 ? 64 : 128;
      auto q = quantize(inputs[0], group, 1.0e-4F,
                        model_ops::Float8ScalePolicy::kPowerOfTwo, s);
      return std::vector{mx::astype(mx::reshape(
          mx::multiply(decode(q.codes, s), q.scales, s), inputs[0].shape(), s), type, s)};
    }
    if (operation.custom_as<model_ops::Float8WeightDecode>()) {
      const auto dims = descriptions[0].dimensions();
      const int k = static_cast<int>(dims[0]), n = static_cast<int>(dims[1]);
      auto w = weight(inputs[0], 1, n, k, s);
      return std::vector{mx::astype(mx::reshape(
          mx::multiply(decode(w.codes, s), w.scales, s), {n, k}, s), type, s)};
    }
    const auto grouped = operation.custom_as<model_ops::GroupedFloat8Linear>();
    const bool four = operation.custom_as<model_ops::GroupedFloat4Linear>().has_value();
    if (grouped || four) {
      const auto dims = descriptions[0].dimensions();
      const int e = static_cast<int>(dims[0]), n = static_cast<int>(dims[1]), k = static_cast<int>(dims[2]);
      const int t = inputs[1].shape(0), a = inputs[1].shape(1), slots = inputs[2].shape(1);
      if (t > (std::numeric_limits<int>::max)() / slots)
        return std::unexpected(Status::OutOfRange("Apple grouped FP8 slot count exceeds native int"));
      const int rows = t * slots;
      auto invalid = mx::logical_or(mx::less(inputs[2], mx::array(0), s),
                                    mx::greater_equal(inputs[2], mx::array(e), s), s);
      auto ids = mx::reshape(mx::where(invalid, mx::array(0), inputs[2], s), {rows}, s);
      auto w = four ? weight4(inputs[0], e, n, k, s)
                    : weight(inputs[0], e, n, k, s, true);
      const auto policy = four ? model_ops::Float8ScalePolicy::kPowerOfTwo
                               : grouped->scale_policy;
      auto q = quantize(inputs[1], 128, four ? 1.0e-4F : grouped->absmax_floor, policy, s);
      const int block = four ? 32 : 128;
      if (four) {
        q.codes = mx::reshape(q.codes, {t, a, k / 32, 32}, s);
        q.scales = mx::repeat(q.scales, 4, 2, s);
      }
      if (a == 1 && slots != 1) {
        q.codes = mx::broadcast_to(q.codes, {t, slots, k / block, block}, s);
        q.scales = mx::broadcast_to(q.scales, {t, slots, k / block, 1}, s);
      }
      q.codes = mx::reshape(q.codes, {rows, 1, k / block, block}, s);
      q.scales = mx::reshape(q.scales, {rows, 1, k / block, 1}, s);
      auto product = packed_fp8_dot(q, w, s, &ids, four);
      auto value = mx::astype(mx::reshape(product, {t, slots, n}, s), type, s);
      auto status = mx::where(mx::any(invalid, s), mx::array(kIndexOutOfRange), mx::array(0), s);
      HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(std::move(value), std::move(status), s));
      return std::vector{std::move(checked)};
    }
    const int m = static_cast<int>(descriptions[0].dimensions()[0]);
    const int k = static_cast<int>(descriptions[0].dimensions()[1]);
    const int n = static_cast<int>(descriptions[1].dimensions()[1]);
    auto w = weight(inputs[1], 1, n, k, s, true);
    Quantized q{inputs[0], inputs[0]};
    if (const auto linear = operation.custom_as<model_ops::Float8BlockLinear>()) {
      q = quantize(inputs[0], 128, linear->absmax_floor, linear->scale_policy, s);
      q.codes = mx::expand_dims(q.codes, 0, s);
      q.scales = mx::expand_dims(q.scales, 0, s);
    } else if (operation.get_if<MatMul>()) {
      HYLE_ASSIGN_OR_RETURN(auto expected, float8_activation_description(m, k, kMode));
      if (descriptions[0] != expected)
        return std::unexpected(Status::Unimplemented("Apple encoded FP8 MatMul requires block128 activation"));
      q.codes = mx::reshape(mx::slice(inputs[0], {0}, {m * k}, s), {1, m, k / 128, 128}, s);
      q.scales = mx::reshape(mx::view(mx::slice(inputs[0], {m * k},
          {static_cast<int>(inputs[0].size())}, s), mx::float32, s), {1, m, k / 128, 1}, s);
    } else {
      return std::unexpected(Status::Unimplemented("Apple FP8 operation is not bound"));
    }
    return std::vector{mx::astype(mx::reshape(packed_fp8_dot(q, w, s), {m, n}, s), type, s)};
  } catch (...) {
    return std::unexpected(native_exception());
  }
}
} // namespace hyle::compute::apple::detail
