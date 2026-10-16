#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <mlx/ops.h>
#include <mlx/fast.h>

#include "apple_checks.h"
#include "apple_internal.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
constexpr auto kMode = Float8ScaleMode::kFloat32Block128;

Result<mx::array> encode_cache(const model_ops::CacheEncode &encode,
                             const mx::array &input, const TensorDesc &output,
                             mx::Stream stream) {
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

Quantized quantize(const mx::array &input, int group, float floor, bool power,
                   mx::Stream s) {
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
                  mx::Stream s) {
  const int payload = n * k, blocks = k / 128;
  const int bytes = count * (payload + ((n + 127) / 128) * blocks * 4);
  auto codes = mx::reshape(mx::slice(carrier, {0}, {count * payload}, s),
                            {count, n, blocks, 128}, s);
  auto scales = mx::view(
      mx::contiguous(mx::slice(carrier, {count * payload}, {bytes}, s), false, s),
      mx::float32, s);
  scales = mx::reshape(scales, {count, (n + 127) / 128, blocks}, s);
  scales = mx::slice(mx::repeat(scales, 128, 1, s), {0, 0, 0},
                      {count, n, blocks}, s);
  return {codes, mx::expand_dims(scales, -1, s)};
}

Quantized weight4(const mx::array &carrier, int count, int n, int k,
                 const mx::array &ids, mx::Stream s) {
  const int payload = count * n * (k / 2);
  auto bytes = mx::reshape(mx::slice(carrier, {0}, {payload}, s), {count, n, k / 2}, s);
  bytes = mx::take(bytes, ids, 0, s);
  auto low = mx::bitwise_and(bytes, mx::array(std::uint8_t{15}), s);
  auto high = mx::right_shift(bytes, mx::array(std::uint8_t{4}), s);
  auto codes = mx::reshape(mx::concatenate({mx::expand_dims(low, -1, s),
      mx::expand_dims(high, -1, s)}, -1, s), {int(ids.size()), n, k / 32, 32}, s);
  // Original fp4_gemm_kernel casts E2M1 through F32 to E4M3FN. Every FP4
  // code is exactly representable; preserve low-first packing and zero sign.
  const mx::array e4m3({std::uint8_t{0}, std::uint8_t{0x30}, std::uint8_t{0x38},
      std::uint8_t{0x3c}, std::uint8_t{0x40}, std::uint8_t{0x44}, std::uint8_t{0x48},
      std::uint8_t{0x4c}, std::uint8_t{0x80}, std::uint8_t{0xb0}, std::uint8_t{0xb8},
      std::uint8_t{0xbc}, std::uint8_t{0xc0}, std::uint8_t{0xc4}, std::uint8_t{0xc8}, std::uint8_t{0xcc}});
  codes = mx::take(e4m3, codes, 0, s);
  auto scales = mx::view(mx::contiguous(mx::slice(carrier, {payload},
      {static_cast<int>(carrier.size())}, s), false, s), mx::float32, s);
  scales = mx::reshape(scales, {count, n, k / 32}, s);
  scales = mx::take(scales, ids, 0, s);
  return {codes, mx::expand_dims(scales, -1, s)};
}

mx::array packed_fp8_dot(const Quantized &a, const Quantized &b, mx::Stream s) {
  // Original MLX SIMDgroup F32 matmul/FP8 decode, with the same per-block
  // scale order as block_dot. Stage only a16x32/b32x16 tile, not float weights
  // or the complete [Kblocks,M,N] dots. MLX owns pipeline/stream/completion.
  static const auto kernel = mx::fast::metal_kernel("hyle_fp8_block128_f32",
      {"qa", "qw", "qs", "ws"}, {"out"}, R"METAL(
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint batch = threadgroup_position_in_grid.z;
    const uint row0 = threadgroup_position_in_grid.y * 16;
    const uint column0 = threadgroup_position_in_grid.x * 16;
    const uint rows = qa_shape[1], columns = qw_shape[1];
    threadgroup float left[512];
    threadgroup float right[512];
    float accumulated[2] = {0.f, 0.f};
    for (uint block = 0; block < uint(qa_shape[2]); ++block) {
      auto partial = make_filled_simdgroup_matrix<float, 8>(0.f);
      for (uint chunk = 0; chunk < 128; chunk += 32) {
        for (uint index = tid; index < 512; index += 128) {
          const uint row = row0 + index / 32;
          const uint k = chunk + index % 32;
          left[index] = row < rows ? hyle_fp8_value(qa[
              batch * qa_strides[0] + row * qa_strides[1] +
              block * qa_strides[2] + k * qa_strides[3]]) : 0.f;
          const uint column = column0 + index % 16;
          const uint wk = chunk + index / 16;
          right[index] = column < columns ? hyle_fp8_value(qw[
              batch * qw_strides[0] + column * qw_strides[1] +
              block * qw_strides[2] + wk * qw_strides[3]]) : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
          simdgroup_float8x8 lhs, rhs;
          simdgroup_load(lhs, left + (sg / 2) * 8 * 32 + k, 32);
          simdgroup_load(rhs, right + k * 16 + (sg % 2) * 8, 16);
          simdgroup_multiply_accumulate(partial, lhs, rhs, partial);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      // Reuse left only after every SIMDgroup has finished reading it.
      simdgroup_store(partial, left + (sg / 2) * 8 * 16 + (sg % 2) * 8, 16);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint part = 0; part < 2; ++part) {
        const uint index = tid + part * 128;
        const uint row = row0 + index / 16;
        const uint column = column0 + index % 16;
        if (row < rows && column < columns) {
          float value = left[index] * qs[batch * qs_strides[0] +
              row * qs_strides[1] + block * qs_strides[2]];
          value = value * ws[batch * ws_strides[0] + column * ws_strides[1] +
              block * ws_strides[2]];
          accumulated[part] = accumulated[part] + value;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint part = 0; part < 2; ++part) {
      const uint index = tid + part * 128;
      const uint row = row0 + index / 16;
      const uint column = column0 + index % 16;
      if (row < rows && column < columns)
        out[(ulong(batch) * rows + row) * columns + column] = accumulated[part];
    }
  )METAL", R"METAL(
    #include <metal_simdgroup_matrix>
    #pragma clang fp contract(off)
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
  return kernel({a.codes, b.codes, a.scales, b.scales}, {{batch, rows, columns}},
      {mx::float32}, {((columns + 15) / 16) * 128, (rows + 15) / 16, batch},
      {128, 1, 1}, {}, std::nullopt, false, s)[0];
}

mx::array block_dot(const Quantized &a, const Quantized &b, mx::Stream s,
                    bool packed_fp8 = true) {
  if (packed_fp8) return packed_fp8_dot(a, b, s);
  // [batch,M,Kblocks,128] x [batch,N,Kblocks,128]. Apply scales
  // after each original F32 dot, before reducing Kblocks.
  auto lhs = mx::transpose(decode(a.codes, s), {0, 2, 1, 3}, s);
  auto rhs = mx::transpose(decode(b.codes, s), {0, 2, 3, 1}, s);
  auto dots = mx::matmul(lhs, rhs, s);
  auto scaled = mx::multiply(dots, mx::transpose(a.scales, {0, 2, 1, 3}, s), s);
  auto weights = mx::transpose(b.scales, {0, 2, 3, 1}, s);
  const int batch = dots.shape(0), rows = dots.shape(2), columns = dots.shape(3);
  auto accumulated = mx::zeros({batch, 1, rows, columns}, mx::float32, s);
  // SGLang applies the activation scale before the weight scale and the
  // ordered F32 accumulation. Combining scales or a tree sum can cross a
  // BF16 midpoint that a later dynamic quantizer makes observable. This loop
  // builds native MLX nodes only at prepare/compile time; MLX owns fusion.
  for (int block = 0; block < dots.shape(1); ++block) {
    auto value = mx::slice(scaled, {0, block, 0, 0},
                            {batch, block + 1, rows, columns}, s);
    auto scale = mx::slice(weights, {0, block, 0, 0},
                            {batch, block + 1, 1, columns}, s);
    accumulated = mx::add(accumulated, mx::multiply(value, scale, s), s);
  }
  return mx::reshape(accumulated, {batch, rows, columns}, s);
}
} // namespace

bool is_mlx_float_operation(const Operation &operation,
                            std::span<const TensorDesc> inputs) noexcept {
  return operation.custom_as<model_ops::CacheEncode>() ||
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
    if (bytes > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
      return std::unexpected(Status::OutOfRange("Apple FP8 carrier exceeds native int shape"));
    return InputSpec{{static_cast<int>(bytes)}, mx::uint8, description};
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

Result<std::vector<mx::array>> lower_mlx_float(
    const Operation &operation, std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream s) noexcept {
  try {
    if (s.device.type != mx::Device::gpu || inputs.size() != descriptions.size())
      return std::unexpected(Status::InvalidArgument("Apple FP8 lowering requires GPU and matching ports"));
    HYLE_ASSIGN_OR_RETURN(auto output, infer_single_output(operation, descriptions));
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(descriptions[i]));
      if (inputs[i].dtype() != spec.dtype || inputs[i].shape() != spec.shape)
        return std::unexpected(Status::InvalidArgument("Apple FP8 carrier differs from logical input"));
      HYLE_ASSIGN_OR_RETURN(auto bytes, descriptions[i].required_bytes());
      if (bytes > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        return std::unexpected(Status::OutOfRange("Apple FP8 input exceeds native index range"));
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
      auto q = quantize(inputs[0], group, 1.0e-4F, true, s);
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
      auto w = four ? weight4(inputs[0], e, n, k, ids, s) : weight(inputs[0], e, n, k, s);
      if (!four) {
        w.codes = mx::take(w.codes, ids, 0, s);
        w.scales = mx::take(w.scales, ids, 0, s);
      }
      auto q = quantize(inputs[1], 128, four ? 1.0e-4F : grouped->absmax_floor, four, s);
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
      auto value = mx::astype(mx::reshape(block_dot(q, w, s, !four), {t, slots, n}, s), type, s);
      auto status = mx::where(mx::any(invalid, s), mx::array(kIndexOutOfRange), mx::array(0), s);
      HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(std::move(value), std::move(status), s));
      return std::vector{std::move(checked)};
    }
    const int m = static_cast<int>(descriptions[0].dimensions()[0]);
    const int k = static_cast<int>(descriptions[0].dimensions()[1]);
    const int n = static_cast<int>(descriptions[1].dimensions()[1]);
    auto w = weight(inputs[1], 1, n, k, s);
    Quantized q{inputs[0], inputs[0]};
    if (const auto linear = operation.custom_as<model_ops::Float8BlockLinear>()) {
      q = quantize(inputs[0], 128, linear->absmax_floor,
                   linear->scale_rounding == model_ops::Float8ScaleRounding::kPowerOfTwo, s);
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
    return std::vector{mx::astype(mx::reshape(block_dot(q, w, s), {m, n}, s), type, s)};
  } catch (...) {
    return std::unexpected(native_exception());
  }
}
} // namespace hyle::compute::apple::detail
