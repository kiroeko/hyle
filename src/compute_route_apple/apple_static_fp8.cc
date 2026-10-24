#include "apple_static_fp8.h"
#include "apple_checks.h"

#include "../compute_shader_shared/ct_packed.h"
#include "../compute_shader_shared/nvfp4_f32.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <mlx/fast.h>
#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
bool floating(DataType t) noexcept {
  return t == DataType::kFloat32 || t == DataType::kFloat16 || t == DataType::kBFloat16;
}
int scalar_tag(DataType t) noexcept {
  return t == DataType::kFloat16 ? 1 : t == DataType::kBFloat16 ? 2 : 0;
}
mx::Dtype raw_type(DataType t) noexcept {
  return t == DataType::kFloat8E4M3FN ? mx::uint8
       : t == DataType::kFloat32 ? mx::uint32 : mx::uint16;
}
mx::Dtype native_type(DataType t) noexcept {
  return t == DataType::kFloat8E4M3FN ? mx::uint8
       : t == DataType::kFloat32 ? mx::float32
       : t == DataType::kFloat16 ? mx::float16 : mx::bfloat16;
}
std::string kernel_header() {
  return std::string(R"METAL(
    inline uint hyle_mul_hi(uint a, uint b) { return uint((ulong(a) * ulong(b)) >> 32); }
    inline uint hyle_clz(uint a) { return clz(a); }
  )METAL") + shader::kNvfp4F32 + shader::kCtPacked + R"METAL(
    inline uint hyle_e4_to_f32(uint code) {
      const uint sign = (code & 128u) << 24u, magnitude = code & 127u;
      const uint exponent = magnitude >> 3u, mantissa = magnitude & 7u;
      if (magnitude == 127u) return sign | 0x7fc00000u;
      if (exponent != 0u)
        return sign | ((exponent + 120u) << 23u) | (mantissa << 20u);
      if (mantissa == 0u) return sign;
      const uint leading = 31u - hyle_clz(mantissa);
      return sign | ((leading + 118u) << 23u) |
          ((mantissa << (23u - leading)) & 0x7fffffu);
    }
    inline uint hyle_f32_to_e4(uint bits) {
      const uint sign = (bits >> 24u) & 128u, magnitude = bits & 0x7fffffffu;
      if (magnitude > 0x7f800000u) return sign | 127u;
      if (magnitude >= 0x43e00000u) return sign | 126u;
      const int exponent = int(magnitude >> 23u) - 120;
      uint high, remainder, midpoint;
      if (exponent >= 1) {
        high = (uint(exponent) << 3u) | ((magnitude >> 20u) & 7u);
        remainder = magnitude & 0xfffffu; midpoint = 0x80000u;
      } else {
        const uint shift = uint(1 - exponent);
        if (shift > 4u) return sign;
        const uint significand = (magnitude & 0x7fffffu) | 0x800000u;
        const uint drop = 20u + shift;
        high = significand >> drop;
        remainder = significand & ((1u << drop) - 1u);
        midpoint = 1u << (drop - 1u);
      }
      high += uint(remainder > midpoint || (remainder == midpoint && (high & 1u) != 0u));
      return sign | min(high, 126u);
    }
  )METAL";
}
Result<std::array<int, 2>> grid(std::uint64_t groups) noexcept {
  const auto x = (std::min)(groups, std::uint64_t{65535});
  const auto y = groups / x + (groups % x != 0);
  if (y > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
    return std::unexpected(Status::OutOfRange("Apple static FP8 grid exceeds native int axes"));
  return std::array{static_cast<int>(x), static_cast<int>(y)};
}
Status carriers(std::span<const TensorDesc> descriptions, std::span<const mx::array> inputs) {
  if (inputs.size() != descriptions.size())
    return Status::InvalidArgument("Apple static FP8 carrier count differs");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(descriptions[i]));
    if (inputs[i].dtype() != spec.dtype || inputs[i].shape() != spec.shape)
      return Status::InvalidArgument("Apple static FP8 carrier differs from logical storage");
  }
  return Status::Ok();
}
} // namespace

bool is_mlx_float8_cast(const Operation &operation, std::span<const TensorDesc> in) noexcept {
  const auto *cast = operation.get_if<Cast>();
  return cast && in.size() == 1 && in[0].is_plain() &&
      ((in[0].data_type() == DataType::kFloat8E4M3FN && floating(cast->output_type)) ||
       (floating(in[0].data_type()) && cast->output_type == DataType::kFloat8E4M3FN));
}

bool is_mlx_exact_binary(const Operation &operation, std::span<const TensorDesc> in) noexcept {
  const auto *binary = operation.get_if<Binary>();
  if (!binary || (binary->kind != BinaryKind::kMultiply && binary->kind != BinaryKind::kDivide) ||
      in.size() != 2) return false;
  const auto type = in[0].data_type();
  if (type != DataType::kFloat32 && type != DataType::kBFloat16) return false;
  bool scalar = false;
  for (const auto &d : in) {
    if (!d.is_plain() || !d.is_dense() || d.data_type() != type) return false;
    const auto count = d.element_count();
    if (!count || *count > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
      return false;
    scalar |= d.rank() == 0 || (d.rank() == 1 && d.dimensions()[0] == 1);
  }
  return type == DataType::kBFloat16 || scalar;
}

Result<std::vector<mx::array>> lower_exact_binary(const Operation &operation,
    std::span<const TensorDesc> descriptions, std::span<const mx::array> inputs, mx::Stream stream) {
  if (!is_mlx_exact_binary(operation, descriptions) || stream.device.type != mx::Device::gpu)
    return std::unexpected(Status::Unimplemented("Apple exact binary requires dense BF16 or scalar F32 operands"));
  HYLE_RETURN_IF_ERROR(carriers(descriptions, inputs));
  HYLE_ASSIGN_OR_RETURN(auto output, infer_single_output(operation,descriptions));
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(output));
  HYLE_ASSIGN_OR_RETURN(auto count, output.element_count());
  if (!count) return std::vector{mx::zeros(spec.shape,spec.dtype,stream)};
  if (count > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
    return std::unexpected(Status::OutOfRange("Apple exact binary exceeds the native flat axis"));
  HYLE_ASSIGN_OR_RETURN(auto shape,grid(count / 256 + (count % 256 != 0)));
  if (output.data_type() == DataType::kBFloat16) {
    // Broadcast only metadata. The kernel walks each actual native stride;
    // a group/token scale does not become an expanded floating allocation.
    const auto broadcast_shape = spec.shape.empty() ? mx::Shape{1} : spec.shape;
    auto left = mx::view(mx::broadcast_to(inputs[0],broadcast_shape,stream),mx::uint16,stream);
    auto right = mx::view(mx::broadcast_to(inputs[1],broadcast_shape,stream),mx::uint16,stream);
    static const auto header = kernel_header();
    static const auto kernel = mx::fast::metal_kernel("hyle_bfloat_binary",
        {"left","right"},{"out"},R"METAL(
      const ulong index = ulong(thread_position_in_grid.y) * GROUPS_X * 256 + thread_position_in_grid.x;
      if (index >= COUNT) return;
      ulong remaining = index, li = 0, ri = 0;
      for (int axis = RANK - 1; axis >= 0; --axis) {
        const ulong coordinate = remaining % ulong(left_shape[axis]);
        remaining /= ulong(left_shape[axis]);
        li += coordinate * left_strides[axis];
        ri += coordinate * right_strides[axis];
      }
      const uint a = uint(left[li]) << 16u, b = uint(right[ri]) << 16u;
      const uint aa = a & 0x7fffffffu, bb = b & 0x7fffffffu;
      uint value;
      if (aa >= 0x00800000u && aa < 0x7f800000u && bb >= 0x00800000u && bb < 0x7f800000u) {
        const float av = as_type<float>(a), bv = as_type<float>(b);
        value = as_type<uint>(DIVIDE ? av / bv : av * bv);
        const uint magnitude = value & 0x7fffffffu, tail = value & 0xffffu;
        // Safe Metal division can differ by a few F32 ulps. Away from a
        // BF16 midpoint those ulps cannot change the stored result. Resolve
        // the midpoint neighborhood with the integer IEEE implementation.
        const bool safe_round = !DIVIDE || tail < 0x7ff0u || tail > 0x8010u;
        if (magnitude >= 0x00800000u && magnitude < 0x7f800000u && safe_round) {
          out[index] = ushort(hyle_ct_narrow(value,2));
          return;
        }
      }
      // Integer carriers preserve subnormal inputs/results, signed zeros,
      // and IEEE exceptional values despite native floating FTZ behavior.
      value = DIVIDE ? hyle_f32_div(a,b) : hyle_f32_mul(a,b);
      out[index] = ushort(hyle_ct_narrow(value,2));
    )METAL",header,false,false,mx::CompileOptions{});
    auto values = kernel({left,right},{spec.shape},{mx::uint16},
        {shape[0]*256,shape[1],1},{256,1,1},
        {{"GROUPS_X",shape[0]},{"COUNT",static_cast<int>(count)},
         {"RANK",static_cast<int>(broadcast_shape.size())},
         {"DIVIDE",operation.get_if<Binary>()->kind == BinaryKind::kDivide}},
        std::nullopt,false,stream);
    return std::vector{mx::view(values[0],mx::bfloat16,stream)};
  }
  auto left = mx::reshape(mx::view(inputs[0],mx::uint32,stream),{-1},stream);
  auto right = mx::reshape(mx::view(inputs[1],mx::uint32,stream),{-1},stream);
  static const auto header = kernel_header();
  static const auto kernel = mx::fast::metal_kernel("hyle_f32_scalar_binary",
      {"left","right"},{"out"},R"METAL(
    const ulong index = ulong(thread_position_in_grid.y) * GROUPS_X * 256 + thread_position_in_grid.x;
    const ulong count = max(ulong(left_shape[0]),ulong(right_shape[0]));
    if (index >= count) return;
    const uint a = left[(left_shape[0] == 1 ? 0 : index) * left_strides[0]];
    const uint b = right[(right_shape[0] == 1 ? 0 : index) * right_strides[0]];
    if (DIVIDE) { out[index] = hyle_f32_div(a,b); return; }
    const uint aa = a & 0x7fffffffu, bb = b & 0x7fffffffu;
    if (aa >= 0x00800000u && aa < 0x7f800000u && bb >= 0x00800000u && bb < 0x7f800000u) {
      // Keep one native multiplication for the normal finite path. Any
      // subnormal, signed zero, overflow or special case uses the same IEEE
      // helper as other native numerical bridges, with no FTZ load/store.
      const uint bits = as_type<uint>(as_type<float>(a) * as_type<float>(b));
      const uint magnitude = bits & 0x7fffffffu;
      if (magnitude >= 0x00800000u && magnitude < 0x7f800000u) { out[index] = bits; return; }
    }
    out[index] = hyle_f32_mul(a,b);
  )METAL",header,false,false,mx::CompileOptions{});
  auto values=kernel({left,right},{spec.shape},{mx::uint32},{shape[0]*256,shape[1],1},{256,1,1},
      {{"GROUPS_X",shape[0]},{"DIVIDE",operation.get_if<Binary>()->kind == BinaryKind::kDivide}},
      std::nullopt,false,stream);
  return std::vector{mx::view(values[0],mx::float32,stream)};
}

Result<std::vector<mx::array>> lower_float8_cast(const Operation &operation,
    std::span<const TensorDesc> descriptions, std::span<const mx::array> inputs, mx::Stream stream) {
  if (!is_mlx_float8_cast(operation, descriptions) || stream.device.type != mx::Device::gpu)
    return std::unexpected(Status::Unimplemented("Apple E4M3 storage Cast requires a floating bridge"));
  HYLE_RETURN_IF_ERROR(carriers(descriptions, inputs));
  HYLE_ASSIGN_OR_RETURN(auto output, infer_single_output(operation, descriptions));
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(output));
  HYLE_ASSIGN_OR_RETURN(auto count, output.element_count());
  if (!count) return std::vector{mx::zeros(spec.shape, spec.dtype, stream)};
  if (count > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
    return std::unexpected(Status::OutOfRange("Apple E4M3 Cast exceeds the native flat axis"));
  HYLE_ASSIGN_OR_RETURN(auto shape, grid(count / 256 + (count % 256 != 0)));
  const auto src = descriptions[0].data_type(), dst = output.data_type();
  auto raw = mx::reshape(mx::view(inputs[0], raw_type(src), stream), {-1}, stream);
  static const auto header = kernel_header();
  static const auto kernel = mx::fast::metal_kernel("hyle_plain_e4m3_storage_cast",
      {"source"}, {"out"}, R"METAL(
    const ulong index = ulong(thread_position_in_grid.y) * GROUPS_X * 256 + thread_position_in_grid.x;
    if (index >= ulong(source_shape[0])) return;
    const uint value = uint(source[index * source_strides[0]]);
    const uint bits = ENCODE ? hyle_ct_widen(value, SOURCE_TYPE) : hyle_e4_to_f32(value);
    out[index] = OutType(ENCODE ? hyle_f32_to_e4(bits) : hyle_ct_narrow(bits, DEST_TYPE));
  )METAL", header, false, false, mx::CompileOptions{});
  auto values = kernel({raw}, {spec.shape}, {raw_type(dst)}, {shape[0]*256,shape[1],1},
      {256,1,1}, {{"GROUPS_X",shape[0]}, {"ENCODE",dst == DataType::kFloat8E4M3FN},
       {"SOURCE_TYPE",scalar_tag(src)}, {"DEST_TYPE",scalar_tag(dst)},
       {"OutType",raw_type(dst)}}, std::nullopt, false, stream);
  return std::vector{mx::view(values[0], native_type(dst), stream)};
}

Result<mx::array> lower_float8_cache_encode(const CacheStorage &storage,
    const mx::array &input, const TensorDesc &output, mx::Stream stream) {
  HYLE_RETURN_IF_ERROR(validate_cache_storage(storage));
  if (storage.data_type() != DataType::kFloat8E4M3FN ||
      stream.device.type != mx::Device::gpu ||
      (input.dtype() != mx::float32 && input.dtype() != mx::float16 && input.dtype() != mx::bfloat16))
    return std::unexpected(Status::Unimplemented("Apple E4 cache encoder requires a floating source on Metal"));
  HYLE_ASSIGN_OR_RETURN(auto spec,input_spec(output));
  HYLE_ASSIGN_OR_RETURN(auto count,output.element_count());
  if (spec.dtype != mx::uint8 || input.shape() != spec.shape ||
      count > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
    return std::unexpected(Status::OutOfRange("Apple E4 cache encoder shape exceeds its native flat axis"));
  if (!count) return mx::zeros(spec.shape,mx::uint8,stream);
  HYLE_ASSIGN_OR_RETURN(auto shape,grid(count / 256 + (count % 256 != 0)));
  const auto source = input.dtype() == mx::float32 ? DataType::kFloat32
      : input.dtype() == mx::float16 ? DataType::kFloat16 : DataType::kBFloat16;
  auto raw = mx::reshape(mx::view(input,raw_type(source),stream),{-1},stream);
  static const auto header = kernel_header();
  static const auto kernel = mx::fast::metal_kernel("hyle_checked_e4_cache_encode",
      {"source"},{"out","status"},R"METAL(
    const ulong index = ulong(thread_position_in_grid.y) * GROUPS_X * 256 + thread_position_in_grid.x;
    if (index >= ulong(source_shape[0])) return;
    const uint value = hyle_ct_widen(uint(source[index * source_strides[0]]),SOURCE_TYPE);
    const bool finite = (value & 0x7fffffffu) < 0x7f800000u;
    status[index] = finite ? 0 : INVALID_NUMERIC;
    out[index] = finite ? uchar(hyle_f32_to_e4(hyle_f32_div(value,uint(SCALE_BITS)))) : uchar(0);
  )METAL",header,false,false,mx::CompileOptions{});
  // Status storage is bounded by the supplied dirty rows, never resident KV.
  // Byte stores stay within the exact output count, including partial tails.
  auto values = kernel({raw},{spec.shape,{static_cast<int>(count)}},
      {mx::uint8,mx::int32},{shape[0]*256,shape[1],1},{256,1,1},
      {{"GROUPS_X",shape[0]},{"SOURCE_TYPE",scalar_tag(source)},
       {"SCALE_BITS",static_cast<int>(std::bit_cast<std::uint32_t>(storage.scale()))},
       {"INVALID_NUMERIC",kInvalidNumericInput}},std::nullopt,false,stream);
  return with_device_status(std::move(values[0]),mx::max(values[1],false,stream),stream);
}

Result<std::vector<mx::array>> lower_static_fp8(
    const packed::internal::StaticFloat8Linear &operation,
    std::span<const TensorDesc> descriptions, std::span<const mx::array> inputs, mx::Stream stream) {
  std::array<TensorDesc,1> output;
  HYLE_RETURN_IF_ERROR(operation.infer_output_descriptions(descriptions, output));
  if (stream.device.type != mx::Device::gpu)
    return std::unexpected(Status::InvalidArgument("Apple static FP8 linear requires the native GPU"));
  HYLE_RETURN_IF_ERROR(carriers(descriptions, inputs));
  const auto rows = static_cast<int>(descriptions[0].dimensions()[0]);
  const auto columns = static_cast<int>(output[0].dimensions()[1]);
  const auto items = std::uint64_t(rows) * columns;
  HYLE_ASSIGN_OR_RETURN(auto shape, grid(items / 4 + (items % 4 != 0)));
  auto a = mx::view(inputs[0],raw_type(descriptions[0].data_type()),stream);
  // Retain every scale element. Scalars still need a one-element array,
  // rather than a Metal scalar parameter; outer scale views flatten only
  // metadata and preserve their row/column selection below.
  auto sa = mx::reshape(mx::view(inputs[2],mx::uint32,stream),{-1},stream);
  auto sw = mx::reshape(mx::view(inputs[3],mx::uint32,stream),{-1},stream);
  static const auto header = kernel_header();
  static const auto kernel = mx::fast::metal_kernel("hyle_static_fp8_reference_linear",
      {"a","w","sa","sw"},{"out"}, R"METAL(
    const uint lane = thread_index_in_simdgroup, subgroup = simdgroup_index_in_threadgroup;
    const ulong group = ulong(threadgroup_position_in_grid.y) * GROUPS_X + threadgroup_position_in_grid.x;
    const ulong index = group * 4 + subgroup, columns = w_shape[0], depth = w_shape[1];
    // The four subgroups can cross a row boundary or contain an output tail.
    // Each valid output needs its own row scale; all subgroups reach the
    // original barrier before the tail returns.
    threadgroup uint reciprocal[4];
    if (lane == 0) reciprocal[subgroup] = index < ulong(a_shape[0]) * columns
        ? hyle_f32_div(0x3f800000u,
            sa[(SA_SCALAR ? 0 : index / columns) * sa_strides[0]]) : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (index >= ulong(a_shape[0]) * columns) return;
    const ulong row = index / columns, column = index % columns;
    float total = 0.0f;
    for (ulong k = lane; k < depth; k += 32) {
      const uint value = hyle_ct_widen(uint(a[row*a_strides[0]+k*a_strides[1]]),A_TYPE);
      const uint normalized = hyle_f32_mul(value,reciprocal[subgroup]);
      const uint code = hyle_f32_to_e4(normalized);
      const uint weight = hyle_e4_to_f32(uint(w[column*w_strides[0]+k*w_strides[1]]));
      // Finite E4 values are multiples of2^-9, so all finite products and
      // intermediate nonzero sums stay above the F32 subnormal range.
      const float product = as_type<float>(hyle_e4_to_f32(code)) * as_type<float>(weight);
      total = total + product;
    }
    for (ushort distance=16; distance>0; distance/=2)
      total = total + simd_shuffle_down(total,distance);
    if (lane == 0) {
      const uint activation_scale = sa[(SA_SCALAR ? 0 : row) * sa_strides[0]];
      const uint weight_scale = sw[(SW_SCALAR ? 0 : column) * sw_strides[0]];
      const uint first = WEIGHT_FIRST ? weight_scale : activation_scale;
      const uint second = WEIGHT_FIRST ? activation_scale : weight_scale;
      const uint scaled_first = hyle_f32_mul(as_type<uint>(total),first);
      const uint scaled_second = hyle_f32_mul(scaled_first,second);
      out[index] = OutType(hyle_ct_narrow(scaled_second,OUTPUT_TYPE));
    }
  )METAL", header, false, false, mx::CompileOptions{});
  auto values = kernel({a,inputs[1],sa,sw}, {{rows,columns}}, {raw_type(operation.output_type)},
      {shape[0]*128,shape[1],1}, {128,1,1}, {{"GROUPS_X",shape[0]},
       {"SA_SCALAR",sa.size() == 1},{"SW_SCALAR",sw.size() == 1},
       {"WEIGHT_FIRST",operation.scale_order ==
           packed::internal::StaticFloat8ScaleOrder::kWeightThenActivation},
       {"A_TYPE",scalar_tag(descriptions[0].data_type())},
       {"OUTPUT_TYPE",scalar_tag(operation.output_type)}, {"OutType",raw_type(operation.output_type)}},
      std::nullopt,false,stream);
  return std::vector{mx::view(values[0],native_type(operation.output_type),stream)};
}
} // namespace hyle::compute::apple::detail
