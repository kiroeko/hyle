#include "apple_nvfp4.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "apple_checks.h"
#include "hyle/compute/model_ops.h"
#include "../compute_shader_shared/nvfp4_f32.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
// All floating boundaries use IEEE bits. Metal's native F32 denormal handling
// must not silently change either BF16 operand or the reference accumulation.
std::string kernel_header() {
  return std::string(R"METAL(
    inline uint hyle_mul_hi(uint a, uint b) {
      return uint((ulong(a) * ulong(b)) >> 32);
    }
    inline uint hyle_clz(uint a) { return clz(a); }
  )METAL") + shader::kNvfp4F32 + R"METAL(
    inline uint hyle_nv_bfloat(uint bits) {
      if ((bits & 0x7fffffffu) > 0x7f800000u)
        return (bits >> 16) | 0x40u;
      return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
    }
    inline bool hyle_nv_positive(uint bits) {
      return bits != 0u && bits < 0x7f800000u;
    }
    inline uint hyle_nv_e4m3(uint code) {
      const uint sign = (code & 128u) << 24;
      const uint exponent = (code >> 3) & 15u, fraction = code & 7u;
      if (exponent == 15u && fraction == 7u) return sign | 0x7fc00000u;
      if (exponent != 0u) return sign | ((exponent + 120u) << 23) | (fraction << 20);
      if (fraction == 0u) return sign;
      const uint top = 31u - clz(fraction);
      return sign | ((top + 118u) << 23) | ((fraction << (23u-top)) & 0x7fffffu);
    }
    inline uint hyle_nv_scale_code(uint bits) {
      // Positive finite F32 -> saturated E4M3, nearest-even, including the
      // subnormal code range. No floating converter or intermediate FTZ.
      if (bits >= 0x43e00000u) return 126u;
      const uint exponent = bits >> 23;
      if (exponent >= 121u) {
        const uint rounded = bits + 0x7ffffu + ((bits >> 20) & 1u);
        return (rounded >> 20) - 960u;
      }
      if (exponent < 117u) return 0u;
      const uint significand = (bits & 0x7fffffu) | 0x800000u;
      const uint shift = 141u-exponent;
      const uint high = significand >> shift;
      const uint low = significand & ((1u << shift)-1u), halfway = 1u << (shift-1u);
      return high + uint(low > halfway || (low == halfway && (high & 1u)));
    }
    inline uint hyle_nv_e2m1(uint code) {
      const uint table[16] = {0u,0x3f000000u,0x3f800000u,0x3fc00000u,
          0x40000000u,0x40400000u,0x40800000u,0x40c00000u,
          0u,0xbf000000u,0xbf800000u,0xbfc00000u,
          0xc0000000u,0xc0400000u,0xc0800000u,0xc0c00000u};
      return table[code]; // The nonfast producer maps code8 to positive zero.
    }
    inline uint hyle_mx_scale(uint maximum) {
      const uint rounded = (maximum + 0x20u) & 0x7f80u;
      const int exponent = rounded == 0u ? -127 : rounded == 0x7f80u ? 127
          : clamp(int(rounded >> 7) - 129,-127,127);
      const uint code = uint(exponent + 127);
      return code == 0u ? 0x00400000u : code << 23;
    }
    inline uint hyle_mx_e8m0(uint code) {
      return code == 0u ? 0x00400000u : code << 23;
    }
    // CT local scales are dequant divisors, with a BF16 boundary before
    // the global multiply. Do not replace division by a cached reciprocal.
    inline uint hyle_nv_local_scale(uint maximum, uint global) {
      const uint local = hyle_nv_bfloat(hyle_f32_div(maximum << 16,0x40c00000u)) << 16;
      uint sf = hyle_nv_e4m3(hyle_nv_scale_code(hyle_f32_mul(global,local)));
      if (sf == 0u) sf = 0x3e000000u; // Only exact zero becomes E4 epsilon .125.
      return hyle_f32_div(sf,global);
    }
    inline uint hyle_nv_nearest(uint bits) {
      if (bits <= 0x3e800000u) return 0u;
      if (bits < 0x3f400000u) return 0x3f000000u;
      if (bits <= 0x3fa00000u) return 0x3f800000u;
      if (bits < 0x3fe00000u) return 0x3fc00000u;
      if (bits <= 0x40200000u) return 0x40000000u;
      if (bits < 0x40600000u) return 0x40400000u;
      return bits <= 0x40a00000u ? 0x40800000u : 0x40c00000u;
    }
    inline uint hyle_mx_quantize(uint value, uint scale) {
      const uint normalized = hyle_nv_bfloat(hyle_f32_div(value << 16,scale)) << 16;
      const uint quantized = hyle_nv_nearest(normalized & 0x7fffffffu) |
          (normalized & 0x80000000u);
      return hyle_f32_mul(quantized,scale);
    }
    inline uint hyle_nv_local_quantize(uint value, uint scale, bool pre_bfloat) {
      uint normalized = hyle_f32_div(value << 16,scale);
      normalized = (normalized & 0x80000000u) |
          min(normalized & 0x7fffffffu,0x40c00000u);
      if (pre_bfloat) normalized = hyle_nv_bfloat(normalized) << 16;
      uint quantized = hyle_nv_nearest(normalized & 0x7fffffffu);
      // Portable torch-sign contract: explicit +/-0 becomes +0; negative
      // nonzero normalized values which quantize to zero retain -0.
      if ((normalized & 0x7fffffffu) != 0u)
        quantized |= normalized & 0x80000000u;
      return hyle_f32_mul(quantized,scale);
    }
  )METAL";
}

// The kernels share one arithmetic body. Their cold signature contains only
// actual operands: MXFP4 has no WG/AG arrays or global-scale resources.
auto fp4_kernel(const std::string &name, const std::vector<std::string> &nv_inputs,
    const std::vector<std::string> &mx_inputs,
    const std::vector<std::string> &outputs, const std::string &source,
    bool mxfp4) {
  static const auto header = kernel_header();
  // MLX discovers shape/stride metadata from the body text before Metal's
  // preprocessor runs, so expand the global-access policy in that text.
  auto body = source;
  const auto expand = [&](const std::string &from, const std::string &to) {
    for (std::size_t at = 0; (at = body.find(from, at)) != std::string::npos;
         at += to.size()) body.replace(at, from.size(), to);
  };
  expand("HYLE_FP4_WG_COUNT", mxfp4 ? "1" : "wg_shape[0]");
  for (const auto index : {"0", "expert"}) {
    expand(std::string("HYLE_FP4_WG(") + index + ")",
        mxfp4 ? "0x3f800000u" : std::string("wg[") + index + "]");
    expand(std::string("HYLE_FP4_AG(") + index + ")",
        mxfp4 ? "0x3f800000u" : std::string("ag[") + index + "]");
  }
  return mx::fast::metal_kernel(name + (mxfp4 ? "_mx" : ""),
      mxfp4 ? mx_inputs : nv_inputs, outputs, body, header,
      false, false, mx::CompileOptions{});
}

enum class Fp4Policy { kNvfp4, kLocalNvfp4, kMxfp4 };

struct Grid final { int x, y, groups_x; };
Result<Grid> grid(std::uint64_t items, unsigned per_group, unsigned threads) {
  const auto groups = items / per_group + (items % per_group != 0);
  const auto x = (std::min)(groups, std::uint64_t{65535});
  const auto y = groups / x + (groups % x != 0);
  if (y > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
    return std::unexpected(Status::OutOfRange("Apple NVFP4 native grid exceeds int axes"));
  return Grid{static_cast<int>(x * threads), static_cast<int>(y), static_cast<int>(x)};
}
} // namespace

static Result<std::vector<mx::array>> lower_nvfp4_impl(const Operation &operation,
    std::span<const TensorDesc> descriptions, std::span<const mx::array> inputs,
    mx::Stream stream, Fp4Policy policy, bool pre_bfloat,
    bool weight_only = false) {
  const bool local = policy == Fp4Policy::kLocalNvfp4;
  const bool mxfp4 = policy == Fp4Policy::kMxfp4;
  const int block_size = mxfp4 ? 32 : 16;
  HYLE_ASSIGN_OR_RETURN(auto output, infer_single_output(operation, descriptions));
  if (inputs.size() != descriptions.size() || stream.device.type != mx::Device::gpu)
    return std::unexpected(Status::InvalidArgument("Apple FP4 needs matching GPU carriers"));
  for (std::size_t i = 0; i != inputs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(descriptions[i]));
    if (inputs[i].dtype() != spec.dtype || inputs[i].shape() != spec.shape)
      return std::unexpected(Status::InvalidArgument("Apple NVFP4 carrier differs from logical storage"));
  }
  const auto rows = static_cast<int>(descriptions[0].dimensions()[0]);
  const auto depth = static_cast<int>(descriptions[0].dimensions()[1]);
  const auto columns = static_cast<int>(output.dimensions()[1]);
  HYLE_ASSIGN_OR_RETURN(auto dot_grid, grid(std::uint64_t(rows) * columns, 4, 128));
  std::vector<mx::array> q, quantize_inputs;
  if (!weight_only) {
    HYLE_ASSIGN_OR_RETURN(auto activation_grid, grid(std::uint64_t(rows) * (depth / block_size), 128, 128));
    static const std::string quantize_source = R"METAL(
    const ulong group = ulong(thread_position_in_grid.y) * GROUP_WIDTH + thread_position_in_grid.x;
    const ulong groups = ulong(a_shape[0]) * (a_shape[1]/BLOCK_SIZE);
    if (group >= groups) return;
    const ulong row = group / (a_shape[1]/BLOCK_SIZE), block = group % (a_shape[1]/BLOCK_SIZE);
    uint values[BLOCK_SIZE], maximum = 0u;
    bool valid = MXFP4 || (hyle_nv_positive(HYLE_FP4_WG(0)) && hyle_nv_positive(HYLE_FP4_AG(0)));
    for (uint j=0; j<BLOCK_SIZE; ++j) {
      values[j] = uint(a[row*a_strides[0] + (block*BLOCK_SIZE+j)*a_strides[1]]);
      const uint magnitude = values[j] & 0x7fffu;
      valid = valid && magnitude < 0x7f80u;
      maximum = max(maximum,magnitude);
    }
    uint scale = 0x3f800000u;
    if (valid && MXFP4) scale = hyle_mx_scale(maximum);
    else if (valid && LOCAL) scale = hyle_nv_local_scale(maximum,HYLE_FP4_AG(0));
    else if (valid) {
      uint ratio = hyle_f32_div(maximum << 16,hyle_f32_mul(0x40c00000u,HYLE_FP4_AG(0)));
      ratio = min(ratio,0x43e00000u);
      scale = hyle_f32_mul(hyle_nv_e4m3(hyle_nv_scale_code(ratio)),HYLE_FP4_AG(0));
      if (scale < 0x3727c5acu) scale = 0x3f800000u; // F32(1e-5).
    }
    for (uint j=0; j<BLOCK_SIZE; ++j) {
      uint result = 0u;
      if (valid && MXFP4) result = hyle_mx_quantize(values[j],scale);
      else if (valid && LOCAL) result = hyle_nv_local_quantize(values[j],scale,PRE_BFLOAT);
      else if (valid) {
        const uint magnitude = (values[j] & 0x7fffu) << 16;
        result = hyle_f32_mul(hyle_nv_nearest(hyle_f32_div(magnitude,scale)),scale);
        // A>=0 maps -0 to +0. Negative nonzero inputs which round to zero
        // retain their sign, even when their F32 value would be flushed.
        if ((values[j] & 0x7fffu) != 0u) result ^= (values[j] & 0x8000u) << 16;
      }
      quantized[row*ulong(a_shape[1]) + block*BLOCK_SIZE+j] = ushort(hyle_nv_bfloat(result));
    }
    checks[group] = valid ? 0 : 3;
  )METAL";
    static const auto quantize_nv = fp4_kernel("hyle_nvfp4_reference_quantize",
        {"a", "wg", "ag"}, {"a"}, {"quantized", "checks"}, quantize_source, false);
    static const auto quantize_mx = fp4_kernel("hyle_nvfp4_reference_quantize",
        {"a", "wg", "ag"}, {"a"}, {"quantized", "checks"}, quantize_source, true);
    const auto &quantize = mxfp4 ? quantize_mx : quantize_nv;
    quantize_inputs = {mx::view(inputs[0], mx::uint16, stream)};
    if (!mxfp4) {
      quantize_inputs.push_back(mx::reshape(mx::view(inputs[3], mx::uint32, stream), {1}, stream));
      quantize_inputs.push_back(mx::reshape(mx::view(inputs[4], mx::uint32, stream), {1}, stream));
    }
    q = quantize(quantize_inputs, {{rows,depth},{rows,depth/block_size}}, {mx::uint16,mx::int32},
        {activation_grid.x,activation_grid.y,1}, {128,1,1},
        {{"GROUP_WIDTH",activation_grid.groups_x*128},
         {"BLOCK_SIZE",block_size}, {"MXFP4",int(mxfp4)}, {"LOCAL",int(local)}, {"PRE_BFLOAT",int(pre_bfloat)}}, std::nullopt, false, stream);
  }
  static const std::string dot_source = R"METAL(
    const uint lane = thread_index_in_simdgroup, subgroup = simdgroup_index_in_threadgroup;
    const ulong group = ulong(threadgroup_position_in_grid.y)*GROUPS_X + threadgroup_position_in_grid.x;
    const ulong index = group*4 + subgroup;
    const ulong columns = w_shape[0], rows = a_shape[0], depth = a_shape[1];
    if (index >= rows*columns) return;
    const ulong row = index/columns, column = index%columns;
    uint total = 0u;
    bool valid = MXFP4 || hyle_nv_positive(HYLE_FP4_WG(0));
    for (ulong k=lane; k<depth; k+=32) {
      const uint scale_code = sf[column*sf_strides[0] + (k/BLOCK_SIZE)*sf_strides[1]];
      valid = valid && (MXFP4 ? scale_code != 255u :
          ((scale_code & 127u) != 127u) &&
          ((scale_code & 128u) == 0u || (scale_code & 127u) == 0u));
      if (WEIGHT_ONLY && scale_code == 255u) continue;
      const uint packed = w[column*w_strides[0] + (k/2)*w_strides[1]];
      const uint code = (packed >> ((k%2)*4)) & 15u;
      const uint scaled = MXFP4 ? hyle_mx_e8m0(scale_code) : LOCAL ? hyle_f32_div(hyle_nv_e4m3(scale_code),HYLE_FP4_WG(0))
          : hyle_f32_mul(hyle_nv_e4m3(scale_code),HYLE_FP4_WG(0));
      const uint decoded = (LOCAL || MXFP4) && code == 8u ? 0x80000000u : hyle_nv_e2m1(code);
      // Weight-only MX explicitly stores BF16(scale), then BF16(code*scale).
      const uint operand_scale = WEIGHT_ONLY ? hyle_nv_bfloat(scaled) << 16 : scaled;
      const uint weight = hyle_nv_bfloat(hyle_f32_mul(decoded,operand_scale)) << 16;
      const uint raw_activation = uint(a[row*a_strides[0] + k*a_strides[1]]);
      if (WEIGHT_ONLY && (raw_activation & 0x7fffu) >= 0x7f80u) {
        valid = false;
        continue;
      }
      const uint value = raw_activation << 16;
      total = hyle_f32_add(total,hyle_f32_mul(value,weight));
    }
    for (ushort distance=16; distance>0; distance/=2)
      total = hyle_f32_add(total,simd_shuffle_down(total,distance));
    const bool invalid = simd_any(!valid);
    if (lane == 0) {
      out[index] = ushort(hyle_nv_bfloat(total));
      checks[index] = invalid ? 3 : 0;
    }
  )METAL";
  static const auto dot_nv = fp4_kernel("hyle_nvfp4_reference_dot",
      {"a", "w", "sf", "wg"}, {"a", "w", "sf"}, {"out", "checks"}, dot_source, false);
  static const auto dot_mx = fp4_kernel("hyle_nvfp4_reference_dot",
      {"a", "w", "sf", "wg"}, {"a", "w", "sf"}, {"out", "checks"}, dot_source, true);
  const auto &dot = mxfp4 ? dot_mx : dot_nv;
  auto activation = weight_only ? mx::view(inputs[0], mx::uint16, stream) : q[0];
  std::vector<mx::array> dot_inputs{activation,inputs[1],inputs[2]};
  if (!mxfp4) dot_inputs.push_back(quantize_inputs[1]);
  auto result = dot(dot_inputs, {{rows,columns},{rows,columns}},
      {mx::uint16,mx::int32}, {dot_grid.x,dot_grid.y,1}, {128,1,1},
      {{"GROUPS_X",dot_grid.groups_x}, {"BLOCK_SIZE",block_size}, {"MXFP4",int(mxfp4)},
       {"WEIGHT_ONLY",int(weight_only)}, {"LOCAL",int(local)}}, std::nullopt, false, stream);
  auto status = mx::max(result[1], false, stream);
  if (!weight_only)
    status = mx::maximum(mx::max(q[1], false, stream), status, stream);
  HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(
      mx::view(result[0], mx::bfloat16, stream), std::move(status), stream));
  return std::vector{std::move(checked)};
}

static Result<std::vector<mx::array>> lower_grouped_nvfp4_impl(
    const Operation &operation, std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream stream, Fp4Policy policy, bool pre_bfloat,
    bool weight_only = false) {
  const bool local = policy == Fp4Policy::kLocalNvfp4;
  const bool mxfp4 = policy == Fp4Policy::kMxfp4;
  const int block_size = mxfp4 ? 32 : 16;
  HYLE_ASSIGN_OR_RETURN(auto output, infer_single_output(operation, descriptions));
  if (inputs.size() != descriptions.size() || stream.device.type != mx::Device::gpu)
    return std::unexpected(Status::InvalidArgument(
        "Apple grouped FP4 needs matching GPU carriers"));
  for (std::size_t i = 0; i != inputs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(descriptions[i]));
    if (inputs[i].dtype() != spec.dtype || inputs[i].shape() != spec.shape)
      return std::unexpected(Status::InvalidArgument(
          "Apple grouped NVFP4 carrier differs from logical storage"));
  }
  const auto tokens = static_cast<int>(output.dimensions()[0]);
  const auto slots = static_cast<int>(output.dimensions()[1]);
  const auto columns = static_cast<int>(output.dimensions()[2]);
  const auto depth = static_cast<int>(descriptions[0].dimensions()[2]);
  if (tokens > (std::numeric_limits<int>::max)() / slots)
    return std::unexpected(Status::OutOfRange(
        "Apple grouped NVFP4 slot count exceeds native int"));
  const auto rows = tokens * slots;
  // MXFP4 activation QDQ is independent of expert. Preserve the real shared
  // storage in the intermediate; NVFP4 still depends on the selected global.
  const bool shared_activation = mxfp4 &&
      (descriptions[0].dimensions()[1] == 1 || descriptions[0].byte_strides()[1] == 0);
  const auto activation_rows = shared_activation ? tokens : rows;
  const auto id_index = mxfp4 ? 3 : 5;
  HYLE_ASSIGN_OR_RETURN(auto dot_grid,
      grid(std::uint64_t(rows) * columns, 4, 128));
  // A GPU dependency gates both numerical phases. Validate the complete ID
  // tensor before either phase may read any selected expert's metadata.
  auto invalid_ids = mx::logical_or(
      mx::less(inputs[id_index], mx::array(std::int32_t{0}), stream),
      mx::greater_equal(inputs[id_index],
          mx::array(static_cast<std::int32_t>(descriptions[1].dimensions()[0])), stream),
      stream);
  auto id_status = mx::reshape(mx::where(mx::any(invalid_ids, stream),
      mx::array(kIndexOutOfRange), mx::array(std::int32_t{0}), stream), {1}, stream);
  std::vector<mx::array> q, quantize_inputs;
  if (!weight_only) {
    if (std::uint64_t(activation_rows) * depth >
        (std::numeric_limits<std::size_t>::max)() / sizeof(std::uint16_t))
      return std::unexpected(Status::OutOfRange(
          "Apple grouped NVFP4 quantized activation exceeds address capacity"));
    HYLE_ASSIGN_OR_RETURN(auto activation_grid,
        grid(std::uint64_t(activation_rows) * (depth / block_size), 128, 128));
    static const std::string quantize_source = R"METAL(
    const ulong group = ulong(thread_position_in_grid.y) * GROUP_WIDTH + thread_position_in_grid.x;
    const ulong depth = a_shape[2], slots = ids_shape[1], blocks = depth/BLOCK_SIZE;
    const ulong groups = ulong(ids_shape[0])*(SHARED_A ? 1 : slots)*blocks;
    if (group >= groups) return;
    const ulong row = group/blocks, block = group%blocks;
    if (id_status[0] != 0) {
      for (uint j=0; j<BLOCK_SIZE; ++j) quantized[row*depth + block*BLOCK_SIZE+j] = 0;
      checks[group] = 1;
      return;
    }
    const ulong token = SHARED_A ? row : row/slots, slot = SHARED_A ? 0 : row%slots;
    const int expert = MXFP4 ? 0 : ids[token*ids_strides[0] + slot*ids_strides[1]];
    // No expert metadata or weight address may be formed from an invalid ID.
    if (expert < 0 || expert >= HYLE_FP4_WG_COUNT) {
      for (uint j=0; j<BLOCK_SIZE; ++j) quantized[row*depth + block*BLOCK_SIZE+j] = 0;
      checks[group] = 1;
      return;
    }
    const uint weight_global = MXFP4 ? 0x3f800000u : HYLE_FP4_WG(expert);
    const uint activation_global = MXFP4 ? 0x3f800000u : HYLE_FP4_AG(expert);
    const ulong activation_slot = a_shape[1] == 1 ? 0 : slot;
    const ulong origin = token*a_strides[0] + activation_slot*a_strides[1];
    uint values[BLOCK_SIZE], maximum = 0u;
    bool valid = hyle_nv_positive(weight_global) && hyle_nv_positive(activation_global);
    for (uint j=0; j<BLOCK_SIZE; ++j) {
      values[j] = uint(a[origin + (block*BLOCK_SIZE+j)*a_strides[2]]);
      const uint magnitude = values[j] & 0x7fffu;
      valid = valid && magnitude < 0x7f80u;
      maximum = max(maximum,magnitude);
    }
    uint scale = 0x3f800000u;
    if (valid && MXFP4) scale = hyle_mx_scale(maximum);
    else if (valid && LOCAL) scale = hyle_nv_local_scale(maximum,activation_global);
    else if (valid) {
      uint ratio = hyle_f32_div(maximum << 16,
          hyle_f32_mul(0x40c00000u,activation_global));
      ratio = min(ratio,0x43e00000u);
      scale = hyle_f32_mul(hyle_nv_e4m3(hyle_nv_scale_code(ratio)),activation_global);
      if (scale < 0x3727c5acu) scale = 0x3f800000u;
    }
    for (uint j=0; j<BLOCK_SIZE; ++j) {
      uint result = 0u;
      if (valid && MXFP4) result = hyle_mx_quantize(values[j],scale);
      else if (valid && LOCAL) result = hyle_nv_local_quantize(values[j],scale,PRE_BFLOAT);
      else if (valid) {
        const uint magnitude = (values[j] & 0x7fffu) << 16;
        result = hyle_f32_mul(
            hyle_nv_nearest(hyle_f32_div(magnitude,scale)),scale);
        if ((values[j] & 0x7fffu) != 0u)
          result ^= (values[j] & 0x8000u) << 16;
      }
      quantized[row*depth + block*BLOCK_SIZE+j] = ushort(hyle_nv_bfloat(result));
    }
    checks[group] = valid ? 0 : 3;
  )METAL";
    static const auto quantize_nv = fp4_kernel("hyle_grouped_nvfp4_reference_quantize",
        {"a", "wg", "ag", "ids", "id_status"}, {"a", "ids", "id_status"}, {"quantized", "checks"}, quantize_source, false);
    static const auto quantize_mx = fp4_kernel("hyle_grouped_nvfp4_reference_quantize",
        {"a", "wg", "ag", "ids", "id_status"}, {"a", "ids", "id_status"}, {"quantized", "checks"}, quantize_source, true);
    const auto &quantize = mxfp4 ? quantize_mx : quantize_nv;
    quantize_inputs = {mx::view(inputs[0], mx::uint16, stream)};
    if (!mxfp4) {
      quantize_inputs.push_back(mx::view(inputs[3], mx::uint32, stream));
      quantize_inputs.push_back(mx::view(inputs[4], mx::uint32, stream));
    }
    quantize_inputs.push_back(inputs[id_index]);
    quantize_inputs.push_back(id_status);
    q = quantize(quantize_inputs,
        {{activation_rows,depth},{activation_rows,depth/block_size}}, {mx::uint16,mx::int32},
        {activation_grid.x,activation_grid.y,1}, {128,1,1},
        {{"GROUP_WIDTH",activation_grid.groups_x*128},
         {"BLOCK_SIZE",block_size}, {"MXFP4",int(mxfp4)},
         {"SHARED_A",int(shared_activation)}, {"LOCAL",int(local)}, {"PRE_BFLOAT",int(pre_bfloat)}}, std::nullopt, false, stream);
  }
  static const std::string dot_source = R"METAL(
    const uint lane = thread_index_in_simdgroup, subgroup = simdgroup_index_in_threadgroup;
    const ulong group = ulong(threadgroup_position_in_grid.y)*GROUPS_X + threadgroup_position_in_grid.x;
    const ulong index = group*4 + subgroup;
    const ulong columns = w_shape[1], rows = ulong(ids_shape[0])*ids_shape[1], depth = a_shape[1];
    if (index >= rows*columns) return;
    if (id_status[0] != 0) {
      if (lane == 0) { out[index] = 0; checks[index] = 1; }
      return;
    }
    const ulong row = index/columns, column = index%columns;
    const ulong token = row/ids_shape[1], slot = row%ids_shape[1];
    const int expert = ids[token*ids_strides[0] + slot*ids_strides[1]];
    if (expert < 0 || expert >= w_shape[0]) {
      if (lane == 0) { out[index] = 0; checks[index] = 1; }
      return;
    }
    const uint weight_global = MXFP4 ? 0x3f800000u : HYLE_FP4_WG(expert);
    uint total = 0u;
    bool valid = hyle_nv_positive(weight_global);
    for (ulong k=lane; k<depth; k+=32) {
      const uint scale_code =
          sf[ulong(expert)*sf_strides[0] + column*sf_strides[1] + (k/BLOCK_SIZE)*sf_strides[2]];
      valid = valid && (MXFP4 ? scale_code != 255u :
          ((scale_code & 127u) != 127u) &&
          ((scale_code & 128u) == 0u || (scale_code & 127u) == 0u));
      if (WEIGHT_ONLY && scale_code == 255u) continue;
      const uint packed =
          w[ulong(expert)*w_strides[0] + column*w_strides[1] + (k/2)*w_strides[2]];
      const uint code = (packed >> ((k%2)*4)) & 15u;
      const uint scaled = MXFP4 ? hyle_mx_e8m0(scale_code) : LOCAL ? hyle_f32_div(hyle_nv_e4m3(scale_code),weight_global)
          : hyle_f32_mul(hyle_nv_e4m3(scale_code),weight_global);
      const uint decoded = (LOCAL || MXFP4) && code == 8u ? 0x80000000u : hyle_nv_e2m1(code);
      // Weight-only MX explicitly stores BF16(scale), then BF16(code*scale).
      const uint operand_scale = WEIGHT_ONLY ? hyle_nv_bfloat(scaled) << 16 : scaled;
      const uint weight = hyle_nv_bfloat(hyle_f32_mul(decoded,operand_scale)) << 16;
      const uint raw_activation = uint(a[(SHARED_A ? token : row)*a_strides[0] + k*a_strides[1]]);
      if (WEIGHT_ONLY && (raw_activation & 0x7fffu) >= 0x7f80u) {
        valid = false;
        continue;
      }
      const uint value = raw_activation << 16;
      total = hyle_f32_add(total,hyle_f32_mul(value,weight));
    }
    for (ushort distance=16; distance>0; distance/=2)
      total = hyle_f32_add(total,simd_shuffle_down(total,distance));
    const bool invalid = simd_any(!valid);
    if (lane == 0) {
      out[index] = ushort(hyle_nv_bfloat(total));
      checks[index] = invalid ? 3 : 0;
    }
  )METAL";
  static const auto dot_nv = fp4_kernel("hyle_grouped_nvfp4_reference_dot",
      {"a", "w", "sf", "wg", "ids", "id_status"}, {"a", "w", "sf", "ids", "id_status"}, {"out", "checks"}, dot_source, false);
  static const auto dot_mx = fp4_kernel("hyle_grouped_nvfp4_reference_dot",
      {"a", "w", "sf", "wg", "ids", "id_status"}, {"a", "w", "sf", "ids", "id_status"}, {"out", "checks"}, dot_source, true);
  const auto &dot = mxfp4 ? dot_mx : dot_nv;
  auto activation = weight_only ? mx::view(inputs[0], mx::uint16, stream) : q[0];
  if (weight_only) {
    if (shared_activation) {
      if (activation.shape(1) != 1)
        activation = mx::slice(activation, {0,0,0}, {tokens,1,depth}, stream);
      activation = mx::squeeze(activation, 1, stream);
    } else {
      activation = mx::reshape(activation, {rows,depth}, stream);
    }
  }
  std::vector<mx::array> dot_inputs{activation,inputs[1],inputs[2]};
  if (!mxfp4) dot_inputs.push_back(quantize_inputs[1]);
  dot_inputs.push_back(inputs[id_index]);
  dot_inputs.push_back(id_status);
  auto result = dot(dot_inputs,
      {{tokens,slots,columns},{rows,columns}}, {mx::uint16,mx::int32},
      {dot_grid.x,dot_grid.y,1}, {128,1,1},
      {{"GROUPS_X",dot_grid.groups_x}, {"BLOCK_SIZE",block_size}, {"MXFP4",int(mxfp4)},
       {"WEIGHT_ONLY",int(weight_only)},
       {"SHARED_A",int(shared_activation)}, {"LOCAL",int(local)}}, std::nullopt, false, stream);
  auto status = mx::max(result[1], false, stream);
  if (!weight_only)
    status = mx::maximum(mx::max(q[1], false, stream), status, stream);
  HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(
      mx::view(result[0], mx::bfloat16, stream), std::move(status), stream));
  return std::vector{std::move(checked)};
}
Result<std::vector<mx::array>> lower_nvfp4(std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream stream) {
  return lower_nvfp4_impl(model_ops::nvfp4_linear(), descriptions, inputs, stream,
      Fp4Policy::kNvfp4, false);
}

Result<std::vector<mx::array>> lower_grouped_nvfp4(
    std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream stream) {
  return lower_grouped_nvfp4_impl(model_ops::grouped_nvfp4_linear(), descriptions,
      inputs, stream, Fp4Policy::kNvfp4, false);
}

Result<std::vector<mx::array>> lower_local_nvfp4(
    const model_ops::LocalNvfp4Linear &operation,
    std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream stream) {
  return lower_nvfp4_impl(Operation::custom(operation), descriptions, inputs, stream,
      Fp4Policy::kLocalNvfp4, operation.rounding == model_ops::Fp4InputRounding::kBFloat16);
}

Result<std::vector<mx::array>> lower_grouped_local_nvfp4(
    const model_ops::GroupedLocalNvfp4Linear &operation,
    std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream stream) {
  return lower_grouped_nvfp4_impl(Operation::custom(operation), descriptions,
      inputs, stream, Fp4Policy::kLocalNvfp4,
      operation.rounding == model_ops::Fp4InputRounding::kBFloat16);
}
Result<std::vector<mx::array>> lower_mxfp4(
    const model_ops::Mxfp4Linear &operation,
    std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream stream) {
  return lower_nvfp4_impl(Operation::custom(operation), descriptions, inputs, stream,
      Fp4Policy::kMxfp4, true,
      operation.activation_mode == model_ops::Mxfp4ActivationMode::kBFloat16);
}

Result<std::vector<mx::array>> lower_grouped_mxfp4(
    const model_ops::GroupedMxfp4Linear &operation,
    std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream stream) {
  return lower_grouped_nvfp4_impl(Operation::custom(operation), descriptions,
      inputs, stream, Fp4Policy::kMxfp4, true,
      operation.activation_mode == model_ops::Mxfp4ActivationMode::kBFloat16);
}
} // namespace hyle::compute::apple::detail
