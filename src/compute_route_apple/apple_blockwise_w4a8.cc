#include "apple_blockwise_w4a8.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "apple_checks.h"
#include "hyle/compute/model_ops.h"
#include "../compute_shader_shared/nvfp4_f32.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
// Reuse the private IEEE arithmetic, not the FP4 operation or its executor.
// Native Metal floating subnormal handling must not change BF16 boundaries.
std::string arithmetic() {
  return std::string(R"METAL(
    inline uint hyle_mul_hi(uint a, uint b) {
      return uint((ulong(a) * ulong(b)) >> 32);
    }
    inline uint hyle_clz(uint a) { return clz(a); }
  )METAL") + shader::kNvfp4F32 + R"METAL(
    inline uint hyle_w4a8_bfloat(uint bits) {
      if ((bits & 0x7fffffffu) > 0x7f800000u)
        return (bits >> 16) | 0x40u;
      return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
    }
    inline bool hyle_w4a8_trunc(uint bits, thread int &code) {
      const uint magnitude = bits & 0x7fffffffu;
      if (magnitude >= 0x7f800000u) return false;
      const int exponent = int(magnitude >> 23) - 127;
      if (exponent < 0) { code = 0; return true; }
      if (exponent > 7) return false;
      code = int(((magnitude & 0x7fffffu) | 0x800000u) >> uint(23-exponent));
      if ((bits & 0x80000000u) != 0u) code = -code;
      return code >= -128 && code <= 127;
    }
  )METAL";
}

auto quantize_kernel(bool grouped) {
  const std::string source = R"METAL(
    const ulong group = ulong(thread_position_in_grid.y)*GROUP_WIDTH + thread_position_in_grid.x;
    const ulong blocks = DEPTH/512;
    if (group >= ulong(QUANT_ROWS)*blocks) return;
    const ulong row = group/blocks, block = group%blocks;
  )METAL" + std::string(grouped ? R"METAL(
    if (id_status[0] != 0) {
      for (uint j=0; j<512; ++j) quantized[row*DEPTH + block*512+j] = 0;
      scales[group] = 0; checks[group] = 1; return;
    }
    const ulong token = SHARED_A ? row : row/SLOTS;
    const ulong slot = SHARED_A ? 0 : row%SLOTS;
    const ulong origin = token*a_strides[0] + slot*a_strides[1];
    const ulong step = a_strides[2];
  )METAL" : R"METAL(
    const ulong origin = row*a_strides[0], step = a_strides[1];
  )METAL") + R"METAL(
    uint maximum = 0u;
    bool valid = true;
    for (uint j=0; j<512; ++j) {
      const uint magnitude = uint(a[origin + (block*512+j)*step]) & 0x7fffu;
      maximum = max(maximum,magnitude);
      valid = valid && magnitude < 0x7f80u;
    }
    uint scale = 0u, inverse = 0u;
    if (valid) {
      // BF16(1/127), not correctly-rounded division by127.
      scale = hyle_w4a8_bfloat(hyle_f32_mul(maximum << 16,0x3c010000u));
      // Source zero scale bypasses reciprocal, including tiny underflow.
      if (scale != 0u)
        inverse = hyle_w4a8_bfloat(hyle_f32_div(0x3f800000u,scale << 16));
    }
    for (uint j=0; j<512; ++j) {
      int code = 0;
      if (valid) {
        const uint value = uint(a[origin + (block*512+j)*step]);
        const uint normalized = hyle_w4a8_bfloat(hyle_f32_mul(value << 16,inverse << 16)) << 16;
        valid = hyle_w4a8_trunc(normalized,code);
      }
      quantized[row*DEPTH + block*512+j] = char(code);
    }
    scales[group] = ushort(scale);
    checks[group] = valid ? 0 : 3;
  )METAL";
  return mx::fast::metal_kernel(
      grouped ? "hyle_grouped_blockwise_w4a8_quantize" : "hyle_blockwise_w4a8_quantize",
      grouped ? std::vector<std::string>{"a", "id_status"} : std::vector<std::string>{"a"},
      {"quantized", "scales", "checks"}, source, arithmetic(),
      false, false, mx::CompileOptions{});
}

auto dot_kernel(bool grouped) {
  const std::string source = R"METAL(
    const uint lane = thread_index_in_simdgroup, subgroup = simdgroup_index_in_threadgroup;
    const ulong group = ulong(threadgroup_position_in_grid.y)*GROUPS_X + threadgroup_position_in_grid.x;
    const ulong index = group*4 + subgroup;
    if (index >= ulong(ROWS)*COLUMNS) return;
    const ulong row = index/COLUMNS, column = index%COLUMNS;
    const ulong blocks = DEPTH/512;
  )METAL" + std::string(grouped ? R"METAL(
    // Full-ID reduction is a data dependency before any expert address.
    if (id_status[0] != 0) {
      if (lane == 0) { out[index] = 0; checks[index] = 1; }
      return;
    }
    const ulong token = row/SLOTS, slot = row%SLOTS;
    const int expert = ids[token*ids_strides[0] + slot*ids_strides[1]];
    if (expert < 0 || expert >= w_shape[0]) {
      if (lane == 0) { out[index] = 0; checks[index] = 1; }
      return;
    }
    const ulong w_origin = ulong(expert)*w_strides[0] + column*w_strides[1];
    const ulong sf_origin = ulong(expert)*sf_strides[0] + column*sf_strides[1];
    const ulong w_step = w_strides[2], sf_step = sf_strides[2];
    const ulong activation_row = SHARED_A ? token : row;
  )METAL" : R"METAL(
    const ulong w_origin = column*w_strides[0], sf_origin = column*sf_strides[0];
    const ulong w_step = w_strides[1], sf_step = sf_strides[1];
    const ulong activation_row = row;
  )METAL") + R"METAL(
    uint total = 0u;
    bool valid = true;
    // Accumulate groups in source K order, rounding to BF16 after each add.
    for (ulong block=0; block<blocks; ++block) {
      int partial = 0;
      for (uint j=lane; j<512; j+=32) {
        const ulong k = block*512+j;
        const uint packed = uint(w[w_origin + (k/8)*w_step]);
        const int weight = int((packed >> uint((k%8)*4)) & 15u)-8;
        partial += int(q[activation_row*DEPTH+k])*weight;
      }
      const int sum = simd_sum(partial);
      if (lane == 0) {
        const uint sf_bits = sf[sf_origin+block*sf_step];
        valid = valid && sf_bits != 0u && sf_bits < 0x7f800000u;
        // INT32 is exact in F32 at this group512 bound. Every following
        // boundary is an explicit BF16 RNE using IEEE integer arithmetic.
        const uint dot = hyle_w4a8_bfloat(as_type<uint>(float(sum))) << 16;
        const uint activation_scale = uint(asf[activation_row*blocks+block]) << 16;
        const uint weight_scale = hyle_w4a8_bfloat(sf_bits) << 16;
        uint value = hyle_w4a8_bfloat(hyle_f32_mul(dot,activation_scale)) << 16;
        value = hyle_w4a8_bfloat(hyle_f32_mul(value,weight_scale)) << 16;
        total = hyle_w4a8_bfloat(hyle_f32_add(total << 16,value));
      }
    }
    if (lane == 0) { out[index] = ushort(total); checks[index] = valid ? 0 : 3; }
  )METAL";
  return mx::fast::metal_kernel(
      grouped ? "hyle_grouped_blockwise_w4a8_dot" : "hyle_blockwise_w4a8_dot",
      grouped ? std::vector<std::string>{"q", "asf", "w", "sf", "ids", "id_status"}
              : std::vector<std::string>{"q", "asf", "w", "sf"},
      {"out", "checks"}, source, arithmetic(), false, false, mx::CompileOptions{});
}

struct Grid final { int x, y, groups_x; };
Result<Grid> grid(std::uint64_t items, unsigned per_group) {
  const auto groups = items/per_group + (items%per_group != 0);
  const auto x = (std::min)(groups,std::uint64_t{65535});
  const auto y = groups/x + (groups%x != 0);
  if (y > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
    return std::unexpected(Status::OutOfRange("Apple W4A8 grid exceeds int axes"));
  return Grid{static_cast<int>(x*128),static_cast<int>(y),static_cast<int>(x)};
}
} // namespace

Result<std::vector<mx::array>> lower_blockwise_w4a8(
    std::span<const TensorDesc> descriptions, std::span<const mx::array> inputs,
    mx::Stream stream, bool grouped) {
  const auto operation = grouped ? model_ops::grouped_blockwise_w4a8_linear()
                                 : model_ops::blockwise_w4a8_linear();
  HYLE_ASSIGN_OR_RETURN(auto output, infer_single_output(operation,descriptions));
  if (inputs.size() != descriptions.size() || stream.device.type != mx::Device::gpu)
    return std::unexpected(Status::InvalidArgument("Apple W4A8 requires matching GPU carriers"));
  for (std::size_t i=0; i<inputs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto spec,input_spec(descriptions[i]));
    if (inputs[i].dtype() != spec.dtype || inputs[i].shape() != spec.shape)
      return std::unexpected(Status::InvalidArgument("Apple W4A8 carrier differs from storage"));
  }
  const auto tokens = static_cast<int>(output.dimensions()[0]);
  const auto slots = grouped ? static_cast<int>(output.dimensions()[1]) : 1;
  const auto columns = static_cast<int>(output.dimensions().back());
  const auto depth = static_cast<int>(descriptions[0].dimensions().back());
  if (tokens > (std::numeric_limits<int>::max)()/slots)
    return std::unexpected(Status::OutOfRange("Apple W4A8 slot count exceeds native int"));
  const int rows = tokens*slots;
  const bool shared = grouped && (descriptions[0].dimensions()[1] == 1 ||
                                  descriptions[0].byte_strides()[1] == 0);
  const int quant_rows = shared ? tokens : rows;
  HYLE_ASSIGN_OR_RETURN(auto qgrid,grid(std::uint64_t(quant_rows)*(depth/512),128));
  HYLE_ASSIGN_OR_RETURN(auto dgrid,grid(std::uint64_t(rows)*columns,4));
  std::optional<mx::array> id_status;
  if (grouped) {
    auto invalid = mx::logical_or(mx::less(inputs[3],mx::array(std::int32_t{0}),stream),
        mx::greater_equal(inputs[3],mx::array(static_cast<std::int32_t>(
            descriptions[1].dimensions()[0])),stream),stream);
    id_status = mx::reshape(mx::where(mx::any(invalid,stream),
        mx::array(kIndexOutOfRange),mx::array(std::int32_t{0}),stream),{1},stream);
  }
  static const auto quantize_dense = quantize_kernel(false);
  static const auto quantize_grouped = quantize_kernel(true);
  const auto &quantize = grouped ? quantize_grouped : quantize_dense;
  std::vector<mx::array> qinputs{mx::view(inputs[0],mx::uint16,stream)};
  if (grouped) qinputs.push_back(*id_status);
  auto q = quantize(qinputs,{{quant_rows,depth},{quant_rows,depth/512},{quant_rows,depth/512}},
      {mx::int8,mx::uint16,mx::int32},{qgrid.x,qgrid.y,1},{128,1,1},
      {{"GROUP_WIDTH",qgrid.groups_x*128},{"DEPTH",depth},{"QUANT_ROWS",quant_rows},
       {"SLOTS",slots},{"SHARED_A",int(shared)}},std::nullopt,false,stream);
  static const auto dot_dense = dot_kernel(false);
  static const auto dot_grouped = dot_kernel(true);
  const auto &dot = grouped ? dot_grouped : dot_dense;
  std::vector<mx::array> dinputs{q[0],q[1],inputs[1],mx::view(inputs[2],mx::uint32,stream)};
  if (grouped) { dinputs.push_back(inputs[3]); dinputs.push_back(*id_status); }
  HYLE_ASSIGN_OR_RETURN(auto ospec,input_spec(output));
  auto result = dot(dinputs,{ospec.shape,{rows,columns}},{mx::uint16,mx::int32},
      {dgrid.x,dgrid.y,1},{128,1,1},{{"GROUPS_X",dgrid.groups_x},{"ROWS",rows},
       {"COLUMNS",columns},{"DEPTH",depth},{"SLOTS",slots},{"SHARED_A",int(shared)}},
      std::nullopt,false,stream);
  auto status = mx::maximum(mx::max(q[2],false,stream),mx::max(result[1],false,stream),stream);
  HYLE_ASSIGN_OR_RETURN(auto checked,with_device_status(
      mx::view(result[0],mx::bfloat16,stream),std::move(status),stream));
  return std::vector{std::move(checked)};
}
} // namespace hyle::compute::apple::detail
