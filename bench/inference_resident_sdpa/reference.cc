#include "reference.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "reference_spirv.h"
#include "vulkan_program_internal.h"

namespace hyle::bench::resident {
namespace c = compute;
namespace v = compute::vulkan;
namespace d = v::detail;
namespace {
template <class T> T take(Result<T> result) {
  if (!result) {
    std::fprintf(stderr, "resident reference: %.*s\n",
                 int(result.error().message().size()),
                 result.error().message().data());
    std::exit(2);
  }
  return std::move(*result);
}
void check(Status status) {
  if (!status.ok()) {
    std::fprintf(stderr, "resident reference: %.*s\n",
                 int(status.message().size()), status.message().data());
    std::exit(2);
  }
}
template <std::size_t N>
std::vector<std::byte> bytes(const std::array<std::uint32_t, N> &words) {
  std::vector<std::byte> result(sizeof(words));
  std::memcpy(result.data(), words.data(), sizeof(words));
  return result;
}
std::array<std::uint32_t, 2> divisor(std::uint32_t value) {
  const auto shift = std::uint32_t(std::bit_width(value - 1));
  const auto multiplier =
      ((std::uint64_t{1} << 32) * ((std::uint64_t{1} << shift) - value)) /
          value +
      1;
  return {std::uint32_t(multiplier), shift};
}
} // namespace

v::Program reference_program(const v::Context &context, const c::TensorDesc &qd,
                             const c::TensorDesc &kd, bool causal) {
  if(qd.rank()!=4){std::fprintf(stderr,"invalid resident reference rank\n");std::exit(2);}
  return encoded_reference_program(context,qd,kd,kd,qd,causal,-1,
      1.F/std::sqrt(float(qd.dimensions().back())));
}

v::Program encoded_reference_program(const v::Context &context,
    const c::TensorDesc &qd,const c::TensorDesc &kd,const c::TensorDesc &vd,
    const c::TensorDesc &od,bool causal,std::int32_t window_left,float scale) {
  const auto &facts = context.environment();
  // These are the two actually available machines. Do not guess AMD/Intel
  // architecture or occupancy properties for an unqualified third machine.
  if (facts.vendor_id != 0x10de && facts.vendor_id != 0x106b) {
    std::fprintf(stderr, "unqualified resident reference device vendor\n");
    std::exit(2);
  }
  const auto type = qd.data_type();
  if (qd.rank() != 4 || kd.rank() != 4 || vd.rank()!=4 || od.rank()!=4) {
    std::fprintf(stderr, "invalid resident reference rank\n");
    std::exit(2);
  }
  const auto shape = qd.dimensions();
  const auto kshape = kd.dimensions();
  const auto vshape = vd.dimensions();
  const auto oshape = od.dimensions();
  const auto batch = std::uint32_t(shape[0]), rows = std::uint32_t(shape[1]),
             heads = std::uint32_t(shape[2]), depth = std::uint32_t(shape[3]),
             keys = std::uint32_t(kshape[1]),
             kv_heads = std::uint32_t(kshape[2]);
  const auto value_depth=std::uint32_t(oshape[3]);
  const bool quantized=kd.data_type()==c::DataType::kUInt8;
  if (qd.rank() != 4 || kd.rank() != 4 || !qd.is_dense() ||
      !kd.is_dense() || !vd.is_dense() || !od.is_dense() ||
      (type != c::DataType::kFloat32 && type != c::DataType::kFloat16) ||
      (!quantized && kd.data_type()!=type) || vd.data_type()!=kd.data_type() ||
      kshape[3]!=(quantized?depth/32*34:depth) ||
      vshape[3]!=(quantized?value_depth/32*34:value_depth) ||
      vshape[0]!=batch || vshape[1]!=keys || vshape[2]!=kv_heads ||
      od.data_type()!=type || oshape[0]!=batch || oshape[1]!=rows || oshape[2]!=heads ||
      !kv_heads || heads % kv_heads != 0 || batch != kshape[0] || !rows || !keys || !heads ||
      (depth != 64 && depth != 128 && depth != 256) ||
      (value_depth!=64 && value_depth!=128 && value_depth!=256) ||
      !std::isfinite(scale) || window_left< -1 || (!causal && window_left>=0) ||
      (causal && keys < rows)) {
    std::fprintf(stderr, "invalid finite resident reference shape\n");
    std::exit(2);
  }
  auto state = std::make_shared<d::ProgramState>();
  state->runtime = d::Access::runtime(context);
  state->values = {{qd, {}}, {kd, {}}, {vd, {}}, {od, {}}};
  state->inputs = {0, 1, 2};
  state->outputs = {3};
  state->input_descriptions = {qd, kd, vd};
  state->output_descriptions = {od};
  auto add = [&](c::DataType value_type, std::span<const std::int64_t> dims) {
    const auto id = std::uint32_t(state->values.size());
    state->values.push_back({take(c::TensorDesc::dense(value_type, dims)), {}});
    return id;
  };
  auto append = [&](std::span<const std::uint32_t> code, std::uint32_t bindings,
                    std::vector<std::uint32_t> values,
                    std::vector<std::byte> push,
                    std::array<std::uint32_t, 3> groups,
                    std::span<const v::SpecializationConstant> spec = {}) {
    auto kernel = take(v::Kernel::create(
        context, {code, bindings, std::uint32_t(push.size()), spec}));
    state->dispatches.push_back({d::Access::pipeline(kernel),
                                 {},
                                 {},
                                 std::move(values),
                                 std::move(push),
                                 groups});
  };
  auto convert = [&](std::uint32_t source, std::uint32_t destination,
                     std::span<const std::uint32_t> code) {
    const auto count =
        std::uint32_t(*state->values[source].description.element_count());
    std::array<std::uint32_t, 30> push{};
    push[0] = count;
    // Flattened contiguous conversion. Original fastdiv constants, no copy
    // kernel authored here and no full K/V conversion in this recipe.
    for (const auto base : {1u, 9u}) {
      push[base] = count;
      push[base + 1] = push[base + 2] = push[base + 3] = 1;
      push[base + 4] = 1;
      push[base + 5] = push[base + 6] = push[base + 7] = count;
    }
    const auto div = divisor(count);
    for (const auto base : {22u, 26u}) {
      push[base] = push[base + 1] = push[base + 2] = div[0];
      push[base + 3] = div[1] | (div[1] << 8) | (div[1] << 16);
    }
    append(code, 2, {source, destination}, bytes(push),
           {1, std::min((count + 511) / 512, 512u), (count + 262143) / 262144});
  };
  auto query = 0u, output = 3u;
  if (type == c::DataType::kFloat16) {
    query = add(c::DataType::kFloat32, shape);
    output = add(c::DataType::kFloat32, oshape);
    convert(0, query, spirv::query_float);
  }
  std::uint32_t mask = query;
  if(causal && (window_left>=0 || quantized)){
    const std::array<std::int64_t,2> mask_shape{rows,keys};
    mask=add(c::DataType::kFloat16,mask_shape);
    std::vector<std::uint16_t> words(std::size_t(rows)*keys,0xfc00);
    for(std::uint32_t row=0;row<rows;++row){
      const auto last=keys-rows+row;
      const auto first=window_left<0?0u:
          (last>std::uint32_t(window_left)?last-std::uint32_t(window_left):0u);
      std::fill(words.begin()+std::size_t(row)*keys+first,
                words.begin()+std::size_t(row)*keys+last+1,std::uint16_t{0});
    }
    state->values[mask].constant=take(context.upload(state->values[mask].description,
        std::as_bytes(std::span(words))));
    std::fprintf(stderr,"# ORIGINAL_Q8_MASK_COLD bytes=%zu window_left=%d\n",
                 words.size()*sizeof(std::uint16_t),int(window_left));
  }else if (causal) {
    const std::array<std::int64_t, 2> mask_shape{rows, keys};
    const auto seed = add(c::DataType::kFloat32, mask_shape);
    const auto masked = seed;
    mask = add(c::DataType::kFloat16, mask_shape);
    append(spirv::fill, 1, {seed},
           bytes(std::array<std::uint32_t, 6>{rows * keys, 0, 0, 0, 0, 0}),
           {(rows * keys + 511) / 512, 1, 1});
    append(spirv::mask, 2, {seed, masked},
           bytes(std::array<std::uint32_t, 3>{keys, rows, keys - rows}),
           {rows, (keys + 511) / 512, 1});
    convert(masked, mask, spirv::half);
  }
  const auto subgroup = facts.subgroup_size;
  const auto ratio = heads / kv_heads;
  const bool reduced = keys < 1024;
  auto effective_rows = rows;
  std::uint32_t group_ratio = 1;
  if (rows <= 8 && ratio > 1 && ratio <= (reduced ? 8u : 16u)) {
    group_ratio = ratio;
    effective_rows = ratio;
  }
  const auto row_split = effective_rows < 4 || depth <= 64 ? 1u : 4u;
  const auto workgroup =
      subgroup > 32 &&
              (effective_rows < 4 || depth < (row_split == 1 ? 128u : 64u))
          ? subgroup * 2
          : subgroup * 4;
  auto br = effective_rows == 1 ? 1u
            : row_split == 1    ? (effective_rows == 2              ? 2u
                                   : effective_rows <= 4 || reduced ? 4u
                                                                    : 8u)
                                : (effective_rows <= 4              ? 4u
                                   : effective_rows <= 8 || reduced ? 8u
                                                                    : 16u);
  const auto bc = effective_rows == 1 ? 64u : 32u;
  const auto ds = std::min({subgroup, 8u, (depth & (~depth + 1)) / 4});
  const auto staged = facts.vendor_id == 0x10de && depth < 256 ? 1u : 0u;
  const auto shmem = [&](std::uint32_t block_rows) {
    return workgroup * 4 + workgroup * 16 + bc * (block_rows + 1) * 4 + 16 * 4 +
           block_rows * (depth / 4 + 1) * 16 +
           (staged ? bc * (std::max(depth,value_depth) / 4 + 1) * 16 : 16);
  };
  if (!reduced && shmem(br) > facts.shared_memory_bytes)
    br /= 2;
  if (!subgroup || !br || shmem(br) > facts.shared_memory_bytes) {
    std::fprintf(stderr, "unqualified original flash shared-memory domain\n");
    std::exit(2);
  }
  // Exact native split-K core count on NVIDIA; upstream's fallback is 16 for
  // the Air. Never assume a Windows GPU's core count from its marketing name.
  std::uint32_t cores = 16;
  if (facts.vendor_id == 0x10de) {
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV sm{};
    sm.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &sm;
    state->runtime->device->instance_fn.GetPhysicalDeviceProperties2(
        state->runtime->device->native.physical_device, &properties);
    cores = sm.shaderSMCount ? sm.shaderSMCount : 16;
  }
  auto split = 1u;
  const auto y = heads / group_ratio;
  const auto tr = (effective_rows + br - 1) / br;
  if (group_ratio > 1 && rows <= br)
    split = cores * 2 / (rows * y * batch);
  else if (group_ratio == 1 && tr * y * batch < cores * 2)
    split = cores * 2 / (tr * y * batch);
  auto split_kv = keys;
  if (split > 1) {
    split_kv = ((std::max(1u, keys / split) + bc - 1) / bc) * bc;
    split = (keys + split_kv - 1) / split_kv;
  }
  const bool half = type == c::DataType::kFloat16;
  const bool mask_opt = causal && rows >= 32 &&
                        std::uint64_t(rows) * keys > 32768 && keys >= bc * 16;
  auto mask_opt_buffer = query;
  if (mask_opt) {
    const auto words = (keys + 16 * bc - 1) / (16 * bc);
    const auto row_tiles = (rows + br - 1) / br;
    const std::array<std::int64_t, 1> mask_size{std::int64_t(words) *
                                                row_tiles};
    mask_opt_buffer = add(c::DataType::kInt32, mask_size);
    const std::array<v::SpecializationConstant, 4> mask_spec{
        {{0, 128}, {1, 128 / subgroup}, {2, br}, {3, bc}}};
    append(spirv::mask_opt, 2, {mask, mask_opt_buffer},
           bytes(std::array<std::uint32_t, 9>{
               keys, rows, 1, keys, keys * rows, keys * rows, words,
               words * row_tiles, words * row_tiles}),
           {words, row_tiles, 1}, mask_spec);
  }
  // Original F32 flash uses vec4 blocks, so nb?1 is in 16-byte blocks.
  const auto kv_size = quantized ? 34u : (half ? 2u : 16u);
  const auto query_strides = state->values[query].description.byte_strides();
  const auto cache_strides = kd.byte_strides();
  const auto value_strides = vd.byte_strides();
  std::array<std::uint32_t, 32> push{
      effective_rows,
      keys,
      heads,
      rows,
      batch,
      heads,
      batch,
      kv_heads,
      batch,
      kv_heads,
      batch,
      causal ? rows : 0u,
      1,
      1,
      std::uint32_t(query_strides[1] / 4),
      std::uint32_t(query_strides[2]),
      std::uint32_t(query_strides[0]),
      std::uint32_t(cache_strides[1] / kv_size),
      std::uint32_t(cache_strides[2]),
      std::uint32_t(cache_strides[0]),
      std::uint32_t(value_strides[1] / kv_size),
      std::uint32_t(value_strides[2]),
      std::uint32_t(value_strides[0]),
      std::bit_cast<std::uint32_t>(scale),
      0,
      0,
      std::bit_floor(heads),
      std::bit_cast<std::uint32_t>(1.F),
      std::bit_cast<std::uint32_t>(1.F),
      group_ratio,
      split_kv,
      split};
  std::array<v::SpecializationConstant, 16> spec{};
  const std::array<std::uint32_t, 16> values{workgroup,
                                             br,
                                             bc,
                                             depth,
                                             value_depth,
                                             keys % bc != 0 ? 1u : 0u,
                                             ds,
                                             row_split,
                                             subgroup,
                                             staged,
                                             (causal ? 2u : 0u) |
                                                 (mask_opt ? 1u : 0u),
                                             0,
                                             quantized ? 8u : (half ? 1u : 0u),
                                             quantized ? 8u : (half ? 1u : 0u),
                                             kv_size,
                                             kv_size};
  for (std::uint32_t i = 0; i < spec.size(); ++i)
    spec[i] = {i, values[i]};
  auto flash_output = output;
  if (split > 1) {
    const std::array<std::int64_t, 1> scratch{std::int64_t(value_depth + 2) * heads *
                                              rows * batch * split};
    flash_output = add(c::DataType::kFloat32, scratch);
  }
  const auto x = group_ratio > 1 ? rows * split : tr * split;
  append(spirv::flash, 7,
         {query, 1, 2, mask, query, flash_output, mask_opt_buffer}, bytes(push),
         {x, y, batch}, spec);
  if (split > 1) {
    const std::array<v::SpecializationConstant, 1> reduce_spec{{{0, subgroup}}};
    append(spirv::split_reduce, 3, {flash_output, query, output},
           bytes(std::array<std::uint32_t, 6>{value_depth, heads, rows, batch, split,
                                              0}),
           {heads, (value_depth + subgroup - 1) / subgroup, rows * batch},
           reduce_spec);
  }
  if (half)
    convert(output, 3, spirv::half);
  for (std::uint32_t value = 0; value < state->values.size(); ++value)
    state->aliases.push_back({value, 0});
  check(d::plan_storage(*state));
  return d::Access::program(std::move(state));
}
} // namespace hyle::bench::resident
