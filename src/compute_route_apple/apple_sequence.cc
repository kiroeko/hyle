#include "apple_sequence.h"
#include "apple_checks.h"
#include "../compute_shader_shared/ct_packed.h"
#include "../compute_shader_shared/nvfp4_f32.h"
#include "hyle/compute/sequence_ops.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <string>
#include <tuple>
#include <mlx/fast.h>
#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;

bool qualifies_short_gdn(const Operation &operation,
                         std::span<const TensorDesc> inputs) noexcept {
  if (operation.get_if<GatedDeltaNet>() == nullptr)
    return false;
  std::array<TensorDesc, 2> outputs;
  if (!infer_output_descriptions(operation, inputs, outputs).ok())
    return false;
  const auto indexable = [](const TensorDesc &value) {
    const auto count = value.element_count();
    return value.is_plain() && value.is_dense() &&
           value.data_type() == DataType::kFloat32 && count && *count <= INT32_MAX;
  };
  if (!std::ranges::all_of(inputs, indexable) ||
      !std::ranges::all_of(outputs, indexable))
    return false;
  if (*outputs[0].element_count() + *outputs[1].element_count() > INT32_MAX)
    return false;
  const auto query = inputs[0].dimensions();
  const auto value = inputs[2].dimensions();
  // Specialize the small verification and decode shapes. Long recurrence
  // continues through the existing native program without another MLX trace.
  return query[1] <= 4 && query[3] == value[3] &&
         (query[3] == 32 || query[3] == 64 || query[3] == 128) &&
         value[2] % query[2] == 0;
}

Result<std::vector<mx::array>> lower_short_gdn(const GatedDeltaNet &operation,
    std::span<const mx::array> inputs, mx::Stream stream) {
  const int depth = inputs[0].shape(3);
  const int heads = inputs[2].shape(2);
  const int per_lane = depth / 32;
  const int snapshots = static_cast<int>(operation.snapshot_count);
  const int score_count = static_cast<int>(inputs[2].size());
  const int packed_count = score_count + snapshots * static_cast<int>(inputs[5].size());
  auto state_shape = inputs[5].shape();
  if (snapshots > 1)
    state_shape.insert(state_shape.begin() + 1, snapshots);

  // Retain ggml Metal's F32 lane and reduction order. The kernel runs in the
  // surrounding MLX encoder and writes the declared chronological state
  // directly, including all requested snapshots, behind one output owner.
  static const auto kernel = mx::fast::metal_kernel("hyle_short_dense_gdn",
      {"q", "k", "v", "gate", "beta", "initial", "scale"}, {"packed"},
      R"METAL(
    const uint batch = threadgroup_position_in_grid.z;
    const uint head = threadgroup_position_in_grid.y;
    const uint lane = thread_position_in_threadgroup.x;
    const uint channel = threadgroup_position_in_grid.x * KEYS_PER_LANE
        + thread_position_in_threadgroup.y;
    const uint state_index = (head * DEPTH + channel) * DEPTH + lane * KEYS_PER_LANE;
    float cells[KEYS_PER_LANE];
    for (uint i = 0; i < KEYS_PER_LANE; ++i)
      cells[i] = initial[batch * VALUE_HEADS * DEPTH * DEPTH + state_index + i];
    for (uint token = 0; token < TOKENS; ++token) {
      const uint head_index = (batch * TOKENS + token) * VALUE_HEADS + head;
      const uint key_index = ((batch * TOKENS + token) * KEY_HEADS + head % KEY_HEADS)
          * DEPTH + lane * KEYS_PER_LANE;
      const uint value_index = head_index * DEPTH + channel;
      const float scalar_decay = GATE_WIDTH == 1 ? exp(gate[head_index]) : 0.0f;
      float prediction = 0.0f;
      for (uint i = 0; i < KEYS_PER_LANE; ++i) {
        cells[i] *= GATE_WIDTH == 1 ? scalar_decay
            : exp(gate[head_index * GATE_WIDTH + lane * KEYS_PER_LANE + i]);
        prediction += cells[i] * k[key_index + i];
      }
      const float delta = (v[value_index] - simd_sum(prediction)) * beta[head_index];
      float mixed = 0.0f;
      for (uint i = 0; i < KEYS_PER_LANE; ++i) {
        cells[i] += k[key_index + i] * delta;
        mixed += cells[i] * q[key_index + i];
      }
      const float dot = simd_sum(mixed);
      if (lane == 0)
        packed[value_index] = dot * (EXPLICIT_SCALE ? scale : 1.0f / sqrt(float(DEPTH)));
      if (token + SNAPSHOTS >= TOKENS) {
        const uint snapshot = token + SNAPSHOTS - TOKENS;
        const uint offset = SCORES + state_index
            + (batch * SNAPSHOTS + snapshot) * VALUE_HEADS * DEPTH * DEPTH;
        for (uint i = 0; i < KEYS_PER_LANE; ++i)
          packed[offset + i] = cells[i];
      }
    }
  )METAL", "", true, false, mx::CompileOptions(static_cast<int>(mx::MathMode::Fast)));
  std::vector<mx::array> args(inputs.begin(), inputs.end());
  args.emplace_back(operation.output_scale.value_or(0.0f));
  // Each SIMD group owns one state row. The X grid contains (depth/per_lane)
  // threadgroups, each with 32 lanes and per_lane state rows.
  auto packed = kernel(args, {{packed_count}}, {mx::float32},
      {32 * depth / per_lane, heads * per_lane, inputs[0].shape(0)}, {32, per_lane, 1},
      {{"KEYS_PER_LANE", per_lane}, {"DEPTH", depth}, {"TOKENS", inputs[0].shape(1)},
       {"KEY_HEADS", inputs[0].shape(2)}, {"VALUE_HEADS", heads},
       {"GATE_WIDTH", inputs[3].shape(3)}, {"SNAPSHOTS", snapshots},
       {"SCORES", score_count},
       {"EXPLICIT_SCALE", operation.output_scale.has_value()}},
      std::nullopt, false, stream)[0];
  auto scores = mx::reshape(mx::slice(packed, {0}, {score_count}, stream),
                            inputs[2].shape(), stream);
  auto state = mx::reshape(mx::slice(packed, {score_count}, {packed_count}, stream),
                           state_shape, stream);
  return std::vector{std::move(scores), std::move(state)};
}

std::vector<mlx::core::array> lower_bfloat_conv_gdn_inputs(
    const mlx::core::array &samples, const mlx::core::array &weights,
    const mlx::core::array &query_epsilon, const mlx::core::array &key_epsilon,
    const mlx::core::Shape &query_shape, const mlx::core::Shape &value_shape,
    std::array<int,3> channel_offsets, mlx::core::Stream stream) {
  // Fuse convolution, BF16 SiLU and the two BF16 L2 chains for short rows.
  // Each SIMD group owns one head and uses the original four-value reduction.
  // The typed casts and exact BF16 multiply boundaries remain explicit.
  static const std::string math = std::string(R"METAL(
inline uint hyle_mul_hi(uint a,uint b) { return uint((ulong(a)*ulong(b))>>32); }
inline uint hyle_clz(uint a) { return clz(a); }
)METAL")+compute::shader::kNvfp4F32+compute::shader::kCtPacked+R"METAL(
inline float hyle_round_bf(float x) { return float(bfloat16_t(x)); }
inline float hyle_bfloat_multiply(float a,float b) {
  const uint left=as_type<uint>(a), right=as_type<uint>(b);
  const uint l=left&0x7fffffffu, r=right&0x7fffffffu;
  if(l>=0x00800000u && l<0x7f800000u && r>=0x00800000u && r<0x7f800000u) {
    const uint product=as_type<uint>(a*b);
    const uint magnitude=product&0x7fffffffu;
    if(magnitude>=0x00800000u && magnitude<0x7f800000u)
      return as_type<float>(hyle_ct_narrow(product,2)<<16u);
  }
  return as_type<float>(hyle_ct_narrow(hyle_f32_mul(left,right),2)<<16u);
}
)METAL";
  static const auto kernel = mx::fast::metal_kernel("hyle_short_causal_conv_gdn_inputs",
      {"samples","weight","qe","ke"},{"out_q","out_k","out_v"},R"METAL(
  const uint lane=thread_index_in_simdgroup, region=threadgroup_position_in_grid.y;
  const uint row=threadgroup_position_in_grid.x, token=row/MAX_HEADS, head=row%MAX_HEADS;
  const uint heads=region==2?V_HEADS:Q_HEADS;
  if(head>=heads) return;
  const uint channels=region==0?Q_START:region==1?K_START:V_START;
  float x[4], total=0.0f;
  for(uint i=0;i<4;++i){
    const uint c=channels+head*128+lane*4+i;
    float convolution=0.0f;
    for(uint tap=0;tap<4;++tap)
      convolution+=float(samples[(token+tap)*samples_strides[1]+c*samples_strides[2]])*
          weight[c*weight_strides[0]+tap*weight_strides[1]];
    const float a=hyle_round_bf(convolution);
    const float s=1.0f/(1.0f+precise::exp(abs(a)));
    x[i]=hyle_round_bf(a==-INFINITY?0.0f:a*(a<0?s:1.0f-s));
    if(region!=2) total=hyle_bfloat_multiply(x[i],x[i])+total;
  }
  float factor=1.0f;
  if(region!=2){
    const float e=region==0?float(qe[0]):float(ke[0]);
    const float sum=hyle_round_bf(hyle_round_bf(simd_sum(total))+e);
    factor=hyle_round_bf(precise::rsqrt(sum));
  }
  for(uint i=0;i<4;++i){
    const uint index=(token*heads+head)*128+lane*4+i;
    if(region==0)out_q[index]=hyle_bfloat_multiply(x[i],factor);
    else if(region==1)out_k[index]=hyle_bfloat_multiply(x[i],factor);
    else out_v[index]=x[i];
  }
)METAL",math,false,false,mx::CompileOptions{});
  const int head_slots = std::max(query_shape[2], value_shape[2]);
  return kernel({samples, weights, query_epsilon, key_epsilon},
      {query_shape, query_shape, value_shape}, {mx::float32, mx::float32, mx::float32},
      {head_slots * query_shape[1] * 32, 3, 1}, {32, 1, 1},
      {{"MAX_HEADS", head_slots}, {"Q_HEADS", query_shape[2]}, {"V_HEADS", value_shape[2]},
       {"Q_START", channel_offsets[0]}, {"K_START", channel_offsets[1]},
       {"V_START", channel_offsets[2]}}, std::nullopt, false, stream);
}

Status qualify_packed_conv(const Operation &operation,
    std::span<const TensorDesc> inputs) noexcept {
  if (!operation.custom_as<sequence_ops::PackedCausalConv1d>())
    return Status::InvalidArgument("Apple packed convolution operation differs");
  std::array<TensorDesc, 2> outputs;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, outputs));
  for (const auto &description : inputs)
    if (*description.element_count() > INT32_MAX - 63)
      return Status::Unimplemented("Apple packed convolution input indexing limit");
  for (const auto &description : outputs)
    if (*description.element_count() > INT32_MAX - 63)
      return Status::Unimplemented("Apple packed convolution output indexing limit");
  return Status::Ok();
}

Result<std::vector<mx::array>> lower_packed_conv(const Operation &operation,
    std::span<const mx::array> inputs, mx::Stream stream) {
  if (!operation.custom_as<sequence_ops::PackedCausalConv1d>() || inputs.size() != 4)
    return std::unexpected(Status::InvalidArgument("Apple packed convolution ports differ"));
  const int rows = inputs[0].shape(0), channels = inputs[0].shape(1);
  const int taps = inputs[1].shape(1), batches = inputs[2].shape(0);
  static const auto check = mx::fast::metal_kernel("hyle_packed_conv_offsets",
      {"offsets"}, {"status"}, R"METAL(
    int previous = offsets[0];
    bool valid = previous == 0;
    for (int sequence = 0; valid && sequence < BATCHES; ++sequence) {
      int next = offsets[sequence + 1];
      valid = next >= previous && next >= 0 && next <= ROWS;
      previous = next;
    }
    status[0] = valid ? 0 : INVALID;
  )METAL", "", true, false, mx::CompileOptions{});
  auto status = check({inputs[3]}, {{1}}, {mx::int32}, {1,1,1}, {1,1,1},
      {{"BATCHES", batches}, {"ROWS", rows}, {"INVALID", kInvalidNumericInput}},
      std::nullopt, false, stream)[0];
  static const auto kernel = mx::fast::metal_kernel("hyle_packed_conv",
      {"x", "w", "history", "offsets", "status", "x_bits", "history_bits"},
      {"y", "next_bits"}, R"METAL(
    if (status[0]) return;
    uint channel = thread_position_in_grid.x, sequence = thread_position_in_grid.y;
    if (channel >= CHANNELS || sequence >= BATCHES) return;
    uint first = uint(offsets[sequence]), last = uint(offsets[sequence + 1]);
    uint length = last - first;
    if (sequence + 1 == BATCHES)
      for (uint row = last; row < ROWS; ++row) y[row * CHANNELS + channel] = OutType(0);
    for (uint row = 0; row < length; ++row) {
      float sum = 0;
      for (uint tap = 0; tap < TAPS; ++tap) {
        uint position = row + tap;
        float sample = position < TAPS - 1
            ? float(history[(sequence * (TAPS - 1) + position) * CHANNELS + channel])
            : float(x[(first + position - (TAPS - 1)) * CHANNELS + channel]);
        sum = fma(sample, float(w[channel * TAPS + tap]), sum);
      }
      y[(first + row) * CHANNELS + channel] = OutType(sum);
    }
    for (uint row = 0; row < TAPS - 1; ++row) {
      uint position = length + row;
      next_bits[(sequence * (TAPS - 1) + row) * CHANNELS + channel] = position < TAPS - 1
          ? history_bits[(sequence * (TAPS - 1) + position) * CHANNELS + channel]
          : x_bits[(first + position - (TAPS - 1)) * CHANNELS + channel];
    }
  )METAL", "", true, false, mx::CompileOptions{});
  // Empty physical ports cannot be bound as Metal buffers. Only these unused
  // arguments/results use a one-element private carrier; the public shapes
  // remain empty and retain the same checked status dependency.
  auto x = rows ? inputs[0] : mx::zeros({1}, inputs[0].dtype(), stream);
  auto history = taps > 1 ? inputs[2] : mx::zeros({1}, inputs[2].dtype(), stream);
  const auto bits = inputs[0].itemsize() == 4 ? mx::uint32 : mx::uint16;
  const std::vector args{x, inputs[1], history, inputs[3], status,
      mx::view(x, bits, stream), mx::view(history, bits, stream)};
  auto outputs = kernel(args,
      {rows ? inputs[0].shape() : mx::Shape{1}, taps > 1 ? inputs[2].shape() : mx::Shape{1}},
      {inputs[0].dtype(), bits}, {((channels + 63) / 64) * 64, batches, 1}, {64,1,1},
      {{"OutType", inputs[0].dtype()}, {"ROWS", rows}, {"CHANNELS", channels},
       {"TAPS", taps}, {"BATCHES", batches}}, std::nullopt, false, stream);
  outputs[0] = rows ? outputs[0] : mx::zeros(inputs[0].shape(), inputs[0].dtype(), stream);
  outputs[1] = taps > 1 ? mx::view(outputs[1], inputs[2].dtype(), stream)
      : mx::zeros(inputs[2].shape(), inputs[2].dtype(), stream);
  for (auto &output : outputs) {
    HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(std::move(output), status, stream));
    output = std::move(checked);
  }
  return outputs;
}

Status qualify_packed_gdn(const Operation &operation,
    std::span<const TensorDesc> inputs) noexcept {
  if (!operation.custom_as<sequence_ops::PackedGatedDeltaNet>() &&
      !operation.custom_as<sequence_ops::PackedGatedDeltaNetPrefixes>())
    return Status::InvalidArgument("Apple packed GDN operation differs");
  std::array<TensorDesc, 2> outputs;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, outputs));
  if (inputs[0].dimensions()[2] > 256)
    return Status::Unimplemented("Apple packed GDN private state depth exceeds 256");
  for (const auto &description : inputs)
    if (*description.element_count() > INT32_MAX - 63)
      return Status::Unimplemented("Apple packed GDN input indexing limit");
  for (const auto &description : outputs)
    if (*description.element_count() > INT32_MAX - 63)
      return Status::Unimplemented("Apple packed GDN output indexing limit");
  return Status::Ok();
}

Result<std::vector<mx::array>> lower_packed_gdn(const Operation &operation,
    std::span<const mx::array> inputs, mx::Stream stream) {
  const auto descriptor = operation.custom_as<sequence_ops::PackedGatedDeltaNet>();
  const auto prefixes = operation.custom_as<sequence_ops::PackedGatedDeltaNetPrefixes>();
  if ((!descriptor && !prefixes) || inputs.size() != 7)
    return std::unexpected(Status::InvalidArgument("Apple packed GDN ports differ"));
  const int rows = inputs[0].shape(0), hk = inputs[0].shape(1), dk = inputs[0].shape(2);
  const int hv = inputs[2].shape(1), dv = inputs[2].shape(2), batches = inputs[5].shape(0);
  const int gates = inputs[3].shape(2), snapshots = static_cast<int>(prefixes
      ? prefixes->max_prefixes : descriptor->snapshot_count);
  auto state_shape = inputs[5].shape();
  if (prefixes || snapshots > 1) state_shape.insert(state_shape.begin() + 1, snapshots);
  static const auto check = mx::fast::metal_kernel("hyle_packed_gdn_offsets",
      {"offsets"}, {"status"}, R"METAL(
    int previous = offsets[0];
    bool valid = previous == 0;
    for (int sequence = 0; valid && sequence < BATCHES; ++sequence) {
      int next = offsets[sequence + 1];
      valid = next >= previous && next >= 0 && next <= ROWS;
      if (valid && PREFIXES) valid = next - previous <= SNAPSHOTS;
      else if (valid && SNAPSHOTS > 1) valid = next - previous >= SNAPSHOTS;
      previous = next;
    }
    status[0] = valid ? 0 : INVALID;
  )METAL", "", true, false, mx::CompileOptions{});
  auto status = check({inputs[6]}, {{1}}, {mx::int32}, {1,1,1}, {1,1,1},
      {{"BATCHES", batches}, {"ROWS", rows}, {"SNAPSHOTS", snapshots}, {"PREFIXES", int(bool(prefixes))},
       {"INVALID", kInvalidNumericInput}},
      std::nullopt, false, stream)[0];
  if (rows == 0) {
    if (prefixes) {
      HYLE_ASSIGN_OR_RETURN(auto y, with_device_status(
          mx::zeros(inputs[2].shape(), inputs[2].dtype(), stream), status, stream));
      HYLE_ASSIGN_OR_RETURN(auto states, with_device_status(
          mx::zeros(state_shape, mx::float32, stream), status, stream));
      return std::vector{std::move(y), std::move(states)};
    }
    // Metal forbids binding empty Q/K/V buffers, even when a shader would not
    // read them. Prepare a state-only copy with the same checked dependency.
    static const auto empty = mx::fast::metal_kernel("hyle_packed_gdn_empty",
        {"initial", "status"}, {"states"}, R"METAL(
      uint index = thread_position_in_grid.x;
      if (!status[0] && index < COUNT) states[index] = initial[index];
    )METAL", "", true, false, mx::CompileOptions{});
    const int count = static_cast<int>(inputs[5].size());
    auto copied = empty({mx::view(inputs[5], mx::uint32, stream), status},
        {inputs[5].shape()}, {mx::uint32}, {((count+63)/64)*64,1,1}, {64,1,1},
        {{"COUNT", count}}, std::nullopt, false, stream)[0];
    HYLE_ASSIGN_OR_RETURN(auto y, with_device_status(
        mx::zeros(inputs[2].shape(), inputs[2].dtype(), stream), status, stream));
    HYLE_ASSIGN_OR_RETURN(auto state, with_device_status(
        mx::view(copied, mx::float32, stream), status, stream));
    return std::vector{std::move(y), std::move(state)};
  }
  static const auto kernel = mx::fast::metal_kernel("hyle_packed_gdn",
      {"q", "k", "v", "g", "beta", "initial", "offsets", "status", "scale"},
      {"y", "states"}, R"METAL(
    // A device dependency orders the whole-input check before either write.
    if (status[0]) return;
    uint lane = thread_position_in_grid.x, sequence = thread_position_in_grid.y;
    if (lane >= HV * DV || sequence >= BATCHES) return;
    uint head = lane / DV, channel = lane % DV;
    uint first = uint(offsets[sequence]), last = uint(offsets[sequence + 1]);
    uint state_row = (sequence * HV * DV + lane) * DK;
    if (sequence + 1 == BATCHES)
      for (uint row = last; row < ROWS; ++row) y[row * HV * DV + lane] = 0;
    if (PREFIXES)
      for (uint snapshot = last - first; snapshot < SNAPSHOTS; ++snapshot) {
        uint output_row = ((sequence * SNAPSHOTS + snapshot) * HV * DV + lane) * DK;
        for (uint key = 0; key < DK; ++key) states[output_row + key] = 0;
      }
    if (first == last) {
      if (!PREFIXES)
        for (uint key = 0; key < DK; ++key) states[state_row + key] = initial[state_row + key];
      return;
    }
    float carry[DK];
    for (uint key = 0; key < DK; ++key) carry[key] = as_type<float>(initial[state_row + key]);
    for (uint row = first; row < last; ++row) {
      uint qk = (row * HK + head % HK) * DK;
      uint value = row * HV * DV + lane;
      uint gate = (row * HV + head) * GATES + (GATES == 1 ? 0 : channel);
      float decay = exp(g[gate]), b = beta[row * HV + head];
      float prediction = 0;
      for (uint key = 0; key < DK; ++key) {
        carry[key] *= decay;
        prediction = fma(carry[key], float(k[qk + key]), prediction);
      }
      float delta = b * (float(v[value]) - prediction), attended = 0;
      for (uint key = 0; key < DK; ++key) {
        carry[key] = fma(delta, float(k[qk + key]), carry[key]);
        attended = fma(carry[key], float(q[qk + key]), attended);
      }
      y[value] = OutType(attended * scale);
      if (PREFIXES || row + SNAPSHOTS >= last) {
        uint snapshot = PREFIXES ? row - first : row + SNAPSHOTS - last;
        uint output_row = ((sequence * SNAPSHOTS + snapshot) * HV * DV + lane) * DK;
        for (uint key = 0; key < DK; ++key) states[output_row + key] = as_type<uint>(carry[key]);
      }
    }
  )METAL", "", true, false, mx::CompileOptions{});
  static const auto cooperative = mx::fast::metal_kernel("hyle_packed_gdn_cooperative",
      {"q", "k", "v", "g", "beta", "initial", "offsets", "status", "scale"},
      {"y", "states"}, R"METAL(
    // The offset check is uniform across the workgroup.
    if (status[0]) return;
    constexpr uint vectors = 32;
    constexpr uint slots = vectors / LANES;
    const uint thread_index = thread_position_in_threadgroup.x;
    const uint simd_lane = thread_index_in_simdgroup;
    const uint row_lane = simd_lane % LANES;
    const uint channel = (threadgroup_position_in_grid.x * 128 + thread_index) / LANES;
    const uint head = threadgroup_position_in_grid.y;
    const uint length = uint(offsets[1]);
    const uint state = (head * DV + channel) * vectors;
    const auto *queries = reinterpret_cast<const device packed_float4 *>(q);
    const auto *keys = reinterpret_cast<const device packed_float4 *>(k);
    const auto *previous = reinterpret_cast<const device packed_uint4 *>(initial);
    auto *next = reinterpret_cast<device packed_uint4 *>(states);
    float4 carry[slots];
    #pragma unroll
    for (uint i = 0; i < slots; ++i)
      carry[i] = as_type<float4>(uint4(previous[state + row_lane + i * LANES]));

    // A workgroup shares the head's key and query rows across its value rows.
    threadgroup float4 prefetched_keys[TILE][vectors];
    threadgroup float4 prefetched_queries[TILE][vectors];
    for (uint begin = 0; begin < length; begin += TILE) {
      const uint count = min(uint(TILE), length - begin);
      for (uint index = thread_index; index < count * vectors; index += 128) {
        const uint token = begin + index / vectors;
        const uint vector_index = (token * HK + head % HK) * vectors + index % vectors;
        prefetched_keys[index / vectors][index % vectors] = float4(keys[vector_index]);
        prefetched_queries[index / vectors][index % vectors] = float4(queries[vector_index]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = 0; i < count; ++i) {
        const uint token = begin + i;
        const uint value_index = (token * HV + head) * DV + channel;
        const float decay = exp(g[token * HV + head]);
        float4 key[slots];
        float prediction = 0.0f;
        #pragma unroll
        for (uint j = 0; j < slots; ++j) {
          const uint index = row_lane + j * LANES;
          key[j] = prefetched_keys[i][index];
          carry[j] *= decay;
          prediction = fma(carry[j].x, key[j].x, prediction);
          prediction = fma(carry[j].y, key[j].y, prediction);
          prediction = fma(carry[j].z, key[j].z, prediction);
          prediction = fma(carry[j].w, key[j].w, prediction);
        }
        #pragma unroll
        for (uint stride = LANES / 2; stride > 0; stride /= 2)
          prediction += simd_shuffle_down(prediction, stride);
        prediction = simd_shuffle(prediction, simd_lane - row_lane);
        float delta = beta[token * HV + head] * (v[value_index] - prediction);
        float attended = 0.0f;
        #pragma unroll
        for (uint j = 0; j < slots; ++j) {
          const uint index = row_lane + j * LANES;
          float4 query = prefetched_queries[i][index];
          carry[j] = fma(float4(delta), key[j], carry[j]);
          attended = fma(carry[j].x, query.x, attended);
          attended = fma(carry[j].y, query.y, attended);
          attended = fma(carry[j].z, query.z, attended);
          attended = fma(carry[j].w, query.w, attended);
        }
        #pragma unroll
        for (uint stride = LANES / 2; stride > 0; stride /= 2)
          attended += simd_shuffle_down(attended, stride);
        if (row_lane == 0) y[value_index] = OutType(attended * scale);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row_lane == 0)
      for (uint row = length; row < ROWS; ++row)
        y[(row * HV + head) * DV + channel] = OutType(0);
    #pragma unroll
    for (uint i = 0; i < slots; ++i)
      next[state + row_lane + i * LANES] = packed_uint4(as_type<uint4>(carry[i]));
  )METAL", "", true, false, mx::CompileOptions{});
  constexpr int lanes = 8, tile = 12, threads = 128;
  const bool use_cooperative = !prefixes && snapshots == 1 && batches == 1 && gates == 1 &&
      dk == 128 && dv % (threads / lanes) == 0 && rows >= tile &&
      inputs[0].dtype() == mx::float32 && inputs[1].dtype() == mx::float32;
  const float factor = (prefixes ? prefixes->output_scale : descriptor->output_scale)
      .value_or(1.F / std::sqrt(float(dk)));
  std::vector args(inputs.begin(), inputs.end());
  args[5] = mx::view(args[5], mx::uint32, stream);
  args.push_back(status);
  args.push_back(mx::array(factor));
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> parameters{
      {"OutType", inputs[2].dtype()}, {"ROWS", rows}, {"HK", hk}, {"HV", hv}, {"DK", dk}, {"DV", dv},
      {"BATCHES", batches}, {"GATES", gates}, {"SNAPSHOTS", snapshots}, {"PREFIXES", int(bool(prefixes))}};
  if (use_cooperative) {
    parameters.emplace_back("LANES", lanes);
    parameters.emplace_back("TILE", tile);
  }
  auto outputs = (use_cooperative ? cooperative : kernel)(args,
      {inputs[2].shape(), state_shape}, {inputs[2].dtype(), mx::uint32},
      use_cooperative ? std::tuple{dv * lanes, hv, 1} : std::tuple{((hv*dv + 63)/64)*64, batches, 1},
      use_cooperative ? std::tuple{threads, 1, 1} : std::tuple{64, 1, 1},
      parameters, std::nullopt, false, stream);
  outputs[1] = mx::view(outputs[1], mx::float32, stream);
  for (auto &output : outputs) {
    HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(std::move(output), status, stream));
    output = std::move(checked);
  }
  return outputs;
}
} // namespace hyle::compute::apple::detail
