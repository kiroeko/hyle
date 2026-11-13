#include "apple_sequence.h"
#include "apple_checks.h"
#include "hyle/compute/sequence_ops.h"

#include <array>
#include <climits>
#include <cmath>
#include <mlx/fast.h>
#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;

Status qualify_packed_gdn(const Operation &operation,
    std::span<const TensorDesc> inputs) noexcept {
  if (!operation.custom_as<sequence_ops::PackedGatedDeltaNet>())
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
  if (!descriptor || inputs.size() != 7)
    return std::unexpected(Status::InvalidArgument("Apple packed GDN ports differ"));
  const int rows = inputs[0].shape(0), hk = inputs[0].shape(1), dk = inputs[0].shape(2);
  const int hv = inputs[2].shape(1), dv = inputs[2].shape(2), batches = inputs[5].shape(0);
  const int gates = inputs[3].shape(2), snapshots = static_cast<int>(descriptor->snapshot_count);
  static const auto check = mx::fast::metal_kernel("hyle_packed_gdn_offsets",
      {"offsets"}, {"status"}, R"METAL(
    int previous = offsets[0];
    bool valid = previous == 0;
    for (int sequence = 0; valid && sequence < BATCHES; ++sequence) {
      int next = offsets[sequence + 1];
      valid = next >= previous && next >= 0 && next <= ROWS;
      if (valid && SNAPSHOTS > 1) valid = next - previous >= SNAPSHOTS;
      previous = next;
    }
    status[0] = valid ? 0 : INVALID;
  )METAL", "", true, false, mx::CompileOptions{});
  auto status = check({inputs[6]}, {{1}}, {mx::int32}, {1,1,1}, {1,1,1},
      {{"BATCHES", batches}, {"ROWS", rows}, {"SNAPSHOTS", snapshots},
       {"INVALID", kInvalidNumericInput}},
      std::nullopt, false, stream)[0];
  if (rows == 0) {
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
    if (first == last) {
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
      if (row + SNAPSHOTS >= last) {
        uint snapshot = row + SNAPSHOTS - last;
        uint output_row = ((sequence * SNAPSHOTS + snapshot) * HV * DV + lane) * DK;
        for (uint key = 0; key < DK; ++key) states[output_row + key] = as_type<uint>(carry[key]);
      }
    }
  )METAL", "", true, false, mx::CompileOptions{});
  auto state_shape = inputs[5].shape();
  if (snapshots > 1) state_shape.insert(state_shape.begin() + 1, snapshots);
  const float factor = descriptor->output_scale.value_or(1.F / std::sqrt(float(dk)));
  std::vector args(inputs.begin(), inputs.end());
  args[5] = mx::view(args[5], mx::uint32, stream);
  args.push_back(status);
  args.push_back(mx::array(factor));
  auto outputs = kernel(args, {inputs[2].shape(), state_shape}, {inputs[2].dtype(), mx::uint32},
      {((hv*dv + 63)/64)*64, batches, 1}, {64,1,1},
      {{"OutType", inputs[2].dtype()}, {"ROWS", rows}, {"HK", hk}, {"HV", hv}, {"DK", dk}, {"DV", dv},
       {"BATCHES", batches}, {"GATES", gates}, {"SNAPSHOTS", snapshots}},
      std::nullopt, false, stream);
  outputs[1] = mx::view(outputs[1], mx::float32, stream);
  for (auto &output : outputs) {
    HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(std::move(output), status, stream));
    output = std::move(checked);
  }
  return outputs;
}
} // namespace hyle::compute::apple::detail
