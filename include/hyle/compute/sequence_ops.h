#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/compute/operation.h"

namespace hyle::compute::sequence_ops {
namespace detail {

inline bool dense(const TensorDesc &description, std::size_t rank) noexcept {
  return description.valid() && description.is_plain() &&
         description.is_dense() && description.rank() == rank;
}

inline bool floating(DataType type) noexcept {
  return type == DataType::kFloat32 || type == DataType::kFloat16 ||
         type == DataType::kBFloat16;
}

inline bool offsets(const TensorDesc &description, std::int64_t batches) noexcept {
  return dense(description, 1) && description.data_type() == DataType::kInt32 &&
         batches > 0 && batches < (std::numeric_limits<std::int64_t>::max)() &&
         description.dimensions()[0] == batches + 1;
}

inline bool rows(std::int64_t count) noexcept {
  return count >= 0 && count <= (std::numeric_limits<std::int32_t>::max)();
}

} // namespace detail

// Packed, independent sequences with explicit state. Inputs are Q, K, V,
// log-decay, beta, initial state and I32 cumulative sequence offsets:
//   Q/K [R,Hk,Dk], V [R,Hv,Dv], g [R,Hv,1 or Dv], beta [R,Hv,1],
//   initial [B,Hv,Dv,Dk], offsets [B+1]. Q/K/V share F16/BF16/F32;
//   gates and state are F32. Hv is a multiple of Hk, with key head h % Hk.
// R is the physical row extent. Offsets start at zero, are nondecreasing and
// end at A <= R. Providers validate ALL offsets and snapshot lengths before
// writing either output. Only [0,A) payload is read; y[A:R] is positive zero.
// Recurrence, F32 accumulation and output_scale follow GatedDeltaNet.
// K=1 returns each sequence's final state; an empty sequence copies its initial
// state bitwise. K>1 requires EVERY sequence length Li >= K and returns the
// chronological post-states S(Li-K+1)..S(Li), not physical-row snapshots.
struct PackedGatedDeltaNet final {
  static constexpr OperationId operation_id{0x48594c4553455131ULL,
                                             0x504b47444e000001ULL};
  static constexpr std::string_view operation_name = "packed-gated-delta-net";
  std::uint32_t snapshot_count = 1;
  std::optional<float> output_scale{};

  constexpr OperationTraits operation_traits() const noexcept {
    return {{7, 7}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 7 || outputs.size() != 2)
      return Status::InvalidArgument("PackedGatedDeltaNet requires seven inputs and two outputs");
    for (std::size_t i = 0; i != 5; ++i)
      if (!detail::dense(inputs[i], 3))
        return Status::InvalidArgument("PackedGatedDeltaNet requires dense packed rank-three inputs");
    if (!detail::dense(inputs[5], 4))
      return Status::InvalidArgument("PackedGatedDeltaNet requires dense initial state [B,Hv,Dv,Dk]");
    const auto q = inputs[0].dimensions(), k = inputs[1].dimensions();
    const auto v = inputs[2].dimensions(), g = inputs[3].dimensions();
    const auto beta = inputs[4].dimensions(), state = inputs[5].dimensions();
    const auto type = inputs[0].data_type();
    if (!detail::floating(type) || inputs[1].data_type() != type ||
        inputs[2].data_type() != type ||
        inputs[3].data_type() != DataType::kFloat32 ||
        inputs[4].data_type() != DataType::kFloat32 ||
        inputs[5].data_type() != DataType::kFloat32)
      return Status::InvalidArgument("PackedGatedDeltaNet requires matching floating QKV and F32 gates/state");
    if (!detail::rows(q[0]) || q[1] <= 0 || q[2] <= 0 ||
        k[0] != q[0] || k[1] != q[1] || k[2] != q[2] ||
        v[0] != q[0] || v[1] <= 0 || v[2] <= 0 || v[1] % q[1] != 0 ||
        g[0] != q[0] || g[1] != v[1] || (g[2] != 1 && g[2] != v[2]) ||
        beta[0] != q[0] || beta[1] != v[1] || beta[2] != 1 ||
        state[0] <= 0 || state[1] != v[1] || state[2] != v[2] || state[3] != q[2] ||
        !detail::offsets(inputs[6], state[0]))
      return Status::InvalidArgument("PackedGatedDeltaNet packed shapes, state or offsets differ");
    if (snapshot_count == 0 ||
        (snapshot_count > 1 && std::uint64_t(snapshot_count) > std::uint64_t(q[0])) ||
        (output_scale && !std::isfinite(*output_scale)))
      return Status::InvalidArgument("PackedGatedDeltaNet snapshot count or output scale is invalid");
    TensorDesc state_output = inputs[5];
    if (snapshot_count > 1) {
      auto description = TensorDesc::dense(DataType::kFloat32,
          std::array{state[0], std::int64_t(snapshot_count), state[1], state[2], state[3]});
      if (!description) return std::move(description).error();
      state_output = *description;
    }
    outputs[0] = inputs[2];
    outputs[1] = state_output;
    return Status::Ok();
  }
};

inline Operation packed_gated_delta_net(std::uint32_t snapshot_count = 1,
                                        std::optional<float> output_scale = {}) noexcept {
  return Operation::custom(PackedGatedDeltaNet{snapshot_count, output_scale});
}

// The same packed recurrence, with capacity for every real prefix instead of
// the final K states. Inputs and y follow PackedGatedDeltaNet. Each actual
// length Li must be <= max_prefixes; zero-length sequences are permitted.
// States are always [B,max_prefixes,Hv,Dv,Dk], including capacity one:
//   states[b,0:Li] = S1..S(Li), states[b,Li:max_prefixes] = positive zero.
// Offsets are the sole source of validity. Unused slots are not additional
// recurrence steps and must never be published as consumed prefixes. An empty
// member has no valid prefix; its initial state remains available at input 5.
// All offsets and length bounds are validated before either output is written.
struct PackedGatedDeltaNetPrefixes final {
  static constexpr OperationId operation_id{0x48594c4553455131ULL,
                                             0x504b47444e000002ULL};
  static constexpr std::string_view operation_name = "packed-gated-delta-net-prefixes";
  std::uint32_t max_prefixes = 1;
  std::optional<float> output_scale{};

  constexpr OperationTraits operation_traits() const noexcept {
    return PackedGatedDeltaNet{}.operation_traits();
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (!max_prefixes || outputs.size() != 2)
      return Status::InvalidArgument("PackedGatedDeltaNetPrefixes requires a positive prefix capacity and two outputs");
    std::array<TensorDesc, 2> inferred;
    HYLE_RETURN_IF_ERROR((PackedGatedDeltaNet{1, output_scale}.infer_output_descriptions(inputs, inferred)));
    const auto state = inputs[5].dimensions();
    HYLE_ASSIGN_OR_RETURN(auto prefixes, TensorDesc::dense(DataType::kFloat32,
        std::array{state[0], std::int64_t(max_prefixes), state[1], state[2], state[3]}));
    outputs[0] = inferred[0]; outputs[1] = std::move(prefixes);
    return Status::Ok();
  }
};

inline Operation packed_gated_delta_net_prefixes(std::uint32_t max_prefixes,
                                                 std::optional<float> output_scale = {}) noexcept {
  return Operation::custom(PackedGatedDeltaNetPrefixes{max_prefixes, output_scale});
}

// Packed depthwise causal convolution with an independent history per sequence.
// Inputs: x [R,C], weights [C,K], history [B,K-1,C], I32 offsets [B+1].
// The floating inputs share F16/BF16/F32. Outputs: y [R,C] and next_history
// [B,K-1,C]. A sequence consumes only its [offset[i],offset[i+1]) rows;
// next_history is the last K-1 rows of (history || those real rows), including
// an exact bitwise history copy for empty sequences. K=1 has empty history.
// Offsets have the same whole-input validation and positive-zero y tail as
// PackedGatedDeltaNet. No padded row is read or appended to history.
struct PackedCausalConv1d final {
  static constexpr OperationId operation_id{0x48594c4553455131ULL,
                                             0x504b434e56000001ULL};
  static constexpr std::string_view operation_name = "packed-causal-conv1d";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{4, 4}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 4 || outputs.size() != 2 ||
        !detail::dense(inputs[0], 2) || !detail::dense(inputs[1], 2) ||
        !detail::dense(inputs[2], 3))
      return Status::InvalidArgument("PackedCausalConv1d requires x, weights, history and offsets");
    const auto x = inputs[0].dimensions(), weight = inputs[1].dimensions();
    const auto history = inputs[2].dimensions();
    const auto type = inputs[0].data_type();
    if (!detail::floating(type) || inputs[1].data_type() != type || inputs[2].data_type() != type)
      return Status::InvalidArgument("PackedCausalConv1d requires matching F16/BF16/F32 inputs");
    if (!detail::rows(x[0]) || x[1] <= 0 || weight[0] != x[1] || weight[1] <= 0 ||
        history[0] <= 0 || history[1] != weight[1] - 1 || history[2] != x[1] ||
        !detail::offsets(inputs[3], history[0]))
      return Status::InvalidArgument("PackedCausalConv1d packed shapes, history or offsets differ");
    outputs[0] = inputs[0];
    outputs[1] = inputs[2];
    return Status::Ok();
  }
};

inline Operation packed_causal_conv1d() noexcept {
  return Operation::custom(PackedCausalConv1d{});
}

} // namespace hyle::compute::sequence_ops
