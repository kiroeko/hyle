#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/compute/operation.h"

namespace hyle::compute::sampling_ops {

// Pure, counter-addressed Philox4x32-10 words. Input is dense U32[R,4]:
// key low/high, draw low/high. Output is dense U32[R,word_count]. Word j uses
// counter {floor(j/4) low/high, draw low/high} and lane j%4. Raw words match
// across qualified backends; prefixes do not depend on word_count or batch
// order. Equal input identities deliberately reproduce the same stream.
//
// No generator state advances at prepare, submit, failure or completion.
// The caller owns draw uniqueness, advance and wrap policy. Inputs remain
// borrowed until actual retirement. Zero rows/count are valid; word_count
// must fit I32. Native resource limits can be narrower than this schema.
struct RandomBits final {
  std::uint32_t word_count = 4;
  static constexpr OperationId operation_id{0x48594c4553414d50ULL,
                                            0x52414e4442495401ULL};
  static constexpr std::string_view operation_name = "counter-random-bits";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{1, 1}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kExact, DataType::kUnknown, true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs,
      std::span<TensorDesc> outputs) const noexcept;
};

// One categorical draw per row of dense F16/BF16/F32[R,V] log-weights, using
// dense U32[R,4] key/draw identities as above. V is positive and fits I32;
// zero rows are valid. -Inf is zero mass. NaN, +Inf or any all--Inf row fails
// through the original checked completion with InvalidArgument and no usable
// result. Inputs are read-only; outputs materialize without input aliasing.
//
// Outputs: I32[R] indices, and optionally F32[R] normalized selected log
// probabilities. Disabling the latter avoids its normalization work. This
// operation owns no temperature, filtering, token or request policy.
//
// The qualified native recipe repeats a fixed input/key/draw independently
// of row permutation or other requests. CPU and GPU may use different mature
// draw algorithms and native floating math; selected IDs need not match
// across backends. Only RandomBits has cross-backend bit equality. Native
// temporaries, checks and waits follow the selected route's normal contract;
// this is neither a capture nor a zero-allocation guarantee.
struct Categorical final {
  bool return_log_probability = false;
  static constexpr OperationId operation_id{0x48594c4553414d50ULL,
                                            0x43415445474f5201ULL};
  static constexpr std::string_view operation_name = "categorical";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, static_cast<std::uint8_t>(return_log_probability ? 2 : 1),
            AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs,
      std::span<TensorDesc> outputs) const noexcept;
};

static_assert(detail::InlineCustomOperationDescriptor<RandomBits>);
static_assert(detail::InlineCustomOperationDescriptor<Categorical>);

inline Operation random_bits(std::uint32_t word_count = 4) noexcept {
  return Operation::custom(RandomBits{word_count});
}
inline Operation categorical(bool return_log_probability = false) noexcept {
  return Operation::custom(Categorical{return_log_probability});
}

} // namespace hyle::compute::sampling_ops
