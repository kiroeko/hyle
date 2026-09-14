#include "custom_operation_fixture.h"

#include <span>
#include <string_view>

namespace hyle::compute::test {
namespace {

struct ProducerIdentity final {
  static constexpr OperationId operation_id{0x48594c4554455354ULL,
                                             0x435553544f4d0001ULL};
  static constexpr std::string_view operation_name = "shared-test-identity";

  std::uint32_t tag = 0;

  OperationTraits operation_traits() const noexcept {
    return {{1, 1},
            1,
            AliasPolicy::kMayAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  }

  Status infer_output_descriptions(
      std::span<const TensorDesc> inputs,
      std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1 || tag != 17)
      return Status::InvalidArgument("invalid shared custom descriptor");
    outputs[0] = inputs[0];
    return Status::Ok();
  }
};

} // namespace

Operation make_shared_custom_operation(std::uint32_t tag) noexcept {
  return Operation::custom(ProducerIdentity{tag});
}

} // namespace hyle::compute::test

