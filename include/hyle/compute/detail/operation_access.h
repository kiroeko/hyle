#pragma once

#include <cstdint>
#include <limits>
#include "hyle/compute/provider.h"

namespace hyle::compute::detail {
// Cold proof from the actual selected implementation: this input is read only
// through checked indices, without importing/copying/converting its full bank.
// It does not authorize arbitrary indices or manufacture page permissions.
inline constexpr ProviderExtensionId kProjectedInputReadExtensionId{
    0x48594c4550524f4aULL, 0x52454144494e5054ULL, 1};
// Same-build facts about the selected prepared implementation, not its schema
// or provider family. An absent input means this implementation only qualifies
// the typed access chain; it does not receive projected storage itself.
struct OperationPreparedAccess final {
  static constexpr auto kNoInput = (std::numeric_limits<std::uint32_t>::max)();
  std::uint32_t projected_input = kNoInput;
  bool allows_empty_footprint = false;
  bool requires_configuration = false;
};
} // namespace hyle::compute::detail
