#include "hyle/compute/provider.h"

#include "provider_internal.h"

namespace hyle::compute {

ProviderKind Provider::kind() const noexcept {
  const detail::ProviderVTable *table = detail::ProviderAccess::vtable(*this);
  return table != nullptr ? table->kind : ProviderKind::kUnknown;
}

std::string_view Provider::name() const noexcept {
  const detail::ProviderVTable *table = detail::ProviderAccess::vtable(*this);
  return table != nullptr ? table->name : std::string_view{"unknown"};
}

std::string_view provider_name(ProviderKind provider) noexcept {
  switch (provider) {
  case ProviderKind::kHyleNative:
    return "hyle-native";
  case ProviderKind::kOneDnn:
    return "onednn";
  case ProviderKind::kMixed:
    return "mixed";
  case ProviderKind::kUnknown:
    break;
  }
  return "unknown";
}

} // namespace hyle::compute
