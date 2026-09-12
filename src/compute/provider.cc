#include "hyle/compute/provider.h"

#include "provider_internal.h"

namespace hyle::compute {

ProviderIdentity Provider::identity() const noexcept {
  const detail::ProviderVTable *table = detail::ProviderAccess::vtable(*this);
  return table != nullptr ? table->identity : ProviderIdentity{};
}

ProviderId Provider::id() const noexcept { return identity().id; }

std::string_view Provider::name() const noexcept {
  const ProviderIdentity value = identity();
  return value.valid() ? value.name : std::string_view{"unknown"};
}

std::string_view Provider::version() const noexcept {
  return identity().version;
}

} // namespace hyle::compute
