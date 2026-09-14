#include "hyle/compute/provider.h"

#include <utility>

#include "provider_internal.h"

namespace hyle::compute {

Provider::Provider(const Provider &other) noexcept
    : vtable_(other.vtable_), payload_(other.payload_) {
  if (vtable_ != nullptr && vtable_->retain_configuration != nullptr)
    vtable_->retain_configuration(*this);
}

Provider &Provider::operator=(const Provider &other) noexcept {
  if (this != &other) {
    Provider copy(other);
    std::swap(vtable_, copy.vtable_);
    std::swap(payload_, copy.payload_);
  }
  return *this;
}

Provider::Provider(Provider &&other) noexcept
    : vtable_(std::exchange(other.vtable_, nullptr)), payload_(other.payload_) {}

Provider &Provider::operator=(Provider &&other) noexcept {
  if (this != &other) {
    if (vtable_ != nullptr && vtable_->release_configuration != nullptr)
      vtable_->release_configuration(*this);
    vtable_ = std::exchange(other.vtable_, nullptr);
    payload_ = other.payload_;
  }
  return *this;
}

Provider::~Provider() {
  if (vtable_ != nullptr && vtable_->release_configuration != nullptr)
    vtable_->release_configuration(*this);
}

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
