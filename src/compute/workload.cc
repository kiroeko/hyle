#include "hyle/compute/workload.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "hyle/util/ref_counted.h"

namespace hyle::compute {
namespace detail {

struct ModelIdentityState final : util::RefCounted<ModelIdentityState> {
  mem::AllocatorRef allocator;
  std::size_t bytes = 0;
  ModelDescription description;

  void destroy() noexcept {
    const auto owner = allocator;
    const auto allocation_bytes = bytes;
    this->~ModelIdentityState();
    owner.deallocate(this, allocation_bytes, alignof(ModelIdentityState));
  }
};

} // namespace detail

Result<ModelIdentity>
ModelIdentity::create(ModelDescription description,
                       mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid() ||
      (description.family.empty() &&
       (!description.name.empty() || !description.revision.empty() ||
        !description.configuration.empty())) ||
      (description.name.empty() &&
       (!description.revision.empty() || !description.configuration.empty())))
    return std::unexpected(Status::InvalidArgument("model identity"));
  if (description.family.empty())
    return ModelIdentity{};

  using State = detail::ModelIdentityState;
  constexpr auto maximum =
      static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)());
  std::size_t bytes = sizeof(State);
  for (const auto field :
       std::array{description.family, description.name, description.revision,
                  description.configuration}) {
    if (field.size() > maximum - bytes)
      return std::unexpected(Status::ResourceExhausted("model identity size"));
    bytes += field.size();
  }
  auto *storage = allocator.allocate(bytes, alignof(State));
  if (storage == nullptr)
    return std::unexpected(Status::ResourceExhausted("model identity allocation"));
  auto *state = ::new (storage) State;
  state->allocator = allocator;
  state->bytes = bytes;
  auto *destination = reinterpret_cast<char *>(state + 1);
  auto copy = [&](std::string_view source) noexcept {
    const std::string_view result(destination, source.size());
    if (!source.empty())
      std::memcpy(destination, source.data(), source.size());
    destination += source.size();
    return result;
  };
  state->description = {copy(description.family), copy(description.name),
                        copy(description.revision),
                        copy(description.configuration)};
  return ModelIdentity(state);
}

ModelIdentity::ModelIdentity(const ModelIdentity &other) noexcept
    : state_(other.state_) {
  if (state_ != nullptr)
    state_->incref();
}

ModelIdentity &ModelIdentity::operator=(const ModelIdentity &other) noexcept {
  if (state_ != other.state_) {
    if (other.state_ != nullptr)
      other.state_->incref();
    if (state_ != nullptr)
      state_->decref();
    state_ = other.state_;
  }
  return *this;
}

ModelIdentity::ModelIdentity(ModelIdentity &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

ModelIdentity &ModelIdentity::operator=(ModelIdentity &&other) noexcept {
  if (this != &other) {
    if (state_ != nullptr)
      state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}

ModelIdentity::~ModelIdentity() {
  if (state_ != nullptr)
    state_->decref();
}

ModelDescription ModelIdentity::description() const noexcept {
  return state_ != nullptr ? state_->description : ModelDescription{};
}

bool ModelIdentity::matches_family(
    std::string_view required_family) const noexcept {
  return !required_family.empty() &&
         description().family == required_family;
}

bool ModelIdentity::matches_model(std::string_view required_family,
                                  std::string_view required_name) const noexcept {
  const auto actual = description();
  return !required_family.empty() && !required_name.empty() &&
         actual.family == required_family && actual.name == required_name;
}

bool ModelIdentity::matches(ModelDescription required) const noexcept {
  return required.complete() && description() == required;
}

bool operator==(const ModelIdentity &left, const ModelIdentity &right) noexcept {
  return left.state_ == right.state_ || left.description() == right.description();
}

} // namespace hyle::compute
