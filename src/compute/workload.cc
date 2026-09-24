#include "hyle/compute/workload.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "hyle/util/ref_counted.h"

namespace hyle::compute {
namespace {

constexpr bool is_lower_hex_digest(std::string_view value) noexcept {
  if (value.size() != 64)
    return false;
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f')))
      return false;
  }
  return true;
}

constexpr bool
valid_description(ModelArtifactProfileDescription description) noexcept {
  if (description.empty())
    return true;
  return description.complete() &&
         is_lower_hex_digest(description.configuration_digest) &&
         is_lower_hex_digest(description.manifest_digest);
}

} // namespace

namespace detail {

struct ModelArtifactProfileState final
    : util::RefCounted<ModelArtifactProfileState> {
  mem::AllocatorRef allocator;
  std::size_t bytes = 0;
  ModelArtifactProfileDescription description;

  void destroy() noexcept {
    const auto owner = allocator;
    const auto allocation_bytes = bytes;
    this->~ModelArtifactProfileState();
    owner.deallocate(this, allocation_bytes,
                     alignof(ModelArtifactProfileState));
  }
};

} // namespace detail

Result<ModelArtifactProfile>
ModelArtifactProfile::create(ModelArtifactProfileDescription description,
                             mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid() || !valid_description(description))
    return std::unexpected(Status::InvalidArgument("model artifact profile"));
  if (description.empty())
    return ModelArtifactProfile{};

  using State = detail::ModelArtifactProfileState;
  constexpr auto maximum =
      static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)());
  const std::array fields{description.model_publisher,
                          description.model_repository,
                          description.model_family,
                          description.model_name,
                          description.model_revision,
                          description.configuration_digest,
                          description.artifact_repository,
                          description.artifact_revision,
                          description.variant,
                          description.manifest_digest,
                          description.format_family};
  std::size_t bytes = sizeof(State);
  for (const auto field : fields) {
    if (field.size() > maximum - bytes)
      return std::unexpected(
          Status::ResourceExhausted("model artifact profile size"));
    bytes += field.size();
  }
  auto *storage = allocator.allocate(bytes, alignof(State));
  if (storage == nullptr)
    return std::unexpected(
        Status::ResourceExhausted("model artifact profile allocation"));
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
  state->description = {copy(description.model_publisher),
                        copy(description.model_repository),
                        copy(description.model_family),
                        copy(description.model_name),
                        copy(description.model_revision),
                        copy(description.configuration_digest),
                        copy(description.artifact_repository),
                        copy(description.artifact_revision),
                        copy(description.variant),
                        copy(description.manifest_digest),
                        copy(description.format_family)};
  return ModelArtifactProfile(state);
}

ModelArtifactProfile::ModelArtifactProfile(
    const ModelArtifactProfile &other) noexcept
    : state_(other.state_) {
  if (state_ != nullptr)
    state_->incref();
}

ModelArtifactProfile &
ModelArtifactProfile::operator=(const ModelArtifactProfile &other) noexcept {
  if (state_ != other.state_) {
    if (other.state_ != nullptr)
      other.state_->incref();
    if (state_ != nullptr)
      state_->decref();
    state_ = other.state_;
  }
  return *this;
}

ModelArtifactProfile::ModelArtifactProfile(
    ModelArtifactProfile &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

ModelArtifactProfile &
ModelArtifactProfile::operator=(ModelArtifactProfile &&other) noexcept {
  if (this != &other) {
    if (state_ != nullptr)
      state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}

ModelArtifactProfile::~ModelArtifactProfile() {
  if (state_ != nullptr)
    state_->decref();
}

ModelArtifactProfileDescription
ModelArtifactProfile::description() const noexcept {
  return state_ != nullptr ? state_->description
                           : ModelArtifactProfileDescription{};
}

bool ModelArtifactProfile::matches(
    ModelArtifactProfileDescription expected) const noexcept {
  return expected.complete() && description() == expected;
}

bool operator==(const ModelArtifactProfile &left,
                const ModelArtifactProfile &right) noexcept {
  return left.state_ == right.state_ ||
         left.description() == right.description();
}

} // namespace hyle::compute
