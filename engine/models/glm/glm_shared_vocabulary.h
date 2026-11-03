#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/mapping.h"

// Test-local immutable GLM shared embedding/head/norm supplier. This header
// neither constructs nor executes a model Graph. Default identity is official
// Flash layer45; explicit geometry and pinned identities can describe other
// GLM artifacts without treating the Flash4096 state calculation as Pro6144.
namespace glm_shared_vocabulary {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;

enum class Role : std::size_t { kEmbedding, kHead, kNorm, kCount };
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::kCount);
struct Geometry final {
  std::int64_t vocabulary = 0, hidden = 0;
  float epsilon = 1.0e-5F;
};
inline constexpr Geometry kFlashGeometry{154880, 4096, 1.0e-5F};
inline constexpr std::string_view kRepository = "zai-org/GLM-5.3-Flash";
inline constexpr std::string_view kRevision = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a";
inline constexpr std::array<std::string_view, kRoles> kFiles{
    "embedding.bf16", "head.bf16", "norm.bf16"};
inline constexpr std::array<std::string_view, kRoles> kNames{
    "shared embedding", "shared LM head", "NextN shared head norm"};
struct Pins final {
  std::string_view repository, revision;
  std::array<std::string_view, kRoles> sha256;
};
// Filled only from the root-owned exact-range acquisition receipt. An empty
// pin is rejected before any file is mapped; it never means unverified input.
inline constexpr Pins kFlashPins{kRepository, kRevision, {
    "4f7aecd51589ce7879d0ae48a5a1ed25750e4e94d2abe2d14c9526e5ce5641d6",
    "aa21c427970f64edd82669db3a8fb46613084e8bc271a3728784a52eb3f25ab4",
    "200c13367cfe6166220338e4c1ccd5d4e8bb414b251cb3c5b73a4bd49bb37188"}};

template<class T> inline T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " +
                                      std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, std::string_view where);
std::size_t index(Role role);
void validate_geometry(Geometry geometry);
c::TensorDesc description(Geometry geometry, Role role);

namespace detail {
// Same bounded SHA-256 block calculation used by the original expert mmap
// supplier: 64-byte mapped blocks and a 128-byte stack tail, never a bank copy.
std::string mapped_sha256(std::span<const std::byte> input);
} // namespace detail

class Data final {
 public:
  Geometry geometry{};
  std::string repository, revision;
  std::array<std::string, kRoles> sha256;

  static Data load(const std::filesystem::path &directory,
                   Geometry selected = kFlashGeometry, Pins pins = kFlashPins) {
    validate_geometry(selected);
    if (pins.repository.empty() || pins.revision.empty())
      throw std::runtime_error("GLM shared vocabulary needs a pinned model identity");
    for (const auto pin : pins.sha256) {
      if (pin.size() != 64 || !std::ranges::all_of(pin, [](char digit) {
            return (digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f');
          }))
        throw std::runtime_error("GLM shared vocabulary needs every actual SHA256 pin");
    }
    Data result;
    result.geometry = selected;
    result.repository = pins.repository;
    result.revision = pins.revision;
    for (std::size_t slot = 0; slot < kRoles; ++slot) {
      const auto role = static_cast<Role>(slot);
      const auto bytes = take(description(selected, role).required_bytes(),
                              "GLM shared role byte extent");
      const auto path = directory / std::string(kFiles[slot]);
      auto mapped = take(m::FileMapping::open(path.string().c_str()),
                         "map GLM original shared role");
      if (mapped.size() != bytes)
        throw std::runtime_error("GLM shared role byte length differs: " + path.string());
      const auto raw = take(mapped.bytes(), "GLM shared mapped bytes");
      if (detail::mapped_sha256(raw) != pins.sha256[slot])
        throw std::runtime_error("GLM shared role SHA256 differs: " + path.string());
      result.sha256[slot] = pins.sha256[slot];
      result.mappings_[slot] = std::move(mapped);
    }
    return result;
  }

  std::span<const std::byte> bytes(Role role) const {
    return take(mappings_[index(role)].bytes(), "GLM original mapped payload");
  }
  // Borrowed view: Data's actual FileMapping must remain alive through use.
  // Vocabulary::source is the shared owner; closing a temporary HostMapping
  // does not remove this underlying original file mapping.
  m::ConstBufferView view(Role role) const {
    return take(mappings_[index(role)].view(), "GLM original readonly mapped view");
  }
  std::vector<std::uint16_t> row(Role role, std::size_t row_index) const {
    validate_geometry(geometry);
    const auto slot = index(role);
    const auto rows = role == Role::kNorm ? std::uint64_t{1} :
                                          static_cast<std::uint64_t>(geometry.vocabulary);
    if (row_index >= rows)
      throw std::runtime_error("GLM original vocabulary row outside original role");
    const auto width = static_cast<std::uint64_t>(geometry.hidden);
    if (width > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))
      throw std::runtime_error("GLM original vocabulary row byte extent overflows");
    const auto stride = static_cast<std::size_t>(width) * sizeof(std::uint16_t);
    const auto raw = bytes(static_cast<Role>(slot));
    if (row_index > raw.size() / stride || stride > raw.size() - row_index * stride)
      throw std::runtime_error("GLM original vocabulary row exceeds mapped bytes");
    const auto selected = raw.subspan(row_index * stride, stride);
    std::vector<std::uint16_t> result(static_cast<std::size_t>(width));
    for (std::size_t element = 0; element < result.size(); ++element) {
      result[element] = static_cast<std::uint16_t>(
          std::to_integer<std::uint32_t>(selected[2 * element]) |
          (std::to_integer<std::uint32_t>(selected[2 * element + 1]) << 8u));
      if (!std::isfinite(std::bit_cast<float>(std::uint32_t{result[element]} << 16u)))
        throw std::runtime_error("GLM original vocabulary row has nonfinite BF16");
    }
    return result;
  }
  std::vector<std::uint16_t> norm() const { return row(Role::kNorm, 0); }
  std::vector<float> norm_f32() const {
    const auto original = norm();
    std::vector<float> result(original.size());
    for (std::size_t i = 0; i < original.size(); ++i)
      result[i] = std::bit_cast<float>(std::uint32_t{original[i]} << 16u);
    return result;
  }
 private:
  std::array<m::FileMapping, kRoles> mappings_;
};

struct ReadonlyRole final {
  std::string_view name;
  c::TensorDesc description;
  c::StorageOwner owner;
  c::ReadLease lease;
  hyle::Result<g::ConstantBinding> binding(g::ValueId value) const {
    if (!owner || !lease)
      return std::unexpected(hyle::Status::InvalidArgument("GLM shared role is not ready"));
    HYLE_ASSIGN_OR_RETURN(auto tensor, lease.tensor(description));
    if (tensor.storage().resource() != owner.ref())
      return std::unexpected(hyle::Status::InvalidArgument("GLM shared role owner differs from its bank"));
    return g::ConstantBinding{.value = value, .tensor = tensor, .owner = owner, .lease = lease};
  }
};

struct Vocabulary final {
  // This shared owner keeps all original mappings and oracle byte spans alive.
  // Device leases independently retain their actual resource through Graph use.
  std::shared_ptr<const Data> source;
  std::array<ReadonlyRole, kRoles> roles;

  static Vocabulary bind(std::shared_ptr<const Data> data, const c::StorageOwner &owner) {
    if (!data || !owner)
      throw std::runtime_error("GLM shared vocabulary needs original data and its Graph storage owner");
    Vocabulary result;
    result.source = std::move(data);
    for (std::size_t slot = 0; slot < kRoles; ++slot) {
      const auto role = static_cast<Role>(slot);
      const auto desc = description(result.source->geometry, role);
      const auto extent = take(desc.required_bytes(), "GLM shared bank extent");
      const auto raw = result.source->bytes(role);
      if (raw.size() != extent)
        throw std::runtime_error("GLM shared bank extent differs from original mapping");
      auto bank = take(m::Buffer::create(owner.ref(), extent, 256), "GLM shared immutable bank");
      check(bank.view().write(raw), "GLM shared original BF16 bank upload");
      result.roles[slot] = {kNames[slot], desc, owner,
          take(owner.adopt_buffer(std::move(bank)), "GLM shared readonly lease")};
    }
    return result;
  }
  static Vocabulary load(const std::filesystem::path &directory,
                         const c::StorageOwner &owner,
                         Geometry geometry = kFlashGeometry, Pins pins = kFlashPins) {
    return bind(std::make_shared<Data>(Data::load(directory, geometry, pins)), owner);
  }
  const ReadonlyRole &at(Role role) const { return roles[index(role)]; }
};
} // namespace glm_shared_vocabulary
