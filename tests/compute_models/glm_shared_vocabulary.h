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
inline void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " +
                                           std::string(status.message()));
}
inline std::size_t index(Role role) {
  const auto result = static_cast<std::size_t>(role);
  if (result >= kRoles) throw std::runtime_error("unknown GLM shared vocabulary role");
  return result;
}
inline void validate_geometry(Geometry geometry) {
  if (geometry.vocabulary <= 0 || geometry.hidden <= 0 ||
      !std::isfinite(geometry.epsilon) || !(geometry.epsilon > 0))
    throw std::runtime_error("GLM shared vocabulary geometry/epsilon is invalid");
}
inline c::TensorDesc description(Geometry geometry, Role role) {
  validate_geometry(geometry);
  if (role == Role::kNorm)
    return take(c::TensorDesc::dense(c::DataType::kBFloat16,
        std::array{geometry.hidden}), "GLM shared norm description");
  (void)index(role);
  return take(c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array{geometry.vocabulary, geometry.hidden}), "GLM shared vocabulary description");
}

namespace detail {
// Same bounded SHA-256 block calculation used by the original expert mmap
// supplier: 64-byte mapped blocks and a 128-byte stack tail, never a bank copy.
inline std::string mapped_sha256(std::span<const std::byte> input) {
  constexpr std::array<std::uint32_t, 64> constants{
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
  std::array<std::uint32_t, 8> hash{
      0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
      0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  if(input.size()>std::numeric_limits<std::uint64_t>::max()/8)throw std::runtime_error("mapped SHA length overflow");
  const auto consume=[&](std::span<const std::byte,64> block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i)
      for (std::size_t byte = 0; byte < 4; ++byte)
        words[i] = (words[i] << 8u) |
            std::to_integer<std::uint32_t>(block[i * 4 + byte]);
    for (std::size_t i = 16; i < words.size(); ++i) {
      const auto x = words[i - 15], y = words[i - 2];
      const auto small0 = std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3u);
      const auto small1 = std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10u);
      words[i] = words[i - 16] + small0 + words[i - 7] + small1;
    }
    auto a=hash[0], b=hash[1], cc=hash[2], d=hash[3];
    auto e=hash[4], f=hash[5], gg=hash[6], h=hash[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const auto big1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const auto first = h + big1 + ((e & f) ^ (~e & gg)) + constants[i] + words[i];
      const auto big0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const auto second = big0 + ((a & b) ^ (a & cc) ^ (b & cc));
      h=gg; gg=f; f=e; e=d+first; d=cc; cc=b; b=a; a=first+second;
    }
    hash[0]+=a; hash[1]+=b; hash[2]+=cc; hash[3]+=d;
    hash[4]+=e; hash[5]+=f; hash[6]+=gg; hash[7]+=h;
  };
  std::size_t offset=0;
  for(;input.size()-offset>=64;offset+=64)consume(std::span<const std::byte,64>(input.data()+offset,64));
  std::array<std::byte,128> tail{};const auto remaining=input.size()-offset;
  std::copy(input.begin()+std::ptrdiff_t(offset),input.end(),tail.begin());tail[remaining]=std::byte{0x80};
  const auto extent=remaining<56?64u:128u;const auto bits=std::uint64_t(input.size())*8;
  for(std::size_t byte=0;byte<8;++byte)tail[extent-1-byte]=std::byte((bits>>(byte*8))&0xffu);
  consume(std::span<const std::byte,64>(tail.data(),64));
  if(extent==128)consume(std::span<const std::byte,64>(tail.data()+64,64));
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (auto word : hash) result << std::setw(8) << word;
  return result.str();
}
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
