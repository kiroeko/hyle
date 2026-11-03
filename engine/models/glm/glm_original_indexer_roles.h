#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "../detail/component_definition.h"
#include "../detail/original_role_input.h"

// Test-local immutable supplier for seven original Flash indexer roles.
// Default pins identify NextN45; another actual layer supplies all seven pins.
// The checkpoint stores BF16; fixed06992 widens APE, LayerNorm and head-gate
// coefficients to F32 when loading. Runtime banks below do that exact widening
// once, while the three dense projection banks retain their original BF16 bits.
namespace glm_original_indexer_roles {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace sem = hyle::inference::model_definition;
namespace original = original_role_input;
using Bytes = std::vector<std::byte>;
inline constexpr std::string_view kRepository = "zai-org/GLM-5.3-Flash";
inline constexpr std::string_view kRevision = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a";
inline constexpr std::size_t kOriginalBytes = 14943744;
enum Parameter : std::size_t { kApe, kGate, kBias, kNorm, kWeights, kKey, kQuery, kParameters };
inline constexpr std::array<std::string_view, kParameters> kNames{
    "index_kpool_compress_ape", "index_kpool_compress_gate", "k_norm.bias",
    "k_norm.weight", "weights_proj.weight", "wk.weight", "wq_b.weight"};
using Pins = std::array<std::string_view, kParameters>;
inline constexpr Pins kDefaultSha256{
    "5409e5d02ce7e14b2b8bca99ec7cb051dd4c2c11d6f9570f5e4dfb827cddbd29",
    "2c4a613a9c17143de1c939032d036b2276c465421837026eb406ef873ef7c229",
    "0550bb42f1eb03b957705e89b41bbe4892cf319961da7cc0559b6e07321b5185",
    "71fb05d01b44ab8b80da4190a803e4fce846bf31009bfd41d267365d392e9359",
    "e31cf93353922020249dbf5b0ee7ca85813308048cac03cc838ca03462d1eada",
    "9348fb28363eed7797f751edf5dfd7be0bea2b397bfa62db1ea3aa4dac7842de",
    "a3ce61c176747c383284a40d75634f70f3fa59dc0f94c523e933ef933cadb6da"};
struct Data final {
  std::vector<std::uint16_t> ape, gate, bias, norm, weights, key, query;
  static Data load(const std::filesystem::path &directory, const Pins &pins = kDefaultSha256) {
    for (const auto pin : pins)
      if (pin.size() != 64) throw std::runtime_error("original GLM indexer requires complete actual-layer SHA256 pins");
    Data r;
    r.ape = original::read_bfloat(directory / "index_kpool_compress_ape.bf16", 4 * 128,
        pins[kApe]);
    r.gate = original::read_bfloat(directory / "index_kpool_compress_gate.bf16", 128 * 4096,
        pins[kGate]);
    r.bias = original::read_bfloat(directory / "k_norm.bias.bf16", 128,
        pins[kBias]);
    r.norm = original::read_bfloat(directory / "k_norm.weight.bf16", 128,
        pins[kNorm]);
    r.weights = original::read_bfloat(directory / "weights_proj.weight.bf16", 32 * 4096,
        pins[kWeights]);
    r.key = original::read_bfloat(directory / "wk.weight.bf16", 128 * 4096,
        pins[kKey]);
    r.query = original::read_bfloat(directory / "wq_b.weight.bf16", 4096 * 1536,
        pins[kQuery]);
    return r;
  }
};
template<class T> T take(hyle::Result<T> result, const char *where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
std::span<const std::uint16_t> values(const Data &data, Parameter parameter);
c::TensorDesc description(Parameter parameter);
std::vector<float> widen(std::span<const std::uint16_t> source);
Bytes payload(const Data &data, Parameter parameter);
struct ReadonlyRole final {
  std::string_view name;
  c::TensorDesc description;
  c::StorageOwner owner;
  c::ReadLease lease;
  hyle::Result<g::ConstantBinding> binding(g::ValueId value) const {
    auto tensor = lease.tensor(description);
    if (!tensor) return std::unexpected(tensor.error());
    return g::ConstantBinding{.value = value, .tensor = *tensor, .owner = owner, .lease = lease};
  }
};
using Roles = std::array<ReadonlyRole, kParameters>;
Roles bind(const Data &data, const c::StorageOwner &owner);
} // namespace glm_original_indexer_roles
