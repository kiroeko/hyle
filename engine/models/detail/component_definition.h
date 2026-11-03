#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/operation.h"
#include "hyle/compute/workload.h"

// The existing cold component definition, shared by compiled model math and
// finite validation callers. This owns descriptions only, never an executor,
// permission ledger, device work or runtime state. Optional validation payloads
// and callbacks are supplied by tests; no numerical oracle lives here.
namespace hyle::inference::model_definition {
namespace c = hyle::compute;
enum class ValueKind : std::uint8_t { kInput, kConstant, kNodeOutput };

struct ValueId final {
  std::uint32_t index = 0;
  ValueKind type = ValueKind::kInput;
  std::uint32_t ordinal() const noexcept { return index; }
  ValueKind kind() const noexcept { return type; }
};

struct DirectNode final {
  c::Operation operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
};

struct Replay final {
  std::vector<std::vector<std::byte>> inputs;
  std::vector<std::vector<float>> outputs;
  std::vector<std::vector<float>> native_outputs{};
  bool reset_state = false;
};

struct Fragment final {
  const c::ModelArtifactProfileDescription *profile;
  std::uint64_t region;
  bool quantized = false;
  float absolute_tolerance = 8.0e-5F;
  std::vector<c::TensorDesc> descriptions{};
  std::vector<ValueId> inputs{}, constants{}, outputs{};
  std::vector<std::vector<std::byte>> constant_bytes{};
  std::vector<DirectNode> nodes{};
  std::vector<Replay> replays{};
  std::vector<std::pair<std::size_t, std::size_t>> state_ports{};
  std::vector<std::size_t> exact_outputs{};
  std::vector<std::size_t> bfloat_chain_outputs{};
  struct UnorderedIds final {
    std::size_t port, row_width;
    std::int64_t extent;
    bool compare_members = false;
    bool allow_padding = false;
  };
  std::optional<UnorderedIds> unordered_ids{};
  std::function<void(std::size_t, std::span<const float>)> observe{};
  // A consumer may validate an ill-conditioned intermediate against its
  // actual predecessor. Final outputs still use the independent chain oracle.
  std::function<bool(std::size_t, std::span<const float>)> validate_intermediate{};
  c::WorkloadPhase phase=c::WorkloadPhase::kDecode;
  // Optional one-time, native weight preparation. Outputs replace these
  // constant slots; no host-generated payload is uploaded into those slots.
  std::shared_ptr<const Fragment> preparation{};
  std::vector<std::size_t> prepared_constants{};
  // A checked-index consumer can exercise accepted failure and fresh-slot
  // recovery after its ordinary changed-input replays.
  std::optional<std::size_t> invalid_index_input{};

  const UnorderedIds *id_validation(std::size_t port) const {
    return unordered_ids && unordered_ids->port == port ? &*unordered_ids : nullptr;
  }

  bool exact_output(std::size_t port) const {
    return std::find(exact_outputs.begin(), exact_outputs.end(), port) !=
           exact_outputs.end();
  }

  bool bfloat_chain_output(std::size_t port) const {
    return std::find(bfloat_chain_outputs.begin(), bfloat_chain_outputs.end(), port) !=
           bfloat_chain_outputs.end();
  }

  const c::TensorDesc &description(ValueId id) const {
    return descriptions[id.ordinal()];
  }
  ValueId value(c::TensorDesc desc, ValueKind kind) {
    ValueId id{static_cast<std::uint32_t>(descriptions.size()), kind};
    descriptions.push_back(std::move(desc));
    return id;
  }
  ValueId input(c::TensorDesc desc) {
    auto id = value(std::move(desc), ValueKind::kInput);
    inputs.push_back(id);
    return id;
  }
  ValueId constant(c::TensorDesc desc, std::vector<std::byte> bytes) {
    auto id = value(std::move(desc), ValueKind::kConstant);
    constants.push_back(id);
    constant_bytes.push_back(std::move(bytes));
    return id;
  }
  std::vector<ValueId> append(c::Operation operation,
                              std::initializer_list<ValueId> ports,
                              std::size_t count = 1) {
    std::vector<c::TensorDesc> in, out(count);
    for (auto id : ports)
      in.push_back(description(id));
    auto status = c::infer_output_descriptions(operation, in, out);
    if (!status.ok())
      throw std::runtime_error(std::string(status.message()));
    DirectNode node{operation, ports, {}};
    for (auto &desc : out)
      node.outputs.push_back(value(std::move(desc), ValueKind::kNodeOutput));
    auto ids = node.outputs;
    nodes.push_back(std::move(node));
    return ids;
  }
  ValueId one(c::Operation operation, std::initializer_list<ValueId> ports) {
    return append(operation, ports)[0];
  }
};

template <class T> std::vector<std::byte> bytes(std::span<const T> values) {
  const auto data = std::as_bytes(values);
  return {data.begin(), data.end()};
}
template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  return bytes<T>(std::span<const T>(values));
}
inline c::TensorDesc dense(std::initializer_list<std::int64_t> shape,
                           c::DataType type = c::DataType::kFloat32) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
} // namespace hyle::inference::model_definition
