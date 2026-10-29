#pragma once

#include "hyle/compute/resource_leases.h"
#include "hyle/compute/tensor_desc.h"
#include "hyle/error/result.h"

#include <memory>
#include <span>
#include <utility>

namespace hyle::inference {

// Ready typed inputs, never KV, a sampling cursor or a grammar-state owner.
// Copies retain the original read permission. Keep any borrowed allocator,
// resource and Context alive through every retained copy and native use, or
// supply leases carrying their existing StorageOwner.
struct LogitsInput final {
  compute::TensorDesc description;
  compute::ReadLease lease;
};

// Counts is the single C0 history. An episode may share bias/mask, or its
// caller may supply one conditional row per target (including bonus); those
// rows contain only bias/mask and cannot replace counts. No payload enters
// Snapshots or committed history, and normalized q is never transformed twice.
struct LogitsInputs final {
  LogitsInput bias, allowed;
  LogitsInput counts{};
};

// An explicit caller freezes only typed metadata and the original read leases,
// never their vocabulary payloads. No mutable metadata alias escapes, and an
// empty/default input allocates nothing. Retain this value through the original
// episode's retirement; a borrowed span alone cannot cross that boundary.
class ConditionalLogitsInputs final {
public:
  ConditionalLogitsInputs() = default;
  explicit ConditionalLogitsInputs(std::span<const LogitsInputs> rows) : count_(rows.size()) {
    if (rows.empty()) return;
    auto copy = std::make_shared<LogitsInputs[]>(count_);
    for (std::size_t row = 0; row != count_; ++row) copy[row] = rows[row];
    owner_ = std::move(copy);
  }
  bool empty() const noexcept { return !owner_; }
  std::span<const LogitsInputs> rows() const noexcept {
    return owner_ ? std::span<const LogitsInputs>{owner_.get(), count_} : std::span<const LogitsInputs>{};
  }
  Result<ConditionalLogitsInputs> prefix(std::size_t count) const noexcept {
    if (count > rows().size())
      return std::unexpected(Status::InvalidArgument("conditional logits prefix exceeds its frozen row count"));
    if (!count) return ConditionalLogitsInputs{};
    auto result = *this; result.count_ = count; return result;
  }
private:
  std::shared_ptr<const LogitsInputs[]> owner_;
  std::size_t count_ = 0;
};

} // namespace hyle::inference
