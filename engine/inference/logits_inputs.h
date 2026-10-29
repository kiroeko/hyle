#pragma once

#include "hyle/compute/resource_leases.h"
#include "hyle/compute/tensor_desc.h"

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

} // namespace hyle::inference
