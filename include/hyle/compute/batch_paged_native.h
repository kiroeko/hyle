#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <utility>

#include "hyle/compute/batch_ops.h"
#include "hyle/compute/resource_leases.h"

namespace hyle::compute {
struct NativePagedBatchOptions final {
  std::size_t max_host_metadata_bytes = 0;
  float scale = 1.0F;
  bool causal = true;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};
struct NativePagedBatchCosts final {
  std::size_t index_bytes = 0;
  bool native_private_bytes_unknown = true;
};
struct NativePagedInvocationOptions final {
  std::size_t max_metadata_backing_bytes = 0;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};
struct NativePagedBatchReport final {
  std::uint64_t layout_generation = 0;
  std::size_t staging_bytes = 0, device_mirror_bytes = 0, backing_bytes = 0, uploaded_bytes = 0;
  std::uint64_t upload_host_waits = 0;
  std::uint32_t device_banks = 2;
};
namespace detail {
struct NativePagedBatchProgramData;
struct NativePagedBatchVersionOwner;
}
class NativePagedBatchVersion;
// Cold schema/attention recipe and compact page metadata, not native math
// qualification. Ports are packed Q, true physical K/V capacity, and I32 page
// IDs. No query/head expansion, mask, or KV payload materialization is stored.
// Injected allocator/budget outlive programs, versions and exported leases.
class NativePagedBatchProgram final {
public:
  NativePagedBatchProgram() noexcept = default;
  HYLE_COMPUTE_BATCH_API NativePagedBatchProgram(const NativePagedBatchProgram &) noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchProgram &operator=(const NativePagedBatchProgram &) noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchProgram(NativePagedBatchProgram &&) noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchProgram &operator=(NativePagedBatchProgram &&) noexcept;
  HYLE_COMPUTE_BATCH_API ~NativePagedBatchProgram();
  HYLE_COMPUTE_BATCH_API static Result<NativePagedBatchProgram> prepare(
      BatchSchema, NativePagedBatchOptions) noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_BATCH_API BatchSchema schema() const noexcept;
  HYLE_COMPUTE_BATCH_API float scale() const noexcept;
  HYLE_COMPUTE_BATCH_API bool causal() const noexcept;
  HYLE_COMPUTE_BATCH_API std::span<const TensorDesc> descriptions() const noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchCosts costs() const noexcept;
  // Exact owned layout identity/generation; caller spans are never retained.
  HYLE_COMPUTE_BATCH_API Result<NativePagedBatchVersion> version(BatchLayoutVersion) const noexcept;
private:
  explicit NativePagedBatchProgram(detail::NativePagedBatchProgramData *data) noexcept : data_(data) {}
  detail::NativePagedBatchProgramData *data_ = nullptr;
};
// Immutable readonly page-ID snapshot. Its exported lease independently retains
// the recipe and owned layout, even after every version handle is released.
class NativePagedBatchVersion final {
public:
  NativePagedBatchVersion() noexcept = default;
  bool valid() const noexcept { return indices_.valid(); }
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion layout() const noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchProgram program() const noexcept;
  HYLE_COMPUTE_BATCH_API Result<ReadLease> indices() const noexcept;
private:
  friend class NativePagedBatchProgram;
  NativePagedBatchVersion(ReadLease indices, detail::NativePagedBatchVersionOwner *owner) noexcept
      : indices_(std::move(indices)), owner_(owner) {}
  ReadLease indices_;
  detail::NativePagedBatchVersionOwner *owner_ = nullptr;
};
} // namespace hyle::compute
