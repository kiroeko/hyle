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
  std::int32_t window_left = -1;
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
  bool reused_native_plan = false;
};
namespace detail {
struct NativePagedBatchProgramData;
struct NativePagedBatchVersionOwner;
struct NativePagedBatchVersionData;
}
class NativePagedBatchVersion;
// Cold schema/attention recipe and compact page metadata, not native math
// qualification. Ports are active packed Q, true physical K/V capacity, and
// I32 page IDs. Q descriptions require the owned active layout; maximum query
// rows remain a schema limit. No mask or KV payload materialization is stored.
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
  HYLE_COMPUTE_BATCH_API std::int32_t window_left() const noexcept;
  HYLE_COMPUTE_BATCH_API std::size_t input_count() const noexcept;
  // Pure descriptor query: Q uses layout.query_rows; KV and page IDs retain
  // schema capacity. The layout must belong to this recipe's schema.
  HYLE_COMPUTE_BATCH_API Result<TensorDesc> input_description(
      const BatchLayoutVersion &layout, std::size_t port) const noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchCosts costs() const noexcept;
  // Exact owned layout identity/generation; caller spans are never retained.
  HYLE_COMPUTE_BATCH_API Result<NativePagedBatchVersion> version(BatchLayoutVersion) const noexcept;
private:
  explicit NativePagedBatchProgram(detail::NativePagedBatchProgramData *data) noexcept : data_(data) {}
  detail::NativePagedBatchProgramData *data_ = nullptr;
};
// Immutable active layout/recipe owner. indices() explicitly materializes the
// fixed-capacity readonly page table on first use and shares its cached export.
// Its lease independently retains recipe/layout after version handles die.
class NativePagedBatchVersion final {
public:
  NativePagedBatchVersion() noexcept = default;
  HYLE_COMPUTE_BATCH_API NativePagedBatchVersion(const NativePagedBatchVersion&) noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchVersion&operator=(const NativePagedBatchVersion&) noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchVersion(NativePagedBatchVersion&&) noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchVersion&operator=(NativePagedBatchVersion&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~NativePagedBatchVersion();
  bool valid() const noexcept { return data_!=nullptr; }
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion layout() const noexcept;
  HYLE_COMPUTE_BATCH_API NativePagedBatchProgram program() const noexcept;
  // Thread-safe explicit export; failure preserves the active version and can
  // be retried. Initial allocation/zeroing is charged only when requested.
  HYLE_COMPUTE_BATCH_API Result<ReadLease> indices() const noexcept;
private:
  friend class NativePagedBatchProgram;
  explicit NativePagedBatchVersion(detail::NativePagedBatchVersionData*data) noexcept:data_(data){}
  detail::NativePagedBatchVersionData*data_=nullptr;
};
} // namespace hyle::compute
