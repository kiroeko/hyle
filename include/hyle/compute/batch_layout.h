#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include "hyle/compute/low_precision.h"
#include "hyle/compute/resource_leases.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_BATCH_API
#elif defined(HYLE_COMPUTE_BATCH_BUILD)
#define HYLE_COMPUTE_BATCH_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_BATCH_API HYLE_IMPORT
#endif

namespace hyle::compute {
enum class BatchStorage : std::uint8_t { kPaged, kRagged };
enum class BatchCacheRole : std::uint8_t { kKey, kValue };
struct BatchCapacity final {
  BatchStorage storage = BatchStorage::kPaged;
  DataType query_type = DataType::kFloat16;
  DataType output_type = DataType::kFloat16;
  CacheStorage key_storage{}, value_storage{};
  std::uint32_t max_requests = 1, max_query_rows = 1, max_kv_rows = 1;
  std::uint32_t max_pages = 1, max_page_refs = 1;
  std::uint32_t query_heads = 1, kv_heads = 1;
  std::uint32_t qk_head_dim = 64, value_head_dim = 64, page_size = 16;
  friend constexpr bool operator==(BatchCapacity, BatchCapacity) noexcept = default;
};
struct HostBatchLayout final {
  std::span<const std::int32_t> query_indptr;
  std::span<const std::int32_t> kv_indptr;
  std::span<const std::int32_t> page_indices;
  std::span<const std::int32_t> last_page_lengths;
};
struct BatchLayoutFacts final {
  std::uint32_t requests = 0, query_rows = 0, kv_rows = 0, page_refs = 0;
};
struct BatchMetadataOffsets final {
  std::size_t query_indptr = 0, kv_indptr = 0, page_indices = 0, last_page_lengths = 0, bytes = 0;
};
struct BatchSchemaOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};
namespace detail { struct BatchSchemaData; struct BatchLayoutData; }
class BatchLayoutVersion;

// Immutable capacity/identity. Cold version creation is thread-safe when the
// supplied allocator supports concurrent allocation; handle mutation needs
// external synchronization. Source spans must not change during version().
// Allocator remains borrowed through
// all versions and exported F1 leases/native uses. No model/page policy here.
class BatchSchema final {
public:
  BatchSchema() noexcept = default;
  HYLE_COMPUTE_BATCH_API BatchSchema(const BatchSchema &) noexcept;
  HYLE_COMPUTE_BATCH_API BatchSchema &operator=(const BatchSchema &) noexcept;
  HYLE_COMPUTE_BATCH_API BatchSchema(BatchSchema &&) noexcept;
  HYLE_COMPUTE_BATCH_API BatchSchema &operator=(BatchSchema &&) noexcept;
  HYLE_COMPUTE_BATCH_API ~BatchSchema();
  HYLE_COMPUTE_BATCH_API static Result<BatchSchema> create(BatchCapacity, BatchSchemaOptions = {}) noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_BATCH_API BatchCapacity capacity() const noexcept;
  HYLE_COMPUTE_BATCH_API BatchMetadataOffsets metadata_offsets() const noexcept;
  HYLE_COMPUTE_BATCH_API Result<TensorDesc> metadata_description() const noexcept;
  // Validates all borrowed input before allocation/copy. Stores an immutable
  // fixed-capacity snapshot; repeated shape/page values still get a new version.
  HYLE_COMPUTE_BATCH_API Result<BatchLayoutVersion> version(HostBatchLayout) const noexcept;
  HYLE_COMPUTE_BATCH_API bool owns(const BatchLayoutVersion &) const noexcept;
private:
  friend class BatchLayoutVersion;
  explicit BatchSchema(detail::BatchSchemaData *data) noexcept : data_(data) {}
  detail::BatchSchemaData *data_ = nullptr;
};

// Shared readonly snapshot and schema owner. Spans borrow this version, never
// the caller's source spans. ReadLease export composes with original F2 and F4;
// no raw mutable/device-generated metadata can mint this identity.
class BatchLayoutVersion final {
public:
  BatchLayoutVersion() noexcept = default;
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion(const BatchLayoutVersion &) noexcept;
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion &operator=(const BatchLayoutVersion &) noexcept;
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion(BatchLayoutVersion &&) noexcept;
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion &operator=(BatchLayoutVersion &&) noexcept;
  HYLE_COMPUTE_BATCH_API ~BatchLayoutVersion();
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_BATCH_API BatchSchema schema() const noexcept;
  HYLE_COMPUTE_BATCH_API std::uint64_t generation() const noexcept;
  HYLE_COMPUTE_BATCH_API BatchLayoutFacts facts() const noexcept;
  HYLE_COMPUTE_BATCH_API HostBatchLayout host() const noexcept;
  HYLE_COMPUTE_BATCH_API Result<ReadLease> metadata() const noexcept;
  HYLE_COMPUTE_BATCH_API bool operator==(const BatchLayoutVersion &) const noexcept;
private:
  friend class BatchSchema;
  explicit BatchLayoutVersion(detail::BatchLayoutData *data) noexcept : data_(data) {}
  detail::BatchLayoutData *data_ = nullptr;
};
// Shared cold validation/schema functions used by typed operations/providers.
HYLE_COMPUTE_BATCH_API Status validate_batch_capacity(BatchCapacity) noexcept;
// Description geometry is separate from native math qualification. Cache
// scales belong to capacity/schema identity, not the encoded TensorDesc bytes.
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_query_description(BatchCapacity) noexcept;
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_cache_description(BatchCapacity, BatchCacheRole) noexcept;
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_output_description(BatchCapacity, std::uint32_t query_rows) noexcept;
HYLE_COMPUTE_BATCH_API Status validate_batch_attention_options(bool causal, std::int32_t window_left) noexcept;
HYLE_COMPUTE_BATCH_API Status validate_batch_attention_layout(
    const BatchLayoutVersion &, bool causal, std::int32_t window_left = -1) noexcept;
HYLE_COMPUTE_BATCH_API Result<BatchMetadataOffsets> batch_metadata_offsets(BatchCapacity) noexcept;
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_metadata_description(BatchCapacity) noexcept;
} // namespace hyle::compute
