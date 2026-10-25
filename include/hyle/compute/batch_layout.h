#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include "hyle/compute/low_precision.h"
#include "hyle/compute/resource_leases.h"

namespace hyle::compute {
enum class BatchStorage : std::uint8_t { kPaged, kRagged };
enum class BatchCacheRole : std::uint8_t { kKey, kValue, kMla, kRow };
enum class MlaCacheLayout : std::uint8_t { kNone, kCkv512, kCkv512Rope64 };
// One physical row is [CKV512, KPE64] (or CKV512 without RoPE).
// V is the CKV prefix of that same row, never an independent allocation.
// One scalar cache interpretation (including its fixed E4 scale) is shared
// by CKV, optional KPE, and the V prefix. Q/output types are independent.
struct MlaBatchCapacity final {
  BatchStorage storage = BatchStorage::kPaged;
  DataType query_type = DataType::kFloat16;
  DataType output_type = DataType::kFloat16;
  CacheStorage cache_storage{};
  MlaCacheLayout layout = MlaCacheLayout::kCkv512Rope64;
  std::uint32_t max_requests = 1, max_query_rows = 1, max_kv_rows = 1;
  std::uint32_t max_pages = 1, max_page_refs = 1;
  std::uint32_t query_heads = 1, page_size = 16;
  friend constexpr bool operator==(MlaBatchCapacity, MlaBatchCapacity) noexcept = default;
};
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
// Paged row storage and selection-row/request grouping, independent of
// attention dtype, heads or Q/K/V geometry. Encoding includes its row depth.
struct RowCacheBatchCapacity final {
  RowCacheEncoding encoding{};
  std::uint32_t max_requests = 1, max_query_rows = 1, max_kv_rows = 1;
  std::uint32_t max_pages = 1, max_page_refs = 1, page_size = 16;
  friend constexpr bool operator==(RowCacheBatchCapacity, RowCacheBatchCapacity) noexcept = default;
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
namespace detail { struct BatchSchemaData; struct BatchLayoutData; struct BatchLayoutAccess; }
class BatchLayoutVersion;
class RowCacheBatchSchema;
class RowCacheLayoutVersion;

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
  HYLE_COMPUTE_BATCH_API static Result<BatchSchema> create_mla(MlaBatchCapacity, BatchSchemaOptions = {}) noexcept;
  // Explicit cache interpretation over this same metadata/topology domain.
  // All geometry, Q/output and MLA layout remain unchanged. Variants retain
  // the original schema, allocator, budget and generation counter; an
  // independent create() remains foreign even with identical capacities.
  // Row caches have their own typed encoding and do not admit this view.
  HYLE_COMPUTE_BATCH_API Result<BatchSchema> with_cache_storage(
      CacheStorage key, CacheStorage value) const noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_BATCH_API BatchCapacity capacity() const noexcept;
  HYLE_COMPUTE_BATCH_API MlaCacheLayout mla_layout() const noexcept;
  HYLE_COMPUTE_BATCH_API BatchMetadataOffsets metadata_offsets() const noexcept;
  HYLE_COMPUTE_BATCH_API Result<TensorDesc> metadata_description() const noexcept;
  // Validates all borrowed input before allocation/copy. Stores an immutable
  // active-sized snapshot; repeated shape/page values still get a new version.
  HYLE_COMPUTE_BATCH_API Result<BatchLayoutVersion> version(HostBatchLayout) const noexcept;
  // Ownership proves the explicit root metadata domain, not that a cache
  // payload has this variant's storage interpretation or page permission.
  HYLE_COMPUTE_BATCH_API bool owns(const BatchLayoutVersion &) const noexcept;
private:
  friend class BatchLayoutVersion;
  friend class RowCacheBatchSchema;
  friend struct detail::BatchLayoutAccess;
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
  // The root that minted this immutable topology, not a cache-storage view.
  HYLE_COMPUTE_BATCH_API BatchSchema origin_schema() const noexcept;
  HYLE_COMPUTE_BATCH_API std::uint64_t generation() const noexcept;
  HYLE_COMPUTE_BATCH_API BatchLayoutFacts facts() const noexcept;
  HYLE_COMPUTE_BATCH_API HostBatchLayout host() const noexcept;
  // Explicit fixed-capacity State/Graph export. First use allocates and zeros
  // the inactive region; later calls share the same readonly backing. Export
  // failure leaves the active version intact and can be retried. Thread-safe.
  HYLE_COMPUTE_BATCH_API Result<ReadLease> metadata() const noexcept;
  HYLE_COMPUTE_BATCH_API bool operator==(const BatchLayoutVersion &) const noexcept;
private:
  friend class BatchSchema;
  friend struct detail::BatchLayoutAccess;
  explicit BatchLayoutVersion(detail::BatchLayoutData *data) noexcept : data_(data) {}
  detail::BatchLayoutData *data_ = nullptr;
};
// Thin typed owners over the same immutable layout storage and F1 metadata
// export. No conversion to an attention schema/layout is exposed. Allocator
// and budget retain the original BatchSchema borrow/lifetime contract.
class RowCacheBatchSchema final {
public:
  RowCacheBatchSchema() noexcept = default;
  HYLE_COMPUTE_BATCH_API static Result<RowCacheBatchSchema> create(
      RowCacheBatchCapacity, BatchSchemaOptions = {}) noexcept;
  bool valid() const noexcept { return owner_.valid(); }
  HYLE_COMPUTE_BATCH_API RowCacheBatchCapacity capacity() const noexcept;
  BatchMetadataOffsets metadata_offsets() const noexcept { return owner_.metadata_offsets(); }
  Result<TensorDesc> metadata_description() const noexcept { return owner_.metadata_description(); }
  HYLE_COMPUTE_BATCH_API Result<RowCacheLayoutVersion> version(HostBatchLayout) const noexcept;
  HYLE_COMPUTE_BATCH_API bool owns(const RowCacheLayoutVersion &) const noexcept;
private:
  friend class RowCacheLayoutVersion;
  friend struct detail::BatchLayoutAccess;
  explicit RowCacheBatchSchema(BatchSchema owner) noexcept : owner_(std::move(owner)) {}
  BatchSchema owner_;
};
class RowCacheLayoutVersion final {
public:
  RowCacheLayoutVersion() noexcept = default;
  bool valid() const noexcept { return owner_.valid(); }
  HYLE_COMPUTE_BATCH_API RowCacheBatchSchema schema() const noexcept;
  std::uint64_t generation() const noexcept { return owner_.generation(); }
  BatchLayoutFacts facts() const noexcept { return owner_.facts(); }
  // query_indptr groups selection rows by request; kv_indptr indexes each
  // request's page references. This is a visibility/physical-page snapshot,
  // not validation of future IDs or their per-query causal policy.
  HostBatchLayout host() const noexcept { return owner_.host(); }
  Result<ReadLease> metadata() const noexcept { return owner_.metadata(); }
  bool operator==(const RowCacheLayoutVersion &other) const noexcept { return owner_ == other.owner_; }
private:
  friend class RowCacheBatchSchema;
  friend struct detail::BatchLayoutAccess;
  explicit RowCacheLayoutVersion(BatchLayoutVersion owner) noexcept : owner_(std::move(owner)) {}
  BatchLayoutVersion owner_;
};
HYLE_COMPUTE_BATCH_API Status validate_row_cache_batch_capacity(RowCacheBatchCapacity) noexcept;
HYLE_COMPUTE_BATCH_API Result<TensorDesc> row_cache_paged_description(RowCacheBatchCapacity) noexcept;
// Shared cold validation/schema functions used by typed operations/providers.
HYLE_COMPUTE_BATCH_API Status validate_batch_capacity(BatchCapacity) noexcept;
// Description geometry is separate from native math qualification. Cache
// scales belong to capacity/schema identity, not the encoded TensorDesc bytes.
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_query_description(BatchCapacity) noexcept;
// An exact packed Q extent within the same schema capacity. The one-argument
// constructor above continues to describe the full max_query_rows allocation.
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_query_description(BatchCapacity, std::uint32_t query_rows) noexcept;
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_cache_description(BatchCapacity, BatchCacheRole) noexcept;
HYLE_COMPUTE_BATCH_API Result<BatchCapacity> mla_batch_capacity(MlaBatchCapacity) noexcept;
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_cache_description(const BatchSchema &, BatchCacheRole) noexcept;
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_output_description(BatchCapacity, std::uint32_t query_rows) noexcept;
HYLE_COMPUTE_BATCH_API Status validate_batch_attention_options(bool causal, std::int32_t window_left) noexcept;
HYLE_COMPUTE_BATCH_API Status validate_batch_attention_layout(
    const BatchLayoutVersion &, bool causal, std::int32_t window_left = -1) noexcept;
HYLE_COMPUTE_BATCH_API Result<BatchMetadataOffsets> batch_metadata_offsets(BatchCapacity) noexcept;
HYLE_COMPUTE_BATCH_API Result<TensorDesc> batch_metadata_description(BatchCapacity) noexcept;
} // namespace hyle::compute
