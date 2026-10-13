#include "hyle/compute/batch_paged_native.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "hyle/compute/detail/lease_access.h"
#include "hyle/util/ref_counted.h"

namespace hyle::compute {
namespace detail {
struct NativePagedBatchProgramData final : util::RefCounted<NativePagedBatchProgramData> {
  NativePagedBatchProgramData(BatchSchema selected, NativePagedBatchOptions settings, mem::AllocatorRef a)
      : schema(std::move(selected)), options(std::move(settings)), allocator(a) {}
  void destroy() noexcept {
    const auto selected = allocator;
    this->~NativePagedBatchProgramData();
    selected.deallocate(this, sizeof(*this), alignof(NativePagedBatchProgramData));
  }
  BatchSchema schema;
  NativePagedBatchOptions options;
  mem::AllocatorRef allocator;
  std::array<TensorDesc, 4> descriptions;
  NativePagedBatchCosts costs;
};
struct NativePagedBatchVersionOwner final {
  NativePagedBatchProgram program;
  BatchLayoutVersion layout;
  mem::AllocatorRef allocator;
};
} // namespace detail
namespace {
void release_version_owner(void *opaque) noexcept {
  auto *owner = static_cast<detail::NativePagedBatchVersionOwner *>(opaque);
  const auto allocator = owner->allocator;
  allocator.destroy(owner);
}
} // namespace

NativePagedBatchProgram::NativePagedBatchProgram(const NativePagedBatchProgram &other) noexcept
    : data_(other.data_) { if (data_) data_->incref(); }
NativePagedBatchProgram &NativePagedBatchProgram::operator=(const NativePagedBatchProgram &other) noexcept {
  if (this != &other) { NativePagedBatchProgram copy(other); std::swap(data_, copy.data_); }
  return *this;
}
NativePagedBatchProgram::NativePagedBatchProgram(NativePagedBatchProgram &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)) {}
NativePagedBatchProgram &NativePagedBatchProgram::operator=(NativePagedBatchProgram &&other) noexcept {
  if (this != &other) { NativePagedBatchProgram moved(std::move(other)); std::swap(data_, moved.data_); }
  return *this;
}
NativePagedBatchProgram::~NativePagedBatchProgram() { if (data_) data_->decref(); }
Result<NativePagedBatchProgram> NativePagedBatchProgram::prepare(
    BatchSchema schema, NativePagedBatchOptions options) noexcept {
  if (!schema.valid() || !options.allocator.valid() || !options.max_host_metadata_bytes ||
      !std::isfinite(options.scale))
    return std::unexpected(Status::InvalidArgument("native page recipe needs schema/allocator/finite budget/scale"));
  const auto capacity = schema.capacity();
  if (capacity.storage != BatchStorage::kPaged || capacity.page_size != 16 ||
      (capacity.data_type != DataType::kFloat16 && capacity.data_type != DataType::kBFloat16))
    return std::unexpected(Status::Unimplemented("native page recipe requires F16/BF16 page16 schema"));
  const std::uint64_t refs = capacity.max_page_refs;
  if (refs > std::uint32_t(INT32_MAX) || refs > SIZE_MAX / sizeof(std::int32_t))
    return std::unexpected(Status::OutOfRange("native page table extent overflow"));
  const auto bytes = std::size_t(capacity.max_page_refs) * sizeof(std::int32_t);
  if (bytes > options.max_host_metadata_bytes)
    return std::unexpected(Status::ResourceExhausted("native page table host byte capacity"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  auto *data = allocator.construct<detail::NativePagedBatchProgramData>(schema, std::move(options), allocator);
  if (!data) return std::unexpected(Status::ResourceExhausted("native page recipe metadata"));
  NativePagedBatchProgram result(data);
  HYLE_ASSIGN_OR_RETURN(data->descriptions[0], TensorDesc::dense(capacity.data_type,
      std::array<std::int64_t, 3>{capacity.max_query_rows, capacity.query_heads, capacity.head_dim}));
  HYLE_ASSIGN_OR_RETURN(data->descriptions[1], TensorDesc::dense(capacity.data_type,
      std::array<std::int64_t, 4>{capacity.max_pages, capacity.page_size, capacity.kv_heads, capacity.head_dim}));
  data->descriptions[2] = data->descriptions[1];
  HYLE_ASSIGN_OR_RETURN(data->descriptions[3], TensorDesc::dense(DataType::kInt32,
      std::array<std::int64_t, 1>{capacity.max_page_refs}));
  data->costs.index_bytes = bytes;
  return result;
}
BatchSchema NativePagedBatchProgram::schema() const noexcept { return data_ ? data_->schema : BatchSchema{}; }
float NativePagedBatchProgram::scale() const noexcept { return data_ ? data_->options.scale : 1.0F; }
bool NativePagedBatchProgram::causal() const noexcept { return data_ ? data_->options.causal : true; }
std::span<const TensorDesc> NativePagedBatchProgram::descriptions() const noexcept {
  return data_ ? std::span<const TensorDesc>(data_->descriptions) : std::span<const TensorDesc>{};
}
NativePagedBatchCosts NativePagedBatchProgram::costs() const noexcept { return data_ ? data_->costs : NativePagedBatchCosts{}; }
Result<NativePagedBatchVersion> NativePagedBatchProgram::version(BatchLayoutVersion layout) const noexcept {
  if (!data_) return std::unexpected(Status::FailedPrecondition("native page recipe is empty"));
  if (!data_->schema.owns(layout))
    return std::unexpected(Status::InvalidArgument("native page version schema differs"));
  HYLE_ASSIGN_OR_RETURN(auto buffer, mem::Buffer::create(mem::system_resource(), data_->costs.index_bytes, 64));
  {
    HYLE_ASSIGN_OR_RETURN(auto mapping, buffer.view().map_host());
    HYLE_ASSIGN_OR_RETURN(auto writable, mapping.writable_bytes());
    auto *indices = reinterpret_cast<std::int32_t *>(writable.data());
    std::fill_n(indices, data_->schema.capacity().max_page_refs, 0);
    const auto pages = layout.host().page_indices;
    std::copy(pages.begin(), pages.end(), indices);
    HYLE_RETURN_IF_ERROR(mapping.close());
  }
  HYLE_ASSIGN_OR_RETURN(auto ready, ReadLease::adopt(std::move(buffer),
      {.allocator = data_->allocator, .resource_budget = data_->options.resource_budget}));
  auto *owner = data_->allocator.construct<detail::NativePagedBatchVersionOwner>(*this, std::move(layout), data_->allocator);
  if (!owner) return std::unexpected(Status::ResourceExhausted("native page version owner"));
  detail::LeaseAccess::retain_resource_owner(ready, owner, &release_version_owner);
  return NativePagedBatchVersion(std::move(ready), owner);
}
BatchLayoutVersion NativePagedBatchVersion::layout() const noexcept { return valid() ? owner_->layout : BatchLayoutVersion{}; }
NativePagedBatchProgram NativePagedBatchVersion::program() const noexcept { return valid() ? owner_->program : NativePagedBatchProgram{}; }
Result<ReadLease> NativePagedBatchVersion::indices() const noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("native page version is empty"));
  return indices_;
}
} // namespace hyle::compute
