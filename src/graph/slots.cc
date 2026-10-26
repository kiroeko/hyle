#include "hyle/graph/slots.h"

#include <array>
#include <utility>
#include <vector>

#include "hyle/mem/std_allocator.h"

namespace hyle::graph::slots {
namespace c = compute;

Result<Schema> Schema::create(c::TensorDesc item, std::uint32_t slots,
    std::size_t stride) noexcept {
  if (!item.valid() || !item.is_plain() || !item.is_dense() || !slots)
    return std::unexpected(Status::InvalidArgument(
        "slot schema needs nonempty dense plain items and positive capacity"));
  HYLE_ASSIGN_OR_RETURN(const auto elements, item.element_count());
  HYLE_ASSIGN_OR_RETURN(const auto bytes, item.required_bytes());
  const auto alignment = item.required_alignment();
  if (!elements || !bytes || !stride || stride < bytes || !alignment ||
      stride % alignment != 0)
    return std::unexpected(Status::InvalidArgument(
        "slot stride must cover and align its nonempty item"));
  if (slots > static_cast<std::uint32_t>(INT32_MAX) ||
      elements > static_cast<std::uint64_t>(INT64_MAX) ||
      stride > static_cast<std::uint64_t>(INT64_MAX))
    return std::unexpected(Status::OutOfRange(
        "slot geometry exceeds I32 indices or signed tensor strides"));
  Schema result;
  result.type_ = item.data_type();
  result.item_elements_ = static_cast<std::int64_t>(elements);
  result.slots_ = slots;
  result.stride_ = stride;
  // TensorDesc checks the complete physical extent before exposing the schema.
  HYLE_ASSIGN_OR_RETURN(const auto capacity, result.capacity_description());
  (void)capacity;
  return result;
}

Result<c::TensorDesc> Schema::capacity_description() const noexcept {
  if (!valid())
    return std::unexpected(Status::FailedPrecondition("slot schema is empty"));
  const std::array<std::int64_t, 2> dimensions{slots_, item_elements_};
  const std::array<std::int64_t, 2> strides{
      static_cast<std::int64_t>(stride_),
      static_cast<std::int64_t>(c::data_type_size(type_))};
  return c::TensorDesc::strided_bytes(type_, dimensions, strides);
}

Status Store::infer_output_descriptions(std::span<const c::TensorDesc> inputs,
    std::span<c::TensorDesc> outputs) const noexcept {
  if (!schema.valid() || inputs.size() != 2 || !outputs.empty())
    return Status::InvalidArgument("slot store needs its schema, rows, I32 IDs and no tensor output");
  const auto &rows = inputs[0];
  const auto &ids = inputs[1];
  if (!rows.valid() || !rows.is_plain() || !rows.is_dense() ||
      rows.rank() != 2 || rows.data_type() != schema.type() ||
      rows.dimensions()[0] <= 0 || rows.dimensions()[0] > schema.slot_count() ||
      rows.dimensions()[1] != schema.item_elements() ||
      !ids.valid() || !ids.is_plain() || !ids.is_dense() ||
      ids.rank() != 1 || ids.data_type() != c::DataType::kInt32 ||
      ids.dimensions()[0] != rows.dimensions()[0])
    return Status::InvalidArgument("slot store needs same-type dense whole items and matching I32 IDs");
  return Status::Ok();
}

namespace detail {
struct SlotBindingsData final {
  explicit SlotBindingsData(mem::AllocatorRef selected) noexcept
      : allocator(selected),
        sources(mem::StdAllocator<c::ReadLease>{selected}),
        destinations(mem::StdAllocator<c::WriteLease>{selected}),
        source_addresses(mem::StdAllocator<c::LeaseSlotAddress>{selected}),
        destination_addresses(mem::StdAllocator<c::LeaseSlotAddress>{selected}) {}
  void destroy() noexcept {
    const auto selected = allocator;
    selected.destroy(this);
  }
  mem::AllocatorRef allocator;
  c::LeaseSlotLayout layout;
  std::vector<c::ReadLease, mem::StdAllocator<c::ReadLease>> sources;
  std::vector<c::WriteLease, mem::StdAllocator<c::WriteLease>> destinations;
  std::vector<c::LeaseSlotAddress, mem::StdAllocator<c::LeaseSlotAddress>> source_addresses;
  std::vector<c::LeaseSlotAddress, mem::StdAllocator<c::LeaseSlotAddress>> destination_addresses;
  bool populated = false;
};
} // namespace detail

SlotBindings::SlotBindings(SlotBindings &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)) {}
SlotBindings &SlotBindings::operator=(SlotBindings &&other) noexcept {
  if (this != &other) {
    SlotBindings next(std::move(other));
    std::swap(data_, next.data_);
  }
  return *this;
}
SlotBindings::~SlotBindings() { if (data_) data_->destroy(); }

Result<SlotBindings> SlotBindings::create(c::LeaseSlotLayout layout,
    std::span<const c::ReadLease> sources, std::span<c::WriteLease> destinations,
    SlotBindingsOptions options) noexcept {
  if (!layout.valid() || destinations.empty() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument(
        "slot bindings need their original layout, destinations and metadata allocator"));
  HYLE_ASSIGN_OR_RETURN(const auto schema, Schema::create(layout.description(),
      layout.slot_count(), layout.stride_bytes()));
  if (destinations.size() > schema.slot_count() ||
      sources.size() > static_cast<std::uint64_t>(INT64_MAX))
    return std::unexpected(Status::OutOfRange("slot binding counts exceed their geometry"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  auto *data = allocator.construct<detail::SlotBindingsData>(allocator);
  if (!data)
    return std::unexpected(Status::ResourceExhausted("slot binding owner metadata"));
  SlotBindings result(data);
  try {
    // Allocate only the actual selected read/write footprint. No caller handle
    // has been copied or consumed if any allocation or late lease check fails.
    data->sources.resize(sources.size());
    data->destinations.resize(destinations.size());
    data->source_addresses.resize(sources.size());
    data->destination_addresses.resize(destinations.size());
  } catch (...) {
    return std::unexpected(Status::ResourceExhausted("slot binding selected metadata"));
  }
  data->layout = std::move(layout);
  HYLE_RETURN_IF_ERROR(result.refill(sources, destinations));
  return result;
}

Status SlotBindings::refill(std::span<const c::ReadLease> sources,
    std::span<c::WriteLease> destinations) noexcept {
  if (!data_)
    return Status::FailedPrecondition("slot refill needs its prepared original owner");
  auto &data = *data_;
  if (sources.size() != data.sources.size() || destinations.size() != data.destinations.size())
    return Status::InvalidArgument("slot refill must preserve its prepared source and destination counts");
  for (const auto &writer : data.destinations)
    if (writer.valid())
      return Status::FailedPrecondition("handle original slot destinations before refilling their transaction");
  HYLE_RETURN_IF_ERROR(data.layout.bind_indices(sources, destinations,
      data.source_addresses, data.destination_addresses));
  // The complete batch was validated under its original domain lock. These
  // assignments cannot allocate, fail or publish a native result.
  for (std::size_t i = 0; i < sources.size(); ++i)
    data.sources[i] = sources[i];
  for (std::size_t i = 0; i < destinations.size(); ++i)
    data.destinations[i] = std::move(destinations[i]);
  data.populated = true;
  return Status::Ok();
}

void SlotBindings::clear() noexcept {
  if (!data_) return;
  for (auto &source : data_->sources) source = {};
  for (auto &destination : data_->destinations) destination = {};
  for (auto &address : data_->source_addresses) address = {};
  for (auto &address : data_->destination_addresses) address = {};
  data_->populated = false;
}

c::LeaseSlotLayout SlotBindings::layout() const noexcept {
  return data_ ? data_->layout : c::LeaseSlotLayout{};
}
std::size_t SlotBindings::source_count() const noexcept {
  return data_ ? data_->sources.size() : 0;
}
std::size_t SlotBindings::destination_count() const noexcept {
  return data_ ? data_->destinations.size() : 0;
}
std::span<const c::ReadLease> SlotBindings::sources() const noexcept {
  return data_ && data_->populated ? std::span<const c::ReadLease>{data_->sources}
               : std::span<const c::ReadLease>{};
}
std::span<c::WriteLease> SlotBindings::destinations() noexcept {
  return data_ && data_->populated ? std::span<c::WriteLease>{data_->destinations}
               : std::span<c::WriteLease>{};
}
std::span<const c::WriteLease> SlotBindings::destinations() const noexcept {
  return data_ && data_->populated ? std::span<const c::WriteLease>{data_->destinations}
               : std::span<const c::WriteLease>{};
}
std::span<const c::LeaseSlotAddress> SlotBindings::source_addresses() const noexcept {
  return data_ && data_->populated ? std::span<const c::LeaseSlotAddress>{data_->source_addresses}
               : std::span<const c::LeaseSlotAddress>{};
}
std::span<const c::LeaseSlotAddress> SlotBindings::destination_addresses() const noexcept {
  return data_ && data_->populated ? std::span<const c::LeaseSlotAddress>{data_->destination_addresses}
               : std::span<const c::LeaseSlotAddress>{};
}

} // namespace hyle::graph::slots
