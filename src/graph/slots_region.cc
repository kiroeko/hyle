#include "slots_region.h"
#include "execution_source_internal.h"
#include "plan_state.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/indexed_copy_backend.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/transfer.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/slots.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::graph::slots {
namespace c = compute;
namespace cd = c::detail;
namespace gd = ::hyle::graph::detail;
namespace {
bool same_storage(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  return a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() == b.offset() && a.size() == b.size();
}
bool same_carrier(c::ConstTensorView a, c::ConstTensorView b) noexcept {
  return a.valid() == b.valid() && (!a.valid() ||
      (a.description() == b.description() && same_storage(a.storage(), b.storage())));
}
bool same_configuration(const detail::StoreConfiguration &a,
    const detail::StoreConfiguration &b) noexcept {
  return same_carrier(a.capacity, b.capacity) && same_carrier(a.source_indices, b.source_indices) &&
      same_carrier(a.destination_indices, b.destination_indices);
}

struct Find final {
  NodeId node{};
  detail::StoreBinding *binding = nullptr;
  const detail::StoreRegionVTable *interface = nullptr;
  ProviderRegionPrepared prepared{};
  void *invocation = nullptr;
  detail::StoreRegionFacts facts{};
  gd::ProjectedWriter proof{};
  ExecutionLaneId lane{};
};
struct IndexFragment final {
  std::size_t offset = 0;
  std::span<const c::LeaseSlotAddress> sources{}, destinations{};
};

// Only private I32 control bytes, never inference payload or slot ownership.
// Uploads use the original shared Queue or native cold source, including its
// actual completion/retirement and accepted-prefix failure ownership.
struct Indices final {
  c::LeasePool device_pool, host_pool;
  c::WriteLease writer, staging;
  c::ReadLease ready, source;
  c::LeaseUsePool uses;
  c::Completion pending;
  c::ResourceBudget::Reservation native_scratch;
  bool direct = false, failed = false, uploaded = false;

  static Result<Indices> create(mem::MemoryResourceRef resource,
      mem::MemoryResourceRef host, std::size_t count, ResourceOptions options,
      bool shared_queue) noexcept {
    if (!resource.valid() || !host.valid() || !count ||
        count > SIZE_MAX / sizeof(std::int32_t) ||
        !host.properties().has(mem::MemoryProperty::kHostAccessible))
      return std::unexpected(Status::InvalidArgument("slot IDs need valid storage, host staging and positive count"));
    const auto bytes = count * sizeof(std::int32_t);
    const c::LeasePoolOptions pool_options{.max_blocks = 1, .backing_alignment = 16,
        .allocator = options.allocator, .resource_budget = options.resource_budget};
    Indices result;
    result.direct = resource.properties().has(mem::MemoryProperty::kHostAccessible);
    HYLE_ASSIGN_OR_RETURN(result.device_pool, c::LeasePool::create(resource, bytes, pool_options));
    HYLE_ASSIGN_OR_RETURN(result.writer, result.device_pool.allocate(bytes, 4));
    if (!result.direct) {
      HYLE_ASSIGN_OR_RETURN(result.host_pool, c::LeasePool::create(host, bytes, pool_options));
      HYLE_ASSIGN_OR_RETURN(result.staging, result.host_pool.allocate(bytes, 4));
      if (shared_queue) {
        HYLE_ASSIGN_OR_RETURN(result.uses, c::LeaseUsePool::create({.max_batches = 1,
            .max_resources_per_batch = 2, .allocator = options.allocator,
            .resource_budget = options.resource_budget}));
      }
    }
    return result;
  }
  static Result<gd::ConstantCopy> native_copy(mem::BufferView output,
      mem::ConstBufferView input) noexcept {
    if (!output.size() || output.size() != input.size() || output.size() % sizeof(std::int32_t) ||
        output.size() / sizeof(std::int32_t) > static_cast<std::uint64_t>(INT64_MAX))
      return std::unexpected(Status::InvalidArgument("slot metadata native copy size differs"));
    HYLE_ASSIGN_OR_RETURN(auto description, c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(output.size() / sizeof(std::int32_t))}));
    HYLE_ASSIGN_OR_RETURN(auto destination, c::TensorView::create(description, output));
    HYLE_ASSIGN_OR_RETURN(auto source, c::ConstTensorView::create(description, input));
    return gd::ConstantCopy{destination, source};
  }
  Status upload(const ExecutionSource &execution, const cd::QueueRef &queue,
      const std::shared_ptr<c::Queue> &metadata_queue,
      ResourceOptions options, std::span<const IndexFragment> fragments,
      std::span<std::int32_t> uploaded_ids) noexcept {
    if (failed || pending.valid() || (!direct && uses.active_batches()))
      return Status::FailedPrecondition("slot index upload is failed or not actually retired");
    std::size_t count = 0;
    bool unchanged = uploaded && ready.valid();
    for (const auto &fragment : fragments) {
      if (fragment.offset != count || fragment.sources.size() > SIZE_MAX - fragment.destinations.size() ||
          count > SIZE_MAX - fragment.sources.size() - fragment.destinations.size() ||
          count > uploaded_ids.size() ||
          fragment.sources.size() + fragment.destinations.size() > uploaded_ids.size() - count)
        return Status::InvalidArgument("slot index slices differ from their fixed carrier");
      for (std::size_t i = 0; i < fragment.sources.size() + fragment.destinations.size(); ++i) {
        const auto &address = i < fragment.sources.size() ? fragment.sources[i]
            : fragment.destinations[i - fragment.sources.size()];
        if (address.index > static_cast<std::uint32_t>(INT32_MAX))
          return Status::InvalidArgument("slot index exceeds its I32 carrier");
        unchanged = unchanged && uploaded_ids[count + i] == static_cast<std::int32_t>(address.index);
      }
      count += fragment.sources.size() + fragment.destinations.size();
    }
    if (count != uploaded_ids.size())
      return Status::InvalidArgument("slot index count differs from its fixed carrier");
    // The bytes are immutable and genuinely published. Reusing them needs no
    // write permission, upload or Queue wait, even while another exact reader
    // retains the same carrier. All actual slot leases were checked by bind.
    if (unchanged) return Status::Ok();
    if (!direct && queue.valid() && (!metadata_queue || !metadata_queue->is_open()))
      return Status::FailedPrecondition("slot metadata Queue is closed or was not cold prepared");
    if (ready.valid()) { HYLE_ASSIGN_OR_RETURN(writer, ready.try_write()); }
    if (source.valid()) { HYLE_ASSIGN_OR_RETURN(staging, source.try_write()); }
    auto &host_writer = direct ? writer : staging;
    HYLE_ASSIGN_OR_RETURN(auto host, host_writer.view());
    if (count != host.size() / sizeof(std::int32_t))
      return Status::InvalidArgument("slot index storage differs from its fixed carrier");
    const auto remember = [&]() noexcept {
      for (const auto &fragment : fragments)
        for (std::size_t i = 0; i < fragment.sources.size() + fragment.destinations.size(); ++i) {
          const auto &address = i < fragment.sources.size() ? fragment.sources[i]
              : fragment.destinations[i - fragment.sources.size()];
          uploaded_ids[fragment.offset + i] = static_cast<std::int32_t>(address.index);
        }
      uploaded = true;
    };
    HYLE_ASSIGN_OR_RETURN(auto mapping, host.map_host());
    HYLE_ASSIGN_OR_RETURN(auto bytes, mapping.writable_bytes());
    for (const auto &fragment : fragments) {
      for (std::size_t i = 0; i < fragment.sources.size() + fragment.destinations.size(); ++i) {
        const auto &address = i < fragment.sources.size() ? fragment.sources[i]
            : fragment.destinations[i - fragment.sources.size()];
        const auto id = static_cast<std::int32_t>(address.index);
        std::memcpy(bytes.data() + (fragment.offset + i) * sizeof(id), &id, sizeof(id));
      }
    }
    HYLE_RETURN_IF_ERROR(mapping.flush());
    HYLE_RETURN_IF_ERROR(mapping.close());
    if (direct) {
      HYLE_ASSIGN_OR_RETURN(ready, writer.publish());
      remember();
      return Status::Ok();
    }
    HYLE_ASSIGN_OR_RETURN(source, staging.publish());
    HYLE_ASSIGN_OR_RETURN(auto input, source.view());
    HYLE_ASSIGN_OR_RETURN(auto output, writer.view());
    if (!queue.valid()) {
      // Original native cold transfer, not a second Queue or runtime. The
      // source prevalidates all ranges and succeeds only after actual native
      // retirement. Its native owners protect an accepted failing prefix.
      HYLE_ASSIGN_OR_RETURN(auto copy, native_copy(output, input));
      HYLE_ASSIGN_OR_RETURN(auto supported, gd::ExecutionSourceAccess::can_copy_constant(execution, copy));
      if (!supported) return Status::Unimplemented("original native source cannot upload slot metadata");
      const std::array copies{copy};
      if (options.resource_budget.valid()) {
        HYLE_ASSIGN_OR_RETURN(auto scratch_bytes, gd::ExecutionSourceAccess::constant_copy_scratch_bytes(execution, copies));
        HYLE_ASSIGN_OR_RETURN(native_scratch, options.resource_budget.reserve_unattributed(scratch_bytes));
      }
      const auto copied = gd::ExecutionSourceAccess::copy_constants(execution, copies, options.allocator);
      if (!copied.ok()) {
        // No retry/fallback/publication after possibly accepted native work.
        // Keep our original endpoints, private backing and scratch charge.
        failed = true;
        return copied;
      }
      native_scratch = {};
      HYLE_ASSIGN_OR_RETURN(ready, writer.publish());
      remember();
      return Status::Ok();
    }
    HYLE_ASSIGN_OR_RETURN(auto use, uses.begin());
    HYLE_RETURN_IF_ERROR(use.read(source));
    HYLE_RETURN_IF_ERROR(use.write(writer));
    HYLE_ASSIGN_OR_RETURN(auto sequence, cd::QueueSequence::begin(*metadata_queue));
    sequence.record(cd::TransferAccess::enqueue_copy(sequence, output, input));
    HYLE_ASSIGN_OR_RETURN(pending, sequence.finish());
    HYLE_FATAL_ASSERT(use.attach(pending).ok());
    const auto terminal = pending.wait();
    if (!terminal.ok() || uses.active_batches()) {
      failed = true;
      return terminal.ok() ? Status::Unavailable("slot metadata upload retirement is unproven") : terminal;
    }
    pending = {};
    HYLE_ASSIGN_OR_RETURN(ready, writer.publish());
    remember();
    return Status::Ok();
  }
};

struct ControlGroup final {
  explicit ControlGroup(mem::AllocatorRef allocator) noexcept
      : fragments(mem::StdAllocator<IndexFragment>{allocator}),
        uploaded_ids(mem::StdAllocator<std::int32_t>{allocator}) {}
  ExecutionSource execution;
  cd::QueueRef queue;
  ExecutionLaneId lane{};
  std::shared_ptr<c::Queue> metadata_queue;
  mem::MemoryResourceRef resource;
  Indices indices;
  std::vector<IndexFragment, mem::StdAllocator<IndexFragment>> fragments;
  std::vector<std::int32_t, mem::StdAllocator<std::int32_t>> uploaded_ids;
  std::size_t count = 0, participants = 0, outstanding = 0;
};
struct BindCandidate final {
  Find found{};
  SlotBindings *bindings = nullptr;
  detail::StoreConfiguration configuration{};
  bool configured = false, source_bound = false, effect_bound = false;
};
// Original cold metadata only. Each prepared region retains this allocator-
// owned object; it has neither a second execution source nor a resource payload.
struct ControlBatch final : util::RefCounted<ControlBatch> {
  explicit ControlBatch(mem::AllocatorRef selected) noexcept
      : allocator(selected), groups(mem::StdAllocator<ControlGroup>{selected}),
        candidates(mem::StdAllocator<BindCandidate>{selected}),
        transactions(mem::StdAllocator<SlotBindings *>{selected}),
        release_readers(mem::StdAllocator<ValueId>{selected}),
        release_writers(mem::StdAllocator<NodeId>{selected}) {}
  void destroy() noexcept { const auto selected = allocator; selected.destroy(this); }
  mem::AllocatorRef allocator;
  ResourceOptions options;
  std::vector<ControlGroup, mem::StdAllocator<ControlGroup>> groups;
  std::vector<BindCandidate, mem::StdAllocator<BindCandidate>> candidates;
  std::vector<SlotBindings *, mem::StdAllocator<SlotBindings *>> transactions;
  std::vector<ValueId, mem::StdAllocator<ValueId>> release_readers;
  std::vector<NodeId, mem::StdAllocator<NodeId>> release_writers;
};
} // namespace

namespace detail {
struct StoreBindingData final {
  explicit StoreBindingData(mem::AllocatorRef selected) noexcept
      : allocator(selected),
        writer_pointers(mem::StdAllocator<const c::WriteLease *>{selected}),
        source_nodes(mem::StdAllocator<NodeId>{selected}) {}
  void destroy() noexcept { const auto selected = allocator; selected.destroy(this); }
  mem::AllocatorRef allocator;
  SlotBindings bindings;
  util::Ref<ControlBatch> controls;
  std::size_t group = 0, fragment = 0, candidate = 0, offset = 0;
  ResourceOptions options;
  std::vector<const c::WriteLease *, mem::StdAllocator<const c::WriteLease *>> writer_pointers;
  std::vector<NodeId, mem::StdAllocator<NodeId>> source_nodes;
  StoreRegionFacts facts;
  StoreSource artifact_source;
  SourceMode mode = SourceMode::kTransaction;
  c::ProviderExtensionId source_capability{};
  bool logical_source = false;
  StoreConfiguration configuration;
  StoreConfiguration last_configuration;
  // Stable original state, not the movable public Invocation wrapper.
  const gd::InvocationState *owner = nullptr;
  gd::ProjectedWriter writer{};
};
static std::size_t generated_source_count(const StoreBindingData &data) noexcept {
  return data.mode == SourceMode::kImported ? 0 : data.facts.source.count;
}
static gd::ResourceProjection imported_projection(const StoreBindingData &data) noexcept {
  const auto &source = data.facts.source;
  return {source.capacity, source.indices, source.gather, data.source_nodes, data.source_capability};
}
struct StoreBindingAccess final {
  static StoreBindingData *get(StoreBinding &binding) noexcept { return binding.data_; }
  static void replace(StoreBinding &binding, StoreBindingData *data) noexcept {
    if (binding.data_) binding.data_->destroy();
    binding.data_ = data;
  }
};
StoreBinding::StoreBinding(StoreBinding &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)) {}
StoreBinding &StoreBinding::operator=(StoreBinding &&other) noexcept {
  if (this != &other) {
    StoreBinding next(std::move(other)); std::swap(data_, next.data_);
  }
  return *this;
}
StoreBinding::~StoreBinding() { if (data_) data_->destroy(); }
bool StoreBinding::prepared() const noexcept { return data_ != nullptr; }
bool StoreBinding::valid() const noexcept { return data_ && data_->bindings.valid(); }
bool StoreBinding::same_queue(const cd::QueueSequence &sequence) const noexcept {
  return data_ && data_->controls->groups[data_->group].queue.same_owner(sequence);
}
StoreConfiguration StoreBinding::configuration() const noexcept {
  return data_ ? data_->configuration : StoreConfiguration{};
}
const SlotBindings *StoreBinding::transaction() const noexcept {
  return valid() ? &data_->bindings : nullptr;
}
Result<StorePendingExport> StoreBinding::export_pending() const noexcept {
  if (!valid() || !data_->controls || data_->group >= data_->controls->groups.size())
    return std::unexpected(Status::FailedPrecondition("slot export has no original transaction/control group"));
  HYLE_RETURN_IF_ERROR(validate_authority());
  const auto &s = *data_;
  const auto &bindings = s.bindings;
  const auto writers = bindings.destinations();
  const auto addresses = bindings.destination_addresses();
  const auto layout = bindings.layout();
  const auto count = s.facts.destination_count;
  if (!count || writers.size() != count || addresses.size() != count || !layout.valid())
    return std::unexpected(Status::FailedPrecondition("slot export lost its populated destination geometry"));
  HYLE_ASSIGN_OR_RETURN(auto schema,
      Schema::create(layout.description(), layout.slot_count(), layout.stride_bytes()));
  if (schema != s.facts.schema)
    return std::unexpected(Status::InvalidArgument("slot export layout differs from its original Store"));
  HYLE_ASSIGN_OR_RETURN(auto description, schema.capacity_description());
  HYLE_ASSIGN_OR_RETURN(auto carrier, cd::LeaseAccess::slot_capacity(layout));
  HYLE_ASSIGN_OR_RETURN(auto capacity, c::ConstTensorView::create(description, carrier));
  if (!s.configuration.capacity.valid() || capacity.description() != s.configuration.capacity.description() ||
      !same_storage(capacity.storage(), s.configuration.capacity.storage()) || carrier.resource() != s.facts.resource)
    return std::unexpected(Status::InvalidArgument("slot export capacity differs from its original pool"));
  HYLE_ASSIGN_OR_RETURN(auto item_bytes, layout.description().required_bytes());
  for (std::size_t i = 0; i < count; ++i) {
    if (addresses[i].index >= layout.slot_count() || writers[i].generation() != addresses[i].generation ||
        writers[i].size() != item_bytes)
      return std::unexpected(Status::FailedPrecondition("slot export destination generation/extent changed"));
    const auto offset = std::size_t{addresses[i].index} * layout.stride_bytes();
    HYLE_ASSIGN_OR_RETURN(auto expected, carrier.subview(offset, item_bytes));
    HYLE_RETURN_IF_ERROR(cd::LeaseAccess::matches_writer_storage(writers[i], expected));
  }
  const auto generated = generated_source_count(s);
  if (generated > SIZE_MAX - s.offset ||
      s.offset + generated > SIZE_MAX / sizeof(std::int32_t) ||
      count > SIZE_MAX / sizeof(std::int32_t))
    return std::unexpected(Status::OutOfRange("slot export destination-ID extent overflows"));
  const auto offset = (s.offset + generated) * sizeof(std::int32_t);
  const auto bytes = count * sizeof(std::int32_t);
  const auto &ready = s.controls->groups[s.group].indices.ready;
  HYLE_ASSIGN_OR_RETURN(auto guard, ready.subview(offset, bytes));
  HYLE_ASSIGN_OR_RETURN(auto indices_description, c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(count)}));
  HYLE_ASSIGN_OR_RETURN(auto indices, guard.tensor(indices_description));
  if (!s.configuration.destination_indices.valid() ||
      indices.description() != s.configuration.destination_indices.description() ||
      !same_storage(indices.storage(), s.configuration.destination_indices.storage()))
    return std::unexpected(Status::InvalidArgument("slot export IDs differ from their original private slice"));
  return StorePendingExport{&bindings, capacity, indices, std::move(guard)};
}
Status StoreBinding::validate_authority() const noexcept {
  if (!valid() || !data_->controls->groups[data_->group].indices.ready.valid())
    return Status::FailedPrecondition("slot store has no current original transaction and controls");
  HYLE_ASSIGN_OR_RETURN(auto bound,
      gd::RegionAccess::has_write_effect(data_->owner, data_->writer, data_->writer_pointers));
  if (!bound)
    return Status::FailedPrecondition("slot store original write permissions were reset or withdrawn");
  const auto input = [&](ValueId value, c::ConstTensorView expected) noexcept -> Status {
    HYLE_ASSIGN_OR_RETURN(auto matches, gd::RegionAccess::has_input_binding(data_->owner, value, expected));
    return matches ? Status::Ok() : Status::FailedPrecondition("slot store original pool/IDs input was replaced or withdrawn");
  };
  HYLE_RETURN_IF_ERROR(input(data_->facts.destination_indices, data_->configuration.destination_indices));
  if (data_->facts.source.count) {
    if (data_->mode == SourceMode::kImported) {
      HYLE_ASSIGN_OR_RETURN(auto source, gd::RegionAccess::store_source_projection(
          data_->owner, imported_projection(*data_)));
      if (!same_configuration({source.capacity, source.indices, data_->configuration.destination_indices},
          data_->configuration))
        return Status::FailedPrecondition("slot store imported source projection was reset or replaced");
    } else {
      HYLE_RETURN_IF_ERROR(input(data_->facts.source.capacity, data_->configuration.capacity));
      HYLE_RETURN_IF_ERROR(input(data_->facts.source.indices, data_->configuration.source_indices));
      HYLE_ASSIGN_OR_RETURN(auto source, gd::RegionAccess::has_read_projection(
          data_->owner, data_->facts.source.capacity, data_->bindings.sources()));
      if (!source)
        return Status::FailedPrecondition("slot store original source permissions were reset or withdrawn");
    }
  }
  return Status::Ok();
}
Status StoreBinding::validate_configuration(const StoreConfiguration &actual) const noexcept {
  HYLE_RETURN_IF_ERROR(validate_authority());
  if (!same_configuration(actual, data_->configuration))
    return Status::InvalidArgument("slot store differs from its private validated capacity/IDs");
  return Status::Ok();
}
} // namespace detail

namespace {

// A byte-copy program of the original backend, not a second executor or slot
// owner. Its Context is retained through the same cold backend capability.
struct NativeStore final {
  NativeStore() noexcept = default;
  NativeStore(const NativeStore &) = delete;
  NativeStore &operator=(const NativeStore &) = delete;
  cd::BackendExtensionRef extension;
  const cd::IndexedCopyBackendVTable *table = nullptr;
  void *program = nullptr;
  mem::AllocatorRef allocator{};

  ~NativeStore() { reset(); }
  bool valid() const noexcept { return program != nullptr; }
  void reset() noexcept {
    if (program) table->destroy(std::exchange(program, nullptr), allocator);
    table = nullptr; extension = {};
  }
  Status prepare(cd::BackendExtensionRef owner, cd::IndexedCopyNativeSpec spec,
      mem::AllocatorRef selected) noexcept {
    extension = std::move(owner); allocator = selected;
    struct Call { NativeStore *self; cd::IndexedCopyNativeSpec spec; } call{this, spec};
    const auto status = cd::BackendExtensionAccess::call(extension,
        [](void *context, const void *opaque, void *user) noexcept -> Status {
      auto &call = *static_cast<Call *>(user);
      const auto *interface = static_cast<const cd::IndexedCopyBackendVTable *>(opaque);
      if (!interface || !interface->prepare || !interface->destroy || !interface->enqueue ||
          !interface->supports_command_capture)
        return Status::Unimplemented("backend has no capturable indexed copy");
      call.self->table = interface;
      return interface->prepare(context, call.spec, call.self->allocator, &call.self->program);
    }, &call);
    if (!status.ok()) { reset(); return status; }
    if (!valid()) { reset(); return Status::Internal("indexed copy returned no native program"); }
    return Status::Ok();
  }
  cd::EnqueueResult enqueue(cd::QueueSequence &sequence,
      cd::IndexedCopyNativeBindings bindings) const noexcept {
    struct Call { const NativeStore *self; cd::IndexedCopyNativeBindings bindings; } call{this, bindings};
    return cd::BackendExtensionAccess::enqueue_admitted(extension, sequence,
        [](void *context, const void *, void *queue, void *user) noexcept {
      const auto &call = *static_cast<Call *>(user);
      return call.self->table->enqueue(context, queue, call.self->program, call.bindings);
    }, &call);
  }
};

struct Prepared final {
  Store store;
  NodeId node;
  ValueId rows_value, ids_value;
  c::TensorDesc rows, ids;
  std::uint32_t rows_port = 0, ids_port = 1;
  mem::MemoryResourceRef resource;
  c::ProviderRuntimeProperties runtime;
  NativeStore native;
};
struct Slot final {
  explicit Slot(mem::AllocatorRef allocator) noexcept
      : destinations(mem::StdAllocator<mem::BufferView>{allocator}) {}
  detail::StoreBinding binding;
  std::vector<mem::BufferView, mem::StdAllocator<mem::BufferView>> destinations;
  c::ConstTensorView rows, ids;
  detail::StoreConfiguration configuration;
  // Keeps only the addressable allocation alive after take(), never an old
  // read/write lease or authority to submit another episode.
  c::LeaseSlotLayout carrier;
  bool sealed = false;
};

Status validate_rows(const Prepared &p, Slot &s,
    c::ConstTensorView rows, c::ConstTensorView ids) noexcept {
  if (rows.description() != p.rows || ids.description() != p.ids)
    return Status::FailedPrecondition("slot store has no current exact transaction and controls");
  auto configuration = s.binding.configuration();
  configuration.destination_indices = ids;
  HYLE_RETURN_IF_ERROR(s.binding.validate_configuration(configuration));
  if (p.native.valid() && !same_configuration(configuration, s.configuration))
    return Status::FailedPrecondition("indexed store private carriers changed without configuration");
  const auto &bindings = *s.binding.transaction();
  const auto input = rows.storage();
  if (input.resource() != p.resource || input.offset() > SIZE_MAX - input.size())
    return Status::InvalidArgument("slot store rows differ from its original Queue storage");
  for (std::size_t i = 0; i < bindings.destination_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto target, bindings.destinations()[i].view());
    if (target.resource() != p.resource || target.offset() > SIZE_MAX - target.size())
      return Status::InvalidArgument("slot store destination range changed");
    if (input.native_handle() == target.native_handle() &&
        input.offset() < target.offset() + target.size() &&
        target.offset() < input.offset() + input.size())
      return Status::InvalidArgument("slot store rows overlap an original destination");
    s.destinations[i] = target;
  }
  if (p.native.valid()) s.carrier = bindings.layout();
  return Status::Ok();
}

const gd::RegionBindingReleaseVTable binding_release{
    [](void *, void *) noexcept { return Status::Ok(); },
    [](void *prepared, void *opaque) noexcept {
      auto &slot = *static_cast<Slot *>(opaque);
      if (!static_cast<Prepared *>(prepared)->native.valid()) { slot.rows = {}; slot.ids = {}; }
      slot.sealed = false;
    }, true};
const gd::RegionCaptureBindingsVTable capture_bindings{
    [](void *prepared, void *opaque) noexcept {
      if (!prepared || !opaque) return false;
      const auto &p = *static_cast<Prepared *>(prepared);
      const auto &s = *static_cast<Slot *>(opaque);
      return p.native.valid() && s.carrier.valid() && s.rows.valid() && s.ids.valid() &&
          s.configuration.capacity.valid() && s.configuration.destination_indices.valid();
    }};
const gd::RegionProjectedWriteVTable projected_write{
    [](const ProviderRegionPrepared &prepared, void *opaque) noexcept {
      return prepared.state && opaque && static_cast<Slot *>(opaque)->sealed &&
          capture_bindings.preserves_private_arguments(prepared.state, opaque);
    }};
const detail::StoreRegionVTable store_binding{
    .inspect = [](const ProviderRegionPrepared &prepared, void *invocation) noexcept
        -> Result<detail::StoreRegionFacts> {
      if (!prepared.state || !invocation)
        return std::unexpected(Status::InvalidArgument("slot store needs its original artifact/slot"));
      const auto &p = *static_cast<Prepared *>(prepared.state);
      return detail::StoreRegionFacts{.store = p.node, .schema = p.store.schema,
          .destination_indices = p.ids_value, .destination_input_port = p.ids_port,
          .destination_count = static_cast<std::size_t>(p.rows.dimensions()[0]),
          .resource = p.resource,
          .permission_changes_preserve_commands = p.native.valid()};
    },
    .binding = [](void *invocation) noexcept {
      return &static_cast<Slot *>(invocation)->binding;
    },
    .configure = [](const ProviderRegionPrepared &prepared, void *invocation,
        const detail::StoreConfiguration &configuration) noexcept -> Status {
      const auto &p = *static_cast<Prepared *>(prepared.state);
      HYLE_ASSIGN_OR_RETURN(auto capacity_description, p.store.schema.capacity_description());
      if (!configuration.capacity.valid() || configuration.capacity.description() != capacity_description ||
          configuration.capacity.storage().resource() != p.resource ||
          !configuration.destination_indices.valid() || configuration.destination_indices.description() != p.ids)
        return Status::InvalidArgument("slot store private configuration differs");
      auto &s = *static_cast<Slot *>(invocation);
      s.configuration = configuration;
      if (!p.native.valid()) { s.rows = {}; s.ids = {}; }
      s.sealed = false;
      return Status::Ok();
    },
    .clear_configuration = [](const ProviderRegionPrepared &prepared, void *invocation) noexcept {
      auto &s = *static_cast<Slot *>(invocation);
      if (!static_cast<Prepared *>(prepared.state)->native.valid()) {
        s.rows = {}; s.ids = {}; s.configuration = {};
      }
      s.sealed = false;
    }};
const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Prepared *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return c::AliasSupport{}; },
    .create_invocation = [](void *opaque, mem::AllocatorRef a, mem::BufferView, void **out) noexcept -> Status {
      *out = nullptr;
      auto *slot = a.construct<Slot>(a);
      if (!slot) return Status::ResourceExhausted("slot store invocation metadata");
      try {
        const auto count = static_cast<std::size_t>(static_cast<Prepared *>(opaque)->rows.dimensions()[0]);
        slot->destinations.resize(count);
      }
      catch (...) { a.destroy(slot); return Status::ResourceExhausted("slot store selected writer pointers"); }
      *out = slot;
      return Status::Ok();
    },
    .destroy_invocation = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Slot *>(p)); },
    .shared_submit = [](void *opaque, void *invocation, cd::QueueSequence &sequence,
        std::span<const c::ConstTensorView> inputs, std::span<const c::TensorView> outputs) noexcept -> cd::EnqueueResult {
      const auto &p = *static_cast<Prepared *>(opaque);
      auto &s = *static_cast<Slot *>(invocation);
      if (inputs.size() != 2 || !outputs.empty() || !s.sealed || !s.binding.valid() ||
          !s.binding.same_queue(sequence))
        return cd::EnqueueResult::rejected(Status::FailedPrecondition("slot store needs its original sealed Queue episode"));
      if (inputs[p.rows_port].description() != p.rows || inputs[p.ids_port].description() != p.ids ||
          !same_storage(inputs[p.rows_port].storage(), s.rows.storage()) ||
          !same_storage(inputs[p.ids_port].storage(), s.ids.storage()))
        return cd::EnqueueResult::rejected(Status::FailedPrecondition("slot store inputs differ from their admitted cold spans"));
      if (p.native.valid())
        return p.native.enqueue(sequence, {s.configuration.capacity.storage(),
            s.rows.storage(), s.ids.storage()});
      const auto bytes = p.rows.required_bytes().value() /
          static_cast<std::size_t>(p.rows.dimensions()[0]);
      bool accepted = false;
      for (std::size_t i = 0; i < s.destinations.size(); ++i) {
        auto source = inputs[p.rows_port].storage().subview(i * bytes, bytes);
        if (!source) return {accepted, source.error()};
        auto command = cd::TransferAccess::enqueue_copy(sequence, s.destinations[i], *source);
        accepted |= command.accepted;
        if (!command.status.ok()) return {accepted, std::move(command.status)};
      }
      return {accepted, Status::Ok()};
    },
    .prepare_bindings = [](void *opaque, void *invocation,
        std::span<const RegionValue> inputs, std::span<const c::TensorView> outputs) noexcept -> Status {
      const auto &p = *static_cast<Prepared *>(opaque);
      auto &s = *static_cast<Slot *>(invocation); s.sealed = false;
      if (inputs.size() != 2 || !outputs.empty()) return Status::InvalidArgument("slot store boundary changed");
      HYLE_RETURN_IF_ERROR(validate_rows(p, s, inputs[p.rows_port].tensor, inputs[p.ids_port].tensor));
      s.rows = inputs[p.rows_port].tensor; s.ids = inputs[p.ids_port].tensor;
      return Status::Ok();
    },
    .seal_bindings = [](void *opaque, void *invocation) noexcept -> Status {
      auto &s = *static_cast<Slot *>(invocation);
      HYLE_RETURN_IF_ERROR(validate_rows(*static_cast<Prepared *>(opaque), s, s.rows, s.ids));
      s.sealed = true; return Status::Ok();
    },
    .query_extension = [](void *, c::ProviderExtensionId id) noexcept -> const void * {
      if (id == gd::kRegionBindingReleaseExtensionId) return &binding_release;
      if (id == detail::kStoreBindingExtensionId) return &store_binding;
      if (id == gd::kRegionCaptureBindingsExtensionId) return &capture_bindings;
      if (id == gd::kRegionProjectedWriteExtensionId) return &projected_write;
      return nullptr;
    }};

struct Description final {
  Store store;
  ValueId rows, ids;
  c::TensorDesc rows_description, ids_description;
  std::uint32_t rows_port = 0, ids_port = 1;
};
Result<Description> describe(const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.nodes.size() != 1 || request.inputs.size() != 2 ||
      !request.outputs.empty() || !request.has_unknown_effects)
    return std::unexpected(Status::Unimplemented("slot store needs its original external-write singleton"));
  const auto &graph = *request.graph;
  HYLE_ASSIGN_OR_RETURN(auto operation, graph.operation(request.nodes[0]));
  const auto store = operation.custom_as<Store>();
  if (!store) return std::unexpected(Status::Unimplemented("not a slot store"));
  Description result; result.store = *store;
  HYLE_ASSIGN_OR_RETURN(result.rows, graph.node_input(request.nodes[0], 0));
  HYLE_ASSIGN_OR_RETURN(result.ids, graph.node_input(request.nodes[0], 1));
  if (result.ids.kind() != ValueKind::kInput)
    return std::unexpected(Status::Unimplemented("slot store IDs must be its private caller input"));
  for (std::size_t i = 0; i < graph.output_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto output, graph.output_id(i));
    if (output == result.ids) return std::unexpected(Status::Unimplemented("private slot IDs cannot be published"));
  }
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    if (node == request.nodes[0]) continue;
    HYLE_ASSIGN_OR_RETURN(auto count, graph.node_input_count(node));
    for (std::size_t port = 0; port < count; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto input, graph.node_input(node, port));
      if (input == result.ids)
        return std::unexpected(Status::Unimplemented("slot store IDs are shared with another node"));
    }
  }
  HYLE_ASSIGN_OR_RETURN(result.rows_description, graph.value_description(result.rows));
  HYLE_ASSIGN_OR_RETURN(result.ids_description, graph.value_description(result.ids));
  const std::array descriptions{result.rows_description, result.ids_description};
  HYLE_RETURN_IF_ERROR(store->infer_output_descriptions(descriptions, {}));
  const auto rows = std::ranges::find(request.inputs, result.rows);
  const auto ids = std::ranges::find(request.inputs, result.ids);
  if (rows == request.inputs.end() || ids == request.inputs.end())
    return std::unexpected(Status::InvalidArgument("slot store original input boundary changed"));
  result.rows_port = static_cast<std::uint32_t>(rows - request.inputs.begin());
  result.ids_port = static_cast<std::uint32_t>(ids - request.inputs.begin());
  return result;
}
c::ProviderRuntimeProperties runtime(const detail::StoreAdapter &adapter) noexcept {
  return {.mode = c::ProviderRuntimeMode::kSharedQueue,
      .workspace_owner = c::ProviderWorkspaceOwner::kCaller,
      .replay_class = c::ProviderReplayClass::kPrepared,
      .blocks_host = adapter.device_class == c::DeviceClass::kCpu,
      .uses_private_threads = false, .supports_concurrent_invocations = true,
      .capture_compatible = adapter.indexed_copy.valid()};
}
Status propose(void *, const RegionDiscoveryRequest &request, mem::AllocatorRef,
    RegionProposalWriter &writer) noexcept {
  for (std::size_t i = 0; i < request.graph.node_count(); ++i) {
    if (!request.allowed_nodes.empty() && !request.allowed_nodes[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, request.graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto kind, request.graph.operation_kind(node));
    if (kind != c::OperationKind::kCustom) continue;
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph.operation(node));
    if (!operation.custom_as<Store>()) continue;
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added, writer.try_add({.nodes = nodes, .route = 1,
        .replay_class = c::ProviderReplayClass::kPrepared,
        .implementation = c::ProviderImplementationId{0x534c4f5453544f52ULL}}));
    if (!added) break;
  }
  return Status::Ok();
}
Status resources(void *opaque, const RegionCompileRequest &request, mem::AllocatorRef,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *out) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto description, describe(request));
  (void)description;
  if (inputs.size() != 2 || !outputs.empty()) return Status::Unimplemented("slot store resource boundary changed");
  const auto &adapter = *static_cast<detail::StoreAdapter *>(opaque);
  for (auto &input : inputs) {
    input.resource.required_properties = adapter.device_class == c::DeviceClass::kCpu
        ? mem::MemoryProperty::kHostAccessible : mem::MemoryProperty::kDeviceAccessible;
    input.resource.minimum_alignment = 1;
  }
  *out = {.runtime = runtime(adapter), .preserves_effects = true};
  return Status::Ok();
}
c::ProviderPrepareOutcome compile(void *opaque, const RegionCompileRequest &request,
    mem::AllocatorRef allocator, ProviderRegionPrepared *out) noexcept {
  *out = {};
  auto description = describe(request);
  if (!description) return description.error().IsUnimplemented()
      ? c::ProviderPrepareOutcome::unsupported(description.error())
      : c::ProviderPrepareOutcome::error(description.error());
  const auto &adapter = *static_cast<detail::StoreAdapter *>(opaque);
  if (request.input_resources.size() != 2 ||
      std::ranges::any_of(request.input_resources, [&](const auto &resource) { return resource != adapter.resource; }))
    return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("slot store needs its original Queue resources"));
  auto *prepared = allocator.construct<Prepared>();
  if (!prepared) return c::ProviderPrepareOutcome::error(Status::ResourceExhausted("slot store prepared metadata"));
  prepared->store = description->store; prepared->node = request.nodes[0];
  prepared->rows_value = description->rows; prepared->ids_value = description->ids;
  prepared->rows = description->rows_description; prepared->ids = description->ids_description;
  prepared->rows_port = description->rows_port; prepared->ids_port = description->ids_port;
  prepared->resource = adapter.resource; prepared->runtime = runtime(adapter);
  if (adapter.indexed_copy.valid()) {
    const auto bytes = prepared->rows.required_bytes();
    if (!bytes) { allocator.destroy(prepared); return c::ProviderPrepareOutcome::error(bytes.error()); }
    const auto count = static_cast<std::uint32_t>(prepared->rows.dimensions()[0]);
    const cd::IndexedCopyNativeSpec spec{prepared->store.schema.slot_count(), count,
        *bytes / count, prepared->store.schema.stride_bytes()};
    const auto native = prepared->native.prepare(adapter.indexed_copy, spec, allocator);
    if (!native.ok() && !native.IsUnimplemented()) {
      allocator.destroy(prepared); return c::ProviderPrepareOutcome::error(native);
    }
    prepared->runtime.capture_compatible = prepared->native.valid();
  }
  *out = {.vtable = &prepared_table, .state = prepared,
      .properties = {.workspace = {0, 1}, .persistent_bytes = sizeof(Prepared),
          .dispatch_count = prepared->native.valid() ? 2u : static_cast<std::uint32_t>(prepared->rows.dimensions()[0]),
          .runtime = prepared->runtime, .preserves_effects = true,
          .invocation_private_bytes = std::nullopt}};
  return c::ProviderPrepareOutcome::prepared();
}
const RegionProviderVTable provider_table{&propose, &resources, &compile};
struct LogicalSource final {
  NodeId store;
  Schema schema;
  ValueId destination_indices;
  std::size_t destination_count = 0;
  detail::StoreSource source;
};
bool same_source(const detail::StoreSource &a, const detail::StoreSource &b) noexcept {
  return a.gather == b.gather && a.capacity == b.capacity && a.indices == b.indices &&
      a.count == b.count && a.capacity_input_port == b.capacity_input_port;
}
bool same_logical_source(const detail::StoreSource &a, const detail::StoreSource &b) noexcept {
  return a.gather == b.gather && a.capacity == b.capacity && a.indices == b.indices && a.count == b.count;
}
Status private_input(const Graph &graph, ValueId value, NodeId reader,
    std::size_t reader_port) noexcept {
  if (!value.valid() || value.kind() != ValueKind::kInput)
    return Status::InvalidArgument("slot pool and controls need original Graph inputs");
  for (std::size_t i = 0; i < graph.output_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto output, graph.output_id(i));
    if (output == value) return Status::InvalidArgument("slot pool/control cannot be published");
  }
  bool consumed = false;
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto count, graph.node_input_count(node));
    for (std::size_t port = 0; port < count; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto input, graph.node_input(node, port));
      if (input != value) continue;
      if (node != reader || port != reader_port)
        return Status::InvalidArgument("slot pool/control has an unrelated logical consumer");
      consumed = true;
    }
  }
  return consumed ? Status::Ok() : Status::InvalidArgument("slot pool/control has no original reader");
}
Result<LogicalSource> describe_source(const Graph &graph, SourceBinding binding) noexcept {
  if (!graph.valid() || !binding.store.valid() || !binding.gather.valid() || binding.store == binding.gather)
    return std::unexpected(Status::InvalidArgument("slot source association needs its original Graph and distinct nodes"));
  HYLE_ASSIGN_OR_RETURN(auto store_operation, graph.operation(binding.store));
  const auto store = store_operation.custom_as<Store>();
  if (!store) return std::unexpected(Status::InvalidArgument("slot source association needs an original Store"));
  HYLE_ASSIGN_OR_RETURN(auto gather_operation, graph.operation(binding.gather));
  const auto *gather = gather_operation.get_if<c::Gather>();
  if (!gather || (gather->axis != 0 && gather->axis != -2) ||
      gather->bounds != c::IndexBoundsPolicy::kChecked)
    return std::unexpected(Status::InvalidArgument("slot source needs a checked axis-zero Gather"));
  HYLE_ASSIGN_OR_RETURN(auto pool, graph.node_input(binding.gather, 0));
  HYLE_ASSIGN_OR_RETURN(auto source_ids, graph.node_input(binding.gather, 1));
  HYLE_ASSIGN_OR_RETURN(auto destination_ids, graph.node_input(binding.store, 1));
  if (pool == source_ids || pool == destination_ids || source_ids == destination_ids)
    return std::unexpected(Status::InvalidArgument("slot source and destination controls must be distinct"));
  HYLE_RETURN_IF_ERROR(private_input(graph, pool, binding.gather, 0));
  HYLE_RETURN_IF_ERROR(private_input(graph, source_ids, binding.gather, 1));
  HYLE_RETURN_IF_ERROR(private_input(graph, destination_ids, binding.store, 1));
  HYLE_ASSIGN_OR_RETURN(auto capacity_description, graph.value_description(pool));
  HYLE_ASSIGN_OR_RETURN(auto expected_capacity, store->schema.capacity_description());
  HYLE_ASSIGN_OR_RETURN(auto ids_description, graph.value_description(source_ids));
  HYLE_ASSIGN_OR_RETURN(auto gathered, graph.result(binding.gather, 0));
  HYLE_ASSIGN_OR_RETURN(auto gathered_description, graph.value_description(gathered));
  if (capacity_description != expected_capacity ||
      !ids_description.is_plain() || !ids_description.is_dense() ||
      ids_description.data_type() != c::DataType::kInt32 || ids_description.rank() != 1 ||
      ids_description.dimensions()[0] <= 0 || ids_description.dimensions()[0] > INT32_MAX)
    return std::unexpected(Status::InvalidArgument("slot Gather pool/IDs differ from Store geometry"));
  const auto source_count = static_cast<std::size_t>(ids_description.dimensions()[0]);
  HYLE_ASSIGN_OR_RETURN(auto expected_gathered, c::TensorDesc::dense(store->schema.type(),
      std::array<std::int64_t, 2>{static_cast<std::int64_t>(source_count), store->schema.item_elements()}));
  if (gathered_description != expected_gathered)
    return std::unexpected(Status::InvalidArgument("slot Gather must materialize same-type whole rows"));
  HYLE_ASSIGN_OR_RETURN(auto rows, graph.node_input(binding.store, 0));
  HYLE_ASSIGN_OR_RETURN(auto rows_description, graph.value_description(rows));
  HYLE_ASSIGN_OR_RETURN(auto destination_description, graph.value_description(destination_ids));
  const std::array inputs{rows_description, destination_description};
  HYLE_RETURN_IF_ERROR(store->infer_output_descriptions(inputs, {}));
  return LogicalSource{binding.store, store->schema, destination_ids,
      static_cast<std::size_t>(rows_description.dimensions()[0]),
      {.gather = binding.gather, .capacity = pool, .indices = source_ids, .count = source_count}};
}
Status matches(const detail::StoreRegionFacts &facts, const LogicalSource &logical) noexcept {
  if (facts.store != logical.store || facts.schema != logical.schema ||
      facts.destination_indices != logical.destination_indices || facts.destination_count != logical.destination_count ||
      (facts.source.count && !same_logical_source(facts.source, logical.source)))
    return Status::InvalidArgument("slot logical association differs from its selected Store/source artifact");
  return Status::Ok();
}
const detail::StoreRegionVTable *store_interface(const gd::RegionAccessView &view) noexcept {
  return view.prepared_vtable && view.prepared_vtable->query_extension
      ? static_cast<const detail::StoreRegionVTable *>(view.prepared_vtable->query_extension(
          view.prepared, detail::kStoreBindingExtensionId)) : nullptr;
}
Result<detail::StoreRegionFacts> inspect(const gd::RegionAccessView &view,
    const detail::StoreRegionVTable &interface) noexcept {
  if (!view.prepared || !view.invocation || !interface.inspect || !interface.binding ||
      !interface.configure || !interface.clear_configuration)
    return std::unexpected(Status::InvalidArgument("slot store cold extension is incomplete"));
  const ProviderRegionPrepared prepared{.vtable = view.prepared_vtable, .state = view.prepared};
  HYLE_ASSIGN_OR_RETURN(auto facts, interface.inspect(prepared, view.invocation));
  if (!facts.store.valid() || !facts.schema.valid() || !facts.resource.valid() ||
      !facts.destination_indices.valid() || facts.destination_indices.kind() != ValueKind::kInput ||
      facts.destination_input_port == UINT32_MAX || !facts.destination_count ||
      facts.destination_count > facts.schema.slot_count() ||
      facts.source.count > static_cast<std::size_t>(INT32_MAX) ||
      facts.source.count > SIZE_MAX - facts.destination_count ||
      std::ranges::find(view.node_ordinals, facts.store.ordinal()) == view.node_ordinals.end())
    return std::unexpected(Status::InvalidArgument("slot store cold geometry/artifact facts differ"));
  if (facts.source.count && (!facts.source.gather.valid() ||
      !facts.source.capacity.valid() || !facts.source.indices.valid() ||
      facts.source.capacity.kind() != ValueKind::kInput ||
      facts.source.indices.kind() != ValueKind::kInput ||
      facts.source.capacity == facts.source.indices ||
      facts.source.capacity == facts.destination_indices ||
      facts.source.indices == facts.destination_indices ||
      facts.source.capacity_input_port == UINT32_MAX ||
      std::ranges::find(view.node_ordinals, facts.source.gather.ordinal()) == view.node_ordinals.end()))
    return std::unexpected(Status::InvalidArgument("slot source cold reader/artifact facts differ"));
  return facts;
}
Status find(void *opaque, const gd::RegionAccessView &view) noexcept {
  auto &found = *static_cast<Find *>(opaque);
  const auto *interface = store_interface(view);
  if (!interface) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto facts, inspect(view, *interface));
  if (facts.store != found.node) return Status::Ok();
  if (found.binding)
    return Status::InvalidArgument("slot store selected artifact/slot identity differs");
  found.binding = interface->binding(view.invocation);
  if (!found.binding) return Status::InvalidArgument("slot store has no original binding owner");
  found.interface = interface; found.prepared = {.vtable = view.prepared_vtable, .state = view.prepared};
  found.invocation = view.invocation; found.facts = facts;
  found.proof = {found.node, view.step, view.prepared_vtable, view.prepared}; found.lane = view.lane;
  return Status::Ok();
}
} // namespace

namespace detail {
const RegionProviderVTable *store_provider() noexcept { return &provider_table; }
Result<gd::ResourceProjection> source_projection(Invocation &invocation, NodeId node) noexcept {
  Find selected{node};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation, node, &find, &selected));
  if (!selected.binding)
    return std::unexpected(Status::Unimplemented("Store has no selected original source binding"));
  const auto *data = StoreBindingAccess::get(*selected.binding);
  if (!data || data->mode != SourceMode::kImported || !data->facts.source.count ||
      data->source_nodes.empty() || !data->source_capability.valid())
    return std::unexpected(Status::FailedPrecondition("Store needs its cold imported source preparation"));
  return imported_projection(*data);
}
} // namespace detail

namespace {
Status prepare_impl(Invocation &invocation, const Graph *graph,
    std::span<const QueueBinding> queues, std::span<const SourceBinding> sources,
    PrepareOptions options) noexcept {
  if (!options.allocator.valid() || !options.host_workspace.valid())
    return Status::InvalidArgument("slot preparation needs allocator and host workspace");
  // Reject foreign/duplicate logical associations before replacing any slot's
  // existing carriers. Selected arithmetic remains owned by its original plan.
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (!graph) return Status::InvalidArgument("slot source associations need their original Graph");
    if (sources[i].mode != SourceMode::kTransaction && sources[i].mode != SourceMode::kImported)
      return Status::InvalidArgument("slot source association mode is invalid");
    HYLE_ASSIGN_OR_RETURN(auto logical, describe_source(*graph, sources[i]));
    Find selected{sources[i].store};
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation, sources[i].store, &find, &selected));
    if (!selected.binding) return Status::Unimplemented("associated Store has no selected slot implementation");
    HYLE_RETURN_IF_ERROR(matches(selected.facts, logical));
    for (std::size_t j = 0; j < i; ++j) {
      if (sources[i].store == sources[j].store || sources[i].gather == sources[j].gather)
        return Status::InvalidArgument("slot source association duplicates a Store, reader or pool");
      HYLE_ASSIGN_OR_RETURN(auto previous_pool, graph->node_input(sources[j].gather, 0));
      if (logical.source.capacity == previous_pool)
        return Status::InvalidArgument("slot source association duplicates a Store, reader or pool");
    }
  }
  const ResourceOptions resources{options.allocator, options.resource_budget};
  HYLE_RETURN_IF_ERROR(invocation.prepare_resources(resources));
  auto allocator = resources.allocator;
  if (resources.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, resources.resource_budget.metadata_allocator());
  }
  struct Pending final {
    Find found{};
    detail::StoreBinding next;
    ExecutionSource execution;
    cd::QueueRef queue;
    ExecutionLaneId lane{};
  };
  std::vector<Pending, mem::StdAllocator<Pending>> pending{mem::StdAllocator<Pending>{allocator}};
  struct Call final {
    Invocation *invocation; const Graph *graph;
    std::span<const QueueBinding> queues; std::span<const SourceBinding> sources;
    mem::AllocatorRef allocator;
    std::vector<Pending, mem::StdAllocator<Pending>> *pending;
  } call{&invocation, graph, queues, sources, allocator, &pending};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,
      [](void *opaque, const gd::RegionAccessView &view) noexcept -> Status {
    const auto *interface = store_interface(view);
    if (!interface) return Status::Ok();
    const auto &call = *static_cast<Call *>(opaque);
    HYLE_ASSIGN_OR_RETURN(auto facts, inspect(view, *interface));
    const auto artifact_source = facts.source;
    bool logical_source = false;
    const auto association = std::ranges::find(call.sources, facts.store, &SourceBinding::store);
    if (association != call.sources.end()) {
      HYLE_ASSIGN_OR_RETURN(auto logical, describe_source(*call.graph, *association));
      HYLE_RETURN_IF_ERROR(matches(facts, logical));
      if (!facts.source.count) { facts.source = logical.source; logical_source = true; }
    }
    auto *binding = interface->binding(view.invocation);
    if (!binding) return Status::InvalidArgument("slot store has no original binding owner");
    if (binding->valid()) return Status::FailedPrecondition("take original slot bindings before preparation");
    HYLE_ASSIGN_OR_RETURN(auto execution,
        gd::RegionAccess::execution_source(call.invocation->executable(), view.lane));
    HYLE_RETURN_IF_ERROR(execution.validate_resource(facts.resource));
    const auto context = gd::ExecutionSourceAccess::compute(execution);
    if (!context && !context.error().IsUnimplemented()) return context.error();
    const bool shared_queue = context.has_value();
    const auto first = std::ranges::find_if(call.queues, [&](const auto &entry) { return entry.lane == view.lane; });
    cd::QueueRef queue;
    if (shared_queue) {
      if (first == call.queues.end() || !first->queue ||
          std::count_if(call.queues.begin(), call.queues.end(), [&](const auto &entry) { return entry.lane == view.lane; }) != 1)
        return Status::InvalidArgument("slot preparation needs one original Queue per shared lane");
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::validate_queue(call.invocation->executable(), view.lane, *first->queue));
      HYLE_ASSIGN_OR_RETURN(queue, cd::QueueRef::acquire(*first->queue));
    } else if (first != call.queues.end()) {
      return Status::InvalidArgument("native slot source must not use a Compute Queue binding");
    }
    auto *data = call.allocator.construct<detail::StoreBindingData>(call.allocator);
    if (!data) return Status::ResourceExhausted("slot store original binding metadata");
    Pending item;
    item.lane = view.lane;
    detail::StoreBindingAccess::replace(item.next, data);
    try { data->writer_pointers.resize(facts.destination_count); }
    catch (...) { return Status::ResourceExhausted("slot store selected writer pointers"); }
    data->facts = facts; data->artifact_source = artifact_source; data->logical_source = logical_source;
    if (association != call.sources.end()) data->mode = association->mode;
    if (data->mode == SourceMode::kImported) {
      data->source_capability = logical_source ? cd::kProjectedInputReadExtensionId
          : detail::kStoreBindingExtensionId;
      try {
        if (logical_source) data->source_nodes.push_back(facts.source.gather);
        else {
          data->source_nodes.reserve(view.node_ordinals.size());
          for (const auto ordinal : view.node_ordinals) {
            HYLE_ASSIGN_OR_RETURN(auto node, call.graph->node_id(ordinal));
            data->source_nodes.push_back(node);
          }
        }
      } catch (...) { return Status::ResourceExhausted("slot imported source access metadata"); }
      HYLE_ASSIGN_OR_RETURN(auto source_resource, gd::RegionAccess::projection_resource(
          call.invocation->executable(), detail::imported_projection(*data)));
      if (source_resource != facts.resource)
        return Status::InvalidArgument("slot imported reader belongs to another original resource");
    }
    item.found = {.node = facts.store, .binding = binding, .interface = interface,
        .prepared = {.vtable = view.prepared_vtable, .state = view.prepared},
        .invocation = view.invocation, .facts = facts,
        .proof = {facts.store, view.step, view.prepared_vtable, view.prepared}, .lane = view.lane};
    item.execution = std::move(execution); item.queue = std::move(queue);
    try { call.pending->push_back(std::move(item)); }
    catch (...) { return Status::ResourceExhausted("slot selected preparation metadata"); }
    return Status::Ok();
  }, &call));
  if (pending.empty()) return Status::Ok();
  auto *allocated = allocator.construct<ControlBatch>(allocator);
  if (!allocated) return Status::ResourceExhausted("slot carrier grouping metadata");
  auto controls = util::adopt_ref(allocated);
  controls->options = resources;
  // Native cold-copy helpers allocate their own temporary metadata through
  // this allocator; their separate scratch query accounts only payload bytes.
  controls->options.allocator = allocator;
  try {
    controls->groups.reserve(pending.size());
    controls->candidates.resize(pending.size());
    controls->transactions.reserve(pending.size());
    controls->release_readers.reserve(pending.size());
    controls->release_writers.reserve(pending.size());
  } catch (...) { return Status::ResourceExhausted("slot carrier group metadata"); }
  for (std::size_t i = 0; i < pending.size(); ++i) {
    auto &item = pending[i];
    auto *data = detail::StoreBindingAccess::get(item.next);
    const auto &facts = data->facts;
    std::size_t selected = 0;
    for (; selected < controls->groups.size(); ++selected) {
      const auto &group = controls->groups[selected];
      if (group.lane == item.lane && group.resource == facts.resource && group.execution.same_owner(item.execution) &&
          ((!group.queue.valid() && !item.queue.valid()) || group.queue.same_owner(item.queue))) break;
    }
    if (selected == controls->groups.size()) {
      controls->groups.emplace_back(allocator); // reserve above: no growth here.
      auto &group = controls->groups.back();
      group.execution = std::move(item.execution); group.queue = std::move(item.queue);
      group.lane = item.lane;
      group.resource = facts.resource;
    }
    auto &group = controls->groups[selected];
    const auto count = detail::generated_source_count(*data) + facts.destination_count;
    if (count > SIZE_MAX / sizeof(std::int32_t) ||
        group.count > SIZE_MAX / sizeof(std::int32_t) - count)
      return Status::ResourceExhausted("slot grouped index carrier overflows");
    data->controls = controls; data->group = selected; data->fragment = group.fragments.size();
    data->candidate = i; data->offset = group.count; data->options = resources;
    try { group.fragments.push_back({.offset = group.count}); }
    catch (...) { return Status::ResourceExhausted("slot grouped index slices"); }
    group.count += count;
  }
  for (auto &group : controls->groups) {
    try { group.uploaded_ids.resize(group.count); }
    catch (...) { return Status::ResourceExhausted("slot successful index metadata"); }
    HYLE_ASSIGN_OR_RETURN(group.indices, Indices::create(group.resource, options.host_workspace,
        group.count, resources, group.queue.valid()));
    if (group.indices.direct) continue;
    HYLE_ASSIGN_OR_RETURN(auto destination, group.indices.writer.view());
    HYLE_ASSIGN_OR_RETURN(auto source, group.indices.staging.view());
    if (group.queue.valid()) {
      HYLE_ASSIGN_OR_RETURN(group.metadata_queue,
          gd::RegionAccess::metadata_queue(invocation, group.lane, resources));
      const auto transfer = c::transfer_properties(*group.metadata_queue, destination, source);
      if (!transfer) return transfer.error().WithContext(
          "slot host_workspace must support the original Context's metadata Queue");
    } else {
      HYLE_ASSIGN_OR_RETURN(auto copy, Indices::native_copy(destination, source));
      HYLE_ASSIGN_OR_RETURN(auto supported, gd::ExecutionSourceAccess::can_copy_constant(group.execution, copy));
      if (!supported) return Status::Unimplemented("slot host_workspace has no original native metadata transfer");
    }
  }
  // Every new owner/carrier exists before any old configuration is changed.
  for (const auto &item : pending)
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation, item.found.proof.step));
  for (auto &item : pending) {
    item.found.interface->clear_configuration(item.found.prepared, item.found.invocation);
    *item.found.binding = std::move(item.next);
  }
  return Status::Ok();
}
} // namespace

Status prepare(Invocation &invocation, std::span<const QueueBinding> queues,
    PrepareOptions options) noexcept {
  return prepare_impl(invocation, nullptr, queues, {}, options);
}
Status prepare(Invocation &invocation, const Graph &graph,
    std::span<const QueueBinding> queues, std::span<const SourceBinding> sources,
    PrepareOptions options) noexcept {
  return prepare_impl(invocation, &graph, queues, sources, options);
}

Status bind(Invocation &invocation, const Graph &graph, NodeId node,
    SlotBindings &&bindings) noexcept {
  const std::array requests{BindRequest{node, &bindings}};
  return ::hyle::graph::slots::bind(invocation, graph, std::span<const BindRequest>{requests});
}

Status bind(Invocation &invocation, const Graph &graph,
    std::span<const BindRequest> requests) noexcept {
  if (requests.empty()) return Status::Ok();
  Find first{requests.front().store};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation, first.node, &find, &first));
  if (!first.binding) return Status::Unimplemented("Store has no selected original slot implementation");
  auto *initial = detail::StoreBindingAccess::get(*first.binding);
  if (!initial || !initial->controls)
    return Status::FailedPrecondition("slot bind needs its original prepared control carrier");
  auto controls = initial->controls;
  struct ScratchReset final {
    ControlBatch &controls;
    ~ScratchReset() {
      for (auto &candidate : controls.candidates) candidate = {};
      for (auto &group : controls.groups) {
        group.participants = 0;
        for (auto &fragment : group.fragments) {
          fragment.sources = {}; fragment.destinations = {};
        }
      }
      controls.transactions.clear();
    }
  } scratch_reset{*controls};
  // Scratch belongs to cold preparation, not to the caller or an executor.
  // A previous rejection never leaves a consumed transaction in these entries.
  for (auto &candidate : controls->candidates) candidate = {};
  for (auto &group : controls->groups) group.participants = 0;
  controls->transactions.clear();
  std::size_t permission_count = 0;
  for (const auto &request : requests) {
    if (!request.bindings || !request.bindings->valid())
      return Status::InvalidArgument("slot batch needs populated original transactions");
    Find found{request.store};
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation, request.store, &find, &found));
    if (!found.binding) return Status::Unimplemented("Store has no selected original slot implementation");
    auto *data = detail::StoreBindingAccess::get(*found.binding);
    if (!data || data->controls.get() != controls.get() || data->bindings.valid())
      return Status::FailedPrecondition("slot batch needs its prepared untaken original transactions");
    auto &group = controls->groups[data->group];
    if (!group.execution.is_open() || (group.queue.valid() && !group.queue.is_open()) ||
        group.indices.failed || group.indices.pending.valid() || group.outstanding)
      return Status::FailedPrecondition("slot carrier is closed, failed, not retired or still owns another transaction");
    auto &candidate = controls->candidates[data->candidate];
    if (candidate.bindings)
      return Status::InvalidArgument("slot batch duplicates an original Store");
    auto &s = *data; const auto &p = s.facts;
    if (found.facts.store != p.store || found.facts.schema != p.schema ||
        found.facts.destination_indices != p.destination_indices ||
        found.facts.destination_count != p.destination_count || found.facts.resource != p.resource ||
        found.facts.permission_changes_preserve_commands != p.permission_changes_preserve_commands ||
        !same_source(found.facts.source, s.artifact_source))
      return Status::InvalidArgument("slot preparation differs from its selected artifact");
    if (s.logical_source) {
      // No rescan of every layer: preparation already checked the immutable
      // logical association, and this lookup checks the original Graph.
      HYLE_ASSIGN_OR_RETURN(auto operation, graph.operation(p.source.gather));
      (void)operation;
    }
    const auto &bindings = *request.bindings;
    const auto layout = bindings.layout();
    HYLE_ASSIGN_OR_RETURN(auto schema, Schema::create(layout.description(), layout.slot_count(), layout.stride_bytes()));
    if (schema != p.schema || bindings.destination_count() != p.destination_count ||
        (s.mode == SourceMode::kImported ? bindings.source_count() != 0
            : (p.source.count && bindings.source_count() != p.source.count)) ||
        s.writer_pointers.size() != p.destination_count ||
        bindings.source_addresses().size() != bindings.source_count() ||
        bindings.destination_addresses().size() != bindings.destination_count())
      return Status::InvalidArgument("slot transaction differs from its selected Store geometry");
    for (std::size_t i = 0; i < bindings.source_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto address, layout.locate(bindings.sources()[i]));
      if (address != bindings.source_addresses()[i]) return Status::InvalidArgument("slot source owner/generation changed");
    }
    for (std::size_t i = 0; i < bindings.destination_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto address, layout.locate(bindings.destinations()[i]));
      if (address != bindings.destination_addresses()[i]) return Status::InvalidArgument("slot destination owner/generation changed");
      s.writer_pointers[i] = &request.bindings->destinations()[i];
    }
    HYLE_ASSIGN_OR_RETURN(auto carrier, cd::LeaseAccess::slot_capacity(layout));
    if (carrier.resource() != p.resource)
      return Status::InvalidArgument("slot pool differs from its selected execution resource");
    HYLE_ASSIGN_OR_RETURN(auto description, schema.capacity_description());
    HYLE_ASSIGN_OR_RETURN(auto capacity, c::ConstTensorView::create(description, carrier));
    c::ConstTensorView imported_indices;
    if (s.mode == SourceMode::kImported) {
      HYLE_ASSIGN_OR_RETURN(auto source, gd::RegionAccess::store_source_projection(
          gd::InvocationAccess::state(invocation), detail::imported_projection(s)));
      if (source.capacity.description() != capacity.description() ||
          !same_storage(source.capacity.storage(), capacity.storage()))
        return Status::InvalidArgument("slot imported source differs from its destination pool");
      imported_indices = source.indices;
    }
    const auto sources = bindings.source_count();
    if (sources > UINT32_MAX - 1u || p.destination_count > UINT32_MAX - 1u - sources ||
        permission_count > UINT32_MAX - 1u - sources - p.destination_count)
      return Status::ResourceExhausted("slot batch projected permission count overflows");
    permission_count += sources + p.destination_count + 1;
    candidate.found = found; candidate.bindings = request.bindings;
    candidate.configuration.capacity = capacity;
    candidate.configuration.source_indices = imported_indices;
    controls->transactions.push_back(request.bindings); // one unique selected Store per entry, reserved cold.
    ++group.participants;
  }
  // Graph permits ordinary writers reused by ordered effects. A move-only
  // transaction still has exactly one owner: reject repeated caller handles
  // without a quadratic scan or a registry/another lifetime mechanism.
  std::sort(controls->transactions.begin(), controls->transactions.end(), std::less<SlotBindings *>{});
  if (std::adjacent_find(controls->transactions.begin(), controls->transactions.end()) != controls->transactions.end())
    return Status::InvalidArgument("slot batch duplicates a caller transaction");
  for (const auto &group : controls->groups)
    if (group.participants && group.participants != group.fragments.size())
      return Status::InvalidArgument("slot batch must include every Store in its shared control carrier group");
  // Reserve the complete actual footprint before installing any permission.
  // The original helpers retain their own geometry/conflict validation.
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::reserve_reads(invocation,
      static_cast<std::uint32_t>(permission_count), controls->options));
  for (auto &candidate : controls->candidates) {
    if (!candidate.bindings) continue;
    auto &s = *detail::StoreBindingAccess::get(*candidate.found.binding);
    const bool preserve = s.facts.permission_changes_preserve_commands;
    if (preserve) HYLE_RETURN_IF_ERROR(gd::RegionAccess::revoke_preparation(invocation));
    else HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation, candidate.found.proof.step));
    candidate.found.interface->clear_configuration(candidate.found.prepared, candidate.found.invocation);
    auto &fragment = controls->groups[s.group].fragments[s.fragment];
    fragment.sources = detail::generated_source_count(s) ? candidate.bindings->source_addresses()
        : std::span<const c::LeaseSlotAddress>{};
    fragment.destinations = candidate.bindings->destination_addresses();
  }
  for (auto &group : controls->groups) {
    if (!group.participants) continue;
    HYLE_RETURN_IF_ERROR(group.indices.upload(group.execution, group.queue, group.metadata_queue, controls->options,
        group.fragments, group.uploaded_ids));
  }
  const auto install = [&]() noexcept -> Status {
    // Configure every selected artifact before installing even the first
    // writer. All controls are exact slices of their single fixed group lease.
    for (auto &candidate : controls->candidates) {
      if (!candidate.bindings) continue;
      auto &s = *detail::StoreBindingAccess::get(*candidate.found.binding);
      const auto &p = s.facts;
      HYLE_ASSIGN_OR_RETURN(auto carrier, controls->groups[s.group].indices.ready.view());
      HYLE_ASSIGN_OR_RETURN(auto destination_description, c::TensorDesc::dense(c::DataType::kInt32,
          std::array<std::int64_t, 1>{static_cast<std::int64_t>(p.destination_count)}));
      HYLE_ASSIGN_OR_RETURN(auto destination_view, carrier.subview(
          (s.offset + detail::generated_source_count(s)) * sizeof(std::int32_t),
          p.destination_count * sizeof(std::int32_t)));
      auto &configuration = candidate.configuration;
      HYLE_ASSIGN_OR_RETURN(configuration.destination_indices,
          c::ConstTensorView::create(destination_description, destination_view));
      if (detail::generated_source_count(s)) {
        HYLE_ASSIGN_OR_RETURN(auto source_description, c::TensorDesc::dense(c::DataType::kInt32,
            std::array<std::int64_t, 1>{static_cast<std::int64_t>(p.source.count)}));
        HYLE_ASSIGN_OR_RETURN(auto source_view, carrier.subview(
            s.offset * sizeof(std::int32_t), p.source.count * sizeof(std::int32_t)));
        HYLE_ASSIGN_OR_RETURN(configuration.source_indices,
            c::ConstTensorView::create(source_description, source_view));
      }
      // The capacity can be a hidden native Store argument. Both ID carriers
      // are original Graph input ports: their ordinary rebind gate revokes
      // launch authority and checks actual capture keys/private contracts.
      // Do not destroy those keys merely because an imported source comes
      // from another producer. Full source/ID authority is still validated.
      if (p.permission_changes_preserve_commands &&
          !same_carrier(configuration.capacity, s.last_configuration.capacity))
        HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation, candidate.found.proof.step));
      // Mark before calling: a rejecting cold hook may have partially changed
      // borrowed views, so rollback clears it too.
      candidate.configured = true;
      HYLE_RETURN_IF_ERROR(candidate.found.interface->configure(
          candidate.found.prepared, candidate.found.invocation, configuration));
    }
    for (auto &candidate : controls->candidates) {
      if (!candidate.bindings) continue;
      auto &s = *detail::StoreBindingAccess::get(*candidate.found.binding);
      const auto &p = s.facts;
      const auto &configuration = candidate.configuration;
      HYLE_RETURN_IF_ERROR(invocation.bind(p.destination_indices, configuration.destination_indices));
      if (p.source.count && s.mode != SourceMode::kImported) {
        HYLE_RETURN_IF_ERROR(invocation.bind(p.source.indices, configuration.source_indices));
        if (s.logical_source) {
          const std::array readers{p.source.gather};
          HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_read_projection(invocation,
              {p.source.capacity, {}, readers}, configuration.capacity, candidate.bindings->sources()));
        } else {
          const std::array readers{gd::ProjectedReader{candidate.found.proof.step,
              candidate.found.proof.implementation, candidate.found.proof.prepared,
              p.source.capacity_input_port, p.permission_changes_preserve_commands}};
          HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_projected(invocation, p.source.capacity,
              configuration.capacity, candidate.bindings->sources(), readers));
        }
        candidate.source_bound = true;
      }
      const auto anonymous_sources = p.source.count ? std::span<const c::ReadLease>{}
          : candidate.bindings->sources();
      const std::array control{controls->groups[s.group].indices.ready};
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_write_effect(invocation, graph,
          configuration.capacity, anonymous_sources, s.writer_pointers, {},
          candidate.found.proof, control, s.options));
      candidate.effect_bound = true;
    }
    return Status::Ok();
  };
  const auto status = install();
  if (!status.ok()) {
    // Withdraw all freshly installed writer pointers BEFORE native reader
    // release callbacks. Caller writers have not moved, even on late refusal.
    Status rollback;
    for (auto &candidate : controls->candidates) {
      if (!candidate.effect_bound) continue;
      const auto released = gd::RegionAccess::release_write_effect(invocation, candidate.found.node);
      if (rollback.ok() && !released.ok()) rollback = released;
    }
    for (auto &candidate : controls->candidates) {
      if (!candidate.source_bound) continue;
      const auto &s = *detail::StoreBindingAccess::get(*candidate.found.binding);
      const auto released = gd::RegionAccess::release_read_projection(invocation, s.facts.source.capacity);
      if (rollback.ok() && !released.ok()) rollback = released;
    }
    for (auto &candidate : controls->candidates)
      if (candidate.configured)
        candidate.found.interface->clear_configuration(candidate.found.prepared, candidate.found.invocation);
    if (!rollback.ok()) return rollback.WithContext("slot batch permission rollback after rejected binding");
    return status;
  }
  // No allocation, callback, work or rejecting check remains. SlotBindings
  // moves preserve the exact Pimpl addresses retained by writer permissions.
  for (auto &candidate : controls->candidates) {
    if (!candidate.bindings) continue;
    auto &s = *detail::StoreBindingAccess::get(*candidate.found.binding);
    s.owner = gd::InvocationAccess::state(invocation); s.writer = candidate.found.proof;
    s.configuration = candidate.configuration; s.last_configuration = candidate.configuration;
    s.bindings = std::move(*candidate.bindings);
    ++controls->groups[s.group].outstanding;
  }
  return Status::Ok();
}

Result<SlotBindings> take(Invocation &invocation, NodeId node) noexcept {
  SlotBindings result;
  const std::array requests{TakeRequest{node, &result}};
  HYLE_RETURN_IF_ERROR(take(invocation, requests));
  return result;
}

Status take(Invocation &invocation, std::span<const TakeRequest> requests) noexcept {
  if (requests.empty()) return Status::Ok();
  Find first{requests.front().store};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit_retired(invocation, first.node, &find, &first));
  if (!first.binding || !first.binding->valid())
    return Status::InvalidArgument("Store has no retained original slot transaction");
  auto controls = detail::StoreBindingAccess::get(*first.binding)->controls;
  if (!controls) return Status::FailedPrecondition("slot take needs its original prepared control carrier");
  if (requests.size() > controls->candidates.size())
    return Status::InvalidArgument("slot take exceeds its original prepared Store count");
  struct ScratchReset final {
    ControlBatch &controls;
    ~ScratchReset() {
      for (auto &candidate : controls.candidates) candidate = {};
      controls.transactions.clear(); controls.release_readers.clear(); controls.release_writers.clear();
    }
  } scratch_reset{*controls};
  for (auto &candidate : controls->candidates) candidate = {};
  controls->transactions.clear(); controls->release_readers.clear(); controls->release_writers.clear();
  for (const auto &request : requests) {
    if (!request.bindings || request.bindings->valid())
      return Status::InvalidArgument("slot take requires empty caller destinations");
    Find found{request.store};
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit_retired(invocation, request.store, &find, &found));
    if (!found.binding || !found.binding->valid())
      return Status::InvalidArgument("Store has no retained original slot transaction");
    auto &s = *detail::StoreBindingAccess::get(*found.binding);
    if (s.controls.get() != controls.get() || !s.bindings.valid())
      return Status::InvalidArgument("slot take needs this preparation's untaken original transaction");
    auto &candidate = controls->candidates[s.candidate];
    if (candidate.bindings) return Status::InvalidArgument("slot take duplicates an original Store");
    // Queries retain the exact old owner, writer pointers and read generations.
    // Reset/closed recovery may have already withdrawn those permissions.
    HYLE_ASSIGN_OR_RETURN(const bool bound, gd::RegionAccess::has_write_effect(
        s.owner, s.writer, s.writer_pointers));
    if (s.facts.source.count && s.mode != SourceMode::kImported) {
      HYLE_ASSIGN_OR_RETURN(const bool source_bound, gd::RegionAccess::has_read_projection(
          s.owner, s.facts.source.capacity, s.bindings.sources()));
      if (source_bound) controls->release_readers.push_back(s.facts.source.capacity);
    }
    if (bound) controls->release_writers.push_back(request.store);
    candidate.found = found; candidate.bindings = request.bindings;
    controls->transactions.push_back(request.bindings);
  }
  std::sort(controls->transactions.begin(), controls->transactions.end(), std::less<SlotBindings *>{});
  if (std::adjacent_find(controls->transactions.begin(), controls->transactions.end()) != controls->transactions.end())
    return Status::InvalidArgument("slot take duplicates a caller destination");
  if (!controls->release_readers.empty())
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::release_read_projections(invocation, controls->release_readers));
  if (!controls->release_writers.empty())
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::release_projected_writes(invocation, controls->release_writers));
  // No allocation, permission admission or rejecting check remains. A move
  // returns the original owner; it neither attests success nor publishes it.
  for (auto &candidate : controls->candidates) {
    if (!candidate.bindings) continue;
    auto &s = *detail::StoreBindingAccess::get(*candidate.found.binding);
    if (s.mode != SourceMode::kImported) {
      candidate.found.interface->clear_configuration(candidate.found.prepared, candidate.found.invocation);
      s.configuration = {};
    }
    // Imported input owners stay with their original Source. Dormant physical
    // arguments authorize no later submission without fresh complete binding.
    s.owner = nullptr; s.writer = {};
    auto &group = controls->groups[s.group];
    HYLE_FATAL_ASSERT(group.outstanding != 0); --group.outstanding;
    *candidate.bindings = std::move(s.bindings);
  }
  return Status::Ok();
}

} // namespace hyle::graph::slots
