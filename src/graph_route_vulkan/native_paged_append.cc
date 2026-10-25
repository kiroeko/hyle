#include "native_paged_append.h"
#include "native_region.h"
#include "vulkan_source.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "../compute_route_vulkan/vulkan_copy_internal.h"
#include "../graph_batch/native_append_regions.h"
#include "hyle/core/assert.h"

namespace hyle::graph::vulkan::detail {
namespace {
namespace cv = compute::vulkan;
namespace vn = cv::detail;
namespace cd = compute::detail;
namespace gd = graph::detail;
namespace bd = batch::detail;
constexpr compute::ProviderRuntimeProperties properties{
    .mode = compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
    .replay_class = compute::ProviderReplayClass::kPrepared};
struct Artifact final {
  std::shared_ptr<vn::Runtime> runtime;
  bd::NativeAppendRegionFacts facts;
};
struct Slot final {
  Slot(mem::AllocatorRef a, const Artifact &owner) : artifact(owner), allocator(a),
      ranges(mem::StdAllocator<bd::NativeAppendRange>(a)),
      copies(mem::StdAllocator<cd::CopyBackendRange>(a)) {}
  const Artifact &artifact;
  mem::AllocatorRef allocator;
  batch::PagedAppendPages pages;
  compute::BatchLayoutVersion layout;
  compute::ConstTensorView capacity;
  std::vector<bd::NativeAppendRange, mem::StdAllocator<bd::NativeAppendRange>> ranges;
  std::vector<cd::CopyBackendRange, mem::StdAllocator<cd::CopyBackendRange>> copies;
  std::shared_ptr<vn::CopyRecording> recording;
  mem::ConstBufferView rows;
  bool configured = false, recorded = false, sealed = false, submitted = false;
};
Status open(const Artifact &a) noexcept {
  std::lock_guard lock(a.runtime->mutex);
  if (a.runtime->closed) return Status::FailedPrecondition("Vulkan append source is closed");
  return a.runtime->failure;
}
bool same(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  return a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() == b.offset() && a.size() == b.size();
}
Status aligned(const std::shared_ptr<vn::Runtime> &runtime, mem::ConstBufferView view) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto span, vn::MemoryAccess::copy_span(runtime, view));
  if (span.offset % 4 || view.size() % 4)
    return Status::Unimplemented("Vulkan append needs exact four-byte aligned copy ranges");
  return Status::Ok();
}
Result<batch::PagedAppend> qualify(const RegionCompileRequest &r) noexcept {
  if (!r.graph || r.graph->dimension_count() || r.route != kNativeAppendRoute ||
      r.nodes.size() != 1 || r.inputs.size() != 1 || !r.outputs.empty() ||
      !r.has_unknown_effects || !r.effect_accesses.empty())
    return std::unexpected(Status::Unimplemented("Vulkan append needs one exact zero-output write node"));
  HYLE_ASSIGN_OR_RETURN(auto operation, r.graph->operation(r.nodes[0]));
  const auto append = operation.custom_as<batch::PagedAppend>();
  if (!append) return std::unexpected(Status::Unimplemented("not a typed paged append"));
  HYLE_ASSIGN_OR_RETURN(auto rows, r.graph->value_description(r.inputs[0]));
  HYLE_RETURN_IF_ERROR(append->infer_output_descriptions(std::span(&rows, 1), {}));
  const auto type = rows.data_type();
  if (type != compute::DataType::kFloat16 && type != compute::DataType::kBFloat16 &&
      type != compute::DataType::kFloat8E4M3FN)
    return std::unexpected(Status::Unimplemented("Vulkan append requires plain F16/BF16/E4 rows"));
  return *append;
}
Status configure(void *prepared, void *invocation,
                 const bd::NativeAppendConfiguration &config) noexcept try {
  const auto &a = *static_cast<const Artifact *>(prepared);
  auto &s = *static_cast<Slot *>(invocation);
  HYLE_RETURN_IF_ERROR(open(a));
  // A retired cancelled/failed copy may be replaced cold. Source/device
  // failure was checked above; this gate never treats pending work as retired.
  if (s.recording) HYLE_RETURN_IF_ERROR(vn::copy_recording_retired_gate(*s.recording));
  if (s.pages.valid()) return Status::FailedPrecondition("take Vulkan append pages before rebinding");
  if (!a.facts.descriptor.schema.owns(config.layout) || !config.capacity.valid() ||
      config.ranges.empty() || config.ranges.size() > UINT32_MAX / 2)
    return Status::InvalidArgument("Vulkan append layout/capacity/range count differs");
  HYLE_RETURN_IF_ERROR(vn::MemoryAccess::validate_resource(a.runtime, config.capacity.storage().resource()));
  HYLE_ASSIGN_OR_RETURN(auto capacity_description,
      compute::batch_cache_description(a.facts.descriptor.schema, a.facts.descriptor.role));
  if (config.capacity.description() != capacity_description)
    return Status::InvalidArgument("Vulkan append bank capacity description differs");
  HYLE_ASSIGN_OR_RETURN(auto bank, vn::MemoryAccess::copy_span(a.runtime, config.capacity.storage()));
  HYLE_ASSIGN_OR_RETURN(auto row_bytes, a.facts.rows.required_bytes());
  std::size_t bytes = 0;
  for (const auto &range : config.ranges) {
    if (!range.bytes || range.destination.size() != range.bytes || bytes > SIZE_MAX - range.bytes)
      return Status::InvalidArgument("Vulkan append exact copy extent differs");
    HYLE_RETURN_IF_ERROR(aligned(a.runtime, range.destination));
    HYLE_ASSIGN_OR_RETURN(auto destination, vn::MemoryAccess::copy_span(a.runtime, range.destination));
    if (destination.physical.address_space != bank.physical.address_space ||
        destination.physical.backing != bank.physical.backing || destination.offset < bank.offset ||
        destination.offset - bank.offset > config.capacity.storage().size() ||
        range.bytes > config.capacity.storage().size() - static_cast<std::size_t>(destination.offset - bank.offset))
      return Status::InvalidArgument("Vulkan append destination is outside its exact bank backing");
    if (range.from_rows) {
      if (range.source_offset > row_bytes || range.bytes > row_bytes - range.source_offset)
        return Status::OutOfRange("Vulkan append row copy exceeds packed input");
    } else {
      HYLE_ASSIGN_OR_RETURN(auto source, range.prefix.subview(0, range.bytes));
      HYLE_RETURN_IF_ERROR(aligned(a.runtime, source));
    }
    bytes += range.bytes;
  }
  s.configured = false; s.recorded = false; s.sealed = false; s.submitted = false; s.rows = {};
  s.ranges.assign(config.ranges.begin(), config.ranges.end());
  s.copies.resize(config.ranges.size());
  // A new cold recording also resets its exact max-byte bound. It enqueues no work.
  HYLE_ASSIGN_OR_RETURN(s.recording, vn::prepare_copy_recording_runtime(a.runtime,
      static_cast<std::uint32_t>(config.ranges.size()), s.allocator, bytes,
      vn::CopyRecordingMode::kReplayable));
  s.layout = config.layout; s.capacity = config.capacity; s.configured = true;
  return Status::Ok();
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan append range metadata");
}
Status reader(void *prepared, void *, cd::CompletionEndpoint endpoint) noexcept {
  const auto &a = *static_cast<const Artifact *>(prepared);
  HYLE_RETURN_IF_ERROR(open(a));
  return imports_region_ordered_endpoint(a.runtime, endpoint) ? Status::Ok() :
      Status::Unimplemented("Vulkan append reader is not the same original compute Runtime");
}
Status before_read(void *, void *invocation) noexcept {
  const auto &s = *static_cast<const Slot *>(invocation);
  if (!s.submitted || !s.recording || !s.pages.valid())
    return Status::FailedPrecondition("Vulkan append writer has not submitted this episode");
  return vn::copy_recording_write_status(*s.recording);
}
const bd::NativeAppendRegionVTable append_table{
    .inspect = [](void *p, void *) noexcept -> Result<bd::NativeAppendRegionFacts> {
      return static_cast<const Artifact *>(p)->facts;
    },
    .pages = [](void *s) noexcept { return &static_cast<Slot *>(s)->pages; },
    .configure = &configure, .qualify_reader = &reader, .before_read = &before_read};
const gd::RegionOrderedWriteVTable ordered_write{
    [](const ProviderRegionPrepared &prepared, void *invocation,
       const cd::CompletionEndpoint *endpoint) noexcept -> Status {
      if (!prepared.state || !invocation) return Status::InvalidArgument("Vulkan append slot is empty");
      const auto &a = *static_cast<const Artifact *>(prepared.state);
      if (&static_cast<const Slot *>(invocation)->artifact != &a)
        return Status::InvalidArgument("Vulkan append slot belongs to a different selected artifact");
      HYLE_RETURN_IF_ERROR(open(a));
      return endpoint ? reader(prepared.state, invocation, *endpoint) : Status::Ok();
    }};
const gd::RegionBindingReleaseVTable release_table{
    .validate = [](void *, void *s) noexcept -> Status {
      const auto &slot = *static_cast<const Slot *>(s);
      return slot.recording ? vn::copy_recording_retired_gate(*slot.recording) : Status::Ok();
    },
    .release = [](void *, void *s) noexcept {
      auto &slot = *static_cast<Slot *>(s);
      if (slot.recording) HYLE_FATAL_ASSERT(vn::clear_copy_recording(*slot.recording).ok());
      slot.rows = {}; slot.recorded = false; slot.sealed = false; slot.submitted = false;
    }};
Status record_rows(void *, void *) noexcept;
Status bind(void *p, void *opaque, std::span<const RegionValue> inputs,
            std::span<const compute::TensorView> outputs) noexcept {
  const auto &a = *static_cast<const Artifact *>(p); auto &s = *static_cast<Slot *>(opaque);
  HYLE_RETURN_IF_ERROR(open(a));
  s.recorded = false; s.sealed = false; s.submitted = false;
  if (!s.configured || !s.pages.valid() || s.pages.layout() != s.layout ||
      inputs.size() != 1 || !outputs.empty() || inputs[0].description != a.facts.rows ||
      !inputs[0].tensor.valid() || inputs[0].tensor.description() != a.facts.rows)
    return Status::FailedPrecondition("Vulkan append original transaction/rows are not bound");
  HYLE_RETURN_IF_ERROR(vn::copy_recording_reuse_gate(*s.recording));
  s.rows = inputs[0].tensor.storage();
  // Acquire original native source owners before final upstream output seals.
  return record_rows(p, opaque);
}
Status record_rows(void *p, void *opaque) noexcept {
  const auto &a = *static_cast<const Artifact *>(p); auto &s = *static_cast<Slot *>(opaque);
  HYLE_RETURN_IF_ERROR(open(a));
  if (!s.configured || !s.pages.valid() || s.pages.layout() != s.layout || !s.rows.valid())
    return Status::FailedPrecondition("Vulkan append exact episode is not configured");
  HYLE_RETURN_IF_ERROR(vn::copy_recording_reuse_gate(*s.recording));
  HYLE_ASSIGN_OR_RETURN(auto row_bytes, a.facts.rows.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto payload, s.rows.subview(0, row_bytes));
  HYLE_ASSIGN_OR_RETURN(auto physical_rows, vn::MemoryAccess::copy_span(a.runtime, payload));
  for (std::size_t i = 0; i < s.ranges.size(); ++i) {
    const auto &range = s.ranges[i];
    auto &copy = s.copies[i]; copy.destination = range.destination;
    if (range.from_rows) {
      HYLE_ASSIGN_OR_RETURN(copy.source, s.rows.subview(range.source_offset, range.bytes));
    } else {
      HYLE_ASSIGN_OR_RETURN(copy.source, range.prefix.subview(0, range.bytes));
    }
    HYLE_RETURN_IF_ERROR(aligned(a.runtime, copy.source));
    HYLE_ASSIGN_OR_RETURN(auto destination, vn::MemoryAccess::copy_span(a.runtime, copy.destination));
    if (physical_rows.physical.address_space == destination.physical.address_space &&
        physical_rows.physical.backing == destination.physical.backing &&
        (physical_rows.offset < destination.offset
            ? destination.offset - physical_rows.offset < row_bytes
            : physical_rows.offset - destination.offset < range.bytes))
      return Status::InvalidArgument("Vulkan append packed input overlaps a dirty destination");
  }
  HYLE_RETURN_IF_ERROR(vn::bind_copy_recording(*s.recording, s.copies));
  s.recorded = true; s.submitted = false; return Status::Ok();
}
Status seal(void *p, void *opaque) noexcept {
  const auto &a = *static_cast<const Artifact *>(p); auto &s = *static_cast<Slot *>(opaque);
  HYLE_RETURN_IF_ERROR(open(a));
  if (!s.recorded || !s.pages.valid() || s.pages.layout() != s.layout || !s.rows.valid())
    return Status::FailedPrecondition("Vulkan append exact recording is not cold bound");
  HYLE_RETURN_IF_ERROR(vn::copy_recording_reuse_gate(*s.recording));
  s.sealed = true; return Status::Ok();
}
Result<compute::Completion> submit(void *p, void *opaque,
    std::span<const cd::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  const auto &a = *static_cast<const Artifact *>(p); auto &s = *static_cast<Slot *>(opaque);
  if (!s.sealed || !s.pages.valid() || s.pages.layout() != s.layout || inputs.size() != 1 ||
      !outputs.empty() || inputs[0].description() != a.facts.rows ||
      !same(inputs[0].storage(), s.rows))
    return std::unexpected(Status::FailedPrecondition("Vulkan append rows/owned pages are not sealed"));
  for (const auto dependency : dependencies)
    HYLE_RETURN_IF_ERROR(vn::qualify_copy_dependency(a.runtime, dependency));
  s.submitted = false;
  auto completion = vn::submit_copy_recording(*s.recording);
  if (completion) s.submitted = true;
  return completion;
}
const ProviderRegionPreparedVTable table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Artifact *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return compute::AliasSupport{}; },
    .create_invocation = [](void *p, mem::AllocatorRef a, mem::BufferView, void **out) noexcept -> Status {
      *out = a.construct<Slot>(a, *static_cast<const Artifact *>(p));
      return *out ? Status::Ok() : Status::ResourceExhausted("Vulkan append slot");
    },
    .destroy_invocation = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Slot *>(p)); },
    .delegated_submit = &submit,
    .endpoint = [](void *p, void *) noexcept { return region_ordered_endpoint(static_cast<Artifact *>(p)->runtime); },
    .can_import_endpoint = [](void *p, void *, cd::CompletionEndpoint source) noexcept {
      return imports_region_ordered_endpoint(static_cast<Artifact *>(p)->runtime, source);
    },
    .prepare_bindings = &bind, .seal_bindings = &seal,
    .query_extension = [](void *, compute::ProviderExtensionId id) noexcept -> const void * {
      if (id == bd::kNativeAppendRegionExtensionId) return &append_table;
      if (id == gd::kRegionOrderedWriteExtensionId) return &ordered_write;
      if (id == gd::kRegionBindingReleaseExtensionId) return &release_table;
      return nullptr;
    }};
} // namespace
bool native_append_operation(const compute::Operation &op) noexcept {
  return op.custom_as<batch::PagedAppend>().has_value();
}
Status propose_native_append(const RegionDiscoveryRequest &r, RegionProposalWriter &writer) noexcept {
  for (std::size_t i = 0; i < r.graph.node_count(); ++i) {
    if (!r.allowed_nodes.empty() && !r.allowed_nodes[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, r.graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto kind, r.graph.operation_kind(node));
    if (kind != compute::OperationKind::kCustom) continue;
    HYLE_ASSIGN_OR_RETURN(auto operation, r.graph.operation(node));
    if (!native_append_operation(operation)) continue;
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added, writer.try_add({.nodes = nodes, .route = kNativeAppendRoute,
        .replay_class = compute::ProviderReplayClass::kPrepared,
        .implementation = compute::ProviderImplementationId{0x415050454e44564bULL}, .preference = 100}));
    if (!added) break;
  }
  return Status::Ok();
}
Status native_append_resources(void *opaque, const RegionCompileRequest &r,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *out) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto descriptor, qualify(r)); (void)descriptor;
  if (inputs.size() != 1 || !outputs.empty()) return Status::InvalidArgument("Vulkan append port arity differs");
  const auto &runtime = static_cast<SourceContext *>(opaque)->runtime;
  inputs[0].resource = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
      .forbidden_properties = mem::MemoryProperty::kHostAccessible, .minimum_alignment = 1,
      .exact_domain = {runtime->device.get(), static_cast<std::uintptr_t>(cv::MemoryKind::kDevice)},
      .requires_exact_domain = true};
  *out = {.runtime = properties, .preserves_effects = true}; return Status::Ok();
}
compute::ProviderPrepareOutcome compile_native_append(void *opaque, const RegionCompileRequest &r,
    mem::AllocatorRef allocator, ProviderRegionPrepared *out) noexcept {
  using Outcome = compute::ProviderPrepareOutcome; *out = {};
  auto descriptor = qualify(r);
  if (!descriptor) return descriptor.error().IsUnimplemented() ? Outcome::unsupported(descriptor.error()) : Outcome::error(descriptor.error());
  if (r.input_resources.size() != 1) return Outcome::error(Status::InvalidArgument("Vulkan append input resource is absent"));
  const auto &runtime = static_cast<SourceContext *>(opaque)->runtime;
  const auto admitted = vn::MemoryAccess::validate_resource(runtime, r.input_resources[0]);
  if (!admitted.ok()) return Outcome::error(admitted);
  auto rows = r.graph->value_description(r.inputs[0]);
  if (!rows) return Outcome::error(rows.error());
  auto *a = allocator.construct<Artifact>();
  if (!a) return Outcome::error(Status::ResourceExhausted("Vulkan append artifact"));
  a->runtime = runtime; a->facts = {r.nodes[0], *descriptor, *rows, r.input_resources[0]};
  *out = {.vtable = &table, .state = a,
      .properties = {.persistent_bytes = sizeof(Artifact), .runtime = properties,
          .preserves_effects = true, .invocation_private_bytes = std::nullopt}};
  return Outcome::prepared();
}
} // namespace hyle::graph::vulkan::detail
