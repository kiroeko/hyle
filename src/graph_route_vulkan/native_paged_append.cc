#include "native_paged_append.h"
#include "native_region.h"
#include "vulkan_source.h"
#include "native_value.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "../compute_route_vulkan/vulkan_copy_internal.h"
#include "../compute_route_vulkan/vulkan_device_append_internal.h"
#include "../graph_batch/native_append_regions.h"
#include "../graph_batch/native_device_append_regions.h"
#include "../graph_batch/paged_append.h"
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
  bool device_lengths = false;
  std::array<ValueId,3> inputs{};
  std::array<std::uint32_t,3> ports{};
  std::array<compute::TensorDesc,3> descriptions;
  std::size_t count = 1;
  cv::InvocationAllocation allocation;
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
  std::size_t dependency_capacity = 0;
  mem::ConstBufferView rows;
  std::array<cv::Tensor,3> device_inputs;
  std::array<compute::ConstTensorView,3> device_views;
  std::array<compute::ReadLease,3> device_owners;
  std::vector<cv::Tensor> checks;
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
struct AppendRecipe final { batch::PagedAppend descriptor; bool device_lengths = false; };
Result<AppendRecipe> qualify(const RegionCompileRequest &r) noexcept {
  if (!r.graph || r.graph->dimension_count() || r.route != kNativeAppendRoute ||
      r.nodes.size() != 1 || r.inputs.empty() || r.inputs.size() > 3 || !r.outputs.empty() ||
      !r.has_unknown_effects || !r.effect_accesses.empty())
    return std::unexpected(Status::Unimplemented("Vulkan append needs one exact zero-output write node"));
  HYLE_ASSIGN_OR_RETURN(auto operation, r.graph->operation(r.nodes[0]));
  const auto append = operation.custom_as<batch::PagedAppend>();
  const auto device = operation.custom_as<batch::DeviceLengthsPagedAppend>();
  if (!append && !device) return std::unexpected(Status::Unimplemented("not a typed paged append"));
  const auto count = device ? 3U : 1U;
  std::array<compute::TensorDesc,3> descriptions;
  std::array<ValueId,3> values;
  for (std::size_t i=0;i<count;++i) {
    HYLE_ASSIGN_OR_RETURN(values[i],r.graph->node_input(r.nodes[0],i));
    if (std::ranges::find(r.inputs,values[i])==r.inputs.end())
      return std::unexpected(Status::Unimplemented("Vulkan append requires each original logical boundary"));
    HYLE_ASSIGN_OR_RETURN(descriptions[i],r.graph->value_description(values[i]));
  }
  for (auto input:r.inputs) if (std::find(values.begin(),values.begin()+count,input)==values.begin()+count)
    return std::unexpected(Status::InvalidArgument("Vulkan append has a foreign boundary input"));
  HYLE_RETURN_IF_ERROR(device ? device->infer_output_descriptions(descriptions,{}) :
      append->infer_output_descriptions(std::span(descriptions).first(1), {}));
  // Common inference proved the complete original cache carrier. This route
  // copies bytes; exact native resources and four-byte ranges are checked below.
  return AppendRecipe{append ? *append : batch::PagedAppend{device->schema,device->role},bool(device)};
}
Status configure(void *prepared, void *invocation,
                 const bd::NativeAppendConfiguration &config) noexcept try {
  const auto &a = *static_cast<const Artifact *>(prepared);
  auto &s = *static_cast<Slot *>(invocation);
  HYLE_RETURN_IF_ERROR(open(a));
  if (a.device_lengths) return Status::InvalidArgument("device append requires its exact capacity configuration");
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
  HYLE_ASSIGN_OR_RETURN(auto carrier_rows, bd::append_query_rows(a.facts.descriptor, a.facts.rows));
  const auto active_rows = config.layout.facts().query_rows;
  if (!active_rows || active_rows > carrier_rows)
    return Status::InvalidArgument("Vulkan append active rows exceed its whole-row carrier");
  HYLE_ASSIGN_OR_RETURN(auto carrier_bytes, a.facts.rows.required_bytes());
  if (!carrier_bytes || carrier_bytes % carrier_rows)
    return Status::InvalidArgument("Vulkan append carrier has incomplete cache rows");
  const auto row_bytes = carrier_bytes / carrier_rows;
  const auto active_bytes = std::size_t(active_rows) * row_bytes;
  std::size_t bytes = 0, new_bytes = 0;
  for (const auto &range : config.ranges) {
    if (!range.bytes || range.bytes % row_bytes || range.destination.size() != range.bytes || bytes > SIZE_MAX - range.bytes)
      return Status::InvalidArgument("Vulkan append exact copy extent differs");
    HYLE_RETURN_IF_ERROR(aligned(a.runtime, range.destination));
    HYLE_ASSIGN_OR_RETURN(auto destination, vn::MemoryAccess::copy_span(a.runtime, range.destination));
    if (destination.physical.address_space != bank.physical.address_space ||
        destination.physical.backing != bank.physical.backing || destination.offset < bank.offset ||
        destination.offset - bank.offset > config.capacity.storage().size() ||
        range.bytes > config.capacity.storage().size() - static_cast<std::size_t>(destination.offset - bank.offset))
      return Status::InvalidArgument("Vulkan append destination is outside its exact bank backing");
    if (range.from_rows) {
      if (range.source_offset % row_bytes || range.source_offset != new_bytes ||
          range.source_offset > active_bytes || range.bytes > active_bytes - range.source_offset)
        return Status::OutOfRange("Vulkan append row copy exceeds packed input");
      new_bytes += range.bytes;
    } else {
      HYLE_ASSIGN_OR_RETURN(auto source, range.prefix.subview(0, range.bytes));
      HYLE_RETURN_IF_ERROR(aligned(a.runtime, source));
    }
    bytes += range.bytes;
  }
  if (new_bytes != active_bytes)
    return Status::InvalidArgument("Vulkan append does not consume exactly its active rows");
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
Status configure_device(void *prepared,void *invocation,const bd::NativeDeviceAppendConfiguration &config) noexcept try {
  const auto &a=*static_cast<Artifact *>(prepared);auto &s=*static_cast<Slot *>(invocation);
  HYLE_RETURN_IF_ERROR(open(a));
  if (s.recording) HYLE_RETURN_IF_ERROR(vn::copy_recording_retired_gate(*s.recording));
  HYLE_ASSIGN_OR_RETURN(auto carrier_rows,bd::append_query_rows(a.facts.descriptor,a.facts.rows));
  if (&s.artifact!=&a || !a.device_lengths || s.pages.valid() ||
      !a.facts.descriptor.schema.owns(config.layout) || !config.capacity.valid() ||
      config.capacity.storage().resource()!=a.facts.resource ||
      config.layout.facts().query_rows!=carrier_rows)
    return Status::InvalidArgument("Vulkan device append needs its exact original capacity episode");
  HYLE_ASSIGN_OR_RETURN(auto expected,compute::batch_cache_description(a.facts.descriptor.schema,a.facts.descriptor.role));
  if (config.capacity.description()!=expected)
    return Status::InvalidArgument("Vulkan device append physical cache description differs");
  const auto capacity=a.facts.descriptor.schema.capacity();
  const auto limit=(std::min)(capacity.max_pages,capacity.max_page_refs);
  HYLE_ASSIGN_OR_RETURN(auto bytes,a.facts.rows.required_bytes());
  const auto row_bytes=bytes/carrier_rows;
  if (config.pages.size()>limit || limit>UINT32_MAX-capacity.max_requests-1 ||
      std::size_t(limit)>SIZE_MAX/capacity.page_size/row_bytes)
    return Status::ResourceExhausted("Vulkan device append original metadata/byte bounds");
  // A terminal episode may be replaced only after its original retirement.
  // No old pages are consumed by constructing this cold native encoding.
  if (!s.recording || !vn::copy_recording_reuse_gate(*s.recording).ok()) {
    HYLE_ASSIGN_OR_RETURN(s.recording,vn::prepare_copy_recording_runtime(a.runtime,limit,s.allocator,
        std::size_t(limit)*capacity.page_size*row_bytes,vn::CopyRecordingMode::kReplayable));
    HYLE_RETURN_IF_ERROR(vn::prepare_device_append_recording(*s.recording,
        {{capacity.page_size,capacity.max_pages,carrier_rows,limit+capacity.max_requests+1,row_bytes},capacity.max_requests},
        a.facts.rows,a.allocation));
  } else HYLE_RETURN_IF_ERROR(vn::clear_copy_recording(*s.recording));
  std::vector<vn::DeviceAppendPageBinding> pages;pages.reserve(config.pages.size());
  for (const auto &page:config.pages) pages.push_back({page.page,page.destination,page.prefix});
  s.configured=s.recorded=s.sealed=s.submitted=false;s.device_inputs={};s.device_views={};s.device_owners={};s.checks.clear();
  HYLE_RETURN_IF_ERROR(vn::configure_device_append_recording(*s.recording,config.capacity.storage(),config.requests,pages));
  s.layout=config.layout;s.capacity=config.capacity;s.rows={};s.configured=true;
  return Status::Ok();
} catch (const std::bad_alloc &) { return Status::ResourceExhausted("Vulkan device append cold bindings"); }
const bd::NativeDeviceAppendRegionVTable device_table{
    [](void *p) noexcept { const auto &a=*static_cast<Artifact *>(p);return std::array{a.inputs[1],a.inputs[2]}; },
    &configure_device,
    [](void *,void *s) noexcept { return vn::device_append_status(*static_cast<Slot *>(s)->recording); }};
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
      slot.device_inputs={};slot.device_views={};slot.device_owners={};slot.checks.clear();
    }};
Status record_rows(void *, void *) noexcept;
Status bind(void *p, void *opaque, std::span<const RegionValue> inputs,
            std::span<const compute::TensorView> outputs) noexcept {
  const auto &a = *static_cast<const Artifact *>(p); auto &s = *static_cast<Slot *>(opaque);
  HYLE_RETURN_IF_ERROR(open(a));
  s.recorded = false; s.sealed = false; s.submitted = false;
  if (a.device_lengths) {
    if (!s.configured || !s.pages.valid() || s.pages.layout()!=s.layout || inputs.size()!=a.count || !outputs.empty())
      return Status::FailedPrecondition("Vulkan device append original transaction is not bound");
    HYLE_RETURN_IF_ERROR(vn::copy_recording_reuse_gate(*s.recording));
    s.device_inputs={};s.device_views={};s.device_owners={};s.checks.clear();
    for (std::size_t i=0;i<3;++i) {
      const auto &input=inputs[a.ports[i]];
      if (input.description!=a.descriptions[i] || !input.tensor.valid() || input.tensor.description()!=a.descriptions[i])
        return Status::InvalidArgument("Vulkan device append original native input differs");
      if (input.lease) {
        HYLE_ASSIGN_OR_RETURN(auto owner,input.lease->view());const auto range=input.tensor.storage();
        if (owner.resource()!=range.resource() || owner.native_handle()!=range.native_handle() ||
            range.offset()<owner.offset() || range.offset()-owner.offset()>owner.size() ||
            range.size()>owner.size()-(range.offset()-owner.offset()))
          return Status::InvalidArgument("Vulkan device append lease does not cover its exact input");
        HYLE_ASSIGN_OR_RETURN(s.device_owners[i],input.lease->subview(range.offset()-owner.offset(),range.size()));
      }
      if (const auto *origin=native_value(input)) {
        HYLE_RETURN_IF_ERROR(append_native_checks(*origin,a.runtime,s.checks));s.device_inputs[i]=*origin->tensor;
      } else {
        if (input.native_type) return Status::Unimplemented("Vulkan device append source has no original native error provenance");
        HYLE_ASSIGN_OR_RETURN(s.device_inputs[i],vn::MemoryAccess::tensor_view(a.runtime,input.tensor,s.allocator,a.allocation.resource_budget));
      }
      s.device_views[i]=input.tensor;
    }
    HYLE_RETURN_IF_ERROR(vn::bind_device_append_inputs(*s.recording,s.device_inputs,s.checks));
    s.rows=s.device_views[0].storage();s.recorded=true;return Status::Ok();
  }
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
  HYLE_RETURN_IF_ERROR(qualify_region_dependencies(s.dependency_capacity, dependencies));
  if (!s.sealed || !s.pages.valid() || s.pages.layout() != s.layout || inputs.size() != a.count ||
      !outputs.empty() || (!a.device_lengths && (inputs[0].description() != a.facts.rows ||
      !same(inputs[0].storage(), s.rows))))
    return std::unexpected(Status::FailedPrecondition("Vulkan append rows/owned pages are not sealed"));
  if (a.device_lengths) for (std::size_t i=0;i<3;++i)
    if (!inputs[a.ports[i]].valid() || !same(inputs[a.ports[i]].storage(),s.device_views[i].storage()))
      return std::unexpected(Status::FailedPrecondition("Vulkan device append input changed after cold binding"));
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
    .query_extension = [](void *p, compute::ProviderExtensionId id) noexcept -> const void * {
      static const gd::RegionDependenciesVTable dependencies_table{
          [](void *prepared, void *invocation, std::size_t capacity) noexcept -> Status {
            auto &s = *static_cast<Slot *>(invocation);
            if (s.recording) HYLE_RETURN_IF_ERROR(vn::copy_recording_reuse_gate(*s.recording));
            return reserve_region_dependencies(static_cast<Artifact *>(prepared)->runtime,
                s.dependency_capacity, capacity);
          }};
      if (id == gd::kRegionDependenciesExtensionId) return &dependencies_table;
      if (id == bd::kNativeAppendRegionExtensionId) return &append_table;
      if (id == bd::kNativeDeviceAppendRegionExtensionId && static_cast<Artifact *>(p)->device_lengths)
        return &device_table;
      if (id == gd::kRegionOrderedWriteExtensionId) return &ordered_write;
      if (id == gd::kRegionBindingReleaseExtensionId) return &release_table;
      return nullptr;
    }};
} // namespace
bool native_append_operation(const compute::Operation &op) noexcept {
  return op.custom_as<batch::PagedAppend>().has_value() || op.custom_as<batch::DeviceLengthsPagedAppend>().has_value();
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
  if (inputs.size() != r.inputs.size() || !outputs.empty()) return Status::InvalidArgument("Vulkan append port arity differs");
  const auto &runtime = static_cast<SourceContext *>(opaque)->runtime;
  const cd::ResourceBindingRequirement resource{.required_properties = mem::MemoryProperty::kDeviceAccessible,
      .forbidden_properties = mem::MemoryProperty::kHostAccessible, .minimum_alignment = 1,
      .exact_domain = {runtime->device.get(), static_cast<std::uintptr_t>(cv::MemoryKind::kDevice)},
      .requires_exact_domain = true};
  std::ranges::fill(inputs,RegionInputRequirement{.resource=resource});
  *out = {.runtime = properties, .preserves_effects = true}; return Status::Ok();
}
compute::ProviderPrepareOutcome compile_native_append(void *opaque, const RegionCompileRequest &r,
    mem::AllocatorRef allocator, ProviderRegionPrepared *out) noexcept {
  using Outcome = compute::ProviderPrepareOutcome; *out = {};
  auto descriptor = qualify(r);
  if (!descriptor) return descriptor.error().IsUnimplemented() ? Outcome::unsupported(descriptor.error()) : Outcome::error(descriptor.error());
  if (r.input_resources.size() != r.inputs.size()) return Outcome::error(Status::InvalidArgument("Vulkan append input resource is absent"));
  const auto &runtime = static_cast<SourceContext *>(opaque)->runtime;
  for(auto resource:r.input_resources) {
    const auto admitted=vn::MemoryAccess::validate_resource(runtime,resource);
    if(!admitted.ok())return Outcome::error(admitted);
  }
  auto *a = allocator.construct<Artifact>();
  if (!a) return Outcome::error(Status::ResourceExhausted("Vulkan append artifact"));
  a->runtime = runtime; a->device_lengths=descriptor->device_lengths;a->count=r.inputs.size();
  *out = {.vtable = &table, .state = a,
      .properties = {.persistent_bytes = sizeof(Artifact), .runtime = properties,
          .preserves_effects = true, .invocation_private_bytes = std::nullopt}};
  const auto build=[&]() -> Status {
    const auto count=a->device_lengths?3U:1U;
    for(std::size_t i=0;i<count;++i) {
      HYLE_ASSIGN_OR_RETURN(a->inputs[i],r.graph->node_input(r.nodes[0],i));
      a->ports[i]=static_cast<std::uint32_t>(std::ranges::find(r.inputs,a->inputs[i])-r.inputs.begin());
      HYLE_ASSIGN_OR_RETURN(a->descriptions[i],r.graph->value_description(a->inputs[i]));
    }
    a->facts={r.nodes[0],descriptor->descriptor,a->descriptions[0],r.input_resources[a->ports[0]]};
    a->allocation.resource_budget=r.resource_budget;a->allocation.device=a->facts.resource;
    if(a->device_lengths) {
      for(const auto &domain:r.available_storage_domains)
        if(vn::MemoryAccess::validate_resource(runtime,domain.resource).ok() &&
            domain.resource.properties().has(mem::MemoryProperty::kHostAccessible)) {
          a->allocation.staging=domain.resource;break;
        }
      if(!a->allocation.staging.valid())return Status::Unimplemented("Vulkan device append needs its original staging metadata domain");
    }
    return Status::Ok();
  };
  const auto status=build();
  if(!status.ok())return status.IsUnimplemented()?Outcome::unsupported(status):Outcome::error(status);
  return Outcome::prepared();
}
} // namespace hyle::graph::vulkan::detail
