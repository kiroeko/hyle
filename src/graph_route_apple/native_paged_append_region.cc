#include "native_paged_append_region.h"

#include "native_region.h"
#include "../compute_route_apple/apple_copy_internal.h"
#include "../compute_route_apple/apple_device_append_internal.h"
#include "../graph_batch/native_append_regions.h"
#include "../graph_batch/native_device_append_regions.h"
#include "../graph_batch/paged_append.h"
#include "hyle/core/assert.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/mem/std_allocator.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <utility>
#include <vector>

namespace hyle::graph::apple::detail {
namespace {
namespace ca = compute::apple;
namespace an = ca::detail;
namespace cd = compute::detail;
namespace bd = graph::batch::detail;
namespace gd = graph::detail;
template<class T> using Vector = std::vector<T, mem::StdAllocator<T>>;
constexpr compute::ProviderRuntimeProperties runtime_properties{
    .mode = compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
    .replay_class = compute::ProviderReplayClass::kManaged,
    .uses_private_threads = true};
struct Artifact final {
  std::shared_ptr<an::Runtime> runtime;
  NodeId node;
  batch::PagedAppend descriptor;
  compute::TensorDesc rows;
  mem::MemoryResourceRef resource;
  std::uint32_t range_capacity = 0, query_rows = 0;
  std::size_t max_bytes = 0, row_bytes = 0, input_bytes = 0;
  bool device_lengths = false;
  std::array<ValueId,3> inputs{};
  std::array<std::uint32_t,3> ports{};
  std::array<compute::TensorDesc,3> descriptions;
  std::uint32_t boundary_count = 0;
};
struct Slot final {
  explicit Slot(mem::AllocatorRef allocator, Artifact &owner)
      : artifact(owner), ranges(mem::StdAllocator<bd::NativeAppendRange>(allocator)),
        copies(mem::StdAllocator<cd::CopyBackendRange>(allocator)),
        device_pages(mem::StdAllocator<an::DeviceAppendPageBinding>(allocator)) {}
  Artifact &artifact;
  batch::PagedAppendPages pages;
  std::shared_ptr<an::CopyRecording> recording;
  Vector<bd::NativeAppendRange> ranges;
  Vector<cd::CopyBackendRange> copies;
  Vector<an::DeviceAppendPageBinding> device_pages;
  std::array<an::InputBinding,3> device_inputs;
  std::array<compute::ReadLease,3> device_owners;
  std::array<compute::ConstTensorView,3> device_views;
  compute::BatchLayoutVersion layout;
  compute::ConstTensorView capacity, rows;
  an::InputSource source;
  compute::ReadLease rows_owner;
  an::CopySpan resolved_rows;
  bool configured = false, bound = false, sealed = false, submitted = false;
};
Status open(const Artifact &artifact) noexcept {
  std::lock_guard lock(artifact.runtime->queue_mutex);
  if (!artifact.runtime->accepting) return Status::FailedPrecondition("Apple append source is closed");
  return artifact.runtime->failure;
}
bool same_range(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  return a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() == b.offset() && a.size() == b.size();
}
bool overlaps(const cd::PhysicalCopyResource &a, const cd::PhysicalCopyResource &b) noexcept {
  if (a.address_space != b.address_space || a.backing != b.backing || !a.capacity || !b.capacity)
    return false;
  return a.origin <= b.origin ? b.origin - a.origin < a.capacity : a.origin - b.origin < b.capacity;
}
struct AppendRecipe final { batch::PagedAppend descriptor; bool device_lengths = false; };
Result<AppendRecipe> qualify(const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.graph->dimension_count() || request.route != kNativePagedAppendRoute ||
      request.nodes.size() != 1 || request.inputs.empty() || request.inputs.size() > 3 || !request.outputs.empty() ||
      !request.has_unknown_effects || !request.effect_accesses.empty())
    return std::unexpected(Status::Unimplemented("Apple append requires one fixed zero-output typed effect"));
  HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(request.nodes[0]));
  const auto append = operation.custom_as<batch::PagedAppend>();
  const auto lengths = operation.custom_as<batch::DeviceLengthsPagedAppend>();
  if (!append && !lengths) return std::unexpected(Status::Unimplemented("not a typed paged append"));
  if(lengths && lengths->schema.capacity().max_requests>32)
    return std::unexpected(Status::Unimplemented("Apple device append supports at most32 requests"));
  AppendRecipe recipe{append?*append:batch::PagedAppend{lengths->schema,lengths->role},bool(lengths)};
  const auto count=recipe.device_lengths?3u:1u;
  std::array<compute::TensorDesc,3> descriptions;
  std::array<ValueId,3> values{};
  for(std::size_t i=0;i<count;++i){
    HYLE_ASSIGN_OR_RETURN(values[i],request.graph->node_input(request.nodes[0],i));
    if(std::ranges::find(request.inputs,values[i])==request.inputs.end())
      return std::unexpected(Status::Unimplemented("Apple append needs each original logical boundary input"));
    HYLE_ASSIGN_OR_RETURN(descriptions[i],request.graph->value_description(values[i]));
  }
  for(auto input:request.inputs)if(std::find(values.begin(),values.begin()+count,input)==values.begin()+count)
    return std::unexpected(Status::InvalidArgument("Apple append has a foreign boundary input"));
  HYLE_RETURN_IF_ERROR(lengths?lengths->infer_output_descriptions(descriptions,{}) :
      append->infer_output_descriptions(std::span(descriptions).first(1),{}));
  return recipe;
}
Status actual_aliases(Slot &slot, const an::CopySpan &rows) noexcept {
  // The complete logical rows input is read authority, not its carrier tail.
  // Resolve deferred producer bytes before checking every dirty destination.
  for (const auto &range : slot.ranges) {
    HYLE_ASSIGN_OR_RETURN(auto target, an::copy_span(range.destination, mem::HostAccess::kWrite));
    if (overlaps(rows.physical, target.physical))
      return Status::InvalidArgument("Apple append rows overlap a dirty destination");
    for (const auto &prefix : slot.ranges) if (!prefix.from_rows) {
      HYLE_ASSIGN_OR_RETURN(auto source, an::copy_span(prefix.prefix, mem::HostAccess::kRead));
      if (overlaps(source.physical, target.physical))
        return Status::InvalidArgument("Apple append inherited prefix overlaps a dirty destination");
    }
  }
  return Status::Ok();
}
Result<an::CopySpan> resolved_source(void *opaque, std::size_t index) noexcept {
  auto &slot = *static_cast<Slot *>(opaque);
  if (index >= slot.ranges.size()) return std::unexpected(Status::OutOfRange("Apple append range index"));
  const auto &range = slot.ranges[index];
  if (!range.from_rows) return an::copy_span(range.prefix, mem::HostAccess::kRead);
  auto source = slot.resolved_rows;
  if (range.source_offset > source.physical.capacity || range.bytes > source.physical.capacity - range.source_offset ||
      range.source_offset > SIZE_MAX - source.physical.origin)
    return std::unexpected(Status::OutOfRange("Apple append source exceeds actual packed rows"));
  source.physical.origin += range.source_offset; source.physical.capacity = range.bytes;
  if (source.physical.origin % source.physical.copy_alignment || range.bytes % source.physical.copy_alignment)
    return std::unexpected(Status::InvalidArgument("Apple append source is not Metal blit aligned"));
  return source;
}
Status preflight(void *opaque) noexcept {
  auto &slot = *static_cast<Slot *>(opaque);
  HYLE_ASSIGN_OR_RETURN(slot.resolved_rows, an::native_copy_source(*slot.recording));
  HYLE_RETURN_IF_ERROR(actual_aliases(slot, slot.resolved_rows));
  return an::resolve_result_copy_recording(*slot.recording, slot.copies, &resolved_source, &slot);
}
Status configure(void *prepared, void *invocation, const bd::NativeAppendConfiguration &configuration) noexcept {
  const auto &artifact = *static_cast<Artifact *>(prepared); auto &slot = *static_cast<Slot *>(invocation);
  HYLE_RETURN_IF_ERROR(open(artifact));
  if(artifact.device_lengths)return Status::InvalidArgument("device append requires its capacity configuration");
  HYLE_RETURN_IF_ERROR(an::copy_recording_reuse_gate(*slot.recording));
  if (slot.pages.valid()) return Status::FailedPrecondition("take Apple append pages before rebinding");
  slot.configured = false; slot.bound = false; slot.sealed = false; slot.submitted = false;
  if (!artifact.descriptor.schema.owns(configuration.layout) || !configuration.capacity.valid() ||
      configuration.capacity.storage().resource() != artifact.resource ||
      !configuration.layout.facts().query_rows || configuration.layout.facts().query_rows > artifact.query_rows ||
      configuration.ranges.size() > artifact.range_capacity)
    return Status::InvalidArgument("Apple append layout/capacity/range geometry differs");
  const auto active_bytes = std::size_t(configuration.layout.facts().query_rows) * artifact.row_bytes;
  std::size_t copied = 0, new_bytes = 0;
  const auto capacity = configuration.capacity.storage();
  for (std::size_t index = 0; index < configuration.ranges.size(); ++index) {
    const auto &range = configuration.ranges[index]; const auto target = range.destination;
    if (!target.valid() || !range.bytes || target.size() != range.bytes || range.bytes % artifact.row_bytes ||
        target.resource() != capacity.resource() || target.native_handle() != capacity.native_handle() ||
        target.offset() < capacity.offset() || target.offset() - capacity.offset() > capacity.size() ||
        range.bytes > capacity.size() - (target.offset() - capacity.offset()) ||
        range.bytes > artifact.max_bytes - copied)
      return Status::InvalidArgument("Apple append dirty range exceeds its original bank");
    copied += range.bytes;
    HYLE_ASSIGN_OR_RETURN(auto physical, an::copy_span(target, mem::HostAccess::kWrite));
    if (physical.physical.origin % physical.physical.copy_alignment || range.bytes % physical.physical.copy_alignment)
      return Status::InvalidArgument("Apple append dirty range is not Metal blit aligned");
    for (std::size_t prior = 0; prior < index; ++prior) {
      HYLE_ASSIGN_OR_RETURN(auto other, an::copy_span(configuration.ranges[prior].destination, mem::HostAccess::kWrite));
      if (overlaps(physical.physical, other.physical)) return Status::InvalidArgument("Apple append dirty ranges overlap");
    }
    if (range.from_rows) {
      if (range.source_offset % artifact.row_bytes || range.source_offset != new_bytes ||
          range.source_offset > active_bytes || range.bytes > active_bytes - range.source_offset)
        return Status::OutOfRange("Apple append new range exceeds packed rows");
      for (std::size_t prior = 0; prior < index; ++prior) if (configuration.ranges[prior].from_rows) {
        const auto &other = configuration.ranges[prior];
        if (range.source_offset <= other.source_offset ? other.source_offset-range.source_offset < range.bytes :
            range.source_offset-other.source_offset < other.bytes)
          return Status::InvalidArgument("Apple append packed row ranges overlap");
      }
      new_bytes += range.bytes;
    } else if (!range.prefix.valid() || range.prefix.size() != range.bytes) {
      return Status::InvalidArgument("Apple append inherited prefix differs from its exact bytes");
    }
  }
  if (new_bytes != active_bytes) return Status::InvalidArgument("Apple append does not consume exactly its active rows");
  try {
    slot.ranges.assign(configuration.ranges.begin(), configuration.ranges.end());
    slot.copies.resize(slot.ranges.size());
  } catch (...) { return Status::ResourceExhausted("Apple append copy descriptors"); }
  HYLE_RETURN_IF_ERROR(an::clear_copy_recording(*slot.recording));
  slot.source.reset(); slot.rows_owner.reset(); slot.rows = {};
  slot.layout = configuration.layout; slot.capacity = configuration.capacity;
  slot.configured = true;
  return Status::Ok();
}
Status configure_device(void *prepared,void *invocation,const bd::NativeDeviceAppendConfiguration &configuration) noexcept {
  const auto &artifact=*static_cast<Artifact *>(prepared);auto &slot=*static_cast<Slot *>(invocation);
  HYLE_RETURN_IF_ERROR(open(artifact));
  HYLE_RETURN_IF_ERROR(an::copy_recording_reuse_gate(*slot.recording));
  if(&slot.artifact!=&artifact || !artifact.device_lengths || slot.pages.valid() ||
      !artifact.descriptor.schema.owns(configuration.layout) || !configuration.capacity.valid() ||
      configuration.capacity.storage().resource()!=artifact.resource ||
      configuration.layout.facts().query_rows!=artifact.query_rows ||
      configuration.requests.size()!=artifact.descriptor.schema.capacity().max_requests ||
      configuration.pages.size()>artifact.range_capacity)
    return Status::InvalidArgument("Apple device append needs its exact original capacity episode");
  try{
    slot.device_pages.clear();
    for(const auto &page:configuration.pages)
      slot.device_pages.push_back({page.page,page.destination,page.prefix});
  }catch(...){return Status::ResourceExhausted("Apple device append original page descriptors");}
  HYLE_RETURN_IF_ERROR(an::clear_copy_recording(*slot.recording));
  slot.configured=slot.bound=slot.sealed=slot.submitted=false;
  slot.device_inputs={};slot.device_owners={};slot.device_views={};
  HYLE_RETURN_IF_ERROR(an::configure_device_append_recording(*slot.recording,
      configuration.capacity.storage(),configuration.requests,slot.device_pages));
  slot.layout=configuration.layout;slot.capacity=configuration.capacity;slot.configured=true;
  return Status::Ok();
}
Status bind_device(Slot &slot,std::span<const RegionValue> inputs,std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact=slot.artifact;
  if(!slot.configured || inputs.size()!=artifact.boundary_count || !outputs.empty())
    return Status::InvalidArgument("Apple device append needs its configured native boundary");
  HYLE_RETURN_IF_ERROR(an::copy_recording_reuse_gate(*slot.recording));
  slot.device_inputs={};slot.device_owners={};slot.device_views={};
  for(std::size_t i=0;i<3;++i){
    if(artifact.ports[i]>=inputs.size())return Status::InvalidArgument("Apple device append logical port is absent");
    const auto &input=inputs[artifact.ports[i]];const auto &description=artifact.descriptions[i];
    if(input.description!=description)return Status::InvalidArgument("Apple device append original input description differs");
    if(input.reference.valid()){
      HYLE_ASSIGN_OR_RETURN(slot.device_inputs[i].source,native_output_source(input.reference,artifact.runtime,description));
    }else{
      if(!input.tensor.valid() || input.tensor.description()!=description)
        return Status::InvalidArgument("Apple device append has no exact concrete native input");
      HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());
      HYLE_ASSIGN_OR_RETURN(auto range,input.tensor.storage().subview(0,bytes));
      if(input.lease){
        HYLE_ASSIGN_OR_RETURN(auto owner,input.lease->view());
        if(owner.resource()!=range.resource() || owner.native_handle()!=range.native_handle() ||
            range.offset()<owner.offset() || range.offset()-owner.offset()>owner.size() ||
            range.size()>owner.size()-(range.offset()-owner.offset()))
          return Status::InvalidArgument("Apple device append lease does not cover its original input");
        HYLE_ASSIGN_OR_RETURN(slot.device_owners[i],input.lease->subview(range.offset()-owner.offset(),range.size()));
      }
      HYLE_ASSIGN_OR_RETURN(slot.device_inputs[i].value,an::MemoryAccess::borrow_tensor(input.tensor));
      slot.device_views[i]=input.tensor;
    }
  }
  HYLE_RETURN_IF_ERROR(an::bind_device_append_inputs(*slot.recording,slot.device_inputs));
  slot.bound=true;return Status::Ok();
}
Status bind(void *prepared, void *invocation, std::span<const RegionValue> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact = *static_cast<Artifact *>(prepared); auto &slot = *static_cast<Slot *>(invocation);
  slot.bound = false; slot.sealed = false; slot.submitted = false;
  if(artifact.device_lengths)return bind_device(slot,inputs,outputs);
  if (!slot.configured || inputs.size() != 1 || !outputs.empty() || inputs[0].description != artifact.rows)
    return Status::InvalidArgument("Apple append requires its configured exact rows input");
  HYLE_RETURN_IF_ERROR(an::copy_recording_reuse_gate(*slot.recording));
  slot.source.reset(); slot.rows_owner.reset(); slot.rows = {};
  if (inputs[0].reference.valid()) {
    HYLE_ASSIGN_OR_RETURN(slot.source, native_output_source(inputs[0].reference, artifact.runtime, artifact.rows));
    for (std::size_t index = 0; index < slot.ranges.size(); ++index)
      slot.copies[index] = {slot.ranges[index].destination, slot.ranges[index].prefix};
    HYLE_RETURN_IF_ERROR(an::bind_result_copy_recording(*slot.recording, slot.copies));
    HYLE_RETURN_IF_ERROR(an::bind_native_copy_source(*slot.recording, slot.source, artifact.rows));
    HYLE_RETURN_IF_ERROR(an::set_copy_preflight(*slot.recording, &preflight, &slot));
  } else {
    const auto &input = inputs[0];
    if (!input.tensor.valid() || input.tensor.description() != artifact.rows)
      return Status::InvalidArgument("Apple append has no concrete or native rows carrier");
    HYLE_ASSIGN_OR_RETURN(auto rows, input.tensor.storage().subview(0, artifact.input_bytes));
    if (input.lease) {
      HYLE_ASSIGN_OR_RETURN(auto owner, input.lease->view());
      if (rows.resource() != owner.resource() || rows.native_handle() != owner.native_handle() ||
          rows.offset() < owner.offset() || rows.offset()-owner.offset() > owner.size() ||
          rows.size() > owner.size()-(rows.offset()-owner.offset()))
        return Status::InvalidArgument("Apple append row lease does not cover its actual input");
      HYLE_ASSIGN_OR_RETURN(slot.rows_owner, input.lease->subview(rows.offset()-owner.offset(), rows.size()));
    }
    HYLE_ASSIGN_OR_RETURN(auto physical, an::copy_span(rows, mem::HostAccess::kRead));
    HYLE_RETURN_IF_ERROR(actual_aliases(slot, physical));
    for (std::size_t index = 0; index < slot.ranges.size(); ++index) {
      const auto &range = slot.ranges[index]; auto source = range.prefix;
      if (range.from_rows) { HYLE_ASSIGN_OR_RETURN(source, rows.subview(range.source_offset, range.bytes)); }
      slot.copies[index] = {range.destination, source};
    }
    HYLE_RETURN_IF_ERROR(an::bind_copy_recording(*slot.recording, slot.copies, true));
    HYLE_RETURN_IF_ERROR(an::set_copy_preflight(*slot.recording, nullptr, nullptr));
    slot.rows = input.tensor;
  }
  slot.bound = true; return Status::Ok();
}
Status before_read(void *, void *invocation) noexcept {
  const auto &slot = *static_cast<const Slot *>(invocation);
  if (!slot.submitted || !slot.pages.valid())
    return Status::FailedPrecondition("Apple append producer has no accepted current page episode");
  // Readability needs the exact original GPU prefix, not a scheduled tail's
  // public metadata callback. This proves all native/check errors and leaves
  // Source permissions and last-reader/slot retirement with their owners.
  // Device-controlled readers import the original event and inspect this
  // writer's GPU verdict before touching a page. Public retirement still
  // observes the full original copy/source outcome.
  return slot.artifact.device_lengths?Status::Ok():an::observe_copy_recording_on_owner(*slot.recording);
}
Status qualify_reader(void *prepared, void *, cd::CompletionEndpoint endpoint) noexcept {
  const auto &artifact = *static_cast<const Artifact *>(prepared);
  HYLE_RETURN_IF_ERROR(open(artifact));
  return imports_region_ordered_endpoint(artifact.runtime, endpoint) ? Status::Ok() :
      Status::InvalidArgument("Apple append reader belongs to a different Runtime");
}
const bd::NativeAppendRegionVTable append_table{
    .inspect = [](void *prepared, void *) noexcept -> Result<bd::NativeAppendRegionFacts> {
      const auto &artifact = *static_cast<const Artifact *>(prepared);
      return bd::NativeAppendRegionFacts{artifact.node, artifact.descriptor, artifact.rows, artifact.resource};
    },
    .pages = [](void *invocation) noexcept { return &static_cast<Slot *>(invocation)->pages; },
    .configure = &configure, .qualify_reader = &qualify_reader, .before_read = &before_read};
const bd::NativeDeviceAppendRegionVTable device_append_table{
    .length_inputs=[](void *prepared) noexcept {
      const auto &a=*static_cast<const Artifact *>(prepared);return std::array{a.inputs[1],a.inputs[2]};
    },
    .configure=&configure_device,
    .status=[](void *prepared,void *invocation) noexcept -> Result<compute::ConstTensorView> {
      auto &slot=*static_cast<Slot *>(invocation);
      if(&slot.artifact!=prepared || !slot.artifact.device_lengths || !slot.configured)
        return std::unexpected(Status::InvalidArgument("Apple append status has no original configured writer"));
      return an::device_append_status(*slot.recording);
    }};
const gd::RegionOrderedWriteVTable ordered_table{
    [](const ProviderRegionPrepared &prepared, void *invocation, const cd::CompletionEndpoint *reader) noexcept -> Status {
      if (!prepared.state || !invocation) return Status::InvalidArgument("Apple append proof has no actual selected slot");
      auto &slot = *static_cast<Slot *>(invocation);
      if (&slot.artifact != prepared.state) return Status::InvalidArgument("Apple append proof artifact differs");
      HYLE_RETURN_IF_ERROR(open(slot.artifact));
      return reader ? qualify_reader(prepared.state, invocation, *reader) : Status::Ok();
    }};
const gd::RegionBindingReleaseVTable release_table{
    .validate = [](void *, void *invocation) noexcept -> Status {
      return an::copy_recording_retired_gate(*static_cast<Slot *>(invocation)->recording);
    },
    .release = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      HYLE_FATAL_ASSERT(an::clear_copy_recording(*slot.recording).ok());
      slot.source.reset(); slot.rows_owner.reset(); slot.rows = {};
      slot.device_inputs={};slot.device_owners={};slot.device_views={};
      slot.bound = false; slot.sealed = false; slot.submitted = false;
    }};
const gd::RegionDependenciesVTable dependencies_table{
    [](void *, void *invocation, std::size_t capacity) noexcept -> Status {
      return an::reserve_copy_dependencies(*static_cast<Slot *>(invocation)->recording, capacity);
    }, true};
const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Artifact *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return compute::AliasSupport{}; },
    .create_invocation = [](void *prepared, mem::AllocatorRef allocator, mem::BufferView, void **out) noexcept -> Status {
      *out = nullptr; auto &artifact = *static_cast<Artifact *>(prepared);
      auto *slot = allocator.construct<Slot>(allocator, artifact);
      if (!slot) return Status::ResourceExhausted("Apple append slot metadata");
      *out = slot;
      HYLE_ASSIGN_OR_RETURN(slot->recording, an::prepare_copy_recording_runtime(artifact.runtime,
          artifact.range_capacity, allocator, artifact.max_bytes, an::CopyRecordingMode::kReplayable));
      if(artifact.device_lengths){
        const auto capacity=artifact.descriptor.schema.capacity();
        HYLE_RETURN_IF_ERROR(an::prepare_device_append_recording(*slot->recording,
            {{capacity.page_size,capacity.max_pages,artifact.query_rows,artifact.range_capacity+capacity.max_requests+1,artifact.row_bytes},capacity.max_requests},artifact.rows));
      }
      try {
        if(artifact.device_lengths)slot->device_pages.reserve(artifact.range_capacity);
        else { slot->ranges.reserve(artifact.range_capacity); slot->copies.reserve(artifact.range_capacity); }
      }
      catch (...) { return Status::ResourceExhausted("Apple append reserved descriptors"); }
      return Status::Ok();
    },
    .destroy_invocation = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Slot *>(p)); },
    .delegated_submit = [](void *, void *invocation, std::span<const cd::CompletionDependency> dependencies,
        std::span<const compute::ConstTensorView> inputs, std::span<const compute::TensorView> outputs) noexcept -> Result<compute::Completion> {
      auto &slot = *static_cast<Slot *>(invocation);
      if(slot.artifact.device_lengths){
        if(!slot.sealed || !slot.pages.valid() || inputs.size()!=slot.artifact.boundary_count || !outputs.empty())
          return std::unexpected(Status::FailedPrecondition("Apple device append current boundary/pages are not sealed"));
        for(std::size_t i=0;i<3;++i)if(!slot.device_inputs[i].source.valid()){
          const auto &actual=inputs[slot.artifact.ports[i]];
          if(!actual.valid() || !same_range(actual.storage(),slot.device_views[i].storage()))
            return std::unexpected(Status::FailedPrecondition("Apple device append original native input changed"));
        }
      }else if (!slot.sealed || !slot.pages.valid() || inputs.size() != 1 || !outputs.empty() ||
          (!slot.source.valid() && (!inputs[0].valid() || !same_range(inputs[0].storage(), slot.rows.storage()))))
        return std::unexpected(Status::FailedPrecondition("Apple append current ports/pages are not sealed"));
      HYLE_ASSIGN_OR_RETURN(auto completion, an::submit_copy_recording(*slot.recording, dependencies));
      slot.submitted = true; return completion;
    },
    .endpoint = [](void *prepared, void *) noexcept { return region_ordered_endpoint(static_cast<Artifact *>(prepared)->runtime); },
    .can_import_endpoint = [](void *prepared, void *, cd::CompletionEndpoint source) noexcept {
      return imports_region_ordered_endpoint(static_cast<Artifact *>(prepared)->runtime, source);
    },
    .prepare_bindings = &bind,
    .seal_bindings = [](void *, void *invocation) noexcept -> Status {
      auto &slot = *static_cast<Slot *>(invocation); slot.sealed = false;
      if (!slot.configured || !slot.bound || !slot.pages.valid() || slot.pages.layout() != slot.layout)
        return Status::FailedPrecondition("Apple append has no exact current page transaction");
      HYLE_RETURN_IF_ERROR(an::copy_recording_reuse_gate(*slot.recording));
      slot.sealed = true; return Status::Ok();
    },
    .query_extension = [](void *prepared, compute::ProviderExtensionId id) noexcept -> const void * {
      if (id == gd::kRegionDependenciesExtensionId) return &dependencies_table;
      if (id == bd::kNativeAppendRegionExtensionId) return &append_table;
      if (id == bd::kNativeDeviceAppendRegionExtensionId && static_cast<Artifact *>(prepared)->device_lengths)
        return &device_append_table;
      if (id == gd::kRegionOrderedWriteExtensionId) return &ordered_table;
      if (id == gd::kRegionBindingReleaseExtensionId) return &release_table;
      return nullptr;
    }};
} // namespace
bool native_paged_append_operation(const compute::Operation &operation) noexcept {
  return operation.custom_as<batch::PagedAppend>().has_value() || operation.custom_as<batch::DeviceLengthsPagedAppend>().has_value();
}
Status propose_native_paged_append(void *, const RegionDiscoveryRequest &request,
    std::span<const std::uint8_t> needed, RegionProposalWriter &writer) noexcept {
  for (std::size_t ordinal = 0; ordinal < request.graph.node_count(); ++ordinal) {
    if (!needed[ordinal] || (!request.allowed_nodes.empty() && !request.allowed_nodes[ordinal])) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, request.graph.node_id(ordinal));
    HYLE_ASSIGN_OR_RETURN(auto kind, request.graph.operation_kind(node));
    if (kind != compute::OperationKind::kCustom) continue;
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph.operation(node));
    if (!native_paged_append_operation(operation)) continue;
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added, writer.try_add({.nodes = nodes, .route = kNativePagedAppendRoute,
        .replay_class = compute::ProviderReplayClass::kManaged,
        .implementation = compute::ProviderImplementationId{0x415050454e444150ULL}, .preference = 100}));
    if (!added) break;
  }
  return Status::Ok();
}
Status native_paged_append_resources(void *, const RegionCompileRequest &request,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *proposal) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto append, qualify(request)); (void)append;
  if (inputs.size() != request.inputs.size() || !outputs.empty()) return Status::InvalidArgument("Apple append physical boundary differs");
  const cd::ResourceBindingRequirement resource{
      .required_properties = mem::MemoryProperty::kHostAccessible | mem::MemoryProperty::kDeviceAccessible |
          mem::MemoryProperty::kHostDeviceCoherent, .minimum_alignment = 1};
  std::ranges::fill(inputs,RegionInputRequirement{resource,true});
  *proposal = {.runtime = runtime_properties, .workspace_resource = resource, .preserves_effects = true};
  return Status::Ok();
}
compute::ProviderPrepareOutcome compile_native_paged_append(void *context,
    const RegionCompileRequest &request, mem::AllocatorRef allocator, ProviderRegionPrepared *out) noexcept {
  using Outcome = compute::ProviderPrepareOutcome; *out = {};
  auto append = qualify(request);
  if (!append) return append.error().IsUnimplemented() ? Outcome::unsupported(append.error()) : Outcome::error(append.error());
  if (request.input_resources.size() != request.inputs.size() ||
      std::ranges::any_of(request.input_resources,[](auto resource){return !an::is_copy_resource(resource);}))
    return Outcome::unsupported(Status::Unimplemented("Apple append needs original shared Metal storage"));
  auto *artifact = allocator.construct<Artifact>();
  if (!artifact) return Outcome::error(Status::ResourceExhausted("Apple append artifact metadata"));
  *out = {.vtable = &prepared_table, .state = artifact};
  const auto build = [&]() noexcept -> Status {
    artifact->runtime = source_runtime(context); artifact->node = request.nodes[0];
    artifact->descriptor = std::move(append->descriptor);artifact->device_lengths=append->device_lengths;
    artifact->boundary_count=static_cast<std::uint32_t>(request.inputs.size());
    const auto count=artifact->device_lengths?3u:1u;
    for(std::size_t i=0;i<count;++i){
      HYLE_ASSIGN_OR_RETURN(artifact->inputs[i],request.graph->node_input(artifact->node,i));
      const auto found=std::ranges::find(request.inputs,artifact->inputs[i]);
      if(found==request.inputs.end())return Status::InvalidArgument("Apple append original logical input is absent");
      artifact->ports[i]=static_cast<std::uint32_t>(found-request.inputs.begin());
      HYLE_ASSIGN_OR_RETURN(artifact->descriptions[i],request.graph->value_description(artifact->inputs[i]));
    }
    artifact->resource=request.input_resources[artifact->ports[0]];artifact->rows=artifact->descriptions[0];
    HYLE_ASSIGN_OR_RETURN(artifact->input_bytes, artifact->rows.required_bytes());
    HYLE_ASSIGN_OR_RETURN(artifact->query_rows, bd::append_query_rows(artifact->descriptor, artifact->rows));
    const auto query = static_cast<std::size_t>(artifact->query_rows);
    artifact->row_bytes = artifact->input_bytes / query;
    const auto capacity = artifact->descriptor.schema.capacity();
    if(artifact->device_lengths){
      artifact->range_capacity=std::min(capacity.max_pages,capacity.max_page_refs);
      if(!artifact->range_capacity || artifact->range_capacity>UINT32_MAX/3-capacity.max_requests-1 ||
          !capacity.page_size || !artifact->row_bytes ||
          std::uint64_t(artifact->range_capacity)*capacity.page_size>SIZE_MAX/artifact->row_bytes)
        return Status::Unimplemented("Apple device append capacity exceeds its native descriptor/dispatch extent");
      artifact->max_bytes=std::size_t(artifact->range_capacity)*capacity.page_size*artifact->row_bytes;
    }else{
    const auto active = std::min(query, static_cast<std::size_t>(capacity.max_requests));
    if (query > UINT32_MAX / 2 || active > UINT32_MAX / 2 - query || !capacity.page_size ||
        (capacity.page_size > 1u && active > (SIZE_MAX - query) / (capacity.page_size - 1u)))
      return Status::OutOfRange("Apple append fixed copy geometry overflows");
    const auto copied_rows = query + active * (capacity.page_size - 1u);
    if (!artifact->row_bytes || copied_rows > SIZE_MAX / artifact->row_bytes)
      return Status::OutOfRange("Apple append byte geometry overflows");
    artifact->range_capacity = static_cast<std::uint32_t>(query + active);
    artifact->max_bytes = copied_rows * artifact->row_bytes;
    }
    out->properties = {.persistent_bytes = sizeof(Artifact), .runtime = runtime_properties,
        .preserves_effects = true, .invocation_private_bytes = std::nullopt};
    return Status::Ok();
  };
  const auto status = build();
  return status.ok()?Outcome::prepared():status.IsUnimplemented()?Outcome::unsupported(status):Outcome::error(status);
}
} // namespace hyle::graph::apple::detail
