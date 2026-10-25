#include "native_paged_region.h"
#include "hyle/compute/detail/batch_provider.h"
#include "native_paged_pending.h"
#include "native_output.h"
#include "native_region.h"

#include "../compute_route_apple/apple_copy_internal.h"
#include "../compute_route_apple/apple_file_stream_access.h"
#include "../graph_batch/native_batch_regions.h"
#include "hyle/compute/batch_ops.h"
#include "hyle/compute/detail/apple_batch_access.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/detail/materialized_native_controller.h"
#include "hyle/graph/apple.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <optional>

namespace hyle::graph::apple::detail {
namespace {
namespace ca = compute::apple;
namespace an = ca::detail;
namespace cd = compute::detail;
namespace bd = graph::batch::detail;
namespace mx = mlx::core;
const unsigned array_type = 0;
constexpr compute::ProviderRuntimeProperties runtime_properties{
    .mode = compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
    .replay_class = compute::ProviderReplayClass::kManaged,
    .uses_private_threads = true};
struct Recipe final {
  compute::BatchLayoutVersion layout;
  batch::PagedAttentionOptions options;
  compute::BatchCapacity capacity{};
  compute::MlaCacheLayout mla = compute::MlaCacheLayout::kNone;
  bool external_metadata = false;
};
Result<Recipe> recipe(const compute::Operation &operation) noexcept {
  if (const auto value = operation.custom_as<batch::PagedAttention>())
    return Recipe{value->layout, value->options, value->layout.origin_schema().capacity(), value->layout.origin_schema().mla_layout()};
  if (const auto value = operation.custom_as<batch::PagedMlaAttention>())
    return Recipe{value->layout, value->options, value->layout.origin_schema().capacity(), value->layout.origin_schema().mla_layout()};
  if (const auto value = operation.custom_as<compute::PagedAttention>())
    return Recipe{{}, {.scale = value->scale, .causal = value->causal,
        .window_left = value->window_left, .allow_bf16_partials = value->allow_bf16_partials},
        value->capacity, compute::MlaCacheLayout::kNone, true};
  if (const auto value = operation.custom_as<compute::PagedMlaAttention>()) {
    HYLE_ASSIGN_OR_RETURN(auto capacity, compute::mla_batch_capacity(value->capacity));
    return Recipe{{}, {.scale = value->scale, .causal = value->causal,
        .window_left = value->window_left}, capacity, value->capacity.layout, true};
  }
  return std::unexpected(Status::Unimplemented("not paged attention"));
}
const std::shared_ptr<an::Runtime> &runtime(void *context) noexcept {
  return source_runtime(context);
}
Status qualify(const std::shared_ptr<an::Runtime> &owner, const Recipe &value,
    const compute::BatchLayoutVersion &layout) noexcept {
  if (!owner || !std::isfinite(value.options.scale))
    return Status::InvalidArgument("Apple page needs its actual Runtime and owned mathematical layout");
  {
    std::lock_guard lock(owner->queue_mutex);
    if (!owner->accepting) return Status::FailedPrecondition("Apple page source is closed");
    if (!owner->failure.ok()) return owner->failure;
  }
  HYLE_RETURN_IF_ERROR(compute::validate_batch_capacity(value.capacity));
  HYLE_RETURN_IF_ERROR(compute::validate_batch_attention_options(value.options.causal, value.options.window_left));
  if (value.capacity.storage != compute::BatchStorage::kPaged || value.capacity.page_size != 16 ||
      value.capacity.max_pages > std::uint32_t(INT32_MAX/16) ||
      !an::PagedBatchAccess::cache_domain(value.capacity))
    return Status::Unimplemented("Apple page dtype/encoding/geometry is unsupported");
  if (!layout.valid())
    return value.external_metadata ? Status::Ok() : Status::InvalidArgument("Apple page owned layout is empty");
  if (!cd::BatchLayoutAccess::same_topology(layout.origin_schema().capacity(), value.capacity) || layout.origin_schema().mla_layout() != value.mla ||
      (!value.external_metadata && !value.layout.origin_schema().owns(layout)))
    return Status::InvalidArgument("Apple page layout schema differs");
  return an::PagedBatchAccess::validate_paged(
      owner, layout, value.capacity, value.options.causal, value.options.window_left);
}
Result<Recipe> qualify(void *context, const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.graph->dimension_count() || request.route != kNativePagedRoute ||
      request.nodes.size() != 1 || request.outputs.size() > 1 ||
      request.has_unknown_effects || !request.effect_accesses.empty())
    return std::unexpected(Status::Unimplemented("Apple page needs one fixed pure typed node"));
  HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(request.nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto value, recipe(operation));
  HYLE_RETURN_IF_ERROR(qualify(runtime(context), value, value.layout));
  return value;
}
struct Preparation final {
  std::shared_ptr<an::Runtime> runtime;
  std::uint32_t query_rows = 0;
};
struct Resource final {
  mem::MemoryResourceRef resource;
  bool valid() const noexcept { return resource.valid(); }
  mem::MemoryResourceRef ref() const noexcept { return resource; }
};
struct Traits final {
  using Context = Preparation; using Memory = Resource; using Program = ca::Program;
  using Invocation = ca::Invocation; using CopyInvocation = ca::CopyBatchInvocation;
  using View = mx::array; using Submission = compute::OwnedResult<ca::Results>;
  using CopyAccess = an::CopyStreamAccess;
  static auto invocation(const Program &program, mem::AllocatorRef allocator) {
    return program.create_invocation(allocator);
  }
  static auto copy_program(const Context &context, std::span<const compute::CopyResourceSpec> specs,
      compute::CopyBatchOptions options) {
    return an::CopyStreamAccess::prepare_program(context.runtime, specs, std::move(options));
  }
  static auto copy_invocation(const ca::CopyBatchProgram &program) { return ca::CopyBatchInvocation::create(program); }
  static auto prepare_program(const Context &context, const compute::NativePagedBatchProgram &program,
      compute::BatchLayoutVersion layout, const compute::WorkloadContext &workload) {
    return an::PagedBatchAccess::prepare_paged_graph(context.runtime, program, std::move(layout), context.query_rows, workload);
  }
  static auto submit(Invocation &invocation, compute::LeaseUseBatch uses) {
    return an::PagedBatchAccess::submit(invocation, std::move(uses));
  }
  static auto submit_pending(Invocation &invocation, compute::LeaseUseBatch uses,
      const ca::CopyBatchUpdate &source) {
    return an::PagedBatchAccess::submit_pending(invocation, std::move(uses), source);
  }
  static auto wait_native(Invocation &invocation) { return an::PagedBatchAccess::wait_native(invocation); }
  static auto reuse_gate(Invocation &invocation) { return an::PagedBatchAccess::reuse_gate(invocation); }
  static auto release_inputs(Invocation &invocation) { return an::PagedBatchAccess::release_inputs(invocation); }
  static auto update_layout(Invocation &invocation, const compute::BatchLayoutVersion &layout) {
    return an::PagedBatchAccess::update_layout(invocation, layout);
  }
};
using Controller = cd::MaterializedNativeState<Traits, cd::PagedMetadataPolicy>;
struct Artifact final {
  Recipe recipe;
  compute::NativePagedBatchProgram native_recipe;
  NodeId node;
  std::array<ValueId,3> inputs{};
  std::array<std::uint32_t,3> ports{};
  std::size_t count = 0, public_outputs = 0;
  ValueId metadata;
  std::uint32_t metadata_port = 0;
  std::array<compute::TensorDesc,4> descriptions;
  compute::TensorDesc output;
  Preparation preparation;
  Resource device, staging;
  ca::Program program;
  compute::ResourceBudget budget;
  compute::WorkloadContext workload;
};
struct Slot final {
  Slot(mem::AllocatorRef selected, Artifact &owner)
      : allocator(selected), artifact(owner) {}
  ~Slot() { if (controller) allocator.destroy(controller); }
  mem::AllocatorRef allocator;
  Artifact &artifact;
  Controller *controller = nullptr;
  std::array<compute::PagedArenaRead,2> frames;
  std::array<compute::PagedArenaPendingRead,2> pending_frames;
  struct Written final {
    compute::BatchLayoutVersion layout;
    compute::ConstTensorView capacity;
    bd::NativeAppendRegionRef producer;
    bool valid() const noexcept { return producer.table != nullptr; }
  };
  std::array<Written,2> written;
  NativePagePendingSource *pending_source = nullptr; // Borrowed from BatchResources.
  std::array<an::InputBinding,4> bindings;
  cd::ResultStorageOwner result;
  std::optional<mx::array> projected;
  std::optional<compute::ReadLease> exported;
  std::mutex output_mutex;
};
void clear_output(Slot &slot) noexcept { slot.exported.reset(); slot.projected.reset(); }
bool empty_program(const Artifact &, const Slot &slot) noexcept {
  if (!slot.controller) return false;
  const auto &controller = *slot.controller;
  const auto &layout = controller.version.layout();
  return controller.prepared && controller.recipe.schema().owns(layout) &&
      !layout.facts().query_rows && !layout.facts().page_refs &&
      an::PagedBatchAccess::empty_cache_reads(controller.native);
}
Result<compute::NativePagedBatchProgram> page_recipe(const Artifact &artifact,
    compute::BatchSchema schema, mem::AllocatorRef allocator) noexcept {
  if (!cd::BatchLayoutAccess::same_topology(schema.capacity(),artifact.recipe.capacity) ||
      schema.mla_layout()!=artifact.recipe.mla)
    return std::unexpected(Status::InvalidArgument("Apple page recipe topology differs"));
  HYLE_ASSIGN_OR_RETURN(schema,schema.with_cache_storage(
      artifact.recipe.capacity.key_storage,artifact.recipe.capacity.value_storage));
  const auto refs = std::size_t(schema.capacity().max_page_refs);
  if (refs > SIZE_MAX / sizeof(std::int32_t))
    return std::unexpected(Status::OutOfRange("Apple page metadata byte extent overflow"));
  return compute::NativePagedBatchProgram::prepare(std::move(schema),
      {.max_host_metadata_bytes = refs * sizeof(std::int32_t),
       .scale = artifact.recipe.options.scale, .causal = artifact.recipe.options.causal,
       .window_left = artifact.recipe.options.window_left,
       .allocator = allocator, .resource_budget = artifact.budget});
}
Result<Controller *> make_controller(Slot &slot, compute::NativePagedBatchProgram recipe,
    ca::Program program = {}) noexcept {
  auto *controller = slot.allocator.construct<Controller>(slot.allocator);
  if (!controller) return std::unexpected(Status::ResourceExhausted("Apple page controller metadata"));
  const auto &artifact = slot.artifact;
  const auto status = controller->initialize(artifact.preparation, std::move(program), std::move(recipe),
      artifact.device, artifact.staging, {.max_metadata_backing_bytes = SIZE_MAX,
          .allocator = slot.allocator, .resource_budget = artifact.budget}, artifact.workload,
      artifact.recipe.external_metadata);
  if (!status.ok()) { slot.allocator.destroy(controller); return std::unexpected(status); }
  return controller;
}
Status prepare(void *prepared, void *invocation, compute::BatchLayoutVersion layout) noexcept {
  auto &artifact = *static_cast<Artifact *>(prepared); auto &slot = *static_cast<Slot *>(invocation);
  if (!layout.valid() || layout.facts().query_rows > artifact.preparation.query_rows ||
      (!artifact.recipe.external_metadata && layout.facts().query_rows != artifact.preparation.query_rows))
    return Status::InvalidArgument("Apple page schema/active output shape changed");
  HYLE_RETURN_IF_ERROR(qualify(artifact.preparation.runtime, artifact.recipe, layout));
  slot.result.reset(); clear_output(slot);
  if (slot.controller) {
    HYLE_RETURN_IF_ERROR(slot.controller->open());
    HYLE_RETURN_IF_ERROR(an::PagedBatchAccess::reuse_gate(slot.controller->native));
    if (slot.controller->uses.active_batches()) return Status::Unavailable("Apple page permissions still in use");
  }
  if (!slot.controller || !slot.controller->recipe.schema().owns(layout) ||
      slot.controller->recipe.schema().capacity()!=artifact.recipe.capacity ||
      slot.controller->recipe.schema().mla_layout()!=artifact.recipe.mla) {
    // A capacity node has no schema at compile time. Retain this actual owned
    // NodeLayout only after its cold replacement has prepared successfully.
    HYLE_ASSIGN_OR_RETURN(auto recipe, page_recipe(artifact, layout.origin_schema(), slot.allocator));
    HYLE_ASSIGN_OR_RETURN(auto replacement, make_controller(slot, std::move(recipe)));
    const auto status = replacement->prepare(layout);
    if (!status.ok()) { slot.allocator.destroy(replacement); return status; }
    auto *previous = slot.controller; slot.controller = replacement;
    if (previous) slot.allocator.destroy(previous);
  } else {
    HYLE_RETURN_IF_ERROR(slot.controller->prepare(std::move(layout)));
  }
  slot.frames = {}; slot.pending_frames = {}; slot.written = {}; slot.pending_source = nullptr;
  return Status::Ok();
}
Status reset(void *, void *invocation) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  slot.result.reset(); clear_output(slot);
  if (!slot.controller) return Status::Ok();
  if (slot.controller->native.valid()) HYLE_RETURN_IF_ERROR(an::PagedBatchAccess::release_inputs(slot.controller->native));
  for (auto &binding : slot.bindings) { binding.value.reset(); binding.source.reset(); }
  slot.frames = {}; slot.pending_frames = {}; slot.written = {}; slot.pending_source = nullptr;
  slot.controller->keys = {}; slot.controller->values = {}; slot.controller->reads = {};
  slot.controller->prepared = false; slot.controller->bound = false;
  return Status::Ok();
}
template<class Frame> Status validate_frame(void *prepared, void *invocation,
    ValueId input, const Frame &value, bool pending = false) noexcept {
  const auto &artifact = *static_cast<Artifact *>(prepared); const auto &slot = *static_cast<Slot *>(invocation);
  if (!slot.controller) return Status::FailedPrecondition("Apple page layout is not prepared");
  const auto &controller = *slot.controller;
  if (!controller.prepared || (pending && controller.bound) || !value.valid() || value.layout() != controller.version.layout())
    return Status::InvalidArgument("Apple page frame needs its prepared owned layout");
  if (pending && std::ranges::any_of(slot.written, [](const auto &frame) { return frame.valid(); }))
    return Status::Unimplemented("Apple page cannot mix external copy Update with same-Graph writers");
  bool selected = false;
  for (std::size_t port = 1; port < artifact.count; ++port) if (artifact.inputs[port] == input) {
    selected = true;
    if ((pending && slot.frames[port-1].valid()) || slot.pending_frames[port-1].valid() || slot.written[port-1].valid())
      return Status::FailedPrecondition("Apple page frame needs reset before rebinding");
    const auto capacity = controller.recipe.schema().capacity();
    HYLE_RETURN_IF_ERROR(cd::PagedArenaAccess::validate_storage(value, port == 1 ? capacity.key_storage : capacity.value_storage));
    HYLE_ASSIGN_OR_RETURN(auto actual, cd::PagedArenaAccess::tensor(value));
    HYLE_ASSIGN_OR_RETURN(auto expected, controller.recipe.input_description(value.layout(), port));
    if (actual.description() != expected || !an::is_copy_resource(actual.storage().resource()))
      return Status::InvalidArgument("Apple page frame physical resource/description differs");
  }
  return selected ? Status::Ok() : Status::InvalidArgument("input is not this Apple page reader");
}
Status frame(void *prepared, void *invocation, ValueId input,
    const compute::PagedArenaRead &value) noexcept {
  return validate_frame(prepared, invocation, input, value);
}
Status validate_written(void *prepared, void *invocation, ValueId input,
    compute::BatchLayoutVersion layout, compute::ConstTensorView capacity,
    const bd::NativeAppendRegionRef &producer) noexcept {
  const auto &artifact = *static_cast<Artifact *>(prepared); auto &slot = *static_cast<Slot *>(invocation);
  if (!slot.controller || !slot.controller->prepared || layout != slot.controller->version.layout() || !capacity.valid() ||
      !producer.table || !producer.table->inspect || !producer.table->qualify_reader || !producer.table->before_read ||
      !producer.implementation || !producer.prepared || !producer.invocation)
    return Status::InvalidArgument("Apple same-Graph page needs its exact prepared layout and native writer");
  if (std::ranges::any_of(slot.pending_frames, [](const auto &frame) { return frame.valid(); }))
    return Status::Unimplemented("Apple page cannot mix external copy Update with same-Graph writers");
  for (std::size_t port = 1; port < artifact.count; ++port) if (artifact.inputs[port] == input) {
    if (slot.frames[port-1].valid() || slot.pending_frames[port-1].valid() || slot.written[port-1].valid())
      return Status::FailedPrecondition("reset Apple page input before changing its writer");
    HYLE_ASSIGN_OR_RETURN(auto facts, producer.table->inspect(producer.prepared, producer.invocation));
    const auto role = port == 1 ? (slot.controller->recipe.schema().mla_layout() == compute::MlaCacheLayout::kNone ?
        compute::BatchCacheRole::kKey : compute::BatchCacheRole::kMla) : compute::BatchCacheRole::kValue;
    const auto writer_capacity=facts.descriptor.schema.capacity();
    const auto reader_capacity=slot.controller->recipe.schema().capacity();
    if (facts.node != producer.node || facts.descriptor.role != role || !facts.descriptor.schema.owns(layout) ||
        (port==1?writer_capacity.key_storage:writer_capacity.value_storage)!=
            (port==1?reader_capacity.key_storage:reader_capacity.value_storage) ||
        facts.resource != capacity.storage().resource())
      return Status::InvalidArgument("Apple same-Graph bank role/schema/selected writer differs");
    HYLE_ASSIGN_OR_RETURN(auto description, slot.controller->recipe.input_description(layout, port));
    if (capacity.description() != description || !an::is_copy_resource(capacity.storage().resource()))
      return Status::InvalidArgument("Apple same-Graph bank description/resource differs");
    HYLE_RETURN_IF_ERROR(producer.table->qualify_reader(producer.prepared, producer.invocation,
        region_ordered_endpoint(artifact.preparation.runtime)));
    // Only addressability is imported; the common Graph owns exact page rights.
    HYLE_ASSIGN_OR_RETURN(auto borrowed, an::MemoryAccess::borrow_tensor(capacity));
    (void)borrowed;
    return Status::Ok();
  }
  return Status::InvalidArgument("input is not this Apple cache reader");
}
void commit_written(void *prepared, void *invocation, ValueId input,
    compute::BatchLayoutVersion layout, compute::ConstTensorView capacity,
    const bd::NativeAppendRegionRef &producer) noexcept {
  const auto &artifact = *static_cast<Artifact *>(prepared); auto &slot = *static_cast<Slot *>(invocation);
  for (std::size_t port = 1; port < artifact.count; ++port)
    if (artifact.inputs[port] == input) slot.written[port-1] = {std::move(layout), capacity, producer};
}
Status before_written_read(const void *opaque) noexcept {
  const auto &slot = *static_cast<const Slot *>(opaque);
  for (const auto &frame : slot.written) if (frame.valid())
    HYLE_RETURN_IF_ERROR(frame.producer.table->before_read(frame.producer.prepared, frame.producer.invocation));
  return Status::Ok();
}
const bd::NativeBatchRegionVTable page_table{
    [](void *prepared, void *invocation) noexcept -> Result<bd::NativeBatchRegionFacts> {
      const auto &artifact = *static_cast<Artifact *>(prepared); const auto &slot = *static_cast<Slot *>(invocation);
      return bd::NativeBatchRegionFacts{.node = artifact.node,
          .paged_program = artifact.recipe.external_metadata ? nullptr : &artifact.native_recipe,
          .capacity = artifact.recipe.capacity,
          .mla = artifact.recipe.mla, .causal = artifact.recipe.options.causal,
          .window_left = artifact.recipe.options.window_left,
          .inputs = std::span(artifact.inputs).first(artifact.count),
          .input_ports = std::span(artifact.ports).first(artifact.count),
          .metadata_input = artifact.metadata,
          .output_query_rows = artifact.preparation.query_rows,
          .layout = slot.controller ? slot.controller->version.layout() : compute::BatchLayoutVersion{}};
    }, &prepare, &frame,
    [](void *prepared, void *invocation, ValueId input, compute::PagedArenaRead value) noexcept {
      const auto &artifact = *static_cast<Artifact *>(prepared); auto &slot = *static_cast<Slot *>(invocation);
      for (std::size_t port = 1; port < artifact.count; ++port)
        if (artifact.inputs[port] == input) slot.frames[port-1] = value;
    }, &reset,
    [](void *, void *invocation) noexcept {
      const auto *controller = static_cast<Slot *>(invocation)->controller;
      return controller ? controller->report : compute::NativePagedBatchReport{};
    },
    &validate_written, &commit_written};
const bd::NativePendingRegionVTable pending_table{
    [](void *prepared, void *) noexcept -> Result<bd::NativePendingRegionFacts> {
      const auto &artifact = *static_cast<Artifact *>(prepared);
      return bd::NativePendingRegionFacts{std::span(artifact.inputs).subspan(1, artifact.count-1),
          std::span(artifact.ports).subspan(1, artifact.count-1),
          {&native_pending_source_table, artifact.preparation.runtime.get(),
              [](void *opaque, std::size_t capacity, mem::AllocatorRef allocator,
                  compute::ResourceBudget budget) noexcept -> Result<void *> {
                return create_native_pending_source(static_cast<Artifact *>(opaque)->preparation.runtime,
                    capacity, allocator, std::move(budget));
              }}};
    },
    [](void *prepared, void *invocation, ValueId input,
        const compute::PagedArenaPendingRead &value,
        const bd::NativePendingSourceOwner &source) noexcept -> Status {
      const auto &artifact = *static_cast<Artifact *>(prepared);
      HYLE_ASSIGN_OR_RETURN(auto actual, native_pending_source(source, artifact.preparation.runtime));
      (void)actual;
      return validate_frame(prepared, invocation, input, value, true);
    },
    [](void *prepared, void *invocation, ValueId input, compute::PagedArenaPendingRead value,
        const bd::NativePendingSourceOwner &source) noexcept {
      const auto &artifact = *static_cast<Artifact *>(prepared); auto &slot = *static_cast<Slot *>(invocation);
      auto actual = native_pending_source(source, artifact.preparation.runtime);
      HYLE_FATAL_ASSERT(actual.has_value());
      slot.pending_source = *actual;
      for (std::size_t port = 1; port < artifact.count; ++port)
        if (artifact.inputs[port] == input) slot.pending_frames[port-1] = value;
    },
    [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      slot.pending_frames = {}; slot.pending_source = nullptr;
    }};
Status create_slot(void *prepared, mem::AllocatorRef allocator, mem::BufferView, void **out) noexcept {
  *out = nullptr; auto &artifact = *static_cast<Artifact *>(prepared);
  auto *slot = allocator.construct<Slot>(allocator, artifact);
  if (!slot) return Status::ResourceExhausted("Apple page slot metadata");
  *out = slot;
  if (artifact.recipe.external_metadata) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(slot->controller, make_controller(*slot, artifact.native_recipe, artifact.program));
  HYLE_ASSIGN_OR_RETURN(slot->controller->version, artifact.native_recipe.version(artifact.recipe.layout));
  return Status::Ok();
}
Status bind(void *prepared, void *invocation, std::span<const RegionValue> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  auto &artifact = *static_cast<Artifact *>(prepared); auto &slot = *static_cast<Slot *>(invocation);
  if (!slot.controller) return Status::FailedPrecondition("Apple page layout is not prepared");
  auto &controller = *slot.controller;
  if (!controller.prepared || outputs.size() != artifact.public_outputs)
    return Status::FailedPrecondition("Apple page layout must be prepared before replay bindings");
  slot.result.reset(); clear_output(slot);
  HYLE_RETURN_IF_ERROR(an::InvocationAccess::reuse_gate(controller.native));
  for (auto &binding : slot.bindings) { binding.value.reset(); binding.source.reset(); }
  for (std::size_t port = 0; port < artifact.count; ++port) {
    HYLE_ASSIGN_OR_RETURN(auto description, controller.recipe.input_description(controller.version.layout(), port));
    if (!port && artifact.recipe.external_metadata) description = artifact.descriptions[0];
    if (artifact.ports[port] >= inputs.size()) return Status::InvalidArgument("Apple page boundary port differs");
    const auto &input = inputs[artifact.ports[port]];
    if (input.description != description) return Status::InvalidArgument("Apple page physical input description differs");
    if (!port) {
      controller.reads[0] = {};
      if (input.reference.valid()) {
        HYLE_ASSIGN_OR_RETURN(slot.bindings[0].source,
            native_output_source(input.reference, artifact.preparation.runtime, description));
      } else if (input.lease) {
        HYLE_ASSIGN_OR_RETURN(auto owner, input.lease->view());
        const auto wanted = input.tensor.storage();
        if (!input.tensor.valid() || owner.resource() != wanted.resource() || owner.native_handle() != wanted.native_handle() ||
            wanted.offset() < owner.offset() || wanted.offset()-owner.offset() > owner.size() ||
            wanted.size() > owner.size()-(wanted.offset()-owner.offset()))
          return Status::InvalidArgument("Apple page query lease does not cover its actual input");
        HYLE_ASSIGN_OR_RETURN(auto selected, input.lease->subview(wanted.offset()-owner.offset(), wanted.size()));
        HYLE_ASSIGN_OR_RETURN(slot.bindings[0].value, an::MemoryAccess::owned_tensor(selected, description, controller.import));
        controller.reads[0] = std::move(selected);
      } else {
        HYLE_ASSIGN_OR_RETURN(slot.bindings[0].value, an::MemoryAccess::borrow_tensor(input.tensor));
      }
    } else {
      const auto &written = slot.written[port-1];
      if (written.valid()) {
        if (!input.tensor.valid() || input.reference.valid())
          return Status::InvalidArgument("Apple same-Graph bank requires its concrete capacity carrier");
        const auto actual = written.capacity.storage(), wanted = input.tensor.storage();
        if (written.layout != controller.version.layout() ||
            actual.resource() != wanted.resource() || actual.native_handle() != wanted.native_handle() ||
            actual.offset() != wanted.offset() || actual.size() != wanted.size())
          return Status::InvalidArgument("Apple same-Graph bank differs from its exact cold writer binding");
        if (!empty_program(artifact, slot)) {
          HYLE_ASSIGN_OR_RETURN(slot.bindings[port].value, an::MemoryAccess::borrow_tensor(input.tensor));
        }
        continue;
      }
      const auto import = [&](const auto &permission) noexcept -> Status {
        if (!permission.valid() || permission.layout() != controller.version.layout() || input.reference.valid())
          return Status::FailedPrecondition("Apple cache port has no exact typed page frame");
        HYLE_ASSIGN_OR_RETURN(auto actual, cd::PagedArenaAccess::tensor(permission));
        const auto x = actual.storage(), y = input.tensor.storage();
        if (x.resource() != y.resource() || x.native_handle() != y.native_handle() || x.offset() != y.offset() || x.size() != y.size())
          return Status::InvalidArgument("Apple cache input differs from its exact bound frame");
        if (!empty_program(artifact, slot)) {
          HYLE_ASSIGN_OR_RETURN(slot.bindings[port].value,
              an::PagedBatchAccess::binding(x.resource(), permission, controller.import));
        }
        return Status::Ok();
      };
      HYLE_RETURN_IF_ERROR(slot.pending_frames[port-1].valid() ?
          import(slot.pending_frames[port-1]) : import(slot.frames[port-1]));
    }
  }
  if (artifact.recipe.external_metadata) {
    if (artifact.metadata_port >= inputs.size()) return Status::InvalidArgument("Apple page metadata boundary is missing");
    const auto &input = inputs[artifact.metadata_port];
    HYLE_ASSIGN_OR_RETURN(auto description, controller.recipe.schema().metadata_description());
    if (input.description != description || !input.tensor.valid() || input.tensor.description() != description ||
        !input.lease || input.reference.valid())
      return Status::InvalidArgument("Apple page metadata needs its exact typed readonly carrier");
    HYLE_ASSIGN_OR_RETURN(auto owner, input.lease->view());
    const auto wanted = input.tensor.storage();
    if (owner.resource() != wanted.resource() || owner.native_handle() != wanted.native_handle() ||
        wanted.offset() < owner.offset() || wanted.offset()-owner.offset() > owner.size() ||
        wanted.size() > owner.size()-(wanted.offset()-owner.offset()))
      return Status::InvalidArgument("Apple page metadata lease does not cover its actual carrier");
    const auto offsets = controller.recipe.schema().metadata_offsets();
    const auto bytes = controller.recipe.costs().index_bytes;
    if (offsets.bytes > wanted.size() || offsets.page_indices > offsets.bytes || bytes > offsets.bytes-offsets.page_indices)
      return Status::InvalidArgument("Apple page table exceeds its canonical metadata carrier");
    // Common preparation uploaded the full encoding once. This view retains
    // that original lease; no second page table allocation or upload exists.
    HYLE_ASSIGN_OR_RETURN(controller.reads[3], input.lease->subview(
        wanted.offset()-owner.offset()+offsets.page_indices, bytes));
  }
  if (empty_program(artifact, slot)) {
    HYLE_RETURN_IF_ERROR(an::InvocationAccess::prepare_inputs(controller.native, std::span(slot.bindings).first(1)));
  } else {
    HYLE_ASSIGN_OR_RETURN(auto metadata, controller.recipe.input_description(controller.version.layout(), artifact.count));
    HYLE_ASSIGN_OR_RETURN(slot.bindings[artifact.count].value,
        an::MemoryAccess::owned_tensor(controller.reads[3], metadata, controller.import));
    HYLE_RETURN_IF_ERROR(an::InvocationAccess::prepare_inputs(controller.native,
        std::span(slot.bindings).first(artifact.count+1)));
  }
  controller.keys = slot.frames[0]; controller.values = artifact.count == 3 ? slot.frames[1] : compute::PagedArenaRead{};
  controller.native_bank = controller.prepared_bank; controller.bound = true;
  return Status::Ok();
}
Status seal(void *, void *invocation) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  // Graph already guards public results; release this completed native owner
  // before asking the original slot whether a new result can be published.
  slot.result.reset();
  if (!slot.controller) return Status::FailedPrecondition("Apple page layout is not prepared");
  return an::InvocationAccess::reuse_gate(slot.controller->native);
}
Status project(Slot &slot, std::size_t port) noexcept {
  if (port >= slot.artifact.public_outputs) return Status::OutOfRange("Apple page output port");
  if (!slot.controller) return Status::FailedPrecondition("Apple page layout is not prepared");
  if (!slot.projected) { HYLE_ASSIGN_OR_RETURN(slot.projected, an::InvocationAccess::output(slot.controller->native, port)); }
  return Status::Ok();
}
Status export_port(Slot &slot, std::size_t port) noexcept {
  HYLE_RETURN_IF_ERROR(project(slot, port));
  if (slot.exported) return Status::Ok();
  if (!slot.projected->nbytes()) return Status::Unimplemented("empty Apple page output has no payload lease");
  HYLE_ASSIGN_OR_RETURN(slot.exported, an::MemoryAccess::export_readonly(*slot.projected, slot.artifact.output,
      {.allocator = slot.allocator, .resource_budget = slot.artifact.budget}));
  return Status::Ok();
}
Result<RegionValue> value(void *, void *invocation, std::size_t port) noexcept {
  auto &slot = *static_cast<Slot *>(invocation); std::lock_guard lock(slot.output_mutex);
  HYLE_RETURN_IF_ERROR(project(slot, port));
  RegionValue result{slot.artifact.output, {}, &array_type, &*slot.projected};
  if (!slot.projected->nbytes()) return result;
  HYLE_RETURN_IF_ERROR(export_port(slot, port));
  HYLE_ASSIGN_OR_RETURN(result.tensor, slot.exported->tensor(result.description));
  return result;
}
Result<compute::ReadLease> lease(void *, void *invocation, std::size_t port) noexcept {
  auto &slot = *static_cast<Slot *>(invocation); std::lock_guard lock(slot.output_mutex);
  HYLE_RETURN_IF_ERROR(export_port(slot, port)); return *slot.exported;
}
const NativeOutputVTable output_table{
    [](void *prepared, void *invocation, std::size_t port, const std::shared_ptr<an::Runtime> &owner,
        const compute::TensorDesc &description) noexcept -> Result<an::InputSource> {
      if (!prepared || !invocation) return std::unexpected(Status::InvalidArgument("Apple page source has no actual slot"));
      const auto &artifact = *static_cast<Artifact *>(prepared); const auto &slot = *static_cast<Slot *>(invocation);
      if (&slot.artifact != &artifact || !slot.controller || owner != artifact.preparation.runtime ||
          port >= artifact.public_outputs || description != artifact.output)
        return std::unexpected(Status::InvalidArgument("Apple page source owner/Runtime/physical port differs"));
      return an::InvocationAccess::source(slot.controller->native, port);
    },
    [](void *prepared, void *invocation, std::size_t port, const ca::NativeStorage &wanted) noexcept -> Result<NativeOutputOrigins> {
      if (!prepared || !invocation) return std::unexpected(Status::InvalidArgument("Apple page output has no actual slot"));
      const auto &artifact = *static_cast<Artifact *>(prepared); auto &slot = *static_cast<Slot *>(invocation);
      if (&slot.artifact != &artifact || port >= artifact.public_outputs)
        return std::unexpected(Status::InvalidArgument("Apple page output owner/physical port differs"));
      std::lock_guard lock(slot.output_mutex);
      HYLE_RETURN_IF_ERROR(project(slot, port));
      HYLE_ASSIGN_OR_RETURN(auto actual, an::MemoryAccess::storage_info(*slot.projected));
      NativeOutputOrigins result;
      if (!actual.view_bytes || !wanted.view_bytes || actual.buffer != wanted.buffer ||
          !(actual.offset_bytes <= wanted.offset_bytes ? wanted.offset_bytes-actual.offset_bytes < actual.view_bytes :
              actual.offset_bytes-wanted.offset_bytes < wanted.view_bytes)) return result;
      result.owned_overlap = true;
      HYLE_RETURN_IF_ERROR(export_port(slot, port));
      result.owner = *slot.exported;
      return result;
    }};
Result<compute::Completion> submit(void *, void *invocation,
    std::span<const cd::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView>, std::span<const compute::TensorView>) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  if (!slot.controller) return std::unexpected(Status::FailedPrecondition("Apple page layout is not prepared"));
  for (const auto dependency : dependencies)
    HYLE_RETURN_IF_ERROR(an::qualify_recording_dependency(slot.artifact.preparation.runtime, dependency));
  slot.result.reset(); clear_output(slot);
  const bool pending = std::ranges::any_of(slot.pending_frames, [](const auto &frame) { return frame.valid(); });
  auto accepted = [&]() -> Result<Traits::Submission> {
    const bool written = std::ranges::any_of(slot.written, [](const auto &frame) { return frame.valid(); });
    if (written) {
      if (pending) return std::unexpected(Status::Unimplemented("Apple page source modes cannot be mixed"));
      HYLE_RETURN_IF_ERROR(slot.controller->open());
      HYLE_RETURN_IF_ERROR(an::PagedBatchAccess::reuse_gate(slot.controller->native));
      if (!slot.controller->prepared || !slot.controller->bound)
        return std::unexpected(Status::FailedPrecondition("Apple same-Graph page is not prepared/bound"));
      HYLE_ASSIGN_OR_RETURN(auto uses, slot.controller->uses.begin());
      auto &reads = slot.controller->page_reads; reads.reset();
      for (const auto &lease : slot.controller->reads) if (lease.valid()) HYLE_RETURN_IF_ERROR(reads.add(lease));
      // Written roots are already admitted exactly once by the outer Graph.
      for (std::size_t port = 1; port < slot.artifact.count; ++port) if (!slot.written[port-1].valid())
        for (const auto &page : cd::PagedArenaAccess::pages(slot.frames[port-1])) HYLE_RETURN_IF_ERROR(reads.add(page));
      HYLE_RETURN_IF_ERROR(reads.admit(uses));
      auto result = an::InvocationAccess::submit_ordered(slot.controller->native, std::move(uses),
          &before_written_read, &slot);
      if (!result) { slot.controller->failed = true; slot.controller->prepared = false; slot.controller->bound = false; }
      return result;
    }
    if (!pending) return slot.controller->replay_ready();
    HYLE_RETURN_IF_ERROR(an::PagedBatchAccess::reuse_gate(slot.controller->native));
    if (!slot.pending_source)
      return std::unexpected(Status::FailedPrecondition("Apple page has no reserved pending source"));
    HYLE_RETURN_IF_ERROR(qualify_native_pending_source(*slot.pending_source, slot.pending_source->source.update()));
    return slot.controller->submit_pending<ca::CopyBatchUpdate, an::CopyStreamAccess>(
        slot.pending_frames[0], slot.pending_frames[1], slot.pending_source->source.update());
  }();
  if (!accepted) return std::unexpected(accepted.error());
  auto transfer = cd::ResultAccess::take_storage(std::move(*accepted));
  slot.result = std::move(transfer.owner); return std::move(transfer.completion);
}
const graph::detail::RegionProjectedReadVTable projected_table{
    [](const ProviderRegionPrepared &prepared, void *invocation, std::uint32_t port,
        cd::OperationPreparedAccess *out) noexcept -> Status {
      *out = {};
      if (!prepared.state || !invocation) return Status::InvalidArgument("Apple empty-page proof has no actual slot");
      const auto &artifact = *static_cast<const Artifact *>(prepared.state); const auto &slot = *static_cast<const Slot *>(invocation);
      bool cache = false;
      for (std::size_t p = 1; p < artifact.count; ++p) cache |= artifact.ports[p] == port;
      if (!cache || !empty_program(artifact, slot)) return Status::Unimplemented("Apple page port is not a prepared Q-only reader");
      *out = {.projected_input = port, .allows_empty_footprint = true}; return Status::Ok();
    }};
const graph::detail::RegionBindingReleaseVTable binding_release_table{
    .validate = [](void *, void *invocation) noexcept -> Status {
      const auto &slot = *static_cast<const Slot *>(invocation);
      if (!slot.controller || !slot.controller->native.valid())
        return slot.result ? Status::FailedPrecondition("native result has no invocation") : Status::Ok();
      return an::InvocationAccess::can_detach_inputs(slot.controller->native, slot.result);
    },
    .release = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      slot.result.reset();
      if (slot.controller && slot.controller->native.valid()) an::InvocationAccess::detach_inputs(slot.controller->native);
      clear_output(slot);
      for (auto &binding : slot.bindings) { binding.value.reset(); binding.source.reset(); }
      // Exact frames/layout are still caller-owned configuration, not withdrawn
      // ports. The next normal bind reconstructs imports from those permissions.
      if (slot.controller) slot.controller->bound = false;
    }};

const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Artifact *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return compute::AliasSupport{}; },
    .create_invocation = &create_slot,
    .destroy_invocation = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Slot *>(p)); },
    .delegated_submit = &submit,
    .endpoint = [](void *p, void *) noexcept { return region_ordered_endpoint(static_cast<Artifact *>(p)->preparation.runtime); },
    .can_import_endpoint = [](void *p, void *, cd::CompletionEndpoint source) noexcept {
      return imports_region_ordered_endpoint(static_cast<Artifact *>(p)->preparation.runtime, source);
    },
    .output_contract = [](void *, std::size_t) noexcept {
      return RegionOutputContract{RegionOutputKind::kDeferredGenerated, RegionOutputProjection::kNativeOnly};
    },
    .output_value = &value, .prepare_bindings = &bind, .seal_bindings = &seal, .output_lease = &lease,
    .query_extension = [](void *, compute::ProviderExtensionId id) noexcept -> const void * {
      if (id == graph::detail::kRegionBindingReleaseExtensionId) return &binding_release_table;
      if (id == bd::kNativeBatchRegionExtensionId) return &page_table;
      if (id == bd::kNativePendingRegionExtensionId) return &pending_table;
      if (id == graph::detail::kRegionProjectedReadExtensionId) return &projected_table;
      if (id == kNativeOutputExtensionId) return &output_table;
      return nullptr;
    }};
} // namespace
bool native_paged_operation(const compute::Operation &operation) noexcept {
  return operation.custom_as<batch::PagedAttention>() || operation.custom_as<batch::PagedMlaAttention>() ||
      operation.custom_as<compute::PagedAttention>() || operation.custom_as<compute::PagedMlaAttention>();
}
Status propose_native_paged(void *context, const RegionDiscoveryRequest &request,
    std::span<const std::uint8_t> needed, RegionProposalWriter &writer) noexcept {
  for (std::size_t ordinal = 0; ordinal < request.graph.node_count(); ++ordinal) {
    if (!needed[ordinal] || (!request.allowed_nodes.empty() && !request.allowed_nodes[ordinal])) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, request.graph.node_id(ordinal));
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph.operation(node));
    if (!native_paged_operation(operation)) continue;
    HYLE_ASSIGN_OR_RETURN(auto selected, recipe(operation));
    const auto admitted = qualify(runtime(context), selected, selected.layout);
    if (!admitted.ok()) { if (admitted.IsUnimplemented()) continue; return admitted; }
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added, writer.try_add({.nodes = nodes, .route = kNativePagedRoute,
        .replay_class = compute::ProviderReplayClass::kManaged,
        .implementation = compute::ProviderImplementationId{0x5041474544415050ULL}, .preference = 100}));
    if (!added) break;
  }
  return Status::Ok();
}
Status native_paged_resources(void *context, const RegionCompileRequest &request,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *proposal) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto selected, qualify(context, request));
  (void)selected;
  if (inputs.size() != request.inputs.size() || outputs.size() != request.outputs.size())
    return Status::InvalidArgument("Apple page resource boundary differs");
  const cd::ResourceBindingRequirement resource{
      .required_properties = mem::MemoryProperty::kHostAccessible | mem::MemoryProperty::kDeviceAccessible |
          mem::MemoryProperty::kHostDeviceCoherent, .minimum_alignment = 1};
  std::ranges::fill(inputs, RegionInputRequirement{resource, false});
  HYLE_ASSIGN_OR_RETURN(auto query, request.graph->node_input(request.nodes[0], 0));
  const auto found = std::ranges::find(request.inputs, query);
  if (found == request.inputs.end()) return Status::InvalidArgument("Apple page query boundary is missing");
  inputs[std::size_t(found-request.inputs.begin())].accepts_reference = true;
  std::ranges::fill(outputs, RegionOutputRequirement{resource,
      {RegionOutputKind::kDeferredGenerated, RegionOutputProjection::kNativeOnly}});
  *proposal = {.runtime = runtime_properties, .workspace_resource = resource}; return Status::Ok();
}
compute::ProviderPrepareOutcome compile_native_paged(void *context, const RegionCompileRequest &request,
    mem::AllocatorRef allocator, ProviderRegionPrepared *out) noexcept {
  using Outcome = compute::ProviderPrepareOutcome; *out = {};
  auto selected = qualify(context, request);
  if (!selected) return selected.error().IsUnimplemented() ? Outcome::unsupported(selected.error()) : Outcome::error(selected.error());
  if (request.input_resources.size() != request.inputs.size() || request.output_resources.size() != request.outputs.size())
    return Outcome::error(Status::InvalidArgument("Apple page physical resources differ"));
  auto *artifact = allocator.construct<Artifact>();
  if (!artifact) return Outcome::error(Status::ResourceExhausted("Apple page artifact metadata"));
  *out = {.vtable = &prepared_table, .state = artifact};
  const auto build = [&]() noexcept -> Status {
    artifact->recipe = std::move(*selected); artifact->node = request.nodes[0];
    artifact->preparation.runtime = runtime(context); artifact->workload = request.workload;
    artifact->budget = request.resource_budget;
    artifact->count = artifact->recipe.mla == compute::MlaCacheLayout::kNone ? 3 : 2;
    artifact->public_outputs = request.outputs.size();
    const auto consider = [&](mem::MemoryResourceRef resource) {
      if (!an::is_copy_resource(resource)) return;
      if (!artifact->device.valid()) artifact->device = {resource};
      if (!artifact->staging.valid()) artifact->staging = {resource};
    };
    consider(request.workspace_resource);
    for (const auto &domain : request.available_storage_domains) consider(domain.resource);
    if (!artifact->device.valid() || !artifact->staging.valid())
      return Status::Unimplemented("Apple page metadata needs actual shared Metal storage");
    const auto input_count = artifact->count + std::size_t(artifact->recipe.external_metadata);
    for (std::size_t port = 0; port < input_count; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto input, request.graph->node_input(artifact->node, port));
      if (port && input.kind() != ValueKind::kInput)
        return Status::InvalidArgument("Apple page capacity needs its exact external frame");
      const auto found = std::ranges::find(request.inputs, input);
      if (found == request.inputs.end()) return Status::InvalidArgument("Apple page boundary input is missing");
      const auto boundary = std::uint32_t(found-request.inputs.begin());
      HYLE_ASSIGN_OR_RETURN(artifact->descriptions[port], request.graph->value_description(input));
      if (port == artifact->count) {
        artifact->metadata = input; artifact->metadata_port = boundary;
        for (std::size_t p = 0; p < artifact->count; ++p)
          if (artifact->inputs[p] == input) return Status::InvalidArgument("Apple page metadata aliases a payload input");
      } else {
        if (port && artifact->inputs[0] == input) return Status::InvalidArgument("Apple page cache aliases its query input");
        artifact->inputs[port] = input; artifact->ports[port] = boundary;
      }
      if (!an::is_copy_resource(request.input_resources[boundary]))
        return Status::InvalidArgument("Apple page input does not belong to shared Metal storage");
    }
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(artifact->node));
    compute::TensorDesc expected;
    HYLE_RETURN_IF_ERROR(compute::infer_output_descriptions(operation,
        std::span(artifact->descriptions).first(input_count), std::span(&expected, 1)));
    HYLE_ASSIGN_OR_RETURN(auto output, request.graph->result(artifact->node));
    HYLE_ASSIGN_OR_RETURN(artifact->output, request.graph->value_description(output));
    if (artifact->output != expected || artifact->output.rank() != 3 ||
        artifact->output.dimensions()[0] < 0 || std::uint64_t(artifact->output.dimensions()[0]) > UINT32_MAX)
      return Status::InvalidArgument("Apple page concrete Q/output descriptions differ");
    artifact->preparation.query_rows = std::uint32_t(artifact->output.dimensions()[0]);
    if (!artifact->recipe.external_metadata) {
      HYLE_ASSIGN_OR_RETURN(artifact->native_recipe, page_recipe(*artifact, artifact->recipe.layout.origin_schema(), allocator));
      HYLE_ASSIGN_OR_RETURN(artifact->program, an::PagedBatchAccess::prepare_program(
          artifact->preparation.runtime, artifact->native_recipe, artifact->recipe.layout, artifact->workload));
    }
    out->properties = {.persistent_bytes = std::nullopt, .runtime = runtime_properties, .invocation_private_bytes = std::nullopt};
    return Status::Ok();
  };
  const auto status = build();
  return status.ok() ? Outcome::prepared() : status.IsUnimplemented() ? Outcome::unsupported(status) : Outcome::error(status);
}
} // namespace hyle::graph::apple::detail
