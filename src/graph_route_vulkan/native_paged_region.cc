#include "native_paged_region.h"
#include "native_paged_pending.h"
#include "native_region.h"
#include "vulkan_source.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <new>

#include "../compute_route_vulkan/vulkan_composition_access.h"
#include "../compute_route_vulkan/vulkan_file_stream_access.h"
#include "../compute_route_vulkan/vulkan_memory_internal.h"
#include "../compute_route_vulkan/vulkan_program_internal.h"
#include "../graph_batch/native_batch_regions.h"
#include "hyle/compute/detail/materialized_native_controller.h"
#include "hyle/compute/detail/vulkan_batch_access.h"

namespace hyle::graph::vulkan::detail {
namespace {
namespace cv = compute::vulkan;
namespace vn = cv::detail;
namespace cd = compute::detail;
namespace bd = graph::batch::detail;
constexpr std::uint64_t kPagedRoute = 2;
const unsigned tensor_type = 0, observed_endpoint = 0;
constexpr compute::ProviderRuntimeProperties runtime_properties{
    .mode = compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
    .replay_class = compute::ProviderReplayClass::kManaged};

struct Recipe final {
  compute::BatchLayoutVersion layout;
  batch::PagedAttentionOptions options;
  compute::BatchCapacity capacity{};
  bool explicit_metadata = false;
};
Result<Recipe> recipe(const compute::Operation &operation) noexcept {
  if (const auto d = operation.custom_as<batch::PagedAttention>())
    return Recipe{d->layout, d->options};
  if (const auto d = operation.custom_as<batch::PagedMlaAttention>())
    return Recipe{d->layout, d->options};
  if (const auto d = operation.custom_as<compute::PagedAttention>())
    return Recipe{{}, {.scale = d->scale, .causal = d->causal,
        .window_left = d->window_left}, d->capacity, true};
  return std::unexpected(Status::Unimplemented("not paged attention"));
}
Status qualify(const Recipe &value) noexcept {
  if ((!value.explicit_metadata && !value.layout.valid()) || !std::isfinite(value.options.scale))
    return Status::InvalidArgument("native Vulkan page layout/scale is invalid");
  const auto capacity = value.explicit_metadata ? value.capacity : value.layout.schema().capacity();
  if (!vn::PagedBatchAccess::cache_domain(capacity) || capacity.page_size != 16)
    return Status::Unimplemented("native Vulkan page dtype/encoding/geometry is unsupported");
  if (value.explicit_metadata) {
    if (capacity.storage != compute::BatchStorage::kPaged)
      return Status::InvalidArgument("native Vulkan page capacity is not paged");
    HYLE_RETURN_IF_ERROR(compute::validate_batch_capacity(capacity));
    return compute::validate_batch_attention_options(value.options.causal, value.options.window_left);
  }
  return compute::validate_batch_attention_layout(value.layout,
      value.options.causal, value.options.window_left);
}
Result<Recipe> qualify(const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.graph->dimension_count() || request.route != kPagedRoute ||
      request.nodes.size() != 1 || request.outputs.size() > 1 ||
      request.has_unknown_effects || !request.effect_accesses.empty())
    return std::unexpected(Status::Unimplemented("native Vulkan page needs one fixed pure typed node"));
  HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(request.nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto value, recipe(operation));
  HYLE_RETURN_IF_ERROR(qualify(value));
  return value;
}

// These are actual native owners and borrowed resource refs. They only adapt
// the original controller's static traits; they are not public Context/Queue
// shells and carry no second execution state.
struct Preparation final {
  std::shared_ptr<vn::Runtime> runtime;
  cv::InvocationAllocation allocation;
  std::uint32_t physical_query_rows = 0;
  bool native_queue_only = false;
};
struct Resource final {
  std::shared_ptr<vn::Runtime> runtime;
  mem::MemoryResourceRef resource;
  bool valid() const noexcept { return runtime && resource.valid(); }
  mem::MemoryResourceRef ref() const noexcept { return resource; }
};
struct PreparedProgram final {
  cv::Program native;
  cv::InvocationAllocation allocation;
  bool valid() const noexcept { return native.valid(); }
};
struct Traits final {
  using Context = Preparation; using Memory = Resource;
  using Program = PreparedProgram; using Invocation = cv::Invocation;
  using CopyInvocation = cv::CopyBatchInvocation; using View = cv::Tensor;
  using Submission = compute::OwnedResult<cv::Results>;
  using CopyAccess = vn::CopyStreamAccess;
  static auto invocation(const Program &p, mem::AllocatorRef a) {
    return p.native.create_invocation(a, p.allocation);
  }
  static Result<Program> prepare_program(const Context &c,
      const compute::NativePagedBatchProgram &p, compute::BatchLayoutVersion layout,
      const compute::WorkloadContext &workload) {
    HYLE_ASSIGN_OR_RETURN(auto native, c.native_queue_only ?
        vn::PagedBatchAccess::prepare_paged_graph(c.runtime, p, std::move(layout), c.physical_query_rows, workload) :
        vn::PagedBatchAccess::prepare_program(c.runtime, p, std::move(layout), workload));
    return Program{std::move(native), c.allocation};
  }
  static auto copy_program(const Context &c, std::span<const compute::CopyResourceSpec> specs,
      compute::CopyBatchOptions options) {
    return vn::CopyStreamAccess::prepare_program(c.runtime, specs, std::move(options));
  }
  static auto copy_invocation(const cv::CopyBatchProgram &p) { return cv::CopyBatchInvocation::create(p); }
  static auto binding(const Memory &m, compute::ReadLease lease,
      compute::TensorDesc description, compute::LeaseImportOptions options) {
    return vn::MemoryAccess::lease_tensor(m.runtime, m.resource, std::move(lease), description, options);
  }
  static auto submit(Invocation &i, compute::LeaseUseBatch uses) {
    return vn::PagedBatchAccess::submit(i, std::move(uses));
  }
  static auto wait_native(Invocation &i) { return vn::PagedBatchAccess::wait_native(i); }
  static auto reuse_gate(Invocation &i) { return vn::PagedBatchAccess::reuse_gate(i); }
  static auto prepare_gate(Invocation &i) { return vn::CompositionAccess::prepare_bindings(i); }
  static auto release_inputs(Invocation &i) { return vn::PagedBatchAccess::release_inputs(i); }
  static auto update_layout(Invocation &i, const compute::BatchLayoutVersion &layout) {
    return vn::PagedBatchAccess::update_layout(i, layout);
  }
};
using Controller = cd::MaterializedNativeState<Traits, cd::PagedMetadataPolicy>;
struct Artifact final {
  Recipe recipe;
  compute::NativePagedBatchProgram native_recipe;
  NodeId node;
  std::array<ValueId, 3> inputs{};
  std::array<std::uint32_t, 3> ports{};
  ValueId metadata_input;
  std::uint32_t metadata_port = 0, output_query_rows = 0;
  compute::TensorDesc query_description;
  std::size_t count = 0, public_outputs = 0;
  Resource device, staging;
  Preparation preparation;
  PreparedProgram program;
  compute::WorkloadContext workload;
  mem::MemoryResourceRef output_resource;
};
struct Slot final {
  struct WrittenBank final {
    compute::BatchLayoutVersion layout;
    compute::ConstTensorView capacity;
    cv::Tensor tensor;
    bd::NativeAppendRegionRef writer;
    bool committed = false;
  };
  Slot(mem::AllocatorRef a, Preparation p) : allocator(a), preparation(std::move(p)) {}
  ~Slot() { if (state) allocator.destroy(state); }
  mem::AllocatorRef allocator;
  Preparation preparation; // Outlives the controller's original cold borrow.
  Controller *state = nullptr;
  Controller &controller() noexcept { return *state; }
  const Controller &controller() const noexcept { return *state; }
  std::array<compute::PagedArenaRead, 2> frames;
  std::array<compute::PagedArenaPendingRead, 2> pending_frames;
  NativePagePendingSource *pending_source = nullptr; // Original resource payload owns it.
  std::array<WrittenBank, 2> written;
  cv::Tensor output;
  mem::Buffer backing;
  cd::ResultStorageOwner result;
  bool sealed = false, needs_seal = false;
};

const compute::NativePagedBatchProgram &page_recipe(const Artifact &a, const Slot &s) noexcept {
  return a.recipe.explicit_metadata ? s.controller().recipe : a.native_recipe;
}
Result<compute::TensorDesc> input_description(const Artifact &a, const Slot &s,
    std::size_t port) noexcept {
  if (!port) return a.query_description;
  return page_recipe(a, s).input_description(s.controller().version.layout(), port);
}

bool empty_program(const Artifact &a, const Slot &s) noexcept {
  const auto &layout = s.controller().version.layout();
  if (!s.controller().prepared || !layout.valid() ||
      !page_recipe(a, s).schema().owns(layout) || layout.facts().query_rows ||
      layout.facts().page_refs || !s.controller().native.valid()) return false;
  const auto &program = *vn::Access::invocation_state(s.controller().native)->program;
  const auto description = input_description(a, s, 0);
  return description && program.inputs.size() == 1 && program.inputs[0] == 0 &&
      program.input_descriptions.size() == 1 &&
      program.input_descriptions[0] == *description;
}

template<class Frame> Status validate_frame(void *prepared, void *invocation, ValueId input,
    const Frame &value, bool pending = false) noexcept {
  const auto &a = *static_cast<Artifact *>(prepared);
  const auto &s = *static_cast<Slot *>(invocation);
  const auto &layout = s.controller().version.layout();
  if (!s.controller().prepared || (pending && s.controller().bound) || !value.valid() || value.layout() != layout ||
      !page_recipe(a, s).schema().owns(value.layout()))
    return Status::InvalidArgument("native Vulkan page frame needs its prepared owned layout");
  if (pending && std::ranges::any_of(s.written, [](const auto &bank) { return bank.committed; }))
    return Status::Unimplemented("Vulkan same-Graph bank cannot borrow an external F2 episode");
  bool selected = false;
  for (std::size_t port = 1; port < a.count; ++port) if (a.inputs[port] == input) {
    selected = true;
    if ((pending && s.frames[port-1].valid()) || s.pending_frames[port-1].valid() ||
        s.written[port-1].committed)
      return Status::FailedPrecondition("native page input needs reset before rebind");
    const auto capacity = page_recipe(a, s).schema().capacity();
    HYLE_RETURN_IF_ERROR(cd::PagedArenaAccess::validate_storage(value,
        port == 1 ? capacity.key_storage : capacity.value_storage));
    HYLE_ASSIGN_OR_RETURN(auto tensor, cd::PagedArenaAccess::tensor(value));
    HYLE_ASSIGN_OR_RETURN(auto description, input_description(a, s, port));
    if (tensor.description() != description)
      return Status::InvalidArgument("native Vulkan page frame physical description differs");
    HYLE_RETURN_IF_ERROR(vn::MemoryAccess::validate_resource(s.preparation.runtime,
        tensor.storage().resource()));
    if (tensor.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
      return Status::InvalidArgument("native page capacity requires actual device storage");
  }
  return selected ? Status::Ok() : Status::InvalidArgument("input is not this native page reader");
}
Status frame(void *prepared, void *invocation, ValueId input,
    const compute::PagedArenaRead &value) noexcept {
  return validate_frame(prepared, invocation, input, value);
}
Status refresh_output(Artifact &a, Slot &s) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto output, vn::CompositionAccess::output(s.controller().native, 0));
  mem::Buffer backing;
  if (a.public_outputs) {
    HYLE_ASSIGN_OR_RETURN(backing, vn::MemoryAccess::export_storage(
        s.preparation.runtime, a.output_resource, output, s.allocator));
  }
  // Keep the projected object addresses stable. A failed export cannot pair a
  // new Tensor with the previous backing or mutate an older readonly view.
  s.output = std::move(output);
  s.backing = std::move(backing);
  return Status::Ok();
}
Result<compute::NativePagedBatchProgram> make_recipe(const Artifact &a,
    compute::BatchSchema schema, mem::AllocatorRef allocator) noexcept {
  const auto refs = std::size_t(schema.capacity().max_page_refs);
  if (refs > SIZE_MAX / sizeof(std::int32_t))
    return std::unexpected(Status::OutOfRange("native page table byte extent overflow"));
  return compute::NativePagedBatchProgram::prepare(std::move(schema),
      {.max_host_metadata_bytes = refs * sizeof(std::int32_t), .scale = a.recipe.options.scale,
       .causal = a.recipe.options.causal, .window_left = a.recipe.options.window_left,
       .allocator = allocator, .resource_budget = a.preparation.allocation.resource_budget});
}
Status qualify_ordered_native(const std::shared_ptr<vn::Runtime> &runtime,
    const cv::Invocation &native) noexcept {
  if (!native.valid())
    return Status::FailedPrecondition("native page has no prepared queue program");
  const auto &run = *vn::Access::invocation_state(native)->run;
  if (run.runtime != runtime)
    return Status::InvalidArgument("native page program Runtime differs");
  if (run.async || run.needs_drain ||
      std::ranges::any_of(run.stages, [](const auto &stage) { return bool(stage.iree); }))
    return Status::Unimplemented("native page capacity requires its original compute queue program");
  return Status::Ok();
}
Status prepare(void *prepared, void *invocation, compute::BatchLayoutVersion layout) noexcept {
  auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
  if (!layout.valid() || (a.recipe.explicit_metadata ?
      layout.schema().capacity() != a.recipe.capacity || layout.schema().mla_layout() != compute::MlaCacheLayout::kNone ||
          layout.facts().query_rows > a.output_query_rows :
      !a.native_recipe.schema().owns(layout) || layout.facts().query_rows != a.output_query_rows))
    return Status::InvalidArgument("native page schema/active output shape changed");
  HYLE_RETURN_IF_ERROR(compute::validate_batch_attention_layout(layout,
      a.recipe.options.causal, a.recipe.options.window_left));
  HYLE_RETURN_IF_ERROR(s.controller().open());
  s.result.reset();
  if (s.controller().native.valid())
    HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_bindings(s.controller().native));
  s.controller().prepared = false; s.controller().bound = false;
  s.needs_seal = false;
  if (a.recipe.explicit_metadata && !s.controller().recipe.schema().owns(layout)) {
    if (s.controller().uses.active_batches())
      return Status::Unavailable("native page prior permission use is retained");
    HYLE_ASSIGN_OR_RETURN(auto next_recipe, make_recipe(a, layout.schema(), s.allocator));
    auto *candidate = s.allocator.construct<Controller>(s.allocator);
    if (!candidate) return Status::ResourceExhausted("native page replacement controller");
    auto destroy = [allocator = s.allocator](Controller *p) noexcept { allocator.destroy(p); };
    std::unique_ptr<Controller, decltype(destroy)> next(candidate, destroy);
    const compute::NativePagedInvocationOptions options{
        .max_metadata_backing_bytes = SIZE_MAX, .allocator = s.allocator,
        .resource_budget = a.preparation.allocation.resource_budget};
    HYLE_RETURN_IF_ERROR(next->initialize(s.preparation, {}, std::move(next_recipe),
        a.device, a.staging, options, a.workload, true));
    HYLE_RETURN_IF_ERROR(next->prepare(layout));
    HYLE_RETURN_IF_ERROR(qualify_ordered_native(s.preparation.runtime, next->native));
    bool refresh = false;
    HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_outputs(next->native, &refresh));
    HYLE_ASSIGN_OR_RETURN(auto output, vn::CompositionAccess::output(next->native, 0));
    mem::Buffer backing;
    if (a.public_outputs) {
      HYLE_ASSIGN_OR_RETURN(backing, vn::MemoryAccess::export_storage(
          s.preparation.runtime, a.output_resource, output, s.allocator));
    }
    // No candidate owns accepted device work. Commit only after every cold
    // allocation/import succeeds; failed replacement leaves the old schema
    // intact but revoked, so a later prepare cannot skip partial initialization.
    auto *old = s.state; s.state = next.release();
    s.output = std::move(output); s.backing = std::move(backing);
    s.sealed = false; s.needs_seal = false;
    s.frames = {}; s.pending_frames = {}; s.pending_source = nullptr; s.written = {};
    s.allocator.destroy(old);
    return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(s.controller().prepare(std::move(layout)));
  if (a.recipe.explicit_metadata) {
    const auto ordered = qualify_ordered_native(s.preparation.runtime, s.controller().native);
    if (!ordered.ok()) {
      s.controller().prepared = false; s.controller().bound = false;
      return ordered;
    }
  }
  s.frames = {}; s.pending_frames = {}; s.pending_source = nullptr;
  s.written = {};
  if (!s.controller().report.reused_native_plan) { s.sealed = false; s.output = {}; s.backing = {}; }
  if (!s.output.valid()) {
    bool refresh = false;
    HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_outputs(s.controller().native, &refresh));
    HYLE_RETURN_IF_ERROR(refresh_output(a, s));
    // The initial projection belongs to the unpublished epoch. Graph's final
    // seal follows the complete input bind and includes downstream references.
  }
  return Status::Ok();
}
Status reset(void *, void *invocation) noexcept {
  auto &s = *static_cast<Slot *>(invocation);
  s.result.reset();
  if (s.controller().native.valid())
    HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_bindings(s.controller().native));
  s.controller().prepared = false; s.controller().bound = false; s.needs_seal = false;
  if (s.controller().native.valid())
    HYLE_RETURN_IF_ERROR(vn::PagedBatchAccess::release_inputs(s.controller().native));
  s.frames = {}; s.pending_frames = {}; s.pending_source = nullptr;
  s.written = {};
  s.controller().keys = {}; s.controller().values = {}; s.controller().reads = {};
  return Status::Ok();
}
cd::CompletionEndpoint endpoint(void *, void *) noexcept;
Status validate_written(void *prepared, void *invocation, ValueId input,
    compute::BatchLayoutVersion layout, compute::ConstTensorView capacity,
    const bd::NativeAppendRegionRef &writer) noexcept {
  const auto &a = *static_cast<const Artifact *>(prepared);
  auto &s = *static_cast<Slot *>(invocation);
  if (!s.controller().prepared || s.controller().bound || layout != s.controller().version.layout() ||
      !page_recipe(a, s).schema().owns(layout) || !capacity.valid() || !writer.table ||
      !writer.table->inspect || !writer.table->qualify_reader || !writer.table->before_read ||
      !writer.prepared || !writer.invocation ||
      std::ranges::any_of(s.pending_frames, [](const auto &f) { return f.valid(); }))
    return Status::FailedPrecondition("Vulkan written bank needs its exact unbound native layout");
  std::size_t port = 1;
  for (; port < a.count; ++port) if (a.inputs[port] == input) break;
  if (port == a.count || s.frames[port - 1].valid() || s.written[port - 1].committed)
    return Status::InvalidArgument("Vulkan written bank is not an available exact native KV port");
  HYLE_ASSIGN_OR_RETURN(auto facts, writer.table->inspect(writer.prepared, writer.invocation));
  const auto role = port == 1 ? compute::BatchCacheRole::kKey : compute::BatchCacheRole::kValue;
  if (facts.node != writer.node || facts.descriptor.role != role ||
      !facts.descriptor.schema.owns(layout))
    return Status::InvalidArgument("Vulkan written bank role/schema differs from its selected writer");
  HYLE_ASSIGN_OR_RETURN(auto description, input_description(a, s, port));
  if (capacity.description() != description)
    return Status::InvalidArgument("Vulkan written bank physical description differs");
  HYLE_RETURN_IF_ERROR(vn::MemoryAccess::validate_resource(s.preparation.runtime, capacity.storage().resource()));
  if (capacity.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
    return Status::InvalidArgument("Vulkan written bank requires device storage");
  HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_bindings(s.controller().native));
  HYLE_RETURN_IF_ERROR(writer.table->qualify_reader(writer.prepared, writer.invocation,
      endpoint(prepared, invocation)));
  // This owns only the native backing. Core has not granted page permission yet.
  HYLE_ASSIGN_OR_RETURN(auto tensor, vn::MemoryAccess::tensor_view(s.preparation.runtime,
      capacity, s.allocator, s.controller().import.resource_budget));
  s.written[port - 1] = {layout, capacity, std::move(tensor), writer, false};
  return Status::Ok();
}
void commit_written(void *prepared, void *invocation, ValueId input,
    compute::BatchLayoutVersion layout, compute::ConstTensorView capacity,
    const bd::NativeAppendRegionRef &writer) noexcept {
  const auto &a = *static_cast<const Artifact *>(prepared);
  auto &s = *static_cast<Slot *>(invocation);
  for (std::size_t port = 1; port < a.count; ++port) if (a.inputs[port] == input) {
    auto &bank = s.written[port - 1];
    HYLE_FATAL_ASSERT(bank.layout == layout && bank.capacity.storage().resource() == capacity.storage().resource() &&
        bank.capacity.storage().native_handle() == capacity.storage().native_handle() &&
        bank.capacity.storage().offset() == capacity.storage().offset() &&
        bank.capacity.storage().size() == capacity.storage().size() &&
        bank.writer.prepared == writer.prepared && bank.writer.invocation == writer.invocation);
    bank.committed = true;
  }
}
const bd::NativeBatchRegionVTable page_table{
    [](void *prepared, void *invocation) noexcept -> Result<bd::NativeBatchRegionFacts> {
      const auto &a = *static_cast<Artifact *>(prepared); const auto &s = *static_cast<Slot *>(invocation);
      const auto schema = a.recipe.explicit_metadata ? compute::BatchSchema{} : a.native_recipe.schema();
      return bd::NativeBatchRegionFacts{
          .node = a.node, .paged_program = a.recipe.explicit_metadata ? nullptr : &a.native_recipe,
          .capacity = a.recipe.explicit_metadata ? a.recipe.capacity : schema.capacity(),
          .mla = a.recipe.explicit_metadata ? compute::MlaCacheLayout::kNone : schema.mla_layout(), .causal = a.recipe.options.causal,
          .window_left = a.recipe.options.window_left,
          .inputs = std::span(a.inputs).first(a.count), .input_ports = std::span(a.ports).first(a.count),
          .metadata_input = a.metadata_input, .output_query_rows = a.output_query_rows,
          .layout = s.controller().version.layout()};
    }, &prepare, &frame,
    [](void *prepared, void *invocation, ValueId input, compute::PagedArenaRead value) noexcept {
      const auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
      for (std::size_t port = 1; port < a.count; ++port)
        if (a.inputs[port] == input) s.frames[port-1] = value;
    }, &reset,
    [](void *, void *invocation) noexcept { return static_cast<Slot *>(invocation)->controller().report; },
    &validate_written, &commit_written};
const bd::NativePendingRegionVTable pending_table{
    [](void *prepared, void *) noexcept -> Result<bd::NativePendingRegionFacts> {
      const auto &a = *static_cast<Artifact *>(prepared);
      return bd::NativePendingRegionFacts{std::span(a.inputs).subspan(1, a.count-1),
          std::span(a.ports).subspan(1, a.count-1),
          {&native_pending_source_table, a.preparation.runtime.get(),
              [](void *opaque, std::size_t capacity, mem::AllocatorRef allocator,
                  compute::ResourceBudget budget) noexcept -> Result<void *> {
                return create_native_pending_source(static_cast<Artifact *>(opaque)->preparation.runtime,
                    capacity, allocator, std::move(budget));
              }}};
    },
    [](void *prepared, void *invocation, ValueId input,
        const compute::PagedArenaPendingRead &value,
        const bd::NativePendingSourceOwner &source) noexcept -> Status {
      auto &s = *static_cast<Slot *>(invocation);
      HYLE_ASSIGN_OR_RETURN(auto actual, native_pending_source(source, s.preparation.runtime));
      (void)actual;
      return validate_frame(prepared, invocation, input, value, true);
    },
    [](void *prepared, void *invocation, ValueId input, compute::PagedArenaPendingRead value,
        const bd::NativePendingSourceOwner &source) noexcept {
      const auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
      auto actual = native_pending_source(source, s.preparation.runtime);
      HYLE_FATAL_ASSERT(actual.has_value());
      s.pending_source = *actual;
      for (std::size_t port = 1; port < a.count; ++port)
        if (a.inputs[port] == input) s.pending_frames[port-1] = value;
    },
    [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      slot.pending_frames = {}; slot.pending_source = nullptr;
    }};

Status create_slot(void *prepared, mem::AllocatorRef allocator, mem::BufferView, void **out) noexcept {
  *out = nullptr; auto &a = *static_cast<Artifact *>(prepared);
  auto *s = allocator.construct<Slot>(allocator, a.preparation);
  if (!s) return Status::ResourceExhausted("native Vulkan page slot metadata");
  *out = s;
  s->state = allocator.construct<Controller>(allocator);
  if (!s->state) return Status::ResourceExhausted("native Vulkan page controller metadata");
  const compute::NativePagedInvocationOptions options{
      .max_metadata_backing_bytes = SIZE_MAX,
      .allocator = allocator, .resource_budget = a.preparation.allocation.resource_budget};
  if (a.recipe.explicit_metadata) return Status::Ok();
  HYLE_RETURN_IF_ERROR(s->controller().initialize(s->preparation, a.program, a.native_recipe,
      a.device, a.staging, options, a.workload));
  HYLE_ASSIGN_OR_RETURN(s->controller().version, a.native_recipe.version(a.recipe.layout));
  return Status::Ok();
}
Status bind(void *prepared, void *invocation, std::span<const RegionValue> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
  auto &c = s.controller();
  if (!c.prepared || outputs.size() != a.public_outputs)
    return Status::FailedPrecondition("native page layout must be prepared before replay bindings");
  s.result.reset();
  bool refresh = false;
  HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_outputs(c.native, &refresh));
  c.bound = false; s.needs_seal = false;
  if (refresh || !s.output.valid()) HYLE_RETURN_IF_ERROR(refresh_output(a, s));
  std::array<cv::Tensor, 4> native;
  for (std::size_t port = 0; port < a.count; ++port) {
    HYLE_ASSIGN_OR_RETURN(auto description, input_description(a, s, port));
    if (a.ports[port] >= inputs.size()) return Status::InvalidArgument("native page boundary port differs");
    const auto &input = inputs[a.ports[port]];
    if (input.description != description) return Status::InvalidArgument("native page physical input description differs");
    if (!port) {
      if (input.lease) {
        HYLE_ASSIGN_OR_RETURN(native[0], vn::MemoryAccess::lease_tensor(s.preparation.runtime,
            input.tensor.storage().resource(), *input.lease, description, c.import));
        c.reads[0] = *input.lease;
      } else {
        HYLE_ASSIGN_OR_RETURN(native[0], vn::MemoryAccess::tensor_view(
            s.preparation.runtime, input.tensor, s.allocator, c.import.resource_budget));
        c.reads[0] = {};
      }
    } else {
      const auto &written = s.written[port - 1];
      if (written.committed) {
        const auto actual = written.capacity.storage(), expected = input.tensor.storage();
        if (written.layout != c.version.layout() || !written.tensor.valid() ||
            actual.resource() != expected.resource() || actual.native_handle() != expected.native_handle() ||
            actual.offset() != expected.offset() || actual.size() != expected.size())
          return Status::InvalidArgument("Vulkan native input differs from its proved same-Graph bank");
        native[port] = written.tensor;
        continue;
      }
      const auto import = [&](const auto &permission) noexcept -> Status {
        if (!permission.valid() || permission.layout() != c.version.layout())
          return Status::FailedPrecondition("native page input has no exact typed frame");
        HYLE_ASSIGN_OR_RETURN(auto actual, cd::PagedArenaAccess::tensor(permission));
        const auto x = actual.storage(), y = input.tensor.storage();
        if (x.resource() != y.resource() || x.native_handle() != y.native_handle() ||
            x.offset() != y.offset() || x.size() != y.size())
          return Status::InvalidArgument("native page input differs from its exact bound frame");
        if (empty_program(a, s)) return Status::Ok();
        HYLE_ASSIGN_OR_RETURN(native[port], vn::MemoryAccess::paged_tensor(
            s.preparation.runtime, actual.storage().resource(), permission, c.import));
        return Status::Ok();
      };
      HYLE_RETURN_IF_ERROR(s.pending_frames[port-1].valid() ?
          import(s.pending_frames[port-1]) : import(s.frames[port-1]));
    }
  }
  if (a.recipe.explicit_metadata) {
    if (a.metadata_port >= inputs.size())
      return Status::InvalidArgument("native page metadata boundary port differs");
    const auto &metadata = inputs[a.metadata_port];
    HYLE_ASSIGN_OR_RETURN(auto description, c.recipe.schema().metadata_description());
    if (metadata.description != description || !metadata.lease)
      return Status::FailedPrecondition("native page needs its original owned full metadata lease");
    HYLE_ASSIGN_OR_RETURN(auto actual, metadata.lease->tensor(description));
    const auto x = actual.storage(), y = metadata.tensor.storage();
    if (x.resource() != y.resource() || x.native_handle() != y.native_handle() ||
        x.offset() != y.offset() || x.size() != y.size())
      return Status::InvalidArgument("native page metadata carrier differs from its owned lease");
    const auto offsets = c.recipe.schema().metadata_offsets();
    HYLE_ASSIGN_OR_RETURN(c.reads[3], metadata.lease->subview(
        offsets.page_indices, c.recipe.costs().index_bytes));
  }
  if (empty_program(a, s)) {
    HYLE_RETURN_IF_ERROR(c.native.bind(std::span(native).first(1)));
  } else {
    HYLE_ASSIGN_OR_RETURN(auto metadata, input_description(a, s, a.count));
    const auto resource = a.recipe.explicit_metadata ?
        inputs[a.metadata_port].tensor.storage().resource() : a.device.resource;
    HYLE_ASSIGN_OR_RETURN(native[a.count], vn::MemoryAccess::lease_tensor(
        s.preparation.runtime, resource, c.reads[3], metadata, c.import));
    HYLE_RETURN_IF_ERROR(c.native.bind(std::span(native).first(a.count+1)));
  }
  c.keys = s.frames[0]; c.values = a.count == 3 ? s.frames[1] : compute::PagedArenaRead{};
  c.native_bank = c.prepared_bank; c.bound = true; s.needs_seal = true;
  return Status::Ok();
}
Status seal(void *, void *invocation) noexcept {
  auto &s = *static_cast<Slot *>(invocation);
  // Graph guards public results; its completed internal owner must not block
  // the original slot's exact reuse gate for unchanged replay bindings.
  s.result.reset();
  if (s.sealed && !s.needs_seal) return vn::PagedBatchAccess::reuse_gate(s.controller().native);
  HYLE_RETURN_IF_ERROR(s.sealed ? vn::PagedBatchAccess::reseal_composition(s.controller().native)
                               : vn::CompositionAccess::seal(s.controller().native));
  s.sealed = true; s.needs_seal = false; return Status::Ok();
}
Result<RegionValue> value(void *prepared, void *invocation, std::size_t port) noexcept {
  const auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
  if (port >= a.public_outputs || !s.output.valid() || !s.backing.valid())
    return std::unexpected(Status::FailedPrecondition("native page output is not cold prepared"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, compute::ConstTensorView::create(s.output.description(), s.backing.view()));
  return RegionValue{s.output.description(), tensor, &tensor_type, &s.output};
}
Result<compute::ReadLease> lease(void *prepared, void *invocation, std::size_t port) noexcept {
  const auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
  if (!s.result || port >= a.public_outputs)
    return std::unexpected(Status::FailedPrecondition("native page has no successful output owner"));
  const auto &run = vn::Access::invocation_state(s.controller().native)->run;
  {
    std::lock_guard lock(s.preparation.runtime->mutex);
    if (run->pending || run->retiring || !run->terminal.ok() ||
        !run->observation_error.ok() || !run->submission_failure.ok())
      return std::unexpected(Status::FailedPrecondition("native page output has not retired successfully"));
  }
  HYLE_ASSIGN_OR_RETURN(auto storage, vn::MemoryAccess::export_storage(
      s.preparation.runtime, a.output_resource, s.output, s.allocator));
  return compute::ReadLease::adopt(std::move(storage), {.allocator = s.allocator});
}
cd::CompletionEndpoint endpoint(void *prepared, void *invocation) noexcept {
  const auto &a = *static_cast<const Artifact *>(prepared);
  const auto &s = *static_cast<Slot *>(invocation);
  // Graph records this endpoint when the Invocation is created, before the
  // first explicit NodeLayout exists. Every later cold native plan must pass
  // qualify_ordered_native before it becomes a prepared capacity episode.
  if (a.recipe.explicit_metadata) return region_ordered_endpoint(s.preparation.runtime);
  if (!s.controller().native.valid()) return {&observed_endpoint, &s};
  const auto &run = *vn::Access::invocation_state(s.controller().native)->run;
  if (run.async || run.needs_drain || std::ranges::any_of(run.stages, [](const auto &stage) { return bool(stage.iree); }))
    return {&observed_endpoint, &run};
  return region_ordered_endpoint(s.preparation.runtime);
}
bool imports(void *prepared, void *invocation, cd::CompletionEndpoint source) noexcept {
  const auto &s = *static_cast<Slot *>(invocation);
  const auto own = endpoint(prepared, invocation);
  return own.authority != &observed_endpoint && imports_region_ordered_endpoint(s.preparation.runtime, source);
}
Result<compute::Completion> submit(void *, void *invocation,
    std::span<const cd::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView>, std::span<const compute::TensorView>) noexcept {
  auto &s = *static_cast<Slot *>(invocation);
  for (const auto dependency : dependencies) HYLE_RETURN_IF_ERROR(vn::qualify_copy_dependency(s.preparation.runtime, dependency));
  s.result.reset();
  const bool pending = std::ranges::any_of(s.pending_frames, [](const auto &frame) { return frame.valid(); });
  for (const auto &bank : s.written) if (bank.committed) {
    if (pending) return std::unexpected(Status::Unimplemented("Vulkan same-Graph bank cannot borrow an external F2 episode"));
    // The writer was accepted on this Runtime's original compute VkQueue.
    // Reject known errors/cancellation here, without waiting for its fence.
    // Queue order and its recorded barrier precede this reader; the original
    // Graph child completions still decide whole-Graph success and retirement.
    HYLE_RETURN_IF_ERROR(bank.writer.table->before_read(bank.writer.prepared, bank.writer.invocation));
  }
  HYLE_RETURN_IF_ERROR(vn::PagedBatchAccess::reuse_gate(s.controller().native));
  auto accepted = [&]() -> Result<Traits::Submission> {
    if (!pending) return s.controller().replay_ready();
    if (!s.pending_source)
      return std::unexpected(Status::FailedPrecondition("native page has no reserved pending source"));
    HYLE_RETURN_IF_ERROR(qualify_native_pending_source(*s.pending_source, s.pending_source->source.update()));
    return s.controller().submit_pending<cv::CopyBatchUpdate, vn::CopyStreamAccess>(
        s.pending_frames[0], s.pending_frames[1], s.pending_source->source.update());
  }();
  if (!accepted) return std::unexpected(accepted.error());
  auto transfer = cd::ResultAccess::take_storage(std::move(*accepted));
  s.result = std::move(transfer.owner); return std::move(transfer.completion);
}
const graph::detail::RegionProjectedReadVTable projected_table{
    [](const ProviderRegionPrepared &prepared, void *invocation,
        std::uint32_t port, cd::OperationPreparedAccess *out) noexcept -> Status {
      *out = {};
      if (!prepared.state || !invocation)
        return Status::InvalidArgument("native page empty projection has no actual slot");
      const auto &a = *static_cast<const Artifact *>(prepared.state);
      const auto &s = *static_cast<const Slot *>(invocation);
      bool selected = false;
      for (std::size_t input = 1; input < a.count; ++input) selected |= a.ports[input] == port;
      if (!selected || !empty_program(a, s))
        return Status::Unimplemented("native page port is not an actual empty Q-only program");
      *out = {.projected_input = port, .allows_empty_footprint = true};
      return Status::Ok();
    }};
const graph::detail::RegionBindingReleaseVTable binding_release_table{
    .validate = [](void *, void *invocation) noexcept -> Status {
      const auto &slot = *static_cast<const Slot *>(invocation);
      if (!slot.controller().native.valid())
        return slot.result ? Status::FailedPrecondition("native result has no invocation") : Status::Ok();
      return vn::PagedBatchAccess::can_detach_inputs(slot.controller().native, slot.result);
    },
    .release = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      slot.result.reset();
      if (slot.controller().native.valid()) vn::PagedBatchAccess::detach_inputs(slot.controller().native);
      slot.written = {};
      // Keep exact caller configuration and the current output projection.
      // The next cold bind rearms its complete owned block before final seal.
      slot.controller().bound = false;
      slot.needs_seal = false;
    }};

const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Artifact *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return compute::AliasSupport{}; },
    .create_invocation = &create_slot,
    .destroy_invocation = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Slot *>(p)); },
    .delegated_submit = &submit, .endpoint = &endpoint, .can_import_endpoint = &imports,
    .output_contract = [](void *, std::size_t) noexcept { return RegionOutputContract{
        .kind = RegionOutputKind::kBoundGenerated, .projection = RegionOutputProjection::kTensorView}; },
    .output_value = &value, .prepare_bindings = &bind, .seal_bindings = &seal, .output_lease = &lease,
    .query_extension = [](void *, compute::ProviderExtensionId id) noexcept -> const void * {
      if (id == graph::detail::kRegionBindingReleaseExtensionId) return &binding_release_table;
      if (id == bd::kNativeBatchRegionExtensionId) return &page_table;
      if (id == bd::kNativePendingRegionExtensionId) return &pending_table;
      if (id == graph::detail::kRegionProjectedReadExtensionId) return &projected_table;
      return nullptr;
    }};
} // namespace

bool native_paged_operation(const compute::Operation &operation) noexcept {
  return operation.custom_as<batch::PagedAttention>() || operation.custom_as<batch::PagedMlaAttention>() ||
      operation.custom_as<compute::PagedAttention>();
}
Status propose_native_paged(void *, const RegionDiscoveryRequest &request,
    std::span<const std::uint8_t> needed, RegionProposalWriter &writer) noexcept {
  for (std::size_t i = 0; i < request.graph.node_count(); ++i) {
    if (!needed[i] || (!request.allowed_nodes.empty() && !request.allowed_nodes[i])) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, request.graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto kind, request.graph.operation_kind(node));
    if (kind != compute::OperationKind::kCustom) continue;
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph.operation(node));
    if (!native_paged_operation(operation)) continue;
    HYLE_ASSIGN_OR_RETURN(auto value, recipe(operation));
    const auto admitted = qualify(value);
    if (!admitted.ok()) { if (admitted.IsUnimplemented()) continue; return admitted; }
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added, writer.try_add({.nodes = nodes, .route = kPagedRoute,
        .replay_class = compute::ProviderReplayClass::kManaged,
        .implementation = compute::ProviderImplementationId{0x5041474544564bULL}, .preference = 100}));
    if (!added) break;
  }
  return Status::Ok();
}
Status native_paged_resources(void *context, const RegionCompileRequest &request,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *proposal) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto value, qualify(request));
  (void)value;
  if (inputs.size() != request.inputs.size() || outputs.size() != request.outputs.size())
    return Status::InvalidArgument("native page resource boundary differs");
  const auto &runtime = static_cast<SourceContext *>(context)->runtime;
  const cd::ResourceBindingRequirement requirement{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .forbidden_properties = mem::MemoryProperty::kHostAccessible, .minimum_alignment = 1,
      .exact_domain = {runtime->device.get(), static_cast<std::uintptr_t>(cv::MemoryKind::kDevice)},
      .requires_exact_domain = true};
  std::ranges::fill(inputs, RegionInputRequirement{.resource = requirement});
  std::ranges::fill(outputs, RegionOutputRequirement{.resource = requirement,
      .value = {.kind = RegionOutputKind::kBoundGenerated, .projection = RegionOutputProjection::kTensorView}});
  *proposal = {.runtime = runtime_properties, .workspace_resource = requirement};
  return Status::Ok();
}
compute::ProviderPrepareOutcome compile_native_paged(void *context,
    const RegionCompileRequest &request, mem::AllocatorRef allocator,
    ProviderRegionPrepared *out) noexcept {
  using Outcome = compute::ProviderPrepareOutcome;
  *out = {};
  auto selected = qualify(request);
  if (!selected) return selected.error().IsUnimplemented() ? Outcome::unsupported(selected.error()) : Outcome::error(selected.error());
  if (request.input_resources.size() != request.inputs.size() ||
      request.output_resources.size() != request.outputs.size())
    return Outcome::error(Status::InvalidArgument("native page physical resource arity differs"));
  auto *a = allocator.construct<Artifact>();
  if (!a) return Outcome::error(Status::ResourceExhausted("native page artifact metadata"));
  *out = {.vtable = &prepared_table, .state = a};
  const auto build = [&]() noexcept -> Status {
    a->recipe = std::move(*selected); a->node = request.nodes[0]; a->workload = request.workload;
    const auto schema = a->recipe.layout.schema();
    const auto capacity = a->recipe.explicit_metadata ? a->recipe.capacity : schema.capacity();
    HYLE_ASSIGN_OR_RETURN(auto query, request.graph->node_input(a->node, 0));
    HYLE_ASSIGN_OR_RETURN(a->query_description, request.graph->value_description(query));
    if (a->query_description.rank() != 3 || a->query_description.dimensions()[0] < 0 ||
        std::uint64_t(a->query_description.dimensions()[0]) > capacity.max_query_rows)
      return Status::InvalidArgument("native page concrete query extent is invalid");
    a->output_query_rows = std::uint32_t(a->query_description.dimensions()[0]);
    // The native flash path keeps its original full-F32 intermediate contract;
    // allowing reduced partial storage does not require it to use that storage.
    a->count = a->recipe.explicit_metadata || schema.mla_layout() == compute::MlaCacheLayout::kNone ? 3 : 2;
    a->public_outputs = request.outputs.size();
    const auto &runtime = static_cast<SourceContext *>(context)->runtime;
    a->preparation.runtime = runtime; a->preparation.allocation.resource_budget = request.resource_budget;
    a->preparation.physical_query_rows = a->output_query_rows;
    a->preparation.native_queue_only = a->recipe.explicit_metadata;
    const auto consider = [&](mem::MemoryResourceRef resource) {
      if (!vn::MemoryAccess::validate_resource(runtime, resource).ok()) return;
      if (resource.properties().has(mem::MemoryProperty::kHostAccessible)) {
        if (!a->staging.valid()) a->staging = {runtime, resource};
      } else if (!a->device.valid()) a->device = {runtime, resource};
    };
    consider(request.workspace_resource);
    for (const auto &domain : request.available_storage_domains) consider(domain.resource);
    if (!a->device.valid() || !a->staging.valid())
      return Status::Unimplemented("native page metadata requires actual Vulkan device and staging domains");
    a->preparation.allocation.device = a->device.resource;
    a->preparation.allocation.staging = a->staging.resource;
    a->output_resource = a->public_outputs ? request.output_resources[0] : a->device.resource;
    for (std::size_t port = 0; port < a->count; ++port) {
      HYLE_ASSIGN_OR_RETURN(a->inputs[port], request.graph->node_input(a->node, port));
      if (port && a->inputs[port].kind() != ValueKind::kInput)
        return Status::InvalidArgument("native page capacity must have an exact external frame");
      const auto found = std::ranges::find(request.inputs, a->inputs[port]);
      if (found == request.inputs.end()) return Status::InvalidArgument("native page physical input is missing");
      a->ports[port] = static_cast<std::uint32_t>(found-request.inputs.begin());
      HYLE_RETURN_IF_ERROR(vn::MemoryAccess::validate_resource(runtime,
          request.input_resources[a->ports[port]]));
    }
    if (a->recipe.explicit_metadata) {
      HYLE_ASSIGN_OR_RETURN(a->metadata_input, request.graph->node_input(a->node, 3));
      if (a->metadata_input.kind() != ValueKind::kInput)
        return Status::InvalidArgument("native page metadata needs an exact external input");
      const auto found = std::ranges::find(request.inputs, a->metadata_input);
      if (found == request.inputs.end())
        return Status::InvalidArgument("native page physical metadata input is missing");
      a->metadata_port = static_cast<std::uint32_t>(found - request.inputs.begin());
      HYLE_ASSIGN_OR_RETURN(auto description, request.graph->value_description(a->metadata_input));
      HYLE_ASSIGN_OR_RETURN(auto expected, compute::batch_metadata_description(capacity));
      if (description != expected)
        return Status::InvalidArgument("native page metadata capacity description differs");
      HYLE_RETURN_IF_ERROR(vn::MemoryAccess::validate_resource(runtime,
          request.input_resources[a->metadata_port]));
    } else {
      HYLE_ASSIGN_OR_RETURN(a->native_recipe, make_recipe(*a, schema, allocator));
      HYLE_ASSIGN_OR_RETURN(a->program, Traits::prepare_program(a->preparation,
          a->native_recipe, a->recipe.layout, a->workload));
    }
    out->properties = {.persistent_bytes = std::nullopt, .runtime = runtime_properties,
                       .invocation_private_bytes = std::nullopt};
    return Status::Ok();
  };
  const auto status = build();
  if (!status.ok()) return status.IsUnimplemented() ? Outcome::unsupported(status) : Outcome::error(status);
  return Outcome::prepared();
}
} // namespace hyle::graph::vulkan::detail
