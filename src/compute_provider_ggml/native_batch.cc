#include "native_batch.h"
#include "program.h"
#include "hyle/mem/std_allocator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

namespace hyle::compute::ggml_cpu::internal {
namespace {
struct Request final {
  mem::AllocatorRef allocator;
  detail::ProviderPrepared prepared;
  void *invocation = nullptr;
  std::array<TensorDesc, 3> inputs;
  TensorDesc output;
  std::size_t query_offset = 0, kv_offset = 0;
  Request() = default;
  Request(const Request &) = delete;
  Request &operator=(const Request &) = delete;
  ~Request() {
    if (invocation)
      prepared.vtable->destroy_invocation(invocation, allocator);
    if (prepared.state)
      prepared.vtable->destroy_prepared(prepared.state, allocator);
  }
};
struct Slot final {
  explicit Slot(mem::AllocatorRef a) : allocator(a), requests(mem::StdAllocator<Request *>(a)) {}
  void *context = nullptr;
  BatchCapacity capacity;
  BatchProgramOptions options;
  mem::AllocatorRef allocator;
  std::vector<Request *, mem::StdAllocator<Request *>> requests;
  BatchLayoutVersion layout;
  std::size_t known_bytes = 0;
  ~Slot() { clear(); }
  void clear() noexcept {
    for (auto *request : requests)
      allocator.destroy(request);
    requests.clear();
    layout = {};
    known_bytes = 0;
  }
};
Status qualify(void *, BatchCapacity capacity,
               const BatchProgramOptions &options) noexcept {
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(capacity));
  if ((capacity.storage != BatchStorage::kRagged && capacity.storage != BatchStorage::kPaged) ||
      (capacity.data_type != DataType::kFloat16 &&
       capacity.data_type != DataType::kBFloat16) ||
      (capacity.head_dim != 64 && capacity.head_dim != 128 &&
       capacity.head_dim != 256) || !std::isfinite(options.scale))
    return Status::Unimplemented("ggml CPU native batch requires qualified resident flash");
  const auto properties = options.resource.properties();
  if (!properties.has(mem::MemoryProperty::kHostAccessible) ||
      !properties.has(mem::MemoryProperty::kHostDeviceCoherent))
    return Status::Unimplemented("ggml CPU native batch requires coherent host storage");
  return Status::Ok();
}
Status create(void *context, BatchCapacity capacity,
              const BatchProgramOptions &options, mem::AllocatorRef allocator,
              void **output) noexcept {
  *output = nullptr;
  HYLE_RETURN_IF_ERROR(qualify(context, capacity, options));
  auto *slot = allocator.construct<Slot>(allocator);
  if (!slot)
    return Status::ResourceExhausted("ggml CPU ragged slot metadata");
  slot->context = context;
  slot->capacity = capacity;
  slot->options = options;
  slot->allocator = allocator;
  *output = slot;
  return Status::Ok();
}
void destroy(void *value, mem::AllocatorRef allocator) noexcept {
  allocator.destroy(static_cast<Slot *>(value));
}
BatchPlanReport report(const Slot &slot, bool reused = false) noexcept {
  return {.layout_generation = slot.layout.generation(),
          .float_workspace_bytes = slot.known_bytes,
          .native_metadata_unknown = true,
          .capture_compatible = false,
          .reused_native_plan = reused};
}
Result<bool> update(void *value, const BatchLayoutVersion &layout,
                    BatchPlanReport *output) noexcept {
  auto &slot = *static_cast<Slot *>(value);
  if (!slot.layout.valid() || !slot.layout.schema().owns(layout))
    return false;
  const auto old = slot.layout.host(), next = layout.host();
  const auto equal = [](auto a, auto b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
  };
  if (!equal(old.query_indptr, next.query_indptr) ||
      !equal(old.kv_indptr, next.kv_indptr) ||
      !equal(old.page_indices,next.page_indices) ||
      !equal(old.last_page_lengths,next.last_page_lengths))
    return false;
  // Paged recipes borrow this immutable topology, not the caller's arrays.
  // Equal content permits reuse while its original owner remains retained.
  if(slot.capacity.storage==BatchStorage::kRagged)slot.layout = layout;
  *output = report(slot, true);
  output->layout_generation=layout.generation();
  return true;
}
detail::EnqueueResult plan(void *value, Queue &, const BatchLayoutVersion &layout,
                          detail::BatchWorkspaceViews,
                          BatchPlanReport *output) noexcept {
  auto &slot = *static_cast<Slot *>(value);
  slot.clear();
  if (!layout.valid() || layout.schema().capacity() != slot.capacity)
    return detail::EnqueueResult::rejected(Status::InvalidArgument("ggml CPU ragged layout capacity"));
  try {
    slot.layout=layout;
    const auto host = layout.host();
    const auto width = data_type_size(slot.capacity.data_type);
    if (layout.facts().requests > slot.options.max_float_workspace_bytes /
        (sizeof(Request *) + sizeof(Request)))
      return detail::EnqueueResult::rejected(Status::ResourceExhausted("ggml CPU ragged request count exceeds budget"));
    slot.requests.reserve(layout.facts().requests);
    const auto metadata_bytes = slot.requests.capacity()*sizeof(Request *) +
        static_cast<std::size_t>(layout.facts().requests)*sizeof(Request);
    if (metadata_bytes > slot.options.max_float_workspace_bytes) {
      slot.clear();
      return detail::EnqueueResult::rejected(Status::ResourceExhausted("ggml CPU ragged request capacity exceeds budget"));
    }
    slot.known_bytes = metadata_bytes;
    for (std::size_t index = 0; index < layout.facts().requests; ++index) {
      const auto queries = host.query_indptr[index + 1] - host.query_indptr[index];
      const auto refs = host.kv_indptr[index + 1] - host.kv_indptr[index];
      const bool paged=slot.capacity.storage==BatchStorage::kPaged;
      const auto keys = paged?(refs?std::uint64_t(refs-1)*slot.capacity.page_size+
          std::uint32_t(host.last_page_lengths[index]):0):std::uint64_t(refs);
      if (!queries)
        continue;
      if (!keys || (slot.options.causal && keys < static_cast<std::uint64_t>(queries))) {
        slot.clear();
        return detail::EnqueueResult::rejected(Status::Unimplemented("ggml CPU ragged flash requires nonempty sufficient KV"));
      }
      const auto mask_bytes = slot.options.causal
          ? static_cast<std::uint64_t>(queries)*static_cast<std::uint64_t>(keys)*2u : 0u;
      const auto query_row_bytes = std::uint64_t(slot.capacity.query_heads)*slot.capacity.head_dim*8u;
      const auto available = slot.options.max_float_workspace_bytes - slot.known_bytes;
      if (mask_bytes > available || (query_row_bytes &&
          std::uint64_t(queries) > (available - mask_bytes)/query_row_bytes)) {
        slot.clear();
        return detail::EnqueueResult::rejected(Status::ResourceExhausted("ggml CPU ragged mask/query lower bound exceeds budget"));
      }
      auto *request = slot.allocator.construct<Request>();
      if (!request) {
        slot.clear();
        return detail::EnqueueResult::rejected(Status::ResourceExhausted("ggml CPU ragged request metadata"));
      }
      request->allocator = slot.allocator;
      // Insert immediately so every partial compile/slot is cleaned on failure.
      slot.requests.push_back(request);
      auto q = TensorDesc::dense(slot.capacity.data_type,
          std::array<std::int64_t, 4>{1, queries, slot.capacity.query_heads, slot.capacity.head_dim});
      auto k = TensorDesc::dense(slot.capacity.data_type,
          std::array<std::int64_t, 4>{1, paged?std::int64_t(slot.capacity.max_pages)*slot.capacity.page_size:
              static_cast<std::int64_t>(keys), slot.capacity.kv_heads, slot.capacity.head_dim});
      if (!q || !k) {
        const auto error = !q ? q.error() : k.error();
        slot.clear();
        return detail::EnqueueResult::rejected(error);
      }
      request->inputs = {*q, *k, *k};
      request->output = *q;
      request->query_offset = static_cast<std::size_t>(host.query_indptr[index]) *
          slot.capacity.query_heads * slot.capacity.head_dim * width;
      request->kv_offset = paged?0:static_cast<std::size_t>(host.kv_indptr[index]) *
          slot.capacity.kv_heads * slot.capacity.head_dim * width;
      Recipe recipe;
      recipe.values = {*q, *k, *k, *q};
      recipe.inputs = {0, 1, 2};
      recipe.outputs = {3};
      const auto operation=paged?Operation::custom(PagedRead{
          .pages=host.page_indices.subspan(static_cast<std::size_t>(host.kv_indptr[index]),static_cast<std::size_t>(refs)),
          .logical_rows=static_cast<std::uint32_t>(keys),.page_size=slot.capacity.page_size,
          .scale=slot.options.scale,.causal=slot.options.causal}):Operation(ScaledDotProductAttention{
          .layout = AttentionLayout::kBatchSequenceHeadDepth,
          .scale = slot.options.scale, .causal = slot.options.causal});
      recipe.nodes.push_back({operation, {0, 1, 2}, {3}});
      const auto compiled = compile(slot.context, std::move(recipe), slot.allocator, &request->prepared);
      if (compiled.kind != ProviderPrepareOutcomeKind::kPrepared) {
        slot.clear();
        return detail::EnqueueResult::rejected(compiled.status);
      }
      auto status = request->prepared.vtable->create_invocation(
          request->prepared.state, slot.allocator, {}, &request->invocation);
      if (!status.ok()) {
        slot.clear();
        return detail::EnqueueResult::rejected(status);
      }
      if (!native_invocation_reads_kv(request->invocation)) {
        slot.clear();
        return detail::EnqueueResult::rejected(Status::Unimplemented("ggml CPU ragged requires original flash direct KV roots"));
      }
      if(paged && !native_invocation_reads_pages(request->invocation,
          host.page_indices.subspan(static_cast<std::size_t>(host.kv_indptr[index]),static_cast<std::size_t>(refs)),
          slot.capacity.page_size,static_cast<std::uint32_t>(keys))){
        slot.clear();
        return detail::EnqueueResult::rejected(Status::Unimplemented("ggml CPU paged flash actual page/root contract differs"));
      }
      const auto bytes = native_invocation_bytes(request->prepared.state, request->invocation);
      if (bytes > slot.options.max_float_workspace_bytes - slot.known_bytes) {
        slot.clear();
        return detail::EnqueueResult::rejected(Status::ResourceExhausted("ggml CPU ragged known native workspace exceeds budget"));
      }
      slot.known_bytes += bytes;
    }
    slot.layout = layout;
    *output = report(slot);
    return detail::EnqueueResult::accepted_command();
  } catch (const std::bad_alloc &) {
    slot.clear();
    return detail::EnqueueResult::rejected(Status::ResourceExhausted("ggml CPU ragged plan allocation"));
  }
}
detail::EnqueueResult run(void *value, Queue &queue,
                         std::span<const ConstTensorView> inputs,
                         detail::BatchWorkspaceViews workspace) noexcept {
  auto &slot = *static_cast<Slot *>(value);
  if (!slot.layout.valid() || inputs.size() != 3)
    return detail::EnqueueResult::rejected(Status::FailedPrecondition("ggml CPU ragged slot is not planned"));
  bool accepted = false;
  for (auto *request : slot.requests) {
    std::array<ConstTensorView, 3> views;
    for (std::size_t port = 0; port < 3; ++port) {
      auto buffer = inputs[port].storage().subview(port == 0 ? request->query_offset : request->kv_offset,
                                                 *request->inputs[port].required_bytes());
      if (!buffer)
        return {accepted, buffer.error()};
      auto view = ConstTensorView::create(request->inputs[port], *buffer);
      if (!view)
        return {accepted, view.error()};
      views[port] = *view;
    }
    auto buffer = workspace.output.subview(request->query_offset, *request->output.required_bytes());
    if (!buffer)
      return {accepted, buffer.error()};
    auto output = TensorView::create(request->output, *buffer);
    if (!output)
      return {accepted, output.error()};
    const auto result = request->prepared.vtable->shared_submit(
        request->prepared.state, request->invocation, queue, views, std::span(&*output, 1));
    accepted |= result.accepted;
    if (!result.status.ok())
      return {accepted, result.status};
  }
  return detail::EnqueueResult::accepted_command();
}
const detail::BatchProviderVTable table{&qualify, &create, &destroy, &update, &plan, &run};
} // namespace
const detail::BatchProviderVTable *native_batch_table() noexcept { return &table; }
} // namespace hyle::compute::ggml_cpu::internal
