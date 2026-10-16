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
  std::array<std::size_t, 3> input_bytes{};
  std::size_t output_bytes = 0;
  std::size_t query_offset = 0, key_offset = 0, value_offset = 0, output_offset = 0;
  BatchLayoutVersion compiled_layout;
  std::uint32_t request_index=0;
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
  HYLE_RETURN_IF_ERROR(validate_batch_attention_options(options.causal,options.window_left));
  const auto q8=CacheStorage::q8_0().value();
  const auto plain=CacheStorage::plain(capacity.query_type).value();
  if(capacity.output_type!=capacity.query_type ||
      !((capacity.key_storage==plain && capacity.value_storage==plain) ||
        (capacity.key_storage==q8 && capacity.value_storage==q8)))
    return Status::Unimplemented("ggml CPU batch requires same-type plain or Q8_0 K/V and query/output dtype");
  if(!((capacity.qk_head_dim==capacity.value_head_dim) ||
       (capacity.qk_head_dim==64 && capacity.value_head_dim==128)))
    return Status::Unimplemented("ggml CPU batch value depth is outside its finite domain");
  if ((capacity.storage != BatchStorage::kRagged && capacity.storage != BatchStorage::kPaged) ||
      (capacity.query_type != DataType::kFloat16 &&
       capacity.query_type != DataType::kBFloat16) ||
      (capacity.qk_head_dim != 64 && capacity.qk_head_dim != 128 &&
       capacity.qk_head_dim != 256) || !std::isfinite(options.scale))
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
  if(!equal(old.query_indptr,next.query_indptr))return false;
  const bool same=equal(old.kv_indptr,next.kv_indptr) && equal(old.page_indices,next.page_indices) &&
      equal(old.last_page_lengths,next.last_page_lengths);
  const bool paged=slot.capacity.storage==BatchStorage::kPaged;bool runtime=true;
  for(std::size_t index=0;runtime && index+1<next.query_indptr.size();++index)
    if(next.query_indptr[index+1]-next.query_indptr[index]>1)runtime=false;
  // No mutation until every native request accepts the same runtime contract.
  if(runtime)for(const auto*request:slot.requests){
    if(paged){
      if(!native_invocation_can_update_q1_pages(request->invocation)){runtime=false;break;}
    }else{
      if(!native_invocation_can_update_q1_ragged(request->invocation)){runtime=false;break;}
      const auto index=request->request_index;const auto rows=next.kv_indptr[index+1]-next.kv_indptr[index];
      for(unsigned port=1;port<3;++port){
        const auto storage=port==1?slot.capacity.key_storage:slot.capacity.value_storage;
        const auto depth=port==1?slot.capacity.qk_head_dim:slot.capacity.value_head_dim;
        HYLE_ASSIGN_OR_RETURN(auto desc,cache_tensor_description(storage,
            std::array<std::int64_t,4>{1,rows,slot.capacity.kv_heads,depth}));
        if(auto bytes=desc.required_bytes();!bytes)return std::unexpected(bytes.error());
      }
    }
  }
  if(!same && !runtime)return false;
  if(runtime){
    for(auto*request:slot.requests){
      const auto index=request->request_index;const auto first=std::size_t(next.kv_indptr[index]);
      const auto refs=std::size_t(next.kv_indptr[index+1])-first;
      if(paged){
        const auto rows=std::uint32_t((refs-1)*slot.capacity.page_size+std::uint32_t(next.last_page_lengths[index]));
        native_invocation_update_q1_pages(request->invocation,next.page_indices.subspan(first,refs),rows);
      }else{
        // All descriptors and native aliases were preflighted before mutation.
        // The private submit maps these actual views; immutable Q/output stay fixed.
        for(unsigned port=1;port<3;++port){
          const auto storage=port==1?slot.capacity.key_storage:slot.capacity.value_storage;
          const auto depth=port==1?slot.capacity.qk_head_dim:slot.capacity.value_head_dim;
          auto desc=cache_tensor_description(storage,
              std::array<std::int64_t,4>{1,std::int64_t(refs),slot.capacity.kv_heads,depth}).value();
          const auto offset=first*std::size_t(desc.byte_strides()[1]);
          (port==1?request->key_offset:request->value_offset)=offset;
          request->input_bytes[port]=desc.required_bytes().value();request->inputs[port]=std::move(desc);
        }
        native_invocation_update_q1_ragged(request->invocation,std::uint32_t(refs));
      }
    }
  }
  // Each recipe retains its first compiled layout separately. Runtime leaf
  // pointers borrow only this current owning version, never caller spans.
  // Equal general paged recipes keep the version their original leaf still
  // points into. Q1 refreshes even equal-content pointers before owner change.
  if(runtime || !paged)slot.layout=layout;
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
    const auto width = data_type_size(slot.capacity.query_type);
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
      const bool boxed=slot.options.window_left>=0 ||
          !slot.capacity.key_storage.format().is_plain() ||
          slot.capacity.qk_head_dim!=slot.capacity.value_head_dim;
      const auto mask_bytes = slot.options.causal && !boxed && queries!=1
          ? static_cast<std::uint64_t>(queries)*static_cast<std::uint64_t>(keys)*2u : 0u;
      const auto query_row_bytes = std::uint64_t(slot.capacity.query_heads)*slot.capacity.qk_head_dim*8u;
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
      request->compiled_layout=layout;request->request_index=static_cast<std::uint32_t>(index);
      // Insert immediately so every partial compile/slot is cleaned on failure.
      slot.requests.push_back(request);
      auto q = TensorDesc::dense(slot.capacity.query_type,
          std::array<std::int64_t, 4>{1, queries, slot.capacity.query_heads, slot.capacity.qk_head_dim});
      auto k = cache_tensor_description(slot.capacity.key_storage,
          std::array<std::int64_t, 4>{1, paged?std::int64_t(slot.capacity.max_pages)*slot.capacity.page_size:
              static_cast<std::int64_t>(keys), slot.capacity.kv_heads, slot.capacity.qk_head_dim});
      auto v = cache_tensor_description(slot.capacity.value_storage,
          std::array<std::int64_t, 4>{1, paged?std::int64_t(slot.capacity.max_pages)*slot.capacity.page_size:
              static_cast<std::int64_t>(keys), slot.capacity.kv_heads, slot.capacity.value_head_dim});
      auto out = TensorDesc::dense(slot.capacity.output_type,
          std::array<std::int64_t,4>{1,queries,slot.capacity.query_heads,slot.capacity.value_head_dim});
      if (!q || !k || !v || !out) {
        const auto error = !q ? q.error() : !k ? k.error() : !v ? v.error() : out.error();
        slot.clear();
        return detail::EnqueueResult::rejected(error);
      }
      request->inputs = {*q, *k, *v};
      request->output = *out;
      // The compiled request owns immutable descriptions. Keep their extents
      // cold; each run still validates the actual storage and TensorViews.
      for (std::size_t port = 0; port < request->inputs.size(); ++port)
        request->input_bytes[port] = *request->inputs[port].required_bytes();
      request->output_bytes = *request->output.required_bytes();
      request->query_offset = static_cast<std::size_t>(host.query_indptr[index]) *
          slot.capacity.query_heads * slot.capacity.qk_head_dim * width;
      const auto key_row=cache_tensor_description(slot.capacity.key_storage,
          std::array<std::int64_t,2>{slot.capacity.kv_heads,slot.capacity.qk_head_dim}).value();
      const auto value_row=cache_tensor_description(slot.capacity.value_storage,
          std::array<std::int64_t,2>{slot.capacity.kv_heads,slot.capacity.value_head_dim}).value();
      request->key_offset=paged?0:static_cast<std::size_t>(host.kv_indptr[index])*key_row.required_bytes().value();
      request->value_offset=paged?0:static_cast<std::size_t>(host.kv_indptr[index])*value_row.required_bytes().value();
      request->output_offset=static_cast<std::size_t>(host.query_indptr[index])*
          slot.capacity.query_heads*slot.capacity.value_head_dim*data_type_size(slot.capacity.output_type);
      Recipe recipe;
      recipe.values = {*q, *k, *v, *out};
      recipe.inputs = {0, 1, 2};
      recipe.outputs = {3};
      const auto operation=Operation::custom(PagedRead{
          .pages=paged?host.page_indices.subspan(static_cast<std::size_t>(host.kv_indptr[index]),static_cast<std::size_t>(refs)):std::span<const std::int32_t>{},
          .logical_rows=static_cast<std::uint32_t>(keys),.page_size=slot.capacity.page_size,
          .scale=slot.options.scale,.causal=slot.options.causal,.window_left=slot.options.window_left});
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
      auto buffer = inputs[port].storage().subview(port == 0 ? request->query_offset : port==1?request->key_offset:request->value_offset,
                                                 request->input_bytes[port]);
      if (!buffer)
        return {accepted, buffer.error()};
      auto view = ConstTensorView::create(request->inputs[port], *buffer);
      if (!view)
        return {accepted, view.error()};
      views[port] = *view;
    }
    auto buffer = workspace.output.subview(request->output_offset, request->output_bytes);
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
