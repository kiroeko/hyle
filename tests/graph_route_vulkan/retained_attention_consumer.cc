// Actual paged/ragged Graph attention rearm with independent retained output.
#include <hyle/compute/batch_ops.h>
#include <hyle/compute/vulkan_memory.h>
#include <hyle/graph/batch.h>
#include <hyle/graph/result.h>
#include <hyle/graph/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace gb = g::batch;
using hyle::Result;
using hyle::Status;
constexpr std::uint32_t query_rows = 2, heads = 2, depth = 64, page_rows = 16, generations = 3;
constexpr std::size_t output_elements = query_rows * heads * depth, slice_first = 64, slice_elements = 64;
constexpr auto dtype = c::DataType::kBFloat16;
enum class Kind { kPagedCapacity, kPagedOwned, kRagged };

Status require(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}
std::uint16_t word(float value) {
  return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
}
struct Data final {
  c::BatchLayoutVersion layout;
  c::ReadLease query, key, value;
  c::PagedArenaRead paged_key, paged_value;
};
struct NativeRange final { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceSize offset = 0; };
struct Output final {
  c::ReadLease whole, slice;
  NativeRange range;
  std::uint64_t whole_generation = 0, slice_generation = 0;
};
struct Ports final { g::ValueId query, key, value, metadata, whole, slice; g::NodeId attention; };

Result<c::ReadLease> upload(const v::Context &context, const v::MemoryResource &device,
                           const c::TensorDesc &description, std::span<const std::uint16_t> payload) {
  HYLE_ASSIGN_OR_RETURN(auto native, context.upload(description, std::as_bytes(payload)));
  return device.import_readonly(native);
}
Result<c::PagedArenaRead> page(c::PagedArena &arena, const c::BatchLayoutVersion &layout,
                              std::uint32_t index, std::span<const std::uint16_t> payload) {
  HYLE_ASSIGN_OR_RETURN(auto writer, arena.acquire(index));
  HYLE_ASSIGN_OR_RETURN(auto storage, writer.lease().view());
  // Public typed resource IO initializes only the fixture's original writer.
  // The real ordinary attention Graph below performs all output computation.
  HYLE_RETURN_IF_ERROR(storage.write(std::as_bytes(payload)));
  HYLE_ASSIGN_OR_RETURN(auto ready, writer.publish(4));
  return arena.read(layout, std::span(&ready, 1));
}
Result<NativeRange> native_range(const v::MemoryResource &device, const c::ReadLease &lease,
                                const c::TensorDesc &description) {
  HYLE_ASSIGN_OR_RETURN(auto tensor, device.binding(lease, description));
  HYLE_ASSIGN_OR_RETURN(auto storage, device.storage_info(tensor));
  return NativeRange{storage.buffer, storage.offset_bytes};
}
Result<std::vector<std::uint16_t>> read(const v::Context &context, const v::MemoryResource &device,
                                       const c::ReadLease &lease, const c::TensorDesc &description) {
  HYLE_ASSIGN_OR_RETURN(auto size, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto storage, lease.view());
  HYLE_RETURN_IF_ERROR(require(storage.resource() == device.ref() && lease.size() == size,
      "retained attention output resource/extent changed"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, device.binding(lease, description));
  std::vector<std::uint16_t> result(size / 2);
  HYLE_RETURN_IF_ERROR(context.download(tensor, std::as_writable_bytes(std::span(result))));
  return result;
}
Status observe(const v::Context &context, const v::MemoryResource &device, Output &output,
               const c::TensorDesc &whole_desc, const c::TensorDesc &slice_desc, unsigned generation) {
  HYLE_RETURN_IF_ERROR(require(!output.whole.try_write() && !output.slice.try_write() &&
      output.whole.valid() && output.slice.valid() && output.whole.generation() == output.whole_generation &&
      output.slice.generation() == output.slice_generation, "retained attention output lost immutable read permission"));
  HYLE_ASSIGN_OR_RETURN(auto whole, read(context, device, output.whole, whole_desc));
  HYLE_ASSIGN_OR_RETURN(auto slice, read(context, device, output.slice, slice_desc));
  HYLE_RETURN_IF_ERROR(require(whole.size() == output_elements && slice.size() == slice_elements,
      "attention output element count differs"));
  for (std::size_t i = 0; i < whole.size(); ++i) {
    // Q=K=0 gives the causal means of the first three/four V rows. These
    // values are exactly BF16-representable in all three generations.
    const float expected = (i < heads * depth ? 2.F : 2.5F) * float(generation + 1);
    HYLE_RETURN_IF_ERROR(require(whole[i] == word(expected), "attention output or retained old full bytes differ from causal mean"));
  }
  HYLE_RETURN_IF_ERROR(require(std::equal(slice.begin(), slice.end(), whole.begin() + slice_first),
      "retained attention offset slice changed bytes"));
  HYLE_ASSIGN_OR_RETURN(auto range, native_range(device, output.whole, whole_desc));
  HYLE_ASSIGN_OR_RETURN(auto slice_range, native_range(device, output.slice, slice_desc));
  return require(range.buffer == output.range.buffer && range.offset == output.range.offset &&
      slice_range.buffer == range.buffer && slice_range.offset == range.offset + slice_first * 2,
      "retained attention whole/offset alias lost its original physical range relationship");
}
Status stale(g::Invocation &invocation) {
  auto submitted = invocation.submit(std::span<const g::QueueBinding>{});
  if (submitted) {
    submitted->wait().IgnoreError();
    return Status::Internal("attention submitted an incomplete/unprepared rearm epoch");
  }
  return require(submitted.error().IsFailedPrecondition(), "incomplete attention rearm rejection category changed");
}
Status withdraw(g::Invocation &invocation, const Ports &ports, const Data &data, bool paged) {
  if (paged) {
    // Typed frame ports remain source-owned configuration. Ordinary unbind
    // must reject without changing their original ready frame permissions;
    // the next typed prepare/bind performs their legal reconfiguration.
    const auto key_layout = data.paged_key.layout(), value_layout = data.paged_value.layout();
    const auto key_pages = data.paged_key.page_count(), value_pages = data.paged_value.page_count();
    HYLE_RETURN_IF_ERROR(require(invocation.unbind(ports.value).IsFailedPrecondition() &&
        invocation.unbind(ports.key).IsFailedPrecondition(),
        "ordinary unbind withdrew a source-owned paged frame port"));
    HYLE_RETURN_IF_ERROR(require(data.paged_key.valid() && data.paged_value.valid() &&
        data.paged_key.layout() == key_layout && data.paged_value.layout() == value_layout &&
        data.paged_key.page_count() == key_pages && data.paged_value.page_count() == value_pages &&
        key_layout == data.layout && value_layout == data.layout,
        "rejected paged unbind changed original ready frame layout or read permission"));
    HYLE_RETURN_IF_ERROR(invocation.unbind(ports.query));
    return stale(invocation);
  }
  // Ragged K/V are ordinary caller inputs. Reverse original bind order;
  // each legal partial withdrawal revokes the prepared submission.
  for (const auto port : {ports.value, ports.key, ports.query}) {
    HYLE_RETURN_IF_ERROR(invocation.unbind(port));
    HYLE_RETURN_IF_ERROR(stale(invocation));
  }
  return Status::Ok();
}
Result<Output> execute(g::Invocation &invocation, const Ports &ports, const Data &data,
                       const v::MemoryResource &device, const v::MemoryResource &staging,
                       const c::TensorDesc &query_desc, const c::TensorDesc &key_desc,
                       const c::TensorDesc &value_desc, const c::TensorDesc &whole_desc,
                       const c::TensorDesc &slice_desc, bool paged) {
  HYLE_RETURN_IF_ERROR(gb::prepare(invocation, {}, std::array{gb::NodeLayout{ports.attention, data.layout}},
      {.host_workspace = staging.ref()}));
  HYLE_RETURN_IF_ERROR(stale(invocation));
  HYLE_RETURN_IF_ERROR(invocation.bind(ports.query, query_desc, data.query));
  HYLE_RETURN_IF_ERROR(stale(invocation));
  if (paged) {
    HYLE_RETURN_IF_ERROR(gb::bind(invocation, ports.key, data.paged_key));
    HYLE_RETURN_IF_ERROR(stale(invocation));
    HYLE_RETURN_IF_ERROR(gb::bind(invocation, ports.value, data.paged_value));
  } else {
    HYLE_RETURN_IF_ERROR(invocation.bind(ports.key, key_desc, data.key));
    HYLE_RETURN_IF_ERROR(stale(invocation));
    HYLE_RETURN_IF_ERROR(invocation.bind(ports.value, value_desc, data.value));
  }
  HYLE_RETURN_IF_ERROR(stale(invocation));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay({}));
  Output result;
  {
    HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(std::span<const g::QueueBinding>{}));
    HYLE_RETURN_IF_ERROR(completion.wait());
    HYLE_ASSIGN_OR_RETURN(auto whole, invocation.result(ports.whole));
    HYLE_ASSIGN_OR_RETURN(auto slice, invocation.result(ports.slice));
    HYLE_ASSIGN_OR_RETURN(auto actual_whole, whole.description());
    HYLE_ASSIGN_OR_RETURN(auto actual_slice, slice.description());
    HYLE_RETURN_IF_ERROR(require(actual_whole == whole_desc && actual_slice == slice_desc,
        "attention output dtype or concrete description changed"));
    HYLE_ASSIGN_OR_RETURN(result.whole, whole.storage_lease());
    HYLE_ASSIGN_OR_RETURN(result.slice, slice.storage_lease());
    HYLE_RETURN_IF_ERROR(require(!invocation.unbind(ports.value).ok(),
        "live attention GraphOutput stopped pinning its original Invocation"));
  }
  HYLE_RETURN_IF_ERROR(require(!invocation.in_flight(), "successful attention Completion did not retire original native work"));
  HYLE_ASSIGN_OR_RETURN(result.range, native_range(device, result.whole, whole_desc));
  result.whole_generation = result.whole.generation(); result.slice_generation = result.slice.generation();
  return result;
}

Status exercise(const v::Context &context, const v::MemoryResource &device,
                const v::MemoryResource &staging, const g::ExecutionSource &source, Kind kind) {
  const bool paged = kind != Kind::kRagged;
  HYLE_ASSIGN_OR_RETURN(auto storage, c::CacheStorage::plain(dtype));
  const c::BatchCapacity capacity{.storage = paged ? c::BatchStorage::kPaged : c::BatchStorage::kRagged,
      .query_type = dtype, .output_type = dtype, .key_storage = storage, .value_storage = storage,
      .max_requests = 1, .max_query_rows = query_rows, .max_kv_rows = paged ? page_rows : 4,
      .max_pages = paged ? generations : 0, .max_page_refs = paged ? 1u : 0u,
      .query_heads = heads, .kv_heads = 1, .qk_head_dim = depth, .value_head_dim = depth,
      .page_size = paged ? page_rows : 1};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(capacity));
  HYLE_ASSIGN_OR_RETURN(auto query_desc, c::batch_query_description(capacity, query_rows));
  HYLE_ASSIGN_OR_RETURN(auto key_desc, c::batch_cache_description(capacity, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto value_desc, c::batch_cache_description(capacity, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto whole_desc, c::batch_output_description(capacity, query_rows));
  HYLE_ASSIGN_OR_RETURN(auto slice_desc, c::TensorDesc::dense(dtype, std::array<std::int64_t, 1>{slice_elements}));
  c::PagedArena keys, values;
  if (paged) {
    constexpr std::size_t bytes = generations * page_rows * depth * 2 + 1;
    HYLE_ASSIGN_OR_RETURN(keys, c::PagedArena::create(schema, c::BatchCacheRole::kKey, device.ref(), {.max_backing_bytes = bytes}));
    HYLE_ASSIGN_OR_RETURN(values, c::PagedArena::create(schema, c::BatchCacheRole::kValue, device.ref(), {.max_backing_bytes = bytes}));
  }
  std::array<Data, generations> data;
  const std::array<std::int32_t, 2> query_pointer{0, 2}, page_pointer{0, 1}, ragged_pointer{0, 4};
  const std::array<std::int32_t, 1> tail{4};
  for (std::uint32_t generation = 0; generation < generations; ++generation) {
    auto &current = data[generation];
    const std::array<std::int32_t, 1> page_index{static_cast<std::int32_t>(generation)};
    if (paged) {
      HYLE_ASSIGN_OR_RETURN(current.layout, schema.version({query_pointer, page_pointer, page_index, tail}));
    } else {
      HYLE_ASSIGN_OR_RETURN(current.layout, schema.version({query_pointer, ragged_pointer, {}, {}}));
    }
    const std::vector<std::uint16_t> query_payload(output_elements, 0);
    HYLE_ASSIGN_OR_RETURN(current.query, upload(context, device, query_desc, query_payload));
    const std::size_t rows = paged ? page_rows : 4;
    std::vector<std::uint16_t> key_payload(rows * depth, 0x7FC1), value_payload(rows * depth, 0x7FC1);
    std::fill_n(key_payload.begin(), 4 * depth, std::uint16_t{0});
    for (unsigned row = 0; row < 4; ++row)
      std::fill_n(value_payload.begin() + row * depth, depth, word(float((generation + 1) * (row + 1))));
    if (paged) {
      HYLE_ASSIGN_OR_RETURN(current.paged_key, page(keys, current.layout, generation, key_payload));
      HYLE_ASSIGN_OR_RETURN(current.paged_value, page(values, current.layout, generation, value_payload));
    } else {
      HYLE_ASSIGN_OR_RETURN(current.key, upload(context, device, key_desc, key_payload));
      HYLE_ASSIGN_OR_RETURN(current.value, upload(context, device, value_desc, value_payload));
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  Ports ports;
  HYLE_ASSIGN_OR_RETURN(ports.query, builder.input(query_desc));
  HYLE_ASSIGN_OR_RETURN(ports.key, builder.input(key_desc));
  HYLE_ASSIGN_OR_RETURN(ports.value, builder.input(value_desc));
  if (kind == Kind::kPagedOwned) {
    HYLE_ASSIGN_OR_RETURN(auto attention, gb::paged_attention(data[0].layout, {.scale = .125F, .causal = true}));
    HYLE_ASSIGN_OR_RETURN(ports.attention, builder.add_node(c::Operation::custom(attention),
        std::array{ports.query, ports.key, ports.value}));
  } else {
    HYLE_ASSIGN_OR_RETURN(auto metadata_desc, schema.metadata_description());
    HYLE_ASSIGN_OR_RETURN(ports.metadata, builder.input(metadata_desc));
    auto attention = paged ? c::Operation::custom(c::PagedAttention{.capacity = capacity, .scale = .125F, .causal = true})
                          : c::Operation::custom(c::RaggedAttention{.capacity = capacity, .scale = .125F, .causal = true});
    HYLE_ASSIGN_OR_RETURN(ports.attention, builder.add_node(std::move(attention),
        std::array{ports.query, ports.key, ports.value, ports.metadata}));
  }
  HYLE_ASSIGN_OR_RETURN(ports.whole, builder.result(ports.attention));
  HYLE_ASSIGN_OR_RETURN(auto shape, c::TensorShape::create(std::array<std::int64_t, 1>{output_elements}));
  HYLE_ASSIGN_OR_RETURN(auto reshape, c::View::reshape(shape));
  HYLE_ASSIGN_OR_RETURN(auto flattened, builder.add(reshape, ports.whole));
  HYLE_ASSIGN_OR_RETURN(auto slicing, c::View::slice(0, slice_first, slice_elements));
  HYLE_ASSIGN_OR_RETURN(ports.slice, builder.add(slicing, flattened));
  HYLE_RETURN_IF_ERROR(builder.publish(ports.whole)); HYLE_RETURN_IF_ERROR(builder.publish(ports.slice));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array domains{g::StorageDomain{device.ref()}, g::StorageDomain{staging.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, gb::compile(graph, {domains, targets}));
  HYLE_ASSIGN_OR_RETURN(auto invocation, compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_ASSIGN_OR_RETURN(auto first, execute(invocation, ports, data[0], device, staging,
      query_desc, key_desc, value_desc, whole_desc, slice_desc, paged));
  HYLE_RETURN_IF_ERROR(observe(context, device, first, whole_desc, slice_desc, 0));
  HYLE_RETURN_IF_ERROR(withdraw(invocation, ports, data[0], paged));
  HYLE_ASSIGN_OR_RETURN(auto second, execute(invocation, ports, data[1], device, staging,
      query_desc, key_desc, value_desc, whole_desc, slice_desc, paged));
  HYLE_RETURN_IF_ERROR(require(first.range.buffer != second.range.buffer || first.range.offset != second.range.offset,
      "attention generation2 reused a still-retained generation1 physical range"));
  HYLE_RETURN_IF_ERROR(observe(context, device, first, whole_desc, slice_desc, 0));
  HYLE_RETURN_IF_ERROR(observe(context, device, second, whole_desc, slice_desc, 1));
  first.whole.reset(); first.slice.reset();
  HYLE_RETURN_IF_ERROR(withdraw(invocation, ports, data[1], paged));
  HYLE_ASSIGN_OR_RETURN(auto third, execute(invocation, ports, data[2], device, staging,
      query_desc, key_desc, value_desc, whole_desc, slice_desc, paged));
  HYLE_RETURN_IF_ERROR(require(third.range.buffer == first.range.buffer && third.range.offset == first.range.offset &&
      (third.range.buffer != second.range.buffer || third.range.offset != second.range.offset),
      "attention generation3 did not reuse only the released original bounded spare"));
  HYLE_RETURN_IF_ERROR(observe(context, device, second, whole_desc, slice_desc, 1));
  HYLE_RETURN_IF_ERROR(observe(context, device, third, whole_desc, slice_desc, 2));
  invocation = {}; compiled.executable = {}; graph = {};
  HYLE_RETURN_IF_ERROR(observe(context, device, second, whole_desc, slice_desc, 1));
  return observe(context, device, third, whole_desc, slice_desc, 2);
}
Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  HYLE_RETURN_IF_ERROR(require(!context.environment().software_device, "retained attention consumer requires actual Vulkan hardware"));
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(context, v::MemoryKind::kStaging));
  HYLE_ASSIGN_OR_RETURN(auto source, g::vulkan::execution_source(context));
  for (const auto kind : {Kind::kPagedCapacity, Kind::kPagedOwned, Kind::kRagged})
    HYLE_RETURN_IF_ERROR(exercise(context, device, staging, source, kind));
  std::printf("PASS Vulkan retained attention: capacity-paged/owned-paged/ragged BF16 D64, one Invocation each, three generations, immutable whole/slice leases, typed-port unbind rejection/ordinary reverse unbind, strict preparation gates and bounded spare reuse; device=%s\n",
      context.environment().device_name.c_str());
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = run();
  if (status.ok()) return 0;
  std::fprintf(stderr, "Vulkan retained attention consumer: %.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
