#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/storage_owner.h>
#include <hyle/graph/batch.h>
#include <hyle/graph/cuda.h>
#include <hyle/simd/dtype.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

#if !defined(HYLE_APPEND_STANDALONE)
#include <gtest/gtest.h>
#endif

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
namespace m = hyle::mem;
using hyle::Result;
using hyle::Status;

Status require(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}
std::uint16_t bf16(float value) { return hyle::simd::f32_to_bf16(value).bits; }

Result<c::ReadLease> upload(c::LeasePool &pool,
                          std::span<const std::uint16_t> words) {
  HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(words.size_bytes(), 256));
  HYLE_ASSIGN_OR_RETURN(auto storage, writer.view());
  HYLE_RETURN_IF_ERROR(storage.write(std::as_bytes(words)));
  return writer.publish();
}

Status journey(const c::Context &context, const c::StorageOwner &memory,
               m::MemoryResourceRef host_workspace, c::Queue &queue,
               bool missing_dependency, bool multi = false) {
  const std::size_t query_rows = multi ? 4u : 2u;
  HYLE_ASSIGN_OR_RETURN(auto plain, c::CacheStorage::plain(c::DataType::kBFloat16));
  const c::BatchCapacity capacity{
      .query_type = c::DataType::kBFloat16,
      .output_type = c::DataType::kBFloat16,
      .key_storage = plain, .value_storage = plain,
      .max_requests = multi ? 2u : 1u, .max_query_rows = multi ? 4u : 2u,
      .max_kv_rows = multi ? 16u : 8u,
      .max_pages = multi ? 7u : 6u, .max_page_refs = multi ? 6u : 4u,
      .query_heads = 2, .kv_heads = 1,
      .qk_head_dim = 64, .value_head_dim = 64, .page_size = 2};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(capacity));
  const std::array<std::int32_t, 2> old_query{0, 1}, next_query{0, 2};
  const std::array<std::int32_t, 2> old_indptr{0, 2}, next_indptr{0, 3};
  const std::array<std::int32_t, 2> old_ids{0, 1}, second_old_ids{0, 4};
  const std::array<std::int32_t, 3> next_ids{0, 2, 3}, second_next_ids{0, 5, 6};
  const std::array<std::int32_t, 1> tail{1};
  std::array<c::BatchLayoutVersion, 2> source_layouts, projected_layouts;
  HYLE_ASSIGN_OR_RETURN(source_layouts[0], schema.version({old_query, old_indptr, old_ids, tail}));
  HYLE_ASSIGN_OR_RETURN(projected_layouts[0], schema.version({next_query, next_indptr, next_ids, tail}));
  auto old_layout = source_layouts[0];
  auto next_layout = projected_layouts[0];
  if (multi) {
    HYLE_ASSIGN_OR_RETURN(source_layouts[1], schema.version({old_query, old_indptr, second_old_ids, tail}));
    HYLE_ASSIGN_OR_RETURN(projected_layouts[1], schema.version({next_query, next_indptr, second_next_ids, tail}));
    const std::array<std::int32_t, 3> old_q{0, 1, 2}, next_q{0, 2, 4};
    const std::array<std::int32_t, 3> old_ptr{0, 2, 4}, next_ptr{0, 3, 6};
    const std::array<std::int32_t, 4> old_pages{0, 1, 0, 4};
    const std::array<std::int32_t, 6> next_pages{0, 2, 3, 0, 5, 6};
    const std::array<std::int32_t, 2> tails{1, 1};
    HYLE_ASSIGN_OR_RETURN(old_layout, schema.version({old_q, old_ptr, old_pages, tails}));
    HYLE_ASSIGN_OR_RETURN(next_layout, schema.version({next_q, next_ptr, next_pages, tails}));
  }
  HYLE_ASSIGN_OR_RETURN(auto keys, c::PagedArena::create(schema,
      c::BatchCacheRole::kKey, memory.ref(), {.max_backing_bytes = 4096}));
  HYLE_ASSIGN_OR_RETURN(auto values, c::PagedArena::create(schema,
      c::BatchCacheRole::kValue, memory.ref(), {.max_backing_bytes = 4096}));

  // Old page 1 has only its first row initialized. The unused NaN tail must
  // remain private to this old page and cannot participate in attention.
  std::array<std::uint16_t, 128> zeros{}, first_page{}, partial_page{}, second_partial{};
  std::fill_n(first_page.begin(), 64, bf16(.125f));
  std::fill(first_page.begin() + 64, first_page.end(), bf16(.25f));
  std::fill_n(partial_page.begin(), 64, bf16(.375f));
  std::fill(partial_page.begin() + 64, partial_page.end(), std::uint16_t{0x7fc1});
  std::fill_n(second_partial.begin(), 64, bf16(.875f));
  std::fill(second_partial.begin() + 64, second_partial.end(), std::uint16_t{0x7fc1});
  std::array<c::ReadLease, 3> old_keys, old_values;
  const std::array<std::uint32_t, 3> source_pages{0, 1, 4};
  const auto source_page_count = multi ? 3u : 2u;
  for (std::size_t page = 0; page < source_page_count; ++page) {
    HYLE_ASSIGN_OR_RETURN(auto key, keys.acquire(source_pages[page]));
    HYLE_ASSIGN_OR_RETURN(auto value, values.acquire(source_pages[page]));
    HYLE_ASSIGN_OR_RETURN(auto key_storage, key.lease().view());
    HYLE_ASSIGN_OR_RETURN(auto value_storage, value.lease().view());
    HYLE_RETURN_IF_ERROR(key_storage.write(std::as_bytes(std::span(zeros))));
    HYLE_RETURN_IF_ERROR(value_storage.write(std::as_bytes(std::span(
        page == 0 ? first_page : (page == 1 ? partial_page : second_partial)))));
    HYLE_ASSIGN_OR_RETURN(old_keys[page], key.publish(page == 0 ? 2 : 1));
    HYLE_ASSIGN_OR_RETURN(old_values[page], value.publish(page == 0 ? 2 : 1));
  }
  std::array<c::PagedArenaRead, 2> source_keys, source_values;
  HYLE_ASSIGN_OR_RETURN(source_keys[0], keys.read(source_layouts[0], std::span(old_keys).first(2)));
  HYLE_ASSIGN_OR_RETURN(source_values[0], values.read(source_layouts[0], std::span(old_values).first(2)));
  auto previous_keys = source_keys[0];
  auto previous_values = source_values[0];
  if (multi) {
    const std::array second_keys{old_keys[0], old_keys[2]};
    const std::array second_values{old_values[0], old_values[2]};
    HYLE_ASSIGN_OR_RETURN(source_keys[1], keys.read(source_layouts[1], second_keys));
    HYLE_ASSIGN_OR_RETURN(source_values[1], values.read(source_layouts[1], second_values));
    HYLE_ASSIGN_OR_RETURN(previous_keys, keys.reframe(old_layout, source_keys));
    HYLE_ASSIGN_OR_RETURN(previous_values, values.reframe(old_layout, source_values));
    HYLE_RETURN_IF_ERROR(require(previous_keys.page_count() == 3 && previous_values.page_count() == 3,
        "composed two-request frame did not deduplicate its shared full-page guard"));
  }
  std::array<c::PagedPageWriter, 4> dirty_keys, dirty_values;
  const std::array<std::uint32_t, 4> target_pages{2, 3, 5, 6};
  const auto dirty_count = multi ? 4u : 2u;
  for (std::size_t page = 0; page < dirty_count; ++page) {
    HYLE_ASSIGN_OR_RETURN(dirty_keys[page], keys.acquire(target_pages[page]));
    HYLE_ASSIGN_OR_RETURN(dirty_values[page], values.acquire(target_pages[page]));
  }
  HYLE_ASSIGN_OR_RETURN(auto append_keys,
      gb::PagedAppendPages::create(keys, next_layout, previous_keys, std::span(dirty_keys).first(dirty_count)));
  HYLE_ASSIGN_OR_RETURN(auto append_values,
      gb::PagedAppendPages::create(values, next_layout, previous_values, std::span(dirty_values).first(dirty_count)));

  HYLE_ASSIGN_OR_RETURN(auto rows, c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{static_cast<std::int64_t>(query_rows), 1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto query_description, c::batch_query_description(capacity, static_cast<std::uint32_t>(query_rows)));
  HYLE_ASSIGN_OR_RETURN(auto key_description,
      c::batch_cache_description(schema, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto value_description,
      c::batch_cache_description(schema, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto query, builder.input(query_description));
  HYLE_ASSIGN_OR_RETURN(auto incoming_keys, builder.input(rows));
  HYLE_ASSIGN_OR_RETURN(auto incoming_values, builder.input(rows));
  HYLE_ASSIGN_OR_RETURN(auto key_bank, builder.input(key_description));
  HYLE_ASSIGN_OR_RETURN(auto value_bank, builder.input(value_description));
  // Discovery sees the original mixed DAG before logical views specialize.
  HYLE_ASSIGN_OR_RETURN(auto row_view, g::ViewSpec::reshape(std::array{
      g::Dimension::fixed(static_cast<std::int64_t>(query_rows)), g::Dimension::fixed(1), g::Dimension::fixed(64)}));
  HYLE_ASSIGN_OR_RETURN(auto viewed_keys, builder.add(row_view, incoming_keys));
  HYLE_ASSIGN_OR_RETURN(auto produced_keys, builder.add(c::Copy{}, viewed_keys));
  HYLE_ASSIGN_OR_RETURN(auto produced_values, builder.add(c::Copy{}, incoming_values));
  HYLE_ASSIGN_OR_RETURN(auto key_append, gb::paged_append(schema, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto value_append, gb::paged_append(schema, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto key_writer,
      builder.add_node(c::Operation::custom(key_append), std::array{produced_keys}));
  HYLE_ASSIGN_OR_RETURN(auto value_writer,
      builder.add_node(c::Operation::custom(value_append), std::array{produced_values}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(value_writer, key_writer));
  HYLE_ASSIGN_OR_RETURN(auto attention,
      gb::paged_attention(next_layout, {.scale = .125f, .causal = true}));
  HYLE_ASSIGN_OR_RETURN(auto reader, builder.add_node(c::Operation::custom(attention),
      std::array{query, key_bank, value_bank}));
  if (!missing_dependency) {
    HYLE_RETURN_IF_ERROR(builder.depends_on(reader, key_writer));
    HYLE_RETURN_IF_ERROR(builder.depends_on(reader, value_writer));
  }
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(reader));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array domains{g::StorageDomain{memory.ref()}};
  HYLE_ASSIGN_OR_RETURN(auto source, g::ExecutionSource::acquire(context));
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, gb::compile(graph,
      {.storage_domains = domains, .targets = targets}));
  HYLE_ASSIGN_OR_RETURN(auto invocation, compiled.executable.create_invocation(
      {.resources = g::ResourceOptions{}}));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  HYLE_RETURN_IF_ERROR(gb::prepare(invocation, queues,
      std::array{gb::NodeLayout{reader, next_layout}},
      {.host_workspace = host_workspace}));
  const auto bound = gb::bind_append(invocation, graph, key_writer, key_bank,
                                   std::move(append_keys));
  if (missing_dependency) {
    HYLE_RETURN_IF_ERROR(require(!bound.ok() && append_keys.valid(),
        "missing DAG edge accepted append or consumed its original pages"));
    HYLE_ASSIGN_OR_RETURN(auto &first, append_keys.page(0));
    HYLE_RETURN_IF_ERROR(require(first.lease().valid(),
        "missing DAG edge withdrew the caller's original writer"));
    std::puts("CUDA paged append missing dependency rejected before ownership transfer PASS");
    return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(bound);
  HYLE_RETURN_IF_ERROR(gb::bind_append(invocation, graph, value_writer, value_bank,
                                     std::move(append_values)));
  HYLE_RETURN_IF_ERROR(require(!append_keys.valid() && !append_values.valid(),
      "append bind did not transfer its page owners"));

  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(memory.ref(), 8192,
      {.max_blocks = 8, .backing_alignment = 256}));
  std::array<std::uint16_t, 512> query_zeros{};
  std::array<std::uint16_t, 256> incoming_zeros{}, appended_values{};
  const std::array<float, 4> incoming{.5f, .75f, 1.f, 1.25f};
  for (std::size_t row = 0; row < query_rows; ++row)
    std::fill_n(appended_values.begin() + row * 64, 64, bf16(incoming[row]));
  HYLE_ASSIGN_OR_RETURN(auto query_input, upload(pool, std::span(query_zeros).first(query_rows * 128)));
  HYLE_ASSIGN_OR_RETURN(auto key_input, upload(pool, std::span(incoming_zeros).first(query_rows * 64)));
  HYLE_ASSIGN_OR_RETURN(auto value_input, upload(pool, std::span(appended_values).first(query_rows * 64)));
  HYLE_RETURN_IF_ERROR(invocation.bind(query, query_description, query_input));
  HYLE_RETURN_IF_ERROR(invocation.bind(incoming_keys, rows, key_input));
  HYLE_RETURN_IF_ERROR(invocation.bind(incoming_values, rows, value_input));
  HYLE_ASSIGN_OR_RETURN(auto output_description, c::batch_output_description(capacity, static_cast<std::uint32_t>(query_rows)));
  HYLE_ASSIGN_OR_RETURN(auto destination, pool.allocate(query_rows * 256, 256));
  HYLE_RETURN_IF_ERROR(invocation.bind(output, output_description, std::move(destination)));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(queues));
  HYLE_RETURN_IF_ERROR(completion.wait());
  completion = {};
  {
    HYLE_ASSIGN_OR_RETURN(auto result, invocation.result(output));
    HYLE_ASSIGN_OR_RETURN(auto output_lease, result.storage_lease());
    HYLE_ASSIGN_OR_RETURN(auto output_storage, output_lease.view());
    std::array<std::uint16_t, 512> actual{};
    HYLE_RETURN_IF_ERROR(output_storage.read(std::as_writable_bytes(std::span(actual).first(query_rows * 128))));
    const std::array<float, 4> expected_means{.3125f, .4f, .5625f, .7f};
    for (std::size_t i = 0; i < query_rows * 128; ++i) {
      const auto expected = bf16(expected_means[i / 128]);
      if (actual[i] != expected) {
        std::fprintf(stderr, "CUDA append oracle index=%zu actual=%04x expected=%04x\n",
            i, unsigned(actual[i]), unsigned(expected));
        return Status::Internal("causal CUDA append attention mean differs");
      }
    }
  } // Release GraphOutput/observer before taking the retired page transactions.
  HYLE_ASSIGN_OR_RETURN(auto returned_keys, gb::take_append(invocation, key_writer));
  HYLE_ASSIGN_OR_RETURN(auto returned_values, gb::take_append(invocation, value_writer));
  {
    // A dormant recording is not page permission or a prepared transaction.
    // Taking both original writers must revoke submit until a fresh bind/prepare.
    auto stale_submit = invocation.submit(queues);
    if (stale_submit) {
      stale_submit->wait().IgnoreError(); // Retire any erroneously accepted work.
      return Status::Internal("taken append transactions still permitted submit");
    }
    HYLE_RETURN_IF_ERROR(require(stale_submit.error().IsFailedPrecondition(),
        "submit without rebound append pages did not reject its missing preparation"));
  }
  HYLE_ASSIGN_OR_RETURN(auto next_keys, returned_keys.ready());
  HYLE_ASSIGN_OR_RETURN(auto next_values, returned_values.ready());
  HYLE_RETURN_IF_ERROR(require(next_keys.layout() == next_layout &&
      next_values.layout() == next_layout && next_keys.page_count() == (multi ? 5u : 3u) &&
      next_values.page_count() == (multi ? 5u : 3u), "new ready frames do not match the exact appended layout"));
  HYLE_RETURN_IF_ERROR(require(previous_keys.layout() == old_layout &&
      previous_values.layout() == old_layout && previous_values.page_count() == (multi ? 3u : 2u),
      "append mutated the original snapshot layout"));
  HYLE_RETURN_IF_ERROR(require(!keys.acquire(1) && !values.acquire(1) &&
      !keys.acquire(2) && !values.acquire(2), "old or new ready pages lost their exact permissions"));
  if (multi) {
    std::array<c::PagedArenaRead, 2> projected_keys, projected_values;
    for (std::size_t request = 0; request < 2; ++request) {
      HYLE_ASSIGN_OR_RETURN(projected_keys[request], keys.reframe(projected_layouts[request], std::span(&next_keys, 1)));
      HYLE_ASSIGN_OR_RETURN(projected_values[request], values.reframe(projected_layouts[request], std::span(&next_values, 1)));
      HYLE_RETURN_IF_ERROR(require(projected_keys[request].layout() == projected_layouts[request] &&
          projected_values[request].layout() == projected_layouts[request] &&
          projected_keys[request].page_count() == 3 && projected_values[request].page_count() == 3 &&
          source_keys[request].layout() == source_layouts[request] && source_values[request].layout() == source_layouts[request],
          "typed request projection changed a source frame or retained the other request's footprint"));
    }
    next_keys = {}; next_values = {}; returned_keys = {}; returned_values = {};
    HYLE_RETURN_IF_ERROR(require(!keys.acquire(2) && !values.acquire(2) && !keys.acquire(5) && !values.acquire(5),
        "projected request frames lost their independently retained page guards"));
  }
  for (std::size_t page = 0; page < source_page_count; ++page) {
    HYLE_ASSIGN_OR_RETURN(auto old_storage, old_values[page].view());
    std::array<std::uint16_t, 128> actual{};
    HYLE_RETURN_IF_ERROR(old_storage.read(std::as_writable_bytes(std::span(actual))));
    HYLE_RETURN_IF_ERROR(require(actual == (page == 0 ? first_page : (page == 1 ? partial_page : second_partial)),
        "COW append changed an old snapshot byte or its unused tail"));
  }
  std::puts(multi
      ? "CUDA two-request compose -> same Graph append/causal attention -> typed request projection PASS"
      : "CUDA ordinary Copy -> exact append/COW -> causal paged attention -> ready frames PASS");
  return Status::Ok();
}

// The metadata cases share one ordinary compiled Graph and its original
// page/lease path. Only page mappings change between these cold episodes.
struct MetadataGraph final {
  c::BatchCapacity capacity;
  c::BatchSchema schema;
  c::TensorDesc rows, query_description, output_description;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::ValueId query, incoming_keys, incoming_values, key_bank, value_bank, output;
  g::NodeId key_writer, value_writer, reader;
};

Result<c::BatchLayoutVersion> metadata_layout(const c::BatchSchema &schema,
    bool next, std::uint32_t tail_page, std::uint32_t last_page = 0) {
  const std::array<std::int32_t, 2> queries{0, next ? 2 : 1};
  const std::array<std::int32_t, 2> indptr{0, next ? 3 : 2};
  const std::array<std::int32_t, 3> ids{0,
      static_cast<std::int32_t>(tail_page), static_cast<std::int32_t>(last_page)};
  const std::array<std::int32_t, 1> tail{1};
  return schema.version({queries, indptr,
      std::span(ids).first(next ? 3 : 2), tail});
}

Result<MetadataGraph> metadata_graph(const c::Context &context,
                                     const c::StorageOwner &memory, bool exclusive = false) {
  MetadataGraph result;
  HYLE_ASSIGN_OR_RETURN(auto plain, c::CacheStorage::plain(c::DataType::kBFloat16));
  result.capacity = {.query_type = c::DataType::kBFloat16,
      .output_type = c::DataType::kBFloat16, .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = 2, .max_kv_rows = 8,
      .max_pages = 6, .max_page_refs = 4, .query_heads = 2, .kv_heads = 1,
      .qk_head_dim = 64, .value_head_dim = 64, .page_size = 2};
  if (exclusive) { result.capacity.max_query_rows = 1; result.capacity.page_size = 4; }
  HYLE_ASSIGN_OR_RETURN(result.schema, c::BatchSchema::create(result.capacity));
  c::BatchLayoutVersion initial;
  if (exclusive) {
    const std::array<std::int32_t, 2> q{0, 1}, ptr{0, 1};
    const std::array<std::int32_t, 1> ids{2}, tail{2};
    HYLE_ASSIGN_OR_RETURN(initial, result.schema.version({q, ptr, ids, tail}));
  } else { HYLE_ASSIGN_OR_RETURN(initial, metadata_layout(result.schema, true, 2, 3)); }
  const auto row_count = result.capacity.max_query_rows;
  HYLE_ASSIGN_OR_RETURN(result.rows, c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{row_count, 1, 64}));
  HYLE_ASSIGN_OR_RETURN(result.query_description,
      c::batch_query_description(result.capacity, row_count));
  HYLE_ASSIGN_OR_RETURN(result.output_description,
      c::batch_output_description(result.capacity, row_count));
  HYLE_ASSIGN_OR_RETURN(auto key_description,
      c::batch_cache_description(result.schema, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto value_description,
      c::batch_cache_description(result.schema, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(result.query, builder.input(result.query_description));
  HYLE_ASSIGN_OR_RETURN(result.incoming_keys, builder.input(result.rows));
  HYLE_ASSIGN_OR_RETURN(result.incoming_values, builder.input(result.rows));
  HYLE_ASSIGN_OR_RETURN(result.key_bank, builder.input(key_description));
  HYLE_ASSIGN_OR_RETURN(result.value_bank, builder.input(value_description));
  HYLE_ASSIGN_OR_RETURN(auto key_append, gb::paged_append(result.schema, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto value_append, gb::paged_append(result.schema, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(result.key_writer,
      builder.add_node(c::Operation::custom(key_append), std::array{result.incoming_keys}));
  HYLE_ASSIGN_OR_RETURN(result.value_writer,
      builder.add_node(c::Operation::custom(value_append), std::array{result.incoming_values}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(result.value_writer, result.key_writer));
  HYLE_ASSIGN_OR_RETURN(auto attention,
      gb::paged_attention(initial, {.scale = .125f, .causal = true}));
  HYLE_ASSIGN_OR_RETURN(result.reader, builder.add_node(c::Operation::custom(attention),
      std::array{result.query, result.key_bank, result.value_bank}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(result.reader, result.key_writer));
  HYLE_RETURN_IF_ERROR(builder.depends_on(result.reader, result.value_writer));
  HYLE_ASSIGN_OR_RETURN(result.output, builder.result(result.reader));
  HYLE_RETURN_IF_ERROR(builder.publish(result.output));
  HYLE_ASSIGN_OR_RETURN(result.graph, std::move(builder).finish());
  const std::array domains{g::StorageDomain{memory.ref()}};
  HYLE_ASSIGN_OR_RETURN(auto source, g::ExecutionSource::acquire(context));
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(result.compiled, gb::compile(result.graph,
      {.storage_domains = domains, .targets = targets}));
  return result;
}

struct MetadataPages final {
  c::PagedArena keys, values;
  std::array<c::ReadLease, 2> old_keys, old_values;
  c::ReadLease decoy;
  c::PagedArenaRead previous_keys, previous_values;
  float scale = 1;
  float tail_value = .375f;
};

Result<MetadataPages> metadata_pages(const MetadataGraph &graph,
    const c::StorageOwner &memory, float scale, bool different_old_tail = false) {
  MetadataPages result;
  result.scale = scale;
  result.tail_value = .375f * scale;
  HYLE_ASSIGN_OR_RETURN(result.keys, c::PagedArena::create(graph.schema,
      c::BatchCacheRole::kKey, memory.ref(), {.max_backing_bytes = 4096}));
  HYLE_ASSIGN_OR_RETURN(result.values, c::PagedArena::create(graph.schema,
      c::BatchCacheRole::kValue, memory.ref(), {.max_backing_bytes = 4096}));
  std::array<std::uint16_t, 128> zeros{}, words{};
  for (std::uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
    HYLE_ASSIGN_OR_RETURN(auto key, result.keys.acquire(ordinal));
    const auto value_id = ordinal == 1 && different_old_tail ? 4u : ordinal;
    HYLE_ASSIGN_OR_RETURN(auto value, result.values.acquire(value_id));
    HYLE_ASSIGN_OR_RETURN(auto key_storage, key.lease().view());
    HYLE_ASSIGN_OR_RETURN(auto value_storage, value.lease().view());
    HYLE_RETURN_IF_ERROR(key_storage.write(std::as_bytes(std::span(zeros))));
    std::fill_n(words.begin(), 64, bf16((ordinal == 0 ? .125f : .375f) * scale));
    std::fill(words.begin() + 64, words.end(),
        ordinal == 0 ? bf16(.25f * scale) : std::uint16_t{0x7fc1});
    HYLE_RETURN_IF_ERROR(value_storage.write(std::as_bytes(std::span(words))));
    HYLE_ASSIGN_OR_RETURN(result.old_keys[ordinal], key.publish(ordinal == 0 ? 2 : 1));
    HYLE_ASSIGN_OR_RETURN(result.old_values[ordinal], value.publish(ordinal == 0 ? 2 : 1));
  }
  if (different_old_tail) {
    // Wrongly sharing K's source-page1 instructions with V must read a
    // deterministic wrong value rather than relying on uninitialized memory.
    HYLE_ASSIGN_OR_RETURN(auto decoy, result.values.acquire(1));
    HYLE_ASSIGN_OR_RETURN(auto storage, decoy.lease().view());
    words.fill(bf16(1.375f * scale));
    HYLE_RETURN_IF_ERROR(storage.write(std::as_bytes(std::span(words))));
    HYLE_ASSIGN_OR_RETURN(result.decoy, decoy.publish(2));
  }
  HYLE_ASSIGN_OR_RETURN(auto key_layout, metadata_layout(graph.schema, false, 1));
  HYLE_ASSIGN_OR_RETURN(auto value_layout,
      metadata_layout(graph.schema, false, different_old_tail ? 4u : 1u));
  HYLE_ASSIGN_OR_RETURN(result.previous_keys, result.keys.read(key_layout, result.old_keys));
  HYLE_ASSIGN_OR_RETURN(result.previous_values, result.values.read(value_layout, result.old_values));
  return result;
}

struct MetadataRun final {
  c::LeasePool pool;
  g::Invocation invocation;
  c::ReadLease query, keys, values;
};

Result<MetadataRun> metadata_run(const MetadataGraph &graph,
    const c::StorageOwner &memory, float scale) {
  MetadataRun result;
  HYLE_ASSIGN_OR_RETURN(result.invocation, graph.compiled.executable.create_invocation(
      {.resources = g::ResourceOptions{}}));
  HYLE_ASSIGN_OR_RETURN(result.pool, c::LeasePool::create(memory.ref(), 8192,
      {.max_blocks = 8, .backing_alignment = 256}));
  std::array<std::uint16_t, 256> query{};
  std::array<std::uint16_t, 128> keys{}, values{};
  std::fill_n(values.begin(), 64, bf16(.5f * scale));
  std::fill(values.begin() + 64, values.end(), bf16(.75f * scale));
  const auto count = graph.capacity.max_query_rows;
  HYLE_ASSIGN_OR_RETURN(result.query, upload(result.pool, std::span(query).first(std::size_t(count) * 128)));
  HYLE_ASSIGN_OR_RETURN(result.keys, upload(result.pool, std::span(keys).first(std::size_t(count) * 64)));
  HYLE_ASSIGN_OR_RETURN(result.values, upload(result.pool, std::span(values).first(std::size_t(count) * 64)));
  HYLE_RETURN_IF_ERROR(result.invocation.bind(graph.query, graph.query_description, result.query));
  HYLE_RETURN_IF_ERROR(result.invocation.bind(graph.incoming_keys, graph.rows, result.keys));
  HYLE_RETURN_IF_ERROR(result.invocation.bind(graph.incoming_values, graph.rows, result.values));
  HYLE_ASSIGN_OR_RETURN(auto bytes, graph.output_description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto destination, result.pool.allocate(bytes, 256));
  HYLE_RETURN_IF_ERROR(result.invocation.bind(graph.output, graph.output_description,
                                           std::move(destination)));
  return result;
}

Status prepare_metadata_episode(const MetadataGraph &graph, MetadataRun &run,
    MetadataPages &pages, m::MemoryResourceRef host_workspace,
    std::span<const g::QueueBinding> queues, c::BatchLayoutVersion layout,
    std::uint32_t first_page, bool expect_reuse) {
  std::array<c::PagedPageWriter, 2> dirty_keys, dirty_values;
  std::array<std::uint16_t, 128> poison;
  poison.fill(0x7fc1);
  for (std::uint32_t i = 0; i < 2; ++i) {
    HYLE_ASSIGN_OR_RETURN(dirty_keys[i], pages.keys.acquire(first_page + i));
    HYLE_ASSIGN_OR_RETURN(dirty_values[i], pages.values.acquire(first_page + i));
    HYLE_ASSIGN_OR_RETURN(auto key, dirty_keys[i].lease().view());
    HYLE_ASSIGN_OR_RETURN(auto value, dirty_values[i].lease().view());
    HYLE_RETURN_IF_ERROR(key.write(std::as_bytes(std::span(poison))));
    HYLE_RETURN_IF_ERROR(value.write(std::as_bytes(std::span(poison))));
  }
  HYLE_ASSIGN_OR_RETURN(auto append_keys, gb::PagedAppendPages::create(
      pages.keys, layout, pages.previous_keys, dirty_keys));
  HYLE_ASSIGN_OR_RETURN(auto append_values, gb::PagedAppendPages::create(
      pages.values, layout, pages.previous_values, dirty_values));
  HYLE_RETURN_IF_ERROR(gb::prepare(run.invocation, queues,
      std::array{gb::NodeLayout{graph.reader, layout}}, {.host_workspace = host_workspace}));
  HYLE_RETURN_IF_ERROR(gb::bind_append(run.invocation, graph.graph, graph.key_writer,
      graph.key_bank, std::move(append_keys)));
  HYLE_RETURN_IF_ERROR(gb::bind_append(run.invocation, graph.graph, graph.value_writer,
      graph.value_bank, std::move(append_values)));
  HYLE_RETURN_IF_ERROR(run.invocation.prepare_replay(queues,
      {.mode = g::ReplayMode::kRequireCapture}));
  HYLE_ASSIGN_OR_RETURN(auto report, run.invocation.replay_report());
  HYLE_RETURN_IF_ERROR(require(report.route == g::ReplayRoute::kCaptured,
      "metadata append did not use the original captured Graph"));
  const bool expected_capture = expect_reuse
      ? report.recorded_capture_range_count == 0 && report.reused_capture_range_count > 0
      : report.recorded_capture_range_count > 0 && report.reused_capture_range_count == 0;
  if (!expected_capture)
    std::fprintf(stderr,
        "CUDA append capture expect_reuse=%u route=%u recorded=%zu reused=%zu total=%zu\n",
        unsigned(expect_reuse), unsigned(report.route), report.recorded_capture_range_count,
        report.reused_capture_range_count,
        report.recorded_capture_range_count + report.reused_capture_range_count);
  HYLE_RETURN_IF_ERROR(require(expected_capture,
      "metadata append capture recording/reuse did not match this episode"));
  return Status::Ok();
}

Result<std::array<c::PagedArenaRead, 2>> finish_metadata_episode(
    const MetadataGraph &graph, MetadataRun &run, MetadataPages &pages,
    std::span<const g::QueueBinding> queues, c::BatchLayoutVersion layout) {
  HYLE_ASSIGN_OR_RETURN(auto done, run.invocation.submit(queues));
  HYLE_RETURN_IF_ERROR(done.wait()); done = {};
  {
    HYLE_ASSIGN_OR_RETURN(auto result, run.invocation.result(graph.output));
    HYLE_ASSIGN_OR_RETURN(auto lease, result.storage_lease());
    HYLE_ASSIGN_OR_RETURN(auto storage, lease.view());
    std::array<std::uint16_t, 256> actual{};
    HYLE_RETURN_IF_ERROR(storage.read(std::as_writable_bytes(std::span(actual))));
    const float first_sum = (.125f + .25f + .5f) * pages.scale + pages.tail_value;
    for (std::size_t i = 0; i < actual.size(); ++i)
      HYLE_RETURN_IF_ERROR(require(actual[i] == bf16(i < 128 ? first_sum / 4.f :
          (first_sum + .75f * pages.scale) / 5.f),
          "dynamic append metadata read the wrong source/target page"));
  }
  HYLE_ASSIGN_OR_RETURN(auto keys, gb::take_append(run.invocation, graph.key_writer));
  HYLE_ASSIGN_OR_RETURN(auto values, gb::take_append(run.invocation, graph.value_writer));
  auto stale = run.invocation.submit(queues);
  if (stale) {
    stale->wait().IgnoreError();
    return std::unexpected(Status::Internal("taken metadata append still allowed submit"));
  }
  HYLE_RETURN_IF_ERROR(require(stale.error().IsFailedPrecondition(),
      "taken metadata append did not revoke preparation"));
  std::array<c::PagedArenaRead, 2> ready;
  HYLE_ASSIGN_OR_RETURN(ready[0], keys.ready());
  HYLE_ASSIGN_OR_RETURN(ready[1], values.ready());
  HYLE_RETURN_IF_ERROR(require(ready[0].layout() == layout && ready[1].layout() == layout,
      "metadata append published a stale layout"));
  HYLE_ASSIGN_OR_RETURN(auto old_storage, pages.old_values[1].view());
  std::array<std::uint16_t, 128> old{};
  HYLE_RETURN_IF_ERROR(old_storage.read(std::as_writable_bytes(std::span(old))));
  for (std::size_t i = 0; i < old.size(); ++i)
    HYLE_RETURN_IF_ERROR(require(old[i] == (i < 64 ? bf16(.375f * pages.scale) : 0x7fc1),
        "metadata append changed the retained old partial page"));
  return ready;
}

Status metadata_mapping_journey(const c::Context &context,
    const c::StorageOwner &memory, m::MemoryResourceRef host_workspace, c::Queue &queue) {
  HYLE_ASSIGN_OR_RETURN(auto graph, metadata_graph(context, memory));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  HYLE_ASSIGN_OR_RETURN(auto pages, metadata_pages(graph, memory, 1));
  HYLE_ASSIGN_OR_RETURN(auto run, metadata_run(graph, memory, 1));
  HYLE_ASSIGN_OR_RETURN(auto first, metadata_layout(graph.schema, true, 2, 3));
  HYLE_RETURN_IF_ERROR(prepare_metadata_episode(graph, run, pages, host_workspace,
      queues, first, 2, false));
  HYLE_ASSIGN_OR_RETURN(auto retained_first,
      finish_metadata_episode(graph, run, pages, queues, first));
  HYLE_ASSIGN_OR_RETURN(auto second, metadata_layout(graph.schema, true, 4, 5));
  HYLE_RETURN_IF_ERROR(prepare_metadata_episode(graph, run, pages, host_workspace,
      queues, second, 4, true));
  HYLE_ASSIGN_OR_RETURN(auto retained_second,
      finish_metadata_episode(graph, run, pages, queues, second));
  HYLE_RETURN_IF_ERROR(require(retained_first[0].layout() == first &&
      retained_first[1].layout() == first && retained_second[0].layout() == second,
      "metadata replay replaced the retained prior ready frame"));
  HYLE_RETURN_IF_ERROR(require(!pages.keys.acquire(2) && !pages.values.acquire(2),
      "metadata replay released the retained first ready page"));
  std::puts("CUDA append dynamic target IDs keep capture and both old/new ready frames PASS");
  // Return the earlier targets only after explicitly dropping their snapshots.
  // The same Invocation must now split its previously shared K/V mirror: K's
  // old tail remains page1, while V reads a new original partial page4.
  retained_first = {};
  retained_second = {};
  HYLE_ASSIGN_OR_RETURN(auto alternate, pages.values.acquire(4));
  HYLE_ASSIGN_OR_RETURN(auto alternate_storage, alternate.lease().view());
  std::array<std::uint16_t, 128> alternate_words;
  std::fill_n(alternate_words.begin(), 64, bf16(1.375f));
  std::fill(alternate_words.begin() + 64, alternate_words.end(), std::uint16_t{0x7fc1});
  HYLE_RETURN_IF_ERROR(alternate_storage.write(std::as_bytes(std::span(alternate_words))));
  HYLE_ASSIGN_OR_RETURN(auto alternate_ready, alternate.publish(1));
  HYLE_ASSIGN_OR_RETURN(auto alternate_layout, metadata_layout(graph.schema, false, 4));
  const std::array alternate_guards{pages.old_values[0], alternate_ready};
  HYLE_ASSIGN_OR_RETURN(pages.previous_values, pages.values.read(alternate_layout, alternate_guards));
  pages.tail_value = 1.375f;
  // Splitting the group creates a real new descriptor carrier, so fresh
  // recording is required; this is distinct from stable-carrier content reuse.
  HYLE_RETURN_IF_ERROR(prepare_metadata_episode(graph, run, pages, host_workspace,
      queues, first, 2, false));
  HYLE_ASSIGN_OR_RETURN(auto split_ready,
      finish_metadata_episode(graph, run, pages, queues, first));
  HYLE_RETURN_IF_ERROR(require(split_ready[0].valid() && split_ready[1].valid(),
      "historically shared append mirror could not split its source mappings"));
  std::puts("CUDA append historical shared metadata splits for distinct old-tail sources PASS");
  {
    HYLE_ASSIGN_OR_RETURN(auto different_pages, metadata_pages(graph, memory, 1, true));
    HYLE_ASSIGN_OR_RETURN(auto different_run, metadata_run(graph, memory, 1));
    HYLE_RETURN_IF_ERROR(prepare_metadata_episode(graph, different_run, different_pages,
        host_workspace, queues, first, 2, false));
    HYLE_ASSIGN_OR_RETURN(auto ready,
        finish_metadata_episode(graph, different_run, different_pages, queues, first));
    HYLE_RETURN_IF_ERROR(require(ready[0].valid() && ready[1].valid(),
        "different old-tail mappings did not publish their actual frames"));
  }
  std::puts("CUDA append same target layout distinguishes K/V old-tail source mappings PASS");
  return Status::Ok();
}

Status metadata_slot_journey(const c::Context &context,
    const c::StorageOwner &memory, m::MemoryResourceRef host_workspace, c::Queue &queue) {
  HYLE_ASSIGN_OR_RETURN(auto graph, metadata_graph(context, memory));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  HYLE_ASSIGN_OR_RETURN(auto pages_a, metadata_pages(graph, memory, 1));
  HYLE_ASSIGN_OR_RETURN(auto pages_b, metadata_pages(graph, memory, 2));
  HYLE_ASSIGN_OR_RETURN(auto run_a, metadata_run(graph, memory, 1));
  HYLE_ASSIGN_OR_RETURN(auto run_b, metadata_run(graph, memory, 2));
  HYLE_ASSIGN_OR_RETURN(auto layout_a, metadata_layout(graph.schema, true, 2, 3));
  HYLE_ASSIGN_OR_RETURN(auto layout_b, metadata_layout(graph.schema, true, 4, 5));
  HYLE_RETURN_IF_ERROR(prepare_metadata_episode(graph, run_a, pages_a, host_workspace,
      queues, layout_a, 2, false));
  HYLE_RETURN_IF_ERROR(prepare_metadata_episode(graph, run_b, pages_b, host_workspace,
      queues, layout_b, 4, false));
  // Both metadata uploads/captures precede either real submit. A global or
  // Executable-shared descriptor mirror would now make at least A incorrect.
  HYLE_ASSIGN_OR_RETURN(auto ready_a,
      finish_metadata_episode(graph, run_a, pages_a, queues, layout_a));
  HYLE_ASSIGN_OR_RETURN(auto ready_b,
      finish_metadata_episode(graph, run_b, pages_b, queues, layout_b));
  HYLE_RETURN_IF_ERROR(require(ready_a[0].layout() == layout_a &&
      ready_b[0].layout() == layout_b, "append metadata leaked between Invocations"));
  std::puts("CUDA append two prepared Invocations retain independent device metadata PASS");
  return Status::Ok();
}

Status exclusive_journey(const c::Context &context, const c::StorageOwner &memory,
    m::MemoryResourceRef host_workspace, c::Queue &queue) {
  HYLE_ASSIGN_OR_RETURN(auto graph, metadata_graph(context, memory, true));
  HYLE_ASSIGN_OR_RETURN(auto run, metadata_run(graph, memory, 1));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const auto layout_for = [&](std::int32_t page, std::int32_t rows) -> Result<c::BatchLayoutVersion> {
    const std::array<std::int32_t, 2> q{0, 1}, ptr{0, 1};
    const std::array ids{page}, tail{rows};
    return graph.schema.version({q, ptr, ids, tail});
  };
  HYLE_ASSIGN_OR_RETURN(auto initial, layout_for(0, 1));
  HYLE_ASSIGN_OR_RETURN(auto first, layout_for(2, 2));
  HYLE_ASSIGN_OR_RETURN(auto second, layout_for(2, 3));
  HYLE_ASSIGN_OR_RETURN(auto branch, layout_for(3, 4));
  HYLE_ASSIGN_OR_RETURN(auto keys, c::PagedArena::create(graph.schema,
      c::BatchCacheRole::kKey, memory.ref(), {.max_backing_bytes = 4096}));
  HYLE_ASSIGN_OR_RETURN(auto values, c::PagedArena::create(graph.schema,
      c::BatchCacheRole::kValue, memory.ref(), {.max_backing_bytes = 4096}));
  HYLE_ASSIGN_OR_RETURN(auto k0, keys.acquire(0));
  HYLE_ASSIGN_OR_RETURN(auto v0, values.acquire(0));
  std::array<std::uint16_t, 256> zeros{}, old_words;
  old_words.fill(0x7fc1); std::fill_n(old_words.begin(), 64, bf16(.125f));
  {
    HYLE_ASSIGN_OR_RETURN(auto ks, k0.lease().view());
    HYLE_ASSIGN_OR_RETURN(auto vs, v0.lease().view());
    HYLE_RETURN_IF_ERROR(ks.write(std::as_bytes(std::span(zeros))));
    HYLE_RETURN_IF_ERROR(vs.write(std::as_bytes(std::span(old_words))));
  }
  HYLE_ASSIGN_OR_RETURN(auto original_k, k0.publish_appendable(1));
  HYLE_ASSIGN_OR_RETURN(auto original_v, v0.publish_appendable(1));
  HYLE_RETURN_IF_ERROR(require(original_k.prefix.size() == 128 && original_v.prefix.size() == 128,
      "initial appendable reader must expose exactly one committed row"));
  HYLE_ASSIGN_OR_RETURN(auto previous_k, keys.read(initial, std::span(&original_k.prefix, 1)));
  HYLE_ASSIGN_OR_RETURN(auto previous_v, values.read(initial, std::span(&original_v.prefix, 1)));
  const auto episode = [&](c::BatchLayoutVersion layout, gb::PagedAppendPages k,
      gb::PagedAppendPages v, float expected, bool reuse) -> Result<std::array<gb::PagedAppendPages, 2>> {
    HYLE_RETURN_IF_ERROR(gb::prepare(run.invocation, queues,
        std::array{gb::NodeLayout{graph.reader, layout}}, {.host_workspace = host_workspace}));
    HYLE_RETURN_IF_ERROR(gb::bind_append(run.invocation, graph.graph, graph.key_writer,
        graph.key_bank, std::move(k)));
    HYLE_RETURN_IF_ERROR(gb::bind_append(run.invocation, graph.graph, graph.value_writer,
        graph.value_bank, std::move(v)));
    HYLE_RETURN_IF_ERROR(run.invocation.prepare_replay(queues, {.mode = g::ReplayMode::kRequireCapture}));
    HYLE_ASSIGN_OR_RETURN(auto report, run.invocation.replay_report());
    const auto correct_capture = report.route == g::ReplayRoute::kCaptured && (reuse
        ? report.recorded_capture_range_count == 0 && report.reused_capture_range_count > 0
        : report.recorded_capture_range_count > 0 && report.reused_capture_range_count == 0);
    if (!correct_capture) std::fprintf(stderr,
        "exclusive capture reuse=%u route=%u recorded=%zu reused=%zu\n", unsigned(reuse),
        unsigned(report.route), report.recorded_capture_range_count, report.reused_capture_range_count);
    HYLE_RETURN_IF_ERROR(require(correct_capture, "exclusive metadata capture did not preserve stable physical carriers"));
    HYLE_ASSIGN_OR_RETURN(auto done, run.invocation.submit(queues));
    HYLE_RETURN_IF_ERROR(done.wait()); done = {};
    {
      HYLE_ASSIGN_OR_RETURN(auto result, run.invocation.result(graph.output));
      HYLE_ASSIGN_OR_RETURN(auto lease, result.storage_lease());
      HYLE_ASSIGN_OR_RETURN(auto storage, lease.view());
      std::array<std::uint16_t, 128> actual{};
      HYLE_RETURN_IF_ERROR(storage.read(std::as_writable_bytes(std::span(actual))));
      for (const auto value : actual)
        HYLE_RETURN_IF_ERROR(require(value == bf16(expected), "exclusive append attention read the wrong committed prefix"));
    }
    std::array<gb::PagedAppendPages, 2> owners;
    HYLE_ASSIGN_OR_RETURN(owners[0], gb::take_append(run.invocation, graph.key_writer));
    HYLE_ASSIGN_OR_RETURN(owners[1], gb::take_append(run.invocation, graph.value_writer));
    auto stale = run.invocation.submit(queues);
    if (stale) { stale->wait().IgnoreError(); return std::unexpected(Status::Internal("taken exclusive append still allowed submit")); }
    HYLE_RETURN_IF_ERROR(require(stale.error().IsFailedPrecondition(), "exclusive take did not revoke preparation"));
    return owners;
  };
  std::array<c::PagedPageWriter, 1> fresh_k, fresh_v;
  HYLE_ASSIGN_OR_RETURN(fresh_k[0], keys.acquire(2));
  HYLE_ASSIGN_OR_RETURN(fresh_v[0], values.acquire(2));
  HYLE_ASSIGN_OR_RETURN(auto ak, gb::PagedAppendPages::create(keys, first, previous_k,
      fresh_k, std::span<c::PagedPageAppender>{}));
  HYLE_ASSIGN_OR_RETURN(auto av, gb::PagedAppendPages::create(values, first, previous_v,
      fresh_v, std::span<c::PagedPageAppender>{}));
  HYLE_ASSIGN_OR_RETURN(auto first_owner, episode(first, std::move(ak), std::move(av), .3125f, false));
  HYLE_ASSIGN_OR_RETURN(auto first_k, first_owner[0].ready());
  HYLE_ASSIGN_OR_RETURN(auto first_v, first_owner[1].ready());
  HYLE_RETURN_IF_ERROR(require(first_owner[0].appender_count() == 1 && first_owner[1].appender_count() == 1,
      "fresh partial publication did not return its unique idle tail"));
  std::array<c::PagedPageAppender, 1> tails_k, tails_v;
  HYLE_ASSIGN_OR_RETURN(auto &idle_k, first_owner[0].appender(0));
  HYLE_ASSIGN_OR_RETURN(auto &idle_v, first_owner[1].appender(0));
  tails_k[0] = std::move(idle_k); tails_v[0] = std::move(idle_v);
  HYLE_RETURN_IF_ERROR(tails_k[0].reserve_rows(1));
  HYLE_RETURN_IF_ERROR(tails_v[0].reserve_rows(1));
  HYLE_ASSIGN_OR_RETURN(ak, gb::PagedAppendPages::create(keys, second, first_k,
      std::span<c::PagedPageWriter>{}, tails_k));
  HYLE_ASSIGN_OR_RETURN(av, gb::PagedAppendPages::create(values, second, first_v,
      std::span<c::PagedPageWriter>{}, tails_v));
  HYLE_ASSIGN_OR_RETURN(auto second_owner, episode(second, std::move(ak), std::move(av), .375f, true));
  HYLE_ASSIGN_OR_RETURN(auto second_k, second_owner[0].ready());
  HYLE_ASSIGN_OR_RETURN(auto second_v, second_owner[1].ready());
  HYLE_ASSIGN_OR_RETURN(auto &same_k, second_owner[0].appender(0));
  HYLE_ASSIGN_OR_RETURN(auto &same_v, second_owner[1].appender(0));
  HYLE_RETURN_IF_ERROR(require(same_k.page() == 2 && same_v.page() == 2 &&
      same_k.initialized_rows() == 3 && same_v.initialized_rows() == 3,
      "exclusive continuation changed its physical tail or committed frontier"));
  // The branch receives no tail authority, so the original COW path is used.
  HYLE_ASSIGN_OR_RETURN(fresh_k[0], keys.acquire(3));
  HYLE_ASSIGN_OR_RETURN(fresh_v[0], values.acquire(3));
  HYLE_ASSIGN_OR_RETURN(ak, gb::PagedAppendPages::create(keys, branch, second_k,
      fresh_k, std::span<c::PagedPageAppender>{}));
  HYLE_ASSIGN_OR_RETURN(av, gb::PagedAppendPages::create(values, branch, second_v,
      fresh_v, std::span<c::PagedPageAppender>{}));
  HYLE_ASSIGN_OR_RETURN(auto branched, episode(branch, std::move(ak), std::move(av), .40625f, true));
  HYLE_ASSIGN_OR_RETURN(auto branch_k, branched[0].ready());
  HYLE_ASSIGN_OR_RETURN(auto branch_v, branched[1].ready());
  HYLE_RETURN_IF_ERROR(require(branch_k.layout() == branch && branch_v.layout() == branch &&
      !branched[0].appender_count() && !branched[1].appender_count(), "full COW branch returned an invalid frontier"));
  HYLE_RETURN_IF_ERROR(require(first_k.layout() == first && first_v.layout() == first &&
      second_k.layout() == second && second_v.layout() == second,
      "same-page continuation widened an older immutable frame"));
  // Observe the retained two-row frame through an ordinary ready reader after
  // the physical page has three rows, rather than inspecting private guards.
  HYLE_ASSIGN_OR_RETURN(auto b, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto q, b.input(graph.query_description));
  HYLE_ASSIGN_OR_RETURN(auto kd, c::batch_cache_description(graph.schema, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto vd, c::batch_cache_description(graph.schema, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto k, b.input(kd)); HYLE_ASSIGN_OR_RETURN(auto v, b.input(vd));
  HYLE_ASSIGN_OR_RETURN(auto op, gb::paged_attention(first, {.scale = .125f, .causal = true}));
  HYLE_ASSIGN_OR_RETURN(auto node, b.add_node(c::Operation::custom(op), std::array{q, k, v}));
  HYLE_ASSIGN_OR_RETURN(auto output, b.result(node)); HYLE_RETURN_IF_ERROR(b.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto old_graph, std::move(b).finish());
  const std::array domains{g::StorageDomain{memory.ref()}};
  HYLE_ASSIGN_OR_RETURN(auto source, g::ExecutionSource::acquire(context));
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto old_program, gb::compile(old_graph, {.storage_domains = domains, .targets = targets}));
  HYLE_ASSIGN_OR_RETURN(auto old_run, old_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(gb::prepare(old_run, queues, std::array{gb::NodeLayout{node, first}}, {.host_workspace = host_workspace}));
  HYLE_RETURN_IF_ERROR(old_run.bind(q, graph.query_description, run.query));
  HYLE_RETURN_IF_ERROR(gb::bind(old_run, k, first_k)); HYLE_RETURN_IF_ERROR(gb::bind(old_run, v, first_v));
  HYLE_ASSIGN_OR_RETURN(auto destination, run.pool.allocate(256, 256));
  HYLE_RETURN_IF_ERROR(old_run.bind(output, graph.output_description, std::move(destination)));
  HYLE_RETURN_IF_ERROR(old_run.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto done, old_run.submit(queues)); HYLE_RETURN_IF_ERROR(done.wait()); done = {};
  {
    HYLE_ASSIGN_OR_RETURN(auto result, old_run.result(output)); HYLE_ASSIGN_OR_RETURN(auto lease, result.storage_lease());
    HYLE_ASSIGN_OR_RETURN(auto storage, lease.view()); std::array<std::uint16_t, 128> actual{};
    HYLE_RETURN_IF_ERROR(storage.read(std::as_writable_bytes(std::span(actual))));
    for (const auto value : actual) HYLE_RETURN_IF_ERROR(require(value == bf16(.3125f),
        "retained bounded frame read the new suffix or damaged old prefix"));
  }
  HYLE_ASSIGN_OR_RETURN(auto old_storage, original_v.prefix.view());
  std::array<std::uint16_t, 64> old_actual{};
  HYLE_RETURN_IF_ERROR(old_storage.read(std::as_writable_bytes(std::span(old_actual))));
  for (const auto value : old_actual) HYLE_RETURN_IF_ERROR(require(value == bf16(.125f), "old committed prefix bytes changed"));
  std::puts("CUDA exclusive same-physical-tail/captured metadata reuse/bounded old snapshot/no-tail COW BF16 PASS");
  return Status::Ok();
}

Status all(int profile = 0) {
  HYLE_ASSIGN_OR_RETURN(auto context, g::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto device,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto memory, c::StorageOwner::adopt(context, std::move(device)));
  HYLE_ASSIGN_OR_RETURN(auto staging,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  struct Drain {
    c::Context &context;
    ~Drain() { context.drain().IgnoreError(); }
  } drain{context};
  if (profile == 0) {
    HYLE_RETURN_IF_ERROR(journey(context, memory, staging.ref(), queue, true));
    return journey(context, memory, staging.ref(), queue, false);
  }
  if (profile == 1)
    return metadata_mapping_journey(context, memory, staging.ref(), queue);
  if (profile == 2)
    return metadata_slot_journey(context, memory, staging.ref(), queue);
  if (profile == 3)
    return exclusive_journey(context, memory, staging.ref(), queue);
  return journey(context, memory, staging.ref(), queue, false, true);
}
} // namespace

#if defined(HYLE_APPEND_STANDALONE)
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  for (int profile = 0; profile < 5; ++profile) {
    const auto result = all(profile);
    if (!result.ok()) {
      char text[1024]{};
      result.ToString(text, sizeof(text));
      std::fprintf(stderr, "%s\n", text);
      return 1;
    }
  }
  return 0;
}
#else
TEST(CudaPagedAppend, OriginalDevicePagesAndSameGraphProof) {
  const auto result = all();
  EXPECT_TRUE(result.ok()) << result.message();
}
TEST(CudaPagedAppend, CapturedMetadataChangesPageMappings) {
  const auto result = all(1);
  EXPECT_TRUE(result.ok()) << result.message();
}
TEST(CudaPagedAppend, InvocationMetadataStaysIndependent) {
  const auto result = all(2);
  EXPECT_TRUE(result.ok()) << result.message();
}
TEST(CudaPagedAppend, ExclusiveTailKeepsBoundedSnapshotsAndCapture) {
  const auto result = all(3);
  EXPECT_TRUE(result.ok()) << result.message();
}
TEST(CudaPagedAppend, TwoRequestFramesComposeAppendAndProject) {
  const auto result = all(4);
  EXPECT_TRUE(result.ok()) << result.message();
}
#endif
