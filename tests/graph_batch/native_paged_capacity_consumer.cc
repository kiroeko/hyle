#include <hyle/graph/batch.h>
#if defined(HYLE_JOINT_PAGE_APPLE)
#include <hyle/graph/apple.h>
#else
#include <hyle/graph/vulkan.h>
#endif
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/sched/blocking.h>
#include <hyle/sched/scheduler.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
#if defined(HYLE_JOINT_PAGE_APPLE)
namespace n = c::apple;
namespace route = g::apple;
#else
namespace n = c::vulkan;
namespace route = g::vulkan;
#endif
using hyle::Result;
using hyle::Status;
constexpr g::ExecutionTargetId target{0};
constexpr g::StorageDomainId device_storage{0};
constexpr std::size_t depth = 256, heads = 2, page_rows = 16;
constexpr std::size_t page_bytes = page_rows * depth * sizeof(std::uint16_t);
constexpr std::uint16_t nan_word = 0x7fc1;
constexpr std::array<std::uint16_t, 5> value_words{0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0};

Status require(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}
struct Drain final {
  hyle::sched::Scheduler &scheduler;
  ~Drain() { scheduler.drain(); }
};
float number(std::uint16_t word) {
  return std::bit_cast<float>(std::uint32_t(word) << 16);
}

// Only fixture I/O. Original native CopyBatch owns every transfer/admission;
// unbind_mutable returns an inspected page writer after actual retirement.
struct Transfers final {
  c::LeasePool host;
  n::CopyBatchInvocation copy;
  static Result<Transfers> create(n::Context &context, n::MemoryResource &device,
                                  n::MemoryResource &staging) {
    Transfers result;
    HYLE_ASSIGN_OR_RETURN(result.host, c::LeasePool::create(staging.ref(), page_bytes * 4 + 256,
        {.max_blocks = 4}));
    const std::array specs{c::CopyResourceSpec{device.ref(), page_bytes},
                           c::CopyResourceSpec{staging.ref(), page_bytes}};
    HYLE_ASSIGN_OR_RETURN(auto program, n::CopyBatchProgram::prepare(context, specs));
    HYLE_ASSIGN_OR_RETURN(result.copy, n::CopyBatchInvocation::create(program));
    return result;
  }
  Status prepare(c::CopyResourceId source, c::CopyResourceId destination, std::size_t bytes) {
    const std::array<c::CopyPhase, 1> phases{{{}}};
    const std::array<c::CopyRange, 1> ranges{{{source, destination, {0}, 0, 0, bytes}}};
    return copy.prepare_replay(phases, ranges);
  }
  Status upload(c::WriteLease &destination, std::span<const std::byte> bytes) {
    HYLE_RETURN_IF_ERROR(copy.reset());
    HYLE_ASSIGN_OR_RETURN(auto writer, host.allocate(bytes.size(), 64));
    HYLE_ASSIGN_OR_RETURN(auto storage, writer.view());
    HYLE_RETURN_IF_ERROR(storage.write(bytes));
    HYLE_ASSIGN_OR_RETURN(auto source, writer.publish());
    HYLE_RETURN_IF_ERROR(copy.bind_source({1}, source));
    HYLE_RETURN_IF_ERROR(copy.bind_mutable({0}, std::move(destination)));
    HYLE_RETURN_IF_ERROR(prepare({1}, {0}, bytes.size()));
    {
      HYLE_ASSIGN_OR_RETURN(auto done, copy.submit());
      HYLE_RETURN_IF_ERROR(done.wait());
    }
    HYLE_ASSIGN_OR_RETURN(destination, copy.unbind_mutable({0}));
    return copy.reset();
  }
  Result<std::vector<std::uint16_t>> read(const c::ReadLease &source, std::size_t bytes) {
    HYLE_RETURN_IF_ERROR(copy.reset());
    HYLE_RETURN_IF_ERROR(copy.bind_source({0}, source));
    return read_bound(bytes, nullptr);
  }
  Result<std::vector<std::uint16_t>> read(c::WriteLease &source, std::size_t bytes) {
    HYLE_RETURN_IF_ERROR(copy.reset());
    HYLE_RETURN_IF_ERROR(copy.bind_mutable({0}, std::move(source)));
    return read_bound(bytes, &source);
  }
  Result<std::vector<std::uint16_t>> read_bound(std::size_t bytes, c::WriteLease *original) {
    HYLE_ASSIGN_OR_RETURN(auto writer, host.allocate(bytes, 64));
    HYLE_RETURN_IF_ERROR(copy.bind_mutable({1}, std::move(writer)));
    HYLE_RETURN_IF_ERROR(prepare({0}, {1}, bytes));
    c::ReadLease ready;
    {
      HYLE_ASSIGN_OR_RETURN(auto done, copy.submit());
      HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_ASSIGN_OR_RETURN(ready, done.output({0}, {1}, 0, bytes));
    }
    if (original) { HYLE_ASSIGN_OR_RETURN(*original, copy.unbind_mutable({0})); }
    std::vector<std::uint16_t> result(bytes / sizeof(std::uint16_t));
    HYLE_ASSIGN_OR_RETURN(auto storage, ready.view());
    HYLE_RETURN_IF_ERROR(storage.read(std::as_writable_bytes(std::span(result))));
    HYLE_RETURN_IF_ERROR(copy.reset());
    return result;
  }
};

struct Ports final {
  g::ValueId incoming_k, incoming_v, q, key_bank, value_bank, metadata, output;
  g::NodeId cast_k, cast_v, kw, vw, reader;
};

Status exercise(n::Context &context, n::MemoryResource &device, n::MemoryResource &staging) {
  HYLE_ASSIGN_OR_RETURN(auto plain, c::CacheStorage::plain(c::DataType::kBFloat16));
  const c::BatchCapacity cap{.query_type = c::DataType::kBFloat16,
      .output_type = c::DataType::kBFloat16, .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = 2, .max_kv_rows = 16, .max_pages = 2,
      .max_page_refs = 1, .query_heads = 2, .kv_heads = 1,
      .qk_head_dim = 256, .value_head_dim = 256, .page_size = 16};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto kdesc, c::batch_cache_description(schema, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto vdesc, c::batch_cache_description(schema, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto metadata, schema.metadata_description());
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto physical_rows, builder.dimension({1, 2, 1}));
  HYLE_ASSIGN_OR_RETURN(auto active_rows, builder.dimension({1, 2, 1}));
  HYLE_ASSIGN_OR_RETURN(auto qspec, g::TensorSpec::dense(c::DataType::kBFloat16,
      std::array{g::Dimension(physical_rows), g::Dimension::fixed(2), g::Dimension::fixed(256)}));
  HYLE_ASSIGN_OR_RETURN(auto source_spec, g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(active_rows), g::Dimension::fixed(1), g::Dimension::fixed(256)}));
  HYLE_ASSIGN_OR_RETURN(auto rows_spec, g::TensorSpec::dense(c::DataType::kBFloat16,
      std::array{g::Dimension(active_rows), g::Dimension::fixed(1), g::Dimension::fixed(256)}));
  Ports ports;
  HYLE_ASSIGN_OR_RETURN(ports.incoming_k, builder.input(source_spec));
  HYLE_ASSIGN_OR_RETURN(ports.incoming_v, builder.input(source_spec));
  HYLE_ASSIGN_OR_RETURN(ports.cast_k, builder.add_node(c::Operation(c::Cast{c::DataType::kBFloat16}),
      std::array{ports.incoming_k}, std::array{rows_spec}));
  HYLE_ASSIGN_OR_RETURN(ports.cast_v, builder.add_node(c::Operation(c::Cast{c::DataType::kBFloat16}),
      std::array{ports.incoming_v}, std::array{rows_spec}));
  HYLE_ASSIGN_OR_RETURN(auto produced_k, builder.result(ports.cast_k));
  HYLE_ASSIGN_OR_RETURN(auto produced_v, builder.result(ports.cast_v));
  HYLE_ASSIGN_OR_RETURN(auto ka, gb::paged_append(schema, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto va, gb::paged_append(schema, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(ports.kw, builder.add_node(c::Operation::custom(ka), std::array{produced_k}, {}));
  HYLE_ASSIGN_OR_RETURN(ports.vw, builder.add_node(c::Operation::custom(va), std::array{produced_v}, {}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(ports.vw, ports.kw));
  HYLE_ASSIGN_OR_RETURN(ports.q, builder.input(qspec));
  HYLE_ASSIGN_OR_RETURN(ports.key_bank, builder.input(kdesc));
  HYLE_ASSIGN_OR_RETURN(ports.value_bank, builder.input(vdesc));
  HYLE_ASSIGN_OR_RETURN(ports.metadata, builder.input(metadata));
  HYLE_ASSIGN_OR_RETURN(ports.reader, builder.add_node(c::Operation::custom(c::PagedAttention{
      .capacity = cap, .scale = .0625F, .causal = true}),
      std::array{ports.q, ports.key_bank, ports.value_bank, ports.metadata}, std::array{qspec}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(ports.reader, ports.kw));
  HYLE_RETURN_IF_ERROR(builder.depends_on(ports.reader, ports.vw));
  HYLE_ASSIGN_OR_RETURN(ports.output, builder.result(ports.reader));
  HYLE_RETURN_IF_ERROR(builder.publish(ports.output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());

  const std::array one{g::ShapeBinding{physical_rows, 1}, g::ShapeBinding{active_rows, 1}};
  const std::array two{g::ShapeBinding{physical_rows, 2}, g::ShapeBinding{active_rows, 2}};
  const std::array inactive{g::ShapeBinding{physical_rows, 2}, g::ShapeBinding{active_rows, 1}};
  const std::array shapes{one, two, inactive};
  const std::array variants{g::ExactShapeVariant{one}, g::ExactShapeVariant{two},
                             g::ExactShapeVariant{inactive}};
  const std::array<std::uint32_t, 3> physical{1, 2, 2}, active{1, 2, 1};
  HYLE_ASSIGN_OR_RETURN(auto source, route::execution_source(context));
  const std::array domains{g::StorageDomain{device.ref()}, g::StorageDomain{staging.ref()}};
  const std::array targets{g::ExecutionTarget{source, device_storage}};
  // One logical DAG and one finite compilation. Runtime NodeLayout carries
  // the actual request rows/history; no Graph is rebuilt for a different T.
  HYLE_ASSIGN_OR_RETURN(auto compiled, gb::compile(graph, {.storage_domains = domains, .targets = targets},
      {.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants}));
  HYLE_RETURN_IF_ERROR(require(compiled.executable.exact_variant_count() == variants.size(),
      "capacity finite family lost a declared child"));
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 32);
  hyle::sched::BlockingPool waiters(hyle::mem::system_allocator(), 2);
  Drain drain{scheduler};
  const std::span<const g::QueueBinding> queues;
  HYLE_ASSIGN_OR_RETURN(auto transfers, Transfers::create(context, device, staging));

  for (std::size_t branch = 0; branch < shapes.size(); ++branch) {
    const auto rows = active[branch], extent = physical[branch];
    HYLE_ASSIGN_OR_RETURN(auto report, compiled.report.exact_variant(branch));
    std::size_t writers = 0, producers = 0, readers = 0;
    for (std::size_t index = 0; index < report.step_count(); ++index) {
      HYLE_ASSIGN_OR_RETURN(auto step, report.step(index));
      for (std::size_t node = 0; node < step.node_count; ++node) {
        HYLE_ASSIGN_OR_RETURN(auto logical, report.step_node(index, node));
        if (logical != ports.kw && logical != ports.vw && logical != ports.cast_k &&
            logical != ports.cast_v && logical != ports.reader) continue;
        HYLE_RETURN_IF_ERROR(require(step.target == target && step.kind == g::CompileStepKind::kProviderRegion &&
            step.runtime.mode == c::ProviderRuntimeMode::kDelegated,
            "capacity Cast/append/attention escaped the actual native Runtime"));
        if (logical == ports.kw || logical == ports.vw) {
          HYLE_RETURN_IF_ERROR(require(step.node_count == 1, "capacity append is not its exact singleton"));
          ++writers;
        } else if (logical == ports.reader) ++readers;
        else ++producers;
      }
    }
    HYLE_RETURN_IF_ERROR(require(writers == 2 && producers == 2 && readers == 1,
        "capacity native selected node coverage differs"));
    for (std::size_t index = 0; index < report.lane_count(); ++index) {
      HYLE_ASSIGN_OR_RETURN(auto lane, report.lane(index));
      HYLE_RETURN_IF_ERROR(require(!lane.queue_required, "capacity reader invented a caller Compute Queue"));
    }
    const std::array<std::int32_t, 2> old_query{0, 1}, query{0, static_cast<std::int32_t>(rows)}, pointer{0, 1};
    const std::array<std::int32_t, 1> old_page{0}, new_page{1}, old_tail{3}, tail{3 + static_cast<std::int32_t>(rows)};
    HYLE_ASSIGN_OR_RETURN(auto previous, schema.version({old_query, pointer, old_page, old_tail}));
    HYLE_ASSIGN_OR_RETURN(auto layout, schema.version({query, pointer, new_page, tail}));
    HYLE_ASSIGN_OR_RETURN(auto keys, c::PagedArena::create(schema, c::BatchCacheRole::kKey,
        device.ref(), {.max_backing_bytes = page_bytes * 2 + 1}));
    HYLE_ASSIGN_OR_RETURN(auto values, c::PagedArena::create(schema, c::BatchCacheRole::kValue,
        device.ref(), {.max_backing_bytes = page_bytes * 2 + 1}));
    std::array<std::uint16_t, page_rows * depth> old_keys, old_values, dirty;
    old_keys.fill(nan_word); old_values.fill(nan_word); dirty.fill(nan_word);
    std::fill_n(old_keys.begin(), 3 * depth, std::uint16_t{0});
    for (std::size_t row = 0; row < 3; ++row)
      std::fill_n(old_values.begin() + static_cast<std::ptrdiff_t>(row * depth), depth, value_words[row]);
    HYLE_ASSIGN_OR_RETURN(auto original_k, keys.acquire(0));
    HYLE_ASSIGN_OR_RETURN(auto original_v, values.acquire(0));
    HYLE_RETURN_IF_ERROR(transfers.upload(original_k.lease(), std::as_bytes(std::span(old_keys))));
    HYLE_RETURN_IF_ERROR(transfers.upload(original_v.lease(), std::as_bytes(std::span(old_values))));
    HYLE_ASSIGN_OR_RETURN(auto old_k, original_k.publish(3));
    HYLE_ASSIGN_OR_RETURN(auto old_v, original_v.publish(3));
    HYLE_ASSIGN_OR_RETURN(auto before_k, keys.read(previous, std::span(&old_k, 1)));
    HYLE_ASSIGN_OR_RETURN(auto before_v, values.read(previous, std::span(&old_v, 1)));
    std::array<c::PagedPageWriter, 1> dirty_k, dirty_v;
    HYLE_ASSIGN_OR_RETURN(dirty_k[0], keys.acquire(1));
    HYLE_ASSIGN_OR_RETURN(dirty_v[0], values.acquire(1));
    HYLE_RETURN_IF_ERROR(transfers.upload(dirty_k[0].lease(), std::as_bytes(std::span(dirty))));
    HYLE_RETURN_IF_ERROR(transfers.upload(dirty_v[0].lease(), std::as_bytes(std::span(dirty))));
    HYLE_ASSIGN_OR_RETURN(auto append_k, gb::PagedAppendPages::create(keys, layout, before_k, dirty_k));
    HYLE_ASSIGN_OR_RETURN(auto append_v, gb::PagedAppendPages::create(values, layout, before_v, dirty_v));
    HYLE_ASSIGN_OR_RETURN(auto run, compiled.executable.create_invocation({.shape = shapes[branch],
        .resources = g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(require(run.exact_variant_index() == branch, "capacity selected a different finite child"));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(run, scheduler, waiters));
    const auto unprepared = gb::bind_append(run, graph, ports.kw, ports.key_bank, std::move(append_k));
    HYLE_RETURN_IF_ERROR(require(unprepared.IsFailedPrecondition() && append_k.valid(),
        "capacity append without typed preparation accepted or consumed original pages"));
    HYLE_RETURN_IF_ERROR(gb::prepare(run, queues, std::array{gb::NodeLayout{ports.reader, layout}}));
    HYLE_RETURN_IF_ERROR(gb::bind_append(run, graph, ports.kw, ports.key_bank, std::move(append_k)));
    HYLE_RETURN_IF_ERROR(gb::bind_append(run, graph, ports.vw, ports.value_bank, std::move(append_v)));
    HYLE_RETURN_IF_ERROR(require(!append_k.valid() && !append_v.valid(), "capacity bind did not move its original page owners"));

    HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(device.ref(), page_bytes + 256, {.max_blocks = 3}));
    const auto input = [&](g::ValueId port, const c::TensorDesc &description,
                           std::span<const std::byte> bytes) -> Status {
      HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(bytes.size(), 64));
      HYLE_RETURN_IF_ERROR(transfers.upload(writer, bytes));
      HYLE_ASSIGN_OR_RETURN(auto ready, writer.publish());
      return run.bind(port, description, ready);
    };
    HYLE_ASSIGN_OR_RETURN(auto rowdesc, c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 3>{rows, 1, 256}));
    HYLE_ASSIGN_OR_RETURN(auto qdesc, c::batch_query_description(cap, extent));
    HYLE_ASSIGN_OR_RETURN(auto outdesc, c::batch_output_description(cap, extent));
    std::vector<float> new_keys(std::size_t(rows) * depth, 0.F), new_values(std::size_t(rows) * depth);
    for (std::size_t row = 0; row < rows; ++row)
      std::fill_n(new_values.begin() + static_cast<std::ptrdiff_t>(row * depth), depth, float(row + 4));
    std::vector<std::uint16_t> queries(std::size_t(extent) * heads * depth, nan_word);
    std::fill_n(queries.begin(), std::size_t(rows) * heads * depth, std::uint16_t{0});
    HYLE_RETURN_IF_ERROR(input(ports.incoming_k, rowdesc, std::as_bytes(std::span(new_keys))));
    HYLE_RETURN_IF_ERROR(input(ports.incoming_v, rowdesc, std::as_bytes(std::span(new_values))));
    HYLE_RETURN_IF_ERROR(input(ports.q, qdesc, std::as_bytes(std::span(queries))));
    // metadata is bound solely by the original typed prepare. The consumer
    // neither builds a page-ID mirror nor binds a second private table.
    HYLE_RETURN_IF_ERROR(run.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto done, run.submit(queues));
    HYLE_RETURN_IF_ERROR(done.wait());
    {
      HYLE_ASSIGN_OR_RETURN(auto result, run.result(ports.output));
      HYLE_ASSIGN_OR_RETURN(auto description, result.description());
      HYLE_RETURN_IF_ERROR(require(description == outdesc, "capacity output changed its concrete physical extent"));
      HYLE_ASSIGN_OR_RETURN(auto lease, result.storage_lease());
      HYLE_ASSIGN_OR_RETURN(auto actual, transfers.read(lease, std::size_t(extent) * heads * depth * 2));
      for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto row = index / (heads * depth);
        // K=Q=0 gives exact uniform weights over four then five visible rows.
        const std::uint16_t expected = row >= rows ? std::uint16_t{0} : row == 0 ? std::uint16_t{0x4020} : std::uint16_t{0x4040};
        HYLE_RETURN_IF_ERROR(require(std::isfinite(number(actual[index])) && actual[index] == expected,
            "capacity causal mean or bitwise-zero inactive output differs"));
      }
    }
    done = {};
    HYLE_RETURN_IF_ERROR(require(!run.reset().ok(), "capacity reset accepted untaken native append owners"));
    HYLE_ASSIGN_OR_RETURN(auto returned_k, gb::take_append(run, ports.kw));
    HYLE_ASSIGN_OR_RETURN(auto returned_v, gb::take_append(run, ports.vw));
    HYLE_ASSIGN_OR_RETURN(auto &key_writer, returned_k.page(0));
    HYLE_ASSIGN_OR_RETURN(auto &value_writer, returned_v.page(0));
    HYLE_ASSIGN_OR_RETURN(auto copied_k, transfers.read(key_writer.lease(), page_bytes));
    HYLE_ASSIGN_OR_RETURN(auto copied_v, transfers.read(value_writer.lease(), page_bytes));
    for (std::size_t index = 0; index < copied_k.size(); ++index) {
      const auto row = index / depth;
      const auto expected_k = row < 3 + rows ? std::uint16_t{0} : nan_word;
      const auto expected_v = row < 3 + rows ? value_words[row] : nan_word;
      HYLE_RETURN_IF_ERROR(require(copied_k[index] == expected_k && copied_v[index] == expected_v,
          "capacity COW/new rows modified a different row or the unused NaN suffix"));
    }
    HYLE_ASSIGN_OR_RETURN(auto next_k, returned_k.ready());
    HYLE_ASSIGN_OR_RETURN(auto next_v, returned_v.ready());
    HYLE_RETURN_IF_ERROR(require(next_k.layout() == layout && next_v.layout() == layout &&
        next_k.page_count() == 1 && next_v.page_count() == 1,
        "capacity dual take lost the whole-Graph success latch or exact ready projection"));
    auto stale = run.submit(queues);
    if (stale) { stale->wait().IgnoreError(); return Status::Internal("capacity taken pages admitted stale submission"); }
    HYLE_RETURN_IF_ERROR(require(stale.error().IsFailedPrecondition(), "capacity stale submit rejection changed"));
    HYLE_ASSIGN_OR_RETURN(auto unchanged_k, transfers.read(old_k, page_bytes));
    HYLE_ASSIGN_OR_RETURN(auto unchanged_v, transfers.read(old_v, page_bytes));
    HYLE_RETURN_IF_ERROR(require(std::equal(unchanged_k.begin(), unchanged_k.end(), old_keys.begin()) &&
        std::equal(unchanged_v.begin(), unchanged_v.end(), old_values.begin()) &&
        before_k.layout() == previous && before_v.layout() == previous &&
        before_k.page_count() == 1 && before_v.page_count() == 1,
        "capacity append changed a retained old frame, bytes or unused NaN tail"));
    HYLE_RETURN_IF_ERROR(require(!keys.acquire(0) && !values.acquire(1), "capacity old/new frames lost their exact page guards"));
    HYLE_RETURN_IF_ERROR(run.reset());
    HYLE_RETURN_IF_ERROR(run.close());
    std::printf("native capacity BF16 D256 R%u active%u finite-child%zu producer/append/reader COW/zero/permissions PASS\n",
        extent, rows, branch);
  }
  return Status::Ok();
}

Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, n::Context::create());
#if defined(HYLE_JOINT_PAGE_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(context));
#else
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(context, n::MemoryKind::kStaging));
#endif
  return exercise(context, device, staging);
}
} // namespace

int main() {
  const auto status = run();
  if (status.ok()) return 0;
  std::fprintf(stderr, "native paged capacity: %.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
