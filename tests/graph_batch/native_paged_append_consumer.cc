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
constexpr std::size_t depth = 64, page_rows = 16, page_bytes = page_rows * depth * 2;
constexpr std::uint16_t nan_word = 0x7e51;

Status require(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}
struct Drain final {
  hyle::sched::Scheduler &scheduler;
  ~Drain() { scheduler.drain(); }
};
float half(std::uint16_t word) {
  const auto exponent = unsigned(word >> 10) & 31U;
  if (exponent == 31U) return std::nanf("");
  const float sign = word & 0x8000U ? -1.F : 1.F;
  const auto fraction = unsigned(word) & 1023U;
  return sign * (exponent ? std::ldexp(1.F + float(fraction) / 1024.F, int(exponent) - 15)
                          : std::ldexp(float(fraction), -24));
}

// Test I/O only. All transfers use the original native CopyBatch/F1 and real
// completion. An inspected page writer is returned with unbind_mutable after
// retirement, without manufacturing a readonly page or consuming its publish.
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
    std::vector<std::uint16_t> result(bytes / 2);
    HYLE_ASSIGN_OR_RETURN(auto storage, ready.view());
    HYLE_RETURN_IF_ERROR(storage.read(std::as_writable_bytes(std::span(result))));
    HYLE_RETURN_IF_ERROR(copy.reset());
    return result;
  }
};

Status exercise(n::Context &context, n::MemoryResource &device, n::MemoryResource &staging,
                bool writer_only, bool missing_dependency) {
  HYLE_ASSIGN_OR_RETURN(auto plain, c::CacheStorage::plain(c::DataType::kFloat16));
  const c::BatchCapacity cap{.query_type = c::DataType::kFloat16,
      .output_type = c::DataType::kFloat16, .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = 2, .max_kv_rows = 16, .max_pages = 2,
      .max_page_refs = 1, .query_heads = 2, .kv_heads = 1,
      .qk_head_dim = 64, .value_head_dim = 64, .page_size = 16};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(cap));
  const std::array<std::int32_t, 2> old_query{0, 1}, query{0, 2}, pointer{0, 1};
  const std::array<std::int32_t, 1> old_page{0}, new_page{1}, old_tail{3}, new_tail{5};
  HYLE_ASSIGN_OR_RETURN(auto previous, schema.version({old_query, pointer, old_page, old_tail}));
  HYLE_ASSIGN_OR_RETURN(auto layout, schema.version({query, pointer, new_page, new_tail}));
  HYLE_ASSIGN_OR_RETURN(auto keys, c::PagedArena::create(schema, c::BatchCacheRole::kKey,
      device.ref(), {.max_backing_bytes = page_bytes * 2 + 1}));
  HYLE_ASSIGN_OR_RETURN(auto values, c::PagedArena::create(schema, c::BatchCacheRole::kValue,
      device.ref(), {.max_backing_bytes = page_bytes * 2 + 1}));
  HYLE_ASSIGN_OR_RETURN(auto transfers, Transfers::create(context, device, staging));
  std::array<std::uint16_t, page_rows * depth> old_keys, old_values;
  old_keys.fill(nan_word); old_values.fill(nan_word);
  std::fill_n(old_keys.begin(), 3 * depth, std::uint16_t{0});
  const std::array<std::uint16_t, 5> value_words{0x3c00, 0x4000, 0x4200, 0x4400, 0x4500};
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
  HYLE_ASSIGN_OR_RETURN(auto append_k, gb::PagedAppendPages::create(keys, layout, before_k, dirty_k));
  HYLE_ASSIGN_OR_RETURN(auto append_v, gb::PagedAppendPages::create(values, layout, before_v, dirty_v));
  HYLE_ASSIGN_OR_RETURN(auto rows, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 3>{2, 1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto qdesc, c::batch_query_description(cap, 2));
  HYLE_ASSIGN_OR_RETURN(auto kdesc, c::batch_cache_description(schema, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto vdesc, c::batch_cache_description(schema, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto incoming_k, builder.input(rows));
  HYLE_ASSIGN_OR_RETURN(auto incoming_v, builder.input(rows));
  HYLE_ASSIGN_OR_RETURN(auto cast_k, builder.add_node(c::Operation(c::Cast{c::DataType::kFloat16}), std::array{incoming_k}));
  HYLE_ASSIGN_OR_RETURN(auto cast_v, builder.add_node(c::Operation(c::Cast{c::DataType::kFloat16}), std::array{incoming_v}));
  HYLE_ASSIGN_OR_RETURN(auto produced_k, builder.result(cast_k));
  HYLE_ASSIGN_OR_RETURN(auto produced_v, builder.result(cast_v));
  HYLE_ASSIGN_OR_RETURN(auto ka, gb::paged_append(schema, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto va, gb::paged_append(schema, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto kw, builder.add_node(c::Operation::custom(ka), std::array{produced_k}));
  HYLE_ASSIGN_OR_RETURN(auto vw, builder.add_node(c::Operation::custom(va), std::array{produced_v}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(vw, kw));
  g::ValueId q, key_bank, value_bank, output;
  g::NodeId reader;
  if (!writer_only) {
    HYLE_ASSIGN_OR_RETURN(q, builder.input(qdesc));
    HYLE_ASSIGN_OR_RETURN(key_bank, builder.input(kdesc));
    HYLE_ASSIGN_OR_RETURN(value_bank, builder.input(vdesc));
    HYLE_ASSIGN_OR_RETURN(auto attention, gb::paged_attention(layout, {.scale = .125F, .causal = true}));
    HYLE_ASSIGN_OR_RETURN(reader, builder.add_node(c::Operation::custom(attention),
        std::array{q, key_bank, value_bank}));
    if (!missing_dependency) HYLE_RETURN_IF_ERROR(builder.depends_on(reader, kw));
    HYLE_RETURN_IF_ERROR(builder.depends_on(reader, vw));
    HYLE_ASSIGN_OR_RETURN(output, builder.result(reader));
    HYLE_RETURN_IF_ERROR(builder.publish(output));
  }
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto source, route::execution_source(context));
  const std::array domains{g::StorageDomain{device.ref()}, g::StorageDomain{staging.ref()}};
  const std::array targets{g::ExecutionTarget{source, device_storage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, gb::compile(graph, {.storage_domains = domains, .targets = targets}));
  std::size_t writer_steps = 0, producer_nodes = 0;
  for (std::size_t index = 0; index < compiled.report.step_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto step, compiled.report.step(index));
    for (std::size_t node = 0; node < step.node_count; ++node) {
      HYLE_ASSIGN_OR_RETURN(auto logical, compiled.report.step_node(index, node));
      if (logical != kw && logical != vw && logical != cast_k && logical != cast_v) continue;
      HYLE_RETURN_IF_ERROR(require(step.target == target && step.kind == g::CompileStepKind::kProviderRegion &&
          step.runtime.mode == c::ProviderRuntimeMode::kDelegated,
          "Cast or external write escaped the actual native Runtime"));
      if (logical == kw || logical == vw) {
        HYLE_RETURN_IF_ERROR(require(step.node_count == 1, "native append is not its exact singleton"));
        ++writer_steps;
      } else ++producer_nodes;
    }
  }
  HYLE_RETURN_IF_ERROR(require(writer_steps == 2 && producer_nodes == 2, "native Cast/write route coverage differs"));
  for (std::size_t index = 0; index < compiled.report.lane_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto lane, compiled.report.lane(index));
    HYLE_RETURN_IF_ERROR(require(!lane.queue_required, "native append invented a caller Compute Queue"));
  }
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 32);
  hyle::sched::BlockingPool waiters(hyle::mem::system_allocator(), 2);
  Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto run, compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(run, scheduler, waiters));
  const std::span<const g::QueueBinding> queues;
  const auto unprepared = writer_only ? gb::bind_append(run, graph, kw, std::move(append_k))
      : gb::bind_append(run, graph, kw, key_bank, std::move(append_k));
  HYLE_RETURN_IF_ERROR(require(unprepared.IsFailedPrecondition() && append_k.valid(),
      "native append without typed prepare accepted or consumed original pages"));
  const std::array layouts{gb::NodeLayout{reader, layout}};
  HYLE_RETURN_IF_ERROR(gb::prepare(run, queues, writer_only ? std::span<const gb::NodeLayout>{}
      : std::span<const gb::NodeLayout>(layouts)));
  const auto bound = writer_only ? gb::bind_append(run, graph, kw, std::move(append_k))
      : gb::bind_append(run, graph, kw, key_bank, std::move(append_k));
  if (missing_dependency) {
    HYLE_RETURN_IF_ERROR(require(!bound.ok() && append_k.valid() && append_k.page(0)->lease().valid() &&
        append_v.valid(), "missing explicit write/read dependency accepted or consumed original pages"));
    std::puts("native paged append missing direct reader edge preserves caller pages PASS");
    return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(bound);
  HYLE_RETURN_IF_ERROR(writer_only ? gb::bind_append(run, graph, vw, std::move(append_v))
      : gb::bind_append(run, graph, vw, value_bank, std::move(append_v)));
  HYLE_RETURN_IF_ERROR(require(!append_k.valid() && !append_v.valid(), "native bind did not move the original page owners"));
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(device.ref(), 4096, {.max_blocks = 3}));
  const auto input = [&](g::ValueId port, const c::TensorDesc &description,
                         std::span<const std::byte> bytes) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(bytes.size(), 64));
    HYLE_RETURN_IF_ERROR(transfers.upload(writer, bytes));
    HYLE_ASSIGN_OR_RETURN(auto ready, writer.publish());
    return run.bind(port, description, ready);
  };
  std::array<float, 2 * depth> new_keys{}, new_values;
  std::fill_n(new_values.begin(), depth, 4.F);
  std::fill_n(new_values.begin() + depth, depth, 5.F);
  HYLE_RETURN_IF_ERROR(input(incoming_k, rows, std::as_bytes(std::span(new_keys))));
  HYLE_RETURN_IF_ERROR(input(incoming_v, rows, std::as_bytes(std::span(new_values))));
  if (!writer_only) {
    const std::array<std::uint16_t, 2 * 2 * depth> zeros{};
    HYLE_RETURN_IF_ERROR(input(q, qdesc, std::as_bytes(std::span(zeros))));
  }
  HYLE_RETURN_IF_ERROR(run.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto done, run.submit(queues));
  HYLE_RETURN_IF_ERROR(done.wait());
  if (!writer_only) {
    HYLE_ASSIGN_OR_RETURN(auto result, run.result(output));
    HYLE_ASSIGN_OR_RETURN(auto lease, result.storage_lease());
    HYLE_ASSIGN_OR_RETURN(auto actual, transfers.read(lease, 2 * 2 * depth * 2));
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const auto number = half(actual[index]);
      const float expected = index < 2 * depth ? 2.5F : 3.F;
      HYLE_RETURN_IF_ERROR(require(std::isfinite(number) && std::abs(number - expected) < .003F,
          "same-Graph native causal attention did not read exactly 4/5 committed rows"));
    }
  }
  done = {};
  HYLE_RETURN_IF_ERROR(require(!run.reset().ok(), "untaken native append pages allowed Graph reset"));
  HYLE_ASSIGN_OR_RETURN(auto returned_k, gb::take_append(run, kw));
  HYLE_ASSIGN_OR_RETURN(auto returned_v, gb::take_append(run, vw));
  HYLE_ASSIGN_OR_RETURN(auto &key_writer, returned_k.page(0));
  HYLE_ASSIGN_OR_RETURN(auto &value_writer, returned_v.page(0));
  HYLE_ASSIGN_OR_RETURN(auto copied_k, transfers.read(key_writer.lease(), page_bytes));
  HYLE_ASSIGN_OR_RETURN(auto copied_v, transfers.read(value_writer.lease(), page_bytes));
  for (std::size_t index = 0; index < 5 * depth; ++index)
    HYLE_RETURN_IF_ERROR(require(copied_k[index] == 0 && copied_v[index] == value_words[index / depth],
        "native COW prefix/new rows differ from their independent exact bytes"));
  HYLE_ASSIGN_OR_RETURN(auto next_k, returned_k.ready());
  HYLE_ASSIGN_OR_RETURN(auto next_v, returned_v.ready());
  HYLE_RETURN_IF_ERROR(require(next_k.layout() == layout && next_v.layout() == layout &&
      next_k.page_count() == 1 && next_v.page_count() == 1,
      "first take lost the sibling native whole-Graph success latch or exact ready frame"));
  auto stale = run.submit(queues);
  if (stale) { stale->wait().IgnoreError(); return Status::Internal("taken native pages admitted a stale submission"); }
  HYLE_RETURN_IF_ERROR(require(stale.error().IsFailedPrecondition(), "native stale submit rejection changed"));
  HYLE_ASSIGN_OR_RETURN(auto unchanged_k, transfers.read(old_k, page_bytes));
  HYLE_ASSIGN_OR_RETURN(auto unchanged_v, transfers.read(old_v, page_bytes));
  HYLE_RETURN_IF_ERROR(require(std::equal(unchanged_k.begin(), unchanged_k.end(), old_keys.begin()) &&
      std::equal(unchanged_v.begin(), unchanged_v.end(), old_values.begin()) &&
      before_k.layout() == previous && before_v.layout() == previous &&
      before_k.page_count() == 1 && before_v.page_count() == 1,
      "native append changed the retained old frame, full old bytes or unused NaN tail"));
  HYLE_RETURN_IF_ERROR(require(!keys.acquire(0) && !values.acquire(1), "native old/new ready frames lost their page guards"));
  HYLE_RETURN_IF_ERROR(run.reset());
  HYLE_RETURN_IF_ERROR(run.close());
  std::printf("native Cast producer -> %s exact COW/old NaN/two-take latch/reset/stale-submit PASS\n",
      writer_only ? "two writer-only effects" : "two effects -> causal attention");
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
  HYLE_RETURN_IF_ERROR(exercise(context, device, staging, false, true));
  HYLE_RETURN_IF_ERROR(exercise(context, device, staging, true, false));
  HYLE_RETURN_IF_ERROR(exercise(context, device, staging, false, false));
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (status.ok()) return 0;
  std::fprintf(stderr, "native paged append: %.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
