#include <hyle/graph/cuda.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/graph/batch.h>

#include "mla_test_support.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace n = c::cuda;
namespace g = hyle::graph;
namespace gb = g::batch;
using hyle::Status;
using mla_test::check;

struct Gate final {
  std::mutex mutex;
  std::condition_variable condition;
  bool entered = false, released = false;
  void wait() {
    std::unique_lock lock(mutex);
    entered = true; condition.notify_all();
    condition.wait(lock, [&] { return released; });
  }
  bool await() {
    std::unique_lock lock(mutex);
    return condition.wait_for(lock, std::chrono::seconds(3), [&] { return entered; });
  }
  void release() {
    std::lock_guard lock(mutex); released = true; condition.notify_all();
  }
};

Status exercise(bool bf) {
  const c::MlaBatchCapacity mc{
      .query_type=bf ? c::DataType::kBFloat16 : c::DataType::kFloat16, .output_type=bf ? c::DataType::kBFloat16 : c::DataType::kFloat16, .cache_storage=c::CacheStorage::plain(bf ? c::DataType::kBFloat16 : c::DataType::kFloat16).value(),
      .max_query_rows = 1, .max_kv_rows = 2, .max_pages = 1,
      .max_page_refs = 1, .query_heads = 64};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create_mla(mc));
  const auto cap = schema.capacity();
  const std::array<std::int32_t, 2> ptr{0, 1};
  const std::array<std::int32_t, 1> pages{0}, tail{2}, short_tail{1};
  HYLE_ASSIGN_OR_RETURN(auto layout, schema.version({ptr, ptr, pages, tail}));
  HYLE_ASSIGN_OR_RETURN(auto different, schema.version({ptr, ptr, pages, short_tail}));
  const c::BatchProgramOptions options{
      .max_float_workspace_bytes = 16, .max_int_workspace_bytes = 4096,
      .max_invocations = 1, .scale = 1.F / 16};
  const auto native_provider = c::flashinfer::provider();
  HYLE_ASSIGN_OR_RETURN(auto id, n::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(id, native_provider));
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host, n::MemoryResource::create(context, n::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  Gate gate;
  struct Drain final {
    Gate &gate; c::Context &context;
    ~Drain() { gate.release(); context.drain().IgnoreError(); }
  } drain{gate, context};

  HYLE_ASSIGN_OR_RETURN(auto qdesc, c::batch_query_description(cap));
  HYLE_ASSIGN_OR_RETURN(auto cdesc, c::batch_cache_description(schema, c::BatchCacheRole::kMla));
  HYLE_ASSIGN_OR_RETURN(auto mdesc, schema.metadata_description());
  HYLE_ASSIGN_OR_RETURN(auto odesc, c::batch_output_description(cap, 1));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  const std::array descriptions{qdesc, cdesc, mdesc};
  std::array<g::ValueId, 3> inputs;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(inputs[i], builder.input(descriptions[i]));
  }
  HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(
      c::Operation::custom(c::PagedMlaAttention{mc, options.scale}), inputs));
  HYLE_ASSIGN_OR_RETURN(auto result, builder.result(node));
  HYLE_RETURN_IF_ERROR(builder.publish(result));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto compiled, g::cuda::compile(graph, context, device.ref(),
      {.operation_fallback = g::OperationFallbackPolicy::kReject}));
  const std::array lanes{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  HYLE_ASSIGN_OR_RETURN(auto original, compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  const gb::PrepareOptions preparation{.host_workspace=host.ref(),
      .max_float_workspace_bytes=options.max_float_workspace_bytes,
      .max_int_workspace_bytes=options.max_int_workspace_bytes,.max_source_completions=3};
  const std::array versions{gb::NodeLayout{node, layout}};
  std::vector<std::uint16_t> q(cap.query_heads * cap.qk_head_dim, mla_test::encode(0, bf));
  std::vector<std::uint16_t> cache(cap.page_size * cap.qk_head_dim, mla_test::encode(0, bf));
  for (unsigned row = 0; row < 2; ++row)
    for (unsigned d = 0; d < cap.qk_head_dim; ++d)
      cache[row * cap.qk_head_dim + d] = mla_test::encode(
          d < 512 ? (row ? .75F : .25F) + float(d % 7) / 128 : .0625F, bf);
  const auto payload_bytes = std::size_t(2) * cap.qk_head_dim * 2;
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(device.ref(), 512u << 10));
  HYLE_ASSIGN_OR_RETURN(auto qw, pool.allocate(q.size() * 2));
  HYLE_RETURN_IF_ERROR(qw.view()->write(std::as_bytes(std::span(q))));
  HYLE_ASSIGN_OR_RETURN(auto query, qw.publish());
  HYLE_ASSIGN_OR_RETURN(auto arena, c::PagedArena::create(schema, c::BatchCacheRole::kMla,
      device.ref(), {.max_backing_bytes = cache.size() * 2 + 1}));
  // Warm this exact Graph/native slot before holding its stream. All native
  // modules, workspace, page backing and Graph command storage are cold here.
  {
    auto warm_cache = cache;
    for (unsigned row = 0; row < 2; ++row)
      for (unsigned d = 0; d < cap.qk_head_dim; ++d)
        warm_cache[row * cap.qk_head_dim + d] = mla_test::encode(
            d < 512 ? -(row ? .75F : .25F) - float(d % 7) / 128 : -.0625F, bf);
    HYLE_ASSIGN_OR_RETURN(auto page, arena.acquire(0));
    HYLE_RETURN_IF_ERROR(page.lease().view()->write(std::as_bytes(std::span(warm_cache))));
    HYLE_ASSIGN_OR_RETURN(auto ready, page.publish(2));
    HYLE_ASSIGN_OR_RETURN(auto frame, arena.read(layout, std::array{ready}));
    HYLE_RETURN_IF_ERROR(gb::prepare(original, lanes, versions, preparation));
    HYLE_RETURN_IF_ERROR(original.bind(inputs[0], qdesc, query));
    HYLE_RETURN_IF_ERROR(gb::bind(original, inputs[1], frame));
    HYLE_ASSIGN_OR_RETURN(auto output, pool.allocate(*odesc.required_bytes()));
    HYLE_RETURN_IF_ERROR(original.bind(result, odesc, std::move(output)));
    HYLE_RETURN_IF_ERROR(original.prepare_replay(lanes,
        {.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kReject}));
    HYLE_ASSIGN_OR_RETURN(auto done, original.submit(lanes)); HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto warm, original.result(result));
    HYLE_ASSIGN_OR_RETURN(auto read, warm.storage_lease());
    HYLE_RETURN_IF_ERROR(mla_test::oracle(cap, layout, q, warm_cache, read, bf));
  }
  HYLE_RETURN_IF_ERROR(gb::prepare(original, lanes, versions, preparation));
  auto receiver = std::move(original);
  HYLE_ASSIGN_OR_RETURN(auto page, arena.acquire(0));
  HYLE_RETURN_IF_ERROR(receiver.bind(inputs[0], qdesc, query));
  HYLE_ASSIGN_OR_RETURN(auto destination_output, pool.allocate(*odesc.required_bytes()));
  HYLE_RETURN_IF_ERROR(receiver.bind(result, odesc, std::move(destination_output)));
  HYLE_ASSIGN_OR_RETURN(auto report, gb::report(receiver, node));
  HYLE_RETURN_IF_ERROR(check(report.float_workspace_bytes == 0 && report.int_workspace_bytes <= 4096,
      "Graph pending MLA did not use the active no-split workspace"));

  HYLE_ASSIGN_OR_RETURN(auto source, pool.allocate(payload_bytes));
  HYLE_ASSIGN_OR_RETURN(auto hp, c::LeasePool::create(host.ref(), payload_bytes + 64));
  HYLE_ASSIGN_OR_RETURN(auto hw, hp.allocate(payload_bytes));
  HYLE_RETURN_IF_ERROR(hw.view()->write(std::as_bytes(std::span(cache)).first(payload_bytes)));
  HYLE_ASSIGN_OR_RETURN(auto staging, hw.publish());
  HYLE_ASSIGN_OR_RETURN(auto destination, n::pointer<std::byte>(source.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto source_pointer, n::pointer<std::byte>(staging.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto guards, c::LeaseUsePool::create());
  HYLE_ASSIGN_OR_RETURN(auto writing, guards.begin());
  HYLE_RETURN_IF_ERROR(writing.write(source)); HYLE_RETURN_IF_ERROR(writing.read(staging));
  const std::array specs{c::CopyResourceSpec{device.ref(), payload_bytes},
                         c::CopyResourceSpec{device.ref(), arena.page_bytes()}};
  HYLE_ASSIGN_OR_RETURN(auto copies, c::CopyBatchProgram::prepare(context, specs));
  HYLE_ASSIGN_OR_RETURN(auto copy, c::CopyBatchInvocation::create(copies, queue));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({0}, std::move(source)));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({1}, std::move(page.lease())));
  const std::array<c::CopyPhase, 2> phases{{{}, {.dependency_mask = 1}}};
  const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {1}, 0, 0, payload_bytes}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases, ranges));
  const std::array<c::PagedPageUpdate, 1> updates{{{&page, {1}}}};
  HYLE_ASSIGN_OR_RETURN(auto frame, arena.pending_read(layout, {}, updates));
  HYLE_ASSIGN_OR_RETURN(auto wrong, arena.pending_read(different, {}, updates));
  HYLE_RETURN_IF_ERROR(check(gb::bind(receiver, inputs[1], wrong).IsInvalidArgument(),
      "MLA pending Graph admitted a different layout"));
  HYLE_RETURN_IF_ERROR(check(gb::bind(receiver, inputs[2], frame).IsInvalidArgument() &&
      gb::bind(receiver, inputs[0], frame).IsInvalidArgument(),
      "MLA pending cache escaped into query or metadata"));
  HYLE_RETURN_IF_ERROR(gb::bind(receiver, inputs[1], frame));
  HYLE_RETURN_IF_ERROR(receiver.prepare_replay(lanes,
      {.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kReject}));

  // Every owner that can retire/free CUDA storage is declared before Release.
  // An early error must release the host callback before destroying those owners.
  c::Completion producer, done;
  c::CopyBatchSubmission copied;
  c::CopyBatchUpdate update;
  g::GraphOutput output, retained_output;
  c::ReadLease observed;
  struct Release final { Gate &gate; ~Release() { gate.release(); } } release{gate};
  HYLE_ASSIGN_OR_RETURN(producer, n::submit(queue, [&](cudaStream_t stream) noexcept -> Status {
    HYLE_RETURN_IF_ERROR(n::status(cudaLaunchHostFunc(stream,
        [](void *p) { static_cast<Gate *>(p)->wait(); }, &gate)));
    return n::status(cudaMemcpyAsync(destination, source_pointer, payload_bytes, cudaMemcpyHostToDevice, stream));
  }));
  HYLE_RETURN_IF_ERROR(writing.attach(producer));
  HYLE_RETURN_IF_ERROR(check(gate.await(), "Graph MLA producer missed checkpoint"));
  std::puts("Graph MLA producer gate held after exact Graph/native warm and cold resource preparation");
  HYLE_RETURN_IF_ERROR(copy.bind_producer({0}, std::move(producer)));
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases, ranges));
  HYLE_ASSIGN_OR_RETURN(copied, copy.submit());
  HYLE_ASSIGN_OR_RETURN(update, c::CopyBatchUpdate::take(std::move(copy), std::move(copied)));
  HYLE_ASSIGN_OR_RETURN(done, gb::submit(receiver, lanes, std::move(update)));
  HYLE_RETURN_IF_ERROR(check(!update.valid() && guards.active_batches() == 1 &&
      !receiver.result(result) && !gb::take_update(receiver),
      "Graph MLA exposed output/source or retired producer before native completion"));
  HYLE_RETURN_IF_ERROR(check(gb::prepare(receiver, lanes, versions, preparation).IsUnavailable(),
      "Graph MLA allowed pending slot reuse"));
  std::puts("Graph MLA submit returned while producer remains held; output/source unavailable");
  gate.release(); HYLE_RETURN_IF_ERROR(done.wait());
  HYLE_ASSIGN_OR_RETURN(output, receiver.result(result));
  HYLE_ASSIGN_OR_RETURN(observed, output.storage_lease());
  HYLE_RETURN_IF_ERROR(mla_test::oracle(cap, layout, q, cache, observed, bf));
  retained_output = output;
  done = {};
  HYLE_RETURN_IF_ERROR(check(!gb::take_update(receiver) && gb::prepare(receiver, lanes, versions, preparation).IsUnavailable(),
      "Graph MLA retained output allowed source or slot reuse"));
  output.reset();
  HYLE_RETURN_IF_ERROR(check(!gb::take_update(receiver) && gb::prepare(receiver, lanes, versions, preparation).IsUnavailable(),
      "copied Graph result did not retain its invocation"));
  retained_output.reset();
  // The exported caller-output lease independently owns its physical backing.
  // It need not retain this Graph's source controls once result handles drop.
  HYLE_ASSIGN_OR_RETURN(update, gb::take_update(receiver));
  HYLE_ASSIGN_OR_RETURN(page.lease(), update.take_mutable({1}));
  HYLE_ASSIGN_OR_RETURN(auto ready, page.publish(2));
  HYLE_RETURN_IF_ERROR(check(arena.read(layout, std::array{ready}).has_value(),
      "Graph MLA did not restore the original page writer"));
  HYLE_ASSIGN_OR_RETURN(copy, update.release_invocation()); HYLE_RETURN_IF_ERROR(copy.reset());
  frame = {}; wrong = {};
  HYLE_RETURN_IF_ERROR(gb::prepare(receiver, lanes, versions, preparation));
  HYLE_RETURN_IF_ERROR(mla_test::oracle(cap, layout, q, cache, observed, bf));
  observed.reset();
  HYLE_RETURN_IF_ERROR(receiver.drain());
  HYLE_RETURN_IF_ERROR(check(guards.active_batches() == 0, "Graph MLA producer native guards leaked"));
  std::printf("MLA %s public Graph 3 ports / one pending CKV: producer->F2->Graph, wrong-layout/ports, full FP64, output+lease retention and writer recovery PASS\n",
              bf ? "BF16" : "F16");
  return Status::Ok();
}
} // namespace
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  for (bool bf : {false, true}) {
    const auto status = exercise(bf);
    if (!status.ok()) {
      char message[1024]{}; status.ToString(message, sizeof(message));
      std::fprintf(stderr, "public pending Graph CUDA MLA: %s\n", message); return 1;
    }
  }
  return 0;
}
