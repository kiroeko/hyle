#include <hyle/graph/cuda.h>
#include <gtest/gtest.h>

#include <hyle/compute/cuda.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#include <hyle/graph/batch.h>

#include "../compute_batch/mla_test_support.h"

#include <array>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = hyle::graph::batch;
using hyle::Status;

Status ragged_mla_case(bool bf, unsigned heads, unsigned depth) {
  const c::MlaBatchCapacity capacity{
      .storage = c::BatchStorage::kRagged,
      .query_type=bf ? c::DataType::kBFloat16 : c::DataType::kFloat16, .output_type=bf ? c::DataType::kBFloat16 : c::DataType::kFloat16, .cache_storage=c::CacheStorage::plain(bf ? c::DataType::kBFloat16 : c::DataType::kFloat16).value(),
      .layout = depth == 576 ? c::MlaCacheLayout::kCkv512Rope64
                            : c::MlaCacheLayout::kCkv512,
      .max_requests = 2, .max_query_rows = 4, .max_kv_rows = 64,
      .max_pages = 0, .max_page_refs = 0, .query_heads = heads, .page_size = 1};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create_mla(capacity));
  const auto cap = schema.capacity();
  c::BatchProgramOptions options{.max_float_workspace_bytes = 16u << 20,
      .max_int_workspace_bytes = 1u << 20, .scale = 1.F / 16};
  // The default roster must discover the same distinct two-payload extension.
  {
    HYLE_ASSIGN_OR_RETURN(auto context, c::cuda::create_context());
    HYLE_ASSIGN_OR_RETURN(auto memory, c::cuda::MemoryResource::create(
        context, c::cuda::MemoryKind::kDevice));
    HYLE_ASSIGN_OR_RETURN(auto host, c::cuda::MemoryResource::create(
        context, c::cuda::MemoryKind::kPinnedHost));
    auto defaults = options; defaults.resource = memory.ref(); defaults.host_workspace = host.ref();
    HYLE_ASSIGN_OR_RETURN(auto program, c::BatchProgram::prepare(context, schema, defaults));
    HYLE_RETURN_IF_ERROR(mla_test::check(program.valid(), "default CUDA did not prepare ragged MLA"));
  }
  const auto native_provider = c::flashinfer::provider();
  HYLE_ASSIGN_OR_RETURN(auto id, c::cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(id, native_provider));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto memory, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
  options.resource = memory.ref(); options.host_workspace = host.ref();
  HYLE_ASSIGN_OR_RETURN(auto program, c::BatchProgram::prepare(context, schema, options));
  std::array<c::BatchInvocation, 2> direct;
  for (auto &slot : direct) { HYLE_ASSIGN_OR_RETURN(slot, c::BatchInvocation::create(program, queue)); }
  HYLE_ASSIGN_OR_RETURN(auto qdesc, c::batch_query_description(cap));
  HYLE_ASSIGN_OR_RETURN(auto cdesc, c::batch_cache_description(schema, c::BatchCacheRole::kMla));
  HYLE_ASSIGN_OR_RETURN(auto mdesc, schema.metadata_description());
  HYLE_ASSIGN_OR_RETURN(auto odesc, c::batch_output_description(cap, cap.max_query_rows));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  std::array<g::ValueId, 3> inputs;
  const std::array descriptions{qdesc, cdesc, mdesc};
  for (unsigned i = 0; i < 3; ++i) { HYLE_ASSIGN_OR_RETURN(inputs[i], builder.input(descriptions[i])); }
  HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(c::Operation::custom(
      c::RaggedMlaAttention{capacity, options.scale}), inputs));
  HYLE_ASSIGN_OR_RETURN(auto result, builder.result(node));
  HYLE_RETURN_IF_ERROR(builder.publish(result));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto compiled, g::cuda::compile(graph, context, memory.ref(),
      {.operation_fallback = g::OperationFallbackPolicy::kReject}));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const gb::PrepareOptions preparation{.host_workspace = host.ref(), .max_float_workspace_bytes = options.max_float_workspace_bytes, .max_int_workspace_bytes = options.max_int_workspace_bytes};
std::array<g::Invocation, 2> invocations;
  for (auto &slot : invocations) {
    HYLE_ASSIGN_OR_RETURN(slot, compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  }
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(memory.ref(), 16u << 20));
  std::array<c::ReadLease, 2> old_outputs;
  std::array<std::vector<std::byte>, 2> saved;
  for (unsigned phase = 0; phase < 2; ++phase) {
    const std::array<std::int32_t, 3> qptr = phase ? std::array<std::int32_t, 3>{0, 1, 2}
                                                : std::array<std::int32_t, 3>{0, 1, 4};
    const std::array<std::int32_t, 3> kptr = phase ? std::array<std::int32_t, 3>{0, 21, 64}
                                                : std::array<std::int32_t, 3>{0, 17, 64};
    HYLE_ASSIGN_OR_RETURN(auto layout, schema.version({qptr, kptr, {}, {}}));
    if (!phase && bf && heads == 64 && depth == 576) {
      auto tiny = options; tiny.max_float_workspace_bytes = tiny.max_int_workspace_bytes = 16;
      HYLE_ASSIGN_OR_RETURN(auto limited, c::BatchProgram::prepare(context, schema, tiny));
      HYLE_ASSIGN_OR_RETURN(auto slot, c::BatchInvocation::create(limited, queue));
      const auto rejected = slot.prepare_batch(layout);
      HYLE_RETURN_IF_ERROR(mla_test::check(!rejected && rejected.error().IsResourceExhausted(),
          "ragged MLA did not reject insufficient original split workspace"));
    }
    std::vector<std::uint16_t> q(std::size_t(4) * heads * depth), cache(std::size_t(64) * depth);
    for (std::size_t i = 0; i < q.size(); ++i)
      q[i] = mla_test::encode(float(int((i + phase * 7) % 29) - 14) / 64, bf);
    for (std::size_t i = 0; i < cache.size(); ++i)
      cache[i] = mla_test::encode(float(int((i * 13 + i / depth + phase * 11) % 37) - 18) / 32, bf);
    HYLE_ASSIGN_OR_RETURN(auto qw, pool.allocate(q.size() * 2));
    HYLE_RETURN_IF_ERROR(qw.view()->write(std::as_bytes(std::span(q))));
    HYLE_ASSIGN_OR_RETURN(auto qr, qw.publish());
    HYLE_ASSIGN_OR_RETURN(auto cw, pool.allocate(cache.size() * 2));
    HYLE_RETURN_IF_ERROR(cw.view()->write(std::as_bytes(std::span(cache))));
    HYLE_ASSIGN_OR_RETURN(auto cr, cw.publish());
    HYLE_ASSIGN_OR_RETURN(auto token, direct[phase].update_batch(layout));
    HYLE_RETURN_IF_ERROR(mla_test::check(direct[phase].report().split_kv,
        "ragged MLA control did not exercise original F32 split partials"));
    HYLE_ASSIGN_OR_RETURN(auto done, token.run(c::MlaBatchInputs{qr, cr}));
    HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto output, direct[phase].output());
    HYLE_RETURN_IF_ERROR(mla_test::oracle(cap, layout, q, cache, output, bf));
    auto &invocation = invocations[phase];
    HYLE_RETURN_IF_ERROR(gb::prepare(invocation, queues, std::array{gb::NodeLayout{node, layout}}, preparation));
    HYLE_RETURN_IF_ERROR(invocation.bind(inputs[0], qdesc, qr));
    HYLE_RETURN_IF_ERROR(invocation.bind(inputs[1], cdesc, cr));
    HYLE_ASSIGN_OR_RETURN(auto ow, pool.allocate(std::size_t(4) * heads * 512 * 2));
    HYLE_RETURN_IF_ERROR(invocation.bind(result, odesc, std::move(ow)));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));HYLE_ASSIGN_OR_RETURN(auto graph_done, invocation.submit(queues));
    HYLE_RETURN_IF_ERROR(graph_done.wait());
    HYLE_ASSIGN_OR_RETURN(auto published, invocation.result(result));
    HYLE_ASSIGN_OR_RETURN(auto read, published.storage_lease());
    HYLE_RETURN_IF_ERROR(mla_test::oracle(cap, layout, q, cache, read, bf));
    if (!phase) {
      old_outputs = {output, read};
      for (unsigned i = 0; i < 2; ++i) {
        saved[i].resize(std::size_t(4) * heads * 512 * 2);
        HYLE_RETURN_IF_ERROR(old_outputs[i].view()->read(saved[i]));
      }
    } else {
      for (unsigned i = 0; i < 2; ++i) {
        std::vector<std::byte> current(saved[i].size());
        HYLE_RETURN_IF_ERROR(old_outputs[i].view()->read(current));
        HYLE_RETURN_IF_ERROR(mla_test::check(current == saved[i], "ragged MLA old output changed across another slot"));
      }
    }
  }
  HYLE_RETURN_IF_ERROR(context.drain());
  return Status::Ok();
}

TEST(FlashInferRaggedMla, Compact512And576KeepGlobalRowsDirectGraphAndOwners) {
  for (bool bf : {false, true}) for (unsigned heads : {64u, 128u})
    for (unsigned depth : {512u, 576u}) {
      SCOPED_TRACE(::testing::Message() << "bf=" << bf << " heads=" << heads << " depth=" << depth);
      const auto status = ragged_mla_case(bf, heads, depth);
      char message[768]{};
      status.ToString(message, sizeof(message));
      ASSERT_TRUE(status.ok()) << message;
    }
}
} // namespace
