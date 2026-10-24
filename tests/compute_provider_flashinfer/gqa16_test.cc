#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/cuda.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/graph/batch.h>
#include <hyle/graph/cuda.h>

#include "../compute_provider_native/half_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
using hyle::Status;
using F16 = hyle::simd::float16_t;
using BF16 = hyle::simd::bfloat16_t;
constexpr unsigned kHeads = 64, kKvHeads = 4, kDepth = 256;
constexpr float kScale = 0.0625F;

Status require(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}

template<class T>
hyle::Result<c::ReadLease> load(c::LeasePool &pool, const std::vector<T> &words) {
  HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(words.size() * sizeof(T)));
  HYLE_ASSIGN_OR_RETURN(auto view, writer.view());
  HYLE_RETURN_IF_ERROR(view.write(std::as_bytes(std::span(words))));
  return writer.publish();
}

template<class T>
Status verify(c::BatchCapacity cap, c::BatchLayoutVersion layout,
              const std::vector<T> &q, const std::vector<T> &k,
              const std::vector<T> &v, const c::ReadLease &output,
              bool graph, unsigned phase) {
  const unsigned heads = cap.query_heads, ratio = heads / kKvHeads;
  std::vector<T> actual(std::size_t(cap.max_query_rows) * heads * kDepth);
  HYLE_ASSIGN_OR_RETURN(auto view, output.view());
  HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(actual))));
  const auto host = layout.host();
  double worst = 0;
  for (unsigned r = 0; r < 2; ++r) {
    const int nq = host.query_indptr[r + 1] - host.query_indptr[r];
    const int count = host.kv_indptr[r + 1] - host.kv_indptr[r];
    const int nk = cap.storage == c::BatchStorage::kPaged
        ? (count - 1) * int(cap.page_size) + host.last_page_lengths[r] : count;
    const auto physical = [&](int j) {
      return cap.storage == c::BatchStorage::kPaged
          ? host.page_indices[host.kv_indptr[r] + j / cap.page_size] * int(cap.page_size) + j % cap.page_size
          : host.kv_indptr[r] + j;
    };
    for (int i = 0; i < nq; ++i) for (unsigned h = 0; h < heads; ++h) {
      const int visible = nk - nq + i + 1;
      std::vector<double> weights(static_cast<std::size_t>(visible));
      double high = -std::numeric_limits<double>::infinity();
      const auto qbase = (std::size_t(host.query_indptr[r] + i) * heads + h) * kDepth;
      for (int j = 0; j < visible; ++j) {
        const auto base = (std::size_t(physical(j)) * kKvHeads + h / ratio) * kDepth;
        double dot = 0;
        for (unsigned d = 0; d < kDepth; ++d)
          dot += half_test::decode(q[qbase + d]) * half_test::decode(k[base + d]);
        weights[j] = dot * kScale;
        high = std::max(high, weights[j]);
      }
      double sum = 0;
      for (auto &w : weights) { w = std::exp(w - high); sum += w; }
      for (unsigned d = 0; d < kDepth; ++d) {
        double expected = 0;
        for (int j = 0; j < visible; ++j)
          expected += weights[j] / sum * half_test::decode(v[
              (std::size_t(physical(j)) * kKvHeads + h / ratio) * kDepth + d]);
        const double observed = half_test::decode(actual[qbase + d]);
        const auto error = std::abs(observed - expected);
        // Original CUDA native batch independent FP64 budget; no oracle input
        // feeds execution and no tolerance is derived from these observations.
        HYLE_RETURN_IF_ERROR(require(std::isfinite(error) && error <= 0.007,
            "GQA16 native output differs from original FP64 batch budget"));
        worst = std::max(worst, error);
      }
    }
  }
  std::printf("GQA%u %s %s %s phase=%u maxAbs=%.9g budget=0.007\n", ratio,
      graph ? "Graph" : "Direct", cap.storage == c::BatchStorage::kPaged ? "paged" : "ragged",
      std::is_same_v<T, BF16> ? "BF16" : "F16", phase, worst);
  return Status::Ok();
}

template<class T>
Status exercise(bool use_graph, bool paged, unsigned heads = kHeads) {
  // Qwen3.8-2.4T's real full-attention geometry, without its giant weights.
  HYLE_ASSIGN_OR_RETURN(auto context, c::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto device, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
  struct Drain { c::Context &context; ~Drain() { context.drain().IgnoreError(); } } drain{context};
  const auto type = half_test::data_type<T>();
  const auto plain = c::CacheStorage::plain(type).value();
  const c::BatchCapacity cap{
      .storage = paged ? c::BatchStorage::kPaged : c::BatchStorage::kRagged,
      .query_type = type, .output_type = type, .key_storage = plain, .value_storage = plain,
      .max_requests = 2, .max_query_rows = 4, .max_kv_rows = 64,
      .max_pages = paged ? 6u : 0u, .max_page_refs = paged ? 6u : 0u,
      .query_heads = heads, .kv_heads = kKvHeads, .qk_head_dim = kDepth,
      .value_head_dim = kDepth, .page_size = paged ? 16u : 1u};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto qdesc, c::batch_query_description(cap));
  HYLE_ASSIGN_OR_RETURN(auto kdesc, c::batch_cache_description(cap, c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto vdesc, c::batch_cache_description(cap, c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto odesc, c::batch_output_description(cap, cap.max_query_rows));
  const c::BatchProgramOptions options{.resource = device.ref(), .host_workspace = host.ref(),
      .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10,
      .max_invocations = 2, .scale = kScale};
  c::BatchProgram direct_program;
  std::array<c::BatchInvocation, 2> direct;
  std::array<g::Invocation, 2> slots;
  g::CompiledGraph compiled;
  g::ValueId result;
  g::NodeId node;
  std::array<g::ValueId, 4> inputs;
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  if (use_graph) {
    HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
    HYLE_ASSIGN_OR_RETURN(auto metadata, schema.metadata_description());
    const std::array descriptions{qdesc, kdesc, vdesc, metadata};
    for (unsigned i = 0; i < 4; ++i) { HYLE_ASSIGN_OR_RETURN(inputs[i], builder.input(descriptions[i])); }
    const auto operation = paged ? c::Operation::custom(c::PagedAttention{cap, kScale})
                                 : c::Operation::custom(c::RaggedAttention{cap, kScale});
    HYLE_ASSIGN_OR_RETURN(node, builder.add_node(operation, inputs));
    HYLE_ASSIGN_OR_RETURN(result, builder.result(node));
    HYLE_RETURN_IF_ERROR(builder.publish(result));
    HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
    HYLE_ASSIGN_OR_RETURN(compiled, g::cuda::compile(graph, context, device.ref(),
        {.operation_fallback = g::OperationFallbackPolicy::kReject}));
    for (auto &slot : slots) {
      HYLE_ASSIGN_OR_RETURN(slot, compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
    }
  } else {
    HYLE_ASSIGN_OR_RETURN(direct_program, c::BatchProgram::prepare(context, schema, options));
    for (auto &slot : direct) { HYLE_ASSIGN_OR_RETURN(slot, c::BatchInvocation::create(direct_program, queue)); }
  }
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(device.ref(), 8u << 20));
  c::ReadLease old_output;
  std::vector<std::byte> old_bytes;
  const std::array<std::array<std::int32_t, 3>, 3> qptr{{{0, 1, 4}, {0, 2, 3}, {0, 1, 2}}};
  const std::array<std::array<unsigned, 2>, 3> lengths{{{17, 31}, {19, 32}, {20, 33}}};
  const std::array<std::int32_t, 5> pages{4, 1, 5, 0, 3};
  for (unsigned phase = 0; phase < 3; ++phase) {
    const auto &qp = qptr[phase];
    const auto &length = lengths[phase];
    const std::array<std::int32_t, 3> kp{0, paged ? 2 : int(length[0]),
        paged ? (phase == 2 ? 5 : 4) : int(length[0] + length[1])};
    const std::array<std::int32_t, 2> tails{int((length[0] - 1) % 16 + 1), int((length[1] - 1) % 16 + 1)};
    HYLE_ASSIGN_OR_RETURN(auto layout, schema.version({qp, kp,
        paged ? std::span(pages).first(kp.back()) : std::span<const std::int32_t>{},
        paged ? std::span(tails) : std::span<const std::int32_t>{}}));
    // Values are defined by request and its logical position. Repacking ragged
    // rows when the first request grows preserves both old logical prefixes.
    const auto nan = half_test::encode<T>(std::numeric_limits<float>::quiet_NaN());
    std::vector<T> q(*qdesc.element_count(), nan), k(*kdesc.element_count(), nan), v(k.size(), nan);
    for (unsigned r = 0; r < 2; ++r) {
      for (int row = qp[r]; row < qp[r + 1]; ++row) for (unsigned h = 0; h < heads; ++h)
        for (unsigned d = 0; d < kDepth; ++d)
          q[(std::size_t(row) * heads + h) * kDepth + d] = half_test::encode<T>(
              float(int((h * 7 + d * 3 + phase * 11 + r * 13 + row) % 53) - 26) / 64);
      for (unsigned j = 0; j < length[r]; ++j) {
        const auto row = paged ? unsigned(pages[kp[r] + j / 16]) * 16 + j % 16 : unsigned(kp[r]) + j;
        for (unsigned h = 0; h < kKvHeads; ++h) for (unsigned d = 0; d < kDepth; ++d) {
          const auto offset = (std::size_t(row) * kKvHeads + h) * kDepth + d;
          k[offset] = half_test::encode<T>(float(int((j * 17 + r * 7 + h * 11 + d) % 47) - 23) / 64);
          v[offset] = half_test::encode<T>(float(int((j * 7 + r * 19 + h * 13 + d * 3) % 61) - 30) / 32);
        }
      }
    }
    HYLE_ASSIGN_OR_RETURN(auto rq, load(pool, q));
    HYLE_ASSIGN_OR_RETURN(auto rk, load(pool, k));
    HYLE_ASSIGN_OR_RETURN(auto rv, load(pool, v));
    const unsigned selected = phase % 2;
    if (phase == 2) old_output.reset();
    c::ReadLease output;
    if (use_graph) {
      auto &slot = slots[selected];
      const gb::PrepareOptions preparation{.host_workspace = host.ref(),
          .max_float_workspace_bytes = options.max_float_workspace_bytes,
          .max_int_workspace_bytes = options.max_int_workspace_bytes};
      HYLE_RETURN_IF_ERROR(gb::prepare(slot, queues, std::array{gb::NodeLayout{node, layout}}, preparation));
      HYLE_RETURN_IF_ERROR(slot.bind(inputs[0], qdesc, rq));
      HYLE_RETURN_IF_ERROR(slot.bind(inputs[1], kdesc, rk));
      HYLE_RETURN_IF_ERROR(slot.bind(inputs[2], vdesc, rv));
      HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(*odesc.required_bytes()));
      HYLE_RETURN_IF_ERROR(slot.bind(result, odesc, std::move(writer)));
      HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
      HYLE_ASSIGN_OR_RETURN(auto done, slot.submit(queues));
      HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_ASSIGN_OR_RETURN(auto published, slot.result(result));
      HYLE_ASSIGN_OR_RETURN(output, published.storage_lease());
    } else {
      HYLE_ASSIGN_OR_RETURN(auto prepared, direct[selected].update_batch(layout));
      HYLE_ASSIGN_OR_RETURN(auto done, prepared.run(std::array{rq, rk, rv}));
      HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_ASSIGN_OR_RETURN(output, direct[selected].output());
    }
    HYLE_RETURN_IF_ERROR(verify(cap, layout, q, k, v, output, use_graph, phase));
    if (phase == 0) {
      old_output = output;
      old_bytes.resize(*odesc.required_bytes());
      HYLE_ASSIGN_OR_RETURN(auto view, old_output.view());
      HYLE_RETURN_IF_ERROR(view.read(old_bytes));
    } else if (phase == 1) {
      std::vector<std::byte> observed(old_bytes.size());
      HYLE_ASSIGN_OR_RETURN(auto view, old_output.view());
      HYLE_RETURN_IF_ERROR(view.read(observed));
      HYLE_RETURN_IF_ERROR(require(observed == old_bytes, "GQA16 second slot overwrote the old output"));
    }
  }
  HYLE_RETURN_IF_ERROR(context.drain());
  return Status::Ok();
}

void run(bool graph) {
  for (bool paged : {false, true}) {
    for (bool bf : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "graph=" << graph << " paged=" << paged << " bf=" << bf);
      const auto status = bf ? exercise<BF16>(graph, paged) : exercise<F16>(graph, paged);
      char message[768]{};
      status.ToString(message, sizeof(message));
      ASSERT_TRUE(status.ok()) << message;
    }
  }
}
TEST(FlashinferGqa16, DefaultTypedDirectPreservesUnequalPrefillContinuationAndDecode) { run(false); }
TEST(FlashinferGqa16, DefaultOrdinaryGraphPreservesUnequalPrefillContinuationAndDecode) { run(true); }
TEST(FlashinferGqa16, ExistingRatioSixKeepsOriginalMathAndCapacityBounds) {
  auto status = exercise<F16>(false, false, 24);
  ASSERT_TRUE(status.ok()) << status.message();
  status = exercise<BF16>(true, false, 24);
  ASSERT_TRUE(status.ok()) << status.message();
  // The natural uint32 tensor bound is cold validation, not a large allocation.
  auto context = c::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto host = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost).value();
  c::BatchCapacity cap{.storage = c::BatchStorage::kRagged,
      .max_query_rows = UINT32_MAX / (kHeads * kDepth) + 1,
      .max_kv_rows = 64, .max_pages = 0, .max_page_refs = 0,
      .query_heads = kHeads, .kv_heads = kKvHeads,
      .qk_head_dim = kDepth, .value_head_dim = kDepth, .page_size = 1};
  auto schema = c::BatchSchema::create(cap).value();
  c::BatchProgramOptions options{.resource = memory.ref(), .host_workspace = host.ref()};
  auto query_overflow = c::BatchProgram::prepare(context, schema, options);
  ASSERT_FALSE(query_overflow);
  EXPECT_TRUE(query_overflow.error().IsOutOfRange()) << query_overflow.error().message();
  cap.max_query_rows = 4;
  cap.max_kv_rows = UINT32_MAX / (kKvHeads * kDepth) + 1;
  schema = c::BatchSchema::create(cap).value();
  auto cache_overflow = c::BatchProgram::prepare(context, schema, options);
  ASSERT_FALSE(cache_overflow);
  EXPECT_TRUE(cache_overflow.error().IsOutOfRange()) << cache_overflow.error().message();
}
} // namespace
