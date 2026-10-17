#include "../compute_provider_flashinfer/attention_test_support.h"

#include <atomic>
#include <limits>
#include <optional>
#include <c10/cuda/CUDACachingAllocator.h>

#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/graph/executable.h"

namespace {
using namespace flashinfer_test;
namespace graph = hyle::graph;
using Bfloat = __nv_bfloat16;

struct Shape {
  int query = 17;
  int kv = 31;
  int heads = 8;
  int kv_heads = 2;
  int depth = 256;
  float scale = 0.0625F;
  bool causal = true;
  bool wide_exponent = false;
};

bool supports_fused_attention(const compute::Context &context) {
  const auto environment = cuda::environment(context).value();
  const int sm = environment.sm_major * 10 + environment.sm_minor;
  return sm >= 80 && sm <= 121;
}

hyle::Result<graph::CompiledGraph> compile_attention(
    const compute::Context &context, mem::MemoryResourceRef memory,
    compute::ScaledDotProductAttention operation,
    const std::array<compute::TensorDesc, 3> &descriptions,
    std::array<graph::ValueId, 4> &ids) {
  auto builder = graph::GraphBuilder::create().value();
  for (std::size_t i = 0; i < descriptions.size(); ++i)
    ids[i] = builder.input(descriptions[i]).value();
  ids[3] = builder.add(operation, ids[0], ids[1], ids[2]).value();
  const auto published = builder.publish(ids[3]);
  if (!published.ok()) return std::unexpected(published);
  auto logical = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{memory}};
  const std::array targets{
      graph::ExecutionTarget{&context, graph::StorageDomainId{0}}};
  return graph::compile(logical,
                        graph::CompileEnvironment{.storage_domains = storage,
                                                   .targets = targets});
}

void bind_graph(graph::Invocation &invocation,
                const std::array<graph::ValueId, 4> &ids,
                const std::array<compute::ConstTensorView, 3> &inputs,
                compute::TensorView output, compute::Queue &queue) {
  for (std::size_t i = 0; i < inputs.size(); ++i)
    ASSERT_TRUE(invocation.bind(ids[i], inputs[i]).ok());
  ASSERT_TRUE(invocation.bind(ids[3], output).ok());
  const std::array queues{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(invocation.prepare_replay(
      queues, {.mode = graph::ReplayMode::kPreferCapture,
               .host_wait = graph::HostWaitPolicy::kReject}).ok());
  EXPECT_EQ(invocation.replay_route(), graph::ReplayRoute::kCommands);
}

void exercise(const Shape &shape) {
  SCOPED_TRACE(::testing::Message() << "Q=" << shape.query << " KV=" << shape.kv
      << " H=" << shape.heads << "/" << shape.kv_heads << " D=" << shape.depth
      << " scale=" << shape.scale << " causal=" << shape.causal);
  auto context = compute::Context::create(cuda::device(0).value(),
                                           compute::torch_cuda::provider()).value();
  if (!supports_fused_attention(context))
    GTEST_SKIP() << "Fused BF16 attention requires SM80..121";
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto q_shape = std::array<std::int64_t, 4>{2, shape.query, shape.heads, shape.depth};
  const auto kv_shape = std::array<std::int64_t, 4>{2, shape.kv, shape.kv_heads, shape.depth};
  constexpr auto type = compute::DataType::kBFloat16;
  Tensor q(type, q_shape, memory.ref()), k(type, kv_shape, memory.ref()),
      v(type, kv_shape, memory.ref()), other_q(type, q_shape, memory.ref()),
      other_k(type, kv_shape, memory.ref()), other_v(type, kv_shape, memory.ref()),
      output(type, q_shape, memory.ref()), other_output(type, q_shape, memory.ref()),
      graph_output(type, q_shape, memory.ref()), other_graph_output(type, q_shape, memory.ref());
  const std::array inputs{q.view.as_const(), k.view.as_const(), v.view.as_const()};
  const std::array other_inputs{other_q.view.as_const(), other_k.view.as_const(), other_v.view.as_const()};
  const std::array outputs{output.view}, other_outputs{other_output.view};
  const compute::ScaledDotProductAttention operation{.scale = shape.scale, .causal = shape.causal};
  auto prepared_result = compute::prepare(context, compute::Operation(operation), inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared_result) << prepared_result.error().message();
  auto prepared = std::move(*prepared_result);
  ASSERT_EQ(prepared.provider().id, compute::torch_cuda::kProviderId);
  EXPECT_EQ(prepared.runtime_properties().mode, compute::ProviderRuntimeMode::kSharedQueue);
  EXPECT_EQ(prepared.runtime_properties().workspace_owner, compute::ProviderWorkspaceOwner::kProvider);
  EXPECT_EQ(prepared.runtime_properties().replay_class, compute::ProviderReplayClass::kManaged);
  EXPECT_FALSE(prepared.runtime_properties().capture_compatible);
  auto first = prepared.create_invocation().value();
  auto second = prepared.create_invocation().value();
  auto first_queue = context.create_queue().value();
  auto second_queue = context.create_queue().value();
  std::array<graph::ValueId, 4> ids;
  auto compiled_result = compile_attention(context, memory.ref(), operation,
      {q.description, k.description, v.description}, ids);
  ASSERT_TRUE(compiled_result) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  ASSERT_EQ(compiled.report.step_count(), 1u);
  ASSERT_EQ(compiled.report.step(0)->provider.id, compute::torch_cuda::kProviderId);
  auto graph_first = compiled.executable.create_invocation().value();
  auto graph_second = compiled.executable.create_invocation().value();

  for (int seed : {0, 3, 9}) {
    auto host_q = payload<Bfloat>(q, seed);
    auto host_k = payload<Bfloat>(k, seed + 5);
    auto host_v = payload<Bfloat>(v, seed + 13);
    if (shape.wide_exponent) {
      std::fill(host_q.begin(), host_q.end(), encode<Bfloat>(0));
      std::fill(host_k.begin(), host_k.end(), encode<Bfloat>(131072));
      for (std::size_t i = 0; i < host_v.size(); ++i)
        host_v[i] = encode<Bfloat>((i % 2 == 0 ? 1.0F : -1.0F) * 131072);
      ASSERT_TRUE(q.storage.view().write(std::as_bytes(std::span(host_q))).ok());
      ASSERT_TRUE(k.storage.view().write(std::as_bytes(std::span(host_k))).ok());
      ASSERT_TRUE(v.storage.view().write(std::as_bytes(std::span(host_v))).ok());
    }
    const auto oq = payload<Bfloat>(other_q, seed + 19);
    const auto ok = payload<Bfloat>(other_k, seed + 23);
    const auto ov = payload<Bfloat>(other_v, seed + 31);
    bind_graph(graph_first, ids, inputs, graph_output.view, first_queue);
    bind_graph(graph_second, ids, other_inputs, other_graph_output.view, second_queue);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto first_result = first.submit(first_queue, inputs, outputs);
    ASSERT_TRUE(first_result) << first_result.error().message();
    auto first_done = std::move(*first_result);
    auto second_result = second.submit(second_queue, other_inputs, other_outputs);
    ASSERT_TRUE(second_result) << second_result.error().message();
    auto second_done = std::move(*second_result);
    auto graph_first_result = graph_first.submit(first_queue);
    ASSERT_TRUE(graph_first_result) << graph_first_result.error().message();
    auto graph_first_done = std::move(*graph_first_result);
    auto graph_second_result = graph_second.submit(second_queue);
    ASSERT_TRUE(graph_second_result) << graph_second_result.error().message();
    auto graph_second_done = std::move(*graph_second_result);
    ASSERT_TRUE(graph_second_done.wait().ok());
    ASSERT_TRUE(graph_first_done.wait().ok());
    ASSERT_TRUE(second_done.wait().ok());
    ASSERT_TRUE(first_done.wait().ok());
    for (Tensor *destination : {&output, &graph_output})
      check_output(*destination, host_q, host_k, host_v, shape.query, shape.kv, 0.01,
                    shape.scale, shape.causal);
    for (Tensor *destination : {&other_output, &other_graph_output})
      check_output(*destination, oq, ok, ov, shape.query, shape.kv, 0.01,
                    shape.scale, shape.causal);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto rebound = first.submit(first_queue, other_inputs, other_outputs).value();
    bind_graph(graph_first, ids, other_inputs, other_graph_output.view, first_queue);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto graph_rebound = graph_first.submit(first_queue).value();
    ASSERT_TRUE(graph_rebound.wait().ok());
    ASSERT_TRUE(rebound.wait().ok());
    for (Tensor *destination : {&other_output, &other_graph_output})
      check_output(*destination, oq, ok, ov, shape.query, shape.kv, 0.01,
                    shape.scale, shape.causal);
    for (Tensor *destination : {&output, &graph_output})
      check_output(*destination, host_q, host_k, host_v, shape.query, shape.kv, 0.01,
                    shape.scale, shape.causal);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}

TEST(TorchCudaAttention, SameHeadAndGroupedBottomRightPrefillDecodeAndRebind) {
  exercise({.query = 7, .kv = 11, .heads = 4, .kv_heads = 4, .depth = 64});
  exercise({});
  exercise({.query = 1, .kv = 513});
}

TEST(TorchCudaAttention, ZeroAndNegativeScaleKeepBottomRightCausality) {
  for (float scale : {0.0F, -0.0625F}) exercise({.scale = scale});
}

TEST(TorchCudaAttention, NoncausalQueriesCanOutnumberKeys) {
  for (float scale : {0.0625F, 0.0F, -0.0625F})
    exercise({.query = 5, .kv = 3, .scale = scale, .causal = false});
}

TEST(TorchCudaAttention, Bfloat16ValuesOutsideFloat16RangeRemainFinite) {
  exercise({.wide_exponent = true});
  exercise({.query = 1, .kv = 513, .wide_exponent = true});
}

struct HostGate {
  std::atomic<bool> released{false};
  static void CUDART_CB wait(void *raw) {
    auto &gate = *static_cast<HostGate *>(raw);
    while (!gate.released.load(std::memory_order_acquire))
      gate.released.wait(false, std::memory_order_relaxed);
  }
  void open() noexcept {
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

class OriginalAllocatorLimit {
public:
  OriginalAllocatorLimit()
      : previous_(c10::cuda::CUDACachingAllocator::getMemoryFraction(0)) {
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::setMemoryFraction(0, 0);
  }
  ~OriginalAllocatorLimit() {
    c10::cuda::CUDACachingAllocator::setMemoryFraction(previous_, 0);
  }
private:
  double previous_;
};

void pending_external_stream(bool graph_path, bool fail_allocation) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                           compute::torch_cuda::provider()).value();
  if (!supports_fused_attention(context))
    GTEST_SKIP() << "Fused BF16 attention requires SM80..121";
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto host = cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost).value();
  cudaStream_t native = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking), cudaSuccess);
  struct NativeOwner {
    cudaStream_t stream;
    ~NativeOwner() { EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess); }
  } native_owner{native};
  auto queue = cuda::borrow_stream(context, native).value();
  EXPECT_EQ(cuda::stream(queue).value(), native);
  constexpr auto type = compute::DataType::kBFloat16;
  const std::array<std::int64_t, 4> qs{1, 7, 4, 64}, kvs{1, 11, 4, 64};
  Tensor q(type, qs, memory.ref()), k(type, kvs, memory.ref()),
      v(type, kvs, memory.ref()), output(type, qs, memory.ref()),
      staging(type, qs, host.ref());
  (void)payload<Bfloat>(q, 0);
  const auto kh = payload<Bfloat>(k, 5), vh = payload<Bfloat>(v, 13);
  auto qh = payload<Bfloat>(staging, 19);
  for (auto &value : qh) value = encode<Bfloat>(4 * decode(value));
  ASSERT_TRUE(staging.storage.view().write(std::as_bytes(std::span(qh))).ok());
  const std::array inputs{q.view.as_const(), k.view.as_const(), v.view.as_const()};
  const std::array outputs{output.view};
  const compute::ScaledDotProductAttention operation{.scale = 0.125F, .causal = true};
  auto prepared = compute::prepare(context, compute::Operation(operation), inputs,
                                    outputs, memory.ref()).value();
  ASSERT_EQ(prepared.provider().id, compute::torch_cuda::kProviderId);
  auto direct = prepared.create_invocation().value();
  std::array<graph::ValueId, 4> ids;
  auto compiled = compile_attention(context, memory.ref(), operation,
      {q.description, k.description, v.description}, ids).value();
  ASSERT_EQ(compiled.report.step(0)->provider.id, compute::torch_cuda::kProviderId);
  auto replay = compiled.executable.create_invocation().value();
  bind_graph(replay, ids, inputs, output.view, queue);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const auto submit = [&]() -> hyle::Result<compute::Completion> {
    return graph_path ? replay.submit(queue) : direct.submit(queue, inputs, outputs);
  };
  {
    // Finish CUDA/Torch first use and prime this exact stream's allocations
    // before a host callback deliberately blocks progress.
    auto warm = submit().value();
    ASSERT_TRUE(warm.wait().ok());
  }
  if (fail_allocation) {
    // A warm Invocation retains its last native result. Release those owners
    // so emptyCache can return whole blocks before the original allocator's
    // limit is applied; a free sub-block could otherwise satisfy this request.
    direct = {};
    replay = {};
    ASSERT_TRUE(context.drain().ok());
    direct = prepared.create_invocation().value();
    replay = compiled.executable.create_invocation().value();
    bind_graph(replay, ids, inputs, output.view, queue);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
  ASSERT_TRUE(q.storage.view().fill(std::byte{0}).ok());
  auto *destination = cuda::pointer<Bfloat>(q.storage.view()).value();
  auto *source = cuda::pointer<Bfloat>(staging.storage.view()).value();
  const auto bytes = q.storage.size();
  std::optional<OriginalAllocatorLimit> limit;
  if (fail_allocation) limit.emplace();
  HostGate gate;
  struct ReleaseGate {
    HostGate &gate;
    compute::Context &context;
    ~ReleaseGate() { gate.open(); context.drain().IgnoreError(); }
  } release{gate, context};
  auto producer = cuda::submit(queue, [&](cudaStream_t stream) noexcept {
    const auto status = cudaLaunchHostFunc(stream, &HostGate::wait, &gate);
    return status == cudaSuccess
               ? cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream)
               : status;
  }).value();
  auto submitted = submit();
  ASSERT_TRUE(submitted) << submitted.error().message();
  EXPECT_EQ(submitted->poll().value(), compute::CompletionPoll::kPending);
  // Caller tensor/resource owners remain live. Artifact/invocation handles may
  // disappear while the accepted Queue prefix and translated failure are pending.
  replay = {};
  compiled = {};
  direct = {};
  prepared = {};
  gate.open();
  const auto terminal = submitted->wait();
  ASSERT_TRUE(producer.wait().ok());
  if (fail_allocation) {
    EXPECT_TRUE(terminal.IsResourceExhausted()) << terminal.message();
    limit.reset();  // Restore the original process-local allocator policy.
    auto recovered = compute::prepare(context, compute::Operation(operation), inputs,
                                        outputs, memory.ref()).value();
    auto invocation = recovered.create_invocation().value();
    auto done = invocation.submit(queue, inputs, outputs).value();
    ASSERT_TRUE(done.wait().ok());
  } else {
    ASSERT_TRUE(terminal.ok()) << terminal.message();
  }
  // A provider using a foreign/default stream could read the old zero Q before
  // the gated producer. This oracle uses the new Q, not the warmup input.
  check_output(output, qh, kh, vh, 7, 11, 0.01, 0.125F, true);
}

class TorchCudaAttentionAsync : public ::testing::TestWithParam<bool> {};
TEST_P(TorchCudaAttentionAsync, BorrowedStreamProducerAndPendingArtifacts) {
  pending_external_stream(GetParam(), false);
}
TEST_P(TorchCudaAttentionAsync, OriginalOomRetainsPrefixAndQueueRecovers) {
  pending_external_stream(GetParam(), true);
}
INSTANTIATE_TEST_SUITE_P(DirectAndGraph, TorchCudaAttentionAsync,
                         ::testing::Values(false, true));

TEST(TorchCudaAttention, UnsupportedDescriptionsDeclineBeforeNativeExecution) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                           compute::torch_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto rejected = [&](compute::DataType type, int depth,
                             compute::ScaledDotProductAttention operation,
                             int query = 7, int kv = 11) {
    const auto q = compute::TensorDesc::dense(type,
        std::array<std::int64_t, 4>{1, query, 4, depth}).value();
    const auto k = compute::TensorDesc::dense(type,
        std::array<std::int64_t, 4>{1, kv, 2, depth}).value();
    const std::array inputs{q, k, k};
    const std::array outputs{q};
    EXPECT_FALSE(compute::prepare(context, compute::Operation(operation), inputs,
                                   outputs, memory.ref()));
  };
  rejected(compute::DataType::kFloat16, 64, {});
  rejected(compute::DataType::kFloat32, 64, {});
  rejected(compute::DataType::kBFloat16, 7, {});
  rejected(compute::DataType::kBFloat16, 264, {});
  rejected(compute::DataType::kBFloat16, 64,
            {.accumulation_type = compute::DataType::kBFloat16});
  rejected(compute::DataType::kBFloat16, 64, {.causal = true}, 11, 7);
  rejected(compute::DataType::kBFloat16, 64,
            {.layout = compute::AttentionLayout::kBatchHeadSequenceDepth}, 4, 2);
  rejected(compute::DataType::kBFloat16, 64,
            {.scale = std::numeric_limits<float>::infinity()});
  const auto q = compute::TensorDesc::dense(compute::DataType::kBFloat16,
      std::array<std::int64_t, 4>{1, 7, 4, 64}).value();
  const auto k = compute::TensorDesc::dense(compute::DataType::kBFloat16,
      std::array<std::int64_t, 4>{1, 11, 2, 64}).value();
  const auto mask = compute::TensorDesc::dense(compute::DataType::kBFloat16,
      std::array<std::int64_t, 4>{1, 4, 7, 11}).value();
  const std::array masked_inputs{q, k, k, mask};
  const std::array outputs{q};
  EXPECT_FALSE(compute::prepare(context,
      compute::Operation(compute::ScaledDotProductAttention{.mask = compute::MaskKind::kAdditive}),
      masked_inputs, outputs, memory.ref()));
}
} // namespace
