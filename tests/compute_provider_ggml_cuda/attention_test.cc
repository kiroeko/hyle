#include "../compute_models/d5_kimi_mla_admission.h"

#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/executable.h"

#include <limits>

#include "../compute_provider_flashinfer/attention_test_support.h"
#include "test_support.h"

namespace {
using namespace flashinfer_test;

template <class T>
void exercise_attention(compute::DataType type, int query_length, int kv_length,
                        double tolerance, float scale = 0.0625F,
                        bool wide_exponent = false, bool causal = true,
                        int depth = 256, bool weak_probabilities = false) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor query(type, {2, query_length, 8, depth}, memory.ref());
  Tensor key(type, {2, kv_length, 2, depth}, memory.ref());
  Tensor value(type, {2, kv_length, 2, depth}, memory.ref());
  Tensor other_query(type, {2, query_length, 8, depth}, memory.ref());
  Tensor other_key(type, {2, kv_length, 2, depth}, memory.ref());
  Tensor other_value(type, {2, kv_length, 2, depth}, memory.ref());
  Tensor output(type, {2, query_length, 8, depth}, memory.ref());
  Tensor other_output(type, {2, query_length, 8, depth}, memory.ref());
  const compute::ScaledDotProductAttention operation{.scale = scale,
                                                     .causal = causal};
  const std::array inputs{query.view.as_const(), key.view.as_const(),
                          value.view.as_const()};
  const std::array other_inputs{other_query.view.as_const(),
                                other_key.view.as_const(),
                                other_value.view.as_const()};
  const std::array outputs{output.view};
  const std::array other_outputs{other_output.view};
  auto prepared = compute::prepare(context, compute::Operation(operation),
                                   inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::ggml_cuda::kProviderId);
  EXPECT_EQ(prepared->runtime_properties().workspace_owner,
            compute::ProviderWorkspaceOwner::kCaller);
  auto first_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto first = prepared->create_invocation(first_workspace.view());
  ASSERT_TRUE(first) << first.error().message();
  auto second_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto second = prepared->create_invocation(second_workspace.view());
  ASSERT_TRUE(second) << second.error().message();
  auto first_queue = context.create_queue().value();
  auto second_queue = context.create_queue().value();
  for (int seed : {0, 3, 9}) {
    auto q = payload<T>(query, seed);
    auto k = payload<T>(key, seed + 5);
    auto v = payload<T>(value, seed + 13);
    if (wide_exponent) {
      std::fill(q.begin(), q.end(), encode<T>(0));
      std::fill(k.begin(), k.end(), encode<T>(131072));
      for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = encode<T>((i % 2 == 0 ? 1.0F : -1.0F) * 131072);
    } else if (weak_probabilities) {
      std::fill(q.begin(), q.end(), encode<T>(0));
      std::fill(k.begin(), k.end(), encode<T>(0));
      std::fill(v.begin(), v.end(), encode<T>(0));
      for (std::size_t i = 0; i < q.size(); i += std::size_t(depth))
        q[i] = encode<T>(1);
      for (std::size_t i = 0; i < k.size(); i += std::size_t(depth)) {
        const auto token = (i / std::size_t(depth) / 2) % std::size_t(kv_length);
        if (token != 0) {
          k[i] = encode<T>(-288);
          std::fill_n(v.begin() + static_cast<std::ptrdiff_t>(i), depth,
                      encode<T>(60000));
        }
      }
    }
    if (wide_exponent || weak_probabilities) {
      ASSERT_TRUE(query.storage.view().write(std::as_bytes(std::span(q))).ok());
      ASSERT_TRUE(key.storage.view().write(std::as_bytes(std::span(k))).ok());
      ASSERT_TRUE(value.storage.view().write(std::as_bytes(std::span(v))).ok());
    }
    const auto oq = payload<T>(other_query, seed + 19);
    const auto ok = payload<T>(other_key, seed + 23);
    const auto ov = payload<T>(other_value, seed + 31);
    auto first_done = first->submit(first_queue, inputs, outputs);
    ASSERT_TRUE(first_done) << first_done.error().message();
    auto second_done =
        second->submit(second_queue, other_inputs, other_outputs);
    ASSERT_TRUE(second_done) << second_done.error().message();
    ASSERT_TRUE(second_done->wait().ok());
    ASSERT_TRUE(first_done->wait().ok());
    check_output(output, q, k, v, query_length, kv_length, tolerance, scale, causal);
    check_output(other_output, oq, ok, ov, query_length, kv_length,
                 wide_exponent ? 0.01 : tolerance, scale, causal);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    // A compiled graph's derived views must follow all changed port addresses.
    auto rebound = first->submit(first_queue, other_inputs, other_outputs);
    ASSERT_TRUE(rebound) << rebound.error().message();
    ASSERT_TRUE(rebound->wait().ok());
    check_output(other_output, oq, ok, ov, query_length, kv_length,
                 wide_exponent ? 0.01 : tolerance, scale, causal);
    check_output(output, q, k, v, query_length, kv_length, tolerance, scale, causal);
  }
  {
    auto abandoned = first->submit(first_queue, inputs, outputs);
    ASSERT_TRUE(abandoned);
  }
  ASSERT_TRUE(first_queue.drain().ok());
}

TEST(GgmlCudaAttention, OriginalPrefillDecodeAndIndependentReplays) {
  exercise_attention<__half>(compute::DataType::kFloat16, 17, 31, 0.002);
  exercise_attention<__half>(compute::DataType::kFloat16, 1, 513, 0.002);
  exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 17, 31, 0.01);
  exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 1, 513, 0.01);
}

TEST(GgmlCudaAttention, ZeroAndNegativeScalePreserveCausalMask) {
  for (float scale : {0.0F, -0.0625F}) {
    exercise_attention<__half>(compute::DataType::kFloat16, 17, 31, 0.002,
                               scale);
    exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 17, 31,
                                      0.01, scale);
  }
}

TEST(GgmlCudaAttention, NoncausalMtpUsesEveryKeyAndIndependentReplays) {
  for (float scale : {0.0625F, 0.0F, -0.0625F}) {
    // More queries than keys distinguishes this domain from causal admission.
    exercise_attention<__half>(compute::DataType::kFloat16, 5, 3, 0.002,
                              scale, false, false);
    exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 5, 3, 0.01,
                                    scale, false, false);
  }
}

// Uniform scores give an independent prefix-mean oracle without constructing
// a score matrix on the host. V varies across every batch/head/token/channel,
// so an incorrect causal offset or a compacted batch stride remains visible.
template <class T>
void prefix_attention(compute::DataType type, int query_length, int kv_length,
                      bool causal, float scale, double tolerance) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider()).value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor query(type, {2, query_length, 8, 256}, memory.ref());
  Tensor key(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor value(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor output(type, {2, query_length, 8, 256}, memory.ref());
  std::vector<T> q(query.storage.size() / sizeof(T), encode<T>(0));
  std::vector<T> k(key.storage.size() / sizeof(T), encode<T>(0));
  std::vector<T> v(value.storage.size() / sizeof(T));
  ASSERT_TRUE(query.storage.view().write(std::as_bytes(std::span(q))).ok());
  ASSERT_TRUE(key.storage.view().write(std::as_bytes(std::span(k))).ok());
  for (int batch = 0; batch < 2; ++batch)
    for (int token = 0; token < kv_length; ++token)
      for (int head = 0; head < 2; ++head)
        for (int depth = 0; depth < 256; ++depth) {
          const auto index = static_cast<std::size_t>(
              ((batch * kv_length + token) * 2 + head) * 256 + depth);
          v[index] = encode<T>(static_cast<float>(
              batch * 128 + head * 32 + depth % 17 + token % 113 - 96) /
                               256.0F);
        }
  ASSERT_TRUE(value.storage.view().write(std::as_bytes(std::span(v))).ok());
  const std::array inputs{query.view.as_const(), key.view.as_const(),
                          value.view.as_const()};
  const std::array outputs{output.view};
  auto prepared = compute::prepare(
      context, compute::Operation(compute::ScaledDotProductAttention{
                   .scale = scale, .causal = causal}),
      inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation = prepared->create_invocation(workspace.view());
  ASSERT_TRUE(invocation) << invocation.error().message();
  auto queue = context.create_queue().value();
  for (int replay = 0; replay < 2; ++replay) {
    // Rebind both the input and output addresses after the first execution.
    Tensor rebound_value(type, {2, kv_length, 2, 256}, memory.ref());
    Tensor rebound_output(type, {2, query_length, 8, 256}, memory.ref());
    ASSERT_TRUE(rebound_value.storage.view()
                    .write(std::as_bytes(std::span(v))).ok());
    const std::array rebound_inputs{inputs[0], inputs[1],
                                    rebound_value.view.as_const()};
    const std::array rebound_outputs{rebound_output.view};
    auto done = invocation->submit(queue,
        replay == 0 ? std::span(inputs) : std::span(rebound_inputs),
        replay == 0 ? std::span(outputs) : std::span(rebound_outputs));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    std::vector<T> actual(q.size());
    auto &observed = replay == 0 ? output : rebound_output;
    ASSERT_TRUE(observed.storage.view()
                    .read(std::as_writable_bytes(std::span(actual))).ok());
    for (int batch = 0; batch < 2; ++batch)
      for (int head = 0; head < 8; ++head)
        for (int depth = 0; depth < 256; ++depth) {
          double sum = 0;
          int visited = 0;
          for (int token = 0; token < query_length; ++token) {
            const auto count = causal ? kv_length - query_length + token + 1
                                      : kv_length;
            while (visited < count) {
              const auto index = static_cast<std::size_t>(
                  ((batch * kv_length + visited) * 2 + head / 4) * 256 + depth);
              sum += decode(v[index]);
              ++visited;
            }
            const auto index = static_cast<std::size_t>(
                ((batch * query_length + token) * 8 + head) * 256 + depth);
            ASSERT_NEAR(decode(actual[index]), sum / count, tolerance)
                << "replay=" << replay << " batch=" << batch
                << " token=" << token << " head=" << head
                << " depth=" << depth;
          }
        }
  }
}

TEST(GgmlCudaAttention, LongCausalPrefixUsesGlobalQueryPositionAndBatchStride) {
  for (float scale : {0.0F, -0.0625F}) {
    prefix_attention<float>(compute::DataType::kFloat32, 257, 273, true,
                             scale, 0.00001);
    prefix_attention<__half>(compute::DataType::kFloat16, 2049, 2065, true,
                              scale, 0.002);
    prefix_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 257, 273,
                                    true, scale, 0.01);
  }
}

TEST(GgmlCudaAttention, LongNoncausalQueryUsesCompleteShorterKv) {
  prefix_attention<float>(compute::DataType::kFloat32, 257, 131, false,
                           0.0625F, 0.00001);
  prefix_attention<__half>(compute::DataType::kFloat16, 257, 131, false,
                            0.0625F, 0.002);
  prefix_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 257, 131,
                                  false, 0.0625F, 0.01);
}

TEST(GgmlCudaAttention, Float32ScoresAndProbabilitiesKeepTheirPrecision) {
  exercise_attention<float>(compute::DataType::kFloat32, 257, 273, 0.00001,
                             0.0625F, false, true, 8);
}

TEST(GgmlCudaAttention, Float16WeakProbabilitiesRetainLargeValueContributions) {
  // exp(-18) underflows in Half. Keep its contribution through F32 probability
  // and PV intermediates, including non-square causal rows.
  exercise_attention<__half>(compute::DataType::kFloat16, 17, 31, 0.002,
                              0.0625F, false, true, 256, true);
}

TEST(GgmlCudaAttention, Bfloat16ValuesOutsideFloat16RangeRemainFinite) {
  exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 17, 31, 0.01,
                                    0.0625F, true);
  exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 1, 513, 0.01,
                                    0.0625F, true);
}

TEST(GgmlCudaAttention, Qwen38GroupedQueryHeadRatioRuns) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor query(compute::DataType::kBFloat16, {1, 1, 24, 256}, memory.ref());
  Tensor key(compute::DataType::kBFloat16, {1, 1, 4, 256}, memory.ref());
  Tensor value(compute::DataType::kBFloat16, {1, 1, 4, 256}, memory.ref());
  Tensor output(compute::DataType::kBFloat16, {1, 1, 24, 256}, memory.ref());
  (void)payload<__nv_bfloat16>(query, 3);
  (void)payload<__nv_bfloat16>(key, 7);
  const auto values = payload<__nv_bfloat16>(value, 11);
  const std::array inputs{query.view.as_const(), key.view.as_const(),
                          value.view.as_const()};
  const std::array outputs{output.view};
  auto prepared = compute::prepare(
      context,
      compute::Operation(compute::ScaledDotProductAttention{.scale = 0.0625F,
                                                            .causal = true}),
      inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::ggml_cuda::kProviderId);
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation = prepared->create_invocation(workspace.view());
  ASSERT_TRUE(invocation) << invocation.error().message();
  auto queue = context.create_queue().value();
  auto done = invocation->submit(queue, inputs, outputs);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  std::vector<__nv_bfloat16> actual(24 * 256);
  ASSERT_TRUE(output.storage.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  for (std::size_t head = 0; head < 24; ++head)
    for (std::size_t depth = 0; depth < 256; ++depth)
      EXPECT_NEAR(decode(actual[head * 256 + depth]),
                  decode(values[(head / 6) * 256 + depth]), 0.01)
          << "head=" << head << " depth=" << depth;
}
TEST(GgmlCudaAttention, D5AbsorbedMlaColdShapeBoundary) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                           compute::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  d5_kimi_mla_admission::check([&](auto operation, const auto &inputs, const auto &outputs) {
    return bool(compute::prepare(context, operation, inputs, outputs, memory.ref()));
  });
}

enum class MaskValues { kFinite, kCausalInfinity, kPrecision, kNonfinite };

struct MaskPort {
  compute::TensorDesc description;
  mem::Buffer storage;
  compute::TensorView view;

  MaskPort(std::span<const std::int64_t> shape, mem::MemoryResourceRef resource)
      : description(compute::TensorDesc::dense(compute::DataType::kFloat32,
                                               shape).value()),
        storage(mem::Buffer::create(resource, *description.required_bytes(),
                                    256).value()),
        view(compute::TensorView::create(description, storage.view()).value()) {}
};

std::size_t mask_index(std::span<const std::int64_t> dimensions,
                      std::array<std::int64_t, 4> coordinate) {
  std::size_t index = 0;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
    index = index * std::size_t(dimensions[axis]) +
        (dimensions[axis] == 1 ? 0 :
         std::size_t(coordinate[4 - dimensions.size() + axis]));
  return index;
}

template <class T>
void check_masked_output(const Tensor &output, const std::vector<T> &q,
                         const std::vector<T> &k, const std::vector<T> &v,
                         const MaskPort &mask, const std::vector<float> &bias,
                         std::int64_t kv_heads, float scale, bool causal,
                         double tolerance) {
  std::vector<T> actual(*output.description.element_count());
  ASSERT_TRUE(output.storage.view().read(
      std::as_writable_bytes(std::span(actual))).ok());
  const auto shape = output.description.dimensions();
  const auto batches = shape[0], queries = shape[1], heads = shape[2];
  const auto depth = shape[3];
  const auto keys = std::int64_t(k.size()) / (batches * kv_heads * depth);
  for (std::int64_t b = 0; b < batches; ++b)
    for (std::int64_t n = 0; n < queries; ++n)
      for (std::int64_t h = 0; h < heads; ++h) {
        const auto qbase = std::size_t(((b * queries + n) * heads + h) * depth);
        std::vector<double> weights(std::size_t(keys), 0.0);
        for (std::int64_t s = 0; s < keys; ++s) {
          const auto base = std::size_t(
              ((b * keys + s) * kv_heads + h / (heads / kv_heads)) * depth);
          double dot = 0;
          for (std::int64_t d = 0; d < depth; ++d)
            dot += double(decode(q[qbase + std::size_t(d)])) *
                   decode(k[base + std::size_t(d)]);
          auto score = dot * scale +
              bias[mask_index(mask.description.dimensions(), {b,h,n,s})];
          if (causal && s > keys - queries + n)
            score = -std::numeric_limits<double>::infinity();
          weights[std::size_t(s)] = score;
        }
        const auto maximum = *std::max_element(weights.begin(), weights.end());
        double total = 0;
        for (auto &weight : weights) {
          weight = std::exp(weight - maximum);
          total += weight;
        }
        for (std::int64_t d = 0; d < depth; ++d) {
          double expected = 0;
          for (std::int64_t s = 0; s < keys; ++s) {
            const auto base = std::size_t(
                ((b * keys + s) * kv_heads + h / (heads / kv_heads)) * depth);
            expected += weights[std::size_t(s)] *
                        decode(v[base + std::size_t(d)]);
          }
          expected /= total;
          const auto value = decode(actual[qbase + std::size_t(d)]);
          if (std::isnan(expected))
            ASSERT_TRUE(std::isnan(value)) << "b=" << b << " q=" << n;
          else
            ASSERT_NEAR(value, expected, tolerance)
                << "b=" << b << " q=" << n << " h=" << h << " d=" << d;
        }
      }
}

template <class T>
void masked_default_routes(compute::DataType type, std::int64_t batches,
                           std::int64_t queries, std::int64_t keys,
                           std::int64_t heads, std::int64_t kv_heads,
                           std::int64_t depth,
                           std::initializer_list<std::int64_t> mask_shape,
                           MaskValues pattern, float scale, bool causal,
                           double tolerance) {
  namespace graph = hyle::graph;
  for (const bool graph_route : {false, true}) {
    SCOPED_TRACE(graph_route ? "default Graph" : "default Direct");
    auto owner = graph_route ? graph::cuda::create_context()
                             : cuda::create_context();
    ASSERT_TRUE(owner) << owner.error().message();
    auto context = std::move(*owner);
    auto memory = cuda::MemoryResource::create(context,
                                               cuda::MemoryKind::kDevice).value();
    const std::array<std::int64_t,4> qs{batches,queries,heads,depth};
    const std::array<std::int64_t,4> ks{batches,keys,kv_heads,depth};
    std::array<Tensor,2> q{Tensor(type,qs,memory.ref()),Tensor(type,qs,memory.ref())};
    std::array<Tensor,2> k{Tensor(type,ks,memory.ref()),Tensor(type,ks,memory.ref())};
    std::array<Tensor,2> v{Tensor(type,ks,memory.ref()),Tensor(type,ks,memory.ref())};
    std::array<Tensor,2> out{Tensor(type,qs,memory.ref()),Tensor(type,qs,memory.ref())};
    const std::span<const std::int64_t> md{mask_shape.begin(),mask_shape.size()};
    std::array<MaskPort,2> masks{MaskPort(md,memory.ref()),MaskPort(md,memory.ref())};
    std::array<compute::Queue,2> queues{
        context.create_queue().value(),context.create_queue().value()};
    const compute::ScaledDotProductAttention operation{
        .mask=compute::MaskKind::kAdditive,.scale=scale,.causal=causal};
    const std::array descriptions{q[0].description,k[0].description,
                                  v[0].description,masks[0].description};
    compute::PreparedOperation prepared;
    std::array<hyle::mem::Buffer,2> direct_workspaces;
    std::array<compute::OperationInvocation,2> direct;
    graph::CompiledGraph compiled;
    std::array<graph::Invocation,2> slots;
    std::array<graph::ValueId,5> ids;
    if (graph_route) {
      auto builder = graph::GraphBuilder::create().value();
      for (std::size_t i=0;i<4;++i) ids[i]=builder.input(descriptions[i]).value();
      ids[4]=builder.add(operation,ids[0],ids[1],ids[2],ids[3]).value();
      ASSERT_TRUE(builder.publish(ids[4]).ok());
      auto logical=std::move(builder).finish().value();
      auto result=graph::cuda::compile(logical,context,memory.ref());
      ASSERT_TRUE(result) << result.error().message();
      compiled=std::move(*result);
      ASSERT_EQ(compiled.report.step_count(),1U);
      EXPECT_EQ(compiled.report.step(0)->provider.id,compute::ggml_cuda::kProviderId);
      for(auto &slot:slots) {
        auto created=compiled.executable.create_invocation();
        ASSERT_TRUE(created) << created.error().message();
        slot=std::move(*created);
      }
    } else {
      auto result=compute::prepare(context,compute::Operation(operation),
          descriptions,std::span(&q[0].description,1),memory.ref());
      ASSERT_TRUE(result) << result.error().message();
      EXPECT_EQ(result->provider().id,compute::ggml_cuda::kProviderId);
      EXPECT_EQ(result->runtime_properties().workspace_owner,
                compute::ProviderWorkspaceOwner::kCaller);
      EXPECT_TRUE(result->runtime_properties().blocks_host);
      prepared=std::move(*result);
      for(std::size_t slot=0;slot<direct.size();++slot) {
        direct_workspaces[slot]=
            ggml_cuda_test::workspace_buffer(prepared,memory.ref());
        auto created=prepared.create_invocation(direct_workspaces[slot].view());
        ASSERT_TRUE(created) << created.error().message();
        direct[slot]=std::move(*created);
      }
    }
    // Each slot changes every actual input address on the second run.
    for(int step=0;step<2;++step)
      for(std::size_t slot=0;slot<2;++slot) {
        const auto address=(slot+std::size_t(step))%2;
        auto qv=payload<T>(q[address],step*13+int(slot)*3);
        auto kv=payload<T>(k[address],step*17+int(slot)*5+2);
        auto vv=payload<T>(v[address],step*19+int(slot)*7+7);
        if(pattern==MaskValues::kPrecision || pattern==MaskValues::kNonfinite) {
          std::fill(qv.begin(),qv.end(),encode<T>(0));
          std::fill(kv.begin(),kv.end(),encode<T>(0));
          for(std::size_t i=0;i<vv.size();++i)
            vv[i]=encode<T>(float((i/std::size_t(depth)/std::size_t(kv_heads))%
                                   std::size_t(keys)));
          ASSERT_TRUE(q[address].storage.view().write(std::as_bytes(std::span(qv))).ok());
          ASSERT_TRUE(k[address].storage.view().write(std::as_bytes(std::span(kv))).ok());
          ASSERT_TRUE(v[address].storage.view().write(std::as_bytes(std::span(vv))).ok());
        }
        std::vector<float> bias(*masks[address].description.element_count());
        for(std::size_t i=0;i<bias.size();++i)
          bias[i]=0.12345679F+float(int((i*7+slot*3+std::size_t(step)*11)%23)-11)*0.03125F;
        if(pattern==MaskValues::kPrecision)
          for(std::size_t i=0;i<bias.size();++i)
            bias[i]=i%2==0 ? 0.0F : 1.0F+0x1p-11F;
        if(pattern==MaskValues::kNonfinite) {
          std::fill(bias.begin(),bias.end(),-std::numeric_limits<float>::infinity());
          bias[std::size_t(keys)]=std::numeric_limits<float>::infinity();
          bias[std::size_t(keys)+1]=0;
        }
        if(pattern==MaskValues::kCausalInfinity)
          for(std::int64_t b=0;b<batches;++b)
            for(std::int64_t h=0;h<heads;++h)
              for(std::int64_t n=0;n<queries;++n)
                for(std::int64_t s=keys-queries+n+1;s<keys;++s)
                  bias[mask_index(md,{b,h,n,s})]=std::numeric_limits<float>::infinity();
        ASSERT_TRUE(masks[address].storage.view().write(std::as_bytes(std::span(bias))).ok());
        const std::array inputs{q[address].view.as_const(),k[address].view.as_const(),
                                v[address].view.as_const(),masks[address].view.as_const()};
        const std::array outputs{out[address].view};
        if(graph_route) {
          for(std::size_t i=0;i<4;++i) ASSERT_TRUE(slots[slot].bind(ids[i],inputs[i]).ok());
          ASSERT_TRUE(slots[slot].bind(ids[4],outputs[0]).ok());
          const std::array bindings{graph::QueueBinding{graph::ExecutionLaneId{0},&queues[slot]}};
          ASSERT_TRUE(slots[slot].prepare_replay(bindings).ok());
          auto done=slots[slot].submit(bindings);
          ASSERT_TRUE(done) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
          *done=compute::Completion{};
          auto held=slots[slot].result(ids[4]);
          ASSERT_TRUE(held) << held.error().message();
          EXPECT_FALSE(slots[slot].bind(ids[0],inputs[0]).ok());
          held->reset();
        } else {
          auto done=direct[slot].submit(queues[slot],inputs,outputs);
          ASSERT_TRUE(done) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
        }
        check_masked_output(out[address],qv,kv,vv,masks[address],bias,kv_heads,
                            scale,causal,tolerance);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        if(pattern==MaskValues::kPrecision) {
          std::vector<T> values(*out[address].description.element_count());
          ASSERT_TRUE(out[address].storage.view().read(
              std::as_writable_bytes(std::span(values))).ok());
          // Both F16 and BF16 lose the supplied 2^-11 bias increment.
          EXPECT_GT(double(decode(values[0]))-1.0/(1.0+std::exp(-1.0)),0.00005);
        }
      }
  }
}

TEST(GgmlCudaMaskedAttention, DefaultDirectAndGraphKeepFloatMaskPrecisionAndNaNs) {
  for(const auto pattern:{MaskValues::kPrecision,MaskValues::kNonfinite})
    masked_default_routes<float>(compute::DataType::kFloat32,1,2,2,4,2,8,
        {2,2},pattern,0.125F,false,0.000002);
}
TEST(GgmlCudaMaskedAttention, DefaultDirectAndGraphQwenShapeUsesOriginalFloatBias) {
  masked_default_routes<__nv_bfloat16>(compute::DataType::kBFloat16,1,23,32,8,2,256,
      {23,32},MaskValues::kFinite,0.0625F,false,0.003);
}
TEST(GgmlCudaMaskedAttention, DefaultDirectAndGraphBroadcastEveryScoreAxis) {
  masked_default_routes<float>(compute::DataType::kFloat32,2,5,7,4,2,8,
      {2,1,1,7},MaskValues::kFinite,-0.125F,false,0.00001);
  masked_default_routes<float>(compute::DataType::kFloat32,2,5,7,4,2,8,
      {1,4,5,1},MaskValues::kFinite,0.0F,false,0.00001);
}
TEST(GgmlCudaMaskedAttention, DefaultDirectAndGraphTilesPreserveMaskPitchAndCausality) {
  masked_default_routes<float>(compute::DataType::kFloat32,2,257,273,8,2,8,
      {2,8,257,273},MaskValues::kCausalInfinity,-0.125F,true,0.00001);
}
} // namespace
