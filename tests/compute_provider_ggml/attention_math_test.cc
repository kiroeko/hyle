#include "../compute_ggml_runtime/gallocr_faults.h"
#include "../compute_provider_native/half_test_support.h"
#include "../compute_models/batched_sink_fixture.h"
#include "cpu_program.h"
#include "ggml-backend-impl.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/stats.h"
#include "metadata_arena.h"
#include "program.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace d = c::ggml_cpu::internal;
using Half = hyle::simd::float16_t;

TEST(GgmlCpuAttention, BatchedSinkMasksAndHeadDenominatorsReplay) {
  auto context = c::cpu::create_context({.native_threads = 2}).value();
  batched_sink_test::direct(context, hyle::mem::system_resource());
}

// Also run with an assertion-enabled native SDK: the sink alone supplies a
// positive denominator, even though every ordinary softmax term is zero.
TEST(GgmlCpuMathAttention, FiniteSinkAcceptsFullyMaskedSingleKey) {
  auto context = c::Context::create(c::cpu::device(), c::ggml_cpu::provider()).value();
  auto queue = context.create_queue().value();
  auto resource = hyle::mem::system_resource();
  batched_sink_test::Fixture fixture(64, false, 1, 1, 1);
  auto prepared = c::prepare(context, fixture.operation, fixture.descriptions,
      std::span(&fixture.descriptions[0], 1), resource).value();
  EXPECT_EQ(prepared.provider().id, c::ggml_cpu::kProviderId);
  auto invocation = prepared.create_invocation().value();
  std::array<hyle::mem::Buffer, 5> buffers;
  std::array<c::ConstTensorView, 5> inputs;
  for (std::size_t port = 0; port < buffers.size(); ++port) {
    buffers[port] = hyle::mem::Buffer::create(resource,
        fixture.descriptions[port].required_bytes().value(), 64).value();
    inputs[port] = c::ConstTensorView::create(fixture.descriptions[port], buffers[port].view()).value();
  }
  auto output = hyle::mem::Buffer::create(resource,
      fixture.descriptions[0].required_bytes().value(), 64).value();
  auto destination = c::TensorView::create(fixture.descriptions[0], output.view()).value();
  for (int phase = 0; phase < 2; ++phase) {
    fixture.update(phase);
    fixture.payloads[3][0] = -std::numeric_limits<float>::infinity();
    for (std::size_t port = 0; port < buffers.size(); ++port)
      ASSERT_TRUE(buffers[port].view().write(std::as_bytes(std::span(fixture.payloads[port]))).ok());
    auto done = invocation.submit(queue, inputs, std::span(&destination, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    for (const auto value : half_test::read<float>(output)) EXPECT_EQ(value, 0.F);
  }
}

struct AttentionPorts final {
  std::vector<hyle::mem::Buffer> owners;
  std::vector<hyle::mem::BufferView> views;
  explicit AttentionPorts(std::span<const c::TensorDesc> descriptions) {
    for (const auto &desc : descriptions) {
      const auto bytes = desc.required_bytes().value();
      owners.push_back(
          hyle::mem::Buffer::create(hyle::mem::system_resource(), bytes + 4, 64)
              .value());
      views.push_back(owners.back().view().subview(4, bytes).value());
    }
  }
  void write(std::size_t port, const c::TensorDesc &desc,
             std::span<const double> values) {
    std::vector<float> data(values.size());
    std::transform(values.begin(), values.end(), data.begin(),
                   [](double x) { return static_cast<float>(x); });
    ASSERT_EQ(data.size(), desc.element_count().value());
    ASSERT_TRUE(views[port].write(std::as_bytes(std::span(data))).ok());
  }
  std::vector<double> read(const c::TensorDesc &desc, std::size_t port) {
    std::vector<float> data(desc.element_count().value());
    auto status = views[port].read(std::as_writable_bytes(std::span(data)));
    if (!status.ok())
      throw std::bad_alloc();
    return {data.begin(), data.end()};
  }
};
struct RawProgram final {
  hyle::mem::AllocatorRef allocator;
  c::Provider provider = c::ggml_cpu::provider({.native_threads = 2});
  const c::detail::ProviderVTable *table =
      c::detail::ProviderAccess::vtable(provider);
  void *context = nullptr;
  c::detail::ProviderPrepared prepared;
  std::vector<void *> slots;
  std::vector<c::TensorDesc> inputs, outputs;
  explicit RawProgram(hyle::mem::AllocatorRef resource) : allocator(resource) {}
  ~RawProgram() {
    for (auto *slot : slots)
      prepared.vtable->destroy_invocation(slot, allocator);
    if (prepared.vtable)
      prepared.vtable->destroy_prepared(prepared.state, allocator);
    if (context)
      table->destroy_context(context, allocator);
  }
  hyle::Status initialize(d::Recipe recipe) {
    HYLE_RETURN_IF_ERROR(table->create_context(provider, c::cpu::device(),
                                               nullptr, allocator, &context));
    for (auto id : recipe.inputs)
      inputs.push_back(recipe.values[id]);
    for (auto id : recipe.outputs)
      outputs.push_back(recipe.values[id]);
    return d::compile(context, std::move(recipe), allocator, &prepared).status;
  }
  hyle::Status add_slot() {
    void *slot = nullptr;
    auto status = prepared.vtable->create_invocation(prepared.state, allocator,
                                                     {}, &slot);
    if (!status.ok()) {
      if (slot)
        prepared.vtable->destroy_invocation(slot, allocator);
      return status;
    }
    slots.push_back(slot);
    return status;
  }
  hyle::Status submit(std::size_t slot, c::Queue &queue, AttentionPorts &ports,
                      std::size_t ninputs) {
    std::vector<c::ConstTensorView> x;
    std::vector<c::TensorView> y;
    for (std::size_t i = 0; i < inputs.size(); ++i)
      x.push_back(
          c::ConstTensorView::create(inputs[i], ports.views[i]).value());
    for (std::size_t i = 0; i < outputs.size(); ++i)
      y.push_back(
          c::TensorView::create(outputs[i], ports.views[ninputs + i]).value());
    HYLE_ASSIGN_OR_RETURN(auto sequence, c::detail::QueueSequence::begin(queue));
    sequence.record(prepared.vtable
        ->shared_submit(prepared.state, slots[slot], sequence, x, y));
    HYLE_ASSIGN_OR_RETURN(auto completion, sequence.finish());
    return completion.wait();
  }
};
struct MathCase final {
  c::DataType type;
  std::int64_t batch, queries, keys, heads, kv_heads, query_depth, value_depth;
  float scale;
  bool causal, sink;
  bool bshd = true;
};

void check_math_attention(const MathCase &test) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  const auto dense = [&](auto shape) {
    if constexpr (std::tuple_size_v<decltype(shape)> == 4)
      if (!test.bshd)
        std::swap(shape[1], shape[2]);
    return c::TensorDesc::dense(test.type, shape).value();
  };
  const auto query =
      dense(std::array{test.batch, test.queries, test.heads, test.query_depth});
  const auto key =
      dense(std::array{test.batch, test.keys, test.kv_heads, test.query_depth});
  const auto value =
      dense(std::array{test.batch, test.keys, test.kv_heads, test.value_depth});
  const auto output =
      dense(std::array{test.batch, test.queries, test.heads, test.value_depth});
  d::Recipe recipe{.values = {query, key, value}, .inputs = {0, 1, 2}};
  if (test.sink) {
    recipe.values.push_back(dense(std::array{test.queries, test.keys}));
    recipe.values.push_back(dense(std::array{test.heads}));
    recipe.inputs = {0, 1, 2, 3, 4};
  }
  const auto result_id = std::uint32_t(recipe.values.size());
  recipe.values.push_back(output);
  recipe.values.push_back(output);
  const auto operation =
      test.sink
          ? c::model_ops::sink_attention(test.scale)
          : c::Operation(c::ScaledDotProductAttention{
                .layout = test.bshd
                              ? c::AttentionLayout::kBatchSequenceHeadDepth
                              : c::AttentionLayout::kBatchHeadSequenceDepth,
                .scale = test.scale,
                .causal = test.causal});
  recipe.nodes = {{operation, recipe.inputs, {result_id}},
                  {c::Operation(c::Unary{c::UnaryKind::kNeg}),
                   {result_id},
                   {result_id + 1}}};
  // Publish both the attention intermediate and its downstream consumer.
  recipe.outputs = {result_id, result_id + 1};
  RawProgram region(stats.get());
  ASSERT_TRUE(region.initialize(recipe).ok());
  ASSERT_TRUE(region.add_slot().ok());
  ASSERT_TRUE(region.add_slot().ok());
  EXPECT_EQ(region.prepared.runtime.replay_class,
            c::ProviderReplayClass::kPrepared);
  auto queue_context =
      c::Context::create(c::cpu::device(), c::ggml_cpu::provider()).value();
  auto queue = queue_context.create_queue().value();
  std::vector<c::TensorDesc> descriptions(recipe.values.begin(),
                                          recipe.values.begin() + result_id);
  descriptions.push_back(output);
  descriptions.push_back(output);
  AttentionPorts first(descriptions), second(descriptions);
  std::vector<double> retained;
  const auto index = [&](auto b, auto row, auto head, auto rows, auto heads,
                         auto depth) {
    return (test.bshd ? (b * rows + row) * heads + head
                      : (b * heads + head) * rows + row) *
           depth;
  };
  for (int replay = 0; replay < 2; ++replay) {
    std::vector<double> q(query.element_count().value());
    std::vector<double> k(key.element_count().value());
    std::vector<double> v(value.element_count().value());
    for (std::int64_t b = 0; b < test.batch; ++b) {
      for (std::int64_t r = 0; r < test.queries; ++r)
        for (std::int64_t h = 0; h < test.heads; ++h)
          q[index(b, r, h, test.queries, test.heads, test.query_depth)] =
              double((b + r + h + replay) % 5 - 2) / 8.;
      for (std::int64_t r = 0; r < test.keys; ++r)
        for (std::int64_t h = 0; h < test.kv_heads; ++h) {
          k[index(b, r, h, test.keys, test.kv_heads, test.query_depth)] =
              double((3 * r + b + h + replay) % 7 - 3) / 8.;
          for (std::int64_t z = 0; z < test.value_depth; ++z)
            v[index(b, r, h, test.keys, test.kv_heads, test.value_depth) + z] =
                double((r + replay) % 11 - 5) / 32. + double(b * 2 + h) / 64. +
                double(z % 5 - 2) / 128.;
        }
    }
    auto &ports = replay ? second : first;
    ports.write(0, query, q);
    ports.write(1, key, k);
    ports.write(2, value, v);
    std::vector<double> mask, sinks;
    if (test.sink) {
      mask.resize(std::size_t(test.queries * test.keys));
      sinks.resize(std::size_t(test.heads));
      for (std::int64_t r = 0; r < test.queries; ++r)
        for (std::int64_t s = 0; s < test.keys; ++s)
          mask[r * test.keys + s] =
              r == test.queries / 2
                  ? -std::numeric_limits<double>::infinity()
                  : double((r + 2 * s + replay) % 7 - 3) / 16.;
      for (std::int64_t h = 0; h < test.heads; ++h)
        sinks[h] = double((h + replay) % 5 - 2) / 4.;
      ports.write(3, descriptions[3], mask);
      ports.write(4, descriptions[4], sinks);
    }
    const auto before = stats.snapshot();
    const auto submitted = region.submit(0, queue, ports, recipe.inputs.size());
    ASSERT_TRUE(submitted.ok()) << submitted.message();
    const auto after = stats.snapshot();
    EXPECT_EQ(after.total_allocs, before.total_allocs);
    EXPECT_EQ(after.total_frees, before.total_frees);
    // A second independent slot installs its first real mappings separately.
    const auto independent =
        region.submit(1, queue, ports, recipe.inputs.size());
    ASSERT_TRUE(independent.ok()) << independent.message();
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    const auto actual = ports.read(output, result_id);
    const auto negative = ports.read(output, result_id + 1);
    double max_error = 0, max_excess = 0;
    for (std::int64_t b = 0; b < test.batch; ++b)
      for (std::int64_t r = 0; r < test.queries; ++r)
        for (std::int64_t h = 0; h < test.heads; ++h) {
          const auto kh = h / (test.heads / test.kv_heads);
          const auto end =
              test.causal ? test.keys - test.queries + r + 1 : test.keys;
          const auto q0 =
              q[index(b, r, h, test.queries, test.heads, test.query_depth)];
          std::vector<double> weights(std::size_t(end), 0.);
          double maximum = test.sink ? sinks[h] : -INFINITY;
          for (std::int64_t s = 0; s < end; ++s) {
            weights[s] = q0 *
                             k[index(b, s, kh, test.keys, test.kv_heads,
                                     test.query_depth)] *
                             double(test.scale) +
                         (test.sink ? mask[r * test.keys + s] : 0.);
            maximum = (std::max)(maximum, weights[s]);
          }
          double sum = test.sink ? std::exp(sinks[h] - maximum) : 0.;
          double weighted_token = 0;
          for (std::int64_t s = 0; s < end; ++s) {
            weights[s] = std::exp(weights[s] - maximum);
            sum += weights[s];
            weighted_token += weights[s] * double((s + replay) % 11 - 5) / 32.;
          }
          const auto mass =
              test.sink ? 1. - std::exp(sinks[h] - maximum) / sum : 1.;
          for (std::int64_t z = 0; z < test.value_depth; ++z) {
            auto expected =
                weighted_token / sum +
                mass * (double(b * 2 + kh) / 64. + double(z % 5 - 2) / 128.);
            expected =
                test.type == c::DataType::kFloat32
                    ? double(float(expected))
                    : half_test::decode(half_test::encode<Half>(expected));
            const auto at =
                index(b, r, h, test.queries, test.heads, test.value_depth) + z;
            const auto tolerance = test.type == c::DataType::kFloat32
                                       ? 6e-6 * std::abs(expected) + 2e-6
                                       : .001 * std::abs(expected) + 2e-7;
            max_error = (std::max)(max_error, std::abs(actual[at] - expected));
            max_excess =
                (std::max)(max_excess,
                           std::abs(actual[at] - expected) - tolerance);
            ASSERT_TRUE(std::isfinite(actual[at]));
            ASSERT_EQ(negative[at], -actual[at]);
          }
        }
    std::cout << "math_attention case=" << test.sink << '/' << test.query_depth
              << '/' << test.value_depth << " replay=" << replay
              << " max_abs=" << max_error << " budget_excess=" << max_excess
              << " allocator_live_bytes=" << stats.snapshot().live_bytes
              << '\n';
    EXPECT_EQ(max_excess, 0.);
    if (replay == 0)
      retained = actual;
    else
      EXPECT_EQ(first.read(output, result_id), retained);
  }
}

d::Recipe sink_recipe(std::int64_t queries, std::int64_t keys) {
  auto dense = [](auto shape) {
    return c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
  };
  const auto q = dense(std::array<std::int64_t, 4>{1, queries, 64, 512});
  const auto kv = dense(std::array<std::int64_t, 4>{1, keys, 1, 512});
  const auto mask = dense(std::array{queries, keys});
  const auto sinks = dense(std::array<std::int64_t, 1>{64});
  return {
      .values = {q, kv, kv, mask, sinks, q},
      .nodes = {{c::model_ops::sink_attention(.125F), {0, 1, 2, 3, 4}, {5}}},
      .inputs = {0, 1, 2, 3, 4},
      .outputs = {5}};
}
TEST(GgmlCpuMathAttention, SinkMaskTailPublishedIntermediateAndRebind) {
  check_math_attention(
      {c::DataType::kFloat32, 1, 3, 17, 64, 1, 512, 512, .125F, false, true});
  check_math_attention({c::DataType::kFloat32, 1, 65, 2049, 64, 1, 512, 512,
                        .125F, false, true});
}
TEST(GgmlCpuMathAttention, LayoutsBatchGqaAndOutputDepth) {
  check_math_attention({c::DataType::kFloat32, 2, 257, 273, 4, 1, 8, 16, -.5F,
                        true, false, false});
  check_math_attention({c::DataType::kFloat32, 2, 513, 129, 4, 1, 8, 16, 0.F,
                        false, false, true});
}
TEST(GgmlCpuMathAttention, ColdQuotaFailureRollsBackAndCanRecover) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  hyle::mem::QuotaAllocator quota(stats.get(), 1ULL << 30);
  {
    RawProgram program(quota.get());
    ASSERT_TRUE(program.initialize(sink_recipe(65, 2049)).ok());
    const auto before = quota.used();
    for (const auto available : {std::size_t{0}, std::size_t{1 << 20}}) {
      auto hold =
          quota.try_reserve_bytes(quota.remaining() - available).value();
      const auto result = program.add_slot();
      EXPECT_TRUE(result.IsResourceExhausted()) << result.message();
      hold.reset();
      EXPECT_EQ(quota.used(), before);
    }
    ASSERT_TRUE(program.add_slot().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
TEST(GgmlCpuMathAttention, ColdSinkAllocationUsesBoundedScores) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    RawProgram program(stats.get());
    auto recipe = sink_recipe(129, 2049);
    ASSERT_TRUE(program.initialize(recipe).ok());
    const auto before = stats.snapshot();
    ASSERT_TRUE(program.add_slot().ok());
    const auto after = stats.snapshot();
    const auto allocated = after.live_bytes - before.live_bytes;
    std::size_t caller = 0;
    for (auto id : recipe.inputs)
      caller += recipe.values[id].required_bytes().value();
    caller += recipe.values.back().required_bytes().value();
    std::cout << "cold_sink_allocator_bytes=" << allocated
              << " caller_logical_required_bytes=" << caller
              << " single_scores_bytes=" << 4ULL * 64 * 129 * 2049 << '\n';
    EXPECT_LT(allocated, 4ULL * 64 * 129 * 2049);
  }
  EXPECT_TRUE(stats.balanced());
}
TEST(GgmlCpuMathAttention, NativeColdPlannerFailuresAndFirstBindDoNotAllocate) {
  auto arena = c::ggml_runtime::MetadataArena::for_graph(
                   8, 8, hyle::mem::system_allocator())
                   .value();
  auto *ctx = arena.get();
  auto *input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
  auto *middle = ggml_scale(ctx, input, 2.F);
  auto *output = ggml_neg(ctx, middle);
  auto *graph = ggml_new_graph_custom(ctx, 8, false);
  ggml_set_input(input);
  ggml_set_output(output);
  ggml_build_forward_expand(graph, output);
  auto *xb = ggml_backend_cpu_buffer_from_ptr(nullptr, 128);
  auto *yb = ggml_backend_cpu_buffer_from_ptr(nullptr, 128);
  input->buffer = xb;
  output->buffer = yb;
  alignas(64) std::array<float, 32> x{}, y{};
  int allocations = 0;
  for (int fail = 0; fail <= allocations; ++fail) {
    input->data = nullptr;
    output->data = nullptr;
    middle->data = nullptr;
    middle->buffer = nullptr;
    hyle_test_gallocr_fail_at(fail);
    auto *allocator = hyle_test_gallocr_new(ggml_backend_cpu_buffer_type());
    const auto reserved =
        allocator && hyle_test_gallocr_reserve(allocator, graph);
    if (fail == 0) {
      ASSERT_TRUE(reserved);
      allocations = hyle_test_gallocr_calls();
      input->data = x.data();
      xb->context = x.data();
      output->data = y.data();
      yb->context = y.data();
      ASSERT_TRUE(hyle_test_gallocr_alloc_graph(allocator, graph));
      EXPECT_EQ(hyle_test_gallocr_calls(), allocations);
      std::cout << "native_cold_c_malloc_calls=" << allocations
                << " after_first_bind_calls=" << hyle_test_gallocr_calls()
                << '\n';
    } else
      EXPECT_FALSE(reserved);
    hyle_test_gallocr_free(allocator);
    EXPECT_EQ(hyle_test_gallocr_live(), 0);
  }
  ggml_backend_buffer_free(xb);
  ggml_backend_buffer_free(yb);
  hyle_test_gallocr_fail_at(0);
}
d::Recipe negative_chain(std::int64_t rows, std::int64_t columns,
                         bool publish_middle = false) {
  const auto description = c::TensorDesc::dense(
      c::DataType::kFloat32, std::array{rows, columns}).value();
  d::Recipe recipe{.values = std::vector<c::TensorDesc>(65, description),
                   .inputs = {0},
                   .outputs = publish_middle ? std::vector<std::uint32_t>{32, 64}
                                             : std::vector<std::uint32_t>{64}};
  for (std::uint32_t i = 1; i <= 64; ++i)
    recipe.nodes.push_back(
        {c::Operation(c::Unary{c::UnaryKind::kNeg}), {i - 1}, {i}});
  return recipe;
}

TEST(GgmlCpuStorage, OrdinaryChainColdAllocationIsBounded) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    RawProgram program(stats.get());
    ASSERT_TRUE(program.initialize(negative_chain(256, 1024)).ok());
    const auto before = stats.snapshot();
    ASSERT_TRUE(program.add_slot().ok());
    const auto bytes = stats.snapshot().live_bytes - before.live_bytes;
    std::cout << "ordinary_chain_cold_allocator_bytes=" << bytes
              << " legacy_owned_payload_bytes=" << 63ULL * 1024 * 1024
              << " caller_logical_required_bytes=" << 2ULL * 1024 * 1024
              << '\n';
    EXPECT_LT(bytes, 8ULL * 1024 * 1024);
  }
  EXPECT_TRUE(stats.balanced());
}

d::Recipe encoded_negative_chain(std::int64_t rows, std::int64_t columns,
                                  c::CacheStorage storage, bool publish = false) {
  auto recipe = negative_chain(rows, columns);
  const auto operation = c::Operation::custom(c::model_ops::CacheEncode{storage});
  recipe.values.push_back(c::infer_single_output(
      operation, std::span(&recipe.values[64], 1)).value());
  recipe.nodes.push_back({operation, {64}, {65}});
  recipe.outputs = publish ? std::vector<std::uint32_t>{31, 65}
                           : std::vector<std::uint32_t>{65};
  return recipe;
}

TEST(GgmlCpuStorage, CacheEncodeChainColdAllocationIsBounded) {
  auto stats = hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    RawProgram program(stats.get());
    ASSERT_TRUE(program.initialize(encoded_negative_chain(
        256, 1024, c::CacheStorage::q8_0().value())).ok());
    const auto before = stats.snapshot();
    ASSERT_TRUE(program.add_slot().ok());
    const auto bytes = stats.snapshot().live_bytes - before.live_bytes;
    std::cout << "cache_encode_chain_cold_allocator_bytes=" << bytes
              << " legacy_owned_float_payload_bytes=" << 64ULL * 1024 * 1024
              << " graph_executed=false\n";
    EXPECT_LT(bytes, 8ULL * 1024 * 1024);
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(GgmlCpuStorage, CacheEncodeKeepsPublishedRootAndIndependentSlotBindings) {
  for (const auto storage : {c::CacheStorage::q8_0().value(),
                             c::CacheStorage::float8(.5F).value()}) {
    RawProgram program(hyle::mem::system_allocator());
    ASSERT_TRUE(program.initialize(encoded_negative_chain(4, 32, storage, true)).ok());
    ASSERT_TRUE(program.add_slot().ok());
    ASSERT_TRUE(program.add_slot().ok());
    std::vector<c::TensorDesc> descriptions = program.inputs;
    descriptions.insert(descriptions.end(), program.outputs.begin(), program.outputs.end());
    AttentionPorts first(descriptions), second(descriptions);
    auto context = c::Context::create(c::cpu::device(), c::ggml_cpu::provider()).value();
    auto queue = context.create_queue().value();
    std::vector<std::uint8_t> retained;
    for (unsigned replay = 0; replay != 2; ++replay) {
      auto &ports = replay ? second : first;
      std::vector<double> input(128), negative(128);
      for (std::size_t i = 0; i != input.size(); ++i) {
        input[i] = ((i / 32 + replay) % 2 ? -1. : 1.) * double(i / 32 % 3 + 1) / 4.;
        negative[i] = -input[i];
      }
      ports.write(0, descriptions[0], input);
      std::vector<std::uint8_t> expected(descriptions[2].required_bytes().value());
      for (std::size_t block = 0; block != 4; ++block) {
        const auto value = input[block * 32];
        if (storage == c::CacheStorage::q8_0().value()) {
          const auto scale = half_test::encode<Half>(float(std::abs(value)) / 127.F).bits;
          expected[block * 34] = std::uint8_t(scale);
          expected[block * 34 + 1] = std::uint8_t(scale >> 8);
          std::fill_n(expected.begin() + block * 34 + 2, 32,
                      std::uint8_t(value < 0 ? -127 : 127));
        } else {
          constexpr std::array<std::uint8_t, 3> codes{0x30, 0x38, 0x3c};
          std::fill_n(expected.begin() + block * 32, 32,
                      std::uint8_t(codes[block % 3] | (value < 0 ? 0x80 : 0)));
        }
      }
      for (std::size_t slot = 0; slot != 2; ++slot) {
        const auto status = program.submit(slot, queue, ports, 1);
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(ports.read(descriptions[0], 0), input);
        EXPECT_EQ(ports.read(descriptions[1], 1), negative);
        std::vector<std::uint8_t> actual(expected.size());
        ASSERT_TRUE(ports.views[2].read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(actual, expected);
      }
      if (!replay) retained = expected;
      else {
        std::vector<std::uint8_t> actual(retained.size());
        ASSERT_TRUE(first.views[2].read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(actual, retained);
      }
    }
  }
}

TEST(GgmlCpuStorage, PublishedViewRootAndTwoSlotsRebindWithoutAllocation) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    RawProgram program(stats.get());
    ASSERT_TRUE(program.initialize(negative_chain(4, 8, true)).ok());
    ASSERT_TRUE(program.add_slot().ok());
    ASSERT_TRUE(program.add_slot().ok());
    std::vector<c::TensorDesc> descriptions = program.inputs;
    descriptions.insert(descriptions.end(), program.outputs.begin(),
                        program.outputs.end());
    AttentionPorts first(descriptions), second(descriptions);
    auto queue_context =
        c::Context::create(c::cpu::device(), c::ggml_cpu::provider()).value();
    auto queue = queue_context.create_queue().value();
    std::vector<double> retained;
    for (int replay = 0; replay != 2; ++replay) {
      auto &ports = replay == 0 ? first : second;
      std::vector<double> input(32);
      for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = double(int(i) - 17 + replay) / 4.;
      ports.write(0, descriptions[0], input);
      for (std::size_t slot = 0; slot != 2; ++slot) {
        const auto before = stats.snapshot();
        const auto status = program.submit(slot, queue, ports, 1);
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
        EXPECT_EQ(stats.snapshot().total_frees, before.total_frees);
        EXPECT_EQ(ports.read(descriptions[0], 0), input);
        EXPECT_EQ(ports.read(descriptions[1], 1), input);
        EXPECT_EQ(ports.read(descriptions[2], 2), input);
      }
      if (replay == 0)
        retained = input;
      else
        EXPECT_EQ(first.read(descriptions[1], 1), retained);
    }
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(GgmlCpuStorage, BranchConstantsAndNanObserversRetainTheirSources) {
  const auto description = c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 2>{4, 8}).value();
  auto narrow = c::TensorDesc::dense(c::DataType::kBFloat16,
                                    std::array<std::int64_t, 2>{4, 8}).value();
  d::Recipe recipe{
      .values = std::vector<c::TensorDesc>(10, description),
      .nodes = {{c::Operation(c::Unary{c::UnaryKind::kNeg}), {0}, {1}},
                {c::Operation(c::Unary{c::UnaryKind::kRelu}), {1}, {2}},
                {c::Operation(c::Unary{c::UnaryKind::kNeg}), {2}, {3}},
                {c::Operation(c::Fill{
                     .output_shape = c::TensorShape::create(
                                         description.dimensions()).value(),
                     .output_type = c::DataType::kFloat32,
                     .value = c::ScalarValue(2.F)}),
                 {},
                 {4}},
                {c::Operation(c::Binary{c::BinaryKind::kAdd}), {3, 4}, {5}},
                {c::Operation(c::Cast{c::DataType::kBFloat16}), {5}, {6}},
                {c::Operation(c::Cast{c::DataType::kFloat32}), {6}, {7}},
                {c::Operation(c::Softmax{}), {7}, {8}},
                {c::Operation(c::Unary{c::UnaryKind::kNeg}), {7}, {9}}},
      .inputs = {0},
      .outputs = {2, 8, 9}};
  recipe.values[6] = narrow;
  RawProgram program(hyle::mem::system_allocator());
  const auto initialized = program.initialize(std::move(recipe));
  ASSERT_TRUE(initialized.ok()) << initialized.message();
  ASSERT_TRUE(program.add_slot().ok());
  std::array descriptions{description, description, description, description};
  AttentionPorts ports(descriptions);
  auto queue_context =
      c::Context::create(c::cpu::device(), c::ggml_cpu::provider()).value();
  auto queue = queue_context.create_queue().value();
  for (int replay = 0; replay != 2; ++replay) {
    std::vector<double> input(32), relu(32), shifted(32);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = double(int(i % 13) - 6 + replay) / 4.;
      relu[i] = (std::max)(-input[i], 0.);
      shifted[i] = 2. - relu[i];
    }
    input[10] = std::numeric_limits<double>::quiet_NaN();
    ports.write(0, description, input);
    const auto status = program.submit(0, queue, ports, 1);
    ASSERT_TRUE(status.ok()) << status.message();
    const auto middle = ports.read(description, 1);
    const auto softmax = ports.read(description, 2);
    const auto branch = ports.read(description, 3);
    for (std::size_t row = 0; row != 4; ++row) {
      double denominator = 0.;
      for (std::size_t column = 0; column != 8; ++column)
        denominator += std::exp(shifted[row * 8 + column]);
      for (std::size_t column = 0; column != 8; ++column) {
        const auto i = row * 8 + column;
        if (i == 10) {
          EXPECT_TRUE(std::isnan(middle[i]));
          EXPECT_TRUE(std::isnan(branch[i]));
        } else {
          EXPECT_EQ(middle[i], relu[i]);
          EXPECT_EQ(branch[i], -shifted[i]);
        }
        if (row == 1)
          EXPECT_TRUE(std::isnan(softmax[i]));
        else
          EXPECT_NEAR(softmax[i], std::exp(shifted[i]) / denominator, 2e-6);
      }
    }
  }
}

TEST(GgmlCpuStorage, ColdQuotaFailureRollsBackAndCanRecover) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  hyle::mem::QuotaAllocator quota(stats.get(), 1ULL << 30);
  {
    RawProgram program(quota.get());
    ASSERT_TRUE(program.initialize(negative_chain(256, 1024)).ok());
    const auto before = quota.used();
    for (const auto available : {std::size_t{0}, std::size_t{1 << 20}}) {
      auto hold = quota.try_reserve_bytes(quota.remaining() - available).value();
      const auto result = program.add_slot();
      EXPECT_TRUE(result.IsResourceExhausted()) << result.message();
      hold.reset();
      EXPECT_EQ(quota.used(), before);
    }
    ASSERT_TRUE(program.add_slot().ok());
  }
  EXPECT_TRUE(stats.balanced());
}

void compare_native_causal_mask(ggml_type type, std::int64_t queries,
                                std::int64_t keys, std::int64_t batch,
                                float scale) {
  const auto selected = c::ggml_runtime::select_cpu_program();
  ASSERT_TRUE(selected.has_value());
  const auto &native = **selected;
  native.initialize();
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({8 * 1024 * 1024, nullptr, false}), &ggml_free);
  ASSERT_NE(context, nullptr);
  constexpr std::int64_t depth = 16, value_depth = 24, heads = 4, kv_heads = 2;
  auto *q = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, depth,
                               queries, heads, batch);
  auto *k = ggml_new_tensor_4d(context.get(), type, depth, keys, kv_heads, batch);
  auto *v = ggml_new_tensor_4d(context.get(), type, value_depth, keys,
                               kv_heads, batch);
  for (std::int64_t i = 0; i != ggml_nelements(q); ++i)
    static_cast<float *>(q->data)[i] = float(i % 17 - 8) / 16.F;
  const auto fill = [](ggml_tensor *tensor, int period) {
    for (std::int64_t i = 0; i != ggml_nelements(tensor); ++i) {
      const float x = float((i * 7 + i / 13) % period - period / 2) / 32.F;
      if (tensor->type == GGML_TYPE_F16)
        static_cast<ggml_fp16_t *>(tensor->data)[i] = ggml_fp32_to_fp16(x);
      else if (tensor->type == GGML_TYPE_BF16)
        static_cast<ggml_bf16_t *>(tensor->data)[i] = ggml_fp32_to_bf16(x);
      else
        static_cast<float *>(tensor->data)[i] = x;
    }
  };
  fill(k, 19);
  fill(v, 23);
  auto *mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, keys, queries);
  for (std::int64_t row = 0; row != queries; ++row)
    for (std::int64_t key = 0; key != keys; ++key)
      static_cast<ggml_fp16_t *>(mask->data)[row * keys + key] =
          ggml_fp32_to_fp16(key <= keys - queries + row ? 0.F : -INFINITY);
  auto *expected = ggml_flash_attn_ext(context.get(), q, k, v, mask,
                                       scale, 0.F, 0.F);
  auto *actual = ggml_flash_attn_ext(context.get(), q, k, v, nullptr,
                                     scale, 0.F, 0.F);
  if (queries > 1)
    actual->op_params[7] = 0x48594341;
  ASSERT_TRUE(ggml_prec_set_acc(expected, GGML_PREC_F32));
  ASSERT_TRUE(ggml_prec_set_acc(actual, GGML_PREC_F32));
  auto *graph = ggml_new_graph_custom(context.get(), 32, false);
  ggml_build_forward_expand(graph, expected);
  ggml_build_forward_expand(graph, actual);
  auto pool_options = ggml_threadpool_params_default(3);
  int native_error = 0;
  std::unique_ptr<ggml_threadpool, decltype(native.free_pool)> pool(
      native.create_pool(&pool_options, &native_error), native.free_pool);
  ASSERT_NE(pool, nullptr) << native_error;
  auto plan = native.plan(graph, 3, pool.get());
  std::vector<std::uint8_t> workspace(plan.work_size);
  plan.work_data = workspace.data();
  ASSERT_EQ(native.compute(graph, &plan), GGML_STATUS_SUCCESS);
  const auto count = ggml_nelements(actual);
  const auto *reference = static_cast<const float *>(expected->data);
  const auto *observed = static_cast<const float *>(actual->data);
  for (std::int64_t i = 0; i != count; ++i)
    ASSERT_EQ(observed[i], reference[i]) << "index=" << i;
  std::cout << "native_causal_type=" << type << " Q=" << queries
            << " KV=" << keys << " B=" << batch << " scale=" << scale
            << " compared_f32_elements=" << count << " exact_match=true\n";
}

TEST(GgmlCpuMathAttention, ImplicitCausalMatchesNativeTiledMask) {
  // Both locked cores use 64x64 tiles; cross query/KV tails and per-head batches.
  compare_native_causal_mask(GGML_TYPE_F16, 129, 193, 2, -.25F);
  compare_native_causal_mask(GGML_TYPE_F16, 65, 65, 2, 0.F);
#if defined(HYLE_GGML_CPU_F32_FLASH)
  compare_native_causal_mask(GGML_TYPE_F32, 65, 69, 2, .25F);
#endif
}

TEST(GgmlCpuMathAttention, ImplicitCausalMatchesNativeVectorMask) {
  // Preserve original Half PV, and BF16's existing non-tiled implementation.
  compare_native_causal_mask(GGML_TYPE_F16, 17, 33, 2, .125F);
  compare_native_causal_mask(GGML_TYPE_BF16, 65, 69, 2, .125F);
}

TEST(GgmlCpuMathAttention, DecodeOmitsZeroMaskAndRetainsNativeSplit) {
  // Q1 has no tag. Three native threads and KV>=512 retain the F16 split path.
  compare_native_causal_mask(GGML_TYPE_F16, 1, 513, 1, .25F);
  compare_native_causal_mask(GGML_TYPE_BF16, 1, 513, 1, .25F);
}

template <class Value> void check_public_causal_rebind() {
  constexpr std::int64_t batch = 2, queries = 65, keys = 69, heads = 4,
                         kv_heads = 2, depth = 16, value_depth = 24;
  const auto dense = [](auto dimensions) {
    return c::TensorDesc::dense(half_test::data_type<Value>(), dimensions).value();
  };
  const std::array inputs{
      dense(std::array{batch, queries, heads, depth}),
      dense(std::array{batch, keys, kv_heads, depth}),
      dense(std::array{batch, keys, kv_heads, value_depth})};
  const auto output = dense(std::array{batch, queries, heads, value_depth});
  auto context = c::Context::create(
      c::cpu::device(), c::ggml_cpu::provider({.native_threads = 3})).value();
  auto queue = context.create_queue().value();
  const c::Operation operation(c::ScaledDotProductAttention{
      .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
      .scale = -.25F,
      .causal = true});
  auto prepared = c::prepare(context, operation, inputs, std::span(&output, 1),
                             hyle::mem::system_resource()).value();
  ASSERT_EQ(prepared.provider().id, c::ggml_cpu::kProviderId);
  auto invocation = prepared.create_invocation().value();
  std::array<std::vector<hyle::mem::Buffer>, 2> retained_owners;
  for (int replay = 0; replay != 2; ++replay) {
    auto &owners = retained_owners[replay];
    for (const auto &desc : inputs)
      owners.push_back(hyle::mem::Buffer::create(
          hyle::mem::system_resource(), desc.required_bytes().value(), 64).value());
    owners.push_back(hyle::mem::Buffer::create(
        hyle::mem::system_resource(), output.required_bytes().value(), 64).value());
    // Q/K are exactly zero. The FP64 oracle is the mean of the allowed V prefix.
    for (std::size_t port = 0; port != 2; ++port) {
      const std::vector<Value> zero(inputs[port].element_count().value(),
                                    half_test::encode<Value>(0.));
      ASSERT_TRUE(owners[port].view().write(std::as_bytes(std::span(zero))).ok());
    }
    const std::array views{
        c::ConstTensorView::create(inputs[0], owners[0].view()).value(),
        c::ConstTensorView::create(inputs[1], owners[1].view()).value(),
        c::ConstTensorView::create(inputs[2], owners[2].view()).value()};
    const auto destination = c::TensorView::create(output, owners[3].view()).value();
    std::vector<Value> values(inputs[2].element_count().value());
    for (std::int64_t b = 0; b != batch; ++b)
      for (std::int64_t row = 0; row != keys; ++row)
        for (std::int64_t head = 0; head != kv_heads; ++head)
          for (std::int64_t z = 0; z != value_depth; ++z)
            values[((b * keys + row) * kv_heads + head) * value_depth + z] =
                half_test::encode<Value>(double(row) / 128. + double(b) / 8. +
                                         double(head) / 16. + double(replay) / 32.);
    ASSERT_TRUE(owners[2].view().write(std::as_bytes(std::span(values))).ok());
    auto completion = invocation.submit(queue, views, std::span(&destination, 1))
                          .value();
    ASSERT_TRUE(completion.wait().ok());
    const auto observed = half_test::read<Value>(owners[3]);
    for (std::int64_t b = 0; b != batch; ++b)
      for (std::int64_t row = 0; row != queries; ++row)
        for (std::int64_t head = 0; head != heads; ++head) {
          const auto kv_head = head / (heads / kv_heads);
          const double expected = double(keys - queries + row) / 256. +
                                  double(b) / 8. + double(kv_head) / 16. +
                                  double(replay) / 32.;
          for (std::int64_t z = 0; z != value_depth; ++z)
            half_test::expect_near(
                observed[((b * queries + row) * heads + head) * value_depth + z],
                expected);
        }
  }
}

TEST(GgmlCpuMathAttention, PublicCausalHalfAndBfloat16Rebind) {
  check_public_causal_rebind<hyle::simd::float16_t>();
  check_public_causal_rebind<hyle::simd::bfloat16_t>();
}
} // namespace
