#include <gtest/gtest.h>
#include <hyle/compute/cuda.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/model_ops.h>
#include <hyle/compute/prepared_operation.h>
#include <hyle/compute/providers/cuda.h>
#include <hyle/graph/executable.h>
#include "../compute_models/row_cache_indexer_fixture.h"
#include <array>
#include <cstdio>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using Fixture = row_cache_indexer_test::Fixture;
struct Tensor final {
  m::Buffer owner;
  c::TensorView view;
  Tensor(c::TensorDesc desc, m::MemoryResourceRef resource,
         std::size_t offset = 0)
      : owner(m::Buffer::create(
                  resource,
                  (std::max)(std::size_t{1}, *desc.required_bytes() + offset),
                  256)
                  .value()),
        view(c::TensorView::create(
                 desc,
                 owner.view().subview(offset, *desc.required_bytes()).value())
                 .value()) {}
  template <class T> void write(const std::vector<T> &v) {
    ASSERT_TRUE(view.storage().write(std::as_bytes(std::span(v))).ok());
  }
  std::vector<float> read() {
    std::vector<float> x(view.storage().size() / 4);
    EXPECT_TRUE(view.storage().read(std::as_writable_bytes(std::span(x))).ok());
    return x;
  }
};
void expect(const std::vector<float> &actual,
            const std::vector<float> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_FLOAT_EQ(actual[i], expected[i]) << i;
}
struct Inputs final {
  Tensor q, bank, w, ranges;
  Inputs(const Fixture &f, m::MemoryResourceRef resource)
      : q(f.descriptions[0], resource), bank(f.descriptions[1], resource, 1),
        w(f.descriptions[2], resource), ranges(f.descriptions[3], resource) {}
  void update(const Fixture &f) {
    q.write(f.query);
    bank.write(f.bank);
    w.write(f.weights);
    ranges.write(f.ranges);
  }
  auto views() const {
    return std::array{q.view.as_const(), bank.view.as_const(),
                      w.view.as_const(), ranges.view.as_const()};
  }
};

TEST(CudaRowCacheIndexer,
     DefaultDirectKeepsBfloatStagesRangesAndIndependentSlots) {
  auto context = c::cuda::create_context().value();
  auto memory =
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)
          .value();
  Fixture f;
  Inputs inputs(f, memory.ref());
  const auto operation = c::model_ops::row_cache_indexer_scores();
  const auto out = c::infer_single_output(operation, f.descriptions).value();
  auto prepared = c::prepare(context, operation, f.descriptions,
                             std::array{out}, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::cuda_native::kProviderId);
  EXPECT_EQ(prepared->runtime_properties().mode,
            c::ProviderRuntimeMode::kSharedQueue);
  const auto requirement = prepared->workspace_requirement();
  std::array workspace{
      m::Buffer::create(memory.ref(), requirement.size, requirement.alignment)
          .value(),
      m::Buffer::create(memory.ref(), requirement.size, requirement.alignment)
          .value()};
  std::array slots{prepared->create_invocation(workspace[0].view()).value(),
                   prepared->create_invocation(workspace[1].view()).value()};
  std::array outputs{Tensor(out, memory.ref()), Tensor(out, memory.ref())};
  std::array queues{context.create_queue().value(),
                    context.create_queue().value()};
  std::vector<float> old;
  for (int phase = 0; phase < 2; ++phase) {
    f.update(phase);
    inputs.update(f);
    const auto expected = f.expected();
    EXPECT_NE(expected, f.expected(false));
    auto done = slots[phase].submit(queues[phase], inputs.views(),
                                    std::array{outputs[phase].view});
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    expect(outputs[phase].read(), expected);
    // A repeat on the same slot reuses descriptors, handle and exact workspace.
    ASSERT_TRUE(slots[phase]
                    .submit(queues[phase], inputs.views(),
                            std::array{outputs[phase].view})
                    ->wait()
                    .ok());
    expect(outputs[phase].read(), expected);
    if (!phase)
      old = outputs[0].read();
    else
      EXPECT_EQ(outputs[0].read(), old);
  }
  const auto sentinel = outputs[1].read();
  f.ranges.back() = std::int32_t(f.rows + 1);
  inputs.ranges.write(f.ranges);
  auto failed = prepared->create_invocation(workspace[1].view()).value();
  auto queue = context.create_queue().value();
  auto done = failed.submit(queue, inputs.views(), std::array{outputs[1].view});
  ASSERT_TRUE(done);
  EXPECT_TRUE(done->wait().IsOutOfRange());
  EXPECT_EQ(outputs[1].read(), sentinel);
  EXPECT_EQ(outputs[0].read(), old);
  EXPECT_FALSE(
      failed.submit(queue, inputs.views(), std::array{outputs[1].view}));
  std::printf(
      "CUDA packed indexer T3 L513 keytile256 caller-workspace=%zu B x2; vendor-private bytes unknown\n",
      requirement.size);
}

TEST(CudaRowCacheIndexer,
     DefaultGraphPublishesScoresAndDownstreamAcrossPhases) {
  auto context = c::cuda::create_context().value();
  auto memory =
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)
          .value();
  Fixture f;
  Inputs inputs(f, memory.ref());
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 4> ports;
  for (unsigned i = 0; i < 4; ++i)
    ports[i] = builder.input(f.descriptions[i]).value();
  const auto score = builder
                         .add(c::model_ops::row_cache_indexer_scores(),
                              ports[0], ports[1], ports[2], ports[3])
                         .value();
  const auto negative =
      builder.add(c::Unary{c::UnaryKind::kNeg}, score).value();
  ASSERT_TRUE(builder.publish(score).ok());
  ASSERT_TRUE(builder.publish(negative).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array storage{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{hyle::graph::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {storage, targets, {}});
  ASSERT_TRUE(compiled) << compiled.error().message();
  bool native = false;
  for (std::size_t i = 0; i < compiled->report.step_count(); ++i)
    native |=
        compiled->report.step(i)->provider.id == c::cuda_native::kProviderId;
  EXPECT_TRUE(native);
  const auto desc =
      c::infer_single_output(c::model_ops::row_cache_indexer_scores(),
                             f.descriptions)
          .value();
  std::array scores{Tensor(desc, memory.ref()), Tensor(desc, memory.ref())};
  std::array negatives{Tensor(desc, memory.ref()), Tensor(desc, memory.ref())};
  std::array invocations{compiled->executable.create_invocation().value(),
                         compiled->executable.create_invocation().value()};
  auto queue = context.create_queue().value();
  std::vector<float> old;
  for (int phase = 0; phase < 2; ++phase) {
    f.update(phase);
    inputs.update(f);
    auto expected = f.expected();
    auto &inv = invocations[phase];
    const auto views = inputs.views();
    for (unsigned i = 0; i < 4; ++i)
      ASSERT_TRUE(inv.bind(ports[i], views[i]).ok());
    ASSERT_TRUE(inv.bind(score, scores[phase].view).ok());
    ASSERT_TRUE(inv.bind(negative, negatives[phase].view).ok());
    auto done = inv.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    expect(scores[phase].read(), expected);
    for (auto &x : expected)
      x = -x;
    expect(negatives[phase].read(), expected);
    if (!phase)
      old = scores[0].read();
    else
      EXPECT_EQ(scores[0].read(), old);
  }
}

TEST(CudaRowCacheIndexer, EmptyIntervalsAndEmptyAxesKeepValidation) {
  auto context = c::cuda::create_context().value();
  auto memory =
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)
          .value();
  for (const auto dims :
       {std::array<std::int64_t, 2>{513, 3}, std::array<std::int64_t, 2>{0, 3},
        std::array<std::int64_t, 2>{513, 0}}) {
    SCOPED_TRACE(::testing::Message() << "L=" << dims[0] << " T=" << dims[1]);
    Fixture f(dims[0], dims[1]);
    std::fill(f.ranges.begin(), f.ranges.end(), 0);
    std::fill(f.bank.begin(), f.bank.end(), 255);
    Inputs inputs(f, memory.ref());
    inputs.update(f);
    const auto op = c::model_ops::row_cache_indexer_scores();
    const auto out = c::infer_single_output(op, f.descriptions).value();
    Tensor output(out, memory.ref());
    auto prepared =
        c::prepare(context, op, f.descriptions, std::array{out}, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    const auto requirement = prepared->workspace_requirement();
    auto workspace =
        m::Buffer::create(memory.ref(),
                          (std::max)(std::size_t{1}, requirement.size),
                          requirement.alignment)
            .value();
    auto invocation = prepared->create_invocation(workspace.view()).value();
    auto queue = context.create_queue().value();
    ASSERT_TRUE(
        invocation.submit(queue, inputs.views(), std::array{output.view})
            ->wait()
            .ok());
    expect(output.read(),
           std::vector<float>(std::size_t(f.queries * f.rows),
                              -std::numeric_limits<float>::infinity()));
    if (!f.rows && f.queries) {
      f.ranges.back() = 1;
      inputs.ranges.write(f.ranges);
      auto failed = prepared->create_invocation(workspace.view()).value();
      auto badqueue = context.create_queue().value();
      auto done =
          failed.submit(badqueue, inputs.views(), std::array{output.view});
      ASSERT_TRUE(done);
      EXPECT_TRUE(done->wait().IsOutOfRange());
    }
  }
}
} // namespace
