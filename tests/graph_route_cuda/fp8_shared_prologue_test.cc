#include "../compute_models/d5_dynamic_fp8_fragments.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/cuda.h"

#include <array>
#include <bit>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace fixture = d5_dynamic_fp8_fragments;
namespace fp8 = d5_official_fp8_fragments;
using Type = c::DataType;
using Policy = c::model_ops::Float8ScalePolicy;
constexpr c::RoutePolicyId kSharedPolicy{0x48594c4546503853ULL, 1, 1};
constexpr std::int64_t kRows = 3, kDepth = 384;
using Bytes = std::vector<std::byte>;

template <class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
c::TensorDesc matrix(Type type, std::int64_t columns) {
  return c::TensorDesc::dense(type, std::array{kRows, columns}).value();
}
Bytes read(c::ConstTensorView tensor) {
  Bytes bytes(tensor.description().required_bytes().value());
  check(tensor.storage().read(bytes), "output BufferView::read");
  return bytes;
}
c::ReadLease upload(const c::StorageOwner &owner, std::span<const std::byte> bytes) {
  auto buffer = take(m::Buffer::create(owner.ref(), bytes.size(), 256), "input Buffer::create");
  check(buffer.view().write(bytes), "input upload");
  return take(owner.adopt_buffer(std::move(buffer)), "input StorageOwner::adopt_buffer");
}

struct Contract final {
  bool bfloat = false;
  std::array<Policy, 2> rounding{Policy::kDivide, Policy::kDivide};
  std::array<float, 2> floor{1.0e-10F, 1.0e-10F};
  bool same_value = true;
  bool packed_single = false;
};
struct Logical final {
  Contract contract;
  g::Graph graph;
  c::TensorDesc activation;
  std::array<c::TensorDesc, 2> weights{}, outputs{};
  std::array<g::ValueId, 2> a{}, w{}, y{};
  std::array<g::NodeId, 2> nodes{};
  std::array<c::Operation, 2> operations{};
  std::size_t linears = 2;
};
Logical build(Contract contract) {
  Logical f{.contract = contract};
  f.activation = matrix(contract.bfloat ? Type::kBFloat16 : Type::kFloat32, kDepth);
  f.linears = contract.packed_single ? 1U : 2U;
  auto builder = g::GraphBuilder::create().value();
  f.a[0] = builder.input(f.activation).value();
  f.a[1] = contract.same_value ? f.a[0] : builder.input(f.activation).value();
  for (std::size_t i = 0; i < f.linears; ++i) {
    const std::int64_t n = contract.packed_single ? 34 : i == 0 ? 17 : 129;
    f.weights[i] = c::float8_weight_description(kDepth, n,
        c::Float8ScaleMode::kFloat32Block128).value();
    f.outputs[i] = matrix(f.activation.data_type(), n);
    f.w[i] = builder.input(f.weights[i]).value();
    f.operations[i] = c::model_ops::float8_block_linear(contract.rounding[i], contract.floor[i]);
    f.nodes[i] = builder.add_node(f.operations[i], std::array{f.a[i], f.w[i]}).value();
    f.y[i] = builder.result(f.nodes[i]).value();
  }
  // Logical publication order deliberately disagrees with native Y0/Y1.
  for (std::size_t i = f.linears; i != 0; --i) check(builder.publish(f.y[i - 1]), "publish linear output");
  // Recover the actual source lease after each completed run. It is an input
  // boundary, not a third output of the native pair or a copied payload.
  check(builder.publish(f.a[0]), "publish activation source");
  f.graph = std::move(builder).finish().value();
  return f;
}
std::array<c::ReadLease, 2> weights(const Logical &f, const c::StorageOwner &owner) {
  std::array<c::ReadLease, 2> result;
  for (std::size_t i = 0; i < f.linears; ++i) {
    const fp8::Case shape{nullptr, f.outputs[i].dimensions()[1], kDepth, false, "shared-prologue test"};
    auto source = fp8::projection(shape, kRows);
    result[i] = upload(owner, source.constant_bytes[0]);
  }
  return result;
}
Bytes activation(const Logical &f, int seed) {
  return fixture::input_bytes(fixture::input_values(kRows, kDepth, seed, f.contract.bfloat), f.contract.bfloat);
}
g::Invocation invocation(const g::Executable &program) {
  return take(program.create_invocation({.resources = g::ResourceOptions{}}), "owned Graph Invocation");
}
void selected(const Logical &f, const g::CompileReport &report, bool pair) {
  ASSERT_EQ(report.step_count(), pair ? 1U : f.linears);
  for (std::size_t step = 0; step < report.step_count(); ++step) {
    const auto item = report.step(step).value();
    EXPECT_EQ(item.provider.id, c::cuda_native::kSglangProviderId);
    if (pair) {
      EXPECT_EQ(item.kind, g::CompileStepKind::kProviderRegion);
      EXPECT_EQ(item.policy, kSharedPolicy);
      EXPECT_EQ(item.implementation, c::ProviderImplementationId{1});
      EXPECT_EQ(item.dispatch_count, 3U); // One quantizer and two unchanged GEMMs.
      EXPECT_EQ(item.input_count, 3U);
      EXPECT_EQ(item.output_count, 2U);
      EXPECT_EQ(item.workspace.size, c::float8_activation_description(kRows, kDepth,
          c::Float8ScaleMode::kFloat32Block128)->required_bytes().value());
      ASSERT_EQ(item.node_count, 2U);
      for (const auto node : f.nodes) {
        bool found = false;
        for (std::size_t j = 0; j < item.node_count; ++j)
          found |= report.step_node(step, j).value() == node;
        EXPECT_TRUE(found);
      }
    } else {
      EXPECT_NE(item.policy, kSharedPolicy);
      EXPECT_EQ(item.kind, g::CompileStepKind::kOperationFallback);
      EXPECT_EQ(item.dispatch_count, 2U);
      EXPECT_EQ(item.node_count, 1U);
    }
  }
  // This is the pair's declared workspace, not a claim that the whole Graph
  // peak halves: separate leaf scratch can already reuse the same allocation.
}

// These are the original public standalone linears: two independent native
// quantizers and GEMMs, not another invocation of the new private pair.
std::array<Bytes, 2> direct(const Logical &f, const c::ReadLease &a,
    const std::array<c::ReadLease, 2> &w, c::Context &context,
    c::Queue &queue, const c::StorageOwner &owner) {
  std::array<Bytes, 2> expected;
  for (std::size_t i = 0; i < f.linears; ++i) {
    const std::array inputs{take(a.tensor(f.activation), "Direct activation"),
        take(w[i].tensor(f.weights[i]), "Direct FP8 weight")};
    auto output = take(m::Buffer::create(owner.ref(), f.outputs[i].required_bytes().value(), 256), "Direct output allocation");
    const auto view = c::TensorView::create(f.outputs[i], output.view()).value();
    auto program = take(c::prepare(context, f.operations[i], inputs, std::span(&view, 1), owner.ref()), "original standalone prepare");
    EXPECT_EQ(program.provider().id, c::cuda_native::kSglangProviderId);
    const auto need = program.workspace_requirement();
    auto scratch = take(m::Buffer::create(owner.ref(), need.size, need.alignment), "original standalone workspace");
    auto call = take(program.create_invocation(scratch.view()), "original standalone invocation");
    auto done = take(call.submit(queue, inputs, std::span(&view, 1)), "original standalone submit");
    check(done.wait(), "original standalone Completion::wait");
    expected[i] = read(view.as_const());
  }
  return expected;
}
std::vector<c::LeasePool> bind_case(const Logical &f, g::Invocation &slot,
    const c::ReadLease &a, const std::array<c::ReadLease, 2> &w,
    const c::StorageOwner &owner) {
  check(slot.bind(f.a[0], f.activation, a), "bind activation");
  if (!f.contract.same_value)
    check(slot.bind(f.a[1], f.activation, a), "bind distinct activation ValueId to same allocation");
  std::vector<c::LeasePool> pools;
  for (std::size_t i = 0; i < f.linears; ++i) {
    check(slot.bind(f.w[i], f.weights[i], w[i]), "bind original FP8 weight");
    const auto size = f.outputs[i].required_bytes().value();
    pools.push_back(take(c::LeasePool::create(owner.ref(), size,
        {.max_blocks = 1, .backing_alignment = 256}), "output LeasePool::create"));
    check(slot.bind(f.y[i], f.outputs[i], take(pools.back().allocate(size, 256), "output WriteLease")), "bind owned output");
  }
  return pools;
}
void prepare(g::Invocation &slot, c::Queue &queue, bool capture) {
  check(g::cuda::prepare_replay(slot, queue,
      {.mode = capture ? g::ReplayMode::kRequireCapture : g::ReplayMode::kCommands,
       .host_wait = g::HostWaitPolicy::kReject}), "Graph prepare_replay");
  EXPECT_EQ(slot.replay_route(), capture ? g::ReplayRoute::kCaptured : g::ReplayRoute::kCommands);
}

void shared_journey(bool bfloat, Policy policy) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  Contract contract{.bfloat = bfloat};
  contract.rounding.fill(policy);
  if (policy == Policy::kPowerOfTwo) contract.floor.fill(1.0e-4F);
  const auto f = build(contract);
  auto program = take(g::cuda::compile(f.graph, context, owner), "shared Graph compile");
  selected(f, program.report, true);
  auto w = weights(f, owner);
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array slots{invocation(program.executable), invocation(program.executable)};
  std::array<c::LeasePool, 2> input_pools;
  std::array<c::ReadLease, 2> a;
  std::array<std::array<Bytes, 2>, 2> first;
  std::array<c::ReadLease, 2> exported;
  g::GraphOutput old;
  for (unsigned round = 0; round < 2; ++round)
    for (const unsigned lane : {1U, 0U}) {
      SCOPED_TRACE("round=" + std::to_string(round) + " lane=" + std::to_string(lane));
      const auto bytes = activation(f, static_cast<int>(round * 2 + lane));
      c::WriteLease writer;
      if (round == 0) {
        input_pools[lane] = take(c::LeasePool::create(owner.ref(), bytes.size(),
            {.max_blocks = 1, .backing_alignment = 256}), "activation LeasePool::create");
        writer = take(input_pools[lane].allocate(bytes.size(), 256), "activation WriteLease");
      } else {
        if (lane == 0) old.reset();
        check(slots[lane].reset(), "Graph reset before source mutation");
        const auto before = take(a[lane].view(), "retained activation view");
        writer = take(a[lane].try_write(), "completed source read-to-write permission");
        EXPECT_EQ(take(writer.view(), "new activation writer").native_handle(), before.native_handle());
        EXPECT_EQ(take(writer.view(), "new activation writer").offset(), before.offset());
      }
      check(take(writer.view(), "activation writer view").write(bytes), "change actual same-address A bytes");
      a[lane] = take(writer.publish(), "publish changed activation");
      const auto expected = direct(f, a[lane], w, context, queues[lane], owner);
      auto output_pools = bind_case(f, slots[lane], a[lane], w, owner);
      a[lane].reset(); // The slot now retains the actual source permission.
      prepare(slots[lane], queues[lane], true);
      {
        auto done = take(slots[lane].submit(queues[lane]), "shared captured submit");
        check(done.wait(), "shared captured Completion::wait");
      }
      auto y0 = take(slots[lane].result(f.y[0]), "first GraphOutput");
      auto y1 = take(slots[lane].result(f.y[1]), "second GraphOutput");
      const auto t0 = take(y0.tensor(), "first GraphOutput tensor");
      const auto t1 = take(y1.tensor(), "second GraphOutput tensor");
      EXPECT_EQ(read(t0), expected[0]);
      EXPECT_EQ(read(t1), expected[1]);
      EXPECT_NE(t0.storage().native_handle(), t1.storage().native_handle());
      EXPECT_TRUE(slots[lane].reset().IsUnavailable());
      if (round == 0) first[lane] = expected;
      else {
        EXPECT_NE(expected[0], first[lane][0]);
        EXPECT_NE(expected[1], first[lane][1]);
        if (lane == 1) EXPECT_EQ(read(take(old.tensor(), "other slot retained old output")), first[0][0]);
      }
      {
        auto source = take(slots[lane].result(f.a[0]), "published source GraphOutput");
        a[lane] = take(source.storage_lease(), "recover original activation storage permission");
      }
      if (round == 0 && lane == 0) old = std::move(y0);
      if (round == 1 && lane == 0) {
        exported[0] = take(y0.storage_lease(), "independent first output lease");
        exported[1] = take(y1.storage_lease(), "independent second output lease");
      }
    }
  std::array<Bytes, 2> before;
  for (std::size_t i = 0; i < exported.size(); ++i)
    before[i] = read(take(exported[i].tensor(f.outputs[i]), "export before teardown"));
  old.reset(); a = {}; w = {}; slots = {}; queues = {}; input_pools = {};
  program.executable = {};
  check(context.close(), "Context::close after completed captured runs");
  context = {};
  // The independent output leases own their allocations. The original typed
  // resource owner remains explicit while reading after Context admission closes.
  for (std::size_t i = 0; i < exported.size(); ++i)
    EXPECT_EQ(read(take(exported[i].tensor(f.outputs[i]), "export after Context close")), before[i]);
  exported = {}; owner = {};
}

TEST(CudaFp8SharedPrologue, FloatSiblingsKeepBothOriginalLinearResultsAcrossCapturedSlots) {
  shared_journey(false, Policy::kDivide);
}
TEST(CudaFp8SharedPrologue, BfloatSiblingsKeepBothOriginalLinearResultsAcrossCapturedSlots) {
  shared_journey(true, Policy::kDivide);
}
TEST(CudaFp8SharedPrologue, PowerOfTwoBfloatSiblingsKeepTheirOriginalScaleLayoutAcrossCapturedSlots) {
  shared_journey(true, Policy::kPowerOfTwo);
}
TEST(CudaFp8SharedPrologue, MultiplyReciprocalBfloatSiblingsKeepTheirOriginalQuantizerAcrossCapturedSlots) {
  shared_journey(true, Policy::kMultiplyReciprocal);
}
TEST(CudaFp8SharedPrologue, DifferentContractsOrValueIdsAndPackedSingleLinearStayStandalone) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  auto queue = context.create_queue().value();
  std::array<Contract, 5> cases;
  for (auto &item : cases) item.bfloat = true;
  cases[0].floor[1] = std::nextafter(cases[0].floor[0], 1.F); // Adjacent F32 floor bits are unequal.
  cases[1].rounding[1] = Policy::kPowerOfTwo;
  cases[1].floor.fill(1.0e-4F);
  cases[2].same_value = false; // Same physical allocation does not identify logical A.
  cases[3].packed_single = true; // Existing packed [2N,K] remains one original linear.
  cases[4].rounding[1] = Policy::kMultiplyReciprocal; // Same floor, distinct complete scale policy.
  for (std::size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    const auto f = build(cases[i]);
    auto program = take(g::cuda::compile(f.graph, context, owner), "standalone boundary Graph compile");
    selected(f, program.report, false);
    if (!f.contract.same_value) EXPECT_NE(f.a[0], f.a[1]);
    const auto a = upload(owner, activation(f, 3));
    const auto w = weights(f, owner);
    const auto expected = direct(f, a, w, context, queue, owner);
    auto slot = invocation(program.executable);
    auto pools = bind_case(f, slot, a, w, owner);
    prepare(slot, queue, false);
    auto done = take(slot.submit(queue), "standalone boundary submit");
    check(done.wait(), "standalone boundary Completion::wait");
    for (std::size_t j = 0; j < f.linears; ++j) {
      auto output = take(slot.result(f.y[j]), "standalone boundary output");
      EXPECT_EQ(read(take(output.tensor(), "standalone boundary tensor")), expected[j]);
    }
  }
}
} // namespace
