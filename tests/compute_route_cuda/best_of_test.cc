#include <gtest/gtest.h>
#include <cstdlib>

#include "../../src/compute_route_cuda/best_of.h"
#include "../../bench/compute_route_cuda/u4/matrix_request.h"
#include "hyle/compute/cuda.h"
#include "hyle/mem/stats.h"

#if HYLE_OS_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
namespace policy = hyle::compute::cuda::internal;
using D = c::DataType;
struct Point { const char *op; D type; int tokens, hidden; };
// Independently sealed complete-choice receipt: partial weighted RMS wins are
// intentionally absent. All outputs are checked against the C2 double oracle.
constexpr std::array kPoints{
    Point{"abs", D::kFloat16, 1, 2048},
    Point{"neg", D::kFloat16, 32, 4096},
    Point{"neg", D::kBFloat16, 32, 4096},
    Point{"exp", D::kBFloat16, 32, 4096},
    Point{"log", D::kFloat16, 1, 2048},
    Point{"log", D::kBFloat16, 1, 2048},
    Point{"sqrt", D::kBFloat16, 1, 2048},
    Point{"sqrt", D::kBFloat16, 32, 4096},
    Point{"rsqrt", D::kBFloat16, 32, 4096},
    Point{"softplus", D::kFloat16, 1, 2048},
    Point{"softplus", D::kBFloat16, 32, 4096},
    Point{"add", D::kFloat16, 1, 2048},
    Point{"sub", D::kFloat16, 32, 4096},
    Point{"sub", D::kBFloat16, 1, 2048},
    Point{"mul", D::kFloat16, 32, 4096},
    Point{"mul", D::kBFloat16, 1, 2048},
    Point{"mul", D::kBFloat16, 32, 4096},
    Point{"div", D::kBFloat16, 32, 4096},
    Point{"rms", D::kFloat16, 32, 4096},
    Point{"layer", D::kFloat32, 1, 2048},
    Point{"layer", D::kFloat32, 32, 4096},
    Point{"affine_layer", D::kFloat32, 32, 4096},
    Point{"affine_layer", D::kBFloat16, 1, 2048},
};

TEST(CudaBestOf, CompletePointsBindOnceAndCheckChangedOutputs) {
#if HYLE_OS_WINDOWS
  // Only the dedicated static qualification child requests this placement.
  // Normal source/shared consumers exercise the mature fallback as-is.
  const bool require_measured = std::getenv("HYLE_CUDA_RULE_TEST_PROFILE") != nullptr;
  if (require_measured) {
    ASSERT_TRUE(::SetProcessAffinityMask(::GetCurrentProcess(), 0x14));
    ASSERT_TRUE(::SetPriorityClass(::GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS));
    ASSERT_NE(::SetThreadAffinityMask(::GetCurrentThread(), 0x4), 0U);
    ASSERT_TRUE(::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_NORMAL));
    ASSERT_TRUE(policy::qualified_recipe());
  }
#endif
  auto stats = take(hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()));
  {
    auto owner = take(c::cuda::create_context({.metadata_allocator = stats.get()}));
    auto memory = take(c::cuda::MemoryResource::create(owner, c::cuda::MemoryKind::kDevice));
    const bool measured = policy::qualified_environment(take(c::cuda::environment(owner))) &&
                          policy::qualified_caller();
#if HYLE_OS_WINDOWS
    if (require_measured)
      ASSERT_TRUE(measured);
#endif
    ::testing::Test::RecordProperty("measured_rule_environment", measured ? "1" : "0");
    auto queue = take(owner.create_queue());
    ASSERT_TRUE(queue.reserve_completions(2).ok());
    for (const auto &point : kPoints) {
      SCOPED_TRACE(point.op);
      SCOPED_TRACE(point.tokens);
      auto r = request(point.op, point.type, point.tokens, point.hidden);
      auto program = take(c::prepare(owner, r.operation, r.inputs,
          std::span(&r.output, 1), memory.ref()));
      const auto certificate = policy::qualified_policy(r.operation, r.inputs, std::span(&r.output, 1));
      ASSERT_TRUE(certificate.valid());
      if (measured) {
        EXPECT_EQ(program.provider().id, c::cuda_native::kProviderId);
        EXPECT_EQ(program.route_policy(), certificate);
      } else {
        EXPECT_NE(program.route_policy().namespace_id, certificate.namespace_id);
        EXPECT_NE(program.provider().id, c::cuda_native::kProviderId);
      }
      const auto bound = program.route_policy();
      const auto requirement = program.workspace_requirement();
      struct Slot {
        std::vector<hyle::mem::Buffer> buffers;
        std::vector<c::ConstTensorView> inputs;
        hyle::mem::Buffer output, workspace;
        c::OperationInvocation invocation;
        std::vector<double> expected;
      };
      std::array<Slot, 2> slots;
      for (std::size_t slot = 0; slot < slots.size(); ++slot) {
        auto &s = slots[slot];
        const auto values = payload(r, slot ? 17 : 5);
        s.expected = oracle(r, values);
        for (std::size_t i = 0; i < r.inputs.size(); ++i) {
          s.buffers.push_back(take(hyle::mem::Buffer::create(memory.ref(),
              take(r.inputs[i].required_bytes()), 256)));
          ASSERT_TRUE(s.buffers.back().view().write(pack(r.inputs[i], values[i])).ok());
          s.inputs.push_back(take(c::ConstTensorView::create(r.inputs[i], s.buffers.back().view())));
        }
        s.output = take(hyle::mem::Buffer::create(memory.ref(), take(r.output.required_bytes()), 256));
        s.workspace = take(hyle::mem::Buffer::create(memory.ref(), requirement.size, requirement.alignment));
        s.invocation = take(requirement.size ? program.create_invocation(s.workspace.view())
                                            : program.create_invocation());
      }
      const auto before = stats.snapshot();
      for (int repeat = 0; repeat < 2; ++repeat)
        for (auto &s : slots) {
          auto output = take(c::TensorView::create(r.output, s.output.view()));
          auto done = take(s.invocation.submit(queue, s.inputs, std::span(&output, 1)));
          ASSERT_TRUE(done.wait().ok());
          std::vector<std::byte> bytes(s.output.size());
          ASSERT_TRUE(s.output.view().read(bytes).ok());
          const auto actual = unpack(point.type, bytes);
          const double tolerance = point.type == D::kBFloat16 ? .008 :
                                   point.type == D::kFloat16 ? .001 : 3.e-5;
          for (std::size_t i = 0; i < actual.size(); ++i)
            ASSERT_LE(std::abs(actual[i] - s.expected[i]), tolerance * (1 + std::abs(s.expected[i])));
          EXPECT_EQ(program.route_policy(), bound);
        }
      EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
      EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
    }
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(CudaBestOf, ShapePortsAttributesAndPartialWinsDoNotInheritCertificates) {
  for (const auto name : {"weighted_rms", "copy", "tanh", "packed_silu"}) {
    const auto r = request(name, D::kFloat16, 1, 2048);
    EXPECT_FALSE(policy::qualified_policy(r.operation, r.inputs, std::span(&r.output, 1)).valid());
  }
  auto r = request("log", D::kFloat16, 1, 4096);
  EXPECT_FALSE(policy::qualified_policy(r.operation, r.inputs, std::span(&r.output, 1)).valid());
  r = request("layer", D::kFloat32, 1, 2048);
  r.operation = c::Operation(c::LayerNorm{.epsilon = 1.e-5F});
  EXPECT_FALSE(policy::qualified_policy(r.operation, r.inputs, std::span(&r.output, 1)).valid());
  r = request("affine_layer", D::kBFloat16, 1, 2048);
  r.inputs[1] = r.inputs[2] = dense(D::kFloat32, {2048});
  EXPECT_FALSE(policy::qualified_policy(r.operation, r.inputs, std::span(&r.output, 1)).valid());
  auto owner = take(c::cuda::create_context());
  auto environment = take(c::cuda::environment(owner));
  ++environment.runtime_version;
  EXPECT_FALSE(policy::qualified_environment(environment));
  environment = take(c::cuda::environment(owner));
  environment.driver_release[0] = '\0';
  EXPECT_FALSE(policy::qualified_environment(environment));
}

TEST(CudaBestOf, ManagedStorageRetainsFallbackWithoutMeasuredLabel) {
  auto owner = take(c::cuda::create_context());
  auto memory = take(c::cuda::MemoryResource::create(owner, c::cuda::MemoryKind::kManaged));
  const auto r = request("log", D::kFloat16, 1, 2048);
  auto program = take(c::prepare(owner, r.operation, r.inputs, std::span(&r.output, 1), memory.ref()));
  EXPECT_NE(program.route_policy().namespace_id, 0x48594c4543555752ULL);
}

TEST(CudaBestOf, ShiftedViewsRetainCapabilityWithoutMeasuredAlignment) {
  auto owner = take(c::cuda::create_context());
  auto memory = take(c::cuda::MemoryResource::create(owner, c::cuda::MemoryKind::kDevice));
  const auto r = request("log", D::kFloat16, 1, 2048);
  const auto bytes = take(r.inputs[0].required_bytes());
  auto input = take(hyle::mem::Buffer::create(memory.ref(), bytes + 2, 256));
  auto output = take(hyle::mem::Buffer::create(memory.ref(), bytes + 2, 256));
  auto input_view = take(c::ConstTensorView::create(r.inputs[0], take(input.view().subview(2, bytes))));
  auto output_view = take(c::TensorView::create(r.output, take(output.view().subview(2, bytes))));
  auto program = take(c::prepare(owner, r.operation, std::array{input_view},
                                 std::array{output_view}, memory.ref()));
  EXPECT_NE(program.route_policy().namespace_id, 0x48594c4543555752ULL);
  const auto requirement = program.workspace_requirement();
  auto workspace = take(hyle::mem::Buffer::create(memory.ref(), requirement.size, requirement.alignment));
  auto invocation = take(requirement.size ? program.create_invocation(workspace.view())
                                         : program.create_invocation());
  auto queue = take(owner.create_queue());
  for (int seed : {5, 17}) {
    const auto values = payload(r, seed);
    ASSERT_TRUE(take(input.view().subview(2, bytes)).write(pack(r.inputs[0], values[0])).ok());
    auto done = take(invocation.submit(queue, input_view, output_view));
    ASSERT_TRUE(done.wait().ok());
    std::vector<std::byte> data(bytes);
    ASSERT_TRUE(take(output.view().subview(2, bytes)).read(data).ok());
    const auto actual = unpack(D::kFloat16, data);
    const auto expected = oracle(r, values);
    for (std::size_t i = 0; i < actual.size(); ++i)
      ASSERT_LE(std::abs(actual[i] - expected[i]), .001 * (1 + std::abs(expected[i])));
  }
  // Descriptor preparation allocates no payload and cannot know a future
  // view's offset. Its bound measured contract must reject a later violation,
  // without an execution-time Provider search or any output write.
  auto declared = take(c::prepare(owner, r.operation, r.inputs, std::span(&r.output, 1), memory.ref()));
  if (declared.route_policy().namespace_id == 0x48594c4543555752ULL) {
    auto slot = take(declared.create_invocation());
    ASSERT_TRUE(output.view().fill(std::byte{0x6f}).ok());
    auto rejected = slot.submit(queue, input_view, output_view);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsInvalidArgument());
    std::vector<std::byte> guard(bytes + 2);
    ASSERT_TRUE(output.view().read(guard).ok());
    EXPECT_TRUE(std::all_of(guard.begin(), guard.end(), [](auto b) { return b == std::byte{0x6f}; }));
  }
}
} // namespace
