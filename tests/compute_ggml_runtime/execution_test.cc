#include "cpu_execution.h"

#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "hyle/mem/quota_allocator.h"

namespace {
using hyle::compute::ggml_runtime::CpuExecution;
using hyle::compute::ggml_runtime::MetadataArena;

// An original ggml RMSNorm -> F16 MatMul -> state addition. Payloads are
// borrowed, metadata is arena-owned, and each invocation owns its workspace.
struct Region {
  MetadataArena arena;
  std::array<float, 16> input{}, norm{};
  std::array<float, 8> product{}, state{}, output{};
  ggml_cgraph *graph = nullptr;

  bool build(std::array<ggml_fp16_t, 32> &constant) {
    auto made = MetadataArena::for_graph(16, 16, hyle::mem::system_allocator());
    if (!made)
      return false;
    arena = std::move(*made);
    auto *x = ggml_new_tensor_2d(arena.get(), GGML_TYPE_F32, 8, 2);
    auto *w = ggml_new_tensor_2d(arena.get(), GGML_TYPE_F16, 8, 4);
    auto *s = ggml_new_tensor_2d(arena.get(), GGML_TYPE_F32, 4, 2);
    x->data = input.data();
    w->data = constant.data();
    s->data = state.data();
    auto *n = ggml_rms_norm(arena.get(), x, 1.0e-5f);
    n->data = norm.data();
    auto *m = ggml_mul_mat(arena.get(), w, n);
    m->data = product.data();
    auto *y = ggml_add(arena.get(), m, s);
    y->data = output.data();
    graph = ggml_new_graph_custom(arena.get(), 16, false);
    ggml_build_forward_expand(graph, y);
    return true;
  }

  void check(const std::array<ggml_fp16_t, 32> &constant) const {
    for (int token = 0; token < 2; ++token) {
      double square = 0;
      for (int k = 0; k < 8; ++k)
        square +=
            static_cast<double>(input[token * 8 + k]) * input[token * 8 + k];
      const auto scale =
          1.0f / std::sqrt(static_cast<float>(square / 8) + 1.0e-5f);
      for (int row = 0; row < 4; ++row) {
        float sum = 0;
        for (int k = 0; k < 8; ++k) {
          const auto rounded = ggml_fp16_to_fp32(
              ggml_fp32_to_fp16(input[token * 8 + k] * scale));
          sum += rounded * ggml_fp16_to_fp32(constant[row * 8 + k]);
        }
        EXPECT_NEAR(output[token * 4 + row], sum + state[token * 4 + row],
                    2.0e-5f);
      }
    }
  }
};

std::array<ggml_fp16_t, 32> weights() {
  std::array<ggml_fp16_t, 32> result{};
  for (int i = 0; i < 32; ++i)
    result[i] = ggml_fp32_to_fp16(static_cast<float>(i % 9 - 4) / 8);
  return result;
}

TEST(GgmlCpuExecution, ReusesGraphPoolWorkspaceAndRetiresExactBacking) {
  auto constant = weights();
  Region region;
  ASSERT_TRUE(region.build(constant));
  hyle::mem::QuotaAllocator workspace(hyle::mem::system_allocator(), 16384);
  {
    auto execution = CpuExecution::prepare(region.graph, 2, workspace.get());
    ASSERT_TRUE(execution);
    ASSERT_GT(execution->workspace_bytes(), 0u);
    const auto allocated = workspace.used();
    for (int pass = 0; pass < 5; ++pass) {
      for (int i = 0; i < 16; ++i)
        region.input[i] =
            static_cast<float>(i - 7) / 4 + static_cast<float>(pass);
      ASSERT_TRUE(execution->run().ok());
      region.check(constant);
      EXPECT_EQ(workspace.used(), allocated);
      region.state = region.output;
    }
    const auto old_result = region.output;
    auto moved = std::move(*execution);
    EXPECT_TRUE(execution->run().IsFailedPrecondition());
    ASSERT_TRUE(moved.run().ok());
    region.check(constant);
    EXPECT_NE(old_result, region.output);
  }
  EXPECT_EQ(workspace.used(), 0u);
}

TEST(GgmlCpuExecution, IndependentSlotsShareOnlyImmutableWeights) {
  auto constant = weights();
  const auto original = constant;
  Region first, second;
  ASSERT_TRUE(first.build(constant));
  ASSERT_TRUE(second.build(constant));
  first.input.fill(0.5f);
  second.input.fill(-1.5f);
  second.state.fill(3.0f);
  auto a = CpuExecution::prepare(first.graph, 2, hyle::mem::system_allocator());
  auto b =
      CpuExecution::prepare(second.graph, 2, hyle::mem::system_allocator());
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  std::atomic<bool> good{true};
  std::thread worker([&] {
    for (int i = 0; i < 32; ++i)
      if (!a->run().ok())
        good.store(false);
  });
  for (int i = 0; i < 32; ++i)
    if (!b->run().ok())
      good.store(false);
  worker.join();
  EXPECT_TRUE(good.load());
  first.check(constant);
  second.check(constant);
  EXPECT_EQ(constant, original);
}

TEST(GgmlCpuExecution, CancellationJoinsWorkersAndAllowsReuse) {
  auto constant = weights();
  Region region;
  ASSERT_TRUE(region.build(constant));
  region.input.fill(1.0f);
  std::atomic<bool> cancelled{true};
  auto execution = CpuExecution::prepare(
      region.graph, 3, hyle::mem::system_allocator(),
      [](void *p) { return static_cast<std::atomic<bool> *>(p)->load(); },
      &cancelled);
  ASSERT_TRUE(execution);
  EXPECT_TRUE(execution->run().IsCancelled());
  cancelled.store(false);
  ASSERT_TRUE(execution->run().ok());
  region.check(constant);
}

TEST(GgmlCpuExecution, InvalidPreparationAndWorkspaceExhaustionAreRecoverable) {
  auto constant = weights();
  Region region;
  ASSERT_TRUE(region.build(constant));
  auto allocator = hyle::mem::system_allocator();
  EXPECT_FALSE(CpuExecution::prepare(nullptr, 2, allocator));
  EXPECT_FALSE(CpuExecution::prepare(region.graph, 0, allocator));
  EXPECT_FALSE(
      CpuExecution::prepare(region.graph, GGML_MAX_N_THREADS + 1, allocator));
  EXPECT_FALSE(CpuExecution::prepare(region.graph, 2, {}));
  hyle::mem::QuotaAllocator exhausted(allocator, 0);
  auto failed = CpuExecution::prepare(region.graph, 2, exhausted.get());
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted());
  EXPECT_EQ(exhausted.used(), 0u);
  EXPECT_TRUE(CpuExecution::prepare(region.graph, 2, allocator));
  EXPECT_FALSE(MetadataArena::for_graph(0, 8, allocator));
  EXPECT_FALSE(MetadataArena::for_graph(
      8, (std::numeric_limits<std::size_t>::max)(), allocator));
  EXPECT_FALSE(MetadataArena::for_graph(
      (std::numeric_limits<std::size_t>::max)(), 8, allocator));
}
} // namespace
