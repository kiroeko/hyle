#include "cpu_execution.h"
#include "cpu_program.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "hyle/mem/quota_allocator.h"

namespace {
using hyle::compute::ggml_runtime::CpuExecution;
using hyle::compute::ggml_runtime::MetadataArena;

TEST(GgmlCpuExecution, Avx2ProgramRequiresCpuAndOsEnabledPrerequisites) {
  using hyle::platform::CpuFeature;
  using hyle::platform::CpuFeatureSet;
  using hyle::compute::ggml_runtime::cpu_avx2_program_eligible;
  const std::array required{CpuFeature::kX86Avx, CpuFeature::kX86Avx2,
                            CpuFeature::kX86Fma, CpuFeature::kX86F16c,
                            CpuFeature::kX86Bmi2};
  CpuFeatureSet complete;
  for (const auto feature : required)
    complete.set(feature);
  EXPECT_TRUE(cpu_avx2_program_eligible(complete));
  EXPECT_FALSE(cpu_avx2_program_eligible({}));
  for (const auto feature : required) {
    auto missing = complete;
    missing.set(feature, false);
    EXPECT_FALSE(cpu_avx2_program_eligible(missing));
  }
  CpuFeatureSet unrelated{CpuFeature::kArmNeon, CpuFeature::kArmBf16};
  EXPECT_FALSE(cpu_avx2_program_eligible(unrelated));
}

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

#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS)
TEST(GgmlCpuExecution, OriginalNarrowIq1MatMulAndGatherReplayEncodedBlocks) {
  const std::array types{GGML_TYPE_IQ1_XS, GGML_TYPE_IQ1_XXS,
                         GGML_TYPE_IQ1_XXXS};
  const std::array<std::size_t, 3> block_bytes{46, 42, 38};
  constexpr int k = 256, n = 4, tokens = 2;
  for (std::size_t type_index = 0; type_index < types.size(); ++type_index) {
    SCOPED_TRACE(type_index);
    const auto type = types[type_index];
    ASSERT_EQ(ggml_blck_size(type), k);
    ASSERT_EQ(ggml_type_size(type), block_bytes[type_index]);
    alignas(64) std::array<std::byte, n * 46> packed{};
    std::array<float, n * k> decoded{};
    for (int row = 0; row < n; ++row) {
      auto *block = packed.data() + row * block_bytes[type_index];
      for (std::size_t byte = 2; byte < block_bytes[type_index]; ++byte)
        block[byte] = std::byte((byte * 37 + row * 19) & 255);
      const auto scale = ggml_fp32_to_fp16(float(row + 1) / 4);
      std::memcpy(block, &scale, sizeof(scale));
      ASSERT_TRUE(ggml_validate_row_data(type, block, block_bytes[type_index]));
    }
    // The all-zero grid index is an independent golden block: original grid
    // entry zero is all -1, the unsigned delta is +1/8, and d is 1/4.
    std::fill(packed.begin() + 2, packed.begin() + block_bytes[type_index],
              std::byte{});
    const auto *traits = ggml_get_type_traits(type);
    ASSERT_NE(traits->to_float, nullptr);
    traits->to_float(packed.data(), decoded.data(), n * k);
    for (int column = 0; column < k; ++column)
      EXPECT_EQ(decoded[column], -0.21875f);

    auto arena = MetadataArena::for_graph(8, 8, hyle::mem::system_allocator());
    ASSERT_TRUE(arena);
    auto *w = ggml_new_tensor_2d(arena->get(), type, k, n);
    auto *x = ggml_new_tensor_2d(arena->get(), GGML_TYPE_F32, k, tokens);
    auto *indices = ggml_new_tensor_1d(arena->get(), GGML_TYPE_I32, 3);
    std::array<float, k * tokens> input{};
    std::array<float, n * tokens> product{};
    std::array<std::int32_t, 3> ids{3, 1, 1};
    std::array<float, k * 3> rows{};
    w->data = packed.data();
    x->data = input.data();
    indices->data = ids.data();
    auto *mul = ggml_mul_mat(arena->get(), w, x);
    auto *gather = ggml_get_rows(arena->get(), w, indices);
    mul->data = product.data();
    gather->data = rows.data();
    auto *graph = ggml_new_graph_custom(arena->get(), 8, false);
    ggml_build_forward_expand(graph, mul);
    ggml_build_forward_expand(graph, gather);
    hyle::mem::QuotaAllocator workspace(hyle::mem::system_allocator(), 65536);
    {
      auto execution = CpuExecution::prepare(graph, 2, workspace.get());
      ASSERT_TRUE(execution);
      const auto allocated = workspace.used();
      for (int pass = 0; pass < 2; ++pass) {
        // Integer [-127,128] (or its negative) is exactly representable by
        // original Q8_K activation conversion; no quantization tolerance hides
        // a wrong codebook, scale, row layout or replayed input.
        for (int token = 0; token < tokens; ++token)
          for (int column = 0; column < k; ++column)
            input[token * k + column] =
                float(column - 127) * ((pass + token) % 2 ? -1 : 1);
        ASSERT_TRUE(execution->run().ok());
        EXPECT_EQ(workspace.used(), allocated);
        for (int token = 0; token < tokens; ++token)
          for (int row = 0; row < n; ++row) {
            float expected = 0;
            for (int column = 0; column < k; ++column)
              expected += input[token * k + column] * decoded[row * k + column];
            EXPECT_NEAR(product[token * n + row], expected, 1.0e-4f);
          }
        for (int index = 0; index < 3; ++index)
          for (int column = 0; column < k; ++column)
            EXPECT_EQ(rows[index * k + column], decoded[ids[index] * k + column]);
      }
    }
    EXPECT_EQ(workspace.used(), 0u);
  }
}
#endif
} // namespace
