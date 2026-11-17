#include "cpu_execution.h"
#include "cpu_program.h"
#include "ggml-quants.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include <vector>
#endif

#include <gtest/gtest.h>

#include "hyle/mem/quota_allocator.h"

namespace {
using hyle::compute::ggml_runtime::CpuExecution;
using hyle::compute::ggml_runtime::CpuTeam;
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

TEST(GgmlCpuExecution, UnaryHadamardOwnsNoMatrixInEitherReferenceSetting) {
  using hyle::compute::ggml_runtime::select_cpu_program;
  auto selected = select_cpu_program();
  ASSERT_TRUE(selected) << selected.error().message();
  const auto *program = *selected;
  ASSERT_NE(program->hadamard, nullptr);
  program->initialize();
  constexpr int width = 64, rows = 2;
  std::array<float, width * rows> input{}, unary_output{}, matrix_output{};
  std::array<float, width * width> matrix{};
  for (int i = 0; i < width * rows; ++i)
    input[i] = float(i % 9 - 4) / 8;
  for (int row = 0; row < width; ++row)
    for (int column = 0; column < width; ++column)
      matrix[row * width + column] =
          (std::popcount(unsigned(row & column)) % 2 ? -1.0F : 1.0F) / 8;
  auto arena = MetadataArena::for_graph(8, 8, hyle::mem::system_allocator());
  ASSERT_TRUE(arena);
  auto *source = ggml_new_tensor_2d(arena->get(), GGML_TYPE_F32, width, rows);
  source->data = input.data();
  auto *unary = ggml_custom_4d(arena->get(), GGML_TYPE_F32, width, rows, 1, 1,
                               &source, 1, program->hadamard,
                               GGML_N_TASKS_MAX, nullptr);
  unary->data = unary_output.data();
  EXPECT_EQ(unary->src[0], source);
  EXPECT_EQ(unary->src[1], nullptr);
  EXPECT_EQ(ggml_nbytes(source), sizeof(input));
  auto *weights = ggml_new_tensor_2d(arena->get(), GGML_TYPE_F32, width, width);
  weights->data = matrix.data(); // The legacy hinted operation keeps real storage.
  auto *hinted = ggml_mul_mat(arena->get(), weights, source);
  ggml_mul_mat_set_hint(hinted, GGML_HINT_SRC0_IS_HADAMARD);
  hinted->data = matrix_output.data();
  auto *graph = ggml_new_graph_custom(arena->get(), 8, false);
  ggml_build_forward_expand(graph, unary);
  ggml_build_forward_expand(graph, hinted);
  for (const bool reference : {false, true}) {
    SCOPED_TRACE(reference);
    auto plan = program->plan(graph, 2, nullptr);
    plan.use_ref = reference;
    alignas(GGML_MEM_ALIGN) std::array<std::uint8_t, 4096> work{};
    ASSERT_LE(plan.work_size, work.size());
    plan.work_data = work.data();
    ASSERT_EQ(program->compute(graph, &plan), GGML_STATUS_SUCCESS);
    for (int row = 0; row < rows; ++row)
      for (int column = 0; column < width; ++column) {
        double expected = 0;
        for (int k = 0; k < width; ++k)
          expected += double(input[row * width + k]) * matrix[column * width + k];
        EXPECT_NEAR(unary_output[row * width + column], expected, 2e-6);
        EXPECT_NEAR(matrix_output[row * width + column], expected, 2e-6);
      }
  }
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
    auto team = CpuTeam::create(2, hyle::mem::system_allocator());
    ASSERT_TRUE(team);
    auto execution = CpuExecution::prepare(region.graph, *team, workspace.get());
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
  auto first_team = CpuTeam::create(2, hyle::mem::system_allocator());
  auto second_team = CpuTeam::create(2, hyle::mem::system_allocator());
  ASSERT_TRUE(first_team);
  ASSERT_TRUE(second_team);
  auto a = CpuExecution::prepare(first.graph, *first_team, hyle::mem::system_allocator());
  auto b =
      CpuExecution::prepare(second.graph, *second_team, hyle::mem::system_allocator());
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
  auto team = CpuTeam::create(3, hyle::mem::system_allocator());
  ASSERT_TRUE(team);
  Region other;
  ASSERT_TRUE(other.build(constant));
  other.input.fill(-0.5f);
  auto independent = CpuExecution::prepare(other.graph, *team, hyle::mem::system_allocator());
  ASSERT_TRUE(independent);
  auto execution = CpuExecution::prepare(
      region.graph, *team, hyle::mem::system_allocator(),
      [](void *p) { return static_cast<std::atomic<bool> *>(p)->load(); },
      &cancelled);
  ASSERT_TRUE(execution);
  EXPECT_TRUE(execution->run().IsCancelled());
  ASSERT_TRUE(independent->run().ok());
  other.check(constant);
  cancelled.store(false);
  ASSERT_TRUE(execution->run().ok());
  region.check(constant);
}

TEST(GgmlCpuExecution, InvalidPreparationAndWorkspaceExhaustionAreRecoverable) {
  auto constant = weights();
  Region region;
  ASSERT_TRUE(region.build(constant));
  auto allocator = hyle::mem::system_allocator();
  EXPECT_FALSE(CpuTeam::create(0, allocator));
  EXPECT_FALSE(CpuTeam::create(GGML_MAX_N_THREADS + 1, allocator));
  EXPECT_FALSE(CpuTeam::create(2, {}));
  auto team = CpuTeam::create(2, allocator);
  ASSERT_TRUE(team);
  EXPECT_FALSE(CpuExecution::prepare(nullptr, *team, allocator));
  EXPECT_FALSE(CpuExecution::prepare(region.graph, CpuTeam{}, allocator));
  EXPECT_FALSE(CpuExecution::prepare(region.graph, *team, {}));
  hyle::mem::QuotaAllocator exhausted(allocator, 0);
  auto no_team = CpuTeam::create(2, exhausted.get());
  ASSERT_FALSE(no_team);
  EXPECT_TRUE(no_team.error().IsResourceExhausted());
  auto failed = CpuExecution::prepare(region.graph, *team, exhausted.get());
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted());
  EXPECT_EQ(exhausted.used(), 0u);
  EXPECT_TRUE(CpuExecution::prepare(region.graph, *team, allocator));
  EXPECT_FALSE(MetadataArena::for_graph(0, 8, allocator));
  EXPECT_FALSE(MetadataArena::for_graph(
      8, (std::numeric_limits<std::size_t>::max)(), allocator));
  EXPECT_FALSE(MetadataArena::for_graph(
      (std::numeric_limits<std::size_t>::max)(), 8, allocator));
}

TEST(GgmlCpuExecution, SharedTeamRetainsOwnerAndSerializesIndependentSlots) {
  auto constant = weights();
  Region first, second;
  ASSERT_TRUE(first.build(constant));
  ASSERT_TRUE(second.build(constant));
  first.input.fill(0.5f);
  second.input.fill(-1.5f);
  second.state.fill(3.0f);
  hyle::mem::QuotaAllocator resources(hyle::mem::system_allocator(), 16384);
  {
    auto team = CpuTeam::create(3, resources.get());
    ASSERT_TRUE(team);
    auto a = CpuExecution::prepare(first.graph, *team, resources.get());
    auto b = CpuExecution::prepare(second.graph, *team, resources.get());
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    *team = CpuTeam{}; // Both slots must keep the pool's owner alive.
    std::atomic<bool> good{true};
    std::thread worker([&] {
      for (int i = 0; i < 16; ++i)
        if (!a->run().ok()) good.store(false);
    });
    for (int i = 0; i < 16; ++i)
      if (!b->run().ok()) good.store(false);
    worker.join();
    EXPECT_TRUE(good.load());
    first.check(constant);
    second.check(constant);
  }
  EXPECT_EQ(resources.used(), 0u);
}

TEST(GgmlCpuExecution, SameTeamReentryFailsBeforePoolMutationAndAllowsLaterRun) {
  auto constant = weights();
  Region outer, inner;
  ASSERT_TRUE(outer.build(constant));
  ASSERT_TRUE(inner.build(constant));
  outer.input.fill(0.5f);
  inner.input.fill(-0.75f);
  auto team = CpuTeam::create(3, hyle::mem::system_allocator());
  ASSERT_TRUE(team);
  auto nested = CpuExecution::prepare(inner.graph, *team, hyle::mem::system_allocator());
  ASSERT_TRUE(nested);
  struct Reentry { CpuExecution *inner; hyle::Status status; unsigned calls = 0; } probe{&*nested, hyle::Status::Ok()};
  auto execution = CpuExecution::prepare(outer.graph, *team, hyle::mem::system_allocator(),
      [](void *value) {
        auto &observed = *static_cast<Reentry *>(value);
        ++observed.calls;
        observed.status = observed.inner->run();
        return false;
      }, &probe);
  ASSERT_TRUE(execution);
  ASSERT_TRUE(execution->run().ok());
  ASSERT_GT(probe.calls, 0u);
  EXPECT_TRUE(probe.status.IsFailedPrecondition());
  outer.check(constant);
  ASSERT_TRUE(nested->run().ok());
  inner.check(constant);
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
      auto team = CpuTeam::create(2, hyle::mem::system_allocator());
      ASSERT_TRUE(team);
      auto execution = CpuExecution::prepare(graph, *team, workspace.get());
      ASSERT_TRUE(execution);
      const auto allocated = workspace.used();
      for (int pass = 0; pass < 2; ++pass) {
        // MatMul's declared original CPU path converts F32 activation to Q8_K.
        // Apply that original encoder to the independent decoded-weight oracle;
        // comparing against unquantized activation would change the contract.
        for (int token = 0; token < tokens; ++token)
          for (int column = 0; column < k; ++column)
            input[token * k + column] =
                float(column - 127) * ((pass + token) % 2 ? -1 : 1);
        ASSERT_TRUE(execution->run().ok());
        EXPECT_EQ(workspace.used(), allocated);
        for (int token = 0; token < tokens; ++token) {
          block_q8_K quantized{};
          const auto *activation_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
          ASSERT_NE(activation_traits->from_float, nullptr);
          activation_traits->from_float(input.data() + token * k, &quantized, k);
          for (int row = 0; row < n; ++row) {
            double expected = 0;
            for (int column = 0; column < k; ++column)
              expected += double(quantized.d) * quantized.qs[column] *
                          decoded[row * k + column];
            // The original dot factors F32 scales outside its integer reduction;
            // allow eight F32 rounding units for that arithmetic association.
            const auto tolerance = 8 * std::numeric_limits<float>::epsilon() *
                                   std::max(1.0, std::abs(expected));
            EXPECT_NEAR(product[token * n + row], expected, tolerance);
          }
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

#if defined(__APPLE__)
struct BlasRegion final {
  int rows, inner, columns;
  std::vector<float> input, scaled, widened, product, bias, output;
  std::vector<ggml_bf16_t> matrix, narrowed;
  MetadataArena arena;
  ggml_tensor *native_matrix = nullptr;
  ggml_cgraph *graph = nullptr;
  BlasRegion(int m, int k, int n, bool explicit_widening=true)
      : rows(m), inner(k), columns(n), input(std::size_t(m)*k), scaled(input.size()),
        widened(input.size()),product(std::size_t(m)*n), bias(n), output(product.size(),77.F),
        matrix(std::size_t(n)*k),narrowed(input.size()) {
    for(std::size_t i=0;i<input.size();++i)input[i]=float(int(i%17)-8)/13.F;
    for(std::size_t i=0;i<matrix.size();++i)matrix[i]=ggml_fp32_to_bf16(float(int(i%19)-9)/23.F);
    for(int i=0;i<n;++i)bias[i]=float(i%5-2)/31.F;
    auto created=MetadataArena::for_graph(20,20,hyle::mem::system_allocator());
    if(!created)throw std::runtime_error(std::string(created.error().message()));
    arena=std::move(*created);
    auto *x=ggml_new_tensor_2d(arena.get(),GGML_TYPE_F32,k,m);x->data=input.data();
    auto *scale=ggml_scale(arena.get(),x,.5F);scale->data=scaled.data();
    auto *activation=scale;
    if(explicit_widening) {
      auto *half=ggml_cast(arena.get(),scale,GGML_TYPE_BF16);half->data=narrowed.data();
      activation=ggml_cast(arena.get(),half,GGML_TYPE_F32);activation->data=widened.data();
    }
    native_matrix=ggml_new_tensor_2d(arena.get(),GGML_TYPE_BF16,k,n);native_matrix->data=matrix.data();
    auto *multiply=ggml_mul_mat(arena.get(),native_matrix,activation);multiply->data=product.data();
    if(!ggml_prec_set_acc(multiply,GGML_PREC_F32))throw std::runtime_error("original F32 accumulation");
    auto *b=ggml_new_tensor_1d(arena.get(),GGML_TYPE_F32,n);b->data=bias.data();
    auto *y=ggml_add(arena.get(),multiply,b);y->data=output.data();
    graph=ggml_new_graph_custom(arena.get(),20,false);ggml_build_forward_expand(graph,y);
  }
  void check() const {
    for(int row=0;row<rows;++row)for(int column=0;column<columns;++column) {
      double expected=bias[column];
      for(int k=0;k<inner;++k)
        expected+=double(ggml_bf16_to_fp32(ggml_fp32_to_bf16(input[std::size_t(row)*inner+k]*.5F)))*
            ggml_bf16_to_fp32(matrix[std::size_t(column)*inner+k]);
      EXPECT_NEAR(output[std::size_t(row)*columns+column],expected,
          3e-5*std::max(1.,std::abs(expected)))<<row<<','<<column;
    }
  }
};

TEST(GgmlCpuExecution, NativeBlasAndCpuSegmentsReuseColdWorkspaceAndMoveSafely) {
  if (__builtin_available(macOS 15.0, *)) {} else { GTEST_SKIP() << "native BLAS thread control requires macOS15"; }
  BlasRegion region(32,64,64);
  auto team=CpuTeam::create(2,hyle::mem::system_allocator());ASSERT_TRUE(team);
  hyle::mem::QuotaAllocator workspace(hyle::mem::system_allocator(),65536);
  {
    auto execution=CpuExecution::prepare(region.graph,*team,workspace.get());
    ASSERT_TRUE(execution)<<execution.error().message();ASSERT_TRUE(execution->uses_blas());
    const auto allocated=workspace.used();
    for(int pass=0;pass<3;++pass) {
      for(auto &value:region.input)value+=.03125F;
      ASSERT_TRUE(execution->run().ok());region.check();EXPECT_EQ(workspace.used(),allocated);
    }
    auto moved=std::move(*execution);EXPECT_TRUE(execution->run().IsFailedPrecondition());
    ASSERT_TRUE(moved.uses_blas());ASSERT_TRUE(moved.run().ok());region.check();
  }
  EXPECT_EQ(workspace.used(),0u);
}

TEST(GgmlCpuExecution, NativeBlasPlansShareOnlyTheOriginalSerializedTeamScratch) {
  if (__builtin_available(macOS 15.0, *)) {} else { GTEST_SKIP() << "native BLAS thread control requires macOS15"; }
  BlasRegion first(32,64,64),second(35,96,96);
  for(auto &value:second.input)value=-value+.25F;
  auto team=CpuTeam::create(2,hyle::mem::system_allocator());ASSERT_TRUE(team);
  auto a=CpuExecution::prepare(first.graph,*team,hyle::mem::system_allocator());
  auto b=CpuExecution::prepare(second.graph,*team,hyle::mem::system_allocator());
  ASSERT_TRUE(a);ASSERT_TRUE(b);ASSERT_TRUE(a->uses_blas());ASSERT_TRUE(b->uses_blas());
  hyle::Status left,right;
  std::thread one([&]{left=a->run();}),two([&]{right=b->run();});
  one.join();two.join();ASSERT_TRUE(left.ok());ASSERT_TRUE(right.ok());first.check();second.check();
  ASSERT_TRUE(a->run().ok());first.check(); // Larger cold reserve did not invalidate the old plan.
}

TEST(GgmlCpuExecution, NativeBlasCancellationLeavesUnsubmittedOutputsUntouchedAndTeamReusable) {
  if (__builtin_available(macOS 15.0, *)) {} else { GTEST_SKIP() << "native BLAS thread control requires macOS15"; }
  BlasRegion region(32,64,64);
  auto team=CpuTeam::create(2,hyle::mem::system_allocator());ASSERT_TRUE(team);
  std::atomic<bool> abort{true};
  auto execution=CpuExecution::prepare(region.graph,*team,hyle::mem::system_allocator(),
      [](void *p){return static_cast<std::atomic<bool> *>(p)->load();},&abort);
  ASSERT_TRUE(execution);ASSERT_TRUE(execution->uses_blas());
  EXPECT_TRUE(execution->run().IsCancelled());
  EXPECT_TRUE(std::ranges::all_of(region.output,[](float x){return x==77.F;}));
  abort=false;ASSERT_TRUE(execution->run().ok());region.check();
}

TEST(GgmlCpuExecution, MixedNativeActivationKeepsItsOriginalImplicitStorageRounding) {
  BlasRegion region(32,64,64,false);
  auto team=CpuTeam::create(2,hyle::mem::system_allocator());ASSERT_TRUE(team);
  auto execution=CpuExecution::prepare(region.graph,*team,hyle::mem::system_allocator());
  ASSERT_TRUE(execution);EXPECT_FALSE(execution->uses_blas());
  ASSERT_TRUE(execution->run().ok());region.check();
}

TEST(GgmlCpuExecution, LosslessBlasWideningUsesActualNativeQualificationAndKeepsDecode) {
  using hyle::compute::ggml_runtime::blas_activation_widening_supported;
  BlasRegion decode(1,64,64),prefill(32,64,64);
  auto team=CpuTeam::create(2,hyle::mem::system_allocator());ASSERT_TRUE(team);
  auto original=CpuExecution::prepare(decode.graph,*team,hyle::mem::system_allocator());
  ASSERT_TRUE(original);EXPECT_FALSE(original->uses_blas());ASSERT_TRUE(original->run().ok());decode.check();
  for(auto *region:{&decode,&prefill}) {
    auto *activation=ggml_new_tensor_2d(region->arena.get(),GGML_TYPE_BF16,region->inner,region->rows);
    auto *product=ggml_mul_mat(region->arena.get(),region->native_matrix,activation);
    ASSERT_TRUE(ggml_prec_set_acc(product,GGML_PREC_F32));
    if (__builtin_available(macOS 15.0, *))
      EXPECT_EQ(blas_activation_widening_supported(*product),region->rows!=1);
    else
      EXPECT_FALSE(blas_activation_widening_supported(*product));
    EXPECT_EQ(activation->type,GGML_TYPE_BF16);EXPECT_EQ(activation->data,nullptr);
  }
}

TEST(GgmlCpuExecution, NativeBlasPreservesTheCallingThreadsPolicyAndOneThreadBudget) {
  if (__builtin_available(macOS 15.0, *)) {
    const auto original = BLASGetThreading();
    struct Restore final {
      enum BLAS_THREADING policy;
      ~Restore() {
        if (__builtin_available(macOS 15.0, *))
          (void)BLASSetThreading(policy);
      }
    } restore{original};
    BlasRegion region(32,64,64);
    auto team=CpuTeam::create(1,hyle::mem::system_allocator());ASSERT_TRUE(team);
    std::atomic<bool> abort{false};
    auto execution=CpuExecution::prepare(region.graph,*team,hyle::mem::system_allocator(),
        [](void *p){return static_cast<std::atomic<bool> *>(p)->load();},&abort);
    ASSERT_TRUE(execution);ASSERT_TRUE(execution->uses_blas());
    for (const auto policy : {BLAS_THREADING_MULTI_THREADED, BLAS_THREADING_SINGLE_THREADED}) {
      ASSERT_EQ(BLASSetThreading(policy),0);
      ASSERT_TRUE(execution->run().ok());region.check();
      EXPECT_EQ(BLASGetThreading(),policy);
      abort=true;EXPECT_TRUE(execution->run().IsCancelled());
      EXPECT_EQ(BLASGetThreading(),policy);abort=false;
    }
  } else {
    GTEST_SKIP() << "native BLAS thread control requires macOS15";
  }
}
#endif
} // namespace
