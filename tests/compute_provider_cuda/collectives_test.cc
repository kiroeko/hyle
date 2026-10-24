#include "collectives_test_support.h"
#include "hyle/mem/stats.h"

#include <atomic>
#include <chrono>
#include <thread>

namespace {
using namespace cccl_test;
struct FaultAllocator {
  std::size_t attempts = 0, fail = SIZE_MAX, live = 0;
  mem::AllocatorRef ref() {
    return {[](void *self, std::size_t bytes,
               std::size_t alignment) noexcept -> void * {
              auto &s = *static_cast<FaultAllocator *>(self);
              if (s.attempts++ == s.fail)
                return nullptr;
              auto *p = mem::system_allocator().allocate(bytes, alignment);
              if (p)
                ++s.live;
              return p;
            },
            [](void *self, void *p, std::size_t bytes,
               std::size_t alignment) noexcept {
              auto &s = *static_cast<FaultAllocator *>(self);
              if (p)
                --s.live;
              mem::system_allocator().deallocate(p, bytes, alignment);
            },
            [](void *self, void *p, std::size_t old_bytes, std::size_t bytes,
               std::size_t alignment) noexcept -> void * {
              auto &s = *static_cast<FaultAllocator *>(self);
              if (bytes && s.attempts++ == s.fail)
                return nullptr;
              auto *next = mem::system_allocator().reallocate(p, old_bytes,
                                                              bytes, alignment);
              if (bytes == 0 || next) {
                if (p)
                  --s.live;
                if (next)
                  ++s.live;
              }
              return next;
            },
            this};
  }
};

TEST(CudaCccl, EveryMetadataAllocationFailureReleasesPartialConstruction) {
  FaultAllocator allocator;
  const auto device = cuda::device(0).value();
  const std::array providers{
      hyle::compute::ProviderBinding{cuda_native::cccl_provider(), 100}};
  const auto create = [&] {
    return Context::create(device, {.metadata_allocator = allocator.ref(),
                                    .providers = providers});
  };
  {
    const auto check_failures = [&](auto make) {
      const auto live = allocator.live, start = allocator.attempts;
      {
        auto value = make();
        ASSERT_TRUE(value) << value.error().message();
      }
      const auto count = allocator.attempts - start;
      ASSERT_GT(count, 0u);
      ASSERT_EQ(allocator.live, live);
      for (std::size_t i = 0; i < count; ++i) {
        SCOPED_TRACE(i);
        allocator.fail = allocator.attempts + i;
        {
          auto failed = make();
          EXPECT_FALSE(failed);
        }
        allocator.fail = SIZE_MAX;
        EXPECT_EQ(allocator.live, live);
      }
      {
        auto retry = make();
        ASSERT_TRUE(retry) << retry.error().message();
      }
      EXPECT_EQ(allocator.live, live);
    };
    check_failures(create);
    auto owner = create().value();
    auto memory =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    const auto desc =
        TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{17})
            .value();
    const auto make_program = [&] {
      return prepare(owner, TopK{.count = 4}, desc, memory.ref());
    };
    check_failures(make_program);
    auto program = make_program().value();
    auto scratch = workspace(program, memory.ref());
    check_failures([&] { return program.create_invocation(scratch.view()); });
  }
  EXPECT_EQ(allocator.live, 0u);
}

Context context() {
  return Context::create(cuda::device(0).value(), cuda_native::cccl_provider())
      .value();
}
TEST(CudaCccl, ScanAxesByteStridesBroadcastAndHighRanks) {
  auto owner = context();
  for (std::int32_t axis : {0, 1, -1})
    for (int mode : {0, 1, 2}) {
      scan<float>(owner, std::array<std::int64_t, 3>{2, 257, 3}, axis, mode);
      scan<std::int32_t>(owner, std::array<std::int64_t, 3>{2, 257, 3}, axis,
                         mode);
    }
  scan<float>(owner, std::array<std::int64_t, 6>{2, 1, 3, 1, 17, 2}, 4, 1);
  scan<std::int32_t>(
      owner, std::array<std::int64_t, 8>{1, 2, 1, 3, 1, 17, 2, 1}, 5, 1);
  scan<float>(owner, std::array<std::int64_t, 2>{2, 8193}, -1, 0);
  scan<std::int32_t>(owner, std::array<std::int64_t, 3>{2, 0, 3}, 1, 0);
}
template <class T> void selections(Context &owner) {
  for (std::int32_t axis : {0, 1, -1})
    for (int mode : {0, 1, 2})
      for (std::size_t k : {1u, axis == 0 ? 2u : axis == 1 ? 17u : 3u})
        topk<T>(owner, std::array<std::int64_t, 3>{2, 17, 3}, axis, mode, k);
  topk<T>(owner, std::array<std::int64_t, 8>{1, 2, 1, 3, 1, 17, 2, 1}, 5, 1, 7);
  topk<T>(owner, std::array<std::int64_t, 2>{2, 8193}, -1, 0, 7);
  topk<T>(owner, std::array<std::int64_t, 3>{0, 17, 3}, 1, 0, 17);
}
TEST(CudaCccl, TopKFourTypesStableTiesNaNsAndOriginalBits) {
  auto owner = context();
  selections<float>(owner);
  selections<std::int32_t>(owner);
  selections<hyle::simd::float16_t>(owner);
  selections<hyle::simd::bfloat16_t>(owner);
}

template <class T> void single_row_special_bits(Context &owner) {
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc = TensorDesc::dense(
      type<T>(), std::array<std::int64_t, 1>{8193}).value();
  Tensor input(desc, memory.ref());
  for (const bool largest : {false, true}) {
    const TopK operation{.count = 1, .largest = largest};
    auto program = prepare(owner, operation, desc, memory.ref());
    ASSERT_TRUE(program) << program.error().message();
    auto scratch = workspace(*program, memory.ref());
    auto slot = program->create_invocation(scratch.view()).value();
    Tensor values(program->output_description(0).value(), memory.ref());
    Tensor indices(program->output_description(1).value(), memory.ref());
    for (const int pattern : {0, 1, 2}) {
      SCOPED_TRACE(::testing::Message()
                   << static_cast<int>(type<T>()) << '/' << largest << '/'
                   << pattern);
      std::vector<T> data(8193);
      for (std::size_t i = 0; i < data.size(); ++i) {
        if (pattern == 0) {
          // Every value is NaN: the first original payload must survive.
          if constexpr (std::is_same_v<T, float>)
            data[i] = std::bit_cast<float>(i % 2 ? 0x7fc001abU : 0xffc002cdU);
          else if constexpr (std::is_same_v<T, hyle::simd::float16_t>)
            data[i] = T::from_bits(
                static_cast<std::uint16_t>(i % 2 ? 0x7eab : 0xfecd));
          else
            data[i] = T::from_bits(
                static_cast<std::uint16_t>(i % 2 ? 0x7feb : 0xffcd));
        } else {
          // Both directions tie +0 with -0; preserve the first sign bit.
          data[i] = encode<T>((i + static_cast<std::size_t>(pattern)) % 2
                                 ? -0.0 : 0.0);
        }
      }
      input.write<T>(data);
      const auto in = input.view.as_const();
      const std::array out{values.view, indices.view};
      auto done = slot.submit(queue, std::span(&in, 1), out);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual = values.read<T>();
      ASSERT_EQ(indices.read<std::int32_t>(), (std::vector<std::int32_t>{0}));
      ASSERT_EQ(actual.size(), 1u);
      EXPECT_EQ(std::memcmp(actual.data(), data.data(), sizeof(T)), 0);
      values.guard();
      indices.guard();
    }
  }
}

TEST(CudaCccl, SingleRowTopOneDenseAndGenericLayoutsKeepStableOriginalBits) {
  auto owner = context();
  const std::array<std::int64_t, 1> row{8193};
  // Reuse the independent stable-sort oracle for all four supported types,
  // both directions, repeated extrema and mixed NaNs/Inf/signed zeros.
  topk<float>(owner, row, 0, 0, 1);
  topk<std::int32_t>(owner, row, 0, 0, 1);
  topk<hyle::simd::float16_t>(owner, row, 0, 0, 1);
  topk<hyle::simd::bfloat16_t>(owner, row, 0, 0, 1);
  single_row_special_bits<float>(owner);
  single_row_special_bits<hyle::simd::float16_t>(owner);
  single_row_special_bits<hyle::simd::bfloat16_t>(owner);
  // An axis followed by singleton dimensions is still one ordered row.
  topk<float>(owner, std::array<std::int64_t, 3>{1, 8193, 1}, 1, 0, 1);
  // Adjacent non-dense and broadcast domains must keep the generic iterator.
  topk<float>(owner, std::array<std::int64_t, 1>{17}, 0, 1, 1);
  topk<float>(owner, std::array<std::int64_t, 1>{17}, 0, 2, 1);
}

TEST(CudaCccl, WideMultiRowTopOneFourTypesAndStridedAxesKeepOracle) {
  auto owner = context();
  const std::array<std::int64_t, 2> rows{3, 8193};
  // More than two independent rows and a partial final tile. The existing
  // stable-sort oracle checks both directions, original bits and I32 values
  // which cannot be compared exactly after promotion to float.
  topk<float>(owner, rows, -1, 0, 1);
  topk<std::int32_t>(owner, rows, -1, 0, 1);
  topk<hyle::simd::float16_t>(owner, rows, -1, 0, 1);
  topk<hyle::simd::bfloat16_t>(owner, rows, -1, 0, 1);
  const std::array<std::int64_t, 3> middle{2, 8193, 3};
  // Padded byte strides include unaligned scalar loads; the zero-stride
  // source must still select the first logical axis element for every row.
  topk<float>(owner, middle, 1, 1, 1);
  topk<float>(owner, middle, 1, 2, 1);
}

template <class T> void multi_row_top_one_bits(Context &owner) {
  constexpr std::size_t width = 8193, row_count = 4;
  constexpr std::size_t first_tie = 4095;
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc = TensorDesc::dense(
      type<T>(), std::array<std::int64_t, 2>{row_count, width}).value();
  Tensor input(desc, memory.ref(), true);
  const std::vector<std::int32_t> expected{
      static_cast<std::int32_t>(width - 1),
      static_cast<std::int32_t>(first_tie), 0, 0};
  for (const bool largest : {false, true}) {
    SCOPED_TRACE(::testing::Message()
                 << static_cast<int>(type<T>()) << '/' << largest);
    std::vector<T> data(row_count * width, encode<T>(0.0));
    if constexpr (std::is_same_v<T, std::int32_t>) {
      data[width - 1] = largest ? INT32_MAX : INT32_MIN;
      const auto tied = largest ? 16777217 : -16777217;
      data[width + first_tie - 1] = largest ? 16777216 : -16777216;
      data[width + first_tie] = data[2 * width - 1] = tied;
      std::fill_n(data.begin() + static_cast<std::ptrdiff_t>(2 * width),
                  width, INT32_MIN);
      std::fill_n(data.begin() + static_cast<std::ptrdiff_t>(3 * width),
                  width, INT32_MAX);
    } else {
      const auto nan_bits = [](std::size_t index) {
        if constexpr (std::is_same_v<T, float>)
          return std::bit_cast<float>(index % 2 ? 0x7fc00456U : 0xff800123U);
        else if constexpr (std::is_same_v<T, hyle::simd::float16_t>)
          return T::from_bits(
              static_cast<std::uint16_t>(index % 2 ? 0x7e23 : 0xfc11));
        else
          return T::from_bits(
              static_cast<std::uint16_t>(index % 2 ? 0x7fc3 : 0xff81));
      };
      data[31] = nan_bits(0);
      data[first_tie] = nan_bits(1);
      data[width - 1] = encode<T>(largest
          ? std::numeric_limits<double>::infinity()
          : -std::numeric_limits<double>::infinity());
      data[width + first_tie] = data[2 * width - 1] =
          encode<T>(largest ? 10.0 : -10.0);
      for (std::size_t i = 0; i < width; ++i) {
        // An all-NaN row preserves its earliest signaling payload; an all-zero
        // row preserves the earliest sign, including ties across distant tiles.
        data[2 * width + i] = nan_bits(i);
        data[3 * width + i] = encode<T>((i + (largest ? 1 : 0)) % 2
                                          ? -0.0 : 0.0);
      }
    }
    input.write<T>(data);
    auto prepared = prepare(owner, TopK{.count = 1, .largest = largest},
                            desc, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    auto scratch = workspace(*prepared, memory.ref());
    auto slot = prepared->create_invocation(scratch.view()).value();
    Tensor values(prepared->output_description(0).value(), memory.ref());
    Tensor indices(prepared->output_description(1).value(), memory.ref());
    const auto in = input.view.as_const();
    const std::array out{values.view, indices.view};
    auto done = slot.submit(queue, std::span(&in, 1), out);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    ASSERT_EQ(indices.read<std::int32_t>(), expected);
    const auto actual = values.read<T>();
    ASSERT_EQ(actual.size(), row_count);
    for (std::size_t row = 0; row < row_count; ++row)
      EXPECT_EQ(std::memcmp(&actual[row],
          &data[row * width + static_cast<std::size_t>(expected[row])], sizeof(T)), 0)
          << row;
    const auto unchanged = input.read<T>();
    EXPECT_EQ(std::memcmp(unchanged.data(), data.data(), data.size() * sizeof(T)), 0);
    input.guard(); values.guard(); indices.guard();
  }
}

TEST(CudaCccl, MultiRowTopOneOriginalBitsAndCrossTileIndices) {
  auto owner = context();
  multi_row_top_one_bits<float>(owner);
  multi_row_top_one_bits<std::int32_t>(owner);
  multi_row_top_one_bits<hyle::simd::float16_t>(owner);
  multi_row_top_one_bits<hyle::simd::bfloat16_t>(owner);
}

TEST(CudaCccl, VocabularyBfloatTopOneMatchesCastFloatAcrossBlocks) {
  using BFloat = hyle::simd::bfloat16_t;
  constexpr std::int64_t vocabulary = 248320;
  const std::array providers{
      ProviderBinding{cuda_native::provider()},
      ProviderBinding{cuda_native::cccl_provider()}};
  auto owner = Context::create(cuda::device(0).value(), {.providers = providers})
                   .value();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const std::array<std::int64_t, 2> shape{1, vocabulary};
  const auto bfloat_desc = TensorDesc::dense(DataType::kBFloat16, shape).value();
  const auto float_desc = TensorDesc::dense(DataType::kFloat32, shape).value();
  Tensor input(bfloat_desc, memory.ref()), widened(float_desc, memory.ref());
  auto cast = prepare(owner, Cast{DataType::kFloat32}, bfloat_desc, memory.ref())
                  .value();
  EXPECT_EQ(cast.provider().id, cuda_native::kProviderId);
  auto cast_slot = cast.create_invocation().value();
  const TopK operation{.count = 1};
  auto bfloat_top = prepare(owner, operation, bfloat_desc, memory.ref()).value();
  auto float_top = prepare(owner, operation, float_desc, memory.ref()).value();
  EXPECT_EQ(bfloat_top.provider().id, cuda_native::kCcclProviderId);
  EXPECT_EQ(float_top.provider().id, cuda_native::kCcclProviderId);
  auto bfloat_scratch = workspace(bfloat_top, memory.ref());
  auto float_scratch = workspace(float_top, memory.ref());
  auto bfloat_slot = bfloat_top.create_invocation(bfloat_scratch.view()).value();
  auto float_slot = float_top.create_invocation(float_scratch.view()).value();
  Tensor bfloat_value(bfloat_top.output_description(0).value(), memory.ref());
  Tensor bfloat_index(bfloat_top.output_description(1).value(), memory.ref());
  Tensor float_value(float_top.output_description(0).value(), memory.ref());
  Tensor float_index(float_top.output_description(1).value(), memory.ref());

  for (int pattern = 0; pattern != 7; ++pattern) {
    SCOPED_TRACE(pattern);
    std::vector<BFloat> data(vocabulary, BFloat::from_bits(0xbf80));
    std::int32_t expected = 0;
    if (pattern == 0 || pattern == 1) {
      // Equal maxima straddle reduction tiles and reach the final vocabulary
      // element. Mixed NaNs must not displace the earliest numeric maximum.
      data[0] = BFloat::from_bits(0xff81);
      data[31] = BFloat::from_bits(0x7fc3);
      expected = pattern == 0 ? 1023 : 4095;
      for (const auto index : {expected, expected + 1, 131073, 248319})
        data[index] = BFloat::from_bits(pattern == 0 ? 0x4120 : 0x7f80);
    } else if (pattern == 2) {
      // All NaNs tie by original index, retaining sign and payload even when
      // the selected source is a signaling NaN rather than a canonical NaN.
      for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = BFloat::from_bits(i % 2 ? 0x7fc3 : 0xff81);
    } else if (pattern == 3 || pattern == 4) {
      // +0 and -0 compare equal; either first source sign must survive.
      for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = BFloat::from_bits((i + pattern) % 2 ? 0x8000 : 0x0000);
    } else if (pattern == 5) {
      std::fill(data.begin(), data.end(), BFloat::from_bits(0x7fc3));
      expected = 65535;
      data[expected] = data[131072] = BFloat::from_bits(0xff80);
    } else {
      std::fill(data.begin(), data.end(), BFloat::from_bits(0x0000));
      expected = 8191;
      data[expected] = data[248319] = BFloat::from_bits(0x0001);
    }
    input.write<BFloat>(data);
    {
      auto done = cast_slot.submit(queue, input.view.as_const(), widened.view);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
    }
    // This is also the exact widening required by the existing F32 trace
    // files: storage bits move into the upper half, without arithmetic or a
    // new BF16 rounding. Check every vocabulary element, including NaNs/zero.
    const auto wide = widened.read<float>();
    ASSERT_EQ(wide.size(), data.size());
    for (std::size_t i = 0; i < wide.size(); ++i)
      ASSERT_EQ(std::bit_cast<std::uint32_t>(wide[i]),
                std::uint32_t(data[i].bits) << 16) << i;
    {
      const auto in = input.view.as_const();
      const std::array out{bfloat_value.view, bfloat_index.view};
      auto done = bfloat_slot.submit(queue, std::span(&in, 1), out);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
    }
    {
      const auto in = widened.view.as_const();
      const std::array out{float_value.view, float_index.view};
      auto done = float_slot.submit(queue, std::span(&in, 1), out);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
    }
    const std::vector<std::int32_t> selected{expected};
    EXPECT_EQ(bfloat_index.read<std::int32_t>(), selected);
    EXPECT_EQ(float_index.read<std::int32_t>(), selected);
    const auto narrow_result = bfloat_value.read<BFloat>();
    const auto wide_result = float_value.read<float>();
    ASSERT_EQ(narrow_result.size(), 1u);
    ASSERT_EQ(wide_result.size(), 1u);
    EXPECT_EQ(narrow_result[0].bits, data[expected].bits);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(wide_result[0]),
              std::uint32_t(narrow_result[0].bits) << 16);
    const auto unchanged = input.read<BFloat>();
    EXPECT_EQ(std::memcmp(unchanged.data(), data.data(), data.size() * sizeof(BFloat)), 0);
    input.guard();
    widened.guard();
    bfloat_value.guard();
    bfloat_index.guard();
    float_value.guard();
    float_index.guard();
  }
}

TEST(CudaCccl, OverflowChecksEveryPrefixAndPreservesEntireOutput) {
  auto owner = context();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 2>{2, 3})
          .value();
  Tensor input(desc, memory.ref()), output(desc, memory.ref());
  for (bool exclusive : {false, true}) {
    auto program =
        prepare(owner, Scan{.exclusive = exclusive}, desc, memory.ref())
            .value();
    auto scratch = workspace(program, memory.ref());
    for (const auto &data : {std::array{1, 2, 3, INT32_MAX, 0, 1},
                             std::array{1, 2, 3, INT32_MAX, 1, -1},
                             std::array{1, 2, 3, INT32_MIN, -1, 1}}) {
      input.write<std::int32_t>(data);
      output.write<std::int32_t>(std::array{-1, -2, -3, -4, -5, -6});
      auto slot = program.create_invocation(scratch.view()).value();
      auto done = slot.submit(queue, input.view.as_const(), output.view);
      ASSERT_TRUE(done) << done.error().message();
      EXPECT_TRUE(done->wait().IsOutOfRange());
      EXPECT_EQ(output.read<std::int32_t>(),
                (std::vector{-1, -2, -3, -4, -5, -6}));
      output.guard();
    }
    input.write<std::int32_t>(std::array{1, 2, 3, INT32_MAX, -1, -1});
    auto fresh = program.create_invocation(scratch.view()).value();
    ASSERT_TRUE(
        fresh.submit(queue, input.view.as_const(), output.view)->wait().ok());
    EXPECT_EQ(output.read<std::int32_t>(),
              exclusive ? (std::vector{0, 1, 3, 0, INT32_MAX, INT32_MAX - 1})
                        : (std::vector{1, 3, 6, INT32_MAX, INT32_MAX - 1,
                                       INT32_MAX - 2}));
  }
}
TEST(CudaCccl, WorkspaceAliasesAndExternalCaptureRejectBeforeWriting) {
  auto owner = context();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{17})
          .value();
  auto program = prepare(owner, Scan{}, desc, memory.ref()).value();
  EXPECT_FALSE(program.create_invocation());
  auto scratch = workspace(program, memory.ref());
  auto short_view = scratch.view().subview(0, scratch.size() - 1).value();
  EXPECT_FALSE(program.create_invocation(short_view));
  auto slot = program.create_invocation(scratch.view()).value();
  Tensor input(desc, memory.ref()), output(desc, memory.ref());
  input.write<float>(std::vector<float>(17, 1));
  output.write<float>(std::vector<float>(17, -9));
  EXPECT_FALSE(slot.submit(queue, input.view.as_const(), input.view));
  auto overlap =
      TensorView::create(desc, input.buffer.view().subview(4, 68).value())
          .value();
  EXPECT_FALSE(slot.submit(queue, input.view.as_const(), overlap));
  const auto stream = cuda::stream(queue).value();
  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  auto rejected = slot.submit(queue, input.view.as_const(), output.view);
  cudaGraph_t graph = nullptr;
  const auto ended = cudaStreamEndCapture(stream, &graph);
  if (graph) {
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  }
  EXPECT_FALSE(rejected);
  EXPECT_EQ(ended, cudaSuccess);
  EXPECT_EQ(output.read<float>(), std::vector<float>(17, -9));
  ASSERT_TRUE(
      slot.submit(queue, input.view.as_const(), output.view)->wait().ok());
}

TEST(CudaCccl, FloatingSpecialPrefixesAndInvalidDescriptions) {
  auto owner = context();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{2, 4})
          .value();
  Tensor input(desc, memory.ref()), output(desc, memory.ref());
  const auto inf = std::numeric_limits<float>::infinity(),
             nan = std::numeric_limits<float>::quiet_NaN();
  input.write<float>(std::array{1.F, inf, nan, 4.F, -inf, 1.F, -2.F, inf});
  for (bool exclusive : {false, true}) {
    auto program =
        prepare(owner, Scan{.exclusive = exclusive}, desc, memory.ref())
            .value();
    auto scratch = workspace(program, memory.ref());
    auto slot = program.create_invocation(scratch.view()).value();
    ASSERT_TRUE(
        slot.submit(queue, input.view.as_const(), output.view)->wait().ok());
    const auto actual = output.read<float>();
    const auto expected =
        exclusive ? std::array{0.F, 1.F, inf, nan, 0.F, -inf, -inf, -inf}
                  : std::array{1.F, inf, nan, nan, -inf, -inf, -inf, nan};
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (std::isnan(expected[i])) {
        EXPECT_TRUE(std::isnan(actual[i]));
      } else {
        EXPECT_EQ(actual[i], expected[i]);
      }
    }
  }
  EXPECT_TRUE(prepare(owner, TopK{.count = 0}, desc, memory.ref()));
  EXPECT_FALSE(prepare(owner, TopK{.count = 5}, desc, memory.ref()));
  const auto empty_axis =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{0})
          .value();
  EXPECT_FALSE(prepare(owner, TopK{}, empty_axis, memory.ref()));
  const auto huge = TensorDesc::dense(DataType::kFloat32,
                                      std::array<std::int64_t, 2>{2, INT32_MAX})
                        .value();
  EXPECT_FALSE(prepare(owner, Scan{}, huge, memory.ref()));
  EXPECT_FALSE(prepare(owner, TopK{}, huge, memory.ref()));
}

TEST(CudaCccl, PendingCompletionRetainsPreparedOwnersWithCallerWorkspace) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    const std::array providers{
        hyle::compute::ProviderBinding{cuda_native::cccl_provider(), 100}};
    auto owner = Context::create(cuda::device(0).value(),
                                 {.metadata_allocator = stats.get(),
                                  .providers = providers})
                     .value();
    auto memory =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    auto queue = owner.create_queue().value();
    const auto desc =
        TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{17})
            .value();
    auto program = prepare(owner, TopK{.count = 4}, desc, memory.ref()).value();
    std::array<TensorDesc, 2> outputs;
    ASSERT_TRUE(infer_output_descriptions(Operation(TopK{.count = 4}),
                                          std::span(&desc, 1), outputs)
                    .ok());
    Tensor input(desc, memory.ref()), values(outputs[0], memory.ref()),
        indices(outputs[1], memory.ref());
    std::vector<float> data(17);
    std::iota(data.begin(), data.end(), 0.F);
    input.write<float>(data);
    auto scratch = workspace(program, memory.ref());
    auto slot = program.create_invocation(scratch.view()).value();
    const auto in = input.view.as_const();
    const std::array out{values.view, indices.view};
    ASSERT_TRUE(slot.submit(queue, std::span(&in, 1), out)->wait().ok());
    struct Gate {
      std::atomic<bool> released{false}, timed_out{false};
      static void CUDART_CB wait(void *raw) {
        auto &g = *static_cast<Gate *>(raw);
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!g.released.load(std::memory_order_acquire)) {
          if (std::chrono::steady_clock::now() >= deadline) {
            g.timed_out = true;
            return;
          }
          std::this_thread::yield();
        }
      }
    } gate;
    const auto stream = cuda::stream(queue).value();
    struct Release {
      Gate &gate;
      cudaStream_t stream;
      ~Release() {
        gate.released.store(true, std::memory_order_release);
        (void)cudaStreamSynchronize(stream);
      }
    } release{gate, stream};
    ASSERT_EQ(cudaLaunchHostFunc(stream, &Gate::wait, &gate), cudaSuccess);
    auto done = slot.submit(queue, std::span(&in, 1), out);
    ASSERT_TRUE(done) << done.error().message();
    EXPECT_EQ(done->poll().value(), CompletionPoll::kPending);
    // The public direct contract borrows the caller's workspace and payload
    // through completion. Only provider handles may be released early.
    slot = {};
    program = {};
    ASSERT_TRUE(owner.close().ok());
    gate.released.store(true, std::memory_order_release);
    ASSERT_TRUE(done->wait().ok());
    EXPECT_FALSE(gate.timed_out.load());
    EXPECT_EQ(values.read<float>(), (std::vector{16.F, 15.F, 14.F, 13.F}));
    EXPECT_EQ(indices.read<std::int32_t>(), (std::vector{16, 15, 14, 13}));
    ASSERT_TRUE(owner.drain().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
