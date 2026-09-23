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
  const std::array providers{cuda_native::cccl_provider()};
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
  EXPECT_FALSE(prepare(owner, TopK{.count = 0}, desc, memory.ref()));
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
    const std::array providers{cuda_native::cccl_provider()};
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
