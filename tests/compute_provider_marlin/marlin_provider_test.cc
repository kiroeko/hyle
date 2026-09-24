#include "hyle/compute/providers/marlin.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"

namespace {
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace marlin = compute::marlin;
namespace mem = hyle::mem;

compute::Context context() {
  return compute::Context::create(cuda::device(0).value(), marlin::provider())
      .value();
}

compute::TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  return compute::TensorDesc::dense(compute::DataType::kFloat16,
                                    std::array{rows, columns})
      .value();
}

struct Matrix final {
  mem::Buffer storage;
  compute::TensorView view;
  Matrix(compute::TensorDesc desc, mem::MemoryResourceRef resource)
      : storage(
            mem::Buffer::create(resource, desc.required_bytes().value(), 256)
                .value()),
        view(compute::TensorView::create(desc, storage.view()).value()) {}
};

struct Weights final {
  std::size_t n, k;
  std::vector<std::int8_t> values;
  std::vector<float> scales;
  Weights(std::size_t columns, std::size_t inner)
      : n(columns), k(inner), values(n * k), scales(n * (k / 128)) {
    for (std::size_t col = 0; col < n; ++col) {
      for (std::size_t i = 0; i < k; ++i)
        values[col * k + i] = static_cast<std::int8_t>(
            static_cast<int>((col * 7 + i * 13 + i / 128) % 16) - 8);
      for (std::size_t group = 0; group < k / 128; ++group)
        scales[col * (k / 128) + group] =
            static_cast<float>((col * 3 + group * 5) % 17) * 0.00173f;
    }
  }
  marlin::QuantizedWeights pack(const compute::Context &owner,
                                mem::MemoryResourceRef resource) const {
    return marlin::QuantizedWeights::pack(
               owner, resource, static_cast<std::int64_t>(n),
               static_cast<std::int64_t>(k), values, scales)
        .value();
  }
};

std::vector<__half> activations(std::size_t count, int seed) {
  std::vector<__half> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = __float2half_rn(
        static_cast<float>(static_cast<int>((i * 11 + i / 128) % 23) - 11 +
                           seed) *
        0.0319f);
  return result;
}

void check_product(const Weights &weights, std::span<const __half> activation,
                   Matrix &output) {
  std::vector<__half> result(output.storage.size() / sizeof(__half));
  ASSERT_TRUE(output.storage.view()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  const auto rows = activation.size() / weights.k;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < weights.n; ++col) {
      double expected = 0;
      for (std::size_t inner = 0; inner < weights.k; ++inner) {
        // The public packing contract rounds scales to FP16. Marlin performs
        // dequantization in FP16, followed by FP32 accumulation/reduction.
        const auto scale = __float2half_rn(
            weights.scales[col * (weights.k / 128) + inner / 128]);
        const auto weight = __float2half_rn(
            static_cast<float>(weights.values[col * weights.k + inner]) *
            __half2float(scale));
        expected += static_cast<double>(
                        __half2float(activation[row * weights.k + inner])) *
                    __half2float(weight);
      }
      ASSERT_NEAR(__half2float(result[row * weights.n + col]), expected,
                  0.0006 * std::abs(expected) + 0.00002)
          << "row=" << row << " column=" << col;
    }
  }
}

void run_matrix(std::int64_t m, std::int64_t k, std::int64_t n) {
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  Weights raw(static_cast<std::size_t>(n), static_cast<std::size_t>(k));
  auto packed = raw.pack(owner, resource.ref());
  EXPECT_EQ(packed.resident_bytes(),
            raw.n * raw.k / 2 + raw.n * (raw.k / 128) * 2);
  EXPECT_EQ(packed.weights().description().storage_format().id(),
            marlin::kWeightFormat);
  auto moved = std::move(packed);
  EXPECT_FALSE(packed.valid());
  EXPECT_FALSE(packed.weights().valid());
  Matrix input(matrix(m, k), resource.ref());
  Matrix first(matrix(m, n), resource.ref()),
      second(matrix(m, n), resource.ref());
  const std::array inputs{input.view.as_const(), moved.weights(),
                          moved.scales()};
  auto preparation = compute::prepare(
      owner, compute::Operation(marlin::QuantizedWeights::operation()), inputs,
      std::array{first.view});
  ASSERT_TRUE(preparation.has_value()) << preparation.error().message();
  auto prepared = std::move(*preparation);
  EXPECT_EQ(prepared.provider().id, marlin::kProviderId);
  EXPECT_TRUE(prepared.capture_compatible());
  EXPECT_EQ(prepared.invocation_private_bytes(), 0u);
  const auto needed = prepared.workspace_requirement();
  std::array workspace{
      mem::Buffer::create(resource.ref(), needed.size, needed.alignment)
          .value(),
      mem::Buffer::create(resource.ref(), needed.size, needed.alignment)
          .value()};
  std::array lanes{prepared.create_invocation(workspace[0].view()).value(),
                   prepared.create_invocation(workspace[1].view()).value()};
  std::array queues{owner.create_queue().value(), owner.create_queue().value()};
  for (auto &queue : queues)
    ASSERT_TRUE(queue.reserve_completions(1).ok());
  for (int seed = 0; seed < 3; ++seed) {
    auto a = activations(static_cast<std::size_t>(m * k), seed);
    ASSERT_TRUE(input.storage.view().write(std::as_bytes(std::span(a))).ok());
    auto first_done =
        lanes[0].submit(queues[0], inputs, std::array{first.view});
    ASSERT_TRUE(first_done.has_value()) << first_done.error().message();
    auto second_done =
        lanes[1].submit(queues[1], inputs, std::array{second.view});
    ASSERT_TRUE(second_done.has_value()) << second_done.error().message();
    ASSERT_TRUE(first_done->wait().ok());
    ASSERT_TRUE(second_done->wait().ok());
    check_product(raw, a, first);
    check_product(raw, a, second);
  }
}

TEST(MarlinProvider, DecodePrefillAndRowTailsMatchDoubleOracle) {
  for (std::int64_t m : {1, 7, 9, 16, 17, 33, 64}) {
    SCOPED_TRACE(m);
    run_matrix(m, 1024, 1024);
  }
}

TEST(MarlinProvider, TileSelectionAndMinimumMatrixMatchDoubleOracle) {
  run_matrix(1, 128, 4096);
  run_matrix(17, 128, 4096);
  run_matrix(17, 128, 2816);
  run_matrix(9, 128, 64);
}

TEST(MarlinProvider, RejectsInvalidPackingAndUnsupportedExecutionDomain) {
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  Weights raw(64, 128);
  auto pack = [&] {
    return marlin::QuantizedWeights::pack(owner, resource.ref(), 64, 128,
                                          raw.values, raw.scales);
  };
  raw.values[0] = 8;
  EXPECT_TRUE(pack().error().IsInvalidArgument());
  raw.values[0] = -9;
  EXPECT_TRUE(pack().error().IsInvalidArgument());
  raw.values[0] = -8;
  raw.scales[0] = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(pack().error().IsInvalidArgument());
  raw.scales[0] = -1;
  EXPECT_TRUE(pack().error().IsInvalidArgument());
  raw.scales[0] = 0.03f;
  EXPECT_FALSE(marlin::QuantizedWeights::pack(owner, mem::system_resource(), 64,
                                              128, raw.values, raw.scales)
                   .has_value());
  EXPECT_FALSE(marlin::QuantizedWeights::pack(owner, resource.ref(), 63, 128,
                                              raw.values, raw.scales)
                   .has_value());
  auto too_large = marlin::QuantizedWeights::pack(
      owner, resource.ref(), 131072, 16384, raw.values, raw.scales);
  ASSERT_FALSE(too_large.has_value());
  EXPECT_TRUE(too_large.error().IsUnimplemented());
  EXPECT_FALSE(marlin::QuantizedWeights::pack(owner, resource.ref(), 64, 128,
                                              std::span(raw.values).first(3),
                                              raw.scales)
                   .has_value());
  auto packed = pack().value();
  auto op = compute::Operation(marlin::QuantizedWeights::operation());
  auto prepare = [&](std::int64_t rows, compute::TensorDesc weights) {
    return compute::prepare(
        owner, op,
        std::array{matrix(rows, 128), weights, packed.scales().description()},
        std::array{matrix(rows, 64)}, resource.ref());
  };
  EXPECT_FALSE(prepare(65, packed.weights().description()).has_value());
  EXPECT_FALSE(prepare(1, matrix(64, 128)).has_value());
  auto prepared = prepare(1, packed.weights().description()).value();
  EXPECT_FALSE(prepared.create_invocation().has_value());
  const auto needed = prepared.workspace_requirement();
  auto short_workspace =
      mem::Buffer::create(resource.ref(), needed.size - 1, needed.alignment)
          .value();
  EXPECT_FALSE(prepared.create_invocation(short_workspace.view()).has_value());
}

TEST(MarlinProvider, CompletionRetainsReleasedArtifactAndLane) {
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  Weights raw(64, 128);
  auto packed = raw.pack(owner, resource.ref());
  Matrix input(matrix(1, 128), resource.ref()),
      output(matrix(1, 64), resource.ref());
  auto a = activations(128, 2);
  ASSERT_TRUE(input.storage.view().write(std::as_bytes(std::span(a))).ok());
  const std::array inputs{input.view.as_const(), packed.weights(),
                          packed.scales()};
  auto prepared =
      compute::prepare(
          owner, compute::Operation(marlin::QuantizedWeights::operation()),
          inputs, std::array{output.view})
          .value();
  const auto needed = prepared.workspace_requirement();
  auto workspace =
      mem::Buffer::create(resource.ref(), needed.size, needed.alignment)
          .value();
  auto lane = prepared.create_invocation(workspace.view()).value();
  auto queue = owner.create_queue().value();
  struct Gate final {
    std::atomic<bool> released{false};
    static void CUDART_CB wait(void *opaque) {
      static_cast<Gate *>(opaque)->released.wait(false,
                                                 std::memory_order_acquire);
    }
    void release() {
      released.store(true, std::memory_order_release);
      released.notify_all();
    }
  } gate;
  struct Release final {
    Gate &gate;
    compute::Context &owner;
    ~Release() {
      gate.release();
      owner.drain().IgnoreError();
    }
  } release{gate, owner};
  ASSERT_EQ(cudaLaunchHostFunc(cuda::stream(queue).value(), &Gate::wait, &gate),
            cudaSuccess);
  auto done = lane.submit(queue, inputs, std::array{output.view});
  ASSERT_TRUE(done.has_value());
  EXPECT_EQ(done->poll().value(), compute::CompletionPoll::kPending);
  lane = {};
  prepared = {};
  ASSERT_TRUE(owner.close().ok());
  gate.release();
  ASSERT_TRUE(done->wait().ok());
  check_product(raw, a, output);
}

struct MetadataAllocator final {
  mem::AllocatorRef backing = mem::system_allocator();
  std::size_t attempts = 0, fail_at = SIZE_MAX, live = 0;
  mem::AllocatorRef ref() noexcept {
    return {[](void *opaque, std::size_t size,
               std::size_t alignment) noexcept -> void * {
              auto &self = *static_cast<MetadataAllocator *>(opaque);
              if (self.attempts++ == self.fail_at)
                return nullptr;
              void *result = self.backing.allocate(size, alignment);
              if (result != nullptr)
                ++self.live;
              return result;
            },
            [](void *opaque, void *pointer, std::size_t size,
               std::size_t alignment) noexcept {
              auto &self = *static_cast<MetadataAllocator *>(opaque);
              if (pointer != nullptr)
                --self.live;
              self.backing.deallocate(pointer, size, alignment);
            },
            [](void *opaque, void *pointer, std::size_t old_size,
               std::size_t size, std::size_t alignment) noexcept -> void * {
              auto &self = *static_cast<MetadataAllocator *>(opaque);
              if (size != 0 && self.attempts++ == self.fail_at)
                return nullptr;
              void *result =
                  self.backing.reallocate(pointer, old_size, size, alignment);
              if (pointer == nullptr && result != nullptr)
                ++self.live;
              else if (pointer != nullptr && size == 0)
                --self.live;
              return result;
            },
            this};
  }
};

class RejectingHostResource final : public mem::MemoryResource {
public:
  std::string_view name() const noexcept override {
    return "rejected host scratch";
  }
  mem::MemoryProperties properties() const noexcept override {
    return mem::system_resource().properties();
  }
  hyle::Result<mem::ResourceAllocation>
  allocate(std::size_t, std::size_t) noexcept override {
    return std::unexpected(
        hyle::Status::ResourceExhausted("test host scratch budget"));
  }
  void deallocate(mem::ResourceAllocation) noexcept override { ADD_FAILURE(); }
};

TEST(MarlinProvider,
     ColdAllocationFailuresRollBackAndReplayDoesNotAllocateMetadata) {
  MetadataAllocator allocator;
  {
    const std::array providers{
        hyle::compute::ProviderBinding{marlin::provider(), 100}};
    auto owner =
        compute::Context::create(
            cuda::device(0).value(),
            {.metadata_allocator = allocator.ref(), .providers = providers})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    Weights raw(64, 128);
    RejectingHostResource rejecting;
    auto rejected = marlin::QuantizedWeights::pack(
        owner, resource.ref(), 64, 128, raw.values, raw.scales,
        mem::MemoryResourceRef(rejecting));
    ASSERT_FALSE(rejected.has_value());
    EXPECT_TRUE(rejected.error().IsResourceExhausted());

    const auto pack_begin = allocator.attempts;
    {
      auto packed = raw.pack(owner, resource.ref());
    }
    const auto pack_allocations = allocator.attempts - pack_begin;
    const auto pack_live = allocator.live;
    ASSERT_GT(pack_allocations, 0u);
    ASSERT_LT(pack_allocations, 32u);
    for (std::size_t offset = 0; offset < pack_allocations; ++offset) {
      SCOPED_TRACE(offset);
      allocator.fail_at = allocator.attempts + offset;
      auto packed = marlin::QuantizedWeights::pack(owner, resource.ref(), 64,
                                                   128, raw.values, raw.scales);
      ASSERT_FALSE(packed.has_value());
      EXPECT_TRUE(packed.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, pack_live);
    }
    allocator.fail_at = SIZE_MAX;
    auto packed = raw.pack(owner, resource.ref());
    Matrix input(matrix(1, 128), resource.ref()),
        output(matrix(1, 64), resource.ref());
    const std::array inputs{input.view.as_const(), packed.weights(),
                            packed.scales()};
    const auto prepare = [&] {
      return compute::prepare(
          owner, compute::Operation(marlin::QuantizedWeights::operation()),
          inputs, std::array{output.view});
    };
    const auto prepare_begin = allocator.attempts;
    {
      auto prepared = prepare().value();
    }
    const auto prepare_allocations = allocator.attempts - prepare_begin;
    const auto prepare_live = allocator.live;
    ASSERT_GT(prepare_allocations, 0u);
    ASSERT_LT(prepare_allocations, 32u);
    for (std::size_t offset = 0; offset < prepare_allocations; ++offset) {
      SCOPED_TRACE(offset);
      allocator.fail_at = allocator.attempts + offset;
      auto prepared = prepare();
      ASSERT_FALSE(prepared.has_value());
      EXPECT_TRUE(prepared.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, prepare_live);
    }
    allocator.fail_at = SIZE_MAX;
    auto prepared = prepare().value();
    const auto needed = prepared.workspace_requirement();
    auto workspace =
        mem::Buffer::create(resource.ref(), needed.size, needed.alignment)
            .value();
    const auto lane_begin = allocator.attempts;
    {
      auto lane = prepared.create_invocation(workspace.view()).value();
    }
    const auto lane_allocations = allocator.attempts - lane_begin;
    const auto lane_live = allocator.live;
    ASSERT_GT(lane_allocations, 0u);
    ASSERT_LT(lane_allocations, 32u);
    for (std::size_t offset = 0; offset < lane_allocations; ++offset) {
      SCOPED_TRACE(offset);
      allocator.fail_at = allocator.attempts + offset;
      auto lane = prepared.create_invocation(workspace.view());
      ASSERT_FALSE(lane.has_value());
      EXPECT_TRUE(lane.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, lane_live);
    }
    allocator.fail_at = SIZE_MAX;
    auto lane = prepared.create_invocation(workspace.view()).value();
    auto queue = owner.create_queue().value();
    ASSERT_TRUE(queue.reserve_completions(1).ok());
    const auto a = activations(128, 1);
    ASSERT_TRUE(input.storage.view().write(std::as_bytes(std::span(a))).ok());
    const auto replay_begin = allocator.attempts;
    for (int iteration = 0; iteration < 3; ++iteration) {
      auto done = lane.submit(queue, inputs, std::array{output.view});
      ASSERT_TRUE(done.has_value());
      ASSERT_TRUE(done->wait().ok());
    }
    EXPECT_EQ(allocator.attempts, replay_begin);
    check_product(raw, a, output);
  }
  EXPECT_EQ(allocator.live, 0u);
}

} // namespace
