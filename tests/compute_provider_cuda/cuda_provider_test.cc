#include "hyle/compute/providers/cuda.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"

namespace {
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;

compute::Context
context(compute::Provider selected = compute::cuda_native::provider()) {
  return compute::Context::create(cuda::device(0).value(), selected).value();
}

compute::TensorDesc matrix(compute::DataType type, std::int64_t rows,
                           std::int64_t hidden) {
  return compute::TensorDesc::dense(type, std::array{rows, hidden}).value();
}

struct Tensor final {
  mem::Buffer storage;
  compute::TensorView view;
  Tensor(compute::TensorDesc desc, mem::MemoryResourceRef resource)
      : storage(
            mem::Buffer::create(resource, desc.required_bytes().value(), 256)
                .value()),
        view(compute::TensorView::create(desc, storage.view()).value()) {}
};

template <class T> T from_float(float value) { return value; }
template <> __half from_float(float value) { return __float2half(value); }
template <> __nv_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}
template <class T> float to_float(T value) { return value; }
template <> float to_float(__half value) { return __half2float(value); }
template <> float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
void run_shape(compute::DataType type, bool packed, std::int64_t rows,
               std::int64_t hidden,
               compute::Provider selected = compute::cuda_native::provider(),
               compute::GatedActivationKind activation =
                   compute::GatedActivationKind::kSilu) {
  auto owner = context(selected);
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto desc = matrix(type, rows, hidden);
  Tensor first(matrix(type, rows, hidden * (packed ? 2 : 1)), resource.ref());
  Tensor second(desc, resource.ref());
  std::array outputs{Tensor(desc, resource.ref()),
                     Tensor(desc, resource.ref())};
  const std::array all_inputs{first.view.as_const(), second.view.as_const()};
  const auto inputs = std::span(all_inputs).first(packed ? 1 : 2);
  const compute::GatedActivation op{
      .activation = activation,
      .inputs = packed ? compute::GatedInputKind::kPackedHalves
                       : compute::GatedInputKind::kSeparate};
  auto result = compute::prepare(owner, compute::Operation(op), inputs,
                                 std::array{outputs[0].view});
  ASSERT_TRUE(result.has_value()) << result.error().message();
  auto prepared = std::move(*result);
  EXPECT_EQ(prepared.provider().id, selected.id());
  EXPECT_EQ(prepared.workspace_requirement().size, 0u);
  EXPECT_EQ(prepared.invocation_private_bytes(), 0u);
  EXPECT_TRUE(prepared.capture_compatible());
  std::array lanes{prepared.create_invocation().value(),
                   prepared.create_invocation().value()};
  std::array queues{owner.create_queue().value(), owner.create_queue().value()};
  const auto count = static_cast<std::size_t>(rows * hidden);
  const auto width = static_cast<std::size_t>(hidden);
  for (int seed = 0; seed < 3; ++seed) {
    std::vector<T> a(count * (packed ? 2 : 1)), b(count);
    std::vector<double> expected(count);
    for (std::size_t index = 0; index < count; ++index) {
      const auto gate_index = packed ? index + (index / width) * width : index;
      float gate = (static_cast<int>(index % 37) - 18 + seed) * 0.331F;
      if (index < 7) {
        const std::array edges{-std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN(),
                               -0.0F,
                               0.0F,
                               -80.0F,
                               80.0F};
        gate = edges[index];
      }
      a[gate_index] = from_float<T>(gate);
      const T value =
          from_float<T>((static_cast<int>(index % 13) - 6 + seed) * 0.173F);
      if (packed)
        a[gate_index + width] = value;
      else
        b[index] = value;
      const double rounded_gate = to_float(a[gate_index]);
      double activated = rounded_gate / (1.0 + std::exp(-rounded_gate));
      if (activation == compute::GatedActivationKind::kGelu)
        activated =
            0.5 * rounded_gate * (1 + std::erf(rounded_gate / std::sqrt(2.0)));
      else if (activation == compute::GatedActivationKind::kGeluTanh)
        activated =
            0.5 * rounded_gate *
            (1 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) *
                           (rounded_gate + 0.044715 * rounded_gate *
                                               rounded_gate * rounded_gate)));
      expected[index] = activated * static_cast<double>(to_float(value));
    }
    ASSERT_TRUE(first.storage.view().write(std::as_bytes(std::span(a))).ok());
    if (!packed) {
      ASSERT_TRUE(
          second.storage.view().write(std::as_bytes(std::span(b))).ok());
    }
    auto first_done =
        lanes[0].submit(queues[0], inputs, std::array{outputs[0].view});
    auto second_done =
        lanes[1].submit(queues[1], inputs, std::array{outputs[1].view});
    ASSERT_TRUE(first_done.has_value());
    ASSERT_TRUE(second_done.has_value());
    ASSERT_TRUE(first_done->wait().ok());
    ASSERT_TRUE(second_done->wait().ok());
    for (auto &output : outputs) {
      std::vector<T> actual(count);
      ASSERT_TRUE(output.storage.view()
                      .read(std::as_writable_bytes(std::span(actual)))
                      .ok());
      for (std::size_t index = 0; index < count; ++index) {
        const double value = to_float(actual[index]);
        if (std::isnan(expected[index]))
          EXPECT_TRUE(std::isnan(value));
        else if (std::isinf(expected[index]))
          EXPECT_EQ(value, expected[index]);
        else {
          const auto epsilon = type == compute::DataType::kBFloat16 ? 0.008
                               : sizeof(T) == 2                     ? 0.001
                                                                    : 0.000001;
          EXPECT_NEAR(value, expected[index],
                      epsilon * (1 + std::abs(expected[index])))
              << "index=" << index << " seed=" << seed;
        }
      }
    }
  }
}

TEST(CudaKernels, FloatAndHalfPackedSeparateTailsAndChangedInputs) {
  for (const bool packed : {false, true}) {
    for (const auto rows : {1, 7, 17}) {
      for (const auto hidden : {1, 127, 256, 513}) {
        SCOPED_TRACE(::testing::Message()
                     << packed << "," << rows << "," << hidden);
        run_shape<float>(compute::DataType::kFloat32, packed, rows, hidden);
        run_shape<__half>(compute::DataType::kFloat16, packed, rows, hidden);
      }
    }
  }
}

TEST(CudaKernels, PreciseSglangVectorsUseIndependentOracleAndSlots) {
  const auto selected = compute::cuda_native::sglang_provider();
  for (const auto rows : {1, 7, 17}) {
    for (const auto hidden : {8, 128, 11008}) {
      SCOPED_TRACE(::testing::Message() << rows << "," << hidden);
      run_shape<float>(compute::DataType::kFloat32, true, rows, hidden,
                       selected);
      run_shape<__half>(compute::DataType::kFloat16, true, rows, hidden,
                        selected);
    }
  }
}

TEST(CudaKernels, AllGatesBfloatAndRealWidthsKeepFloatIntermediates) {
  for (const auto activation : {compute::GatedActivationKind::kSilu,
                                compute::GatedActivationKind::kGelu,
                                compute::GatedActivationKind::kGeluTanh}) {
    for (const bool packed : {false, true}) {
      for (const auto width : {129, 2048, 6144}) {
        SCOPED_TRACE(static_cast<int>(activation));
        SCOPED_TRACE(width);
        SCOPED_TRACE(packed);
        run_shape<__nv_bfloat16>(compute::DataType::kBFloat16, packed, 2, width,
                                 compute::cuda_native::provider(), activation);
        if (activation != compute::GatedActivationKind::kSilu) {
          run_shape<__half>(compute::DataType::kFloat16, packed, 2, width,
                            compute::cuda_native::provider(), activation);
          run_shape<float>(compute::DataType::kFloat32, packed, 2, width,
                           compute::cuda_native::provider(), activation);
        }
        if (packed && width % 8 == 0) {
          run_shape<__nv_bfloat16>(compute::DataType::kBFloat16, true, 2, width,
                                   compute::cuda_native::sglang_provider(),
                                   activation);
          if (activation != compute::GatedActivationKind::kSilu) {
            run_shape<__half>(compute::DataType::kFloat16, true, 2, width,
                              compute::cuda_native::sglang_provider(),
                              activation);
            run_shape<float>(compute::DataType::kFloat32, true, 2, width,
                             compute::cuda_native::sglang_provider(),
                             activation);
          }
        }
      }
    }
  }
}

TEST(CudaKernels, EmptyGatesDoNotRequireDevicePayloadAllocations) {
  for (const auto selected : {compute::cuda_native::provider(),
                              compute::cuda_native::sglang_provider()}) {
    auto owner = context(selected);
    auto queue = owner.create_queue().value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    auto empty = mem::Buffer::create(resource.ref(), 0, 256).value();
    EXPECT_EQ(empty.native_handle(), nullptr);
    for (const auto type :
         {compute::DataType::kFloat32, compute::DataType::kFloat16,
          compute::DataType::kBFloat16}) {
      for (const auto dimensions : {std::array<std::int64_t, 2>{0, 256},
                                    std::array<std::int64_t, 2>{2, 0}}) {
        const auto input = compute::TensorDesc::dense(type, dimensions).value();
        auto prepared = compute::prepare(owner, compute::GatedActivation{},
                                         input, resource.ref());
        ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
        auto invocation = prepared->create_invocation().value();
        auto done = invocation.submit(
            queue,
            compute::ConstTensorView::create(input, empty.view()).value(),
            compute::TensorView::create(prepared->output_description().value(),
                                        empty.view())
                .value());
        ASSERT_TRUE(done.has_value()) << done.error().message();
        ASSERT_TRUE(done->wait().ok());
      }
    }
  }
}

template <class T> void vector_binding_boundaries(compute::DataType type) {
  auto owner = context(compute::cuda_native::sglang_provider());
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto in = matrix(type, 1, 256), out = matrix(type, 1, 128);
  const auto in_bytes = in.required_bytes().value();
  const auto out_bytes = out.required_bytes().value();
  auto input = mem::Buffer::create(resource.ref(), in_bytes + 16, 256).value();
  auto output =
      mem::Buffer::create(resource.ref(), out_bytes + 16, 256).value();
  std::vector<T> values((in_bytes + 16) / sizeof(T), from_float<T>(1.0F));
  ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
  const auto first = compute::ConstTensorView::create(
                         in, input.view().subview(0, in_bytes).value())
                         .value();
  const auto shifted_first =
      compute::ConstTensorView::create(
          in, input.view().subview(sizeof(T), in_bytes).value())
          .value();
  const auto result = compute::TensorView::create(
                          out, output.view().subview(0, out_bytes).value())
                          .value();
  const auto shifted_result =
      compute::TensorView::create(
          out, output.view().subview(sizeof(T), out_bytes).value())
          .value();
  auto program =
      compute::prepare(owner, compute::Operation(compute::GatedActivation{}),
                       std::array{first}, std::array{result})
          .value();
  auto lane = program.create_invocation().value();
  auto queue = owner.create_queue().value();
  EXPECT_FALSE(lane.submit(queue, shifted_first, result).has_value());
  EXPECT_FALSE(lane.submit(queue, first, shifted_result).has_value());
  const auto alias = compute::TensorView::create(
                         out, input.view().subview(0, out_bytes).value())
                         .value();
  auto aliased = lane.submit(queue, first, alias);
  ASSERT_FALSE(aliased.has_value());
  EXPECT_TRUE(aliased.error().IsInvalidArgument());
  EXPECT_FALSE(lane.in_flight());
  auto done = lane.submit(queue, first, result);
  ASSERT_TRUE(done.has_value());
  ASSERT_TRUE(done->wait().ok());
  std::vector<T> actual(128);
  ASSERT_TRUE(
      result.storage().read(std::as_writable_bytes(std::span(actual))).ok());
  for (const auto value : actual)
    EXPECT_NEAR(to_float(value), 1.0 / (1.0 + std::exp(-1.0)),
                sizeof(T) == 2 ? 0.001 : 0.000001);

  // The scalar implementation retains its smaller native alignment. The
  // explicit expert choice requires no hidden copy or runtime route search.
  auto scalar_owner = context();
  auto scalar =
      compute::prepare(scalar_owner,
                       compute::Operation(compute::GatedActivation{}),
                       std::array{shifted_first}, std::array{shifted_result})
          .value();
  auto scalar_lane = scalar.create_invocation().value();
  auto scalar_queue = scalar_owner.create_queue().value();
  auto scalar_done =
      scalar_lane.submit(scalar_queue, shifted_first, shifted_result);
  ASSERT_TRUE(scalar_done.has_value());
  ASSERT_TRUE(scalar_done->wait().ok());
  ASSERT_TRUE(shifted_result.storage()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  for (const auto value : actual)
    EXPECT_NEAR(to_float(value), 1.0 / (1.0 + std::exp(-1.0)),
                sizeof(T) == 2 ? 0.001 : 0.000001);

  auto tail = compute::prepare(owner, compute::GatedActivation{},
                               matrix(type, 1, 254), resource.ref());
  ASSERT_FALSE(tail.has_value());
  EXPECT_TRUE(tail.error().IsUnimplemented());
  auto separate =
      compute::prepare(owner,
                       compute::Operation(compute::GatedActivation{
                           .inputs = compute::GatedInputKind::kSeparate}),
                       std::array{out, out}, std::array{out}, resource.ref());
  ASSERT_FALSE(separate.has_value());
  EXPECT_TRUE(separate.error().IsUnimplemented());
}

TEST(CudaKernels, VectorRejectionPreservesSlotAndScalarAlignmentCapability) {
  vector_binding_boundaries<float>(compute::DataType::kFloat32);
  vector_binding_boundaries<__half>(compute::DataType::kFloat16);
  vector_binding_boundaries<__nv_bfloat16>(compute::DataType::kBFloat16);
}

TEST(CudaKernels, RejectsUnsupportedShapesResourcesAndAliasing) {
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const compute::GatedActivation op{.inputs =
                                        compute::GatedInputKind::kSeparate};
  const auto desc = matrix(compute::DataType::kFloat32, 1, 128);
  auto unsupported = [&](compute::GatedActivation selected,
                         compute::TensorDesc input, compute::TensorDesc output,
                         mem::MemoryResourceRef memory) {
    return compute::prepare(owner, compute::Operation(selected),
                            std::array{input, input}, std::array{output},
                            memory);
  };
  EXPECT_FALSE(
      unsupported({.activation = static_cast<compute::GatedActivationKind>(255),
                   .inputs = compute::GatedInputKind::kSeparate},
                  desc, desc, resource.ref())
          .has_value());
  EXPECT_FALSE(unsupported(op, desc, desc, mem::system_resource()).has_value());
  const auto huge = matrix(compute::DataType::kFloat32, 2, INT32_MAX);
  EXPECT_FALSE(unsupported(op, huge, huge, resource.ref()).has_value());
  const auto strided =
      compute::TensorDesc::strided_bytes(compute::DataType::kFloat32,
                                         std::array<std::int64_t, 2>{2, 128},
                                         std::array<std::int64_t, 2>{1024, 4})
          .value();
  EXPECT_FALSE(unsupported(op, strided,
                           matrix(compute::DataType::kFloat32, 2, 128),
                           resource.ref())
                   .has_value());
  Tensor first(desc, resource.ref()), second(desc, resource.ref());
  auto alias = compute::prepare(
      owner, compute::Operation(op),
      std::array{first.view.as_const(), second.view.as_const()},
      std::array{first.view});
  // Preparation binds descriptions/resources. Actual storage overlap is
  // rejected at submit, before the Provider records a native command.
  ASSERT_TRUE(alias.has_value());
  auto lane = alias->create_invocation().value();
  auto queue = owner.create_queue().value();
  auto rejected = lane.submit(queue, first.view.as_const(),
                               second.view.as_const(), first.view);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  EXPECT_FALSE(lane.in_flight());
}

void pending_completion_retains_program(
    compute::Provider selected,
    compute::Operation operation =
        compute::Operation(compute::GatedActivation{})) {
  auto owner = context(selected);
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  Tensor input(
      matrix(compute::DataType::kFloat32, 1,
             operation.kind() == compute::OperationKind::kGatedActivation
                 ? 512
                 : 256),
      resource.ref());
  std::array<compute::TensorDesc, 1> output_desc;
  ASSERT_TRUE(compute::infer_output_descriptions(
                  operation, std::array{input.view.description()}, output_desc)
                  .ok());
  Tensor output(output_desc[0], resource.ref());
  std::vector<float> values(
      operation.kind() == compute::OperationKind::kGatedActivation ? 512 : 256,
      1.0F);
  ASSERT_TRUE(
      input.storage.view().write(std::as_bytes(std::span(values))).ok());
  const std::array inputs{input.view.as_const()};
  const std::array outputs{output.view};
  auto prepared = compute::prepare(owner, operation, inputs, outputs).value();
  auto lane = prepared.create_invocation().value();
  auto warm = lane.submit(queue, inputs, outputs).value();
  ASSERT_TRUE(warm.wait().ok());
  struct Gate final {
    std::atomic<bool> released{false};
    static void CUDART_CB wait(void *raw) {
      static_cast<Gate *>(raw)->released.wait(false, std::memory_order_acquire);
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
  auto done = lane.submit(queue, inputs, outputs);
  ASSERT_TRUE(done.has_value());
  EXPECT_EQ(done->poll().value(), compute::CompletionPoll::kPending);
  lane = {};
  prepared = {};
  ASSERT_TRUE(owner.close().ok());
  gate.release();
  ASSERT_TRUE(done->wait().ok());
  std::vector<float> actual(output.storage.size() / sizeof(float));
  ASSERT_TRUE(output.storage.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  const double expected =
      operation.kind() == compute::OperationKind::kReduce      ? 256
      : operation.kind() == compute::OperationKind::kSoftmax   ? 1.0 / 256
      : operation.kind() == compute::OperationKind::kLayerNorm ? 0
      : operation.kind() == compute::OperationKind::kRmsNorm
          ? 1 / std::sqrt(1.0 + 1e-5)
          : 1.0 / (1.0 + std::exp(-1.0));
  for (const auto value : actual)
    EXPECT_NEAR(value, expected, 0.000001);
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

void metadata_failures_rollback(
    compute::Provider selected,
    compute::Operation operation =
        compute::Operation(compute::GatedActivation{})) {
  MetadataAllocator allocator;
  {
    const std::array providers{selected};
    auto owner =
        compute::Context::create(
            cuda::device(0).value(),
            {.metadata_allocator = allocator.ref(), .providers = providers})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    Tensor input(
        matrix(compute::DataType::kFloat32, 1,
               operation.kind() == compute::OperationKind::kGatedActivation
                   ? 512
                   : 256),
        resource.ref());
    std::array<compute::TensorDesc, 1> output_desc;
    ASSERT_TRUE(
        compute::infer_output_descriptions(
            operation, std::array{input.view.description()}, output_desc)
            .ok());
    Tensor output(output_desc[0], resource.ref());
    const std::array inputs{input.view.as_const()};
    const std::array outputs{output.view};
    const auto prepare = [&] {
      return compute::prepare(owner, operation, inputs, outputs);
    };
    const auto begin = allocator.attempts;
    {
      auto prepared = prepare().value();
    }
    const auto allocations = allocator.attempts - begin;
    const auto retained = allocator.live;
    ASSERT_GT(allocations, 0u);
    ASSERT_LT(allocations, 32u);
    for (std::size_t offset = 0; offset < allocations; ++offset) {
      allocator.fail_at = allocator.attempts + offset;
      auto prepared = prepare();
      ASSERT_FALSE(prepared.has_value());
      EXPECT_TRUE(prepared.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, retained);
    }
    allocator.fail_at = SIZE_MAX;
    auto prepared = prepare().value();
    const auto lane_begin = allocator.attempts;
    {
      auto lane = prepared.create_invocation().value();
    }
    const auto lane_allocations = allocator.attempts - lane_begin;
    const auto lane_retained = allocator.live;
    ASSERT_GT(lane_allocations, 0u);
    ASSERT_LT(lane_allocations, 32u);
    for (std::size_t offset = 0; offset < lane_allocations; ++offset) {
      allocator.fail_at = allocator.attempts + offset;
      auto lane = prepared.create_invocation();
      ASSERT_FALSE(lane.has_value());
      EXPECT_TRUE(lane.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, lane_retained);
    }
    allocator.fail_at = SIZE_MAX;
    auto lane = prepared.create_invocation().value();
    auto queue = owner.create_queue().value();
    ASSERT_TRUE(queue.reserve_completions(1).ok());
    std::vector<float> values(
        operation.kind() == compute::OperationKind::kGatedActivation ? 512
                                                                     : 256,
        1.0F);
    ASSERT_TRUE(
        input.storage.view().write(std::as_bytes(std::span(values))).ok());
    const auto replay_begin = allocator.attempts;
    for (int iteration = 0; iteration < 3; ++iteration) {
      auto done = lane.submit(queue, inputs, outputs);
      ASSERT_TRUE(done.has_value());
      ASSERT_TRUE(done->wait().ok());
    }
    EXPECT_EQ(allocator.attempts, replay_begin);
  }
  EXPECT_EQ(allocator.live, 0u);
}

TEST(CudaKernels, PendingCompletionRetainsProgramAndSlotAfterClose) {
  pending_completion_retains_program(compute::cuda_native::provider());
  pending_completion_retains_program(compute::cuda_native::sglang_provider());
  pending_completion_retains_program(
      compute::cuda_native::provider(),
      compute::Operation(compute::Unary{compute::UnaryKind::kSilu}));
  pending_completion_retains_program(compute::cuda_native::provider(),
                                     compute::Operation(compute::RmsNorm{}));
  pending_completion_retains_program(compute::cuda_native::provider(),
                                     compute::Operation(compute::LayerNorm{}));
  pending_completion_retains_program(compute::cuda_native::provider(),
                                     compute::Operation(compute::Reduce{}));
  pending_completion_retains_program(compute::cuda_native::provider(),
                                     compute::Operation(compute::Softmax{}));
}

TEST(CudaKernels, ColdMetadataFailuresRollbackAndReplayUsesReservedState) {
  metadata_failures_rollback(compute::cuda_native::provider());
  metadata_failures_rollback(compute::cuda_native::sglang_provider());
  metadata_failures_rollback(
      compute::cuda_native::provider(),
      compute::Operation(compute::Unary{compute::UnaryKind::kSilu}));
  metadata_failures_rollback(compute::cuda_native::provider(),
                             compute::Operation(compute::RmsNorm{}));
  metadata_failures_rollback(compute::cuda_native::provider(),
                             compute::Operation(compute::LayerNorm{}));
  metadata_failures_rollback(compute::cuda_native::provider(),
                             compute::Operation(compute::Reduce{}));
  metadata_failures_rollback(compute::cuda_native::provider(),
                             compute::Operation(compute::Softmax{}));
}

} // namespace
