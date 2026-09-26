#include "hyle/compute/providers/cuda.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/low_precision.h"
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
        if (std::isnan(expected[index])) {
          EXPECT_TRUE(std::isnan(value));
        } else if (std::isinf(expected[index])) {
          EXPECT_EQ(value, expected[index]);
        } else {
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
  for (const auto &selected : {compute::cuda_native::provider(),
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

void append_index_operands(compute::Operation operation,
                           mem::MemoryResourceRef resource,
                           std::vector<compute::TensorDesc> &descriptions,
                           std::vector<compute::ConstTensorView> &inputs,
                           std::optional<Tensor> &indices,
                           std::optional<Tensor> &updates) {
  const bool scatter = operation.kind() == compute::OperationKind::kScatter;
  if (operation.kind() != compute::OperationKind::kGather && !scatter)
    return;
  indices.emplace(
      scatter ? compute::TensorDesc::dense(compute::DataType::kInt32,
                                           std::array<std::int64_t, 1>{2})
                    .value()
              : matrix(compute::DataType::kInt32, 1, 1),
      resource);
  const std::vector<std::int32_t> zero(scatter ? 2 : 1, 0);
  EXPECT_TRUE(
      indices->storage.view().write(std::as_bytes(std::span(zero))).ok());
  descriptions.push_back(indices->view.description());
  inputs.push_back(indices->view.as_const());
  if (scatter) {
    updates.emplace(matrix(compute::DataType::kFloat32, 2, 256), resource);
    const std::vector<float> ones(512, 1.0F);
    EXPECT_TRUE(
        updates->storage.view().write(std::as_bytes(std::span(ones))).ok());
    descriptions.push_back(updates->view.description());
    inputs.push_back(updates->view.as_const());
  }
}

void append_hash_operands(compute::Operation operation, mem::MemoryResourceRef resource,
                           std::vector<compute::TensorDesc> &descriptions,
                           std::vector<compute::ConstTensorView> &inputs,
                           std::optional<Tensor> &tokens, std::optional<Tensor> &table) {
  if (!operation.custom_as<compute::model_ops::SqrtSoftplusHashRouting>()) return;
  tokens.emplace(compute::TensorDesc::dense(compute::DataType::kInt64,
                                            std::array<std::int64_t, 1>{1}).value(), resource);
  table.emplace(matrix(compute::DataType::kInt32, 129280, 6), resource);
  const std::array<std::int64_t, 1> token{129279};
  std::vector<std::int32_t> lookup(129280 * 6);
  for (std::size_t i = 0; i < lookup.size(); ++i) lookup[i] = std::int32_t(i % 6);
  ASSERT_TRUE(tokens->storage.view().write(std::as_bytes(std::span(token))).ok());
  ASSERT_TRUE(table->storage.view().write(std::as_bytes(std::span(lookup))).ok());
  for (auto *tensor : {&*tokens, &*table}) {
    descriptions.push_back(tensor->view.description());
    inputs.push_back(tensor->view.as_const());
  }
}

void pending_completion_retains_program(
    compute::Provider selected,
    compute::Operation operation =
        compute::Operation(compute::GatedActivation{})) {
  const bool sglang_norm =
      selected.id() == compute::cuda_native::kSglangProviderId &&
      operation.kind() == compute::OperationKind::kRmsNorm;
  auto owner = context(selected);
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  Tensor input(
      matrix(sglang_norm ? compute::DataType::kFloat16
                         : compute::DataType::kFloat32,
             1,
             operation.kind() == compute::OperationKind::kGatedActivation
                 ? 512
                 : 256),
      resource.ref());
  std::array<compute::TensorDesc, 2> output_desc;
  const auto arity =
      operation.kind() == compute::OperationKind::kConcat ? 9U : 1U;
  std::vector input_desc(arity, input.view.description());
  std::vector inputs(arity, input.view.as_const());
  std::optional<Tensor> norm_weight;
  if (sglang_norm) {
    norm_weight.emplace(
        compute::TensorDesc::dense(compute::DataType::kFloat16,
                                   std::array<std::int64_t, 1>{256})
            .value(),
        resource.ref());
    const std::vector<__half> ones(256, __float2half(1));
    ASSERT_TRUE(
        norm_weight->storage.view().write(std::as_bytes(std::span(ones))).ok());
    input_desc.push_back(norm_weight->view.description());
    inputs.push_back(norm_weight->view.as_const());
  }
  std::optional<Tensor> gather_indices, scatter_updates;
  append_index_operands(operation, resource.ref(), input_desc, inputs,
                        gather_indices, scatter_updates);
  std::optional<Tensor> hash_tokens, hash_table;
  append_hash_operands(operation, resource.ref(), input_desc, inputs, hash_tokens, hash_table);
  const auto descriptions =
      std::span(output_desc)
          .first(compute::operation_traits(operation).outputs);
  ASSERT_TRUE(
      compute::infer_output_descriptions(operation, input_desc, descriptions)
          .ok());
  std::vector<Tensor> output_storage;
  std::vector<compute::TensorView> outputs;
  output_storage.reserve(descriptions.size());
  for (const auto &desc : descriptions) {
    output_storage.emplace_back(desc, resource.ref());
    outputs.push_back(output_storage.back().view);
  }
  std::vector<float> values(
      operation.kind() == compute::OperationKind::kGatedActivation ? 512 : 256,
      1.0F);
  if (sglang_norm) {
    const std::vector<__half> ones(256, __float2half(1));
    ASSERT_TRUE(
        input.storage.view().write(std::as_bytes(std::span(ones))).ok());
  } else {
    ASSERT_TRUE(
        input.storage.view().write(std::as_bytes(std::span(values))).ok());
  }
  auto prepared = compute::prepare(owner, operation, inputs, outputs).value();
  mem::Buffer scratch;
  if (const auto need = prepared.workspace_requirement(); need.size != 0)
    scratch =
        mem::Buffer::create(resource.ref(), need.size, need.alignment).value();
  auto lane = prepared.create_invocation(scratch.view()).value();
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
  const auto &output = output_storage[0];
  std::vector<float> actual;
  if (sglang_norm) {
    std::vector<__half> half_values(256);
    ASSERT_TRUE(output.storage.view()
                    .read(std::as_writable_bytes(std::span(half_values)))
                    .ok());
    for (const auto value : half_values)
      actual.push_back(__half2float(value));
  } else {
    actual.resize(output.storage.size() / sizeof(float));
    ASSERT_TRUE(output.storage.view()
                    .read(std::as_writable_bytes(std::span(actual)))
                    .ok());
  }
  const double expected =
      operation.custom_as<compute::model_ops::SqrtSoftplusHashRouting>() ? 1.0 / 6
      : operation.kind() == compute::OperationKind::kTopK ||
              operation.kind() == compute::OperationKind::kGather ||
              operation.kind() == compute::OperationKind::kScatter ||
              operation.kind() == compute::OperationKind::kConcat
          ? 1
      : operation.kind() == compute::OperationKind::kReduce    ? 256
      : operation.kind() == compute::OperationKind::kSoftmax   ? 1.0 / 256
      : operation.kind() == compute::OperationKind::kLayerNorm ? 0
      : operation.kind() == compute::OperationKind::kRmsNorm
          ? (sglang_norm ? 1.0 : 1 / std::sqrt(1.0 + 1e-5))
          : 1.0 / (1.0 + std::exp(-1.0));
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_NEAR(actual[index],
                operation.kind() == compute::OperationKind::kScan
                    ? static_cast<double>(index + 1)
                    : expected,
                0.000001);
  if (operation.kind() == compute::OperationKind::kTopK) {
    std::vector<std::int32_t> indices(actual.size());
    ASSERT_TRUE(output_storage[1]
                    .storage.view()
                    .read(std::as_writable_bytes(std::span(indices)))
                    .ok());
    for (std::size_t index = 0; index < indices.size(); ++index)
      EXPECT_EQ(indices[index], index);
  }
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
  const bool sglang_norm =
      selected.id() == compute::cuda_native::kSglangProviderId &&
      operation.kind() == compute::OperationKind::kRmsNorm;
  MetadataAllocator allocator;
  {
    const std::array providers{hyle::compute::ProviderBinding{selected, 100}};
    auto owner =
        compute::Context::create(
            cuda::device(0).value(),
            {.metadata_allocator = allocator.ref(), .providers = providers})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    Tensor input(
        matrix(sglang_norm ? compute::DataType::kFloat16
                           : compute::DataType::kFloat32,
               1,
               operation.kind() == compute::OperationKind::kGatedActivation
                   ? 512
                   : 256),
        resource.ref());
    std::array<compute::TensorDesc, 2> output_desc;
    const auto arity =
        operation.kind() == compute::OperationKind::kConcat ? 9U : 1U;
    std::vector input_desc(arity, input.view.description());
    std::vector inputs(arity, input.view.as_const());
    std::optional<Tensor> norm_weight;
    if (sglang_norm) {
      norm_weight.emplace(
          compute::TensorDesc::dense(compute::DataType::kFloat16,
                                     std::array<std::int64_t, 1>{256})
              .value(),
          resource.ref());
      const std::vector<__half> ones(256, __float2half(1));
      ASSERT_TRUE(norm_weight->storage.view()
                      .write(std::as_bytes(std::span(ones)))
                      .ok());
      input_desc.push_back(norm_weight->view.description());
      inputs.push_back(norm_weight->view.as_const());
    }
    std::optional<Tensor> gather_indices, scatter_updates;
    append_index_operands(operation, resource.ref(), input_desc, inputs,
                          gather_indices, scatter_updates);
    std::optional<Tensor> hash_tokens, hash_table;
    append_hash_operands(operation, resource.ref(), input_desc, inputs, hash_tokens, hash_table);
    const auto descriptions =
        std::span(output_desc)
            .first(compute::operation_traits(operation).outputs);
    ASSERT_TRUE(
        compute::infer_output_descriptions(operation, input_desc, descriptions)
            .ok());
    std::vector<Tensor> output_storage;
    std::vector<compute::TensorView> outputs;
    output_storage.reserve(descriptions.size());
    for (const auto &desc : descriptions) {
      output_storage.emplace_back(desc, resource.ref());
      outputs.push_back(output_storage.back().view);
    }
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
    mem::Buffer scratch;
    if (const auto need = prepared.workspace_requirement(); need.size != 0)
      scratch = mem::Buffer::create(resource.ref(), need.size, need.alignment)
                    .value();
    const auto lane_begin = allocator.attempts;
    {
      auto lane = prepared.create_invocation(scratch.view()).value();
    }
    const auto lane_allocations = allocator.attempts - lane_begin;
    const auto lane_retained = allocator.live;
    ASSERT_GT(lane_allocations, 0u);
    ASSERT_LT(lane_allocations, 32u);
    for (std::size_t offset = 0; offset < lane_allocations; ++offset) {
      allocator.fail_at = allocator.attempts + offset;
      auto lane = prepared.create_invocation(scratch.view());
      ASSERT_FALSE(lane.has_value());
      EXPECT_TRUE(lane.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, lane_retained);
    }
    allocator.fail_at = SIZE_MAX;
    auto lane = prepared.create_invocation(scratch.view()).value();
    auto queue = owner.create_queue().value();
    ASSERT_TRUE(queue.reserve_completions(1).ok());
    std::vector<float> values(
        operation.kind() == compute::OperationKind::kGatedActivation ? 512
                                                                     : 256,
        1.0F);
    if (sglang_norm) {
      const std::vector<__half> ones(256, __float2half(1));
      ASSERT_TRUE(
          input.storage.view().write(std::as_bytes(std::span(ones))).ok());
    } else {
      ASSERT_TRUE(
          input.storage.view().write(std::as_bytes(std::span(values))).ok());
    }
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
  pending_completion_retains_program(compute::cuda_native::sglang_provider(),
                                     compute::model_ops::sqrt_softplus_hash_routing());
  pending_completion_retains_program(compute::cuda_native::provider(),
                                     compute::Operation(compute::Scatter{}));
  pending_completion_retains_program(compute::cuda_native::provider());
  pending_completion_retains_program(compute::cuda_native::sglang_provider());
  pending_completion_retains_program(compute::cuda_native::sglang_provider(),
                                     compute::Operation(compute::RmsNorm{}));
  pending_completion_retains_program(
      compute::cuda_native::sglang_provider(),
      compute::Operation(compute::Unary{compute::UnaryKind::kSilu}));
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
  pending_completion_retains_program(compute::cuda_native::provider(),
                                     compute::Operation(compute::Concat{}));
}


TEST(DeepSeekRoundtrip, WorkspaceRejectionsRollbackConcurrencyAndLifetime) {
  using Format = compute::model_ops::BlockQuantizationFormat;
  for (auto format : {Format::kFloat8E4M3Block64, Format::kFloat4E2M1Block32}) {
    MetadataAllocator allocator;
    {
      const std::array providers{compute::ProviderBinding{compute::cuda_native::deepseek_provider(), 100}};
      auto owner = compute::Context::create(cuda::device(0).value(),
          {.metadata_allocator = allocator.ref(), .providers = providers}).value();
      auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
      const std::int64_t group = format == Format::kFloat8E4M3Block64 ? 64 : 32;
      const auto desc = matrix(compute::DataType::kBFloat16, 33, group);
      const auto op = compute::model_ops::block_quantization_roundtrip(format);
      Tensor input(desc, resource.ref());
      std::array output{Tensor(desc, resource.ref()), Tensor(desc, resource.ref())};
      const std::array inputs{input.view.as_const()};
      const auto make = [&] { return compute::prepare(owner, op, inputs, std::array{output[0].view}); };
      const auto start = allocator.attempts, live = allocator.live;
      { auto program = make(); ASSERT_TRUE(program) << program.error().message(); }
      const auto count = allocator.attempts - start;
      ASSERT_GT(count, 0u);
      ASSERT_LT(count, 32u);
      for (std::size_t i = 0; i < count; ++i) {
        allocator.fail_at = allocator.attempts + i;
        auto failed = make();
        ASSERT_FALSE(failed);
        EXPECT_TRUE(failed.error().IsResourceExhausted());
        EXPECT_EQ(allocator.live, live);
        allocator.fail_at = SIZE_MAX;
      }
      const auto huge = matrix(compute::DataType::kBFloat16, std::int64_t(INT32_MAX) + 1, group);
      auto too_big = compute::prepare(owner, op, std::array{huge}, std::array{huge}, resource.ref());
      ASSERT_FALSE(too_big);
      EXPECT_TRUE(too_big.error().IsUnimplemented());
      EXPECT_FALSE(compute::prepare(owner, op, std::array{desc}, std::array{desc}, mem::system_resource()));
      auto program = make().value();
      EXPECT_EQ(program.provider().id, compute::cuda_native::kDeepSeekProviderId);
      EXPECT_EQ(program.workspace_requirement().size, 33u * sizeof(float));
      EXPECT_EQ(program.invocation_private_bytes(), 0u);
      EXPECT_TRUE(program.capture_compatible());
      auto short_workspace = mem::Buffer::create(resource.ref(), 128, 256).value();
      EXPECT_FALSE(program.create_invocation(short_workspace.view()));
      auto host_workspace = mem::Buffer::create(mem::system_resource(), 132, 256).value();
      EXPECT_FALSE(program.create_invocation(host_workspace.view()));
      std::array workspace{mem::Buffer::create(resource.ref(), 132, 256).value(),
                           mem::Buffer::create(resource.ref(), 132, 256).value()};
      const auto before_lane = allocator.attempts, lane_live = allocator.live;
      { auto lane = program.create_invocation(workspace[0].view()); ASSERT_TRUE(lane) << lane.error().message(); }
      const auto lane_count = allocator.attempts - before_lane;
      ASSERT_LT(lane_count, 16u);
      for (std::size_t i = 0; i < lane_count; ++i) {
        allocator.fail_at = allocator.attempts + i;
        auto failed = program.create_invocation(workspace[0].view());
        ASSERT_FALSE(failed);
        EXPECT_TRUE(failed.error().IsResourceExhausted());
        EXPECT_EQ(allocator.live, lane_live);
        allocator.fail_at = SIZE_MAX;
      }
      std::array lanes{program.create_invocation(workspace[0].view()).value(),
                       program.create_invocation(workspace[1].view()).value()};
      std::array queues{owner.create_queue().value(), owner.create_queue().value()};
      for (auto &queue : queues) ASSERT_TRUE(queue.reserve_completions(1).ok());
      auto reject = [&](const auto &in, const auto &out) {
        auto result = lanes[0].submit(queues[0], in, out);
        EXPECT_FALSE(result);
        EXPECT_FALSE(lanes[0].in_flight());
      };
      reject(inputs, std::array{input.view});
      Tensor host_input(desc, mem::system_resource());
      reject(std::array{host_input.view.as_const()}, std::array{output[0].view});
      auto unaligned = mem::Buffer::create(resource.ref(), desc.required_bytes().value() + 2, 256).value();
      const auto wrong_alignment = compute::ConstTensorView::create(desc, unaligned.view().subview(2).value()).value();
      reject(std::array{wrong_alignment}, std::array{output[0].view});
      for (int replay = 0; replay < 2; ++replay) {
        std::vector<__nv_bfloat16> values(std::size_t(33 * group), __float2bfloat16(float(replay + 1)));
        ASSERT_TRUE(input.storage.view().write(std::as_bytes(std::span(values))).ok());
        const auto before = allocator.attempts;
        std::array<compute::Completion, 2> done;
        for (std::size_t i = 0; i < lanes.size(); ++i) {
          auto completion = lanes[i].submit(queues[i], inputs, std::array{output[i].view});
          ASSERT_TRUE(completion) << completion.error().message();
          done[i] = std::move(*completion);
        }
        if (replay == 1) { program = {}; lanes = {}; ASSERT_TRUE(owner.close().ok()); }
        for (std::size_t i = 0; i < done.size(); ++i) {
          ASSERT_TRUE(done[i].wait().ok());
          ASSERT_TRUE(output[i].storage.view().read(std::as_writable_bytes(std::span(values))).ok());
          for (auto value : values) ASSERT_EQ(__bfloat162float(value), float(replay + 1));
        }
        EXPECT_EQ(allocator.attempts, before);
      }
    }
    EXPECT_EQ(allocator.live, 0u);
  }
}

TEST(DeepSeekFp8Decode, AllCodesTailAndOutputGuard) {
  auto owner = context(compute::cuda_native::deepseek_provider());
  auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  constexpr std::int64_t n = 129, k = 256;
  const auto wd = compute::float8_weight_description(k, n,
      compute::Float8ScaleMode::kFloat32Block128).value();
  Tensor source(wd, resource.ref());
  std::vector<std::byte> storage(wd.required_bytes().value());
  for (bool bf16 : {false, true}) {
    const auto type = bf16 ? compute::DataType::kBFloat16 : compute::DataType::kFloat32;
    const auto od = matrix(type, n, k);
    const auto size = od.required_bytes().value();
    auto output = mem::Buffer::create(resource.ref(), size + 64, 256).value();
    const auto view = compute::TensorView::create(od, output.view().subview(0, size).value()).value();
    auto program = compute::prepare(owner, compute::model_ops::float8_weight_decode(type),
        std::array{source.view.as_const()}, std::array{view}).value();
    auto lane = program.create_invocation().value();
    for (int replay = 0; replay < 2; ++replay) {
      SCOPED_TRACE(bf16);
      SCOPED_TRACE(replay);
      std::array<float, 4> scales{0, 0.0012345F, -1.75F, 2.03125F};
      if (replay) std::reverse(scales.begin(), scales.end());
      std::memcpy(storage.data() + n * k, scales.data(), sizeof(scales));
      for (std::size_t i = 0; i < n * k; ++i)
        storage[i] = std::byte{std::uint8_t((i + replay * 37) % 256)};
      ASSERT_TRUE(source.storage.view().write(storage).ok());
      const std::vector<std::byte> sentinel(size + 64, std::byte{0x5a});
      ASSERT_TRUE(output.view().write(sentinel).ok());
      auto done = lane.submit(queue, std::array{source.view.as_const()}, std::array{view});
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      std::vector<std::byte> actual(size + 64);
      ASSERT_TRUE(output.view().read(actual).ok());
      for (std::size_t i = size; i < actual.size(); ++i) ASSERT_EQ(actual[i], std::byte{0x5a});
      for (std::size_t i = 0; i < n * k; ++i) {
        const auto code = std::to_integer<unsigned>(storage[i]);
        float value;
        if (bf16) {
          __nv_bfloat16 stored;
          std::memcpy(&stored, actual.data() + i * 2, 2);
          value = __bfloat162float(stored);
        } else std::memcpy(&value, actual.data() + i * 4, 4);
        if ((code & 127) == 127) { ASSERT_TRUE(std::isnan(value)) << i; continue; }
        const auto exponent = (code >> 3) & 15, mantissa = code & 7;
        const float magnitude = exponent ? std::ldexp(1.0F + float(mantissa) / 8, int(exponent) - 7)
                                          : std::ldexp(float(mantissa), -9);
        float expected = std::copysign(magnitude, code & 128 ? -1.0F : 1.0F) *
                         scales[((i / k) / 128) * 2 + (i % k) / 128];
        if (bf16) expected = __bfloat162float(__float2bfloat16(expected));
        ASSERT_EQ(value, expected) << i;
        if (expected == 0) EXPECT_EQ(std::signbit(value), std::signbit(expected)) << i;
      }
    }
  }
}

TEST(DeepSeekFp8Decode, RejectionsRollbackConcurrentReplayAndCompletionOwnership) {
  for (bool bf16 : {false, true}) {
    MetadataAllocator allocator;
    {
      const std::array providers{compute::ProviderBinding{compute::cuda_native::deepseek_provider(), 100}};
      auto owner = compute::Context::create(cuda::device(0).value(),
          {.metadata_allocator = allocator.ref(), .providers = providers}).value();
      auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
      const auto type = bf16 ? compute::DataType::kBFloat16 : compute::DataType::kFloat32;
      constexpr auto mode = compute::Float8ScaleMode::kFloat32Block128;
      constexpr std::int64_t n = 129, k = 128;
      const auto wd = compute::float8_weight_description(k, n, mode).value();
      const auto od = matrix(type, n, k);
      const auto operation = compute::model_ops::float8_weight_decode(type);
      Tensor input(wd, resource.ref());
      std::array output{Tensor(od, resource.ref()), Tensor(od, resource.ref())};
      const std::array inputs{input.view.as_const()};
      const auto make = [&] { return compute::prepare(owner, operation, inputs, std::array{output[0].view}); };
      const auto start = allocator.attempts, live = allocator.live;
      { auto program = make(); ASSERT_TRUE(program) << program.error().message(); }
      const auto count = allocator.attempts - start;
      ASSERT_GT(count, 0u);
      ASSERT_LT(count, 32u);
      for (std::size_t i = 0; i < count; ++i) {
        allocator.fail_at = allocator.attempts + i;
        auto failed = make();
        ASSERT_FALSE(failed);
        EXPECT_TRUE(failed.error().IsResourceExhausted());
        EXPECT_EQ(allocator.live, live);
        allocator.fail_at = SIZE_MAX;
      }
      for (const auto shape : {std::array<std::int64_t, 2>{1LL << 24, 128}, {1, 65536LL * 128}}) {
        const auto w = compute::float8_weight_description(shape[1], shape[0], mode).value();
        auto rejected = compute::prepare(owner, operation, std::array{w},
            std::array{matrix(type, shape[0], shape[1])}, resource.ref());
        ASSERT_FALSE(rejected);
        EXPECT_TRUE(rejected.error().IsUnimplemented()) << rejected.error().message();
      }
      auto host = compute::prepare(owner, operation, std::array{wd}, std::array{od}, mem::system_resource());
      EXPECT_FALSE(host);
      auto prepared = make().value();
      EXPECT_EQ(prepared.provider().id, compute::cuda_native::kDeepSeekProviderId);
      EXPECT_EQ(prepared.workspace_requirement().size, 0u);
      EXPECT_EQ(prepared.invocation_private_bytes(), 0u);
      EXPECT_TRUE(prepared.capture_compatible());
      std::array lanes{prepared.create_invocation().value(), prepared.create_invocation().value()};
      std::array queues{owner.create_queue().value(), owner.create_queue().value()};
      for (auto &queue : queues) ASSERT_TRUE(queue.reserve_completions(1).ok());
      const auto reject = [&](std::span<const compute::ConstTensorView> in,
                              std::span<const compute::TensorView> out) {
        auto done = lanes[0].submit(queues[0], in, out);
        EXPECT_FALSE(done);
        EXPECT_FALSE(lanes[0].in_flight());
      };
      reject({}, std::array{output[0].view});
      Tensor host_input(wd, mem::system_resource());
      reject(std::array{host_input.view.as_const()}, std::array{output[0].view});
      auto unaligned = mem::Buffer::create(resource.ref(), wd.required_bytes().value() + 1, 256).value();
      // Encoded format alignment is rejected when forming the public view,
      // before a malformed binding can reach Invocation::submit.
      auto misaligned = compute::ConstTensorView::create(wd, unaligned.view().subview(1).value());
      ASSERT_FALSE(misaligned);
      EXPECT_TRUE(misaligned.error().IsInvalidArgument());
      reject(std::array{compute::ConstTensorView::create(wd, output[0].storage.view()).value()},
             std::array{output[0].view});
      std::vector<std::byte> bytes(wd.required_bytes().value(), std::byte{0x38});
      const float scale = 1.0F;
      for (int block = 0; block < 2; ++block) std::memcpy(bytes.data() + n*k + block*4, &scale, 4);
      for (int replay = 0; replay < 2; ++replay) {
        std::fill_n(bytes.begin(), n*k, replay ? std::byte{0x40} : std::byte{0x38});
        ASSERT_TRUE(input.storage.view().write(bytes).ok());
        const auto before = allocator.attempts;
        std::array<compute::Completion, 2> done;
        for (std::size_t i = 0; i < lanes.size(); ++i) {
          auto result = lanes[i].submit(queues[i], inputs, std::array{output[i].view});
          ASSERT_TRUE(result) << result.error().message();
          done[i] = std::move(*result);
        }
        if (replay == 1) {
          prepared = {};
          lanes = {};
          ASSERT_TRUE(owner.close().ok());
        }
        for (std::size_t i = 0; i < done.size(); ++i) {
          ASSERT_TRUE(done[i].wait().ok());
          if (bf16) {
            std::vector<__nv_bfloat16> actual(n*k);
            ASSERT_TRUE(output[i].storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
            for (auto value : actual) ASSERT_EQ(__bfloat162float(value), float(replay + 1));
          } else {
            std::vector<float> actual(n*k);
            ASSERT_TRUE(output[i].storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
            for (auto value : actual) ASSERT_EQ(value, float(replay + 1));
          }
        }
        EXPECT_EQ(allocator.attempts, before);
      }
    }
    EXPECT_EQ(allocator.live, 0u);
  }
}

TEST(SglangFp8, AllocationRollbackConcurrentReplayAndProgramLifetime) {
  MetadataAllocator allocator;
  {
    const std::array providers{compute::ProviderBinding{
        compute::cuda_native::sglang_provider(), 100}};
    auto owner = compute::Context::create(cuda::device(0).value(),
        {.metadata_allocator = allocator.ref(), .providers = providers}).value();
    auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    constexpr auto mode = compute::Float8ScaleMode::kFloat32Block128;
    constexpr std::int64_t m = 3, n = 129, k = 128;
    const auto ad = compute::float8_activation_description(m, k, mode).value();
    const auto bd = compute::float8_weight_description(k, n, mode).value();
    const auto cd = matrix(compute::DataType::kFloat32, m, n);
    Tensor a(ad, resource.ref()), b(bd, resource.ref());
    std::array output{Tensor(cd, resource.ref()), Tensor(cd, resource.ref())};
    const std::array inputs{a.view.as_const(), b.view.as_const()};
    const auto make = [&] {
      return compute::prepare(owner, compute::Operation(compute::MatMul{}),
                              inputs, std::array{output[0].view});
    };
    const auto start = allocator.attempts, live = allocator.live;
    { auto program = make(); ASSERT_TRUE(program) << program.error().message(); }
    const auto count = allocator.attempts - start;
    ASSERT_GT(count, 0u);
    ASSERT_LT(count, 32u);
    for (std::size_t i = 0; i < count; ++i) {
      SCOPED_TRACE(i);
      allocator.fail_at = allocator.attempts + i;
      auto failed = make();
      ASSERT_FALSE(failed);
      EXPECT_TRUE(failed.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, live);
      allocator.fail_at = SIZE_MAX;
    }
    auto prepared = make().value();
    EXPECT_EQ(prepared.provider().id, compute::cuda_native::kSglangProviderId);
    EXPECT_EQ(prepared.workspace_requirement().size, 0u);
    EXPECT_EQ(prepared.invocation_private_bytes(), 0u);
    EXPECT_TRUE(prepared.capture_compatible());
    std::array lanes{prepared.create_invocation().value(), prepared.create_invocation().value()};
    std::array queues{owner.create_queue().value(), owner.create_queue().value()};
    for (auto &queue : queues) ASSERT_TRUE(queue.reserve_completions(1).ok());
    std::vector<std::byte> ab(ad.required_bytes().value(), std::byte{0x38});
    std::vector<std::byte> bb(bd.required_bytes().value(), std::byte{0x38});
    const float scale = 1.0F;
    for (int i = 0; i < m; ++i) std::memcpy(ab.data() + m*k + i*4, &scale, 4);
    for (int i = 0; i < 2; ++i) std::memcpy(bb.data() + n*k + i*4, &scale, 4);
    ASSERT_TRUE(b.storage.view().write(bb).ok());
    for (int replay = 0; replay < 2; ++replay) {
      std::fill_n(ab.begin(), m*k, replay ? std::byte{0x40} : std::byte{0x38});
      ASSERT_TRUE(a.storage.view().write(ab).ok());
      const auto before = allocator.attempts;
      std::array<compute::Completion, 2> done;
      for (std::size_t i = 0; i < lanes.size(); ++i) {
        auto result = lanes[i].submit(queues[i], inputs, std::array{output[i].view});
        ASSERT_TRUE(result) << result.error().message();
        done[i] = std::move(*result);
      }
      if (replay == 1) {
        prepared = {};
        lanes = {};
        // Only real pending completions now retain native modules and slots.
        ASSERT_TRUE(owner.close().ok());
      }
      for (std::size_t i = 0; i < done.size(); ++i) {
        ASSERT_TRUE(done[i].wait().ok());
        std::vector<float> actual(m*n);
        ASSERT_TRUE(output[i].storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
        for (float value : actual) EXPECT_EQ(value, float(k * (replay + 1)));
      }
      EXPECT_EQ(allocator.attempts, before);
    }
  }
  EXPECT_EQ(allocator.live, 0u);
}

TEST(SglangFp8, DynamicWorkspaceOomIsolationReplayAndCompletionOwnership) {
  for (bool bf16 : {false, true}) {
    SCOPED_TRACE(bf16);
    MetadataAllocator allocator;
    {
      const std::array providers{compute::ProviderBinding{compute::cuda_native::sglang_provider(), 100}};
      auto owner = compute::Context::create(cuda::device(0).value(),
          {.metadata_allocator = allocator.ref(), .providers = providers}).value();
      auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
      const auto ad = matrix(bf16 ? compute::DataType::kBFloat16 : compute::DataType::kFloat32, 3, 128);
      const auto bd = compute::float8_weight_description(128, 129, compute::Float8ScaleMode::kFloat32Block128).value();
      const auto cd = matrix(ad.data_type(), 3, 129);
      std::array inputs{Tensor(ad, resource.ref()), Tensor(ad, resource.ref())};
      Tensor weight(bd, resource.ref());
      std::array outputs{Tensor(cd, resource.ref()), Tensor(cd, resource.ref())};
      const auto op = compute::model_ops::float8_block_linear(
          bf16 ? compute::model_ops::Float8ScaleRounding::kPowerOfTwo
               : compute::model_ops::Float8ScaleRounding::kNone, bf16 ? 1e-4F : 1e-10F);
      const auto make = [&] { return compute::prepare(owner, op,
          std::array{inputs[0].view.as_const(), weight.view.as_const()}, std::array{outputs[0].view}); };
      const auto check_failures = [&](auto create) {
        const auto live = allocator.live, start = allocator.attempts;
        { auto value = create(); ASSERT_TRUE(value) << value.error().message(); }
        const auto count = allocator.attempts - start;
        ASSERT_GT(count, 0u); ASSERT_LT(count, 32u);
        for (std::size_t i = 0; i < count; ++i) {
          allocator.fail_at = allocator.attempts + i;
          { auto result = create(); ASSERT_FALSE(result); EXPECT_TRUE(result.error().IsResourceExhausted()); }
          EXPECT_EQ(allocator.live, live);
          allocator.fail_at = SIZE_MAX;
        }
      };
      check_failures(make);
      auto prepared = make().value();
      EXPECT_EQ(prepared.workspace_requirement().size, 3u*128 + 3u*4);
      EXPECT_GT(prepared.invocation_private_bytes().value(), 0u);
      EXPECT_TRUE(prepared.capture_compatible());
      const auto need = prepared.workspace_requirement();
      std::array workspace{
          mem::Buffer::create(resource.ref(), need.size, need.alignment).value(),
          mem::Buffer::create(resource.ref(), need.size, need.alignment).value()};
      check_failures([&] { return prepared.create_invocation(workspace[0].view()); });
      std::array lanes{prepared.create_invocation(workspace[0].view()).value(),
                       prepared.create_invocation(workspace[1].view()).value()};
      std::array queues{owner.create_queue().value(), owner.create_queue().value()};
      for (auto &queue : queues) ASSERT_TRUE(queue.reserve_completions(1).ok());
      std::vector<std::byte> weights(bd.required_bytes().value(), std::byte{0x38});
      const float scale = 1;
      std::memcpy(weights.data()+129*128, &scale, 4);
      std::memcpy(weights.data()+129*128+4, &scale, 4);
      ASSERT_TRUE(weight.storage.view().write(weights).ok());
      for (std::size_t lane = 0; lane < 2; ++lane) {
        if (bf16) {
          std::vector<__nv_bfloat16> x(3*128, __float2bfloat16(float(lane+1)));
          ASSERT_TRUE(inputs[lane].storage.view().write(std::as_bytes(std::span(x))).ok());
        } else {
          std::vector<float> x(3*128, float(lane+1));
          ASSERT_TRUE(inputs[lane].storage.view().write(std::as_bytes(std::span(x))).ok());
        }
      }
      for (int replay = 0; replay < 2; ++replay) {
        const auto before = allocator.attempts;
        std::array<compute::Completion, 2> done;
        for (std::size_t lane = 0; lane < 2; ++lane) {
          auto result = lanes[lane].submit(queues[lane],
              std::array{inputs[lane].view.as_const(), weight.view.as_const()}, std::array{outputs[lane].view});
          ASSERT_TRUE(result) << result.error().message();
          done[lane] = std::move(*result);
        }
        if (replay == 1) { prepared = {}; lanes = {}; ASSERT_TRUE(owner.close().ok()); }
        for (std::size_t lane = 0; lane < 2; ++lane) {
          ASSERT_TRUE(done[lane].wait().ok());
          if (bf16) {
            std::vector<__nv_bfloat16> actual(3*129);
            ASSERT_TRUE(outputs[lane].storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
            for (auto value : actual) EXPECT_EQ(__bfloat162float(value), float(128*(lane+1)));
          } else {
            std::vector<float> actual(3*129);
            ASSERT_TRUE(outputs[lane].storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
            for (auto value : actual) EXPECT_NEAR(value, float(128*(lane+1)), 1e-4F);
          }
        }
        EXPECT_EQ(allocator.attempts, before);
      }
    }
    EXPECT_EQ(allocator.live, 0u);
  }
}

template <class T>
void dynamic_fp8_vector_boundaries(compute::DataType type) {
  auto owner = context(compute::cuda_native::sglang_provider());
  auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto weight_desc = compute::float8_weight_description(
      128, 1, compute::Float8ScaleMode::kFloat32Block128).value();
  Tensor weight(weight_desc, resource.ref());
  std::vector<std::byte> weights(132, std::byte{0x38});
  const float scale = 1;
  std::memcpy(weights.data() + 128, &scale, sizeof(scale));
  ASSERT_TRUE(weight.storage.view().write(weights).ok());
  for (int rows : {1, 2, 3, 4, 8, 16}) {
    SCOPED_TRACE(rows);
    const auto desc = matrix(type, rows, 128);
    const auto size = desc.required_bytes().value();
    auto input = mem::Buffer::create(resource.ref(), size + 16, 256).value();
    const auto aligned = compute::ConstTensorView::create(
        desc, input.view().subview(0, size).value()).value();
    const auto shifted = compute::ConstTensorView::create(
        desc, input.view().subview(sizeof(T), size).value()).value();
    std::vector<T> values(rows * 128, from_float<T>(1));
    ASSERT_TRUE(input.view().subview(0, size).value().write(std::as_bytes(std::span(values))).ok());
    Tensor output(matrix(type, rows, 1), resource.ref());
    auto program = compute::prepare(owner, compute::model_ops::float8_block_linear(),
        std::array{aligned, weight.view.as_const()}, std::array{output.view}).value();
    const auto need = program.workspace_requirement();
    auto workspace = mem::Buffer::create(resource.ref(), need.size, need.alignment).value();
    auto lane = program.create_invocation(workspace.view()).value();
    auto queue = owner.create_queue().value();
    const auto rejected = lane.submit(queue, std::array{shifted, weight.view.as_const()},
                                      std::array{output.view});
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsInvalidArgument());
    EXPECT_FALSE(lane.in_flight());
    auto done = lane.submit(queue, std::array{aligned, weight.view.as_const()},
                            std::array{output.view});
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    std::vector<T> actual(rows);
    ASSERT_TRUE(output.storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
    for (auto value : actual) EXPECT_NEAR(to_float(value), 128.0F, 1e-4F);
  }
}

TEST(SglangFp8, DynamicVectorAlignmentAndWholeGroupLaunch) {
  dynamic_fp8_vector_boundaries<float>(compute::DataType::kFloat32);
  dynamic_fp8_vector_boundaries<__nv_bfloat16>(compute::DataType::kBFloat16);
}

TEST(SglangFp8, GroupedWorkspaceOomConcurrencyCompletionAndInvalidIds) {
  for (int experts : {9, 256}) {
    SCOPED_TRACE(experts);
    MetadataAllocator allocator;
    {
      const std::array providers{compute::ProviderBinding{compute::cuda_native::sglang_provider(), 100}};
      auto owner = compute::Context::create(cuda::device(0).value(),
          {.metadata_allocator = allocator.ref(), .providers = providers}).value();
      auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
      const auto a = compute::TensorDesc::dense(compute::DataType::kFloat32,
          std::array<std::int64_t, 3>{3, 1, 128}).value();
      const auto w = compute::float8_grouped_weight_description(experts, 129, 128).value();
      const auto id = matrix(compute::DataType::kInt32, 3, 8);
      const auto c = compute::TensorDesc::dense(compute::DataType::kFloat32,
          std::array<std::int64_t, 3>{3, 8, 129}).value();
      Tensor input(a, resource.ref()), weight(w, resource.ref());
      std::array ids{Tensor(id, resource.ref()), Tensor(id, resource.ref())};
      std::array outputs{Tensor(c, resource.ref()), Tensor(c, resource.ref())};
      const auto make = [&] { return compute::prepare(owner, compute::model_ops::grouped_float8_linear(),
          std::array{weight.view.as_const(), input.view.as_const(), ids[0].view.as_const()},
          std::array{outputs[0].view}); };
      const auto check_failures = [&](auto create) {
        const auto live = allocator.live, start = allocator.attempts;
        { auto result = create(); ASSERT_TRUE(result) << result.error().message(); }
        const auto count = allocator.attempts - start;
        ASSERT_GT(count, 0u); ASSERT_LT(count, 32u);
        for (std::size_t i = 0; i < count; ++i) {
          allocator.fail_at = allocator.attempts + i;
          { auto result = create(); ASSERT_FALSE(result); EXPECT_TRUE(result.error().IsResourceExhausted()); }
          EXPECT_EQ(allocator.live, live);
          allocator.fail_at = SIZE_MAX;
        }
      };
      check_failures(make);
      auto prepared = make().value();
      const auto need = prepared.workspace_requirement();
      EXPECT_EQ(need.alignment, 16u);
      EXPECT_GT(need.size, 3u * 128 + 12);
      std::array workspace{
          mem::Buffer::create(resource.ref(), need.size + 16, need.alignment).value(),
          mem::Buffer::create(resource.ref(), need.size, need.alignment).value()};
      EXPECT_FALSE(prepared.create_invocation(workspace[0].view().subview(0, need.size - 1).value()));
      EXPECT_FALSE(prepared.create_invocation(workspace[0].view().subview(4, need.size).value()));
      check_failures([&] { return prepared.create_invocation(workspace[0].view()); });
      std::array lanes{prepared.create_invocation(workspace[0].view()).value(),
                       prepared.create_invocation(workspace[1].view()).value()};
      std::array queues{owner.create_queue().value(), owner.create_queue().value()};
      for (auto &queue : queues) ASSERT_TRUE(queue.reserve_completions(1).ok());
      std::vector<std::byte> weights(w.required_bytes().value(), std::byte{0x38});
      for (int ex = 0; ex < experts; ++ex) for (int block = 0; block < 2; ++block) {
        const float scale = float(ex + 1);
        std::memcpy(weights.data() + experts * 129 * 128 + (ex * 2 + block) * 4, &scale, 4);
      }
      ASSERT_TRUE(weight.storage.view().write(weights).ok());
      std::vector<float> x(3 * 128, 1);
      ASSERT_TRUE(input.storage.view().write(std::as_bytes(std::span(x))).ok());
      for (int replay = 0; replay < 3; ++replay) {
        std::array<std::vector<std::int32_t>, 2> selected;
        for (int lane = 0; lane < 2; ++lane) {
          selected[lane].resize(24);
          for (int i = 0; i < 24; ++i) selected[lane][i] = (i + lane + replay) % experts;
          if (replay == 2 && lane == 1) {
            selected[lane][0] = -1; selected[lane][1] = experts;
            selected[lane][2] = INT32_MIN; selected[lane][3] = INT32_MAX;
          }
          ASSERT_TRUE(ids[lane].storage.view().write(std::as_bytes(std::span(selected[lane]))).ok());
        }
        const auto before = allocator.attempts;
        std::array<compute::Completion, 2> done;
        for (std::size_t lane = 0; lane < 2; ++lane) {
          auto result = lanes[lane].submit(queues[lane],
              std::array{weight.view.as_const(), input.view.as_const(), ids[lane].view.as_const()},
              std::array{outputs[lane].view});
          ASSERT_TRUE(result) << result.error().message();
          done[lane] = std::move(*result);
        }
        if (replay == 2) { prepared = {}; lanes = {}; ASSERT_TRUE(owner.close().ok()); }
        for (int lane = 0; lane < 2; ++lane) {
          const auto status = done[lane].wait();
          if (replay == 2 && lane == 1) { EXPECT_TRUE(status.IsOutOfRange()); continue; }
          ASSERT_TRUE(status.ok()) << status.message();
          std::vector<float> actual(24 * 129);
          ASSERT_TRUE(outputs[lane].storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
          for (int slot = 0; slot < 24; ++slot) for (int col = 0; col < 129; ++col) {
            const float expected = 128.0F * (selected[lane][slot] + 1);
            // Same predeclared F32 abs+relative gate as the native/entry batch.
            EXPECT_NEAR(actual[slot * 129 + col], expected, 2e-6F + 2e-6F * expected);
          }
        }
        EXPECT_EQ(allocator.attempts, before);
      }
    }
    EXPECT_EQ(allocator.live, 0u);
  }
}

TEST(SglangFp8, GroupedColdDomainAndActivationAlignment) {
  auto owner = context(compute::cuda_native::sglang_provider());
  auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto rejected = [&](std::int64_t experts, std::int64_t tokens,
                             std::int64_t slots, std::int64_t n, std::int64_t k) {
    const auto w = compute::float8_grouped_weight_description(experts, n, k).value();
    const auto a = compute::TensorDesc::dense(compute::DataType::kFloat32,
        std::array<std::int64_t, 3>{tokens, 1, k}).value();
    const auto ids = matrix(compute::DataType::kInt32, tokens, slots);
    const auto c = compute::TensorDesc::dense(compute::DataType::kFloat32,
        std::array<std::int64_t, 3>{tokens, slots, n}).value();
    auto result = compute::prepare(owner, compute::model_ops::grouped_float8_linear(),
        std::array{w, a, ids}, std::array{c}, resource.ref());
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
  };
  rejected(9, 1, 4, 129, 128);
  rejected(1023, 1, 8, 129, 128);
  rejected(9, 1LL << 25, 8, 129, 128);
  rejected(9, 1, 8, 1LL << 24, 128);
  for (auto type : {compute::DataType::kFloat32, compute::DataType::kBFloat16}) {
    const auto desc = compute::TensorDesc::dense(type, std::array<std::int64_t, 3>{1, 1, 128}).value();
    const auto size = desc.required_bytes().value();
    auto storage = mem::Buffer::create(resource.ref(), size + 16, 256).value();
    const auto aligned = compute::ConstTensorView::create(desc, storage.view().subview(0, size).value()).value();
    const auto shifted = compute::ConstTensorView::create(desc, storage.view().subview(4, size).value()).value();
    Tensor w(compute::float8_grouped_weight_description(9, 129, 128).value(), resource.ref());
    Tensor ids(matrix(compute::DataType::kInt32, 1, 8), resource.ref());
    Tensor c(compute::TensorDesc::dense(type, std::array<std::int64_t, 3>{1, 8, 129}).value(), resource.ref());
    auto prepared = compute::prepare(owner, compute::model_ops::grouped_float8_linear(),
        std::array{w.view.as_const(), aligned, ids.view.as_const()}, std::array{c.view}).value();
    auto need = prepared.workspace_requirement();
    auto workspace = mem::Buffer::create(resource.ref(), need.size, need.alignment).value();
    auto invocation = prepared.create_invocation(workspace.view()).value();
    auto queue = owner.create_queue().value();
    const auto result = invocation.submit(queue, std::array{w.view.as_const(), shifted, ids.view.as_const()},
                                          std::array{c.view});
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsInvalidArgument());
    EXPECT_FALSE(invocation.in_flight());
  }
}

TEST(SglangFp8, RejectsDifferentEncodingMalformedPayloadAndIndexOverflow) {
  auto owner = context(compute::cuda_native::sglang_provider());
  auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto rejected = [&](compute::TensorDesc a, compute::TensorDesc b) {
    const auto c = matrix(compute::DataType::kFloat32, a.dimensions()[0], b.dimensions()[1]);
    auto result = compute::prepare(owner, compute::Operation(compute::MatMul{}),
                                  std::array{a, b}, std::array{c}, resource.ref());
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
  };
  constexpr auto f32 = compute::Float8ScaleMode::kFloat32Block128;
  constexpr auto ue8 = compute::Float8ScaleMode::kUnsignedExponent8Group32;
  rejected(compute::float8_activation_description(3, 128, ue8).value(),
           compute::float8_weight_description(128, 129, ue8).value());
  rejected(matrix(compute::DataType::kFloat32, 3, 128),
           compute::float8_weight_description(128, 129, f32).value());
  auto a = compute::float8_activation_description(3, 128, f32).value();
  auto malformed = compute::TensorDesc::opaque(a.data_type(), a.dimensions(),
      a.storage_format(), a.required_bytes().value() + 4).value();
  rejected(malformed, compute::float8_weight_description(128, 129, f32).value());
  rejected(compute::float8_activation_description(1LL << 25, 128, f32).value(),
           compute::float8_weight_description(128, 129, f32).value());
}

TEST(CudaKernels, ColdMetadataFailuresRollbackAndReplayUsesReservedState) {
  metadata_failures_rollback(compute::cuda_native::sglang_provider(),
                             compute::model_ops::sqrt_softplus_hash_routing());
  metadata_failures_rollback(compute::cuda_native::provider(),
                             compute::Operation(compute::Scatter{}));
  metadata_failures_rollback(compute::cuda_native::provider());
  metadata_failures_rollback(compute::cuda_native::sglang_provider());
  metadata_failures_rollback(compute::cuda_native::sglang_provider(),
                             compute::Operation(compute::RmsNorm{}));
  metadata_failures_rollback(
      compute::cuda_native::sglang_provider(),
      compute::Operation(compute::Unary{compute::UnaryKind::kSilu}));
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
  metadata_failures_rollback(compute::cuda_native::provider(),
                             compute::Operation(compute::Concat{}));
}

TEST(CudaConcat, LaterGroupPointerFailureKeepsRealPrefixCompletion) {
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc = matrix(compute::DataType::kFloat32, 1, 129);
  Tensor input(desc, resource.ref());
  const std::vector<float> values(129, 3.0F);
  ASSERT_TRUE(
      input.storage.view().write(std::as_bytes(std::span(values))).ok());
  std::array<compute::TensorDesc, 9> descriptions;
  descriptions.fill(desc);
  auto program = compute::prepare(owner, compute::Concat{.axis = 1},
                                  descriptions, resource.ref())
                     .value();
  auto slot = program.create_invocation().value();
  Tensor output(program.output_description().value(), resource.ref());
  const std::vector<float> sentinel(9 * 129, -7.0F);
  ASSERT_TRUE(
      output.storage.view().write(std::as_bytes(std::span(sentinel))).ok());
  std::array<compute::ConstTensorView, 9> inputs;
  inputs.fill(input.view.as_const());
  // Public submit checks bindings before acceptance. This deliberately injects
  // the fault at the existing provider/Graph seam after that check, so the
  // second group's pointer failure is exercised without an illegal GPU access
  // or a device assertion that would poison the process's CUDA context.
  inputs[4] = {};
  ASSERT_TRUE(queue.reserve_completions(1).ok());
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
  auto sequence = compute::detail::QueueSequence::begin(queue).value();
  auto enqueued =
      compute::detail::OperationInvocationAccess::enqueue_prevalidated(
          slot, sequence, inputs, std::array{output.view});
  EXPECT_TRUE(enqueued.accepted);
  EXPECT_TRUE(enqueued.status.IsInvalidArgument());
  sequence.record(std::move(enqueued));
  auto done = sequence.finish();
  ASSERT_TRUE(done.has_value()) << done.error().message();
  EXPECT_EQ(done->poll().value(), compute::CompletionPoll::kPending);
  ASSERT_TRUE(owner.close().ok());
  gate.release();
  EXPECT_TRUE(done->wait().IsInvalidArgument());
  std::vector<float> actual(sentinel.size());
  ASSERT_TRUE(output.storage.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_EQ(actual[index], index < 4 * 129 ? 3.0F : -7.0F);
  // The prevalidated seam borrows program/slot owners through Completion.
  // Ordinary submit's retained-owner path is covered above for all nine inputs.
  slot = {};
  program = {};
}

} // namespace
