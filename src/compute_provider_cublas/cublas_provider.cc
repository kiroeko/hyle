#include "hyle/compute/providers/cublas.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <utility>

#include <cublasLt.h>

#include "hyle/compute/detail/capture_bindings.h"
#include "native_context.h"
#include "provider_internal.h"
#include "hyle/compute/model_ops.h"
#include "hyle/container/small_vector.h"

namespace hyle::compute::cublas {
namespace {

constexpr std::size_t kTensorAlignment = 16;
constexpr std::size_t kWorkspaceAlignment = 256;
constexpr std::size_t kMaximumWorkspaceBytes = 32 * 1024 * 1024;

// No mutable vendor handle is shared by concurrent prepare or execution.
struct CublasContext final {
  CUcontext primary = nullptr;
};

struct CublasRun final {
  std::array<cublasLtMatrixLayout_t, 3> layouts{};
  cublasLtMatmulAlgo_t algorithm{};
  std::array<std::size_t, 3> offsets{};
  std::array<std::size_t, 3> byte_strides{};
  std::int32_t batch_count = 1;
};

struct CublasPrepared final {
  CUcontext primary = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  container::SmallVector<CublasRun, 1, mem::AllocatorRef> runs;
  std::size_t workspace_bytes = 0;
  CublasPrepared(CUcontext context, mem::AllocatorRef allocator) noexcept
      : primary(context), runs(allocator) {}
};

struct CublasInvocation final {
  CUcontext primary = nullptr;
  cublasLtHandle_t handle = nullptr;
  void *workspace = nullptr;
};

Status translate(cublasStatus_t result, const char *operation) noexcept {
  if (result == CUBLAS_STATUS_SUCCESS)
    return Status::Ok();
  StatusCode code = StatusCode::Internal;
  switch (result) {
  case CUBLAS_STATUS_ALLOC_FAILED:
    code = StatusCode::ResourceExhausted;
    break;
  case CUBLAS_STATUS_INVALID_VALUE:
    code = StatusCode::InvalidArgument;
    break;
  case CUBLAS_STATUS_NOT_SUPPORTED:
  case CUBLAS_STATUS_ARCH_MISMATCH:
    code = StatusCode::Unimplemented;
    break;
  case CUBLAS_STATUS_NOT_INITIALIZED:
    code = StatusCode::FailedPrecondition;
    break;
  default:
    break;
  }
  // cuBLAS numbers are not CUDA Runtime/Driver numbers. Preserve the exact
  // vendor code with its cuBLAS operation name, using the existing unknown
  // native namespace instead of mislabelling it as a CUDA error domain.
  return Status::FromNative(code,
                            NativeError{NativeErrorDomain::Unknown,
                                        static_cast<std::int64_t>(result)},
                            operation);
}

struct ColdHandle final {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  ~ColdHandle() {
    if (preference != nullptr)
      (void)cublasLtMatmulPreferenceDestroy(preference);
    if (handle != nullptr)
      (void)cublasLtDestroy(handle);
  }
};

Status create_context(const Provider &, Device device, void *backend_state,
                      mem::AllocatorRef allocator, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("cuBLAS context output is null");
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("cuBLAS requires the CUDA backend");
  const CUcontext primary = cuda::detail::borrow_primary_context(backend_state);
  if (primary == nullptr)
    return Status::FailedPrecondition("cuBLAS CUDA context is invalid");
  void *storage =
      allocator.allocate(sizeof(CublasContext), alignof(CublasContext));
  if (storage == nullptr)
    return Status::ResourceExhausted("cuBLAS context allocation failed");
  *output = ::new (storage) CublasContext{primary};
  return Status::Ok();
}

void destroy_context(void *state, mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<CublasContext *>(state);
  if (context == nullptr)
    return;
  context->~CublasContext();
  allocator.deallocate(context, sizeof(CublasContext), alignof(CublasContext));
}

struct MatrixLayout final {
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::int64_t leading = 0;
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
};

Status matrix_layout(const TensorDesc &description,
                     MatrixLayout &layout) noexcept {
  if (!description.valid() || !description.is_plain() ||
      description.rank() < 2 ||
      (description.data_type() != DataType::kFloat16 &&
       description.data_type() != DataType::kBFloat16 &&
       description.data_type() != DataType::kFloat32)) {
    return Status::Unimplemented(
        "cuBLAS requires plain matrix FP16/BF16/FP32");
  }
  const auto dimensions = description.dimensions().last(2);
  const auto strides = description.byte_strides().last(2);
  const auto width =
      static_cast<std::int64_t>(data_type_size(description.data_type()));
  if (dimensions[0] <= 0 || dimensions[1] <= 0 ||
      dimensions[0] > std::numeric_limits<int>::max() ||
      dimensions[1] > std::numeric_limits<int>::max()) {
    return Status::Unimplemented(
        "cuBLAS matrix dimensions exceed the supported domain");
  }
  layout.rows = static_cast<std::uint64_t>(dimensions[0]);
  layout.columns = static_cast<std::uint64_t>(dimensions[1]);
  if (strides[1] == width && strides[0] % width == 0 &&
      strides[0] / width >= dimensions[1]) {
    layout.order = CUBLASLT_ORDER_ROW;
    layout.leading = strides[0] / width;
  } else if (strides[0] == width && strides[1] % width == 0 &&
             strides[1] / width >= dimensions[0]) {
    layout.order = CUBLASLT_ORDER_COL;
    layout.leading = strides[1] / width;
  } else {
    return Status::Unimplemented("cuBLAS requires row/column-major storage");
  }
  // Keep byte leading dimensions inside the SDK's qualified integer domain.
  if (layout.leading > std::numeric_limits<int>::max() / width)
    return Status::Unimplemented(
        "cuBLAS matrix leading dimension is too large");
  return Status::Ok();
}

struct BatchGeometry final {
  std::array<MatrixLayout, 3> matrices{};
  std::array<std::size_t, kMaxTensorRank - 2> dimensions{};
  std::array<std::array<std::size_t, kMaxTensorRank - 2>, 3> strides{};
  std::array<DataType, 3> data_types{};
  std::array<std::size_t, 3> element_bytes{};
  std::size_t rank = 0;
  std::size_t count = 1;
};

Status select_layouts(Operation operation, std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs,
                      BatchGeometry &geometry) noexcept {
  const auto *matmul = operation.get_if<MatMul>();
  const bool scores = operation.custom_as<model_ops::DenseLinearF32>().has_value();
  if ((!matmul && !scores) || inputs.size() != 2 || outputs.size() != 1)
    return Status::Unimplemented(
        "cuBLAS provider implements MatMul and DenseLinearF32 only");
  if (matmul && matmul->accumulation_type != DataType::kFloat32)
    return Status::Unimplemented("cuBLAS requires FP32 accumulation");
  if (scores && inputs[0].data_type() != inputs[1].data_type())
    return Status::Unimplemented(
        "cuBLAS mixed activation/weight precision is served by the direct native matvec");
  auto &layouts = geometry.matrices;
  HYLE_RETURN_IF_ERROR(matrix_layout(inputs[0], layouts[0]));
  HYLE_RETURN_IF_ERROR(matrix_layout(inputs[1], layouts[1]));
  HYLE_RETURN_IF_ERROR(matrix_layout(outputs[0], layouts[2]));
  if (matmul && (inputs[0].data_type() != inputs[1].data_type() ||
                 inputs[0].data_type() != outputs[0].data_type()))
    return Status::Unimplemented("cuBLAS requires matching matrix data types");
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, inferred));
  if (scores) {
    if (outputs[0] != inferred[0])
      return Status::InvalidArgument(
          "cuBLAS DenseLinearF32 output differs from inference");
    // Dense stored [E,K] is the same allocation as column-major [K,E].
    // Only the descriptor changes; the original floating weight address is used.
    std::swap(layouts[1].rows, layouts[1].columns);
    layouts[1].order = CUBLASLT_ORDER_COL;
  }
  if (layouts[0].columns != layouts[1].rows ||
      layouts[2].rows != layouts[0].rows ||
      layouts[2].columns != layouts[1].columns)
    return Status::InvalidArgument("cuBLAS MatMul shapes do not agree");
  if (!std::ranges::equal(inferred[0].dimensions(), outputs[0].dimensions()))
    return Status::InvalidArgument("cuBLAS MatMul batch shapes do not agree");
  geometry.rank = outputs[0].rank() - 2;
  const std::array descriptions{inputs[0], inputs[1], outputs[0]};
  for (std::size_t port = 0; port < descriptions.size(); ++port) {
    geometry.data_types[port] = descriptions[port].data_type();
    geometry.element_bytes[port] = data_type_size(geometry.data_types[port]);
  }
  for (std::size_t axis = 0; axis < geometry.rank; ++axis) {
    const auto extent = outputs[0].dimensions()[axis];
    if (extent <= 0)
      return Status::Unimplemented("cuBLAS requires positive batch extents");
    const auto size = static_cast<std::size_t>(extent);
    if (geometry.count > SIZE_MAX / size)
      return Status::OutOfRange("cuBLAS batch count overflows");
    geometry.dimensions[axis] = size;
    geometry.count *= size;
    for (std::size_t port = 0; port < descriptions.size(); ++port) {
      const auto &description = descriptions[port];
      const auto batch_rank = description.rank() - 2;
      if (axis + batch_rank < geometry.rank)
        continue;
      const auto source_axis = axis + batch_rank - geometry.rank;
      if (description.dimensions()[source_axis] == 1)
        continue;
      const auto signed_stride = description.byte_strides()[source_axis];
      if (signed_stride < 0)
        return Status::Unimplemented("cuBLAS requires nonnegative batch strides");
      const auto stride = static_cast<std::size_t>(signed_stride);
      if (stride % geometry.element_bytes[port] != 0)
        return Status::Unimplemented(
            "cuBLAS batch strides require whole storage elements");
      geometry.strides[port][axis] = stride;
    }
  }
  for (std::size_t port = 0; port < descriptions.size(); ++port) {
    const auto &layout = layouts[port];
    const auto major = layout.order == CUBLASLT_ORDER_ROW ? layout.rows
                                                         : layout.columns;
    const auto minor = layout.order == CUBLASLT_ORDER_ROW ? layout.columns
                                                         : layout.rows;
    const auto leading = static_cast<std::size_t>(layout.leading);
    const auto width = geometry.element_bytes[port];
    if (major - 1 > (SIZE_MAX / width - minor) / leading)
      return Status::OutOfRange("cuBLAS matrix storage span overflows");
    const auto matrix_bytes =
        ((major - 1) * leading + minor) * width;
    std::size_t last = matrix_bytes;
    for (std::size_t axis = 0; axis < geometry.rank; ++axis) {
      const auto repeats = geometry.dimensions[axis] - 1;
      const auto stride = geometry.strides[port][axis];
      if (repeats != 0 && stride > (SIZE_MAX - last) / repeats)
        return Status::OutOfRange("cuBLAS batch storage span overflows");
      last += repeats * stride;
    }
    const auto available = descriptions[port].required_bytes();
    if (!available || last > *available)
      return Status::InvalidArgument("cuBLAS batch storage exceeds its tensor");
    if (port != 2)
      continue;
    // Output matrices must not overlap. Sorting physical batch strides admits
    // an outer permutation without assuming logical order is physical order.
    std::array<std::size_t, kMaxTensorRank - 2> axes{};
    for (std::size_t axis = 0; axis < geometry.rank; ++axis)
      axes[axis] = axis;
    std::sort(axes.begin(), axes.begin() + geometry.rank,
              [&](std::size_t a, std::size_t b) {
                return geometry.strides[port][a] < geometry.strides[port][b];
              });
    auto occupied = matrix_bytes;
    for (std::size_t i = 0; i < geometry.rank; ++i) {
      const auto axis = axes[i];
      if (geometry.dimensions[axis] == 1)
        continue;
      const auto stride = geometry.strides[port][axis];
      if (stride < occupied)
        return Status::Unimplemented("cuBLAS output batches must not overlap");
      occupied += (geometry.dimensions[axis] - 1) * stride;
    }
  }
  return Status::Ok();
}

Status create_runs(CublasPrepared &prepared,
                    const BatchGeometry &geometry) noexcept {
  // Merge the maximal common affine suffix. Broadcast is exactly stride0;
  // an outer-axis reset is a cold run boundary, never a hot pointer upload.
  std::size_t first = geometry.rank, inner_count = 1;
  std::array<std::size_t, 3> steps{};
  bool active = false;
  for (auto axis = geometry.rank; axis-- > 0;) {
    const auto extent = geometry.dimensions[axis];
    if (extent == 1) {
      first = axis;
      continue;
    }
    if (active) {
      bool continuous = true;
      for (std::size_t port = 0; port < steps.size(); ++port)
        if (steps[port] > SIZE_MAX / inner_count ||
            geometry.strides[port][axis] != steps[port] * inner_count)
          continuous = false;
      if (!continuous)
        break;
    } else {
      for (std::size_t port = 0; port < steps.size(); ++port)
        steps[port] = geometry.strides[port][axis];
      active = true;
    }
    inner_count *= extent;
    first = axis;
  }
  const auto outer_count = geometry.count / inner_count;
  for (std::size_t outer = 0; outer < outer_count; ++outer) {
    std::array<std::size_t, 3> offsets{};
    auto coordinate = outer;
    for (auto axis = first; axis-- > 0;) {
      const auto digit = coordinate % geometry.dimensions[axis];
      coordinate /= geometry.dimensions[axis];
      for (std::size_t port = 0; port < offsets.size(); ++port)
        offsets[port] += digit * geometry.strides[port][axis];
    }
    for (std::size_t begin = 0; begin < inner_count;) {
      const auto count = std::min(inner_count - begin,
                                  static_cast<std::size_t>(INT32_MAX));
      if (!prepared.runs.try_emplace_back())
        return Status::ResourceExhausted("cuBLAS batch run allocation failed");
      auto &run = prepared.runs.back();
      run.batch_count = static_cast<std::int32_t>(count);
      run.byte_strides = steps;
      for (std::size_t port = 0; port < offsets.size(); ++port)
        run.offsets[port] = offsets[port] + begin * steps[port];
      begin += count;
    }
  }
  return Status::Ok();
}

constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = true,
};

Status propose(void *, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (proposal == nullptr)
    return Status::InvalidArgument("cuBLAS proposal output is null");
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("cuBLAS requires an exact prepared shape");
  if (inputs.size() != request.inputs.size() ||
      outputs.size() != request.outputs.size())
    return Status::InvalidArgument("cuBLAS proposal binding counts differ");
  BatchGeometry geometry;
  HYLE_RETURN_IF_ERROR(select_layouts(request.operation, request.inputs,
                                      request.outputs, geometry));
  const detail::ResourceBindingRequirement tensor{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = kTensorAlignment,
  };
  for (auto &requirement : inputs)
    requirement = tensor;
  for (auto &requirement : outputs)
    requirement = tensor;
  *proposal = detail::ProviderOperationProposal{
      .route = 1,
      .runtime = kRuntime,
      .workspace_resource =
          {
              .required_properties = mem::MemoryProperty::kDeviceAccessible,
              .minimum_alignment = kWorkspaceAlignment,
          },
      .cost = {.workspace = std::nullopt,
               .persistent_bytes = std::nullopt,
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count = std::nullopt},
      .implementation = ProviderImplementationId{1},
  };
  return Status::Ok();
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<CublasPrepared *>(state);
  if (prepared == nullptr)
    return;
  for (const auto &run : prepared->runs)
    for (auto layout : run.layouts)
      if (layout != nullptr)
        (void)cublasLtMatrixLayoutDestroy(layout);
  if (prepared->operation != nullptr)
    (void)cublasLtMatmulDescDestroy(prepared->operation);
  prepared->~CublasPrepared();
  allocator.deallocate(prepared, sizeof(CublasPrepared),
                       alignof(CublasPrepared));
}

Status configure_run(CublasPrepared &prepared, CublasRun &run,
                      const BatchGeometry &geometry,
                      ColdHandle &cold) noexcept {
  const auto &layouts = geometry.matrices;
  for (std::size_t index = 0; index < layouts.size(); ++index) {
    const auto &layout = layouts[index];
    const auto type = geometry.data_types[index];
    const cudaDataType_t data_type =
        type == DataType::kFloat16
            ? CUDA_R_16F
            : (type == DataType::kBFloat16 ? CUDA_R_16BF : CUDA_R_32F);
    HYLE_RETURN_IF_ERROR(translate(
        cublasLtMatrixLayoutCreate(&run.layouts[index], data_type,
                                   layout.rows, layout.columns, layout.leading),
        "cublasLtMatrixLayoutCreate"));
    HYLE_RETURN_IF_ERROR(translate(
        cublasLtMatrixLayoutSetAttribute(run.layouts[index],
                                         CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &layout.order, sizeof(layout.order)),
        "cublasLtMatrixLayoutSetAttribute(order)"));
    if (run.batch_count > 1) {
      const auto stride = static_cast<std::int64_t>(
          run.byte_strides[index] / geometry.element_bytes[index]);
      HYLE_RETURN_IF_ERROR(translate(
          cublasLtMatrixLayoutSetAttribute(run.layouts[index],
              CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &run.batch_count,
              sizeof(run.batch_count)),
          "cublasLtMatrixLayoutSetAttribute(batch_count)"));
      HYLE_RETURN_IF_ERROR(translate(
          cublasLtMatrixLayoutSetAttribute(run.layouts[index],
              CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
              sizeof(stride)),
          "cublasLtMatrixLayoutSetAttribute(batch_stride)"));
    }
  }
  const std::array attributes{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
  for (std::size_t index = 0; index < attributes.size(); ++index) {
    const auto port = std::min(index, std::size_t{2});
    auto alignment = std::gcd(kTensorAlignment, run.offsets[port]);
    if (run.batch_count > 1)
      alignment = std::gcd(alignment, run.byte_strides[port]);
    const auto qualified_alignment = static_cast<std::uint32_t>(alignment);
    HYLE_RETURN_IF_ERROR(translate(
        cublasLtMatmulPreferenceSetAttribute(cold.preference, attributes[index],
            &qualified_alignment, sizeof(qualified_alignment)),
        "cublasLtMatmulPreferenceSetAttribute(alignment)"));
  }
  cublasLtMatmulHeuristicResult_t result{};
  int count = 0;
  HYLE_RETURN_IF_ERROR(translate(
      cublasLtMatmulAlgoGetHeuristic(cold.handle, prepared.operation,
                                     run.layouts[0], run.layouts[1],
                                     run.layouts[2], run.layouts[2],
                                     cold.preference, 1, &result, &count),
      "cublasLtMatmulAlgoGetHeuristic"));
  if (count != 1)
    return Status::Unimplemented("cuBLAS found no algorithm for these layouts");
  HYLE_RETURN_IF_ERROR(translate(result.state, "cuBLAS selected algorithm"));
  run.algorithm = result.algo;
  prepared.workspace_bytes =
      std::max(prepared.workspace_bytes, result.workspaceSize);
  return Status::Ok();
}

Status configure(CublasPrepared &prepared,
                 const BatchGeometry &geometry) noexcept {
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  ColdHandle cold;
  HYLE_RETURN_IF_ERROR(
      translate(cublasLtCreate(&cold.handle), "cublasLtCreate"));
  // F32 scores retain BF16 input arithmetic; only real F32 inputs use pedantic.
  const auto type = geometry.data_types[0];
  const cublasComputeType_t compute_type = type == DataType::kFloat32
                                               ? CUBLAS_COMPUTE_32F_PEDANTIC
                                               : CUBLAS_COMPUTE_32F;
  HYLE_RETURN_IF_ERROR(translate(
      cublasLtMatmulDescCreate(&prepared.operation, compute_type, CUDA_R_32F),
      "cublasLtMatmulDescCreate"));
  HYLE_RETURN_IF_ERROR(create_runs(prepared, geometry));
  HYLE_RETURN_IF_ERROR(
      translate(cublasLtMatmulPreferenceCreate(&cold.preference),
                "cublasLtMatmulPreferenceCreate"));
  HYLE_RETURN_IF_ERROR(translate(
      cublasLtMatmulPreferenceSetAttribute(
          cold.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &kMaximumWorkspaceBytes, sizeof(kMaximumWorkspaceBytes)),
      "cublasLtMatmulPreferenceSetAttribute(workspace)"));
  if (type != DataType::kFloat32) {
    // Split-K retains F32 partial sums, as in the original rank-2 path.
    const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    HYLE_RETURN_IF_ERROR(translate(
        cublasLtMatmulPreferenceSetAttribute(
            cold.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            &reduction, sizeof(reduction)),
        "cublasLtMatmulPreferenceSetAttribute(reduction)"));
  }
  for (auto &run : prepared.runs)
    HYLE_RETURN_IF_ERROR(configure_run(prepared, run, geometry, cold));
  return Status::Ok();
}

Status create_invocation(void *state, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("cuBLAS invocation output is null");
  *output = nullptr;
  const auto &prepared = *static_cast<CublasPrepared *>(state);
  void *scratch = nullptr;
  if (prepared.workspace_bytes != 0) {
    if (workspace.size() < prepared.workspace_bytes)
      return Status::InvalidArgument("cuBLAS workspace is too small");
    auto pointer =
        cuda::detail::mutable_pointer(workspace, kWorkspaceAlignment, 1);
    if (!pointer)
      return std::move(pointer).error();
    scratch = *pointer;
  }
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  void *storage =
      allocator.allocate(sizeof(CublasInvocation), alignof(CublasInvocation));
  if (storage == nullptr)
    return Status::ResourceExhausted("cuBLAS invocation allocation failed");
  auto *invocation = ::new (storage)
      CublasInvocation{.primary = prepared.primary, .workspace = scratch};
  *output = invocation; // Compute destroys partial state on failure.
  return translate(cublasLtCreate(&invocation->handle),
                   "cublasLtCreate(invocation)");
}

void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<CublasInvocation *>(state);
  if (invocation == nullptr)
    return;
  cuda::detail::ScopedPrimaryContext scope(invocation->primary);
  if (invocation->handle != nullptr)
    (void)cublasLtDestroy(invocation->handle);
  invocation->~CublasInvocation();
  allocator.deallocate(invocation, sizeof(CublasInvocation),
                       alignof(CublasInvocation));
}

detail::EnqueueResult submit(void *state, void *lane, detail::QueueSequence &sequence,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  auto &prepared = *static_cast<CublasPrepared *>(state);
  auto &invocation = *static_cast<CublasInvocation *>(lane);
  auto stream = cuda::stream(sequence);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  auto left =
      cuda::detail::const_pointer(inputs[0].storage(), kTensorAlignment, 1);
  if (!left)
    return detail::EnqueueResult::rejected(std::move(left).error());
  auto right =
      cuda::detail::const_pointer(inputs[1].storage(), kTensorAlignment, 1);
  if (!right)
    return detail::EnqueueResult::rejected(std::move(right).error());
  auto output =
      cuda::detail::mutable_pointer(outputs[0].storage(), kTensorAlignment, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  const float alpha = 1.0f;
  const float beta = 0.0f;
  for (const auto &run : prepared.runs) {
    const auto *a = static_cast<const char *>(*left) + run.offsets[0];
    const auto *b = static_cast<const char *>(*right) + run.offsets[1];
    auto *c = static_cast<char *>(*output) + run.offsets[2];
    const auto result = cublasLtMatmul(
        invocation.handle, prepared.operation, &alpha, a, run.layouts[0],
        b, run.layouts[1], &beta, c, run.layouts[2], c, run.layouts[2],
        &run.algorithm, invocation.workspace, prepared.workspace_bytes, *stream);
    // Even a first vendor error may have enqueued work: preserve the real
    // accepted prefix and the original Invocation handle/workspace lifetime.
    if (result != CUBLAS_STATUS_SUCCESS)
      return detail::EnqueueResult::accepted_command(
          translate(result, "cublasLtMatmul"));
  }
  return detail::EnqueueResult::accepted_command(Status::Ok());
}

AliasSupport alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}

const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
};

ProviderPrepareOutcome prepare(void *context_value,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  if (output == nullptr)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("cuBLAS prepared output is null"));
  *output = {};
  if (request.route != 1 || !request.shape.exact())
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("cuBLAS exact prepare request is invalid"));
  BatchGeometry geometry;
  Status status = select_layouts(request.operation, request.inputs,
                                 request.outputs, geometry);
  if (!status.ok())
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  void *storage =
      allocator.allocate(sizeof(CublasPrepared), alignof(CublasPrepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::error(
        Status::ResourceExhausted("cuBLAS prepared allocation failed"));
  auto *prepared = ::new (storage) CublasPrepared(
      static_cast<CublasContext *>(context_value)->primary, allocator);
  status = configure(*prepared, geometry);
  if (!status.ok()) {
    destroy_prepared(prepared, allocator);
    if (status.IsUnimplemented())
      return ProviderPrepareOutcome::unsupported(std::move(status));
    if (status.IsResourceExhausted())
      return ProviderPrepareOutcome::resource_declined(std::move(status));
    return ProviderPrepareOutcome::error(std::move(status));
  }
  *output = detail::ProviderPrepared{
      .vtable = &kPreparedVTable,
      .state = prepared,
      .workspace = {prepared->workspace_bytes, kWorkspaceAlignment},
      .runtime = kRuntime,
      .persistent_bytes = std::nullopt,
      .invocation_private_bytes = std::nullopt,
  };
  return ProviderPrepareOutcome::prepared();
}

Status drain_context(void *) noexcept { return Status::Ok(); }
Status inspect_capture_bindings(const detail::ProviderPrepared &value,
                                bool *preserves) noexcept {
  if (!preserves)
    return Status::InvalidArgument("cuBLAS capture bindings output is null");
  *preserves = false;
  if (value.vtable != &kPreparedVTable || !value.state)
    return Status::Unimplemented("foreign cuBLAS prepared implementation");
  const auto &prepared = *static_cast<const CublasPrepared *>(value.state);
  if (!prepared.primary || !prepared.operation || prepared.runs.empty())
    return Status::Internal("cuBLAS capture bindings prepared state is invalid");
  for (const auto &run : prepared.runs)
    for (const auto layout : run.layouts)
      if (!layout)
        return Status::Internal("cuBLAS capture bindings layout is invalid");
  // Algorithms/layouts are immutable. The original invocation installs its
  // handle and caller workspace once; submit changes only explicit ports.
  *preserves = value.runtime.capture_compatible &&
      value.runtime.mode == ProviderRuntimeMode::kSharedQueue &&
      value.runtime.workspace_owner == ProviderWorkspaceOwner::kCaller;
  return Status::Ok();
}
const void *query_extension(void *, ProviderExtensionId id) noexcept {
  static constexpr detail::CaptureBindingsVTable capture{
      .inspect = &inspect_capture_bindings};
  return id == detail::kCaptureBindingsExtensionId ? &capture : nullptr;
}

#define HYLE_CUBLAS_STRINGIFY_IMPL(value) #value
#define HYLE_CUBLAS_STRINGIFY(value) HYLE_CUBLAS_STRINGIFY_IMPL(value)
const detail::ProviderVTable kProviderVTable{
    .identity =
        {.id = kProviderId,
         .name = "cublasLt",
         .version =
             HYLE_CUBLAS_STRINGIFY(CUBLAS_VER_MAJOR) "." HYLE_CUBLAS_STRINGIFY(
                 CUBLAS_VER_MINOR) "." HYLE_CUBLAS_STRINGIFY(CUBLAS_VER_PATCH)},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain_context,
    .query_extension = &query_extension,
};
#undef HYLE_CUBLAS_STRINGIFY
#undef HYLE_CUBLAS_STRINGIFY_IMPL

} // namespace

Provider provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, std::uint8_t{0});
}

} // namespace hyle::compute::cublas
