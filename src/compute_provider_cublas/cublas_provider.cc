#include "hyle/compute/providers/cublas.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <utility>

#include <cublasLt.h>

#include "native_context.h"
#include "provider_internal.h"

namespace hyle::compute::cublas {
namespace {

constexpr std::size_t kTensorAlignment = 16;
constexpr std::size_t kWorkspaceAlignment = 256;
constexpr std::size_t kMaximumWorkspaceBytes = 32 * 1024 * 1024;

// No mutable vendor handle is shared by concurrent prepare or execution.
struct CublasContext final {
  CUcontext primary = nullptr;
};

struct CublasPrepared final {
  CUcontext primary = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  std::array<cublasLtMatrixLayout_t, 3> layouts{};
  cublasLtMatmulAlgo_t algorithm{};
  std::size_t workspace_bytes = 0;
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
      description.rank() != 2 ||
      (description.data_type() != DataType::kFloat16 &&
       description.data_type() != DataType::kFloat32)) {
    return Status::Unimplemented("cuBLAS requires plain rank-2 FP16/FP32");
  }
  const auto dimensions = description.dimensions();
  const auto strides = description.byte_strides();
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

Status select_layouts(Operation operation, std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs,
                      std::array<MatrixLayout, 3> &layouts) noexcept {
  if (operation.kind() != OperationKind::kMatMul || inputs.size() != 2 ||
      outputs.size() != 1)
    return Status::Unimplemented("cuBLAS provider implements MatMul only");
  if (operation.get_if<MatMul>()->accumulation_type != DataType::kFloat32)
    return Status::Unimplemented("cuBLAS requires FP32 accumulation");
  HYLE_RETURN_IF_ERROR(matrix_layout(inputs[0], layouts[0]));
  HYLE_RETURN_IF_ERROR(matrix_layout(inputs[1], layouts[1]));
  HYLE_RETURN_IF_ERROR(matrix_layout(outputs[0], layouts[2]));
  if (inputs[0].data_type() != inputs[1].data_type() ||
      inputs[0].data_type() != outputs[0].data_type())
    return Status::Unimplemented("cuBLAS requires matching matrix data types");
  if (layouts[0].columns != layouts[1].rows ||
      layouts[2].rows != layouts[0].rows ||
      layouts[2].columns != layouts[1].columns)
    return Status::InvalidArgument("cuBLAS MatMul shapes do not agree");
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
  std::array<MatrixLayout, 3> layouts{};
  HYLE_RETURN_IF_ERROR(select_layouts(request.operation, request.inputs,
                                      request.outputs, layouts));
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
      .profile_key = ProviderRouteKey{1},
  };
  return Status::Ok();
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<CublasPrepared *>(state);
  if (prepared == nullptr)
    return;
  for (auto layout : prepared->layouts) {
    if (layout != nullptr)
      (void)cublasLtMatrixLayoutDestroy(layout);
  }
  if (prepared->operation != nullptr)
    (void)cublasLtMatmulDescDestroy(prepared->operation);
  prepared->~CublasPrepared();
  allocator.deallocate(prepared, sizeof(CublasPrepared),
                       alignof(CublasPrepared));
}

Status configure(CublasPrepared &prepared, DataType type,
                 const std::array<MatrixLayout, 3> &layouts) noexcept {
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  ColdHandle cold;
  HYLE_RETURN_IF_ERROR(
      translate(cublasLtCreate(&cold.handle), "cublasLtCreate"));
  const cudaDataType_t data_type =
      type == DataType::kFloat16 ? CUDA_R_16F : CUDA_R_32F;
  const cublasComputeType_t compute_type = type == DataType::kFloat16
                                               ? CUBLAS_COMPUTE_32F
                                               : CUBLAS_COMPUTE_32F_PEDANTIC;
  HYLE_RETURN_IF_ERROR(translate(
      cublasLtMatmulDescCreate(&prepared.operation, compute_type, CUDA_R_32F),
      "cublasLtMatmulDescCreate"));
  for (std::size_t index = 0; index < layouts.size(); ++index) {
    const auto &layout = layouts[index];
    HYLE_RETURN_IF_ERROR(translate(
        cublasLtMatrixLayoutCreate(&prepared.layouts[index], data_type,
                                   layout.rows, layout.columns, layout.leading),
        "cublasLtMatrixLayoutCreate"));
    HYLE_RETURN_IF_ERROR(translate(
        cublasLtMatrixLayoutSetAttribute(prepared.layouts[index],
                                         CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &layout.order, sizeof(layout.order)),
        "cublasLtMatrixLayoutSetAttribute(order)"));
  }
  HYLE_RETURN_IF_ERROR(
      translate(cublasLtMatmulPreferenceCreate(&cold.preference),
                "cublasLtMatmulPreferenceCreate"));
  HYLE_RETURN_IF_ERROR(
      translate(cublasLtMatmulPreferenceSetAttribute(
                    cold.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &kMaximumWorkspaceBytes, sizeof(kMaximumWorkspaceBytes)),
                "cublasLtMatmulPreferenceSetAttribute(workspace)"));
  if (type == DataType::kFloat16) {
    // Split-K must not round partial sums to the FP16 output type. The NONE
    // scheme remains available; a separate reduction must retain FP32 scratch.
    const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    HYLE_RETURN_IF_ERROR(translate(
        cublasLtMatmulPreferenceSetAttribute(
            cold.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            &reduction, sizeof(reduction)),
        "cublasLtMatmulPreferenceSetAttribute(reduction)"));
  }
  const std::uint32_t alignment = kTensorAlignment;
  for (const auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    HYLE_RETURN_IF_ERROR(translate(
        cublasLtMatmulPreferenceSetAttribute(cold.preference, attribute,
                                             &alignment, sizeof(alignment)),
        "cublasLtMatmulPreferenceSetAttribute(alignment)"));
  }
  cublasLtMatmulHeuristicResult_t result{};
  int count = 0;
  HYLE_RETURN_IF_ERROR(translate(
      cublasLtMatmulAlgoGetHeuristic(cold.handle, prepared.operation,
                                     prepared.layouts[0], prepared.layouts[1],
                                     prepared.layouts[2], prepared.layouts[2],
                                     cold.preference, 1, &result, &count),
      "cublasLtMatmulAlgoGetHeuristic"));
  if (count != 1)
    return Status::Unimplemented("cuBLAS found no algorithm for these layouts");
  HYLE_RETURN_IF_ERROR(translate(result.state, "cuBLAS selected algorithm"));
  prepared.algorithm = result.algo;
  prepared.workspace_bytes = result.workspaceSize;
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

detail::EnqueueResult submit(void *state, void *lane, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  auto &prepared = *static_cast<CublasPrepared *>(state);
  auto &invocation = *static_cast<CublasInvocation *>(lane);
  auto stream = cuda::stream(queue);
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
  const cublasStatus_t result = cublasLtMatmul(
      invocation.handle, prepared.operation, &alpha, *left, prepared.layouts[0],
      *right, prepared.layouts[1], &beta, *output, prepared.layouts[2], *output,
      prepared.layouts[2], &prepared.algorithm, invocation.workspace,
      prepared.workspace_bytes, *stream);
  // A vendor error does not prove that no kernel was recorded. Conservatively
  // keep the Queue prefix and lane alive through the backend's real terminal.
  return detail::EnqueueResult::accepted_command(
      translate(result, "cublasLtMatmul"));
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
  std::array<MatrixLayout, 3> layouts{};
  Status status = select_layouts(request.operation, request.inputs,
                                 request.outputs, layouts);
  if (!status.ok())
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  void *storage =
      allocator.allocate(sizeof(CublasPrepared), alignof(CublasPrepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("cuBLAS prepared allocation failed"));
  auto *prepared = ::new (storage) CublasPrepared{
      .primary = static_cast<CublasContext *>(context_value)->primary};
  status = configure(*prepared, request.inputs[0].data_type(), layouts);
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
};
#undef HYLE_CUBLAS_STRINGIFY
#undef HYLE_CUBLAS_STRINGIFY_IMPL

} // namespace

Provider provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, std::uint8_t{0});
}

} // namespace hyle::compute::cublas
