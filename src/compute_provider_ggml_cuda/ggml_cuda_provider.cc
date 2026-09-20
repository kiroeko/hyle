#include "program.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <new>

#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <ggml.h>

#include "metadata_arena.h"
#include "native_bridge.h"
#include "native_context.h"
#include "tensor_program.h"

namespace hyle::compute::ggml_cuda {
namespace {

bool default_cublas_precision() noexcept {
#if defined(_WIN32)
  std::size_t bytes = 0;
  return ::getenv_s(&bytes, nullptr, 0, "GGML_CUDA_CUBLAS_COMPUTE_TYPE") == 0 &&
         bytes == 0;
#else
  return std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE") == nullptr;
#endif
}

struct Extension final {
  ProviderExtensionId id{};
  const void *table = nullptr;
};
struct Context final {
  CUcontext primary = nullptr;
  int device = -1;
  bool qualified = false;
  bool bfloat16_attention = false;
  Extension extension;
};
struct Program final {
  CUcontext primary = nullptr;
  int device = -1;
  internal::Recipe recipe;
  std::vector<std::size_t> alignments;
};

// Called only from catch blocks; native namespaces and numbers survive the
// private vendor exception boundary. No exception crosses a public callback.
Status failure() noexcept {
  try {
    throw;
  } catch (const bridge::NativeFailure<cudaError_t> &error) {
    return cuda::status(error.code);
  } catch (const bridge::NativeFailure<CUresult> &error) {
    return cuda::status(error.code);
  } catch (const bridge::NativeFailure<cublasStatus_t> &error) {
    const auto code = error.code == CUBLAS_STATUS_ALLOC_FAILED
                          ? StatusCode::ResourceExhausted
                          : StatusCode::Internal;
    return Status::FromNative(
        code,
        NativeError{NativeErrorDomain::Unknown,
                    static_cast<std::int64_t>(error.code)},
        error.operation);
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("ggml CUDA host allocation failed");
  } catch (const std::exception &error) {
    return Status::Internal(error.what());
  } catch (...) {
    return Status::Internal("ggml CUDA unknown native failure");
  }
}

struct Invocation final {
  CUcontext primary = nullptr;
  ggml_runtime::MetadataArena arena;
  ggml_backend *backend = nullptr;
  cudaStream_t root = nullptr;
  cudaEvent_t entry = nullptr;
  cudaEvent_t exit = nullptr;
  ggml_cgraph *graph = nullptr;
  std::vector<ggml_tensor *> tensors;
  std::vector<ggml_tensor *> roots;
  std::vector<ggml_backend_buffer *> buffers;
  std::vector<ggml_backend_buffer *> scratch;
  std::vector<ggml_tensor *> masks;
  std::vector<internal::ScalarConstant> constants;
  std::vector<ggml_tensor *> views;
  std::vector<void *> pointers;
  bool failed = false;

  Invocation() = default;
  Invocation(const Invocation &) = delete;
  Invocation &operator=(const Invocation &) = delete;
  ~Invocation() {
    cuda::detail::ScopedPrimaryContext scope(primary);
    if (backend)
      (void)bridge::drain(backend);
    if (entry)
      (void)cudaEventDestroy(entry);
    if (exit)
      (void)cudaEventDestroy(exit);
    for (auto *buffer : buffers)
      ggml_backend_buffer_free(buffer);
    for (auto *buffer : scratch)
      ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
  }

  ggml_tensor *attention(const internal::Node &node) {
    const auto &operation = *node.operation.get_if<ScaledDotProductAttention>();
    auto *context = arena.get();
    auto *query = tensors[node.inputs[0]];
    auto *key = tensors[node.inputs[1]];
    auto *value = tensors[node.inputs[2]];
    const auto type = query->type;
    // BSHD is [D,H,S,B] in ggml. Its attention/GQA inputs are [D,S,H,B].
    query = ggml_permute(context, ggml_cast(context, query, GGML_TYPE_F32), 0,
                         2, 1, 3);
    if (type == GGML_TYPE_F16) {
      key = ggml_permute(context, key, 0, 2, 1, 3);
      value = ggml_permute(context, value, 0, 2, 1, 3);
      auto *mask =
          ggml_new_tensor_2d(context, GGML_TYPE_F16, key->ne[1], query->ne[1]);
      masks.push_back(mask);
      auto *result = ggml_flash_attn_ext(context, query, key, value, mask,
                                         operation.scale, 0, 0);
      (void)ggml_prec_set_acc(result, GGML_PREC_F32);
      return ggml_cast(context, result, type);
    }
    // This upstream pin narrows BF16 K/V in its SM89 MMA flash path. Use
    // its original F32 graph to preserve the BF16 exponent range instead.
    key = ggml_permute(context, ggml_cast(context, key, GGML_TYPE_F32), 0, 2, 1,
                       3);
    value = ggml_permute(context, ggml_cast(context, value, GGML_TYPE_F32), 0,
                         2, 1, 3);
    auto *scores = ggml_mul_mat(context, key, query);
    (void)ggml_prec_set_acc(scores, GGML_PREC_F32);
    (void)ggml_prec_set_src(scores, GGML_PREC_F32, 1);
    // Scale before masking: a zero or negative scale must not turn -inf
    // into NaN or +inf. Bottom-right causal offset is KV length - Q length.
    scores = ggml_scale(context, scores, operation.scale);
    scores = ggml_diag_mask_inf(context, scores,
                                static_cast<int>(key->ne[1] - query->ne[1]));
    auto *probabilities = ggml_soft_max(context, scores);
    value = ggml_cont(context, ggml_transpose(context, value));
    auto *result = ggml_mul_mat(context, value, probabilities);
    (void)ggml_prec_set_acc(result, GGML_PREC_F32);
    (void)ggml_prec_set_src(result, GGML_PREC_F32, 1);
    return ggml_cast(context, ggml_permute(context, result, 0, 2, 1, 3), type);
  }

  Status allocate(ggml_tensor *tensor, ggml_backend_buffer_type_t type) {
    if (tensor->view_src) {
      views.push_back(tensor);
      return Status::Ok();
    }
    if (tensor->buffer)
      return Status::Ok(); // Already bound external tensor.
    auto *buffer = ggml_backend_buft_alloc_buffer(
        type, ggml_backend_buft_get_alloc_size(type, tensor));
    if (!buffer)
      return Status::ResourceExhausted("ggml CUDA scratch allocation failed");
    scratch.push_back(buffer); // Capacity reserved before acquiring resources.
    if (ggml_backend_tensor_alloc(buffer, tensor,
                                  ggml_backend_buffer_get_base(buffer)) !=
        GGML_STATUS_SUCCESS)
      return Status::Internal("ggml CUDA scratch binding failed");
    return Status::Ok();
  }

  void bind_views() noexcept {
    for (auto *tensor : views) {
      const auto *source = tensor->view_src;
      tensor->buffer = source->buffer;
      tensor->data = source->data ? static_cast<std::byte *>(source->data) +
                                        tensor->view_offs
                                  : nullptr;
    }
  }

  Status initialize(const Program &program, mem::AllocatorRef allocator) {
    primary = program.primary;
    cuda::detail::ScopedPrimaryContext scope(primary);
    HYLE_RETURN_IF_ERROR(scope.status());
    const auto &recipe = program.recipe;
    const auto count = recipe.values.size();
    constexpr auto maximum = static_cast<std::size_t>(INT32_MAX) - 16;
    if (count > maximum / 4 || recipe.nodes.size() > (maximum - count * 4) / 64)
      return Status::OutOfRange("ggml CUDA metadata count overflows");
    auto capacity = count * 4 + recipe.nodes.size() * 64;
    for (const auto &node : recipe.nodes) {
      if (!node.operation.get_if<Concat>())
        continue;
      // A region can concatenate the same value many times. Count ports,
      // not just unique recipe values, before reserving the fixed arena.
      if (node.inputs.size() > (maximum - capacity) / 4)
        return Status::OutOfRange("ggml CUDA concat metadata count overflows");
      capacity += node.inputs.size() * 4;
    }
    capacity += 16;
    auto metadata =
        ggml_runtime::MetadataArena::for_graph(capacity, capacity, allocator);
    if (!metadata)
      return std::move(metadata).error();
    arena = std::move(*metadata);
    tensors.resize(count, nullptr);
    roots.resize(count, nullptr);
    buffers.resize(count, nullptr);
    pointers.resize(recipe.inputs.size() + recipe.outputs.size());
    backend = ggml_backend_cuda_init(program.device);
    if (!backend)
      return Status::ResourceExhausted("ggml CUDA backend allocation failed");
    root = bridge::root_stream(backend);
    HYLE_RETURN_IF_ERROR(
        cuda::status(cudaEventCreateWithFlags(&entry, cudaEventDisableTiming)));
    HYLE_RETURN_IF_ERROR(
        cuda::status(cudaEventCreateWithFlags(&exit, cudaEventDisableTiming)));
    for (const auto id : recipe.inputs) {
      const auto &description = recipe.values[id];
      tensors[id] = internal::create_tensor_input(arena.get(), description);
      ggml_set_input(tensors[id]);
    }
    for (const auto &node : recipe.nodes) {
      ggml_tensor *result = nullptr;
      if (node.operation.get_if<ScaledDotProductAttention>())
        result = attention(node);
      else
        result = internal::lower_tensor_operation(arena.get(), recipe, node,
                                                  tensors, constants);
      if (!result)
        return Status::Unimplemented("ggml CUDA tensor lowering declined");
      if (!ggml_backend_supports_op(backend, result))
        return Status::Unimplemented("ggml CUDA native operation declined");
      tensors[node.output] = result;
    }
    for (std::size_t id = 0; id < count; ++id) {
      roots[id] = tensors[id];
      while (roots[id] && roots[id]->view_src)
        roots[id] = roots[id]->view_src;
    }
    graph = ggml_new_graph_custom(arena.get(), capacity, false);
    for (const auto id : recipe.outputs) {
      ggml_set_output(tensors[id]);
      // Reshape is a view: bind and protect the actual producer as well.
      // Native fusion must not consume a separately published intermediate.
      ggml_set_output(roots[id]);
      ggml_build_forward_expand(graph, tensors[id]);
    }
    for (std::size_t id = 0; id < count; ++id) {
      if (!tensors[id])
        return Status::InvalidArgument("ggml CUDA recipe has an unused value");
      const auto bytes = ggml_nbytes(roots[id]);
      const bool external =
          std::ranges::find(recipe.inputs, id) != recipe.inputs.end() ||
          std::ranges::find(recipe.outputs, id) != recipe.outputs.end();
      if (external) {
        buffers[id] = bridge::binding_buffer(program.device, bytes);
        // An idle external binding has no payload. Submission installs both
        // the buffer base and tensor address after provenance/size admission.
        roots[id]->buffer = buffers[id];
      }
    }
    const auto nodes = ggml_graph_n_nodes(graph);
    std::size_t resources = 0;
    for (auto *tensor = ggml_get_first_tensor(arena.get()); tensor;
         tensor = ggml_get_next_tensor(arena.get(), tensor))
      ++resources;
    scratch.reserve(resources);
    views.reserve(resources);
    auto *type = ggml_backend_cuda_buffer_type(program.device);
    for (auto *tensor = ggml_get_first_tensor(arena.get()); tensor;
         tensor = ggml_get_next_tensor(arena.get(), tensor))
      HYLE_RETURN_IF_ERROR(allocate(tensor, type));
    for (int i = 0; i < nodes; ++i) {
      auto *tensor = ggml_graph_node(graph, i);
      if (!ggml_backend_supports_op(backend, tensor))
        return Status::Unimplemented("ggml CUDA native graph node declined");
    }
    bind_views();
    for (const auto &constant : constants)
      ggml_backend_tensor_set(constant.tensor, &constant.value, 0,
                              sizeof(constant.value));
    for (auto *mask : masks) {
      const auto columns = static_cast<std::size_t>(mask->ne[0]);
      const auto rows = static_cast<std::size_t>(mask->ne[1]);
      std::vector<std::uint16_t> data(columns * rows, 0);
      for (std::size_t row = 0; row < rows; ++row)
        std::fill(data.begin() + static_cast<std::ptrdiff_t>(
                                     row * columns + columns - rows + row + 1),
                  data.begin() +
                      static_cast<std::ptrdiff_t>((row + 1) * columns),
                  std::uint16_t{0xfc00}); // Exact IEEE FP16 negative infinity.
      ggml_backend_tensor_set(mask, data.data(), 0, ggml_nbytes(mask));
    }
    return Status::Ok();
  }
};

Status create_context(const Provider &provider, Device device, void *backend,
                      mem::AllocatorRef allocator, void **output) noexcept {
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("ggml CUDA requires a CUDA device");
  const auto primary = cuda::detail::borrow_primary_context(backend);
  const auto *environment = cuda::detail::borrow_environment(backend);
  if (!primary || !environment)
    return Status::FailedPrecondition("ggml CUDA backend context is invalid");
  cuda::detail::ScopedPrimaryContext scope(primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  try {
#if defined(_WIN32)
    // This DLL embeds its own ggml copy. MetadataArena initializes a separate
    // copy; native graph-cache timestamps need this copy's thread-safe timer.
    ggml_time_init();
#endif
    const int native_device = bridge::device_index(environment->device_ordinal);
    if (native_device < 0)
      return Status::Unimplemented("ggml CUDA physical device is unavailable");
    void *storage = allocator.allocate(sizeof(Context), alignof(Context));
    if (!storage)
      return Status::ResourceExhausted("ggml CUDA context allocation failed");
    *output = ::new (storage)
        Context{primary, native_device,
                environment->sm_major == 8 && environment->sm_minor == 9,
                default_cublas_precision(),
                detail::ProviderAccess::payload<Extension>(provider)};
    return Status::Ok();
  } catch (...) {
    return failure();
  }
}
void destroy_context(void *value, mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<Context *>(value);
  if (context) {
    context->~Context();
    allocator.deallocate(context, sizeof(Context), alignof(Context));
  }
}
void destroy_program(void *value, mem::AllocatorRef allocator) noexcept {
  auto *program = static_cast<Program *>(value);
  if (program) {
    program->~Program();
    allocator.deallocate(program, sizeof(Program), alignof(Program));
  }
}
void destroy_invocation(void *value, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(value);
  if (invocation) {
    invocation->~Invocation();
    allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
  }
}
Status create_invocation(void *value, mem::AllocatorRef allocator,
                         mem::BufferView, void **output) noexcept {
  *output = nullptr;
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage)
    return Status::ResourceExhausted("ggml CUDA invocation allocation failed");
  auto *invocation = ::new (storage) Invocation;
  *output = invocation; // The caller owns cleanup even after partial failure.
  try {
    return invocation->initialize(*static_cast<Program *>(value), allocator);
  } catch (...) {
    return failure();
  }
}
detail::EnqueueResult submit(void *value, void *lane, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &program = *static_cast<Program *>(value);
  auto &invocation = *static_cast<Invocation *>(lane);
  const auto &recipe = program.recipe;
  if (invocation.failed)
    return detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "ggml CUDA invocation failed; instantiate another"));
  if (inputs.size() != recipe.inputs.size() ||
      outputs.size() != recipe.outputs.size())
    return detail::EnqueueResult::rejected(
        Status::InvalidArgument("ggml CUDA binding counts differ"));
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto pointer = cuda::detail::const_pointer(
        inputs[i].storage(), program.alignments[recipe.inputs[i]], 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    invocation.pointers[i] = const_cast<void *>(*pointer);
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto pointer = cuda::detail::mutable_pointer(
        outputs[i].storage(), program.alignments[recipe.outputs[i]], 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    invocation.pointers[inputs.size() + i] = *pointer;
  }
  cuda::detail::ScopedPrimaryContext scope(program.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  const auto bind = [&](std::uint32_t id, void *pointer) {
    bridge::bind_buffer(invocation.buffers[id], pointer);
    invocation.roots[id]->data = pointer;
  };
  for (std::size_t i = 0; i < inputs.size(); ++i)
    bind(recipe.inputs[i], invocation.pointers[i]);
  for (std::size_t i = 0; i < outputs.size(); ++i)
    bind(recipe.outputs[i], invocation.pointers[inputs.size() + i]);
  invocation.bind_views();

  Status terminal = cuda::status(cudaEventRecord(invocation.entry, *stream));
  if (terminal.ok())
    terminal =
        cuda::status(cudaStreamWaitEvent(invocation.root, invocation.entry, 0));
  if (terminal.ok()) {
    try {
      if (ggml_backend_graph_compute_async(
              invocation.backend, invocation.graph) != GGML_STATUS_SUCCESS)
        terminal = Status::Internal("ggml CUDA native graph failed");
    } catch (...) {
      terminal = failure();
    }
  }
  if (terminal.ok())
    terminal = cuda::status(cudaEventRecord(invocation.exit, invocation.root));
  if (terminal.ok())
    terminal = cuda::status(cudaStreamWaitEvent(*stream, invocation.exit, 0));
  if (!terminal.ok()) {
    invocation.failed = true;
    // A failed exit event cannot transfer native ownership to HYLE's stream.
    // Quiesce every accepted upstream stream before exposing the failure.
    (void)bridge::drain(invocation.backend);
  }
  return detail::EnqueueResult::accepted_command(std::move(terminal));
}
AliasSupport alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy_program,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};

Status propose(void *context, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *output) noexcept {
  *output = {};
  if (!request.shape.exact() ||
      !internal::supports(context, request.operation, request.inputs,
                          request.outputs))
    return Status::Unimplemented("ggml CUDA operation domain is not admitted");
  std::ranges::fill(inputs, internal::kBinding);
  std::ranges::fill(outputs, internal::kBinding);
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i].minimum_alignment = internal::boundary_alignment(
        request.operation, request.inputs[i].data_type());
  for (std::size_t i = 0; i < outputs.size(); ++i)
    outputs[i].minimum_alignment = internal::boundary_alignment(
        request.operation, request.outputs[i].data_type());
  *output = {.route = 1,
             .runtime = internal::kRuntime,
             .workspace_resource = internal::kBinding,
             .cost = {.workspace = WorkspaceRequirement{},
                      .persistent_bytes = std::nullopt,
                      .copy_bytes = std::nullopt,
                      .repack_bytes = std::nullopt,
                      .synchronization_count = 2,
                      .dispatch_count = std::nullopt},
             .profile_key = ProviderRouteKey{1}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *context,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  if (request.route != 1 || !request.shape.exact() ||
      !internal::supports(context, request.operation, request.inputs,
                          request.outputs))
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("ggml CUDA prepare domain differs"));
  try {
    internal::Recipe recipe;
    recipe.values.assign(request.inputs.begin(), request.inputs.end());
    recipe.values.push_back(request.outputs[0]);
    internal::Node node{request.operation,
                        {},
                        static_cast<std::uint32_t>(request.inputs.size())};
    for (std::uint32_t i = 0; i < request.inputs.size(); ++i) {
      recipe.inputs.push_back(i);
      node.inputs.push_back(i);
    }
    recipe.outputs.push_back(node.output);
    recipe.nodes.push_back(std::move(node));
    return internal::compile(context, std::move(recipe), allocator, output);
  } catch (...) {
    return ProviderPrepareOutcome::error(failure());
  }
}
Status drain_context(void *) noexcept { return Status::Ok(); }
const void *query_extension(void *value, ProviderExtensionId id) noexcept {
  const auto &extension = static_cast<Context *>(value)->extension;
  return id == extension.id ? extension.table : nullptr;
}
const detail::ProviderVTable kProvider{
    .identity = {.id = kProviderId,
                 .name = "ggml-cuda",
                 .version = "456172ec+hyle-resource-v1"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain_context,
    .query_extension = &query_extension};
} // namespace

namespace internal {
bool supports(void *value, Operation operation,
              std::span<const TensorDesc> inputs,
              std::span<const TensorDesc> outputs) noexcept {
  if (!value || !static_cast<Context *>(value)->qualified ||
      outputs.size() != 1)
    return false;
  if (const auto *attention = operation.get_if<ScaledDotProductAttention>()) {
    if (inputs.size() != 3 || !attention->causal ||
        attention->layout != AttentionLayout::kBatchSequenceHeadDepth ||
        attention->mask != MaskKind::kNone ||
        attention->accumulation_type != DataType::kFloat32 ||
        !std::isfinite(attention->scale))
      return false;
    const auto type = inputs[0].data_type();
    if ((type != DataType::kFloat16 && type != DataType::kBFloat16) ||
        (type == DataType::kBFloat16 &&
         !static_cast<Context *>(value)->bfloat16_attention))
      return false;
    for (const auto &description : inputs)
      if (!description.valid() || !description.is_plain() ||
          !description.is_dense() || description.rank() != 4 ||
          description.data_type() != type || !description.element_count() ||
          *description.element_count() > INT32_MAX)
        return false;
    const auto q = inputs[0].dimensions();
    const auto k = inputs[1].dimensions();
    if (q[0] <= 0 || q[0] > 65535 || q[1] <= 0 || k[1] < q[1] || q[2] != 8 ||
        k[2] != 2 || q[3] != 256 || k[3] != 256 || q[0] != k[0] ||
        inputs[1] != inputs[2] || outputs[0] != inputs[0])
      return false;
    // Bound native indexing and the BF16 score/mask allocation before any
    // upstream API asserts or buffer-size arithmetic can be reached.
    return k[1] <= INT32_MAX / q[1] / q[2] / q[0];
  }
  return tensor_operation_supports(operation, inputs, outputs);
}

ProviderPrepareOutcome compile(void *value, Recipe recipe,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  const auto &context = *static_cast<Context *>(value);
  std::vector<std::size_t> alignments;
  try {
    alignments.resize(recipe.values.size(), 1);
    for (const auto &node : recipe.nodes) {
      const auto require = [&](std::uint32_t id) {
        alignments[id] = std::max(
            alignments[id],
            boundary_alignment(node.operation, recipe.values[id].data_type()));
      };
      for (const auto id : node.inputs)
        require(id);
      require(node.output);
    }
  } catch (...) {
    return ProviderPrepareOutcome::error(failure());
  }
  void *storage = allocator.allocate(sizeof(Program), alignof(Program));
  if (!storage)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("ggml CUDA program allocation failed"));
  *output = {.vtable = &kPrepared,
             .state = ::new (storage)
                 Program{context.primary, context.device, std::move(recipe),
                         std::move(alignments)},
             .workspace = {},
             .runtime = kRuntime,
             .persistent_bytes = std::nullopt,
             .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}
Provider with_extension(ProviderExtensionId id, const void *table) noexcept {
  return detail::ProviderAccess::make(&kProvider, Extension{id, table});
}
} // namespace internal

Provider provider() noexcept { return internal::with_extension({}, nullptr); }
} // namespace hyle::compute::ggml_cuda
