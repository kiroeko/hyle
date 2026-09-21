#include "normalization_internal.h"
#include "tensor_internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <new>
#include <string_view>
#include <utility>

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_threadpool.h"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

namespace hyle::compute::onednn::internal {
namespace {

struct VendorFailure final {
  dnnl_status_t code;
};

void check(dnnl_status_t code) {
  if (code != dnnl_success)
    throw VendorFailure{code};
}

template <class Function> Status vendor_call(Function &&function) noexcept {
  try {
    function();
    return Status::Ok();
  } catch (const VendorFailure &failure) {
    if (failure.code == dnnl_out_of_memory)
      return Status::ResourceExhausted("oneDNN primitive allocation failed");
    if (failure.code == dnnl_unimplemented)
      return Status::Unimplemented("oneDNN primitive domain is unsupported");
    return Status::Internal("oneDNN primitive rejected a validated request");
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("oneDNN primitive allocation threw");
  } catch (...) {
    return Status::Internal("oneDNN primitive raised a foreign exception");
  }
}

class Threadpool final : public dnnl::threadpool_interop::threadpool_iface {
public:
  explicit Threadpool(ParallelExecutorRef executor) noexcept
      : executor_(executor) {}
  int get_num_threads() const override {
    return static_cast<int>(executor_.concurrency());
  }
  bool get_in_parallel() const override { return executor_.in_parallel(); }
  std::uint64_t get_flags() const override { return 0; }
  void wait() override {}
  void parallel_for(int count,
                    const std::function<void(int, int)> &function) override {
    // A foreign exception may not cross HYLE's infallible worker callback.
    // The borrowed synchronous executor joins every task before we report it.
    struct Work final {
      const std::function<void(int, int)> *function;
      std::atomic<int> failure{0};
    } work{&function};
    executor_.parallel_for(
        static_cast<std::size_t>(count),
        [](void *value, std::size_t index, std::size_t size) noexcept {
          auto &work = *static_cast<Work *>(value);
          try {
            (*work.function)(static_cast<int>(index), static_cast<int>(size));
          } catch (const std::bad_alloc &) {
            work.failure.store(1, std::memory_order_relaxed);
          } catch (...) {
            work.failure.store(2, std::memory_order_relaxed);
          }
        },
        &work);
    const int failure = work.failure.load(std::memory_order_relaxed);
    if (failure != 0)
      throw VendorFailure{failure == 1 ? dnnl_out_of_memory
                                       : dnnl_runtime_error};
  }

private:
  ParallelExecutorRef executor_{};
};

struct ConcurrencyScope final {
  int previous = 0;
  bool active = false;
  explicit ConcurrencyScope(ParallelExecutorRef executor) {
    check(dnnl_threadpool_interop_get_max_concurrency(&previous));
    check(dnnl_threadpool_interop_set_max_concurrency(
        static_cast<int>(executor.concurrency())));
    active = true;
  }
  ~ConcurrencyScope() {
    if (active)
      dnnl_threadpool_interop_set_max_concurrency(previous);
  }
};

// Vendor owners live only inside this adapter; their allocation sizes are not
// represented as zero or charged to HYLE's allocator quota.
struct PrimitivePlan final {
  ParallelExecutorRef executor{};
  dnnl_engine_t engine = nullptr;
  dnnl_primitive_desc_t descriptor = nullptr;
  dnnl_primitive_t primitive = nullptr;
  std::array<int, kNormMaxInputs> input_args{};
  std::size_t input_count = 0;
  std::size_t scratch_size = 0;
  ~PrimitivePlan() {
    dnnl_primitive_destroy(primitive);
    dnnl_primitive_desc_destroy(descriptor);
    dnnl_engine_destroy(engine);
  }
};

struct PrimitiveInvocation final {
  explicit PrimitiveInvocation(ParallelExecutorRef executor) noexcept
      : threadpool(executor) {}
  Threadpool threadpool;
  mem::HostMapping workspace{};
  dnnl_stream_t stream = nullptr;
  std::array<dnnl_exec_arg_t, kNormMaxInputs + 2> args{};
  std::size_t arg_count = 0;
  ~PrimitiveInvocation() {
    for (auto &arg : args)
      dnnl_memory_destroy(arg.memory);
    dnnl_stream_destroy(stream);
    workspace.close().IgnoreError();
  }
};

struct BuildMemory final {
  dnnl_memory_desc_t data = nullptr;
  std::array<dnnl_memory_desc_t, kNormMaxPostOps> rhs{};
  dnnl_primitive_attr_t attributes = nullptr;
  dnnl_post_ops_t post_ops = nullptr;
  ~BuildMemory() {
    dnnl_primitive_attr_destroy(attributes);
    dnnl_post_ops_destroy(post_ops);
    for (auto descriptor : rhs)
      dnnl_memory_desc_destroy(descriptor);
    dnnl_memory_desc_destroy(data);
  }
};

dnnl_data_type_t vendor_type(DataType type) noexcept {
  switch (type) {
  case DataType::kFloat32:
    return dnnl_f32;
  case DataType::kFloat16:
    return dnnl_f16;
  case DataType::kBFloat16:
    return dnnl_bf16;
  default:
    return dnnl_data_type_undef;
  }
}

std::array<dnnl_dim_t, 2> matrix_shape(const TensorDesc &description) noexcept {
  const auto dimensions = description.dimensions();
  dnnl_dim_t rows = 1;
  for (std::size_t axis = 0; axis + 1 < dimensions.size(); ++axis)
    rows *= dimensions[axis];
  return {rows, dimensions.back()};
}

void destroy_prepared(void *value, mem::AllocatorRef allocator) noexcept {
  auto *plan = static_cast<PrimitivePlan *>(value);
  if (plan == nullptr)
    return;
  plan->~PrimitivePlan();
  allocator.deallocate(plan, sizeof(PrimitivePlan), alignof(PrimitivePlan));
}

void destroy_invocation(void *value, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<PrimitiveInvocation *>(value);
  if (invocation == nullptr)
    return;
  invocation->~PrimitiveInvocation();
  allocator.deallocate(invocation, sizeof(PrimitiveInvocation),
                       alignof(PrimitiveInvocation));
}

Status create_invocation(void *value, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  if (output == nullptr || value == nullptr)
    return Status::InvalidArgument("oneDNN primitive invocation is invalid");
  *output = nullptr;
  const auto &plan = *static_cast<PrimitivePlan *>(value);
  void *storage = allocator.allocate(sizeof(PrimitiveInvocation),
                                     alignof(PrimitiveInvocation));
  if (storage == nullptr)
    return Status::ResourceExhausted(
        "oneDNN primitive invocation allocation failed");
  auto *invocation = ::new (storage) PrimitiveInvocation(plan.executor);
  void *scratch = nullptr;
  Status status = Status::Ok();
  if (plan.scratch_size != 0) {
    auto mapped = workspace.map_host(mem::HostAccess::kReadWrite);
    if (!mapped) {
      status = std::move(mapped).error();
    } else {
      invocation->workspace = std::move(*mapped);
      auto bytes = invocation->workspace.writable_bytes();
      if (!bytes) {
        status = std::move(bytes).error();
      } else if (bytes->size() < plan.scratch_size ||
                 reinterpret_cast<std::uintptr_t>(bytes->data()) %
                         kNormWorkspaceAlignment !=
                     0) {
        status = Status::InvalidArgument(
            "oneDNN primitive scratchpad is insufficient or misaligned");
      } else {
        scratch = bytes->data();
      }
    }
  }
  if (status.ok()) {
    status = vendor_call([&] {
      check(dnnl_threadpool_interop_stream_create(
          &invocation->stream, plan.engine, &invocation->threadpool));
      auto add_argument = [&](int argument, void *handle) {
        auto &slot = invocation->args[invocation->arg_count++];
        slot.arg = argument;
        const auto description = dnnl_primitive_desc_query_md(
            plan.descriptor, dnnl_query_exec_arg_md, argument);
        check(
            dnnl_memory_create(&slot.memory, description, plan.engine, handle));
      };
      for (std::size_t index = 0; index < plan.input_count; ++index)
        add_argument(plan.input_args[index], DNNL_MEMORY_NONE);
      add_argument(DNNL_ARG_DST, DNNL_MEMORY_NONE);
      if (plan.scratch_size != 0)
        add_argument(DNNL_ARG_SCRATCHPAD, scratch);
    });
  }
  if (!status.ok()) {
    destroy_invocation(invocation, allocator);
    return status;
  }
  *output = invocation;
  return Status::Ok();
}

void preserve_first(Status &status, Status candidate) noexcept {
  if (status.ok() && !candidate.ok())
    status = std::move(candidate);
}

Status execute(PrimitivePlan &plan, PrimitiveInvocation &invocation,
               std::span<const ConstTensorView> inputs,
               std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != plan.input_count || outputs.size() != 1)
    return Status::Internal("oneDNN primitive binding count is invalid");
  std::array<mem::HostMapping, kNormMaxInputs> mappings{};
  mem::HostMapping output;
  std::array<void *, kNormMaxInputs + 1> handles{};
  Status status = Status::Ok();
  bool exact_alias = false;
  const auto destination = outputs[0].storage();
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto source = inputs[index].storage();
    for (std::size_t prior = 0; prior < index; ++prior) {
      const auto previous = inputs[prior].storage();
      if (source.resource() == previous.resource() &&
          source.native_handle() == previous.native_handle() &&
          source.offset() == previous.offset() &&
          source.size() == previous.size()) {
        handles[index] = handles[prior];
        break;
      }
    }
    if (handles[index] != nullptr)
      continue;
    if (!source.resource().properties().has(
            mem::MemoryProperty::kHostDeviceCoherent)) {
      status = source.invalidate();
      if (!status.ok())
        break;
    }
    exact_alias = index == 0 && source.resource() == destination.resource() &&
                  source.native_handle() == destination.native_handle() &&
                  source.offset() == destination.offset() &&
                  source.size() == destination.size();
    // One read/write mapping for the supported exact alias. Do not request two
    // simultaneously conflicting maps from resources with exclusive mapping.
    auto mapped = exact_alias
                      ? destination.map_host(mem::HostAccess::kReadWrite)
                      : source.map_host();
    if (!mapped) {
      status = std::move(mapped).error();
      break;
    }
    mappings[index] = std::move(*mapped);
    handles[index] = const_cast<std::byte *>(mappings[index].bytes().data());
    if (exact_alias)
      handles[plan.input_count] = handles[index];
  }
  if (status.ok() && handles[plan.input_count] == nullptr) {
    auto mapped = destination.map_host(mem::HostAccess::kWrite);
    if (!mapped) {
      status = std::move(mapped).error();
    } else {
      output = std::move(*mapped);
      auto bytes = output.writable_bytes();
      if (!bytes)
        status = std::move(bytes).error();
      else
        handles[plan.input_count] = bytes->data();
    }
  }
  if (status.ok()) {
    status = vendor_call([&] {
      for (std::size_t index = 0; index <= plan.input_count; ++index)
        check(dnnl_memory_set_data_handle(invocation.args[index].memory,
                                          handles[index]));
      check(dnnl_primitive_execute(plan.primitive, invocation.stream,
                                   static_cast<int>(invocation.arg_count),
                                   invocation.args.data()));
    });
  }
  if (status.ok() && !destination.resource().properties().has(
                         mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first(status,
                   output.valid() ? output.flush() : mappings[0].flush());
  }
  preserve_first(status, output.close());
  for (auto &mapping : mappings)
    preserve_first(status, mapping.close());
  return status;
}

detail::EnqueueResult submit(void *plan, void *invocation, Queue &,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  return detail::EnqueueResult::accepted_command(execute(
      *static_cast<PrimitivePlan *>(plan),
      *static_cast<PrimitiveInvocation *>(invocation), inputs, outputs));
}

AliasSupport alias_support(void *, std::size_t output,
                           std::size_t input) noexcept {
  return {.supported = output == 0 && input == 0};
}

const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
};

AliasSupport no_alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}

const detail::ProviderPreparedVTable kTensorPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .alias_support = &no_alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
};

// The byte axis preserves storage bits. Floating reorder/select may round,
// quiet NaNs or flush subnormals even when the public operation only moves
// bits.
void byte_description(dnnl_memory_desc_t *output, const TensorDesc &input,
                      const TensorDesc &result, bool condition) {
  std::array<dnnl_dim_t, kMaxTensorRank + 1> dimensions{};
  std::array<dnnl_dim_t, kMaxTensorRank + 1> strides{};
  const auto leading = result.rank() - input.rank();
  for (std::size_t axis = 0; axis < result.rank(); ++axis) {
    dimensions[axis] = result.dimensions()[axis];
    // Broadcast by descriptor, retaining the original storage and byte stride.
    strides[axis] = axis < leading || input.dimensions()[axis - leading] == 1
                        ? 0
                        : input.byte_strides()[axis - leading];
  }
  dimensions[result.rank()] =
      condition ? 1
                : static_cast<dnnl_dim_t>(data_type_size(input.data_type()));
  strides[result.rank()] = 1;
  check(dnnl_memory_desc_create_with_strides(
      output, static_cast<int>(result.rank() + 1), dimensions.data(),
      condition ? dnnl_s8 : dnnl_u8, strides.data()));
}

} // namespace

Status validate_normalization(const NormRequest &request,
                              ParallelExecutorRef executor) noexcept {
  const auto *rms = request.operation.get_if<RmsNorm>();
  const auto *layer = request.operation.get_if<LayerNorm>();
  if (rms == nullptr && layer == nullptr)
    return Status::Unimplemented("oneDNN primitive requires a norm operation");
  if ((rms != nullptr ? rms->accumulation_type : layer->accumulation_type) !=
      DataType::kFloat32)
    return Status::Unimplemented(
        "oneDNN primitive supports float32 accumulation");
  if (request.norm_inputs == 0 ||
      request.norm_inputs > (rms != nullptr ? 2U : 3U) ||
      request.inputs.size() != request.norm_inputs + request.post_ops.size() ||
      request.post_ops.size() > kNormMaxPostOps)
    return Status::InvalidArgument("oneDNN primitive arity is invalid");
  if (executor.concurrency() >
      static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    return Status::Unimplemented("oneDNN executor concurrency exceeds int");
  const auto &input = request.inputs[0];
  if (!input.valid() || !input.is_plain() || !input.is_dense() ||
      input.rank() == 0 ||
      vendor_type(input.data_type()) == dnnl_data_type_undef)
    return Status::Unimplemented(
        "oneDNN primitive requires dense floating input");
  dnnl_dim_t elements = 1;
  for (const auto dimension : input.dimensions()) {
    if (dimension == 0)
      return Status::Unimplemented("empty normalization uses the native route");
    if (dimension < 0 ||
        dimension > (std::numeric_limits<dnnl_dim_t>::max)() / elements)
      return Status::OutOfRange("oneDNN primitive dimensions overflow");
    elements *= dimension;
  }
  auto inferred = infer_single_output(
      request.operation, request.inputs.first(request.norm_inputs));
  if (!inferred)
    return std::move(inferred).error();
  if (*inferred != request.output)
    return Status::InvalidArgument(
        "oneDNN primitive output does not match schema");
  for (std::size_t index = 1; index < request.inputs.size(); ++index) {
    if (!request.inputs[index].is_plain() || !request.inputs[index].is_dense())
      return Status::Unimplemented(
          "oneDNN primitive requires dense parameters");
  }
  if (request.norm_inputs == 3 &&
      request.inputs[1].data_type() != request.inputs[2].data_type())
    return Status::Unimplemented(
        "oneDNN primitive scale and bias types must match");
  if (!request.post_ops.empty() && input.data_type() != DataType::kFloat32)
    return Status::Unimplemented(
        "oneDNN norm post-ops preserve only float32 intermediate semantics");
  for (std::size_t index = 0; index < request.post_ops.size(); ++index) {
    const auto &post = request.post_ops[index];
    if (post.input != request.norm_inputs + index ||
        (post.kind != BinaryKind::kAdd && post.kind != BinaryKind::kMultiply))
      return Status::InvalidArgument("oneDNN norm post-op binding is invalid");
    const auto &rhs = request.inputs[post.input];
    const bool scalar =
        rhs.rank() == 0 || (rhs.rank() == 1 && rhs.dimensions()[0] == 1);
    const bool vector =
        rhs.rank() == 1 && rhs.dimensions()[0] == input.dimensions().back();
    if (rhs.data_type() != DataType::kFloat32 ||
        (!scalar && !vector && rhs != input))
      return Status::Unimplemented("oneDNN norm post-op requires scalar, row "
                                   "or full-shape float32 operand");
    // The pinned optimized norm supports scalar/last-axis broadcast. A full
    // multirow operand silently selects simple:any and generic per-element
    // post-ops. Keep the optimized prefix; Graph owns the remaining operation.
    if (!scalar && !vector && matrix_shape(input)[0] > 1)
      return Status::Unimplemented(
          "oneDNN optimized norm post-ops exclude full multirow operands");
  }
  return Status::Ok();
}

ProviderPrepareOutcome
prepare_normalization(const NormRequest &request, ParallelExecutorRef executor,
                      mem::AllocatorRef allocator,
                      detail::ProviderPrepared *prepared) noexcept {
  if (prepared == nullptr)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("oneDNN norm prepared output is null"));
  *prepared = {};
  Status status = validate_normalization(request, executor);
  if (!status.ok())
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  void *storage =
      allocator.allocate(sizeof(PrimitivePlan), alignof(PrimitivePlan));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("oneDNN primitive plan allocation failed"));
  auto *plan = ::new (storage) PrimitivePlan{.executor = executor};
  status = vendor_call([&] {
    ConcurrencyScope concurrency(executor);
    BuildMemory temporary;
    check(dnnl_engine_create(&plan->engine, dnnl_cpu, 0));
    const auto dimensions = matrix_shape(request.inputs[0]);
    check(dnnl_memory_desc_create_with_tag(
        &temporary.data, 2, dimensions.data(),
        vendor_type(request.inputs[0].data_type()), dnnl_ab));
    check(dnnl_primitive_attr_create(&temporary.attributes));
    check(dnnl_primitive_attr_set_scratchpad_mode(temporary.attributes,
                                                  dnnl_scratchpad_mode_user));
    check(dnnl_primitive_attr_set_fpmath_mode(temporary.attributes,
                                              dnnl_fpmath_mode_strict));
    check(dnnl_post_ops_create(&temporary.post_ops));
    for (std::size_t index = 0; index < request.post_ops.size(); ++index) {
      const auto &post = request.post_ops[index];
      const auto &rhs = request.inputs[post.input];
      const auto rhs_shape =
          rhs.rank() == 0 ? std::array<dnnl_dim_t, 2>{1, 1} : matrix_shape(rhs);
      check(dnnl_memory_desc_create_with_tag(
          &temporary.rhs[index], 2, rhs_shape.data(), dnnl_f32, dnnl_ab));
      check(dnnl_post_ops_append_binary(
          temporary.post_ops,
          post.kind == BinaryKind::kAdd ? dnnl_binary_add : dnnl_binary_mul,
          temporary.rhs[index]));
      plan->input_args[post.input] =
          DNNL_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>(index)) |
          DNNL_ARG_SRC_1;
    }
    check(dnnl_primitive_attr_set_post_ops(temporary.attributes,
                                           temporary.post_ops));
    const auto *rms = request.operation.get_if<RmsNorm>();
    const auto *layer = request.operation.get_if<LayerNorm>();
    unsigned flags =
        rms != nullptr ? dnnl_rms_norm : dnnl_normalization_flags_none;
    plan->input_args[0] = DNNL_ARG_SRC;
    if (request.norm_inputs > 1) {
      flags |= dnnl_use_scale;
      plan->input_args[1] = DNNL_ARG_SCALE;
    }
    if (request.norm_inputs > 2) {
      flags |= dnnl_use_shift;
      plan->input_args[2] = DNNL_ARG_SHIFT;
    }
    check(dnnl_layer_normalization_forward_primitive_desc_create_v2(
        &plan->descriptor, plan->engine, dnnl_forward_inference, temporary.data,
        temporary.data, nullptr,
        request.norm_inputs > 1 ? vendor_type(request.inputs[1].data_type())
                                : dnnl_f32,
        rms != nullptr ? rms->epsilon : layer->epsilon, flags,
        temporary.attributes));
    if (!request.post_ops.empty()) {
      const char *implementation = nullptr;
      check(dnnl_primitive_desc_query(plan->descriptor, dnnl_query_impl_info_str,
                                      0, &implementation));
      if (implementation == nullptr)
        throw VendorFailure{dnnl_runtime_error};
      const std::string_view name(implementation);
      if (name == "simple:any" || name == "ref:any")
        throw VendorFailure{dnnl_unimplemented};
    }
    check(dnnl_primitive_create(&plan->primitive, plan->descriptor));
    const auto scratch = dnnl_primitive_desc_query_md(
        plan->descriptor, dnnl_query_scratchpad_md, 0);
    plan->scratch_size = dnnl_memory_desc_get_size(scratch);
    plan->input_count = request.inputs.size();
  });
  if (!status.ok()) {
    destroy_prepared(plan, allocator);
    if (status.IsUnimplemented())
      return ProviderPrepareOutcome::unsupported(std::move(status));
    return status.IsResourceExhausted()
               ? ProviderPrepareOutcome::resource_declined(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  }
  *prepared = {.vtable = &kPreparedVTable,
               .state = plan,
               .workspace = {.size = plan->scratch_size,
                             .alignment = kNormWorkspaceAlignment},
               .runtime = kNormRuntime,
               .persistent_bytes = std::nullopt,
               .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}

Status validate_tensor(const TensorRequest &request,
                       ParallelExecutorRef executor) noexcept {
  const bool select = request.operation.kind() == OperationKind::kSelect;
  if (!select && request.operation.kind() != OperationKind::kCopy)
    return Status::Unimplemented("oneDNN tensor operation is not mapped");
  if (request.inputs.size() != (select ? 3U : 1U))
    return Status::InvalidArgument("oneDNN tensor input count is invalid");
  if (executor.concurrency() >
      static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    return Status::Unimplemented("oneDNN executor concurrency exceeds int");
  if (!request.output.valid() || !request.output.is_plain() ||
      !request.output.is_dense() ||
      data_type_size(request.output.data_type()) == 0)
    return Status::Unimplemented("oneDNN tensor requires plain dense output");
  const auto output_bytes = request.output.required_bytes();
  if (!output_bytes ||
      *output_bytes >
          static_cast<std::size_t>((std::numeric_limits<dnnl_dim_t>::max)()))
    return Status::Unimplemented("oneDNN tensor output span exceeds int64");
  for (std::size_t index = 0; index < request.inputs.size(); ++index) {
    const auto &input = request.inputs[index];
    if (!input.valid() || !input.is_plain() ||
        input.rank() > request.output.rank())
      return Status::Unimplemented("oneDNN tensor layout is unsupported");
    if (input.data_type() !=
        (select && index == 0 ? DataType::kBool : request.output.data_type()))
      return Status::InvalidArgument(
          "oneDNN tensor storage type is inconsistent");
    const auto leading = request.output.rank() - input.rank();
    for (std::size_t axis = 0; axis < input.rank(); ++axis) {
      const auto extent = input.dimensions()[axis];
      if (extent != request.output.dimensions()[leading + axis] &&
          (!select || extent != 1))
        return Status::InvalidArgument("oneDNN tensor shape is inconsistent");
    }
    const auto bytes = input.required_bytes();
    if (!bytes || *bytes > static_cast<std::size_t>(
                               (std::numeric_limits<dnnl_dim_t>::max)()))
      return Status::Unimplemented("oneDNN tensor storage span exceeds int64");
  }
  return Status::Ok();
}

ProviderPrepareOutcome
prepare_tensor(const TensorRequest &request, ParallelExecutorRef executor,
               mem::AllocatorRef allocator,
               detail::ProviderPrepared *prepared) noexcept {
  if (prepared == nullptr)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("oneDNN tensor prepared output is null"));
  *prepared = {};
  Status status = validate_tensor(request, executor);
  if (!status.ok())
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  void *storage =
      allocator.allocate(sizeof(PrimitivePlan), alignof(PrimitivePlan));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("oneDNN tensor plan allocation failed"));
  auto *plan = ::new (storage) PrimitivePlan{.executor = executor};
  status = vendor_call([&] {
    ConcurrencyScope concurrency(executor);
    BuildMemory temporary;
    check(dnnl_engine_create(&plan->engine, dnnl_cpu, 0));
    check(dnnl_primitive_attr_create(&temporary.attributes));
    check(dnnl_primitive_attr_set_scratchpad_mode(temporary.attributes,
                                                  dnnl_scratchpad_mode_user));
    const bool select = request.operation.kind() == OperationKind::kSelect;
    for (std::size_t index = 0; index < request.inputs.size(); ++index)
      byte_description(&temporary.rhs[index], request.inputs[index],
                       request.output, select && index == 0);
    byte_description(&temporary.data, request.output, request.output, false);
    if (select) {
      check(dnnl_binary_primitive_desc_create_v2(
          &plan->descriptor, plan->engine, dnnl_binary_select, temporary.rhs[1],
          temporary.rhs[2], temporary.rhs[0], temporary.data,
          temporary.attributes));
      plan->input_args[0] = DNNL_ARG_SRC_2;
      plan->input_args[1] = DNNL_ARG_SRC_0;
      plan->input_args[2] = DNNL_ARG_SRC_1;
    } else {
      check(dnnl_reorder_primitive_desc_create(
          &plan->descriptor, temporary.rhs[0], plan->engine, temporary.data,
          plan->engine, temporary.attributes));
      plan->input_args[0] = DNNL_ARG_FROM;
    }
    check(dnnl_primitive_create(&plan->primitive, plan->descriptor));
    const auto scratch = dnnl_primitive_desc_query_md(
        plan->descriptor, dnnl_query_scratchpad_md, 0);
    plan->scratch_size = dnnl_memory_desc_get_size(scratch);
    plan->input_count = request.inputs.size();
  });
  if (!status.ok()) {
    destroy_prepared(plan, allocator);
    if (status.IsUnimplemented())
      return ProviderPrepareOutcome::unsupported(std::move(status));
    return status.IsResourceExhausted()
               ? ProviderPrepareOutcome::resource_declined(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  }
  *prepared = {.vtable = &kTensorPreparedVTable,
               .state = plan,
               .workspace = {.size = plan->scratch_size,
                             .alignment = kNormWorkspaceAlignment},
               .runtime = kNormRuntime,
               .persistent_bytes = std::nullopt,
               .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}

} // namespace hyle::compute::onednn::internal
