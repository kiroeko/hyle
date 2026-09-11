#include "hyle/compute/providers/onednn.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
#include <span>
#include <utility>

#include "provider_internal.h"
#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_threadpool.h"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

namespace hyle::compute::onednn {
namespace {

static_assert(DNNL_VERSION_MAJOR == 3 && DNNL_VERSION_MINOR == 13 &&
              DNNL_VERSION_PATCH == 2,
              "update the HYLE oneDNN provider identity with its source lock");

Status translate_status(dnnl_status_t status,
                        const char *operation) noexcept {
  switch (status) {
  case dnnl_success:
    return Status::Ok();
  case dnnl_out_of_memory:
    return Status::ResourceExhausted(operation);
  case dnnl_invalid_arguments:
  case dnnl_invalid_graph:
  case dnnl_invalid_graph_op:
  case dnnl_invalid_shape:
  case dnnl_invalid_data_type:
    return Status::InvalidArgument(operation);
  case dnnl_unimplemented:
  case dnnl_last_impl_reached:
  case dnnl_not_required:
    return Status::Unimplemented(operation);
  case dnnl_runtime_error:
    return Status::Internal(operation);
  }
  return Status::Internal(operation);
}

class ThreadPoolAdapter final
    : public dnnl::threadpool_interop::threadpool_iface {
public:
  explicit ThreadPoolAdapter(ParallelExecutorRef executor) noexcept
      : executor_(executor) {}

  int get_num_threads() const override {
    return static_cast<int>((std::min)(executor_.concurrency(),
                                       static_cast<std::size_t>(INT_MAX)));
  }

  bool get_in_parallel() const override { return executor_.in_parallel(); }

  void parallel_for(int count,
                    const std::function<void(int, int)> &task) override {
    if (count <= 0)
      return;
    struct Bridge final {
      const std::function<void(int, int)> *task = nullptr;
      int count = 0;
    } bridge{&task, count};
    executor_.parallel_for(
        static_cast<std::size_t>(count),
        [](void *context, std::size_t index, std::size_t) noexcept {
          auto &value = *static_cast<Bridge *>(context);
          (*value.task)(static_cast<int>(index), value.count);
        },
        &bridge);
  }

  std::uint64_t get_flags() const override { return 0; }
  void wait() override {}

private:
  ParallelExecutorRef executor_{};
};

struct OneDnnContext final {
  dnnl_engine_t engine = nullptr;
  ParallelExecutorRef executor{};
};

struct OneDnnPlan final {
  OneDnnContext *context = nullptr;
  dnnl_primitive_desc_t descriptor = nullptr;
  dnnl_primitive_t primitive = nullptr;
  std::size_t scratchpad_size = 0;
};

struct OneDnnInvocation final {
  explicit OneDnnInvocation(ParallelExecutorRef executor) noexcept
      : thread_pool(executor) {}

  ThreadPoolAdapter thread_pool;
  dnnl_stream_t stream = nullptr;
  dnnl_memory_t source = nullptr;
  dnnl_memory_t weights = nullptr;
  dnnl_memory_t destination = nullptr;
  dnnl_memory_t scratchpad = nullptr;
  mem::HostMapping scratchpad_mapping{};
};

void destroy_memory(dnnl_memory_t &memory) noexcept {
  if (memory != nullptr) {
    dnnl_memory_destroy(memory);
    memory = nullptr;
  }
}

void destroy_invocation_contents(OneDnnInvocation &invocation) noexcept {
  destroy_memory(invocation.scratchpad);
  destroy_memory(invocation.destination);
  destroy_memory(invocation.weights);
  destroy_memory(invocation.source);
  if (invocation.stream != nullptr) {
    dnnl_stream_destroy(invocation.stream);
    invocation.stream = nullptr;
  }
  invocation.scratchpad_mapping.close().IgnoreError();
}

Status create_memory(dnnl_memory_t *memory,
                     const_dnnl_memory_desc_t description,
                     dnnl_engine_t engine) noexcept {
  if (description == nullptr) {
    return Status::Internal(
        "oneDNN did not expose a required MatMul memory description");
  }
  return translate_status(
      dnnl_memory_create(memory, description, engine, DNNL_MEMORY_NONE),
      "oneDNN failed to create MatMul memory metadata");
}

bool is_row_major_f32_matrix(const TensorDesc &description) noexcept {
  return description.valid() && description.data_type() == DataType::kFloat32 &&
         description.rank() == 2 && description.is_dense();
}

Status create_context(const Provider &provider_value, Device device,
                      mem::AllocatorRef allocator,
                      void **context_output) noexcept {
  if (context_output == nullptr)
    return Status::InvalidArgument("oneDNN context output is null");
  *context_output = nullptr;
  if (device != Device::cpu())
    return Status::Unimplemented("oneDNN provider requires a CPU device");

  void *storage =
      allocator.allocate(sizeof(OneDnnContext), alignof(OneDnnContext));
  if (storage == nullptr)
    return Status::ResourceExhausted("oneDNN context allocation failed");
  auto *context = ::new (storage) OneDnnContext{
      .engine = nullptr,
      .executor = detail::ProviderAccess::payload<Options>(provider_value)
                      .executor,
  };
  Status status = translate_status(
      dnnl_engine_create(&context->engine, dnnl_cpu, 0),
      "oneDNN failed to create its CPU engine");
  if (!status.ok()) {
    context->~OneDnnContext();
    allocator.deallocate(context, sizeof(OneDnnContext),
                         alignof(OneDnnContext));
    return status;
  }
  *context_output = context;
  return Status::Ok();
}

void destroy_context(void *context_value,
                     mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<OneDnnContext *>(context_value);
  if (context == nullptr)
    return;
  if (context->engine != nullptr)
    dnnl_engine_destroy(context->engine);
  context->~OneDnnContext();
  allocator.deallocate(context, sizeof(OneDnnContext),
                       alignof(OneDnnContext));
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *plan = static_cast<OneDnnPlan *>(state);
  if (plan == nullptr)
    return;
  if (plan->primitive != nullptr)
    dnnl_primitive_destroy(plan->primitive);
  if (plan->descriptor != nullptr)
    dnnl_primitive_desc_destroy(plan->descriptor);
  plan->~OneDnnPlan();
  allocator.deallocate(plan, sizeof(OneDnnPlan), alignof(OneDnnPlan));
}

Status create_invocation(void *prepared_value, mem::AllocatorRef allocator,
                         mem::BufferView workspace,
                         void **invocation_output) noexcept {
  if (invocation_output == nullptr)
    return Status::InvalidArgument("oneDNN invocation output is null");
  *invocation_output = nullptr;
  auto *plan = static_cast<OneDnnPlan *>(prepared_value);
  if (plan == nullptr || plan->context == nullptr)
    return Status::Internal("oneDNN MatMul plan is invalid");

  void *storage = allocator.allocate(sizeof(OneDnnInvocation),
                                     alignof(OneDnnInvocation));
  if (storage == nullptr)
    return Status::ResourceExhausted("oneDNN invocation allocation failed");
  auto *invocation =
      ::new (storage) OneDnnInvocation(plan->context->executor);

  Status status = translate_status(
      dnnl_threadpool_interop_stream_create(
          &invocation->stream, plan->context->engine,
          static_cast<void *>(&invocation->thread_pool)),
      "oneDNN failed to create an execution stream");
  if (status.ok()) {
    status = create_memory(
        &invocation->source,
        dnnl_primitive_desc_query_md(plan->descriptor,
                                     dnnl_query_exec_arg_md, DNNL_ARG_SRC),
        plan->context->engine);
  }
  if (status.ok()) {
    status = create_memory(
        &invocation->weights,
        dnnl_primitive_desc_query_md(plan->descriptor,
                                     dnnl_query_exec_arg_md,
                                     DNNL_ARG_WEIGHTS),
        plan->context->engine);
  }
  if (status.ok()) {
    status = create_memory(
        &invocation->destination,
        dnnl_primitive_desc_query_md(plan->descriptor,
                                     dnnl_query_exec_arg_md, DNNL_ARG_DST),
        plan->context->engine);
  }
  if (status.ok() && plan->scratchpad_size != 0) {
    const_dnnl_memory_desc_t scratchpad_description =
        dnnl_primitive_desc_query_md(plan->descriptor,
                                     dnnl_query_exec_arg_md,
                                     DNNL_ARG_SCRATCHPAD);
    status = create_memory(&invocation->scratchpad, scratchpad_description,
                           plan->context->engine);
    if (status.ok()) {
      auto mapping = workspace.map_host(mem::HostAccess::kReadWrite);
      if (!mapping) {
        status = std::move(mapping).error();
      } else {
        invocation->scratchpad_mapping = std::move(*mapping);
        auto bytes = invocation->scratchpad_mapping.writable_bytes();
        if (!bytes) {
          status = std::move(bytes).error();
        } else if (bytes->size() < plan->scratchpad_size ||
                   reinterpret_cast<std::uintptr_t>(bytes->data()) % 64 != 0) {
          status = Status::InvalidArgument(
              "oneDNN workspace must satisfy its size and 64-byte alignment");
        } else {
          status = translate_status(
              dnnl_memory_set_data_handle(invocation->scratchpad,
                                          bytes->data()),
              "oneDNN rejected the caller-owned scratchpad");
        }
      }
    }
  }

  if (!status.ok()) {
    destroy_invocation_contents(*invocation);
    invocation->~OneDnnInvocation();
    allocator.deallocate(invocation, sizeof(OneDnnInvocation),
                         alignof(OneDnnInvocation));
    return status;
  }
  *invocation_output = invocation;
  return Status::Ok();
}

void destroy_invocation(void *invocation_value,
                        mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<OneDnnInvocation *>(invocation_value);
  if (invocation == nullptr)
    return;
  destroy_invocation_contents(*invocation);
  invocation->~OneDnnInvocation();
  allocator.deallocate(invocation, sizeof(OneDnnInvocation),
                       alignof(OneDnnInvocation));
}

Status clear_data_handle(dnnl_memory_t memory) noexcept {
  return translate_status(dnnl_memory_set_data_handle(memory, DNNL_MEMORY_NONE),
                          "oneDNN failed to release a tensor binding");
}

Status execute(void *prepared_value, void *invocation_value,
               std::span<const void *const> inputs,
               std::span<void *const> outputs) noexcept {
  auto *plan = static_cast<OneDnnPlan *>(prepared_value);
  auto *invocation = static_cast<OneDnnInvocation *>(invocation_value);
  if (plan == nullptr || invocation == nullptr || inputs.size() != 2 ||
      outputs.size() != 1 || inputs[0] == nullptr || inputs[1] == nullptr ||
      outputs[0] == nullptr) {
    return Status::Internal("oneDNN MatMul execution state is invalid");
  }

  Status status = translate_status(
      dnnl_memory_set_data_handle(invocation->source,
                                  const_cast<void *>(inputs[0])),
      "oneDNN rejected the MatMul source binding");
  if (status.ok()) {
    status = translate_status(
        dnnl_memory_set_data_handle(invocation->weights,
                                    const_cast<void *>(inputs[1])),
        "oneDNN rejected the MatMul weight binding");
  }
  if (status.ok()) {
    status = translate_status(
        dnnl_memory_set_data_handle(invocation->destination, outputs[0]),
        "oneDNN rejected the MatMul output binding");
  }

  std::array<dnnl_exec_arg_t, 4> arguments{{
      {DNNL_ARG_SRC, invocation->source},
      {DNNL_ARG_WEIGHTS, invocation->weights},
      {DNNL_ARG_DST, invocation->destination},
      {DNNL_ARG_SCRATCHPAD, invocation->scratchpad},
  }};
  const int argument_count = plan->scratchpad_size == 0 ? 3 : 4;
  if (status.ok()) {
    status = translate_status(
        dnnl_primitive_execute(plan->primitive, invocation->stream,
                               argument_count, arguments.data()),
        "oneDNN MatMul execution failed");
  }
  if (status.ok()) {
    status = translate_status(dnnl_stream_wait(invocation->stream),
                              "oneDNN MatMul stream wait failed");
  }

  // An invocation retains provider metadata and caller-owned scratchpad only;
  // tensor mappings belong to this submit and must never escape it.
  Status clear = clear_data_handle(invocation->destination);
  if (status.ok() && !clear.ok())
    status = std::move(clear);
  clear = clear_data_handle(invocation->weights);
  if (status.ok() && !clear.ok())
    status = std::move(clear);
  clear = clear_data_handle(invocation->source);
  if (status.ok() && !clear.ok())
    status = std::move(clear);
  return status;
}

const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .execute = &execute,
};

Status prepare(void *context_value, Operation operation,
               std::span<const TensorDesc> inputs,
               std::span<const TensorDesc> outputs,
               mem::AllocatorRef allocator,
               detail::ProviderPrepared *prepared_output) noexcept {
  if (prepared_output == nullptr)
    return Status::InvalidArgument("oneDNN prepared output is null");
  *prepared_output = {};
  auto *context = static_cast<OneDnnContext *>(context_value);
  if (context == nullptr || context->engine == nullptr)
    return Status::Internal("oneDNN provider context is invalid");
  if (operation.kind() != OperationKind::kMatMul || inputs.size() != 2 ||
      outputs.size() != 1) {
    return Status::Unimplemented(
        "oneDNN provider does not implement this operation");
  }
  if (!is_row_major_f32_matrix(inputs[0]) ||
      !is_row_major_f32_matrix(inputs[1]) ||
      !is_row_major_f32_matrix(outputs[0])) {
    return Status::Unimplemented(
        "oneDNN provider route requires dense row-major float32 matrices");
  }
  const auto left_dimensions = inputs[0].dimensions();
  const auto right_dimensions = inputs[1].dimensions();
  if (left_dimensions[0] == 0 || left_dimensions[1] == 0 ||
      right_dimensions[1] == 0) {
    return Status::Unimplemented(
        "zero-size MatMul remains on the HYLE-native route");
  }

  dnnl_memory_desc_t source_description = nullptr;
  dnnl_memory_desc_t weights_description = nullptr;
  dnnl_memory_desc_t destination_description = nullptr;
  dnnl_primitive_attr_t attributes = nullptr;
  dnnl_primitive_desc_t descriptor = nullptr;
  dnnl_primitive_t primitive = nullptr;

  auto make_description = [](dnnl_memory_desc_t *description,
                             const TensorDesc &tensor) noexcept {
    const auto dimensions = tensor.dimensions();
    const auto strides = tensor.element_strides();
    return dnnl_memory_desc_create_with_strides(
        description, 2, dimensions.data(), dnnl_f32, strides.data());
  };

  Status status = translate_status(make_description(&source_description,
                                                    inputs[0]),
                                   "oneDNN rejected the MatMul source shape");
  if (status.ok()) {
    status = translate_status(make_description(&weights_description, inputs[1]),
                              "oneDNN rejected the MatMul weight shape");
  }
  if (status.ok()) {
    status = translate_status(
        make_description(&destination_description, outputs[0]),
        "oneDNN rejected the MatMul destination shape");
  }
  if (status.ok()) {
    status = translate_status(dnnl_primitive_attr_create(&attributes),
                              "oneDNN MatMul attribute allocation failed");
  }
  if (status.ok()) {
    status = translate_status(
        dnnl_primitive_attr_set_scratchpad_mode(attributes,
                                                dnnl_scratchpad_mode_user),
        "oneDNN rejected caller-owned MatMul scratchpad policy");
  }
  if (status.ok()) {
    status = translate_status(
        dnnl_matmul_primitive_desc_create(
            &descriptor, context->engine, source_description,
            weights_description, nullptr, destination_description, attributes),
        "oneDNN cannot prepare this MatMul shape");
  }
  if (status.ok()) {
    status = translate_status(dnnl_primitive_create(&primitive, descriptor),
                              "oneDNN failed to prepare MatMul");
  }

  if (attributes != nullptr)
    dnnl_primitive_attr_destroy(attributes);
  if (destination_description != nullptr)
    dnnl_memory_desc_destroy(destination_description);
  if (weights_description != nullptr)
    dnnl_memory_desc_destroy(weights_description);
  if (source_description != nullptr)
    dnnl_memory_desc_destroy(source_description);

  if (!status.ok()) {
    if (primitive != nullptr)
      dnnl_primitive_destroy(primitive);
    if (descriptor != nullptr)
      dnnl_primitive_desc_destroy(descriptor);
    return status;
  }

  const_dnnl_memory_desc_t scratchpad_description =
      dnnl_primitive_desc_query_md(descriptor, dnnl_query_exec_arg_md,
                                   DNNL_ARG_SCRATCHPAD);
  if (scratchpad_description == nullptr) {
    dnnl_primitive_destroy(primitive);
    dnnl_primitive_desc_destroy(descriptor);
    return Status::Internal(
        "oneDNN did not expose its MatMul scratchpad requirement");
  }
  const std::size_t scratchpad_size =
      dnnl_memory_desc_get_size(scratchpad_description);

  void *storage = allocator.allocate(sizeof(OneDnnPlan), alignof(OneDnnPlan));
  if (storage == nullptr) {
    dnnl_primitive_destroy(primitive);
    dnnl_primitive_desc_destroy(descriptor);
    return Status::ResourceExhausted("oneDNN MatMul plan allocation failed");
  }
  auto *plan = ::new (storage) OneDnnPlan{
      .context = context,
      .descriptor = descriptor,
      .primitive = primitive,
      .scratchpad_size = scratchpad_size,
  };
  prepared_output->vtable = &kPreparedVTable;
  prepared_output->state = plan;
  prepared_output->workspace = WorkspaceRequirement{
      .size = scratchpad_size,
      .alignment = scratchpad_size == 0 ? 1U : 64U,
  };
  return Status::Ok();
}

const detail::ProviderVTable kProviderVTable{
    .kind = ProviderKind::kOneDnn,
    .name = "onednn",
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .prepare = &prepare,
};

} // namespace

Provider provider(Options options) noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, options);
}

std::string_view version() noexcept { return "3.13.2"; }

} // namespace hyle::compute::onednn
