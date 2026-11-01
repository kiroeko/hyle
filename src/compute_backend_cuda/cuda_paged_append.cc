#include "cuda_internal.h"
#include "cuda_paged_append.h"

#include <memory>
#include <utility>

#include "hyle/compute/detail/paged_append_backend.h"
#include "hyle/compute/detail/device_choice_paged_append_backend.h"

namespace hyle::compute::cuda::detail {
namespace {
using compute::detail::EnqueueResult;
using compute::detail::PagedAppendNativeBindings;
using compute::detail::PagedAppendNativeSpec;

struct AppendProgram final {
  ContextState *context = nullptr; // borrowed from the original extension owner
  const void *kernel = nullptr;
  PagedAppendNativeSpec spec;
  std::size_t arena_bytes = 0, input_bytes = 0, metadata_bytes = 0;
};

Status prepare_append(void *opaque, PagedAppendNativeSpec spec,
                      mem::AllocatorRef allocator, void **output) noexcept {
  if (!output) return Status::InvalidArgument("CUDA paged append output is null");
  *output = nullptr;
  if (!opaque || !allocator.valid() || !spec.page_rows || !spec.physical_pages ||
      !spec.input_rows || !spec.instruction_capacity || !spec.row_bytes ||
      spec.row_bytes % sizeof(std::uint16_t) != 0)
    return Status::InvalidArgument("CUDA paged append needs finite nonempty raw16 geometry");
  if (spec.row_bytes > SIZE_MAX / spec.page_rows ||
      spec.row_bytes > SIZE_MAX / spec.input_rows)
    return Status::ResourceExhausted("CUDA paged append row extent overflows");
  const auto page_bytes = spec.row_bytes * spec.page_rows;
  if (page_bytes > SIZE_MAX / spec.physical_pages ||
      spec.instruction_capacity >
          (SIZE_MAX - sizeof(compute::detail::PagedAppendMetadata)) /
              sizeof(compute::detail::PagedAppendInstruction))
    return Status::ResourceExhausted("CUDA paged append capacity overflows");
  auto *context = static_cast<ContextState *>(opaque);
  ScopedPrimaryContext scope(context->primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  const auto *kernel = paged_append_kernel();
  cudaFuncAttributes attributes{};
  HYLE_RETURN_IF_ERROR(runtime_status(cudaFuncGetAttributes(&attributes, kernel)));
  if (attributes.maxThreadsPerBlock < 256)
    return Status::Unimplemented("CUDA paged append kernel has insufficient block capacity");
  auto *storage = allocator.allocate(sizeof(AppendProgram), alignof(AppendProgram));
  if (!storage) return Status::ResourceExhausted("CUDA paged append program metadata");
  auto *program = std::construct_at(static_cast<AppendProgram *>(storage));
  program->context = context; program->kernel = kernel; program->spec = spec;
  program->arena_bytes = page_bytes * spec.physical_pages;
  program->input_bytes = spec.row_bytes * spec.input_rows;
  program->metadata_bytes = sizeof(compute::detail::PagedAppendMetadata) +
      std::size_t(spec.instruction_capacity) * sizeof(compute::detail::PagedAppendInstruction);
  *output = program;
  return Status::Ok();
}

void destroy_append(void *opaque, mem::AllocatorRef allocator) noexcept {
  auto *program = static_cast<AppendProgram *>(opaque);
  if (!program) return;
  std::destroy_at(program);
  allocator.deallocate(program, sizeof(AppendProgram), alignof(AppendProgram));
}

Status validate_carrier(ContextState *context, mem::ConstBufferView view,
                        std::size_t bytes, std::size_t alignment) noexcept {
  const auto *selected = resource_const(view.resource());
  if (!view.valid() || !selected || selected->context() != context ||
      selected->kind() == MemoryKind::kPinnedHost || view.size() < bytes)
    return Status::InvalidArgument("CUDA paged append carrier context/capacity is invalid");
  const auto address = reinterpret_cast<std::uintptr_t>(byte_pointer(view));
  if (!address || address % alignment != 0 || view.size() > SIZE_MAX - address)
    return Status::InvalidArgument("CUDA paged append carrier address is invalid");
  return Status::Ok();
}

EnqueueResult enqueue_append(void *opaque, void *queue_opaque, void *program_opaque,
                              PagedAppendNativeBindings bindings) noexcept {
  auto *queue = static_cast<QueueState *>(queue_opaque);
  auto *program = static_cast<AppendProgram *>(program_opaque);
  if (!queue || !program || queue->context != opaque ||
      program->context != opaque || !queue->active_sequence)
    return EnqueueResult::rejected(Status::InvalidArgument(
        "CUDA paged append requires its original admitted context/queue"));
  auto checked = validate_carrier(queue->context, bindings.arena,
      program->arena_bytes, alignof(std::uint16_t));
  if (!checked.ok()) return EnqueueResult::rejected(std::move(checked));
  checked = validate_carrier(queue->context, bindings.rows,
      program->input_bytes, alignof(std::uint16_t));
  if (!checked.ok()) return EnqueueResult::rejected(std::move(checked));
  checked = validate_carrier(queue->context, bindings.metadata,
      program->metadata_bytes, alignof(compute::detail::PagedAppendInstruction));
  if (!checked.ok()) return EnqueueResult::rejected(std::move(checked));
  ScopedPrimaryContext scope(queue->context->primary);
  if (!scope) return EnqueueResult::rejected(scope.status());
  // The original admitted sequence owns completion/error propagation. A
  // native launch error may include an earlier asynchronous error, so retain
  // the accepted prefix rather than attempting another execution route.
  return EnqueueResult::accepted_command(runtime_status(launch_paged_append(
      program->kernel, const_cast<void *>(byte_pointer(bindings.arena)),
      byte_pointer(bindings.rows), byte_pointer(bindings.metadata),
      program->spec, queue->stream)));
}

const compute::detail::PagedAppendBackendVTable table{
    &prepare_append, &destroy_append, &enqueue_append, true};
Status prepare_choice(void*opaque,PagedAppendNativeSpec spec,mem::AllocatorRef allocator,void**output)noexcept{
  if(spec.instruction_capacity<3)return Status::InvalidArgument("device-choice append needs two nonempty streams and a directory");
  HYLE_RETURN_IF_ERROR(prepare_append(opaque,spec,allocator,output));
  auto*program=static_cast<AppendProgram*>(*output);program->kernel=device_choice_paged_append_kernel();
  ScopedPrimaryContext scope(program->context->primary);
  auto status=scope.status();
  if(status.ok()){cudaFuncAttributes attributes{};status=runtime_status(cudaFuncGetAttributes(&attributes,program->kernel));
    if(status.ok() && attributes.maxThreadsPerBlock<256)status=Status::Unimplemented("device-choice append kernel block capacity is insufficient");}
  if(!status.ok()){destroy_append(program,allocator);*output=nullptr;}
  return status;
}
EnqueueResult enqueue_choice(void*opaque,void*queue_opaque,void*program_opaque,
    compute::detail::DeviceChoicePagedAppendNativeBindings bindings)noexcept{
  auto*queue=static_cast<QueueState*>(queue_opaque);auto*program=static_cast<AppendProgram*>(program_opaque);
  if(!queue || !program || queue->context!=opaque || program->context!=opaque || !queue->active_sequence)
    return EnqueueResult::rejected(Status::InvalidArgument("device-choice append needs its original admitted Queue"));
  struct Check final{mem::ConstBufferView view;std::size_t bytes,alignment;};
  for(const auto&entry:std::array<Check,4>{{
      {bindings.arena,program->arena_bytes,alignof(std::uint16_t)},
      {bindings.rows,program->input_bytes,alignof(std::uint16_t)},
      {bindings.metadata,program->metadata_bytes,alignof(compute::detail::PagedAppendInstruction)},
      {bindings.selector,sizeof(std::int32_t),alignof(std::int32_t)}}}){
    auto status=validate_carrier(queue->context,entry.view,entry.bytes,entry.alignment);
    if(!status.ok())return EnqueueResult::rejected(std::move(status));
  }
  ScopedPrimaryContext scope(queue->context->primary);if(!scope)return EnqueueResult::rejected(scope.status());
  return EnqueueResult::accepted_command(runtime_status(launch_device_choice_paged_append(program->kernel,
      const_cast<void*>(byte_pointer(bindings.arena)),byte_pointer(bindings.rows),byte_pointer(bindings.metadata),
      byte_pointer(bindings.selector),program->spec,queue->stream)));
}
const compute::detail::DeviceChoicePagedAppendBackendVTable choice_table{
    &prepare_choice,&destroy_append,&enqueue_choice,true};
} // namespace

const compute::detail::PagedAppendBackendVTable *paged_append_interface() noexcept {
  return &table;
}
const compute::detail::DeviceChoicePagedAppendBackendVTable*device_choice_paged_append_interface()noexcept{return &choice_table;}
} // namespace hyle::compute::cuda::detail
