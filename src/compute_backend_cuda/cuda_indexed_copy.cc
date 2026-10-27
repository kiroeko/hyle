#include "cuda_internal.h"
#include "cuda_indexed_copy.h"

#include <initializer_list>
#include <memory>
#include <utility>

#include "hyle/compute/detail/indexed_copy_backend.h"

namespace hyle::compute::cuda::detail {
namespace {
using compute::detail::EnqueueResult;
using compute::detail::IndexedCopyNativeBindings;
using compute::detail::IndexedCopyNativeSpec;

struct IndexedCopyProgram final {
  // Borrowed from the original retained backend extension, like paged append.
  ContextState *context = nullptr;
  const void *check_kernel = nullptr, *copy_kernel = nullptr;
  IndexedCopyLayout layout;
  std::size_t capacity_bytes = 0, input_bytes = 0, indices_bytes = 0;
};

Status prepare_copy(void *opaque, IndexedCopyNativeSpec spec,
                    mem::AllocatorRef allocator, void **output) noexcept {
  if (!output)
    return Status::InvalidArgument("CUDA indexed-copy output is null");
  *output = nullptr;
  if (!opaque || !allocator.valid() || !spec.slot_count || !spec.row_count ||
      spec.row_count > spec.slot_count || !spec.row_bytes || spec.slot_stride < spec.row_bytes)
    return Status::InvalidArgument("CUDA indexed-copy needs nonempty disjoint row geometry");
  // The capacity stops at the final logical item's end, not its padding.
  if (spec.row_count > SIZE_MAX / spec.row_bytes ||
      std::size_t(spec.slot_count - 1) > (SIZE_MAX - spec.row_bytes) / spec.slot_stride ||
      spec.row_count > SIZE_MAX / sizeof(std::int32_t))
    return Status::ResourceExhausted("CUDA indexed-copy byte extent overflows");
  auto *context = static_cast<ContextState *>(opaque);
  ScopedPrimaryContext scope(context->primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  const auto *check_kernel = indexed_copy_check_kernel();
  const auto *copy_kernel = indexed_copy_rows_kernel();
  for (const auto *kernel : {check_kernel, copy_kernel}) {
    cudaFuncAttributes attributes{};
    HYLE_RETURN_IF_ERROR(runtime_status(cudaFuncGetAttributes(&attributes, kernel)));
    if (attributes.maxThreadsPerBlock < 256)
      return Status::Unimplemented("CUDA indexed-copy kernel has insufficient block capacity");
  }
  auto *storage = allocator.allocate(sizeof(IndexedCopyProgram), alignof(IndexedCopyProgram));
  if (!storage)
    return Status::ResourceExhausted("CUDA indexed-copy program metadata");
  auto *program = std::construct_at(static_cast<IndexedCopyProgram *>(storage));
  program->context = context;
  program->check_kernel = check_kernel; program->copy_kernel = copy_kernel;
  program->layout = {spec.slot_count, spec.row_count, spec.row_bytes, spec.slot_stride};
  program->capacity_bytes = std::size_t(spec.slot_count - 1) * spec.slot_stride + spec.row_bytes;
  program->input_bytes = std::size_t(spec.row_count) * spec.row_bytes;
  program->indices_bytes = std::size_t(spec.row_count) * sizeof(std::int32_t);
  *output = program;
  return Status::Ok();
}

void destroy_copy(void *opaque, mem::AllocatorRef allocator) noexcept {
  auto *program = static_cast<IndexedCopyProgram *>(opaque);
  if (!program)
    return;
  std::destroy_at(program);
  allocator.deallocate(program, sizeof(IndexedCopyProgram), alignof(IndexedCopyProgram));
}

Status validate_carrier(ContextState *context, mem::ConstBufferView view,
                        std::size_t required, std::size_t alignment) noexcept {
  const auto *resource = resource_const(view.resource());
  if (!view.valid() || !resource || resource->context() != context ||
      resource->kind() == MemoryKind::kPinnedHost || view.size() < required)
    return Status::InvalidArgument("CUDA indexed-copy carrier context/capacity is invalid");
  const auto address = reinterpret_cast<std::uintptr_t>(byte_pointer(view));
  if (!address || address % alignment != 0 || view.size() > SIZE_MAX - address)
    return Status::InvalidArgument("CUDA indexed-copy carrier address is invalid");
  return Status::Ok();
}

EnqueueResult enqueue_copy(void *opaque, void *queue_opaque, void *program_opaque,
                           IndexedCopyNativeBindings bindings) noexcept {
  auto *queue = static_cast<QueueState *>(queue_opaque);
  auto *program = static_cast<IndexedCopyProgram *>(program_opaque);
  if (!queue || !program || queue->context != opaque || program->context != opaque ||
      !queue->active_sequence)
    return EnqueueResult::rejected(Status::InvalidArgument(
        "CUDA indexed-copy needs its original admitted context/queue"));
  auto checked = validate_carrier(queue->context, bindings.capacity, program->capacity_bytes, 1);
  if (!checked.ok()) return EnqueueResult::rejected(std::move(checked));
  checked = validate_carrier(queue->context, bindings.rows, program->input_bytes, 1);
  if (!checked.ok()) return EnqueueResult::rejected(std::move(checked));
  checked = validate_carrier(queue->context, bindings.indices, program->indices_bytes, alignof(std::int32_t));
  if (!checked.ok()) return EnqueueResult::rejected(std::move(checked));
  ScopedPrimaryContext scope(queue->context->primary);
  if (!scope) return EnqueueResult::rejected(scope.status());
  unsigned *word = nullptr;
  auto initialized = acquire_device_status(queue, &word);
  if (!initialized.status.ok()) return initialized;
  // All rejecting host checks precede any native prefix. From status-word
  // initialization onward the original Completion owns errors and retirement.
  return EnqueueResult::accepted_command(runtime_status(launch_indexed_copy(
      program->check_kernel, program->copy_kernel,
      const_cast<void *>(byte_pointer(bindings.capacity)), byte_pointer(bindings.rows),
      reinterpret_cast<const std::int32_t *>(byte_pointer(bindings.indices)),
      program->layout, word, queue->stream)));
}

const compute::detail::IndexedCopyBackendVTable table{
    &prepare_copy, &destroy_copy, &enqueue_copy, true};
} // namespace

const compute::detail::IndexedCopyBackendVTable *indexed_copy_interface() noexcept {
  return &table;
}
} // namespace hyle::compute::cuda::detail
