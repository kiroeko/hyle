#include "hyle/compute/cuda.h"

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/copy_stream_access.h"
#include "hyle/compute/detail/file_stream_copy.h"

namespace hyle::compute::cuda {
Result<FileResourcePipeline> create_file_pipeline(io::IoContext &io, const Context &context,
    Queue &queue, mem::MemoryResourceRef staging, std::span<const CopyResourceSpec> targets,
    FileStreamOptions options) noexcept {
  if (!context.valid() || !queue.valid() || context.device().backend_id() != kBackendId ||
      queue.device() != context.device() || !staging.properties().has(mem::MemoryProperty::kHostPinned))
    return std::unexpected(Status::InvalidArgument("CUDA file pipeline requires its original Context/Queue and pinned staging"));
  HYLE_ASSIGN_OR_RETURN(auto endpoint, (compute::detail::prepare_file_copy_endpoint<CopyBatchProgram,
      CopyBatchInvocation, CopyBatchSubmission, compute::detail::CopyStreamAccess>(staging, targets, options,
        [&context](std::span<const CopyResourceSpec> resources, CopyBatchOptions options) noexcept {
          return CopyBatchProgram::prepare(context, resources, std::move(options)); },
        [&queue](const CopyBatchProgram &program) noexcept { return CopyBatchInvocation::create(program, queue); })));
  return compute::detail::FileStreamAccess::create(io, staging, targets, std::move(options), std::move(endpoint));
}
} // namespace hyle::compute::cuda
