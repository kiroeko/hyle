#include "hyle/compute/cpu_file_stream.h"

#include "hyle/compute/detail/copy_stream_access.h"
#include "hyle/compute/detail/file_stream_copy.h"

namespace hyle::compute::cpu {
Result<FileResourcePipeline> create_file_pipeline(io::IoContext &io, const Context &context,
    Queue &queue, mem::MemoryResourceRef staging, std::span<const CopyResourceSpec> targets,
    FileStreamOptions options) noexcept {
  if (!context.valid() || !queue.valid() || context.device() != device() || queue.device() != context.device())
    return std::unexpected(Status::InvalidArgument("CPU file pipeline requires its original CPU Context/Queue"));
  HYLE_ASSIGN_OR_RETURN(auto endpoint, (detail::prepare_file_copy_endpoint<CopyBatchProgram, CopyBatchInvocation,
      CopyBatchSubmission, detail::CopyStreamAccess>(staging, targets, options,
        [&context](std::span<const CopyResourceSpec> resources, CopyBatchOptions options) noexcept {
          return CopyBatchProgram::prepare(context, resources, std::move(options)); },
        [&queue](const CopyBatchProgram &program) noexcept { return CopyBatchInvocation::create(program, queue); })));
  return detail::FileStreamAccess::create(io, staging, targets, std::move(options), std::move(endpoint));
}
} // namespace hyle::compute::cpu
