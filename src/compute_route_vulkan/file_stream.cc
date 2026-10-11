#include "hyle/compute/vulkan_memory.h"

#include "vulkan_file_stream_access.h"
#include "hyle/compute/detail/file_stream_copy.h"

namespace hyle::compute::vulkan {
Result<FileResourcePipeline> create_file_pipeline(io::IoContext &io, const Context &context,
    mem::MemoryResourceRef staging, std::span<const CopyResourceSpec> targets, FileStreamOptions options) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto endpoint, (compute::detail::prepare_file_copy_endpoint<CopyBatchProgram,
      CopyBatchInvocation, CopyBatchSubmission, detail::CopyStreamAccess>(staging, targets, options,
        [&context](std::span<const CopyResourceSpec> resources, CopyBatchOptions options) noexcept {
          return CopyBatchProgram::prepare(context, resources, std::move(options)); },
        [](const CopyBatchProgram &program) noexcept { return CopyBatchInvocation::create(program); })));
  return compute::detail::FileStreamAccess::create(io, staging, targets, std::move(options), std::move(endpoint));
}
} // namespace hyle::compute::vulkan
