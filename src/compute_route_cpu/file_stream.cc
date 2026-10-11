#include "hyle/compute/cpu.h"

#include <vector>

#include "hyle/compute/detail/copy_stream_access.h"
#include "hyle/compute/detail/file_stream_copy.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::cpu {
Result<FileResourcePipeline> create_file_pipeline(io::IoContext &io, const Context &context,
    Queue &queue, mem::MemoryResourceRef staging, std::span<const CopyResourceSpec> targets,
    FileStreamOptions options) noexcept {
  if (!context.valid() || !queue.valid() || context.device() != device() || queue.device() != context.device())
    return std::unexpected(Status::InvalidArgument("CPU file pipeline requires its original CPU Context/Queue"));
  if (!options.allocator.valid() || targets.size() >= UINT32_MAX || !options.in_flight_slots)
    return std::unexpected(Status::InvalidArgument("invalid CPU stream capacities/allocator"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) { HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator()); }
  std::vector<CopyResourceSpec, mem::StdAllocator<CopyResourceSpec>> resources{mem::StdAllocator<CopyResourceSpec>(allocator)};
  try {
    resources.reserve(targets.size() + 1);
    resources.push_back({staging, options.staging_bytes_per_slot}); resources.insert(resources.end(), targets.begin(), targets.end());
  } catch (...) { return std::unexpected(Status::ResourceExhausted("CPU stream resource catalog")); }
  HYLE_ASSIGN_OR_RETURN(auto program, CopyBatchProgram::prepare(context, resources,
      {.max_resources = static_cast<std::uint32_t>(resources.size()), .max_copies = 1, .max_phases = 1,
       .max_bytes = options.staging_bytes_per_slot, .allocator = allocator, .resource_budget = options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(auto endpoint, (detail::make_file_copy_endpoint<CopyBatchProgram, CopyBatchInvocation,
      CopyBatchSubmission, detail::CopyStreamAccess>(std::move(program), resources, options,
        [&queue](const CopyBatchProgram &program) noexcept { return CopyBatchInvocation::create(program, queue); })));
  return detail::FileStreamAccess::create(io, staging, targets, std::move(options), std::move(endpoint));
}
} // namespace hyle::compute::cpu
