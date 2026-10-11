#include "vulkan_copy_internal.h"
#include "vulkan_transfer_internal.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>

namespace hyle::compute::vulkan::detail {
template<class T> using CopyArray = std::vector<T, mem::StdAllocator<T>>;
struct CopyPair final {
  std::shared_ptr<Buffer> source, destination;
  std::uint64_t source_key = 0, destination_key = 0;
  VkBufferCopy region{};
  std::size_t input = 0;
};
struct CopyRecording final {
  explicit CopyRecording(mem::AllocatorRef allocator)
      : physical(mem::StdAllocator<cd::PhysicalCopyResource>(allocator)),
        inputs(mem::StdAllocator<CopyRange>(allocator)), intervals(mem::StdAllocator<cd::CopyInterval>(allocator)),
        resolved(mem::StdAllocator<cd::ResolvedCopyRange>(allocator)),
        pairs(mem::StdAllocator<CopyPair>(allocator)), regions(mem::StdAllocator<VkBufferCopy>(allocator)) {}
  std::shared_ptr<Runtime> runtime;
  std::shared_ptr<Run> run;
  CopyBatchOptions limits;
  CopyBatchReport report;
  bool prepared = false;
  bool prefix_marker = false;
  CopyArray<cd::PhysicalCopyResource> physical;
  CopyArray<CopyRange> inputs;
  CopyArray<cd::CopyInterval> intervals;
  CopyArray<cd::ResolvedCopyRange> resolved;
  CopyArray<CopyPair> pairs;
  CopyArray<VkBufferCopy> regions;
  std::array<std::uint64_t, 1> closure{};
};
namespace {
std::uint64_t key(VkBuffer buffer) noexcept {
  std::uint64_t result = 0;
  static_assert(sizeof(buffer) <= sizeof(result));
  std::memcpy(&result, &buffer, sizeof(buffer));
  return result;
}
Status mutable_recording(const CopyRecording &recording) noexcept {
  if (recording.runtime->closed) return Status::FailedPrecondition("Vulkan copy Context is closed");
  if (!recording.runtime->failure.ok()) return recording.runtime->failure;
  const auto &run = *recording.run;
  if (run.pending || run.completion_owner || run.observers || run.retiring)
    return Status::Unavailable("Vulkan copy phase is retained");
  if (!run.terminal.ok()) return run.terminal;
  if (!run.submission_failure.ok()) return run.submission_failure;
  return Status::Ok();
}
void barrier(const Device &device, VkCommandBuffer command, const Buffer &buffer,
             VkDeviceSize offset, VkDeviceSize bytes, VkPipelineStageFlags source_stage,
             VkAccessFlags source_access, VkPipelineStageFlags destination_stage,
             VkAccessFlags destination_access) noexcept {
  VkBufferMemoryBarrier range{}; range.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  range.srcQueueFamilyIndex = range.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  range.buffer = buffer.handle; range.offset = offset; range.size = bytes;
  range.srcAccessMask = source_access; range.dstAccessMask = destination_access;
  device.fn.CmdPipelineBarrier(command, source_stage, destination_stage, 0,
      0, nullptr, 1, &range, 0, nullptr);
}
} // namespace

Result<std::shared_ptr<CopyRecording>> prepare_copy_recording(const Context &context,
    std::uint32_t capacity, mem::AllocatorRef allocator) noexcept {
  if (!context.valid()) return std::unexpected(Status::InvalidArgument("Vulkan copy Context is invalid"));
  return prepare_copy_recording_runtime(Access::runtime(context), capacity, allocator);
}
Result<std::shared_ptr<CopyRecording>> prepare_copy_recording_runtime(const std::shared_ptr<Runtime> &runtime,
    std::uint32_t capacity, mem::AllocatorRef allocator, std::size_t max_bytes) noexcept {
  if (!runtime || !allocator.valid() || capacity == 0 || capacity > UINT32_MAX / 2 || max_bytes == 0)
    return std::unexpected(Status::InvalidArgument("Vulkan copy capacity/allocator is invalid"));
  std::lock_guard lock(runtime->mutex);
  if (runtime->closed) return std::unexpected(Status::FailedPrecondition("Vulkan copy Context is closed"));
  if (!runtime->failure.ok()) return std::unexpected(runtime->failure);
  try {
    auto recording = std::allocate_shared<CopyRecording>(mem::StdAllocator<CopyRecording>(allocator), allocator);
    recording->runtime = runtime;
    recording->limits.max_resources = capacity * 2; recording->limits.max_copies = capacity;
    recording->limits.max_phases = 1;
    recording->limits.max_bytes = max_bytes;
    recording->physical.resize(static_cast<std::size_t>(capacity) * 2);
    recording->inputs.resize(capacity); recording->intervals.resize(static_cast<std::size_t>(capacity) * 2);
    recording->resolved.resize(capacity); recording->pairs.resize(capacity); recording->regions.resize(capacity);
    HYLE_ASSIGN_OR_RETURN(recording->run, create_run(runtime, allocator));
    recording->run->retained_buffers.resize(static_cast<std::size_t>(capacity) * 2);
    HYLE_RETURN_IF_ERROR(prepare_transfer_run(*recording->run));
    return recording;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan copy cold metadata"));
  }
}
Status bind_copy_recording(CopyRecording &recording,
    std::span<const cd::CopyBackendRange> ranges, bool host_upload_ready, bool prefix_marker) noexcept {
  std::lock_guard lock(recording.runtime->mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(recording));
  recording.prepared = false;
  recording.prefix_marker = prefix_marker;
  if (ranges.size() > recording.limits.max_copies)
    return Status::ResourceExhausted("Vulkan copy range capacity exhausted");
  for (auto &pair : recording.pairs) pair = {};
  for (auto &buffer : recording.run->retained_buffers) buffer.reset();
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    if (ranges[index].source.size() != ranges[index].destination.size())
      return Status::InvalidArgument("Vulkan copy range sizes differ");
    HYLE_ASSIGN_OR_RETURN(auto source, MemoryAccess::copy_span(recording.runtime, ranges[index].source));
    HYLE_ASSIGN_OR_RETURN(auto destination, MemoryAccess::copy_span(recording.runtime, ranges[index].destination));
    recording.physical[2 * index] = source.physical;
    recording.physical[2 * index + 1] = destination.physical;
    const auto id = static_cast<std::uint32_t>(index);
    recording.inputs[index] = {{2 * id}, {2 * id + 1}, {0}, 0, 0, ranges[index].source.size()};
    recording.pairs[index] = {std::move(source.backing), std::move(destination.backing), 0, 0,
        {source.offset, destination.offset, ranges[index].source.size()}, index};
  }
  const std::array<CopyPhase, 1> phase{{{}}};
  HYLE_ASSIGN_OR_RETURN(auto report, cd::validate_copy_ranges(
      std::span(recording.physical).first(ranges.size() * 2), phase,
      std::span(recording.inputs).first(ranges.size()), recording.limits,
      {recording.intervals, recording.resolved, recording.closure}));
  // Compaction follows the validator's stable input indices, before sorting
  // by actual native buffer pair. No pairwise overlap scan or hot allocation.
  for (std::size_t index = 0; index < report.active_copies; ++index) {
    const auto original = recording.resolved[index].input_index;
    if (index != original) recording.pairs[index] = std::move(recording.pairs[original]);
    recording.pairs[index].source_key = key(recording.pairs[index].source->handle);
    recording.pairs[index].destination_key = key(recording.pairs[index].destination->handle);
  }
  auto active = std::span(recording.pairs).first(report.active_copies);
  std::sort(active.begin(), active.end(), [](const auto &left, const auto &right) noexcept {
    return std::tie(left.source_key, left.destination_key, left.region.srcOffset) <
           std::tie(right.source_key, right.destination_key, right.region.srcOffset);
  });
  for (std::size_t index = 0; index < active.size(); ++index) recording.regions[index] = active[index].region;
  auto &run = *recording.run; const auto &device = *recording.runtime->device;
  run.transfer_submission = host_upload_ready && run.transfer_command != VK_NULL_HANDLE && !active.empty();
  for (const auto &pair : active)
    run.transfer_submission = run.transfer_submission && pair.source->mapped != nullptr &&
        pair.destination->mapped == nullptr && pair.source->transfer_eligible && pair.destination->transfer_eligible;
  report.native_metadata_unknown = true; report.native_calls = 0; report.phase_completions = 0;
  recording.report = report;
  if (active.empty() && !prefix_marker) { recording.prepared = true; return Status::Ok(); }
  // Visibility failure precedes command-buffer recording or native acceptance.
  for (const auto &pair : active) {
    if (host_upload_ready && pair.source->mapped != nullptr) {
      const auto source = ranges[pair.input].source;
      HYLE_RETURN_IF_ERROR(source.resource().resource().flush(source.native_handle(), source.offset(), source.size()));
    }
  }
  const auto command = run.transfer_submission ? run.transfer_command : run.command;
  VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  HYLE_RETURN_IF_ERROR(native_status(device.fn.BeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(batch copy)"));
  for (std::size_t index = 0; index < active.size(); ++index) {
    const auto &pair = active[index];
    barrier(device, command, *pair.source, pair.region.srcOffset, pair.region.size,
        run.transfer_submission ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        run.transfer_submission ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    if (!run.transfer_submission)
      barrier(device, command, *pair.destination, pair.region.dstOffset, pair.region.size,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    run.retained_buffers[2 * index] = pair.source; run.retained_buffers[2 * index + 1] = pair.destination;
  }
  for (std::size_t first = 0; first < active.size();) {
    auto last = first + 1;
    while (last < active.size() && active[last].source_key == active[first].source_key &&
        active[last].destination_key == active[first].destination_key) ++last;
    device.fn.CmdCopyBuffer(command, active[first].source->handle, active[first].destination->handle,
        static_cast<std::uint32_t>(last - first), recording.regions.data() + first);
    ++recording.report.native_calls; first = last;
  }
  if (!run.transfer_submission) {
    for (const auto &pair : active)
      barrier(device, command, *pair.destination, pair.region.dstOffset, pair.region.size,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  }
  HYLE_RETURN_IF_ERROR(native_status(device.fn.EndCommandBuffer(command), "vkEndCommandBuffer(batch copy)"));
  if (run.transfer_submission) {
    HYLE_RETURN_IF_ERROR(native_status(device.fn.BeginCommandBuffer(run.command, &begin), "vkBeginCommandBuffer(batch handoff)"));
    HYLE_RETURN_IF_ERROR(native_status(device.fn.EndCommandBuffer(run.command), "vkEndCommandBuffer(batch handoff)"));
  }
  recording.prepared = true;
  return Status::Ok();
}
Result<Completion> submit_copy_recording(CopyRecording &recording) noexcept {
  std::lock_guard lock(recording.runtime->mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(recording));
  if (!recording.prepared) return std::unexpected(Status::FailedPrecondition("Vulkan copy ranges are not prepared"));
  recording.prepared = false;
  if (recording.report.active_copies == 0 && !recording.prefix_marker) return Completion::completed();
  auto completion = submit_locked(recording.run);
  if (completion) ++recording.report.phase_completions;
  return completion;
}
CopyBatchReport copy_recording_report(const CopyRecording &recording) noexcept { return recording.report; }
Status copy_recording_terminal_hint(const CopyRecording &recording) noexcept {
  std::lock_guard lock(recording.runtime->mutex);
  return recording.run->submission_failure;
}
Status clear_copy_recording(CopyRecording &recording) noexcept {
  std::lock_guard lock(recording.runtime->mutex);
  const auto &run = *recording.run;
  if (run.pending || run.completion_owner || run.observers || run.retiring)
    return Status::Unavailable("Vulkan copy recording still has actual work");
  for (auto &pair : recording.pairs) pair = {};
  for (auto &buffer : recording.run->retained_buffers) buffer.reset();
  recording.prepared = false;
  return Status::Ok();
}
} // namespace hyle::compute::vulkan::detail
