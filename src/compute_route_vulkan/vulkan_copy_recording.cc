#include "vulkan_copy_internal.h"
#include "vulkan_transfer_internal.h"
#include "vulkan_program_internal.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>

// These are the original locked ggml copy shader, already linked by the
// managed Vulkan runtime. No conversion flags or floating arithmetic occur
// in this same-type shader. Copy into aligned words before pipeline creation.
extern const std::uint64_t contig_cpy_f16_f16_len;
extern const unsigned char contig_cpy_f16_f16_data[];

namespace hyle::compute::vulkan::detail {
template<class T> using CopyArray = std::vector<T, mem::StdAllocator<T>>;
struct CopyPair final {
  std::shared_ptr<Buffer> source, destination;
  std::uint64_t source_key = 0, destination_key = 0;
  VkBufferCopy region{};
  std::size_t input = 0;
};
struct HalfwordCopy final {
  VkDescriptorSet descriptor = VK_NULL_HANDLE;
  std::array<VkDescriptorBufferInfo, 2> buffers{};
  // Original generic_unary_head.glsl push layout. Only ne (word 0) and
  // misalign_offsets (word 17) survive this contiguous-copy specialization.
  std::array<std::uint32_t, 32> push{};
  std::array<std::uint32_t, 3> groups{};
  bool active = false;
};
struct CopyRecording final {
  explicit CopyRecording(mem::AllocatorRef allocator)
      : physical(mem::StdAllocator<cd::PhysicalCopyResource>(allocator)),
        inputs(mem::StdAllocator<CopyRange>(allocator)), intervals(mem::StdAllocator<cd::CopyInterval>(allocator)),
        resolved(mem::StdAllocator<cd::ResolvedCopyRange>(allocator)),
        pairs(mem::StdAllocator<CopyPair>(allocator)), regions(mem::StdAllocator<VkBufferCopy>(allocator)),
        halfwords(mem::StdAllocator<HalfwordCopy>(allocator)) {}
  std::shared_ptr<Runtime> runtime;
  std::shared_ptr<Run> run;
  CopyBatchOptions limits;
  CopyBatchReport report;
  CopyRecordingMode mode = CopyRecordingMode::kOnce;
  Status replay_failure;
  bool prepared = false;
  bool prefix_marker = false;
  CopyArray<cd::PhysicalCopyResource> physical;
  CopyArray<CopyRange> inputs;
  CopyArray<cd::CopyInterval> intervals;
  CopyArray<cd::ResolvedCopyRange> resolved;
  CopyArray<CopyPair> pairs;
  CopyArray<VkBufferCopy> regions;
  CopyArray<HalfwordCopy> halfwords;
  std::shared_ptr<Pipeline> halfword_pipeline;
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
  if (!recording.replay_failure.ok()) return recording.replay_failure;
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
Status prepare_halfwords(CopyRecording &recording, std::uint32_t capacity,
                        mem::AllocatorRef allocator) {
  auto &device = *recording.runtime->device;
  if (!device.provider.native().storage_buffer16 ||
      device.properties.limits.maxComputeWorkGroupInvocations < 128 ||
      device.properties.limits.maxComputeWorkGroupSize[0] < 128)
    return Status::Unimplemented("Vulkan exact halfword copy needs enabled storage16 and 128 lanes");
  if (contig_cpy_f16_f16_len % 4 || contig_cpy_f16_f16_len > SIZE_MAX)
    return Status::Internal("Vulkan locked halfword copy shader size");
  CopyArray<std::uint32_t> code{mem::StdAllocator<std::uint32_t>(allocator)};
  code.resize(static_cast<std::size_t>(contig_cpy_f16_f16_len / 4));
  std::memcpy(code.data(), contig_cpy_f16_f16_data, static_cast<std::size_t>(contig_cpy_f16_f16_len));
  HYLE_ASSIGN_OR_RETURN(recording.halfword_pipeline, create_pipeline(recording.runtime->device,
      {code, 2, sizeof(HalfwordCopy::push), {}}));
  // A submitted Completion retains Run, not this reusable recording wrapper.
  // Keep every native command dependency alive until actual Run retirement.
  recording.run->recording_owner = recording.halfword_pipeline;
  const auto descriptors = capacity * 2;
  const bool rebindable = device.environment.storage_update_after_bind;
  if (rebindable) {
    const auto limit = device.descriptor_properties.maxUpdateAfterBindDescriptorsInAllPools;
    auto current = device.descriptors_in_pools.load(std::memory_order_relaxed);
    do {
      if (descriptors > limit || current > limit - descriptors)
        return Status::ResourceExhausted("Vulkan halfword descriptor budget");
    } while (!device.descriptors_in_pools.compare_exchange_weak(
        current, current + descriptors, std::memory_order_relaxed));
    recording.run->descriptor_count = descriptors;
  }
  VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptors};
  VkDescriptorPoolCreateInfo pool{}; pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool.flags = rebindable ? VkDescriptorPoolCreateFlags{VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT} : 0u;
  pool.maxSets = capacity; pool.poolSizeCount = 1; pool.pPoolSizes = &size;
  HYLE_RETURN_IF_ERROR(native_status(device.fn.CreateDescriptorPool(device.native.device, &pool,
      nullptr, &recording.run->descriptor_pool), "vkCreateDescriptorPool(halfword copy)"));
  recording.halfwords.resize(capacity);
  for (auto &copy : recording.halfwords) {
    VkDescriptorSetAllocateInfo allocate{}; allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = recording.run->descriptor_pool; allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &recording.halfword_pipeline->descriptor_layout;
    HYLE_RETURN_IF_ERROR(native_status(device.fn.AllocateDescriptorSets(device.native.device,
        &allocate, &copy.descriptor), "vkAllocateDescriptorSets(halfword copy)"));
  }
  return Status::Ok();
}
Status halfword_range(const Device &device, const CopyPair &pair, HalfwordCopy &copy) noexcept {
  copy.active = pair.region.srcOffset % 4 || pair.region.dstOffset % 4 || pair.region.size % 4;
  if (!copy.active) return Status::Ok();
  // Staging resources have TRANSFER usage only. Only original device buffers
  // (including generated results) have the required STORAGE_BUFFER usage.
  if (pair.source->mapped || pair.destination->mapped)
    return Status::Unimplemented("Vulkan halfword copy requires device storage buffers");
  const auto alignment = (std::max)(VkDeviceSize{4}, device.properties.limits.minStorageBufferOffsetAlignment);
  const std::array offsets{pair.region.srcOffset, pair.region.dstOffset};
  const std::array buffers{pair.source.get(), pair.destination.get()};
  std::uint32_t packed = 0;
  for (std::size_t port = 0; port < 2; ++port) {
    const auto prefix = offsets[port] % alignment;
    if (prefix % 2 || prefix / 2 > UINT16_MAX || prefix > device.properties.limits.maxStorageBufferRange ||
        pair.region.size > device.properties.limits.maxStorageBufferRange - prefix)
      return Status::Unimplemented("Vulkan halfword copy descriptor origin/range exceeds domain");
    copy.buffers[port] = {buffers[port]->handle, offsets[port] - prefix, prefix + pair.region.size};
    packed |= static_cast<std::uint32_t>(prefix / 2) << (port == 0 ? 16 : 0);
  }
  const auto elements = pair.region.size / 2;
  if (elements > UINT32_MAX)
    return Status::Unimplemented("Vulkan halfword copy exceeds shader index range");
  copy.push = {}; copy.push[0] = static_cast<std::uint32_t>(elements); copy.push[17] = packed;
  copy.groups = {1, static_cast<std::uint32_t>((std::min)(VkDeviceSize{512}, (elements + 511) / 512)),
      static_cast<std::uint32_t>((elements + 262143) / 262144)};
  for (std::size_t axis = 0; axis < 3; ++axis)
    if (!copy.groups[axis] || copy.groups[axis] > device.properties.limits.maxComputeWorkGroupCount[axis])
      return Status::Unimplemented("Vulkan halfword copy dispatch exceeds device limits");
  return Status::Ok();
}
} // namespace

Result<std::shared_ptr<CopyRecording>> prepare_copy_recording(const Context &context,
    std::uint32_t capacity, mem::AllocatorRef allocator) noexcept {
  if (!context.valid()) return std::unexpected(Status::InvalidArgument("Vulkan copy Context is invalid"));
  return prepare_copy_recording_runtime(Access::runtime(context), capacity, allocator);
}
Result<std::shared_ptr<CopyRecording>> prepare_copy_recording_runtime(const std::shared_ptr<Runtime> &runtime,
    std::uint32_t capacity, mem::AllocatorRef allocator, std::size_t max_bytes,
    CopyRecordingMode mode) noexcept {
  return prepare_copy_recording_runtime(runtime, capacity, allocator, max_bytes, mode, CopyGranularity::kWord);
}
Result<std::shared_ptr<CopyRecording>> prepare_copy_recording_runtime(const std::shared_ptr<Runtime> &runtime,
    std::uint32_t capacity, mem::AllocatorRef allocator, std::size_t max_bytes,
    CopyRecordingMode mode, CopyGranularity granularity) noexcept {
  if (!runtime || !allocator.valid() || capacity == 0 || capacity > UINT32_MAX / 2 || max_bytes == 0)
    return std::unexpected(Status::InvalidArgument("Vulkan copy capacity/allocator is invalid"));
  std::lock_guard lock(runtime->mutex);
  if (runtime->closed) return std::unexpected(Status::FailedPrecondition("Vulkan copy Context is closed"));
  if (!runtime->failure.ok()) return std::unexpected(runtime->failure);
  try {
    auto recording = std::allocate_shared<CopyRecording>(mem::StdAllocator<CopyRecording>(allocator), allocator);
    recording->runtime = runtime;
    recording->mode = mode;
    recording->limits.max_resources = capacity * 2; recording->limits.max_copies = capacity;
    recording->limits.max_phases = 1;
    recording->limits.max_bytes = max_bytes;
    recording->physical.resize(static_cast<std::size_t>(capacity) * 2);
    recording->inputs.resize(capacity); recording->intervals.resize(static_cast<std::size_t>(capacity) * 2);
    recording->resolved.resize(capacity); recording->pairs.resize(capacity); recording->regions.resize(capacity);
    HYLE_ASSIGN_OR_RETURN(recording->run, create_run(runtime, allocator));
    recording->run->retained_buffers.resize(static_cast<std::size_t>(capacity) * 2);
    HYLE_RETURN_IF_ERROR(prepare_transfer_run(*recording->run));
    if (granularity == CopyGranularity::kHalfword)
      HYLE_RETURN_IF_ERROR(prepare_halfwords(*recording, capacity, allocator));
    return recording;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan copy cold metadata"));
  }
}
Status bind_copy_recording(CopyRecording &recording,
    std::span<const cd::CopyBackendRange> ranges, bool host_upload_ready, bool prefix_marker,
    ResultCopySourceFn resolve_source,void*source_owner) noexcept {
  std::lock_guard lock(recording.runtime->mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(recording));
  recording.prepared = false;
  recording.prefix_marker = prefix_marker;
  if (ranges.size() > recording.limits.max_copies)
    return Status::ResourceExhausted("Vulkan copy range capacity exhausted");
  for (auto &pair : recording.pairs) pair = {};
  for (auto &buffer : recording.run->retained_buffers) buffer.reset();
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    if (!resolve_source && ranges[index].source.size() != ranges[index].destination.size())
      return Status::InvalidArgument("Vulkan copy range sizes differ");
    HYLE_ASSIGN_OR_RETURN(auto source,resolve_source?resolve_source(source_owner,index):
        MemoryAccess::copy_span(recording.runtime,ranges[index].source));
    HYLE_ASSIGN_OR_RETURN(auto destination, MemoryAccess::copy_span(recording.runtime, ranges[index].destination));
    recording.physical[2 * index] = source.physical;
    recording.physical[2 * index + 1] = destination.physical;
    if (recording.halfword_pipeline) {
      recording.physical[2 * index].copy_alignment = 2;
      recording.physical[2 * index + 1].copy_alignment = 2;
    }
    const auto id = static_cast<std::uint32_t>(index);
    if(source.physical.capacity!=ranges[index].destination.size())return Status::InvalidArgument("Vulkan native source size differs");
    recording.inputs[index] = {{2 * id}, {2 * id + 1}, {0}, 0, 0, ranges[index].destination.size()};
    recording.pairs[index] = {std::move(source.backing), std::move(destination.backing), 0, 0,
        {source.offset, destination.offset, ranges[index].destination.size()}, index};
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
  // Validate every shader descriptor/dispatch before recording any data
  // command. The exact shared interval validator above still owns alias proof.
  for (std::size_t i = 0; i < active.size() && recording.halfword_pipeline; ++i) {
    HYLE_RETURN_IF_ERROR(halfword_range(device, active[i], recording.halfwords[i]));
    if (recording.halfwords[i].active) run.transfer_submission = false;
  }
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
  begin.flags = recording.mode == CopyRecordingMode::kOnce
      ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0;
  HYLE_RETURN_IF_ERROR(native_status(device.fn.BeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(batch copy)"));
  for (std::size_t index = 0; index < active.size(); ++index) {
    const auto &pair = active[index];
    const bool halfword = recording.halfword_pipeline && recording.halfwords[index].active;
    barrier(device, command, *pair.source, pair.region.srcOffset, pair.region.size,
        run.transfer_submission ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        run.transfer_submission ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_MEMORY_WRITE_BIT,
        halfword ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
        halfword ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT);
    if (!run.transfer_submission)
      barrier(device, command, *pair.destination, pair.region.dstOffset, pair.region.size,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
          halfword ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
          halfword ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT);
    run.retained_buffers[2 * index] = pair.source; run.retained_buffers[2 * index + 1] = pair.destination;
  }
  for (std::size_t first = 0; first < active.size();) {
    if (recording.halfword_pipeline && recording.halfwords[first].active) {
      const auto &copy = recording.halfwords[first]; const auto &pipeline = *recording.halfword_pipeline;
      std::array<VkWriteDescriptorSet, 2> writes{};
      for (std::uint32_t port = 0; port < writes.size(); ++port) {
        writes[port].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[port].dstSet = copy.descriptor; writes[port].dstBinding = port;
        writes[port].descriptorCount = 1; writes[port].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[port].pBufferInfo = &copy.buffers[port];
      }
      device.fn.UpdateDescriptorSets(device.native.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
      device.fn.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
      device.fn.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &copy.descriptor, 0, nullptr);
      device.fn.CmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
          static_cast<std::uint32_t>(sizeof(copy.push)), copy.push.data());
      device.fn.CmdDispatch(command, copy.groups[0], copy.groups[1], copy.groups[2]);
      ++recording.report.native_calls; ++first; continue;
    }
    auto last = first + 1;
    while (last < active.size() && active[last].source_key == active[first].source_key &&
        active[last].destination_key == active[first].destination_key &&
        !(recording.halfword_pipeline && recording.halfwords[last].active)) ++last;
    device.fn.CmdCopyBuffer(command, active[first].source->handle, active[first].destination->handle,
        static_cast<std::uint32_t>(last - first), recording.regions.data() + first);
    ++recording.report.native_calls; first = last;
  }
  if (!run.transfer_submission) {
    for (std::size_t index = 0; index < active.size(); ++index) {
      const auto &pair = active[index];
      const bool halfword = recording.halfword_pipeline && recording.halfwords[index].active;
      barrier(device, command, *pair.destination, pair.region.dstOffset, pair.region.size,
          halfword ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
          halfword ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    }
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
  if (recording.mode == CopyRecordingMode::kOnce) recording.prepared = false;
  if (recording.report.active_copies == 0 && !recording.prefix_marker) return Completion::completed();
  // Native retirement may release episode pins. The recording keeps the real
  // backing owners, and these pre-sized slots need no allocation on replay.
  for (std::size_t index = 0; index < recording.report.active_copies; ++index) {
    recording.run->retained_buffers[2 * index] = recording.pairs[index].source;
    recording.run->retained_buffers[2 * index + 1] = recording.pairs[index].destination;
  }
  auto completion = submit_locked(recording.run);
  if (completion) ++recording.report.phase_completions;
  else if (recording.mode == CopyRecordingMode::kReplayable)
    recording.replay_failure = completion.error();
  return completion;
}
CopyBatchReport copy_recording_report(const CopyRecording &recording) noexcept { return recording.report; }
Status copy_recording_reuse_gate(const CopyRecording &recording) noexcept {
  std::lock_guard lock(recording.runtime->mutex);
  return mutable_recording(recording);
}
Status copy_recording_terminal_hint(const CopyRecording &recording) noexcept {
  std::lock_guard lock(recording.runtime->mutex);
  if (!recording.replay_failure.ok()) return recording.replay_failure;
  return recording.run->submission_failure;
}
Status copy_recording_write_status(const CopyRecording &recording) noexcept {
  std::lock_guard lock(recording.runtime->mutex);
  if (!recording.replay_failure.ok()) return recording.replay_failure;
  const auto &run = *recording.run;
  HYLE_RETURN_IF_ERROR(run.submission_failure);
  HYLE_RETURN_IF_ERROR(run.observation_error);
  HYLE_RETURN_IF_ERROR(run.terminal);
  HYLE_RETURN_IF_ERROR(recording.runtime->failure);
  return run.pending && run.cancelled ? Status::Cancelled("Vulkan append writer was cancelled") : Status::Ok();
}
Status copy_recording_retired_gate(const CopyRecording &recording) noexcept {
  std::lock_guard lock(recording.runtime->mutex);
  const auto &run = *recording.run;
  return run.pending || run.completion_owner || run.observers || run.retiring
      ? Status::Unavailable("Vulkan append recording still has actual work") : Status::Ok();
}
Result<CompletionPoll> poll_copy_recording(const CopyRecording &recording) noexcept { return poll_native(recording.run); }
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
