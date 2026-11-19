// Included after native_runtime_impl.h inside the locked Vulkan translation
// unit. Graph ownership/lowering lives above this native completion seam.
#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>

namespace hyle::compute::ggml_vulkan::bridge {
struct Work {
  Runtime *runtime = nullptr;
  ggml_backend *backend = nullptr;
  vk::Fence fence; // Borrowed from this Work's exclusive native backend.
  Failure failure;
  bool pending = false;
  bool needs_drain = false;
  bool ordered_completion = false;
  ggml_cgraph *recorded_graph = nullptr;
  std::vector<vk::CommandBuffer> recorded_commands;
  bool recorded = false;
  ggml_cgraph *assessed_graph = nullptr;
  bool fixed_attention = false;
  bool needs_rms_partials = true;
  bool has_dispatch = false;
  DescriptorPoolBlock *descriptor_pool = nullptr;
  std::vector<vk::DescriptorSet> borrowed_descriptors;
};

VkFence work_fence(const Work *work) noexcept {
  return work ? static_cast<VkFence>(work->fence) : VK_NULL_HANDLE;
}
bool work_recording_ready(const Work *work) noexcept {
  return work && work->recorded && !work->pending;
}

bool supports_ordered_submits(Runtime *runtime) noexcept {
#if defined(__APPLE__) && !defined(GGML_VULKAN_CHECK_RESULTS)
  Active active(runtime);
  const auto &device = runtime->instance.devices[0];
  return device->driver_id == vk::DriverId::eMoltenvk && device->support_async &&
         !device->serialize_submissions && !vk_perf_logger_enabled;
#else
  (void)runtime;
  return false;
#endif
}

Failure flush_ordered_submits(Runtime *runtime, OrderedSubmitBatch &batch,
                             VkFence fence) noexcept {
  if (batch.count == 0 && fence == VK_NULL_HANDLE)
    return {};
  VkSubmitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  info.commandBufferCount = std::exchange(batch.count, 0);
  info.pCommandBuffers = batch.commands.data();
  struct Call {
    PFN_vkQueueSubmit submit;
    const VkSubmitInfo &info;
    VkFence fence;
  } call{batch.submit, info, fence};
  const auto submitted = with_queue(runtime, [](void *state, VkQueue queue) {
    const auto &call = *static_cast<const Call *>(state);
    return call.submit(queue, 1, &call.info, call.fence);
  }, &call);
  batch.accepted = batch.accepted || submitted.error == Error::None;
  return submitted;
}

Failure append_ordered_submit(Runtime *runtime, OrderedSubmitBatch &batch,
                              VkCommandBuffer command) noexcept {
  if (batch.count == batch.commands.size()) {
    const auto flushed = flush_ordered_submits(runtime, batch, VK_NULL_HANDLE);
    if (flushed.error != Error::None)
      return flushed;
  }
  batch.commands[batch.count++] = command;
  return {};
}

namespace {
struct OrderedWorkScope;
thread_local OrderedWorkScope *active_ordered_work = nullptr;

// Vendor encoding may submit a serial prefix or need host staging. Only this
// Work's resident final batch may join the parent's ordered command list.
struct OrderedWorkScope {
  Work &work;
  ggml_backend_vk_context *context;
  OrderedSubmitBatch *batch;
  OrderedWorkScope *previous = nullptr;

  OrderedWorkScope(Work &work, ggml_backend_vk_context *ctx,
                   OrderedSubmitBatch *batch) noexcept
      : work(work), context(ctx), batch(batch) {
    if (batch)
      previous = std::exchange(active_ordered_work, this);
  }
  ~OrderedWorkScope() {
    if (batch)
      active_ordered_work = previous;
  }
};

void check_ordered_submit(Failure failure) {
  if (failure.error == Error::None)
    return;
  if (failure.native != VK_SUCCESS)
    throw vk::SystemError(vk::make_error_code(static_cast<vk::Result>(failure.native)),
                          "Vulkan ordered submit");
  if (failure.error == Error::Exhausted)
    throw std::bad_alloc();
  throw std::runtime_error("Vulkan ordered submit");
}

void flush_active_ordered_work() {
  if (const auto *active = active_ordered_work)
    check_ordered_submit(flush_ordered_submits(active->work.runtime, *active->batch,
                                               VK_NULL_HANDLE));
}

void append_work_command(Work &work, OrderedSubmitBatch &batch,
                         vk::CommandBuffer command) {
  work.ordered_completion = true;
  check_ordered_submit(append_ordered_submit(work.runtime, batch,
                                             static_cast<VkCommandBuffer>(command)));
}

bool defer_work_submit(ggml_backend_vk_context *ctx, const vk_context &commands) {
  auto *active = active_ordered_work;
  if (!active || active->context != ctx || commands->p != &ctx->compute_cmd_pool ||
      !commands->in_memcpys.empty() || !commands->out_memcpys.empty() ||
      !commands->memsets.empty())
    return false;
  for (const auto &sequence : commands->seqs)
    for (const auto &submission : sequence)
      if (!submission.wait_semaphores.empty() || !submission.signal_semaphores.empty())
        return false;

  for (const auto &sequence : commands->seqs) {
    for (const auto &submission : sequence) {
      if (commands->p->hyle_record_commands)
        commands->p->hyle_record_commands->push_back(submission.buffer->buf);
      append_work_command(active->work, *active->batch, submission.buffer->buf);
    }
  }
  commands->seqs.clear();
  return true;
}

void prepare_descriptors(Work &work, ggml_backend_vk_context &context,
                         const ggml_cgraph &graph) {
  if (work.descriptor_pool || !context.descriptor_sets.empty() ||
      graph.n_nodes >= VK_DEVICE_DESCRIPTOR_POOL_SIZE) return;
  // The usual finite region has only a few nodes. Reserve a bounded first
  // slice, then let the unchanged native allocator grow private overflow pools
  // when a node needs several passes. Native pool indexing remains unchanged.
  const auto count = static_cast<std::size_t>(graph.n_nodes);
  work.borrowed_descriptors.reserve(count);
  context.descriptor_sets.reserve(count);
  std::lock_guard lock(work.runtime->descriptor_mutex);
  DescriptorPoolBlock *selected = nullptr;
  for (const auto &pool : work.runtime->descriptor_pools)
    if (pool->available.size() >= count) { selected = pool.get(); break; }
  if (!selected) {
    auto pool = std::make_unique<DescriptorPoolBlock>();
    pool->device = context.device;
    const vk::DescriptorPoolSize size(vk::DescriptorType::eStorageBuffer,
        MAX_PARAMETER_COUNT * VK_DEVICE_DESCRIPTOR_POOL_SIZE);
    pool->pool = context.device->device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo({}, VK_DEVICE_DESCRIPTOR_POOL_SIZE, size));
    const std::vector<vk::DescriptorSetLayout> layouts(
        VK_DEVICE_DESCRIPTOR_POOL_SIZE, context.device->dsl);
    pool->available = context.device->device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo(pool->pool, VK_DEVICE_DESCRIPTOR_POOL_SIZE, layouts.data()));
    selected = pool.get();
    work.runtime->descriptor_pools.push_back(std::move(pool));
  }
  const auto first = selected->available.end() - count;
  work.borrowed_descriptors.assign(first, selected->available.end());
  context.descriptor_sets.assign(work.borrowed_descriptors.begin(), work.borrowed_descriptors.end());
  selected->available.resize(selected->available.size() - count);
  ++selected->borrowers;
  work.descriptor_pool = selected;
}
void release_descriptors(Work &work) noexcept {
  if (!work.descriptor_pool) return;
  std::lock_guard lock(work.runtime->descriptor_mutex);
  auto *pool = work.descriptor_pool;
  // Capacity was allocated once with the native pool; returning a retired
  // slice cannot allocate metadata or recycle another Work's live sets.
  GGML_ASSERT(pool->borrowers && pool->available.size() + work.borrowed_descriptors.size() <=
      VK_DEVICE_DESCRIPTOR_POOL_SIZE);
  pool->available.insert(pool->available.end(), work.borrowed_descriptors.begin(), work.borrowed_descriptors.end());
  work.borrowed_descriptors.clear();
  work.descriptor_pool = nullptr;
  if (--pool->borrowers == 0) {
    auto &pools = work.runtime->descriptor_pools;
    const auto found = std::find_if(pools.begin(), pools.end(),
        [&](const auto &candidate) { return candidate.get() == pool; });
    GGML_ASSERT(found != pools.end());
    pools.erase(found);
  }
}
WorkReport report(const Work &work) noexcept {
  return {work.failure, work.pending, work.runtime->device_lost.load()};
}
void remember(Work &work, Failure failure) noexcept {
  if (work.failure.error == Error::None)
    work.failure = failure;
  if (failure.native == VK_ERROR_DEVICE_LOST)
    work.runtime->device_lost.store(true);
}
VkResult drain_queue(const std::shared_ptr<vk_queue_handle> &handle) {
  std::unique_lock<std::mutex> device_guard;
  if (handle->device_submit_mutex)
    device_guard = std::unique_lock<std::mutex>(*handle->device_submit_mutex);
  std::lock_guard<vk_queue_handle> queue_guard(*handle);
  return VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueWaitIdle(handle->queue);
}
bool drain(Work &work) {
  auto device = work.runtime->instance.devices[0];
  const auto compute = device->compute_queue->handle;
  const auto transfer =
      device->transfer_queue && device->transfer_queue->handle != compute
          ? device->transfer_queue->handle
          : nullptr;
  // Only the exceptional path drains queues. A failed observation never
  // licenses freeing the accepted prefix or resetting its command buffers.
  for (const auto &handle : {compute, transfer}) {
    if (!handle)
      continue;
    const auto result = drain_queue(handle);
    if (result == VK_ERROR_DEVICE_LOST) {
      remember(work, {Error::Native, result});
      return true;
    }
    if (result != VK_SUCCESS) {
      remember(work, {Error::Native, result});
      return false;
    }
  }
  return true;
}
void retire(Work &work) noexcept {
  auto *ctx = static_cast<ggml_backend_vk_context *>(work.backend->context);
  // This function is called only after the completion fence/all relevant
  // queues have retired, or device loss has terminated their work.
  work.pending = false;
  work.ordered_completion = false;
  ctx->hyle_completion_fence = nullptr;
  ctx->compute_ctx.reset();
  ctx->transfer_ctx.reset();
  ctx->submit_pending = false;
  ctx->hyle_retired_failure =
      work.failure.error != Error::None || work.runtime->device_lost.load();
  try {
    if (!ctx->hyle_retired_failure && ctx->almost_ready_fence_pending)
      ctx->device->device.resetFences({ctx->almost_ready_fence});
    ctx->almost_ready_fence_pending = false;
    if (!work.recorded || ctx->hyle_retired_failure) {
      work.recorded = false;
      work.recorded_commands.clear();
      ggml_vk_graph_cleanup(ctx, !ctx->hyle_retired_failure);
    }
  } catch (...) {
    remember(work, current_failure());
    ctx->hyle_retired_failure = true;
  }
  ctx->device->diag_cgraph = nullptr;
}
bool may_use_rms_partials(const ggml_backend_vk_context &context,
                          const ggml_cgraph &graph) {
  if (!context.device->add_rms_fusion)
    return false;
  // A fused producer starts with an Add. Its last node must feed the next
  // RMS norm directly. Keep the original scratch for every such candidate.
  bool seen_add = false;
  for (int index = 0; index < graph.n_nodes; ++index) {
    if (seen_add && ggml_vk_rms_partials_consumer(graph, index))
      return true;
    seen_add = seen_add || graph.nodes[index]->op == GGML_OP_ADD;
  }
  return false;
}

bool fixed_attention_graph(const ggml_cgraph *graph) {
  bool attention = false;
  for (int i = 0; i < graph->n_nodes; ++i) {
    const auto *node = graph->nodes[i];
    switch (node->op) {
    case GGML_OP_FLASH_ATTN_EXT:
      if (node->op_params[kFlashAttentionMathSlot] != kFlashAttentionFullF32)
        return false;
      attention = true;
      break;
    case GGML_OP_NONE:
    case GGML_OP_VIEW:
    case GGML_OP_RESHAPE:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_CONT:
    case GGML_OP_CPY:
    case GGML_OP_CONCAT:
    case GGML_OP_FILL:
    case GGML_OP_DIAG_MASK_INF:
      break;
    default:
      return false;
    }
  }
  return attention;
}
} // namespace

Failure invalidate_work_recording(Work *work) noexcept {
  if (!work || work->pending)
    return {Error::InvalidState};
  work->assessed_graph = nullptr;
  work->fixed_attention = false;
  if (!work->recorded && work->recorded_commands.empty())
    return {};
  Active active(work->runtime);
  auto *ctx = static_cast<ggml_backend_vk_context *>(work->backend->context);
  work->recorded = false;
  work->recorded_graph = nullptr;
  work->recorded_commands.clear();
  try {
    ggml_vk_graph_cleanup(ctx, !work->runtime->device_lost.load());
    return {};
  } catch (...) {
    const auto failure = current_failure();
    remember(*work, failure);
    return failure;
  }
}

Failure create_work(Runtime *runtime, Work **result) noexcept {
  *result = nullptr;
  Work *work = nullptr;
  try {
    Active active(runtime);
    work = new Work();
    work->runtime = runtime;
    work->backend = ggml_backend_vk_init(0);
    if (!work->backend) {
      destroy_work(work);
      return {Error::Unavailable};
    }
    // The backend already owns an independent completion fence. During an
    // asynchronous region only its final batch signals that fence; serial
    // prefixes retire/reset it before our existing final marker is submitted.
    // Keep the original per-Work submission/retirement, without a third fence
    // per native context or sharing completion between independent Works.
    work->fence = static_cast<ggml_backend_vk_context *>(work->backend->context)->fence;
    *result = work;
    return {};
  } catch (...) {
    const auto failure = current_failure();
    destroy_work(work);
    return failure;
  }
}

static WorkReport submit_work_impl(Work *work, ggml_cgraph *graph,
                                  OrderedSubmitBatch *batch) noexcept {
  if (!work || !graph || graph->n_nodes <= 0)
    return {{Error::InvalidArgument}};
  if (work->pending)
    return {{Error::InvalidState}, true, work->runtime->device_lost.load()};
  if (work->failure.error != Error::None)
    return report(*work); // Failed native slots are never reused.
  if (work->runtime->device_lost.load()) {
    remember(*work, {Error::Native, VK_ERROR_DEVICE_LOST});
    return report(*work);
  }
  Active active(work->runtime);
  auto *ctx = static_cast<ggml_backend_vk_context *>(work->backend->context);
  OrderedWorkScope ordered(*work, ctx, batch);
  if (work->recorded && work->recorded_graph != graph) {
    const auto invalidated = invalidate_work_recording(work);
    if (invalidated.error != Error::None)
      return report(*work);
  }
  const auto original_rms_partials = ctx->prealloc_size_add_rms_partials;
  try {
    ctx->device->device.resetFences({work->fence});
    // The caller binds resident tensors; no unfinished host upload/download
    // context may sneak into this pure region completion contract.
    if (!ctx->compute_ctx.expired() || !ctx->transfer_ctx.expired())
      return {{Error::InvalidState}};
    const bool attach_to_last = ctx->device->support_async &&
                                !ctx->device->serialize_submissions &&
                                !vk_perf_logger_enabled;
    ctx->hyle_completion_fence = attach_to_last ? work->fence : vk::Fence{};
    ctx->hyle_completion_submitted = false;
    work->pending = true; // Conservatively retained before any fallible submit.
    if (work->recorded) {
      ctx->device->diag_cgraph = graph;
      const vk::SubmitInfo submitted{
          0,
          nullptr,
          nullptr,
          static_cast<uint32_t>(work->recorded_commands.size()),
          work->recorded_commands.data(),
          0,
          nullptr};
      if (batch) {
        for (const auto command : work->recorded_commands)
          append_work_command(*work, *batch, command);
      } else {
        ctx->device->compute_queue->handle->submit(submitted, work->fence);
      }
      ctx->submit_pending = true;
      ctx->hyle_completion_submitted = true;
      ctx->hyle_completion_fence = nullptr;
      return report(*work);
    }
    prepare_descriptors(*work, *ctx, *graph);
    if (work->assessed_graph != graph) {
      work->fixed_attention = fixed_attention_graph(graph);
      work->needs_rms_partials = may_use_rms_partials(*ctx, *graph);
      work->has_dispatch = std::any_of(graph->nodes, graph->nodes + graph->n_nodes,
          [](ggml_tensor *node) {
            return !ggml_vk_is_empty(node) &&
                   (node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
          });
      work->assessed_graph = graph;
    }
    const bool eligible = attach_to_last && work->fixed_attention;
    // Upstream can retire an already submitted prefix and grow its scratch
    // buffers while recording later attention nodes. Such a first execution
    // is valid, but its saved prefix commands refer to the replaced storage.
    // Weak owners preserve control-block identity without retaining old GPU
    // allocations. Admit replay only after all four scratch owners are stable.
    const std::array<std::weak_ptr<vk_buffer_struct>, 4> recording_scratch{
        ctx->prealloc_x, ctx->prealloc_y, ctx->prealloc_split_k,
        ctx->prealloc_add_rms_partials};
    // A graph without a candidate consumer does not read the partial sums.
    // Avoid its private allocation and host clear. Restore the original
    // sizing metadata after execution so another graph can use it.
    if (!work->needs_rms_partials)
      ctx->prealloc_size_add_rms_partials = 0;
    ctx->compute_cmd_pool.hyle_record_commands =
        eligible ? &work->recorded_commands : nullptr;
    ctx->compute_cmd_pool.hyle_record_safe = eligible;
    // Native graph dependency tracking covers this Work's own tensors only.
    // Imported inputs may have been written by a different Work or route on
    // the same ordered queue. Establish their device visibility in the first
    // original command buffer, after recording is enabled so replay keeps it.
    // Perf and RMS-partial initialization already emit this entry dependency.
    // A graph made only of views must not acquire an unused compute context.
    vk_context imported_inputs;
    if (work->has_dispatch && !vk_perf_logger_enabled &&
        !ctx->prealloc_size_add_rms_partials) {
      imported_inputs = ggml_vk_get_compute_ctx(ctx);
      ggml_vk_sync_buffers(ctx, imported_inputs);
    }
    // This ordered Work is already a prepared graph fragment. MoltenVK
    // allocates Metal command buffers for its split submissions. Keep the
    // original node and completion boundaries without its FLOP heuristic.
    if (ctx->device->driver_id == vk::DriverId::eMoltenvk)
      ctx->last_total_flops = 0;
    if (ggml_backend_vk_graph_compute(work->backend, graph) !=
        GGML_STATUS_SUCCESS) {
      remember(*work, {Error::Internal});
      work->needs_drain = true;
    } else if (!ctx->hyle_completion_submitted) {
      // Upstream's explicit serial/diagnostic mode, or a graph without an
      // emitted final batch. One region marker, never a per-kernel wrapper.
      flush_active_ordered_work();
      ctx->device->compute_queue->handle->submit({}, work->fence);
    }
    const std::array<vk_buffer, 4> current_scratch{
        ctx->prealloc_x, ctx->prealloc_y, ctx->prealloc_split_k,
        ctx->prealloc_add_rms_partials};
    const bool scratch_stable = std::equal(
        recording_scratch.begin(), recording_scratch.end(), current_scratch.begin(),
        [](const auto& before, const auto& after) {
          return !before.owner_before(after) && !after.owner_before(before);
        });
    work->recorded = eligible && work->failure.error == Error::None &&
                     scratch_stable &&
                     ctx->compute_cmd_pool.hyle_record_safe &&
                     !work->recorded_commands.empty() &&
                     ctx->gc.semaphores.empty() &&
                     ctx->gc.tl_semaphores.empty() && ctx->gc.events.empty();
    work->recorded_graph = work->recorded ? graph : nullptr;
    ctx->prealloc_size_add_rms_partials = original_rms_partials;
    ctx->compute_cmd_pool.hyle_record_commands = nullptr;
    ctx->hyle_completion_fence = nullptr;
  } catch (...) {
    ctx->prealloc_size_add_rms_partials = original_rms_partials;
    ctx->compute_cmd_pool.hyle_record_commands = nullptr;
    work->recorded = false;
    ctx->hyle_completion_fence = nullptr;
    remember(*work, current_failure());
    work->needs_drain = work->pending;
    if (work->runtime->device_lost.load() && work->pending)
      retire(*work);
  }
  return report(*work);
}

WorkReport submit_work(Work *work, ggml_cgraph *graph) noexcept {
  return submit_work_impl(work, graph, nullptr);
}
WorkReport submit_work(Work *work, ggml_cgraph *graph,
                       OrderedSubmitBatch &batch) noexcept {
  return submit_work_impl(work, graph, &batch);
}

WorkReport observe_work(Work *work, bool wait) noexcept {
  if (!work)
    return {{Error::InvalidArgument}};
  if (!work->pending)
    return report(*work);
  Active active(work->runtime);
  try {
    if (work->runtime->device_lost.load()) {
      remember(*work, {Error::Native, VK_ERROR_DEVICE_LOST});
      retire(*work);
    } else if (work->needs_drain) {
      // Nonblocking poll reports both the diagnosis and retained pending
      // ownership. Explicit wait/close performs exceptional queue-wide drain.
      if (wait && drain(*work))
        retire(*work);
    } else if (!work->ordered_completion) {
      const auto device =
          static_cast<VkDevice>(work->runtime->instance.devices[0]->device);
      const auto fence = static_cast<VkFence>(work->fence);
      const auto &api = work->runtime->instance.dispatcher;
      const auto result =
          wait ? api.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX)
               : api.vkGetFenceStatus(device, fence);
      if (result == VK_SUCCESS) {
        retire(*work);
      } else if (result == VK_ERROR_DEVICE_LOST) {
        remember(*work, {Error::Native, result});
        retire(*work);
      } else if (result != VK_NOT_READY && result != VK_TIMEOUT) {
        remember(*work, {Error::Native, result});
        work->needs_drain = true;
        if (wait && drain(*work))
          retire(*work);
      }
    }
  } catch (...) {
    remember(*work, current_failure());
    work->needs_drain = true;
    if (work->runtime->device_lost.load())
      retire(*work);
  }
  return report(*work);
}

WorkReport retire_ordered_work(Work *work, bool wait) noexcept {
  if (!work || !work->ordered_completion || work->needs_drain)
    return observe_work(work, wait);
  // The parent proof covers every accepted command on its compute queue.
  Active active(work->runtime);
  if (work->runtime->device_lost.load())
    remember(*work, {Error::Native, VK_ERROR_DEVICE_LOST});
  retire(*work);
  return report(*work);
}

void destroy_work(Work *work) noexcept {
  if (!work)
    return;
  GGML_ASSERT(!work->ordered_completion);
  while (work->pending) {
    observe_work(work, true);
    if (work->pending)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  Active active(work->runtime);
  if (work->backend) {
    auto *ctx = static_cast<ggml_backend_vk_context *>(work->backend->context);
    // Final destruction needs no native command/event resets. In particular,
    // device-loss teardown must not throw while resetting a lost device.
    ctx->hyle_retired_failure = true;
    ggml_backend_free(work->backend);
  }
  release_descriptors(*work);
  // ggml_backend_free destroys the borrowed native completion fence once.
  delete work;
}
} // namespace hyle::compute::ggml_vulkan::bridge

static void hyle_flush_ordered_work() {
#if defined(__APPLE__)
    hyle::compute::ggml_vulkan::bridge::flush_active_ordered_work();
#endif
}

static bool hyle_defer_work_submit(ggml_backend_vk_context *ctx,
                                   const vk_context &commands) {
#if defined(__APPLE__)
    return hyle::compute::ggml_vulkan::bridge::defer_work_submit(ctx, commands);
#else
    (void)ctx;
    (void)commands;
    return false;
#endif
}
