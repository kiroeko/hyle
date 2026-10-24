#pragma once

// Included at the end of the locked ggml-cuda.cu, where its resource types are
// visible. All computation stays upstream; the admitted stream is borrowed
// only while that same backend submits its original graph.
#include "native_bridge.h"

namespace hyle::compute::ggml_cuda::bridge {

int device_index(int physical_device) {
  const auto &info = ggml_cuda_info();
  for (int index = 0; index < info.device_count; ++index)
    if (ggml_cuda_get_physical_device(index) == physical_device)
      return index;
  return -1;
}

bool supports_bfloat_paired_projection(int device, std::uint64_t inner,
                                      std::uint64_t rows) {
  return ggml_cuda_supports_bf16_paired_projection_geometry(device, inner, rows);
}

void initialize_backend(ggml_backend *backend) {
  auto *context = static_cast<ggml_backend_cuda_context *>(backend->context);
  // Keep the original caching allocator without creating an unused root
  // stream. Admitted compute installs its real caller stream before any
  // native stream() call; standalone cold packing keeps native lazy creation.
  for (int stream = 0; stream < GGML_CUDA_MAX_STREAMS; ++stream) {
    auto &pool = context->pools[context->device][stream];
    GGML_ASSERT(pool == nullptr);
    pool = std::make_unique<ggml_cuda_pool_leg>(context->device);
  }
}

void collect_allocation_dependencies(
    ggml_backend *backend, ggml_cgraph *graph,
    void (*add)(void *user, ggml_tensor *root, int until_node), void *user) {
  GGML_ASSERT(backend && graph && add);
  ggml_cuda_collect_fusion_alloc_deps(
      static_cast<ggml_backend_cuda_context *>(backend->context), graph,
      [&](ggml_tensor *root, int until_node) { add(user, root, until_node); });
}

cudaError_t drain(ggml_backend *backend) noexcept {
  auto *context = static_cast<ggml_backend_cuda_context *>(backend->context);
  cudaError_t first = cudaSuccess;
  for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
    for (int index = 0; index < GGML_CUDA_MAX_STREAMS; ++index) {
      if (context->streams[device][index] == nullptr)
        continue;
      const auto status =
          cudaStreamSynchronize(context->streams[device][index]);
      if (first == cudaSuccess)
        first = status;
    }
  }
  return first;
}

cudaError_t query(ggml_backend *backend) noexcept {
  auto *context = static_cast<ggml_backend_cuda_context *>(backend->context);
  cudaError_t first = cudaSuccess;
  for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
    for (int index = 0; index < GGML_CUDA_MAX_STREAMS; ++index) {
      if (context->streams[device][index] == nullptr)
        continue;
      const auto status = cudaStreamQuery(context->streams[device][index]);
      if (first == cudaSuccess)
        first = status;
    }
  }
  return first;
}

namespace {

class AdmittedStreamScope final {
public:
  AdmittedStreamScope(ggml_backend *backend, cudaStream_t stream,
                      bool recording = false)
      : backend_(backend),
        context_(static_cast<ggml_backend_cuda_context *>(backend->context)),
        stream_(stream), original_stream_(context_->streams[context_->device][0]),
        original_stream_no_(context_->curr_stream_no),
        original_handle_(context_->cublas_handles[context_->device][0]),
        recording_(recording) {
    if (original_handle_ != nullptr) {
      const auto status = cublasGetStream(original_handle_, &original_handle_stream_);
      if (status != CUBLAS_STATUS_SUCCESS)
        fail(status, "cublasGetStream");
    }
  }

  AdmittedStreamScope(const AdmittedStreamScope &) = delete;
  AdmittedStreamScope &operator=(const AdmittedStreamScope &) = delete;

  ~AdmittedStreamScope() noexcept {
    if (installed_)
      (void)recover();
  }

  void install() {
    context_->streams[context_->device][0] = stream_;
    context_->curr_stream_no = 0;
    installed_ = true;
    const auto status = bind_root_handle(stream_);
    if (status != CUBLAS_STATUS_SUCCESS)
      fail(status, "binding admitted cuBLAS stream and workspace");
  }

  void begin_compute() noexcept { native_started_ = true; }

  // Native graph submission may have created the root handle for the first
  // time. Restore the original root binding, including the default stream
  // when this backend never needed an owned root stream.
  cublasStatus_t restore() noexcept {
    const auto target = original_handle_ != nullptr ? original_handle_stream_
                                                   : original_stream_;
    const auto status = bind_root_handle(target);
    if (status != CUBLAS_STATUS_SUCCESS) {
      // A restore failure follows real submission. Quiesce while the table
      // still names the borrowed stream, before restoring the owned entries.
      if (native_started_)
        (void)retire_prefix();
      (void)bind_root_handle(target);
    }
    restore_table();
    return status;
  }

  cudaError_t recover() noexcept {
    if (installed_) {
      if (native_started_)
        (void)retire_prefix();
      const auto target = original_handle_ != nullptr ? original_handle_stream_
                                                     : original_stream_;
      (void)bind_root_handle(target);
      restore_table();
    }
    return drain_status_;
  }

  cudaError_t drain_all() noexcept {
    if (!drained_) {
      drained_ = true;
      drain_status_ = drain(backend_);
    }
    return drain_status_;
  }

private:
  cudaError_t retire_prefix() noexcept {
    // Synchronizing a participating stream before outer End/abort would
    // invalidate capture. Its original Completion observes all owned native
    // streams after the caller has ended that recording instead.
    if (recording_) {
      drain_status_ = cudaErrorNotReady;
      return drain_status_;
    }
    return drain_all();
  }

  cublasStatus_t bind_root_handle(cudaStream_t stream) noexcept {
    const auto device = context_->device;
    const auto handle = context_->cublas_handles[device][0];
    if (handle == nullptr)
      return CUBLAS_STATUS_SUCCESS;
    auto status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;
    // cublasSetStream resets a user workspace. Reapply the original native
    // allocation, without replacing its owner, size, math mode or pool.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) &&                        \
    (CUBLAS_VER_MAJOR > 11 ||                                                \
     (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR >= 2))
    if (context_->cublas_workspaces[device][0] != nullptr)
      status = cublasSetWorkspace(handle, context_->cublas_workspaces[device][0],
                                  context_->cublas_workspace_sizes[device]);
#endif
    return status;
  }

  void restore_table() noexcept {
    context_->streams[context_->device][0] = original_stream_;
    context_->curr_stream_no = original_stream_no_;
    installed_ = false;
  }

  ggml_backend *backend_;
  ggml_backend_cuda_context *context_;
  cudaStream_t stream_;
  cudaStream_t original_stream_;
  int original_stream_no_;
  cublasHandle_t original_handle_;
  cudaStream_t original_handle_stream_ = nullptr;
  cudaError_t drain_status_ = cudaSuccess;
  bool installed_ = false;
  bool drained_ = false;
  bool native_started_ = false;
  bool recording_ = false;
};

class RecordingModeScope final {
public:
  RecordingModeScope() {
    CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&previous_));
    active_ = true;
  }
  ~RecordingModeScope() noexcept {
    if (active_)
      (void)cudaThreadExchangeStreamCaptureMode(&previous_);
  }
  void restore() {
    const auto status = cudaThreadExchangeStreamCaptureMode(&previous_);
    if (status == cudaSuccess)
      active_ = false;
    CUDA_CHECK(status);
  }
private:
  cudaStreamCaptureMode previous_ = cudaStreamCaptureModeRelaxed;
  bool active_ = false;
};

} // namespace

void retain_recording_guard() {
  std::lock_guard<std::mutex> lock(ggml_cuda_lock);
  ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
}

void release_recording_guard() noexcept {
  std::lock_guard<std::mutex> lock(ggml_cuda_lock);
  if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1)
    ggml_cuda_lock_cv.notify_all();
}

bool record_compatible(ggml_backend *backend, ggml_cgraph *graph) {
#if defined(USE_CUDA_GRAPH)
  if (backend == nullptr || graph == nullptr)
    return false;
  auto *context =
      static_cast<ggml_backend_cuda_context *>(backend->context);
  ggml_cuda_set_device(context->device);
  // Match ordinary graph_compute's original metadata initialization. Mean's
  // CUB choice observes any_cuda_graph_enabled(): this enabled entry keeps its
  // original >65536 threshold in Commands and outer capture alike. No graph
  // math, warmup, capture or instance is created by this metadata operation.
  return ggml_cuda_graph_set_enabled(context, ggml_cuda_graph_get_key(graph)) &&
         ggml_cuda_graph_check_compability(graph);
#else
  GGML_UNUSED(backend);
  GGML_UNUSED(graph);
  return false;
#endif
}

ggml_status record_on_admitted_stream(ggml_backend *backend,
                                     ggml_cgraph *graph, cudaStream_t stream,
                                     cudaError_t *retirement_error) {
  if (retirement_error == nullptr)
    fail(cudaErrorInvalidValue, "recording retirement output is null");
  *retirement_error = cudaSuccess;
  if (backend == nullptr || graph == nullptr || stream == nullptr)
    fail(cudaErrorInvalidValue, "recording graph or stream is null");
  RecordingModeScope mode;
  AdmittedStreamScope scope(backend, stream, true);
  try {
    // Use this Invocation's actual cache, not the temporary compile context.
    // Refresh the original key even after the native idle eviction interval.
    if (!record_compatible(backend, graph))
      fail(cudaErrorStreamCaptureUnsupported,
           "native graph configuration does not support recording");
    scope.install();
    scope.begin_compute();
    auto *context =
        static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_hyle_recording_scope recording;
    ggml_cuda_graph_encode(context, graph);
    const auto restored = scope.restore();
    if (restored != CUBLAS_STATUS_SUCCESS)
      fail(restored, "restoring recorded native cuBLAS stream and workspace");
    mode.restore();
    return GGML_STATUS_SUCCESS;
  } catch (...) {
    *retirement_error = scope.recover();
    // Successful restore has removed the borrowed stream already. A later
    // mode-restore failure still belongs to this unfinished outer recording.
    if (*retirement_error == cudaSuccess)
      *retirement_error = cudaErrorNotReady;
    throw;
  }
}

ggml_status compute_on_admitted_stream(ggml_backend *backend,
                                      ggml_cgraph *graph, cudaStream_t stream,
                                      cudaError_t *retirement_error) {
  if (retirement_error == nullptr)
    fail(cudaErrorInvalidValue, "admitted retirement output is null");
  *retirement_error = cudaSuccess;
  // A null table entry means 'create an owned stream' to the native backend,
  // so the CUDA legacy default stream cannot be borrowed through this seam.
  if (backend == nullptr || graph == nullptr || stream == nullptr)
    fail(cudaErrorInvalidValue, "admitted graph or stream is null");
  AdmittedStreamScope scope(backend, stream);
  try {
    scope.install();
    scope.begin_compute();
    const auto status = ggml_backend_graph_compute_async(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
      (void)scope.drain_all();
      // Retain the native graph status if restoring a handle also fails.
      (void)scope.restore();
      *retirement_error = scope.recover();
      return status;
    }
    const auto restored = scope.restore();
    *retirement_error = scope.recover();
    if (restored != CUBLAS_STATUS_SUCCESS)
      fail(restored, "restoring native cuBLAS stream and workspace");
    return status;
  } catch (...) {
    // The original native capture guard has already unwound here. Keep the
    // borrowed stream installed through all-stream drain, including streams
    // whose ordinary native join node was never reached. An unsuccessful
    // drain is an explicit retirement fact for the original Queue owner.
    *retirement_error = scope.recover();
    throw;
  }
}

ggml_backend_buffer *binding_buffer(int device, std::size_t capacity) {
  auto context =
      std::make_unique<ggml_backend_cuda_buffer_context>(device, nullptr);
  context->owns_payload = false;
  auto *buffer = ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(device),
                                          ggml_backend_cuda_buffer_interface,
                                          context.get(), capacity);
  context.release();
  return buffer;
}

void bind_buffer(ggml_backend_buffer *buffer, void *pointer) noexcept {
  auto *context =
      static_cast<ggml_backend_cuda_buffer_context *>(buffer->context);
  context->dev_ptr = pointer;
}

} // namespace hyle::compute::ggml_cuda::bridge
