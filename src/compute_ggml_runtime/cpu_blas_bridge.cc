// Keep the original mature BLAS implementation and its private cold reserve
// together, just as the CPU vector bridge retains its locked source unit.
#include HYLE_GGML_BLAS_SOURCE

#include "cpu_blas_bridge.h"

#include <new>

namespace {
bool native_threading_available() noexcept {
  // The original native_threads budget must not silently become an
  // unconstrained Accelerate worker pool. The API is local to this thread;
  // never change a process environment setting or another caller's policy.
  static const bool available = [] {
    if (__builtin_available(macOS 15.0, *)) {
      const auto previous = BLASGetThreading();
      if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) != 0)
        return false;
      return BLASSetThreading(previous) == 0;
    }
    return false;
  }();
  return available;
}
} // namespace

extern "C" bool ggml_hyle_blas_supports(const ggml_tensor *node) noexcept {
  if (!native_threading_available() || !node ||
      node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1] ||
      node->type != GGML_TYPE_F32)
    return false;
  const auto type = node->src[0]->type;
  // This recipe widens floating storage exactly. Encoded activation/dot
  // contracts stay with the original native CPU implementation.
  if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16)
    return false;
  // The original CPU dot narrows a mixed F32 activation to its half storage.
  // Only an explicit exact widening of that storage can avoid the narrowing
  // without changing this native graph's mathematical operand contract.
  if (type != GGML_TYPE_F32) {
    const auto *activation = node->src[1];
    if (activation->op != GGML_OP_CPY || !activation->src[0] ||
        activation->src[0]->type != type)
      return false;
  }
  auto *device = ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0);
  return ggml_backend_dev_supports_op(device, node);
}

extern "C" bool ggml_hyle_blas_reserve(ggml_backend_t backend,
                                       std::size_t bytes) noexcept {
  if (!backend || !native_threading_available())
    return false;
  auto &context = *static_cast<ggml_backend_blas_context *>(backend->context);
  if (bytes > context.work_size) {
    std::unique_ptr<char[]> storage(new (std::nothrow) char[bytes]);
    if (!storage)
      return false;
    context.work_data = std::move(storage);
    context.work_size = bytes;
  }
  // The original HYLE native team owns worker admission. Weight conversion
  // stays on this submitting thread, avoiding per-call async task allocation.
  ggml_backend_blas_set_n_threads(backend, 1);
  return true;
}

extern "C" ggml_status ggml_hyle_blas_compute(ggml_backend_t backend,
                                              ggml_cgraph *graph) noexcept {
  if (__builtin_available(macOS 15.0, *)) {
    const auto previous = BLASGetThreading();
    if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) != 0)
      return GGML_STATUS_FAILED;
    const auto result = ggml_backend_graph_compute(backend, graph);
    if (BLASSetThreading(previous) != 0)
      return GGML_STATUS_FAILED;
    return result;
  }
  return GGML_STATUS_FAILED;
}
