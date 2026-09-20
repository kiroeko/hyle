#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <ggml.h>

#include "metadata_arena.h"
#include "native_bridge.h"

namespace bridge = hyle::compute::ggml_cuda::bridge;

#if defined(HYLE_GGML_CUDA_FAULT_INJECTION)
namespace {
enum class Fault { kNone, kBegin, kEncode, kEnd };
thread_local Fault fault = Fault::kNone;
} // namespace
extern "C" cudaError_t __real_cudaStreamBeginCapture(cudaStream_t,
                                                     cudaStreamCaptureMode);
extern "C" cudaError_t
__wrap_cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  if (fault == Fault::kBegin) {
    fault = Fault::kNone;
    return cudaErrorStreamCaptureUnsupported;
  }
  return __real_cudaStreamBeginCapture(stream, mode);
}
extern "C" cudaError_t __real_cudaStreamEndCapture(cudaStream_t, cudaGraph_t *);
extern "C" cudaError_t __wrap_cudaStreamEndCapture(cudaStream_t stream,
                                                   cudaGraph_t *graph) {
  const auto result = __real_cudaStreamEndCapture(stream, graph);
  if (fault == Fault::kEnd) {
    fault = Fault::kNone;
    return cudaErrorStreamCaptureInvalidated;
  }
  return result;
}
extern "C" cudaError_t __real_cudaGetLastError();
extern "C" cudaError_t __wrap_cudaGetLastError() {
  if (fault == Fault::kEncode) {
    fault = Fault::kNone;
    return cudaErrorLaunchOutOfResources;
  }
  return __real_cudaGetLastError();
}
#endif

namespace {

struct NativeProgram final {
  static constexpr int kWidth = 128;
  static constexpr int kRows = 4;
  static constexpr std::size_t kElements = kWidth * kRows;
  static constexpr std::size_t kBytes = kElements * sizeof(float);
  hyle::compute::ggml_runtime::MetadataArena arena;
  ggml_backend *backend = nullptr;
  std::array<ggml_backend_buffer *, 3> buffers{};
  float *input = nullptr;
  float *output = nullptr;
  ggml_cgraph *graph = nullptr;

  NativeProgram() {
    try {
      // This executable embeds ggml separately from the MetadataArena DLL.
      // Initialize its upstream timer just as the ordinary CUDA provider does.
      ggml_time_init();
      arena = hyle::compute::ggml_runtime::MetadataArena::for_graph(
                  8, 16, hyle::mem::system_allocator())
                  .value();
      const int device = bridge::device_index(0);
      if (device < 0)
        throw std::runtime_error("CUDA device missing from ggml table");
      backend = ggml_backend_cuda_init(device);
      if (!backend)
        throw std::runtime_error("ggml CUDA backend creation failed");
      auto *x = ggml_new_tensor_2d(arena.get(), GGML_TYPE_F32, kWidth, kRows);
      auto *norm = ggml_rms_norm(arena.get(), x, 1e-5F);
      auto *result = ggml_add(arena.get(), norm, x);
      buffers[0] = bridge::binding_buffer(device, kBytes);
      buffers[1] = ggml_backend_buft_alloc_buffer(
          ggml_backend_cuda_buffer_type(device), kBytes);
      buffers[2] = bridge::binding_buffer(device, kBytes);
      if (cudaMalloc(reinterpret_cast<void **>(&input), kBytes) !=
              cudaSuccess ||
          cudaMalloc(reinterpret_cast<void **>(&output), kBytes) != cudaSuccess)
        throw std::runtime_error("native test allocation failed");
      bridge::bind_buffer(buffers[0], input);
      bridge::bind_buffer(buffers[2], output);
      if (ggml_backend_tensor_alloc(buffers[0], x, input) !=
              GGML_STATUS_SUCCESS ||
          ggml_backend_tensor_alloc(buffers[1], norm,
                                    ggml_backend_buffer_get_base(buffers[1])) !=
              GGML_STATUS_SUCCESS ||
          ggml_backend_tensor_alloc(buffers[2], result, output) !=
              GGML_STATUS_SUCCESS)
        throw std::runtime_error("native test tensor binding failed");
      graph = ggml_new_graph_custom(arena.get(), 16, false);
      ggml_build_forward_expand(graph, result);
    } catch (...) {
      reset();
      throw;
    }
  }
  NativeProgram(const NativeProgram &) = delete;
  NativeProgram &operator=(const NativeProgram &) = delete;
  ~NativeProgram() { reset(); }

  void reset() noexcept {
    if (backend)
      (void)bridge::drain(backend);
    for (auto *buffer : buffers)
      ggml_backend_buffer_free(buffer);
    // Foreign wrappers are already gone. These are the only payload owners.
    (void)cudaFree(input);
    (void)cudaFree(output);
    ggml_backend_free(backend);
  }
  std::vector<float> values(int seed) const {
    std::vector<float> result(kElements);
    for (std::size_t i = 0; i < result.size(); ++i)
      result[i] =
          static_cast<float>(static_cast<int>(i % 31) - 15 + seed) / 8.0F;
    return result;
  }
  void execute() {
    if (ggml_backend_graph_compute_async(backend, graph) != GGML_STATUS_SUCCESS)
      throw std::runtime_error("native graph compute failed");
  }
};

TEST(GgmlCudaResources,
     OriginalRegionReplaysWithBorrowedPayloadAndIndependentBackends) {
  NativeProgram first, second;
  ASSERT_NE(bridge::root_stream(first.backend),
            bridge::root_stream(second.backend));
  for (int seed : {0, 3, 7}) {
    const auto values = first.values(seed);
    ASSERT_EQ(cudaMemcpy(first.input, values.data(), first.kBytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(second.input, values.data(), second.kBytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    first.execute();
    second.execute();
    ASSERT_EQ(bridge::drain(second.backend), cudaSuccess);
    ASSERT_EQ(bridge::drain(first.backend), cudaSuccess);
    std::vector<float> actual(first.kElements), other(first.kElements);
    ASSERT_EQ(cudaMemcpy(actual.data(), first.output, first.kBytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(other.data(), second.output, second.kBytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int row = 0; row < first.kRows; ++row) {
      double sum = 0;
      for (int col = 0; col < first.kWidth; ++col) {
        const double value =
            values[static_cast<std::size_t>(row * first.kWidth + col)];
        sum += value * value;
      }
      const double scale = 1 / std::sqrt(sum / first.kWidth + 1e-5);
      for (int col = 0; col < first.kWidth; ++col) {
        const auto index = static_cast<std::size_t>(row * first.kWidth + col);
        EXPECT_NEAR(actual[index], values[index] * (scale + 1), 1e-5);
        EXPECT_EQ(actual[index], other[index]);
      }
    }
  }
  for (auto &buffer : first.buffers) {
    ggml_backend_buffer_free(buffer);
    buffer = nullptr;
  }
  // A non-owning buffer must leave external storage usable after teardown.
  EXPECT_EQ(cudaMemset(first.input, 0, first.kBytes), cudaSuccess);
  EXPECT_EQ(cudaMemset(first.output, 0, first.kBytes), cudaSuccess);
}

#if defined(HYLE_GGML_CUDA_FAULT_INJECTION)
TEST(GgmlCudaResources,
     CaptureFailuresKeepNativeCodesAndDoNotDeadlockTeardown) {
  for (auto failure : {Fault::kBegin, Fault::kEncode, Fault::kEnd}) {
    SCOPED_TRACE(static_cast<int>(failure));
    {
      NativeProgram program;
      const auto values = program.values(0);
      ASSERT_EQ(cudaMemcpy(program.input, values.data(), program.kBytes,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      program.execute();
      ASSERT_EQ(bridge::drain(program.backend), cudaSuccess);
      fault = failure;
      bool caught = false;
      try {
        program.execute();
      } catch (const bridge::NativeFailure<cudaError_t> &error) {
        caught = true;
        const auto expected =
            failure == Fault::kBegin    ? cudaErrorStreamCaptureUnsupported
            : failure == Fault::kEncode ? cudaErrorLaunchOutOfResources
                                        : cudaErrorStreamCaptureInvalidated;
        EXPECT_EQ(error.code, expected);
        EXPECT_NE(error.operation, nullptr);
      }
      EXPECT_TRUE(caught);
      EXPECT_EQ(fault, Fault::kNone);
      fault = Fault::kNone;
      EXPECT_EQ(bridge::drain(program.backend), cudaSuccess);
    }
    NativeProgram recovered;
    const auto values = recovered.values(1);
    ASSERT_EQ(cudaMemcpy(recovered.input, values.data(), recovered.kBytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    recovered.execute();
    EXPECT_EQ(bridge::drain(recovered.backend), cudaSuccess);
  }
}
#endif

} // namespace
