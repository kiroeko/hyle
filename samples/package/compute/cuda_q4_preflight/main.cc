#include "kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

#include <cublasLt.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/error/result.h>

namespace {
using hyle::Status;
namespace cuda = hyle::compute::cuda;
namespace mem = hyle::mem;

Status check(cublasStatus_t result, const char *operation) noexcept {
  if (result == CUBLAS_STATUS_SUCCESS)
    return Status::Ok();
  std::fprintf(stderr, "%s: cuBLAS status %d\n", operation,
               static_cast<int>(result));
  return result == CUBLAS_STATUS_ALLOC_FAILED
             ? Status::ResourceExhausted(operation)
             : Status::Internal(operation);
}

// Pure vendor owner. Every descriptor/algorithm query is cold. This sample
// does not claim that vendor-private host allocation has been qualified.
struct VendorPlan final {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, y = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t selected{};

  VendorPlan() noexcept = default;
  VendorPlan(const VendorPlan &) = delete;
  VendorPlan &operator=(const VendorPlan &) = delete;
  ~VendorPlan() {
    if (preference)
      (void)cublasLtMatmulPreferenceDestroy(preference);
    if (y)
      (void)cublasLtMatrixLayoutDestroy(y);
    if (b)
      (void)cublasLtMatrixLayoutDestroy(b);
    if (a)
      (void)cublasLtMatrixLayoutDestroy(a);
    if (operation)
      (void)cublasLtMatmulDescDestroy(operation);
    if (handle)
      (void)cublasLtDestroy(handle);
  }

  Status initialize(int rows, int columns, int inner) noexcept {
    HYLE_RETURN_IF_ERROR(check(cublasLtCreate(&handle), "create cuBLASLt"));
    HYLE_RETURN_IF_ERROR(
        check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F_PEDANTIC,
                                       CUDA_R_32F),
              "create matmul"));
    const cublasOperation_t transpose = CUBLAS_OP_T;
    HYLE_RETURN_IF_ERROR(check(
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &transpose, sizeof(transpose)),
        "set weight transpose"));
    HYLE_RETURN_IF_ERROR(
        check(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, rows, inner, inner),
              "create input layout"));
    HYLE_RETURN_IF_ERROR(
        check(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, columns, inner, inner),
              "create weight layout"));
    HYLE_RETURN_IF_ERROR(check(
        cublasLtMatrixLayoutCreate(&y, CUDA_R_32F, rows, columns, columns),
        "create output layout"));
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    for (auto layout : {a, b, y})
      HYLE_RETURN_IF_ERROR(check(
          cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                           &order, sizeof(order)),
          "set row-major layout"));
    HYLE_RETURN_IF_ERROR(check(cublasLtMatmulPreferenceCreate(&preference),
                               "create matmul preference"));
    const std::size_t max_workspace = 4 * 1024 * 1024;
    HYLE_RETURN_IF_ERROR(
        check(cublasLtMatmulPreferenceSetAttribute(
                  preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &max_workspace, sizeof(max_workspace)),
              "set workspace budget"));
    int count = 0;
    HYLE_RETURN_IF_ERROR(
        check(cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, y, y,
                                             preference, 1, &selected, &count),
              "query matmul algorithm"));
    if (count != 1 || selected.state != CUBLAS_STATUS_SUCCESS)
      return Status::Unimplemented("no cuBLASLt algorithm for preflight shape");
    return Status::Ok();
  }

  Status enqueue(cudaStream_t stream, const float *input, const float *weights,
                 float *output, void *workspace) const noexcept {
    const float alpha = 1, beta = 0;
    return check(cublasLtMatmul(handle, operation, &alpha, input, a, weights, b,
                                &beta, output, y, output, y, &selected.algo,
                                workspace, selected.workspaceSize, stream),
                 "enqueue cuBLASLt matmul");
  }
};

Status exercise(int rows) {
  constexpr int columns = 65, inner = 96;
  constexpr std::size_t weight_values = columns * inner;
  constexpr std::size_t weight_blocks = weight_values / 32;
  constexpr std::size_t q4_bytes = weight_blocks * sizeof(q4_preflight::Block);
  // Include negative and subnormal FP16 scales. Values are known exactly;
  // the numerical oracle does not reuse the CUDA decode implementation.
  struct Scale {
    std::uint16_t bits;
    float value;
  };
  constexpr std::array scales{Scale{0x3400, 0.25F}, Scale{0xbc00, -1.0F},
                              Scale{0x4000, 2.0F}, Scale{0x3c01, 1.0009765625F},
                              Scale{0x0001, 0x1p-24F}};
  std::array<q4_preflight::Block, weight_blocks> blocks{};
  std::array<float, weight_values> decoded{};
  for (std::size_t block = 0; block < blocks.size(); ++block) {
    const auto scale = scales[block % scales.size()];
    blocks[block].scale = scale.bits;
    for (std::size_t lane = 0; lane < 16; ++lane) {
      const auto low = static_cast<int>((block * 3 + lane * 7) % 16);
      const auto high = static_cast<int>((block * 11 + lane * 5) % 16);
      blocks[block].quants[lane] = static_cast<std::uint8_t>(low | (high << 4));
      decoded[block * 32 + lane] = static_cast<float>(low - 8) * scale.value;
      decoded[block * 32 + lane + 16] =
          static_cast<float>(high - 8) * scale.value;
    }
  }

  HYLE_RETURN_IF_ERROR(cuda::status(cudaSetDevice(0)));
  HYLE_ASSIGN_OR_RETURN(auto device, cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context, hyle::compute::Context::create(device));
  HYLE_ASSIGN_OR_RETURN(auto memory, cuda::MemoryResource::create(
                                         context, cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_RETURN_IF_ERROR(queue.reserve_completions(4));
  VendorPlan vendor;
  HYLE_RETURN_IF_ERROR(vendor.initialize(rows, columns, inner));
  HYLE_ASSIGN_OR_RETURN(auto input,
                        mem::Buffer::create(memory.ref(),
                                            static_cast<std::size_t>(rows) *
                                                inner * sizeof(float),
                                            256));
  HYLE_ASSIGN_OR_RETURN(auto weight,
                        mem::Buffer::create(memory.ref(), q4_bytes, 256));
  HYLE_ASSIGN_OR_RETURN(
      auto expanded,
      mem::Buffer::create(memory.ref(), decoded.size() * sizeof(float), 256));
  HYLE_ASSIGN_OR_RETURN(auto output,
                        mem::Buffer::create(memory.ref(),
                                            static_cast<std::size_t>(rows) *
                                                columns * sizeof(float),
                                            256));
  HYLE_ASSIGN_OR_RETURN(
      auto workspace,
      mem::Buffer::create(
          memory.ref(), std::max<std::size_t>(1, vendor.selected.workspaceSize),
          256));
  HYLE_RETURN_IF_ERROR(weight.view().write(std::as_bytes(std::span(blocks))));
  // Explicit initialization-time representation conversion, never replay work.
  HYLE_RETURN_IF_ERROR(
      expanded.view().write(std::as_bytes(std::span(decoded))));
  HYLE_ASSIGN_OR_RETURN(auto a, cuda::pointer<float>(input.view()));
  HYLE_ASSIGN_OR_RETURN(auto q4,
                        cuda::pointer<q4_preflight::Block>(weight.view()));
  HYLE_ASSIGN_OR_RETURN(auto b, cuda::pointer<float>(expanded.view()));
  HYLE_ASSIGN_OR_RETURN(auto y, cuda::pointer<float>(output.view()));
  HYLE_ASSIGN_OR_RETURN(auto scratch,
                        cuda::pointer<std::byte>(workspace.view()));
  HYLE_ASSIGN_OR_RETURN(auto stream, cuda::stream(queue));
  if (q4_preflight::matmul(stream, a, q4, y, rows, columns, 31) !=
      cudaErrorInvalidValue)
    return Status::Internal("invalid Q4 block dimension was not rejected");

  std::vector<float> values(static_cast<std::size_t>(rows) * inner);
  std::vector<float> observed(static_cast<std::size_t>(rows) * columns);
  std::vector<double> expected(observed.size());
  for (int replay = 0; replay < 3; ++replay) {
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(
                      static_cast<int>((i * 13 + replay * 29) % 127) - 63) /
                  32;
    HYLE_RETURN_IF_ERROR(input.view().write(std::as_bytes(std::span(values))));
    for (int row = 0; row < rows; ++row)
      for (int column = 0; column < columns; ++column) {
        double sum = 0;
        for (int k = 0; k < inner; ++k)
          sum += static_cast<double>(values[row * inner + k]) *
                 decoded[column * inner + k];
        expected[row * columns + column] = sum;
      }
    for (bool use_vendor : {false, true}) {
      std::fill(observed.begin(), observed.end(), 12345.0F);
      HYLE_RETURN_IF_ERROR(
          output.view().write(std::as_bytes(std::span(observed))));
      HYLE_ASSIGN_OR_RETURN(
          auto done, cuda::submit(queue, [&](cudaStream_t native) noexcept {
            return use_vendor ? vendor.enqueue(native, a, b, y, scratch)
                              : cuda::status(q4_preflight::matmul(
                                    native, a, q4, y, rows, columns, inner));
          }));
      HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_RETURN_IF_ERROR(
          output.view().read(std::as_writable_bytes(std::span(observed))));
      for (std::size_t i = 0; i < observed.size(); ++i) {
        const double tolerance = 2e-4 * std::max(1.0, std::abs(expected[i]));
        if (!std::isfinite(observed[i]) ||
            std::abs(observed[i] - expected[i]) > tolerance) {
          std::fprintf(stderr, "%s M=%d replay=%d index=%zu: %.9g vs %.12g\n",
                       use_vendor ? "cuBLASLt" : "Q4 native", rows, replay, i,
                       static_cast<double>(observed[i]), expected[i]);
          return Status::DataLoss("Q4 carrier numerical mismatch");
        }
      }
    }
  }
  std::printf("Q4 carrier PASS M=%d N=%d K=%d: Q4=%zu bytes, cold F32=%zu "
              "bytes, vendor workspace=%zu bytes\n",
              rows, columns, inner, q4_bytes, decoded.size() * sizeof(float),
              vendor.selected.workspaceSize);
  return Status::Ok();
}
} // namespace

int main() {
  for (int rows : {1, 7, 33}) {
    const auto status = exercise(rows);
    if (!status.ok()) {
      const auto message = status.message();
      std::fprintf(stderr, "Q4 preflight: %.*s\n",
                   static_cast<int>(message.size()), message.data());
      return 1;
    }
  }
}
