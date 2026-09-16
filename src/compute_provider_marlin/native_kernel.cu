// Kernel selection/cache arithmetic adapted from vLLM marlin.cu at
// 51918e252af4dfff381001f42b7ebe6a0454501b, Apache-2.0.
// Copyright (C) Marlin.2024 Elias Frantar; modifications by Neural Magic.
// Device arithmetic remains in the pinned third_party/marlin source subset.
#include "native_kernel.h"

#include <algorithm>
#include <array>

#define MARLIN_NAMESPACE_NAME hyle_marlin_51918
#include "kernel.h"
#include "marlin_template.h"
#include "repack_kernel.cuh"

namespace hyle::compute::marlin::native {
namespace {

struct Config final {
  int inner;
  int columns;
  int threads;
};
constexpr std::array<Config, 4> kConfigs{
    {{128, 128, 256}, {64, 256, 256}, {64, 128, 128}, {128, 64, 128}}};

int cache_bytes(Config config, int row_blocks) noexcept {
  constexpr int stages = 4;
  const int rows = row_blocks * 16;
  const int activation = stages * rows * config.inner * 2;
  const int weight = stages * config.inner * config.columns / 2;
  const int reduction = rows * (config.columns + 8) * 2;
  const int bias = config.columns * 2;
  const int temporary =
      std::max(std::max(weight, reduction), std::min(weight, reduction) + bias);
  const int scales = ((config.inner + 127) / 128) * config.columns * 2 * stages;
  return temporary + activation + scales;
}

template <int RowBlocks, bool Small> const void *kernel(int index) noexcept {
  constexpr auto a = vllm::kFloat16.id();
  constexpr auto b = vllm::kU4B8.id();
  switch (index) {
  case 0:
    return reinterpret_cast<const void *>(
        hyle_marlin_51918::Marlin<a, b, a, a, 256, RowBlocks, 8, 8, Small, 4, 8,
                                  false>);
  case 1:
    return reinterpret_cast<const void *>(
        hyle_marlin_51918::Marlin<a, b, a, a, 256, RowBlocks, 16, 4, Small, 4,
                                  8, false>);
  case 2:
    return reinterpret_cast<const void *>(
        hyle_marlin_51918::Marlin<a, b, a, a, 128, RowBlocks, 8, 4, Small, 4, 8,
                                  false>);
  default:
    return reinterpret_cast<const void *>(
        hyle_marlin_51918::Marlin<a, b, a, a, 128, RowBlocks, 4, 8, Small, 4, 8,
                                  false>);
  }
}

} // namespace

cudaError_t prepare(int rows, int columns, int inner, int multiprocessors,
                    int maximum_shared_bytes, Program *output) noexcept {
  if (output == nullptr || rows < 1 || rows > 64 || columns < 64 ||
      columns % 64 != 0 || inner < 128 || inner % 128 != 0 ||
      multiprocessors < 1 || maximum_shared_bytes <= 512)
    return cudaErrorInvalidValue;
  *output = {};
  const int row_blocks = (rows + 15) / 16;
  const std::array priorities{row_blocks > 1 ? 1 : 0, 2, 3};
  const auto valid = [&](int index, int limit) {
    const auto config = kConfigs[index];
    return inner % config.inner == 0 && columns % config.columns == 0 &&
           cache_bytes(config, row_blocks) <= limit;
  };
  int selected = -1;
  for (int index : priorities) {
    if (valid(index, maximum_shared_bytes - 512)) {
      selected = index;
      break;
    }
  }
  if (selected < 0)
    return cudaErrorNotSupported;
  // Preserve the upstream small-grid refinement; this is cold rule matching,
  // not a benchmark or a runtime timing search.
  if (columns / kConfigs[selected].columns <= multiprocessors / 4 &&
      valid(3, maximum_shared_bytes))
    selected = 3;
  const void *function = nullptr;
  switch (row_blocks) {
  case 1:
    function =
        rows <= 8 ? kernel<1, true>(selected) : kernel<1, false>(selected);
    break;
  case 2:
    function = kernel<2, false>(selected);
    break;
  case 3:
    function = kernel<3, false>(selected);
    break;
  default:
    function = kernel<4, false>(selected);
    break;
  }
  const auto configured = cudaFuncSetAttribute(
      function, cudaFuncAttributeMaxDynamicSharedMemorySize,
      maximum_shared_bytes);
  if (configured != cudaSuccess)
    return configured;
  const std::size_t reduction_bytes =
      static_cast<std::size_t>(multiprocessors) * row_blocks * 16 * 256 *
      sizeof(float);
  const std::size_t locks_offset = (reduction_bytes + 255) & ~std::size_t{255};
  *output = {function,
             rows,
             columns,
             inner,
             multiprocessors,
             kConfigs[selected].threads,
             maximum_shared_bytes,
             locks_offset,
             locks_offset +
                 static_cast<std::size_t>(multiprocessors) * sizeof(int)};
  return cudaSuccess;
}

cudaError_t launch(const Program &program, const void *activation,
                   const void *weights, const void *scales, void *output,
                   void *workspace, cudaStream_t stream) noexcept {
  const auto *a = static_cast<const int4 *>(activation);
  const auto *b = static_cast<const int4 *>(weights);
  auto *c = static_cast<int4 *>(output);
  auto *temporary = static_cast<int4 *>(workspace);
  const int4 *bias = nullptr;
  const float *activation_scale = nullptr;
  const auto *weight_scale = static_cast<const int4 *>(scales);
  const float *global_scale = nullptr;
  const int4 *zero_points = nullptr;
  int rows = program.rows, columns = program.columns, inner = program.inner;
  int leading = inner;
  auto *locks = reinterpret_cast<int *>(static_cast<std::byte *>(workspace) +
                                        program.locks_offset);
  bool has_bias = false, atomic_add = false, fp32_reduce = true;
  int shared = program.shared_bytes;
  void *arguments[] = {&a,
                       &b,
                       &c,
                       &temporary,
                       &bias,
                       &activation_scale,
                       &weight_scale,
                       &global_scale,
                       &zero_points,
                       &rows,
                       &columns,
                       &inner,
                       &leading,
                       &locks,
                       &has_bias,
                       &atomic_add,
                       &fp32_reduce,
                       &shared};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks),
                          dim3(program.threads), arguments,
                          static_cast<std::size_t>(shared), stream);
}

cudaError_t prepare_repack(int maximum_shared_bytes) noexcept {
  return cudaFuncSetAttribute(
      hyle_marlin_51918::gptq_marlin_repack_kernel<256, 4, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, maximum_shared_bytes);
}

cudaError_t repack(const std::uint32_t *source, void *destination, int inner,
                   int columns, int blocks, int shared_bytes,
                   cudaStream_t stream) noexcept {
  hyle_marlin_51918::gptq_marlin_repack_kernel<256, 4, false>
      <<<blocks, 256, shared_bytes, stream>>>(
          source, static_cast<std::uint32_t *>(destination), inner, columns);
  return cudaGetLastError();
}

} // namespace hyle::compute::marlin::native
