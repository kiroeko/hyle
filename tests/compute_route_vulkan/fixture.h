#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <span>
#include <vector>

#include "hyle/compute/vulkan.h"

namespace vulkan_test {
namespace c = hyle::compute;
namespace v = c::vulkan;

inline c::TensorDesc dense(std::initializer_list<std::int64_t> dimensions) {
  return *c::TensorDesc::dense(c::DataType::kFloat32,
                               {dimensions.begin(), dimensions.size()});
}

inline hyle::Result<v::Context>
context(hyle::mem::QuotaAllocator *quota = nullptr) {
  // Test execution opts into a real installed validation layer explicitly.
  bool enabled = false;
#if defined(_WIN32)
  char *validation = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&validation, &length, "HYLE_VULKAN_VALIDATION") == 0 && validation)
    enabled = validation[0] == '1';
  std::free(validation);
#else
  const auto *validation = std::getenv("HYLE_VULKAN_VALIDATION");
  enabled = validation && validation[0] == '1';
#endif
  return v::Context::create({.validation = enabled, .memory_quota = quota});
}

inline std::vector<float> data(std::size_t count, int seed) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    // Non-degenerate signed values; no constant rows or identity matrices.
    const auto residue =
        (index * 37 + static_cast<std::size_t>(seed) * 19) % 103;
    values[index] = static_cast<float>(static_cast<int>(residue) - 51) / 64.f;
  }
  return values;
}

inline hyle::Result<v::Tensor> upload(const v::Context &context,
                                      const c::TensorDesc &description,
                                      std::span<const float> values) {
  return context.upload(description, std::as_bytes(values));
}

inline hyle::Result<std::vector<float>> read(const v::Context &context,
                                             const v::Tensor &tensor) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, tensor.description().required_bytes());
  std::vector<float> result(bytes / sizeof(float));
  HYLE_RETURN_IF_ERROR(context.download(
      tensor, std::as_writable_bytes(std::span<float>(result))));
  return result;
}

inline std::vector<float> matmul(std::span<const float> a,
                                 std::span<const float> b, int m, int k,
                                 int n) {
  std::vector<float> output(static_cast<std::size_t>(m) * n);
  for (int row = 0; row < m; ++row)
    for (int column = 0; column < n; ++column) {
      double sum = 0;
      for (int inner = 0; inner < k; ++inner)
        sum += static_cast<double>(a[row * k + inner]) * b[inner * n + column];
      output[row * n + column] = static_cast<float>(sum);
    }
  return output;
}

struct Ffn final {
  static constexpr int k = 67, hidden = 37, output = 13;
  std::vector<float> gate = data(k * hidden, 1);
  std::vector<float> up = data(k * hidden, 2);
  std::vector<float> down = data(hidden * output, 3);

  hyle::Result<std::array<v::Tensor, 3>>
  constants(const v::Context &context) const {
    HYLE_ASSIGN_OR_RETURN(auto g, upload(context, dense({k, hidden}), gate));
    HYLE_ASSIGN_OR_RETURN(auto u, upload(context, dense({k, hidden}), up));
    HYLE_ASSIGN_OR_RETURN(auto d,
                          upload(context, dense({hidden, output}), down));
    return std::array{std::move(g), std::move(u), std::move(d)};
  }
  std::vector<float> oracle(std::span<const float> x, int rows) const {
    auto g = matmul(x, gate, rows, k, hidden);
    const auto u = matmul(x, up, rows, k, hidden);
    for (std::size_t index = 0; index < g.size(); ++index)
      g[index] = g[index] / (1.f + std::exp(-g[index])) * u[index];
    return matmul(g, down, rows, hidden, output);
  }
};

inline hyle::Result<v::Kernel> affine(const v::Context &context) {
  std::ifstream stream(HYLE_VULKAN_TEST_SHADER,
                       std::ios::binary | std::ios::ate);
  if (!stream)
    return std::unexpected(hyle::Status::NotFound("test SPIR-V file"));
  const auto size = static_cast<std::size_t>(stream.tellg());
  if (size == 0 || size % 4 != 0)
    return std::unexpected(hyle::Status::DataLoss("test SPIR-V word size"));
  std::vector<std::uint32_t> words(size / 4);
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(words.data()),
                   static_cast<std::streamsize>(size)))
    return std::unexpected(hyle::Status::DataLoss("test SPIR-V truncated"));
  return v::Kernel::create(context, {words, 2, 4, {}});
}

inline std::vector<std::byte> count_push(std::uint32_t count) {
  std::vector<std::byte> bytes(sizeof(count));
  std::memcpy(bytes.data(), &count, sizeof(count));
  return bytes;
}

} // namespace vulkan_test
