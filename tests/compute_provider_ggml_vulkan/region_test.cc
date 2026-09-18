#include "metadata_arena.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {
using hyle::compute::ggml_vulkan::detail::MetadataArena;

// Internal admission: the CMake-owned complete upstream GPU implementation,
// independent scalar arithmetic, and the HYLE metadata owner. This is not an
// ordinary HYLE Direct/Graph acceptance or a performance comparison.
TEST(GgmlVulkanRegion, ModelWidthNormGateResidualWithChangingInputs) {
#if defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
               SEM_NOOPENFILEERRORBOX);
#endif
  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
      ggml_backend_vk_init(0), &ggml_backend_free);
  ASSERT_NE(backend, nullptr);
  constexpr std::size_t width = 2048;
  constexpr float epsilon = 1e-6f;
  for (const std::size_t rows : {1u, 17u, 128u}) {
    SCOPED_TRACE(rows);
    const auto bytes =
        32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false);
    auto arena = MetadataArena::create(bytes, hyle::mem::system_allocator());
    ASSERT_TRUE(arena);
    auto *context = arena->get();
    auto *input = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, rows);
    auto *weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
    auto *norm = ggml_rms_norm(context, input, epsilon);
    auto *scaled = ggml_mul(context, norm, weight);
    auto *gate = ggml_silu(context, scaled);
    auto *result = ggml_add(context, gate, input);
    ggml_set_input(input);
    ggml_set_input(weight);
    ggml_set_output(result);
    auto *graph = ggml_new_graph_custom(context, 32, false);
    ggml_build_forward_expand(graph, result);
    for (auto *tensor : {norm, scaled, gate, result})
      ASSERT_TRUE(ggml_backend_supports_op(backend.get(), tensor));
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
        storage(ggml_backend_alloc_ctx_tensors(context, backend.get()),
                &ggml_backend_buffer_free);
    ASSERT_NE(storage, nullptr);
    std::vector<float> x(width * rows), w(width), actual(width * rows);
    for (std::size_t i = 0; i < width; ++i)
      w[i] = 0.5f + static_cast<float>(i % 19) / 19.0f;
    ggml_backend_tensor_set(weight, w.data(), 0, w.size() * sizeof(float));
    for (std::size_t iteration = 0; iteration < 3; ++iteration) {
      SCOPED_TRACE(iteration);
      for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(
                   static_cast<int>((i * 13 + iteration * 17) % 193) - 96) /
               37.0f;
      ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
      ASSERT_EQ(ggml_backend_graph_compute(backend.get(), graph),
                GGML_STATUS_SUCCESS);
      ggml_backend_tensor_get(result, actual.data(), 0,
                              actual.size() * sizeof(float));
      for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0;
        for (std::size_t column = 0; column < width; ++column) {
          const double value = x[row * width + column];
          sum += value * value;
        }
        const double inverse = 1 / std::sqrt(sum / width + epsilon);
        for (std::size_t column = 0; column < width; ++column) {
          const auto index = row * width + column;
          const double z = x[index] * inverse * w[column];
          const double expected = z / (1 + std::exp(-z)) + x[index];
          ASSERT_NEAR(actual[index], expected, 6e-6 * (1 + std::abs(expected)))
              << "index=" << index;
        }
      }
    }
  }
}
} // namespace
