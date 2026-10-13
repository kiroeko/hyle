#include "native_graph.h"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include <gtest/gtest.h>

namespace {
namespace n = hyle::compute::ggml_vulkan::detail;
namespace b = hyle::compute::ggml_vulkan::bridge;

TEST(GgmlVulkanAttention,
     FullF32AndOriginalScalarPipelinesStayDistinctInBothOrders) {
  // N=1/MHA makes both modes select the same scalar tiles. Only the precision
  // key can distinguish the original half-internal shader from full F32.
  for (bool strict_first : {false, true}) {
    auto device = n::NativeDevice::create();
    ASSERT_TRUE(device) << device.error().message();
    auto active = device->activate();
    std::vector<float> normal_before;
    for (bool strict :
         {strict_first, !strict_first, strict_first, !strict_first}) {
      auto arena =
          n::MetadataArena::for_graph(8, 16, hyle::mem::system_allocator());
      ASSERT_TRUE(arena) << arena.error().message();
      auto *context = arena->get();
      auto *query = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 1, 2, 1);
      auto *keys = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
      auto *values = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
      auto *output = ggml_flash_attn_ext(context, query, keys, values, nullptr,
                                         0.125F, 0.F, 0.F);
      ASSERT_TRUE(ggml_prec_set_acc(output, GGML_PREC_F32));
      if (strict)
        output->op_params[b::kFlashAttentionMathSlot] =
            b::kFlashAttentionFullF32;
      for (auto *input : {query, keys, values})
        ggml_set_input(input);
      ggml_set_output(output);
      auto *graph = ggml_new_graph_custom(context, 16, false);
      ggml_build_forward_expand(graph, output);
      auto native = n::NativeGraph::allocate(*device, std::move(*arena), graph);
      ASSERT_TRUE(native) << native.error().message();
      auto work = device->create_work();
      ASSERT_TRUE(work) << work.error().message();
      const std::vector<float> q(128, 0.F), k(31 * 128, 0.F),
          v(31 * 128, 131072.F);
      ggml_backend_tensor_set(query, q.data(), 0, q.size() * sizeof(float));
      ggml_backend_tensor_set(keys, k.data(), 0, k.size() * sizeof(float));
      ggml_backend_tensor_set(values, v.data(), 0, v.size() * sizeof(float));
      ASSERT_TRUE(work->submit(native->get(), native->lease()).status.ok());
      ASSERT_TRUE(work->wait().status.ok());
      std::vector<float> result(128);
      ggml_backend_tensor_get(output, result.data(), 0,
                              result.size() * sizeof(float));
      if (strict) {
        for (const auto value : result) {
          ASSERT_TRUE(std::isfinite(value));
          EXPECT_NEAR(value, 131072.F, 0.1F);
        }
      } else if (normal_before.empty()) {
        normal_before = result;
      } else {
        EXPECT_EQ(std::memcmp(result.data(), normal_before.data(),
                              result.size() * sizeof(float)),
                  0);
      }
    }
  }
}
} // namespace
