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
     RecordedFlashReadsCurrentPayloadAcross64StepsAndNewGraphOwner) {
  auto device = n::NativeDevice::create();
  ASSERT_TRUE(device) << device.error().message();
  auto active = device->activate();
  auto work = device->create_work();
  ASSERT_TRUE(work) << work.error().message();
  for (int owner : {0, 1}) {
    auto arena =
        n::MetadataArena::for_graph(8, 16, hyle::mem::system_allocator());
    ASSERT_TRUE(arena) << arena.error().message();
    auto *context = arena->get();
    auto *q = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 1, 2, 1);
    auto *k = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
    auto *v = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
    auto *output =
        ggml_flash_attn_ext(context, q, k, v, nullptr, .125F, 0.F, 0.F);
    ASSERT_TRUE(ggml_prec_set_acc(output, GGML_PREC_F32));
    output->op_params[b::kFlashAttentionMathSlot] = b::kFlashAttentionFullF32;
    for (auto *input : {q, k, v})
      ggml_set_input(input);
    // Persistent ready ports live outside gallocr's reusable workspace.
    auto imports =
        std::make_shared<std::array<std::shared_ptr<ggml_backend_buffer>, 3>>();
    const std::array<ggml_tensor *, 3> inputs{q, k, v};
    for (std::size_t port = 0; port < inputs.size(); ++port) {
      auto allocation = device->allocate_buffer(ggml_nbytes(inputs[port]));
      ASSERT_TRUE(allocation) << allocation.error().message();
      (*imports)[port] = std::shared_ptr<ggml_backend_buffer>(
          *allocation, [owner = *device](ggml_backend_buffer *buffer) {
            auto active = owner.activate();
            ggml_backend_buffer_free(buffer);
          });
      ASSERT_EQ(ggml_backend_tensor_alloc(
                    (*imports)[port].get(), inputs[port],
                    ggml_backend_buffer_get_base((*imports)[port].get())),
                GGML_STATUS_SUCCESS);
    }
    ggml_set_output(output);
    auto *graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);
    auto native =
        n::NativeGraph::allocate(*device, std::move(*arena), graph, imports);
    ASSERT_TRUE(native) << native.error().message();
    const std::vector<float> query(128, 0.F), keys(31 * 128, 0.F);
    ggml_backend_tensor_set(q, query.data(), 0, query.size() * sizeof(float));
    ggml_backend_tensor_set(k, keys.data(), 0, keys.size() * sizeof(float));
    for (int step = 0; step < 64; ++step) {
      const auto expected = float(owner * 64 + step) * .03125F + .125F;
      const std::vector<float> values(31 * 128, expected);
      ggml_backend_tensor_set(v, values.data(), 0,
                              values.size() * sizeof(float));
      if (step == 32) {
        ASSERT_TRUE(work->invalidate_recording().ok());
        EXPECT_FALSE(work->recording_ready());
      }
      auto submitted = work->submit(native->get(), native->lease());
      ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
      EXPECT_FALSE(work->recording_ready());
      auto completed = work->wait();
      ASSERT_TRUE(completed.status.ok()) << completed.status.message();
      ASSERT_TRUE(work->recording_ready())
          << "fixed original flash recording was not retained";
      std::array<float, 128> actual{};
      ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
      for (const auto value : actual) {
        ASSERT_TRUE(std::isfinite(value));
        EXPECT_NEAR(value, expected, 1.e-5F);
      }
    }
  }
}

TEST(GgmlVulkanAttention,
     TaggedNativeDomainRejectsUnqualifiedDepthBeforeExecution) {
  auto device = n::NativeDevice::create();
  ASSERT_TRUE(device) << device.error().message();
  auto active = device->activate();
  for (const auto depth : {8, 72, 512}) {
    auto arena =
        n::MetadataArena::for_graph(8, 16, hyle::mem::system_allocator());
    ASSERT_TRUE(arena) << arena.error().message();
    auto *context = arena->get();
    auto *q = ggml_new_tensor_4d(context, GGML_TYPE_F32, depth, 1, 2, 1);
    auto *k = ggml_new_tensor_4d(context, GGML_TYPE_F32, depth, 31, 2, 1);
    auto *v = ggml_new_tensor_4d(context, GGML_TYPE_F32, depth, 31, 2, 1);
    auto *output =
        ggml_flash_attn_ext(context, q, k, v, nullptr, 0.125F, 0.F, 0.F);
    ASSERT_TRUE(ggml_prec_set_acc(output, GGML_PREC_F32));
    output->op_params[b::kFlashAttentionMathSlot] = b::kFlashAttentionFullF32;
    for (auto *input : {q, k, v})
      ggml_set_input(input);
    ggml_set_output(output);
    auto *graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);
    auto rejected = n::NativeGraph::allocate(*device, std::move(*arena), graph);
    EXPECT_FALSE(rejected) << "native full-F32 tag admitted depth " << depth;
    if (!rejected)
      EXPECT_EQ(rejected.error().code(), hyle::StatusCode::Unimplemented);
  }
}

TEST(GgmlVulkanAttention,
     TaggedColdResourceExhaustionRollsBackOwnedGraphStorage) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 0);
  n::NativeOptions options;
  options.quota = &quota;
  auto device = n::NativeDevice::create(options);
  ASSERT_TRUE(device) << device.error().message();
  auto active = device->activate();
  auto arena =
      n::MetadataArena::for_graph(8, 16, hyle::mem::system_allocator());
  ASSERT_TRUE(arena) << arena.error().message();
  auto *context = arena->get();
  auto *q = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 1, 2, 1);
  auto *k = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
  auto *v = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
  auto *output =
      ggml_flash_attn_ext(context, q, k, v, nullptr, 0.125F, 0.F, 0.F);
  ASSERT_TRUE(ggml_prec_set_acc(output, GGML_PREC_F32));
  output->op_params[b::kFlashAttentionMathSlot] = b::kFlashAttentionFullF32;
  for (auto *input : {q, k, v})
    ggml_set_input(input);
  ggml_set_output(output);
  auto *graph = ggml_new_graph_custom(context, 16, false);
  ggml_build_forward_expand(graph, output);
  auto rejected = n::NativeGraph::allocate(*device, std::move(*arena), graph);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code(), hyle::StatusCode::ResourceExhausted);
  EXPECT_EQ(quota.used(), 0U);
}

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
