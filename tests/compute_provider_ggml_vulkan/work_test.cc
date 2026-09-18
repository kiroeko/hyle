#include "native_device.h"
#include "native_graph.h"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "ggml-backend.h"
#include "ggml.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {
using hyle::compute::ggml_vulkan::detail::MetadataArena;
using hyle::compute::ggml_vulkan::detail::NativeDevice;
using hyle::compute::ggml_vulkan::detail::NativeGraph;

struct GraphStorage {
  NativeDevice device;
  NativeGraph storage;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *weight = nullptr;
  ggml_tensor *output = nullptr;
};

hyle::Result<std::shared_ptr<GraphStorage>> make_graph(const NativeDevice &device) {
  auto result = std::make_shared<GraphStorage>();
  result->device = device;
  auto active = device.activate();
  HYLE_ASSIGN_OR_RETURN(auto arena, MetadataArena::create(
      24 * ggml_tensor_overhead() + ggml_graph_overhead_custom(24, false),
      hyle::mem::system_allocator()));
  auto *context = arena.get();
  result->input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2048);
  result->weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2048);
  auto *norm = ggml_rms_norm(context, result->input, 1e-6f);
  auto *scaled = ggml_mul(context, norm, result->weight);
  result->output = ggml_add(context, ggml_silu(context, scaled), result->input);
  ggml_set_input(result->input);
  ggml_set_input(result->weight);
  ggml_set_output(result->output);
  result->graph = ggml_new_graph_custom(context, 24, false);
  ggml_build_forward_expand(result->graph, result->output);
  HYLE_ASSIGN_OR_RETURN(result->storage,
      NativeGraph::allocate(device, std::move(arena), result->graph));
  return result;
}

TEST(GgmlVulkanWork, RepeatedIndependentSlotsRetainTheirResourceLeases) {
#if defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
               SEM_NOOPENFILEERRORBOX);
#endif
  auto device = NativeDevice::create();
  ASSERT_TRUE(device);
  auto first = device->create_work();
  auto second = device->create_work();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto a = make_graph(*device);
  auto b = make_graph(*device);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  const std::array frames{*a, *b};
  const std::array slots{&*first, &*second};
  std::array<std::vector<float>, 2> inputs, weights;
  for (std::size_t slot = 0; slot < 2; ++slot) {
    inputs[slot].resize(2048);
    weights[slot].assign(2048, 0.75f + 0.25f * static_cast<float>(slot));
  }
  for (int iteration = 0; iteration < 3; ++iteration) {
    std::array<std::weak_ptr<std::shared_ptr<GraphStorage>>, 2> leases;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      for (std::size_t i = 0; i < 2048; ++i)
        inputs[slot][i] = static_cast<float>(static_cast<int>(i % 31) - 15) *
                         (0.1f + 0.05f * iteration + 0.02f * static_cast<float>(slot));
      {
        auto active = device->activate();
        ggml_backend_tensor_set(frames[slot]->input, inputs[slot].data(), 0, 2048 * sizeof(float));
        ggml_backend_tensor_set(frames[slot]->weight, weights[slot].data(), 0, 2048 * sizeof(float));
      }
    }
    // Finish synchronous uploads before admitting either region, so uploading
    // the second input cannot accidentally wait for the first execution.
    for (std::size_t slot = 0; slot < 2; ++slot) {
      auto lease = std::make_shared<std::shared_ptr<GraphStorage>>(frames[slot]);
      leases[slot] = lease;
      const auto submitted = slots[slot]->submit(frames[slot]->graph, std::move(lease));
      ASSERT_TRUE(submitted.status.ok());
      EXPECT_TRUE(submitted.pending);
      EXPECT_FALSE(leases[slot].expired());
      const auto rejected = slots[slot]->submit(frames[slot]->graph, frames[slot]);
      EXPECT_FALSE(rejected.status.ok());
      EXPECT_TRUE(rejected.pending);
    }
    // Both slots have been admitted before either owning observation.
    for (std::size_t slot = 0; slot < 2; ++slot) {
      const auto done = slots[slot]->wait();
      ASSERT_TRUE(done.status.ok());
      EXPECT_FALSE(done.pending);
      EXPECT_TRUE(leases[slot].expired());
      std::vector<float> actual(2048);
      auto active = device->activate();
      ggml_backend_tensor_get(frames[slot]->output, actual.data(), 0, actual.size() * sizeof(float));
      double sum = 0;
      for (const auto value : inputs[slot]) sum += static_cast<double>(value) * value;
      const double inverse = 1.0 / std::sqrt(sum / 2048 + 1e-6);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const double z = inputs[slot][i] * inverse * weights[slot][i];
        const double expected = z / (1 + std::exp(-z)) + inputs[slot][i];
        ASSERT_NEAR(actual[i], expected, 6e-6 * (1 + std::abs(expected)));
      }
      EXPECT_FALSE(slots[slot]->poll().pending);
    }
  }
}

TEST(GgmlVulkanWork, MovedPendingOwnerDrainsBeforeDroppingItsLastGraphLease) {
  auto device = NativeDevice::create();
  ASSERT_TRUE(device);
  auto graph = make_graph(*device);
  ASSERT_TRUE(graph);
  std::weak_ptr<GraphStorage> lifetime = *graph;
  {
    auto work = device->create_work();
    ASSERT_TRUE(work);
    auto active = device->activate();
    const std::vector<float> values(2048, 0.5f);
    ggml_backend_tensor_set((*graph)->input, values.data(), 0, 2048 * sizeof(float));
    ggml_backend_tensor_set((*graph)->weight, values.data(), 0, 2048 * sizeof(float));
    ASSERT_TRUE(work->submit((*graph)->graph, *graph).status.ok());
    graph->reset();
    EXPECT_FALSE(lifetime.expired());
    auto moved = std::move(*work);
    EXPECT_FALSE(work->poll().status.ok());
    // Emergency destruction observes native retirement before the graph and
    // payload lease disappear. Ordinary public abandonment retains this owner.
  }
  EXPECT_TRUE(lifetime.expired());
}
} // namespace
