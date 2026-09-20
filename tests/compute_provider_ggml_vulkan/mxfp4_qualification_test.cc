#include "native_graph.h"

#include <array>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "ggml-backend.h"
#include "ggml.h"

namespace {
using hyle::compute::ggml_vulkan::detail::MetadataArena;
using hyle::compute::ggml_vulkan::detail::NativeDevice;
using hyle::compute::ggml_vulkan::detail::NativeGraph;

// Frozen Q-VK logical inputs and NumPy double -> F16 oracle, variants 0/1.
// This qualifies the original native provider only, not an ordinary HYLE
// quantized-operation entry or complete model coverage. The 17-byte blocks
// change packing only: UE8 exponent, then low/high 16-element halves.
constexpr std::array<std::uint16_t, 17> kActivation{
    0xbc00, 0xbb00, 0xba00, 0xb900, 0xb800, 0xb600, 0xb400, 0xb000,
    0x0000, 0x3000, 0x3400, 0x3600, 0x3800, 0x3900, 0x3a00, 0x3b00, 0x3c00};
constexpr std::array<std::array<std::uint16_t, 32>, 2> kExpected{{
    {0x4dcd, 0x5389, 0xd112, 0x52a3, 0x51cd, 0x4dcd, 0x5389, 0xd112,
     0x52a3, 0x51cd, 0x4dcd, 0x5389, 0xd112, 0x52a3, 0x51cd, 0x4dcd,
     0xce38, 0xd42b, 0xcabe, 0xcdfc, 0xd238, 0xce38, 0xd42b, 0xcabe,
     0xcdfc, 0xd238, 0xce38, 0xd42b, 0xcabe, 0xcdfc, 0xd238, 0xce38},
    {0xd334, 0xcf34, 0xcab8, 0xcf9d, 0x51f0, 0xd334, 0xcf34, 0xcab8,
     0xcf9d, 0x51f0, 0xd334, 0xcf34, 0xcab8, 0xcf9d, 0x51f0, 0xd334,
     0x50e2, 0x4ce2, 0xc5bc, 0x516c, 0xc628, 0x50e2, 0x4ce2, 0xc5bc,
     0x516c, 0xc628, 0x50e2, 0x4ce2, 0xc5bc, 0x516c, 0xc628, 0x50e2}
}};

TEST(GgmlVulkanQualification, Mxfp4PreservesFrozenQvkValues) {
  auto device = NativeDevice::create();
  ASSERT_TRUE(device) << device.error().message();
  for (std::size_t variant = 0; variant < kExpected.size(); ++variant) {
    SCOPED_TRACE(variant);
    auto arena = MetadataArena::create(
        16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
        hyle::mem::system_allocator());
    ASSERT_TRUE(arena);
    auto *context = arena->get();
    auto *weights = ggml_new_tensor_2d(context, GGML_TYPE_MXFP4, 128, 16);
    auto *activation = ggml_new_tensor_2d(context, GGML_TYPE_F16, 128, 2);
    auto *product = ggml_mul_mat(context, weights, activation);
    ASSERT_TRUE(ggml_prec_set_acc(product, GGML_PREC_F32));
    auto *output = ggml_cast(context, product, GGML_TYPE_F16);
    ggml_set_input(weights);
    ggml_set_input(activation);
    ggml_set_output(output);
    // Persistent inputs must live outside gallocr's reusable graph workspace.
    // GGML_TENSOR_FLAG_INPUT reserves an input early; it does not keep it live
    // after its last consumer. This mirrors the provider's external port owner.
    auto imports = std::make_shared<
        std::array<std::shared_ptr<ggml_backend_buffer>, 2>>();
    const std::array<ggml_tensor *, 2> inputs{weights, activation};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto allocation = device->allocate_buffer(ggml_nbytes(inputs[i]));
      ASSERT_TRUE(allocation);
      (*imports)[i] = std::shared_ptr<ggml_backend_buffer>(*allocation,
          [owner = *device](ggml_backend_buffer *buffer) {
            auto active = owner.activate();
            ggml_backend_buffer_free(buffer);
          });
      auto active = device->activate();
      ASSERT_EQ(ggml_backend_tensor_alloc((*imports)[i].get(), inputs[i],
          ggml_backend_buffer_get_base((*imports)[i].get())), GGML_STATUS_SUCCESS);
    }
    auto *graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);
    auto storage = NativeGraph::allocate(*device, std::move(*arena), graph, imports);
    ASSERT_TRUE(storage) << storage.error().message();
    std::array<std::uint8_t, 16 * 4 * 17> packed{};
    std::array<std::uint16_t, 256> input{};
    for (std::size_t block = 0; block < 64; ++block) {
      packed[block * 17] = std::uint8_t((block + variant) % 5 + 125);
      for (std::size_t lane = 0; lane < 16; ++lane) {
        const auto low = ((block * 32 + lane) * 3 + variant) % 16;
        const auto high = ((block * 32 + lane + 16) * 3 + variant) % 16;
        packed[block * 17 + lane + 1] = std::uint8_t(low | (high << 4));
      }
    }
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = kActivation[(i * 7 + variant) % kActivation.size()];
    ASSERT_EQ(ggml_nbytes(weights), packed.size());
    {
      auto active = device->activate();
      ggml_backend_tensor_set(weights, packed.data(), 0, sizeof(packed));
      ggml_backend_tensor_set(activation, input.data(), 0, sizeof(input));
    }
    auto work = device->create_work();
    ASSERT_TRUE(work);
    for (int repeat = 0; repeat < 3; ++repeat) {
      ASSERT_TRUE(work->submit(storage->get(), storage->lease()).status.ok());
      ASSERT_TRUE(work->wait().status.ok());
      std::array<std::uint16_t, 32> actual{};
      auto active = device->activate();
      ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
      EXPECT_EQ(actual, kExpected[variant]);
      decltype(packed) retained_weights{};
      decltype(input) retained_activation{};
      ggml_backend_tensor_get(weights, retained_weights.data(), 0, sizeof(packed));
      ggml_backend_tensor_get(activation, retained_activation.data(), 0, sizeof(input));
      EXPECT_EQ(retained_weights, packed);
      EXPECT_EQ(retained_activation, input);
    }
  }
}
} // namespace
