#include "fixture.h"
#include "vulkan_internal.h"
#include "../compute_provider_ggml_vulkan/native_graph.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ggml-backend.h"
#include "ggml.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace p = c::ggml_vulkan::detail;
namespace t = vulkan_test;

// This proves the actual native boundary using an ordinary HYLE producer and
// readback. The private middle region is not a managed Program acceptance.
TEST(VulkanProviderBridge, RouteProducerAndGgmlRegionShareOneDevice) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 32 * 1024 * 1024);
  {
    auto context = t::context(&quota);
    ASSERT_TRUE(context) << context.error().message();
    const auto &route = *v::detail::Access::runtime(*context)->device;
    const auto provider = route.provider;
    const auto &native = provider.native();
    EXPECT_EQ(context->native_device().instance, native.instance);
    EXPECT_EQ(context->native_device().physical_device, native.physical);
    EXPECT_EQ(context->native_device().device, native.device);
    EXPECT_EQ(context->native_device().queue_family, native.queue_family);
    EXPECT_EQ(route.queue, native.queue);
    EXPECT_EQ(context->environment().storage_update_after_bind,
              native.storage_update_after_bind);
    constexpr int width = 2048, rows = 17;
    constexpr float epsilon = 1e-6f;
    const auto description = t::dense({rows, width});
    const auto host = t::data(width * rows, 3);
    auto input = t::upload(*context, description, host);
    ASSERT_TRUE(input);
    auto program = v::prepare(*context, c::Unary{c::UnaryKind::kNeg}, description);
    ASSERT_TRUE(program) << program.error().message();
    auto invocation = program->create_invocation();
    ASSERT_TRUE(invocation);
    ASSERT_TRUE(invocation->bind(*input).ok());
    auto submitted = invocation->submit();
    ASSERT_TRUE(submitted);
    auto produced = submitted->wait();
    ASSERT_TRUE(produced);
    const auto &source = (*produced)->tensors()[0];

    auto active = provider.activate();
    auto arena = p::MetadataArena::create(
        16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
        hyle::mem::system_allocator());
    ASSERT_TRUE(arena);
    auto *x = ggml_new_tensor_2d(arena->get(), GGML_TYPE_F32, width, rows);
    auto *normalized = ggml_rms_norm(arena->get(), x, epsilon);
    ggml_set_input(x);
    ggml_set_output(normalized);
    auto *graph = ggml_new_graph_custom(arena->get(), 16, false);
    ggml_build_forward_expand(graph, normalized);
    const auto capacity = static_cast<std::size_t>(source.native_offset()) +
                          *description.required_bytes();
    // A regular HYLE buffer has storage/transfer usage, no BDA requirement.
    auto imported = provider.borrow(source.native_buffer(), capacity,
                                    source.native_owner());
    ASSERT_TRUE(imported) << imported.error().message();
    std::shared_ptr<ggml_backend_buffer> borrowed(*imported,
        [provider](ggml_backend_buffer *buffer) {
          auto active = provider.activate();
          ggml_backend_buffer_free(buffer);
        });
    auto *base = static_cast<std::byte *>(ggml_backend_buffer_get_base(borrowed.get()));
    ASSERT_EQ(ggml_backend_tensor_alloc(borrowed.get(), x,
        base + source.native_offset()), GGML_STATUS_SUCCESS);
    auto storage = p::NativeGraph::allocate(provider, std::move(*arena), graph, borrowed);
    ASSERT_TRUE(storage) << storage.error().message();
    auto work = provider.create_work();
    ASSERT_TRUE(work);
    auto accepted = work->submit(graph, storage->lease());
    ASSERT_TRUE(accepted.status.ok()) << accepted.status.message();
    ASSERT_TRUE(accepted.pending);
    auto completed = work->wait();
    ASSERT_FALSE(completed.pending);
    ASSERT_TRUE(completed.status.ok()) << completed.status.message();
    const auto offset = reinterpret_cast<std::uintptr_t>(normalized->data) -
        reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(normalized->buffer));
    auto result = context->borrow_ready(description,
        provider.buffer_handle(normalized->buffer), offset,
        ggml_backend_buffer_get_size(normalized->buffer) - offset, storage->lease());
    ASSERT_TRUE(result) << result.error().message();
    std::weak_ptr<const void> lifetime = storage->lease();
    *storage = {};
    EXPECT_FALSE(lifetime.expired());
    ASSERT_TRUE(context->close().ok());
    const auto actual = t::read(*context, *result);
    ASSERT_TRUE(actual);
    for (int row = 0; row < rows; ++row) {
      double sum = 0;
      for (int column = 0; column < width; ++column) {
        const double value = host[row * width + column];
        sum += value * value;
      }
      const double inverse = 1.0 / std::sqrt(sum / width + epsilon);
      for (int column = 0; column < width; ++column) {
        const auto index = row * width + column;
        const auto expected = -host[index] * inverse;
        EXPECT_NEAR((*actual)[index], expected, 6e-6 * (1.0 + std::abs(expected)));
      }
    }
    *result = {};
    EXPECT_TRUE(lifetime.expired());
  }
  EXPECT_EQ(quota.used(), 0u);
}

TEST(VulkanProviderBridge, ExpertCallbackFailureReleasesSharedQueueLock) {
  auto context = t::context();
  ASSERT_TRUE(context);
  auto status = context->with_queue([](VkQueue) -> hyle::Status {
    throw std::runtime_error("native consumer failure");
  });
  EXPECT_TRUE(status.IsInternal());
  const auto provider = v::detail::Access::runtime(*context)->device->provider;
  bool called = false;
  status = provider.with_queue([](void *state, VkQueue queue) -> VkResult {
    *static_cast<bool *>(state) = queue != VK_NULL_HANDLE;
    return VK_SUCCESS;
  }, &called);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(called);
  const auto values = t::data(257, 1);
  auto tensor = t::upload(*context, t::dense({257}), values);
  ASSERT_TRUE(tensor);
  const auto actual = t::read(*context, *tensor);
  ASSERT_TRUE(actual);
  EXPECT_EQ(*actual, values);
}
} // namespace
