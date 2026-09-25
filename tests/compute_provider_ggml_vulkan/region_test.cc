#include "metadata_arena.h"
#include "native_device.h"
#include "native_graph.h"
#include "region.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {
namespace c = hyle::compute;
using hyle::compute::ggml_vulkan::detail::MetadataArena;
using hyle::compute::ggml_vulkan::detail::NativeDevice;
using hyle::compute::ggml_vulkan::detail::NativeGraph;
using hyle::compute::ggml_vulkan::detail::region_supports;

// A native producer uses the exported device and the same upstream queue lock.
// Its write is made visible before the original ggml region consumes the buffer.
hyle::Status native_fill(const NativeDevice &owner, VkBuffer buffer,
                          VkDeviceSize bytes) {
  const auto &native = owner.native();
  const auto get = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      native.get_instance_proc_addr(native.instance, "vkGetDeviceProcAddr"));
#define LOAD(name) const auto name = reinterpret_cast<PFN_vk##name>( \
    get(native.device, "vk" #name)); \
    if (!name) return hyle::Status::Unavailable("native fixture PFN " #name)
  LOAD(CreateCommandPool);
  LOAD(DestroyCommandPool);
  LOAD(AllocateCommandBuffers);
  LOAD(BeginCommandBuffer);
  LOAD(CmdFillBuffer);
  LOAD(CmdPipelineBarrier);
  LOAD(EndCommandBuffer);
  LOAD(QueueSubmit);
  LOAD(QueueWaitIdle);
#undef LOAD
  const auto check = [](VkResult result) {
    return result == VK_SUCCESS ? hyle::Status::Ok() :
        hyle::Status::FromNative(hyle::StatusCode::Unknown,
            {hyle::NativeErrorDomain::Vulkan, result}, "native producer fixture");
  };
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = native.queue_family;
  VkCommandPool pool = VK_NULL_HANDLE;
  HYLE_RETURN_IF_ERROR(check(CreateCommandPool(native.device, &pool_info, nullptr, &pool)));
  VkCommandBufferAllocateInfo allocation{};
  allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocation.commandPool = pool;
  allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocation.commandBufferCount = 1;
  VkCommandBuffer command = VK_NULL_HANDLE;
  auto status = check(AllocateCommandBuffers(native.device, &allocation, &command));
  if (!status.ok()) {
    DestroyCommandPool(native.device, pool, nullptr);
    return status;
  }
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  status = check(BeginCommandBuffer(command, &begin));
  if (status.ok()) {
    CmdFillBuffer(command, buffer, 0, bytes, 0x3f000000u); // 0.5f
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    status = check(EndCommandBuffer(command));
  }
  bool accepted = false;
  if (status.ok()) {
    status = owner.with_queue([&](VkQueue queue) -> hyle::Status {
      VkSubmitInfo submit{};
      submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command;
      HYLE_RETURN_IF_ERROR(check(QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE)));
      accepted = true;
      return check(QueueWaitIdle(queue));
    });
  }
  // An unexpected observation error fails this fixture and keeps pending
  // commands alive. Ordinary pending/terminal handling is the separate Y3 work.
  if (status.ok() || !accepted)
    DestroyCommandPool(native.device, pool, nullptr);
  return status;
}

// Internal admission: the CMake-owned complete upstream GPU implementation,
// independent scalar arithmetic, and the HYLE metadata owner. This is not an
// ordinary HYLE Direct/Graph acceptance or a performance comparison.
TEST(GgmlVulkanRegion, TargetEmbeddingFormatsAdmitOnlyInternalUncheckedRows) {
  const auto ids =
      c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{2})
          .value();
  for (const auto type :
       {c::GgufQuantizedType::kQ4K, c::GgufQuantizedType::kQ6K,
        c::GgufQuantizedType::kQ8_0}) {
    const auto stored =
        c::gguf_weight_description(type, c::DataType::kFloat32, 256, 3).value();
    const auto table = stored.transposed().value();
    const std::array inputs{table, ids};
    const std::array outputs{
        c::infer_single_output(c::Operation(c::Gather{}), inputs).value()};
    EXPECT_TRUE(region_supports(
        c::Operation(c::Gather{0, c::IndexBoundsPolicy::kUnchecked}), inputs,
        outputs));
    EXPECT_TRUE(region_supports(
        c::Operation(c::Gather{-2, c::IndexBoundsPolicy::kUnchecked}), inputs,
        outputs));
    EXPECT_FALSE(region_supports(c::Operation(c::Gather{}), inputs, outputs));
    EXPECT_FALSE(region_supports(
        c::Operation(c::Gather{1, c::IndexBoundsPolicy::kUnchecked}), inputs,
        outputs));
  }
  const auto other = c::gguf_weight_description(c::GgufQuantizedType::kQ5K,
                                                c::DataType::kFloat32, 256, 3)
                         .value()
                         .transposed()
                         .value();
  const std::array other_inputs{other, ids};
  const std::array other_outputs{
      c::infer_single_output(c::Operation(c::Gather{}), other_inputs).value()};
  EXPECT_FALSE(region_supports(
      c::Operation(c::Gather{0, c::IndexBoundsPolicy::kUnchecked}),
      other_inputs, other_outputs));
}

TEST(GgmlVulkanRegion, DenseBfloatGroupedExpertsRequireF32Activations) {
  const auto operation = c::model_ops::grouped_matmul();
  const auto weights = c::TensorDesc::dense(
                           c::DataType::kBFloat16,
                           std::array<std::int64_t, 3>{3, 8, 64})
                           .value();
  const auto activations = c::TensorDesc::dense(
                               c::DataType::kFloat32,
                               std::array<std::int64_t, 3>{1, 2, 64})
                               .value();
  const auto ids = c::TensorDesc::dense(
                       c::DataType::kInt32,
                       std::array<std::int64_t, 2>{1, 2})
                       .value();
  const std::array inputs{weights, activations, ids};
  const std::array outputs{
      c::infer_single_output(operation, inputs).value()};
  EXPECT_TRUE(region_supports(operation, inputs, outputs));

  auto invalid_inputs = inputs;
  invalid_inputs[1] = c::TensorDesc::dense(
                          c::DataType::kBFloat16,
                          std::array<std::int64_t, 3>{1, 2, 64})
                          .value();
  EXPECT_FALSE(region_supports(operation, invalid_inputs, outputs));
}

TEST(GgmlVulkanRegion, BfloatRouterMatMulRetainsF32OutputDomain) {
  const auto operation = c::model_ops::bfloat_linear_f32();
  const auto activation =
      c::TensorDesc::dense(c::DataType::kBFloat16,
                           std::array<std::int64_t, 2>{1, 8192})
          .value();
  const auto stored =
      c::TensorDesc::dense(c::DataType::kBFloat16,
                           std::array<std::int64_t, 2>{512, 8192})
          .value();
  const std::array inputs{activation, stored};
  const std::array outputs{
      c::infer_single_output(operation, inputs).value()};
  EXPECT_TRUE(region_supports(operation, inputs, outputs));
  auto invalid_outputs = outputs;
  invalid_outputs[0] =
      c::TensorDesc::dense(c::DataType::kBFloat16,
                           std::array<std::int64_t, 2>{1, 512})
          .value();
  EXPECT_FALSE(region_supports(operation, inputs, invalid_outputs));
}

TEST(GgmlVulkanRegion, ModelWidthNormGateResidualWithChangingInputs) {
#if defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
               SEM_NOOPENFILEERRORBOX);
#endif
  auto device = NativeDevice::create();
  ASSERT_TRUE(device);
  auto active = device->activate();
  auto backend_result = device->create_backend();
  ASSERT_TRUE(backend_result);
  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
      *backend_result, &ggml_backend_free);
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
    auto input_allocation = device->allocate_buffer(width * rows * sizeof(float));
    ASSERT_TRUE(input_allocation);
    std::shared_ptr<ggml_backend_buffer> input_owner(*input_allocation,
                                                   ggml_backend_buffer_free);
    const auto native_input = device->buffer_handle(input_owner.get());
    ASSERT_NE(native_input, VK_NULL_HANDLE);
    auto imported = device->borrow(native_input, width * rows * sizeof(float), input_owner);
    ASSERT_TRUE(imported);
    std::shared_ptr<ggml_backend_buffer> borrowed(*imported,
        [device = *device](ggml_backend_buffer *buffer) {
          auto active = device.activate();
          ggml_backend_buffer_free(buffer);
        });
    ASSERT_EQ(ggml_backend_tensor_alloc(borrowed.get(), input,
        ggml_backend_buffer_get_base(borrowed.get())), GGML_STATUS_SUCCESS);
    auto storage = NativeGraph::allocate(*device, std::move(*arena), graph, borrowed);
    ASSERT_TRUE(storage);
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
      if (iteration == 1) {
        std::fill(x.begin(), x.end(), 0.5f);
        ASSERT_TRUE(native_fill(*device, native_input, x.size() * sizeof(float)).ok());
      } else {
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
      }
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
    borrowed.reset();
    *storage = {};
    ASSERT_TRUE(native_fill(*device, native_input, x.size() * sizeof(float)).ok());
  }
}
} // namespace
