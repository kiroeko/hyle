#if defined(HYLE_COMPUTE_CPU)
#include <hyle/compute/cpu.h>
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include "cuda_ffn.h"
#include <cstdio>
#endif
#if defined(HYLE_COMPUTE_APPLE)
#include "apple_ffn.h"
#include <cstdio>
#endif
#include <hyle/compute/operation.h>
#if defined(HYLE_COMPUTE_VULKAN)
#include "vulkan_ffn.h"
#include <cstdio>
#endif
#include <hyle/compute/prepared_operation.h>
#if defined(HYLE_COMPUTE_ONEDNN)
#include <hyle/compute/providers/onednn.h>
#endif
#include <hyle/compute/tensor.h>
#include <hyle/mem/buffer.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

int main() {
  using namespace hyle::compute;

  const std::array<std::int64_t, 1> shape{4};
  auto description = TensorDesc::dense(DataType::kFloat32, shape);
  if (!description)
    return 1;

#if defined(HYLE_COMPUTE_CPU)
  const std::array<float, 4> input{1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 4> output{};
  auto input_buffer = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                                sizeof(input), alignof(float));
  auto output_buffer = hyle::mem::Buffer::create(
      hyle::mem::system_resource(), sizeof(output), alignof(float));
  if (!input_buffer || !output_buffer ||
      !input_buffer->view().write(std::as_bytes(std::span(input))).ok()) {
    return 2;
  }

  auto input_view = ConstTensorView::create(*description, input_buffer->view());
  auto output_view = TensorView::create(*description, output_buffer->view());
  auto context = cpu::create_context();
  if (!input_view || !output_view || !context)
    return 3;
  auto queue = context->create_queue();
  auto prepared = prepare(*context, RmsNorm{}, *description);
  if (!queue || !prepared)
    return 4;
  auto invocation = prepared->create_invocation();
  if (!invocation)
    return 5;
  auto completion = invocation->submit(*queue, *input_view, *output_view);
  if (!completion || !completion->wait().ok() ||
      !output_buffer->view()
           .read(std::as_writable_bytes(std::span(output)))
           .ok()) {
    return 6;
  }

  float sum = 0.0F;
  for (float value : input)
    sum += value * value;
  const float scale = 1.0F / std::sqrt(sum / static_cast<float>(input.size()) +
                                       RmsNorm{}.epsilon);
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (std::abs(output[index] - input[index] * scale) > 1.0e-5F)
      return 7;
  }

  int engineering_value = 40;
  auto host = cpu::submit_host(
      *queue, [](int &value) noexcept { value += 2; }, engineering_value);
  if (!host || !host->wait().ok() || engineering_value != 42)
    return 8;

#if defined(HYLE_COMPUTE_ONEDNN)
  if (onednn::version() != "3.13.2")
    return 9;
  auto provider_context = cpu::create_context();
  if (!provider_context)
    return 10;

  constexpr std::size_t rows = 16;
  constexpr std::size_t inner = 4;
  constexpr std::size_t columns = 4;
  auto left_description = TensorDesc::dense(
      DataType::kFloat32, std::array{static_cast<std::int64_t>(rows),
                                     static_cast<std::int64_t>(inner)});
  auto right_description = TensorDesc::dense(
      DataType::kFloat32, std::array{static_cast<std::int64_t>(inner),
                                     static_cast<std::int64_t>(columns)});
  auto result_description = TensorDesc::dense(
      DataType::kFloat32, std::array{static_cast<std::int64_t>(rows),
                                     static_cast<std::int64_t>(columns)});
  if (!left_description || !right_description || !result_description)
    return 11;

  std::array<float, rows * inner> left_values{};
  std::array<float, inner * columns> right_values{};
  for (std::size_t index = 0; index < left_values.size(); ++index)
    left_values[index] = static_cast<float>(index + 1);
  for (std::size_t diagonal = 0; diagonal < inner; ++diagonal)
    right_values[diagonal * columns + diagonal] = 1.0F;

  auto left_storage = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                                sizeof(left_values), 64);
  auto right_storage = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                                 sizeof(right_values), 64);
  auto result_storage = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                                  sizeof(left_values), 64);
  if (!left_storage || !right_storage || !result_storage ||
      !left_storage->view().write(std::as_bytes(std::span(left_values))).ok() ||
      !right_storage->view()
           .write(std::as_bytes(std::span(right_values)))
           .ok()) {
    return 12;
  }

  auto provider_queue = provider_context->create_queue();
  auto provider_prepared =
      prepare(*provider_context, MatMul{}, *left_description,
              *right_description, *result_description);
  if (!provider_queue || !provider_prepared ||
      provider_prepared->provider().id != onednn::kProviderId) {
    return 13;
  }
  const auto requirement = provider_prepared->workspace_requirement();
  auto workspace = hyle::mem::Buffer::create(
      hyle::mem::system_resource(), requirement.size, requirement.alignment);
  if (!workspace)
    return 14;
  auto provider_invocation =
      provider_prepared->create_invocation(workspace->view());
  auto left_tensor =
      ConstTensorView::create(*left_description, left_storage->view());
  auto right_tensor =
      ConstTensorView::create(*right_description, right_storage->view());
  auto result_tensor =
      TensorView::create(*result_description, result_storage->view());
  if (!provider_invocation || !left_tensor || !right_tensor || !result_tensor)
    return 15;
  auto provider_done = provider_invocation->submit(
      *provider_queue, *left_tensor, *right_tensor, *result_tensor);
  std::array<float, rows * columns> provider_output{};
  if (!provider_done || !provider_done->wait().ok() ||
      !result_storage->view()
           .read(std::as_writable_bytes(std::span(provider_output)))
           .ok()) {
    return 16;
  }
  if (provider_output != left_values)
    return 17;
#endif
#endif // HYLE_COMPUTE_CPU

#if defined(HYLE_COMPUTE_CUDA)
  const auto cuda_status = sample_cuda_direct::run();
  if (!cuda_status.ok()) {
    const auto message = cuda_status.message();
    std::fprintf(stderr, "CUDA FFN: %.*s\n", static_cast<int>(message.size()),
                 message.data());
    return 18;
  }
#endif
#if defined(HYLE_COMPUTE_APPLE)
  const auto apple_status = sample_apple_direct::run();
  if (!apple_status.ok()) {
    const auto message = apple_status.message();
    std::fprintf(stderr, "Apple FFN: %.*s\n", static_cast<int>(message.size()), message.data());
    return 19;
  }
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  const auto vulkan_status = sample_vulkan_direct::run();
  if (!vulkan_status.ok()) {
    const auto message = vulkan_status.message();
    std::fprintf(stderr, "Vulkan FFN: %.*s\n", static_cast<int>(message.size()), message.data());
    return 20;
  }
#endif
  return 0;
}
