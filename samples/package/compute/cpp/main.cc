#include <hyle/compute/host_kernel.h>
#include <hyle/compute/operation.h>
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
  if (!description) return 1;

  const std::array<float, 4> input{1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 4> output{};
  auto input_buffer = hyle::mem::Buffer::create(
      hyle::mem::system_resource(), sizeof(input), alignof(float));
  auto output_buffer = hyle::mem::Buffer::create(
      hyle::mem::system_resource(), sizeof(output), alignof(float));
  if (!input_buffer || !output_buffer ||
      !input_buffer->view().write(std::as_bytes(std::span(input))).ok()) {
    return 2;
  }

  auto input_view =
      ConstTensorView::create(*description, input_buffer->view());
  auto output_view = TensorView::create(*description, output_buffer->view());
  auto context = Context::create(Device::cpu());
  if (!input_view || !output_view || !context) return 3;
  auto queue = context->create_queue();
  auto prepared = prepare(*context, RmsNorm{}, *description, *description);
  if (!queue || !prepared) return 4;
  auto completion = prepared->submit(*queue, *input_view, *output_view);
  if (!completion || !completion->wait().ok() ||
      !output_buffer->view()
           .read(std::as_writable_bytes(std::span(output)))
           .ok()) {
    return 5;
  }

  float sum = 0.0F;
  for (float value : input) sum += value * value;
  const float scale =
      1.0F / std::sqrt(sum / static_cast<float>(input.size()) +
                       RmsNorm{}.epsilon);
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (std::abs(output[index] - input[index] * scale) > 1.0e-5F) return 6;
  }

  int engineering_value = 40;
  auto host = submit_host(
      *queue,
      [](int& value) noexcept { value += 2; }, engineering_value);
  return host && host->wait().ok() && engineering_value == 42 ? 0 : 7;
}
