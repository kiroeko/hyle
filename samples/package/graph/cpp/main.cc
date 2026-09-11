#include <hyle/graph/executable.h>
#include <hyle/mem/buffer.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

int main() {
  using namespace hyle::compute;
  using namespace hyle::graph;

  constexpr std::array<std::int64_t, 1> shape{8};
  auto description = TensorDesc::dense(DataType::kFloat32, shape);
  if (!description) return 1;

  GraphBuilder builder;
  auto input_id = builder.input(*description);
  if (!input_id) return 2;
  auto output_id = builder.add(RmsNorm{}, *input_id);
  if (!output_id || !builder.output(*output_id).ok()) return 3;
  auto logical = std::move(builder).finish();

  auto context = Context::create(Device::cpu());
  if (!logical || !context) return 4;
  auto queue = context->create_queue();
  auto compiled = compile(*logical, *context);
  if (!queue || !compiled || compiled->report.workspace_bytes != 0 ||
      compiled->report.copy_bytes != 0) {
    return 5;
  }

  constexpr std::size_t bytes = shape[0] * sizeof(float);
  auto input_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), bytes);
  auto output_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), bytes);
  if (!input_buffer || !output_buffer) return 6;
  const std::array<float, shape[0]> values{1, 2, 3, 4, 5, 6, 7, 8};
  if (!input_buffer->view().write(std::as_bytes(std::span(values))).ok()) {
    return 7;
  }
  auto input = TensorView::create(*description, input_buffer->view());
  auto output = TensorView::create(*description, output_buffer->view());
  auto invocation = compiled->executable.create_invocation();
  if (!input || !output || !invocation ||
      !invocation->bind(*input_id, *input).ok() ||
      !invocation->bind(*output_id, *output).ok()) {
    return 8;
  }
  auto completion = invocation->submit(*queue);
  return completion && completion->wait().ok() ? 0 : 9;
}
