#include <hyle/compute/cpu.h>
#include <hyle/graph/executable.h>
#include <hyle/mem/buffer.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace {

hyle::Result<hyle::mem::Buffer>
make_buffer(hyle::compute::TensorDesc description) noexcept {
  auto bytes = description.required_bytes();
  if (!bytes)
    return std::unexpected(std::move(bytes).error());
  return hyle::mem::Buffer::create(hyle::mem::system_resource(), *bytes, 64);
}

} // namespace

int main() {
  using namespace hyle::compute;
  using namespace hyle::graph;

  constexpr std::array<std::int64_t, 2> activation_shape{1, 4};
  constexpr std::array<std::int64_t, 1> norm_shape{4};
  constexpr std::array<std::int64_t, 2> stored_weight_shape{8, 4};
  constexpr std::array<std::int64_t, 2> output_shape{1, 4};
  auto activation_desc =
      TensorDesc::dense(DataType::kFloat32, activation_shape);
  auto norm_desc = TensorDesc::dense(DataType::kFloat32, norm_shape);
  auto stored_weight_desc =
      TensorDesc::dense(DataType::kFloat32, stored_weight_shape);
  auto output_desc = TensorDesc::dense(DataType::kFloat32, output_shape);
  if (!activation_desc || !norm_desc || !stored_weight_desc || !output_desc) {
    return 1;
  }
  auto logical_weight_desc = stored_weight_desc->transposed();
  if (!logical_weight_desc)
    return 2;

  auto builder_result = GraphBuilder::create();
  if (!builder_result)
    return 3;
  GraphBuilder builder = std::move(*builder_result);
  auto activation_id = builder.input(*activation_desc);
  auto norm_id = builder.input(*norm_desc);
  auto weight_id = builder.input(*logical_weight_desc);
  if (!activation_id || !norm_id || !weight_id)
    return 4;
  auto normalized = builder.add(RmsNorm{}, *activation_id, *norm_id);
  if (!normalized)
    return 5;
  auto projected = builder.add(MatMul{}, *normalized, *weight_id);
  if (!projected)
    return 6;
  auto output_id = builder.add(GatedActivation{}, *projected);
  if (!output_id || !builder.publish(*output_id).ok())
    return 7;
  auto logical = std::move(builder).finish();

  auto context = cpu::create_context();
  if (!logical || !context)
    return 8;
  auto queue = context->create_queue();
  auto compiled = compile(*logical, *context);
  if (!queue || !compiled || compiled->report.node_count() != 3 ||
      compiled->report.workspace_bytes() == 0 ||
      compiled->report.copy_bytes() != 0 ||
      compiled->report.mixed_providers()) {
    return 9;
  }

  auto activation_storage = make_buffer(*activation_desc);
  auto norm_storage = make_buffer(*norm_desc);
  auto weight_storage = make_buffer(*logical_weight_desc);
  auto output_storage = make_buffer(*output_desc);
  if (!activation_storage || !norm_storage || !weight_storage ||
      !output_storage) {
    return 10;
  }
  constexpr std::array<float, 4> activations{1, 2, 3, 4};
  constexpr std::array<float, 4> norm_weight{1, 1, 1, 1};
  std::array<float, 32> matrix_weight{};
  matrix_weight.fill(0.25F);
  if (!activation_storage->view()
           .write(std::as_bytes(std::span(activations)))
           .ok() ||
      !norm_storage->view().write(std::as_bytes(std::span(norm_weight))).ok() ||
      !weight_storage->view()
           .write(std::as_bytes(std::span(matrix_weight)))
           .ok()) {
    return 11;
  }

  auto activation =
      TensorView::create(*activation_desc, activation_storage->view());
  auto norm = TensorView::create(*norm_desc, norm_storage->view());
  auto weight =
      TensorView::create(*logical_weight_desc, weight_storage->view());
  auto output = TensorView::create(*output_desc, output_storage->view());
  auto invocation = compiled->executable.create_invocation();
  if (!activation || !norm || !weight || !output || !invocation ||
      !invocation->bind(*activation_id, *activation).ok() ||
      !invocation->bind(*norm_id, *norm).ok() ||
      !invocation->bind(*weight_id, *weight).ok() ||
      !invocation->bind(*output_id, *output).ok()) {
    return 12;
  }
  auto completion = invocation->submit(*queue);
  return completion && completion->wait().ok() ? 0 : 13;
}
