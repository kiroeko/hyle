#if defined(HYLE_COMPUTE_CPU)
#include <hyle/compute/cpu.h>
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/transfer.h>
#endif
#include <hyle/graph/executable.h>
#include <hyle/mem/buffer.h>

#if defined(HYLE_GRAPH_ONEDNN)
#include <hyle/graph/providers/onednn.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#if defined(HYLE_COMPUTE_CPU)
namespace {

hyle::Result<hyle::mem::Buffer>
make_buffer(hyle::compute::TensorDesc description) noexcept {
  auto bytes = description.required_bytes();
  if (!bytes)
    return std::unexpected(std::move(bytes).error());
  return hyle::mem::Buffer::create(hyle::mem::system_resource(), *bytes, 64);
}

} // namespace
#endif

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
  if (!logical)
    return 8;

#if defined(HYLE_COMPUTE_CPU)
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
  if (!completion || !completion->wait().ok())
    return 13;

#if defined(HYLE_GRAPH_ONEDNN)
  constexpr std::array<std::int64_t, 2> region_activation_shape{16, 4};
  constexpr std::array<std::int64_t, 2> region_weight_shape{4, 8};
  constexpr std::array<std::int64_t, 2> region_down_shape{8, 4};
  auto region_activation_desc =
      TensorDesc::dense(DataType::kFloat32, region_activation_shape);
  auto region_weight_desc =
      TensorDesc::dense(DataType::kFloat32, region_weight_shape);
  auto region_down_desc =
      TensorDesc::dense(DataType::kFloat32, region_down_shape);
  if (!region_activation_desc || !region_weight_desc || !region_down_desc) {
    return 14;
  }

  auto region_builder_result = GraphBuilder::create();
  if (!region_builder_result)
    return 15;
  GraphBuilder region_builder = std::move(*region_builder_result);
  auto region_activation = region_builder.input(*region_activation_desc);
  auto gate_weight = region_builder.constant(*region_weight_desc);
  auto up_weight = region_builder.constant(*region_weight_desc);
  auto down_weight = region_builder.constant(*region_down_desc);
  if (!region_activation || !gate_weight || !up_weight || !down_weight)
    return 16;
  auto gate = region_builder.add(MatMul{}, *region_activation, *gate_weight);
  auto up = region_builder.add(MatMul{}, *region_activation, *up_weight);
  if (!gate || !up)
    return 17;
  auto gated = region_builder.add(
      GatedActivation{.activation = GatedActivationKind::kSilu,
                      .inputs = GatedInputKind::kSeparate},
      *gate, *up);
  if (!gated)
    return 18;
  auto region_output = region_builder.add(MatMul{}, *gated, *down_weight);
  if (!region_output || !region_builder.publish(*region_output).ok())
    return 19;
  auto region_graph = std::move(region_builder).finish();
  auto gate_storage = make_buffer(*region_weight_desc);
  auto up_storage = make_buffer(*region_weight_desc);
  auto down_storage = make_buffer(*region_down_desc);
  if (!region_graph || !gate_storage || !up_storage || !down_storage)
    return 20;
  auto gate_tensor =
      TensorView::create(*region_weight_desc, gate_storage->view());
  auto up_tensor = TensorView::create(*region_weight_desc, up_storage->view());
  auto down_tensor =
      TensorView::create(*region_down_desc, down_storage->view());
  if (!gate_tensor || !up_tensor || !down_tensor)
    return 21;
  const std::array region_constants{
      ConstantBinding{*gate_weight, gate_tensor->as_const()},
      ConstantBinding{*up_weight, up_tensor->as_const()},
      ConstantBinding{*down_weight, down_tensor->as_const()},
  };
  auto region_context =
      Context::create(cpu::device(), hyle::graph::onednn::provider());
  if (!region_context)
    return 22;
  auto region_compiled = compile(*region_graph, *region_context,
                                 CompileOptions{.constants = region_constants});
  if (!region_compiled || region_compiled->report.region_count() != 1 ||
      region_compiled->report.step_count() != 1 ||
      region_compiled->report.provider().id !=
          hyle::graph::onednn::kProviderId) {
    return 23;
  }
#endif
#endif // HYLE_COMPUTE_CPU

#if defined(HYLE_COMPUTE_CUDA)
  auto cuda_device = hyle::compute::cuda::device(0);
  if (!cuda_device)
    return 24;
  auto cuda_context = Context::create(*cuda_device);
  if (!cuda_context)
    return 25;
  auto cuda_queue = cuda_context->create_queue();
  auto cuda_resource = hyle::compute::cuda::MemoryResource::create(
      *cuda_context, hyle::compute::cuda::MemoryKind::kDevice);
  if (!cuda_queue || !cuda_resource ||
      !cuda_queue->reserve_completions(1).ok()) {
    return 26;
  }
  auto cuda_buffer = hyle::mem::Buffer::create(cuda_resource->ref(), 256);
  if (!cuda_buffer)
    return 27;
  auto cuda_done = hyle::compute::fill_async(
      *cuda_queue, cuda_buffer->view(), std::byte{0x2a});
  if (!cuda_done || !cuda_done->wait().ok())
    return 28;
  std::array<std::byte, 256> downloaded{};
  if (!cuda_buffer->view().read(downloaded).ok())
    return 29;
  for (auto value : downloaded) {
    if (value != std::byte{0x2a})
      return 30;
  }
#endif

  return 0;
}
