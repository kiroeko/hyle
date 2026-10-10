#pragma once

// Ordinary inference fragment: CPU normalization/casts around an intact CUDA
// gated FFN. Both weight matrices are selected host fragments of one DMA batch.
inline hyle::Status inference_case(const compute::Context &cpu,
    const compute::Context &cuda, const compute::cuda::MemoryResource &host,
    const compute::cuda::MemoryResource &device,
    hyle::sched::Scheduler &scheduler) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto workload, sample_workload::create());
  HYLE_ASSIGN_OR_RETURN(auto x_desc, compute::TensorDesc::dense(
      compute::DataType::kFloat32, std::array<std::int64_t, 2>{1, 128}));
  HYLE_ASSIGN_OR_RETURN(auto norm_desc, compute::TensorDesc::dense(
      compute::DataType::kFloat32, std::array<std::int64_t, 1>{128}));
  HYLE_ASSIGN_OR_RETURN(auto gate_desc, compute::TensorDesc::dense(
      compute::DataType::kFloat16, std::array<std::int64_t, 2>{128, 256}));
  HYLE_ASSIGN_OR_RETURN(auto down_desc, compute::TensorDesc::dense(
      compute::DataType::kFloat16, std::array<std::int64_t, 2>{128, 64}));
  HYLE_ASSIGN_OR_RETURN(auto out_desc, compute::TensorDesc::dense(
      compute::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto x_buffer, tensor(mem::system_resource(), x_desc));
  HYLE_ASSIGN_OR_RETURN(auto norm_buffer, tensor(mem::system_resource(), norm_desc));
  HYLE_ASSIGN_OR_RETURN(auto gate_buffer, tensor(mem::system_resource(), gate_desc));
  HYLE_ASSIGN_OR_RETURN(auto down_buffer, tensor(mem::system_resource(), down_desc));
  HYLE_ASSIGN_OR_RETURN(auto out_buffer, tensor(mem::system_resource(), out_desc));
  std::array<float, 128> ones;
  ones.fill(1.0f);
  HYLE_RETURN_IF_ERROR(x_buffer.buffer.view().write(std::as_bytes(std::span(ones))));
  HYLE_RETURN_IF_ERROR(norm_buffer.buffer.view().write(std::as_bytes(std::span(ones))));
  std::array<__half, 32768> gate_values;
  for (std::size_t i = 0; i < gate_values.size(); ++i)
    gate_values[i] = __float2half(i % 256 < 128 ? 1.0f / 32 : 1.0f / 16);
  std::array<__half, 8192> down_values;
  down_values.fill(__float2half(1.0f / 64));
  HYLE_RETURN_IF_ERROR(gate_buffer.buffer.view().write(std::as_bytes(std::span(gate_values))));
  HYLE_RETURN_IF_ERROR(down_buffer.buffer.view().write(std::as_bytes(std::span(down_values))));
  const std::array catalog{compute::cuda::WeightSpec{101, gate_desc},
                           compute::cuda::WeightSpec{102, down_desc}};
  HYLE_ASSIGN_OR_RETURN(auto slots,
      compute::cuda::WeightSlots::create(cuda, host, device, catalog, 81920));
  HYLE_ASSIGN_OR_RETURN(auto transfer, cuda.create_queue());
  HYLE_RETURN_IF_ERROR(transfer.reserve_completions(8));
  const std::array selection{
      compute::cuda::WeightFragment{101, gate_buffer.view.as_const()},
      compute::cuda::WeightFragment{102, down_buffer.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto weights, slots.upload(0, selection, transfer));
  HYLE_ASSIGN_OR_RETURN(auto builder, graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(x_desc));
  HYLE_ASSIGN_OR_RETURN(auto scale, builder.input(norm_desc));
  HYLE_ASSIGN_OR_RETURN(auto gate, builder.input(gate_desc));
  HYLE_ASSIGN_OR_RETURN(auto down, builder.input(down_desc));
  HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(compute::RmsNorm{}, x, scale));
  HYLE_ASSIGN_OR_RETURN(auto half, builder.add(compute::Cast{compute::DataType::kFloat16}, normalized));
  HYLE_ASSIGN_OR_RETURN(auto projected, builder.add(compute::MatMul{}, half, gate));
  HYLE_ASSIGN_OR_RETURN(auto activated, builder.add(compute::GatedActivation{}, projected));
  HYLE_ASSIGN_OR_RETURN(auto lowered, builder.add(compute::MatMul{}, activated, down));
  HYLE_ASSIGN_OR_RETURN(auto y, builder.add(compute::Cast{compute::DataType::kFloat32}, lowered));
  HYLE_RETURN_IF_ERROR(builder.publish(y));
  HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
  std::array<graph::NodePlacement, 6> placement;
  for (std::size_t i = 0; i < placement.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, logical.node_id(i));
    placement[i] = {node, i < 2 || i == 5 ? bridge::kCpuTarget : bridge::kCudaTarget,
                    graph::PlacementStrength::kRequired};
  }
  const std::array external{
      graph::ExternalResourceContract{x, bridge::kHostStorage},
      graph::ExternalResourceContract{scale, bridge::kHostStorage},
      graph::ExternalResourceContract{gate, bridge::kDeviceStorage},
      graph::ExternalResourceContract{down, bridge::kDeviceStorage},
      graph::ExternalResourceContract{y, bridge::kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,
      bridge::compile(logical, {&cpu, &cuda, mem::system_resource(), &host, &device},
          {.node_placements = placement, .external_resources = external,
           .workload = workload}));
  HYLE_ASSIGN_OR_RETURN(auto native, compiled.executable.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto gpu_queue, cuda.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue, cpu.create_queue());
  const std::array queues{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &gpu_queue},
      graph::QueueBinding{graph::ExecutionLaneId{1}, &cpu_queue}};
  HYLE_ASSIGN_OR_RETURN(auto invocation,
      bridge::Invocation::create(std::move(native), scheduler, queues));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, x_buffer.view.as_const()));
  HYLE_RETURN_IF_ERROR(invocation.bind(scale, norm_buffer.view.as_const()));
  HYLE_RETURN_IF_ERROR(invocation.bind(gate, weights, 101));
  HYLE_RETURN_IF_ERROR(invocation.bind(down, weights, 102));
  HYLE_RETURN_IF_ERROR(invocation.bind(y, out_buffer.view));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay());
  Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto result, invocation.run());
  HYLE_ASSIGN_OR_RETURN(auto output, result->output(y));
  std::array<float, 64> actual;
  HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(actual))));
  const float intermediate = __half2float(__float2half(4.0f / (1 + std::exp(-4.0f)) * 8.0f));
  const float expected = __half2float(__float2half(2 * intermediate));
  for (auto value : actual)
    HYLE_RETURN_IF_ERROR(require(std::abs(value - expected) < 0.01f,
                                 "CPU CUDA gated FFN numeric result differs"));
  std::printf("CPU+CUDA inference: CPU norm/casts, CUDA gated FFN, FP16 weight fragments passed\n");
  return Status::Ok();
}
