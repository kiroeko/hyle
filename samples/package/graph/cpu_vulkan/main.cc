#include <hyle/graph/cpu.h>
#include <hyle/graph/cpu_vulkan.h>
#include <hyle/compute/low_precision.h>
#include <hyle/sched/scheduler.h>

#include "../../compute/cpp/workload.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace v = c::vulkan;
namespace b = g::cpu_vulkan;
namespace m = hyle::mem;
using hyle::Status;
template<class T> using Result = hyle::Result<T>;
struct Tensor final { m::Buffer buffer; c::TensorView view; };
Result<Tensor> tensor(c::TensorDesc description) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto buffer, m::Buffer::create(m::system_resource(), bytes, 64));
  HYLE_ASSIGN_OR_RETURN(auto view, c::TensorView::create(description, buffer.view()));
  return Tensor{std::move(buffer), view};
}
struct Drain final { hyle::sched::Scheduler &scheduler; ~Drain() { scheduler.drain(); } };

// Exact Q8_0 fixture: each 32-value block has an IEEE half scale followed by
// signed bytes. The oracle uses decoded scalar values, never provider math.
Result<Tensor> q8_weight(int projection, std::vector<float> &decoded) {
  HYLE_ASSIGN_OR_RETURN(auto description, c::gguf_weight_description(c::GgufQuantizedType::kQ8_0, c::DataType::kFloat32, 64, 64));
  HYLE_ASSIGN_OR_RETURN(auto result, tensor(description));
  std::array<std::byte, 64 * 2 * 34> encoded{};
  decoded.resize(64 * 64);
  for (int n = 0; n < 64; ++n) for (int block = 0; block < 2; ++block) {
    auto *data = encoded.data() + (n * 2 + block) * 34;
    const std::uint16_t scale = 0x2400; // 1/64.
    std::memcpy(data, &scale, 2);
    for (int j = 0; j < 32; ++j) {
      const auto k = block * 32 + j;
      const auto q = (n * 5 + k * 3 + projection * 7) % 17 - 8;
      data[2 + j] = std::byte(static_cast<unsigned char>(q));
      decoded[n * 64 + k] = q / 64.0f;
    }
  }
  HYLE_RETURN_IF_ERROR(result.buffer.view().write(encoded));
  return result;
}
Status ffn(b::Environment environment, hyle::sched::Scheduler &scheduler, c::ResourceBudget budget) {
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto scale, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto down_desc, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 16}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(row));
  HYLE_ASSIGN_OR_RETURN(auto normalization, tensor(scale));
  HYLE_ASSIGN_OR_RETURN(auto cpu_weight, tensor(matrix));
  std::array<float, 64> ones; ones.fill(1.0f);
  std::array<float, 4096> identity{};
  for (std::size_t i = 0; i < 64; ++i) identity[i * 64 + i] = 1.0f;
  HYLE_RETURN_IF_ERROR(normalization.buffer.view().write(std::as_bytes(std::span(ones))));
  HYLE_RETURN_IF_ERROR(cpu_weight.buffer.view().write(std::as_bytes(std::span(identity))));
  std::vector<float> gate_values, up_values;
  HYLE_ASSIGN_OR_RETURN(auto gate_weight, q8_weight(0, gate_values));
  HYLE_ASSIGN_OR_RETURN(auto up_weight, q8_weight(1, up_values));
  HYLE_ASSIGN_OR_RETURN(auto down_weight, tensor(down_desc));
  std::array<float, 64 * 16> down_values;
  for (std::size_t i = 0; i < down_values.size(); ++i) down_values[i] = float(int(i % 17) - 8) * 0.015f;
  HYLE_RETURN_IF_ERROR(down_weight.buffer.view().write(std::as_bytes(std::span(down_values))));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row));
  HYLE_ASSIGN_OR_RETURN(auto s, builder.input(scale));
  HYLE_ASSIGN_OR_RETURN(auto cw, builder.input(matrix));
  HYLE_ASSIGN_OR_RETURN(auto gw, builder.constant(gate_weight.view.description()));
  HYLE_ASSIGN_OR_RETURN(auto uw, builder.constant(up_weight.view.description()));
  HYLE_ASSIGN_OR_RETURN(auto dw, builder.constant(down_desc));
  const std::array constants{
      g::ConstantBinding{gw, gate_weight.view.as_const(), g::ConstantPolicy::kMaterialize},
      g::ConstantBinding{uw, up_weight.view.as_const(), g::ConstantPolicy::kMaterialize},
      g::ConstantBinding{dw, down_weight.view.as_const(), g::ConstantPolicy::kMaterialize}};
  HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(c::RmsNorm{}, x, s));
  HYLE_ASSIGN_OR_RETURN(auto projected, builder.add(c::MatMul{}, normalized, cw));
  HYLE_ASSIGN_OR_RETURN(auto gate, builder.add(c::MatMul{}, projected, gw));
  HYLE_ASSIGN_OR_RETURN(auto up, builder.add(c::MatMul{}, projected, uw));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate, up));
  HYLE_ASSIGN_OR_RETURN(auto down, builder.add(c::MatMul{}, product, dw));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kAbs}, down));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placement{
      g::NodePlacement{*graph.node_id(0), b::kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(1), b::kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(6), b::kCpuTarget, g::PlacementStrength::kRequired}};
  HYLE_ASSIGN_OR_RETURN(auto workload, sample_workload::create());
  b::CompileReport compile_report;
  environment.native_boundary_budget_bytes = 512;
  HYLE_ASSIGN_OR_RETURN(auto program, b::compile(graph, environment,
      {.node_placements = placement, .constants = constants, .workload = workload, .resource_budget = budget}, &compile_report));
  if (compile_report.fragments.size() != 3 || compile_report.fragments[1].node_count != 4)
    return Status::Internal("CPU+Vulkan split the mature FFN or lost node coverage");
  gate_weight = {}; up_weight = {}; down_weight = {}; // Materialization must own bytes.
  HYLE_ASSIGN_OR_RETURN(auto queue0, environment.cpu->create_queue());
  HYLE_ASSIGN_OR_RETURN(auto queue1, environment.cpu->create_queue());
  HYLE_ASSIGN_OR_RETURN(auto first, program.create_invocation(scheduler, queue0));
  HYLE_ASSIGN_OR_RETURN(auto second, program.create_invocation(scheduler, queue1));
  Drain drain{scheduler};
  for (auto *run : {&first, &second}) {
    HYLE_RETURN_IF_ERROR(run->bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(run->bind(s, normalization.view.as_const()));
    HYLE_RETURN_IF_ERROR(run->bind(cw, cpu_weight.view.as_const()));
    HYLE_RETURN_IF_ERROR(run->prepare_replay());
    HYLE_ASSIGN_OR_RETURN(auto report, run->replay_report());
    if (report.cpu_fragments != 2 || report.vulkan_fragments != 1 || report.host_copy_count != 2)
      return Status::Internal("CPU+Vulkan replay report differs");
    if (report.independent_upload_count + report.compute_queue_upload_count != 1 ||
        (environment.vulkan->environment().distinct_transfer_queue && report.independent_upload_count != 1))
      return Status::Internal("CPU+Vulkan cold upload queue report differs");
  }
  for (int replay = 0; replay < 3; ++replay) {
    std::array<float, 64> values; double square = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = float((int(i) * 7 + replay * 3) % 23 - 11) * 0.09f;
      square += double(values[i]) * values[i] / values.size();
    }
    HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(auto work0, first.submit());
    HYLE_ASSIGN_OR_RETURN(auto work1, second.submit());
    HYLE_ASSIGN_OR_RETURN(auto lease0, work0.wait());
    HYLE_ASSIGN_OR_RETURN(auto lease1, work1.wait());
    if (first.bind(x, input.view.as_const()).ok()) return Status::Internal("old result lease did not guard reuse");
    HYLE_ASSIGN_OR_RETURN(auto result, lease0->output(output));
    std::array<float, 16> actual;
    HYLE_RETURN_IF_ERROR(result.storage().read(std::as_writable_bytes(std::span(actual))));
    for (int n = 0; n < 16; ++n) {
      double expected = 0;
      for (int h = 0; h < 64; ++h) {
        double gv = 0, uv = 0;
        for (int k = 0; k < 64; ++k) {
          const auto value = values[k] / std::sqrt(square + 1e-5);
          gv += value * gate_values[h * 64 + k]; uv += value * up_values[h * 64 + k];
        }
        expected += gv / (1.0 + std::exp(-gv)) * uv * down_values[h * 16 + n];
      }
      if (!std::isfinite(actual[n]) || std::abs(actual[n] - std::abs(expected)) > 5e-5)
        return Status::Internal("CPU+Vulkan Q8 FFN independent oracle differs");
    }
    HYLE_ASSIGN_OR_RETURN(auto resources, first.resource_report());
    if (!resources.has_shared_budget || !resources.has_unknown_native_storage ||
        resources.managed_native_boundary_bytes > 512 || resources.observed_native_backing_count == 0)
      return Status::Internal("CPU+Vulkan resource accounting omitted native/ordinary storage");
  }
  HYLE_RETURN_IF_ERROR(first.close()); HYLE_RETURN_IF_ERROR(second.close());
  return Status::Ok();
}
Status chunk(b::Environment environment, hyle::sched::Scheduler &scheduler, c::ResourceBudget budget) {
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{2, 64}));
  HYLE_ASSIGN_OR_RETURN(auto scale_desc, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(row)); HYLE_ASSIGN_OR_RETURN(auto scale, tensor(scale_desc)); HYLE_ASSIGN_OR_RETURN(auto weight, tensor(matrix));
  std::array<float, 128> values;
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 13) - 6) * 0.25f;
  std::array<float, 64> ones; ones.fill(1.0f);
  std::array<float, 4096> identity{};
  for (std::size_t i = 0; i < 64; ++i) identity[i * 64 + i] = 2.0f;
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
  HYLE_RETURN_IF_ERROR(scale.buffer.view().write(std::as_bytes(std::span(ones))));
  HYLE_RETURN_IF_ERROR(weight.buffer.view().write(std::as_bytes(std::span(identity))));
  const std::array catalog{v::WeightSpec{19, matrix}};
  HYLE_ASSIGN_OR_RETURN(auto slots, v::WeightSlots::create(*environment.vulkan, *environment.staging, *environment.device, catalog, 16384, {.resource_budget = budget}));
  const std::array selection{v::WeightFragment{19, weight.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto batch, slots.upload(0, selection));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row)); HYLE_ASSIGN_OR_RETURN(auto w, builder.input(matrix)); HYLE_ASSIGN_OR_RETURN(auto s, builder.input(scale_desc));
  HYLE_ASSIGN_OR_RETURN(auto positive, builder.add(c::Unary{c::UnaryKind::kAbs}, x));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::MatMul{}, positive, w));
  HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(c::RmsNorm{}, product, s));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kNeg}, normalized));
  HYLE_RETURN_IF_ERROR(builder.publish(output)); HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placement{g::NodePlacement{*graph.node_id(0), b::kCpuTarget, g::PlacementStrength::kRequired}, g::NodePlacement{*graph.node_id(3), b::kCpuTarget, g::PlacementStrength::kRequired}};
  HYLE_ASSIGN_OR_RETURN(auto program, b::compile(graph, environment, {.node_placements = placement, .resource_budget = budget}));
  HYLE_ASSIGN_OR_RETURN(auto queue, environment.cpu->create_queue()); HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation(scheduler, queue));
  Drain drain{scheduler};
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const())); HYLE_RETURN_IF_ERROR(invocation.bind(w, batch, 19)); HYLE_RETURN_IF_ERROR(invocation.bind(s, scale.view.as_const()));
  if (invocation.prepare_replay({.host_wait = g::HostWaitPolicy::kReject}).ok()) return Status::Internal("CPU+Vulkan hid its native completion barrier");
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay()); HYLE_ASSIGN_OR_RETURN(auto lease, invocation.run()); HYLE_ASSIGN_OR_RETURN(auto result, lease->output(output));
  std::array<float, 128> actual; HYLE_RETURN_IF_ERROR(result.storage().read(std::as_writable_bytes(std::span(actual))));
  for (std::size_t r = 0; r < 2; ++r) {
    double square = 0;
    for (std::size_t i = 0; i < 64; ++i) square += 4.0 * values[r * 64 + i] * values[r * 64 + i] / 64;
    for (std::size_t i = 0; i < 64; ++i)
      if (!std::isfinite(actual[r * 64 + i]) || std::abs(actual[r * 64 + i] + 2.0 * std::abs(values[r * 64 + i]) / std::sqrt(square + 1e-5)) > 1e-5)
        return Status::Internal("CPU+Vulkan orthogonal chunk oracle differs");
  }
  HYLE_RETURN_IF_ERROR(invocation.close()); HYLE_RETURN_IF_ERROR(slots.close());
  return Status::Ok();
}
Status run() {
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context()); HYLE_ASSIGN_OR_RETURN(auto vulkan, v::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(vulkan, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(vulkan, v::MemoryKind::kStaging));
  hyle::sched::Scheduler scheduler(2, m::system_allocator(), 64);
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 64 * 1024 * 1024}, c::ResourceBudgetLimit{device.ref(), 64 * 1024 * 1024}, c::ResourceBudgetLimit{staging.ref(), 64 * 1024 * 1024}};
  HYLE_ASSIGN_OR_RETURN(auto budget, c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
  const b::Environment environment{.cpu = &cpu, .vulkan = &vulkan, .device = &device, .staging = &staging};
  HYLE_RETURN_IF_ERROR(ffn(environment, scheduler, budget)); HYLE_RETURN_IF_ERROR(chunk(environment, scheduler, budget));
  std::printf("CPU+Vulkan: complete Q8 FFN, orthogonal chunk, two slots and owned replay passed\n");
  return Status::Ok();
}
}
int main() {
  const auto status = run(); if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data()); return 1;
}
