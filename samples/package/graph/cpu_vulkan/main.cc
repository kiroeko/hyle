#include <hyle/graph/cpu.h>
#include <hyle/graph/vulkan.h>
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/compute/vulkan_weights.h>
#include <hyle/compute/low_precision.h>
#include <hyle/sched/scheduler.h>
#include <hyle/sched/blocking.h>

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

constexpr g::ExecutionTargetId kVulkanTarget{0}, kCpuTarget{1};
constexpr g::StorageDomainId kHostStorage{0}, kDeviceStorage{1}, kStagingStorage{2};
struct Resources final {
  const c::Context &cpu;
  const v::Context &vulkan;
  const v::MemoryResource &device, &staging;
};
Result<g::CompiledGraph> compile_graph(const g::Graph &graph, const Resources &resources,
    g::CompileOptions options, std::size_t device_workspace = SIZE_MAX) {
  HYLE_ASSIGN_OR_RETURN(auto native, g::vulkan::execution_source(resources.vulkan));
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::ExecutionSource::acquire(resources.cpu));
  const std::array domains{g::StorageDomain{m::system_resource()},
      g::StorageDomain{resources.device.ref(), device_workspace},
      g::StorageDomain{resources.staging.ref()}};
  const std::array targets{g::ExecutionTarget{native, kDeviceStorage},
      g::ExecutionTarget{cpu, kHostStorage}};
  return g::compile(graph, {.storage_domains = domains, .targets = targets}, options);
}
Result<std::vector<g::QueueBinding>> cpu_queues(const g::CompileReport &report, c::Queue &queue) {
  std::vector<g::QueueBinding> bindings;
  for (std::size_t i = 0; i < report.lane_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto lane, report.lane(i));
    if (!lane.queue_required) continue;
    if (lane.target != kCpuTarget || lane.runtime_mode != c::ProviderRuntimeMode::kSharedQueue)
      return std::unexpected(Status::Internal("native Vulkan requested a caller Compute Queue"));
    bindings.push_back({lane.lane, &queue});
  }
  return bindings;
}

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
Status ffn(const Resources &environment, hyle::sched::Scheduler &scheduler,
           hyle::sched::BlockingPool &workers, c::ResourceBudget budget) {
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
  // Observe the already-existing native boundary. This adds no arithmetic or
  // fusion cut, but keeps its output live through the public GraphOutput.
  HYLE_RETURN_IF_ERROR(builder.publish(down));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  std::array<g::NodePlacement, 7> placement;
  for (std::size_t i = 0; i < placement.size(); ++i)
    placement[i] = {*graph.node_id(i), i < 2 || i == 6 ? kCpuTarget : kVulkanTarget,
                    g::PlacementStrength::kRequired};
  const std::array external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{s, kHostStorage}, g::ExternalResourceContract{cw, kHostStorage},
      g::ExternalResourceContract{output, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto workload, sample_workload::create());
  HYLE_ASSIGN_OR_RETURN(auto compiled, compile_graph(graph, environment,
      {.constants = constants, .resource_budget = budget, .node_placements = placement,
       .external_resources = external, .workload = workload}, 512));
  std::size_t cpu_nodes = 0, native_nodes = 0, full_ffn = 0, copies = 0;
  for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto step, compiled.report.step(i));
    if (step.kind == g::CompileStepKind::kCopy) { ++copies; continue; }
    if (step.target == kCpuTarget) cpu_nodes += step.node_count;
    else native_nodes += step.node_count;
    full_ffn += step.target == kVulkanTarget && step.kind == g::CompileStepKind::kProviderRegion && step.node_count == 4;
  }
  if (cpu_nodes != 3 || native_nodes != 4 || full_ffn != 1)
    return Status::Internal("CPU+Vulkan split the mature FFN or lost node coverage");
  std::array<bool, 2> boundary_seen{};
  std::size_t boundary_bytes = 0;
  for (std::size_t i = 0; i < compiled.report.physical_value_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto value, compiled.report.physical_value(i));
    if (value.storage != kDeviceStorage) continue;
    const auto boundary = value.value == projected ? 0 : value.value == down ? 1 : -1;
    if (boundary < 0 || boundary_seen[std::size_t(boundary)]) continue;
    boundary_seen[std::size_t(boundary)] = true;
    HYLE_ASSIGN_OR_RETURN(auto bytes, value.description.required_bytes());
    boundary_bytes += bytes;
  }
  if (!boundary_seen[0] || !boundary_seen[1] || boundary_bytes > 512)
    return Status::Internal("CPU+Vulkan moved more than its declared tensor boundaries");
  gate_weight = {}; up_weight = {}; down_weight = {}; // Materialization must own bytes.
  HYLE_ASSIGN_OR_RETURN(auto queue0, environment.cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto queue1, environment.cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto queues0, cpu_queues(compiled.report, queue0));
  HYLE_ASSIGN_OR_RETURN(auto queues1, cpu_queues(compiled.report, queue1));
  HYLE_ASSIGN_OR_RETURN(auto first, compiled.executable.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto second, compiled.executable.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto output_desc, graph.value_description(output));
  HYLE_ASSIGN_OR_RETURN(auto output0, tensor(output_desc));
  HYLE_ASSIGN_OR_RETURN(auto output1, tensor(output_desc));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(first, scheduler, workers));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(second, scheduler, workers));
  HYLE_RETURN_IF_ERROR(first.bind(output, output0.view));
  HYLE_RETURN_IF_ERROR(second.bind(output, output1.view));
  Drain drain{scheduler};
  for (auto *run : {&first, &second}) {
    HYLE_RETURN_IF_ERROR(run->bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(run->bind(s, normalization.view.as_const()));
    HYLE_RETURN_IF_ERROR(run->bind(cw, cpu_weight.view.as_const()));
    HYLE_RETURN_IF_ERROR(run->prepare_replay(run == &first ? queues0 : queues1));
    HYLE_ASSIGN_OR_RETURN(auto report, run->replay_report());
    if (copies != 4 || report.host_wait_dependency_count == 0)
      return Status::Internal("CPU+Vulkan replay report differs");
  }
  for (int replay = 0; replay < 3; ++replay) {
    std::array<float, 64> values; double square = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = float((int(i) * 7 + replay * 3) % 23 - 11) * 0.09f;
      square += double(values[i]) * values[i] / values.size();
    }
    HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(auto work0, first.submit(queues0));
    HYLE_ASSIGN_OR_RETURN(auto work1, second.submit(queues1));
    HYLE_RETURN_IF_ERROR(work0.wait());
    HYLE_RETURN_IF_ERROR(work1.wait());
    HYLE_ASSIGN_OR_RETURN(auto lease0, first.result(output));
    HYLE_ASSIGN_OR_RETURN(auto lease1, second.result(output));
    if (first.bind(x, input.view.as_const()).ok()) return Status::Internal("old result lease did not guard reuse");
    HYLE_ASSIGN_OR_RETURN(auto result, lease0.tensor());
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
    HYLE_ASSIGN_OR_RETURN(auto native_result, first.result(down));
    HYLE_ASSIGN_OR_RETURN(auto native_owner, native_result.storage_lease());
    HYLE_ASSIGN_OR_RETURN(auto native_tensor, environment.device.binding(std::move(native_owner), output_desc));
    HYLE_ASSIGN_OR_RETURN(auto backing, environment.device.storage_info(native_tensor));
    HYLE_ASSIGN_OR_RETURN(auto domain, compiled.report.storage_domain(kDeviceStorage.ordinal()));
    if (resources.storage.size() != 3 || !resources.has_unknown_graph_storage_history ||
        resources.storage[kDeviceStorage.ordinal()].resource != environment.device.ref() ||
        domain.workspace_bytes > 512 || backing.buffer == VK_NULL_HANDLE || backing.capacity_bytes < 64)
      return Status::Internal("CPU+Vulkan resource accounting omitted native/ordinary storage");
  }
  HYLE_RETURN_IF_ERROR(first.close()); HYLE_RETURN_IF_ERROR(second.close());
  return Status::Ok();
}
Status chunk(const Resources &environment, hyle::sched::Scheduler &scheduler,
             hyle::sched::BlockingPool &workers, c::ResourceBudget budget) {
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
  HYLE_ASSIGN_OR_RETURN(auto slots, v::WeightSlots::create(environment.vulkan, environment.staging, environment.device, catalog, 16384, {.resource_budget = budget}));
  const std::array selection{v::WeightFragment{19, weight.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto batch, slots.upload(0, selection));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row)); HYLE_ASSIGN_OR_RETURN(auto w, builder.input(matrix)); HYLE_ASSIGN_OR_RETURN(auto s, builder.input(scale_desc));
  HYLE_ASSIGN_OR_RETURN(auto positive, builder.add(c::Unary{c::UnaryKind::kAbs}, x));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::MatMul{}, positive, w));
  HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(c::RmsNorm{}, product, s));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kNeg}, normalized));
  HYLE_RETURN_IF_ERROR(builder.publish(output)); HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placement{g::NodePlacement{*graph.node_id(0), kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(1), kVulkanTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(2), kVulkanTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(3), kCpuTarget, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{w, kDeviceStorage}, g::ExternalResourceContract{s, kHostStorage},
      g::ExternalResourceContract{output, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, compile_graph(graph, environment,
      {.resource_budget = budget, .node_placements = placement, .external_resources = external}));
  HYLE_ASSIGN_OR_RETURN(auto queue, environment.cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto queues, cpu_queues(compiled.report, queue));
  HYLE_ASSIGN_OR_RETURN(auto invocation, compiled.executable.create_invocation());
  HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, workers));
  HYLE_ASSIGN_OR_RETURN(auto destination, tensor(row));
  Drain drain{scheduler};
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(g::vulkan::bind(invocation, w, batch, 19));
  HYLE_RETURN_IF_ERROR(invocation.bind(s, scale.view.as_const()));
  HYLE_RETURN_IF_ERROR(invocation.bind(output, destination.view));
  if (invocation.prepare_replay(queues, {.host_wait = g::HostWaitPolicy::kReject}).ok()) return Status::Internal("CPU+Vulkan hid its native completion barrier");
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(queues));
  HYLE_RETURN_IF_ERROR(completion.wait());
  HYLE_ASSIGN_OR_RETURN(auto lease, invocation.result(output));
  HYLE_ASSIGN_OR_RETURN(auto result, lease.tensor());
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
  hyle::sched::BlockingPool workers(m::system_allocator(), 2);
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 64 * 1024 * 1024}, c::ResourceBudgetLimit{device.ref(), 64 * 1024 * 1024}, c::ResourceBudgetLimit{staging.ref(), 64 * 1024 * 1024}};
  HYLE_ASSIGN_OR_RETURN(auto budget, c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
  const Resources environment{cpu, vulkan, device, staging};
  HYLE_RETURN_IF_ERROR(ffn(environment, scheduler, workers, budget).WithContext("mixed FFN"));
  HYLE_RETURN_IF_ERROR(chunk(environment, scheduler, workers, budget).WithContext("mixed weight chunk"));
  std::printf("CPU+Vulkan: complete Q8 FFN, orthogonal chunk, two slots and owned replay passed\n");
  return Status::Ok();
}
}
int main() {
  const auto status = run(); if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data()); return 1;
}
