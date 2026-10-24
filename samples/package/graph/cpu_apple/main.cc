#include <hyle/graph/cpu.h>
#include <hyle/graph/apple.h>
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/sched/blocking.h>
#include <hyle/sched/scheduler.h>

#include "../../compute/cpp/apple_ffn.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace a = c::apple;
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

constexpr g::ExecutionTargetId kAppleTarget{0}, kCpuTarget{1};
constexpr g::StorageDomainId kHostStorage{0}, kNativeStorage{1};
Result<g::CompiledGraph> compile_graph(const g::Graph &graph, const c::Context &cpu,
    const a::Context &apple, const a::MemoryResource &memory, g::CompileOptions options,
    std::size_t workspace = SIZE_MAX) {
  HYLE_ASSIGN_OR_RETURN(auto native, g::apple::execution_source(apple));
  HYLE_ASSIGN_OR_RETURN(auto host, g::ExecutionSource::acquire(cpu));
  const std::array domains{g::StorageDomain{m::system_resource()}, g::StorageDomain{memory.ref(), workspace}};
  const std::array targets{g::ExecutionTarget{native, kNativeStorage}, g::ExecutionTarget{host, kHostStorage}};
  return g::compile(graph, {.storage_domains = domains, .targets = targets}, options);
}
Result<std::vector<g::QueueBinding>> cpu_queues(const g::CompileReport &report, c::Queue &queue) {
  std::vector<g::QueueBinding> bindings;
  for (std::size_t i = 0; i < report.lane_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto lane, report.lane(i));
    if (!lane.queue_required) continue;
    if (lane.target != kCpuTarget || lane.runtime_mode != c::ProviderRuntimeMode::kSharedQueue)
      return std::unexpected(Status::Internal("Apple native lane requested a public Compute Queue"));
    bindings.push_back({lane.lane, &queue});
  }
  return bindings;
}

Status ffn(const c::Context &cpu, const a::Context &apple, const a::MemoryResource &memory,
           hyle::sched::Scheduler &scheduler, hyle::sched::BlockingPool &workers, c::ResourceBudget budget) {
  sample_apple_direct::Weights weights;
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto scale, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto q4, a::affine_q4_description(64, 64));
  HYLE_ASSIGN_OR_RETURN(auto qs, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 1}));
  HYLE_ASSIGN_OR_RETURN(auto down_desc, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 16}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(row));
  HYLE_ASSIGN_OR_RETURN(auto normalization, tensor(scale));
  HYLE_ASSIGN_OR_RETURN(auto cpu_weight, tensor(matrix));
  std::array<float, 64> ones; ones.fill(1.0f);
  std::array<float, 4096> identity{};
  for (std::size_t i = 0; i < 64; ++i) identity[i * 64 + i] = 1.0f;
  HYLE_RETURN_IF_ERROR(normalization.buffer.view().write(std::as_bytes(std::span(ones))));
  HYLE_RETURN_IF_ERROR(cpu_weight.buffer.view().write(std::as_bytes(std::span(identity))));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row));
  HYLE_ASSIGN_OR_RETURN(auto s, builder.input(scale));
  HYLE_ASSIGN_OR_RETURN(auto cw, builder.input(matrix));
  std::vector<Tensor> host_constants;
  std::vector<g::ConstantBinding> constants;
  std::array<g::ValueId, 7> ids;
  host_constants.reserve(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto desc = i == 6 ? down_desc : i % 3 == 0 ? q4 : qs;
    HYLE_ASSIGN_OR_RETURN(auto host, tensor(desc));
    HYLE_RETURN_IF_ERROR(memory.copy_to_host(weights.constants[i], host.view));
    HYLE_ASSIGN_OR_RETURN(ids[i], builder.constant(desc));
    constants.push_back({ids[i], host.view.as_const(), g::ConstantPolicy::kMaterialize});
    host_constants.push_back(std::move(host));
  }
  HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(c::RmsNorm{}, x, s));
  HYLE_ASSIGN_OR_RETURN(auto projected, builder.add(c::MatMul{}, normalized, cw));
  const c::QuantizedMatMul qmm{.mode = c::QuantizedWeightMode::kUnsignedAffine4};
  HYLE_ASSIGN_OR_RETURN(auto gate, builder.add(qmm, projected, ids[0], ids[1], ids[2]));
  HYLE_ASSIGN_OR_RETURN(auto up, builder.add(qmm, projected, ids[3], ids[4], ids[5]));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate, up));
  HYLE_ASSIGN_OR_RETURN(auto down, builder.add(c::MatMul{}, product, ids[6]));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kAbs}, down));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placement{
      g::NodePlacement{*graph.node_id(0), kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(1), kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(6), kCpuTarget, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{s, kHostStorage}, g::ExternalResourceContract{cw, kHostStorage},
      g::ExternalResourceContract{output, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto workload, sample_workload::create());
  HYLE_ASSIGN_OR_RETURN(auto compiled, compile_graph(graph, cpu, apple, memory,
      {.constants = constants, .resource_budget = budget, .node_placements = placement,
       .external_resources = external, .workload = workload}, 512));
  std::size_t cpu_nodes = 0, native_nodes = 0, full_ffn = 0, copies = 0;
  for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto step, compiled.report.step(i));
    if (step.kind == g::CompileStepKind::kCopy) { ++copies; continue; }
    if (step.target == kCpuTarget) cpu_nodes += step.node_count;
    else native_nodes += step.node_count;
    full_ffn += step.target == kAppleTarget && step.kind == g::CompileStepKind::kProviderRegion && step.node_count == 4;
  }
  if (cpu_nodes != 3 || native_nodes != 4 || full_ffn != 1 || copies != 2)
    return Status::Internal("CPU+Apple split a mature FFN or lost node coverage");
  host_constants.clear(); // Materialized constants must really own their bytes.
  HYLE_ASSIGN_OR_RETURN(auto queue0, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto queue1, cpu.create_queue());
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
  const auto bind = [&](g::Invocation &run, std::span<const g::QueueBinding> queues) -> Status {
    HYLE_RETURN_IF_ERROR(run.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(run.bind(s, normalization.view.as_const()));
    HYLE_RETURN_IF_ERROR(run.bind(cw, cpu_weight.view.as_const()));
    HYLE_RETURN_IF_ERROR(run.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto report, run.replay_report());
    if (report.host_wait_dependency_count == 0)
      return Status::Internal("CPU+Apple replay boundary report differs");
    return Status::Ok();
  };
  HYLE_RETURN_IF_ERROR(bind(first, queues0)); HYLE_RETURN_IF_ERROR(bind(second, queues1));
  for (int replay = 0; replay < 3; ++replay) {
    std::array<float, 64> values;
    double square = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = float((int(i) * 7 + replay * 3) % 23 - 11) * 0.09f;
      square += double(values[i]) * values[i] / values.size();
    }
    HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(auto work0, first.submit(queues0));
    HYLE_ASSIGN_OR_RETURN(auto work1, second.submit(queues1));
    HYLE_RETURN_IF_ERROR(work0.wait()); HYLE_RETURN_IF_ERROR(work1.wait());
    HYLE_ASSIGN_OR_RETURN(auto lease0, first.result(output));
    HYLE_ASSIGN_OR_RETURN(auto lease1, second.result(output));
    if (first.bind(x, input.view.as_const()).ok())
      return Status::Internal("CPU+Apple old result lease did not guard reuse");
    HYLE_ASSIGN_OR_RETURN(auto result, lease0.tensor());
    std::array<float, 16> actual;
    HYLE_RETURN_IF_ERROR(result.storage().read(std::as_writable_bytes(std::span(actual))));
    for (std::size_t n = 0; n < actual.size(); ++n) {
      double expected = 0;
      for (int h = 0; h < 64; ++h) {
        double gate_value = 0, up_value = 0;
        for (int i = 0; i < 64; ++i) {
          const auto value = values[i] / std::sqrt(square + 1e-5);
          gate_value += value * weights.gate[h * 64 + i];
          up_value += value * weights.up[h * 64 + i];
        }
        expected += gate_value / (1.0 + std::exp(-gate_value)) * up_value * weights.down[h * 16 + n];
      }
      if (std::abs(actual[n] - std::abs(expected)) > 5e-5)
        return Status::Internal("CPU+Apple FFN independent oracle differs");
    }
    HYLE_ASSIGN_OR_RETURN(auto resources, first.resource_report());
    HYLE_ASSIGN_OR_RETURN(auto domain, compiled.report.storage_domain(kNativeStorage.ordinal()));
    if (resources.storage.size() != 2 || !resources.has_unknown_graph_storage_history ||
        resources.storage[kNativeStorage.ordinal()].resource != memory.ref() || domain.workspace_bytes > 512)
      return Status::Internal("CPU+Apple resource accounting omitted native/ordinary storage");
  }
  HYLE_RETURN_IF_ERROR(first.close()); HYLE_RETURN_IF_ERROR(second.close());
  return Status::Ok();
}

Status chunk(const c::Context &cpu, const a::Context &apple, const a::MemoryResource &memory,
             hyle::sched::Scheduler &scheduler, hyle::sched::BlockingPool &workers, c::ResourceBudget budget) {
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{2, 64}));
  HYLE_ASSIGN_OR_RETURN(auto scale_desc, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(row));
  HYLE_ASSIGN_OR_RETURN(auto scale, tensor(scale_desc));
  HYLE_ASSIGN_OR_RETURN(auto weight, tensor(matrix));
  std::array<float, 128> values;
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 13) - 6) * 0.25f;
  std::array<float, 64> ones; ones.fill(1.0f);
  std::array<float, 4096> identity{};
  for (std::size_t i = 0; i < 64; ++i) identity[i * 64 + i] = 2.0f;
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
  HYLE_RETURN_IF_ERROR(scale.buffer.view().write(std::as_bytes(std::span(ones))));
  HYLE_RETURN_IF_ERROR(weight.buffer.view().write(std::as_bytes(std::span(identity))));
  const std::array catalog{a::WeightSpec{19, matrix}};
  HYLE_ASSIGN_OR_RETURN(auto slots, a::WeightSlots::create(apple, memory, catalog, 16384, {.resource_budget = budget}));
  const std::array selection{a::WeightFragment{19, weight.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto batch, slots.upload(0, selection));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row));
  HYLE_ASSIGN_OR_RETURN(auto w, builder.input(matrix));
  HYLE_ASSIGN_OR_RETURN(auto s, builder.input(scale_desc));
  HYLE_ASSIGN_OR_RETURN(auto positive, builder.add(c::Unary{c::UnaryKind::kAbs}, x));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::MatMul{}, positive, w));
  HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(c::RmsNorm{}, product, s));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kNeg}, normalized));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placement{
      g::NodePlacement{*graph.node_id(0), kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(3), kCpuTarget, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{w, kNativeStorage}, g::ExternalResourceContract{s, kHostStorage},
      g::ExternalResourceContract{output, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, compile_graph(graph, cpu, apple, memory,
      {.resource_budget = budget, .node_placements = placement, .external_resources = external}));
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto queues, cpu_queues(compiled.report, queue));
  HYLE_ASSIGN_OR_RETURN(auto invocation, compiled.executable.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto destination, tensor(row));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, workers));
  Drain drain{scheduler};
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(g::apple::bind(invocation, w, batch, 19));
  HYLE_RETURN_IF_ERROR(invocation.bind(s, scale.view.as_const()));
  HYLE_RETURN_IF_ERROR(invocation.bind(output, destination.view));
  if (invocation.prepare_replay(queues, {.host_wait = g::HostWaitPolicy::kReject}).ok())
    return Status::Internal("CPU+Apple hid its native completion barrier");
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(queues));
  HYLE_RETURN_IF_ERROR(completion.wait());
  HYLE_ASSIGN_OR_RETURN(auto lease, invocation.result(output));
  HYLE_ASSIGN_OR_RETURN(auto result, lease.tensor());
  std::array<float, 128> actual;
  HYLE_RETURN_IF_ERROR(result.storage().read(std::as_writable_bytes(std::span(actual))));
  for (std::size_t r = 0; r < 2; ++r) {
    double square = 0;
    for (std::size_t i = 0; i < 64; ++i) square += 4.0 * values[r * 64 + i] * values[r * 64 + i] / 64;
    for (std::size_t i = 0; i < 64; ++i)
      if (std::abs(actual[r * 64 + i] + 2.0 * std::abs(values[r * 64 + i]) / std::sqrt(square + 1e-5)) > 1e-5)
        return Status::Internal("CPU+Apple orthogonal chunk oracle differs");
  }
  HYLE_RETURN_IF_ERROR(invocation.close());
  HYLE_RETURN_IF_ERROR(slots.close());
  return Status::Ok();
}
Status run() {
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto apple, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(apple));
  hyle::sched::Scheduler scheduler(2, m::system_allocator(), 64);
  hyle::sched::BlockingPool workers(m::system_allocator(), 2);
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 64 * 1024 * 1024},
                           c::ResourceBudgetLimit{memory.ref(), 64 * 1024 * 1024}};
  HYLE_ASSIGN_OR_RETURN(auto budget, c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
  HYLE_RETURN_IF_ERROR(ffn(cpu, apple, memory, scheduler, workers, budget));
  HYLE_RETURN_IF_ERROR(chunk(cpu, apple, memory, scheduler, workers, budget));
  std::printf("CPU+Apple: complete Q4 FFN, orthogonal chunk, two slots and owned replay passed\n");
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
