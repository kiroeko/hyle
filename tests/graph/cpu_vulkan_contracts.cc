#define main hyle_cpu_vulkan_consumer_main
#include "../../samples/package/graph/cpu_vulkan/main.cc"
#undef main

#include <hyle/mem/quota_allocator.h>

namespace {
Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}
Status read_native(const v::Context &context, const v::MemoryResource &staging,
                   const v::Tensor &source, c::TensorView destination) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, destination.description().required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto transfer, v::TransferSlot::create(context, staging, std::max<std::size_t>(1, bytes)));
  HYLE_ASSIGN_OR_RETURN(auto copy, transfer.download(source));
  HYLE_RETURN_IF_ERROR(copy.wait());
  return transfer.read(destination);
}
Status contracts() {
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto vulkan, v::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(vulkan, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(vulkan, v::MemoryKind::kStaging));
  hyle::sched::Scheduler scheduler(1, m::system_allocator(), 32);
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(row)); HYLE_ASSIGN_OR_RETURN(auto source, tensor(matrix));
  std::array<float, 64> ones; ones.fill(1.0f);
  std::array<float, 4096> diagonal{};
  for (std::size_t i = 0; i < 64; ++i) diagonal[i * 64 + i] = 2.0f;
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(ones))));
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(diagonal))));
  const std::array catalog{v::WeightSpec{7, matrix}, v::WeightSpec{8, matrix}};
  const std::array selected{v::WeightFragment{7, source.view.as_const()}};
  const std::array over{selected[0], v::WeightFragment{8, source.view.as_const()}};
  const std::array duplicate{selected[0], selected[0]};
  const std::array bad_id{v::WeightFragment{99, source.view.as_const()}};
  const std::array bad_shape{v::WeightFragment{7, input.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto slots, v::WeightSlots::create(vulkan, staging, device, catalog, 16384));
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, over), "weight capacity was exceeded"));
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, duplicate), "duplicate weight ID was accepted"));
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, bad_id), "unknown weight ID was accepted"));
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, bad_shape), "weight layout mismatch was accepted"));
  const std::array duplicate_catalog{catalog[0], catalog[0]};
  HYLE_RETURN_IF_ERROR(require(!v::WeightSlots::create(vulkan, staging, device, duplicate_catalog, 16384), "duplicate catalog was accepted"));
  HYLE_RETURN_IF_ERROR(require(!v::WeightSlots::create(vulkan, staging, device, catalog, 0), "zero capacity was accepted"));
  HYLE_RETURN_IF_ERROR(require(!v::WeightSlots::create(vulkan, staging, device, catalog, SIZE_MAX), "overflow capacity was accepted"));
  HYLE_RETURN_IF_ERROR(require(!v::TransferSlot::create(vulkan, device, 256), "unmapped device storage used as staging"));
  {
    HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 1>{3}));
    HYLE_ASSIGN_OR_RETURN(auto bytes, tensor(desc));
    const std::array<std::byte, 3> pattern{std::byte{1}, std::byte{2}, std::byte{255}};
    HYLE_RETURN_IF_ERROR(bytes.buffer.view().write(pattern));
    const std::array tiny_catalog{v::WeightSpec{30, desc}};
    HYLE_ASSIGN_OR_RETURN(auto pool, v::WeightSlots::create(vulkan, staging, device, tiny_catalog, 3));
    const std::array selection{v::WeightFragment{30, bytes.view.as_const()}};
    HYLE_ASSIGN_OR_RETURN(auto batch, pool.upload(0, selection)); HYLE_RETURN_IF_ERROR(batch.wait());
    HYLE_ASSIGN_OR_RETURN(auto native, batch.binding(30));
    HYLE_RETURN_IF_ERROR(read_native(vulkan, staging, native, bytes.view));
    std::array<std::byte, 3> actual; HYLE_RETURN_IF_ERROR(bytes.buffer.view().read(actual));
    HYLE_RETURN_IF_ERROR(require(actual == pattern, "native padding changed logical bytes"));
    HYLE_ASSIGN_OR_RETURN(auto small, v::TransferSlot::create(vulkan, staging, 2));
    HYLE_RETURN_IF_ERROR(require(!small.download(native), "transfer ignored logical capacity"));
  }
  {
    HYLE_ASSIGN_OR_RETURN(auto q8, c::gguf_weight_description(c::GgufQuantizedType::kQ8_0, c::DataType::kFloat32, 32, 1));
    HYLE_ASSIGN_OR_RETURN(auto bytes, tensor(q8));
    std::array<std::byte, 34> pattern;
    for (std::size_t i = 0; i < pattern.size(); ++i) pattern[i] = std::byte(i);
    HYLE_RETURN_IF_ERROR(bytes.buffer.view().write(pattern));
    const std::array q8_catalog{v::WeightSpec{31, q8}};
    HYLE_ASSIGN_OR_RETURN(auto pool, v::WeightSlots::create(vulkan, staging, device, q8_catalog, 34));
    const std::array selection{v::WeightFragment{31, bytes.view.as_const()}};
    HYLE_ASSIGN_OR_RETURN(auto batch, pool.upload(0, selection)); HYLE_RETURN_IF_ERROR(batch.wait());
    HYLE_ASSIGN_OR_RETURN(auto native, batch.binding(31));
    HYLE_RETURN_IF_ERROR(read_native(vulkan, staging, native, bytes.view));
    std::array<std::byte, 34> actual; HYLE_RETURN_IF_ERROR(bytes.buffer.view().read(actual));
    HYLE_RETURN_IF_ERROR(require(actual == pattern, "GGUF encoded bytes were reinterpreted"));
  }
  {
    m::QuotaAllocator quota(m::system_allocator(), 1);
    HYLE_RETURN_IF_ERROR(require(!v::WeightSlots::create(vulkan, staging, device, catalog, 16384, {.allocator = quota.get()}), "metadata OOM was ignored"));
    HYLE_RETURN_IF_ERROR(require(quota.used() == 0, "metadata OOM leaked"));
  }
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 8 * 1024 * 1024}, c::ResourceBudgetLimit{device.ref(), 8 * 1024 * 1024}, c::ResourceBudgetLimit{staging.ref(), 8 * 1024 * 1024}};
  HYLE_ASSIGN_OR_RETURN(auto strict, c::ResourceBudget::create({.limits = limits}));
  const std::array tiny_limits{c::ResourceBudgetLimit{m::system_resource(), 1024 * 1024}, c::ResourceBudgetLimit{device.ref(), 16384}, c::ResourceBudgetLimit{staging.ref(), 32768}};
  HYLE_ASSIGN_OR_RETURN(auto tiny, c::ResourceBudget::create({.limits = tiny_limits, .allow_unattributed = true}));
  HYLE_RETURN_IF_ERROR(require(!v::WeightSlots::create(vulkan, staging, device, catalog, 16384, {.resource_budget = tiny}), "two-slot payload budget was ignored"));
  HYLE_ASSIGN_OR_RETURN(auto usage, tiny.usage(device.ref()));
  HYLE_RETURN_IF_ERROR(require(usage.live_bytes == 0, "partial slot admission leaked device budget"));
  HYLE_ASSIGN_OR_RETURN(usage, tiny.usage(staging.ref()));
  HYLE_RETURN_IF_ERROR(require(usage.live_bytes == 0, "partial slot admission leaked staging budget"));
  {
    HYLE_ASSIGN_OR_RETURN(auto pool, v::WeightSlots::create(vulkan, staging, device, catalog, 16384));
    HYLE_ASSIGN_OR_RETURN(auto batch, pool.upload(0, selected)); HYLE_RETURN_IF_ERROR(batch.wait());
    HYLE_ASSIGN_OR_RETURN(auto native, batch.binding(7));
    auto owner = native.native_owner(); native = {}; batch = {};
    HYLE_RETURN_IF_ERROR(require(!pool.upload(0, selected), "exported native owner did not protect slot"));
    owner.reset(); HYLE_ASSIGN_OR_RETURN(auto reused, pool.upload(0, selected)); HYLE_RETURN_IF_ERROR(reused.wait());
  }
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row)); HYLE_ASSIGN_OR_RETURN(auto w, builder.input(matrix));
  HYLE_ASSIGN_OR_RETURN(auto positive, builder.add(c::Unary{c::UnaryKind::kAbs}, x));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::MatMul{}, positive, w));
  HYLE_ASSIGN_OR_RETURN(auto y, builder.add(c::Unary{c::UnaryKind::kNeg}, product));
  HYLE_RETURN_IF_ERROR(builder.publish(y)); HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placement{g::NodePlacement{*graph.node_id(0), b::kCpuTarget, g::PlacementStrength::kRequired}, g::NodePlacement{*graph.node_id(2), b::kCpuTarget, g::PlacementStrength::kRequired}};
  const b::Environment environment{.cpu = &cpu, .vulkan = &vulkan, .device = &device, .staging = &staging};
  HYLE_RETURN_IF_ERROR(require(!b::compile(graph, environment, {.node_placements = placement, .resource_budget = strict}), "strict graph admitted unknown private storage"));
  HYLE_ASSIGN_OR_RETURN(auto program, b::compile(graph, environment, {.node_placements = placement}));
  {
    auto bounded_environment = environment; bounded_environment.native_boundary_budget_bytes = 1;
    HYLE_ASSIGN_OR_RETURN(auto bounded, b::compile(graph, bounded_environment, {.node_placements = placement}));
    HYLE_RETURN_IF_ERROR(require(!bounded.create_invocation(scheduler, queue), "native boundary capacity ignored"));
  }
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation(scheduler, queue)); Drain drain{scheduler};
  HYLE_RETURN_IF_ERROR(require(!invocation.submit(), "unprepared graph submitted"));
  HYLE_RETURN_IF_ERROR(require(!invocation.prepare_replay().ok(), "unbound graph prepared"));
  HYLE_RETURN_IF_ERROR(require(!invocation.bind(x, source.view.as_const()).ok(), "wrong graph shape bound"));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
  HYLE_ASSIGN_OR_RETURN(auto first, slots.upload(0, selected));
  for (std::size_t i = 0; i < 64; ++i) diagonal[i * 64 + i] = 3.0f;
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(diagonal))));
  const std::array next_expert{v::WeightFragment{8, source.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto second, slots.upload(1, next_expert));
  HYLE_RETURN_IF_ERROR(require(!second.description(7), "new expert selection retained old ID"));
  HYLE_RETURN_IF_ERROR(invocation.bind(w, first, 7));
  HYLE_RETURN_IF_ERROR(require(!invocation.prepare_replay({.host_wait = g::HostWaitPolicy::kReject}).ok(), "host wait rejection ignored"));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay());
  {
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit()); HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    HYLE_ASSIGN_OR_RETURN(auto output, result->output(y)); std::array<float, 64> values;
    HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(values))));
    for (auto value : values) HYLE_RETURN_IF_ERROR(require(std::abs(value + 2.0f) < 1e-5f, "weight graph differs"));
    HYLE_RETURN_IF_ERROR(require(!invocation.reset().ok(), "old lease did not guard reset"));
  }
  HYLE_RETURN_IF_ERROR(invocation.reset()); HYLE_RETURN_IF_ERROR(require(!invocation.replay_report(), "reset kept stale report"));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const())); HYLE_RETURN_IF_ERROR(invocation.bind(w, second, 8)); HYLE_RETURN_IF_ERROR(invocation.prepare_replay());
  {
    HYLE_ASSIGN_OR_RETURN(auto result, invocation.run()); HYLE_ASSIGN_OR_RETURN(auto output, result->output(y)); std::array<float, 64> values;
    HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(values))));
    for (auto value : values) HYLE_RETURN_IF_ERROR(require(std::abs(value + 3.0f) < 1e-5f, "changed expert ID/payload did not replay"));
  }
  {
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit()); HYLE_RETURN_IF_ERROR(work.request_cancel());
    const auto result = work.wait(); if (!result && !result.error().IsCancelled()) return result.error();
  }
  const auto closed = invocation.close(); if (!closed.ok() && !closed.IsCancelled()) return closed;
  HYLE_RETURN_IF_ERROR(require(!invocation.submit(), "closed graph submitted"));
  {
    HYLE_ASSIGN_OR_RETURN(auto abandoned, program.create_invocation(scheduler, queue));
    HYLE_RETURN_IF_ERROR(abandoned.bind(x, input.view.as_const())); HYLE_RETURN_IF_ERROR(abandoned.bind(w, first, 7)); HYLE_RETURN_IF_ERROR(abandoned.prepare_replay());
    HYLE_ASSIGN_OR_RETURN(auto pending, abandoned.submit()); pending = {}; abandoned = {}; scheduler.drain();
  }
  HYLE_RETURN_IF_ERROR(slots.close()); HYLE_RETURN_IF_ERROR(require(!slots.upload(0, selected), "closed pool submitted"));
  HYLE_ASSIGN_OR_RETURN(auto late, program.create_invocation(scheduler, queue));
  HYLE_RETURN_IF_ERROR(late.bind(x, input.view.as_const())); HYLE_RETURN_IF_ERROR(late.bind(w, first, 7)); HYLE_RETURN_IF_ERROR(late.prepare_replay());
  HYLE_RETURN_IF_ERROR(vulkan.close()); HYLE_ASSIGN_OR_RETURN(auto accepted, late.submit());
  HYLE_RETURN_IF_ERROR(require(!accepted.wait(), "closed GPU silently ran elsewhere"));
  HYLE_RETURN_IF_ERROR(require(!late.submit(), "failed accepted prefix did not retire"));
  return Status::Ok();
}
}
int main() {
  const auto status = contracts();
  if (status.ok()) { std::printf("CPU+Vulkan: rejection, capacity, rollback, encoded bytes, leases, cancellation and retirement passed\n"); return 0; }
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data()); return 1;
}
