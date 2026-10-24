#define main hyle_cpu_vulkan_consumer_main
#include "../../samples/package/graph/cpu_vulkan/main.cc"
#undef main

#include <hyle/mem/quota_allocator.h>
#include <hyle/mem/stats.h>
#include <thread>

#include "weight_alias_allocator.h"

#include "cpu_vulkan_waiter_cases.h"

namespace {
Status require(bool condition, const char *message) {
  if (!condition) std::fprintf(stderr, "CPU+Vulkan contract assertion: %s\n", message);
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
Status collect_abandoned_uses(const v::Context &context, const c::LeasePool &pool) {
  const auto deadline = std::chrono::steady_clock::now() + vulkan_waiter_cases::gate_timeout;
  do {
    HYLE_RETURN_IF_ERROR(context.collect());
    if (pool.stats().active_uses == 0) return Status::Ok();
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < deadline);
  return require(false, "abandoned mixed lease uses did not retire before watchdog");
}
Result<g::GraphOutput> run_once(g::Invocation &invocation,
    std::span<const g::QueueBinding> queues, g::ValueId output) {
  HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(queues));
  HYLE_RETURN_IF_ERROR(completion.wait());
  return invocation.result(output);
}
Status bind_owned_output(g::Invocation &invocation, g::ValueId output,
    c::TensorDesc description, c::LeasePool &pool) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(bytes));
  return invocation.bind(output, description, std::move(writer));
}
Status leased_contracts() {
  // The real native retirement owners may outlive an early public-handle drop.
  // Their metadata allocator must outlive the Context that finally collects them.
  auto stats = m::StatsAllocator::create(m::system_allocator()).value();
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto vulkan, v::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(vulkan, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(vulkan, v::MemoryKind::kStaging));
  const Resources environment{cpu, vulkan, device, staging};
  hyle::sched::Scheduler scheduler(1, m::system_allocator(), 32);
  hyle::sched::BlockingPool completion_waiters(m::system_allocator(), 1);
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue()); Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(m::system_resource(), 20 * 1024,
      {.max_blocks = 2, .allocator = stats.get()}));
  HYLE_ASSIGN_OR_RETURN(auto outputs, c::LeasePool::create(m::system_resource(), 4096,
      {.max_blocks = 8}));
  HYLE_ASSIGN_OR_RETURN(auto input_write, pool.allocate(256));
  HYLE_ASSIGN_OR_RETURN(auto weight_write, pool.allocate(16384));
  std::array<float, 64> values; for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(i + 1) / 64;
  std::array<float, 4096> diagonal{}; for (std::size_t i = 0; i < 64; ++i) diagonal[i * 64 + i] = 2;
  HYLE_RETURN_IF_ERROR(input_write.view()->write(std::as_bytes(std::span(values))));
  HYLE_RETURN_IF_ERROR(weight_write.view()->write(std::as_bytes(std::span(diagonal))));
  HYLE_ASSIGN_OR_RETURN(auto input, input_write.publish());
  HYLE_ASSIGN_OR_RETURN(auto weight, weight_write.publish());
  auto reader = input;
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create({.metadata_allocator = stats.get()}));
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row));
  HYLE_ASSIGN_OR_RETURN(auto w, builder.input(matrix));
  HYLE_ASSIGN_OR_RETURN(auto first, builder.add(c::Unary{c::UnaryKind::kAbs}, x));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::MatMul{}, first, w));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kNeg}, product));
  HYLE_RETURN_IF_ERROR(builder.publish(output)); HYLE_RETURN_IF_ERROR(builder.publish(x));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placement{g::NodePlacement{*graph.node_id(0), kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(1), kVulkanTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(2), kCpuTarget, g::PlacementStrength::kRequired}};
  const std::array host_external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{w, kHostStorage}, g::ExternalResourceContract{output, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto program, compile_graph(graph, environment,
      {.node_placements = placement, .external_resources = host_external}));
  HYLE_ASSIGN_OR_RETURN(auto host_queues, cpu_queues(program.report, queue));
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, completion_waiters));
  HYLE_RETURN_IF_ERROR(bind_owned_output(invocation, output, row, outputs));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, row, input)); HYLE_RETURN_IF_ERROR(invocation.bind(w, matrix, weight));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(host_queues));
  { HYLE_ASSIGN_OR_RETURN(auto warm, run_once(invocation, host_queues, output)); }
  const auto cold = stats.snapshot().total_allocs;
  for (int iteration = 0; iteration < 8; ++iteration) {
    HYLE_ASSIGN_OR_RETURN(auto result, run_once(invocation, host_queues, output));
    HYLE_ASSIGN_OR_RETURN(auto tensor, result.tensor());
    std::array<float, 64> actual;
    HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(actual))));
    for (std::size_t i = 0; i < actual.size(); ++i)
      HYLE_RETURN_IF_ERROR(require(std::abs(actual[i] + 2 * values[i]) < 1e-5f, "leased mixed math differs"));
    HYLE_RETURN_IF_ERROR(require(!invocation.reset().ok(), "old mixed result did not guard reset"));
    HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses == 0, "successful native prefixes did not retire"));
  }
  HYLE_RETURN_IF_ERROR(require(stats.snapshot().total_allocs == cold, "mixed replay allocated lease metadata"));
  HYLE_RETURN_IF_ERROR(require(!input.try_write().has_value(), "shared mixed source became writable"));
  HYLE_RETURN_IF_ERROR(invocation.reset());
  HYLE_ASSIGN_OR_RETURN(auto native_weight, vulkan.upload(matrix, std::as_bytes(std::span(diagonal))));
  HYLE_ASSIGN_OR_RETURN(auto imported, device.import_readonly(native_weight));
  // External storage is a cold contract. Compile the same mathematics for the
  // actual native weight owner rather than disguising its resource identity.
  const std::array native_external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{w, kDeviceStorage}, g::ExternalResourceContract{output, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto native_program, compile_graph(graph, environment,
      {.node_placements = placement, .external_resources = native_external}));
  HYLE_ASSIGN_OR_RETURN(auto queues, cpu_queues(native_program.report, queue));
  invocation = {};
  HYLE_ASSIGN_OR_RETURN(invocation, native_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, completion_waiters));
  HYLE_RETURN_IF_ERROR(bind_owned_output(invocation, output, row, outputs));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, row, input)); HYLE_RETURN_IF_ERROR(invocation.bind(w, matrix, imported));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto retained, run_once(invocation, queues, output));
  HYLE_ASSIGN_OR_RETURN(auto retained_input, invocation.result(x));
  HYLE_ASSIGN_OR_RETURN(auto native_product, retained.tensor());
  std::array<float, 64> native_values;
  HYLE_RETURN_IF_ERROR(native_product.storage().read(std::as_writable_bytes(std::span(native_values))));
  for (std::size_t i = 0; i < native_values.size(); ++i)
    HYLE_RETURN_IF_ERROR(require(std::abs(native_values[i] + 2 * values[i]) < 1e-5f, "native leased mixed math differs"));
  // The original Invocation and sources exercise the opt-in lease branch on
  // cancellation, public-handle abandonment and an accepted CPU prefix failure.
  {
    HYLE_ASSIGN_OR_RETURN(auto cancelled_slot, native_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(cancelled_slot, scheduler, completion_waiters));
    HYLE_RETURN_IF_ERROR(bind_owned_output(cancelled_slot, output, row, outputs));
    HYLE_RETURN_IF_ERROR(cancelled_slot.bind(x, row, input));
    HYLE_RETURN_IF_ERROR(cancelled_slot.bind(w, matrix, imported));
    HYLE_RETURN_IF_ERROR(cancelled_slot.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto work, cancelled_slot.submit(queues));
    HYLE_RETURN_IF_ERROR(work.request_cancel());
    auto completed = work.wait();
    if (!completed.ok()) {
      HYLE_RETURN_IF_ERROR(require(completed.IsCancelled(), "mixed lease cancellation lost its status"));
    } else {
      HYLE_ASSIGN_OR_RETURN(auto result, cancelled_slot.result(output));
      HYLE_ASSIGN_OR_RETURN(auto output_view, result.tensor());
      std::array<float, 64> completed_values;
      HYLE_RETURN_IF_ERROR(output_view.storage().read(std::as_writable_bytes(std::span(completed_values))));
      for (std::size_t i = 0; i < completed_values.size(); ++i)
        HYLE_RETURN_IF_ERROR(require(std::abs(completed_values[i] + 2 * values[i]) < 1e-5f,
            "completion won cancellation with invalid math"));
    }
    HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses == 0, "cancelled mixed lease uses did not retire"));
  }
  {
    HYLE_ASSIGN_OR_RETURN(auto abandoned_slot, native_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(abandoned_slot, scheduler, completion_waiters));
    HYLE_RETURN_IF_ERROR(bind_owned_output(abandoned_slot, output, row, outputs));
    HYLE_RETURN_IF_ERROR(abandoned_slot.bind(x, row, input));
    HYLE_RETURN_IF_ERROR(abandoned_slot.bind(w, matrix, imported));
    HYLE_RETURN_IF_ERROR(abandoned_slot.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto abandoned_work, abandoned_slot.submit(queues));
    abandoned_work = {}; abandoned_slot = {};
    scheduler.drain();
    // Scheduler completion is not native fence retirement. Both public Graph
    // handles stay abandoned; collect polls the original Runtime owners only.
    HYLE_RETURN_IF_ERROR(collect_abandoned_uses(vulkan, pool));
    HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses == 0, "abandoned mixed lease uses did not retire"));
  }
  HYLE_ASSIGN_OR_RETURN(auto rejected_slot, native_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(rejected_slot, scheduler, completion_waiters));
  HYLE_RETURN_IF_ERROR(bind_owned_output(rejected_slot, output, row, outputs));
  HYLE_RETURN_IF_ERROR(rejected_slot.bind(x, row, input)); HYLE_RETURN_IF_ERROR(rejected_slot.bind(w, matrix, imported));
  HYLE_RETURN_IF_ERROR(rejected_slot.prepare_replay(queues));
  {
    HYLE_ASSIGN_OR_RETURN(auto late_slot, native_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(late_slot, scheduler, completion_waiters));
    HYLE_RETURN_IF_ERROR(bind_owned_output(late_slot, output, row, outputs));
    HYLE_RETURN_IF_ERROR(late_slot.bind(x, row, input));
    HYLE_RETURN_IF_ERROR(late_slot.bind(w, matrix, imported));
    HYLE_RETURN_IF_ERROR(late_slot.prepare_replay(queues));
    HYLE_RETURN_IF_ERROR(vulkan.close());
    HYLE_ASSIGN_OR_RETURN(auto accepted_prefix, late_slot.submit(queues));
    auto failed = accepted_prefix.wait();
    HYLE_RETURN_IF_ERROR(require(failed.IsFailedPrecondition(),
        "accepted mixed lease CPU prefix did not preserve native refusal"));
    HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses == 0, "failed mixed lease prefix did not retire"));
    HYLE_RETURN_IF_ERROR(require(!late_slot.submit(queues), "failed mixed lease slot accepted a successor"));
  }
  std::printf("F1 CPU+vulkan: leased cancellation, abandonment and accepted CPU prefix refusal passed\n");
  HYLE_RETURN_IF_ERROR(pool.close());
  HYLE_RETURN_IF_ERROR(require(!rejected_slot.submit(queues).has_value(), "closed mixed source accepted math"));
  HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses == 0, "closed source rejection leaked guards"));
  rejected_slot = {}; invocation = {}; input.reset(); weight.reset(); reader.reset();
  HYLE_ASSIGN_OR_RETURN(auto alias, retained_input.tensor());
  std::array<float, 64> original;
  HYLE_RETURN_IF_ERROR(alias.storage().read(std::as_writable_bytes(std::span(original))));
  HYLE_RETURN_IF_ERROR(require(original == values, "mixed source alias lost its lease backing"));
  HYLE_RETURN_IF_ERROR(require(pool.stats().live_blocks == 1, "old mixed result did not retain source"));
  retained = {}; retained_input = {};
  // Work.wait has already proved permissions retired. Observe physical source
  // reclamation after the original Scheduler's admitted callback tails return.
  const auto before_drain = pool.stats().live_blocks;
  scheduler.drain();
  std::printf("F1 CPU+vulkan: source blocks before/after callback drain: %u/%u\n",
      static_cast<unsigned>(before_drain), static_cast<unsigned>(pool.stats().live_blocks));
  HYLE_RETURN_IF_ERROR(require(pool.stats().live_blocks == 0, "mixed source lease was not reclaimed"));
  std::printf("F1 CPU+vulkan: leased host/native math, eight replays without lease metadata allocation, closed admission and retained source alias passed\n");
  return Status::Ok();
}

Status contracts() {
  HYLE_RETURN_IF_ERROR(vulkan_waiter_cases::run());
  HYLE_RETURN_IF_ERROR(leased_contracts());
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto vulkan, v::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(vulkan, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(vulkan, v::MemoryKind::kStaging));
  hyle::sched::Scheduler scheduler(1, m::system_allocator(), 32);
  hyle::sched::BlockingPool completion_waiters(m::system_allocator(), 2);
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(row)); HYLE_ASSIGN_OR_RETURN(auto source, tensor(matrix));
  HYLE_ASSIGN_OR_RETURN(auto outputs, c::LeasePool::create(m::system_resource(), 4096, {.max_blocks = 8}));
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
  g::test::WeightAliasAllocator weight_alias;
  HYLE_ASSIGN_OR_RETURN(auto slots, v::WeightSlots::create(vulkan, staging, device, catalog, 16384, {.allocator = weight_alias.get()}));
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
  const std::array placement{g::NodePlacement{*graph.node_id(0), kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(1), kVulkanTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(2), kCpuTarget, g::PlacementStrength::kRequired}};
  const Resources environment{cpu, vulkan, device, staging};
  const std::array external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{w, kDeviceStorage}, g::ExternalResourceContract{y, kHostStorage}};
  HYLE_RETURN_IF_ERROR(require(!compile_graph(graph, environment,
      {.resource_budget = strict, .node_placements = placement, .external_resources = external}),
      "strict graph admitted unknown private storage"));
  HYLE_ASSIGN_OR_RETURN(auto program, compile_graph(graph, environment,
      {.node_placements = placement, .external_resources = external}));
  HYLE_ASSIGN_OR_RETURN(auto queues, cpu_queues(program.report, queue));
  {
    auto bounded = compile_graph(graph, environment,
        {.node_placements = placement, .external_resources = external}, 1);
    HYLE_RETURN_IF_ERROR(require(!bounded && bounded.error().IsResourceExhausted(),
        "native boundary domain budget did not cold-reject its actual copy storage"));
  }
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, completion_waiters));
  Drain drain{scheduler};
  HYLE_RETURN_IF_ERROR(require(!invocation.submit(queues), "unprepared graph submitted"));
  HYLE_RETURN_IF_ERROR(require(!invocation.prepare_replay(queues).ok(), "unbound graph prepared"));
  HYLE_RETURN_IF_ERROR(require(!invocation.bind(x, source.view.as_const()).ok(), "wrong graph shape bound"));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(bind_owned_output(invocation, y, row, outputs));
  HYLE_ASSIGN_OR_RETURN(auto first, slots.upload(0, selected));
  for (std::size_t i = 0; i < 64; ++i) diagonal[i * 64 + i] = 3.0f;
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(diagonal))));
  const std::array next_expert{v::WeightFragment{8, source.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto second, slots.upload(1, next_expert));
  HYLE_RETURN_IF_ERROR(require(!second.description(7), "new expert selection retained old ID"));
  HYLE_RETURN_IF_ERROR(g::vulkan::bind(invocation, w, first, 7));
  HYLE_RETURN_IF_ERROR(require(!invocation.prepare_replay(queues, {.host_wait = g::HostWaitPolicy::kReject}).ok(), "host wait rejection ignored"));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  {
    HYLE_ASSIGN_OR_RETURN(auto result, run_once(invocation, queues, y));
    HYLE_ASSIGN_OR_RETURN(auto output, result.tensor()); std::array<float, 64> values;
    HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(values))));
    for (auto value : values) HYLE_RETURN_IF_ERROR(require(std::abs(value + 2.0f) < 1e-5f, "weight graph differs"));
    HYLE_RETURN_IF_ERROR(require(!invocation.reset().ok(), "old lease did not guard reset"));
  }
  const auto alias_calls = weight_alias.calls.load();
  weight_alias.deny.store(true);
  for (int replay = 0; replay < 2; ++replay) {
    ones.fill(static_cast<float>(replay + 2));
    HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(ones))));
    HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto report, invocation.replay_report());
    HYLE_ASSIGN_OR_RETURN(auto source_report, invocation.input_source_report());
    HYLE_RETURN_IF_ERROR(require(source_report.source_count == 1 && source_report.host_wait_count == 1 &&
                                 report.host_wait_dependency_count != 0,
                                 "immutable weight source or native host wait disappeared"));
    HYLE_ASSIGN_OR_RETURN(auto result, run_once(invocation, queues, y));
    HYLE_ASSIGN_OR_RETURN(auto output, result.tensor());
    std::array<float, 64> values;
    HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(values))));
    for (auto value : values)
      HYLE_RETURN_IF_ERROR(require(std::abs(value + 2.0f * static_cast<float>(replay + 2)) < 1e-5f,
                                   "cached weight alias changed the result"));
  }
  ones.fill(1.0f);
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(ones))));
  HYLE_RETURN_IF_ERROR(require(weight_alias.calls.load() == alias_calls,
                               "unchanged weight allocated new alias metadata"));
  // Alias ownership is acquired by the actual immutable-source bind. A failed
  // bind preserves the previous live source and can be retried.
  const auto rebound = g::vulkan::bind(invocation, w, first, 7);
  weight_alias.deny.store(false);
  HYLE_RETURN_IF_ERROR(require(!rebound.ok() && rebound.IsResourceExhausted() &&
                               weight_alias.rejected.load() == 1,
                               "explicit weight rebind did not invalidate its alias"));
  HYLE_RETURN_IF_ERROR(g::vulkan::bind(invocation, w, first, 7));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto rebound_report, invocation.input_source_report());
  HYLE_RETURN_IF_ERROR(require(rebound_report.source_count == 1 && rebound_report.host_wait_count == 1,
                               "retried weight rebind lost its immutable source route"));
  std::puts("ready weight alias: two activation rebinds allocate zero metadata; weight rebind OOM/retry passed");
  HYLE_RETURN_IF_ERROR(invocation.reset()); HYLE_RETURN_IF_ERROR(require(!invocation.replay_report(), "reset kept stale report"));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(bind_owned_output(invocation, y, row, outputs));
  HYLE_RETURN_IF_ERROR(g::vulkan::bind(invocation, w, second, 8)); HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  {
    HYLE_ASSIGN_OR_RETURN(auto result, run_once(invocation, queues, y)); HYLE_ASSIGN_OR_RETURN(auto output, result.tensor()); std::array<float, 64> values;
    HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(values))));
    for (auto value : values) HYLE_RETURN_IF_ERROR(require(std::abs(value + 3.0f) < 1e-5f, "changed expert ID/payload did not replay"));
  }
  {
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit(queues)); HYLE_RETURN_IF_ERROR(work.request_cancel());
    const auto result = work.wait(); if (!result.ok() && !result.IsCancelled()) return result;
  }
  const auto closed = invocation.close(); if (!closed.ok() && !closed.IsCancelled()) return closed;
  HYLE_RETURN_IF_ERROR(require(!invocation.submit(queues), "closed graph submitted"));
  {
    HYLE_ASSIGN_OR_RETURN(auto abandoned, program.executable.create_invocation({.resources = g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(abandoned, scheduler, completion_waiters));
    HYLE_RETURN_IF_ERROR(bind_owned_output(abandoned, y, row, outputs));
    HYLE_RETURN_IF_ERROR(abandoned.bind(x, input.view.as_const())); HYLE_RETURN_IF_ERROR(g::vulkan::bind(abandoned, w, first, 7)); HYLE_RETURN_IF_ERROR(abandoned.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto pending, abandoned.submit(queues)); pending = {}; abandoned = {}; scheduler.drain();
  }
  HYLE_RETURN_IF_ERROR(slots.close()); HYLE_RETURN_IF_ERROR(require(!slots.upload(0, selected), "closed pool submitted"));
  HYLE_ASSIGN_OR_RETURN(auto late, program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(late, scheduler, completion_waiters));
  HYLE_RETURN_IF_ERROR(bind_owned_output(late, y, row, outputs));
  HYLE_RETURN_IF_ERROR(late.bind(x, input.view.as_const())); HYLE_RETURN_IF_ERROR(g::vulkan::bind(late, w, first, 7)); HYLE_RETURN_IF_ERROR(late.prepare_replay(queues));
  HYLE_RETURN_IF_ERROR(vulkan.close()); HYLE_ASSIGN_OR_RETURN(auto accepted, late.submit(queues));
  HYLE_RETURN_IF_ERROR(require(!accepted.wait().ok(), "closed GPU silently ran elsewhere"));
  HYLE_RETURN_IF_ERROR(require(!late.submit(queues), "failed accepted prefix did not retire"));
  return Status::Ok();
}
}
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const auto status = contracts();
  if (status.ok()) { std::printf("CPU+Vulkan: rejection, capacity, rollback, encoded bytes, leases, cancellation and retirement passed\n"); return 0; }
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data()); return 1;
}
