// The consumer provides public setup helpers; these checks exercise distinct
// rejection, quota rollback and asynchronous lifetime contracts.
#define main hyle_cpu_apple_consumer_main
#include "../../samples/package/graph/cpu_apple/main.cc"
#undef main

#include <hyle/mem/quota_allocator.h>
#include <hyle/mem/stats.h>

#include "weight_alias_allocator.h"
#include <hyle/compute/low_precision.h>

namespace {
Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
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
Status require_native_product(const g::CompileReport &report, g::NodeId product) {
  for (std::size_t index = 0; index < report.step_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto step, report.step(index));
    if (step.kind == g::CompileStepKind::kCopy) continue;
    for (std::size_t node = 0; node < step.node_count; ++node) {
      HYLE_ASSIGN_OR_RETURN(auto selected, report.step_node(index, node));
      if (selected == product)
        return require(step.target == kAppleTarget, "mixed MatMul did not use its actual native target");
    }
  }
  return Status::Internal("mixed MatMul is absent from the execution plan");
}
Status leased_contracts() {
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto apple, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(apple));
  auto stats = m::StatsAllocator::create(m::system_allocator()).value();
  hyle::sched::Scheduler scheduler(1, m::system_allocator(), 32);
  hyle::sched::BlockingPool workers(m::system_allocator(), 1);
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
                            g::NodePlacement{*graph.node_id(1), kAppleTarget, g::PlacementStrength::kRequired},
                            g::NodePlacement{*graph.node_id(2), kCpuTarget, g::PlacementStrength::kRequired}};
  const std::array host_external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{w, kHostStorage}, g::ExternalResourceContract{output, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto program, compile_graph(graph, cpu, apple, memory,
      {.node_placements = placement, .external_resources = host_external}));
  HYLE_RETURN_IF_ERROR(require_native_product(program.report, *graph.node_id(1)));
  HYLE_ASSIGN_OR_RETURN(auto host_queues, cpu_queues(program.report, queue));
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, workers));
  HYLE_RETURN_IF_ERROR(bind_owned_output(invocation, output, row, outputs));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, row, input).WithContext("leased host input bind"));
  HYLE_RETURN_IF_ERROR(invocation.bind(w, matrix, weight).WithContext("leased host weight bind"));
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
  HYLE_ASSIGN_OR_RETURN(auto native_weight, memory.allocate(matrix));
  HYLE_ASSIGN_OR_RETURN(auto host_weight, weight.tensor(matrix));
  HYLE_RETURN_IF_ERROR(memory.copy_from_host(native_weight, host_weight));
  HYLE_ASSIGN_OR_RETURN(auto imported, memory.import_readonly(native_weight, matrix));
  // External storage is a cold contract. Compile the same mathematics for the
  // actual native weight owner rather than disguising its resource identity.
  const std::array native_external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{w, kNativeStorage}, g::ExternalResourceContract{output, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto native_program, compile_graph(graph, cpu, apple, memory,
      {.node_placements = placement, .external_resources = native_external}));
  HYLE_RETURN_IF_ERROR(require_native_product(native_program.report, *graph.node_id(1)));
  HYLE_ASSIGN_OR_RETURN(auto queues, cpu_queues(native_program.report, queue));
  invocation = {};
  HYLE_ASSIGN_OR_RETURN(invocation, native_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, workers));
  HYLE_RETURN_IF_ERROR(bind_owned_output(invocation, output, row, outputs));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, row, input).WithContext("leased native input bind"));
  HYLE_RETURN_IF_ERROR(invocation.bind(w, matrix, imported).WithContext("leased readonly native weight bind"));
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
    HYLE_RETURN_IF_ERROR(g::prepare_progress(cancelled_slot, scheduler, workers));
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
    HYLE_RETURN_IF_ERROR(g::prepare_progress(abandoned_slot, scheduler, workers));
    HYLE_RETURN_IF_ERROR(bind_owned_output(abandoned_slot, output, row, outputs));
    HYLE_RETURN_IF_ERROR(abandoned_slot.bind(x, row, input));
    HYLE_RETURN_IF_ERROR(abandoned_slot.bind(w, matrix, imported));
    HYLE_RETURN_IF_ERROR(abandoned_slot.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto abandoned_work, abandoned_slot.submit(queues));
    abandoned_work = {}; abandoned_slot = {};
    scheduler.drain();
    HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses == 0, "abandoned mixed lease uses did not retire"));
  }
  HYLE_ASSIGN_OR_RETURN(auto rejected_slot, native_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(rejected_slot, scheduler, workers));
  HYLE_RETURN_IF_ERROR(bind_owned_output(rejected_slot, output, row, outputs));
  HYLE_RETURN_IF_ERROR(rejected_slot.bind(x, row, input)); HYLE_RETURN_IF_ERROR(rejected_slot.bind(w, matrix, imported));
  HYLE_RETURN_IF_ERROR(rejected_slot.prepare_replay(queues));
  {
    HYLE_ASSIGN_OR_RETURN(auto late_slot, native_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(late_slot, scheduler, workers));
    HYLE_RETURN_IF_ERROR(bind_owned_output(late_slot, output, row, outputs));
    HYLE_RETURN_IF_ERROR(late_slot.bind(x, row, input));
    HYLE_RETURN_IF_ERROR(late_slot.bind(w, matrix, imported));
    HYLE_RETURN_IF_ERROR(late_slot.prepare_replay(queues));
    HYLE_RETURN_IF_ERROR(apple.close());
    HYLE_ASSIGN_OR_RETURN(auto accepted_prefix, late_slot.submit(queues));
    auto failed = accepted_prefix.wait();
    HYLE_RETURN_IF_ERROR(require(failed.IsFailedPrecondition(),
        "accepted mixed lease CPU prefix did not preserve native refusal"));
    HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses == 0, "failed mixed lease prefix did not retire"));
    HYLE_RETURN_IF_ERROR(require(!late_slot.submit(queues), "failed mixed lease slot accepted a successor"));
  }
  std::printf("F1 CPU+apple: leased cancellation, abandonment and accepted CPU prefix refusal passed\n");
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
  std::printf("F1 CPU+apple: source blocks before/after callback drain: %u/%u\n",
      static_cast<unsigned>(before_drain), static_cast<unsigned>(pool.stats().live_blocks));
  HYLE_RETURN_IF_ERROR(require(pool.stats().live_blocks == 0, "mixed source lease was not reclaimed"));
  std::printf("F1 CPU+apple: leased host/native math, eight replays without lease metadata allocation, closed admission and retained source alias passed\n");
  return Status::Ok();
}

Status contracts() {
  HYLE_RETURN_IF_ERROR(leased_contracts());
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto apple, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(apple));
  hyle::sched::Scheduler scheduler(1, m::system_allocator(), 32);
  hyle::sched::BlockingPool workers(m::system_allocator(), 1);
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(row));
  HYLE_ASSIGN_OR_RETURN(auto source, tensor(matrix));
  HYLE_ASSIGN_OR_RETURN(auto outputs, c::LeasePool::create(m::system_resource(), 4096,
      {.max_blocks = 8}));
  std::array<float, 64> ones; ones.fill(1.0f);
  std::array<float, 4096> diagonal{};
  for (std::size_t i = 0; i < 64; ++i) diagonal[i * 64 + i] = 2.0f;
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(ones))));
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(diagonal))));
  const std::array catalog{a::WeightSpec{7, matrix}, a::WeightSpec{8, matrix}};
  const std::array selected{a::WeightFragment{7, source.view.as_const()}};
  const std::array over{selected[0], a::WeightFragment{8, source.view.as_const()}};
  const std::array duplicate{selected[0], selected[0]};
  const std::array bad_id{a::WeightFragment{99, source.view.as_const()}};
  const std::array bad_shape{a::WeightFragment{7, input.view.as_const()}};
  g::test::WeightAliasAllocator weight_alias;
  HYLE_ASSIGN_OR_RETURN(auto slots, a::WeightSlots::create(apple, memory, catalog, 16384, {.allocator = weight_alias.get()}));
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, over), "weight capacity was exceeded"));
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, duplicate), "duplicate weight ID was accepted"));
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, bad_id), "unknown weight ID was accepted"));
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, bad_shape), "weight layout mismatch was accepted"));
  HYLE_ASSIGN_OR_RETURN(auto half, c::TensorDesc::dense(c::DataType::kFloat16, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto half_source, c::ConstTensorView::create(half, source.buffer.view()));
  const std::array bad_encoding{a::WeightFragment{7, half_source}};
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, bad_encoding), "different physical weight dtype was accepted"));
  const std::array duplicate_catalog{catalog[0], catalog[0]};
  HYLE_RETURN_IF_ERROR(require(!a::WeightSlots::create(apple, memory, duplicate_catalog, 16384), "duplicate catalog was accepted"));
  HYLE_RETURN_IF_ERROR(require(!a::WeightSlots::create(apple, memory, catalog, 0), "zero capacity was accepted"));
  HYLE_RETURN_IF_ERROR(require(!a::WeightSlots::create(apple, memory, catalog, SIZE_MAX), "overflow capacity was accepted"));
  {
    HYLE_ASSIGN_OR_RETURN(auto bytes_desc, c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 1>{3}));
    HYLE_ASSIGN_OR_RETURN(auto bytes, tensor(bytes_desc));
    const std::array<std::byte, 3> pattern{std::byte{1}, std::byte{2}, std::byte{255}};
    HYLE_RETURN_IF_ERROR(bytes.buffer.view().write(pattern));
    const std::array byte_catalog{a::WeightSpec{30, bytes_desc}};
    HYLE_ASSIGN_OR_RETURN(auto byte_pool, a::WeightSlots::create(apple, memory, byte_catalog, 3));
    const std::array byte_selection{a::WeightFragment{30, bytes.view.as_const()}};
    HYLE_ASSIGN_OR_RETURN(auto batch, byte_pool.upload(0, byte_selection));
    HYLE_RETURN_IF_ERROR(batch.wait());
    HYLE_ASSIGN_OR_RETURN(auto array, batch.binding(30));
    HYLE_RETURN_IF_ERROR(memory.copy_to_host(array, bytes.view));
    std::array<std::byte, 3> actual;
    HYLE_RETURN_IF_ERROR(bytes.buffer.view().read(actual));
    HYLE_RETURN_IF_ERROR(require(actual == pattern, "four-byte native padding changed byte-carrier contents"));
    mlx::core::array outside(array.shape(), array.dtype(), nullptr, {});
    outside.copy_shared_buffer(array, outside.strides(), outside.flags(), outside.size(), 512 * 1024 * 1024);
    outside.set_status(mlx::core::array::available);
    HYLE_RETURN_IF_ERROR(require(!memory.copy_to_host(outside, bytes.view).ok(), "native byte range was not checked"));
  }
  {
    // A real GGUF row is 34 bytes and its logical [K,N] axes do not describe
    // the native uint8 carrier. Preserve both encoding and padding boundary.
    HYLE_ASSIGN_OR_RETURN(auto q8, c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,
        c::DataType::kFloat32, 32, 1));
    HYLE_ASSIGN_OR_RETURN(auto encoded, tensor(q8));
    std::array<std::byte, 34> pattern;
    for (std::size_t i = 0; i < pattern.size(); ++i) pattern[i] = std::byte(i);
    HYLE_RETURN_IF_ERROR(encoded.buffer.view().write(pattern));
    const std::array q8_catalog{a::WeightSpec{31, q8}};
    HYLE_ASSIGN_OR_RETURN(auto q8_pool, a::WeightSlots::create(apple, memory, q8_catalog, 34));
    const std::array selection{a::WeightFragment{31, encoded.view.as_const()}};
    HYLE_ASSIGN_OR_RETURN(auto batch, q8_pool.upload(0, selection));
    HYLE_RETURN_IF_ERROR(batch.wait());
    HYLE_ASSIGN_OR_RETURN(auto native, batch.binding(31));
    HYLE_RETURN_IF_ERROR(memory.copy_to_host(native, encoded.view));
    std::array<std::byte, 34> actual;
    HYLE_RETURN_IF_ERROR(encoded.buffer.view().read(actual));
    HYLE_RETURN_IF_ERROR(require(actual == pattern, "GGUF exact byte carrier was reinterpreted"));
  }
  {
    m::QuotaAllocator quota(m::system_allocator(), 1);
    HYLE_RETURN_IF_ERROR(require(!a::WeightSlots::create(apple, memory, catalog, 16384,
        {.allocator = quota.get()}), "weight metadata OOM was accepted"));
    HYLE_RETURN_IF_ERROR(require(quota.used() == 0, "weight metadata OOM leaked"));
  }
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 8 * 1024 * 1024},
                           c::ResourceBudgetLimit{memory.ref(), 8 * 1024 * 1024}};
  HYLE_ASSIGN_OR_RETURN(auto strict, c::ResourceBudget::create({.limits = limits}));
  HYLE_RETURN_IF_ERROR(require(!memory.allocate(row, strict), "opaque MLX metadata passed strict total budget"));
  HYLE_ASSIGN_OR_RETURN(auto native_usage, strict.usage(memory.ref()));
  HYLE_RETURN_IF_ERROR(require(native_usage.live_bytes == 0, "strict rejection retained a native allocation"));
  const std::array tiny_limits{c::ResourceBudgetLimit{m::system_resource(), 1024 * 1024},
                                c::ResourceBudgetLimit{memory.ref(), 16384}};
  HYLE_ASSIGN_OR_RETURN(auto tiny, c::ResourceBudget::create({.limits = tiny_limits, .allow_unattributed = true}));
  HYLE_RETURN_IF_ERROR(require(!a::WeightSlots::create(apple, memory, catalog, 16384,
      {.resource_budget = tiny}), "two-slot payload budget was ignored"));
  HYLE_ASSIGN_OR_RETURN(native_usage, tiny.usage(memory.ref()));
  HYLE_RETURN_IF_ERROR(require(native_usage.live_bytes == 0, "partial slot admission did not roll back"));

  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row));
  HYLE_ASSIGN_OR_RETURN(auto w, builder.input(matrix));
  HYLE_ASSIGN_OR_RETURN(auto positive, builder.add(c::Unary{c::UnaryKind::kAbs}, x));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::MatMul{}, positive, w));
  HYLE_ASSIGN_OR_RETURN(auto y, builder.add(c::Unary{c::UnaryKind::kNeg}, product));
  HYLE_RETURN_IF_ERROR(builder.publish(y));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placement{
      g::NodePlacement{*graph.node_id(0), kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(1), kAppleTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(2), kCpuTarget, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{w, kNativeStorage}, g::ExternalResourceContract{y, kHostStorage}};
  HYLE_RETURN_IF_ERROR(require(!compile_graph(graph, cpu, apple, memory,
      {.resource_budget = strict, .node_placements = placement, .external_resources = external}),
      "strict graph admitted private MLX storage"));
  HYLE_ASSIGN_OR_RETURN(auto program, compile_graph(graph, cpu, apple, memory,
      {.node_placements = placement, .external_resources = external}));
  HYLE_RETURN_IF_ERROR(require_native_product(program.report, *graph.node_id(1)));
  HYLE_ASSIGN_OR_RETURN(auto queues, cpu_queues(program.report, queue));
  {
    auto bounded = compile_graph(graph, cpu, apple, memory,
        {.node_placements = placement, .external_resources = external}, 1);
    if (!bounded) {
      HYLE_RETURN_IF_ERROR(require(bounded.error().IsResourceExhausted(),
          "native boundary budget failed for another reason"));
    } else {
      auto slot = bounded->executable.create_invocation();
      HYLE_RETURN_IF_ERROR(require(!slot && slot.error().IsResourceExhausted(),
          "native boundary capacity was ignored"));
    }
  }
  {
    HYLE_ASSIGN_OR_RETURN(auto alias_pool, a::WeightSlots::create(apple, memory, catalog, 16384));
    HYLE_ASSIGN_OR_RETURN(auto batch, alias_pool.upload(0, selected));
    HYLE_RETURN_IF_ERROR(batch.wait());
    HYLE_ASSIGN_OR_RETURN(auto array, batch.binding(7));
    std::optional<mlx::core::array> alias(array);
    array = mlx::core::array(0.0f);
    batch = {};
    HYLE_RETURN_IF_ERROR(require(!alias_pool.upload(0, selected), "last array alias did not guard its slot"));
    alias.reset();
    HYLE_ASSIGN_OR_RETURN(auto reused, alias_pool.upload(0, selected));
    HYLE_RETURN_IF_ERROR(reused.wait());
  }
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, workers));
  Drain drain{scheduler};
  HYLE_RETURN_IF_ERROR(require(!invocation.submit(queues), "unprepared graph submitted"));
  HYLE_RETURN_IF_ERROR(require(!invocation.prepare_replay(queues).ok(), "unbound graph prepared"));
  HYLE_RETURN_IF_ERROR(require(!invocation.bind(x, source.view.as_const()).ok(), "wrong graph shape bound"));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(bind_owned_output(invocation, y, row, outputs));
  HYLE_ASSIGN_OR_RETURN(auto first, slots.upload(0, selected));
  for (std::size_t i = 0; i < 64; ++i) diagonal[i * 64 + i] = 3.0f;
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(diagonal))));
  const std::array next_expert{a::WeightFragment{8, source.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto second, slots.upload(1, next_expert));
  HYLE_RETURN_IF_ERROR(g::apple::bind(invocation, w, first, 7));
  HYLE_RETURN_IF_ERROR(require(!invocation.prepare_replay(queues, {.host_wait = g::HostWaitPolicy::kReject}).ok(),
      "host-wait rejection was ignored"));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  {
    HYLE_ASSIGN_OR_RETURN(auto result, run_once(invocation, queues, y));
    HYLE_ASSIGN_OR_RETURN(auto output, result.tensor());
    std::array<float, 64> values;
    HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(values))));
    for (auto value : values) HYLE_RETURN_IF_ERROR(require(std::abs(value + 2.0f) < 1e-5f, "weight graph result differs"));
    HYLE_RETURN_IF_ERROR(require(!invocation.reset().ok(), "old result lease did not guard reset"));
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
  // The immutable source owns its alias metadata at cold bind, so allocation
  // failure is reported here without replacing the previous live binding.
  const auto rebound = g::apple::bind(invocation, w, first, 7);
  weight_alias.deny.store(false);
  HYLE_RETURN_IF_ERROR(require(!rebound.ok() && rebound.IsResourceExhausted() &&
                               weight_alias.rejected.load() == 1,
                               "explicit weight rebind did not report metadata OOM"));
  HYLE_RETURN_IF_ERROR(g::apple::bind(invocation, w, first, 7));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto rebound_report, invocation.input_source_report());
  HYLE_RETURN_IF_ERROR(require(rebound_report.source_count == 1 && rebound_report.host_wait_count == 1,
                               "retried weight rebind lost its immutable source route"));
  std::puts("ready weight alias: two activation rebinds allocate zero metadata; weight rebind OOM/retry passed");
  HYLE_RETURN_IF_ERROR(invocation.reset());
  HYLE_RETURN_IF_ERROR(require(!invocation.replay_report(), "reset left a valid replay report"));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(bind_owned_output(invocation, y, row, outputs));
  HYLE_RETURN_IF_ERROR(g::apple::bind(invocation, w, second, 8));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  {
    HYLE_ASSIGN_OR_RETURN(auto result, run_once(invocation, queues, y));
    HYLE_ASSIGN_OR_RETURN(auto output, result.tensor());
    std::array<float, 64> values;
    HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(values))));
    for (auto value : values)
      HYLE_RETURN_IF_ERROR(require(std::abs(value + 3.0f) < 1e-5f, "updated weight payload did not replay"));
  }
  {
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit(queues));
    HYLE_RETURN_IF_ERROR(work.request_cancel());
    auto result = work.wait();
    if (!result.ok() && !result.IsCancelled()) return result;
  }
  const auto closed = invocation.close();
  if (!closed.ok() && !closed.IsCancelled()) return closed;
  HYLE_RETURN_IF_ERROR(require(!invocation.submit(queues), "closed invocation submitted"));
  {
    HYLE_ASSIGN_OR_RETURN(auto abandoned, program.executable.create_invocation({.resources = g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(abandoned, scheduler, workers));
    HYLE_RETURN_IF_ERROR(abandoned.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(bind_owned_output(abandoned, y, row, outputs));
    HYLE_RETURN_IF_ERROR(g::apple::bind(abandoned, w, first, 7));
    HYLE_RETURN_IF_ERROR(abandoned.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto pending, abandoned.submit(queues));
    pending = {}; abandoned = {};
    scheduler.drain(); // Borrowed owners remain alive until actual retirement.
  }
  HYLE_RETURN_IF_ERROR(slots.close());
  HYLE_RETURN_IF_ERROR(require(!slots.upload(0, selected), "closed weight pool submitted"));

  // Accepted CPU prefix followed by a GPU runtime admission failure. The
  // prefix must drain and the slot must retire; GPU failure never falls to CPU.
  HYLE_ASSIGN_OR_RETURN(auto late, program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(late, scheduler, workers));
  HYLE_RETURN_IF_ERROR(late.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(bind_owned_output(late, y, row, outputs));
  HYLE_RETURN_IF_ERROR(g::apple::bind(late, w, first, 7));
  HYLE_RETURN_IF_ERROR(late.prepare_replay(queues));
  HYLE_RETURN_IF_ERROR(apple.close());
  HYLE_ASSIGN_OR_RETURN(auto accepted, late.submit(queues));
  HYLE_RETURN_IF_ERROR(require(!accepted.wait().ok(), "closed GPU runtime silently ran elsewhere"));
  HYLE_RETURN_IF_ERROR(require(!late.submit(queues), "failed accepted prefix did not retire the slot"));
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = contracts();
  if (status.ok()) { std::printf("CPU+Apple contracts: rejection, OOM, old leases, cancel, abandon and late failure passed\n"); return 0; }
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
