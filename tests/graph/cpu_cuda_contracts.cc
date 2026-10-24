// Reuse the actual consumer's public C++ setup helpers. Its main is not run:
// these checks cover distinct rejection/rollback/terminal contracts.
#define main hyle_cpu_cuda_consumer_main
#include "../../samples/package/graph/cpu_cuda/main.cc"
#undef main

#include <atomic>
#include <climits>
#include <chrono>
#include <thread>
#include <bit>
#include <vector>
#include <hyle/compute/detail/queue_sequence.h>
#include <hyle/graph/executable.h>
#include <hyle/mem/stats.h>

namespace {
struct NativeGate final {
  std::atomic<bool> entered{false}, open{false};
  compute::Completion prefix{};
  static void CUDART_CB block(void *opaque) noexcept {
    auto &self = *static_cast<NativeGate *>(opaque);
    self.entered.store(true, std::memory_order_release);
    self.open.wait(false, std::memory_order_acquire);
  }
  void release() noexcept {
    open.store(true, std::memory_order_release);
    open.notify_all();
  }
  ~NativeGate() {
    release();
    if (prefix.valid()) prefix.wait().IgnoreError();
  }
};
struct AllocationBudget final {
  std::atomic<int> remaining{0};
  std::atomic<int> live{0};
  static void *allocate(void *opaque, std::size_t size, std::size_t alignment) noexcept {
    auto &self = *static_cast<AllocationBudget *>(opaque);
    if (self.remaining.fetch_sub(1) <= 0)
      return nullptr;
    auto *result = mem::system_allocator().allocate(size, alignment);
    if (result != nullptr)
      ++self.live;
    return result;
  }
  static void free(void *opaque, void *pointer, std::size_t size, std::size_t alignment) noexcept {
    if (pointer != nullptr)
      --static_cast<AllocationBudget *>(opaque)->live;
    mem::system_allocator().deallocate(pointer, size, alignment);
  }
  static void *resize(void *opaque, void *pointer, std::size_t old_size,
                      std::size_t new_size, std::size_t alignment) noexcept {
    if (pointer == nullptr)
      return allocate(opaque, new_size, alignment);
    if (new_size == 0) {
      free(opaque, pointer, old_size, alignment);
      return nullptr;
    }
    if (static_cast<AllocationBudget *>(opaque)->remaining.fetch_sub(1) <= 0)
      return nullptr;
    return mem::system_allocator().reallocate(pointer, old_size, new_size, alignment);
  }
  mem::AllocatorRef ref() noexcept { return {allocate, free, resize, this}; }
};

Status derived_constant_control(const compute::Context &cpu,
                                const compute::Context &cuda,
                                const compute::cuda::MemoryResource &host,
                                const compute::cuda::MemoryResource &device,
                                hyle::sched::Scheduler &scheduler) noexcept {
  HYLE_ASSIGN_OR_RETURN(
      auto desc, compute::TensorDesc::dense(compute::DataType::kFloat32,
                                            std::array<std::int64_t, 2>{2, 9}));
  HYLE_ASSIGN_OR_RETURN(auto bf,
                        compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                                   desc.dimensions()));
  HYLE_ASSIGN_OR_RETURN(auto source, tensor(mem::system_resource(), desc));
  HYLE_ASSIGN_OR_RETURN(auto weight, tensor(device.ref(), bf));
  std::array<std::uint16_t, 18> bits;
  for (std::size_t i = 0; i < bits.size(); ++i)
    bits[i] = std::uint16_t(0x3e80 + i * 32);
  HYLE_RETURN_IF_ERROR(
      weight.buffer.view().write(std::as_bytes(std::span(bits))));
  HYLE_ASSIGN_OR_RETURN(auto builder, graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto w, builder.constant(bf));
  HYLE_ASSIGN_OR_RETURN(
      auto before,
      builder.add_node(compute::Operation(compute::TopK{.count = 3}),
                       std::array{x}));
  HYLE_ASSIGN_OR_RETURN(auto cast,
                        builder.add_node(compute::Operation(compute::Cast{
                                             compute::DataType::kFloat32}),
                                         std::array{w}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(cast, before));
  HYLE_ASSIGN_OR_RETURN(
      auto neg, builder.add_node(compute::Operation(
                                     compute::Unary{compute::UnaryKind::kNeg}),
                                 std::array{x}));
  HYLE_ASSIGN_OR_RETURN(auto wide, builder.result(cast));
  HYLE_ASSIGN_OR_RETURN(
      auto after,
      builder.add_node(compute::Operation(compute::TopK{.count = 3}),
                       std::array{wide}));
  HYLE_ASSIGN_OR_RETURN(auto negative, builder.result(neg));
  HYLE_ASSIGN_OR_RETURN(auto selected, builder.result(after));
  HYLE_RETURN_IF_ERROR(builder.publish(negative));
  HYLE_RETURN_IF_ERROR(builder.publish(selected));
  HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
  const std::array placements{
      graph::NodePlacement{before, kCpuTarget,
                           graph::PlacementStrength::kRequired},
      graph::NodePlacement{cast, kCudaTarget,
                           graph::PlacementStrength::kRequired},
      graph::NodePlacement{neg, kCudaTarget,
                           graph::PlacementStrength::kRequired},
      graph::NodePlacement{after, kCpuTarget,
                           graph::PlacementStrength::kRequired}};
  const std::array external{
      graph::ExternalResourceContract{x, kHostStorage},
      graph::ExternalResourceContract{negative, kHostStorage},
      graph::ExternalResourceContract{selected, kHostStorage}};
  const std::array constants{graph::ConstantBinding{w, weight.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(
      auto compiled,
      compile_mixed(logical, cpu, cuda, host, device,
                      {.constants = constants,
                       .node_placements = placements,
                       .external_resources = external}));
  HYLE_RETURN_IF_ERROR(
      require(compiled.report.packed_constant_count() == 1,
              "mixed test did not prepare the BF16-derived constant"));
  const auto segment_for = [&](graph::NodeId node) {
    for (std::size_t s = 0; s < compiled.report.segment_count(); ++s) {
      const auto segment = compiled.report.segment(s).value();
      for (std::size_t i = segment.step_begin;
           i < segment.step_begin + segment.step_count; ++i) {
        const auto step = compiled.report.step(i).value();
        for (std::size_t n = 0; n < step.node_count; ++n)
          if (compiled.report.step_node(i, n).value() == node)
            return s;
      }
    }
    return SIZE_MAX;
  };
  const auto a = segment_for(before), b = segment_for(cast),
             d = segment_for(after);
  HYLE_RETURN_IF_ERROR(
      require(a != SIZE_MAX && b != SIZE_MAX && d != SIZE_MAX && a != d,
              "mixed constant control segments were coalesced"));
  const auto reaches = [&](std::size_t destination,
                           std::size_t source_segment) {
    std::vector<bool> seen(compiled.report.segment_count(), false);
    std::vector<std::size_t> pending{destination};
    while (!pending.empty()) {
      const auto at = pending.back();
      pending.pop_back();
      if (at == source_segment)
        return true;
      if (seen[at])
        continue;
      seen[at] = true;
      for (std::size_t i = 0; i < compiled.report.segment(at)->dependency_count;
           ++i)
        pending.push_back(
            compiled.report.segment_dependency(at, i)->source_segment);
    }
    return false;
  };
  HYLE_RETURN_IF_ERROR(require(reaches(b, a) && reaches(d, b),
                               "mixed prepared constant lost its incoming "
                               "control or outgoing data dependency"));
  HYLE_ASSIGN_OR_RETURN(auto selected_desc,
                        logical.value_description(selected));
  std::array<graph::Invocation, 2> slots;
  std::array<std::array<graph::QueueBinding, 2>, 2> queues;
  hyle::sched::BlockingPool workers(mem::system_allocator(), 2);
  std::array<compute::Queue, 2> gpu_queues, cpu_queues;
  std::array<Tensor, 2> negatives, outputs;
  for (std::size_t i = 0; i < slots.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(slots[i], compiled.executable.create_invocation());
    HYLE_ASSIGN_OR_RETURN(gpu_queues[i], cuda.create_queue());
    HYLE_ASSIGN_OR_RETURN(cpu_queues[i], cpu.create_queue());
    queues[i] = {
        graph::QueueBinding{graph::ExecutionLaneId{0}, &gpu_queues[i]},
        graph::QueueBinding{graph::ExecutionLaneId{1}, &cpu_queues[i]}};
    HYLE_RETURN_IF_ERROR(graph::prepare_progress(slots[i], scheduler, workers));
    HYLE_ASSIGN_OR_RETURN(negatives[i], tensor(mem::system_resource(), desc));
    HYLE_ASSIGN_OR_RETURN(outputs[i],
                          tensor(mem::system_resource(), selected_desc));
  }
  Drain drain{scheduler};
  for (int phase = 0; phase < 2; ++phase) {
    std::array<float, 18> input;
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = float(int((i * 13 + phase * 7) % 37) - 18) / 8;
    HYLE_RETURN_IF_ERROR(
        source.buffer.view().write(std::as_bytes(std::span(input))));
    for (std::size_t i = 0; i < slots.size(); ++i) {
      auto &slot = slots[i];
      HYLE_RETURN_IF_ERROR(slot.bind(x, source.view.as_const()));
      HYLE_RETURN_IF_ERROR(slot.bind(negative, negatives[i].view));
      HYLE_RETURN_IF_ERROR(slot.bind(selected, outputs[i].view));
      HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues[i]));
      HYLE_ASSIGN_OR_RETURN(auto completion, slot.submit(queues[i]));
      HYLE_RETURN_IF_ERROR(completion.wait());
      HYLE_ASSIGN_OR_RETURN(auto result, slot.result(selected));
      std::array<float, 18> actual_neg;
      std::array<float, 6> actual;
      HYLE_RETURN_IF_ERROR(negatives[i].buffer.view().read(
          std::as_writable_bytes(std::span(actual_neg))));
      HYLE_RETURN_IF_ERROR(outputs[i].buffer.view().read(
          std::as_writable_bytes(std::span(actual))));
      for (std::size_t j = 0; j < input.size(); ++j)
        HYLE_RETURN_IF_ERROR(
            require(actual_neg[j] == -input[j],
                    "mixed derived region lost changed input"));
      for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t j = 0; j < 3; ++j)
          HYLE_RETURN_IF_ERROR(
              require(actual[r * 3 + j] ==
                          std::bit_cast<float>(
                              std::uint32_t(bits[r * 9 + 8 - j]) << 16),
                      "mixed prepared constant transfer changed TopK"));
      HYLE_RETURN_IF_ERROR(require(!slot.bind(x, source.view.as_const()).ok(),
                                   "mixed retained result allowed slot reuse"));
      HYLE_ASSIGN_OR_RETURN(auto exported, result.tensor());
      (void)exported;
    }
  }
  std::puts("CPU+CUDA derived BF16 constant: incoming CPU control, CUDA "
            "region, D2H data consumer, two slots/two phases PASS");
  return Status::Ok();
}

Status weight_dependencies(const compute::Context &cpu,
                           const compute::Context &cuda,
                           const compute::cuda::MemoryResource &host,
                           const compute::cuda::MemoryResource &device,
                           hyle::sched::Scheduler &scheduler) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto description, compute::TensorDesc::dense(
      compute::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(mem::system_resource(), description));
  HYLE_ASSIGN_OR_RETURN(auto first, tensor(mem::system_resource(), description));
  HYLE_ASSIGN_OR_RETURN(auto second, tensor(mem::system_resource(), description));
  HYLE_ASSIGN_OR_RETURN(auto output, tensor(mem::system_resource(), description));
  std::array<float, 64> values;
  values.fill(-2.0f);
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
  values.fill(3.0f);
  HYLE_RETURN_IF_ERROR(first.buffer.view().write(std::as_bytes(std::span(values))));
  values.fill(5.0f);
  HYLE_RETURN_IF_ERROR(second.buffer.view().write(std::as_bytes(std::span(values))));
  const std::array catalog{compute::cuda::WeightSpec{7, description},
                           compute::cuda::WeightSpec{8, description}};
  HYLE_ASSIGN_OR_RETURN(auto pool,
      compute::cuda::WeightSlots::create(cuda, host, device, catalog, 512));
  HYLE_ASSIGN_OR_RETURN(auto copy, cuda.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto gpu, cuda.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue, cpu.create_queue());
  HYLE_RETURN_IF_ERROR(copy.reserve_completions(16));
  const std::array fragments{compute::cuda::WeightFragment{7, first.view.as_const()},
                             compute::cuda::WeightFragment{8, second.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto both, pool.upload(0, fragments, copy));
  HYLE_ASSIGN_OR_RETURN(auto builder, graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(description));
  HYLE_ASSIGN_OR_RETURN(auto w1, builder.input(description));
  HYLE_ASSIGN_OR_RETURN(auto w2, builder.input(description));
  HYLE_ASSIGN_OR_RETURN(auto absolute,
      builder.add(compute::Unary{compute::UnaryKind::kAbs}, x));
  HYLE_ASSIGN_OR_RETURN(auto multiplied,
      builder.add(compute::Binary{compute::BinaryKind::kMultiply}, absolute, w1));
  HYLE_ASSIGN_OR_RETURN(auto added,
      builder.add(compute::Binary{compute::BinaryKind::kAdd}, multiplied, w2));
  HYLE_ASSIGN_OR_RETURN(auto y,
      builder.add(compute::Unary{compute::UnaryKind::kAbs}, added));
  HYLE_RETURN_IF_ERROR(builder.publish(y));
  HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
  std::array<graph::NodePlacement, 4> placements;
  for (std::size_t i = 0; i < placements.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, logical.node_id(i));
    placements[i] = {node, i == 0 || i == 3 ? kCpuTarget : kCudaTarget,
                     graph::PlacementStrength::kRequired};
  }
  const std::array external{
      graph::ExternalResourceContract{x, kHostStorage},
      graph::ExternalResourceContract{w1, kDeviceStorage},
      graph::ExternalResourceContract{w2, kDeviceStorage},
      graph::ExternalResourceContract{y, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, compile_mixed(logical, cpu, cuda, host, device,
      {.node_placements = placements, .external_resources = external}));
  HYLE_ASSIGN_OR_RETURN(auto native, compiled.executable.create_invocation());
  const std::array queues{graph::QueueBinding{graph::ExecutionLaneId{0}, &gpu},
                          graph::QueueBinding{graph::ExecutionLaneId{1}, &cpu_queue}};
  auto invocation = std::move(native);
  hyle::sched::BlockingPool workers(mem::system_allocator(), 1);
  HYLE_RETURN_IF_ERROR(graph::prepare_progress(invocation, scheduler, workers));
  Drain drain{scheduler};
  const auto bind = [&](const compute::cuda::WeightBatch &left, std::uint64_t left_id,
                        const compute::cuda::WeightBatch &right, std::uint64_t right_id,
                        std::size_t dependencies) noexcept -> Status {
    HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(graph::cuda::bind(invocation, w1, left, left_id));
    HYLE_RETURN_IF_ERROR(graph::cuda::bind(invocation, w2, right, right_id));
    HYLE_RETURN_IF_ERROR(invocation.bind(y, output.view));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto report, invocation.input_source_report());
    return require(report.device_wait_count == dependencies &&
                       report.host_wait_count == 0,
                   "weight report did not count distinct upload owners");
  };
  const auto check = [&](float expected) noexcept -> Status {
    HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(queues));
    HYLE_RETURN_IF_ERROR(completion.wait());
    HYLE_ASSIGN_OR_RETURN(auto result, invocation.result(y));
    HYLE_ASSIGN_OR_RETURN(auto view, result.tensor());
    std::array<float, 64> actual;
    HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(actual))));
    for (auto value : actual)
      HYLE_RETURN_IF_ERROR(require(value == expected,
                                   "sorted weight bindings changed a tensor ID"));
    return Status::Ok();
  };
  HYLE_RETURN_IF_ERROR(bind(both, 7, both, 8, 1));
  HYLE_RETURN_IF_ERROR(check(11.0f));
  HYLE_RETURN_IF_ERROR(bind(both, 8, both, 7, 1));
  HYLE_RETURN_IF_ERROR(check(13.0f));
  values.fill(7.0f);
  HYLE_RETURN_IF_ERROR(second.buffer.view().write(std::as_bytes(std::span(values))));
  const std::array replacement{compute::cuda::WeightFragment{8, second.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto other, pool.upload(1, replacement, copy));
  HYLE_RETURN_IF_ERROR(bind(both, 7, other, 8, 2));
  HYLE_RETURN_IF_ERROR(check(13.0f));
  HYLE_RETURN_IF_ERROR(invocation.reset());
  HYLE_RETURN_IF_ERROR(bind(both, 7, both, 8, 1));
  HYLE_RETURN_IF_ERROR(check(11.0f));
  HYLE_RETURN_IF_ERROR(invocation.close());
  std::puts("H1 weights: shared batch/two IDs, distinct batches, rebind/reset passed");
  return Status::Ok();
}

Status contracts() noexcept {
  HYLE_ASSIGN_OR_RETURN(auto cpu, graph::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto cuda, graph::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto host, compute::cuda::MemoryResource::create(cuda, compute::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto device, compute::cuda::MemoryResource::create(cuda, compute::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto copy, cuda.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto gpu, cuda.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue, cpu.create_queue());
  HYLE_RETURN_IF_ERROR(copy.reserve_completions(16));
  HYLE_ASSIGN_OR_RETURN(auto row, compute::TensorDesc::dense(compute::DataType::kFloat32, std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, compute::TensorDesc::dense(compute::DataType::kFloat32, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto scale_desc, compute::TensorDesc::dense(compute::DataType::kFloat32, std::array<std::int64_t, 1>{64}));
  HYLE_ASSIGN_OR_RETURN(auto source, tensor(mem::system_resource(), matrix));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(mem::system_resource(), row));
  HYLE_ASSIGN_OR_RETURN(auto scale, tensor(mem::system_resource(), scale_desc));
  HYLE_ASSIGN_OR_RETURN(auto output, tensor(mem::system_resource(), row));
  std::array<float, 4096> data{};
  for (std::size_t i = 0; i < 64; ++i) data[i * 64 + i] = 2.0f;
  std::array<float, 64> ones; ones.fill(1.0f);
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(data))));
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(ones))));
  HYLE_RETURN_IF_ERROR(scale.buffer.view().write(std::as_bytes(std::span(ones))));
  const std::array catalog{compute::cuda::WeightSpec{7, matrix}, compute::cuda::WeightSpec{8, matrix}};
  HYLE_ASSIGN_OR_RETURN(auto pool, compute::cuda::WeightSlots::create(cuda, host, device, catalog, 16384));
  const std::array selected{compute::cuda::WeightFragment{7, source.view.as_const()}};
  const std::array over_capacity{selected[0], compute::cuda::WeightFragment{8, source.view.as_const()}};
  HYLE_RETURN_IF_ERROR(require(!pool.upload(0, over_capacity, copy), "selection exceeded fixed capacity"));
  const std::array wrong_shape{compute::cuda::WeightFragment{7, input.view.as_const()}};
  HYLE_RETURN_IF_ERROR(require(!pool.upload(0, wrong_shape, copy), "wrong shape was accepted"));
  HYLE_ASSIGN_OR_RETURN(auto half_desc, compute::TensorDesc::dense(compute::DataType::kFloat16, std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto half_view, compute::ConstTensorView::create(half_desc, source.buffer.view()));
  const std::array wrong_format{compute::cuda::WeightFragment{7, half_view}};
  HYLE_RETURN_IF_ERROR(require(!pool.upload(0, wrong_format, copy), "wrong physical description was accepted"));
  HYLE_RETURN_IF_ERROR(require(!source.buffer.view().subview(16383, 2), "invalid byte range was accepted"));
  HYLE_RETURN_IF_ERROR(require(!source.buffer.view().subview(SIZE_MAX, 2), "overflow byte range was accepted"));
  auto enormous = compute::TensorDesc::dense(compute::DataType::kFloat32, std::array<std::int64_t, 2>{INT64_MAX, INT64_MAX});
  HYLE_RETURN_IF_ERROR(require(!enormous || !enormous->required_bytes(), "tensor extent overflow was accepted"));
  const std::array duplicate_catalog{catalog[0], catalog[0]};
  HYLE_RETURN_IF_ERROR(require(!compute::cuda::WeightSlots::create(cuda, host, device, duplicate_catalog, 16384), "duplicate catalog ID was accepted"));
  HYLE_RETURN_IF_ERROR(require(!compute::cuda::WeightSlots::create(cuda, host, device, {}, 16384), "empty catalog was accepted"));
  HYLE_RETURN_IF_ERROR(require(!compute::cuda::WeightSlots::create(cuda, host, device, catalog, 0), "zero capacity was accepted"));
  HYLE_ASSIGN_OR_RETURN(auto other, compute::Context::create(cuda.device()));
  HYLE_ASSIGN_OR_RETURN(auto foreign_host, compute::cuda::MemoryResource::create(other, compute::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto foreign_device, compute::cuda::MemoryResource::create(other, compute::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto foreign_queue, other.create_queue());
  HYLE_RETURN_IF_ERROR(require(!compute::cuda::WeightSlots::create(cuda, foreign_host, device, catalog, 16384), "foreign resource Context was accepted"));
  HYLE_RETURN_IF_ERROR(require(!pool.upload(0, selected, foreign_queue), "foreign upload Queue was accepted"));
  HYLE_RETURN_IF_ERROR(require(!pool.upload(2, selected, copy), "invalid slot ID was accepted"));
  std::array<compute::cuda::WeightSpec, 8> bigger_catalog;
  for (std::size_t i = 0; i < 8; ++i) bigger_catalog[i] = {i, matrix};
  AllocationBudget budget;
  for (int limit = 0; limit < 8; ++limit) {
    budget.remaining = limit;
    auto failed = compute::cuda::WeightSlots::create(cuda, host, device, bigger_catalog, 16384, {.allocator = budget.ref()});
    HYLE_RETURN_IF_ERROR(require(!failed && failed.error().IsResourceExhausted(), "weight metadata OOM did not reject"));
    HYLE_RETURN_IF_ERROR(require(budget.live == 0, "partial weight allocation leaked"));
  }
  HYLE_ASSIGN_OR_RETURN(auto weights, pool.upload(0, selected, copy));
  HYLE_RETURN_IF_ERROR(weights.wait());
  // WeightBatch is move-only; the live batch itself guards reuse.
  HYLE_RETURN_IF_ERROR(require(!pool.upload(0, selected, copy), "live batch did not guard reuse"));
  HYLE_ASSIGN_OR_RETURN(auto builder, graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row));
  HYLE_ASSIGN_OR_RETURN(auto s, builder.input(scale_desc));
  HYLE_ASSIGN_OR_RETURN(auto w, builder.input(matrix));
  HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(compute::RmsNorm{}, x, s));
  HYLE_ASSIGN_OR_RETURN(auto projected, builder.add(compute::MatMul{}, normalized, w));
  HYLE_ASSIGN_OR_RETURN(auto y, builder.add(compute::Unary{compute::UnaryKind::kAbs}, projected));
  HYLE_RETURN_IF_ERROR(builder.publish(y));
  HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
  std::array<graph::NodePlacement, 3> placements;
  for (std::size_t i = 0; i < 3; ++i) { HYLE_ASSIGN_OR_RETURN(auto id, logical.node_id(i)); placements[i] = {id, i == 1 ? kCudaTarget : kCpuTarget, graph::PlacementStrength::kRequired}; }
  const std::array external{graph::ExternalResourceContract{x, kHostStorage}, graph::ExternalResourceContract{s, kHostStorage}, graph::ExternalResourceContract{w, kDeviceStorage}, graph::ExternalResourceContract{y, kHostStorage}};
  graph::CompileOptions options{.node_placements = placements, .external_resources = external};
  HYLE_RETURN_IF_ERROR(require(!compile_mixed(logical, cpu, cuda, device, device, options), "wrong resource kind was accepted"));
  HYLE_ASSIGN_OR_RETURN(auto foreign_compiled,
      compile_mixed(logical, cpu, cuda, host, foreign_device, options));
  HYLE_RETURN_IF_ERROR(require(!compile_mixed(logical, cpu, cuda, host, device, options, {0, 0, 0}), "zero workspace budget was ignored"));
  HYLE_ASSIGN_OR_RETURN(auto compiled, compile_mixed(logical, cpu, cuda, host, device, options));
  const std::array ordinary_storage{graph::StorageDomain{device.ref()}, graph::StorageDomain{host.ref()}};
  HYLE_ASSIGN_OR_RETURN(auto target_source_0, hyle::graph::ExecutionSource::acquire(cuda));
  HYLE_ASSIGN_OR_RETURN(auto target_source_1, hyle::graph::ExecutionSource::acquire(cpu));
  const std::array ordinary_targets{graph::ExecutionTarget{target_source_0, kDeviceStorage}, graph::ExecutionTarget{target_source_1, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto ordinary,
      graph::compile(logical, {.storage_domains = ordinary_storage, .targets = ordinary_targets}, options));
  HYLE_RETURN_IF_ERROR(require(ordinary.report.mixed_backends() && ordinary.report.copy_bytes() != 0,
                               "ordinary actual sources did not retain their physical copy route"));
  HYLE_RETURN_IF_ERROR(require(!graph::compile(logical,
      {.storage_domains = ordinary_storage, .targets = std::span(ordinary_targets).first(1)}, options),
      "missing required CPU execution target was accepted"));
  const std::array bindings{graph::QueueBinding{graph::ExecutionLaneId{0}, &gpu}, graph::QueueBinding{graph::ExecutionLaneId{1}, &cpu_queue}};
  {
    // Compilation describes same-device resources. The original CUDA transfer
    // checks its exact Context once the real buffers and Queue are prepared.
    HYLE_ASSIGN_OR_RETURN(auto foreign_weight, tensor(foreign_device.ref(), matrix));
    HYLE_ASSIGN_OR_RETURN(auto foreign_run, foreign_compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(foreign_run.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(foreign_run.bind(s, scale.view.as_const()));
    HYLE_RETURN_IF_ERROR(foreign_run.bind(w, foreign_weight.view.as_const()));
    HYLE_RETURN_IF_ERROR(foreign_run.bind(y, output.view));
    HYLE_RETURN_IF_ERROR(require(!foreign_run.prepare_replay(bindings).ok(),
                                 "original Queue prepared a foreign Context buffer"));
  }
  {
    // Owned host/device bindings use this original executable and its actual
    // domains, plan and Queue bindings.
    auto metadata = mem::StatsAllocator::create(mem::system_allocator()).value();
    HYLE_ASSIGN_OR_RETURN(auto host_leases, compute::LeasePool::create(mem::system_resource(), 1024,
        {.max_blocks = 3, .allocator = metadata.get()}));
    HYLE_ASSIGN_OR_RETURN(auto device_leases, compute::LeasePool::create(device.ref(), 16384,
        {.max_blocks = 1, .allocator = metadata.get()}));
    HYLE_ASSIGN_OR_RETURN(auto x_write, host_leases.allocate(256));
    HYLE_ASSIGN_OR_RETURN(auto s_write, host_leases.allocate(256));
    HYLE_ASSIGN_OR_RETURN(auto destination, host_leases.allocate(256));
    HYLE_ASSIGN_OR_RETURN(auto w_write, device_leases.allocate(16384));
    HYLE_RETURN_IF_ERROR(x_write.view()->write(std::as_bytes(std::span(ones))));
    HYLE_RETURN_IF_ERROR(s_write.view()->write(std::as_bytes(std::span(ones))));
    HYLE_RETURN_IF_ERROR(w_write.view()->write(std::as_bytes(std::span(data))));
    HYLE_ASSIGN_OR_RETURN(auto x_read, x_write.publish());
    HYLE_ASSIGN_OR_RETURN(auto s_read, s_write.publish());
    HYLE_ASSIGN_OR_RETURN(auto w_read, w_write.publish());
    HYLE_ASSIGN_OR_RETURN(auto leased, compiled.executable.create_invocation({.resources = graph::ResourceOptions{.allocator = metadata.get()}}));
    HYLE_RETURN_IF_ERROR(leased.bind(x, row, x_read));
    HYLE_RETURN_IF_ERROR(leased.bind(s, scale_desc, s_read));
    HYLE_RETURN_IF_ERROR(leased.bind(w, matrix, w_read));
    HYLE_RETURN_IF_ERROR(leased.bind(y, row, std::move(destination)));
    x_read.reset(); s_read.reset(); w_read.reset();
    HYLE_RETURN_IF_ERROR(leased.prepare_replay(bindings));
    { HYLE_ASSIGN_OR_RETURN(auto warm, leased.submit(bindings)); HYLE_RETURN_IF_ERROR(warm.wait()); }
    const auto cold = metadata.snapshot().total_allocs;
    for (int iteration = 0; iteration < 8; ++iteration) {
      HYLE_ASSIGN_OR_RETURN(auto result, leased.result(y));
      HYLE_ASSIGN_OR_RETURN(auto view, result.tensor());
      std::array<float, 64> actual;
      HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(actual))));
      for (auto value : actual) HYLE_RETURN_IF_ERROR(require(std::abs(value - 2 / std::sqrt(1.00001f)) < 1e-4f,
          "leased CPU+CUDA mixed math differs"));
      HYLE_RETURN_IF_ERROR(require(!leased.submit(bindings).has_value(), "old mixed output allowed overwrite"));
      result.reset();
      HYLE_ASSIGN_OR_RETURN(auto next, leased.submit(bindings)); HYLE_RETURN_IF_ERROR(next.wait());
    }
    HYLE_RETURN_IF_ERROR(require(metadata.snapshot().total_allocs == cold, "CPU+CUDA lease replay allocated metadata"));
    HYLE_ASSIGN_OR_RETURN(auto held, leased.submit(bindings));
    HYLE_RETURN_IF_ERROR(leased.drain()); HYLE_RETURN_IF_ERROR(held.wait());
    HYLE_RETURN_IF_ERROR(require(host_leases.stats().active_uses == 0 && device_leases.stats().active_uses == 0,
        "CPU+CUDA drain did not retire both domains"));
    HYLE_RETURN_IF_ERROR(require(!leased.submit(bindings).has_value(), "closed leased CPU+CUDA accepted work"));
    std::printf("F1 CPU+CUDA: original mixed plan, leased host/device math, eight allocation-free replays and held completion drain passed\n");
  }
  hyle::sched::Scheduler scheduler(1, mem::system_allocator(), 16);
  hyle::sched::BlockingPool workers(mem::system_allocator(), 2);
  // Exercise the original slot's real cold allocations, rather than an old
  // wrapper's fixed allocation count. The shared budget supplies its allocator.
  budget.remaining = 100000;
  {
    const std::array limits{
        compute::ResourceBudgetLimit{mem::system_resource(), 64 * 1024 * 1024},
        compute::ResourceBudgetLimit{device.ref(), 64 * 1024 * 1024},
        compute::ResourceBudgetLimit{host.ref(), 4 * 1024 * 1024}};
    HYLE_ASSIGN_OR_RETURN(auto metadata_budget, compute::ResourceBudget::create(
        {.limits = limits, .allocator = budget.ref(), .allow_unattributed = true}));
    auto bounded_options = options;
    bounded_options.resource_budget = metadata_budget;
    HYLE_ASSIGN_OR_RETURN(auto bounded,
        compile_mixed(logical, cpu, cuda, host, device, bounded_options));
    HYLE_ASSIGN_OR_RETURN(auto candidate, bounded.executable.create_invocation());
    const auto live = budget.live.load();
    budget.remaining = 0;
    auto failed = graph::cuda::bind(candidate, w, weights, 7);
    HYLE_RETURN_IF_ERROR(require(failed.IsResourceExhausted() && candidate.valid() &&
                                 budget.live == live && !candidate.submit(bindings),
                                 "input-source metadata OOM corrupted the original slot"));
    budget.remaining = 100000;
    HYLE_RETURN_IF_ERROR(graph::cuda::bind(candidate, w, weights, 7));
    const auto with_source = budget.live.load();
    budget.remaining = 0;
    failed = graph::prepare_progress(candidate, scheduler, workers);
    HYLE_RETURN_IF_ERROR(require(failed.IsResourceExhausted() && candidate.valid() &&
                                 budget.live == with_source && !candidate.submit(bindings),
                                 "progress metadata OOM leaked or consumed the slot"));
    budget.remaining = 100000;
    HYLE_RETURN_IF_ERROR(graph::prepare_progress(candidate, scheduler, workers));
    HYLE_RETURN_IF_ERROR(candidate.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(candidate.bind(s, scale.view.as_const()));
    HYLE_RETURN_IF_ERROR(candidate.bind(y, output.view));
    HYLE_RETURN_IF_ERROR(candidate.prepare_replay(bindings));
    HYLE_ASSIGN_OR_RETURN(auto recovered, candidate.submit(bindings));
    HYLE_RETURN_IF_ERROR(recovered.wait());
  }
  scheduler.drain();
  HYLE_RETURN_IF_ERROR(require(budget.live == 0, "cold source/progress OOM left metadata alive"));
  HYLE_ASSIGN_OR_RETURN(auto invocation, compiled.executable.create_invocation());
  HYLE_RETURN_IF_ERROR(graph::prepare_progress(invocation, scheduler, workers));
  HYLE_RETURN_IF_ERROR(require(!invocation.submit(bindings), "unbound graph was submitted"));
  HYLE_RETURN_IF_ERROR(require(!invocation.bind(x, source.view.as_const()).ok(), "wrong input shape was bound"));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(invocation.bind(s, scale.view.as_const()));
  HYLE_RETURN_IF_ERROR(graph::cuda::bind(invocation, w, weights, 7));
  HYLE_RETURN_IF_ERROR(invocation.bind(y, output.view));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(bindings));
  Drain drain{scheduler};
  {
    // Keep the already-ready device bytes borrowed through this scope, rather
    // than inserting a typed weight wait before the CPU prefix under test.
    HYLE_ASSIGN_OR_RETURN(auto ready_weight, weights.binding(7));
    HYLE_RETURN_IF_ERROR(invocation.bind(w, ready_weight));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(bindings));
    // Real busy endpoint rejects the GPU successor after a synchronous CPU
    // prefix was accepted. No mock provider or device-loss injection is used.
    HYLE_ASSIGN_OR_RETURN(auto occupied, compute::detail::QueueSequence::begin(gpu));
    HYLE_ASSIGN_OR_RETURN(auto accepted, invocation.submit(bindings));
    auto terminal = accepted.wait();
    HYLE_RETURN_IF_ERROR(require(terminal.IsFailedPrecondition(), "accepted tail rejection did not fail the result"));
    HYLE_RETURN_IF_ERROR(require(!invocation.result(y), "failed run exposed ready output"));
    HYLE_RETURN_IF_ERROR(require(!invocation.submit(bindings) && !invocation.reset().ok(), "failed native prefix slot was reused"));
  }
  invocation.drain().IgnoreError();
  auto make_run = [&]() noexcept -> Result<graph::Invocation> {
    HYLE_ASSIGN_OR_RETURN(auto next, compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(graph::prepare_progress(next, scheduler, workers));
    return next;
  };
  HYLE_ASSIGN_OR_RETURN(auto reusable, make_run());
  auto bind_run = [&](const compute::cuda::WeightBatch &batch, std::uint64_t id) noexcept -> Status {
    HYLE_RETURN_IF_ERROR(reusable.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(reusable.bind(s, scale.view.as_const()));
    HYLE_RETURN_IF_ERROR(graph::cuda::bind(reusable, w, batch, id));
    HYLE_RETURN_IF_ERROR(reusable.bind(y, output.view));
    return reusable.prepare_replay(bindings);
  };
  HYLE_RETURN_IF_ERROR(bind_run(weights, 7));
  {
    HYLE_ASSIGN_OR_RETURN(auto completion, reusable.submit(bindings));
    HYLE_RETURN_IF_ERROR(completion.wait());
    HYLE_ASSIGN_OR_RETURN(auto initial_result, reusable.result(y));
    HYLE_ASSIGN_OR_RETURN(auto view, initial_result.tensor());
    std::array<float, 64> actual;
    HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(actual))));
    HYLE_RETURN_IF_ERROR(require(std::abs(actual[0] - 2.0f / std::sqrt(1.00001f)) < 0.0001f,
                                 "initial selected payload differs"));
  }
  HYLE_RETURN_IF_ERROR(reusable.reset());
  for (std::size_t i = 0; i < 64; ++i) data[i * 64 + i] = 3.0f;
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(data))));
  const std::array alternate{compute::cuda::WeightFragment{8, source.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto next_batch, pool.upload(1, alternate, copy));
  HYLE_RETURN_IF_ERROR(bind_run(next_batch, 8));
  {
    HYLE_ASSIGN_OR_RETURN(auto completion, reusable.submit(bindings));
    HYLE_RETURN_IF_ERROR(completion.wait());
    HYLE_ASSIGN_OR_RETURN(auto updated_result, reusable.result(y));
    HYLE_ASSIGN_OR_RETURN(auto view, updated_result.tensor());
    std::array<float, 64> actual;
    HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(actual))));
    HYLE_RETURN_IF_ERROR(require(std::abs(actual[0] - 3.0f / std::sqrt(1.00001f)) < 0.0001f,
                                 "changed expert ID/payload did not update fixed program"));
  }
  HYLE_RETURN_IF_ERROR(reusable.close());
  HYLE_ASSIGN_OR_RETURN(auto controlled, make_run());
  HYLE_RETURN_IF_ERROR(controlled.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(controlled.bind(s, scale.view.as_const()));
  HYLE_RETURN_IF_ERROR(graph::cuda::bind(controlled, w, next_batch, 8));
  HYLE_RETURN_IF_ERROR(controlled.bind(y, output.view));
  HYLE_RETURN_IF_ERROR(controlled.prepare_replay(bindings));
  {
    // A bounded real native dependency keeps the run pending without a mock
    // provider. The callback waits only on host input, never on CUDA work.
    NativeGate gate;
    HYLE_ASSIGN_OR_RETURN(gate.prefix, compute::cuda::submit(gpu, [&gate](cudaStream_t stream) noexcept {
      return cudaLaunchHostFunc(stream, &NativeGate::block, &gate);
    }));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    HYLE_RETURN_IF_ERROR(require(gate.entered.load(), "native dependency gate did not enter"));
    HYLE_ASSIGN_OR_RETURN(auto pending, controlled.submit(bindings));
    HYLE_RETURN_IF_ERROR(require(!controlled.result(y), "pending run published a ready result"));
    HYLE_RETURN_IF_ERROR(require(!controlled.submit(bindings) && !controlled.bind(x, input.view.as_const()).ok(),
                                 "in-flight slot admitted reentry or rebind"));
    HYLE_RETURN_IF_ERROR(pending.request_cancel());
    gate.release();
    auto cancelled = pending.wait();
    HYLE_RETURN_IF_ERROR(require(cancelled.IsCancelled(),
                                 "pending native dependency did not preserve cancellation"));
  }
  controlled.drain().IgnoreError();
  HYLE_ASSIGN_OR_RETURN(auto empty_desc, compute::TensorDesc::dense(compute::DataType::kFloat32, std::array<std::int64_t, 1>{0}));
  HYLE_ASSIGN_OR_RETURN(auto empty_storage, tensor(mem::system_resource(), empty_desc));
  HYLE_ASSIGN_OR_RETURN(auto empty_builder, graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto empty_input, empty_builder.input(empty_desc));
  HYLE_RETURN_IF_ERROR(empty_builder.publish(empty_input));
  HYLE_ASSIGN_OR_RETURN(auto empty_logical, std::move(empty_builder).finish());
  const std::array empty_external{graph::ExternalResourceContract{empty_input, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto empty_compiled, compile_mixed(empty_logical, cpu, cuda, host, device, {.external_resources = empty_external}));
  HYLE_ASSIGN_OR_RETURN(auto empty_run, empty_compiled.executable.create_invocation());
  HYLE_RETURN_IF_ERROR(graph::prepare_progress(empty_run, scheduler, workers));
  HYLE_RETURN_IF_ERROR(empty_run.bind(empty_input, empty_storage.view.as_const()));
  HYLE_RETURN_IF_ERROR(empty_run.prepare_replay({}));
  {
    HYLE_ASSIGN_OR_RETURN(auto completion, empty_run.submit());
    HYLE_RETURN_IF_ERROR(completion.wait());
    HYLE_ASSIGN_OR_RETURN(auto empty_result, empty_run.result(empty_input));
    HYLE_ASSIGN_OR_RETURN(auto view, empty_result.tensor());
    HYLE_RETURN_IF_ERROR(require(view.description() == empty_desc, "empty output signature differs"));
  }
  HYLE_RETURN_IF_ERROR(empty_run.close());
  budget.remaining = 100000;
  {
    HYLE_ASSIGN_OR_RETURN(auto tracked_pool,
        compute::cuda::WeightSlots::create(cuda, host, device, catalog, 16384, {.allocator = budget.ref()}));
    HYLE_ASSIGN_OR_RETURN(auto tracked_batch, tracked_pool.upload(0, selected, copy));
    auto tracked_options = options;
    const std::array tracked_limits{
        compute::ResourceBudgetLimit{mem::system_resource(), 64 * 1024 * 1024},
        compute::ResourceBudgetLimit{device.ref(), 64 * 1024 * 1024},
        compute::ResourceBudgetLimit{host.ref(), 4 * 1024 * 1024}};
    HYLE_ASSIGN_OR_RETURN(auto tracked_budget, compute::ResourceBudget::create(
        {.limits = tracked_limits, .allocator = budget.ref(), .allow_unattributed = true}));
    tracked_options.resource_budget = tracked_budget;
    HYLE_ASSIGN_OR_RETURN(auto tracked_compiled,
        compile_mixed(logical, cpu, cuda, host, device, tracked_options));
    HYLE_ASSIGN_OR_RETURN(auto tracked, tracked_compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(graph::prepare_progress(tracked, scheduler, workers));
    HYLE_RETURN_IF_ERROR(tracked.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(tracked.bind(s, scale.view.as_const()));
    HYLE_RETURN_IF_ERROR(graph::cuda::bind(tracked, w, tracked_batch, 7));
    HYLE_RETURN_IF_ERROR(tracked.bind(y, output.view));
    HYLE_RETURN_IF_ERROR(tracked.prepare_replay(bindings));
    HYLE_ASSIGN_OR_RETURN(auto completion, tracked.submit(bindings));
    HYLE_RETURN_IF_ERROR(completion.wait());
    HYLE_ASSIGN_OR_RETURN(auto kept, tracked.result(y));
    tracked = {};
    tracked_batch = {};
    tracked_pool = {};
    tracked_compiled = {};
    HYLE_RETURN_IF_ERROR(require(budget.live > 0, "old lease did not retain its own metadata/backing"));
    HYLE_ASSIGN_OR_RETURN(auto view, kept.tensor());
    std::array<float, 64> actual;
    HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(actual))));
    HYLE_RETURN_IF_ERROR(require(actual[0] > 0, "result lost backing after owners were dropped"));
  }
  scheduler.drain();
  HYLE_RETURN_IF_ERROR(require(budget.live == 0, "late result owner or bound weight metadata leaked"));
  {
    const std::array tight_limits{
        compute::ResourceBudgetLimit{mem::system_resource(), 1024 * 1024},
        compute::ResourceBudgetLimit{device.ref(), 32767},
        compute::ResourceBudgetLimit{host.ref(), 65536}};
    HYLE_ASSIGN_OR_RETURN(auto tight, compute::ResourceBudget::create({.limits = tight_limits}));
    HYLE_ASSIGN_OR_RETURN(auto initial_host, tight.usage(mem::system_resource()));
    HYLE_RETURN_IF_ERROR(require(!tight.reserve_unattributed(1) &&
                                 !tight.reserve_unattributed(std::nullopt),
                                 "strict budget accepted unassigned private storage"));
    auto failed = compute::cuda::WeightSlots::create(cuda, host, device, catalog, 16384,
                                                   {.resource_budget = tight});
    HYLE_RETURN_IF_ERROR(require(!failed && failed.error().IsResourceExhausted(),
                                 "two-slot migration exceeded shared device budget"));
    HYLE_ASSIGN_OR_RETURN(auto failed_device, tight.usage(device.ref()));
    HYLE_ASSIGN_OR_RETURN(auto failed_host, tight.usage(mem::system_resource()));
    HYLE_ASSIGN_OR_RETURN(auto failed_staging, tight.usage(host.ref()));
    HYLE_RETURN_IF_ERROR(require(failed_device.live_bytes == 0 &&
                                 failed_staging.live_bytes == 0 &&
                                 failed_host.live_bytes == initial_host.live_bytes,
                                 "failed pool did not roll back shared reservations"));
    const std::array full_limits{
        compute::ResourceBudgetLimit{mem::system_resource(), 64 * 1024 * 1024},
        compute::ResourceBudgetLimit{device.ref(), 64 * 1024 * 1024},
        compute::ResourceBudgetLimit{host.ref(), 4 * 1024 * 1024}};
    HYLE_ASSIGN_OR_RETURN(auto shared, compute::ResourceBudget::create(
        {.limits = full_limits, .allow_unattributed = true}));
    auto scoped_options = options;
    scoped_options.resource_budget = shared;
    HYLE_ASSIGN_OR_RETURN(auto scoped_compiled,
        compile_mixed(logical, cpu, cuda, host, device, scoped_options));
    HYLE_ASSIGN_OR_RETURN(auto scoped_pool,
        compute::cuda::WeightSlots::create(cuda, host, device, catalog, 16384,
                                         {.resource_budget = shared}));
    HYLE_ASSIGN_OR_RETURN(auto scoped_batch, scoped_pool.upload(0, selected, copy));
    HYLE_RETURN_IF_ERROR(scoped_batch.wait());
    HYLE_ASSIGN_OR_RETURN(auto scoped, scoped_compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(graph::prepare_progress(scoped, scheduler, workers));
    HYLE_RETURN_IF_ERROR(require(!scoped.replay_report(), "unprepared report was published"));
    HYLE_ASSIGN_OR_RETURN(auto aliased_scale,
        compute::ConstTensorView::create(scale_desc, input.buffer.view()));
    HYLE_RETURN_IF_ERROR(scoped.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(scoped.bind(s, aliased_scale));
    HYLE_RETURN_IF_ERROR(graph::cuda::bind(scoped, w, scoped_batch, 7));
    HYLE_RETURN_IF_ERROR(scoped.bind(y, output.view));
    HYLE_RETURN_IF_ERROR(require(!scoped.prepare_replay(bindings, {.host_wait = graph::HostWaitPolicy::kReject}).ok(),
                                 "CPU successor ignored host-wait rejection"));
    HYLE_RETURN_IF_ERROR(scoped.prepare_replay(bindings));
    HYLE_ASSIGN_OR_RETURN(auto scoped_replay, scoped.replay_report());
    HYLE_ASSIGN_OR_RETURN(auto scoped_inputs, scoped.input_source_report());
    HYLE_ASSIGN_OR_RETURN(auto scoped_resources, scoped.resource_report());
    HYLE_ASSIGN_OR_RETURN(auto copy_count, staging_copies(scoped_compiled.report));
    const auto migration = static_cast<std::size_t>(compute::ResourceBudgetCategory::kMigration);
    HYLE_RETURN_IF_ERROR(require(copy_count == 4 && scoped_inputs.device_wait_count == 1 &&
                                 scoped_replay.host_wait_dependency_count != 0,
                                 "prepared paths do not describe actual native boundaries"));
    HYLE_RETURN_IF_ERROR(require(scoped_resources.storage[kHostStorage.ordinal()].boundary_view_bytes == 512 &&
                                 scoped_resources.storage[kDeviceStorage.ordinal()].usage.live_by_category[migration] == 32768 &&
                                 scoped_resources.storage[kStagingStorage.ordinal()].usage.live_by_category[migration] == 32768,
                                 "resource report double-counted aliases or lost migration slots"));
    HYLE_RETURN_IF_ERROR(scoped.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(require(!scoped.replay_report() && !scoped.submit(bindings),
                                 "rebinding did not invalidate preparation"));
    HYLE_RETURN_IF_ERROR(scoped.prepare_replay(bindings));
    HYLE_ASSIGN_OR_RETURN(auto completion, scoped.submit(bindings));
    HYLE_RETURN_IF_ERROR(completion.wait());
    completion = {};
    HYLE_ASSIGN_OR_RETURN(auto retained, scoped.result(y));
    scoped = {};
    scoped_batch = {};
    scoped_pool = {};
    scoped_compiled = {};
    HYLE_ASSIGN_OR_RETURN(auto retained_usage, shared.usage(device.ref()));
    HYLE_RETURN_IF_ERROR(require(retained_usage.live_bytes >= 16384,
                                 "old result returned live device quota too early"));
    retained = {};
    scheduler.drain();
    HYLE_ASSIGN_OR_RETURN(auto released_device, shared.usage(device.ref()));
    HYLE_ASSIGN_OR_RETURN(auto released_staging, shared.usage(host.ref()));
    HYLE_RETURN_IF_ERROR(require(released_device.live_bytes == 0 && released_staging.live_bytes == 0 &&
                                 shared.unattributed_usage().unknown_private_owners == 0,
                                 "last result did not retire shared resource charges"));
  }
  HYLE_RETURN_IF_ERROR(pool.close());
  HYLE_RETURN_IF_ERROR(require(!pool.upload(1, selected, copy), "closed weight pool accepted work"));
  HYLE_RETURN_IF_ERROR(weight_dependencies(cpu, cuda, host, device, scheduler));
  HYLE_RETURN_IF_ERROR(derived_constant_control(cpu,cuda,host,device,scheduler));
  std::puts("H1 contracts: shape/format/context/range/overflow, bounded OOM rollback and accepted prefix failure passed");
  return Status::Ok();
}
}
int main() {
  auto result = contracts();
  if (!result.ok()) { std::fprintf(stderr, "%.*s\n", int(result.message().size()), result.message().data()); return 1; }
  return 0;
}
