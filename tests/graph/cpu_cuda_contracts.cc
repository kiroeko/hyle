// Reuse the actual consumer's public C++ setup helpers. Its main is not run:
// these checks cover distinct rejection/rollback/terminal contracts.
#define main hyle_cpu_cuda_consumer_main
#include "../../samples/package/graph/cpu_cuda/main.cc"
#undef main

#include <atomic>
#include <climits>
#include <chrono>
#include <thread>
#include <hyle/compute/detail/queue_sequence.h>

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
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(host.ref(), row));
  HYLE_ASSIGN_OR_RETURN(auto scale, tensor(host.ref(), scale_desc));
  HYLE_ASSIGN_OR_RETURN(auto output, tensor(host.ref(), row));
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
  HYLE_ASSIGN_OR_RETURN(auto foreign_queue, other.create_queue());
  HYLE_RETURN_IF_ERROR(require(!compute::cuda::WeightSlots::create(cuda, foreign_host, device, catalog, 16384), "foreign resource Context was accepted"));
  HYLE_RETURN_IF_ERROR(require(!pool.upload(0, selected, foreign_queue), "foreign upload Queue was accepted"));
  HYLE_RETURN_IF_ERROR(require(!pool.upload(2, selected, copy), "invalid slot ID was accepted"));
  std::array<compute::cuda::WeightSpec, 8> bigger_catalog;
  for (std::size_t i = 0; i < 8; ++i) bigger_catalog[i] = {i, matrix};
  AllocationBudget budget;
  for (int limit = 0; limit < 8; ++limit) {
    budget.remaining = limit;
    auto failed = compute::cuda::WeightSlots::create(cuda, host, device, bigger_catalog, 16384, budget.ref());
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
  for (std::size_t i = 0; i < 3; ++i) { HYLE_ASSIGN_OR_RETURN(auto id, logical.node_id(i)); placements[i] = {id, i == 1 ? bridge::kCudaTarget : bridge::kCpuTarget, graph::PlacementStrength::kRequired}; }
  const std::array external{graph::ExternalResourceContract{x, bridge::kHostStorage}, graph::ExternalResourceContract{s, bridge::kHostStorage}, graph::ExternalResourceContract{w, bridge::kDeviceStorage}, graph::ExternalResourceContract{y, bridge::kHostStorage}};
  graph::CompileOptions options{.node_placements = placements, .external_resources = external};
  HYLE_RETURN_IF_ERROR(require(!bridge::compile(logical, {&cpu, &cuda, &device, &device}, options), "wrong resource kind was accepted"));
  HYLE_RETURN_IF_ERROR(require(!bridge::compile(logical, {&cpu, &cuda, &foreign_host, &device}, options), "mixed graph foreign resource was accepted"));
  HYLE_RETURN_IF_ERROR(require(!bridge::compile(logical, {&cpu, &cuda, &host, &device, 0, 0}, options), "zero workspace budget was ignored"));
  HYLE_ASSIGN_OR_RETURN(auto compiled, bridge::compile(logical, {&cpu, &cuda, &host, &device}, options));
  const std::array ordinary_storage{graph::StorageDomain{device.ref()}, graph::StorageDomain{host.ref()}};
  const std::array ordinary_targets{graph::ExecutionTarget{&cuda, bridge::kDeviceStorage}, graph::ExecutionTarget{&cpu, bridge::kHostStorage}};
  HYLE_RETURN_IF_ERROR(require(!graph::compile(logical, {.storage_domains = ordinary_storage, .targets = ordinary_targets}, options),
                               "ordinary compile silently enabled cross-device work"));
  const std::array bindings{graph::QueueBinding{graph::ExecutionLaneId{0}, &gpu}, graph::QueueBinding{graph::ExecutionLaneId{1}, &cpu_queue}};
  hyle::sched::Scheduler scheduler(1, mem::system_allocator(), 16);
  HYLE_ASSIGN_OR_RETURN(auto native, compiled.executable.create_invocation());
  for (int limit = 0; limit < 3; ++limit) {
    budget.remaining = limit;
    auto failed = bridge::Invocation::create(std::move(native), scheduler, bindings, budget.ref());
    HYLE_RETURN_IF_ERROR(require(!failed && failed.error().IsResourceExhausted() && native.valid(), "invocation OOM consumed its caller input"));
    HYLE_RETURN_IF_ERROR(require(budget.live == 0, "partial invocation metadata leaked"));
  }
  HYLE_ASSIGN_OR_RETURN(auto invocation, bridge::Invocation::create(std::move(native), scheduler, bindings));
  HYLE_RETURN_IF_ERROR(require(!invocation.submit(), "unbound graph was submitted"));
  HYLE_RETURN_IF_ERROR(require(!invocation.bind(x, source.view.as_const()).ok(), "wrong input shape was bound"));
  HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(invocation.bind(s, scale.view.as_const()));
  HYLE_RETURN_IF_ERROR(invocation.bind(w, weights, 7));
  HYLE_RETURN_IF_ERROR(invocation.bind(y, output.view));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay());
  Drain drain{scheduler};
  {
    // Real busy endpoint rejects the GPU successor after a synchronous CPU
    // prefix was accepted. No mock provider or device-loss injection is used.
    HYLE_ASSIGN_OR_RETURN(auto occupied, compute::detail::QueueSequence::begin(gpu));
    HYLE_ASSIGN_OR_RETURN(auto accepted, invocation.submit());
    auto terminal = accepted.wait();
    HYLE_RETURN_IF_ERROR(require(!terminal && terminal.error().IsFailedPrecondition(), "accepted tail rejection did not fail the result"));
    HYLE_RETURN_IF_ERROR(require(!accepted.result(), "failed run exposed ready output"));
    HYLE_RETURN_IF_ERROR(require(!invocation.submit() && !invocation.reset().ok(), "failed native prefix slot was reused"));
  }
  invocation.close().IgnoreError();
  auto make_run = [&]() noexcept -> Result<bridge::Invocation> {
    HYLE_ASSIGN_OR_RETURN(auto next, compiled.executable.create_invocation());
    return bridge::Invocation::create(std::move(next), scheduler, bindings);
  };
  HYLE_ASSIGN_OR_RETURN(auto reusable, make_run());
  auto bind_run = [&](const compute::cuda::WeightBatch &batch, std::uint64_t id) noexcept -> Status {
    HYLE_RETURN_IF_ERROR(reusable.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(reusable.bind(s, scale.view.as_const()));
    HYLE_RETURN_IF_ERROR(reusable.bind(w, batch, id));
    HYLE_RETURN_IF_ERROR(reusable.bind(y, output.view));
    return reusable.prepare_replay();
  };
  HYLE_RETURN_IF_ERROR(bind_run(weights, 7));
  {
    HYLE_ASSIGN_OR_RETURN(auto initial_result, reusable.run());
    HYLE_ASSIGN_OR_RETURN(auto view, initial_result->output(y));
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
    HYLE_ASSIGN_OR_RETURN(auto updated_result, reusable.run());
    HYLE_ASSIGN_OR_RETURN(auto view, updated_result->output(y));
    std::array<float, 64> actual;
    HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(actual))));
    HYLE_RETURN_IF_ERROR(require(std::abs(actual[0] - 3.0f / std::sqrt(1.00001f)) < 0.0001f,
                                 "changed expert ID/payload did not update fixed program"));
  }
  HYLE_RETURN_IF_ERROR(reusable.close());
  HYLE_ASSIGN_OR_RETURN(auto controlled, make_run());
  HYLE_RETURN_IF_ERROR(controlled.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(controlled.bind(s, scale.view.as_const()));
  HYLE_RETURN_IF_ERROR(controlled.bind(w, next_batch, 8));
  HYLE_RETURN_IF_ERROR(controlled.bind(y, output.view));
  HYLE_RETURN_IF_ERROR(controlled.prepare_replay());
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
    HYLE_ASSIGN_OR_RETURN(auto pending, controlled.submit());
    HYLE_RETURN_IF_ERROR(require(!pending.result(), "pending run published a ready result"));
    HYLE_RETURN_IF_ERROR(require(!controlled.submit() && !controlled.bind(x, input.view.as_const()).ok(),
                                 "in-flight slot admitted reentry or rebind"));
    HYLE_RETURN_IF_ERROR(pending.request_cancel());
    gate.release();
    auto cancelled = pending.wait();
    HYLE_RETURN_IF_ERROR(require(!cancelled && cancelled.error().IsCancelled(),
                                 "pending native dependency did not preserve cancellation"));
  }
  controlled.close().IgnoreError();
  HYLE_ASSIGN_OR_RETURN(auto empty_desc, compute::TensorDesc::dense(compute::DataType::kFloat32, std::array<std::int64_t, 1>{0}));
  HYLE_ASSIGN_OR_RETURN(auto empty_storage, tensor(host.ref(), empty_desc));
  HYLE_ASSIGN_OR_RETURN(auto empty_builder, graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto empty_input, empty_builder.input(empty_desc));
  HYLE_RETURN_IF_ERROR(empty_builder.publish(empty_input));
  HYLE_ASSIGN_OR_RETURN(auto empty_logical, std::move(empty_builder).finish());
  const std::array empty_external{graph::ExternalResourceContract{empty_input, bridge::kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto empty_compiled, bridge::compile(empty_logical, {&cpu, &cuda, &host, &device}, {.external_resources = empty_external}));
  HYLE_ASSIGN_OR_RETURN(auto empty_native, empty_compiled.executable.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto empty_run, bridge::Invocation::create(std::move(empty_native), scheduler, {}));
  HYLE_RETURN_IF_ERROR(empty_run.bind(empty_input, empty_storage.view.as_const()));
  HYLE_RETURN_IF_ERROR(empty_run.prepare_replay());
  {
    HYLE_ASSIGN_OR_RETURN(auto empty_result, empty_run.run());
    HYLE_ASSIGN_OR_RETURN(auto view, empty_result->output(empty_input));
    HYLE_RETURN_IF_ERROR(require(view.description() == empty_desc, "empty output signature differs"));
  }
  HYLE_RETURN_IF_ERROR(empty_run.close());
  budget.remaining = 1000;
  {
    HYLE_ASSIGN_OR_RETURN(auto tracked_pool,
        compute::cuda::WeightSlots::create(cuda, host, device, catalog, 16384, budget.ref()));
    HYLE_ASSIGN_OR_RETURN(auto tracked_batch, tracked_pool.upload(0, selected, copy));
    HYLE_ASSIGN_OR_RETURN(auto next, compiled.executable.create_invocation());
    HYLE_ASSIGN_OR_RETURN(auto tracked,
        bridge::Invocation::create(std::move(next), scheduler, bindings, budget.ref()));
    HYLE_RETURN_IF_ERROR(tracked.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(tracked.bind(s, scale.view.as_const()));
    HYLE_RETURN_IF_ERROR(tracked.bind(w, tracked_batch, 7));
    HYLE_RETURN_IF_ERROR(tracked.bind(y, output.view));
    HYLE_RETURN_IF_ERROR(tracked.prepare_replay());
    HYLE_ASSIGN_OR_RETURN(auto kept, tracked.run());
    tracked = {};
    tracked_batch = {};
    tracked_pool = {};
    HYLE_RETURN_IF_ERROR(require(budget.live > 0, "old lease did not retain its own metadata/backing"));
    HYLE_ASSIGN_OR_RETURN(auto view, kept->output(y));
    std::array<float, 64> actual;
    HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(actual))));
    HYLE_RETURN_IF_ERROR(require(actual[0] > 0, "result lost backing after owners were dropped"));
  }
  scheduler.drain();
  HYLE_RETURN_IF_ERROR(require(budget.live == 0, "late result owner or bound weight metadata leaked"));
  HYLE_RETURN_IF_ERROR(pool.close());
  HYLE_RETURN_IF_ERROR(require(!pool.upload(1, selected, copy), "closed weight pool accepted work"));
  std::puts("H1 contracts: shape/format/context/range/overflow, bounded OOM rollback and accepted prefix failure passed");
  return Status::Ok();
}
}
int main() {
  auto result = contracts();
  if (!result.ok()) { std::fprintf(stderr, "%.*s\n", int(result.message().size()), result.message().data()); return 1; }
  return 0;
}
