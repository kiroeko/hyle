#include <hyle/graph/cpu.h>
#include <hyle/graph/cuda.h>
#include <hyle/graph/cpu_cuda.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(HYLE_G09_HAS_CUPTI)
#include <cupti.h>
#endif

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace b = g::cpu_cuda;
namespace m = hyle::mem;
using hyle::Status;

struct Interval { std::uint64_t begin = 0, end = 0; unsigned phase = 0; };
struct Trace {
  std::atomic<unsigned> phase{0};
  std::mutex mutex;
  std::array<Interval, 4096> cpu{}, gpu{};
  std::size_t cpu_count = 0, gpu_count = 0;
  bool overflow = false;
  void record(bool native, Interval interval) noexcept {
    std::lock_guard lock(mutex);
    auto &count = native ? gpu_count : cpu_count;
    auto &records = native ? gpu : cpu;
    if (count == records.size()) { overflow = true; return; }
    records[count++] = interval;
  }
};
Trace trace;

std::uint64_t timestamp() noexcept {
#if defined(HYLE_G09_HAS_CUPTI)
  std::uint64_t value = 0;
  (void)cuptiGetTimestamp(&value);
  return value;
#else
  return 0;
#endif
}
void cpu_parallel(void *, std::size_t count, c::ParallelTaskFn task, void *context) noexcept {
  const auto phase = trace.phase.load(std::memory_order_acquire);
  const auto begin = phase == 0 ? 0 : timestamp();
  for (std::size_t i = 0; i < count; ++i) task(context, i, count);
  if (phase != 0) trace.record(false, {begin, timestamp(), phase});
}

#if defined(HYLE_G09_HAS_CUPTI)
void CUPTIAPI request_buffer(std::uint8_t **buffer, std::size_t *size, std::size_t *records) {
  *size = 1024 * 1024;
  *buffer = static_cast<std::uint8_t *>(std::malloc(*size));
  if (*buffer == nullptr) *size = 0;
  *records = 0;
}
void CUPTIAPI complete_buffer(CUcontext context, std::uint32_t stream,
    std::uint8_t *buffer, std::size_t, std::size_t valid) {
  CUpti_Activity *record = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
    if (record->kind != CUPTI_ACTIVITY_KIND_KERNEL &&
        record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
    // This route's qualified Toolkit floor is CUDA/CUPTI 13.3.
    const auto *kernel = reinterpret_cast<const CUpti_ActivityKernel12 *>(record);
    if (kernel->start != 0 && kernel->end >= kernel->start)
      trace.record(true, {kernel->start, kernel->end, 0});
  }
  std::size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(context, stream, &dropped) != CUPTI_SUCCESS || dropped != 0) {
    std::lock_guard lock(trace.mutex); trace.overflow = true;
  }
  std::free(buffer);
}
#endif

struct Tensor { m::Buffer buffer; c::TensorView view; };
hyle::Result<Tensor> tensor(m::MemoryResourceRef resource, c::TensorDesc desc) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, desc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto buffer, m::Buffer::create(resource, bytes, 256));
  HYLE_ASSIGN_OR_RETURN(auto view, c::TensorView::create(desc, buffer.view()));
  return Tensor{std::move(buffer), view};
}
Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}

Status branches(bool cpu_first, bool tracing, unsigned workers) {
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context(
      {.executor = c::ParallelExecutorRef{&trace, 1, &cpu_parallel}}));
  HYLE_ASSIGN_OR_RETURN(auto cuda, g::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto pinned, c::cuda::MemoryResource::create(cuda, c::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto device, c::cuda::MemoryResource::create(cuda, c::cuda::MemoryKind::kDevice));
  constexpr std::size_t elements = 4 * 1024 * 1024;
  HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{elements}));
  HYLE_ASSIGN_OR_RETURN(auto a, tensor(device.ref(), desc));
  std::vector<float> values(elements, 1.25f);
  HYLE_RETURN_IF_ERROR(a.buffer.view().write(std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto xi, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto ai, builder.input(desc));
  g::ValueId cv, av;
  if (cpu_first) {
    HYLE_ASSIGN_OR_RETURN(cv, builder.add(c::Unary{c::UnaryKind::kAbs}, xi));
    HYLE_ASSIGN_OR_RETURN(av, builder.add(c::Unary{c::UnaryKind::kNeg}, ai));
  } else {
    HYLE_ASSIGN_OR_RETURN(av, builder.add(c::Unary{c::UnaryKind::kNeg}, ai));
    HYLE_ASSIGN_OR_RETURN(cv, builder.add(c::Unary{c::UnaryKind::kAbs}, xi));
  }
  HYLE_ASSIGN_OR_RETURN(auto out, builder.add(c::Binary{}, cv, av));
  HYLE_RETURN_IF_ERROR(builder.publish(out));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placements{
      g::NodePlacement{*graph.node_id(cpu_first ? 0 : 1), b::kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(cpu_first ? 1 : 0), b::kCudaTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(2), b::kCpuTarget, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{xi, b::kHostStorage},
      g::ExternalResourceContract{ai, b::kDeviceStorage}, g::ExternalResourceContract{out, b::kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, b::compile(graph,
      {.cpu = &cpu, .cuda = &cuda, .staging = &pinned, .device = &device},
      {.node_placements = placements, .external_resources = external}));
  // Both orders must preserve an independent CPU segment before the join.
  HYLE_RETURN_IF_ERROR(require(compiled.report.segment_count() >= 3,
      "independent CPU prefix still inherits the GPU join entry wait"));
  hyle::sched::Scheduler scheduler(workers, m::system_allocator(), 64);
  std::array<c::Queue, 2> cq, gq;
  std::array<Tensor, 2> x, y;
  struct Drain { hyle::sched::Scheduler &scheduler; ~Drain() { scheduler.drain(); } } drain{scheduler};
  std::array<b::Invocation, 2> slots;
  for (unsigned i = 0; i < 2; ++i) {
    HYLE_ASSIGN_OR_RETURN(cq[i], cpu.create_queue());
    HYLE_ASSIGN_OR_RETURN(gq[i], cuda.create_queue());
    HYLE_ASSIGN_OR_RETURN(x[i], tensor(m::system_resource(), desc));
    HYLE_ASSIGN_OR_RETURN(y[i], tensor(m::system_resource(), desc));
    HYLE_ASSIGN_OR_RETURN(auto native, compiled.executable.create_invocation());
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &gq[i]},
        g::QueueBinding{g::ExecutionLaneId{1}, &cq[i]}};
    HYLE_ASSIGN_OR_RETURN(slots[i], b::Invocation::create(std::move(native), scheduler, queues));
  }
  auto bind = [&](unsigned slot, unsigned step) -> Status {
    for (std::size_t i = 0; i < elements; ++i)
      values[i] = float(int(i % 101) - 50) * 0.125f + float(step) * 0.0625f + float(slot) * 0.25f;
    HYLE_RETURN_IF_ERROR(x[slot].buffer.view().write(std::as_bytes(std::span(values))));
    HYLE_RETURN_IF_ERROR(slots[slot].bind(xi, x[slot].view.as_const()));
    HYLE_RETURN_IF_ERROR(slots[slot].bind(ai, a.view.as_const()));
    HYLE_RETURN_IF_ERROR(slots[slot].bind(out, y[slot].view));
    return slots[slot].prepare_replay();
  };
  auto verify = [&](unsigned slot, unsigned step) -> Status {
    HYLE_RETURN_IF_ERROR(y[slot].buffer.view().read(std::as_writable_bytes(std::span(values))));
    for (std::size_t i = 0; i < elements; ++i) {
      const auto source = float(int(i % 101) - 50) * 0.125f + float(step) * 0.0625f + float(slot) * 0.25f;
      if (values[i] != std::abs(source) - 1.25f)
        return Status::Internal("independent CPU/CUDA branch+join oracle differs");
    }
    return Status::Ok();
  };
  // Warm original kernels/programs before observing the single-DAG timeline.
  HYLE_RETURN_IF_ERROR(bind(0, 0));
  { HYLE_ASSIGN_OR_RETURN(auto warm, slots[0].run()); }
  for (unsigned step = 0; step < 8; ++step) {
    HYLE_RETURN_IF_ERROR(bind(0, step));
    trace.phase.store(tracing ? (cpu_first ? 1u : 2u) : 0u, std::memory_order_release);
    { HYLE_ASSIGN_OR_RETURN(auto result, slots[0].run()); }
    trace.phase.store(0, std::memory_order_release);
    HYLE_RETURN_IF_ERROR(verify(0, step));
  }
  for (unsigned step = 0; step < 65; ++step) {
    HYLE_RETURN_IF_ERROR(bind(0, step)); HYLE_RETURN_IF_ERROR(bind(1, step));
    trace.phase.store(tracing ? (cpu_first ? 3u : 4u) : 0u, std::memory_order_release);
    {
      HYLE_ASSIGN_OR_RETURN(auto first, slots[0].submit());
      HYLE_ASSIGN_OR_RETURN(auto second, slots[1].submit());
      HYLE_ASSIGN_OR_RETURN(auto old, first.wait());
      HYLE_ASSIGN_OR_RETURN(auto other, second.wait());
      HYLE_RETURN_IF_ERROR(require(!slots[0].bind(xi, x[0].view.as_const()).ok(),
          "retained result did not prevent slot overwrite"));
      HYLE_RETURN_IF_ERROR(verify(0, step)); HYLE_RETURN_IF_ERROR(verify(1, step));
    }
    trace.phase.store(0, std::memory_order_release);
  }
  HYLE_RETURN_IF_ERROR(bind(0, 91));
  {
    HYLE_ASSIGN_OR_RETURN(auto pending, slots[0].submit());
    HYLE_RETURN_IF_ERROR(pending.request_cancel());
    auto result = pending.wait();
    if (result) HYLE_RETURN_IF_ERROR(verify(0, 91));
    else HYLE_RETURN_IF_ERROR(require(result.error().IsCancelled(), "cancellation lost its status"));
  }
  const auto closed = slots[0].close();
  HYLE_RETURN_IF_ERROR(require(closed.ok() || closed.IsCancelled(), "cancelled slot did not close"));
  HYLE_RETURN_IF_ERROR(bind(1, 92));
  {
    HYLE_ASSIGN_OR_RETURN(auto abandoned, slots[1].submit());
    auto moved = std::move(slots[1]);
    abandoned = {};
    moved = {};
  }
  scheduler.drain();
  std::printf("{\"case\":\"%s\",\"workers\":%u,\"oracle\":\"PASS\",\"single_episodes\":8,\"slots\":2,\"changed_steps\":65,\"cancel_abandon_drain\":\"PASS\"}\n",
      cpu_first ? "cpu-first" : "gpu-first", workers);
  return Status::Ok();
}

#if defined(HYLE_G09_HAS_CUPTI)
Status trace_verdict() {
  std::lock_guard lock(trace.mutex);
  std::array<unsigned, 5> overlap{};
  for (std::size_t i = 0; i < trace.cpu_count; ++i) {
    const auto &cpu = trace.cpu[i];
    std::printf("{\"cpu\":true,\"phase\":%u,\"begin\":%llu,\"end\":%llu}\n", cpu.phase,
        static_cast<unsigned long long>(cpu.begin), static_cast<unsigned long long>(cpu.end));
    for (std::size_t j = 0; j < trace.gpu_count; ++j) {
      const auto &gpu = trace.gpu[j];
      if (cpu.begin < gpu.end && gpu.begin < cpu.end) { ++overlap[cpu.phase]; break; }
    }
  }
  for (std::size_t i = 0; i < trace.gpu_count; ++i)
    std::printf("{\"gpu\":true,\"begin\":%llu,\"end\":%llu}\n",
        static_cast<unsigned long long>(trace.gpu[i].begin), static_cast<unsigned long long>(trace.gpu[i].end));
  std::printf("{\"actual_kernel_intervals\":%zu,\"cpu_intervals\":%zu,\"single_cpu_first_overlap\":%u,\"single_gpu_first_overlap\":%u,\"multi_cpu_first_overlap\":%u,\"multi_gpu_first_overlap\":%u}\n",
      trace.gpu_count, trace.cpu_count, overlap[1], overlap[2], overlap[3], overlap[4]);
  return require(!trace.overflow && overlap[1] != 0 && overlap[2] != 0 && overlap[3] != 0 && overlap[4] != 0,
      "required real CPU/kernel interval overlap not observed or activity records dropped");
}
#endif
} // namespace

int main(int argc, char **argv) {
  const bool tracing = argc == 2 && std::strcmp(argv[1], "--trace") == 0;
#if defined(HYLE_G09_HAS_CUPTI)
  if (tracing && (cuptiActivityRegisterCallbacks(&request_buffer, &complete_buffer) != CUPTI_SUCCESS ||
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS)) {
    std::fprintf(stderr, "CUPTI activity setup failed\n"); return 2;
  }
#else
  if (tracing) { std::fprintf(stderr, "CUPTI was not built for this trace fixture\n"); return 2; }
#endif
  auto status = branches(true, tracing, 2);
  if (status.ok()) status = branches(false, tracing, 2);
  if (status.ok() && !tracing) status = branches(true, false, 1);
  if (status.ok() && !tracing) status = branches(false, false, 1);
#if defined(HYLE_G09_HAS_CUPTI)
  if (tracing) {
    const auto flush = cuptiActivityFlushAll(0);
    const auto disable = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (status.ok() && (flush != CUPTI_SUCCESS || disable != CUPTI_SUCCESS))
      status = Status::Internal("CUPTI activity finalization failed");
    if (status.ok()) status = trace_verdict();
  }
#endif
  if (!status.ok()) {
    std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data()); return 1;
  }
  return 0;
}
