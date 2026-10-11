#include <hyle/compute/batch_transfer.h>
#include <hyle/compute/leased_operation.h>
#include <hyle/compute/low_precision.h>
#include <hyle/graph/leased_invocation.h>
#include <hyle/mem/stats.h>

#if defined(HYLE_COMPUTE_CPU)
#include <hyle/graph/cpu.h>
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include <hyle/compute/backends/cuda.h>
#include <hyle/graph/cuda.h>
#endif
#if defined(HYLE_COMPUTE_VULKAN)
#include <hyle/compute/vulkan_memory.h>
#include <hyle/graph/vulkan.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using hyle::Status;
template <class T> using Result = hyle::Result<T>;

// One BF16 group and one GGUF Q8_0 group. Q8_0 has a half scale followed
// by 32 signed codes per row; the sample keeps that existing file encoding.
struct Payload final {
  std::array<std::uint16_t, 4> bf16_left{0x3f80, 0x4000, 0x4040, 0x4080};
  std::array<std::uint16_t, 4> bf16_right{0x40a0, 0x40c0, 0x40e0, 0x4100};
  std::array<float, 32> activation{};
  std::array<std::byte, 68> q8{};
  explicit Payload(unsigned lane) {
    activation.fill(lane == 0 ? 0.25f : 0.5f);
    if (lane != 0)
      for (auto &bits : bf16_left) bits ^= 0x8000;
    for (unsigned row = 0; row < 2; ++row) {
      q8[row * 34] = std::byte{0};
      q8[row * 34 + 1] = std::byte{0x3c}; // IEEE half 1.0
      for (unsigned col = 0; col < 32; ++col)
        q8[row * 34 + 2 + col] = row == 0 ? std::byte{1} : std::byte{0xfe};
    }
  }
  std::span<const std::byte> bytes(unsigned id) const {
    switch (id) {
    case 0: return std::as_bytes(std::span(bf16_left));
    case 1: return std::as_bytes(std::span(bf16_right));
    case 2: return std::as_bytes(std::span(activation));
    default: return q8;
    }
  }
};
constexpr std::array<std::size_t, 4> kSizes{8, 8, 128, 68};
constexpr std::array<c::CopyPhase, 6> kPhases{{
    {0, 10}, {1, 10}, {2, 10}, {0, 20}, {8, 20}, {16, 20}}};

Result<std::array<c::TensorDesc, 4>> descriptions() {
  HYLE_ASSIGN_OR_RETURN(auto bf, c::TensorDesc::dense(
      c::DataType::kBFloat16, std::array<std::int64_t, 2>{2, 2}));
  HYLE_ASSIGN_OR_RETURN(auto activation, c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 2>{1, 32}));
  HYLE_ASSIGN_OR_RETURN(auto q8, c::gguf_weight_description(
      c::GgufQuantizedType::kQ8_0, c::DataType::kFloat32, 32, 2));
  return std::array{bf, bf, activation, q8};
}

Status check(const c::ReadLease &lease, c::DataType type, unsigned lane,
             unsigned group) {
  HYLE_ASSIGN_OR_RETURN(auto view, lease.view());
  if (type == c::DataType::kBFloat16) {
    std::array<std::uint16_t, 4> bits{};
    HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(bits))));
    const std::array<float, 4> expected{19, 22, 43, 50};
    for (unsigned i = 0; i < bits.size(); ++i) {
      const auto value = std::bit_cast<float>(std::uint32_t(bits[i]) << 16);
      if (value != (lane == 0 ? expected[i] : -expected[i]))
        return Status::Internal("copied BF16 Direct/Graph mathematical result");
    }
  } else {
    std::array<float, 2> values{};
    HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(values))));
    const float scale = lane == 0 ? 1.0f : 2.0f;
    if (group != 1 || !std::isfinite(values[0]) || !std::isfinite(values[1]) ||
        std::abs(values[0] - 8 * scale) > 0.1f ||
        std::abs(values[1] + 16 * scale) > 0.1f)
      return Status::Internal("copied Q8 Direct/Graph mathematical result");
  }
  return Status::Ok();
}

#if defined(HYLE_COMPUTE_CPU) || defined(HYLE_COMPUTE_CUDA)
template <class Compile>
Status consume(c::Context &context, c::Queue &queue,
               m::MemoryResourceRef resource, Compile compile,
               const std::array<c::ReadLease, 4> &ready,
               unsigned lane) {
  HYLE_ASSIGN_OR_RETURN(auto desc, descriptions());
  for (unsigned group = 0; group < 2; ++group) {
    const auto offset = group * 2;
    HYLE_ASSIGN_OR_RETURN(auto out_desc, c::TensorDesc::dense(
        group == 0 ? c::DataType::kBFloat16 : c::DataType::kFloat32,
        std::array<std::int64_t, 2>{group == 0 ? 2 : 1, 2}));
    HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(resource, 4096));
    HYLE_ASSIGN_OR_RETURN(auto output, pool.allocate(*out_desc.required_bytes()));
    HYLE_ASSIGN_OR_RETURN(auto left, ready[offset].tensor(desc[offset]));
    HYLE_ASSIGN_OR_RETURN(auto right, ready[offset + 1].tensor(desc[offset + 1]));
    HYLE_ASSIGN_OR_RETURN(auto destination, output.tensor(out_desc));
    HYLE_ASSIGN_OR_RETURN(auto prepared, c::prepare(
        context, c::Operation(c::MatMul{}), std::array{left, right},
        std::array{destination}));
    c::LeasePool workspace_pool;
    c::WriteLease workspace;
    const auto required = prepared.workspace_requirement();
    if (required.size) {
      const auto alignment = (std::max)(required.alignment, std::size_t{256});
      HYLE_ASSIGN_OR_RETURN(workspace_pool, c::LeasePool::create(
          resource, required.size, {.max_blocks = 1, .backing_alignment = alignment}));
      HYLE_ASSIGN_OR_RETURN(workspace, workspace_pool.allocate(required.size, alignment));
    }
    HYLE_ASSIGN_OR_RETURN(auto direct, c::LeasedOperationInvocation::create(
        prepared, std::move(workspace)));
    const std::array inputs{c::LeasedInput(desc[offset], ready[offset]),
                            c::LeasedInput(desc[offset + 1], ready[offset + 1])};
    const std::array outputs{c::LeasedOutput(out_desc, output)};
    HYLE_ASSIGN_OR_RETURN(auto done, direct.submit(queue, inputs, outputs));
    HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto direct_ready, output.publish());
    HYLE_RETURN_IF_ERROR(check(direct_ready, out_desc.data_type(), lane, group));
    done = {};
    HYLE_RETURN_IF_ERROR(direct.close());

    HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
    HYLE_ASSIGN_OR_RETURN(auto a, builder.input(desc[offset]));
    HYLE_ASSIGN_OR_RETURN(auto b, builder.input(desc[offset + 1]));
    HYLE_ASSIGN_OR_RETURN(auto y, builder.add(c::MatMul{}, a, b));
    HYLE_RETURN_IF_ERROR(builder.publish(y));
    HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
    HYLE_ASSIGN_OR_RETURN(auto compiled, compile(graph));
    HYLE_ASSIGN_OR_RETURN(auto invocation, g::LeasedInvocation::create(compiled.executable));
    HYLE_RETURN_IF_ERROR(invocation.bind(a, desc[offset], ready[offset]));
    HYLE_RETURN_IF_ERROR(invocation.bind(b, desc[offset + 1], ready[offset + 1]));
    HYLE_ASSIGN_OR_RETURN(auto graph_output, pool.allocate(*out_desc.required_bytes()));
    HYLE_RETURN_IF_ERROR(invocation.bind(y, out_desc, std::move(graph_output)));
    HYLE_ASSIGN_OR_RETURN(auto graph_done, invocation.submit(queue));
    HYLE_RETURN_IF_ERROR(graph_done.wait());
    HYLE_ASSIGN_OR_RETURN(auto result, invocation.output(y));
    HYLE_ASSIGN_OR_RETURN(auto lease, result.storage_lease());
    HYLE_RETURN_IF_ERROR(check(lease, out_desc.data_type(), lane, group));
  }
  return Status::Ok();
}

#endif

// The application describes data and stages; every route still executes on
// its own original owner. This helper never dispatches a native copy itself.
template <class Invocation, class Submission, class Prepare, class Create,
          class Consume>
Status transfer_groups(m::MemoryResourceRef host, m::MemoryResourceRef device,
                       Prepare prepare, Create create, Consume consume_ready) {
  std::array<c::CopyResourceSpec, 16> resources;
  std::array<c::LeasePool, 16> pools;
  for (unsigned stage = 0; stage < 4; ++stage)
    for (unsigned id = 0; id < 4; ++id) {
      const auto index = stage * 4 + id;
      resources[index] = {stage == 0 || stage == 3 ? host : device, kSizes[id]};
      HYLE_ASSIGN_OR_RETURN(pools[index], c::LeasePool::create(
          resources[index].resource, 1024, {.max_blocks = 2}));
    }
  HYLE_ASSIGN_OR_RETURN(auto stats, m::StatsAllocator::create(m::system_allocator()));
  HYLE_ASSIGN_OR_RETURN(auto program, prepare(
      resources, c::CopyBatchOptions{.max_copies = 12, .max_phases = 6, .allocator = stats.get()}));
  std::array<Invocation, 2> slots;
  // Observe an accepted lane even when another lane returns an error, before
  // these borrowed resources/Queues leave scope. Submission destruction never waits.
  struct Drain final {
    std::array<Invocation, 2> &slots;
    ~Drain() { for (auto &slot : slots) if (slot.valid()) (void)slot.drain(); }
  } drain{slots};
  for (unsigned lane = 0; lane < 2; ++lane) {
    HYLE_ASSIGN_OR_RETURN(slots[lane], create(program, lane));
  }
  std::array<c::CopyRange, 12> ranges;
  for (unsigned stage = 0; stage < 3; ++stage)
    for (unsigned id = 0; id < 4; ++id)
      ranges[stage * 4 + id] = {{stage * 4 + id}, {(stage + 1) * 4 + id},
                               {id / 2 * 3 + stage}, 0, 0, kSizes[id]};
  const auto cold_allocations = stats.snapshot().total_allocs;
  for (unsigned replay = 0; replay < 3; ++replay) {
    std::array<Submission, 2> submissions;
    for (unsigned lane = 0; lane < 2; ++lane) {
      const Payload payload(lane);
      for (unsigned id = 0; id < 16; ++id) {
        HYLE_ASSIGN_OR_RETURN(auto writer, pools[id].allocate(kSizes[id % 4]));
        if (id < 4) {
          HYLE_RETURN_IF_ERROR(writer.view()->write(payload.bytes(id)));
          HYLE_ASSIGN_OR_RETURN(auto source, writer.publish());
          HYLE_RETURN_IF_ERROR(slots[lane].bind_source({id}, source));
        } else {
          HYLE_RETURN_IF_ERROR(slots[lane].bind_mutable({id}, std::move(writer)));
        }
      }
      HYLE_RETURN_IF_ERROR(slots[lane].prepare_replay(kPhases, ranges));
      HYLE_ASSIGN_OR_RETURN(submissions[lane], slots[lane].submit());
    }
    for (unsigned lane = 0; lane < 2; ++lane) {
      HYLE_RETURN_IF_ERROR(submissions[lane].wait());
      std::array<c::ReadLease, 4> ready;
      const Payload payload(lane);
      for (unsigned id = 0; id < 4; ++id) {
        HYLE_ASSIGN_OR_RETURN(ready[id], submissions[lane].output(
            {id / 2 * 3 + 1}, {8 + id}, 0, kSizes[id]));
        HYLE_ASSIGN_OR_RETURN(auto readback, submissions[lane].output(
            {id / 2 * 3 + 2}, {12 + id}, 0, kSizes[id]));
        std::array<std::byte, 128> bytes{};
        const auto actual = std::span(bytes).first(kSizes[id]);
        HYLE_RETURN_IF_ERROR(readback.view()->read(actual));
        if (!std::ranges::equal(actual, payload.bytes(id)))
          return Status::Internal("batch changed BF16 or GGUF encoded bytes");
      }
      HYLE_RETURN_IF_ERROR(consume_ready(ready, lane));
      const auto report = submissions[lane].report();
      if (report.copies != 12 || report.phase_completions != 6 ||
          report.logical_bytes != 3 * (8 + 8 + 128 + 68))
        return Status::Internal("copy phase accounting differs");
    }
    if (stats.snapshot().total_allocs != cold_allocations)
      return Status::Internal("batch metadata grew during replay");
    submissions = {};
    for (auto &slot : slots) HYLE_RETURN_IF_ERROR(slot.reset());
    for (auto &pool : pools)
      if (pool.stats().live_blocks != 0 || pool.stats().active_uses != 0)
        return Status::Internal("math consumer retained a retired copy block");
  }
  return Status::Ok();
}

#if defined(HYLE_COMPUTE_CPU) || defined(HYLE_COMPUTE_CUDA)
template <class Compile>
Status common(c::Context &context, m::MemoryResourceRef host,
              m::MemoryResourceRef device, Compile compile) {
  std::array<c::Queue, 2> queues;
  for (auto &queue : queues) {
    HYLE_ASSIGN_OR_RETURN(queue, context.create_queue());
  }
  return transfer_groups<c::CopyBatchInvocation, c::CopyBatchSubmission>(
      host, device,
      [&](auto resources, auto options) { return c::CopyBatchProgram::prepare(context, resources, options); },
      [&](const auto &program, unsigned lane) { return c::CopyBatchInvocation::create(program, queues[lane]); },
      [&](const auto &ready, unsigned lane) { return consume(context, queues[lane], device, compile, ready, lane); });
}
#endif

#if defined(HYLE_COMPUTE_VULKAN)
Status vulkan() {
  namespace v = c::vulkan;
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  const auto &environment = context.environment();
  std::printf("Vulkan device: %s; driver: %s %s; software: %s\n",
              environment.device_name.c_str(), environment.driver_name.c_str(),
              environment.driver_info.c_str(), environment.software_device ? "yes" : "no");
  HYLE_ASSIGN_OR_RETURN(auto host, v::MemoryResource::create(context, v::MemoryKind::kStaging));
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  const auto readable = [&](const v::Tensor &tensor) -> Result<c::ReadLease> {
    HYLE_ASSIGN_OR_RETURN(auto bytes, tensor.description().required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto host_buffer, m::Buffer::create(m::system_resource(), bytes, 64));
    HYLE_ASSIGN_OR_RETURN(auto mapped, host_buffer.view().map_host(m::HostAccess::kWrite));
    HYLE_ASSIGN_OR_RETURN(auto destination, mapped.writable_bytes());
    HYLE_RETURN_IF_ERROR(context.download(tensor, destination));
    HYLE_RETURN_IF_ERROR(mapped.close());
    return c::ReadLease::adopt(std::move(host_buffer));
  };
  const auto consume_ready = [&](const std::array<c::ReadLease, 4> &ready, unsigned lane) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto desc, descriptions());
    for (unsigned group = 0; group < 2; ++group) {
      const auto offset = group * 2;
      const std::array inputs{v::LeasedInput{desc[offset], ready[offset]},
                              v::LeasedInput{desc[offset + 1], ready[offset + 1]}};
      HYLE_ASSIGN_OR_RETURN(auto direct_program, v::prepare(context, c::MatMul{}, desc[offset], desc[offset + 1]));
      HYLE_ASSIGN_OR_RETURN(auto direct, v::LeasedInvocation::create(direct_program, device));
      HYLE_RETURN_IF_ERROR(direct.bind(inputs));
      HYLE_ASSIGN_OR_RETURN(auto done, direct.submit());
      HYLE_ASSIGN_OR_RETURN(auto result, done.wait());
      HYLE_ASSIGN_OR_RETURN(auto output, readable(result->tensors()[0]));
      HYLE_RETURN_IF_ERROR(check(output, desc[offset].data_type(), lane, group));

      HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
      HYLE_ASSIGN_OR_RETURN(auto a, builder.input(desc[offset]));
      HYLE_ASSIGN_OR_RETURN(auto b, builder.input(desc[offset + 1]));
      HYLE_ASSIGN_OR_RETURN(auto y, builder.add(c::MatMul{}, a, b));
      HYLE_RETURN_IF_ERROR(builder.publish(y));
      HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
      HYLE_ASSIGN_OR_RETURN(auto program, g::vulkan::compile(context, graph));
      HYLE_ASSIGN_OR_RETURN(auto invocation, v::LeasedInvocation::create(program, device));
      HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
      HYLE_ASSIGN_OR_RETURN(auto graph_done, invocation.submit());
      HYLE_ASSIGN_OR_RETURN(auto graph_result, graph_done.wait());
      HYLE_ASSIGN_OR_RETURN(auto graph_output, readable(graph_result->tensors()[0]));
      HYLE_RETURN_IF_ERROR(check(graph_output, desc[offset].data_type(), lane, group));
    }
    return Status::Ok();
  };
  return transfer_groups<v::CopyBatchInvocation, v::CopyBatchSubmission>(
      host.ref(), device.ref(),
      [&](auto resources, auto options) { return v::CopyBatchProgram::prepare(context, resources, options); },
      [](const auto &program, unsigned) { return v::CopyBatchInvocation::create(program); }, consume_ready);
}
#endif

Status run() {
#if defined(HYLE_COMPUTE_CPU)
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_RETURN_IF_ERROR(common(cpu, m::system_resource(), m::system_resource(),
      [&](const g::Graph &graph) { return g::compile(graph, cpu); }));
  std::puts("CPU: two copy groups/two slots, BF16+Q8 Direct/Graph and three replays passed");
#endif
#if defined(HYLE_COMPUTE_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto cuda, g::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto host, c::cuda::MemoryResource::create(cuda, c::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto device, c::cuda::MemoryResource::create(cuda, c::cuda::MemoryKind::kDevice));
  HYLE_RETURN_IF_ERROR(common(cuda, host.ref(), device.ref(),
      [&](const g::Graph &graph) { return g::cuda::compile(graph, cuda, device.ref()); }));
  std::puts("CUDA: staged upload/D2D/readback, BF16+Q8 Direct/Graph and three replays passed");
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  HYLE_RETURN_IF_ERROR(vulkan());
  std::puts("Vulkan: staged upload/D2D/readback, BF16+Q8 Direct/Graph and three replays passed");
#endif
#if !defined(HYLE_COMPUTE_CPU) && !defined(HYLE_COMPUTE_CUDA) && !defined(HYLE_COMPUTE_VULKAN)
  return Status::Unimplemented("batched transfer math sample requires a connected route");
#endif
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = run();
  if (status.ok()) return 0;
  const auto message = status.message();
  std::fprintf(stderr, "batched transfers: %.*s\n", static_cast<int>(message.size()), message.data());
  return 1;
}
