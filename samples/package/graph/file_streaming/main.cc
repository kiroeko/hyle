#include <hyle/compute/file_stream.h>
#include <hyle/platform/fs.h>
#include <hyle/sched/join.h>
#include <hyle/compute/leased_operation.h>
#include <hyle/compute/low_precision.h>
#include <hyle/graph/executable.h>
#include <hyle/mem/stats.h>
#include "gguf_input.h"

#if defined(HYLE_COMPUTE_CPU)
#include <hyle/compute/cpu_file_stream.h>
#include <hyle/graph/cpu.h>
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include <hyle/compute/cuda_file_stream.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/graph/cuda.h>
#endif
#if defined(HYLE_COMPUTE_APPLE)
#include <hyle/compute/apple_file_stream.h>
#include <hyle/compute/apple_memory.h>
#include <hyle/graph/apple.h>
#endif
#if defined(HYLE_COMPUTE_VULKAN)
#include <hyle/compute/vulkan_file_stream.h>
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
#include <vector>
#include <string>
#include <atomic>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using hyle::Status;
template <class T> using Result = hyle::Result<T>;
const gguf_input::Fixture *gguf_fixture = nullptr;

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
               unsigned lane, unsigned selected) {
  HYLE_ASSIGN_OR_RETURN(auto desc, descriptions());
  for (unsigned group = selected; group < selected + 1; ++group) {
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
    HYLE_ASSIGN_OR_RETURN(auto invocation, compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(invocation.bind(a, desc[offset], ready[offset]));
    HYLE_RETURN_IF_ERROR(invocation.bind(b, desc[offset + 1], ready[offset + 1]));
    HYLE_ASSIGN_OR_RETURN(auto graph_output, pool.allocate(*out_desc.required_bytes()));
    HYLE_RETURN_IF_ERROR(invocation.bind(y, out_desc, std::move(graph_output)));
    HYLE_ASSIGN_OR_RETURN(auto graph_done, invocation.submit(queue));
    HYLE_RETURN_IF_ERROR(graph_done.wait());
    HYLE_ASSIGN_OR_RETURN(auto result, invocation.result(y));
    HYLE_ASSIGN_OR_RETURN(auto lease, result.storage_lease());
    HYLE_RETURN_IF_ERROR(check(lease, out_desc.data_type(), lane, group));
  }
  return Status::Ok();
}

#endif

// The application supplies ranges/encodings; the ordinary route factory
// owns IO/native ordering and bounded retirement. Math uses the existing
// Direct/Graph consumers above, never a copy or math kernel in this sample.
hyle::sched::Task<Result<hyle::io::File>> open_source(m::AllocatorRef,
    hyle::io::IoContext *io, std::string path, bool read_only) {
  co_return co_await hyle::io::open_file(*io, 0, path,
      {.access = read_only ? hyle::io::FileAccess::kRead : hyle::io::FileAccess::kReadWrite});
}
template<std::size_t N, class Create, class MakePayload, class Consume>
Status transfer_groups(m::MemoryResourceRef host, m::MemoryResourceRef device,
    const std::array<std::size_t, N> &sizes, Create create,
    MakePayload make_payload, Consume consume_ready) {
  static_assert(N >= 4 && N <= 8);
  hyle::sched::Scheduler scheduler(2, m::system_allocator(), 256);
  hyle::io::IoContext io(scheduler, {.max_blocking_threads = 2});
  if (!io.valid()) return io.status();
  struct Shutdown final { hyle::sched::Scheduler &scheduler; ~Shutdown() { scheduler.shutdown(); } } shutdown{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto stats, m::StatsAllocator::create(m::system_allocator()));
  const std::array targets{c::CopyResourceSpec{device, 4096}};
  HYLE_ASSIGN_OR_RETURN(auto pipeline, create(io, targets, c::FileStreamOptions{
      .allocator = stats.get(), .max_requests = 4, .max_groups_per_request = 2,
      .max_ranges_per_group = static_cast<std::uint32_t>(N), .in_flight_slots = 2,
      .staging_bytes_per_slot = 64, .staging_alignment = 64}));
  struct Drain final { c::FileResourcePipeline &pipeline; ~Drain() { pipeline.drain().IgnoreError(); } } drain{pipeline};
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(device, 16384));
  std::array<std::uint64_t, N> offsets{};
  std::size_t total = 0;
  for (std::size_t i = 0; i < N; ++i) { offsets[i] = total; total += sizes[i]; }
  static std::atomic<unsigned> sequence{0};
  for (unsigned replay = 0; replay < 3; ++replay) {
    for (unsigned lane = 0; lane < 2; ++lane) {
      const auto payload = make_payload(lane);
      std::vector<std::byte> file_bytes;
      for (unsigned i = 0; i < N; ++i) file_bytes.insert(file_bytes.end(), payload.bytes(i).begin(), payload.bytes(i).end());
      const auto stem = "hyle_f3_math_" + std::to_string(sequence.fetch_add(1));
      const auto external = N == 4 && gguf_fixture;
      const auto source_path = external ? gguf_fixture->file : stem + "_source.tmp", sink_path = stem + "_sink.tmp";
      struct Files final { const std::string &a, &b; bool external; ~Files() {
        if (!external) hyle::platform::remove_file(a.c_str()).IgnoreError();
        hyle::platform::remove_file(b.c_str()).IgnoreError(); } } files{source_path, sink_path, external};
      if (!external) {
        HYLE_ASSIGN_OR_RETURN(auto raw, hyle::platform::open_file(source_path.c_str(),
            hyle::platform::kOpenRead | hyle::platform::kOpenWrite | hyle::platform::kOpenCreate | hyle::platform::kOpenTruncate));
        HYLE_RETURN_IF_ERROR(raw.write_all(file_bytes)); HYLE_RETURN_IF_ERROR(raw.close());
      }
      HYLE_ASSIGN_OR_RETURN(auto empty, hyle::platform::open_file(sink_path.c_str(),
          hyle::platform::kOpenRead | hyle::platform::kOpenWrite | hyle::platform::kOpenCreate | hyle::platform::kOpenTruncate));
      HYLE_RETURN_IF_ERROR(empty.close());
      auto open = [&](const std::string &path) -> Result<c::StreamFile> {
        HYLE_ASSIGN_OR_RETURN(auto opened, hyle::sched::spawn_task_on(scheduler, 0, m::system_allocator(),
            open_source(m::system_allocator(), &io, path, path == source_path)).join());
        HYLE_ASSIGN_OR_RETURN(auto file, std::move(opened));
        return c::StreamFile::adopt(std::move(file), external && path == source_path ? gguf_fixture->capacity : total);
      };
      HYLE_ASSIGN_OR_RETURN(auto source, open(source_path)); HYLE_ASSIGN_OR_RETURN(auto sink, open(sink_path));
      auto make_ranges = [&](bool original) -> Result<std::array<c::StreamReadRange, N>> {
        std::array<c::StreamReadRange, N> ranges;
        for (unsigned i = 0; i < N; ++i) {
          ranges[i].file_offset = offsets[i]; ranges[i].bytes = sizes[i];
          if constexpr (N == 4) {
            if (external && original) ranges[i].file_offset = gguf_fixture->offsets[lane][i];
          }
          HYLE_ASSIGN_OR_RETURN(ranges[i].destination, pool.allocate(sizes[i]));
        } return ranges;
      };
      auto read_and_consume = [&](c::StreamFile file, bool original) -> Result<std::array<c::ReadLease, N>> {
        HYLE_ASSIGN_OR_RETURN(auto ranges, make_ranges(original));
        std::array<c::StreamReadGroup, 2> groups{{{10, std::span(ranges).first(2)}, {20, std::span(ranges).subspan(2)}}};
        HYLE_ASSIGN_OR_RETURN(auto ticket, pipeline.read(c::StreamSource::file(std::move(file)), groups));
        std::array<c::ReadLease, N> ready;
        for (unsigned n = 0; n < 2; ++n) {
          HYLE_ASSIGN_OR_RETURN(auto group, ticket.wait_next_ready());
          const auto selected = group.tag() == 10 ? 0u : 1u, first = selected == 0 ? 0u : 2u;
          for (unsigned i = 0; i < group.range_count(); ++i) { HYLE_ASSIGN_OR_RETURN(ready[first + i], group.output(i)); }
          HYLE_RETURN_IF_ERROR(consume_ready(ready, lane, selected)); // no whole-request barrier before early group math
        }
        const auto result = ticket.wait(); HYLE_RETURN_IF_ERROR(result.status);
        if (!result.retirement_proven || result.read_bytes != total || result.copied_bytes != total)
          return std::unexpected(Status::Internal("file stream read progress/retirement mismatch"));
        return ready;
      };
      HYLE_ASSIGN_OR_RETURN(auto ready, read_and_consume(source, true));
      std::array<c::StreamWriteRange, N> writes;
      for (unsigned i = 0; i < N; ++i) writes[i] = c::StreamWriteRange::ready(ready[i], offsets[i], 0, sizes[i]);
      std::array<c::StreamWriteGroup, 2> write_groups{{{30, std::span(writes).first(2)}, {40, std::span(writes).subspan(2)}}};
      HYLE_ASSIGN_OR_RETURN(auto written, pipeline.write(sink, write_groups));
      const auto saved = written.wait(); HYLE_RETURN_IF_ERROR(saved.status);
      if (saved.written_bytes != total) return Status::Internal("file stream write progress mismatch");
      written = {}; HYLE_RETURN_IF_ERROR(pipeline.collect());
      HYLE_ASSIGN_OR_RETURN(auto stored, hyle::platform::open_random_access_file(sink_path.c_str(), hyle::platform::kOpenRead));
      std::vector<std::byte> bytes(total); HYLE_ASSIGN_OR_RETURN(auto actual, stored.read_at(0, bytes));
      if (actual != total || bytes != file_bytes) return Status::Internal("file stream changed original encoded bytes");
      HYLE_RETURN_IF_ERROR(stored.close());
      HYLE_ASSIGN_OR_RETURN(auto reloaded, read_and_consume(sink, false));
      ready = {}; reloaded = {}; HYLE_RETURN_IF_ERROR(pipeline.collect());
    }
  }
  const auto report = pipeline.report();
  if (report.peak_slots > 2 || report.staging_bytes != 128 || report.mapping_copy_bytes)
    return Status::Internal("file staging/slot bound mismatch");
  std::printf("F3 logical/physical native bytes=%llu; stage=%zu; peak slots=%u; native calls=%llu; host waits=%llu\n",
      static_cast<unsigned long long>(report.copied_physical_bytes), report.staging_bytes, report.peak_slots,
      static_cast<unsigned long long>(report.native_calls), static_cast<unsigned long long>(report.host_waits));
  HYLE_RETURN_IF_ERROR(pipeline.drain());
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
  HYLE_ASSIGN_OR_RETURN(auto transfer_queue, context.create_queue());
  return transfer_groups(host, device, kSizes,
      [&](hyle::io::IoContext &io, auto targets, auto options) -> Result<c::FileResourcePipeline> {
#if defined(HYLE_COMPUTE_CUDA)
        if (context.device().backend_id() == c::cuda::kBackendId)
          return c::cuda::create_file_pipeline(io, context, transfer_queue, host, targets, options);
#endif
#if defined(HYLE_COMPUTE_CPU)
        return c::cpu::create_file_pipeline(io, context, transfer_queue, host, targets, options);
#else
        return Status::Unimplemented("no common file route");
#endif
      },
      [](unsigned lane) { return Payload(lane); },
      [&](const auto &ready, unsigned lane, unsigned selected) { return consume(context, queues[lane], device, compile, ready, lane, selected); });
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
  const auto consume_ready = [&](const std::array<c::ReadLease, 4> &ready, unsigned lane, unsigned selected) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto desc, descriptions());
    for (unsigned group = selected; group < selected + 1; ++group) {
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
  return transfer_groups(host.ref(), device.ref(), kSizes,
      [&](hyle::io::IoContext &io, auto targets, auto options) { return v::create_file_pipeline(io, context, host.ref(), targets, options); },
      [](unsigned lane) { return Payload(lane); }, consume_ready);
}
#endif

#if defined(HYLE_COMPUTE_APPLE)
struct ApplePayload final {
  Payload dense;
  std::array<float, 64> activation;
  std::array<std::uint32_t, 16> q4;
  std::array<float, 2> scales{1, 1}, biases{0, -4};
  explicit ApplePayload(unsigned lane) : dense(lane) {
    activation.fill(lane == 0 ? 0.125f : 0.25f);
    std::fill_n(q4.begin(), 8, 0x11111111u);
    std::fill_n(q4.begin() + 8, 8, 0x22222222u);
  }
  std::span<const std::byte> bytes(unsigned id) const {
    switch (id) {
    case 0: case 1: return dense.bytes(id);
    case 2: return std::as_bytes(std::span(activation));
    case 3: return std::as_bytes(std::span(q4));
    case 4: return std::as_bytes(std::span(scales));
    default: return std::as_bytes(std::span(biases));
    }
  }
};

Status apple() {
  namespace a = c::apple;
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(const auto &environment, context.environment());
  std::printf("Apple device: %s; OS: %s\n", environment.device_name.c_str(), environment.os_runtime.c_str());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(context));
  constexpr std::array<std::size_t, 6> sizes{8, 8, 256, 64, 8, 8};
  const auto consume_ready = [&](const std::array<c::ReadLease, 6> &ready, unsigned lane, unsigned selected) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto bf, c::TensorDesc::dense(c::DataType::kBFloat16,
        std::array<std::int64_t, 2>{2, 2}));
    HYLE_ASSIGN_OR_RETURN(auto activation, c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{1, 64}));
    HYLE_ASSIGN_OR_RETURN(auto q4, a::affine_q4_description(2, 64));
    HYLE_ASSIGN_OR_RETURN(auto scale, c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{2, 1}));
    HYLE_ASSIGN_OR_RETURN(auto q_output, c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{1, 2}));
    const std::array desc{bf, bf, activation, q4, scale, scale};
    for (unsigned group = selected; group < selected + 1; ++group) {
      const auto first = group == 0 ? 0u : 2u, count = group == 0 ? 2u : 4u;
      const c::Operation operation = group == 0 ? c::Operation(c::MatMul{})
          : c::Operation(c::QuantizedMatMul{.mode = c::QuantizedWeightMode::kUnsignedAffine4});
      std::vector<a::InputSpec> specs;
      std::vector<a::LeasedInput> inputs;
      std::vector<g::ValueId> ids;
      HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
      for (unsigned index = first; index < first + count; ++index) {
        HYLE_ASSIGN_OR_RETURN(auto spec, a::input_spec(desc[index]));
        specs.push_back(std::move(spec));
        inputs.push_back({desc[index], ready[index]});
        HYLE_ASSIGN_OR_RETURN(auto id, builder.input(desc[index]));
        ids.push_back(id);
      }
      HYLE_ASSIGN_OR_RETURN(auto direct_program, context.prepare(operation, specs));
      HYLE_ASSIGN_OR_RETURN(auto direct, a::LeasedInvocation::create(direct_program, memory));
      HYLE_RETURN_IF_ERROR(direct.bind(inputs));
      HYLE_ASSIGN_OR_RETURN(auto done, direct.submit());
      HYLE_ASSIGN_OR_RETURN(auto result, done.wait());
      HYLE_ASSIGN_OR_RETURN(auto output, memory.import_readonly(result->arrays()[0], group == 0 ? bf : q_output));
      HYLE_RETURN_IF_ERROR(check(output, group == 0 ? c::DataType::kBFloat16 : c::DataType::kFloat32, lane, group));

      HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(operation, ids));
      HYLE_ASSIGN_OR_RETURN(auto y, builder.result(node));
      HYLE_RETURN_IF_ERROR(builder.publish(y));
      HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
      HYLE_ASSIGN_OR_RETURN(auto program, g::apple::compile(context, graph));
      HYLE_ASSIGN_OR_RETURN(auto invocation, a::LeasedInvocation::create(program, memory));
      HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
      HYLE_ASSIGN_OR_RETURN(auto graph_done, invocation.submit());
      HYLE_ASSIGN_OR_RETURN(auto graph_result, graph_done.wait());
      HYLE_ASSIGN_OR_RETURN(auto graph_output, memory.import_readonly(graph_result->arrays()[0], group == 0 ? bf : q_output));
      HYLE_RETURN_IF_ERROR(check(graph_output, group == 0 ? c::DataType::kBFloat16 : c::DataType::kFloat32, lane, group));
    }
    return Status::Ok();
  };
  return transfer_groups(memory.ref(), memory.ref(), sizes,
      [&](hyle::io::IoContext &io, auto targets, auto options) { return a::create_file_pipeline(io, context, memory.ref(), targets, options); },
      [](unsigned lane) { return ApplePayload(lane); }, consume_ready);
}
#endif

Status run() {
#if defined(HYLE_COMPUTE_CPU)
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_RETURN_IF_ERROR(common(cpu, m::system_resource(), m::system_resource(),
      [&](const g::Graph &graph) { return g::compile(graph, cpu); }));
  std::puts("CPU: file read/write/reload groups, BF16+Q8 Direct/Graph and three replays passed");
#endif
#if defined(HYLE_COMPUTE_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto cuda, g::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto host, c::cuda::MemoryResource::create(cuda, c::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto device, c::cuda::MemoryResource::create(cuda, c::cuda::MemoryKind::kDevice));
  HYLE_RETURN_IF_ERROR(common(cuda, host.ref(), device.ref(),
      [&](const g::Graph &graph) { return g::cuda::compile(graph, cuda, device.ref()); }));
  std::puts("CUDA: file read/write/reload, BF16+Q8 Direct/Graph and three replays passed");
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  HYLE_RETURN_IF_ERROR(vulkan());
  std::puts("Vulkan: file read/write/reload, BF16+Q8 Direct/Graph and three replays passed");
#endif
#if defined(HYLE_COMPUTE_APPLE)
  HYLE_RETURN_IF_ERROR(apple());
  std::puts("Apple: file/shared-resource GPU blits, BF16+original affine Q4 Direct/Graph and three replays passed");
#endif
#if !defined(HYLE_COMPUTE_CPU) && !defined(HYLE_COMPUTE_CUDA) && !defined(HYLE_COMPUTE_VULKAN) && !defined(HYLE_COMPUTE_APPLE)
  return Status::Unimplemented("file streaming math sample requires a connected route");
#endif
  return Status::Ok();
}
} // namespace

int main(int argc, char **argv) {
  gguf_input::Fixture input;
  if (argc > 2) { std::fprintf(stderr, "usage: hyle_file_streaming_sample [mature-tool-checked gguf-ranges.tsv]\n"); return 2; }
  if (argc == 2) {
    auto parsed = gguf_input::read(argv[1]);
    if (!parsed) { std::fprintf(stderr, "invalid checked GGUF fixture\n"); return 2; }
    input = std::move(*parsed); gguf_fixture = &input;
  }
  const auto status = run();
  if (status.ok()) return 0;
  const auto message = status.message();
  std::fprintf(stderr, "file streaming: %.*s\n", static_cast<int>(message.size()), message.data());
  return 1;
}
