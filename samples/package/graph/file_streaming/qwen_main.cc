#include "qwen_input.h"

#include <hyle/compute/file_stream.h>
#include <hyle/compute/leased_operation.h>
#include <hyle/graph/leased_invocation.h>
#include <hyle/mem/stats.h>
#include <hyle/platform/fs.h>
#include <hyle/sched/join.h>
#if defined(HYLE_COMPUTE_CPU)
#include <hyle/graph/cpu.h>
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include <hyle/compute/backends/cuda.h>
#include <hyle/graph/cuda.h>
#endif
#if defined(HYLE_COMPUTE_APPLE)
#include <hyle/compute/apple_memory.h>
#include <hyle/graph/apple.h>
#endif
#if defined(HYLE_COMPUTE_VULKAN)
#include <hyle/compute/vulkan_memory.h>
#include <hyle/graph/vulkan.h>
#endif

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using hyle::Status;
template<class T> using Result = hyle::Result<T>;
struct Work final {
  c::Operation operation;
  c::TensorDesc activation, weight, output;
  std::vector<std::byte> input;
};
Result<Work> describe(const qwen_input::Range &range) {
  Work work;
  if (range.description.data_type() == c::DataType::kBFloat16) {
    const auto dims = range.description.dimensions();
    HYLE_ASSIGN_OR_RETURN(work.activation, c::TensorDesc::dense(c::DataType::kBFloat16, std::array<std::int64_t, 2>{1, dims[1]}));
    HYLE_ASSIGN_OR_RETURN(work.weight, range.description.transposed());
    HYLE_ASSIGN_OR_RETURN(work.output, c::TensorDesc::dense(c::DataType::kBFloat16, std::array<std::int64_t, 2>{1, dims[0]}));
    work.operation = c::Operation(c::MatMul{});
    std::vector<std::uint16_t> input(static_cast<std::size_t>(dims[1]));
    input.front() = 0x3e00; input.back() = 0xbd80; // 0.125 and -0.0625, exact BF16
    const auto bytes = std::as_bytes(std::span(input)); work.input.assign(bytes.begin(), bytes.end());
  } else {
    work.activation = work.weight = work.output = range.description;
    work.operation = c::Operation(c::Binary{.kind = c::BinaryKind::kMultiply});
    std::array<float, 128> input;
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = 1.0f + (i % 7) * 0.125f;
    const auto bytes = std::as_bytes(std::span(input)); work.input.assign(bytes.begin(), bytes.end());
  }
  return work;
}
Status check(const c::ReadLease &result, const Work &work, const qwen_input::Range &range) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, work.output.required_bytes());
  std::vector<std::byte> data(bytes); HYLE_RETURN_IF_ERROR(result.view()->read(data));
  const auto bf16 = work.output.data_type() == c::DataType::kBFloat16;
  for (std::size_t index = 0; index < range.reference.size(); ++index) {
    float actual;
    if (bf16) {
      std::uint16_t bits; std::memcpy(&bits, data.data() + index * 2, 2);
      actual = std::bit_cast<float>(std::uint32_t(bits) << 16);
    } else { std::memcpy(&actual, data.data() + index * 4, 4); }
    const auto expected = range.reference[index];
    const auto tolerance = bf16 ? 0.0001f + 0.015f * std::abs(expected) : 0.00001f + 0.00001f * std::abs(expected);
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
      return Status::Internal("original Qwen range mathematical reference differs");
  }
  return Status::Ok();
}
#if defined(HYLE_COMPUTE_CPU) || defined(HYLE_COMPUTE_CUDA)
template<class Compile>
Status consume(c::Context &context, c::Queue &queue, m::MemoryResourceRef resource,
    Compile compile, const c::ReadLease &weight, const qwen_input::Range &range) {
  HYLE_ASSIGN_OR_RETURN(auto work, describe(range));
  HYLE_ASSIGN_OR_RETURN(auto scratch, c::LeasePool::create(resource, 1024u * 1024u));
  HYLE_ASSIGN_OR_RETURN(auto input_writer, scratch.allocate(work.input.size()));
  HYLE_RETURN_IF_ERROR(input_writer.view()->write(work.input));
  HYLE_ASSIGN_OR_RETURN(auto input, input_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto input_view, input.tensor(work.activation));
  HYLE_ASSIGN_OR_RETURN(auto weight_view, weight.tensor(work.weight));
  HYLE_ASSIGN_OR_RETURN(auto bytes, work.output.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto output, scratch.allocate(bytes));
  HYLE_ASSIGN_OR_RETURN(auto output_view, output.tensor(work.output));
  HYLE_ASSIGN_OR_RETURN(auto prepared, c::prepare(context, work.operation, std::array{input_view, weight_view}, std::array{output_view}));
  c::LeasePool workspace_pool; c::WriteLease workspace;
  const auto required = prepared.workspace_requirement();
  if (required.size) {
    const auto alignment = (std::max)(required.alignment, std::size_t{256});
    HYLE_ASSIGN_OR_RETURN(workspace_pool, c::LeasePool::create(resource, required.size, {.max_blocks = 1, .backing_alignment = alignment}));
    HYLE_ASSIGN_OR_RETURN(workspace, workspace_pool.allocate(required.size, alignment));
  }
  HYLE_ASSIGN_OR_RETURN(auto direct, c::LeasedOperationInvocation::create(prepared, std::move(workspace)));
  const std::array inputs{c::LeasedInput(work.activation, input), c::LeasedInput(work.weight, weight)};
  const std::array outputs{c::LeasedOutput(work.output, output)};
  HYLE_ASSIGN_OR_RETURN(auto done, direct.submit(queue, inputs, outputs)); HYLE_RETURN_IF_ERROR(done.wait());
  HYLE_ASSIGN_OR_RETURN(auto direct_output, output.publish()); HYLE_RETURN_IF_ERROR(check(direct_output, work, range));
  done = {}; HYLE_RETURN_IF_ERROR(direct.close());
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto a, builder.input(work.activation)); HYLE_ASSIGN_OR_RETURN(auto b, builder.input(work.weight));
  HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(work.operation, std::array{a, b}));
  HYLE_ASSIGN_OR_RETURN(auto y, builder.result(node)); HYLE_RETURN_IF_ERROR(builder.publish(y));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish()); HYLE_ASSIGN_OR_RETURN(auto compiled, compile(graph));
  HYLE_ASSIGN_OR_RETURN(auto invocation, g::LeasedInvocation::create(compiled.executable));
  HYLE_RETURN_IF_ERROR(invocation.bind(a, work.activation, input)); HYLE_RETURN_IF_ERROR(invocation.bind(b, work.weight, weight));
  HYLE_ASSIGN_OR_RETURN(auto graph_writer, scratch.allocate(bytes)); HYLE_RETURN_IF_ERROR(invocation.bind(y, work.output, std::move(graph_writer)));
  HYLE_ASSIGN_OR_RETURN(auto graph_done, invocation.submit(queue)); HYLE_RETURN_IF_ERROR(graph_done.wait());
  HYLE_ASSIGN_OR_RETURN(auto result, invocation.output(y)); HYLE_ASSIGN_OR_RETURN(auto lease, result.storage_lease());
  return check(lease, work, range);
}
#endif
hyle::sched::Task<Result<hyle::io::File>> open_model(m::AllocatorRef, hyle::io::IoContext *io, std::string path, bool read_only) {
  co_return co_await hyle::io::open_file(*io, 0, path,
      {.access = read_only ? hyle::io::FileAccess::kRead : hyle::io::FileAccess::kReadWrite});
}
template<class Create, class Consume>
Status stream(const qwen_input::Manifest &manifest, m::MemoryResourceRef resource, Create create, Consume consume_ready) {
  hyle::sched::Scheduler scheduler(2, m::system_allocator(), 256);
  hyle::io::IoContext io(scheduler, {.max_blocking_threads = 2}); if (!io.valid()) return io.status();
  struct Shutdown { hyle::sched::Scheduler &scheduler; ~Shutdown() { scheduler.shutdown(); } } shutdown{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto opened, hyle::sched::spawn_task_on(scheduler, 0, m::system_allocator(), open_model(m::system_allocator(), &io, manifest.file, true)).join());
  HYLE_ASSIGN_OR_RETURN(auto model, std::move(opened)); HYLE_ASSIGN_OR_RETURN(auto source, c::StreamFile::adopt(std::move(model), manifest.capacity));
  std::size_t capacity = 0, largest = 0;
  for (const auto &range : manifest.ranges) { capacity += (range.bytes + 255) & ~std::size_t{255}; largest = (std::max)(largest, range.bytes); }
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(resource, capacity));
  const std::array specs{c::CopyResourceSpec{resource, largest}};
  HYLE_ASSIGN_OR_RETURN(auto pipeline, create(io, specs, c::FileStreamOptions{.max_groups_per_request = 4, .max_ranges_per_group = 1}));
  struct Drain { c::FileResourcePipeline &pipeline; ~Drain() { pipeline.drain().IgnoreError(); } } drain{pipeline};
  std::array<std::uint64_t, 4> saved_offsets{};
  std::size_t total = 0;
  for (std::size_t i = 0; i < saved_offsets.size(); ++i) { saved_offsets[i] = total; total += manifest.ranges[i].bytes; }
  const auto read_math = [&](c::StreamFile file, bool original) -> Result<std::array<c::ReadLease, 4>> {
    std::array<c::StreamReadRange, 4> ranges; std::array<c::StreamReadGroup, 4> groups;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      const auto &input = manifest.ranges[i]; ranges[i].file_offset = original ? input.offset : saved_offsets[i];
      ranges[i].bytes = input.bytes; ranges[i].description = input.description;
      HYLE_ASSIGN_OR_RETURN(ranges[i].destination, pool.allocate(input.bytes)); groups[i] = {i, std::span(ranges).subspan(i, 1)};
    }
    HYLE_ASSIGN_OR_RETURN(auto ticket, pipeline.read(c::StreamSource::file(std::move(file)), groups));
    std::array<c::ReadLease, 4> ready;
    for (std::size_t count = 0; count < ranges.size(); ++count) {
      HYLE_ASSIGN_OR_RETURN(auto group, ticket.wait_next_ready()); HYLE_ASSIGN_OR_RETURN(ready[group.tag()], group.output(0));
      HYLE_RETURN_IF_ERROR(consume_ready(ready[group.tag()], manifest.ranges[group.tag()]));
      std::printf("Qwen %s %s: Direct/Graph reference passed\n", original ? "original" : "reloaded", manifest.ranges[group.tag()].name.c_str());
    }
    const auto result = ticket.wait(); HYLE_RETURN_IF_ERROR(result.status);
    if (!result.retirement_proven || result.read_bytes != total || result.copied_bytes != total)
      return std::unexpected(Status::Internal("Qwen range progress/retirement"));
    return ready;
  };
  HYLE_ASSIGN_OR_RETURN(auto ready, read_math(source, true));
  static std::atomic<unsigned> sequence{0};
  const auto sink_path = "hyle_f3_qwen_" + std::to_string(sequence.fetch_add(1)) + ".tmp";
  struct Remove { const std::string &path; ~Remove() { hyle::platform::remove_file(path.c_str()).IgnoreError(); } } remove{sink_path};
  HYLE_ASSIGN_OR_RETURN(auto empty, hyle::platform::open_file(sink_path.c_str(),
      hyle::platform::kOpenRead | hyle::platform::kOpenWrite | hyle::platform::kOpenCreate | hyle::platform::kOpenTruncate));
  HYLE_RETURN_IF_ERROR(empty.close());
  HYLE_ASSIGN_OR_RETURN(auto opened_sink, hyle::sched::spawn_task_on(scheduler, 0, m::system_allocator(), open_model(m::system_allocator(), &io, sink_path, false)).join());
  HYLE_ASSIGN_OR_RETURN(auto sink_file, std::move(opened_sink));
  HYLE_ASSIGN_OR_RETURN(auto sink, c::StreamFile::adopt(std::move(sink_file), total));
  std::array<c::StreamWriteRange, 4> writes; std::array<c::StreamWriteGroup, 4> groups;
  for (std::size_t i = 0; i < writes.size(); ++i) {
    writes[i] = c::StreamWriteRange::ready(ready[i], saved_offsets[i], 0, manifest.ranges[i].bytes);
    groups[i] = {i, std::span(writes).subspan(i, 1)};
  }
  HYLE_ASSIGN_OR_RETURN(auto written, pipeline.write(sink, groups));
  const auto result = written.wait(); HYLE_RETURN_IF_ERROR(result.status);
  if (!result.retirement_proven || result.written_bytes != total) return Status::Internal("Qwen write progress/retirement");
  written = {}; writes = {}; ready = {}; HYLE_RETURN_IF_ERROR(pipeline.collect());
  HYLE_ASSIGN_OR_RETURN(auto original_file, hyle::platform::open_random_access_file(manifest.file.c_str(), hyle::platform::kOpenRead));
  HYLE_ASSIGN_OR_RETURN(auto saved_file, hyle::platform::open_random_access_file(sink_path.c_str(), hyle::platform::kOpenRead));
  std::array<std::byte, 64 * 1024> original_bytes{}, saved_bytes{};
  for (std::size_t i = 0; i < saved_offsets.size(); ++i) {
    for (std::size_t offset = 0; offset < manifest.ranges[i].bytes;) {
      const auto chunk = (std::min)(original_bytes.size(), manifest.ranges[i].bytes - offset);
      auto a = std::span(original_bytes).first(chunk), b = std::span(saved_bytes).first(chunk);
      HYLE_ASSIGN_OR_RETURN(auto a_bytes, original_file.read_at(manifest.ranges[i].offset + offset, a));
      HYLE_ASSIGN_OR_RETURN(auto b_bytes, saved_file.read_at(saved_offsets[i] + offset, b));
      if (a_bytes != chunk || b_bytes != chunk || !std::equal(a.begin(), a.end(), b.begin()))
        return Status::Internal("Qwen file roundtrip changed original encoded bytes");
      offset += chunk;
    }
  }
  HYLE_RETURN_IF_ERROR(original_file.close()); HYLE_RETURN_IF_ERROR(saved_file.close());
  HYLE_ASSIGN_OR_RETURN(auto reloaded, read_math(sink, false));
  reloaded = {}; HYLE_RETURN_IF_ERROR(pipeline.collect());
  std::printf("Qwen roundtrip logical bytes=%zu; staging=%zu; peak slots=%u; native bytes=%llu\n",
      total, pipeline.report().staging_bytes, pipeline.report().peak_slots,
      static_cast<unsigned long long>(pipeline.report().copied_physical_bytes));
  return pipeline.drain();
}

#if defined(HYLE_COMPUTE_APPLE)
Status apple(const qwen_input::Manifest &manifest) {
  namespace a = c::apple;
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(context));
  return stream(manifest, memory.ref(),
      [&](auto &io, auto specs, auto options) { return a::create_file_pipeline(io, context, memory.ref(), specs, options); },
      [&](const c::ReadLease &weight, const qwen_input::Range &range) -> Status {
        HYLE_ASSIGN_OR_RETURN(auto work, describe(range));
        HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(memory.ref(), work.input.size()));
        HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(work.input.size()));
        HYLE_RETURN_IF_ERROR(writer.view()->write(work.input));
        HYLE_ASSIGN_OR_RETURN(auto input, writer.publish());
        HYLE_ASSIGN_OR_RETURN(auto input_spec, a::input_spec(work.activation));
        HYLE_ASSIGN_OR_RETURN(auto weight_spec, a::input_spec(work.weight));
        const std::array specs{input_spec, weight_spec};
        const std::array inputs{a::LeasedInput{work.activation, input}, a::LeasedInput{work.weight, weight}};
        HYLE_ASSIGN_OR_RETURN(auto program, context.prepare(work.operation, specs));
        HYLE_ASSIGN_OR_RETURN(auto direct, a::LeasedInvocation::create(program, memory));
        HYLE_RETURN_IF_ERROR(direct.bind(inputs));
        HYLE_ASSIGN_OR_RETURN(auto done, direct.submit()); HYLE_ASSIGN_OR_RETURN(auto result, done.wait());
        HYLE_ASSIGN_OR_RETURN(auto output, memory.import_readonly(result->arrays()[0], work.output));
        HYLE_RETURN_IF_ERROR(check(output, work, range));
        HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
        HYLE_ASSIGN_OR_RETURN(auto x, builder.input(work.activation)); HYLE_ASSIGN_OR_RETURN(auto w, builder.input(work.weight));
        HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(work.operation, std::array{x, w}));
        HYLE_ASSIGN_OR_RETURN(auto y, builder.result(node)); HYLE_RETURN_IF_ERROR(builder.publish(y));
        HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
        HYLE_ASSIGN_OR_RETURN(auto compiled, g::apple::compile(context, graph));
        HYLE_ASSIGN_OR_RETURN(auto invocation, a::LeasedInvocation::create(compiled, memory));
        HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
        HYLE_ASSIGN_OR_RETURN(auto graph_done, invocation.submit()); HYLE_ASSIGN_OR_RETURN(auto graph_result, graph_done.wait());
        HYLE_ASSIGN_OR_RETURN(auto graph_output, memory.import_readonly(graph_result->arrays()[0], work.output));
        return check(graph_output, work, range);
      });
}
#endif

#if defined(HYLE_COMPUTE_VULKAN)
Status vulkan(const qwen_input::Manifest &manifest) {
  namespace v = c::vulkan;
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  const auto &environment = context.environment();
  std::printf("Qwen Vulkan device: %s; driver: %s %s; software: %s\n",
      environment.device_name.c_str(), environment.driver_name.c_str(),
      environment.driver_info.c_str(), environment.software_device ? "yes" : "no");
  HYLE_ASSIGN_OR_RETURN(auto host, v::MemoryResource::create(context, v::MemoryKind::kStaging));
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  return stream(manifest, device.ref(),
      [&](auto &io, auto specs, auto options) { return v::create_file_pipeline(io, context, host.ref(), specs, options); },
      [&](const c::ReadLease &weight, const qwen_input::Range &range) -> Status {
        HYLE_ASSIGN_OR_RETURN(auto work, describe(range));
        HYLE_ASSIGN_OR_RETURN(auto activation, context.upload(work.activation, work.input));
        HYLE_ASSIGN_OR_RETURN(auto input, device.import_readonly(activation));
        const std::array inputs{v::LeasedInput{work.activation, input}, v::LeasedInput{work.weight, weight}};
        const std::array arguments{v::Argument(work.activation), v::Argument(work.weight)};
        HYLE_ASSIGN_OR_RETURN(auto program, v::prepare(context, work.operation, arguments));
        const auto checked_output = [&](const v::Tensor &tensor) -> Status {
          HYLE_ASSIGN_OR_RETURN(auto bytes, work.output.required_bytes());
          HYLE_ASSIGN_OR_RETURN(auto buffer, m::Buffer::create(m::system_resource(), bytes, 64));
          HYLE_ASSIGN_OR_RETURN(auto mapping, buffer.view().map_host(m::HostAccess::kWrite));
          HYLE_ASSIGN_OR_RETURN(auto destination, mapping.writable_bytes());
          HYLE_RETURN_IF_ERROR(context.download(tensor, destination)); HYLE_RETURN_IF_ERROR(mapping.close());
          HYLE_ASSIGN_OR_RETURN(auto output, c::ReadLease::adopt(std::move(buffer)));
          return check(output, work, range);
        };
        HYLE_ASSIGN_OR_RETURN(auto direct, v::LeasedInvocation::create(program, device));
        HYLE_RETURN_IF_ERROR(direct.bind(inputs));
        HYLE_ASSIGN_OR_RETURN(auto done, direct.submit()); HYLE_ASSIGN_OR_RETURN(auto result, done.wait());
        HYLE_RETURN_IF_ERROR(checked_output(result->tensors()[0]));
        HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
        HYLE_ASSIGN_OR_RETURN(auto x, builder.input(work.activation)); HYLE_ASSIGN_OR_RETURN(auto w, builder.input(work.weight));
        HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(work.operation, std::array{x, w}));
        HYLE_ASSIGN_OR_RETURN(auto y, builder.result(node)); HYLE_RETURN_IF_ERROR(builder.publish(y));
        HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
        HYLE_ASSIGN_OR_RETURN(auto compiled, g::vulkan::compile(context, graph));
        HYLE_ASSIGN_OR_RETURN(auto invocation, v::LeasedInvocation::create(compiled, device));
        HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
        HYLE_ASSIGN_OR_RETURN(auto graph_done, invocation.submit()); HYLE_ASSIGN_OR_RETURN(auto graph_result, graph_done.wait());
        return checked_output(graph_result->tensors()[0]);
      });
}
#endif
Status run(const char *path) {
  HYLE_ASSIGN_OR_RETURN(auto input, qwen_input::read(path));
#if !defined(HYLE_COMPUTE_CPU) && !defined(HYLE_COMPUTE_CUDA) && !defined(HYLE_COMPUTE_APPLE) && !defined(HYLE_COMPUTE_VULKAN)
  return Status::Unimplemented("Qwen file math consumer requires a connected route");
#endif
#if defined(HYLE_COMPUTE_CPU)
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context()); HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto transfer, cpu.create_queue());
  HYLE_RETURN_IF_ERROR(stream(input, m::system_resource(),
      [&](auto &io, auto specs, auto options) { return c::cpu::create_file_pipeline(io, cpu, transfer, m::system_resource(), specs, options); },
      [&](const auto &weight, const auto &range) { return consume(cpu, queue, m::system_resource(), [&](const g::Graph &graph) { return g::compile(graph, cpu); }, weight, range); }));
#endif
#if defined(HYLE_COMPUTE_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto cuda, g::cuda::create_context()); HYLE_ASSIGN_OR_RETURN(auto queue_cuda, cuda.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto transfer_cuda, cuda.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto host, c::cuda::MemoryResource::create(cuda, c::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto device, c::cuda::MemoryResource::create(cuda, c::cuda::MemoryKind::kDevice));
  HYLE_RETURN_IF_ERROR(stream(input, device.ref(),
      [&](auto &io, auto specs, auto options) { return c::cuda::create_file_pipeline(io, cuda, transfer_cuda, host.ref(), specs, options); },
      [&](const auto &weight, const auto &range) { return consume(cuda, queue_cuda, device.ref(), [&](const g::Graph &graph) { return g::cuda::compile(graph, cuda, device.ref()); }, weight, range); }));
#endif
#if defined(HYLE_COMPUTE_APPLE)
  HYLE_RETURN_IF_ERROR(apple(input));
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  HYLE_RETURN_IF_ERROR(vulkan(input));
#endif
  return Status::Ok();
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: hyle_file_streaming_qwen <official-tool-checked ranges.tsv>\n"); return 2; }
  const auto status = run(argv[1]); if (status.ok()) return 0;
  char message[512]{}; status.ToString(message, sizeof(message)); std::fprintf(stderr, "%s\n", message); return 1;
}
