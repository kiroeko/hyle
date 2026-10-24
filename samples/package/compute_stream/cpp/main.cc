#include <hyle/compute/file_stream.h>
#include <hyle/platform/fs.h>
#include <hyle/sched/join.h>

#if defined(HYLE_COMPUTE_CPU)
#include <hyle/compute/cpu_file_stream.h>
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include <hyle/compute/cuda_file_stream.h>
#endif
#if defined(HYLE_COMPUTE_APPLE)
#include <hyle/compute/apple_file_stream.h>
#include <hyle/compute/apple_memory.h>
#endif
#if defined(HYLE_COMPUTE_VULKAN)
#include <hyle/compute/vulkan_file_stream.h>
#include <hyle/compute/vulkan_memory.h>
#endif

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string>
#include <utility>

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace p = hyle::platform;
namespace io = hyle::io;
namespace s = hyle::sched;
using hyle::Status;
template <class T> using Result = hyle::Result<T>;

#if defined(HYLE_COMPUTE_CPU) || defined(HYLE_COMPUTE_CUDA) || \
    defined(HYLE_COMPUTE_APPLE) || defined(HYLE_COMPUTE_VULKAN)
struct TemporaryFiles final {
  std::string source, sink;
  explicit TemporaryFiles(const char *route) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto prefix = std::string("hyle_compute_stream_") + route + "_" + std::to_string(stamp);
    source = prefix + "_input.tmp";
    sink = prefix + "_output.tmp";
  }
  ~TemporaryFiles() {
    p::remove_file(source.c_str()).IgnoreError();
    p::remove_file(sink.c_str()).IgnoreError();
  }
};

struct Runtime final {
  s::Scheduler scheduler{2, m::system_allocator(), 256};
  io::IoContext io{scheduler, {.max_blocking_threads = 2}};
  // IO borrows its Scheduler, which must shut down before IO is destroyed.
  ~Runtime() { scheduler.shutdown(); }
};

struct Drain final {
  c::FileResourcePipeline &pipeline;
  ~Drain() { pipeline.drain().IgnoreError(); pipeline = {}; }
};

s::Task<Result<io::File>> open_file(m::AllocatorRef, io::IoContext *runtime,
                                  std::string path, bool readonly) {
  co_return co_await io::open_file(*runtime, 0, path,
      {.access = readonly ? io::FileAccess::kRead : io::FileAccess::kReadWrite});
}

Result<c::StreamFile> stream_file(Runtime &runtime, const std::string &path,
                                std::size_t capacity, bool readonly) {
  const auto allocator = m::system_allocator();
  HYLE_ASSIGN_OR_RETURN(auto opened, s::spawn_task_on(runtime.scheduler, 0, allocator,
      open_file(allocator, &runtime.io, path, readonly)).join());
  HYLE_ASSIGN_OR_RETURN(auto file, std::move(opened));
  return c::StreamFile::adopt(std::move(file), capacity);
}

Status create_file(const std::string &path, std::span<const std::byte> bytes) {
  HYLE_ASSIGN_OR_RETURN(auto file, p::open_file(path.c_str(),
      p::kOpenRead | p::kOpenWrite | p::kOpenCreate | p::kOpenTruncate));
  HYLE_RETURN_IF_ERROR(file.write_all(bytes));
  return file.close();
}

template <class Factory>
Status roundtrip(const char *route, m::MemoryResourceRef target, Factory factory) {
  std::array<std::byte, 512> input{};
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<std::byte>((i * 17 + 31) % 251);
  TemporaryFiles paths(route);
  HYLE_RETURN_IF_ERROR(create_file(paths.source, input));
  HYLE_RETURN_IF_ERROR(create_file(paths.sink, {}));
  Runtime runtime;
  HYLE_RETURN_IF_ERROR(runtime.io.status());
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(target, input.size()));
  const std::array targets{c::CopyResourceSpec{target, input.size()}};
  const c::FileStreamOptions options{
      .max_requests = 2, .max_groups_per_request = 1, .max_ranges_per_group = 1,
      .in_flight_slots = 2, .staging_bytes_per_slot = 64,
      .staging_alignment = 64, .max_active_backing_bytes = 4096};
  HYLE_ASSIGN_OR_RETURN(auto pipeline, factory(runtime.io, targets, options));
  Drain drain{pipeline};
  HYLE_ASSIGN_OR_RETURN(auto sink, stream_file(runtime, paths.sink, input.size(), false));
  for (const bool mapped : std::array{false, true}) {
    const char *stage = "source open";
    const auto attempt = [&]() -> Status {
      c::StreamSource source;
      if (mapped) {
        HYLE_ASSIGN_OR_RETURN(auto mapping, m::FileMapping::open(paths.source.c_str()));
        HYLE_ASSIGN_OR_RETURN(source, c::StreamSource::adopt(std::move(mapping)));
      } else {
        HYLE_ASSIGN_OR_RETURN(auto file, stream_file(runtime, paths.source, input.size(), true));
        source = c::StreamSource::file(std::move(file));
      }
      std::array<c::StreamReadRange, 1> reading{};
      reading[0].bytes = input.size();
      stage = "target lease allocation";
      HYLE_ASSIGN_OR_RETURN(reading[0].destination, pool.allocate(input.size()));
      std::array<c::StreamReadGroup, 1> groups{{{1, reading}}};
      stage = "read admission";
      HYLE_ASSIGN_OR_RETURN(auto read, pipeline.read(std::move(source), groups));
      stage = "read ready";
      HYLE_ASSIGN_OR_RETURN(auto ready, read.wait_next_ready());
      stage = "read completion";
      const auto loaded = read.wait();
      HYLE_RETURN_IF_ERROR(loaded.status);
      if (!loaded.retirement_proven || loaded.copied_bytes != input.size())
        return Status::Internal("file resource read did not retire every byte");
      stage = "ready output lease";
      HYLE_ASSIGN_OR_RETURN(auto lease, ready.output(0));
      // Read the real target through its copy route, including device-only memory.
      std::array writing{c::StreamWriteRange::ready(lease, 0, 0, input.size())};
      std::array<c::StreamWriteGroup, 1> writes{{{2, writing}}};
      stage = "write admission";
      HYLE_ASSIGN_OR_RETURN(auto write, pipeline.write(sink, writes));
      stage = "write completion";
      const auto saved = write.wait();
      HYLE_RETURN_IF_ERROR(saved.status);
      if (!saved.retirement_proven || saved.written_bytes != input.size())
        return Status::Internal("file resource write did not retire every byte");
      stage = "file byte verification";
      HYLE_ASSIGN_OR_RETURN(auto stored, p::open_random_access_file(paths.sink.c_str(), p::kOpenRead));
      std::array<std::byte, 512> output{};
      HYLE_ASSIGN_OR_RETURN(auto bytes, stored.read_at(0, output));
      HYLE_RETURN_IF_ERROR(stored.close());
      if (bytes != input.size() || output != input)
        return Status::DataLoss("file resource roundtrip changed the input bytes");
      return Status::Ok();
    }(); // Tickets, mappings and exported leases are dropped before collect.
    if (!attempt.ok()) {
      const auto leases = pool.stats();
      const auto current = pipeline.report();
      std::fprintf(stderr, "%s %s %s: lease bytes=%zu/%zu blocks=%u uses=%llu; "
          "pipeline requests=%u slots=%u backing=%zu\n", route, mapped ? "mapping" : "file", stage,
          leases.allocated_bytes, leases.capacity_bytes, leases.live_blocks,
          static_cast<unsigned long long>(leases.active_uses), current.live_requests,
          current.active_slots, current.active_backing_bytes);
      return attempt.WithContext(stage).WithContext(mapped ? "mapping source" : "file source");
    }
    HYLE_RETURN_IF_ERROR(pipeline.collect());
  }
  HYLE_RETURN_IF_ERROR(pipeline.drain());
  const auto report = pipeline.report();
  if (report.staging_bytes != 128 || report.peak_slots > 2 || report.live_requests != 0)
    return Status::Internal("file resource pipeline exceeded its configured bounds");
  std::printf("%s: file + mapping roundtrips verified, 512 bytes each, %zu staging bytes\n",
      route, report.staging_bytes);
  return Status::Ok();
}
#endif

Status run() {
#if defined(HYLE_COMPUTE_CPU)
  {
    // No math Provider roster: the backend alone supplies the copy Queue.
    HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(c::cpu::device()));
    HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
    const auto result = roundtrip("CPU", m::system_resource(),
        [&](io::IoContext &runtime, auto targets, auto options) {
          return c::cpu::create_file_pipeline(runtime, context, queue, m::system_resource(), targets, options);
        });
    const auto retired = context.drain();
    HYLE_RETURN_IF_ERROR(result);
    HYLE_RETURN_IF_ERROR(retired);
  }
#endif
#if defined(HYLE_COMPUTE_CUDA)
  {
    HYLE_ASSIGN_OR_RETURN(auto device, c::cuda::device(0));
    HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(device));
    HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
    HYLE_ASSIGN_OR_RETURN(auto host, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
    HYLE_ASSIGN_OR_RETURN(auto memory, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
    const auto result = roundtrip("CUDA", memory.ref(),
        [&](io::IoContext &runtime, auto targets, auto options) {
          return c::cuda::create_file_pipeline(runtime, context, queue, host.ref(), targets, options);
        });
    const auto retired = context.drain();
    HYLE_RETURN_IF_ERROR(result);
    HYLE_RETURN_IF_ERROR(retired);
  }
#endif
#if defined(HYLE_COMPUTE_APPLE)
  {
    HYLE_ASSIGN_OR_RETURN(auto context, c::apple::Context::create());
    HYLE_ASSIGN_OR_RETURN(auto memory, c::apple::MemoryResource::create(context));
    const auto result = roundtrip("Apple", memory.ref(),
        [&](io::IoContext &runtime, auto targets, auto options) {
          return c::apple::create_file_pipeline(runtime, context, memory.ref(), targets, options);
        });
    const auto retired = context.close();
    HYLE_RETURN_IF_ERROR(result);
    HYLE_RETURN_IF_ERROR(retired);
  }
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  {
    HYLE_ASSIGN_OR_RETURN(auto context, c::vulkan::Context::create());
    HYLE_ASSIGN_OR_RETURN(auto host, c::vulkan::MemoryResource::create(context, c::vulkan::MemoryKind::kStaging));
    HYLE_ASSIGN_OR_RETURN(auto memory, c::vulkan::MemoryResource::create(context, c::vulkan::MemoryKind::kDevice));
    const auto result = roundtrip("Vulkan", memory.ref(),
        [&](io::IoContext &runtime, auto targets, auto options) {
          return c::vulkan::create_file_pipeline(runtime, context, host.ref(), targets, options);
        });
    const auto retired = context.close();
    HYLE_RETURN_IF_ERROR(result);
    HYLE_RETURN_IF_ERROR(retired);
  }
#endif
#if !defined(HYLE_COMPUTE_CPU) && !defined(HYLE_COMPUTE_CUDA) && \
    !defined(HYLE_COMPUTE_APPLE) && !defined(HYLE_COMPUTE_VULKAN)
  return Status::Unimplemented("compute_stream sample requires a selected copy route");
#endif
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = run();
  if (status.ok()) return 0;
  char message[512]{};
  status.ToString(message, sizeof(message));
  std::fprintf(stderr, "compute_stream: %s\n", message);
  return 1;
}
