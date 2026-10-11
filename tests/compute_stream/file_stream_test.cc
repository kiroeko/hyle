#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <csignal>
#include <sys/resource.h>
#endif

#include "hyle/compute/cpu.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/stats.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/join.h"

namespace {
using namespace hyle;
using namespace hyle::compute;
namespace m = hyle::mem;
namespace p = hyle::platform;

sched::Task<Result<io::File>> open_async(m::AllocatorRef, io::IoContext *io, std::string path, unsigned owner) noexcept {
  co_return co_await io::open_file(*io, owner, path, {.access = io::FileAccess::kReadWrite});
}
struct Runtime final {
  m::StatsAllocator metadata = m::StatsAllocator::create(m::system_allocator()).value();
  Context context = Context::create(cpu::device()).value();
  Queue queue = context.create_queue().value();
  sched::Scheduler scheduler{2, m::system_allocator(), 256};
  io::IoContext io{scheduler, {.max_blocking_threads = 2}};
  FileResourcePipeline pipeline;
  std::vector<std::string> paths;
  explicit Runtime(FileStreamOptions options = {}) {
    options.allocator = metadata.get(); options.max_groups_per_request = 4; options.max_ranges_per_group = 4;
    options.staging_bytes_per_slot = 32; options.staging_alignment = 16;
    options.max_active_backing_bytes = 4096;
    const std::array targets{CopyResourceSpec{m::system_resource(), 2048}};
    pipeline = cpu::create_file_pipeline(io, context, queue, m::system_resource(), targets, options).value();
  }
  ~Runtime() {
    pipeline.drain().IgnoreError(); pipeline = {}; scheduler.shutdown();
    EXPECT_TRUE(metadata.balanced());
    for (const auto &path : paths) p::remove_file(path.c_str()).IgnoreError();
  }
  StreamFile file(std::span<const std::byte> bytes, std::uint64_t declared = UINT64_MAX, unsigned owner = 0) {
    static std::atomic<unsigned> sequence{0};
    auto path = "hyle_f3_cpu_" + std::to_string(sequence.fetch_add(1)) + ".tmp";
    paths.push_back(path);
    auto raw = p::open_file(path.c_str(), p::kOpenRead | p::kOpenWrite | p::kOpenCreate | p::kOpenTruncate).value();
    raw.write_all(bytes).IgnoreError(); raw.close().IgnoreError();
    auto opened = sched::spawn_task_on(scheduler, owner, m::system_allocator(), open_async(m::system_allocator(), &io, path, owner)).join().value().value();
    return StreamFile::adopt(std::move(opened), declared == UINT64_MAX ? bytes.size() : declared).value();
  }
};
std::array<std::byte, 256> bytes() {
  std::array<std::byte, 256> result;
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = std::byte(i % 251);
  return result;
}
std::array<StreamReadRange, 2> read_ranges(LeasePool &pool) {
  std::array<StreamReadRange, 2> ranges;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    ranges[i].file_offset = i * 128; ranges[i].bytes = 128;
    ranges[i].destination = pool.allocate(128, 1).value();
  } return ranges;
}
std::array<StreamReadGroup, 2> read_groups(std::array<StreamReadRange, 2> &ranges) {
  return {{{10, std::span(ranges).first(1)}, {11, std::span(ranges).subspan(1)}}};
}
void expect_bytes(const ReadLease &lease, std::span<const std::byte> expected) {
  std::vector<std::byte> output(expected.size());
  ASSERT_TRUE(lease.view()->read(output).ok()); EXPECT_TRUE(std::equal(output.begin(), output.end(), expected.begin()));
}

TEST(FileStream, RealPositionalReadWriteReloadAcrossTwoSlotsAndMultipleChunks) {
  Runtime runtime;
  const auto input = bytes(); auto source = runtime.file(input, UINT64_MAX, 1); auto sink = runtime.file({}, 256, 0);
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
  auto read = runtime.pipeline.read(StreamSource::file(source), groups).value();
  auto first = read.wait_next_ready().value(); auto second = read.wait_next_ready().value();
  std::array<ReadLease, 2> outputs;
  outputs[first.tag() - 10] = first.output(0).value(); outputs[second.tag() - 10] = second.output(0).value();
  auto loaded = read.wait(); ASSERT_TRUE(loaded.status.ok()); EXPECT_TRUE(loaded.retirement_proven);
  EXPECT_EQ(loaded.ready_groups, 2u); EXPECT_EQ(loaded.read_bytes, 256u); EXPECT_EQ(loaded.copied_bytes, 256u);
  for (std::size_t i = 0; i < 2; ++i) expect_bytes(outputs[i], std::span(input).subspan(i * 128, 128));
  std::array<StreamWriteRange, 2> writes{
      StreamWriteRange::ready(outputs[0], 0, 0, 128), StreamWriteRange::ready(outputs[1], 128, 0, 128)};
  std::array<StreamWriteGroup, 2> writing{{{20, std::span(writes).first(1)}, {21, std::span(writes).subspan(1)}}};
  auto write = runtime.pipeline.write(sink, writing).value();
  auto written = write.wait(); ASSERT_TRUE(written.status.ok()); EXPECT_EQ(written.written_bytes, 256u);
  EXPECT_EQ(written.copied_bytes, 256u); EXPECT_EQ(written.ready_groups, 2u);
  write = {}; ASSERT_TRUE(runtime.pipeline.collect().ok());
  auto reload_ranges = read_ranges(pool); auto reload_groups = read_groups(reload_ranges);
  auto reload = runtime.pipeline.read(StreamSource::file(sink), reload_groups).value();
  ASSERT_TRUE(reload.wait().status.ok());
  for (unsigned i = 0; i < 2; ++i) {
    auto ready = reload.next_ready().value(); expect_bytes(ready.output(0).value(), std::span(input).subspan((ready.tag() - 10) * 128, 128));
  }
  auto report = runtime.pipeline.report(); EXPECT_EQ(report.staging_bytes, 64u);
  EXPECT_LE(report.peak_slots, 2u); EXPECT_GE(report.native_calls, 24u); EXPECT_GE(report.io_calls, 24u);
  EXPECT_EQ(report.active_backing_bytes, 64u + 1024u); EXPECT_FALSE(report.native_metadata_unknown);
  ASSERT_TRUE(runtime.pipeline.drain().ok());
}

TEST(FileStream, RetainedReadyAndExportedLeaseKeepRequestAndBackingCapacity) {
  Runtime runtime({.max_requests = 1}); const auto input = bytes(); auto source = runtime.file(input);
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
  auto ticket = runtime.pipeline.read(StreamSource::file(source), groups).value();
  ASSERT_TRUE(ticket.wait().status.ok());
  auto ready = ticket.next_ready().value(); auto output = ready.output(0).value();
  ticket = {}; ready = {}; ASSERT_TRUE(runtime.pipeline.collect().ok());
  EXPECT_FALSE(runtime.pipeline.read({}, {}).has_value());
  expect_bytes(output, std::span(input).first(128));
  output.reset(); ASSERT_TRUE(runtime.pipeline.collect().ok());
  auto empty = runtime.pipeline.read({}, {}).value(); EXPECT_TRUE(empty.wait().complete);
  EXPECT_EQ(empty.result().total_groups, 0u); EXPECT_EQ(runtime.pipeline.report().active_backing_bytes, 64u);
}

TEST(FileStream, LaterEofPreservesEarlierReadyAndActualPartialProgress) {
  Runtime runtime({.in_flight_slots = 1}); const auto input = bytes(); auto source = runtime.file(std::span(input).first(138), 256);
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
  auto ticket = runtime.pipeline.read(StreamSource::file(source), groups).value();
  auto earlier = ticket.wait_next_ready().value(); EXPECT_EQ(earlier.tag(), 10u);
  auto finished = ticket.wait(); EXPECT_FALSE(finished.status.ok()); EXPECT_TRUE(finished.retirement_proven);
  EXPECT_EQ(finished.ready_groups, 1u); EXPECT_EQ(finished.read_bytes, 138u); EXPECT_EQ(finished.copied_bytes, 128u);
  expect_bytes(earlier.output(0).value(), std::span(input).first(128));
  EXPECT_FALSE(ticket.next_ready().has_value());
}

TEST(FileStream, InvalidBoundsAndOverlappingSinkRejectBeforeConsumingLeases) {
  Runtime runtime; const auto input = bytes(); auto source = runtime.file(input);
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
  ranges[0].file_offset = UINT64_MAX;
  EXPECT_FALSE(runtime.pipeline.read(StreamSource::file(source), groups).has_value());
  EXPECT_TRUE(ranges[0].destination.valid()); EXPECT_TRUE(ranges[1].destination.valid());
  ranges[0].file_offset = 0;
  auto ready_source = ranges[0].destination.publish().value();
  std::array<StreamWriteRange, 2> overlap{StreamWriteRange::ready(ready_source, 0, 0, 32),
                                        StreamWriteRange::ready(ready_source, 16, 0, 32)};
  std::array<StreamWriteGroup, 1> writing{{{1, overlap}}};
  EXPECT_FALSE(runtime.pipeline.write(source, writing).has_value());
  EXPECT_TRUE(overlap[0].source.valid()); EXPECT_TRUE(overlap[1].source.valid());
  EXPECT_EQ(runtime.pipeline.report().native_calls, 0u); EXPECT_EQ(runtime.pipeline.report().io_calls, 0u);
}

TEST(FileStream, ImmutableMappingUsesBoundedStageAndCancellationDoesNotPublish) {
  Runtime runtime; const auto input = bytes(); auto source = runtime.file(input);
  auto mapping = m::FileMapping::open(runtime.paths.back().c_str()).value();
  auto mapped = StreamSource::adopt(std::move(mapping)).value(); EXPECT_FALSE(mapping.valid());
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
  auto ticket = runtime.pipeline.read(mapped, groups).value(); ASSERT_TRUE(ticket.wait().status.ok());
  EXPECT_EQ(runtime.pipeline.report().mapping_copy_bytes, 256u);
  auto cancellation = sched::CancelToken::root(m::system_allocator()); cancellation.cancel();
  auto cancelled_ranges = read_ranges(pool); auto cancelled_groups = read_groups(cancelled_ranges);
  auto cancelled = runtime.pipeline.read(StreamSource::file(source), cancelled_groups, {.deadline = {}, .cancel = &cancellation}).value();
  EXPECT_TRUE(cancelled.wait().status.IsCancelled()); EXPECT_EQ(cancelled.result().ready_groups, 0u);
  EXPECT_EQ(cancelled.result().read_bytes, 0u); EXPECT_FALSE(cancelled.next_ready().has_value());
}

TEST(FileStream, ColdOomAndOwnedFileFailureAreRecoverableWithoutMovingInputs) {
  Runtime runtime; const auto input = bytes(); runtime.file(input);
  auto raw = p::open_file(runtime.paths.back().c_str(), p::kOpenRead).value();
  auto file = io::File::adopt(runtime.io, 0, std::move(raw)).value();
  m::QuotaAllocator no_metadata(m::system_allocator(), 1);
  EXPECT_FALSE(StreamFile::adopt(std::move(file), 256, no_metadata.get()).has_value()); EXPECT_TRUE(file.valid());
  const std::array specs{CopyResourceSpec{m::system_resource(), 256}};
  EXPECT_FALSE(cpu::create_file_pipeline(runtime.io, runtime.context, runtime.queue, m::system_resource(), specs,
      {.allocator = no_metadata.get()}).has_value());
  EXPECT_EQ(no_metadata.used(), 0u);
}
TEST(FileStream, DescriptionOffsetAndBackingQuotaRejectBeforeMovingOrDoingIo) {
  Runtime runtime; const auto input = bytes(); auto source = runtime.file(input, UINT64_MAX);
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
  ranges[0].description = TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{16}).value();
  EXPECT_FALSE(runtime.pipeline.read(StreamSource::file(source), groups));
  EXPECT_TRUE(ranges[0].destination.valid()); EXPECT_TRUE(ranges[1].destination.valid());
  ranges[0].description.reset(); ranges[0].file_offset = UINT64_MAX - 8;
  EXPECT_FALSE(runtime.pipeline.read(StreamSource::file(source), groups));
  EXPECT_TRUE(ranges[0].destination.valid()); ranges[0].file_offset = 0;
  auto oversized_pool = LeasePool::create(m::system_resource(), 8192).value();
  ranges[0].destination = oversized_pool.allocate(128).value();
  auto rejected = runtime.pipeline.read(StreamSource::file(source), groups);
  ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsUnavailable()); EXPECT_TRUE(ranges[0].destination.valid());
  EXPECT_EQ(runtime.pipeline.report().io_calls, 0u); EXPECT_EQ(runtime.pipeline.report().native_calls, 0u);
}
TEST(FileStream, AcceptedSourceOwnsFileAndDescriptorsAfterCallerDropsThem) {
  Runtime runtime; const auto input = bytes(); auto source = runtime.file(input);
  auto pool = LeasePool::create(m::system_resource(), 1024).value(); StreamTicket ticket;
  {
    auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
    ticket = runtime.pipeline.read(StreamSource::file(source), groups).value();
  }
  source = {}; ASSERT_TRUE(ticket.wait().status.ok());
  for (unsigned i = 0; i < 2; ++i) {
    auto ready = ticket.next_ready().value();
    expect_bytes(ready.output(0).value(), std::span(input).subspan((ready.tag() - 10) * 128, 128));
  }
}
TEST(FileStream, OriginalReadonlySinkErrorKeepsZeroFileBytesAndDoesNotPublish) {
  Runtime runtime; const auto input = bytes(); runtime.file(input);
  auto raw = p::open_file(runtime.paths.back().c_str(), p::kOpenRead).value();
  auto file = io::File::adopt(runtime.io, 0, std::move(raw)).value();
  auto sink = StreamFile::adopt(std::move(file), input.size()).value();
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto writer = pool.allocate(input.size()).value(); ASSERT_TRUE(writer.view()->write(input).ok());
  auto source = writer.publish().value();
  std::array<StreamWriteRange, 1> ranges{StreamWriteRange::ready(source, 0, 0, input.size())};
  std::array<StreamWriteGroup, 1> groups{{{1, ranges}}};
  auto ticket = runtime.pipeline.write(sink, groups).value(); auto result = ticket.wait();
  EXPECT_FALSE(result.status.ok()); EXPECT_TRUE(result.retirement_proven); EXPECT_EQ(result.ready_groups, 0u);
  EXPECT_EQ(result.written_bytes, 0u); EXPECT_EQ(result.copied_bytes, 32u); EXPECT_FALSE(ticket.next_ready());
  auto unchanged = p::open_random_access_file(runtime.paths.back().c_str(), p::kOpenRead).value();
  std::array<std::byte, 256> output; ASSERT_EQ(unchanged.read_at(0, output).value(), output.size()); EXPECT_EQ(output, input);
}
TEST(FileStream, AcceptedHotMetadataOomReportsFailureAndReleasesOriginalOwners) {
  Runtime runtime; const auto input = bytes(); auto source = runtime.file(input);
  m::QuotaAllocator metadata(m::system_allocator());
  const std::array specs{CopyResourceSpec{m::system_resource(), 2048}};
  auto pipeline = cpu::create_file_pipeline(runtime.io, runtime.context, runtime.queue, m::system_resource(), specs,
      {.allocator = metadata.get(), .staging_bytes_per_slot = 32, .staging_alignment = 16}).value();
  struct Drain { FileResourcePipeline &pipeline; ~Drain() { pipeline.drain().IgnoreError(); pipeline = {}; } } drain{pipeline};
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
  auto cancellation = sched::CancelToken::root(m::system_allocator());
  auto no_hot_bytes = metadata.try_reserve_bytes(metadata.limit() - metadata.used()).value();
  auto admitted = pipeline.read(StreamSource::file(source), groups, {.deadline = {}, .cancel = &cancellation});
  ASSERT_TRUE(admitted); auto ticket = std::move(*admitted); source = {};
  auto result = ticket.wait(); EXPECT_TRUE(result.status.IsResourceExhausted()); EXPECT_TRUE(result.retirement_proven);
  EXPECT_EQ(result.ready_groups, 0u); EXPECT_EQ(result.read_bytes, 0u); EXPECT_EQ(result.copied_bytes, 0u);
  EXPECT_FALSE(ticket.next_ready()); EXPECT_EQ(pipeline.report().io_calls, 0u);
  ticket = {}; pipeline.drain().IgnoreError(); no_hot_bytes.reset(); pipeline = {}; EXPECT_EQ(metadata.used(), 0u);
}
TEST(FileStream, ExportedReadyAndLeaseCanRetireOnAnotherThreadWithoutReusingHeldRecord) {
  Runtime runtime({.max_requests = 1}); const auto input = bytes(); auto source = runtime.file(input);
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
  auto ticket = runtime.pipeline.read(StreamSource::file(source), groups).value(); ASSERT_TRUE(ticket.wait().status.ok());
  auto ready = ticket.next_ready().value(); auto lease = ready.output(0).value();
  std::atomic<bool> release{false};
  std::thread consumer([ready = std::move(ready), lease = std::move(lease), &release]() mutable {
    release.wait(false, std::memory_order_acquire); lease.reset(); ready = {};
  });
  struct Join {
    std::thread &thread; std::atomic<bool> &release;
    ~Join() { release.store(true, std::memory_order_release); release.notify_all(); if (thread.joinable()) thread.join(); }
  } join{consumer, release};
  ticket = {}; ASSERT_TRUE(runtime.pipeline.collect().ok());
  EXPECT_EQ(runtime.pipeline.report().live_requests, 1u); EXPECT_FALSE(runtime.pipeline.read({}, {}));
  release.store(true, std::memory_order_release); release.notify_all(); consumer.join();
  ASSERT_TRUE(runtime.pipeline.collect().ok()); EXPECT_EQ(pool.stats().live_blocks, 0u);
  auto next = runtime.pipeline.read({}, {}).value(); EXPECT_TRUE(next.wait().status.ok());
}
TEST(FileStream, ExpiredDeadlineAndRepeatedClosePreserveUserResourcesAndRejectNewWork) {
  Runtime runtime; const auto input = bytes(); auto source = runtime.file(input);
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto ranges = read_ranges(pool); auto groups = read_groups(ranges);
  auto ticket = runtime.pipeline.read(StreamSource::file(source), groups,
      {.deadline = time::monotonic_now() - time::Duration{1}, .cancel = nullptr}).value();
  auto result = ticket.wait(); EXPECT_TRUE(result.status.IsDeadlineExceeded());
  EXPECT_TRUE(result.retirement_proven); EXPECT_EQ(result.ready_groups, 0u);
  EXPECT_EQ(result.read_bytes, 0u); EXPECT_EQ(result.copied_bytes, 0u); EXPECT_FALSE(ticket.next_ready());
  EXPECT_EQ(runtime.pipeline.report().io_calls, 0u); EXPECT_EQ(runtime.pipeline.report().native_calls, 0u);
  ticket = {}; ASSERT_TRUE(runtime.pipeline.collect().ok());
  runtime.pipeline.stop_admission(); runtime.pipeline.stop_admission();
  auto rejected_ranges = read_ranges(pool); auto rejected_groups = read_groups(rejected_ranges);
  auto rejected = runtime.pipeline.read(StreamSource::file(source), rejected_groups);
  ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_TRUE(rejected_ranges[0].destination.valid()); EXPECT_TRUE(rejected_ranges[1].destination.valid());
  ASSERT_TRUE(runtime.pipeline.close().ok()); ASSERT_TRUE(runtime.pipeline.close().ok());
  ASSERT_TRUE(runtime.pipeline.drain().ok()); ASSERT_TRUE(runtime.pipeline.collect().ok());
  EXPECT_TRUE(source.valid()); EXPECT_TRUE(runtime.context.valid()); EXPECT_TRUE(runtime.queue.valid());
  EXPECT_TRUE(runtime.io.valid()); EXPECT_TRUE(runtime.scheduler.accepting());
  EXPECT_TRUE(runtime.file(input).valid());
  auto stored = p::open_random_access_file(runtime.paths.front().c_str(), p::kOpenRead).value();
  std::array<std::byte, 256> actual{}; ASSERT_EQ(stored.read_at(0, actual).value(), actual.size());
  EXPECT_EQ(actual, input);
}
#if !defined(_WIN32)
TEST(FileStream, OriginalFileSizeLimitPreservesPositiveShortWritePrefixAndNoGroupReady) {
  Runtime runtime; const auto input = bytes(); auto sink = runtime.file({}, 128);
  auto pool = LeasePool::create(m::system_resource(), 1024).value();
  auto writer = pool.allocate(128).value(); ASSERT_TRUE(writer.view()->write(std::span(input).first(128)).ok());
  auto source = writer.publish().value();
  std::array<StreamWriteRange, 1> ranges{StreamWriteRange::ready(source, 0, 0, 128)};
  std::array<StreamWriteGroup, 1> groups{{{1, ranges}}};
  struct Limit {
    struct rlimit original{}; struct sigaction signal{};
    bool limit_changed = false, signal_changed = false;
    void restore() {
      if (limit_changed) { EXPECT_EQ(setrlimit(RLIMIT_FSIZE, &original), 0); limit_changed = false; }
      if (signal_changed) { EXPECT_EQ(sigaction(SIGXFSZ, &signal, nullptr), 0); signal_changed = false; }
    }
    ~Limit() { restore(); }
  } limit;
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &limit.original), 0); ASSERT_GE(limit.original.rlim_max, 80u);
  struct sigaction ignored{}; ignored.sa_handler = SIG_IGN; sigemptyset(&ignored.sa_mask);
  ASSERT_EQ(sigaction(SIGXFSZ, &ignored, &limit.signal), 0); limit.signal_changed = true;
  auto bounded = limit.original; bounded.rlim_cur = 80;
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &bounded), 0); limit.limit_changed = true;
  auto admitted = runtime.pipeline.write(sink, groups);
  // Restore before diagnostics/XML/log writes; only this child process's
  // actual positional file IO is constrained, with its original signal kept.
  if (!admitted) { limit.restore(); FAIL() << "bounded file write admission failed"; }
  auto ticket = std::move(*admitted); auto result = ticket.wait(); limit.restore();
  EXPECT_FALSE(result.status.ok()); EXPECT_TRUE(result.retirement_proven); EXPECT_EQ(result.ready_groups, 0u);
  EXPECT_EQ(result.written_bytes, 80u); EXPECT_EQ(result.copied_bytes, 96u); EXPECT_FALSE(ticket.next_ready());
  auto stored = p::open_random_access_file(runtime.paths.back().c_str(), p::kOpenRead).value();
  std::array<std::byte, 128> prefix; ASSERT_EQ(stored.read_at(0, prefix).value(), 80u);
  EXPECT_TRUE(std::equal(prefix.begin(), prefix.begin() + 80, input.begin()));
}
#endif
} // namespace
