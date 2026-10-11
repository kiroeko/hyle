#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

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
  auto cancelled = runtime.pipeline.read(StreamSource::file(source), cancelled_groups, {.cancel = &cancellation}).value();
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
} // namespace
