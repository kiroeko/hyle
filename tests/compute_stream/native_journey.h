#pragma once

#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <string>
#include <vector>

#include "hyle/compute/file_stream.h"
#include "hyle/mem/stats.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/join.h"

namespace file_stream_test {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace p = hyle::platform;
namespace io = hyle::io;
namespace s = hyle::sched;
struct Host final {
  m::StatsAllocator metadata = m::StatsAllocator::create(m::system_allocator()).value();
  s::Scheduler scheduler{2, m::system_allocator(), 256};
  io::IoContext io{scheduler, {.max_blocking_threads = 2}};
  std::vector<std::string> paths;
  ~Host() { scheduler.shutdown(); for (const auto &path : paths) p::remove_file(path.c_str()).IgnoreError(); }
  static s::Task<hyle::Result<io::File>> open(m::AllocatorRef, io::IoContext *io, std::string path, unsigned owner) noexcept {
    co_return co_await io::open_file(*io, owner, path, {.access = io::FileAccess::kReadWrite});
  }
  c::StreamFile file(std::span<const std::byte> bytes, std::uint64_t capacity, unsigned owner = 0) {
    static std::atomic<unsigned> sequence{0};
    auto path = "hyle_f3_native_" + std::to_string(sequence.fetch_add(1)) + ".tmp"; paths.push_back(path);
    auto raw = p::open_file(path.c_str(), p::kOpenRead | p::kOpenWrite | p::kOpenCreate | p::kOpenTruncate).value();
    EXPECT_TRUE(raw.write_all(bytes).ok()); EXPECT_TRUE(raw.close().ok());
    auto file = s::spawn_task_on(scheduler, owner, m::system_allocator(), open(m::system_allocator(), &io, path, owner)).join().value().value();
    return c::StreamFile::adopt(std::move(file), capacity).value();
  }
  c::FileStreamOptions options(unsigned slots = 2) {
    return {.allocator = metadata.get(), .max_requests = 2, .max_groups_per_request = 4,
            .max_ranges_per_group = 4, .in_flight_slots = slots,
            .staging_bytes_per_slot = 64, .staging_alignment = 64, .max_active_backing_bytes = 4096};
  }
};
struct Drain final {
  c::FileResourcePipeline &pipeline;
  ~Drain() { pipeline.drain().IgnoreError(); pipeline = {}; }
};
inline std::array<std::byte, 512> input(unsigned salt = 0) {
  std::array<std::byte, 512> bytes;
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::byte((i * 17 + salt) % 251);
  return bytes;
}
inline std::array<c::StreamReadRange, 2> ranges(c::LeasePool &pool) {
  std::array<c::StreamReadRange, 2> result;
  for (std::size_t i = 0; i < 2; ++i) {
    result[i].file_offset = i * 256; result[i].bytes = 256; result[i].destination = pool.allocate(256).value();
  } return result;
}
inline std::array<c::StreamReadGroup, 2> groups(std::array<c::StreamReadRange, 2> &ranges) {
  return {{{41, std::span(ranges).first(1)}, {42, std::span(ranges).subspan(1)}}};
}
inline void roundtrip(Host &host, c::FileResourcePipeline &pipeline, c::LeasePool &pool) {
  Drain drain{pipeline};
  for (unsigned replay = 0; replay < 3; ++replay) {
    auto bytes = input(replay * 29); auto source = host.file(bytes, bytes.size(), 1); auto sink = host.file({}, bytes.size(), 0);
    auto reading = ranges(pool); auto requests = groups(reading);
    auto ticket = pipeline.read(c::StreamSource::file(source), requests).value();
    auto first = ticket.wait_next_ready().value(), second = ticket.wait_next_ready().value();
    auto result = ticket.wait(); ASSERT_TRUE(result.status.ok()); EXPECT_TRUE(result.retirement_proven);
    EXPECT_EQ(result.ready_groups, 2u); EXPECT_EQ(result.read_bytes, 512u); EXPECT_EQ(result.copied_bytes, 512u);
    std::array<c::ReadLease, 2> ready;
    ready[first.tag() - 41] = first.output(0).value(); ready[second.tag() - 41] = second.output(0).value();
    std::array<c::StreamWriteRange, 2> writing{
        c::StreamWriteRange::ready(ready[0], 0, 0, 256), c::StreamWriteRange::ready(ready[1], 256, 0, 256)};
    std::array<c::StreamWriteGroup, 2> write_groups{{{51, std::span(writing).first(1)}, {52, std::span(writing).subspan(1)}}};
    auto write = pipeline.write(sink, write_groups).value();
    auto saved = write.wait(); char failure[512]{}; saved.status.ToString(failure, sizeof(failure));
    ASSERT_TRUE(saved.status.ok()) << failure << " copied=" << saved.copied_bytes << " written=" << saved.written_bytes;
    EXPECT_EQ(saved.written_bytes, 512u);
    auto stored = p::open_random_access_file(host.paths.back().c_str(), p::kOpenRead).value();
    std::array<std::byte, 512> output; EXPECT_EQ(stored.read_at(0, output).value(), output.size()); EXPECT_EQ(output, bytes);
    stored.close().IgnoreError();
    ticket = {}; write = {}; first = {}; second = {}; EXPECT_TRUE(pipeline.collect().ok());
    EXPECT_EQ(pipeline.report().live_requests, 1u); // exported read leases still own their record
    for (auto &lease : ready) lease.reset(); EXPECT_TRUE(pipeline.collect().ok());
    EXPECT_EQ(pipeline.report().live_requests, 0u); EXPECT_EQ(pool.stats().live_blocks, 0u);
  }
  const auto report = pipeline.report(); EXPECT_LE(report.peak_slots, 2u);
  EXPECT_GE(report.native_calls, 48u); EXPECT_GE(report.phase_completions, 48u);
  EXPECT_TRUE(report.native_metadata_unknown); // driver/runtime internals are not claimed zero
  ::testing::Test::RecordProperty("native_calls", static_cast<int>(report.native_calls));
  ::testing::Test::RecordProperty("phase_completions", static_cast<int>(report.phase_completions));
}
inline void late_eof(Host &host, c::FileResourcePipeline &pipeline, c::LeasePool &pool) {
  Drain drain{pipeline}; const auto bytes = input(); auto source = host.file(std::span(bytes).first(270), 512, 1);
  auto reading = ranges(pool); auto requests = groups(reading);
  auto ticket = pipeline.read(c::StreamSource::file(source), requests).value();
  auto early = ticket.wait_next_ready().value(); EXPECT_EQ(early.tag(), 41u);
  auto result = ticket.wait(); EXPECT_FALSE(result.status.ok()); EXPECT_TRUE(result.retirement_proven);
  EXPECT_EQ(result.read_bytes, 270u); EXPECT_EQ(result.copied_bytes, 256u); EXPECT_EQ(result.ready_groups, 1u);
  auto sink = host.file({}, 256);
  std::array<c::StreamWriteRange, 1> writing{c::StreamWriteRange::ready(early.output(0).value(), 0, 0, 256)};
  std::array<c::StreamWriteGroup, 1> write_groups{{{71, writing}}};
  auto write = pipeline.write(sink, write_groups).value(); ASSERT_TRUE(write.wait().status.ok());
  auto stored = p::open_random_access_file(host.paths.back().c_str(), p::kOpenRead).value();
  std::array<std::byte, 256> output; ASSERT_EQ(stored.read_at(0, output).value(), output.size());
  EXPECT_TRUE(std::equal(output.begin(), output.end(), bytes.begin()));
}
} // namespace file_stream_test
