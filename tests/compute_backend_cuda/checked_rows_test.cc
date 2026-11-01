#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "../../src/compute_backend_cuda/checked_rows.h"
#include "../../src/compute_backend_cuda/native_context.h"

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
namespace cuda = c::cuda;
namespace m = hyle::mem;

struct Fixture final {
  c::Context context = c::Context::create(cuda::device(0).value()).value();
  cuda::MemoryResource memory = cuda::MemoryResource::create(
      context, cuda::MemoryKind::kDevice).value();
  c::Queue queue = context.create_queue().value();
  Fixture() { EXPECT_TRUE(queue.reserve_completions(8).ok()); }
  ~Fixture() { context.drain().IgnoreError(); }
};

struct Bytes final {
  Fixture &fixture;
  std::size_t active, total, source_offset, destination_offset;
  m::Buffer source, destination, offsets, good_offsets;
  std::vector<std::byte> input, initial;

  Bytes(Fixture &f, std::size_t prefix, std::size_t extent,
        std::size_t in_offset = 0, std::size_t out_offset = 32)
      : fixture(f), active(prefix), total(extent), source_offset(in_offset),
        destination_offset(out_offset) {
    input.assign(source_offset + active + 31, std::byte{0xa5});
    for (std::size_t i = 0; i < active; ++i)
      input[source_offset + i] = std::byte((i * 37 + i / 257 + 19) & 255);
    initial.assign(destination_offset + total + 31, std::byte{0x5a});
    source = m::Buffer::create(f.memory.ref(), input.size(), 256).value();
    destination = m::Buffer::create(f.memory.ref(), initial.size(), 256).value();
    offsets = m::Buffer::create(f.memory.ref(), 2 * sizeof(std::int32_t), 256).value();
    good_offsets = m::Buffer::create(f.memory.ref(), 2 * sizeof(std::int32_t), 256).value();
    EXPECT_TRUE(source.view().write(input).ok());
    EXPECT_TRUE(destination.view().write(initial).ok());
    const std::array<std::int32_t, 2> valid{0, 1};
    EXPECT_TRUE(good_offsets.view().write(std::as_bytes(std::span(valid))).ok());
  }
  ~Bytes() { fixture.queue.drain().IgnoreError(); }

  hyle::Status execute(std::array<std::int32_t, 2> values = {0, 1},
      bool check_again = false, bool null_source = false) {
    SCOPED_TRACE(::testing::Message() << "active_bytes=" << active
        << " total_bytes=" << total << " source_offset=" << source_offset
        << " destination_offset=" << destination_offset);
    HYLE_RETURN_IF_ERROR(offsets.view().write(std::as_bytes(std::span(values))));
    HYLE_RETURN_IF_ERROR(destination.view().write(initial));
    HYLE_ASSIGN_OR_RETURN(auto primary, cuda::primary_context(fixture.context));
    cuda::detail::ScopedPrimaryContext scope(primary);
    HYLE_RETURN_IF_ERROR(scope.status());
    const void *source_pointer = nullptr;
    if (active) {
      HYLE_ASSIGN_OR_RETURN(source_pointer,
          cuda::detail::const_pointer(source.view().subview(source_offset, active).value(), 1, 1));
    }
    void *destination_pointer = nullptr;
    if (total) {
      HYLE_ASSIGN_OR_RETURN(destination_pointer,
          cuda::detail::mutable_pointer(destination.view().subview(destination_offset, total).value(), 1, 1));
    }
    HYLE_ASSIGN_OR_RETURN(auto control, cuda::pointer<std::int32_t>(offsets.view()));
    HYLE_ASSIGN_OR_RETURN(auto valid_control, cuda::pointer<std::int32_t>(good_offsets.view()));
    HYLE_ASSIGN_OR_RETURN(auto sequence, cd::QueueSequence::begin(fixture.queue));
    unsigned *status = nullptr;
    auto initialized = cuda::detail::acquire_device_status(sequence, &status);
    const auto initialization_status = initialized.status;
    sequence.record(std::move(initialized));
    auto terminal = initialization_status;
    if (terminal.ok()) {
      const auto stream = cuda::stream(sequence);
      if (!stream) {
        terminal = stream.error();
        sequence.record(cd::EnqueueResult::rejected(terminal));
      } else {
        terminal = cuda::status(cuda::detail::check_packed_r1_offsets_expected(
            control, 4, active ? 1 : 0, status, *stream));
        sequence.record(cd::EnqueueResult::accepted_command(terminal));
        if (terminal.ok() && check_again) {
          // Acquiring again in this SAME sequence must preserve its earlier
          // device error. No host observation separates these two checks.
          auto inherited = cuda::detail::acquire_device_status(sequence, &status);
          terminal = inherited.status;
          sequence.record(std::move(inherited));
          if (terminal.ok()) {
            terminal = cuda::status(cuda::detail::check_packed_r1_offsets_expected(
                valid_control, 4, 1, status, *stream));
            sequence.record(cd::EnqueueResult::accepted_command(terminal));
          }
        }
        if (terminal.ok()) {
          terminal = cuda::status(cuda::detail::publish_checked_prefix(
              null_source ? nullptr : source_pointer, destination_pointer,
              active, total, status, *stream));
          sequence.record(cd::EnqueueResult::accepted_command(terminal));
        }
      }
    }
    auto done = sequence.finish();
    if (!done) return done.error();
    const auto completed = done->wait();
    *done = {};
    return completed;
  }

  hyle::Status execute_packed(std::array<std::int32_t, 2> values,
      bool inherited_error = false, bool original_after = false,
      unsigned alias = 0) {
    SCOPED_TRACE(::testing::Message() << "combined active_bytes=" << active
        << " total_bytes=" << total << " source_offset=" << source_offset
        << " destination_offset=" << destination_offset << " alias=" << alias);
    HYLE_RETURN_IF_ERROR(offsets.view().write(std::as_bytes(std::span(values))));
    HYLE_RETURN_IF_ERROR(destination.view().write(initial));
    const std::array<std::int32_t, 2> invalid{0, -1};
    HYLE_RETURN_IF_ERROR(good_offsets.view().write(std::as_bytes(std::span(invalid))));
    HYLE_ASSIGN_OR_RETURN(auto primary, cuda::primary_context(fixture.context));
    cuda::detail::ScopedPrimaryContext scope(primary);
    HYLE_RETURN_IF_ERROR(scope.status());
    const void *source_pointer = nullptr;
    if (active) {
      HYLE_ASSIGN_OR_RETURN(source_pointer,
          cuda::detail::const_pointer(source.view().subview(source_offset, active).value(), 1, 1));
    }
    HYLE_ASSIGN_OR_RETURN(auto destination_pointer,
        cuda::detail::mutable_pointer(destination.view().subview(destination_offset, total).value(), 1, 1));
    HYLE_ASSIGN_OR_RETURN(auto control, cuda::pointer<std::int32_t>(offsets.view()));
    HYLE_ASSIGN_OR_RETURN(auto prior_control, cuda::pointer<std::int32_t>(good_offsets.view()));
    HYLE_ASSIGN_OR_RETURN(auto sequence, cd::QueueSequence::begin(fixture.queue));
    unsigned *status = nullptr;
    auto initialized = cuda::detail::acquire_device_status(sequence, &status);
    auto terminal = initialized.status;
    sequence.record(std::move(initialized));
    const auto stream = cuda::stream(sequence);
    if (terminal.ok() && !stream) {
      terminal = stream.error();
      sequence.record(cd::EnqueueResult::rejected(terminal));
    }
    if (terminal.ok() && inherited_error) {
      terminal = cuda::status(cuda::detail::check_packed_r1_offsets_expected(
          prior_control, 4, 1, status, *stream));
      sequence.record(cd::EnqueueResult::accepted_command(terminal));
    }
    if (terminal.ok()) {
      terminal = cuda::status(cuda::detail::publish_packed_r1_prefix(
          alias == 3 ? destination_pointer : source_pointer, destination_pointer,
          1, static_cast<std::int64_t>(total), static_cast<std::int32_t>(active),
          alias == 1 ? static_cast<const std::int32_t *>(destination_pointer) : control,
          alias == 2 ? static_cast<unsigned *>(destination_pointer) : status, *stream));
      sequence.record(cd::EnqueueResult::accepted_command(terminal));
    }
    if (terminal.ok() && original_after) {
      auto inherited = cuda::detail::acquire_device_status(sequence, &status);
      terminal = inherited.status;
      sequence.record(std::move(inherited));
      if (terminal.ok()) {
        terminal = cuda::status(cuda::detail::publish_checked_prefix(
            source_pointer, destination_pointer, active, total, status, *stream));
        sequence.record(cd::EnqueueResult::accepted_command(terminal));
      }
    }
    auto done = sequence.finish();
    if (!done) return done.error();
    const auto completed = done->wait();
    *done = {};
    return completed;
  }

  void expect(bool published) const {
    auto wanted = initial;
    if (published) {
      std::copy_n(input.begin() + source_offset, active, wanted.begin() + destination_offset);
      std::fill(wanted.begin() + destination_offset + active,
          wanted.begin() + destination_offset + total, std::byte{0});
    }
    std::vector<std::byte> actual(initial.size()), unchanged(input.size());
    ASSERT_TRUE(destination.view().read(actual).ok());
    ASSERT_TRUE(source.view().read(unchanged).ok());
    EXPECT_EQ(actual, wanted);
    EXPECT_EQ(unchanged, input);
  }
};

TEST(CudaCheckedRows, AlignedTilesAndUnalignedByteTailsKeepEveryBoundary) {
  Fixture fixture;
  for (const bool aligned : {false, true}) {
    SCOPED_TRACE(aligned);
    Bytes bytes(fixture, 16 * 1024 + (aligned ? 16 : 3), 48 * 1024 + 7,
        aligned ? 0 : 1, aligned ? 32 : 3);
    const auto status = bytes.execute();
    ASSERT_TRUE(status.ok()) << status.message();
    bytes.expect(true);
  }
  // No private payload exists for L0. Its full output still becomes +0.
  Bytes empty(fixture, 0, 16 * 1024 + 3);
  const auto status = empty.execute({0, 0}, false, true);
  ASSERT_TRUE(status.ok()) << status.message();
  empty.expect(true);
  // A zero-byte destination is a real no-op and leaves both guards intact.
  Bytes no_output(fixture, 0, 0);
  const auto no_op = no_output.execute({0, 0}, false, true);
  ASSERT_TRUE(no_op.ok()) << no_op.message();
  no_output.expect(true);
}

TEST(CudaCheckedRows, OffsetMismatchAndInheritedStatusPublishNoBytesAndRetry) {
  Fixture fixture;
  Bytes bytes(fixture, 32, 96);
  for (const auto invalid : {std::array<std::int32_t, 2>{1, 1},
      std::array<std::int32_t, 2>{0, -1}, std::array<std::int32_t, 2>{0, 5},
      std::array<std::int32_t, 2>{0, 2}}) {
    SCOPED_TRACE(invalid[0]);
    SCOPED_TRACE(invalid[1]);
    const auto bad = bytes.execute(invalid);
    EXPECT_TRUE(bad.IsInvalidArgument()) << bad.message();
    bytes.expect(false);
    const auto good = bytes.execute();
    ASSERT_TRUE(good.ok()) << good.message();
    bytes.expect(true);
  }
  const auto inherited = bytes.execute({0, 5}, true);
  EXPECT_TRUE(inherited.IsInvalidArgument()) << inherited.message();
  bytes.expect(false);
  // Host-invalid publication is rejected after the check but never writes.
  const auto invalid_source = bytes.execute({0, 1}, false, true);
  EXPECT_TRUE(invalid_source.IsInvalidArgument()) << invalid_source.message();
  bytes.expect(false);
}

TEST(CudaCheckedRows, CombinedPublicationChecksEveryTileAndInheritsTheOriginalStatus) {
  Fixture fixture;
  for (const bool aligned : {false, true}) {
    SCOPED_TRACE(aligned);
    Bytes bytes(fixture, 16 * 1024 + (aligned ? 16 : 3), 48 * 1024 + 7,
        aligned ? 0 : 1, aligned ? 32 : 3);
    const auto status = bytes.execute_packed({0, static_cast<std::int32_t>(bytes.active)});
    ASSERT_TRUE(status.ok()) << status.message();
    bytes.expect(true);
    for (const auto invalid : {std::array<std::int32_t, 2>{1, std::int32_t(bytes.active)},
        std::array<std::int32_t, 2>{0, -1},
        std::array<std::int32_t, 2>{0, std::int32_t(bytes.total + 1)},
        std::array<std::int32_t, 2>{0, std::int32_t(bytes.active - 1)}}) {
      const auto bad = bytes.execute_packed(invalid, false, true);
      EXPECT_TRUE(bad.IsInvalidArgument()) << bad.message();
      bytes.expect(false);
    }
    const auto inherited = bytes.execute_packed(
        {0, static_cast<std::int32_t>(bytes.active)}, true, true);
    EXPECT_TRUE(inherited.IsInvalidArgument()) << inherited.message();
    bytes.expect(false);
  }
  Bytes empty(fixture, 0, 16 * 1024 + 3);
  const auto zero = empty.execute_packed({0, 0});
  ASSERT_TRUE(zero.ok()) << zero.message();
  empty.expect(true);
  Bytes aliases(fixture, 32, 96);
  for (unsigned alias : {1u, 2u, 3u}) {
    SCOPED_TRACE(alias);
    const auto rejected = aliases.execute_packed({0, 32}, false, false, alias);
    EXPECT_TRUE(rejected.IsInvalidArgument()) << rejected.message();
    aliases.expect(false);
  }
  const auto retry = aliases.execute_packed({0, 32});
  ASSERT_TRUE(retry.ok()) << retry.message();
  aliases.expect(true);
}
} // namespace
