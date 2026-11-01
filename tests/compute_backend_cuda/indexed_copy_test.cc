#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/command_capture.h"
#include "hyle/compute/detail/indexed_copy_backend.h"

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
namespace m = hyle::mem;

struct IndexedCopyFixture final {
  c::Context context = c::Context::create(c::cuda::device(0).value()).value();
  c::cuda::MemoryResource memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  c::Queue queue = context.create_queue().value();
  cd::BackendExtensionRef extension = cd::BackendExtensionAccess::query(context, cd::kIndexedCopyExtensionId).value();
  IndexedCopyFixture() { EXPECT_TRUE(queue.reserve_completions(8).ok()); }
  ~IndexedCopyFixture() { context.drain().IgnoreError(); }
};

struct IndexedProgram final {
  const cd::IndexedCopyBackendVTable *table = nullptr;
  void *native = nullptr;
  ~IndexedProgram() { if (native) table->destroy(native, m::system_allocator()); }
  hyle::Status prepare(const cd::BackendExtensionRef &extension, cd::IndexedCopyNativeSpec spec) noexcept {
    struct Call { IndexedProgram *self; cd::IndexedCopyNativeSpec spec; } call{this, spec};
    return cd::BackendExtensionAccess::call(extension,
        +[](void *context, const void *opaque_table, void *opaque) noexcept -> hyle::Status {
          auto &call = *static_cast<Call *>(opaque);
          call.self->table = static_cast<const cd::IndexedCopyBackendVTable *>(opaque_table);
          return call.self->table->prepare(context, call.spec, m::system_allocator(), &call.self->native);
        }, &call);
  }
  hyle::Status enqueue(const cd::BackendExtensionRef &extension, cd::QueueSequence &sequence,
                       cd::IndexedCopyNativeBindings bindings) const noexcept {
    struct Call { const IndexedProgram *self; cd::IndexedCopyNativeBindings bindings; } call{this, bindings};
    auto result = cd::BackendExtensionAccess::enqueue_admitted(extension, sequence,
        +[](void *context, const void *, void *queue, void *opaque) noexcept -> cd::EnqueueResult {
          const auto &call = *static_cast<Call *>(opaque);
          return call.self->table->enqueue(context, queue, call.self->native, call.bindings);
        }, &call);
    const auto status = result.status;
    sequence.record(std::move(result));
    return status;
  }
};

struct CopyBytes final {
  c::Queue &queue;
  cd::IndexedCopyNativeSpec spec;
  m::Buffer capacity, rows, indices;
  m::BufferView capacity_view, rows_view;
  std::vector<std::byte> initial, expected, payload;
  std::size_t destination_offset;

  CopyBytes(m::MemoryResourceRef resource, c::Queue &original_queue, cd::IndexedCopyNativeSpec geometry,
            std::span<const std::int32_t> destinations, std::size_t source_offset = 0,
            std::size_t output_offset = 0)
      : queue(original_queue), spec(geometry), destination_offset(output_offset) {
    const auto capacity_bytes = std::size_t(spec.slot_count - 1) * spec.slot_stride + spec.row_bytes;
    initial.assign(output_offset + capacity_bytes + 31, std::byte{0xa5});
    expected = initial;
    payload.resize(std::size_t(spec.row_count) * spec.row_bytes);
    // All byte patterns, including NaN/sign-bit encodings, must survive exactly.
    for (std::size_t i = 0; i < payload.size(); ++i)
      payload[i] = std::byte((i * 37 + i / 257 + 19) & 255);
    std::vector<std::byte> source(source_offset + payload.size() + 31, std::byte{0x5a});
    std::copy(payload.begin(), payload.end(), source.begin() + source_offset);
    capacity = m::Buffer::create(resource, initial.size(), 256).value();
    rows = m::Buffer::create(resource, source.size(), 256).value();
    indices = m::Buffer::create(resource, destinations.size_bytes(), 256).value();
    EXPECT_TRUE(capacity.view().write(initial).ok());
    EXPECT_TRUE(rows.view().write(source).ok());
    EXPECT_TRUE(indices.view().write(std::as_bytes(destinations)).ok());
    capacity_view = capacity.view().subview(output_offset, capacity_bytes).value();
    rows_view = rows.view().subview(source_offset, payload.size()).value();
  }
  // Even a fatal assertion after acceptance must retire original native work
  // before the raw caller-owned payload/control buffers are released.
  ~CopyBytes() { queue.drain().IgnoreError(); }
  cd::IndexedCopyNativeBindings bindings() const { return {capacity_view, rows_view, indices.view()}; }
  void select(std::span<const std::int32_t> destinations) {
    expected = initial;
    for (std::size_t row = 0; row < destinations.size(); ++row)
      std::copy_n(payload.begin() + row * spec.row_bytes, spec.row_bytes,
          expected.begin() + destination_offset + std::size_t(destinations[row]) * spec.slot_stride);
  }
  void expect_bytes(const std::vector<std::byte> &wanted) const {
    std::vector<std::byte> actual(wanted.size());
    ASSERT_TRUE(capacity.view().read(actual).ok());
    EXPECT_EQ(actual, wanted);
  }
};

TEST(CudaIndexedCopy, MultiBlockRowsAreByteExactWithAlignmentTailsAndPadding) {
  IndexedCopyFixture fixture;
  struct Geometry { std::size_t bytes, stride, source_offset, output_offset; };
  const std::array cases{
      Geometry{2u * 1024u * 1024u + 32u, 2u * 1024u * 1024u + 256u, 0, 0},
      Geometry{16u * 1024u + 7u, 16u * 1024u + 256u, 0, 0},
      Geometry{16u * 1024u + 13u, 16u * 1024u + 30u, 1, 3},
      Geometry{1, 19, 0, 0}, Geometry{15, 32, 0, 0},
      Geometry{16, 32, 0, 0}, Geometry{17, 33, 0, 0}};
  const std::array<std::int32_t, 3> destinations{4, 0, 2};
  for (const auto geometry : cases) {
    SCOPED_TRACE(geometry.bytes);
    const cd::IndexedCopyNativeSpec spec{5, 3, geometry.bytes, geometry.stride};
    CopyBytes bytes(fixture.memory.ref(), fixture.queue, spec, destinations, geometry.source_offset, geometry.output_offset);
    bytes.select(destinations);
    IndexedProgram program;
    const auto prepared = program.prepare(fixture.extension, spec);
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    auto sequence = cd::QueueSequence::begin(fixture.queue).value();
    const auto enqueued = program.enqueue(fixture.extension, sequence, bytes.bindings());
    ASSERT_TRUE(enqueued.ok()) << enqueued.message();
    auto complete = sequence.finish();
    ASSERT_TRUE(complete) << complete.error().message();
    ASSERT_TRUE(complete->wait().ok());
    bytes.expect_bytes(bytes.expected);
  }
}

TEST(CudaIndexedCopy, InvalidLastIndexPreventsEveryPayloadWriteAcrossCheckBlocks) {
  IndexedCopyFixture fixture;
  constexpr std::uint32_t count = 513;
  const cd::IndexedCopyNativeSpec spec{count, count, 19, 37};
  std::vector<std::int32_t> destinations(count);
  for (std::uint32_t i = 0; i < count; ++i) destinations[i] = std::int32_t(i);
  IndexedProgram program;
  ASSERT_TRUE(program.prepare(fixture.extension, spec).ok());
  for (const auto invalid : {-1, std::int32_t(count)}) {
    SCOPED_TRACE(invalid);
    destinations.back() = invalid;
    CopyBytes bytes(fixture.memory.ref(), fixture.queue, spec, destinations, 1, 3);
    auto sequence = cd::QueueSequence::begin(fixture.queue).value();
    ASSERT_TRUE(program.enqueue(fixture.extension, sequence, bytes.bindings()).ok());
    auto complete = sequence.finish();
    ASSERT_TRUE(complete) << complete.error().message();
    const auto status = complete->wait();
    EXPECT_TRUE(status.IsOutOfRange()) << status.message();
    bytes.expect_bytes(bytes.initial);
  }
}

TEST(CudaIndexedCopy, RowsPastGridLimitRetainTheirDistinctByteDestinations) {
  IndexedCopyFixture fixture;
  constexpr std::uint32_t count = 65539;
  const cd::IndexedCopyNativeSpec spec{count, count, 1, 3};
  std::vector<std::int32_t> destinations(count);
  for (std::uint32_t i = 0; i < count; ++i) destinations[i] = std::int32_t(count - 1 - i);
  CopyBytes bytes(fixture.memory.ref(), fixture.queue, spec, destinations, 1, 3);
  bytes.select(destinations);
  IndexedProgram program;
  ASSERT_TRUE(program.prepare(fixture.extension, spec).ok());
  auto sequence = cd::QueueSequence::begin(fixture.queue).value();
  ASSERT_TRUE(program.enqueue(fixture.extension, sequence, bytes.bindings()).ok());
  auto complete = sequence.finish();
  ASSERT_TRUE(complete) << complete.error().message();
  ASSERT_TRUE(complete->wait().ok());
  bytes.expect_bytes(bytes.expected);
}

TEST(CudaIndexedCopy, CapturedMultiBlockCopyInheritsErrorAndNextSequenceReseedsStatus) {
  IndexedCopyFixture fixture;
  const cd::IndexedCopyNativeSpec spec{3, 2, 1024u * 1024u + 3u, 1024u * 1024u + 256u};
  const std::array<std::int32_t, 2> invalid{0, 3}, valid{2, 0};
  CopyBytes bad(fixture.memory.ref(), fixture.queue, spec, invalid);
  CopyBytes good(fixture.memory.ref(), fixture.queue, spec, valid);
  good.select(valid);
  IndexedProgram program;
  ASSERT_TRUE(program.prepare(fixture.extension, spec).ok());
  const auto capture_extension = cd::CommandCaptureAccess::query(fixture.queue).value();
  auto recording = cd::QueueSequence::begin(fixture.queue).value();
  ASSERT_TRUE(cd::CommandCaptureAccess::begin(capture_extension, recording).ok());
  ASSERT_TRUE(program.enqueue(fixture.extension, recording, good.bindings()).ok());
  auto captured = cd::CommandCaptureAccess::end(capture_extension, recording, m::system_allocator());
  ASSERT_TRUE(captured) << captured.error().message();
  auto recorded = recording.finish();
  ASSERT_TRUE(recorded) << recorded.error().message();
  ASSERT_TRUE(recorded->wait().ok());
  *recorded = {};
  good.expect_bytes(good.initial); // Recording never executes the payload copy.

  auto failing = cd::QueueSequence::begin(fixture.queue).value();
  ASSERT_TRUE(program.enqueue(fixture.extension, failing, bad.bindings()).ok());
  auto replay = captured->enqueue(failing);
  ASSERT_TRUE(replay.accepted && replay.status.ok()) << replay.status.message();
  failing.record(std::move(replay));
  auto failed = failing.finish();
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_TRUE(failed->wait().IsOutOfRange());
  *failed = {};
  bad.expect_bytes(bad.initial); good.expect_bytes(good.initial);

  auto succeeding = cd::QueueSequence::begin(fixture.queue).value();
  auto success = captured->enqueue(succeeding);
  ASSERT_TRUE(success.accepted && success.status.ok()) << success.status.message();
  succeeding.record(std::move(success));
  auto complete = succeeding.finish();
  ASSERT_TRUE(complete) << complete.error().message();
  ASSERT_TRUE(complete->wait().ok());
  good.expect_bytes(good.expected);
}

TEST(CudaIndexedCopy, RejectsOverflowAndEmptyGeometryBeforeNativeAdmission) {
  IndexedCopyFixture fixture;
  const auto maximum = std::numeric_limits<std::size_t>::max();
  const std::array cases{
      cd::IndexedCopyNativeSpec{0, 0, 1, 1},
      cd::IndexedCopyNativeSpec{2, 2, maximum / 2 + 1, maximum / 2 + 1},
      cd::IndexedCopyNativeSpec{3, 1, 1, maximum},
      cd::IndexedCopyNativeSpec{2, 2, 17, 16}};
  for (const auto spec : cases) {
    IndexedProgram program;
    const auto status = program.prepare(fixture.extension, spec);
    EXPECT_FALSE(status.ok()); EXPECT_EQ(program.native, nullptr);
  }
}
} // namespace
