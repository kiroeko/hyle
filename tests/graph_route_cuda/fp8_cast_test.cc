#include "../compute_provider_native/aten_cast_reference.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/cuda.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace ref = cast_reference;
using Type = c::DataType;
using Bytes = std::vector<std::byte>;

template <class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
c::TensorDesc dense(Type type, std::size_t count) {
  return c::TensorDesc::dense(type, std::array{static_cast<std::int64_t>(count)}).value();
}
Bytes payload(std::span<const std::uint32_t> codes, int width) {
  Bytes bytes(codes.size() * static_cast<std::size_t>(width));
  for (std::size_t i = 0; i < codes.size(); ++i)
    std::memcpy(bytes.data() + i * static_cast<std::size_t>(width), &codes[i], width);
  return bytes;
}
Bytes read(c::ConstTensorView view) {
  Bytes result(view.description().required_bytes().value());
  check(view.storage().read(result), "BufferView::read");
  return result;
}
std::uint32_t expected(std::uint32_t bits, const ref::Format &source, const ref::Format &target) {
  if (source.type == target.type) return bits;
  const auto wide = ref::float_bits(bits, source);
  return target.width == 4 ? wide : ref::encode(std::bit_cast<float>(wide), target, (wide >> 31) != 0);
}
void exact(const Bytes &bytes, std::span<const std::uint32_t> codes,
           const ref::Format &source, const ref::Format &target) {
  ASSERT_EQ(bytes.size(), codes.size() * static_cast<std::size_t>(target.width));
  for (std::size_t i = 0; i < codes.size(); ++i) {
    std::uint32_t actual = 0;
    std::memcpy(&actual, bytes.data() + i * static_cast<std::size_t>(target.width), target.width);
    ASSERT_EQ(actual, expected(codes[i], source, target))
        << "index=" << i << " source=" << int(source.type) << " target=" << int(target.type)
        << " raw=" << codes[i];
  }
}
std::vector<std::uint32_t> all_codes(unsigned count) {
  std::vector<std::uint32_t> result(count);
  for (unsigned i = 0; i < count; ++i) result[i] = i;
  return result;
}
Bytes direct(c::Context &context, c::Queue &queue, m::MemoryResourceRef memory,
             const ref::Format &source, const ref::Format &target,
             std::span<const std::uint32_t> codes) {
  const auto in = dense(source.type, codes.size()), out = dense(target.type, codes.size());
  auto source_buffer = take(m::Buffer::create(memory, in.required_bytes().value(), 256), "Direct input buffer");
  const auto bytes = payload(codes, source.width);
  check(source_buffer.view().write(bytes), "Direct source upload");
  auto destination = take(m::Buffer::create(memory, out.required_bytes().value(), 256), "Direct output buffer");
  const auto a = c::ConstTensorView::create(in, source_buffer.view()).value();
  const auto y = c::TensorView::create(out, destination.view()).value();
  auto prepared = take(c::prepare(context, c::Cast{target.type}, in, memory), "FP8 Cast prepare");
  EXPECT_EQ(prepared.provider().id, c::cuda_native::kProviderId);
  EXPECT_FALSE(prepared.alias_support(0, 0)->supported);
  EXPECT_TRUE(prepared.runtime_properties().capture_compatible);
  EXPECT_EQ(prepared.runtime_properties().replay_class, c::ProviderReplayClass::kPrepared);
  EXPECT_EQ(prepared.workspace_requirement().size, 0u);
  EXPECT_EQ(prepared.invocation_private_bytes(), 0u);
  auto call = take(prepared.create_invocation(), "FP8 Cast invocation");
  auto done = take(call.submit(queue, a, y), "FP8 Cast submit");
  check(done.wait(), "FP8 Cast Completion::wait");
  EXPECT_EQ(read(a), bytes);
  return read(y.as_const());
}

TEST(CudaFp8Cast, EveryFp8StorageCodeWidensAndSameTypeCopyKeepsBits) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  const auto codes = all_codes(256);
  for (const auto &source : std::span(ref::formats).subspan(3)) {
    for (const auto &target : std::span(ref::formats).first(3)) {
      const auto actual = direct(context, queue, memory.ref(), source, target, codes);
      exact(actual, codes, source, target);
    }
    EXPECT_EQ(direct(context, queue, memory.ref(), source, source, codes), payload(codes, 1));
  }
}

TEST(CudaFp8Cast, Float32MidpointsNeighborsOverflowInfinitiesAndNanBits) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  for (const auto &target : std::span(ref::formats).subspan(3)) {
    std::vector<std::uint32_t> codes{0, 0x80000000, 1, 0x80000001,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7f800001, 0xff800001, 0x7fc12345, 0xffc12345};
    const unsigned maximum = target.infinity ? 0x7bU : 0x7eU;
    for (unsigned i = 0; i < maximum; ++i) {
      const auto midpoint = static_cast<float>((ref::decode(i, target) + ref::decode(i + 1, target)) / 2);
      for (const float x : {std::nextafter(midpoint, -INFINITY), midpoint, std::nextafter(midpoint, INFINITY)})
        for (const float y : {x, -x}) codes.push_back(std::bit_cast<std::uint32_t>(y));
    }
    for (const float value : {448.F, 449.F, 464.F, 57344.F, 61440.F, 65504.F})
      for (const float signed_value : {value, -value}) codes.push_back(std::bit_cast<std::uint32_t>(signed_value));
    exact(direct(context, queue, memory.ref(), ref::formats[0], target, codes), codes, ref::formats[0], target);
  }
}

TEST(CudaFp8Cast, LowPrecisionPatternsAndCrossFp8UseThePublicStorageOracle) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  const auto low_codes = all_codes(65536), fp8_codes = all_codes(256);
  for (const auto &source : std::span(ref::formats).subspan(1, 2))
    for (const auto &target : std::span(ref::formats).subspan(3))
      exact(direct(context, queue, memory.ref(), source, target, low_codes), low_codes, source, target);
  for (const auto &source : std::span(ref::formats).subspan(3))
    for (const auto &target : std::span(ref::formats).subspan(3))
      if (source.type != target.type)
        exact(direct(context, queue, memory.ref(), source, target, fp8_codes), fp8_codes, source, target);
}

c::ReadLease upload(const c::StorageOwner &owner, const Bytes &bytes) {
  auto buffer = take(m::Buffer::create(owner.ref(), bytes.size(), 256), "owned input buffer");
  check(buffer.view().write(bytes), "owned input upload");
  return take(owner.adopt_buffer(std::move(buffer)), "adopt owned input");
}

TEST(CudaFp8Cast, PublicClipGraphRebindsTwoOwnedSlotsWithRequiredCapture) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  constexpr std::size_t count = 9;
  const auto input = dense(Type::kFloat32, count), limit = dense(Type::kFloat32, 1);
  const auto fp8 = dense(Type::kFloat8E4M3FN, count);
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(input).value(), lo = builder.input(limit).value(), hi = builder.input(limit).value();
  const auto lower = builder.add_node(c::Operation(c::Binary{c::BinaryKind::kMaximum}), std::array{x, lo}).value();
  const auto upper = builder.add_node(c::Operation(c::Binary{c::BinaryKind::kMinimum}), std::array{builder.result(lower).value(), hi}).value();
  const auto narrow = builder.add_node(c::Operation(c::Cast{Type::kFloat8E4M3FN}), std::array{builder.result(upper).value()}).value();
  const auto encoded = builder.result(narrow).value();
  const auto widen = builder.add_node(c::Operation(c::Cast{Type::kFloat32}), std::array{encoded}).value();
  const auto y = builder.result(widen).value();
  check(builder.publish(encoded), "publish encoded storage"); check(builder.publish(y), "publish decoded storage");
  const auto graph = std::move(builder).finish().value();
  auto program = take(g::cuda::compile(graph, context, owner), "public clip/Cast Graph compile");
  std::size_t managed_steps = 0, native_steps = 0;
  for (std::size_t i = 0; i < program.report.step_count(); ++i) {
    native_steps += program.report.step(i)->provider.id == c::cuda_native::kProviderId;
    managed_steps += program.report.step(i)->runtime.replay_class == c::ProviderReplayClass::kManaged;
  }
  EXPECT_GE(native_steps, 2U);
  EXPECT_EQ(managed_steps, 0U);
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array slots{
      program.executable.create_invocation({.resources = g::ResourceOptions{}}).value(),
      program.executable.create_invocation({.resources = g::ResourceOptions{}}).value()};
  const std::array lower_bits{std::bit_cast<std::uint32_t>(-448.F)}, upper_bits{std::bit_cast<std::uint32_t>(448.F)};
  auto low = upload(owner, payload(lower_bits, 4)), high = upload(owner, payload(upper_bits, 4));
  std::array<c::ReadLease, 2> sources;
  std::array<Bytes, 2> first;
  g::GraphOutput old;
  for (unsigned round = 0; round < 2; ++round)
    for (const unsigned lane : {1U, 0U}) {
      if (round) { if (lane == 0) old.reset(); check(slots[lane].reset(), "reset completed Graph episode"); }
      std::vector<std::uint32_t> codes;
      for (const float value : {-INFINITY, -449.F, -1.0625F, -0.F, 0.F, .1F, 1.1875F, 449.F, INFINITY})
        codes.push_back(std::bit_cast<std::uint32_t>(std::isfinite(value) && value != 0
            ? value * (round ? .5F : 1.F) * (lane ? -1.F : 1.F) : value));
      auto fresh = upload(owner, payload(codes, 4));
      if (round) {
        const auto before = take(sources[lane].view(), "old source backing"), after = take(fresh.view(), "fresh source backing");
        EXPECT_TRUE(before.native_handle() != after.native_handle() || before.offset() != after.offset());
      }
      sources[lane] = std::move(fresh);
      check(slots[lane].bind(x, input, sources[lane]), "bind current input permission");
      check(slots[lane].bind(lo, limit, low), "bind lower limit");
      check(slots[lane].bind(hi, limit, high), "bind upper limit");
      if (round == 0 && lane == 1)
        EXPECT_FALSE(g::cuda::prepare_replay(slots[lane], queues[lane], {.mode = g::ReplayMode::kCommands}).ok());
      auto p8 = take(c::LeasePool::create(owner.ref(), count, {.max_blocks = 1, .backing_alignment = 256}), "encoded LeasePool");
      auto p32 = take(c::LeasePool::create(owner.ref(), count * 4, {.max_blocks = 1, .backing_alignment = 256}), "decoded LeasePool");
      check(slots[lane].bind(encoded, fp8, take(p8.allocate(count, 256), "encoded WriteLease")), "bind encoded output");
      check(slots[lane].bind(y, input, take(p32.allocate(count * 4, 256), "decoded WriteLease")), "bind decoded output");
      const auto capture = g::cuda::prepare_replay(slots[lane], queues[lane], {.mode = g::ReplayMode::kRequireCapture});
      check(capture, "native FP8 Cast required capture");
      EXPECT_EQ(slots[lane].replay_route(), g::ReplayRoute::kCaptured);
      auto done = take(slots[lane].submit(queues[lane]), "captured Graph submit"); check(done.wait(), "captured Graph wait");
      auto code_result = take(slots[lane].result(encoded), "encoded GraphOutput");
      auto decoded_result = take(slots[lane].result(y), "decoded GraphOutput");
      const auto code_bytes = read(take(code_result.tensor(), "encoded output tensor"));
      std::vector<std::uint32_t> clipped, encoded_codes;
      for (const auto bits : codes) {
        const float value = std::bit_cast<float>(bits);
        clipped.push_back(std::bit_cast<std::uint32_t>((std::min)(448.F, (std::max)(-448.F, value))));
        encoded_codes.push_back(expected(clipped.back(), ref::formats[0], ref::formats[3]));
      }
      exact(code_bytes, clipped, ref::formats[0], ref::formats[3]);
      exact(read(take(decoded_result.tensor(), "decoded output tensor")), encoded_codes, ref::formats[3], ref::formats[0]);
      EXPECT_TRUE(slots[lane].reset().IsUnavailable()); // A real result(), not the borrowed output() view.
      if (!round) first[lane] = code_bytes;
      else {
        EXPECT_NE(code_bytes, first[lane]);
        if (lane == 1) EXPECT_EQ(read(take(old.tensor(), "other slot retained GraphOutput")), first[0]);
      }
      if (!round && lane == 0) old = std::move(code_result);
    }
}

TEST(CudaFp8Cast, NativeAdmissionRejectsOverlapButHandlesEmptyScalarAndWideShape) {
  const std::array bindings{c::ProviderBinding{c::cuda_native::provider(), 100}};
  auto context = c::Context::create(c::cuda::device(0).value(), {.providers = bindings}).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  const auto wide = c::TensorDesc::dense(Type::kFloat8E4M3FN,
      std::array<std::int64_t, 1>{INT64_C(1) << 33}).value();
  auto large = c::prepare(context, c::Cast{Type::kFloat32}, wide, memory.ref());
  ASSERT_TRUE(large) << large.error().message();
  EXPECT_EQ(large->output_description()->element_count().value(), UINT64_C(1) << 33);
  EXPECT_EQ(large->workspace_requirement().size, 0u);
  // Descriptor-only cold preparation; no giant payload allocation or submit.
  const auto d = dense(Type::kFloat8E4M3FN, 4);
  auto storage = take(m::Buffer::create(memory.ref(), 4, 256), "alias input buffer");
  const auto codes = std::array<std::uint32_t, 4>{0, 0x80, 0x7f, 0xff};
  check(storage.view().write(payload(codes, 1)), "alias input upload");
  auto prepared = take(c::prepare(context, c::Cast{Type::kFloat8E4M3FN}, d, memory.ref()), "same-type Cast prepare");
  auto call = take(prepared.create_invocation(), "same-type Cast invocation");
  EXPECT_FALSE(call.submit(queue, c::ConstTensorView::create(d, storage.view()).value(),
      c::TensorView::create(d, storage.view()).value()));
  EXPECT_EQ(read(c::ConstTensorView::create(d, storage.view()).value()), payload(codes, 1));
  for (const auto &shape : std::array<std::vector<std::int64_t>, 2>{std::vector<std::int64_t>{}, {0}}) {
    const auto source = c::TensorDesc::dense(Type::kFloat8E4M3FN, shape).value();
    auto p = take(c::prepare(context, c::Cast{Type::kFloat32}, source, memory.ref()), "scalar/empty Cast prepare");
    const auto target = p.output_description(0).value();
    auto a = take(m::Buffer::create(memory.ref(), source.required_bytes().value(), 256), "scalar/empty source");
    auto y = take(m::Buffer::create(memory.ref(), target.required_bytes().value(), 256), "scalar/empty output");
    const auto av = c::ConstTensorView::create(source, a.view()).value();
    if (!shape.empty()) EXPECT_EQ(source.element_count().value(), 0U);
    else check(a.view().write(payload(std::array<std::uint32_t, 1>{0x80}, 1)), "scalar signed-zero upload");
    auto invocation = take(p.create_invocation(), "scalar/empty invocation");
    auto done = take(invocation.submit(queue, av, c::TensorView::create(target, y.view()).value()), "scalar/empty submit");
    check(done.wait(), "scalar/empty wait");
    const auto actual = read(c::ConstTensorView::create(target, y.view()).value());
    if (shape.empty()) exact(actual, std::array<std::uint32_t, 1>{0x80}, ref::formats[3], ref::formats[0]);
    else EXPECT_TRUE(actual.empty());
  }
}
TEST(CudaFp8Cast, OddByteStridesAndBroadcastPreserveBitsAndOutputGuards) {
  auto context = c::Context::create(c::cuda::device(0).value(), c::cuda_native::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  constexpr std::size_t count = 258;
  for (const auto &source : ref::formats) for (const auto &target : ref::formats) {
    if (source.width != 1 && target.width != 1) continue;
    for (const bool broadcast : {false, true}) {
      SCOPED_TRACE(::testing::Message() << int(source.type) << '/' << int(target.type) << '/' << broadcast);
      const auto width = std::int64_t(source.width);
      const auto input = c::TensorDesc::strided_bytes(source.type,
          std::array<std::int64_t, 2>{2, 129},
          std::array<std::int64_t, 2>{broadcast ? 0 : 129 * (width + 1) + 3, broadcast ? 0 : width + 1}).value();
      const auto output = c::TensorDesc::dense(target.type, input.dimensions()).value();
      std::vector<std::byte> physical(input.required_bytes().value(), std::byte{0x5a});
      std::vector<std::uint32_t> codes(count);
      for (std::size_t i = 0; i < count; ++i) {
        const auto code = broadcast ? 0x80U : static_cast<std::uint32_t>(i & (source.width == 1 ? 255 : 65535));
        codes[i] = source.width == 4 ? std::bit_cast<std::uint32_t>((float(code) - 128) * .03125F) : code;
        const auto offset = broadcast ? 0 : (i / 129) * std::size_t(input.byte_strides()[0]) + (i % 129) * std::size_t(input.byte_strides()[1]);
        std::memcpy(physical.data() + offset, &codes[i], source.width);
      }
      auto a = take(m::Buffer::create(memory.ref(), physical.size(), 256), "strided source");
      check(a.view().write(physical), "strided upload");
      auto y = take(m::Buffer::create(memory.ref(), count * target.width + 32, 256), "guarded output");
      Bytes actual(y.size(), std::byte{0xa5}); check(y.view().write(actual), "guard initialization");
      auto prepared = take(c::prepare(context, c::Cast{target.type}, input, memory.ref()), "strided FP8 Cast");
      EXPECT_TRUE(prepared.capture_compatible()); EXPECT_EQ(prepared.workspace_requirement().size, 0U);
      const auto result = c::TensorView::create(output, y.view().subview(16, count * target.width).value()).value();
      auto invocation = take(prepared.create_invocation(), "strided invocation");
      auto done = take(invocation.submit(queue, c::ConstTensorView::create(input, a.view()).value(), result), "strided submit");
      check(done.wait(), "strided wait"); check(y.view().read(actual), "guarded read");
      const auto code_bytes = Bytes(actual.begin() + 16, actual.end() - 16);
      exact(code_bytes, codes, source, target);
      for (std::size_t i = 0; i < 16; ++i) { EXPECT_EQ(actual[i], std::byte{0xa5}); EXPECT_EQ(actual[actual.size()-1-i], std::byte{0xa5}); }
      Bytes after(physical.size()); check(a.view().read(after), "source unchanged read"); EXPECT_EQ(after, physical);
    }
  }
}
TEST(CudaFp8Cast, BoundedGridStrideWritesDenseAndBroadcastTails) {
  auto context = c::Context::create(c::cuda::device(0).value(), c::cuda_native::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  constexpr std::size_t count = (std::size_t{1} << 24) + 257;
  const auto output = dense(Type::kFloat8E5M2, count);
  const auto code = std::byte(expected(0xb8, ref::formats[3], ref::formats[4]));
  ASSERT_EQ(code, std::byte{0xbc}); // Exact -1 in the two public FP8 formats.
  for (const bool broadcast : {false, true}) {
    SCOPED_TRACE(broadcast);
    const auto input = c::TensorDesc::strided_bytes(Type::kFloat8E4M3FN,
        output.dimensions(), std::array<std::int64_t, 1>{broadcast ? 0 : 1}).value();
    auto a = take(m::Buffer::create(memory.ref(), input.required_bytes().value(), 256), "grid source");
    check(a.view().write(Bytes(a.size(), std::byte{0xb8})), "grid source upload");
    auto y = take(m::Buffer::create(memory.ref(), count + 32, 256), "grid guarded output");
    check(y.view().subview(0, 80).value().write(Bytes(80, std::byte{0xa5})), "grid prefix poison");
    check(y.view().subview(16 + count - 64, 80).value().write(Bytes(80, std::byte{0xa5})), "grid tail poison");
    auto program = take(c::prepare(context, c::Cast{Type::kFloat8E5M2}, input, memory.ref()), "grid cast prepare");
    auto invocation = take(program.create_invocation(), "grid cast invocation");
    auto done = take(invocation.submit(queue, c::ConstTensorView::create(input, a.view()).value(),
        c::TensorView::create(output, y.view().subview(16, count).value()).value()), "grid cast submit");
    check(done.wait(), "grid cast wait");
    Bytes prefix(80), tail(80), middle(64);
    check(y.view().subview(0, 80).value().read(prefix), "grid prefix read");
    check(y.view().subview(16 + count - 64, 80).value().read(tail), "grid tail read");
    check(y.view().subview(16 + count / 2, 64).value().read(middle), "grid middle read");
    for (std::size_t i = 0; i < 80; ++i) {
      EXPECT_EQ(prefix[i], i < 16 ? std::byte{0xa5} : code);
      EXPECT_EQ(tail[i], i < 64 ? code : std::byte{0xa5});
    }
    EXPECT_EQ(middle, Bytes(64, code));
  }
}
} // namespace
