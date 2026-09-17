#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"

namespace {
using namespace hyle::compute;
namespace mem = hyle::mem;
struct Tensor final {
  mem::Buffer storage;
  TensorView view;
  Tensor(const TensorDesc &desc, mem::MemoryResourceRef resource)
      : storage(mem::Buffer::create(resource,
                                    desc.required_bytes().value() + 16, 256)
                    .value()),
        view(TensorView::create(desc,
                                storage.view()
                                    .subview(0, desc.required_bytes().value())
                                    .value())
                 .value()) {
    EXPECT_TRUE(storage.view().fill(std::byte{0xa5}).ok());
  }
  template <class T> hyle::Status write(std::span<const T> values) {
    const auto &desc = view.description();
    std::vector<std::byte> bytes(desc.required_bytes().value(),
                                 std::byte{0x5a});
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::size_t remaining = i, offset = 0;
      for (std::size_t axis = desc.rank(); axis-- > 0;) {
        offset +=
            (remaining % desc.dimensions()[axis]) * desc.byte_strides()[axis];
        remaining /= desc.dimensions()[axis];
      }
      std::memcpy(bytes.data() + offset, &values[i], sizeof(T));
    }
    return view.storage().write(bytes);
  }
  template <class T> std::vector<T> read() const {
    std::vector<T> result(view.storage().size() / sizeof(T));
    EXPECT_TRUE(
        view.storage().read(std::as_writable_bytes(std::span(result))).ok());
    return result;
  }
};
struct Fixture final {
  Context owner =
      Context::create(cuda::device(0).value(), cuda_native::provider()).value();
  cuda::MemoryResource resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  Queue queue = owner.create_queue().value();
};
TensorDesc description(DataType type, std::span<const std::int64_t> dimensions,
                       bool strided) {
  if (!strided)
    return TensorDesc::dense(type, dimensions).value();
  std::vector<std::int64_t> strides(dimensions.size());
  std::int64_t step = static_cast<std::int64_t>(data_type_size(type)) + 1;
  for (std::size_t axis = dimensions.size(); axis-- > 0;) {
    strides[axis] = step;
    step *= std::max<std::int64_t>(1, dimensions[axis]);
  }
  return TensorDesc::strided_bytes(type, dimensions, strides).value();
}
mem::Buffer workspace(const PreparedOperation &program,
                      mem::MemoryResourceRef resource) {
  const auto requirement = program.workspace_requirement();
  if (requirement.size == 0)
    return {};
  return mem::Buffer::create(resource, requirement.size, requirement.alignment)
      .value();
}
template <class Bits, class Index>
void assign_case(Fixture &fixture, DataType type,
                 std::span<const std::int64_t> shape,
                 std::int32_t requested_axis,
                 std::span<const std::int64_t> index_shape, bool strided,
                 bool in_place = false) {
  SCOPED_TRACE(::testing::Message()
               << static_cast<int>(type) << '/' << requested_axis << '/'
               << strided << '/' << in_place);
  const auto axis = static_cast<std::size_t>(
      requested_axis < 0 ? requested_axis + static_cast<int>(shape.size())
                         : requested_axis);
  const auto source_desc = description(type, shape, strided && !in_place);
  const auto index_desc =
      description(sizeof(Index) == 8 ? DataType::kInt64 : DataType::kInt32,
                  index_shape, strided);
  std::array<TensorDesc, 1> update_shape;
  ASSERT_TRUE(infer_output_descriptions(
                  Operation(Gather{.axis = requested_axis}),
                  std::array{source_desc, index_desc}, update_shape)
                  .ok());
  const auto update_desc =
      description(type, update_shape[0].dimensions(), strided);
  auto program =
      prepare(fixture.owner, Scatter{.axis = requested_axis}, source_desc,
              index_desc, update_desc, fixture.resource.ref());
  ASSERT_TRUE(program.has_value()) << program.error().message();
  const auto count = index_desc.element_count().value();
  EXPECT_EQ(program->workspace_requirement().size == 0, count <= 1);
  if (count > 1) {
    EXPECT_GE(program->workspace_requirement().size, count * 16);
    EXPECT_FALSE(program->invocation_private_bytes().has_value());
  }
  Tensor source(source_desc, fixture.resource.ref()),
      ids(index_desc, fixture.resource.ref()),
      updates(update_desc, fixture.resource.ref()),
      output(program->output_description(0).value(), fixture.resource.ref());
  std::vector<Bits> values(source_desc.element_count().value()),
      changes(update_desc.element_count().value());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<Bits>(0x7fc000018123ULL + i * 0x9e3779b1ULL);
  for (std::size_t i = 0; i < changes.size(); ++i)
    changes[i] = static_cast<Bits>(0x800000004321ULL + i * 0x7f4a7c15ULL);
  std::vector<Index> indices(count);
  for (std::size_t i = 0; i < count; ++i)
    indices[i] = static_cast<Index>((i * 3 + shape[axis] - 1) % shape[axis]);
  ASSERT_TRUE(source.write<Bits>(values).ok());
  ASSERT_TRUE(ids.write<Index>(indices).ok());
  ASSERT_TRUE(updates.write<Bits>(changes).ok());
  auto scratch = workspace(*program, fixture.resource.ref());
  auto slot = program->create_invocation(scratch.view()).value();
  std::array inputs{source.view.as_const(), ids.view.as_const(),
                    updates.view.as_const()};
  const std::array outputs{in_place ? source.view : output.view};
  auto done = slot.submit(fixture.queue, inputs, outputs);
  ASSERT_TRUE(done.has_value()) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  std::size_t outer = 1, inner = 1;
  for (std::size_t dim = 0; dim < axis; ++dim)
    outer *= shape[dim];
  for (std::size_t dim = axis + 1; dim < shape.size(); ++dim)
    inner *= shape[dim];
  auto expected = values;
  for (std::size_t prefix = 0; prefix < outer; ++prefix)
    for (std::size_t i = 0; i < indices.size(); ++i)
      for (std::size_t suffix = 0; suffix < inner; ++suffix)
        expected[(prefix * shape[axis] + indices[i]) * inner + suffix] =
            changes[(prefix * indices.size() + i) * inner + suffix];
  EXPECT_EQ((in_place ? source : output).template read<Bits>(), expected);
  std::array<std::byte, 16> guard{};
  const auto &result = in_place ? source : output;
  ASSERT_TRUE(result.storage.read(result.storage.size() - 16, guard).ok());
  for (auto byte : guard)
    EXPECT_EQ(byte, std::byte{0xa5});
}
template <class Bits> void assign_types(Fixture &fixture, DataType type) {
  for (std::int64_t count : {0, 1, 17, 129}) {
    assign_case<Bits, std::int32_t>(fixture, type,
                                    std::array<std::int64_t, 2>{7, 129}, 0,
                                    std::array{count}, false);
    assign_case<Bits, std::int64_t>(fixture, type,
                                    std::array<std::int64_t, 3>{2, 7, 3}, -2,
                                    std::array{count}, true);
  }
  assign_case<Bits, std::int64_t>(
      fixture, type, std::array<std::int64_t, 2>{7, 128}, 0,
      std::array<std::int64_t, 2>{2, 3}, false, true);
  assign_case<Bits, std::int32_t>(fixture, type,
                                  std::array<std::int64_t, 2>{7, 2048}, 0,
                                  std::span<const std::int64_t>{}, true, true);
}
TEST(CudaScatter, AssignPreservesBitsAxesStridesLastWriterAndExactAlias) {
  Fixture fixture;
  assign_types<std::uint8_t>(fixture, DataType::kUInt8);
  assign_types<std::uint16_t>(fixture, DataType::kFloat16);
  assign_types<std::uint16_t>(fixture, DataType::kBFloat16);
  assign_types<std::uint32_t>(fixture, DataType::kFloat32);
  assign_types<std::uint64_t>(fixture, DataType::kInt64);
  assign_case<std::uint32_t, std::int64_t>(
      fixture, DataType::kInt32,
      std::array<std::int64_t, 8>{2, 1, 2, 3, 1, 2, 1, 2}, 3,
      std::array<std::int64_t, 1>{257}, true);
  assign_case<std::uint32_t, std::int64_t>(
      fixture, DataType::kInt32, std::array<std::int64_t, 2>{1031, 3}, 0,
      std::array<std::int64_t, 1>{4097}, false);
}
template <class Value>
void add_case(Fixture &fixture, std::int64_t count, std::int64_t width,
              bool strided, bool in_place) {
  SCOPED_TRACE(::testing::Message()
               << static_cast<int>(half_test::data_type<Value>()) << '/'
               << count << '/' << width << '/' << in_place);
  const auto source_desc = description(half_test::data_type<Value>(),
                                       std::array<std::int64_t, 3>{2, 7, width},
                                       strided && !in_place);
  const auto index_desc =
      description(DataType::kInt64, std::array{count}, strided);
  const auto update_desc =
      description(half_test::data_type<Value>(),
                  std::array<std::int64_t, 3>{2, count, width}, strided);
  auto program =
      prepare(fixture.owner, Scatter{.axis = 1, .kind = ScatterKind::kAdd},
              source_desc, index_desc, update_desc, fixture.resource.ref())
          .value();
  Tensor source(source_desc, fixture.resource.ref()),
      ids(index_desc, fixture.resource.ref()),
      updates(update_desc, fixture.resource.ref()),
      output(program.output_description(0).value(), fixture.resource.ref());
  std::vector<Value> values(source_desc.element_count().value()),
      changes(update_desc.element_count().value());
  const double large = std::is_same_v<Value, hyle::simd::bfloat16_t> ? 256
                       : std::is_same_v<Value, hyle::simd::float16_t>
                           ? 2048
                           : 16777216;
  std::fill(values.begin(), values.end(), half_test::encode<Value>(large));
  std::vector<std::int64_t> indices(static_cast<std::size_t>(count), 3);
  for (std::size_t p = 0; p < 2; ++p)
    for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i)
      for (std::size_t c = 0; c < static_cast<std::size_t>(width); ++c) {
        double value = i % 3 == 0 ? 1 : i % 3 == 1 ? -large : large;
        if (c == 1)
          value = i % 2 == 0 ? std::numeric_limits<double>::infinity()
                             : -std::numeric_limits<double>::infinity();
        if (c == 2)
          value = i == 0 ? std::numeric_limits<double>::quiet_NaN() : 1;
        changes[(p * count + i) * width + c] = half_test::encode<Value>(value);
      }
  auto expected = values;
  for (std::size_t p = 0; p < 2; ++p)
    for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i)
      for (std::size_t c = 0; c < static_cast<std::size_t>(width); ++c) {
        const auto destination = (p * 7 + indices[i]) * width + c;
        const float sum =
            static_cast<float>(half_test::decode(expected[destination])) +
            static_cast<float>(
                half_test::decode(changes[(p * count + i) * width + c]));
        expected[destination] = half_test::encode<Value>(sum);
      }
  ASSERT_TRUE(source.write<Value>(values).ok());
  ASSERT_TRUE(ids.write<std::int64_t>(indices).ok());
  ASSERT_TRUE(updates.write<Value>(changes).ok());
  auto scratch = workspace(program, fixture.resource.ref());
  auto slot = program.create_invocation(scratch.view()).value();
  auto done =
      slot.submit(fixture.queue,
                  std::array{source.view.as_const(), ids.view.as_const(),
                             updates.view.as_const()},
                  std::array{in_place ? source.view : output.view})
          .value();
  ASSERT_TRUE(done.wait().ok());
  const auto actual = (in_place ? source : output).template read<Value>();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::isnan(half_test::decode(expected[i])))
      ASSERT_TRUE(std::isnan(half_test::decode(actual[i]))) << i;
    else if constexpr (std::is_same_v<Value, float>)
      ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                std::bit_cast<std::uint32_t>(expected[i]))
          << i;
    else
      ASSERT_EQ(actual[i].bits, expected[i].bits) << i;
  }
}
TEST(CudaScatter, AddPreservesEveryRoundingStepAndStableSkewedUpdateOrder) {
  Fixture fixture;
  for (auto count : {1, 17, 128, 129, 4097}) {
    const auto width = count == 4097 ? 3 : count == 128 ? 2048 : 129;
    for (bool in_place : {false, true}) {
      add_case<float>(fixture, count, width, true, in_place);
      add_case<hyle::simd::float16_t>(fixture, count, width, true, in_place);
      add_case<hyle::simd::bfloat16_t>(fixture, count, width, true, in_place);
    }
  }
}
TEST(CudaScatter, CheckedFailurePreservesAllOutputAndFreshSortSlotRecovers) {
  Fixture fixture;
  const auto source_desc = description(
      DataType::kBFloat16, std::array<std::int64_t, 2>{7, 128}, false);
  for (std::int64_t count : {1, 257}) {
    const auto index_desc =
        description(DataType::kInt64, std::array{count}, true);
    const auto update_desc = description(
        DataType::kBFloat16, std::array<std::int64_t, 2>{count, 128}, false);
    auto program =
        prepare(fixture.owner, Scatter{.kind = ScatterKind::kAdd}, source_desc,
                index_desc, update_desc, fixture.resource.ref())
            .value();
    Tensor source(source_desc, fixture.resource.ref()),
        ids(index_desc, fixture.resource.ref()),
        updates(update_desc, fixture.resource.ref()),
        output(source_desc, fixture.resource.ref());
    ASSERT_TRUE(
        source.write<std::uint16_t>(std::vector<std::uint16_t>(7 * 128, 0))
            .ok());
    ASSERT_TRUE(updates
                    .write<std::uint16_t>(
                        std::vector<std::uint16_t>(count * 128, 0x3f80))
                    .ok());
    const std::array inputs{source.view.as_const(), ids.view.as_const(),
                            updates.view.as_const()};
    auto scratch = workspace(program, fixture.resource.ref());
    for (auto bad : {std::int64_t{-1}, std::int64_t{7}, INT64_MIN, INT64_MAX}) {
      std::vector<std::int64_t> indices(static_cast<std::size_t>(count), 0);
      indices.back() = bad;
      ASSERT_TRUE(ids.write<std::int64_t>(indices).ok());
      ASSERT_TRUE(output.view.storage().fill(std::byte{0xa5}).ok());
      auto slot = program.create_invocation(scratch.view()).value();
      auto done = slot.submit(fixture.queue, inputs, std::array{output.view});
      ASSERT_TRUE(done.has_value());
      EXPECT_TRUE(done->wait().IsOutOfRange());
      EXPECT_EQ(output.read<std::uint16_t>(),
                std::vector<std::uint16_t>(7 * 128, 0xa5a5));
      auto retired =
          slot.submit(fixture.queue, inputs, std::array{output.view});
      ASSERT_FALSE(retired.has_value());
      EXPECT_TRUE(retired.error().IsFailedPrecondition());
      auto fresh = program.create_invocation(scratch.view()).value();
      std::fill(indices.begin(), indices.end(), 6);
      ASSERT_TRUE(ids.write<std::int64_t>(indices).ok());
      auto good =
          fresh.submit(fixture.queue, inputs, std::array{output.view}).value();
      ASSERT_TRUE(good.wait().ok());
      const auto actual = output.read<std::uint16_t>();
      for (std::size_t i = 0; i < 6 * 128; ++i)
        EXPECT_EQ(actual[i], 0U);
      EXPECT_EQ(actual.back(), half_test::encode<hyle::simd::bfloat16_t>(
                                   count == 257 ? 256 : 1)
                                   .bits);
    }
  }
}
TEST(CudaScatter, ZeroStrideUncheckedEmptyWorkspaceAndAliasBoundaries) {
  Fixture fixture;
  const auto sd = TensorDesc::strided_bytes(DataType::kInt32,
                                            std::array<std::int64_t, 2>{7, 3},
                                            std::array<std::int64_t, 2>{0, 4})
                      .value();
  const auto id = TensorDesc::strided_bytes(DataType::kInt64,
                                            std::array<std::int64_t, 1>{17},
                                            std::array<std::int64_t, 1>{0})
                      .value();
  const auto ud = TensorDesc::strided_bytes(DataType::kInt32,
                                            std::array<std::int64_t, 2>{17, 3},
                                            std::array<std::int64_t, 2>{0, 4})
                      .value();
  auto program =
      prepare(fixture.owner, Scatter{.bounds = IndexBoundsPolicy::kUnchecked},
              sd, id, ud, fixture.resource.ref())
          .value();
  Tensor source(sd, fixture.resource.ref()), ids(id, fixture.resource.ref()),
      updates(ud, fixture.resource.ref()),
      output(program.output_description(0).value(), fixture.resource.ref());
  ASSERT_TRUE(source.write<int>(std::array{1, 2, 3}).ok());
  ASSERT_TRUE(ids.write<std::int64_t>(std::array<std::int64_t, 1>{6}).ok());
  ASSERT_TRUE(updates.write<int>(std::array{9, 8, 7}).ok());
  ASSERT_FALSE(program.create_invocation().has_value());
  auto scratch = workspace(program, fixture.resource.ref());
  auto too_small = program.create_invocation(
      scratch.view().subview(0, scratch.size() - 1).value());
  ASSERT_FALSE(too_small.has_value());
  auto host =
      mem::Buffer::create(mem::system_resource(), scratch.size(), 256).value();
  ASSERT_FALSE(program.create_invocation(host.view()).has_value());
  auto slot = program.create_invocation(scratch.view()).value();
  auto done =
      slot.submit(fixture.queue,
                  std::array{source.view.as_const(), ids.view.as_const(),
                             updates.view.as_const()},
                  std::array{output.view})
          .value();
  ASSERT_TRUE(done.wait().ok());
  std::vector<int> expected;
  for (int i = 0; i < 6; ++i)
    expected.insert(expected.end(), {1, 2, 3});
  expected.insert(expected.end(), {9, 8, 7});
  EXPECT_EQ(output.read<int>(), expected);
  const auto dense =
      description(DataType::kInt32, std::array<std::int64_t, 1>{7}, false);
  const auto did =
      description(DataType::kInt32, std::array<std::int64_t, 1>{7}, false);
  auto aliases = prepare(fixture.owner, Scatter{}, dense, did, dense,
                         fixture.resource.ref())
                     .value();
  Tensor a(dense, fixture.resource.ref()), b(did, fixture.resource.ref()),
      c(dense, fixture.resource.ref());
  ASSERT_TRUE(a.write<int>(std::array{1, 2, 3, 4, 5, 6, 7}).ok());
  ASSERT_TRUE(b.write<int>(std::array{0, 1, 2, 3, 4, 5, 6}).ok());
  ASSERT_TRUE(c.write<int>(std::array{7, 6, 5, 4, 3, 2, 1}).ok());
  auto ascratch = workspace(aliases, fixture.resource.ref());
  auto healthy = aliases.create_invocation(ascratch.view()).value();
  const std::array ain{a.view.as_const(), b.view.as_const(), c.view.as_const()};
  EXPECT_FALSE(
      healthy.submit(fixture.queue, ain, std::array{b.view}).has_value());
  EXPECT_FALSE(
      healthy.submit(fixture.queue, ain, std::array{c.view}).has_value());
  auto accepted =
      healthy.submit(fixture.queue, ain, std::array{a.view}).value();
  ASSERT_TRUE(accepted.wait().ok());
  EXPECT_EQ(a.read<int>(), (std::vector{7, 6, 5, 4, 3, 2, 1}));
  const auto empty =
      description(DataType::kInt32, std::array<std::int64_t, 1>{0}, false);
  auto empty_program =
      prepare(fixture.owner, Scatter{}, empty, did, did, fixture.resource.ref())
          .value();
  Tensor e(empty, fixture.resource.ref());
  auto empty_slot = empty_program.create_invocation().value();
  auto empty_done = empty_slot
                        .submit(fixture.queue,
                                std::array{e.view.as_const(), b.view.as_const(),
                                           c.view.as_const()},
                                std::array{e.view})
                        .value();
  ASSERT_TRUE(empty_done.wait().ok());
}
} // namespace
