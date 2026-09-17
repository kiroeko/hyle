#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"

namespace {
using namespace hyle::compute;
namespace mem = hyle::mem;
struct Tensor final {
  mem::Buffer storage;
  TensorView view;
  Tensor(const TensorDesc &desc, mem::MemoryResourceRef resource,
         std::size_t offset = 0)
      : storage(mem::Buffer::create(
                    resource, desc.required_bytes().value() + offset + 16, 256)
                    .value()),
        view(TensorView::create(
                 desc, storage.view()
                           .subview(offset, desc.required_bytes().value())
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
    step *= dimensions[axis];
  }
  return TensorDesc::strided_bytes(type, dimensions, strides).value();
}
template <class Bits, class Index>
void bits_case(Fixture &fixture, DataType type,
               std::span<const std::int64_t> shape, std::int32_t requested_axis,
               std::span<const std::int64_t> index_shape, bool source_stride,
               bool index_stride, bool unaligned = false) {
  const auto axis = static_cast<std::size_t>(
      requested_axis < 0 ? requested_axis + static_cast<int>(shape.size())
                         : requested_axis);
  const auto desc = description(type, shape, source_stride);
  const auto idesc =
      description(sizeof(Index) == 8 ? DataType::kInt64 : DataType::kInt32,
                  index_shape, index_stride);
  std::vector<Bits> values(desc.element_count().value());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<Bits>(0x7fc000018123ULL + i * 0x9e3779b1ULL);
  std::vector<Index> ids(idesc.element_count().value());
  for (std::size_t i = 0; i < ids.size(); ++i)
    ids[i] = static_cast<Index>((i * 3 + 1) % shape[axis]);
  const auto offset = unaligned ? sizeof(Bits) : 0;
  Tensor source(desc, fixture.resource.ref(), offset),
      indices(idesc, fixture.resource.ref());
  auto prepared = prepare(fixture.owner, Gather{.axis = requested_axis}, desc,
                          idesc, fixture.resource.ref());
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_EQ(prepared->workspace_requirement().size, 0U);
  EXPECT_EQ(prepared->invocation_private_bytes(), 0U);
  EXPECT_TRUE(prepared->capture_compatible());
  Tensor output(prepared->output_description(0).value(), fixture.resource.ref(),
                offset);
  ASSERT_TRUE(source.write<Bits>(values).ok());
  ASSERT_TRUE(indices.write<Index>(ids).ok());
  auto slot = prepared->create_invocation().value();
  auto done = slot.submit(fixture.queue, source.view.as_const(),
                          indices.view.as_const(), output.view);
  ASSERT_TRUE(done.has_value()) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  std::size_t outer = 1, inner = 1;
  for (std::size_t dim = 0; dim < axis; ++dim)
    outer *= shape[dim];
  for (std::size_t dim = axis + 1; dim < shape.size(); ++dim)
    inner *= shape[dim];
  std::vector<Bits> expected;
  for (std::size_t prefix = 0; prefix < outer; ++prefix)
    for (auto id : ids)
      for (std::size_t suffix = 0; suffix < inner; ++suffix)
        expected.push_back(
            values[(prefix * shape[axis] + id) * inner + suffix]);
  EXPECT_EQ(output.read<Bits>(), expected);
  std::array<std::byte, 16> suffix_guard{};
  ASSERT_TRUE(
      output.storage.read(output.storage.size() - 16, suffix_guard).ok());
  for (auto byte : suffix_guard)
    EXPECT_EQ(byte, std::byte{0xa5});
  if (offset != 0) {
    std::vector<std::byte> prefix_guard(offset);
    ASSERT_TRUE(output.storage.read(0, prefix_guard).ok());
    for (auto byte : prefix_guard)
      EXPECT_EQ(byte, std::byte{0xa5});
  }
}
template <class Bits> void type_cases(Fixture &fixture, DataType type) {
  for (std::int64_t width : {1, 128, 129, 2048}) {
    for (bool source_stride : {false, true}) {
      for (bool index_stride : {false, true}) {
        const std::array<std::int64_t, 2> shape{7, width}, ids{2, 3};
        SCOPED_TRACE(::testing::Message()
                     << static_cast<int>(type) << '/' << width << '/'
                     << source_stride << '/' << index_stride);
        bits_case<Bits, std::int32_t>(fixture, type, shape, 0, ids,
                                      source_stride, index_stride);
        bits_case<Bits, std::int64_t>(fixture, type, shape, -2, ids,
                                      source_stride, index_stride);
      }
    }
  }
  for (std::int64_t width : {128, 129})
    bits_case<Bits, std::int64_t>(
        fixture, type, std::array<std::int64_t, 2>{7, width}, 0,
        std::array<std::int64_t, 1>{3}, false, false, true);
  bits_case<Bits, std::int64_t>(fixture, type,
                                std::array<std::int64_t, 3>{2, 7, 3}, 1,
                                std::span<const std::int64_t>{}, true, false);
  bits_case<Bits, std::int32_t>(
      fixture, type, std::array<std::int64_t, 8>{2, 1, 3, 1, 1, 1, 2, 3}, -6,
      std::array<std::int64_t, 1>{4}, true, true);
}
TEST(CudaGather,
     ExactBitsForScalarTypesIndexWidthsAxesStridesAndUnalignedVectors) {
  Fixture fixture;
  type_cases<std::uint8_t>(fixture, DataType::kUInt8);
  type_cases<std::uint16_t>(fixture, DataType::kFloat16);
  type_cases<std::uint16_t>(fixture, DataType::kBFloat16);
  type_cases<std::uint32_t>(fixture, DataType::kFloat32);
  type_cases<std::uint64_t>(fixture, DataType::kInt64);
}
TEST(CudaGather, LargeInt64IndicesAndZeroStridesKeepFullAddressDomain) {
  Fixture fixture;
  constexpr std::int64_t extent = 1LL << 40;
  const auto desc =
      TensorDesc::strided_bytes(DataType::kInt32,
                                std::array<std::int64_t, 2>{extent, 3},
                                std::array<std::int64_t, 2>{0, 4})
          .value();
  const auto idesc = TensorDesc::strided_bytes(
                         DataType::kInt64, std::array<std::int64_t, 2>{2, 3},
                         std::array<std::int64_t, 2>{0, 9})
                         .value();
  Tensor source(desc, fixture.resource.ref()),
      indices(idesc, fixture.resource.ref());
  const std::array values{11, -37, INT_MAX};
  ASSERT_TRUE(
      source.view.storage().write(std::as_bytes(std::span(values))).ok());
  const std::array<std::int64_t, 6> ids{INT32_MAX + 13LL, 0, extent - 1,
                                        INT32_MAX + 13LL, 0, extent - 1};
  ASSERT_TRUE(indices.write<std::int64_t>(ids).ok());
  for (auto policy :
       {IndexBoundsPolicy::kChecked, IndexBoundsPolicy::kUnchecked}) {
    auto program = prepare(fixture.owner, Gather{.bounds = policy}, desc, idesc,
                           fixture.resource.ref())
                       .value();
    Tensor output(program.output_description(0).value(),
                  fixture.resource.ref());
    auto slot = program.create_invocation().value();
    auto done = slot.submit(fixture.queue, source.view.as_const(),
                            indices.view.as_const(), output.view);
    ASSERT_TRUE(done.has_value());
    ASSERT_TRUE(done->wait().ok());
    const auto actual = output.read<int>();
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], values[i % 3]);
  }
}
template <class Index> void invalid_cases() {
  Fixture fixture;
  const auto desc = TensorDesc::dense(DataType::kBFloat16,
                                      std::array<std::int64_t, 2>{7, 128})
                        .value();
  const auto idesc =
      description(sizeof(Index) == 8 ? DataType::kInt64 : DataType::kInt32,
                  std::array<std::int64_t, 1>{257}, true);
  Tensor source(desc, fixture.resource.ref()),
      indices(idesc, fixture.resource.ref());
  const std::vector<std::uint16_t> values(7 * 128, 0x3f80);
  ASSERT_TRUE(source.write<std::uint16_t>(values).ok());
  auto program =
      prepare(fixture.owner, Gather{}, desc, idesc, fixture.resource.ref())
          .value();
  Tensor output(program.output_description(0).value(), fixture.resource.ref());
  for (Index bad : {Index{-1}, Index{7}, std::numeric_limits<Index>::min(),
                    std::numeric_limits<Index>::max()}) {
    std::vector<Index> ids(257, 3);
    ids.back() = bad;
    ASSERT_TRUE(indices.write<Index>(ids).ok());
    ASSERT_TRUE(output.view.storage().fill(std::byte{0xa5}).ok());
    auto slot = program.create_invocation().value();
    auto done = slot.submit(fixture.queue, source.view.as_const(),
                            indices.view.as_const(), output.view);
    ASSERT_TRUE(done.has_value()) << done.error().message();
    EXPECT_TRUE(done->wait().IsOutOfRange());
    const auto unchanged = output.read<std::uint16_t>();
    for (auto value : unchanged)
      ASSERT_EQ(value, 0xa5a5);
    ids.back() = 0;
    ASSERT_TRUE(indices.write<Index>(ids).ok());
    const auto retired = slot.submit(fixture.queue, source.view.as_const(),
                                     indices.view.as_const(), output.view);
    ASSERT_FALSE(retired.has_value());
    EXPECT_TRUE(retired.error().IsFailedPrecondition());
    auto fresh = program.create_invocation().value();
    auto recovered = fresh.submit(fixture.queue, source.view.as_const(),
                                  indices.view.as_const(), output.view);
    ASSERT_TRUE(recovered.has_value());
    ASSERT_TRUE(recovered->wait().ok());
    for (auto value : output.read<std::uint16_t>())
      ASSERT_EQ(value, 0x3f80);
    ASSERT_TRUE(fixture.owner.is_open());
  }
}
TEST(CudaGather, CheckedFailuresWriteNothingRetireOnlyFailedSlotAndRecover) {
  invalid_cases<std::int32_t>();
  invalid_cases<std::int64_t>();
}
TEST(CudaGather, EmptyOutputSkipsPayloadAndImpossibleAxisOrOverlapsReject) {
  Fixture fixture;
  const auto empty =
      TensorDesc::strided_bytes(DataType::kFloat32,
                                std::array<std::int64_t, 2>{INT64_MAX, 0},
                                std::array<std::int64_t, 2>{0, 4})
          .value();
  const auto idesc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{4})
          .value();
  Tensor indices(idesc, fixture.resource.ref());
  ASSERT_TRUE(indices.write<int>(std::array{-1, INT_MIN, INT_MAX, 7}).ok());
  auto empty_program =
      prepare(fixture.owner, Gather{}, empty, idesc, fixture.resource.ref())
          .value();
  Tensor source_empty(empty, fixture.resource.ref()),
      empty_output(empty_program.output_description(0).value(),
                   fixture.resource.ref());
  auto no_op = empty_program.create_invocation().value();
  auto done = no_op.submit(fixture.queue, source_empty.view.as_const(),
                           indices.view.as_const(), empty_output.view);
  ASSERT_TRUE(done.has_value());
  ASSERT_TRUE(done->wait().ok());
  const auto impossible =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{0, 3})
          .value();
  const auto invalid = prepare(fixture.owner, Gather{}, impossible, idesc,
                               fixture.resource.ref());
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsOutOfRange());
  const auto desc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{4})
          .value();
  Tensor source(desc, fixture.resource.ref()),
      output(desc, fixture.resource.ref());
  ASSERT_TRUE(source.write<int>(std::array{5, 6, 7, 8}).ok());
  ASSERT_TRUE(indices.write<int>(std::array{3, 2, 1, 0}).ok());
  auto program =
      prepare(fixture.owner, Gather{}, desc, idesc, fixture.resource.ref())
          .value();
  auto slot = program.create_invocation().value();
  for (auto alias : {source.view, indices.view}) {
    const auto rejected = slot.submit(fixture.queue, source.view.as_const(),
                                      indices.view.as_const(), alias);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_TRUE(rejected.error().IsInvalidArgument());
  }
  done = slot.submit(fixture.queue, source.view.as_const(),
                     indices.view.as_const(), output.view);
  ASSERT_TRUE(done.has_value());
  ASSERT_TRUE(done->wait().ok());
  EXPECT_EQ(output.read<int>(), (std::vector{8, 7, 6, 5}));
}
} // namespace
