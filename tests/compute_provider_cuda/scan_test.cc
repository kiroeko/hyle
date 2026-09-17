#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
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
template <class T>
constexpr DataType type =
    std::is_same_v<T, float> ? DataType::kFloat32 : DataType::kInt32;
TensorDesc description(DataType type, std::span<const std::int64_t> shape,
                       std::size_t axis, int mode) {
  if (mode == 0)
    return TensorDesc::dense(type, shape).value();
  std::vector<std::int64_t> strides(shape.size());
  std::int64_t step = 5;
  for (std::size_t dim = shape.size(); dim-- > 0;) {
    strides[dim] = mode == 2 && dim == axis ? 0 : step;
    step *= std::max<std::int64_t>(shape[dim], 1);
  }
  return TensorDesc::strided_bytes(type, shape, strides).value();
}
template <class T>
void numeric_case(Fixture &fixture, std::span<const std::int64_t> shape,
                  std::int32_t requested_axis, int mode) {
  const auto axis = static_cast<std::size_t>(
      requested_axis < 0 ? requested_axis + static_cast<int>(shape.size())
                         : requested_axis);
  SCOPED_TRACE(::testing::Message()
               << static_cast<int>(type<T>) << '/' << shape[axis] << '/'
               << requested_axis << '/' << mode);
  const auto input_desc = description(type<T>, shape, axis, mode);
  const auto output_desc = TensorDesc::dense(type<T>, shape).value();
  std::size_t outer = 1, inner = 1;
  for (std::size_t dim = 0; dim < axis; ++dim)
    outer *= shape[dim];
  for (std::size_t dim = axis + 1; dim < shape.size(); ++dim)
    inner *= shape[dim];
  const auto width = static_cast<std::size_t>(shape[axis]);
  std::vector<T> original(output_desc.element_count().value());
  for (std::size_t index = 0; index < original.size(); ++index) {
    const auto key =
        mode == 2 ? index / (width * inner) * inner + index % inner : index;
    original[index] = static_cast<T>(static_cast<int>(key % 29) - 13);
    if constexpr (std::is_same_v<T, float>)
      original[index] *= 0.07F;
  }
  Tensor input(input_desc, fixture.resource.ref()),
      output(output_desc, fixture.resource.ref());
  for (bool exclusive : {false, true}) {
    ASSERT_TRUE(input.write<T>(original).ok());
    auto program = prepare(fixture.owner,
                           Scan{.axis = requested_axis, .exclusive = exclusive},
                           input_desc, fixture.resource.ref());
    ASSERT_TRUE(program.has_value()) << program.error().message();
    EXPECT_EQ(program->workspace_requirement().size, 0U);
    EXPECT_EQ(program->invocation_private_bytes(), 0U);
    auto slot = program->create_invocation().value();
    for (bool in_place : {false, true}) {
      if (in_place && mode != 0)
        continue;
      auto done = slot.submit(fixture.queue, input.view.as_const(),
                              in_place ? input.view : output.view);
      ASSERT_TRUE(done.has_value()) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual = (in_place ? input : output).template read<T>();
      for (std::size_t group = 0; group < outer; ++group)
        for (std::size_t column = 0; column < inner; ++column) {
          using Sum = std::conditional_t<std::is_same_v<T, float>, double,
                                         std::int64_t>;
          Sum carry = 0;
          double absolute = 0;
          for (std::size_t position = 0; position < width; ++position) {
            const auto index = (group * width + position) * inner + column;
            const Sum expected = exclusive ? carry : carry + original[index];
            carry += original[index];
            absolute += std::abs(static_cast<double>(original[index]));
            if constexpr (std::is_same_v<T, float>) {
              EXPECT_NEAR(actual[index], expected,
                          3e-6 * std::max(absolute, 1.0));
            } else {
              ASSERT_EQ(actual[index], expected) << index;
            }
          }
        }
      std::array<std::byte, 16> guard{};
      const auto &result = in_place ? input : output;
      ASSERT_TRUE(result.storage.read(result.storage.size() - 16, guard).ok());
      for (auto value : guard)
        EXPECT_EQ(value, std::byte{0xa5});
    }
  }
}
TEST(CudaScan, IntegerAndFloatPrefixesCoverTilesAxesStridesAndExactAlias) {
  Fixture fixture;
  for (std::int64_t width :
       {0, 1, 17, 63, 64, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 8193})
    for (std::int64_t inner : {1, 3})
      for (int mode : {0, 1}) {
        const std::array<std::int64_t, 3> shape{2, width, inner};
        numeric_case<std::int32_t>(fixture, shape, -2, mode);
        numeric_case<float>(fixture, shape, 1, mode);
      }
  for (int mode : {0, 1, 2}) {
    numeric_case<std::int32_t>(
        fixture, std::array<std::int64_t, 8>{2, 1, 3, 1, 129, 1, 2, 1}, 4,
        mode);
    numeric_case<float>(fixture,
                        std::array<std::int64_t, 8>{2, 1, 3, 1, 129, 1, 2, 1},
                        -4, mode);
  }
  numeric_case<std::int32_t>(fixture, std::array<std::int64_t, 1>{65537}, 0, 0);
  numeric_case<float>(fixture, std::array<std::int64_t, 1>{65537}, -1, 0);
}
TEST(CudaScan,
     EveryIntegerPrefixIncludingExclusiveFinalTotalIsCheckedBeforeAnyWrite) {
  Fixture fixture;
  for (std::int64_t width : {3, 1025}) {
    for (bool exclusive : {false, true}) {
      const auto desc = TensorDesc::dense(DataType::kInt32,
                                          std::array<std::int64_t, 2>{2, width})
                            .value();
      auto program = prepare(fixture.owner, Scan{.exclusive = exclusive}, desc,
                             fixture.resource.ref())
                         .value();
      Tensor input(desc, fixture.resource.ref()),
          output(desc, fixture.resource.ref());
      for (bool negative : {false, true})
        for (bool total_recovers : {false, true}) {
          std::vector<std::int32_t> original(
              static_cast<std::size_t>(2 * width), 0);
          original[0] = 7;
          const auto start = static_cast<std::size_t>(2 * width - 3);
          original[start] = negative ? INT32_MIN : INT32_MAX;
          original[start + (total_recovers ? 1 : 2)] = negative ? -1 : 1;
          if (total_recovers)
            original[start + 2] = negative ? 1 : -1;
          ASSERT_TRUE(input.write<std::int32_t>(original).ok());
          const std::vector<std::int32_t> sentinel(original.size(), 0x25252525);
          ASSERT_TRUE(output.write<std::int32_t>(sentinel).ok());
          for (bool in_place : {false, true}) {
            auto slot = program.create_invocation().value();
            auto done = slot.submit(fixture.queue, input.view.as_const(),
                                    in_place ? input.view : output.view);
            ASSERT_TRUE(done.has_value());
            EXPECT_TRUE(done->wait().IsOutOfRange());
            EXPECT_EQ(input.read<std::int32_t>(), original);
            EXPECT_EQ(output.read<std::int32_t>(), sentinel);
            auto retired =
                slot.submit(fixture.queue, input.view.as_const(), output.view);
            ASSERT_FALSE(retired.has_value());
            EXPECT_TRUE(retired.error().IsFailedPrecondition());
          }
          const std::vector<std::int32_t> ones(original.size(), 1);
          ASSERT_TRUE(input.write<std::int32_t>(ones).ok());
          auto fresh = program.create_invocation().value();
          auto good =
              fresh.submit(fixture.queue, input.view.as_const(), output.view)
                  .value();
          ASSERT_TRUE(good.wait().ok());
          const auto actual = output.read<std::int32_t>();
          for (std::size_t index = 0; index < actual.size(); ++index)
            EXPECT_EQ(actual[index], index % width + (exclusive ? 0 : 1));
        }
    }
  }
}
TEST(CudaScan, FloatExclusiveDoesNotSubtractCurrentInfinityOrNaN) {
  Fixture fixture;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (std::int64_t width : {17, 129, 1025}) {
    const auto desc = TensorDesc::dense(DataType::kFloat32,
                                        std::array<std::int64_t, 2>{3, width})
                          .value();
    std::vector<float> values(static_cast<std::size_t>(3 * width), 0.125F);
    values[0] = inf;
    values[width / 2] = -inf;
    values[width] = nan;
    values[2 * width + width / 2] = nan;
    Tensor input(desc, fixture.resource.ref()),
        output(desc, fixture.resource.ref());
    for (bool exclusive : {false, true}) {
      ASSERT_TRUE(input.write<float>(values).ok());
      auto program = prepare(fixture.owner, Scan{.exclusive = exclusive}, desc,
                             fixture.resource.ref())
                         .value();
      auto slot = program.create_invocation().value();
      auto done = slot.submit(fixture.queue, input.view.as_const(), output.view)
                      .value();
      ASSERT_TRUE(done.wait().ok());
      const auto actual = output.read<float>();
      for (std::size_t row = 0; row < 3; ++row) {
        float carry = 0;
        for (std::size_t position = 0;
             position < static_cast<std::size_t>(width); ++position) {
          const auto index = row * width + position;
          const float expected = exclusive ? carry : carry + values[index];
          carry += values[index];
          if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(actual[index])) << index;
          else
            EXPECT_EQ(actual[index], expected) << index;
        }
      }
    }
  }
}
TEST(CudaScan, EmptyDomainsSkipPayloadAndPartialAliasRejectsBeforeAcceptance) {
  Fixture fixture;
  for (auto type : {DataType::kInt32, DataType::kFloat32}) {
    const auto empty = TensorDesc::strided_bytes(
                           type, std::array<std::int64_t, 2>{INT64_MAX, 0},
                           std::array<std::int64_t, 2>{0, 0})
                           .value();
    auto program =
        prepare(fixture.owner, Scan{.axis = 0}, empty, fixture.resource.ref())
            .value();
    Tensor input(empty, fixture.resource.ref()),
        output(program.output_description(0).value(), fixture.resource.ref());
    auto slot = program.create_invocation().value();
    auto done =
        slot.submit(fixture.queue, input.view.as_const(), output.view).value();
    ASSERT_TRUE(done.wait().ok());
  }
  const auto desc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{17})
          .value();
  auto program =
      prepare(fixture.owner, Scan{}, desc, fixture.resource.ref()).value();
  Tensor input(desc, fixture.resource.ref()),
      output(desc, fixture.resource.ref());
  ASSERT_TRUE(input.write<std::int32_t>(std::vector<std::int32_t>(17, 1)).ok());
  auto partial =
      TensorView::create(desc, input.storage.view()
                                   .subview(4, desc.required_bytes().value())
                                   .value())
          .value();
  auto slot = program.create_invocation().value();
  auto rejected = slot.submit(fixture.queue, input.view.as_const(), partial);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  auto good =
      slot.submit(fixture.queue, input.view.as_const(), output.view).value();
  ASSERT_TRUE(good.wait().ok());
  EXPECT_EQ(output.read<std::int32_t>().back(), 17);
  const auto huge =
      TensorDesc::dense(
          DataType::kInt32,
          std::array<std::int64_t, 1>{static_cast<std::int64_t>(INT32_MAX) + 1})
          .value();
  auto unsupported =
      prepare(fixture.owner, Scan{}, huge, fixture.resource.ref());
  ASSERT_FALSE(unsupported.has_value());
  EXPECT_TRUE(unsupported.error().IsUnimplemented());
}
} // namespace
