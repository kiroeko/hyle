#include <gtest/gtest.h>

#include <array>
#include <tuple>

#include "reduction_test_support.h"
#include "scatter_test_support.h"

namespace {
using namespace apple_tensor_test;

mx::array storage_pattern(mx::Dtype type, mx::Shape shape, int phase = 0) {
  std::size_t count = 1;
  for (int dimension : shape)
    count *= dimension;
  if (type == mx::int64) {
    std::vector<std::uint64_t> data(count);
    for (std::size_t i = 0; i < count; ++i)
      data[i] = 0xffffeeee00000000ULL + i * 31 + phase;
    return ready(
        mx::view(mx::array(data.data(), shape, mx::uint64), type, stream()));
  }
  if (type == mx::float16 || type == mx::bfloat16) {
    const std::array<std::uint16_t, 7> raw{0xffc3, 0x7fc7, 0x8000, 0,
                                           0x7c00, 0xfc00, 0x37b1};
    std::vector<std::uint16_t> data(count);
    for (std::size_t i = 0; i < count; ++i)
      data[i] = raw[(i + phase) % raw.size()];
    return ready(
        mx::view(mx::array(data.data(), shape, mx::uint16), type, stream()));
  }
  const std::array<std::uint32_t, 7> raw{0xffc00003, 0x7fc00007, 0x80000000, 0,
                                         0x7f800000, 0xff800000, 0x37b12345};
  std::vector<std::uint32_t> data(count);
  for (std::size_t i = 0; i < count; ++i)
    data[i] = raw[(i + phase) % raw.size()];
  const auto native = mx::array(data.data(), shape, mx::uint32);
  return ready(type == mx::bool_ ? mx::astype(native, type, stream())
                                 : mx::view(native, type, stream()));
}

TEST(AppleScatter, AssignKeepsRawBitsAndLastLogicalUpdateAcrossNativeLayouts) {
  auto context = a::Context::create().value();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16, mx::int32,
                    mx::int64, mx::bool_}) {
    for (auto index_type : {mx::int32, mx::int64}) {
      for (auto index_shape : {mx::Shape{}, mx::Shape{4}, mx::Shape{2, 2}}) {
        const std::vector<std::int64_t> ids =
            index_shape.empty() ? std::vector<std::int64_t>{6}
                                : std::vector<std::int64_t>{6, 0, 6, 3};
        const auto indices =
            ready(mx::astype(mx::array(ids.data(), index_shape, mx::int64),
                             index_type, stream()));
        auto input = ready(mx::transpose(storage_pattern(type, {7, 2, 3}),
                                         {1, 0, 2}, stream()));
        auto updates = ready(mx::transpose(
            storage_pattern(type, {static_cast<int>(ids.size()), 2, 3}, 2),
            {1, 0, 2}, stream()));
        mx::Shape update_shape{2};
        update_shape.insert(update_shape.end(), index_shape.begin(),
                            index_shape.end());
        update_shape.push_back(3);
        updates = ready(mx::reshape(updates, update_shape, stream()));
        for (bool broadcast : {false, true}) {
          if (broadcast) {
            input = ready(mx::broadcast_to(
                mx::slice(input, {0, 0, 0}, {1, 7, 3}, stream()), {2, 7, 3},
                stream()));
            auto stop = update_shape;
            stop[0] = 1;
            updates = ready(mx::broadcast_to(
                mx::slice(updates, mx::Shape(stop.size(), 0), stop, stream()),
                update_shape, stream()));
          }
          const auto source_bits = bits(input), update_bits = bits(updates);
          auto expected = source_bits;
          for (int row = 0; row < 2; ++row)
            for (std::size_t index = 0; index < ids.size(); ++index)
              for (int lane = 0; lane < 3; ++lane)
                expected[(row * 7 + ids[index]) * 3 + lane] =
                    update_bits[(row * ids.size() + index) * 3 + lane];
          for (auto policy : {c::IndexBoundsPolicy::kChecked,
                              c::IndexBoundsPolicy::kUnchecked}) {
            auto result = execute_all(
                context, c::Operation(c::Scatter{.axis = -2, .bounds = policy}),
                std::array{input, indices, updates});
            ASSERT_TRUE(result) << result.error().message();
            ASSERT_EQ(result->size(), 1U);
            EXPECT_EQ(bits((*result)[0]), expected);
            EXPECT_EQ(bits(input), source_bits);
            EXPECT_EQ(bits(updates), update_bits);
          }
        }
      }
    }
  }
}

TEST(AppleScatter, AddRoundsEveryDuplicateInOriginalOrderIncludingLongSorts) {
  auto context = a::Context::create().value();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    for (const auto &[count, width] :
         {std::pair{1, 1}, {17, 129}, {129, 128}, {4097, 3}, {17, 2048}}) {
      for (bool skew : {false, true}) {
        const double large = type == mx::float32   ? 16777216.0
                             : type == mx::float16 ? 2048.0
                                                   : 256.0;
        const std::array sequence{large, 1.0, -large, 0.125, -0.0625};
        std::vector<std::int64_t> ids(count);
        for (int index = 0; index < count; ++index)
          ids[index] = skew ? 3 : (index * 5) % 7;
        std::vector<double> data(static_cast<std::size_t>(count) * 2 * width);
        for (int index = 0; index < count; ++index)
          for (int row = 0; row < 2; ++row)
            for (int lane = 0; lane < width; ++lane)
              data[(index * 2 + row) * width + lane] = sequence[index % 5];
        if (count > 1 && width > 1) {
          data[0] = std::numeric_limits<double>::infinity();
          data[2 * width] = -std::numeric_limits<double>::infinity();
          data[1] = std::numeric_limits<double>::quiet_NaN();
        }
        auto updates = ready(mx::transpose(make(data, {count, 2, width}, type),
                                           {1, 0, 2}, stream()));
        auto source_values = pattern(static_cast<std::size_t>(2 * 7 * width));
        source_values[0] = -0.0;
        const auto input = make(source_values, {2, 7, width}, type);
        const auto index_type = skew ? mx::int64 : mx::int32;
        const auto indices = ready(mx::astype(
            mx::array(ids.data(), {count}, mx::int64), index_type, stream()));
        const auto input_bits = bits(input), update_bits = bits(updates);
        const auto expected = scatter_reference(values(input), values(updates),
                                                ids, input.shape(), 1, type);
        auto result = execute(
            context, c::Scatter{.axis = 1, .kind = c::ScatterKind::kAdd},
            std::array{input, indices, updates});
        ASSERT_TRUE(result) << result.error().message();
        expect_ordered(*result, expected);
        EXPECT_EQ(bits(input), input_bits);
        EXPECT_EQ(bits(updates), update_bits);
      }
    }
    // A reduction in F32 followed by one storage cast would produce one.
    const double large = type == mx::float32   ? 16777216.0
                         : type == mx::float16 ? 2048.0
                                               : 256.0;
    auto result = execute(
        context, c::Scatter{.kind = c::ScatterKind::kAdd},
        std::array{make(std::array{0.0}, {1}, type), mx::array({0, 0, 0}),
                   make(std::array{large, 1.0, -large}, {3}, type)});
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(values(*result), std::vector<double>{0.0});
  }
}

TEST(AppleScatter, CheckedFailuresDoNotPublishOrChangeRetainedState) {
  auto context = a::Context::create().value();
  for (auto index_type : {mx::int32, mx::int64}) {
    for (auto kind : {c::ScatterKind::kAssign, c::ScatterKind::kAdd}) {
      for (int count : {1, 257}) {
        const auto source = make(pattern(7 * 129), {7, 129}, mx::bfloat16);
        const auto updates =
            make(pattern(count * 129, 2), {count, 129}, mx::bfloat16);
        const auto original = bits(source), update_bits = bits(updates);
        const std::array specs{a::InputSpec{{7, 129}, mx::bfloat16},
                               a::InputSpec{{count}, index_type},
                               a::InputSpec{{count, 129}, mx::bfloat16}};
        auto program = context.prepare(c::Scatter{.kind = kind}, specs);
        ASSERT_TRUE(program) << program.error().message();
        std::vector<std::int64_t> ids(count, 3);
        const auto make_ids = [&] {
          return ready(mx::astype(mx::array(ids.data(), {count}, mx::int64),
                                  index_type, stream()));
        };
        auto retained = program->create_invocation().value();
        ASSERT_TRUE(
            retained.bind(std::array{source, make_ids(), updates}).ok());
        auto old_work = retained.submit().value();
        auto old = old_work.wait().value();
        const auto old_bits = bits(old.get()->arrays()[0]);
        const auto minimum =
            index_type == mx::int32
                ? std::int64_t{std::numeric_limits<std::int32_t>::min()}
                : std::numeric_limits<std::int64_t>::min();
        const auto maximum =
            index_type == mx::int32
                ? std::int64_t{std::numeric_limits<std::int32_t>::max()}
                : std::numeric_limits<std::int64_t>::max();
        for (auto bad : std::array<std::int64_t, 4>{-1, 7, minimum, maximum}) {
          ids.back() = bad;
          auto slot = program->create_invocation().value();
          ASSERT_TRUE(slot.bind(std::array{source, make_ids(), updates}).ok());
          auto work = slot.submit().value();
          auto result = work.wait();
          ASSERT_FALSE(result);
          EXPECT_TRUE(result.error().IsOutOfRange());
          EXPECT_FALSE(work.result());
          EXPECT_FALSE(slot.submit());
          EXPECT_EQ(bits(source), original);
          EXPECT_EQ(bits(updates), update_bits);
          EXPECT_EQ(bits(old.get()->arrays()[0]), old_bits);
          ids.back() = 3;
          auto fresh = program->create_invocation().value();
          ASSERT_TRUE(fresh.bind(std::array{source, make_ids(), updates}).ok());
          auto again = fresh.submit().value();
          auto good = again.wait();
          ASSERT_TRUE(good) << good.error().message();
          EXPECT_EQ(bits(good->get()->arrays()[0]), old_bits);
        }
      }
    }
  }
  EXPECT_TRUE(context.close().ok());
}

TEST(AppleScatter, EmptyScalarRankEightAndPreparationBoundaries) {
  auto context = a::Context::create().value();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16, mx::int32,
                    mx::int64, mx::bool_}) {
    const auto input = storage_pattern(type, {2, 7, 3});
    const auto indices = ready(mx::zeros({0}, mx::int64, stream()));
    const auto updates = storage_pattern(type, {2, 0, 3});
    auto result = execute(context, c::Scatter{.axis = 1},
                          std::array{input, indices, updates});
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(bits(*result), bits(input));
    for (auto shape : {mx::Shape{2, 0, 3}, mx::Shape{0, 7, 3}}) {
      const auto empty = storage_pattern(type, shape);
      shape[1] = 2;
      const auto values = storage_pattern(type, shape);
      auto out = execute(context, c::Scatter{.axis = 1},
                         std::array{empty, mx::array({-1, 999}), values});
      ASSERT_TRUE(out) << out.error().message();
      EXPECT_EQ(out->size(), 0U);
    }
    const auto rank8 =
        ready(mx::reshape(input, {1, 2, 1, 7, 1, 1, 3, 1}, stream()));
    const auto scalar_update = storage_pattern(type, {1, 2, 1, 1, 1, 3, 1}, 3);
    auto rank_result =
        execute(context, c::Scatter{.axis = 3},
                std::array{rank8, mx::array(std::int64_t{6}), scalar_update});
    ASSERT_TRUE(rank_result) << rank_result.error().message();
    auto expected = bits(rank8);
    const auto raw_updates = bits(scalar_update);
    for (int row = 0; row < 2; ++row)
      for (int lane = 0; lane < 3; ++lane)
        expected[(row * 7 + 6) * 3 + lane] = raw_updates[row * 3 + lane];
    EXPECT_EQ(bits(*rank_result), expected);
    if (type == mx::int32 || type == mx::int64 || type == mx::bool_) {
      auto unsupported =
          context.prepare(c::Scatter{.axis = 1, .kind = c::ScatterKind::kAdd},
                          std::array{a::InputSpec{{2, 7, 3}, type},
                                     a::InputSpec{{0}, mx::int64},
                                     a::InputSpec{{2, 0, 3}, type}});
      ASSERT_FALSE(unsupported);
      EXPECT_TRUE(unsupported.error().IsUnimplemented());
    }
  }
  auto mismatch = context.prepare(
      c::Scatter{}, std::array{a::InputSpec{{7, 3}, mx::float32},
                               a::InputSpec{{2}, mx::int64},
                               a::InputSpec{{2, 4}, mx::float32}});
  ASSERT_FALSE(mismatch);
  EXPECT_TRUE(mismatch.error().IsInvalidArgument());
}
} // namespace
