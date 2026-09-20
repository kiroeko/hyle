#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <set>
#include <string>
#include <string_view>

#include <mlx/primitives.h>

#include "reduction_test_support.h"

namespace {
using namespace apple_tensor_test;

TEST(AppleIndexing, OrdinaryLoweringUsesOriginalMlxPrimitives) {
  auto context = a::Context::create().value();
  auto names = std::make_shared<std::set<std::string>>();
  auto program = context.compile(
      [names](const auto &input,
              mx::Stream stream) -> hyle::Result<std::vector<mx::array>> {
        HYLE_ASSIGN_OR_RETURN(
            auto gathered, a::lower(c::Operation(c::Gather{}),
                                    std::array{input[0], input[1]}, stream));
        HYLE_ASSIGN_OR_RETURN(
            auto prefix,
            a::lower(c::Operation(c::Scan{.exclusive = true}),
                     std::array{input[2]}, stream));
        HYLE_ASSIGN_OR_RETURN(
            auto top, a::lower(c::Operation(c::TopK{.count = 3}), prefix,
                               stream));
        std::vector outputs{gathered[0], prefix[0], top[0], top[1]};
        auto pending = outputs;
        std::set<std::uintptr_t> seen;
        while (!pending.empty()) {
          auto value = std::move(pending.back());
          pending.pop_back();
          if (!seen.insert(value.id()).second || !value.has_primitive())
            continue;
          names->insert(value.primitive().name());
          for (const auto &argument : value.inputs())
            pending.push_back(argument);
        }
        return outputs;
      },
      std::array{a::InputSpec{{7}, mx::float32},
                 a::InputSpec{{3}, mx::int32}, a::InputSpec{{7}, mx::int32}});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_TRUE(names->contains("Gather"));
  EXPECT_TRUE(names->contains("CumSum"));
  EXPECT_TRUE(names->contains("ArgSort"));
  for (const auto &name : *names) {
    EXPECT_EQ(name.find("Custom"), std::string::npos) << name;
    EXPECT_EQ(name.find("MetalKernel"), std::string::npos) << name;
  }
  auto slot = program->create_invocation().value();
  ASSERT_TRUE(slot.bind(std::array{
      mx::array({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F}),
      mx::array({1, 0, 6}), mx::array({3, 1, 4, 1, 5, 9, 2})}).ok());
  auto work = slot.submit().value();
  auto output = work.wait();
  ASSERT_TRUE(output) << output.error().message();
  const auto arrays = output->get()->arrays();
  ASSERT_EQ(arrays.size(), 4U);
  EXPECT_EQ(values(arrays[0]), (std::vector<double>{2, 1, 7}));
  EXPECT_EQ(values(arrays[1]), (std::vector<double>{0, 3, 4, 8, 9, 14, 23}));
  EXPECT_EQ(values(arrays[2]), (std::vector<double>{23, 14, 9}));
  EXPECT_EQ(values(arrays[3]), (std::vector<double>{6, 5, 4}));
}

mx::array source(mx::Dtype type) {
  if (type == mx::int64) {
    std::vector<std::int64_t> data(42);
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = (i % 2 ? std::numeric_limits<std::int64_t>::min()
                       : std::numeric_limits<std::int64_t>::max()) +
                (i % 2 ? static_cast<std::int64_t>(i)
                       : -static_cast<std::int64_t>(i));
    return mx::array(data.data(), {7, 2, 3}, type);
  }
  if (type == mx::int32 || type == mx::bool_) {
    std::vector<std::int32_t> data(42);
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = static_cast<std::int32_t>(i) - 17;
    return ready(mx::astype(mx::array(data.data(), {7, 2, 3}, mx::int32), type,
                            stream()));
  }
  auto data = pattern(42);
  data[0] = std::numeric_limits<double>::quiet_NaN();
  data[1] = -0.0;
  return make(data, {7, 2, 3}, type);
}

TEST(AppleIndexing, GatherPreservesStorageAcrossIndexWidthsShapesAndLayouts) {
  auto context = a::Context::create().value();
  for (auto type : {mx::float32, mx::float16, mx::bfloat16, mx::int32,
                    mx::int64, mx::bool_}) {
    auto input = ready(mx::transpose(source(type), {1, 0, 2}, stream()));
    for (bool broadcast : {false, true}) {
      if (broadcast)
        input = ready(
            mx::broadcast_to(mx::slice(input, {0, 0, 0}, {1, 7, 3}, stream()),
                             {2, 7, 3}, stream()));
      const auto raw = bits(input);
      for (auto index_type : {mx::int32, mx::int64}) {
        for (auto shape : {mx::Shape{}, mx::Shape{4}, mx::Shape{2, 2}}) {
          const std::vector<std::int64_t> ids =
              shape.empty() ? std::vector<std::int64_t>{6}
                            : std::vector<std::int64_t>{6, 0, 6, 3};
          const auto indices = ready(mx::astype(
              mx::array(ids.data(), shape, mx::int64), index_type, stream()));
          for (auto policy : {c::IndexBoundsPolicy::kChecked,
                              c::IndexBoundsPolicy::kUnchecked}) {
            auto out = execute_all(
                context, c::Operation(c::Gather{.axis = -2, .bounds = policy}),
                std::array{input, indices});
            ASSERT_TRUE(out) << out.error().message();
            ASSERT_EQ(out->size(), 1U); // The device status is never public.
            const auto observed = bits((*out)[0]);
            std::vector<std::uint64_t> expected;
            for (int row = 0; row < 2; ++row)
              for (auto index : ids)
                for (int lane = 0; lane < 3; ++lane)
                  expected.push_back(raw[(row * 7 + index) * 3 + lane]);
            EXPECT_EQ(observed, expected);
            EXPECT_EQ(bits(input), raw);
          }
        }
      }
    }
    const auto rank8 =
        ready(mx::reshape(input, {1, 2, 1, 7, 1, 1, 3, 1}, stream()));
    const auto index = mx::array(std::int64_t{6});
    auto out = execute(context, c::Gather{.axis = 3}, std::array{rank8, index});
    ASSERT_TRUE(out) << out.error().message();
    EXPECT_EQ(out->shape(), (mx::Shape{1, 2, 1, 1, 1, 3, 1}));
    const auto raw = bits(input), actual = bits(*out);
    for (int row = 0; row < 2; ++row)
      for (int lane = 0; lane < 3; ++lane)
        EXPECT_EQ(actual[row * 3 + lane], raw[(row * 7 + 6) * 3 + lane]);
  }
}

TEST(AppleIndexing, BadIndicesRetireOnlyTheFailedInvocation) {
  auto context = a::Context::create().value();
  for (auto type : {mx::int32, mx::int64}) {
    const auto table = source(mx::bfloat16);
    const auto original = bits(table);
    const std::array specs{a::InputSpec{table.shape(), table.dtype()},
                           a::InputSpec{{3}, type}};
    auto program = context.prepare(c::Gather{}, specs);
    ASSERT_TRUE(program) << program.error().message();
    const auto valid = ready(mx::astype(mx::array({0, 6, 0}), type, stream()));
    auto retained_slot = program->create_invocation().value();
    ASSERT_TRUE(retained_slot.bind(std::array{table, valid}).ok());
    auto retained_work = retained_slot.submit().value();
    auto old = retained_work.wait().value();
    const auto old_bits = bits(old.get()->arrays()[0]);
    const std::array bad_values =
        type == mx::int32
            ? std::array<std::int64_t,
                         4>{-1, 7, std::numeric_limits<std::int32_t>::min(),
                            std::numeric_limits<std::int32_t>::max()}
            : std::array<std::int64_t, 4>{
                  -1, 7, std::numeric_limits<std::int64_t>::min(),
                  std::numeric_limits<std::int64_t>::max()};
    for (auto invalid : bad_values) {
      const std::array<std::int64_t, 3> ids{0, 6, invalid};
      const auto indices = ready(
          mx::astype(mx::array(ids.data(), {3}, mx::int64), type, stream()));
      auto failed = program->create_invocation().value();
      ASSERT_TRUE(failed.bind(std::array{table, indices}).ok());
      auto work = failed.submit();
      ASSERT_TRUE(work) << work.error().message();
      auto result = work->wait();
      ASSERT_FALSE(result);
      EXPECT_TRUE(result.error().IsOutOfRange());
      EXPECT_FALSE(work->result());
      EXPECT_FALSE(failed.submit());
      EXPECT_EQ(bits(table), original);
      EXPECT_EQ(bits(old.get()->arrays()[0]), old_bits);
      auto fresh = program->create_invocation().value();
      ASSERT_TRUE(fresh.bind(std::array{table, valid}).ok());
      auto again = fresh.submit().value();
      auto good = again.wait();
      ASSERT_TRUE(good) << good.error().message();
      EXPECT_EQ(bits(good->get()->arrays()[0]), old_bits);
    }
  }
  EXPECT_TRUE(context.close().ok()); // No native device fault poisoned it.
}

TEST(AppleIndexing,
     ScanMatchesIntegerAndFloatingPrefixesIncludingExclusiveTotal) {
  auto context = a::Context::create().value();
  for (auto type : {mx::int32, mx::float32}) {
    for (int width : {0, 1, 17, 127, 129, 1025, 8193, 65537}) {
      std::vector<float> floats(static_cast<std::size_t>(width) * 6);
      std::vector<std::int32_t> integers(floats.size());
      for (std::size_t i = 0; i < floats.size(); ++i) {
        floats[i] = 1.0F + static_cast<float>(i % 13) * 0.0037F;
        integers[i] = static_cast<std::int32_t>(i % 7) - 2;
      }
      auto input = ready(mx::transpose(
          type == mx::float32 ? mx::array(floats.data(), {width, 2, 3}, type)
                              : mx::array(integers.data(), {width, 2, 3}, type),
          {1, 0, 2}, stream()));
      const auto data = values(input);
      const auto original = bits(input);
      for (bool exclusive : {false, true}) {
        auto outputs = execute_all(
            context, c::Operation(c::Scan{.axis = -2, .exclusive = exclusive}),
            std::array{input});
        ASSERT_TRUE(outputs) << outputs.error().message();
        ASSERT_EQ(outputs->size(), 1U);
        std::vector<double> expected(data.size());
        for (int row = 0; row < 2; ++row)
          for (int lane = 0; lane < 3; ++lane) {
            double sum = 0;
            for (int col = 0; col < width; ++col) {
              const auto index = (row * width + col) * 3 + lane;
              expected[index] = sum;
              sum += data[index];
              if (!exclusive)
                expected[index] = sum;
            }
          }
        if (type == mx::int32)
          EXPECT_EQ(values((*outputs)[0]), expected);
        else
          expect((*outputs)[0], expected);
        EXPECT_EQ(bits(input), original);
      }
    }
  }
  const auto inf = std::numeric_limits<float>::infinity();
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const auto special = mx::array({1.0F, inf, 2.0F, -inf, 3.0F, nan});
  for (bool exclusive : {false, true}) {
    auto output =
        execute(context, c::Scan{.exclusive = exclusive}, std::array{special});
    ASSERT_TRUE(output) << output.error().message();
    std::vector<double> expected;
    double total = 0;
    for (double x : values(special)) {
      const auto before = total;
      total += x;
      expected.push_back(exclusive ? before : total);
    }
    expect(*output, expected);
  }
  for (bool exclusive : {false, true}) {
    for (const auto &last_row :
         {std::array<std::int32_t, 3>{std::numeric_limits<std::int32_t>::max(),
                                      1, -1},
          {std::numeric_limits<std::int32_t>::min(), -1, 1},
          {std::numeric_limits<std::int32_t>::max(), 0, 1}}) {
      const std::array<std::int32_t, 6> data{
          1, 2, 3, last_row[0], last_row[1], last_row[2]};
      const auto input = mx::array(data.data(), {2, 3}, mx::int32);
      const auto original = bits(input);
      auto out =
          execute(context, c::Scan{.exclusive = exclusive}, std::array{input});
      ASSERT_FALSE(out);
      EXPECT_TRUE(out.error().IsOutOfRange());
      EXPECT_EQ(bits(input), original);
    }
  }
  EXPECT_TRUE(context.close().ok());
}

TEST(AppleIndexing, ReachabilityWarmupAndFirstErrorAreBoundDuringPreparation) {
  auto context = a::Context::create().value();
  const auto table = mx::array({1.0F, 2.0F, 3.0F});
  const auto bad_index = mx::array({-1, 3});
  const auto overflow =
      mx::array({std::numeric_limits<std::int32_t>::max(), 1});
  const auto build = [&](bool keep_bad, bool scan_first) {
    return context.compile(
        [=](const auto &,
            mx::Stream stream) -> hyle::Result<std::vector<mx::array>> {
          if (scan_first) {
            HYLE_ASSIGN_OR_RETURN(auto prefix,
                                  a::lower(c::Operation(c::Scan{}),
                                           std::array{overflow}, stream));
            HYLE_ASSIGN_OR_RETURN(
                auto gather, a::lower(c::Operation(c::Gather{}),
                                      std::array{table, bad_index}, stream));
            return std::vector{prefix[0], gather[0]};
          }
          HYLE_ASSIGN_OR_RETURN(auto gather,
                                a::lower(c::Operation(c::Gather{}),
                                         std::array{table, bad_index}, stream));
          HYLE_ASSIGN_OR_RETURN(
              auto prefix,
              a::lower(c::Operation(c::Scan{}), std::array{overflow}, stream));
          return keep_bad ? std::vector{gather[0], prefix[0]}
                          : std::vector{table};
        },
        std::span<const a::InputSpec>{});
  };
  for (bool first_scan : {false, true}) {
    auto program = build(true, first_scan);
    ASSERT_TRUE(program)
        << program.error().message(); // Warm status isn't a payload verdict.
    auto slot = program->create_invocation().value();
    ASSERT_TRUE(slot.bind(std::span<const mx::array>{}).ok());
    auto work = slot.submit().value();
    auto failed = work.wait();
    ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsOutOfRange());
    EXPECT_NE(failed.error().message().find(first_scan ? "Scan" : "index"),
              std::string_view::npos);
  }
  auto dead = build(false, false);
  ASSERT_TRUE(dead) << dead.error().message();
  auto slot = dead->create_invocation().value();
  ASSERT_TRUE(slot.bind(std::span<const mx::array>{}).ok());
  auto work = slot.submit().value();
  auto good = work.wait();
  ASSERT_TRUE(good) << good.error().message();
  ASSERT_EQ(good->get()->arrays().size(), 1U);
  EXPECT_EQ(bits(good->get()->arrays()[0]), bits(table));
  auto unmanaged = a::lower(c::Operation(c::Gather{}),
                            std::array{table, bad_index}, stream());
  ASSERT_FALSE(unmanaged);
  EXPECT_TRUE(unmanaged.error().IsFailedPrecondition());
  auto failed_trace = context.compile(
      [=](const auto &,
          mx::Stream stream) -> hyle::Result<std::vector<mx::array>> {
        HYLE_ASSIGN_OR_RETURN(auto output,
                              a::lower(c::Operation(c::Gather{}),
                                       std::array{table, bad_index}, stream));
        return std::unexpected(
            hyle::Status::InvalidArgument("injected cold trace failure"));
      },
      std::span<const a::InputSpec>{});
  EXPECT_FALSE(failed_trace);
  auto plain =
      execute(context, c::Unary{.kind = c::UnaryKind::kNeg}, std::array{table});
  ASSERT_TRUE(plain) << plain.error().message();
  expect(*plain, std::array{-1.0, -2.0, -3.0});
  EXPECT_TRUE(context.close().ok());
}

TEST(AppleIndexing, EmptyDomainsAndAcceptedFailuresKeepTheirOwners) {
  auto context = a::Context::create().value();
  const auto indices = mx::array({0, -1});
  const auto empty_source = ready(mx::zeros({0, 3}, mx::bfloat16, stream()));
  auto failed =
      execute(context, c::Gather{}, std::array{empty_source, indices});
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsOutOfRange());
  const auto empty_indices = ready(mx::zeros({0}, mx::int64, stream()));
  auto empty =
      execute(context, c::Gather{}, std::array{empty_source, empty_indices});
  ASSERT_TRUE(empty) << empty.error().message();
  EXPECT_EQ(empty->shape(), (mx::Shape{0, 3}));
  const auto no_output = ready(mx::zeros({7, 0}, mx::float32, stream()));
  auto skipped = execute(context, c::Gather{}, std::array{no_output, indices});
  ASSERT_TRUE(skipped) << skipped.error().message();
  EXPECT_EQ(skipped->size(), 0U);
  auto program =
      context
          .prepare(c::Gather{}, std::array{a::InputSpec{{3}, mx::float32},
                                           a::InputSpec{{2}, mx::int32}})
          .value();
  auto slot = program.create_invocation().value();
  const std::array inputs{mx::array({1.0F, 2.0F, 3.0F}), mx::array({0, 3})};
  ASSERT_TRUE(slot.bind(inputs).ok());
  auto work = slot.submit().value();
  program = {};
  slot = {};
  ASSERT_TRUE(context.close().ok());
  auto result = work.wait();
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsOutOfRange());
  auto abandoned_context = a::Context::create().value();
  {
    auto p = abandoned_context.prepare(c::Gather{}, inputs).value();
    auto i = p.create_invocation().value();
    ASSERT_TRUE(i.bind(inputs).ok());
    auto abandoned = i.submit();
    ASSERT_TRUE(abandoned);
  }
  EXPECT_TRUE(abandoned_context.close().ok());
}
} // namespace
