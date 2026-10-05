#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "aten_cast_reference.h"
#include "half_test_support.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/stats.h"

namespace {
using namespace hyle::compute;
namespace reference = cast_reference;
using hyle::mem::Buffer;
Context
context(hyle::mem::AllocatorRef allocator = hyle::mem::system_allocator()) {
  const std::array bindings{ProviderBinding{aten_cpu::provider(), 100}};
  return Context::create(cpu::device(), {.metadata_allocator = allocator,
                                         .providers = bindings})
      .value();
}
Buffer storage(const TensorDesc &d) {
  return Buffer::create(hyle::mem::system_resource(),
                        d.required_bytes().value(), 64)
      .value();
}
std::vector<std::uint32_t> codes(const reference::Format &f) {
  if (f.width != 4) {
    std::vector<std::uint32_t> result(std::size_t(1) << (8 * f.width));
    for (std::size_t i = 0; i < result.size(); ++i)
      result[i] = static_cast<std::uint32_t>(i);
    return result;
  }
  std::vector<std::uint32_t> result{
      0,          0x80000000, 1,          0x80000001, 0x7f800000, 0xff800000,
      0x7f800001, 0xff800001, 0x7fc12345, 0xffc12345, 0x7f7fffff, 0xff7fffff};
  std::uint32_t state = 0x8372916d;
  for (int i = 0; i < 65536; ++i) {
    state = 1664525U * state + 1013904223U;
    result.push_back(state);
  }
  for (const auto &format : reference::formats)
    if (format.width == 1) {
      const auto maximum = format.infinity ? 0x7bU : 0x7eU;
      for (unsigned i = 0; i < maximum; ++i) {
        const auto midpoint = static_cast<float>(
            (reference::decode(i, format) + reference::decode(i + 1, format)) /
            2);
        for (float x : {std::nextafter(midpoint, -INFINITY), midpoint,
                        std::nextafter(midpoint, INFINITY)})
          for (float y : {x, -x})
            result.push_back(std::bit_cast<std::uint32_t>(y));
      }
      for (float x : {448.F, 449.F, 464.F, 57344.F, 61440.F, 65504.F, 65520.F,
                      INFINITY, -INFINITY})
        result.push_back(std::bit_cast<std::uint32_t>(x));
    }
  return result;
}
void write_codes(Buffer &buffer, const TensorDesc &d,
                 std::span<const std::uint32_t> input) {
  std::vector<std::byte> bytes(buffer.size(), std::byte{0xd7});
  const auto width = data_type_size(d.data_type());
  for (std::size_t i = 0; i < input.size(); ++i) {
    std::size_t offset = 0, rest = i;
    for (std::size_t axis = d.rank(); axis-- != 0;) {
      const auto extent = static_cast<std::size_t>(d.dimensions()[axis]);
      offset +=
          (rest % extent) * static_cast<std::size_t>(d.byte_strides()[axis]);
      rest /= extent;
    }
    std::memcpy(bytes.data() + offset, &input[i], width);
  }
  ASSERT_TRUE(buffer.view().write(bytes).ok());
}
std::uint32_t expected(std::uint32_t bits, const reference::Format &source,
                       const reference::Format &destination) {
  if (source.type == destination.type)
    return bits;
  const auto wide = reference::float_bits(bits, source);
  return destination.width == 4
             ? wide
             : reference::encode(std::bit_cast<float>(wide), destination,
                                 (wide >> 31) != 0);
}
void check_codes(const Buffer &buffer,
                 std::span<const std::uint32_t> source_codes,
                 const reference::Format &source,
                 const reference::Format &destination) {
  std::vector<std::byte> bytes(buffer.size());
  ASSERT_TRUE(buffer.view().read(bytes).ok());
  ASSERT_EQ(bytes.size(), source_codes.size() * destination.width);
  for (std::size_t i = 0; i < source_codes.size(); ++i) {
    std::uint32_t actual = 0;
    std::memcpy(&actual, bytes.data() + i * destination.width,
                destination.width);
    ASSERT_EQ(actual, expected(source_codes[i], source, destination))
        << "index=" << i << " source=" << int(source.type)
        << " target=" << int(destination.type) << " bits=" << source_codes[i];
  }
}

TEST(AtenCpuCast,
     AllTwentyFiveStoragePairsPreserveExactContractAcrossOddBytes) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto ctx = context();
  auto q = ctx.create_queue().value();
  for (const auto &source : reference::formats) {
    const auto input = codes(source);
    const std::array shape{static_cast<std::int64_t>(input.size())};
    for (bool odd : {false, true}) {
      const auto d = odd ? TensorDesc::strided_bytes(
                               source.type, shape,
                               std::array<std::int64_t, 1>{source.width + 1})
                               .value()
                         : TensorDesc::dense(source.type, shape).value();
      auto in = storage(d);
      write_codes(in, d, input);
      const auto view = ConstTensorView::create(d, in.view()).value();
      for (const auto &destination : reference::formats) {
        auto p = prepare(ctx, Cast{destination.type}, d).value();
        EXPECT_EQ(p.provider().id, aten_cpu::kProviderId);
        EXPECT_FALSE(p.alias_support(0, 0)->supported);
        auto lane = p.create_invocation().value();
        const auto output = p.output_description(0).value();
        auto out = storage(output);
        const auto target = TensorView::create(output, out.view()).value();
        ASSERT_TRUE(lane.submit(q, view, target)->wait().ok());
        check_codes(out, input, source, destination);
      }
    }
  }
  ASSERT_TRUE(ctx.close().ok());
}

TEST(AtenCpuCast, ScalarRankEightEmptyAndZeroStrideHaveRealStorageResults) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto ctx = context();
  auto q = ctx.create_queue().value();
  for (const auto &source : reference::formats)
    for (const auto &destination : reference::formats) {
      for (const auto &shape : std::vector<std::vector<std::int64_t>>{
               {}, {0}, {1, 1, 1, 1, 1, 1, 1, 2}}) {
        auto d = TensorDesc::dense(source.type, shape).value();
        std::vector<std::uint32_t> input(d.element_count().value(),
                                         source.width == 1   ? 0x80U
                                         : source.width == 2 ? 0x8000U
                                                             : 0x80000000U);
        auto in = storage(d);
        write_codes(in, d, input);
        auto p = prepare(ctx, Cast{destination.type}, d).value();
        auto lane = p.create_invocation().value();
        auto od = p.output_description(0).value();
        auto out = storage(od);
        ASSERT_TRUE(lane.submit(q,
                                ConstTensorView::create(d, in.view()).value(),
                                TensorView::create(od, out.view()).value())
                        ->wait()
                        .ok());
        check_codes(out, input, source, destination);
      }
      auto d =
          TensorDesc::strided_bytes(source.type, std::array<std::int64_t, 1>{3},
                                    std::array<std::int64_t, 1>{0})
              .value();
      std::vector<std::uint32_t> input(3, source.width == 1   ? 0x3cU
                                          : source.width == 2 ? 0x3c00U
                                                              : 0x3f800000U);
      auto in = storage(d);
      write_codes(in, d, input);
      auto p = prepare(ctx, Cast{destination.type}, d).value();
      auto od = p.output_description(0).value();
      auto out = storage(od);
      auto lane = p.create_invocation().value();
      ASSERT_TRUE(lane.submit(q, ConstTensorView::create(d, in.view()).value(),
                              TensorView::create(od, out.view()).value())
                      ->wait()
                      .ok());
      check_codes(out, input, source, destination);
    }
}

TEST(AtenCpuCast, PrivatePackingReplaysRejectsAliasAndClosesWithoutLeaks) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto ctx = context(stats.get());
    auto q = ctx.create_queue().value();
    auto d =
        TensorDesc::dense(DataType::kFloat8E5M2, std::array<std::int64_t, 1>{4})
            .value();
    auto p = prepare(ctx, Cast{DataType::kFloat8E4M3FN}, d).value();
    EXPECT_EQ(p.runtime_properties().replay_class,
              ProviderReplayClass::kManaged);
    EXPECT_TRUE(p.runtime_properties().uses_private_threads);
    EXPECT_FALSE(p.alias_support(0, 0)->supported);
    auto in = storage(d), out = storage(p.output_description(0).value());
    auto lane = p.create_invocation().value();
    auto view = ConstTensorView::create(d, in.view()).value();
    auto target =
        TensorView::create(p.output_description(0).value(), out.view()).value();
    for (const auto input :
         std::array{std::array<std::uint32_t, 4>{0x00, 0x80, 0x7c, 0xfe},
                    std::array<std::uint32_t, 4>{0x3c, 0xbc, 0x7b, 0xff}}) {
      write_codes(in, d, input);
      ASSERT_TRUE(lane.submit(q, view, target)->wait().ok());
      check_codes(out, input, reference::formats[4], reference::formats[3]);
    }
    auto same = prepare(ctx, Cast{DataType::kFloat8E5M2}, d).value();
    auto bad = same.create_invocation().value();
    EXPECT_FALSE(bad.submit(q, view, TensorView::create(d, in.view()).value()));
    auto unsupported = prepare(ctx, Cast{DataType::kBool}, d);
    EXPECT_FALSE(unsupported);
    EXPECT_TRUE(unsupported.error().IsUnimplemented());
    ASSERT_TRUE(ctx.close().ok());
    EXPECT_FALSE(lane.submit(q, view, target));
  }
  EXPECT_TRUE(stats.balanced());
}

template <class T> void finite_gate() {
  auto ctx = context();
  auto q = ctx.create_queue().value();
  const auto d = TensorDesc::dense(half_test::data_type<T>(),
                                   std::array<std::int64_t, 1>{8})
                     .value();
  auto gate = storage(d), value = storage(d), out = storage(d);
  auto p =
      prepare(ctx, GatedActivation{.inputs = GatedInputKind::kSeparate}, d, d)
          .value();
  auto lane = p.create_invocation().value();
  for (double scale :
       {std::is_same_v<T, float> || std::is_same_v<T, hyle::simd::bfloat16_t>
            ? 1e38
            : 60000.,
        double(INFINITY)}) {
    std::array<T, 8> gates{}, values{};
    std::array<float, 8> samples{-100.F, -90.F, -80.F,    -0.F,
                                 0.F,    2.F,   INFINITY, -INFINITY};
    for (std::size_t i = 0; i < gates.size(); ++i) {
      gates[i] = half_test::encode<T>(samples[i]);
      values[i] = half_test::encode<T>(scale);
    }
    ASSERT_TRUE(gate.view().write(std::as_bytes(std::span(gates))).ok());
    ASSERT_TRUE(value.view().write(std::as_bytes(std::span(values))).ok());
    ASSERT_TRUE(lane.submit(q, ConstTensorView::create(d, gate.view()).value(),
                            ConstTensorView::create(d, value.view()).value(),
                            TensorView::create(d, out.view()).value())
                    ->wait()
                    .ok());
    const auto actual = half_test::read<T>(out);
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const auto x = half_test::decode(gates[i]);
      const auto e = std::exp(-std::abs(x));
      const float activation =
          x == -INFINITY
              ? 0.F
              : static_cast<float>(x * (x < 0 ? e / (1 + e) : 1 / (1 + e)));
      const auto expected = half_test::decode(half_test::encode<T>(
          activation * static_cast<float>(half_test::decode(values[i]))));
      const auto result = half_test::decode(actual[i]);
      if (std::isnan(expected))
        EXPECT_TRUE(std::isnan(result));
      else if (std::isinf(expected) || expected == 0) {
        EXPECT_EQ(result, expected);
        EXPECT_EQ(std::signbit(result), std::signbit(expected));
      } else
        EXPECT_LE(std::abs(result - expected),
                  std::abs(expected) *
                          (std::is_same_v<T, float>                   ? 3e-6
                           : std::is_same_v<T, hyle::simd::float16_t> ? .001
                                                                      : .008) +
                      4 * std::numeric_limits<float>::denorm_min());
    }
  }
}
TEST(AtenCpuCast, FiniteTailAndInfiniteValueGateUsesStableF32Activation) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  finite_gate<float>();
  finite_gate<hyle::simd::float16_t>();
  finite_gate<hyle::simd::bfloat16_t>();
}
} // namespace
