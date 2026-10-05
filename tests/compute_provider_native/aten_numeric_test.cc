#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <thread>
#include <vector>

#include "half_test_support.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/stats.h"

namespace {
using namespace hyle::compute;
using hyle::mem::Buffer;

TensorDesc description(DataType type, std::span<const std::int64_t> shape,
                       bool odd = false) {
  if (!odd)
    return TensorDesc::dense(type, shape).value();
  std::vector<std::int64_t> strides(shape.size());
  auto stride = static_cast<std::int64_t>(data_type_size(type)) + 1;
  for (std::size_t i = shape.size(); i-- != 0;) {
    strides[i] = stride;
    stride = stride * shape[i] + 1;
  }
  return TensorDesc::strided_bytes(type, shape, strides).value();
}
std::size_t offset(const TensorDesc &desc, std::size_t linear) {
  std::size_t result = 0;
  for (std::size_t i = desc.rank(); i-- != 0;) {
    const auto dim = static_cast<std::size_t>(desc.dimensions()[i]);
    result += linear % dim * static_cast<std::size_t>(desc.byte_strides()[i]);
    linear /= dim;
  }
  return result;
}
Buffer storage(const TensorDesc &desc) {
  return Buffer::create(hyle::mem::system_resource(),
                        desc.required_bytes().value(), 64)
      .value();
}
template <class T>
void write(Buffer &buffer, const TensorDesc &desc, std::span<const T> values) {
  std::vector<std::byte> bytes(buffer.size(), std::byte{0xc7});
  for (std::size_t i = 0; i < values.size(); ++i)
    std::memcpy(bytes.data() + offset(desc, i), &values[i], sizeof(T));
  ASSERT_TRUE(buffer.view().write(bytes).ok());
}
template <class T> std::vector<T> read(const Buffer &buffer) {
  std::vector<T> result(buffer.size() / sizeof(T));
  EXPECT_TRUE(
      buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}
Context
context(hyle::mem::AllocatorRef allocator = hyle::mem::system_allocator()) {
  const std::array bindings{ProviderBinding{aten_cpu::provider(), 100}};
  return Context::create(cpu::device(), {.metadata_allocator = allocator,
                                         .providers = bindings})
      .value();
}
double activation_reference(double x, GatedActivationKind kind) {
  if (x == -(std::numeric_limits<double>::infinity)())
    return 0;
  if (kind == GatedActivationKind::kGelu)
    return 0.5 * x * std::erfc(-x / std::numbers::sqrt2);
  double t = x;
  if (kind == GatedActivationKind::kGeluTanh)
    t = 2 * std::sqrt(2 / std::numbers::pi) * (x + 0.044715 * x * x * x);
  const auto e = std::exp(-std::abs(t));
  return x * (t < 0 ? e / (1 + e) : 1 / (1 + e));
}
template <class T> void equal_numeric(T actual, T expected) {
  const auto a = half_test::decode(actual), e = half_test::decode(expected);
  if (std::isnan(e)) {
    EXPECT_TRUE(std::isnan(a));
  } else if (std::isinf(e) || e == 0) {
    EXPECT_EQ(a, e);
    EXPECT_EQ(std::signbit(a), std::signbit(e));
  } else if constexpr (std::is_same_v<T, float>) {
    EXPECT_LE(std::abs(a - e),
              3e-6 * std::abs(e) +
                  4 * (std::numeric_limits<float>::denorm_min)());
  } else {
    // Independent RNE encoder determines the storage result; one storage ULP
    // covers a library transcendental result at a rounding boundary.
    EXPECT_LE(std::abs(a - e),
              std::abs(e) *
                  (std::is_same_v<T, hyle::simd::float16_t> ? 0.001 : 0.008));
  }
}

template <class T> void activations() {
  const auto type = half_test::data_type<T>();
  const auto infinity = (std::numeric_limits<float>::infinity)();
  std::vector<T> gate;
  for (float x : {-infinity, -100.F, -90.F, -40.F, -20.F, -12.F, -8.3625F, -6.F,
                  -5.F, -0.F, 0.F, 1.F, 5.F, 20.F, 100.F, infinity,
                  (std::numeric_limits<float>::quiet_NaN)()})
    gate.push_back(half_test::encode<T>(x));
  for (int i = -200; i <= 200; ++i)
    gate.push_back(half_test::encode<T>(i / 20.0));
  const auto n = static_cast<std::int64_t>(gate.size());
  auto ctx = context();
  auto queue = ctx.create_queue().value();
  for (bool odd : {false, true}) {
    const auto desc = description(type, std::array<std::int64_t, 2>{2, n}, odd);
    std::vector<T> g = gate;
    g.insert(g.end(), gate.begin(), gate.end());
    auto input = storage(desc);
    write(input, desc, std::span<const T>(g));
    const auto view = ConstTensorView::create(desc, input.view()).value();
    for (auto kind : {UnaryKind::kSilu, UnaryKind::kGelu, UnaryKind::kGeluTanh,
                      UnaryKind::kSoftplus}) {
      auto prepared = prepare(ctx, Unary{kind}, desc).value();
      EXPECT_EQ(prepared.provider().id, aten_cpu::kProviderId);
      auto inv = prepared.create_invocation().value();
      auto output = storage(prepared.output_description(0).value());
      const auto out =
          TensorView::create(prepared.output_description(0).value(),
                             output.view())
              .value();
      ASSERT_TRUE(inv.submit(queue, view, out)->wait().ok());
      auto actual = read<T>(output);
      for (std::size_t i = 0; i < g.size(); ++i) {
        SCOPED_TRACE(i);
        const double x = half_test::decode(g[i]);
        const float expected =
            kind == UnaryKind::kSoftplus
                ? static_cast<float>((std::max)(x, 0.0) +
                                     std::log1p(std::exp(-std::abs(x))))
                : static_cast<float>(activation_reference(
                      x, kind == UnaryKind::kSilu ? GatedActivationKind::kSilu
                         : kind == UnaryKind::kGelu
                             ? GatedActivationKind::kGelu
                             : GatedActivationKind::kGeluTanh));
        equal_numeric(actual[i], half_test::encode<T>(expected));
      }
    }
    const auto packed_desc =
        description(type, std::array<std::int64_t, 2>{2, 2 * n}, odd);
    auto packed_buffer = storage(packed_desc), values_buffer = storage(desc);
    for (double scale :
         {0.5, std::is_same_v<T, hyle::simd::float16_t> ? 60000.0 : 1e30}) {
      std::vector<T> values(g.size(), half_test::encode<T>(scale));
      std::vector<T> packed;
      for (int row = 0; row < 2; ++row) {
        packed.insert(packed.end(), gate.begin(), gate.end());
        packed.insert(packed.end(), values.begin() + row * n,
                      values.begin() + (row + 1) * n);
      }
      write(packed_buffer, packed_desc, std::span<const T>(packed));
      write(values_buffer, desc, std::span<const T>(values));
      for (auto kind : {GatedActivationKind::kSilu, GatedActivationKind::kGelu,
                        GatedActivationKind::kGeluTanh})
        for (bool is_packed : {false, true}) {
          const auto op =
              GatedActivation{kind, is_packed ? GatedInputKind::kPackedHalves
                                              : GatedInputKind::kSeparate};
          auto prepared = (is_packed ? prepare(ctx, op, packed_desc)
                                     : prepare(ctx, op, desc, desc))
                              .value();
          auto output = storage(prepared.output_description(0).value());
          auto out = TensorView::create(prepared.output_description(0).value(),
                                        output.view())
                         .value();
          std::vector<ConstTensorView> inputs;
          if (is_packed)
            inputs.push_back(
                ConstTensorView::create(packed_desc, packed_buffer.view())
                    .value());
          else {
            inputs.push_back(view);
            inputs.push_back(
                ConstTensorView::create(desc, values_buffer.view()).value());
          }
          auto inv = prepared.create_invocation().value();
          ASSERT_TRUE(
              inv.submit(queue, inputs, std::span(&out, 1))->wait().ok());
          const auto actual = read<T>(output);
          for (std::size_t i = 0; i < g.size(); ++i) {
            SCOPED_TRACE(i);
            const float activated = static_cast<float>(
                activation_reference(half_test::decode(g[i]), kind));
            const float expected =
                activated * static_cast<float>(half_test::decode(values[i]));
            equal_numeric(actual[i], half_test::encode<T>(expected));
          }
        }
    }
  }
}
TEST(AtenCpuNumeric,
     StableActivationAndGateTailsUseF32IntermediatesAndOneStorageRound) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  activations<float>();
  activations<hyle::simd::float16_t>();
  activations<hyle::simd::bfloat16_t>();
}

template <class T> void numbers() {
  const auto type = half_test::data_type<T>();
  auto ctx = context();
  auto queue = ctx.create_queue().value();
  const float nan = (std::numeric_limits<float>::quiet_NaN)(),
              inf = (std::numeric_limits<float>::infinity)();
  const std::array<float, 8> a{0, -0.F, 0, -0.F, nan, 1, nan, -inf},
      b{-0.F, 0, 0, -0.F, 1, nan, nan, inf};
  const auto ad = description(type, std::array<std::int64_t, 2>{2, 8}, true);
  const auto bd = description(type, std::array<std::int64_t, 2>{1, 8}, true);
  auto ab = storage(ad), bb = storage(bd);
  std::vector<T> av, bv;
  for (int row = 0; row < 2; ++row)
    for (float x : a)
      av.push_back(half_test::encode<T>(x));
  for (float x : b)
    bv.push_back(half_test::encode<T>(x));
  write(ab, ad, std::span<const T>(av));
  write(bb, bd, std::span<const T>(bv));
  std::array inputs{ConstTensorView::create(ad, ab.view()).value(),
                    ConstTensorView::create(bd, bb.view()).value()};
  for (auto kind : {BinaryKind::kMinimum, BinaryKind::kMaximum}) {
    auto p = prepare(ctx, Binary{kind}, ad, bd).value();
    auto inv = p.create_invocation().value();
    auto outb = storage(p.output_description(0).value());
    auto out = TensorView::create(p.output_description(0).value(), outb.view())
                   .value();
    ASSERT_TRUE(inv.submit(queue, inputs, std::span(&out, 1))->wait().ok());
    auto actual = read<T>(outb);
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const auto x = a[i % 8], y = b[i % 8];
      float expected =
          kind == BinaryKind::kMinimum ? std::fmin(x, y) : std::fmax(x, y);
      if (x == 0 && y == 0)
        expected =
            (kind == BinaryKind::kMinimum ? std::signbit(x) || std::signbit(y)
                                          : std::signbit(x) && std::signbit(y))
                ? -0.F
                : 0.F;
      equal_numeric(actual[i], half_test::encode<T>(expected));
    }
  }
}
TEST(AtenCpuNumeric,
     NumberMinMaxPreserveNaNAndBothZeroOrdersAcrossBroadcastOddBytes) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  numbers<float>();
  numbers<hyle::simd::float16_t>();
  numbers<hyle::simd::bfloat16_t>();
}

template <class T> void reductions() {
  const auto type = half_test::data_type<T>();
  auto ctx = context();
  auto queue = ctx.create_queue().value();
  const float nan = (std::numeric_limits<float>::quiet_NaN)();
  for (const auto shape : std::vector<std::vector<std::int64_t>>{
           {}, {2, 3, 4}, {2, 0, 4}, {1, 1, 1, 1, 1, 2, 3, 4}}) {
    for (bool odd : {false, true}) {
      auto desc = description(type, shape, odd);
      auto inb = storage(desc);
      std::vector<T> values(desc.element_count().value());
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = half_test::encode<T>(
            static_cast<float>(static_cast<int>(i % 9) - 4) / 4);
      write(inb, desc, std::span<const T>(values));
      auto input = ConstTensorView::create(desc, inb.view()).value();
      std::vector<Axes> all{
          Axes::all(), Axes::from(std::span<const std::int32_t>{}).value()};
      if (!shape.empty())
        all.push_back(Axes::from(std::array<std::int32_t, 1>{-1}).value());
      if (shape.size() > 1)
        all.push_back(Axes::from(std::array<std::int32_t, 2>{0, -1}).value());
      for (const auto axes : all)
        for (bool keep : {false, true})
          for (auto kind : {ReductionKind::kSum, ReductionKind::kMean,
                            ReductionKind::kMinimum, ReductionKind::kMaximum}) {
            const Reduce op{.kind = kind,
                            .axes = axes,
                            .keep_dimensions = keep,
                            .accumulation_type = DataType::kFloat32};
            auto p = prepare(ctx, op, desc).value();
            auto outb = storage(p.output_description(0).value());
            auto out =
                TensorView::create(p.output_description(0).value(), outb.view())
                    .value();
            auto inv = p.create_invocation().value();
            ASSERT_TRUE(inv.submit(queue, input, out)->wait().ok());
            std::vector<bool> selected(shape.size(), axes.is_all());
            if (!axes.is_all())
              for (auto axis : axes.values())
                selected[axis < 0 ? axis + shape.size() : axis] = true;
            const auto output = p.output_description(0).value();
            std::vector<float> accum(
                output.element_count().value(),
                kind == ReductionKind::kMinimum
                    ? (std::numeric_limits<float>::infinity)()
                : kind == ReductionKind::kMaximum
                    ? -(std::numeric_limits<float>::infinity)()
                    : 0);
            std::size_t count = 1;
            for (std::size_t i = 0; i < shape.size(); ++i)
              if (selected[i])
                count *= shape[i];
            for (std::size_t linear = 0; linear < values.size(); ++linear) {
              std::vector<std::size_t> coordinates(shape.size());
              auto rest = linear;
              for (std::size_t i = shape.size(); i-- != 0;) {
                coordinates[i] = rest % shape[i];
                rest /= shape[i];
              }
              std::size_t target = 0;
              for (std::size_t i = 0; i < shape.size(); ++i)
                if (!selected[i])
                  target = target * shape[i] + coordinates[i];
              const auto value =
                  static_cast<float>(half_test::decode(values[linear]));
              if (kind == ReductionKind::kMinimum)
                accum[target] = std::fmin(accum[target], value);
              else if (kind == ReductionKind::kMaximum)
                accum[target] = std::fmax(accum[target], value);
              else
                accum[target] += value;
            }
            if (kind == ReductionKind::kMean)
              for (auto &x : accum)
                x = count == 0 ? nan : x / static_cast<float>(count);
            auto actual = read<T>(outb);
            for (std::size_t i = 0; i < actual.size(); ++i)
              equal_numeric(actual[i], half_test::encode<T>(accum[i]));
          }
    }
  }
  // Mixed and all-negative zero rows plus a NaN row: both orders and keep-dim.
  auto desc = description(type, std::array<std::int64_t, 2>{3, 4}, true);
  auto inb = storage(desc);
  std::vector<T> values;
  for (float x :
       {0.F, -0.F, 0.F, 0.F, -0.F, -0.F, -0.F, -0.F, 1.F, nan, 2.F, 3.F})
    values.push_back(half_test::encode<T>(x));
  write(inb, desc, std::span<const T>(values));
  auto input = ConstTensorView::create(desc, inb.view()).value();
  for (bool keep : {false, true})
    for (auto kind : {ReductionKind::kMinimum, ReductionKind::kMaximum}) {
      auto p =
          prepare(
              ctx,
              Reduce{.kind = kind,
                     .axes = Axes::from(std::array<std::int32_t, 1>{1}).value(),
                     .keep_dimensions = keep,
                     .accumulation_type = DataType::kFloat32},
              desc)
              .value();
      auto outb = storage(p.output_description(0).value());
      auto out =
          TensorView::create(p.output_description(0).value(), outb.view())
              .value();
      auto inv = p.create_invocation().value();
      ASSERT_TRUE(inv.submit(queue, input, out)->wait().ok());
      const auto actual = read<T>(outb);
      equal_numeric(
          actual[0],
          half_test::encode<T>(kind == ReductionKind::kMinimum ? -0.F : 0.F));
      equal_numeric(actual[1], half_test::encode<T>(-0.F));
      EXPECT_TRUE(std::isnan(half_test::decode(actual[2])));
    }
}
TEST(AtenCpuNumeric,
     ArbitraryAxesRankZeroEightEmptyIdentityAndZeroNaNReduction) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  reductions<float>();
  reductions<hyle::simd::float16_t>();
  reductions<hyle::simd::bfloat16_t>();
}

TEST(AtenCpuNumeric,
     AliasRuntimeCostsCloseAndPrivateNumericInvocationsAreExplicit) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto ctx = context(stats.get());
    auto queue = ctx.create_queue().value();
    const auto desc =
        description(DataType::kFloat32, std::array<std::int64_t, 1>{3});
    auto buffer = storage(desc);
    const std::array<float, 3> data{-5, -0.F, 5};
    write(buffer, desc, std::span<const float>(data));
    auto view = TensorView::create(desc, buffer.view()).value();
    auto p = prepare(ctx, Unary{UnaryKind::kGelu}, desc).value();
    EXPECT_TRUE(p.alias_support(0, 0)->supported);
    EXPECT_EQ(p.runtime_properties().replay_class,
              ProviderReplayClass::kManaged);
    EXPECT_TRUE(p.runtime_properties().uses_private_threads);
    EXPECT_FALSE(p.runtime_properties().capture_compatible);
    auto inv = p.create_invocation().value();
    ASSERT_TRUE(inv.submit(queue, view.as_const(), view)->wait().ok());
    auto actual = read<float>(buffer);
    for (std::size_t i = 0; i < data.size(); ++i)
      equal_numeric(actual[i], static_cast<float>(activation_reference(
                                   data[i], GatedActivationKind::kGelu)));
    auto binary =
        prepare(ctx, Binary{BinaryKind::kMaximum}, desc, desc).value();
    EXPECT_TRUE(binary.alias_support(0, 0)->supported);
    EXPECT_TRUE(binary.alias_support(0, 1)->supported);
    auto gate =
        prepare(ctx,
                GatedActivation{.activation = GatedActivationKind::kGelu,
                                .inputs = GatedInputKind::kSeparate},
                desc, desc)
            .value();
    EXPECT_FALSE(gate.alias_support(0, 0)->supported);
    auto reduce = prepare(ctx, Reduce{}, desc).value();
    EXPECT_FALSE(reduce.alias_support(0, 0)->supported);
    const auto strided =
        description(DataType::kFloat32, std::array<std::int64_t, 1>{3}, true);
    auto sp = prepare(ctx, Unary{UnaryKind::kGelu}, strided).value();
    EXPECT_FALSE(sp.alias_support(0, 0)->supported);
    // Each packed native input belongs to one invocation; changed sources do
    // not change another invocation's already submitted output.
    auto a = p.create_invocation().value(), b = p.create_invocation().value();
    for (float seed : {1.F, 3.F}) {
      const std::array<float, 3> changed{seed, seed + 1, seed + 2};
      write(buffer, desc, std::span<const float>(changed));
      ASSERT_TRUE(a.submit(queue, view.as_const(), view)->wait().ok());
      write(buffer, desc, std::span<const float>(data));
      ASSERT_TRUE(b.submit(queue, view.as_const(), view)->wait().ok());
      actual = read<float>(buffer);
      for (std::size_t i = 0; i < data.size(); ++i)
        equal_numeric(actual[i], static_cast<float>(activation_reference(
                                     data[i], GatedActivationKind::kGelu)));
    }
    ASSERT_TRUE(ctx.close().ok());
    auto late = inv.submit(queue, view.as_const(), view);
    EXPECT_FALSE(late);
    const auto half =
        description(DataType::kFloat16, std::array<std::int64_t, 1>{3});
    auto fresh = context();
    EXPECT_FALSE(prepare(fresh, Reduce{}, half));
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(AtenCpuNumeric,
     NumericInputPackingSlotsRemainPrivateDuringConcurrentReplay) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto ctx = context();
  auto q0 = ctx.create_queue().value(), q1 = ctx.create_queue().value();
  const auto d =
      description(DataType::kFloat32, std::array<std::int64_t, 1>{31}, true);
  auto p = prepare(ctx, Unary{UnaryKind::kGeluTanh}, d).value();
  auto in0 = storage(d), in1 = storage(d);
  auto out0 = storage(p.output_description().value());
  auto out1 = storage(p.output_description().value());
  const auto i0 = ConstTensorView::create(d, in0.view()).value();
  const auto i1 = ConstTensorView::create(d, in1.view()).value();
  const auto o0 =
      TensorView::create(p.output_description().value(), out0.view()).value();
  const auto o1 =
      TensorView::create(p.output_description().value(), out1.view()).value();
  auto a = p.create_invocation().value(), b = p.create_invocation().value();
  std::atomic<bool> okay{true};
  const auto run = [&](auto &inv, auto &queue, auto &input_buffer,
                       const auto &input, const auto &output,
                       auto &output_buffer, float shift) {
    for (int round = 0; round < 12; ++round) {
      std::vector<float> values(31);
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = shift + static_cast<float>(i + round) / 16;
      write(input_buffer, d, std::span<const float>(values));
      auto done = inv.submit(queue, input, output);
      if (!done || !done->wait().ok()) {
        okay.store(false);
        return;
      }
      const auto actual = read<float>(output_buffer);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto expected = static_cast<float>(
            activation_reference(values[i], GatedActivationKind::kGeluTanh));
        if (std::abs(actual[i] - expected) > std::abs(expected) * 3e-6F)
          okay.store(false);
      }
    }
  };
  std::thread first([&] { run(a, q0, in0, i0, o0, out0, -6.F); });
  std::thread second([&] { run(b, q1, in1, i1, o1, out1, 1.F); });
  first.join();
  second.join();
  EXPECT_TRUE(okay.load());
}

struct FaultAllocator final {
  hyle::mem::AllocatorRef backing;
  std::size_t attempts = 0, fail = SIZE_MAX;
  hyle::mem::AllocatorRef get() noexcept {
    return {[](void *self, std::size_t bytes,
               std::size_t alignment) noexcept -> void * {
              auto &owner = *static_cast<FaultAllocator *>(self);
              return owner.attempts++ == owner.fail
                         ? nullptr
                         : owner.backing.allocate(bytes, alignment);
            },
            [](void *self, void *data, std::size_t bytes,
               std::size_t alignment) noexcept {
              static_cast<FaultAllocator *>(self)->backing.deallocate(
                  data, bytes, alignment);
            },
            [](void *self, void *data, std::size_t old_bytes, std::size_t bytes,
               std::size_t alignment) noexcept -> void * {
              auto &owner = *static_cast<FaultAllocator *>(self);
              return bytes != 0 && owner.attempts++ == owner.fail
                         ? nullptr
                         : owner.backing.reallocate(data, old_bytes, bytes,
                                                    alignment);
            },
            this};
  }
};
TEST(AtenCpuNumeric,
     TrackedColdNumericAllocationFailuresRecoverAndReleaseResources) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  FaultAllocator owner{.backing = stats.get()};
  {
    auto ctx = context(owner.get());
    const auto d = description(DataType::kBFloat16,
                               std::array<std::int64_t, 2>{2, 7}, true);
    const auto begin = owner.attempts;
    {
      auto p = prepare(ctx, Unary{UnaryKind::kSoftplus}, d);
      ASSERT_TRUE(p);
    }
    const auto count = owner.attempts - begin;
    for (std::size_t i = 0; i < count; ++i) {
      owner.fail = owner.attempts + i;
      auto p = prepare(ctx, Unary{UnaryKind::kSoftplus}, d);
      ASSERT_FALSE(p);
      EXPECT_TRUE(p.error().IsResourceExhausted());
      owner.fail = SIZE_MAX;
    }
    auto p = prepare(ctx, Unary{UnaryKind::kSoftplus}, d).value();
    const auto start = owner.attempts;
    {
      auto inv = p.create_invocation();
      ASSERT_TRUE(inv);
    }
    const auto slots = owner.attempts - start;
    for (std::size_t i = 0; i < slots; ++i) {
      owner.fail = owner.attempts + i;
      auto inv = p.create_invocation();
      ASSERT_FALSE(inv);
      EXPECT_TRUE(inv.error().IsResourceExhausted());
      owner.fail = SIZE_MAX;
    }
    EXPECT_TRUE(p.create_invocation());
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
