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
    for (auto kind : {UnaryKind::kSigmoid, UnaryKind::kSilu, UnaryKind::kGelu,
                      UnaryKind::kGeluTanh, UnaryKind::kSoftplus}) {
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
            kind == UnaryKind::kSigmoid
                ? static_cast<float>(x < 0 ? std::exp(x) / (1 + std::exp(x))
                                           : 1 / (1 + std::exp(-x)))
            : kind == UnaryKind::kSoftplus
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
  for (const auto &shape : std::vector<std::vector<std::int64_t>>{
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
      std::vector<Axes> all{Axes::all()};
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

template <class T>
void scatter_add_case(std::span<const std::int64_t> base_shape,
                      std::span<const std::int64_t> index_shape,
                      DataType index_type, std::int32_t axis, bool odd,
                      bool alias = false) {
  const auto base = description(half_test::data_type<T>(), base_shape, odd);
  const auto index = description(index_type, index_shape, odd);
  const auto update_shape = infer_single_output(Operation(Gather{.axis = axis}),
                                                std::array{base, index})
                                .value();
  const auto updates =
      description(base.data_type(), update_shape.dimensions(), odd);
  auto ctx = context();
  auto queue = ctx.create_queue().value();
  auto p = prepare(ctx, Scatter{.axis = axis, .kind = ScatterKind::kAdd}, base,
                   index, updates)
               .value();
  EXPECT_EQ(p.provider().id, aten_cpu::kProviderId);
  EXPECT_EQ(p.runtime_properties().replay_class, ProviderReplayClass::kManaged);
  EXPECT_EQ(p.alias_support(0, 0)->supported, base.is_dense());
  EXPECT_FALSE(p.alias_support(0, 2)->supported);
  auto inv = p.create_invocation().value();
  auto b = storage(base), i = storage(index), u = storage(updates);
  auto o = storage(p.output_description().value());
  const auto output = TensorView::create(p.output_description().value(),
                                         alias ? b.view() : o.view())
                          .value();
  const std::array inputs{ConstTensorView::create(base, b.view()).value(),
                          ConstTensorView::create(index, i.view()).value(),
                          ConstTensorView::create(updates, u.view()).value()};
  const auto positive_axis = static_cast<std::size_t>(
      axis < 0 ? axis + static_cast<std::int32_t>(base.rank()) : axis);
  const auto extent = base.dimensions()[positive_axis];
  const auto count = index.element_count().value();
  std::vector<std::int64_t> indices(count);
  for (std::size_t n = 0; n < count; ++n)
    indices[n] = n % 2 == 0 ? extent - 1 : 0;
  if (index_type == DataType::kInt64)
    write(i, index, std::span<const std::int64_t>(indices));
  else {
    const std::vector<std::int32_t> small(indices.begin(), indices.end());
    write(i, index, std::span<const std::int32_t>(small));
  }
  std::size_t prefix = 1, suffix = 1;
  for (std::size_t n = 0; n < positive_axis; ++n)
    prefix *= static_cast<std::size_t>(base.dimensions()[n]);
  for (std::size_t n = positive_axis + 1; n < base.rank(); ++n)
    suffix *= static_cast<std::size_t>(base.dimensions()[n]);
  for (int replay = 0; replay < 2; ++replay) {
    std::vector<T> initial(base.element_count().value());
    std::vector<T> added(updates.element_count().value());
    for (std::size_t n = 0; n < initial.size(); ++n)
      initial[n] = half_test::encode<T>(float(n + replay) * .125F);
    for (std::size_t n = 0; n < added.size(); ++n)
      added[n] = half_test::encode<T>(float(n % 7 + replay) * .25F);
    write(b, base, std::span<const T>(initial));
    write(u, updates, std::span<const T>(added));
    std::vector<float> expected(initial.size());
    for (std::size_t n = 0; n < initial.size(); ++n)
      expected[n] = static_cast<float>(half_test::decode(initial[n]));
    for (std::size_t outer = 0; outer < prefix; ++outer)
      for (std::size_t n = 0; n < count; ++n)
        for (std::size_t inner = 0; inner < suffix; ++inner) {
          const auto destination = (outer * static_cast<std::size_t>(extent) +
                                    static_cast<std::size_t>(indices[n])) *
                                       suffix +
                                   inner;
          expected[destination] += static_cast<float>(
              half_test::decode(added[(outer * count + n) * suffix + inner]));
        }
    auto done = inv.submit(queue, inputs, output);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = read<T>(alias ? b : o);
    for (std::size_t n = 0; n < actual.size(); ++n)
      equal_numeric(actual[n], half_test::encode<T>(expected[n]));
  }
}
template <class T> void scatter_add_shapes() {
  for (auto index : {DataType::kInt32, DataType::kInt64})
    for (bool odd : {false, true}) {
      for (int axis = 0; axis < 3; ++axis) {
        const std::array<std::int64_t, 3> base{2, 4, 3};
        scatter_add_case<T>(base, std::array<std::int64_t, 2>{2, 2}, index,
                            axis, odd);
        scatter_add_case<T>(base, std::span<const std::int64_t>{}, index,
                            axis - 3, odd);
        scatter_add_case<T>(base, std::array<std::int64_t, 1>{0}, index, axis,
                            odd);
      }
      scatter_add_case<T>(std::array<std::int64_t, 8>{1, 1, 1, 1, 1, 1, 4, 3},
                          std::array<std::int64_t, 1>{4}, index, -2, odd);
    }
  scatter_add_case<T>(std::array<std::int64_t, 2>{4, 3},
                      std::array<std::int64_t, 1>{4}, DataType::kInt64, 0,
                      false, true);
}
TEST(AtenCpuNumeric, ScatterAddUsesMatureMathForBothIndexWidthsAndAllAxes) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  scatter_add_shapes<float>();
  scatter_add_shapes<hyle::simd::float16_t>();
  scatter_add_shapes<hyle::simd::bfloat16_t>();
}

TEST(AtenCpuNumeric, ScatterAddChecksLateIndicesBeforeWritingAndRecovers) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto ctx = context(stats.get());
    auto queue = ctx.create_queue().value();
    const auto bdesc =
        description(DataType::kFloat32, std::array<std::int64_t, 1>{3});
    const auto idesc =
        description(DataType::kInt64, std::array<std::int64_t, 1>{3}, true);
    auto p =
        prepare(ctx, Scatter{.kind = ScatterKind::kAdd}, bdesc, idesc, bdesc)
            .value();
    auto inv = p.create_invocation().value();
    auto b = storage(bdesc), i = storage(idesc), u = storage(bdesc);
    auto o = storage(bdesc);
    const std::array inputs{ConstTensorView::create(bdesc, b.view()).value(),
                            ConstTensorView::create(idesc, i.view()).value(),
                            ConstTensorView::create(bdesc, u.view()).value()};
    const auto output = TensorView::create(bdesc, o.view()).value();
    const std::array<float, 3> base{1, 2, 3}, updates{4, 5, 6};
    write(b, bdesc, std::span<const float>(base));
    write(u, bdesc, std::span<const float>(updates));
    const std::array<float, 3> poison{-7, -7, -7};
    for (auto bad : {std::int64_t{-1}, std::int64_t{3}, INT64_MAX}) {
      const std::array<std::int64_t, 3> indices{2, 0, bad};
      write(i, idesc, std::span<const std::int64_t>(indices));
      write(o, bdesc, std::span<const float>(poison));
      auto done = inv.submit(queue, inputs, output);
      ASSERT_TRUE(done);
      EXPECT_TRUE(done->wait().IsOutOfRange());
      EXPECT_EQ(read<float>(o),
                std::vector<float>(poison.begin(), poison.end()));
      auto stale = inv.submit(queue, inputs, output);
      ASSERT_FALSE(stale);
      EXPECT_TRUE(stale.error().IsFailedPrecondition());
      inv = p.create_invocation().value();
    }
    const std::array<std::int64_t, 3> indices{2, 0, 2};
    write(i, idesc, std::span<const std::int64_t>(indices));
    ASSERT_TRUE(inv.submit(queue, inputs, output)->wait().ok());
    EXPECT_EQ(read<float>(o), (std::vector<float>{6, 2, 13}));
    EXPECT_TRUE(ctx.close().ok());
    EXPECT_FALSE(inv.submit(queue, inputs, output));
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(AtenCpuNumeric, ScatterAddEmptyIndicesPreserveBitsAndAcceptsEmptyOutput) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto ctx = context();
  auto queue = ctx.create_queue().value();
  const auto bdesc =
      description(DataType::kFloat16, std::array<std::int64_t, 1>{3});
  const auto idesc =
      description(DataType::kInt64, std::array<std::int64_t, 1>{0});
  const auto udesc =
      description(DataType::kFloat16, std::array<std::int64_t, 1>{0});
  auto p = prepare(ctx, Scatter{.kind = ScatterKind::kAdd}, bdesc, idesc, udesc)
               .value();
  auto inv = p.create_invocation().value();
  auto b = storage(bdesc), i = storage(idesc), u = storage(udesc);
  auto o = storage(bdesc);
  const std::array<std::uint16_t, 3> bits{0x8000, 0x7c01, 0xffff};
  write(b, bdesc, std::span<const std::uint16_t>(bits));
  const std::array inputs{ConstTensorView::create(bdesc, b.view()).value(),
                          ConstTensorView::create(idesc, i.view()).value(),
                          ConstTensorView::create(udesc, u.view()).value()};
  const auto output = TensorView::create(bdesc, o.view()).value();
  ASSERT_TRUE(inv.submit(queue, inputs, output)->wait().ok());
  EXPECT_EQ(read<std::uint16_t>(o),
            std::vector<std::uint16_t>(bits.begin(), bits.end()));
  scatter_add_case<float>(std::array<std::int64_t, 2>{0, 4},
                          std::array<std::int64_t, 1>{0}, DataType::kInt64, 0,
                          false);
}

template <class T> void scatter_add_special() {
  const auto infinity = (std::numeric_limits<float>::infinity)();
  const auto nan = (std::numeric_limits<float>::quiet_NaN)();
  auto ctx = context();
  auto queue = ctx.create_queue().value();
  const auto desc = description(half_test::data_type<T>(),
                                std::array<std::int64_t, 1>{4}, true);
  const auto idesc =
      description(DataType::kInt64, std::array<std::int64_t, 1>{4}, true);
  auto p = prepare(ctx, Scatter{.kind = ScatterKind::kAdd}, desc, idesc, desc)
               .value();
  auto inv = p.create_invocation().value();
  auto b = storage(desc), i = storage(idesc), u = storage(desc);
  auto o = storage(p.output_description().value());
  const std::array<T, 4> base{
      half_test::encode<T>(nan), half_test::encode<T>(infinity),
      half_test::encode<T>(-infinity), half_test::encode<T>(16)};
  const std::array<T, 4> added{
      half_test::encode<T>(2), half_test::encode<T>(-infinity),
      half_test::encode<T>(3), half_test::encode<T>(.015625)};
  const std::array<std::int64_t, 4> indices{0, 1, 2, 3};
  write(b, desc, std::span<const T>(base));
  write(i, idesc, std::span<const std::int64_t>(indices));
  write(u, desc, std::span<const T>(added));
  const std::array inputs{ConstTensorView::create(desc, b.view()).value(),
                          ConstTensorView::create(idesc, i.view()).value(),
                          ConstTensorView::create(desc, u.view()).value()};
  ASSERT_TRUE(
      inv.submit(queue, inputs,
                 TensorView::create(p.output_description().value(), o.view())
                     .value())
          ->wait()
          .ok());
  const auto actual = read<T>(o);
  EXPECT_TRUE(std::isnan(half_test::decode(actual[0])));
  EXPECT_TRUE(std::isnan(half_test::decode(actual[1])));
  EXPECT_EQ(half_test::decode(actual[2]), -infinity);
  equal_numeric(actual[3], half_test::encode<T>(16.015625));
}
TEST(AtenCpuNumeric, ScatterAddKeepsFloatingSpecialValueSemantics) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  scatter_add_special<float>();
  scatter_add_special<hyle::simd::float16_t>();
  scatter_add_special<hyle::simd::bfloat16_t>();
}

TEST(AtenCpuNumeric, ScatterAddPrivateLanesReplayConcurrently) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto ctx = context(stats.get());
    const auto desc =
        description(DataType::kFloat32, std::array<std::int64_t, 1>{3});
    const auto idesc =
        description(DataType::kInt64, std::array<std::int64_t, 1>{3}, true);
    auto p = prepare(ctx, Scatter{.kind = ScatterKind::kAdd}, desc, idesc, desc)
                 .value();
    std::array queues{ctx.create_queue().value(), ctx.create_queue().value()};
    std::array lanes{p.create_invocation().value(),
                     p.create_invocation().value()};
    std::array base{storage(desc), storage(desc)},
        ids{storage(idesc), storage(idesc)},
        updates{storage(desc), storage(desc)},
        output{storage(desc), storage(desc)};
    for (int replay = 0; replay < 4; ++replay) {
      for (std::size_t slot = 0; slot < 2; ++slot) {
        const std::array<float, 3> b{float(replay + slot), 2, 3};
        const std::array<float, 3> u{float(slot + 4), 5, 6};
        const std::array<std::int64_t, 3> index{2, 0, 2};
        write(base[slot], desc, std::span<const float>(b));
        write(updates[slot], desc, std::span<const float>(u));
        write(ids[slot], idesc, std::span<const std::int64_t>(index));
      }
      std::array<hyle::Status, 2> status;
      std::array<std::thread, 2> threads;
      for (std::size_t slot = 0; slot < 2; ++slot)
        threads[slot] = std::thread([&, slot] {
          const std::array inputs{
              ConstTensorView::create(desc, base[slot].view()).value(),
              ConstTensorView::create(idesc, ids[slot].view()).value(),
              ConstTensorView::create(desc, updates[slot].view()).value()};
          auto done = lanes[slot].submit(
              queues[slot], inputs,
              TensorView::create(desc, output[slot].view()).value());
          if (done)
            status[slot] = done->wait();
          else
            status[slot] = std::move(done).error();
        });
      for (auto &thread : threads)
        thread.join();
      for (std::size_t slot = 0; slot < 2; ++slot) {
        ASSERT_TRUE(status[slot].ok()) << status[slot].message();
        EXPECT_EQ(read<float>(output[slot]),
                  (std::vector<float>{float(replay + slot + 5), 2,
                                      float(slot + 13)}));
      }
    }
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(AtenCpuNumeric, ScatterAddColdOomRestoresLiveBytesAndRetries) {
  if (!aten_cpu::provider())
    GTEST_SKIP();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  FaultAllocator owner{.backing = stats.get()};
  {
    auto ctx = context(owner.get());
    const auto desc = description(DataType::kBFloat16,
                                  std::array<std::int64_t, 2>{4, 3}, true);
    const auto idesc =
        description(DataType::kInt64, std::array<std::int64_t, 1>{4}, true);
    const auto op = Scatter{.kind = ScatterKind::kAdd};
    const auto start = owner.attempts;
    {
      auto p = prepare(ctx, op, desc, idesc, desc);
      ASSERT_TRUE(p);
    }
    const auto attempts = owner.attempts - start;
    ASSERT_GT(attempts, 0U);
    for (std::size_t failure = 0; failure < attempts; ++failure) {
      const auto live = stats.snapshot().live_bytes;
      owner.fail = owner.attempts + failure;
      auto p = prepare(ctx, op, desc, idesc, desc);
      ASSERT_FALSE(p);
      EXPECT_TRUE(p.error().IsResourceExhausted());
      EXPECT_EQ(stats.snapshot().live_bytes, live);
      owner.fail = SIZE_MAX;
    }
    auto p = prepare(ctx, op, desc, idesc, desc).value();
    const auto begin = owner.attempts;
    {
      auto inv = p.create_invocation();
      ASSERT_TRUE(inv);
    }
    const auto count = owner.attempts - begin;
    ASSERT_GT(count, 0U);
    for (std::size_t failure = 0; failure < count; ++failure) {
      const auto live = stats.snapshot().live_bytes;
      owner.fail = owner.attempts + failure;
      auto inv = p.create_invocation();
      ASSERT_FALSE(inv);
      EXPECT_TRUE(inv.error().IsResourceExhausted());
      EXPECT_EQ(stats.snapshot().live_bytes, live);
      owner.fail = SIZE_MAX;
      EXPECT_TRUE(p.create_invocation());
    }
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
