#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "iq1_original_grid.h"
#include "tensor_test_support.h"
#include "../../src/compute_route_vulkan/vulkan_iq1.h"

namespace vulkan_iq1_test {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace t = vulkan_tensor_test;
inline constexpr std::array types{c::GgufQuantizedType::kIq1Xs,
    c::GgufQuantizedType::kIq1Xxs, c::GgufQuantizedType::kIq1Xxxs};
inline v::Tensor upload_ids(const v::Context &context, const c::TensorDesc &description,
                           std::span<const double> values) {
  std::vector<std::int32_t> ids;
  for (auto value : values) ids.push_back(static_cast<std::int32_t>(value));
  return context.upload(description, std::as_bytes(std::span(ids))).value();
}

struct Inputs {
  c::GgufQuantizedType type;
  int k, n, e, s;
  std::array<c::TensorDesc, 3> descriptions;
  std::vector<std::byte> weight;
  std::vector<double> activation, ids;
  Inputs(c::GgufQuantizedType format, int depth = 1024, int slots = 2, int seed = 0)
      : type(format), k(depth), n(5), e(3), s(slots),
        descriptions{c::gguf_grouped_weight_description(format, c::DataType::kFloat32, e, n, k).value(),
                     t::dense(c::DataType::kFloat32, {2, s, k}),
                     t::dense(c::DataType::kInt32, {2, s})},
        weight(*descriptions[0].required_bytes()), activation(2 * s * k), ids(2 * s) {
    const auto block = block_bytes();
    for (std::size_t offset = 0; offset < weight.size(); offset += block) {
      const auto index = static_cast<unsigned>(offset / block);
      const auto scale = static_cast<std::uint16_t>(0x2000 + ((index + seed) % 4) * 0x400);
      std::memcpy(weight.data() + offset, &scale, 2);
      for (int b = 2; b < block; ++b)
        weight[offset + b] = static_cast<std::byte>((index * 37 + b * 19 + seed * 71) & 255);
    }
    for (std::size_t i = 0; i < activation.size(); ++i)
      activation[i] = static_cast<float>(0.1 * std::cos(0.017 * (i + seed * 3)));
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<double>((i + seed) % e);
    if (ids.size() >= 2) ids[1] = ids[0]; // Same expert selected by independent activation rows.
  }
  int block_bytes() const {
    return type == types[0] ? 46 : type == types[1] ? 42 : 38;
  }
  unsigned byte(std::size_t offset) const {
    return std::to_integer<unsigned>(weight[offset]);
  }
  float decoded(int expert, int col, int j) const {
    const auto base = (static_cast<std::size_t>(expert * n + col) * (k / 256) + j / 256) * block_bytes();
    const auto within = j % 256, group = within / 32, grid_lane = within / 8 % 4;
    const auto index0 = byte(base + 2 + within / 8);
    unsigned index = index0, nibble;
    if (type == types[0]) {
      index |= ((byte(base + 34 + group) >> (2 * grid_lane)) & 3) << 8;
      nibble = (byte(base + 42 + group / 2) >> (4 * (group & 1))) & 15;
    } else if (type == types[1]) {
      const auto qh = byte(base + 34 + group);
      index |= ((qh >> grid_lane) & 1) << 8;
      nibble = (qh >> 4) & 15;
    } else nibble = (byte(base + 34 + group / 2) >> (4 * (group & 1))) & 15;
    const auto word = type == types[0] ? vulkan_iq1_original::xs[index]
                    : type == types[1] ? vulkan_iq1_original::xxs[index]
                                      : vulkan_iq1_original::xxxs[index];
    const auto raw = static_cast<unsigned>((word >> (8 * (j % 8))) & 255);
    const auto grid = raw < 128 ? static_cast<int>(raw) : static_cast<int>(raw) - 256;
    std::uint16_t bits = 0;std::memcpy(&bits, weight.data() + base, 2);
    const auto d = static_cast<float>(t::decode(bits, c::DataType::kFloat16));
    const auto dl = d * static_cast<float>(2 * (nibble & 7) + 1);
    return dl * (static_cast<float>(grid) + ((nibble & 8) ? -0.125F : 0.125F));
  }
  std::vector<double> expected() const {
    std::vector<double> result(2 * s * n);
    for (int row = 0; row < 2 * s; ++row)
      for (int col = 0; col < n; ++col)
        for (int j = 0; j < k; ++j)
          result[row * n + col] += activation[row * k + j] * decoded(static_cast<int>(ids[row]), col, j);
    return result;
  }
  std::array<v::Tensor, 3> upload(const v::Context &context) const {
    return {context.upload(descriptions[0], weight).value(),
            t::make(context, descriptions[1], activation), upload_ids(context, descriptions[2], ids)};
  }
};

template <class Prepare> void batch(Prepare prepare) {
  auto context = vulkan_test::context().value();
  for (auto type : types) for (int k : {256, 1024, 8192}) for (int s : {1, 2}) {
    SCOPED_TRACE(::testing::Message() << "IQ1 format=" << static_cast<unsigned>(type) << " k=" << k << " slots=" << s);
    Inputs first(type, k, s), second(type, k, s, 1);
    auto prepared = prepare(context, c::model_ops::grouped_matmul(), first.descriptions);
    ASSERT_TRUE(prepared) << prepared.error().message();auto invocation = prepared->create_invocation();
    ASSERT_TRUE(invocation) << invocation.error().message();auto slot = std::move(*invocation);
    for (const auto *data : {&first, &second}) {
      ASSERT_TRUE(slot.bind(data->upload(context)).ok());auto work = slot.submit().value();auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();t::expect(context, result->get()->tensors()[0], data->expected());
    }
    for (double bad : {-1.0, 3.0, 2147483647.0}) {
      auto failed = prepared->create_invocation().value();auto inputs = first.upload(context);auto ids = first.ids;ids.back() = bad;
      inputs[2] = upload_ids(context, first.descriptions[2], ids);
      ASSERT_TRUE(failed.bind(inputs).ok());auto work = failed.submit().value();auto result = work.wait();
      ASSERT_FALSE(result);EXPECT_TRUE(result.error().IsOutOfRange()) << result.error().message();EXPECT_FALSE(failed.submit());
    }
  }
}

template <class Prepare> void domains(Prepare prepare) {
  auto context = vulkan_test::context().value();
  struct Shape { c::GgufQuantizedType type;int k, n, e; };
  const std::array shapes{Shape{types[0],3072,3584,896},Shape{types[0],3584,3072,896},
      Shape{types[1],3072,3584,896},Shape{types[1],3584,3072,896},
      Shape{types[2],3072,3584,896},Shape{types[2],3584,3072,896},
      Shape{types[2],8192,2048,512},Shape{types[2],2048,8192,512}};
  for (auto shape : shapes) {
    const auto bank = c::gguf_grouped_weight_description(shape.type, c::DataType::kFloat32, shape.e, shape.n, shape.k).value();
    EXPECT_GT(*bank.element_count(), static_cast<std::size_t>(INT32_MAX));
    const std::array desc{bank, t::dense(c::DataType::kFloat32,{1,2,shape.k}), t::dense(c::DataType::kInt32,{1,2})};
    auto program = prepare(context, c::model_ops::grouped_matmul(), desc);
    EXPECT_TRUE(program) << program.error().message(); // Cold geometry only, not full-bank GPU qualification.
  }
  Inputs data(types[0]);auto &device = *v::detail::Access::runtime(context)->device;
  const auto saved = device.properties.limits.maxComputeWorkGroupInvocations;
  device.properties.limits.maxComputeWorkGroupInvocations = 63;
  EXPECT_FALSE(prepare(context,c::model_ops::grouped_matmul(),data.descriptions));
  device.properties.limits.maxComputeWorkGroupInvocations = saved;
}

template <class Prepare> void lifecycle(Prepare prepare) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),512 * 1024 * 1024);
  {
    auto context = vulkan_test::context(&quota).value();Inputs a(types[2]), b(types[2],1024,2,1);
    auto program = prepare(context,c::model_ops::grouped_matmul(),a.descriptions).value();
    { auto reserved = quota.try_reserve_bytes(quota.limit()-quota.used());ASSERT_TRUE(reserved);
      auto denied = program.create_invocation();ASSERT_FALSE(denied);EXPECT_TRUE(denied.error().IsResourceExhausted()); }
    auto first = program.create_invocation().value(), second = program.create_invocation().value();
    ASSERT_TRUE(first.bind(a.upload(context)).ok());ASSERT_TRUE(second.bind(b.upload(context)).ok());v::Tensor retained;
    { auto one = first.submit().value(), two = second.submit().value();auto r2 = two.wait(), r1 = one.wait();
      ASSERT_TRUE(r1) << r1.error().message();ASSERT_TRUE(r2) << r2.error().message();t::expect(context,r1->get()->tensors()[0],a.expected());t::expect(context,r2->get()->tensors()[0],b.expected());
      retained = r1->get()->tensors()[0]; }
    EXPECT_FALSE(first.submit());
    { auto cancelled = second.submit().value();ASSERT_TRUE(cancelled.request_cancel().ok());auto result = cancelled.wait();
      if (result) t::expect(context,result->get()->tensors()[0],b.expected());else EXPECT_TRUE(result.error().IsCancelled()); }
    { auto replay = second.submit().value();auto result = replay.wait();ASSERT_TRUE(result);t::expect(context,result->get()->tensors()[0],b.expected()); }
    t::expect(context,retained,a.expected());EXPECT_TRUE(context.close().ok());EXPECT_FALSE(second.submit());
  }
  EXPECT_EQ(quota.used(),0U);
}
} // namespace vulkan_iq1_test
