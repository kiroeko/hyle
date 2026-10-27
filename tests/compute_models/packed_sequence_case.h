#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/sequence_ops.h"

namespace packed_sequence_test {
namespace c = hyle::compute;
namespace m = hyle::mem;

inline c::TensorDesc desc(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, std::span(shape.begin(), shape.size())).value();
}

template <class Value> std::vector<std::byte> bytes(const std::vector<Value> &values) {
  const auto source = std::as_bytes(std::span(values));
  return {source.begin(), source.end()};
}

template <class Value> std::vector<Value> values(std::size_t count, int phase = 0) {
  std::vector<Value> result(count);
  for (std::size_t i = 0; i != count; ++i)
    result[i] = half_test::encode<Value>(double(int((i * 7 + std::size_t(phase)) % 19) - 9) / 32.0);
  return result;
}

// One original prepared operation/invocation, rebound through public views and
// a real Queue. Empty tensors still have ordinary valid zero-byte buffers.
struct Case final {
  c::Queue queue;
  c::PreparedOperation prepared;
  c::OperationInvocation invocation;
  std::vector<m::Buffer> input_buffers;
  std::array<m::Buffer, 2> output_buffers;
  std::vector<c::ConstTensorView> inputs;
  std::array<c::TensorView, 2> outputs;
  std::array<c::TensorDesc, 2> output_desc;

  Case(c::Context &context, m::MemoryResourceRef resource, c::Operation operation,
       std::span<const c::TensorDesc> descriptions,
       const std::vector<std::vector<std::byte>> &data)
      : queue(context.create_queue().value()) {
    const auto inferred = c::infer_output_descriptions(operation, descriptions, output_desc);
    EXPECT_TRUE(inferred.ok()) << inferred.message();
    prepared = c::prepare(context, operation, descriptions, output_desc, resource).value();
    EXPECT_EQ(prepared.workspace_requirement().size, 0U);
    invocation = prepared.create_invocation().value();
    input_buffers.reserve(descriptions.size());
    inputs.reserve(descriptions.size());
    for (std::size_t i = 0; i != descriptions.size(); ++i) {
      input_buffers.push_back(m::Buffer::create(resource, descriptions[i].required_bytes().value(), 256).value());
      EXPECT_EQ(data[i].size(), input_buffers.back().size());
      EXPECT_TRUE(input_buffers.back().view().write(data[i]).ok());
      inputs.push_back(c::ConstTensorView::create(descriptions[i], input_buffers.back().view()).value());
    }
    for (std::size_t i = 0; i != outputs.size(); ++i) {
      output_buffers[i] = m::Buffer::create(resource, output_desc[i].required_bytes().value() + 32, 256).value();
      EXPECT_TRUE(output_buffers[i].view().fill(std::byte{0xa5}).ok());
      outputs[i] = c::TensorView::create(output_desc[i], output_buffers[i].view()).value();
    }
  }

  void offsets(std::span<const std::int32_t> selected) {
    EXPECT_TRUE(input_buffers.back().view().write(std::as_bytes(selected)).ok());
  }

  hyle::Status submit() {
    auto done = invocation.submit(queue, inputs, outputs);
    if (!done) return done.error();
    auto status = done->wait();
    expect_tail();
    return status;
  }

  template <class Value> std::vector<Value> read(std::size_t index) const {
    std::vector<Value> result(output_desc[index].required_bytes().value() / sizeof(Value));
    EXPECT_TRUE(outputs[index].storage().read(std::as_writable_bytes(std::span(result))).ok());
    return result;
  }

  void expect_tail() const {
    for (std::size_t i = 0; i != output_buffers.size(); ++i) {
      std::array<std::byte, 32> actual{};
      EXPECT_TRUE(output_buffers[i].view().subview(output_desc[i].required_bytes().value(),32)
                      .value().read(actual).ok());
      for (const auto value : actual) EXPECT_EQ(value,std::byte{0xa5});
    }
  }

  void poison() {
    for (auto &output : outputs) EXPECT_TRUE(output.storage().fill(std::byte{0x5a}).ok());
  }

  void expect_poison() {
    for (std::size_t i = 0; i != outputs.size(); ++i)
      EXPECT_EQ(read<std::byte>(i), std::vector<std::byte>(output_desc[i].required_bytes().value(),std::byte{0x5a}));
    expect_tail();
  }
};

template <class Value> void expect_value(Value actual, double reference) {
  const double rounded = half_test::decode(half_test::encode<Value>(reference));
  constexpr double tolerance = std::is_same_v<Value, float> ? 0.000003 :
                               std::is_same_v<Value, hyle::simd::float16_t> ? 0.001 : 0.009;
  EXPECT_NEAR(half_test::decode(actual), rounded, tolerance * (1 + std::abs(rounded)));
}

template <class Value>
void gdn_math(c::Context &context, m::MemoryResourceRef resource, std::size_t dk = 3,
              std::optional<float> output_scale = 1.0F) {
  constexpr std::size_t rows = 8, batches = 2, hk = 2, hv = 4, dv = 2;
  const auto type = half_test::data_type<Value>();
  auto q = values<Value>(rows * hk * dk, 1), k = values<Value>(rows * hk * dk, 2);
  auto v = values<Value>(rows * hv * dv, 3);
  auto initial = values<float>(batches * hv * dv * dk, 4);
  // Padding contains NaN. A correct packed operation never reads these rows.
  for (std::size_t i = 4 * hk * dk; i != q.size(); ++i) q[i] = k[i] = half_test::encode<Value>(NAN);
  for (std::size_t i = 4 * hv * dv; i != v.size(); ++i) v[i] = half_test::encode<Value>(NAN);
  for (const auto gate_depth : {std::size_t{1}, dv}) {
    SCOPED_TRACE(gate_depth);
    const auto f32 = c::DataType::kFloat32;
    const std::array descriptions{desc(type,{rows,hk,std::int64_t(dk)}), desc(type,{rows,hk,std::int64_t(dk)}),
        desc(type,{rows,hv,dv}), desc(f32,{rows,hv,std::int64_t(gate_depth)}),
        desc(f32,{rows,hv,1}), desc(f32,{batches,hv,dv,std::int64_t(dk)}), desc(c::DataType::kInt32,{batches+1})};
    std::vector<float> g(rows * hv * gate_depth, -.17F), beta(rows * hv, .4F);
    for (std::size_t i = 0; i != 4 * hv * gate_depth; ++i) g[i] -= float(i % 3) * .025F;
    for (std::size_t i = 4 * hv * gate_depth; i != g.size(); ++i) g[i] = NAN;
    for (std::size_t i = 4 * hv; i != beta.size(); ++i) beta[i] = NAN;
    Case test(context, resource, c::sequence_ops::packed_gated_delta_net(1, output_scale), descriptions,
        {bytes(q),bytes(k),bytes(v),bytes(g),bytes(beta),bytes(initial),bytes(std::vector<std::int32_t>{0,1,4})});
    for (const auto offsets : {std::array<std::int32_t,3>{0,1,4}, std::array<std::int32_t,3>{0,0,3}}) {
      SCOPED_TRACE(offsets[1]);
      test.offsets(offsets);
      const auto status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
      const auto actual_y = test.read<Value>(0);
      const auto actual_state = test.read<float>(1);
      std::vector<double> expected_state(initial.begin(), initial.end()), expected_y(rows * hv * dv, 0.0);
      for (std::size_t b = 0; b != batches; ++b)
        for (auto t = std::size_t(offsets[b]); t != std::size_t(offsets[b+1]); ++t)
          for (std::size_t h = 0; h != hv; ++h)
            for (std::size_t channel = 0; channel != dv; ++channel) {
              const auto base = ((b * hv + h) * dv + channel) * dk;
              const auto key_base = (t * hk + h % hk) * dk;
              const auto gate_index = (t * hv + h) * gate_depth + (gate_depth == 1 ? 0 : channel);
              double prediction = 0;
              for (std::size_t d = 0; d != dk; ++d) {
                expected_state[base+d] *= std::exp(double(g[gate_index]));
                prediction += expected_state[base+d] * half_test::decode(k[key_base+d]);
              }
              const double delta = beta[t*hv+h] * (half_test::decode(v[(t*hv+h)*dv+channel]) - prediction);
              for (std::size_t d = 0; d != dk; ++d) {
                expected_state[base+d] += delta * half_test::decode(k[key_base+d]);
                expected_y[(t*hv+h)*dv+channel] += expected_state[base+d] * half_test::decode(q[key_base+d]);
              }
            }
      const double scale = output_scale ? double(*output_scale) : 1.0 / std::sqrt(double(dk));
      for (std::size_t i = 0; i != actual_y.size(); ++i) expect_value(actual_y[i],expected_y[i] * scale);
      for (std::size_t i = 0; i != actual_state.size(); ++i) EXPECT_NEAR(actual_state[i],expected_state[i],.000003);
      for (std::size_t i = std::size_t(offsets.back()) * hv * dv; i != actual_y.size(); ++i)
        EXPECT_EQ(bytes(std::vector<Value>{actual_y[i]}),std::vector<std::byte>(sizeof(Value),std::byte{0}));
    }
  }
}

inline void gdn_snapshots(c::Context &context, m::MemoryResourceRef resource) {
  const auto f32 = c::DataType::kFloat32;
  const std::array descriptions{desc(f32,{8,1,1}),desc(f32,{8,1,1}),desc(f32,{8,1,1}),
      desc(f32,{8,1,1}),desc(f32,{8,1,1}),desc(f32,{2,1,1,1}),desc(c::DataType::kInt32,{3})};
  const std::vector<float> ones(8,1.0F), zeros(8,0.0F), beta(8,.5F), v{2,4,6,8,10,NAN,NAN,NAN};
  Case test(context, resource, c::sequence_ops::packed_gated_delta_net(2, 1.0F), descriptions,
      {bytes(ones),bytes(ones),bytes(v),bytes(zeros),bytes(beta),bytes(std::vector<float>{0,1}),
       bytes(std::vector<std::int32_t>{0,2,5})});
  auto status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(test.read<float>(0), (std::vector<float>{1,2.5,3.5,5.75,7.875,0,0,0}));
  EXPECT_EQ(test.read<float>(1), (std::vector<float>{1,2.5,5.75,7.875}));
  test.poison(); test.offsets(std::array<std::int32_t,3>{0,2,3});
  status = test.submit(); EXPECT_TRUE(status.IsInvalidArgument()) << status.message(); test.expect_poison();
  ASSERT_FALSE(test.invocation.in_flight());
  // A failed invocation is terminal. Reuse the prepared plan and fixed buffers,
  // but recreate the invocation only after observing its actual retirement.
  test.invocation = test.prepared.create_invocation().value();
  test.offsets(std::array<std::int32_t,3>{0,2,5});
  status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(test.read<float>(1), (std::vector<float>{1,2.5,5.75,7.875}));
}

inline void gdn_empty(c::Context &context, m::MemoryResourceRef resource) {
  const auto f32 = c::DataType::kFloat32;
  const std::vector<std::uint32_t> initial{0x80000000U,0x7fc01234U,0x7f800000U,0xff800000U};
  for (const auto rows : {std::int64_t{0},std::int64_t{8}}) {
    const std::array descriptions{desc(f32,{rows,1,2}),desc(f32,{rows,1,2}),desc(f32,{rows,1,2}),
        desc(f32,{rows,1,1}),desc(f32,{rows,1,1}),desc(f32,{1,1,2,2}),desc(c::DataType::kInt32,{2})};
    const auto nan = std::vector<float>(std::size_t(rows)*2,NAN), gates = std::vector<float>(std::size_t(rows),NAN);
    Case test(context, resource, c::sequence_ops::packed_gated_delta_net(), descriptions,
        {bytes(nan),bytes(nan),bytes(nan),bytes(gates),bytes(gates),bytes(initial),
         bytes(std::vector<std::int32_t>{0,0})});
    const auto status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(test.read<std::uint32_t>(1), initial);
    EXPECT_EQ(test.read<std::uint32_t>(0), std::vector<std::uint32_t>(std::size_t(rows)*2,0));
    test.poison(); test.offsets(std::array<std::int32_t,2>{0,std::int32_t(rows+1)});
    const auto failed = test.submit(); EXPECT_TRUE(failed.IsInvalidArgument()) << failed.message(); test.expect_poison();
  }
}

template <class Value>
void conv_math(c::Context &context, m::MemoryResourceRef resource) {
  constexpr std::size_t rows = 8, batches = 2, channels = 3;
  const auto type = half_test::data_type<Value>();
  for (const auto taps : {std::size_t{1},std::size_t{4}}) {
    SCOPED_TRACE(taps);
    auto x = values<Value>(rows * channels,1), weight = values<Value>(channels * taps,2);
    auto history = values<Value>(batches * (taps-1) * channels,3);
    for (std::size_t i = 4 * channels; i != x.size(); ++i) x[i] = half_test::encode<Value>(NAN);
    const std::array descriptions{desc(type,{rows,channels}),desc(type,{channels,std::int64_t(taps)}),
        desc(type,{batches,std::int64_t(taps-1),channels}),desc(c::DataType::kInt32,{batches+1})};
    Case test(context, resource, c::sequence_ops::packed_causal_conv1d(), descriptions,
        {bytes(x),bytes(weight),bytes(history),bytes(std::vector<std::int32_t>{0,1,4})});
    for (const auto offsets : {std::array<std::int32_t,3>{0,1,4}, std::array<std::int32_t,3>{0,0,3}}) {
      test.offsets(offsets);
      const auto status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
      const auto y = test.read<Value>(0);
      const auto next = test.read<Value>(1);
      std::vector<double> expected_y(rows * channels,0);
      std::vector<Value> expected_history(history.size());
      for (std::size_t b = 0; b != batches; ++b) {
        std::vector<Value> sequence;
        sequence.insert(sequence.end(), history.begin()+b*(taps-1)*channels, history.begin()+(b+1)*(taps-1)*channels);
        sequence.insert(sequence.end(), x.begin()+std::size_t(offsets[b])*channels, x.begin()+std::size_t(offsets[b+1])*channels);
        const auto length = std::size_t(offsets[b+1]-offsets[b]);
        for (std::size_t t = 0; t != length; ++t)
          for (std::size_t ch = 0; ch != channels; ++ch)
            for (std::size_t tap = 0; tap != taps; ++tap)
              expected_y[(std::size_t(offsets[b])+t)*channels+ch] +=
                  half_test::decode(sequence[(t+tap)*channels+ch]) * half_test::decode(weight[ch*taps+tap]);
        std::copy(sequence.end()-(taps-1)*channels,sequence.end(),expected_history.begin()+b*(taps-1)*channels);
      }
      for (std::size_t i = 0; i != y.size(); ++i) expect_value(y[i],expected_y[i]);
      EXPECT_EQ(bytes(next),bytes(expected_history));
      for (std::size_t i = std::size_t(offsets.back())*channels; i != y.size(); ++i) {
        const auto encoded = bytes(std::vector<Value>{y[i]});
        EXPECT_EQ(encoded,std::vector<std::byte>(sizeof(Value),std::byte{0}));
      }
    }
  }
  // All-empty, zero-physical-row convolution must still copy nonfinite history.
  const std::array descriptions{desc(type,{0,1}),desc(type,{1,2}),desc(type,{1,1,1}),desc(c::DataType::kInt32,{2})};
  const std::vector<Value> history{half_test::encode<Value>(NAN)};
  Case empty(context,resource,c::sequence_ops::packed_causal_conv1d(),descriptions,
      {bytes(std::vector<Value>{}),bytes(values<Value>(2)),bytes(history),bytes(std::vector<std::int32_t>{0,0})});
  const auto status = empty.submit(); ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(bytes(empty.read<Value>(1)),bytes(history));
}

inline void invalid_offsets(c::Context &context, m::MemoryResourceRef resource) {
  const auto f32 = c::DataType::kFloat32;
  const std::array gd{desc(f32,{8,1,1}),desc(f32,{8,1,1}),desc(f32,{8,1,1}),
      desc(f32,{8,1,1}),desc(f32,{8,1,1}),desc(f32,{2,1,1,1}),desc(c::DataType::kInt32,{3})};
  const std::array cd{desc(f32,{8,1}),desc(f32,{1,2}),desc(f32,{2,1,1}),desc(c::DataType::kInt32,{3})};
  const auto input = bytes(std::vector<float>(8,.25F));
  Case gdn(context,resource,c::sequence_ops::packed_gated_delta_net(),gd,
      {input,input,input,input,input,bytes(std::vector<float>(2,1)),bytes(std::vector<std::int32_t>{0,1,4})});
  Case conv(context,resource,c::sequence_ops::packed_causal_conv1d(),cd,
      {input,bytes(std::vector<float>(2,1)),bytes(std::vector<float>(2,1)),bytes(std::vector<std::int32_t>{0,1,4})});
  for (auto *test : {&gdn,&conv}) {
    for (const auto selected : {std::array<std::int32_t,3>{1,2,4}, std::array<std::int32_t,3>{0,2,1}, std::array<std::int32_t,3>{0,1,9}, std::array<std::int32_t,3>{0,1,-1}}) {
      test->poison(); test->offsets(selected);
      const auto status = test->submit();
      EXPECT_TRUE(status.IsInvalidArgument()) << status.message();
      EXPECT_EQ(status.message(),"packed sequence offsets or per-sequence snapshot lengths are invalid");
      test->expect_poison();
      ASSERT_FALSE(test->invocation.in_flight());
      test->invocation = test->prepared.create_invocation().value();
    }
    test->offsets(std::array<std::int32_t,3>{0,1,4});
    const auto status = test->submit(); EXPECT_TRUE(status.ok()) << status.message();
  }
}

} // namespace packed_sequence_test
