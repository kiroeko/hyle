#pragma once

#include "packed_sequence_case.h"

namespace packed_gdn_prefix_test {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace p = packed_sequence_test;

// Independent FP64 recurrence over already rounded source values. Physical
// padding is deliberately NaN; no provider supplies a reference state.
template<class Value> struct Fixture final {
  static constexpr std::size_t rows = 8, batches = 2, hk = 2, hv = 4, dv = 2;
  static constexpr std::uint32_t prefixes = 3;
  std::size_t dk, gate_depth;
  std::optional<float> scale;
  std::array<c::TensorDesc, 7> descriptions;
  std::vector<Value> q, k, v;
  std::vector<float> g, beta, initial;

  explicit Fixture(bool per_channel = false, std::size_t key_depth = 3,
                   std::optional<float> output_scale = 1.0F)
      : dk(key_depth), gate_depth(per_channel ? dv : 1), scale(output_scale),
        q(p::values<Value>(rows * hk * dk, 1)), k(p::values<Value>(rows * hk * dk, 2)),
        v(p::values<Value>(rows * hv * dv, 3)), g(rows * hv * gate_depth, -.17F),
        beta(rows * hv, .4F), initial(p::values<float>(batches * hv * dv * dk, 4)) {
    const auto type = half_test::data_type<Value>(), f32 = c::DataType::kFloat32;
    descriptions = {p::desc(type,{rows,hk,std::int64_t(dk)}), p::desc(type,{rows,hk,std::int64_t(dk)}),
        p::desc(type,{rows,hv,dv}), p::desc(f32,{rows,hv,std::int64_t(gate_depth)}),
        p::desc(f32,{rows,hv,1}), p::desc(f32,{batches,hv,dv,std::int64_t(dk)}),
        p::desc(c::DataType::kInt32,{batches+1})};
    for (std::size_t i = 0; i < 4 * hv * gate_depth; ++i) g[i] -= float(i % 3) * .025F;
    for (std::size_t i = 4 * hk * dk; i < q.size(); ++i) q[i] = k[i] = half_test::encode<Value>(NAN);
    for (std::size_t i = 4 * hv * dv; i < v.size(); ++i) v[i] = half_test::encode<Value>(NAN);
    for (std::size_t i = 4 * hv * gate_depth; i < g.size(); ++i) g[i] = NAN;
    for (std::size_t i = 4 * hv; i < beta.size(); ++i) beta[i] = NAN;
  }

  std::vector<std::vector<std::byte>> data() const {
    return {p::bytes(q),p::bytes(k),p::bytes(v),p::bytes(g),p::bytes(beta),p::bytes(initial),
        p::bytes(std::vector<std::int32_t>{0,1,4})};
  }

  std::array<std::vector<double>, 2> oracle(std::span<const std::int32_t> offsets) const {
    const auto state_width = hv * dv * dk;
    std::array<std::vector<double>, 2> expected{std::vector<double>(rows * hv * dv, 0),
        std::vector<double>(batches * prefixes * state_width, 0)};
    std::vector<double> state(initial.begin(), initial.end());
    for (std::size_t b = 0; b < batches; ++b)
      for (auto t = std::size_t(offsets[b]); t < std::size_t(offsets[b+1]); ++t) {
        for (std::size_t h = 0; h < hv; ++h)
          for (std::size_t channel = 0; channel < dv; ++channel) {
            const auto base = ((b * hv + h) * dv + channel) * dk;
            const auto key_base = (t * hk + h % hk) * dk;
            const auto gate = (t * hv + h) * gate_depth + (gate_depth == 1 ? 0 : channel);
            double prediction = 0;
            for (std::size_t d = 0; d < dk; ++d) {
              state[base+d] *= std::exp(double(g[gate]));
              prediction += state[base+d] * half_test::decode(k[key_base+d]);
            }
            const double delta = beta[t*hv+h] * (half_test::decode(v[(t*hv+h)*dv+channel]) - prediction);
            for (std::size_t d = 0; d < dk; ++d) {
              state[base+d] += delta * half_test::decode(k[key_base+d]);
              expected[0][(t*hv+h)*dv+channel] += state[base+d] * half_test::decode(q[key_base+d]);
            }
          }
        const auto slot = t - std::size_t(offsets[b]);
        std::copy_n(state.begin() + b * state_width, state_width,
            expected[1].begin() + (b * prefixes + slot) * state_width);
      }
    const auto factor = scale ? double(*scale) : 1.0 / std::sqrt(double(dk));
    for (auto &value : expected[0]) value *= factor;
    return expected;
  }

  void check(const p::Case &test, std::span<const std::int32_t> offsets) const {
    const auto expected = oracle(offsets);
    const auto actual_y = test.read<Value>(0);
    const auto actual_states = test.read<float>(1);
    ASSERT_EQ(actual_y.size(), expected[0].size()); ASSERT_EQ(actual_states.size(), expected[1].size());
    for (std::size_t index = 0; index < actual_y.size(); ++index) p::expect_value(actual_y[index], expected[0][index]);
    for (std::size_t index = 0; index < actual_states.size(); ++index)
      EXPECT_NEAR(actual_states[index], expected[1][index], .000003) << index;
    for (std::size_t index = std::size_t(offsets.back()) * hv * dv; index < actual_y.size(); ++index)
      EXPECT_EQ(p::bytes(std::vector<Value>{actual_y[index]}), std::vector<std::byte>(sizeof(Value),std::byte{0}));
    const auto state_width = hv * dv * dk;
    for (std::size_t b = 0; b < batches; ++b)
      for (auto slot = std::size_t(offsets[b+1] - offsets[b]); slot < prefixes; ++slot)
        for (std::size_t index = 0; index < state_width; ++index)
          EXPECT_EQ(std::bit_cast<std::uint32_t>(actual_states[(b * prefixes + slot) * state_width + index]), 0U);
    std::vector<std::byte> original(initial.size() * sizeof(float));
    ASSERT_TRUE(test.input_buffers[5].view().read(original).ok());
    EXPECT_EQ(original, p::bytes(initial)); test.expect_tail();
  }
};

// The second sequence's S2 is fed directly from its device-resident output
// subrange into another original operation. No state readback feeds execution.
inline void continue_from_prefix(c::Context &context, m::MemoryResourceRef resource,
    p::Case &test, const Fixture<float> &fixture) {
  constexpr std::size_t batch = 1, prefix = 2, row = 3;
  std::array<c::ConstTensorView, 7> inputs;
  std::array<c::TensorDesc, 7> descriptions;
  for (std::size_t port = 0; port < 5; ++port) {
    const auto shape = fixture.descriptions[port].dimensions();
    descriptions[port] = p::desc(c::DataType::kFloat32,{1,shape[1],shape[2]});
    const auto bytes = descriptions[port].required_bytes().value();
    inputs[port] = c::ConstTensorView::create(descriptions[port],
        test.inputs[port].storage().subview(row * bytes, bytes).value()).value();
  }
  descriptions[5] = p::desc(c::DataType::kFloat32,{1,Fixture<float>::hv,Fixture<float>::dv,std::int64_t(fixture.dk)});
  const auto state_bytes = descriptions[5].required_bytes().value();
  inputs[5] = c::ConstTensorView::create(descriptions[5], test.outputs[1].storage().subview(
      (batch * fixture.prefixes + prefix - 1) * state_bytes, state_bytes).value()).value();
  std::vector<std::byte> saved(state_bytes);
  ASSERT_TRUE(inputs[5].storage().read(saved).ok());
  descriptions[6] = p::desc(c::DataType::kInt32,{2});
  auto offsets = m::Buffer::create(resource, 2 * sizeof(std::int32_t), 256).value();
  const std::array<std::int32_t, 2> one{0,1};
  ASSERT_TRUE(offsets.view().write(std::as_bytes(std::span(one))).ok());
  inputs[6] = c::ConstTensorView::create(descriptions[6], offsets.view()).value();
  const auto operation = c::sequence_ops::packed_gated_delta_net_prefixes(1,fixture.scale);
  std::array<c::TensorDesc, 2> outputs;
  ASSERT_TRUE(c::infer_output_descriptions(operation,descriptions,outputs).ok());
  std::array<m::Buffer, 2> storage;
  std::array<c::TensorView, 2> destinations;
  for (std::size_t port = 0; port < 2; ++port) {
    storage[port] = m::Buffer::create(resource, outputs[port].required_bytes().value(),256).value();
    destinations[port] = c::TensorView::create(outputs[port],storage[port].view()).value();
  }
  auto prepared = c::prepare(context,operation,inputs,destinations);
  ASSERT_TRUE(prepared) << prepared.error().message();
  ASSERT_EQ(prepared->workspace_requirement().size,0U);
  auto invocation = prepared->create_invocation().value();
  auto done = invocation.submit(test.queue,inputs,destinations);
  ASSERT_TRUE(done) << done.error().message(); ASSERT_TRUE(done->wait().ok());
  const std::array<std::int32_t,3> layout{0,1,4}; const auto expected = fixture.oracle(layout);
  const std::array expected_begin{row * Fixture<float>::hv * Fixture<float>::dv,
      (batch * fixture.prefixes + prefix) * state_bytes / sizeof(float)};
  for (std::size_t port = 0; port < 2; ++port) {
    std::vector<float> actual(outputs[port].element_count().value());
    ASSERT_TRUE(storage[port].view().read(std::as_writable_bytes(std::span(actual))).ok());
    for (std::size_t index = 0; index < actual.size(); ++index)
      EXPECT_NEAR(actual[index],expected[port][expected_begin[port]+index],.000003) << port << ':' << index;
  }
  std::vector<std::byte> unchanged(state_bytes);
  ASSERT_TRUE(inputs[5].storage().read(unchanged).ok()); EXPECT_EQ(unchanged,saved);
}

template<class Value> void math(c::Context &context, m::MemoryResourceRef resource,
    bool per_channel, std::optional<float> scale = 1.0F) {
  const Fixture<Value> fixture(per_channel,3,scale);
  p::Case test(context,resource,c::sequence_ops::packed_gated_delta_net_prefixes(fixture.prefixes,scale),
      fixture.descriptions,fixture.data());
  EXPECT_EQ(test.output_desc[1],p::desc(c::DataType::kFloat32,{2,3,4,2,3}));
  for (const auto selected : {std::array<std::int32_t,3>{0,1,4},std::array<std::int32_t,3>{0,3,4}}) {
    test.offsets(selected); const auto status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
    fixture.check(test,selected);
    if constexpr (std::is_same_v<Value,float>)
      if (selected[1] == 1) continue_from_prefix(context,resource,test,fixture);
  }
}

inline void schema() {
  const Fixture<float> fixture;
  const auto sentinel = p::desc(c::DataType::kFloat32,{7});
  for (const auto capacity : {1U,3U}) {
    std::array<c::TensorDesc,2> outputs;
    ASSERT_TRUE(c::infer_output_descriptions(c::sequence_ops::packed_gated_delta_net_prefixes(capacity),
        fixture.descriptions,outputs).ok());
    EXPECT_EQ(outputs[0],fixture.descriptions[2]);
    EXPECT_EQ(outputs[1],p::desc(c::DataType::kFloat32,{2,std::int64_t(capacity),4,2,3}));
  }
  const auto reject = [&](const auto &inputs, c::Operation operation) {
    std::array result{sentinel,sentinel};
    EXPECT_TRUE(c::infer_output_descriptions(operation,inputs,result).IsInvalidArgument());
    EXPECT_EQ(result[0],sentinel); EXPECT_EQ(result[1],sentinel);
  };
  reject(fixture.descriptions,c::sequence_ops::packed_gated_delta_net_prefixes(0));
  reject(fixture.descriptions,c::sequence_ops::packed_gated_delta_net_prefixes(3,std::numeric_limits<float>::infinity()));
  auto wrong = fixture.descriptions; wrong[6] = p::desc(c::DataType::kInt64,{3});
  reject(wrong,c::sequence_ops::packed_gated_delta_net_prefixes(3));
  wrong = fixture.descriptions; wrong[3] = p::desc(c::DataType::kFloat32,{8,4,3});
  reject(wrong,c::sequence_ops::packed_gated_delta_net_prefixes(3));
}

inline void empty_and_failure(c::Context &context, m::MemoryResourceRef resource) {
  const auto f32 = c::DataType::kFloat32;
  const std::vector<std::uint32_t> initial{0x80000000U,0x7fc01234U,0x7f800000U,0xff800000U};
  for (const auto rows : {std::int64_t{0},std::int64_t{8}}) {
    const std::array descriptions{p::desc(f32,{rows,1,1}),p::desc(f32,{rows,1,1}),p::desc(f32,{rows,1,1}),
        p::desc(f32,{rows,1,1}),p::desc(f32,{rows,1,1}),p::desc(f32,{4,1,1,1}),p::desc(c::DataType::kInt32,{5})};
    const auto nan = p::bytes(std::vector<float>(std::size_t(rows),NAN));
    p::Case test(context,resource,c::sequence_ops::packed_gated_delta_net_prefixes(3),descriptions,
        {nan,nan,nan,nan,nan,p::bytes(initial),p::bytes(std::vector<std::int32_t>{0,0,0,0,0})});
    const auto status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
    for (std::size_t port = 0; port < 2; ++port)
      EXPECT_EQ(test.read<std::uint32_t>(port),std::vector<std::uint32_t>(test.output_desc[port].element_count().value(),0));
    std::vector<std::uint32_t> actual(initial.size());
    ASSERT_TRUE(test.input_buffers[5].view().read(std::as_writable_bytes(std::span(actual))).ok()); EXPECT_EQ(actual,initial);
  }
  const Fixture<float> fixture;
  p::Case test(context,resource,c::sequence_ops::packed_gated_delta_net_prefixes(3,fixture.scale),fixture.descriptions,fixture.data());
  test.poison(); test.offsets(std::array<std::int32_t,3>{0,1,5}); // Last member Li=4>K, wholly in R=8.
  const auto failed = test.submit(); EXPECT_TRUE(failed.IsInvalidArgument()) << failed.message(); test.expect_poison();
  ASSERT_FALSE(test.invocation.in_flight());
  auto denied = test.invocation.submit(test.queue,test.inputs,test.outputs);
  ASSERT_FALSE(denied); EXPECT_TRUE(denied.error().IsFailedPrecondition());
  test.invocation = test.prepared.create_invocation().value();
  const std::array<std::int32_t,3> valid{0,0,3}; test.offsets(valid);
  const auto retried = test.submit(); ASSERT_TRUE(retried.ok()) << retried.message(); fixture.check(test,valid);
}
} // namespace packed_gdn_prefix_test
