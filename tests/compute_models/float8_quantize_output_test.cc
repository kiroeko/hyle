#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/model_ops.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"
#if defined(HYLE_TEST_FP8_QUANT_CPU)
#include "hyle/graph/cpu.h"
#endif
#if defined(HYLE_TEST_FP8_QUANT_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif
#if defined(HYLE_TEST_FP8_QUANT_APPLE)
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#endif
#if defined(HYLE_TEST_FP8_QUANT_VULKAN)
#include "../compute_route_vulkan/fixture.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"
#include "hyle/graph/vulkan.h"
#endif

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using Type = c::DataType;
using Policy = c::model_ops::Float8ScalePolicy;
using Bytes = std::vector<std::byte>;
constexpr std::int64_t kRows = 2, kDepth = 256, kGroup = 128;
constexpr std::size_t kPolicies = 4, kOutputs = 2 * kPolicies;

template<class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
c::TensorDesc dense(Type type, std::initializer_list<std::int64_t> shape) {
  return take(c::TensorDesc::dense(type, {shape.begin(), shape.size()}), "quantize dense description");
}
template<class T> Bytes bytes(std::span<const T> values) {
  const auto source = std::as_bytes(values); return {source.begin(), source.end()};
}

// Independent scalar oracle: named source arithmetic rounds through a volatile
// F32 store. No product quantizer, codec, SIMD or backend helper is called.
float f32(double value) { volatile float rounded = static_cast<float>(value); return rounded; }
float multiply(float a, float b) { return f32(double(a) * double(b)); }
std::uint16_t bfloat(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u)
    return static_cast<std::uint16_t>((bits >> 16) | ((bits & 0x7fffffu) ? 0x40u : 0u));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(bits >> 16);
}
float wide(std::uint16_t value) { return std::bit_cast<float>(std::uint32_t(value) << 16); }
float e4(unsigned magnitude) {
  const auto exponent = int(magnitude >> 3), fraction = int(magnitude & 7u);
  return exponent ? std::ldexp(1.F + float(fraction) / 8.F, exponent - 7)
                  : std::ldexp(float(fraction), -9);
}
std::uint8_t encode(float value) {
  const double magnitude = (std::min)(std::abs(double(value)), 448.);
  unsigned best = 0; double distance = magnitude;
  for (unsigned code = 1; code <= 0x7e; ++code) {
    const double next = std::abs(magnitude - double(e4(code)));
    if (next < distance || (next == distance && !(code & 1u))) {
      best = code; distance = next;
    }
  }
  return static_cast<std::uint8_t>(best | (std::signbit(value) ? 0x80u : 0u));
}
float scale(float maximum, Policy policy, float floor) {
  maximum = (std::max)(maximum, floor);
  if (policy == Policy::kDivide) return f32(double(maximum) / 448.);
  const float raw = multiply(maximum, f32(1. / 448.));
  if (policy == Policy::kMultiplyReciprocal) return raw;
  // Mathematical ceil to a power of two, independent of native bit ladders.
  int exponent = 0; const float fraction = std::frexp(raw, &exponent);
  return std::ldexp(1.F, fraction == .5F ? exponent - 1 : exponent);
}
constexpr std::array<Policy, kPolicies> kPolicy{
    Policy::kDivide, Policy::kPowerOfTwo, Policy::kMultiplyReciprocal, Policy::kDivide};
constexpr std::array<float, kPolicies> kFloor{1.e-4F, 1.e-4F, 1.e-4F, 1.e-10F};
struct Input final { Bytes payload; std::vector<float> actual; };
Input input(Type type, unsigned phase, bool invalid = false) {
  std::vector<float> values(std::size_t(kRows * kDepth));
  const std::array<float, 4> maximum = phase == 0
      ? std::array<float, 4>{450.F, 0x1p-126F, 1.00371F, .00339871F}
      : std::array<float, 4>{887.311F, 0x1p-125F, .291873F, .00712793F};
  constexpr std::array<unsigned, 8> thresholds{1, 2, 7, 8, 15, 32, 63, 124};
  for (std::size_t group = 0; group < maximum.size(); ++group) {
    const auto start = group * std::size_t(kGroup);
    for (std::size_t lane = 0; lane < std::size_t(kGroup); ++lane) {
      const int residue = int((lane * 37 + group * 19 + phase * 23) % 101) - 50;
      values[start + lane] = multiply(maximum[group], f32(double(residue) / 83.));
    }
    values[start] = maximum[group];
    values[start + 1] = 0.F; values[start + 2] = -0.F;
    values[start + 3] = std::numeric_limits<float>::denorm_min();
    values[start + 4] = -std::numeric_limits<float>::denorm_min();
    values[start + 5] = 0x1p-133F; values[start + 6] = -0x1p-133F;
    if (group == 1) continue; // An entire tiny group exercises both floors.
    const float group_scale = scale(maximum[group], Policy::kDivide, 1.e-4F);
    for (std::size_t i = 0; i < thresholds.size(); ++i) {
      const auto code = thresholds[i];
      const float midpoint = f32((double(e4(code)) + double(e4(code + 1))) / 2.);
      const float boundary = multiply(midpoint, group_scale);
      const auto lane = start + 16 + i * 3;
      values[lane] = std::nextafter(boundary, -std::numeric_limits<float>::infinity());
      values[lane + 1] = boundary;
      values[lane + 2] = -std::nextafter(boundary, std::numeric_limits<float>::infinity());
    }
  }
  values[7] = phase == 0 ? .006866455078125F : -.01373291015625F;
  if (invalid) values[173] = std::numeric_limits<float>::quiet_NaN();
  Input result;
  if (type == Type::kFloat32) { result.actual = values; result.payload = bytes<float>(values); }
  else {
    std::vector<std::uint16_t> stored; stored.reserve(values.size()); result.actual.reserve(values.size());
    for (const auto value : values) { stored.push_back(bfloat(value)); result.actual.push_back(wide(stored.back())); }
    result.payload = bytes<std::uint16_t>(stored);
  }
  return result;
}
using Image = std::array<Bytes, kOutputs>;
Image oracle(std::span<const float> values) {
  Image output;
  for (std::size_t policy = 0; policy < kPolicies; ++policy) {
    std::vector<std::uint8_t> codes(values.size()); std::array<float, 4> scales{};
    for (std::size_t group = 0; group < scales.size(); ++group) {
      float maximum = 0;
      for (std::size_t lane = 0; lane < std::size_t(kGroup); ++lane)
        maximum = (std::max)(maximum, std::abs(values[group * std::size_t(kGroup) + lane]));
      scales[group] = scale(maximum, kPolicy[policy], kFloor[policy]);
      for (std::size_t lane = 0; lane < std::size_t(kGroup); ++lane) {
        const auto i = group * std::size_t(kGroup) + lane;
        const auto normalized = kPolicy[policy] == Policy::kMultiplyReciprocal
            ? multiply(values[i], f32(1. / double(scales[group])))
            : f32(double(values[i]) / double(scales[group]));
        codes[i] = encode(normalized);
      }
    }
    output[policy * 2] = bytes<std::uint8_t>(codes);
    output[policy * 2 + 1] = bytes<float>(scales);
  }
  return output;
}
struct Logical final {
  c::TensorDesc input_desc;
  std::array<c::TensorDesc, kOutputs> desc;
  g::ValueId source{};
  std::array<g::ValueId, kOutputs> outputs{};
  g::Graph graph;
  explicit Logical(Type type) : input_desc(dense(type, {kRows, kDepth})) {
    auto builder = take(g::GraphBuilder::create(), "two-output quantize GraphBuilder");
    source = take(builder.input(input_desc), "shared activation source");
    for (std::size_t i = 0; i < kPolicies; ++i) {
      const auto op = i == kPolicies - 1 ? c::model_ops::float8_block_quantize()
          : c::model_ops::float8_block_quantize(kPolicy[i], kFloor[i]);
      const auto node = take(builder.add_node(op, std::array{source}), "ordinary two-output quantize node");
      for (std::size_t port = 0; port < 2; ++port) {
        const auto index = i * 2 + port;
        outputs[index] = take(builder.result(node, port), "real code/scale result");
        desc[index] = dense(port ? Type::kFloat32 : Type::kFloat8E4M3FN,
                            {kRows, port ? kDepth / kGroup : kDepth});
        check(builder.publish(outputs[index]), "publish typed quantize output");
      }
    }
    graph = take(std::move(builder).finish(), "quantize Graph");
  }
};
bool overlaps(m::ConstBufferView a, m::ConstBufferView b) {
  return a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() < b.offset() + b.size() && b.offset() < a.offset() + a.size();
}
template<class Read> Image image(const std::array<g::GraphOutput, kOutputs> &results, Read read) {
  Image result;
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto view = take(results[i].tensor(), "actual typed quantize result");
    result[i].resize(view.storage().size()); check(read(view.storage(), result[i]), "read actual code/scale bytes");
  }
  return result;
}
void expect_image(const Image &actual, const Image &expected) {
  for (std::size_t port = 0; port < expected.size(); ++port) {
    ASSERT_EQ(actual[port].size(), expected[port].size());
    if (!(port & 1u)) {
      std::size_t mismatches = 0, first = 0;
      for (std::size_t i = 0; i < expected[port].size(); ++i)
        if (actual[port][i] != expected[port][i]) { if (!mismatches) first = i; ++mismatches; }
      EXPECT_EQ(mismatches, 0u) << "code port=" << port << " first=" << first
          << " actual=" << unsigned(std::to_integer<std::uint8_t>(actual[port][first]))
          << " expected=" << unsigned(std::to_integer<std::uint8_t>(expected[port][first]));
    } else {
      for (std::size_t group = 0; group < 4; ++group) {
        std::uint32_t a = 0, b = 0;
        std::memcpy(&a, actual[port].data() + group * sizeof(float), sizeof(a));
        std::memcpy(&b, expected[port].data() + group * sizeof(float), sizeof(b));
        EXPECT_EQ(a, b) << "scale port=" << port << " row-major group=" << group;
      }
    }
  }
}

template<class Compile, class Write, class Read, class Prepare>
void run(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
         Compile compile, Write write, Read read, Prepare prepare, const c::StorageOwner &owner = {}) {
  unsigned successful = 0, failed = 0;
  for (const auto type : {Type::kFloat32, Type::kBFloat16}) {
    SCOPED_TRACE(::testing::Message() << "input dtype=" << unsigned(type));
    Logical logical(type);
    auto compiled = take(compile(logical.graph), "compile shared-input code/scale Graph");
    std::array<bool, kOutputs> caller{}; std::size_t footprint = 0;
    for (std::size_t i = 0; i < kOutputs; ++i) {
      caller[i] = take(compiled.executable.output_ownership(i), "quantize output ownership") == g::OutputOwnership::kCaller;
      if (caller[i]) footprint += (take(logical.desc[i].required_bytes(), "output bytes") + 255u) / 256u * 256u;
    }
    c::LeasePool pool;
    if (footprint) pool = take(c::LeasePool::create(resource, footprint * 4,
        {.max_blocks = std::uint32_t(kOutputs * 4), .backing_alignment = 256}), "bounded code/scale generations");
    if (caller[0] && caller[1]) {
      // These deliberately borrowed buffers exercise actual binding rejection,
      // independently of the exclusive leases used by every submitted run.
      auto backing = take(m::Buffer::create(resource, take(logical.input_desc.required_bytes(), "alias backing extent"), 256), "alias probe backing");
      {
        auto probe = take(compiled.executable.create_invocation(), "input/output alias probe");
        check(probe.bind(logical.source, take(c::ConstTensorView::create(logical.input_desc, backing.view()), "borrowed probe input")), "bind probe input");
        const auto range = take(backing.view().subview(0, take(logical.desc[0].required_bytes(), "code extent")), "aliased code range");
        const auto status = probe.bind(logical.outputs[0], take(c::TensorView::create(logical.desc[0], range), "aliased code view"));
        EXPECT_FALSE(status.ok()); EXPECT_TRUE(status.IsInvalidArgument());
      }
      {
        auto probe = take(compiled.executable.create_invocation(), "code/scale alias probe");
        const auto code_range = take(backing.view().subview(0, take(logical.desc[0].required_bytes(), "code extent")), "code probe range");
        check(probe.bind(logical.outputs[0], take(c::TensorView::create(logical.desc[0], code_range), "code probe view")), "bind standalone code output");
        const auto scale_range = take(backing.view().subview(0, take(logical.desc[1].required_bytes(), "scale extent")), "overlapping scale range");
        const auto status = probe.bind(logical.outputs[1], take(c::TensorView::create(logical.desc[1], scale_range), "overlapping scale view"));
        EXPECT_FALSE(status.ok()); EXPECT_TRUE(status.IsInvalidArgument());
      }
    }
    std::array<std::array<g::GraphOutput, kOutputs>, 3> retained;
    std::array<std::array<c::ReadLease, kOutputs>, 3> exported;
    std::array<Image, 3> expected;
    const auto execute = [&](unsigned phase, bool invalid, std::size_t generation) {
      const auto data = input(type, phase, invalid);
      auto backing = take(m::Buffer::create(resource, data.payload.size(), 256), "actual activation backing");
      check(write(backing.view(), data.payload), "upload current activation bytes");
      auto ready = owner ? take(owner.adopt_buffer(std::move(backing)), "retained activation owner")
                         : take(c::ReadLease::adopt(std::move(backing)), "retained native activation");
      const auto original_input = take(ready.view(), "readonly current activation");
      auto invocation = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "independent quantize Invocation");
      check(invocation.bind(logical.source, logical.input_desc, ready), "bind shared actual input lease");
      for (std::size_t i = 0; i < kOutputs; ++i)
        if (caller[i]) check(invocation.bind(logical.outputs[i], logical.desc[i],
            take(pool.allocate(take(logical.desc[i].required_bytes(), "output extent")), "exclusive code/scale writer")), "bind independent output writer");
      check(prepare(invocation, queues), "prepare ordinary quantize invocation");
      { auto done = take(invocation.submit(queues), "submit actual quantization");
        const auto status = done.wait();
        if (invalid) {
          EXPECT_FALSE(status.ok());
          for (const auto port : logical.outputs) EXPECT_FALSE(invocation.result(port));
          ++failed; return;
        }
        check(status, "quantize Completion"); }
      ++successful; expected[generation] = oracle(data.actual);
      for (std::size_t i = 0; i < kOutputs; ++i) {
        retained[generation][i] = take(invocation.result(logical.outputs[i]), "retain actual code/scale result");
        EXPECT_EQ(take(retained[generation][i].description(), "retained typed description"), logical.desc[i]);
        exported[generation][i] = take(retained[generation][i].storage_lease(), "independent result owner");
        EXPECT_FALSE(exported[generation][i].try_write());
        const auto view = take(exported[generation][i].view(), "readonly exported result");
        EXPECT_FALSE(overlaps(view, original_input));
        for (std::size_t prior = 0; prior < i; ++prior)
          EXPECT_FALSE(overlaps(view, take(exported[generation][prior].view(), "other output range")));
      }
      expect_image(image(retained[generation], read), expected[generation]);
      for (std::size_t prior = 0; prior < generation; ++prior)
        expect_image(image(retained[prior], read), expected[prior]);
    };
    execute(0, false, 0); execute(1, false, 1);
    EXPECT_NE(expected[0], expected[1]);
    EXPECT_NE(expected[0][0], expected[0][2]);
    EXPECT_NE(expected[0][0], expected[0][4]);
    execute(0, true, 0); // Failed outputs are never read or exported.
    expect_image(image(retained[0], read), expected[0]);
    expect_image(image(retained[1], read), expected[1]);
    execute(0, false, 2); // New Invocation after failed Completion.
    EXPECT_EQ(expected[0], expected[2]);
    compiled = {};
    for (std::size_t generation = 0; generation < retained.size(); ++generation)
      expect_image(image(retained[generation], read), expected[generation]);
    retained = {}; exported = {};
    if (pool.valid()) { EXPECT_EQ(pool.stats().active_uses, 0u); EXPECT_EQ(pool.stats().live_blocks, 0u); }
  }
  EXPECT_EQ(successful, 6u); EXPECT_EQ(failed, 2u);
  ::testing::Test::RecordProperty("successful_submissions", int(successful));
  ::testing::Test::RecordProperty("failed_submissions", int(failed));
  ::testing::Test::RecordProperty("scope", "two dense M2/K256 Graphs:F32/BF16;four policies,eight real outputs;two generations,NaN Completion,fresh recovery,retained readonly results;not throughput/fullmodel");
}

TEST(Float8QuantizeOutput, ContractRejectsInvalidDescriptionsAndParameters) {
  const auto op = c::model_ops::float8_block_quantize();
  EXPECT_EQ(c::operation_traits(op).outputs, 2u);
  EXPECT_EQ(c::operation_traits(op).alias, c::AliasPolicy::kNone);
  EXPECT_EQ(c::operation_traits(op).effect, c::EffectKind::kPure);
  const auto descriptor = op.custom_as<c::model_ops::Float8BlockQuantize>();
  ASSERT_TRUE(descriptor); EXPECT_EQ(descriptor->scale_policy, Policy::kDivide); EXPECT_EQ(descriptor->absmax_floor, 1.e-10F);
  std::array<c::TensorDesc, 2> output;
  const std::array good{dense(Type::kFloat32, {kRows, kDepth})};
  check(c::infer_output_descriptions(op, good, output), "ordinary two-output inference");
  EXPECT_EQ(output[0], dense(Type::kFloat8E4M3FN, {kRows, kDepth}));
  EXPECT_EQ(output[1], dense(Type::kFloat32, {kRows, kDepth / kGroup}));
  for (const auto desc : {dense(Type::kFloat16, {kRows, kDepth}), dense(Type::kFloat32, {kRows, 129}),
                          dense(Type::kFloat32, {0, kDepth})})
    EXPECT_FALSE(c::infer_output_descriptions(op, std::array{desc}, output).ok());
  for (const auto floor : {0.F, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::quiet_NaN()})
    EXPECT_FALSE(c::infer_output_descriptions(c::model_ops::float8_block_quantize(Policy::kDivide, floor), good, output).ok());
  EXPECT_FALSE(c::infer_output_descriptions(c::model_ops::float8_block_quantize(static_cast<Policy>(99)), good, output).ok());
  EXPECT_FALSE(c::infer_output_descriptions(op, good, std::span(output).first(1)).ok());
}

#if defined(HYLE_TEST_FP8_QUANT_CPU) || defined(HYLE_TEST_FP8_QUANT_CUDA)
// The same declared F32 dynamic linear is executed by CPU and CUDA. Sparse
// code-domain dots isolate scale layout/output precision from reduction order.
template<class Compile>
void run_f32_power2_linear(const c::StorageOwner &owner, c::Queue &queue, Compile compile) {
  struct Shape { std::int64_t m, n, k; bool paired = false; };
  unsigned successes = 0, failures = 0, checked_values = 0, f32_operand_witnesses = 0;
  std::size_t paired_graph_steps = 0;
  for (const auto shape : {Shape{3,3,128}, Shape{3,129,384}, Shape{5,65,256,true}}) {
    SCOPED_TRACE(::testing::Message() << "M=" << shape.m << " N=" << shape.n << " K=" << shape.k);
    const auto groups = shape.k / 128;
    auto builder = take(g::GraphBuilder::create(), "F32 pow2 linear builder");
    const auto ad = dense(Type::kFloat32, {shape.m, shape.k});
    const auto a = take(builder.input(ad), "original F32 A");
    std::vector<g::ValueId> ports;
    std::vector<c::TensorDesc> outputs;
    std::vector<std::vector<std::byte>> original_weights;
    std::vector<c::ReadLease> weight_leases;
    std::vector<g::ConstantBinding> constants;
    const std::vector<std::int64_t> columns = shape.paired ? std::vector<std::int64_t>{shape.n,129} : std::vector<std::int64_t>{shape.n};
    for (const auto n : columns) {
      const auto wd = take(c::float8_weight_description(shape.k,n,c::Float8ScaleMode::kFloat32Block128), "encoded F32SF W");
      auto &raw = original_weights.emplace_back(take(wd.required_bytes(), "encoded W extent"),std::byte{});
      for (std::int64_t row = 0; row < n; ++row) for (std::int64_t group = 0; group < groups; ++group)
        raw[std::size_t(row*shape.k+group*128+1)] = std::byte(row%2 ? 0xb8 : 0x38);
      constexpr std::array<float,4> sw{1.00390625F,.625F,.03125F,3.25F};
      for (std::int64_t block = 0; block < (n+127)/128; ++block) for (std::int64_t group = 0; group < groups; ++group) {
        const auto value = sw[std::size_t(block+group)%sw.size()];
        std::memcpy(raw.data()+n*shape.k+(block*groups+group)*4,&value,4);
      }
      auto backing = take(m::Buffer::create(owner.ref(),raw.size(),256), "encoded W backing");
      check(backing.view().write(raw), "write original encoded W");
      weight_leases.push_back(take(owner.adopt_buffer(std::move(backing)), "retain original W"));
      const auto w = take(builder.constant(wd), "encoded W constant");
      constants.push_back({w,take(weight_leases.back().tensor(wd), "typed original W"),g::ConstantPolicy::kBorrow,owner,weight_leases.back()});
      const auto y = take(builder.add(c::model_ops::float8_block_linear(Policy::kPowerOfTwo,1.e-4F),a,w), "F32 power2 dynamic linear");
      check(builder.publish(y), "publish F32 projection"); ports.push_back(y); outputs.push_back(dense(Type::kFloat32,{shape.m,n}));
    }
    auto graph = take(std::move(builder).finish(), "F32 dynamic linear graph");
    auto compiled = take(compile(graph,g::CompileOptions{.constants=constants}), "compile F32 dynamic linear");
    if(shape.paired)paired_graph_steps=compiled.report.step_count();
    const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
    auto pool = take(c::LeasePool::create(owner.ref(),std::size_t(shape.m*129*4+256)*12,{.max_blocks=12}), "finite F32 outputs");
    const auto input = [&](unsigned phase) {
      std::vector<float> values(std::size_t(shape.m*shape.k));
      for (std::int64_t row=0;row<shape.m;++row) for (std::int64_t group=0;group<groups;++group) {
        const auto first=std::size_t(row*shape.k+group*128);
        const auto maximum=std::ldexp(1.3F+float(row%3)*.137F,int(group-row)+int(phase));
        const auto sf=scale(maximum,Policy::kPowerOfTwo,1.e-4F);
        values[first]=maximum;
        values[first+1]=std::nextafter(multiply(1.0625F,sf),std::numeric_limits<float>::infinity());
        if ((row+group+phase)%2) values[first+1]=-values[first+1];
      }
      return values;
    };
    const auto oracle = [&](std::span<const float> values,std::size_t port,bool wrong_bf16=false) {
      const auto n=columns[port];std::vector<float> expected(std::size_t(shape.m*n));
      for (std::int64_t row=0;row<shape.m;++row) for (std::int64_t column=0;column<n;++column) {
        float total=0;
        for (std::int64_t group=0;group<groups;++group) {
          const auto first=std::size_t(row*shape.k+group*128);float maximum=0;
          for (std::size_t k=0;k<128;++k)maximum=(std::max)(maximum,std::abs(wrong_bf16?wide(bfloat(values[first+k])):values[first+k]));
          const auto sf=scale(maximum,Policy::kPowerOfTwo,1.e-4F);
          const auto source=wrong_bf16?wide(bfloat(values[first+1])):values[first+1];
          const auto code=encode(f32(double(source)/double(sf)));
          const auto decoded=(code&128 ? -1.F : 1.F)*e4(code&127);
          float sw=0;std::memcpy(&sw,original_weights[port].data()+n*shape.k+((column/128)*groups+group)*4,4);
          total=f32(double(total)+double(multiply(multiply(decoded,column%2?-1.F:1.F),sf))*double(sw));
        }
        expected[std::size_t(row*n+column)]=total;
      }
      return expected;
    };
    const auto read_output = [&](const g::GraphOutput &output) {
      const auto tensor=take(output.tensor(), "retained F32 result");
      std::vector<float> values(take(tensor.description().element_count(), "F32 result count"));
      check(tensor.storage().read(std::as_writable_bytes(std::span(values))), "read F32 result");return values;
    };
    const auto execute = [&](unsigned phase,float bad) {
      auto values=input(phase);if (!std::isfinite(bad)) values.back()=bad;
      auto backing=take(m::Buffer::create(owner.ref(),values.size()*4,256), "independent F32 A backing");
      check(backing.view().write(std::as_bytes(std::span(values))), "write F32 A");
      const auto ready=take(owner.adopt_buffer(std::move(backing)), "retain F32 A");
      auto invocation=take(compiled.executable.create_invocation(), "independent F32 linear invocation");
      check(invocation.prepare_resources(), "prepare owned F32 linear bindings");
      check(invocation.bind(a,ad,ready), "bind F32 A lease");
      for (std::size_t port=0;port<ports.size();++port)
        if(take(compiled.executable.output_ownership(port), "F32 output ownership")==g::OutputOwnership::kCaller)
          check(invocation.bind(ports[port],outputs[port],take(pool.allocate(take(outputs[port].required_bytes(), "F32 output bytes")), "finite output writer")), "bind F32 output writer");
      check(invocation.prepare_replay(bindings,{.mode=g::ReplayMode::kCommands}), "prepare F32 linear commands");
      auto done=take(invocation.submit(bindings), "submit F32 dynamic linear");const auto status=done.wait();
      std::vector<g::GraphOutput> retained;
      if(!std::isfinite(bad)) {
        EXPECT_FALSE(status.ok());for(auto port:ports)EXPECT_FALSE(invocation.result(port));++failures;return retained;
      }
      check(status, "F32 linear Completion");++successes;
      for(std::size_t port=0;port<ports.size();++port) {
        retained.push_back(take(invocation.result(ports[port]), "retain F32 linear result"));
        const auto actual=read_output(retained.back()), expected=oracle(values,port), wrong=oracle(values,port,true);
        if(actual.size()!=expected.size())throw std::runtime_error("F32 linear result extent differs");
        for(std::size_t i=0;i<actual.size();++i){EXPECT_EQ(actual[i],expected[i])<<i;checked_values+=1;f32_operand_witnesses+=expected[i]!=wrong[i];}
      }
      return retained;
    };
    auto old=execute(0,0);std::vector<std::vector<float>> image;for(const auto &value:old)image.push_back(read_output(value));
    (void)execute(0,std::numeric_limits<float>::quiet_NaN());
    (void)execute(0,std::numeric_limits<float>::infinity());
    auto recovered=execute(1,0);compiled={};
    for(std::size_t port=0;port<old.size();++port)EXPECT_EQ(read_output(old[port]),image[port]);
    EXPECT_FALSE(recovered.empty());
  }
  EXPECT_EQ(successes,6u);EXPECT_EQ(failures,6u);EXPECT_GT(f32_operand_witnesses,0u);
  ::testing::Test::RecordProperty("successful_submissions",successes);
  ::testing::Test::RecordProperty("nonfinite_failed_Completions",failures);
  ::testing::Test::RecordProperty("checked_F32_values",checked_values);
  ::testing::Test::RecordProperty("wrong_BF16_operand_witnesses",f32_operand_witnesses);
  ::testing::Test::RecordProperty("paired_graph_steps",paired_graph_steps);
}
#endif

#if defined(HYLE_TEST_FP8_QUANT_CPU)
struct SystemResource final { m::MemoryResourceRef ref() noexcept { return m::system_resource(); } };
TEST(Float8QuantizeOutput, CpuCodesScalesAndLifetimes) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  const auto owner = take(c::StorageOwner::adopt(context, SystemResource{}), "CPU allocation owner");
  auto queue = take(context.create_queue(), "CPU queue");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  run(owner.ref(), queues,
      [&](const g::Graph &graph) { return g::cpu::compile(graph, context); },
      [](m::BufferView to, std::span<const std::byte> from) { return to.write(from); },
      [](m::ConstBufferView from, std::span<std::byte> to) { return from.read(to); },
      [](g::Invocation &invocation, std::span<const g::QueueBinding> bindings) { return invocation.prepare_replay(bindings); }, owner);
}
TEST(Float8QuantizeOutput, CpuF32Power2Linear) {
  auto context=take(g::cpu::create_context(), "CPU F32 linear context");
  auto owner=take(c::StorageOwner::adopt(context,SystemResource{}), "CPU F32 linear owner");
  auto queue=take(context.create_queue(), "CPU F32 linear queue");
  run_f32_power2_linear(owner,queue,[&](const g::Graph &graph,g::CompileOptions options){return g::cpu::compile(graph,context,options);});
}
#endif
#if defined(HYLE_TEST_FP8_QUANT_CUDA)
TEST(Float8QuantizeOutput, CudaCodesScalesAndLifetimes) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice), "CUDA memory")), "CUDA allocation owner");
  auto queue = take(context.create_queue(), "CUDA queue");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  run(owner.ref(), queues,
      [&](const g::Graph &graph) { return g::cuda::compile(graph, context, owner); },
      [](m::BufferView to, std::span<const std::byte> from) { return to.write(from); },
      [](m::ConstBufferView from, std::span<std::byte> to) { return from.read(to); },
      [](g::Invocation &invocation, std::span<const g::QueueBinding> bindings) { return invocation.prepare_replay(bindings); }, owner);
}
TEST(Float8QuantizeOutput, CudaF32Power2Linear) {
  auto context=take(g::cuda::create_context(), "CUDA F32 linear context");
  const auto environment=take(c::cuda::environment(context), "CUDA F32 linear environment");
  if(environment.sm_major!=8 || environment.sm_minor!=9)
    GTEST_SKIP() << "The existing encoded FP8 GEMM AOT requires SM89";
  auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice), "CUDA F32 linear memory")), "CUDA F32 linear owner");
  auto queue=take(context.create_queue(), "CUDA F32 linear queue");
  run_f32_power2_linear(owner,queue,[&](const g::Graph &graph,g::CompileOptions options){return g::cuda::compile(graph,context,owner,options);});
}
#endif
#if defined(HYLE_TEST_FP8_QUANT_APPLE)
TEST(Float8QuantizeOutput, AppleCodesScalesAndLifetimes) {
  auto context = take(c::apple::Context::create(), "ordinary Apple context");
  auto memory = take(c::apple::MemoryResource::create(context), "Apple shared memory");
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{take(g::apple::execution_source(context), "Apple execution source"), g::StorageDomainId{0}}};
  run(memory.ref(), {},
      [&](const g::Graph &graph) { return g::compile(graph, {.storage_domains = domains, .targets = targets}); },
      [](m::BufferView to, std::span<const std::byte> from) { return to.write(from); },
      [](m::ConstBufferView from, std::span<std::byte> to) { return from.read(to); },
      [](g::Invocation &invocation, std::span<const g::QueueBinding> bindings) { return invocation.prepare_replay(bindings); });
}
#endif
#if defined(HYLE_TEST_FP8_QUANT_VULKAN)
TEST(Float8QuantizeOutput, VulkanCodesScalesAndLifetimes) {
  namespace v = c::vulkan;
  auto context = take(vulkan_test::context(), "ordinary Vulkan context");
  auto device = take(v::MemoryResource::create(context, v::MemoryKind::kDevice), "Vulkan device memory");
  auto staging = take(v::MemoryResource::create(context, v::MemoryKind::kStaging), "Vulkan staging memory");
  constexpr std::size_t capacity = 4096;
  auto transfer = take(v::TransferSlot::create(context, staging, capacity), "Vulkan transfer slot");
  auto host = take(m::Buffer::create(m::system_resource(), capacity, 64), "Vulkan host bytes");
  const auto description = [](std::size_t size) { return c::TensorDesc::dense(Type::kUInt8, std::array{std::int64_t(size)}); };
  const auto write = [&](m::BufferView destination, std::span<const std::byte> source) -> hyle::Status {
    HYLE_ASSIGN_OR_RETURN(auto desc, description(source.size()));
    HYLE_ASSIGN_OR_RETURN(auto host_bytes, host.view().subview(0, source.size()));
    HYLE_RETURN_IF_ERROR(host_bytes.write(source));
    HYLE_ASSIGN_OR_RETURN(auto view, c::ConstTensorView::create(desc, destination));
    HYLE_ASSIGN_OR_RETURN(auto native, v::detail::MemoryAccess::tensor_view(v::detail::Access::runtime(context), view, m::system_allocator()));
    HYLE_ASSIGN_OR_RETURN(auto uploaded, c::ConstTensorView::create(desc, host_bytes));
    HYLE_ASSIGN_OR_RETURN(auto done, transfer.upload(native, uploaded)); return done.wait();
  };
  const auto read = [&](m::ConstBufferView source, std::span<std::byte> destination) -> hyle::Status {
    HYLE_ASSIGN_OR_RETURN(auto desc, description(destination.size()));
    HYLE_ASSIGN_OR_RETURN(auto view, c::ConstTensorView::create(desc, source));
    HYLE_ASSIGN_OR_RETURN(auto native, v::detail::MemoryAccess::tensor_view(v::detail::Access::runtime(context), view, m::system_allocator()));
    HYLE_ASSIGN_OR_RETURN(auto done, transfer.download(native)); HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto host_bytes, host.view().subview(0, destination.size()));
    HYLE_ASSIGN_OR_RETURN(auto target, c::TensorView::create(desc, host_bytes));
    HYLE_RETURN_IF_ERROR(transfer.read(target)); return host_bytes.read(destination);
  };
  const std::array domains{g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{take(g::vulkan::execution_source(context), "Vulkan execution source"), g::StorageDomainId{0}}};
  run(device.ref(), {},
      [&](const g::Graph &graph) { return g::compile(graph, {.storage_domains = domains, .targets = targets}); }, write, read,
      [](g::Invocation &invocation, std::span<const g::QueueBinding> bindings) { return invocation.prepare_replay(bindings); });
}
#endif
} // namespace
