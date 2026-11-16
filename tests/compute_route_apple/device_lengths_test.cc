#include <gtest/gtest.h>

#include "../../src/compute_route_apple/apple_device_lengths.h"
#include "../../src/compute_route_apple/apple_carrier.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/compute/detail/apple_plan.h"
#include "../compute_provider_native/half_test_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <mlx/ops.h>

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace d = a::detail;
using F16 = hyle::simd::float16_t;
using BF16 = hyle::simd::bfloat16_t;
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error(std::string(value.error().message()));
  return std::move(*value);
}
void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

template<class T> struct Cell final {
  static constexpr std::uint32_t heads = 4, kv_heads = 2, queries = 6;
  a::Context context = take(a::Context::create());
  a::MemoryResource memory = take(a::MemoryResource::create(context));
  c::BatchSchema schema;
  c::BatchLayoutVersion envelope, control_envelope;
  c::BatchCapacity capacity;
  a::Program program;
  std::uint32_t qk, dv, per_request_pages;
  std::vector<std::int32_t> page_indices;
  c::TensorDesc output;

  Cell(std::uint32_t key_depth, std::uint32_t value_depth, std::uint32_t history,
      bool causal = true, std::int32_t window = -1)
      : qk(key_depth), dv(value_depth), per_request_pages((history+15)/16) {
    const auto storage = take(c::CacheStorage::plain(half_test::data_type<T>()));
    capacity = {.query_type = half_test::data_type<T>(), .output_type = half_test::data_type<T>(),
        .key_storage = storage, .value_storage = storage, .max_requests = 2,
        .max_query_rows = queries, .max_kv_rows = 2*history,
        .max_pages = 2*per_request_pages+2, .max_page_refs = 2*per_request_pages,
        .query_heads = heads, .kv_heads = kv_heads, .qk_head_dim = qk, .value_head_dim = dv, .page_size = 16};
    schema = take(c::BatchSchema::create(capacity));
    for (std::uint32_t request = 0; request < 2; ++request)
      for (std::uint32_t page = 0; page < per_request_pages; ++page)
        page_indices.push_back(static_cast<std::int32_t>(request*per_request_pages+per_request_pages-page));
    const std::array<std::int32_t,3> qp{0,2,6}, kp{0,static_cast<std::int32_t>(per_request_pages),static_cast<std::int32_t>(2*per_request_pages)};
    const std::array<std::int32_t,2> tails{static_cast<std::int32_t>((history-1)%16+1),static_cast<std::int32_t>((history-1)%16+1)};
    envelope = take(schema.version({qp,kp,page_indices,tails}));
    control_envelope = envelope;
    output = take(c::batch_output_description(capacity, queries));
    program = take(d::prepare_device_lengths_paged_graph(d::ContextAccess::runtime(context),
        schema, queries, .125f, causal, window));
    EXPECT_EQ(program.input_count(),6u);
  }

  template<class U> mlx::core::array binding(std::span<const U> payload, c::TensorDesc description,
      std::vector<c::LeasePool> &pools, std::vector<c::ReadLease> &owners) {
    const auto bytes = std::as_bytes(payload);
    pools.push_back(take(c::LeasePool::create(memory.ref(), bytes.size()+64)));
    auto writer = take(pools.back().allocate(bytes.size()));
    check(take(writer.view()).write(bytes));
    owners.push_back(take(writer.publish()));
    return take(memory.binding(owners.back(), description));
  }
  static float quantized(float value) { return half_test::decode(half_test::encode<T>(value)); }
  static float query_value(std::uint32_t row, std::uint32_t head, std::uint32_t dim) {
    return quantized(.5f*std::sin(float(row*17+head*3+dim)*.19f));
  }
  static float key_value(std::uint32_t request, std::uint32_t row, std::uint32_t head, std::uint32_t dim) {
    return quantized(.4f*std::cos(float(request*31+row*13+head*11+dim)*.13f));
  }
  static float value_value(std::uint32_t request, std::uint32_t row, std::uint32_t head, std::uint32_t dim) {
    return quantized(.3f*std::sin(float(request*23+row*7+head*5+dim)*.11f));
  }
  void run(a::Invocation &invocation, std::array<std::int32_t,2> active,
      std::array<std::int32_t,2> lengths, bool causal = true, std::int32_t window = -1,
      int invalid = 0, bool bad_page = false, bool device_controls = false) {
    const auto nan = half_test::encode<T>(std::numeric_limits<float>::quiet_NaN());
    std::vector<T> q(queries*heads*qk,nan), k(capacity.max_pages*16*kv_heads*qk,nan), v(capacity.max_pages*16*kv_heads*dv,nan);
    const auto actual_rows = invalid ? 0 : active[0]+active[1];
    for (int row = 0; row < actual_rows; ++row)
      for (std::uint32_t h = 0; h < heads; ++h) for (std::uint32_t dim = 0; dim < qk; ++dim)
        q[(std::size_t(row)*heads+h)*qk+dim] = half_test::encode<T>(query_value(row,h,dim));
    if (!invalid) for (std::uint32_t r = 0; r < 2; ++r)
      for (int row = 0; row < lengths[r]; ++row) {
        const auto page = static_cast<std::uint32_t>(page_indices[r*per_request_pages+row/16]);
        for (std::uint32_t h = 0; h < kv_heads; ++h) {
          const auto physical = (std::size_t(page)*16+row%16)*kv_heads+h;
          for (std::uint32_t dim = 0; dim < qk; ++dim) k[physical*qk+dim] = half_test::encode<T>(key_value(r,row,h,dim));
          for (std::uint32_t dim = 0; dim < dv; ++dim) v[physical*dv+dim] = half_test::encode<T>(value_value(r,row,h,dim));
        }
      }
    auto original_metadata = take(control_envelope.metadata());
    std::vector<std::int32_t> metadata(schema.metadata_offsets().bytes/4);
    check(take(original_metadata.view()).read(std::as_writable_bytes(std::span(metadata))));
    if (bad_page) {
      const auto invalid_page = static_cast<std::int32_t>(capacity.max_pages);
      std::memcpy(reinterpret_cast<std::byte *>(metadata.data())+schema.metadata_offsets().page_indices,&invalid_page,sizeof(invalid_page));
    }
    std::vector<c::LeasePool> pools; pools.reserve(6);
    std::vector<c::ReadLease> owners; owners.reserve(6);
    std::vector<mlx::core::array> inputs;
    inputs.push_back(binding<T>(q,take(c::batch_query_description(capacity,queries)),pools,owners));
    inputs.push_back(binding<T>(k,take(c::batch_cache_description(schema,c::BatchCacheRole::kKey)),pools,owners));
    inputs.push_back(binding<T>(v,take(c::batch_cache_description(schema,c::BatchCacheRole::kValue)),pools,owners));
    const auto indices = take(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{2}));
    inputs.push_back(binding<std::int32_t>(active,indices,pools,owners));
    inputs.push_back(binding<std::int32_t>(lengths,indices,pools,owners));
    inputs.push_back(binding<std::int32_t>(metadata,take(c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t,1>{static_cast<std::int64_t>(metadata.size())})),pools,owners));
    std::optional<a::Invocation> producer;
    std::optional<c::OwnedResult<a::Results>> produced;
    if (device_controls) {
      const auto spec = take(a::input_spec(indices));
      const std::array specifications{spec,spec};
      auto generator = take(context.compile([](const std::vector<mlx::core::array> &args,
          mlx::core::Stream stream) -> hyle::Result<std::vector<mlx::core::array>> {
        return std::vector{mlx::core::add(args[0],mlx::core::array(0),stream),
            mlx::core::add(args[1],mlx::core::array(0),stream)};
      },specifications));
      producer.emplace(take(generator.create_invocation()));
      const std::array controls{inputs[3],inputs[4]}; check(producer->bind(controls));
      std::array<d::InputBinding,6> bindings;
      for (std::size_t i = 0; i < inputs.size(); ++i)
        if (i != 3 && i != 4) bindings[i].value = inputs[i];
      bindings[3].source = take(d::InvocationAccess::source(*producer,0));
      bindings[4].source = take(d::InvocationAccess::source(*producer,1));
      check(d::InvocationAccess::prepare_inputs(invocation,bindings));
      produced.emplace(take(producer->submit()));
    } else check(invocation.bind(inputs));
    auto completion = take(invocation.submit());
    auto result = completion.wait();
    if (produced) check(produced->completion().wait());
    if (invalid || bad_page) {
      ASSERT_FALSE(result);
      EXPECT_TRUE(bad_page ? result.error().IsOutOfRange() : result.error().IsInvalidArgument()) << result.error().message();
      return;
    }
    ASSERT_TRUE(result) << result.error().message();
    auto read = take(memory.import_readonly((*result)->arrays()[0],output));
    std::vector<T> values(queries*heads*dv);
    check(take(read.view()).read(std::as_writable_bytes(std::span(values))));
    std::uint32_t packed = 0;
    for (std::uint32_t request = 0; request < 2; ++request)
      for (int row = 0; row < active[request]; ++row,++packed)
        for (std::uint32_t h = 0; h < heads; ++h) {
          const auto end = causal ? lengths[request]-active[request]+row+1 : lengths[request];
          const auto begin = window < 0 ? 0 : std::max(0,lengths[request]-active[request]+row-window);
          const auto kh = h/(heads/kv_heads);
          std::vector<double> weights;
          double maximum = -std::numeric_limits<double>::infinity(), denominator = 0;
          for (int j = begin; j < end; ++j) {
            double score = 0;
            for (std::uint32_t dim = 0; dim < qk; ++dim) score += double(query_value(packed,h,dim))*key_value(request,j,kh,dim);
            weights.push_back(score*.125); maximum = std::max(maximum,weights.back());
          }
          for (auto &weight : weights) { weight = std::exp(weight-maximum); denominator += weight; }
          for (std::uint32_t dim = 0; dim < dv; ++dim) {
            double expected = 0;
            for (int j = begin; j < end; ++j) expected += weights[j-begin]*value_value(request,j,kh,dim);
            expected = quantized(static_cast<float>(expected/denominator));
            EXPECT_NEAR(half_test::decode(values[(packed*heads+h)*dv+dim]),expected,.003) << request << ',' << row << ',' << h << ',' << dim;
          }
        }
    for (std::size_t i = std::size_t(actual_rows)*heads*dv; i < values.size(); ++i) EXPECT_EQ(values[i].bits,0u);
  }
};

TEST(AppleDeviceLengths, PackedRequestsUseActualDeviceLengthsAndZeroInactiveRows) {
  Cell<BF16> cell(256,256,39);
  auto invocation = take(cell.program.create_invocation());
  cell.run(invocation,{1,3},{17,31});
  cell.run(invocation,{0,2},{0,19});
  cell.run(invocation,{2,4},{37,39});
  const std::array<std::int32_t,3> qp{0,4,6}, kp{0,3,6};
  const std::array<std::int32_t,2> tails{7,7};
  cell.control_envelope = take(cell.schema.version({qp,kp,cell.page_indices,tails}));
  // The same actual Program and Invocation consume changed owned bounds.
  // No rebuilding/tracing is needed when the per-request capacities change.
  cell.run(invocation,{3,1},{31,17});
}
TEST(AppleDeviceLengths, F32SplitPartialsKeepBothDeclaredStorageTypes) {
  for (const auto depth : {64u,128u,256u}) {
    Cell<F16> half(depth,depth,1031);
    auto first = take(half.program.create_invocation()); half.run(first,{1,3},{513,1019});
    Cell<BF16> bfloat(depth,depth,1031);
    auto second = take(bfloat.program.create_invocation()); bfloat.run(second,{1,3},{513,1019});
  }
}
TEST(AppleDeviceLengths, SeparateValueDepthAndSlidingWindowArePreserved) {
  Cell<BF16> cell(64,128,39,true,7);
  auto invocation = take(cell.program.create_invocation());
  cell.run(invocation,{1,3},{17,31},true,7);
}
TEST(AppleDeviceLengths, FullContextCapacityDoesNotReadInactiveCapacityPages) {
  Cell<BF16> cell(256,256,32768);
  auto invocation = take(cell.program.create_invocation());
  cell.run(invocation,{1,3},{513,1019});
}
TEST(AppleDeviceLengths, NoncausalAttentionUsesTheActualHistory) {
  Cell<F16> cell(64,64,39,false);
  auto invocation = take(cell.program.create_invocation());
  cell.run(invocation,{2,4},{1,17},false);
}
TEST(AppleDeviceLengths, InvalidControlsFailBeforeAnyPoisonedCacheCanBeRead) {
  Cell<BF16> cell(64,64,39);
  const std::array<std::pair<std::array<std::int32_t,2>,std::array<std::int32_t,2>>,5> cases{{
      {{-1,3},{17,31}},{{3,3},{17,31}},{{1,3},{40,31}},{{2,3},{1,31}},
      {{INT32_MAX,INT32_MAX},{INT32_MAX,INT32_MAX}}}};
  for (const auto &[queries,lengths] : cases) {
    auto invocation = take(cell.program.create_invocation());
    cell.run(invocation,queries,lengths,true,-1,1);
  }
}
TEST(AppleDeviceLengths, InvalidPageDoesNotBecomeAReadableOutput) {
  Cell<BF16> cell(64,64,39);
  auto invocation = take(cell.program.create_invocation());
  cell.run(invocation,{1,3},{17,31},true,-1,0,true);
}
TEST(AppleDeviceLengths, RejectsPhysicalRowsOutsideTheOriginalSchema) {
  Cell<BF16> cell(64,64,39);
  const auto result = d::prepare_device_lengths_paged_graph(d::ContextAccess::runtime(cell.context),
      cell.schema,7,.125f,true,-1);
  ASSERT_FALSE(result); EXPECT_TRUE(result.error().IsInvalidArgument());
}
TEST(AppleDeviceLengths, DeviceProducedLengthsUseTheOriginalNativeSources) {
  Cell<BF16> cell(256,256,39);
  auto invocation = take(cell.program.create_invocation());
  cell.run(invocation,{1,3},{17,31},true,-1,0,false,true);
  cell.run(invocation,{2,1},{31,17},true,-1,0,false,true);
}
} // namespace
