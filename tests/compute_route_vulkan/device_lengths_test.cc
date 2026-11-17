#include <gtest/gtest.h>

#include "../../src/compute_route_vulkan/vulkan_device_lengths.h"
#include "../../src/compute_route_vulkan/vulkan_program_internal.h"
#include "../../src/compute_route_vulkan/vulkan_composition_access.h"
#include "tensor_test_support.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;
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
  static constexpr std::uint32_t rows = 6, heads = 4, kv_heads = 2;
  v::Context context = take(vulkan_test::context());
  c::BatchCapacity capacity;
  c::BatchSchema schema;
  c::BatchLayoutVersion envelope;
  std::vector<std::int32_t> pages;
  std::uint32_t qk, dv, page_count;
  bool causal, guarded;
  std::int32_t window;
  v::Program program;
  Cell(std::uint32_t depth, std::uint32_t value_depth, std::uint32_t history,
      bool is_causal = true, std::int32_t window_left = -1, bool guard = false)
      : qk(depth), dv(value_depth), page_count((history+15)/16), causal(is_causal),
        guarded(guard), window(window_left) {
    const auto storage = take(c::CacheStorage::plain(half_test::data_type<T>()));
    capacity = {.query_type=half_test::data_type<T>(), .output_type=half_test::data_type<T>(),
        .key_storage=storage, .value_storage=storage, .max_requests=2, .max_query_rows=rows,
        .max_kv_rows=2*history, .max_pages=2*page_count+2, .max_page_refs=2*page_count,
        .query_heads=heads, .kv_heads=kv_heads, .qk_head_dim=qk, .value_head_dim=dv, .page_size=16};
    schema = take(c::BatchSchema::create(capacity));
    for (std::uint32_t r=0;r<2;++r) for (std::uint32_t p=0;p<page_count;++p)
      pages.push_back(static_cast<std::int32_t>(r*page_count+page_count-p));
    change_envelope({0,2,6}, history);
    auto prepared = d::prepare_device_lengths_paged_graph(d::Access::runtime(context),
        schema, rows, .125F, causal, window, {}, guarded);
    if (!prepared) {
      std::fprintf(stderr,"Vulkan device max allocation=%llu\n",
          static_cast<unsigned long long>(d::Access::runtime(context)->device->maximum_allocation_bytes));
      const std::array descriptions{take(c::batch_query_description(capacity,rows)),
          take(c::batch_cache_description(schema,c::BatchCacheRole::kKey)),
          take(c::batch_cache_description(schema,c::BatchCacheRole::kValue)),
          take(schema.metadata_description()),take(c::batch_output_description(capacity,rows))};
      for (const auto &desc:descriptions) std::fprintf(stderr,"valid=%d plain=%d type=%d bytes=%llu elements=%llu\n",
          int(desc.valid()),int(desc.is_plain()),int(desc.data_type()),
          static_cast<unsigned long long>(*desc.required_bytes()),static_cast<unsigned long long>(*desc.element_count()));
    }
    program = take(std::move(prepared));
    EXPECT_EQ(program.input_descriptions().size(), guarded ? 8U : 6U);
    EXPECT_EQ(d::Access::program(program)->dispatches.size(), 3U);
    EXPECT_TRUE(d::Access::program(program)->status_value.has_value());
  }
  void change_envelope(std::array<std::int32_t,3> qp, std::uint32_t history) {
    const std::array<std::int32_t,3> kp{0,static_cast<std::int32_t>(page_count),static_cast<std::int32_t>(2*page_count)};
    const std::array<std::int32_t,2> tails{static_cast<std::int32_t>((history-1)%16+1),static_cast<std::int32_t>((history-1)%16+1)};
    envelope = take(schema.version({qp,kp,pages,tails}));
  }
  static float quantize(float value) { return static_cast<float>(half_test::decode(half_test::encode<T>(value))); }
  static float qvalue(std::uint32_t row, std::uint32_t h, std::uint32_t i) {
    return quantize(.5F*std::sin(float(row*17+h*3+i)*.19F));
  }
  static float kvalue(std::uint32_t r, std::uint32_t row, std::uint32_t h, std::uint32_t i) {
    return quantize(.4F*std::cos(float(r*31+row*13+h*11+i)*.13F));
  }
  static float vvalue(std::uint32_t r, std::uint32_t row, std::uint32_t h, std::uint32_t i) {
    return quantize(.3F*std::sin(float(r*23+row*7+h*5+i)*.11F));
  }
  template<class U> v::Tensor upload(std::span<const U> payload, c::TensorDesc description) {
    return take(context.upload(description, std::as_bytes(payload)));
  }
  void run(v::Invocation &invocation, std::array<std::int32_t,2> active,
      std::array<std::int32_t,2> lengths, int invalid=0, bool bad_page=false,
      bool device_controls=false, std::array<std::int32_t,2> write_status={0,0}) {
    const auto nan = half_test::encode<T>(std::numeric_limits<float>::quiet_NaN());
    std::vector<T> q(rows*heads*qk,nan), k(capacity.max_pages*16*kv_heads*qk,nan),
        values(capacity.max_pages*16*kv_heads*dv,nan);
    const bool poisoned = invalid || write_status[0] || write_status[1];
    const auto actual_rows = poisoned ? 0 : active[0]+active[1];
    for (int row=0;row<actual_rows;++row) for (std::uint32_t h=0;h<heads;++h)
      for (std::uint32_t i=0;i<qk;++i) q[(row*heads+h)*qk+i]=half_test::encode<T>(qvalue(row,h,i));
    if (!poisoned) for (std::uint32_t r=0;r<2;++r) for (int row=0;row<lengths[r];++row) {
      const auto physical_page=std::uint32_t(pages[r*page_count+row/16]);
      for (std::uint32_t h=0;h<kv_heads;++h) {
        const auto physical=(physical_page*16+row%16)*kv_heads+h;
        for (std::uint32_t i=0;i<qk;++i) k[physical*qk+i]=half_test::encode<T>(kvalue(r,row,h,i));
        for (std::uint32_t i=0;i<dv;++i) values[physical*dv+i]=half_test::encode<T>(vvalue(r,row,h,i));
      }
    }
    auto metadata_lease = take(envelope.metadata());
    std::vector<std::int32_t> metadata(schema.metadata_offsets().bytes/4);
    check(take(metadata_lease.view()).read(std::as_writable_bytes(std::span(metadata))));
    if (bad_page) metadata[schema.metadata_offsets().page_indices/4]=static_cast<std::int32_t>(capacity.max_pages);
    const auto index = take(c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t,1>{2}));
    std::vector<v::Tensor> inputs{
      upload<T>(q,take(c::batch_query_description(capacity,rows))),
      upload<T>(k,take(c::batch_cache_description(schema,c::BatchCacheRole::kKey))),
      upload<T>(values,take(c::batch_cache_description(schema,c::BatchCacheRole::kValue))),
      upload<std::int32_t>(active,index), upload<std::int32_t>(lengths,index),
      upload<std::int32_t>(metadata,program.input_descriptions()[5])};
    if (guarded) for (std::size_t i=0;i<2;++i) inputs.push_back(upload<std::int32_t>(
        std::span(write_status).subspan(i,1),take(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{1}))));
    std::optional<v::Invocation> producer;
    std::optional<c::OwnedResult<v::Results>> produced;
    if (device_controls) {
      d::Plan plan; plan.values={{index,{}},{index,{}},{index,{}},{index,{}}};
      plan.inputs={0,1}; plan.outputs={2,3};
      plan.steps={{c::Operation(c::Copy{}),{0},{2}},{c::Operation(c::Copy{}),{1},{3}}};
      auto generation = take(d::prepare_plan(context,plan));
      producer.emplace(take(generation.create_invocation()));
      check(producer->bind(std::array{inputs[3],inputs[4]}));
      inputs[3]=take(d::CompositionAccess::output(*producer,0));
      inputs[4]=take(d::CompositionAccess::output(*producer,1));
      check(invocation.bind(inputs));
      check(d::CompositionAccess::seal(*producer));
      produced.emplace(take(producer->submit())); // Original Runtime FIFO, no host wait.
    } else check(invocation.bind(inputs));
    auto work=take(invocation.submit()); auto result=work.wait();
    if (produced) check(produced->completion().wait());
    if (invalid || bad_page || write_status[0] || write_status[1]) {
      ASSERT_FALSE(result);
      EXPECT_TRUE(bad_page ? result.error().IsOutOfRange() : result.error().IsInvalidArgument()) << result.error().message();
      // A failed original Invocation is terminal. Never clear its error or
      // make its output ready. The same Context/immutable Program may prepare
      // a fresh slot after the failed native work has actually retired.
      EXPECT_FALSE(invocation.bind(inputs).ok());
      const auto description=program.output_descriptions()[0];
      std::vector<std::byte> poison(*description.required_bytes(),std::byte{0x5a});
      const auto backing=take(context.upload(description,poison));
      auto recorder=take(v::Recorder::create(context));
      check(recorder.append(program,inputs,std::array{backing}));
      auto recorded=take(recorder.finish()); auto completion=take(recorded.submit());
      const auto failed=completion.wait();
      EXPECT_TRUE(bad_page ? failed.IsOutOfRange() : failed.IsInvalidArgument());
      std::vector<std::byte> observed(poison.size()); check(context.download(backing,observed));
      EXPECT_EQ(observed,poison); // The complete output is untouched on failure.
      return;
    }
    ASSERT_TRUE(result) << result.error().message();
    std::vector<T> output(rows*heads*dv);
    check(context.download(result->get()->tensors()[0],std::as_writable_bytes(std::span(output))));
    std::uint32_t packed=0;
    for (std::uint32_t r=0;r<2;++r) for (int row=0;row<active[r];++row,++packed)
      for (std::uint32_t h=0;h<heads;++h) {
        const int end=causal ? lengths[r]-active[r]+row+1 : lengths[r];
        const int begin=window<0 ? 0 : std::max(0,lengths[r]-active[r]+row-window);
        const auto kh=h/(heads/kv_heads);
        std::vector<double> weights;
        double maximum=-std::numeric_limits<double>::infinity(), denominator=0;
        for (int j=begin;j<end;++j) {
          double score=0;
          for (std::uint32_t i=0;i<qk;++i) score+=double(qvalue(packed,h,i))*kvalue(r,j,kh,i);
          weights.push_back(score*.125); maximum=std::max(maximum,weights.back());
        }
        for (auto &weight:weights) { weight=std::exp(weight-maximum); denominator+=weight; }
        for (std::uint32_t i=0;i<dv;++i) {
          double expected=0;
          for (int j=begin;j<end;++j) expected+=weights[j-begin]*vvalue(r,j,kh,i);
          expected=quantize(float(expected/denominator));
          EXPECT_NEAR(half_test::decode(output[(packed*heads+h)*dv+i]),expected,.003) << r << ',' << row << ',' << h << ',' << i;
        }
      }
    for (std::size_t i=std::size_t(actual_rows)*heads*dv;i<output.size();++i) EXPECT_EQ(output[i].bits,0U);
  }
};

TEST(VulkanDeviceLengths, ActualPackedLengthsAndChangedEnvelopeReuseOriginalProgram) {
  Cell<BF16> cell(256,256,39); auto invocation=take(cell.program.create_invocation());
  cell.run(invocation,{1,3},{17,31}); cell.run(invocation,{0,2},{0,19});
  cell.run(invocation,{2,4},{37,39}); cell.run(invocation,{0,0},{0,0});
  cell.change_envelope({0,4,6},39); cell.run(invocation,{3,1},{31,17});
}
TEST(VulkanDeviceLengths, BothStorageTypesAndLongSplitHistoriesHaveIndependentDoubleOracle) {
  for (auto depth:{64U,96U,128U,192U,256U}) {
    Cell<F16> half(depth,depth,1031); auto first=take(half.program.create_invocation());
    half.run(first,{1,3},{1024,1029});
    Cell<BF16> bfloat(depth,depth,1031); auto second=take(bfloat.program.create_invocation());
    bfloat.run(second,{2,4},{1023,1031});
  }
}
TEST(VulkanDeviceLengths, WindowNoncausalAndAsymmetricDepthKeepDeclaredMathematics) {
  Cell<BF16> window(64,128,39,true,0); auto first=take(window.program.create_invocation());
  window.run(first,{2,4},{17,31});
  Cell<F16> noncausal(320,96,39,false); auto second=take(noncausal.program.create_invocation());
  noncausal.run(second,{1,3},{17,31});
}
TEST(VulkanDeviceLengths, ControlsCanComeFromAnOriginalNativeProducerWithoutHostWait) {
  Cell<BF16> cell(64,64,39); auto invocation=take(cell.program.create_invocation());
  cell.run(invocation,{1,3},{17,31},0,false,true);
}
TEST(VulkanDeviceLengths, InvalidLengthsWriteNothingAndFailedSlotsStayTerminalWhileContextRecovers) {
  Cell<BF16> cell(64,64,39);
  for (auto active:std::array<std::array<std::int32_t,2>,3>{{{-1,1},{3,0},{1,5}}}) {
    auto invocation=take(cell.program.create_invocation()); cell.run(invocation,active,{17,31},1);
  }
  for (auto lengths:std::array<std::array<std::int32_t,2>,2>{{{0,31},{17,40}}}) {
    auto invocation=take(cell.program.create_invocation()); cell.run(invocation,{1,3},lengths,1);
  }
  auto recovered=take(cell.program.create_invocation()); cell.run(recovered,{1,3},{17,31});
}
TEST(VulkanDeviceLengths, BadPageRejectsBeforeArithmeticAndSameContextCanPrepareFreshWork) {
  Cell<F16> cell(64,64,39); auto invocation=take(cell.program.create_invocation());
  cell.run(invocation,{1,3},{17,31},0,true);
  auto recovered=take(cell.program.create_invocation()); cell.run(recovered,{1,3},{17,31});
}
TEST(VulkanDeviceLengths, FailedCacheWriterGuardsBothPayloadsBeforeAnyRead) {
  Cell<BF16> cell(64,64,39,true,-1,true);
  for (auto status:std::array<std::array<std::int32_t,2>,2>{{{4,0},{0,4}}}) {
    auto invocation=take(cell.program.create_invocation()); cell.run(invocation,{1,3},{17,31},1,false,false,status);
  }
  auto recovered=take(cell.program.create_invocation()); cell.run(recovered,{1,3},{17,31});
}
TEST(VulkanDeviceLengths, ColdAdmissionRejectsEmptyAndUnsupportedMathematicalSchemas) {
  EXPECT_FALSE(d::qualify_device_lengths_paged_schema({},1.F,true,-1).ok());
  Cell<BF16> cell(64,64,39);
  auto rows=d::prepare_device_lengths_paged_graph(d::Access::runtime(cell.context),cell.schema,7,.125F,true,-1);
  ASSERT_FALSE(rows); EXPECT_TRUE(rows.error().IsInvalidArgument());
  auto window=d::qualify_device_lengths_paged_schema(cell.schema,1.F,false,0);
  ASSERT_FALSE(window.ok()); EXPECT_TRUE(window.IsInvalidArgument());
  auto capacity=cell.capacity; capacity.query_type=c::DataType::kFloat32;
  auto schema=take(c::BatchSchema::create(capacity));
  EXPECT_TRUE(d::qualify_device_lengths_paged_schema(schema,1.F,true,-1).IsUnimplemented());
  for (auto depths:std::array<std::array<std::uint32_t,2>,3>{{{63,64},{64,63},{64,258}}}) {
    capacity=cell.capacity; capacity.qk_head_dim=depths[0]; capacity.value_head_dim=depths[1];
    auto unsupported=take(c::BatchSchema::create(capacity));
    EXPECT_TRUE(d::qualify_device_lengths_paged_schema(unsupported,1.F,true,-1).IsUnimplemented());
  }
}
} // namespace
