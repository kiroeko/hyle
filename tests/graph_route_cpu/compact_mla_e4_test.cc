#include "../compute_models/compact_mla_e4_fixture.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/providers/ggml_cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f=compact_mla_e4_test;
  auto context=f::take(f::c::Context::create(f::c::cpu::device(),
      f::c::ggml_cpu::provider({.native_threads=2})), "CPU paged context");
  auto queue=f::take(context.create_queue(), "CPU paged queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  const std::array domains{f::g::StorageDomain{f::m::system_resource()}};
  const std::array targets{f::g::ExecutionTarget{
      f::take(f::g::ExecutionSource::acquire(context), "CPU source"),f::g::StorageDomainId{0}}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        return f::g::batch::compile(graph,{domains,targets},options);
      },
      [](f::m::BufferView to,std::span<const std::byte> from) {return to.write(from);},
      [](f::m::ConstBufferView from,std::span<std::byte> to) {return from.read(to);});
  f::check(context.drain(), "CPU context retirement");
}
}
TEST(CpuCompactMlaE4, PagedRope64SharesScaleAndOldResults) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read) {
    compact_mla_e4_test::run(resource,queues,compile,write,read,576,true,false);
  });
}

TEST(CpuCompactMlaE4, RaggedNoPeSharesScaleAndOldResults) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read) {
    compact_mla_e4_test::run(resource,queues,compile,write,read,512,false,false);
  });
}

TEST(CpuCompactMlaE4, PagedNoPeQ1GrowsThroughOriginalPlan) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read) {
    compact_mla_e4_test::run(resource,queues,compile,write,read,512,true,true);
  });
}

TEST(CpuCompactMlaE4, RaggedRope64Q1GrowthMatchesFreshInvocation) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read) {
    compact_mla_e4_test::run(resource,queues,compile,write,read,576,false,true);
  });
}

TEST(CpuCompactMlaE4, SchemaKeepsIndependentQueryOutputAndOneCacheInterpretation) {
  namespace c=hyle::compute;
  c::MlaBatchCapacity capacity{.query_type=c::DataType::kBFloat16,
      .output_type=c::DataType::kFloat16,
      .cache_storage=c::CacheStorage::float8(.5F).value()};
  const auto schema=c::BatchSchema::create_mla(capacity);ASSERT_TRUE(schema);
  const auto internal=schema->capacity();
  EXPECT_EQ(internal.key_storage,internal.value_storage);
  EXPECT_EQ(c::batch_query_description(internal)->data_type(),c::DataType::kBFloat16);
  EXPECT_EQ(c::batch_output_description(internal,1)->data_type(),c::DataType::kFloat16);
  const auto cache=c::batch_cache_description(*schema,c::BatchCacheRole::kMla);ASSERT_TRUE(cache);
  EXPECT_EQ(cache->data_type(),c::DataType::kFloat8E4M3FN);
  EXPECT_EQ(cache->required_bytes().value(),16u*576u);
  EXPECT_FALSE(c::batch_cache_description(*schema,c::BatchCacheRole::kValue));
  capacity.cache_storage=c::CacheStorage::q8_0().value();
  EXPECT_TRUE(c::BatchSchema::create_mla(capacity).error().IsUnimplemented());
}
