#include "../compute_route_apple/tensor_test_support.h"

#include "hyle/compute/apple_batch.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"

#include <numeric>

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;

// Uniform softmax has an independent exact FP64 oracle. The BF16 cases
// separate large positive/negative partials and retain one encoded residual.
// These values fit the final type; only the old unnormalized partial narrows.
struct Case final {
  int queries, keys, heads;
  bool bf16, grouped;
};
constexpr int kv_heads = 2, depth = 64;
constexpr std::array cases{
    Case{1,4096,8,false,false}, Case{8,4096,8,true,false},
    Case{1,8192,16,false,true}, Case{1,8192,16,true,true}};

std::vector<std::uint16_t> value_words(Case test) {
  std::vector<std::uint16_t> result(std::size_t(test.keys)*kv_heads*depth);
  for (int row=0;row<test.keys;++row) {
    const bool positive = test.grouped ? row<test.keys/2 : row%32<16;
    const float value = test.bf16 ? (positive ? 16384.F : -16384.F) +
        (row==0 ? 128.F : 0.F) : 2048.F;
    const auto encoded = test.bf16 ? half_test::encode<B16>(value).bits :
                                    half_test::encode<F16>(value).bits;
    std::fill_n(result.begin()+std::ptrdiff_t(row)*kv_heads*depth,
                kv_heads*depth, encoded);
  }
  return result;
}
double expected_value(Case test, const std::vector<std::uint16_t>& values) {
  double sum=0;
  for(int row=0;row<test.keys;++row) {
    const auto word=values[std::size_t(row)*kv_heads*depth];
    sum += test.bf16 ? half_test::decode_bits<B16>(word) :
                       half_test::decode_bits<F16>(word);
  }
  return sum/test.keys;
}
void check_words(c::ConstTensorView output, Case test, double expected) {
  const auto count=std::size_t(test.queries)*test.heads*depth;
  std::vector<std::uint16_t> words(count);
  ASSERT_TRUE(output.storage().read(std::as_writable_bytes(std::span(words))).ok());
  const auto type=test.bf16 ? mx::bfloat16 : mx::float16;
  const auto wanted=rounded(expected,type);
  const double tolerance=(test.bf16 ? .008 : .001)*std::abs(wanted)+2e-7;
  for(std::size_t i=0;i<count;++i) {
    const auto observed=test.bf16 ? half_test::decode_bits<B16>(words[i]) :
                                   half_test::decode_bits<F16>(words[i]);
    ASSERT_TRUE(std::isfinite(observed)) << i;
    ASSERT_NEAR(observed,wanted,tolerance) << i;
  }
}

void paged_case(Case test, bool graph) {
  SCOPED_TRACE(::testing::Message() << "Q=" << test.queries << " KV=" << test.keys
      << " H=" << test.heads << " BF16=" << test.bf16 << " Graph=" << graph);
  const auto type=test.bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat16;
  auto native=a::Context::create().value();
  auto memory=a::MemoryResource::create(native).value();
  const auto pages=std::uint32_t(test.keys/16);
  const c::BatchCapacity cap{.query_type=type,.output_type=type,
      .key_storage=c::CacheStorage::plain(type).value(),
      .value_storage=c::CacheStorage::plain(type).value(),
      .max_requests=1,.max_query_rows=std::uint32_t(test.queries),
      .max_kv_rows=std::uint32_t(test.keys),.max_pages=pages,.max_page_refs=pages,
      .query_heads=std::uint32_t(test.heads),.kv_heads=kv_heads,
      .qk_head_dim=depth,.value_head_dim=depth};
  auto schema=c::BatchSchema::create(cap).value();
  const std::array<std::int32_t,2> qp{0,test.queries},kp{0,std::int32_t(pages)};
  const std::array<std::int32_t,1> tails{16};
  std::vector<std::int32_t> ids(pages);
  std::iota(ids.rbegin(),ids.rend(),0); // Deliberately reverse physical pages.
  auto layout=schema.version({qp,kp,ids,tails}).value();
  auto recipe=c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes=pages*sizeof(std::int32_t),.scale=.125F,.causal=false}).value();
  const auto qd=recipe.input_description(layout,0).value();
  const auto kd=recipe.input_description(layout,1).value();
  const auto vd=recipe.input_description(layout,2).value();
  auto keys=c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory.ref(),
      {.max_backing_bytes=std::size_t(test.keys)*kv_heads*depth*2+64}).value();
  auto values=c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory.ref(),
      {.max_backing_bytes=std::size_t(test.keys)*kv_heads*depth*2+64}).value();
  const auto encoded=value_words(test);
  const auto expected=expected_value(test,encoded);
  std::vector<std::uint16_t> zero(16*kv_heads*depth);
  std::vector<c::ReadLease> kr(pages),vr(pages);
  for(std::size_t logical=0;logical<pages;++logical) {
    const auto physical=std::uint32_t(ids[logical]);
    auto kw=keys.acquire(physical).value(),vw=values.acquire(physical).value();
    ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(std::span(zero))).ok());
    const auto data=std::span(encoded).subspan(logical*zero.size(),zero.size());
    ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(data)).ok());
    kr[physical]=kw.publish(16).value();vr[physical]=vw.publish(16).value();
  }
  auto key=keys.read(layout,kr).value(),value=values.read(layout,vr).value();
  auto pool=c::LeasePool::create(memory.ref(),qd.required_bytes().value()+64).value();
  auto writer=pool.allocate(qd.required_bytes().value()).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{0}).ok());
  auto query=writer.publish().value();
  const auto od=c::batch_output_description(cap,test.queries).value();
  if(!graph) {
    auto created=a::PagedBatchInvocation::create(native,recipe,memory,memory,
        {.max_metadata_backing_bytes=3*pages*sizeof(std::int32_t)+256});
    ASSERT_TRUE(created) << created.error().message();
    auto slot=std::move(*created);
    ASSERT_TRUE(slot.prepare_batch(layout).ok());
    ASSERT_TRUE(slot.bind(query,key,value).ok());
    auto submitted=slot.submit();ASSERT_TRUE(submitted) << submitted.error().message();
    auto result=submitted->wait();ASSERT_TRUE(result) << result.error().message();
    auto read=memory.import_readonly((*result)->arrays()[0],od).value();
    check_words(c::ConstTensorView::create(od,read.view().value()).value(),test,expected);
    return;
  }

  auto builder=g::GraphBuilder::create().value();
  const auto qi=builder.input(qd).value(),ki=builder.input(kd).value(),vi=builder.input(vd).value();
  auto descriptor=g::batch::paged_attention(layout,{.scale=.125F,.causal=false}).value();
  EXPECT_FALSE(descriptor.options.allow_bf16_partials);
  const auto node=builder.add_node(c::Operation::custom(descriptor),std::array{qi,ki,vi}).value();
  const auto output=builder.result(node).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto logical=std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{g::apple::execution_source(native).value(),g::StorageDomainId{0}}};
  auto compiled=g::batch::compile(logical,{.storage_domains=domains,.targets=targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  std::array<g::Invocation,2> slots;
  std::array<g::GraphOutput,2> held;
  for(std::size_t i=0;i<slots.size();++i) {
    slots[i]=compiled->executable.create_invocation({.resources=g::ResourceOptions{}}).value();
    const std::array layouts{g::batch::NodeLayout{node,layout}};
    const auto prepared=g::batch::prepare(slots[i],{},layouts);
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    ASSERT_TRUE(slots[i].bind(qi,qd,query).ok());
    ASSERT_TRUE(g::batch::bind(slots[i],ki,key).ok());
    ASSERT_TRUE(g::batch::bind(slots[i],vi,value).ok());
    ASSERT_TRUE(slots[i].prepare_replay({}).ok());
    auto done=slots[i].submit();ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    *done=c::Completion{};
    held[i]=slots[i].result(output).value();
    check_words(held[i].tensor().value(),test,expected);
  }
  check_words(held[0].tensor().value(),test,expected);
  const std::array layouts{g::batch::NodeLayout{node,layout}};
  EXPECT_FALSE(g::batch::prepare(slots[0],{},layouts).ok());
  held[1].reset();
  ASSERT_TRUE(slots[1].prepare_replay({}).ok());
  auto done=slots[1].submit();ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  check_words(slots[1].result(output)->tensor().value(),test,expected);
  check_words(held[0].tensor().value(),test,expected);
}

TEST(AppleAttentionPartials, DirectKeepsRepresentableResultsAcrossBothTwoPassKernels) {
  for(const auto test:cases) paged_case(test,false);
}
TEST(AppleAttentionPartials, DefaultGraphKeepsMathAndIndependentResultLifetimes) {
  for(const auto test:cases) paged_case(test,true);
}
TEST(AppleAttentionPartials, OrdinaryRaggedAndAdjacentBranchesUseMatchingPartials) {
  for(const auto test : {cases[0],cases[3],Case{1,32,8,false,false},Case{16,4096,8,true,false}}) {
    SCOPED_TRACE(::testing::Message() << "Q=" << test.queries << " KV=" << test.keys
        << " H=" << test.heads << " BF16=" << test.bf16);
    const auto dtype=test.bf16 ? mx::bfloat16 : mx::float16;
    const auto values=value_words(test);
    std::vector<double> vd(values.size());
    for(std::size_t i=0;i<values.size();++i)
      vd[i]=test.bf16 ? half_test::decode_bits<B16>(values[i]) : half_test::decode_bits<F16>(values[i]);
    const auto expected=expected_value(test,values);
    const std::array inputs{
        make(std::vector<double>(std::size_t(test.queries)*test.heads*depth),{1,test.queries,test.heads,depth},dtype),
        make(std::vector<double>(std::size_t(test.keys)*kv_heads*depth),{1,test.keys,kv_heads,depth},dtype),
        make(vd,{1,test.keys,kv_heads,depth},dtype)};
    auto context=a::Context::create().value();
    const c::ScaledDotProductAttention op{.scale=.125F};
    auto ordinary=execute(context,op,inputs);ASSERT_TRUE(ordinary) << ordinary.error().message();
    expect(*ordinary,std::vector<double>(std::size_t(test.queries)*test.heads*depth,expected));
    const auto type=logical(dtype);
    const c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.query_type=type,.output_type=type,
        .key_storage=c::CacheStorage::plain(type).value(),.value_storage=c::CacheStorage::plain(type).value(),
        .max_requests=1,.max_query_rows=std::uint32_t(test.queries),.max_kv_rows=std::uint32_t(test.keys),
        .max_pages=0,.max_page_refs=0,
        .query_heads=std::uint32_t(test.heads),.kv_heads=kv_heads,.qk_head_dim=depth,.value_head_dim=depth,.page_size=1};
    auto schema=c::BatchSchema::create(cap).value();
    const std::array<std::int32_t,2> qp{0,test.queries},kp{0,test.keys};
    auto layout=schema.version({qp,kp,{},{}}).value();
    auto program=a::prepare_ragged_batch(context,layout,{cap,.125F,false});
    ASSERT_TRUE(program) << program.error().message();
    auto slot=program->create_invocation().value();
    const std::array packed{
        ready(mx::reshape(inputs[0],{test.queries,test.heads,depth},stream())),
        ready(mx::reshape(inputs[1],{test.keys,kv_heads,depth},stream())),
        ready(mx::reshape(inputs[2],{test.keys,kv_heads,depth},stream()))};
    const auto bound=slot.bind(packed);
    ASSERT_TRUE(bound.ok()) << bound.message();
    auto work=slot.submit();ASSERT_TRUE(work) << work.error().message();
    auto result=work->wait();ASSERT_TRUE(result) << result.error().message();
    expect((*result)->arrays()[0],std::vector<double>(std::size_t(test.queries)*test.heads*depth,expected));
  }
}
TEST(AppleAttentionPartials, GroupedDecodeUsesEachBatchesOwnKeysAndValues) {
  constexpr int batches=2,heads=16,keys=8192;
  std::vector<double> q(batches*heads*depth);
  std::vector<double> k(std::size_t(batches)*keys*kv_heads*depth);
  std::vector<double> v(std::size_t(batches)*keys*kv_heads*depth);
  std::vector<double> expected(std::size_t(batches)*heads*depth);
  for(int b=0;b<batches;++b) {
    const double even_key=b==0 ? .5 : -.25,odd_key=b==0 ? -.5 : .75;
    for(int row=0;row<keys;++row)
      for(int h=0;h<kv_heads;++h) {
        k[((b*keys+row)*kv_heads+h)*depth]=row%2==0 ? even_key : odd_key;
        for(int d=0;d<depth;++d)
          v[((b*keys+row)*kv_heads+h)*depth+d]=(b+1)*4+h+(d%4)*.25+(row%2)*2;
      }
    const double odd_probability=1/(1+std::exp(-(b+1)*.125*(odd_key-even_key)));
    for(int h=0;h<heads;++h) {
      q[(b*heads+h)*depth]=b+1;
      for(int d=0;d<depth;++d)
        expected[(b*heads+h)*depth+d]=(b+1)*4+h/8+(d%4)*.25+2*odd_probability;
    }
  }
  const std::array inputs{
      make(q,{batches,1,heads,depth},mx::float16),
      make(k,{batches,keys,kv_heads,depth},mx::float16),
      make(v,{batches,keys,kv_heads,depth},mx::float16)};
  auto context=a::Context::create().value();
  auto output=execute(context,c::ScaledDotProductAttention{.scale=.125F},inputs);
  ASSERT_TRUE(output) << output.error().message();
  expect(*output,expected);
}
} // namespace
