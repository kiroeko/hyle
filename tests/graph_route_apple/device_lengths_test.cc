#include <gtest/gtest.h>

#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/result.h"
#include "hyle/graph/detail/region_access.h"
#include "../../src/graph_batch/native_batch_regions.h"
#include "../compute_provider_native/half_test_support.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace c=hyle::compute; namespace g=hyle::graph; namespace b=g::batch;
namespace a=c::apple; using BF16=hyle::simd::bfloat16_t;
template<class T> T take(hyle::Result<T> r) {
  if (!r) throw std::runtime_error(std::string(r.error().message()));
  return std::move(*r);
}
void check(hyle::Status s) { if (!s.ok()) throw std::runtime_error(std::string(s.message())); }
constexpr std::uint32_t heads=4,kv_heads=2,depth=64,rows=4,page_size=16;

struct Cell final {
  a::Context context=take(a::Context::create());
  a::MemoryResource memory=take(a::MemoryResource::create(context));
  c::BatchSchema schema;
  c::BatchLayoutVersion envelope;
  c::PagedArena keys,values;
  std::vector<c::ReadLease> key_pages,value_pages;
  c::PagedArenaRead key_frame,value_frame;
  c::LeasePool inputs;
  c::ReadLease query;
  std::vector<c::ReadLease> controls;
  c::TensorDesc query_description,index_description,output_description;
  g::Graph graph;
  g::CompiledGraph compiled;
  std::array<g::ValueId,5> ports{};
  std::array<g::ValueId,2> lengths{};
  g::NodeId reader;
  g::ValueId output;
  const std::array<std::int32_t,6> pages{5,1,3,0,4,2};

  Cell() {
    const auto storage=take(c::CacheStorage::plain(c::DataType::kBFloat16));
    const c::BatchCapacity capacity{.query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
        .key_storage=storage,.value_storage=storage,.max_requests=2,.max_query_rows=rows,.max_kv_rows=78,
        .max_pages=6,.max_page_refs=6,.query_heads=heads,.kv_heads=kv_heads,
        .qk_head_dim=depth,.value_head_dim=depth,.page_size=page_size};
    schema=take(c::BatchSchema::create(capacity)); envelope=layout(schema);
    const auto backing=std::size_t(capacity.max_pages)*page_size*kv_heads*depth*2+1;
    keys=take(c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory.ref(),{.max_backing_bytes=backing}));
    values=take(c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory.ref(),{.max_backing_bytes=backing}));
    std::array<std::uint32_t,6> logical{};
    for (std::uint32_t i=0;i<pages.size();++i) logical[pages[i]]=i;
    for (std::uint32_t page=0;page<6;++page) {
      auto k=take(keys.acquire(page)),v=take(values.acquire(page));
      std::vector<BF16> payload(page_size*kv_heads*depth);
      for (std::uint32_t row=0;row<page_size;++row)
        for (std::uint32_t h=0;h<kv_heads;++h) for (std::uint32_t d=0;d<depth;++d)
          payload[(row*kv_heads+h)*depth+d]=half_test::encode<BF16>(sample(logical[page]/3,(logical[page]%3)*16+row,h,d));
      check(take(k.lease().view()).fill(std::byte{0}));
      check(take(v.lease().view()).write(std::as_bytes(std::span(payload))));
      key_pages.push_back(take(k.publish(page_size)));value_pages.push_back(take(v.publish(page_size)));
    }
    key_frame=take(keys.read(envelope,key_pages));value_frame=take(values.read(envelope,value_pages));
    query_description=take(c::batch_query_description(capacity,rows));
    output_description=take(c::batch_output_description(capacity,rows));
    index_description=take(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{2}));
    inputs=take(c::LeasePool::create(memory.ref(),take(query_description.required_bytes())+4096,{.max_blocks=20}));
    auto q=take(inputs.allocate(take(query_description.required_bytes()))); check(take(q.view()).fill(std::byte{0})); query=take(q.publish());
    auto builder=take(g::GraphBuilder::create());
    ports[0]=take(builder.input(query_description));
    ports[1]=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kKey))));
    ports[2]=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kValue))));
    ports[3]=take(builder.input(index_description));ports[4]=take(builder.input(index_description));
    lengths[0]=take(builder.add(c::Copy{},ports[3]));lengths[1]=take(builder.add(c::Copy{},ports[4]));
    const auto attention=take(b::device_lengths_paged_attention(schema,rows,{.scale=.125f}));
    reader=take(builder.add_node(c::Operation::custom(attention),std::array{ports[0],ports[1],ports[2],lengths[0],lengths[1]}));
    output=take(builder.result(reader));check(builder.publish(output));graph=take(std::move(builder).finish());
    const auto source=take(g::apple::execution_source(context));
    const std::array domains{g::StorageDomain{memory.ref()}};
    const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
    compiled=take(g::compile(graph,{.storage_domains=domains,.targets=targets}));
  }
  c::BatchLayoutVersion layout(const c::BatchSchema &domain,bool alternate=false) {
    const std::array<std::int32_t,3> q=alternate?std::array<std::int32_t,3>{0,1,4}:std::array<std::int32_t,3>{0,2,4};
    const std::array<std::int32_t,3> k{0,3,6}; const std::array<std::int32_t,2> tails{7,7};
    return take(domain.version({q,k,pages,tails}));
  }
  static float sample(std::uint32_t request,std::uint32_t row,std::uint32_t head,std::uint32_t dim) {
    return half_test::decode(half_test::encode<BF16>(float(request)+float(row)/64+float(head)/32+float(dim%7)/128));
  }
  void configure(g::Invocation &invocation) {
    const std::array layouts{b::NodeLayout{reader,envelope}};
    check(b::prepare_lengths(invocation,{},layouts,{.host_workspace=hyle::mem::system_resource()}));
    key_frame=take(keys.read(envelope,key_pages));value_frame=take(values.read(envelope,value_pages));
    check(b::bind(invocation,ports[1],key_frame));check(b::bind(invocation,ports[2],value_frame));
  }
  g::Invocation prepare() {
    auto invocation=take(compiled.executable.create_invocation());check(invocation.prepare_resources());
    configure(invocation);return invocation;
  }
  void run(g::Invocation &invocation,std::array<std::int32_t,2> active,
      std::array<std::int32_t,2> histories,bool invalid=false) {
    check(invocation.bind(ports[0],query_description,query));
    for (std::size_t i=0;i<2;++i) {
      const auto &data=i?histories:active;
      auto writer=take(inputs.allocate(sizeof(data),8));check(take(writer.view()).write(std::as_bytes(std::span(data))));
      controls.push_back(take(writer.publish()));check(invocation.bind(ports[3+i],index_description,controls.back()));
    }
    check(invocation.prepare_replay({}));
    auto completion=take(invocation.submit());auto status=completion.wait();completion={};
    if (invalid) {
      EXPECT_TRUE(status.IsInvalidArgument()) << status.message(); EXPECT_FALSE(invocation.output(output));return;
    }
    check(status);
    auto tensor=take(invocation.output(output));
    std::vector<BF16> actual(rows*heads*depth);check(tensor.storage().read(std::as_writable_bytes(std::span(actual))));
    std::uint32_t packed=0;
    for (std::uint32_t r=0;r<2;++r) for (int q=0;q<active[r];++q,++packed)
      for (std::uint32_t h=0;h<heads;++h) for (std::uint32_t d=0;d<depth;++d) {
        const auto count=histories[r]-active[r]+q+1;double sum=0;
        for (int k=0;k<count;++k) sum+=sample(r,k,h/(heads/kv_heads),d);
        const auto expected=half_test::decode(half_test::encode<BF16>(float(sum/count)));
        EXPECT_NEAR(half_test::decode(actual[(packed*heads+h)*depth+d]),expected,.008) << r << ',' << q << ',' << h << ',' << d;
      }
    for (std::size_t i=std::size_t(packed)*heads*depth;i<actual.size();++i) EXPECT_EQ(actual[i].bits,0u);
    struct Facts { g::NodeId node; std::array<g::ValueId,2> controls; bool found=false; } facts{reader,lengths};
    check(g::detail::RegionAccess::visit(invocation,[](void *opaque,const g::detail::RegionAccessView &view) noexcept -> hyle::Status {
      auto &wanted=*static_cast<Facts *>(opaque);
      if (!view.prepared_vtable || !view.prepared_vtable->query_extension) return hyle::Status::Ok();
      const auto *table=static_cast<const b::detail::NativeBatchRegionVTable *>(view.prepared_vtable->query_extension(view.prepared,b::detail::kNativeBatchRegionExtensionId));
      if (!table) return hyle::Status::Ok();
      auto actual=table->inspect(view.prepared,view.invocation);if (!actual) return actual.error();
      if (actual->node==wanted.node) wanted.found=actual->device_lengths && actual->length_inputs==wanted.controls;
      return hyle::Status::Ok();
    },&facts));
    EXPECT_TRUE(facts.found) << "actual device-length Graph region was not selected";
    EXPECT_TRUE(take(b::native_report(invocation,reader)).reused_native_plan);
  }
};

TEST(AppleGraphDeviceLengths, NativeGraphConsumesDeviceProducedLengthsAcrossRepeatedInvocations) {
  Cell cell;auto invocation=cell.prepare();
  cell.run(invocation,{1,2},{17,31});cell.run(invocation,{2,1},{31,17});cell.run(invocation,{0,2},{0,19});
}
TEST(AppleGraphDeviceLengths, InvalidDeviceControlFailsTheWholeGraph) {
  Cell cell;auto invocation=cell.prepare();cell.run(invocation,{3,1},{31,17},true);
}
TEST(AppleGraphDeviceLengths, NewOwnedEnvelopeReusesTheOriginalPreparedSlot) {
  Cell cell;auto invocation=cell.prepare();cell.run(invocation,{1,2},{17,31});
  cell.envelope=cell.layout(cell.schema,true);cell.configure(invocation);cell.run(invocation,{1,3},{17,31});
}
TEST(AppleGraphDeviceLengths, ForeignEnvelopeDoesNotRevokeThePreparedConfiguration) {
  Cell cell;auto invocation=cell.prepare();cell.run(invocation,{1,2},{17,31});
  auto foreign=take(c::BatchSchema::create(cell.schema.capacity()));
  const std::array layouts{b::NodeLayout{cell.reader,cell.layout(foreign)}};
  EXPECT_TRUE(b::prepare_lengths(invocation,{},layouts,{.host_workspace=hyle::mem::system_resource()}).IsInvalidArgument());
  cell.run(invocation,{2,1},{31,17});
}
TEST(AppleGraphDeviceLengths, PreparedSlotsKeepTheirOwnMetadataAndControlInputs) {
  Cell cell;auto first=cell.prepare();
  cell.envelope=cell.layout(cell.schema,true);auto second=cell.prepare();
  cell.run(first,{2,1},{31,17});cell.run(second,{1,3},{17,31});cell.run(first,{1,2},{17,31});
}
TEST(AppleGraphDeviceLengths, InvalidControlDoesNotPoisonAnotherPreparedSlot) {
  Cell cell;auto first=cell.prepare();cell.run(first,{3,1},{31,17},true);
  auto second=cell.prepare();cell.run(second,{1,2},{17,31});
}
} // namespace
