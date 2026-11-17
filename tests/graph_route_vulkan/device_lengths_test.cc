#include <gtest/gtest.h>

#include "hyle/compute/vulkan_memory.h"
#include "hyle/graph/vulkan.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/result.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/batch_provider.h"
#include "../../src/graph_batch/native_batch_regions.h"
#include "../../src/graph_batch/native_device_append_regions.h"
#include "../compute_provider_native/half_test_support.h"
#include "../compute_route_vulkan/fixture.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <numeric>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace c=hyle::compute; namespace g=hyle::graph; namespace b=g::batch;
namespace a=c::vulkan; using BF16=hyle::simd::bfloat16_t;
template<class T> T take(hyle::Result<T> r,std::source_location at=std::source_location::current()) {
  if (!r) throw std::runtime_error(std::to_string(at.line())+": "+std::string(r.error().message()));
  return std::move(*r);
}
void check(hyle::Status s,std::source_location at=std::source_location::current()) {
  if (!s.ok()) throw std::runtime_error(std::to_string(at.line())+": "+std::string(s.message()));
}
constexpr std::uint32_t heads=4,kv_heads=2,depth=64,rows=4,page_size=16;

struct Cell final {
  a::Context context=take(vulkan_test::context());
  a::MemoryResource memory=take(a::MemoryResource::create(context,a::MemoryKind::kDevice));
  a::MemoryResource staging=take(a::MemoryResource::create(context,a::MemoryKind::kStaging));
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
    const auto source=take(g::vulkan::execution_source(context));
    const std::array domains{g::StorageDomain{memory.ref()},g::StorageDomain{staging.ref()}};
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

TEST(VulkanGraphDeviceLengths, NativeGraphConsumesDeviceProducedLengthsAcrossRepeatedInvocations) {
  Cell cell;auto invocation=cell.prepare();
  cell.run(invocation,{1,2},{17,31});cell.run(invocation,{2,1},{31,17});cell.run(invocation,{0,2},{0,19});
}
TEST(VulkanGraphDeviceLengths, InvalidDeviceControlFailsTheWholeGraph) {
  Cell cell;auto invocation=cell.prepare();cell.run(invocation,{3,1},{31,17},true);
}
TEST(VulkanGraphDeviceLengths, NewOwnedEnvelopeReusesTheOriginalPreparedSlot) {
  Cell cell;auto invocation=cell.prepare();cell.run(invocation,{1,2},{17,31});
  cell.envelope=cell.layout(cell.schema,true);cell.configure(invocation);cell.run(invocation,{1,3},{17,31});
}
TEST(VulkanGraphDeviceLengths, ForeignEnvelopeDoesNotRevokeThePreparedConfiguration) {
  Cell cell;auto invocation=cell.prepare();cell.run(invocation,{1,2},{17,31});
  auto foreign=take(c::BatchSchema::create(cell.schema.capacity()));
  const std::array layouts{b::NodeLayout{cell.reader,cell.layout(foreign)}};
  EXPECT_TRUE(b::prepare_lengths(invocation,{},layouts,{.host_workspace=hyle::mem::system_resource()}).IsInvalidArgument());
  cell.run(invocation,{2,1},{31,17});
}
TEST(VulkanGraphDeviceLengths, PreparedSlotsKeepTheirOwnMetadataAndControlInputs) {
  Cell cell;auto first=cell.prepare();
  cell.envelope=cell.layout(cell.schema,true);auto second=cell.prepare();
  cell.run(first,{2,1},{31,17});cell.run(second,{1,3},{17,31});cell.run(first,{1,2},{17,31});
}
TEST(VulkanGraphDeviceLengths, InvalidControlDoesNotPoisonAnotherPreparedSlot) {
  Cell cell;auto first=cell.prepare();cell.run(first,{3,1},{31,17},true);
  auto second=cell.prepare();cell.run(second,{1,2},{17,31});
}


struct AppendCell final {
  enum class QueryMode { kAppendedRows,kLast,kForeignKv,kFailingControl,kTransitiveFailingControl };
  static constexpr std::uint32_t requests=2,query_capacity=3,minimum=15,prior=18,maximum=21;
  static constexpr std::uint32_t old_pages=2,new_pages=2,physical_pages=9,dim=64,row_bytes=dim*2;
  a::Context context=take(vulkan_test::context());
  a::MemoryResource memory=take(a::MemoryResource::create(context,a::MemoryKind::kDevice));
  a::MemoryResource staging=take(a::MemoryResource::create(context,a::MemoryKind::kStaging));
  c::BatchSchema schema;
  c::PagedArena keys,values;
  std::vector<c::ReadLease> old_guards,value_guards;
  c::PagedArenaRead frame,value_frame;
  c::BatchLayoutVersion previous,lower,capacity,reader_capacity,source_capacity;
  c::TensorDesc row_desc,query_desc,control_desc;
  c::LeasePool payload;
  c::ReadLease rows,query;
  g::Graph graph;
  g::CompiledGraph compiled;
  std::array<g::ValueId,3> ports{};
  std::array<g::ValueId,2> lengths{};
  g::ValueId key_port,value_port,query_port,reader_query_port,output;
  g::ValueId failure_indices,source_key_port,source_value_port,source_kv_port;
  g::NodeId append,reader,source_reader;
  bool last_query,checked_producer,transitive;
  c::BatchCacheRole role;

  explicit AppendCell(QueryMode mode=QueryMode::kAppendedRows,c::BatchCacheRole written=c::BatchCacheRole::kKey)
      :last_query(mode!=QueryMode::kAppendedRows),
       checked_producer(mode==QueryMode::kFailingControl || mode==QueryMode::kTransitiveFailingControl),
       transitive(mode==QueryMode::kTransitiveFailingControl),role(written) {
    const auto plain=take(c::CacheStorage::plain(c::DataType::kBFloat16));
    schema=take(c::BatchSchema::create({.query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
        .key_storage=plain,.value_storage=plain,.max_requests=requests,.max_query_rows=requests*query_capacity,
        .max_kv_rows=requests*maximum,.max_pages=physical_pages,.max_page_refs=requests*new_pages,
        .query_heads=1,.kv_heads=1,.qk_head_dim=dim,.value_head_dim=dim,.page_size=page_size}));
    previous=layout(0,prior,false);lower=layout(0,minimum,false);capacity=layout(query_capacity,maximum,true);
    reader_capacity=last_query?layout(1,maximum,true):capacity;
    const c::PagedArenaOptions options{.max_backing_bytes=std::size_t(physical_pages)*page_size*row_bytes+1};
    keys=take(c::PagedArena::create(schema,role,memory.ref(),options));
    values=take(c::PagedArena::create(schema,role==c::BatchCacheRole::kKey?c::BatchCacheRole::kValue:c::BatchCacheRole::kKey,memory.ref(),options));
    for (std::uint32_t p=0;p<requests*old_pages;++p) {
      auto writer=take(keys.acquire(p));std::vector<std::uint16_t> words(page_size*dim);
      for (std::uint32_t row=0;row<page_size;++row)
        std::fill_n(words.begin()+row*dim,dim,old_value(p/old_pages,(p%old_pages)*page_size+row));
      check(take(writer.lease().view()).write(std::as_bytes(std::span(words))));
      old_guards.push_back(take(writer.publish(page_size)));
    }
    frame=take(keys.read(previous,old_guards));
    for (std::uint32_t p=0;p<physical_pages;++p) {
      auto writer=take(values.acquire(p));const std::vector<std::uint16_t> ones(page_size*dim,0x3f80);
      check(take(writer.lease().view()).write(std::as_bytes(std::span(ones))));
      value_guards.push_back(take(writer.publish(page_size)));
    }
    std::vector<c::ReadLease> used;
    for (auto page:take(c::detail::BatchLayoutAccess::page_footprint(reader_capacity)))
      used.push_back(value_guards[std::size_t(page)]);
    value_frame=take(values.read(reader_capacity,used));
    row_desc=take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{requests*query_capacity,1,dim}));
    query_desc=last_query?take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{requests,1,dim})):row_desc;
    control_desc=take(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{requests}));
    payload=take(c::LeasePool::create(memory.ref(),2*take(row_desc.required_bytes())+4096,{.max_blocks=8}));
    auto packed=take(payload.allocate(take(row_desc.required_bytes())));
    std::vector<std::uint16_t> words(requests*query_capacity*dim);
    for (std::uint32_t row=0;row<requests*query_capacity;++row)
      std::fill_n(words.begin()+row*dim,dim,input_value(row));
    check(take(packed.view()).write(std::as_bytes(std::span(words))));rows=take(packed.publish());
    auto q=take(payload.allocate(take(query_desc.required_bytes())));check(take(q.view()).fill(std::byte{0}));query=take(q.publish());
    auto builder=take(g::GraphBuilder::create());
    ports={take(builder.input(row_desc)),take(builder.input(control_desc)),take(builder.input(control_desc))};
    lengths={take(builder.add(c::Copy{},ports[1])),take(builder.add(c::Copy{},ports[2]))};
    g::NodeId checked_node;
    auto append_rows=ports[0];
    if (checked_producer) {
      failure_indices=take(builder.input(take(c::TensorDesc::dense(c::DataType::kInt32,
          std::array<std::int64_t,1>{requests*query_capacity}))));
      checked_node=take(builder.add_node(c::Operation(c::Gather{}),std::array{ports[0],failure_indices}));
      append_rows=take(builder.result(checked_node));
      if (transitive) {
        // A real attention region separates the checked producer from append;
        // its output must retain the original Gather error provenance.
        source_capacity=layout(query_capacity,prior,false);
        source_key_port=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kKey))));
        source_value_port=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kValue))));
        source_kv_port=take(builder.input(control_desc));
        const auto histories=take(builder.add(c::Copy{},source_kv_port));
        source_reader=take(builder.add_node(c::Operation::custom(take(b::device_lengths_paged_attention(
            schema,requests*query_capacity,{.scale=.125f,.causal=true}))),
            std::array{append_rows,source_key_port,source_value_port,lengths[0],histories}));
        append_rows=take(builder.result(source_reader));
      }
    }
    append=take(builder.add_node(c::Operation::custom(take(b::device_lengths_paged_append(schema,role))),
        std::array{append_rows,lengths[0],lengths[1]}));
    key_port=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kKey))));
    value_port=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kValue))));
    query_port=take(builder.input(query_desc));
    auto reader_q=lengths[0];
    if (last_query) { reader_query_port=take(builder.input(control_desc));reader_q=take(builder.add(c::Copy{},reader_query_port)); }
    const auto reader_kv=mode==QueryMode::kForeignKv?take(builder.add(c::Copy{},lengths[1])):lengths[1];
    reader=take(builder.add_node(c::Operation::custom(take(b::device_lengths_paged_attention(schema,
        last_query?requests:requests*query_capacity,{.scale=.125f,.causal=true}))),
        std::array{query_port,key_port,value_port,reader_q,reader_kv}));
    check(builder.depends_on(reader,append));output=take(builder.result(reader));check(builder.publish(output));
    graph=take(std::move(builder).finish());const auto source=take(g::vulkan::execution_source(context));
    const std::array domains{g::StorageDomain{memory.ref()},g::StorageDomain{staging.ref()}};
    const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
    compiled=take(g::compile(graph,{.storage_domains=domains,.targets=targets}));
    if (transitive) {
      std::array<std::size_t,2> steps{SIZE_MAX,SIZE_MAX};
      for (std::size_t i=0;i<compiled.report.step_count();++i) {
        const auto step=take(compiled.report.step(i));
        for (std::size_t j=0;j<step.node_count;++j) {
          const auto node=take(compiled.report.step_node(i,j));
          if (node==checked_node) steps[0]=i;
          if (node==source_reader) steps[1]=i;
        }
      }
      if (steps[0]==SIZE_MAX || steps[1]==SIZE_MAX || steps[0]==steps[1])
        throw std::runtime_error("checked producer and attention require distinct original native regions");
    }
  }
  static std::uint16_t old_value(std::uint32_t request,std::uint32_t row) {
    return std::uint16_t(0x3f00+(request*13+row)%128);
  }
  static std::uint16_t input_value(std::uint32_t row) { return std::uint16_t(0x4000+row); }
  static std::uint32_t destination(std::uint32_t request,std::uint32_t page) {
    return requests*old_pages+request*new_pages+page;
  }
  c::BatchLayoutVersion layout(std::uint32_t q,std::uint32_t kv,bool next) {
    std::array<std::int32_t,3> queries{},ptr{};std::vector<std::int32_t> ids;
    std::array<std::int32_t,2> tails{};const auto pages=(kv+page_size-1)/page_size;
    for (std::uint32_t r=0;r<requests;++r) {
      queries[r+1]=queries[r]+q;ptr[r+1]=ptr[r]+pages;tails[r]=(kv-1)%page_size+1;
      for (std::uint32_t p=0;p<pages;++p) ids.push_back(next?destination(r,p):r*old_pages+p);
    }
    return take(schema.version({queries,ptr,ids,tails}));
  }
  g::Invocation prepare() {
    auto invocation=take(compiled.executable.create_invocation());check(invocation.prepare_resources());
    std::vector layouts{b::NodeLayout{reader,reader_capacity}};
    if (transitive) layouts.push_back({source_reader,source_capacity});
    check(b::prepare_lengths(invocation,{},layouts,{.host_workspace=hyle::mem::system_resource()}));
    return invocation;
  }
  b::PagedAppendPages pages() {
    std::vector<c::PagedPageWriter> writers;
    for (std::uint32_t p=requests*old_pages;p<physical_pages;++p) {
      auto writer=take(keys.acquire(p));check(take(writer.lease().view()).fill(std::byte{0x5a}));
      if (p+1!=physical_pages) writers.push_back(std::move(writer));
    }
    const std::array sources{take(c::PagedArenaFootprint::from_read(frame))};
    return take(b::PagedAppendPages::create_lengths(keys,capacity,lower,previous,sources,writers));
  }
  void run(std::array<std::int32_t,2> q,std::array<std::int32_t,2> kv,bool invalid=false,bool producer_failed=false) {
    const bool failed=invalid || producer_failed;
    auto invocation=prepare();check(invocation.bind(ports[0],row_desc,rows));
    check(invocation.bind(query_port,query_desc,query));
    check(b::bind(invocation,role==c::BatchCacheRole::kKey?value_port:key_port,value_frame));
    std::vector<c::ReadLease> controls;
    for (std::size_t i=0;i<(last_query?3u:2u);++i) {
      const auto data=i==0?q:i==1?kv:std::array<std::int32_t,2>{1,1};
      auto writer=take(payload.allocate(sizeof(data),8));check(take(writer.view()).write(std::as_bytes(std::span(data))));
      controls.push_back(take(writer.publish()));
      check(invocation.bind(i<2?ports[i+1]:reader_query_port,control_desc,controls.back()));
    }
    if (checked_producer) {
      std::array<std::int32_t,requests*query_capacity> indices{};
      std::iota(indices.begin(),indices.end(),0);
      if (producer_failed) indices.back()=std::int32_t(indices.size());
      auto writer=take(payload.allocate(sizeof(indices),8));
      check(take(writer.view()).write(std::as_bytes(std::span(indices))));
      controls.push_back(take(writer.publish()));
      check(invocation.bind(failure_indices,take(c::TensorDesc::dense(c::DataType::kInt32,
          std::array<std::int64_t,1>{requests*query_capacity})),controls.back()));
      if (transitive) {
        const std::array<std::int32_t,2> histories{prior,prior};
        auto writer=take(payload.allocate(sizeof(histories),8));
        check(take(writer.view()).write(std::as_bytes(std::span(histories))));
        controls.push_back(take(writer.publish()));
        check(invocation.bind(source_kv_port,control_desc,controls.back()));
        auto old_keys=take(keys.read(source_capacity,old_guards));
        const auto used=std::span(value_guards).first(requests*old_pages);
        auto old_values=take(values.read(source_capacity,used));
        check(b::bind(invocation,source_key_port,old_keys));check(b::bind(invocation,source_value_port,old_values));
      }
    }
    auto transaction=pages();EXPECT_FALSE(transaction.select_lengths(q,kv).ok());
    check(b::bind_append(invocation,graph,append,role==c::BatchCacheRole::kKey?key_port:value_port,std::move(transaction)));
    struct Selection { std::array<g::ValueId,2> controls;bool found=false; } selection{lengths};
    check(g::detail::RegionAccess::visit(invocation,[](void *opaque,const g::detail::RegionAccessView &view) noexcept -> hyle::Status {
      if (!view.prepared_vtable || !view.prepared_vtable->query_extension) return hyle::Status::Ok();
      const auto *table=static_cast<const b::detail::NativeDeviceAppendRegionVTable *>(
          view.prepared_vtable->query_extension(view.prepared,b::detail::kNativeDeviceAppendRegionExtensionId));
      if (!table) return hyle::Status::Ok();
      auto &wanted=*static_cast<Selection *>(opaque);
      wanted.found=table->length_inputs(view.prepared)==wanted.controls;
      return hyle::Status::Ok();
    },&selection));
    EXPECT_TRUE(selection.found)<<"original native device-controlled append was not selected";
    check(invocation.prepare_replay({}));auto completion=take(invocation.submit());
    const auto status=completion.wait();completion={};
    if (failed) {
      if (producer_failed) EXPECT_TRUE(status.IsOutOfRange())<<status.message();
      else EXPECT_TRUE(status.IsInvalidArgument())<<status.message();
      EXPECT_FALSE(invocation.output(output));
    }
    else {
      check(status);auto tensor=take(invocation.output(output));
      std::vector<std::uint16_t> actual(take(query_desc.required_bytes())/2);
      check(tensor.storage().read(std::as_writable_bytes(std::span(actual))));
      std::uint32_t packed=0,output_row=0;
      for (std::uint32_t r=0;r<requests;++r) {
        const auto queries=last_query?1:q[r];
        for (std::int32_t row=0;row<queries;++row,++output_row) {
          const auto previous_rows=kv[r]-q[r],end=last_query?kv[r]:previous_rows+row+1;
          double sum=0;
          if (role==c::BatchCacheRole::kValue) for (std::int32_t j=0;j<end;++j) {
            BF16 word;word.bits=j<previous_rows?old_value(r,j):input_value(packed+j-previous_rows);
            sum+=half_test::decode(word);
          }
          const auto expected=role==c::BatchCacheRole::kKey?1.0:sum/end;
          for (std::uint32_t d=0;d<dim;++d) {
            BF16 word;word.bits=actual[output_row*dim+d];
            EXPECT_NEAR(half_test::decode(word),expected,.008)<<r<<','<<row<<','<<d;
          }
        }
        packed+=q[r];
      }
      for (std::size_t i=std::size_t(output_row)*dim;i<actual.size();++i) EXPECT_EQ(actual[i],0u)<<i;
    }
    auto returned=take(b::take_append(invocation,append));
    if (failed) { EXPECT_FALSE(returned.select_lengths(q,kv).ok());EXPECT_FALSE(returned.ready()); }
    else {
      EXPECT_FALSE(returned.ready());check(returned.select_lengths(q,kv));
      auto published=take(returned.ready());EXPECT_EQ(published.layout().facts().query_rows,std::uint32_t(q[0]+q[1]));
    }
    auto bank=take(c::detail::PagedArenaAccess::tensor(keys));
    std::vector<std::uint16_t> actual(take(bank.description().required_bytes())/2);
    check(bank.storage().read(std::as_writable_bytes(std::span(actual))));std::uint32_t packed=0;
    for (std::uint32_t r=0;r<requests;++r) {
      for (std::uint32_t row=0;row<old_pages*page_size;++row) for (std::uint32_t d=0;d<dim;++d)
        EXPECT_EQ(actual[(r*old_pages*page_size+row)*dim+d],old_value(r,row));
      for (std::uint32_t row=0;row<new_pages*page_size;++row) {
        auto expected=std::uint16_t(0x5a5a);
        if (!failed && row<maximum) {
          const auto previous_rows=std::uint32_t(kv[r]-q[r]);expected=0;
          if (row<previous_rows) expected=old_value(r,row);
          else if (row<std::uint32_t(kv[r])) expected=transitive?0x3f80:input_value(packed+row-previous_rows);
        }
        for (std::uint32_t d=0;d<dim;++d)
          EXPECT_EQ(actual[(destination(r,row/page_size)*page_size+row%page_size)*dim+d],expected)<<r<<','<<row<<','<<d;
      }
      if (!failed) packed+=q[r];
    }
    for (std::size_t i=(physical_pages-1)*page_size*dim;i<actual.size();++i) EXPECT_EQ(actual[i],0x5a5a);
    if (failed) {
      invocation={};
      for (std::size_t i=0;i<returned.page_count();++i) {
        auto item=returned.page(i);ASSERT_TRUE(item);auto &writer=*item;writer.lease().reset();check(writer.discard());
      }
    }
  }
};

TEST(VulkanGraphDeviceAppend, NativeCowWriteFeedsAttentionWithActualDeviceLengths) {
  AppendCell cell;cell.run({1,3},{17,21});
}
TEST(VulkanGraphDeviceAppend, LastQueryHasItsOwnControlAndTheSameOriginalKvIdentity) {
  AppendCell cell(AppendCell::QueryMode::kLast);cell.run({3,1},{21,17});cell.run({0,2},{18,20});
}
TEST(VulkanGraphDeviceAppend, ValueWriterUsesItsOriginalRoleAndDeviceVerdict) {
  AppendCell cell(AppendCell::QueryMode::kLast,c::BatchCacheRole::kValue);cell.run({3,1},{21,17});
}
TEST(VulkanGraphDeviceAppend, InvalidWriterFailsEvenWhenReaderControlsWouldBeValid) {
  AppendCell cell(AppendCell::QueryMode::kLast);cell.run({0,0},{14,18},true);cell.run({1,2},{17,20});
}
TEST(VulkanGraphDeviceAppend, OriginalProducerErrorPreventsWritesDespiteLegalNumericControls) {
  AppendCell cell(AppendCell::QueryMode::kFailingControl);
  cell.run({1,2},{17,20},false,true);cell.run({1,2},{17,20});
}
TEST(VulkanGraphDeviceAppend, OriginalErrorSurvivesAnotherNativeRegionAndFreshRetiredWork) {
  AppendCell cell(AppendCell::QueryMode::kTransitiveFailingControl);
  cell.run({1,2},{17,20},false,true);cell.run({1,2},{17,20});
}
TEST(VulkanGraphDeviceAppend, EqualValuesFromAnotherKvProducerCannotReplaceOriginalIdentity) {
  AppendCell cell(AppendCell::QueryMode::kForeignKv);auto invocation=cell.prepare();auto pages=cell.pages();
  const auto status=b::bind_append(invocation,cell.graph,cell.append,cell.key_port,std::move(pages));
  EXPECT_TRUE(status.IsInvalidArgument());EXPECT_NE(status.message().find("original KV"),std::string_view::npos);
  EXPECT_TRUE(pages.valid());
  for (std::size_t i=0;i<pages.page_count();++i) {
    auto writer=pages.page(i);ASSERT_TRUE(writer);EXPECT_TRUE(writer->valid() && writer->lease().valid());
  }
}
} // namespace
