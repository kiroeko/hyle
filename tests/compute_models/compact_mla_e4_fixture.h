#pragma once
#include "paged_e4_kv_fixture.h"

namespace compact_mla_e4_test {
namespace c=hyle::compute;namespace g=hyle::graph;namespace m=hyle::mem;
namespace scalar=nvfp4_reference_test;
using scalar::take;using scalar::check;using scalar::dense;using Type=c::DataType;

struct Case final {
  static constexpr unsigned pages=5,page_rows=16,heads=64,dv=512;
  static constexpr float attention_scale=.0625F;
  unsigned depth;bool paged,single,exact;float scale;
  c::BatchSchema schema;std::array<c::BatchLayoutVersion,2> layouts;
  c::TensorDesc query_description,cache_description,output_description;
  std::array<g::ValueId,3> inputs;g::NodeId node;g::ValueId output;g::Graph graph;
  std::vector<std::uint8_t> bank;std::vector<std::uint16_t> query;
  Case(unsigned d,bool p,bool q1,bool scale_witness=false)
      :depth(d),paged(p),single(q1),exact(scale_witness),
       scale(scale_witness?std::bit_cast<float>(0x3f805586u):.5F) {
    const c::MlaBatchCapacity cap{.storage=p?c::BatchStorage::kPaged:c::BatchStorage::kRagged,
        .query_type=Type::kBFloat16,.output_type=Type::kBFloat16,
        .cache_storage=take(c::CacheStorage::float8(scale),"shared CKV scale"),
        .layout=d==576?c::MlaCacheLayout::kCkv512Rope64:c::MlaCacheLayout::kCkv512,
        .max_requests=q1?1u:2u,.max_query_rows=q1?1u:4u,.max_kv_rows=64,
        .max_pages=p?pages:0,.max_page_refs=p?4u:0u,.query_heads=heads,.page_size=p?page_rows:1u};
    schema=take(c::BatchSchema::create_mla(cap),"uniform compact E4 schema");
    const std::array<std::int32_t,3> qp{0,1,4},pk{0,2,4};
    const std::array<std::int32_t,4> ids{2,0,1,0};
    for(unsigned epoch=0;epoch<2;++epoch) {
      auto refs=pk;std::array<std::int32_t,3> rk{0,q1?(epoch?31:17):17,epoch?48:40};
      std::array<std::int32_t,2> tails{q1?(epoch?15:1):1,epoch?15:7};
      if(exact){refs[1]=1;rk[1]=1;tails[0]=1;}
      layouts[epoch]=take(schema.version({std::span(qp).first(q1?2:3),
          std::span(p?refs:rk).first(q1?2:3),p?std::span(ids).first(exact?1:q1?2:4):std::span<const std::int32_t>{},
          p?std::span(tails).first(q1?1:2):std::span<const std::int32_t>{}}),"compact layout");
    }
    query_description=take(c::batch_query_description(schema.capacity()),"compact Q");
    cache_description=take(c::batch_cache_description(schema,c::BatchCacheRole::kMla),"one compact bank");
    output_description=take(c::batch_output_description(schema.capacity(),q1?1:4),"compact Y");
    const auto metadata=take(schema.metadata_description(),"MLA metadata");
    auto builder=take(g::GraphBuilder::create(),"compact builder");
    const std::array descriptions{query_description,cache_description,metadata};
    for(unsigned port=0;port<(p?2u:3u);++port)inputs[port]=take(builder.input(descriptions[port]),"MLA port");
    const auto operation=p?c::Operation::custom(take(g::batch::paged_mla_attention(layouts[0],
        {.scale=attention_scale}),"owned compact policy")):
        c::Operation::custom(c::RaggedMlaAttention{cap,attention_scale});
    node=take(builder.add_node(operation,std::span(inputs).first(p?2:3)),"original MLA node");output=take(builder.result(node),"MLA Y");
    check(builder.publish(output),"publish MLA");graph=take(std::move(builder).finish(),"compact graph");
    bank.resize(cache_description.required_bytes().value(),0x7f);
    const unsigned valid_rows=p?3*page_rows:48;
    for(unsigned row=0;row<valid_rows;++row) {
      if(p && row==15)continue;
      for(unsigned lane=0;lane<depth;++lane) {
        const auto value=lane>=512?float(int(row%5)-2)*.25F:
            float(int((row*5+lane%11)%13)-6)*.0625F;
        const auto code=exact?std::uint8_t{3}:paged_e4_kv_test::encode(value,scale);
        bank[row*depth+lane]=code;
      }
    }
    query.resize(query_description.element_count().value());
    for(unsigned t=0;t<(q1?1u:4u);++t)for(unsigned h=0;h<heads;++h)for(unsigned lane=0;lane<depth;++lane)
      query[(t*heads+h)*depth+lane]=scalar::bf16(lane>=512?.25F*float((h+t)%3+1):
          float(int((h+t+lane)%7)-3)*.0625F);
  }
  std::vector<std::uint16_t> oracle(unsigned epoch,bool wrong_scale=false,bool omit_kpe=false)const {
    const auto host=layouts[epoch].host();std::vector<std::uint16_t> result(layouts[epoch].facts().query_rows*heads*dv);
    const auto value=[&](unsigned row,unsigned lane) {
      return double(scalar::e4m3(bank[row*depth+lane]))*(wrong_scale?1.:double(scale));
    };
    for(unsigned r=0;r<layouts[epoch].facts().requests;++r) {
      const auto first=unsigned(host.query_indptr[r]),nq=unsigned(host.query_indptr[r+1])-first;
      const auto count=unsigned(host.kv_indptr[r+1]-host.kv_indptr[r]);
      const auto length=paged?(count-1)*page_rows+unsigned(host.last_page_lengths[r]):count;
      const auto physical=[&](unsigned row){return paged?unsigned(host.page_indices[unsigned(host.kv_indptr[r])+row/page_rows])*page_rows+row%page_rows:unsigned(host.kv_indptr[r])+row;};
      for(unsigned t=0;t<nq;++t)for(unsigned h=0;h<heads;++h) {
        const auto visible=length-nq+t+1;std::vector<double> scores(visible);double maximum=-INFINITY;
        for(unsigned row=0;row<visible;++row) {
          double dot=0;for(unsigned lane=0;lane<(omit_kpe?std::min(depth,512u):depth);++lane)
            dot+=double(scalar::wide(query[((first+t)*heads+h)*depth+lane]))*value(physical(row),lane);
          scores[row]=dot*attention_scale;maximum=std::max(maximum,scores[row]);
        }
        double sum=0;for(auto &s:scores){s=std::exp(s-maximum);sum+=s;}
        for(unsigned lane=0;lane<dv;++lane) {
          double output_value=0;for(unsigned row=0;row<visible;++row)output_value+=scores[row]*value(physical(row),lane);
          result[((first+t)*heads+h)*dv+lane]=scalar::bf16(scalar::f32(output_value/sum));
        }
      }
    }
    return result;
  }
};

template<class Compile,class Write,class Read>
void run(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
         Compile compile,Write write,Read read,unsigned depth,bool paged,bool single,
         m::MemoryResourceRef host_workspace=m::system_resource(),bool exact=false,
         bool require_plan_reuse=true) {
  Case shape(depth,paged,single,exact);c::PagedArena arena;
  std::array<c::ReadLease,Case::pages> ready;c::ReadLease ragged;
  if(paged) {
    arena=take(c::PagedArena::create(shape.schema,c::BatchCacheRole::kMla,resource,
        {.max_backing_bytes=shape.bank.size()+1}),"single physical MLA arena");
    EXPECT_EQ(arena.page_bytes(),Case::page_rows*depth);
    EXPECT_FALSE(c::PagedArena::create(shape.schema,c::BatchCacheRole::kValue,resource,{}));
    for(unsigned p=0;p<Case::pages;++p) {
      auto writer=take(arena.acquire(p),"CKV page");const auto bytes=arena.page_bytes();
      check(write(take(writer.lease().view(),"CKV bytes"),std::as_bytes(std::span(shape.bank).subspan(p*bytes,bytes))),"upload original compact bytes");
      ready[p]=take(writer.publish(p?16:15),"initialized compact prefix");
    }
  } else {
    auto b=take(m::Buffer::create(resource,shape.bank.size(),64),"packed CKV bytes");
    check(write(b.view(),std::as_bytes(std::span(shape.bank))),"upload packed CKV");
    ragged=take(c::ReadLease::adopt(std::move(b)),"packed CKV lease");
  }
  auto qb=take(m::Buffer::create(resource,shape.query.size()*2,64),"BF16 Q");
  check(write(qb.view(),std::as_bytes(std::span(shape.query))),"upload Q");
  auto query=take(c::ReadLease::adopt(std::move(qb)),"Q lease");
  auto compiled=take(compile(shape.graph,{}),"compile original compact graph");
  const auto ownership=take(compiled.executable.output_ownership(),"MLA ownership");
  const auto bytes=shape.output_description.required_bytes().value();
  auto pool=take(c::LeasePool::create(resource,bytes*4+1024,{.max_blocks=4}),"independent MLA outputs");
  std::array<g::Invocation,3> slots;
  for(auto &slot:slots)slot=take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"MLA slot");
  std::array<g::GraphOutput,3> observers;std::array<c::ReadLease,3> held;
  std::array<std::vector<std::uint16_t>,3> saved;
  const auto observe=[&](const c::ReadLease &lease) {
    std::vector<std::uint16_t> words(bytes/2);check(read(*lease.view(),std::as_writable_bytes(std::span(words))),"read MLA");return words;
  };
  unsigned submissions=0,reused_plans=0,cold_replans=0,unreported_plans=0;
  for(unsigned epoch=0;epoch<2;++epoch) {
    const auto expected=shape.oracle(epoch),wrong=shape.oracle(epoch,true);
    float witness=0;for(std::size_t i=0;i<expected.size();++i)witness=std::max(witness,std::abs(scalar::wide(expected[i])-scalar::wide(wrong[i])));
    if(!exact)ASSERT_GT(witness,.01F);
    if(depth==576 && !exact) {
      const auto omitted=shape.oracle(epoch,false,true);float kpe_witness=0;
      for(std::size_t i=0;i<expected.size();++i)kpe_witness=std::max(kpe_witness,std::abs(scalar::wide(expected[i])-scalar::wide(omitted[i])));
      ASSERT_GT(kpe_witness,.016F)<<"fixture must detect dropping all64 KPE lanes";
    }
    for(unsigned index=0;index<(single && epoch?3u:2u);++index) {
      auto &slot=slots[index];observers[index].reset();held[index]={};
      const std::array versions{g::batch::NodeLayout{shape.node,shape.layouts[epoch]}};
      const g::batch::PrepareOptions options{.host_workspace=host_workspace};
      check(g::batch::prepare(slot,queues,versions,options),"prepare compact layout");
      if(single && epoch && index<2) {
        const auto common=g::batch::report(slot,shape.node);
        // Delegated ragged families have no public page-plan report. Their
        // warm/fresh and lifetime checks below still execute in full.
        if(common || paged) {
          const bool reused=common?common->reused_native_plan:take(g::batch::native_report(slot,shape.node),"native Q1 report").reused_native_plan;
          if(require_plan_reuse)EXPECT_TRUE(reused);
          reused?++reused_plans:++cold_replans;
        }else ++unreported_plans;
      }
      check(slot.bind(shape.inputs[0],shape.query_description,query),"bind Q");
      if(paged) {
        const std::array refs{ready[exact?2:0],ready[2],ready[1]};
        auto frame=take(arena.read(shape.layouts[epoch],std::span(refs).first(exact?1:single?2:3)),"exact shared page permissions");
        check(g::batch::bind(slot,shape.inputs[1],frame),"bind single compact arena");
      }else check(slot.bind(shape.inputs[1],shape.cache_description,ragged),"bind single packed cache");
      if(ownership==g::OutputOwnership::kCaller)
        check(slot.bind(shape.output,shape.output_description,take(pool.allocate(bytes,64),"MLA output writer")),"bind Y");
      check(slot.prepare_replay(queues),"prepare MLA replay");
      auto done=take(slot.submit(queues),"MLA submit");check(done.wait(),"MLA Completion");++submissions;
      observers[index]=take(slot.result(shape.output),"published MLA");held[index]=take(observers[index].storage_lease(),"retained MLA");
      saved[index]=observe(held[index]);
      if(exact) {
        ASSERT_EQ(expected.front(),0x3bc1u);EXPECT_EQ(saved[index],expected);
      }else for(std::size_t i=0;i<expected.size();++i)
        ASSERT_NEAR(scalar::wide(saved[index][i]),scalar::wide(expected[i]),.008F)<<"epoch="<<epoch<<" slot="<<index<<" lane="<<i;
      EXPECT_FALSE(g::batch::prepare(slot,queues,versions,options).ok());
      if(index)EXPECT_EQ(observe(held[0]),saved[0]);
      if(epoch && index==0 && held[1].valid())EXPECT_EQ(observe(held[1]),saved[1]);
      if(index==2)EXPECT_EQ(saved[2],saved[0]);
    }
  }
  slots={};compiled={};shape.graph={};
  for(unsigned i=0;i<3;++i)if(held[i].valid())EXPECT_EQ(observe(held[i]),saved[i]);
  ::testing::Test::RecordProperty("successful_submissions",int(submissions));
  ::testing::Test::RecordProperty("physical_cache_bytes",int(shape.bank.size()));
  ::testing::Test::RecordProperty("q1_reused_plans",int(reused_plans));
  ::testing::Test::RecordProperty("q1_cold_replans",int(cold_replans));
  ::testing::Test::RecordProperty("q1_unreported_plans",int(unreported_plans));
}
} // namespace compact_mla_e4_test
