#include "apple_encoded_page_reference.h"
#include <cstring>
#include "hyle/compute/detail/materialized_native_controller.h"
#include "hyle/compute/detail/apple_batch_access.h"
#include "../../src/compute_route_apple/apple_file_stream_access.h"
#include "perf_process.h"
#include <nanobench.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
namespace c=hyle::compute;namespace nb=ankerl::nanobench;
namespace n=c::apple;
constexpr const char* backend="Apple";
#ifndef HYLE_ENCODED_PAGE_PLACEMENT
#define HYLE_ENCODED_PAGE_PLACEMENT 0
#endif
void check(hyle::Status status){if(!status.ok()){std::fprintf(stderr,"native page: %.*s\n",int(status.message().size()),status.message().data());std::exit(2);}}
template<class T>T take(hyle::Result<T> value){if(!value)check(value.error());return std::move(*value);}
bool set(const char* key){const auto* value=std::getenv(key);return value && *value;}
std::uint16_t encode(float x){if(x==0)return 0;const auto bits=std::bit_cast<std::uint32_t>(x);return std::uint16_t(((bits>>16)&0x8000U)|(((bits>>23&255U)-112U)<<10)|((bits&0x7fffffU)>>13));}
double decode(std::uint16_t bits){const unsigned e=(bits>>10)&31U,f=bits&1023U;const double sign=(bits&0x8000U)?-1.:1.;return sign*(e?std::ldexp(1.+double(f)/1024.,int(e)-15):std::ldexp(double(f),-24));}
struct Data final {
  c::BatchCapacity cap;std::array<std::int32_t,3> qp,kp;std::array<std::int32_t,2> tails;
  std::vector<std::int32_t> pages;std::vector<std::uint16_t> query;
  std::array<std::vector<std::byte>,2> packed;
  std::array<std::vector<double>,2> decoded;
  float scale;bool causal;int window;
  Data(int q1,int q2,int k1,int k2,int dk,int dv,bool masked,int left):
      cap{.storage=c::BatchStorage::kPaged,.query_type=c::DataType::kFloat16,
          .output_type=c::DataType::kFloat16,.key_storage=take(c::CacheStorage::q8_0()),
          .value_storage=take(c::CacheStorage::q8_0()),.max_requests=2,
          .max_query_rows=unsigned(q1+q2+3),.max_kv_rows=unsigned(k1+k2+13),
          .max_pages=unsigned((k1+15)/16+(k2+15)/16+2),
          .max_page_refs=unsigned((k1+15)/16+(k2+15)/16),.query_heads=8,.kv_heads=2,
          .qk_head_dim=unsigned(dk),.value_head_dim=unsigned(dv),.page_size=16},
      qp{0,q1,q1+q2},kp{0,(k1+15)/16,(k1+15)/16+(k2+15)/16},
      tails{(k1-1)%16+1,(k2-1)%16+1},scale(1.F/std::sqrt(float(dk))),causal(masked),window(left){
    for(unsigned i=0;i<cap.max_page_refs;++i)pages.push_back(int(cap.max_pages-1-i));
    pages[unsigned(kp[1])]=pages[0]; // A real shared readonly physical page.
    query.resize(std::size_t(cap.max_query_rows)*8*dk,encode(128.F));
    for(std::size_t i=0;i<std::size_t(qp[2])*8*dk;++i)
      query[i]=encode(float(int((i*13)%31)-15)*.015625F);
    for(unsigned port=0;port<2;++port){
      const auto depth=port?dv:dk;auto& bytes=packed[port];auto& values=decoded[port];
      values.resize(std::size_t(cap.max_pages)*16*2*depth);bytes.resize(values.size()/32*34);
      for(unsigned page=0;page<cap.max_pages;++page){
        unsigned valid=0;
        for(unsigned r=0;r<2;++r)for(int ref=kp[r];ref<kp[r+1];++ref)
          if(pages[unsigned(ref)]==int(page))valid=std::max(valid,ref+1==kp[r+1]?unsigned(tails[r]):16u);
        for(unsigned row=0;row<16;++row)for(unsigned h=0;h<2;++h)for(int block=0;block<depth/32;++block){
          const auto index=((std::size_t(page)*16+row)*2+h)*depth/32+unsigned(block);
          const auto bits=std::uint16_t(row<valid?0x1800+((index+17*port)%5)*0x400:0x5400);
          std::memcpy(bytes.data()+index*34,&bits,2);
          for(unsigned d=0;d<32;++d){
            const auto code=std::int8_t(int((index*13+d*7+17*port)%31)-15);
            bytes[index*34+2+d]=std::byte(std::uint8_t(code));
            values[index*32+d]=decode(bits)*code;
          }
        }
      }
    }
  }
  std::vector<double> oracle()const{
    const auto dk=cap.qk_head_dim,dv=cap.value_head_dim;
    std::vector<double> result(std::size_t(qp[2])*8*dv),weights;
    for(unsigned r=0;r<2;++r){
      const int nq=qp[r+1]-qp[r],nk=(kp[r+1]-kp[r]-1)*16+tails[r];weights.resize(nk);
      const auto physical=[&](int row){return pages[unsigned(kp[r]+row/16)]*16+row%16;};
      for(int row=0;row<nq;++row)for(unsigned h=0;h<8;++h){
        double sum=0;const int position=nk-nq+row;
        for(int j=0;j<nk;++j){
          if(causal&&(j>position||(window>=0&&j<std::max(0,position-window)))){weights[j]=0;continue;}
          double dot=0;for(unsigned d=0;d<dk;++d)
            dot+=decode(query[(std::size_t(qp[r]+row)*8+h)*dk+d])*decoded[0][(std::size_t(physical(j))*2+h/4)*dk+d];
          weights[j]=std::exp(dot*scale);sum+=weights[j];
        }
        for(unsigned d=0;d<dv;++d)for(int j=0;j<nk;++j)
          result[(std::size_t(qp[r]+row)*8+h)*dv+d]+=weights[j]/sum*decoded[1][(std::size_t(physical(j))*2+h/4)*dv+d];
      }
    }
    return result;
  }
};
// Only the benchmark's peer cold recipe differs. Both arms use the existing
// fixed table banks/F2 upload/real F1 frame guards/native slot retirement.
struct PeerTraits final {
  using Context=hyle::bench::resident::AppleEncodedPeerContext;using Program=n::Program;using Memory=n::MemoryResource;
  using Invocation=n::Invocation;using CopyInvocation=n::CopyBatchInvocation;
  using CopyAccess=n::detail::CopyStreamAccess;
  using View=mlx::core::array;
  using Submission=c::OwnedResult<n::Results>;
  static auto invocation(const Program& p,hyle::mem::AllocatorRef a){return p.create_invocation(a);}
  static auto copy_program(const Context& x,std::span<const c::CopyResourceSpec> s,c::CopyBatchOptions o){return n::CopyBatchProgram::prepare(*x.context,s,o);}
  static auto copy_invocation(const n::CopyBatchProgram& p){return n::CopyBatchInvocation::create(p);}
  static auto binding(const Memory& m,c::ReadLease r,c::TensorDesc d,c::LeaseImportOptions o){return m.binding(std::move(r),d,o);}
  static auto paged_binding(const Memory& m,c::PagedArenaRead r,c::LeaseImportOptions o){return n::detail::PagedBatchAccess::binding(m,std::move(r),o);}
  static auto submit(Invocation& i,c::LeaseUseBatch u){return n::detail::PagedBatchAccess::submit(i,std::move(u));}
  static auto wait_native(Invocation& i){return n::detail::PagedBatchAccess::wait_native(i);}
  static auto reuse_gate(Invocation& i){return n::detail::PagedBatchAccess::reuse_gate(i);}
  static auto release_inputs(Invocation& i){return n::detail::PagedBatchAccess::release_inputs(i);}
  static auto update_layout(Invocation& i,const c::BatchLayoutVersion& l){return n::detail::PagedBatchAccess::update_layout(i,l);}
  static auto prepare_program(const Context& x,const c::NativePagedBatchProgram& p,c::BatchLayoutVersion l,const c::WorkloadContext& w){return hyle::bench::resident::encoded_page_reference_program(x,p,std::move(l),w);}
};
using PeerState=c::detail::MaterializedNativeState<PeerTraits,c::detail::PagedMetadataPolicy>;
struct Harness final {
  n::Context context;std::array<hyle::bench::resident::AppleEncodedPeerContext,2> peer_contexts;n::MemoryResource device,staging;c::BatchSchema schema;c::BatchLayoutVersion layout;
  c::NativePagedBatchProgram recipe;c::PagedArena keys,values;c::PagedArenaRead key_frame,value_frame;
  c::LeasePool query_pool;c::ReadLease query;std::vector<c::ReadLease> key_reads,value_reads;
  std::array<n::PagedBatchInvocation,2> selected;std::array<std::unique_ptr<PeerState>,2> peers;
  bool original;unsigned next=0;
  Harness(const Data& data,bool use_original):context(take(n::Context::create())),
      device(take(n::MemoryResource::create(context))),staging(take(n::MemoryResource::create(context))),
      original(use_original){
    const auto env=take(context.environment());
    std::fprintf(stderr,"# ENCODED_PAGE_ENV backend=Apple device=%s architecture=%s runtime=%s revision=%s\n",env.device_name.c_str(),env.architecture.c_str(),env.os_runtime.c_str(),env.ggml_revision.c_str());
    schema=take(c::BatchSchema::create(data.cap));layout=take(schema.version({data.qp,data.kp,data.pages,data.tails}));
    recipe=take(c::NativePagedBatchProgram::prepare(schema,{.max_host_metadata_bytes=1<<20,.scale=data.scale,.causal=data.causal,.window_left=data.window}));
    keys=take(c::PagedArena::create(schema,c::BatchCacheRole::kKey,device.ref(),{.max_backing_bytes=1<<24}));values=take(c::PagedArena::create(schema,c::BatchCacheRole::kValue,device.ref(),{.max_backing_bytes=1<<24}));
    const auto page_bytes=std::max(keys.page_bytes(),values.page_bytes()),query_bytes=data.query.size()*2;
    query_pool=take(c::LeasePool::create(device.ref(),query_bytes+512));auto host=take(c::LeasePool::create(staging.ref(),2*std::max(page_bytes,query_bytes)+512));
    const auto upload=[&](c::WriteLease& writer,std::span<const std::byte> bytes){
      auto sw=take(host.allocate(bytes.size(),64));check(sw.view()->write(bytes));auto source=take(sw.publish());
      const std::array specs{c::CopyResourceSpec{staging.ref(),bytes.size()},c::CopyResourceSpec{device.ref(),bytes.size()}};
      auto cp=take(n::CopyBatchProgram::prepare(context,specs));auto copy=take(n::CopyBatchInvocation::create(cp));
      check(copy.bind_source({0},source));check(copy.bind_mutable({1},std::move(writer)));
      const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,bytes.size()}}};
      check(copy.prepare_replay(phases,ranges));auto done=take(copy.submit());check(done.wait());return take(done.output({0},{1},0,bytes.size()));
    };
    auto qw=take(query_pool.allocate(query_bytes,64));query=upload(qw,std::as_bytes(std::span(data.query)));
    for(unsigned p=0;p<data.cap.max_pages;++p){
      auto k=take(keys.acquire(p));key_reads.push_back(upload(k.lease(),std::span(data.packed[0]).subspan(p*keys.page_bytes(),keys.page_bytes())));check(k.complete(key_reads.back(),16));
      auto v=take(values.acquire(p));value_reads.push_back(upload(v.lease(),std::span(data.packed[1]).subspan(p*values.page_bytes(),values.page_bytes())));check(v.complete(value_reads.back(),16));
    }
    auto ids=data.pages;std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
    std::vector<c::ReadLease> exact_keys,exact_values;
    for(const auto id:ids){exact_keys.push_back(key_reads[unsigned(id)]);exact_values.push_back(value_reads[unsigned(id)]);}
    key_frame=take(keys.read(layout,exact_keys));value_frame=take(values.read(layout,exact_values));
    const c::NativePagedInvocationOptions options{.max_metadata_backing_bytes=1<<20};
    for(unsigned i=0;i<2;++i){if(original){peer_contexts[i]={&context,std::make_shared<hyle::bench::resident::OriginalMetalCompletion>()};peers[i]=std::make_unique<PeerState>(hyle::mem::system_allocator());check(peers[i]->initialize(peer_contexts[i],n::Program{},recipe,device,staging,options));}
      else selected[i]=take(n::PagedBatchInvocation::create(context,recipe,device,staging,options));}
    std::fprintf(stderr,"# ENCODED_PAGE_COLD backend=%s peer=%d page16 refs=%u physical=%u table_bytes=%zu private_pool_upper=UNKNOWN\n",backend,original,layout.facts().page_refs,data.cap.max_pages,recipe.costs().index_bytes);
  }
  ~Harness(){for(auto& p:peers)if(p){p->drain().IgnoreError();p.reset();}for(auto& p:selected){p.drain().IgnoreError();p={};}key_frame={};value_frame={};key_reads.clear();value_reads.clear();query={};check(context.close());}
  void execute(std::vector<std::uint16_t>* host=nullptr){
    const auto slot=next++%2;
    // Both arms charge checked prepare/version, idle input release and any F2
    // metadata delta. This fixed layout reuses its prepared native slot.
    if(original){check(peers[slot]->prepare(layout));check(peers[slot]->bind(query,key_frame,value_frame));}
    else{check(selected[slot].prepare_batch(layout));check(selected[slot].bind(query,key_frame,value_frame));}
    auto submitted=original?take(peers[slot]->submit()):take(selected[slot].submit());auto result=take(submitted.wait());
    if(original)check(peer_contexts[slot].completion->check_completed());
    const auto output=result->arrays()[0];if(host){auto description=take(c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{layout.facts().query_rows,8,schema.capacity().value_head_dim}));auto read=take(device.import_readonly(output,description));check(read.view()->read(std::as_writable_bytes(std::span(*host))));}
    nb::doNotOptimizeAway(output);
  }
  void verify(const Data& data,const std::vector<double>& expected){std::vector<std::uint16_t> actual(expected.size());double worst=0;
    for(unsigned replay=0;replay<4;++replay){
      execute(&actual);
      for(std::size_t i=0;i<actual.size();++i){if((actual[i]&0x7c00)==0x7c00)check(hyle::Status::Internal("page P1 nonfinite output"));worst=std::max(worst,std::abs(decode(actual[i])-expected[i]));}
    }
    if(worst>2e-3)check(hyle::Status::Internal("page P1 independent double oracle differs"));
    for(unsigned port=0;port<2;++port){
      const auto bytes=port?values.page_bytes():keys.page_bytes();
      auto host=take(c::LeasePool::create(staging.ref(),2*bytes+512));
      const std::array specs{c::CopyResourceSpec{device.ref(),bytes},c::CopyResourceSpec{staging.ref(),bytes}};
      auto cp=take(n::CopyBatchProgram::prepare(context,specs));auto copy=take(n::CopyBatchInvocation::create(cp));
      const auto& reads=port?value_reads:key_reads;
      for(unsigned page=0;page<reads.size();++page){
        check(copy.reset());auto writer=take(host.allocate(bytes,64));
        check(copy.bind_source({0},reads[page]));check(copy.bind_mutable({1},std::move(writer)));
        const std::array<c::CopyPhase,1> phases{{{}}};
        const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,bytes}}};
        check(copy.prepare_replay(phases,ranges));auto done=take(copy.submit());check(done.wait());
        auto read=take(done.output({0},{1},0,bytes));std::vector<std::byte> after(bytes);
        check(read.view()->read(after));
        if(!std::equal(after.begin(),after.end(),data.packed[port].begin()+page*bytes))
          check(hyle::Status::Internal("encoded page P1 readonly physical KV changed"));
      }
    }
    std::fprintf(stderr,"# ENCODED_PAGE_CORRECTNESS peer=%d replays=4 slots=2 readonly=PASS max_abs_error=%.9g elements=%zu\n",int(original),worst,expected.size());
  }
};
void measure(int q1,int q2,int k1,int k2,int depth,int dv,bool causal,int window){Data data(q1,q2,k1,k2,depth,dv,causal,window);
  const auto* p=std::getenv("HYLE_BENCH_AA_ROUTE");const std::string aa=p?p:"";
  if(!aa.empty() && aa!="native" && aa!="selected")check(hyle::Status::InvalidArgument("page AA route"));
  std::unique_ptr<Harness> candidate,peer;
  if(set("HYLE_BENCH_PEER_ALLOCATION_FIRST")){peer=std::make_unique<Harness>(data,aa!="selected");candidate=std::make_unique<Harness>(data,aa=="native");}
  else{candidate=std::make_unique<Harness>(data,aa=="native");peer=std::make_unique<Harness>(data,aa!="selected");}
  const auto expected=data.oracle();candidate->verify(data,expected);peer->verify(data,expected);
  for(unsigned i=0;i<4;++i){candidate->execute();peer->execute();}if(set("HYLE_BENCH_VERIFY_ONLY"))return;
  nb::Bench bench;bench.title(std::string(backend)+" page16 F16/Q8 Q="+std::to_string(q1)+","+std::to_string(q2)+" KV="+std::to_string(k1)+","+std::to_string(k2)+" D="+std::to_string(depth)+" Dv="+std::to_string(dv)+" causal="+std::to_string(causal)+" window="+std::to_string(window)).batch(1).epochs(16).minEpochIterations(1).minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
  const auto a=[&]{candidate->execute();};const auto b=[&]{peer->execute();};const std::string cn=std::string("HYLE ")+backend+" encoded page16",pn=std::string("Original ")+backend+" Q8 gathered page lifecycle";
  if(set("HYLE_BENCH_RIVAL_FIRST"))(void)bench.compare(pn,b,cn,a);else(void)bench.compare(cn,a,pn,b);
}
}
int main(int,char **){const auto qos=hyle::bench::perf::configure_thread_qos();
  const auto* aa=std::getenv("HYLE_BENCH_AA_ROUTE");
  std::fprintf(stderr,"# HYLE_ENCODED_PAGE_RUN run_nonce=%s apple_qos=%s qos_ok=%s backend=%s verify_only=%d paired_rounds=16 placement=%d execution_order=%s allocation_order=%s aa_route=%s\n",hyle::bench::perf::run_nonce(),qos.name,qos.ok,backend,set("HYLE_BENCH_VERIFY_ONLY"),HYLE_ENCODED_PAGE_PLACEMENT,set("HYLE_BENCH_RIVAL_FIRST")?"peer-first":"candidate-first",set("HYLE_BENCH_PEER_ALLOCATION_FIRST")?"peer-first":"candidate-first",aa&&*aa?aa:"none");
  measure(1,17,1024,768,256,256,true,16);measure(64,128,1024,768,256,256,true,-1);measure(17,1,7,47,64,128,false,-1);
}
