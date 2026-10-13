#include "native_page_reference.h"
#include "hyle/compute/detail/materialized_native_controller.h"
#if defined(HYLE_PAGE_APPLE_BENCH)
#include "hyle/compute/detail/apple_batch_access.h"
#else
#include "hyle/compute/detail/vulkan_batch_access.h"
#endif
#include "../common/perf_process.h"
#include <nanobench.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
namespace c=hyle::compute;namespace nb=ankerl::nanobench;
#if defined(HYLE_PAGE_APPLE_BENCH)
namespace n=c::apple;
constexpr const char* backend="Apple";
#else
namespace n=c::vulkan;
constexpr const char* backend="Vulkan";
#endif
#ifndef HYLE_PAGE_PLACEMENT
#define HYLE_PAGE_PLACEMENT 0
#endif
void check(hyle::Status status){if(!status.ok()){std::fprintf(stderr,"native page: %.*s\n",int(status.message().size()),status.message().data());std::exit(2);}}
template<class T>T take(hyle::Result<T> value){if(!value)check(value.error());return std::move(*value);}
bool set(const char* key){const auto* value=std::getenv(key);return value && *value;}
std::uint16_t encode(float x){if(x==0)return 0;const auto bits=std::bit_cast<std::uint32_t>(x);return std::uint16_t(((bits>>16)&0x8000U)|(((bits>>23&255U)-112U)<<10)|((bits&0x7fffffU)>>13));}
double decode(std::uint16_t bits){const unsigned e=(bits>>10)&31U,f=bits&1023U;const double sign=(bits&0x8000U)?-1.:1.;return sign*(e?std::ldexp(1.+double(f)/1024.,int(e)-15):std::ldexp(double(f),-24));}
struct Data final {
  c::BatchCapacity cap;std::array<std::int32_t,3> qp,kp;std::array<std::int32_t,2> tails;
  std::vector<std::int32_t> pages;std::array<std::vector<std::uint16_t>,3> values;float scale;bool causal;
  Data(int q1,int q2,int k1,int k2,int depth,bool masked):
      cap{.storage=c::BatchStorage::kPaged,.max_requests=2,.max_query_rows=unsigned(q1+q2+3),
          .max_kv_rows=unsigned(k1+k2+13),.max_pages=unsigned((k1+15)/16+(k2+15)/16+4),
          .max_page_refs=unsigned((k1+15)/16+(k2+15)/16+3),.query_heads=8,.kv_heads=2,.head_dim=unsigned(depth),.page_size=16},
      qp{0,q1,q1+q2},kp{0,(k1+15)/16,(k1+15)/16+(k2+15)/16},
      tails{(k1-1)%16+1,(k2-1)%16+1},scale(1.F/std::sqrt(float(depth))),causal(masked){
    pages.resize(kp[2]);for(unsigned i=0;i<pages.size();++i)pages[i]=int(pages.size()-1-i);
    values[0].resize(std::size_t(cap.max_query_rows)*8*depth);
    for(unsigned p=1;p<3;++p)values[p].resize(std::size_t(cap.max_pages)*16*2*depth);
    for(unsigned p=0;p<3;++p)for(std::size_t i=0;i<values[p].size();++i)
      values[p][i]=encode(float(int((i*13+p*17)%31)-15)*.015625F);
    for(unsigned p=1;p<3;++p)std::fill(values[p].begin()+std::size_t(kp[2])*16*2*depth,values[p].end(),encode(128.F));
  }
  std::vector<double> oracle()const{
    const auto depth=cap.head_dim;std::vector<double> result(std::size_t(qp[2])*8*depth),weights;
    for(unsigned r=0;r<2;++r){const int nq=qp[r+1]-qp[r],nk=(kp[r+1]-kp[r]-1)*16+tails[r];weights.resize(nk);
      const auto physical=[&](int row){return pages[kp[r]+row/16]*16+row%16;};
      for(int row=0;row<nq;++row)for(int h=0;h<8;++h){double sum=0;
        for(int j=0;j<nk;++j){double dot=0;for(unsigned d=0;d<depth;++d)dot+=decode(values[0][((qp[r]+row)*8+h)*depth+d])*decode(values[1][(physical(j)*2+h/4)*depth+d]);
          weights[j]=causal && j>nk-nq+row?0.:std::exp(dot*scale);sum+=weights[j];}
        for(unsigned d=0;d<depth;++d)for(int j=0;j<nk;++j)result[((qp[r]+row)*8+h)*depth+d]+=weights[j]/sum*decode(values[2][(physical(j)*2+h/4)*depth+d]);
      }
    }return result;
  }
};
// Only the benchmark's peer cold recipe differs. Both arms use the existing
// fixed table banks/F2 upload/real F1 frame guards/native slot retirement.
struct PeerTraits final {
  using Context=n::Context;using Program=n::Program;using Memory=n::MemoryResource;
  using Invocation=n::Invocation;using CopyInvocation=n::CopyBatchInvocation;
#if defined(HYLE_PAGE_APPLE_BENCH)
  using View=mlx::core::array;
#else
  using View=n::Tensor;
#endif
  using Submission=c::OwnedResult<n::Results>;
  static auto invocation(const Program& p,hyle::mem::AllocatorRef a){return p.create_invocation(a);}
  static auto copy_program(const Context& x,std::span<const c::CopyResourceSpec> s,c::CopyBatchOptions o){return n::CopyBatchProgram::prepare(x,s,o);}
  static auto copy_invocation(const n::CopyBatchProgram& p){return n::CopyBatchInvocation::create(p);}
  static auto binding(const Memory& m,c::ReadLease r,c::TensorDesc d,c::LeaseImportOptions o){return m.binding(std::move(r),d,o);}
  static auto paged_binding(const Memory& m,c::PagedArenaRead r,c::LeaseImportOptions o){return n::detail::PagedBatchAccess::binding(m,std::move(r),o);}
  static auto submit(Invocation& i,c::LeaseUseBatch u){return n::detail::PagedBatchAccess::submit(i,std::move(u));}
  static auto wait_native(Invocation& i){return n::detail::PagedBatchAccess::wait_native(i);}
  static auto reuse_gate(Invocation& i){return n::detail::PagedBatchAccess::reuse_gate(i);}
  static auto prepare_program(const Context& x,const c::NativePagedBatchProgram& p,c::BatchLayoutVersion l,const c::WorkloadContext& w){return hyle::bench::resident::page_reference_program(x,p,std::move(l),w);}
};
using PeerState=c::detail::MaterializedNativeState<PeerTraits,c::detail::PagedMetadataPolicy>;
struct Harness final {
  n::Context context;n::MemoryResource device,staging;c::BatchSchema schema;c::BatchLayoutVersion layout;
  c::NativePagedBatchProgram recipe;c::PagedArena keys,values;c::PagedArenaRead key_frame,value_frame;
  c::LeasePool query_pool;c::ReadLease query;std::vector<c::ReadLease> key_reads,value_reads;
  std::array<n::PagedBatchInvocation,2> selected;std::array<std::unique_ptr<PeerState>,2> peers;
  bool original;unsigned next=0;
  Harness(const Data& data,bool use_original):context(take(n::Context::create())),
#if defined(HYLE_PAGE_APPLE_BENCH)
      device(take(n::MemoryResource::create(context))),staging(take(n::MemoryResource::create(context))),
#else
      device(take(n::MemoryResource::create(context,n::MemoryKind::kDevice))),staging(take(n::MemoryResource::create(context,n::MemoryKind::kStaging))),
#endif
      original(use_original){
#if defined(HYLE_PAGE_APPLE_BENCH)
    const auto env=take(context.environment());std::fprintf(stderr,"# NATIVE_PAGE_ENV backend=Apple device=%s architecture=%s runtime=%s revision=%s\n",env.device_name.c_str(),env.architecture.c_str(),env.os_runtime.c_str(),env.mlx_revision.c_str());
#else
    const auto& env=context.environment();if(env.software_device)check(hyle::Status::Unimplemented("page P1 requires real GPU"));
    std::fprintf(stderr,"# NATIVE_PAGE_ENV backend=Vulkan device=%s driver=%s revision=%s\n",env.device_name.c_str(),env.driver_name.c_str(),env.ggml_revision.c_str());
#endif
    schema=take(c::BatchSchema::create(data.cap));layout=take(schema.version({data.qp,data.kp,data.pages,data.tails}));
    recipe=take(c::NativePagedBatchProgram::prepare(schema,{.max_host_metadata_bytes=1<<20,.scale=data.scale,.causal=data.causal}));
    keys=take(c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1<<24}));values=take(c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1<<24}));
    const auto page_bytes=keys.page_bytes(),query_bytes=data.values[0].size()*2;
    query_pool=take(c::LeasePool::create(device.ref(),query_bytes+512));auto host=take(c::LeasePool::create(staging.ref(),2*std::max(page_bytes,query_bytes)+512));
    const auto upload=[&](c::WriteLease& writer,std::span<const std::uint16_t> words){
      auto sw=take(host.allocate(words.size_bytes(),64));check(sw.view()->write(std::as_bytes(words)));auto source=take(sw.publish());
      const std::array specs{c::CopyResourceSpec{staging.ref(),words.size_bytes()},c::CopyResourceSpec{device.ref(),words.size_bytes()}};
      auto cp=take(n::CopyBatchProgram::prepare(context,specs));auto copy=take(n::CopyBatchInvocation::create(cp));
      check(copy.bind_source({0},source));check(copy.bind_mutable({1},std::move(writer)));
      const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,words.size_bytes()}}};
      check(copy.prepare_replay(phases,ranges));auto done=take(copy.submit());check(done.wait());return take(done.output({0},{1},0,words.size_bytes()));
    };
    auto qw=take(query_pool.allocate(query_bytes,64));query=upload(qw,data.values[0]);
    for(unsigned p=0;p<unsigned(data.kp[2]);++p){
      auto k=take(keys.acquire(p));key_reads.push_back(upload(k.lease(),std::span(data.values[1]).subspan(p*page_bytes/2,page_bytes/2)));check(k.complete(key_reads.back()));
      auto v=take(values.acquire(p));value_reads.push_back(upload(v.lease(),std::span(data.values[2]).subspan(p*page_bytes/2,page_bytes/2)));check(v.complete(value_reads.back()));
    }
    key_frame=take(keys.read(layout,key_reads));value_frame=take(values.read(layout,value_reads));
    const c::NativePagedInvocationOptions options{.max_metadata_backing_bytes=1<<20};
    for(unsigned i=0;i<2;++i){if(original){peers[i]=std::make_unique<PeerState>(hyle::mem::system_allocator());check(peers[i]->initialize(context,n::Program{},recipe,device,staging,options));}
      else selected[i]=take(n::PagedBatchInvocation::create(context,recipe,device,staging,options));}
    std::fprintf(stderr,"# NATIVE_PAGE_COLD backend=%s peer=%d page16 refs=%u physical=%u table_bytes=%zu private_pool_upper=UNKNOWN\n",backend,original,layout.facts().page_refs,data.cap.max_pages,recipe.costs().index_bytes);
  }
  ~Harness(){for(auto& p:peers)if(p){p->drain().IgnoreError();p.reset();}for(auto& p:selected){p.drain().IgnoreError();p={};}key_frame={};value_frame={};key_reads.clear();value_reads.clear();query={};check(context.close());}
  void execute(std::vector<std::uint16_t>* host=nullptr){
    const auto slot=next++%2;
    // Explicit prepare/version/F2 upload/cold original Program and native slot
    // replacement are charged in both arms; no hidden kernel-only shortcut.
    if(original){check(peers[slot]->prepare(layout));check(peers[slot]->bind(query,key_frame,value_frame));}
    else{check(selected[slot].prepare_batch(layout));check(selected[slot].bind(query,key_frame,value_frame));}
    auto submitted=original?take(peers[slot]->submit()):take(selected[slot].submit());auto result=take(submitted.wait());
#if defined(HYLE_PAGE_APPLE_BENCH)
    const auto output=result->arrays()[0];if(host){auto desc=take(c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{layout.facts().query_rows,8,schema.capacity().head_dim}));auto read=take(device.import_readonly(output,desc));check(read.view()->read(std::as_writable_bytes(std::span(*host))));}
#else
    const auto output=result->tensors()[0];if(host)check(context.download(output,std::as_writable_bytes(std::span(*host))));
#endif
    nb::doNotOptimizeAway(output);
  }
  void verify(const Data& data,const std::vector<double>& expected){std::vector<std::uint16_t> actual(expected.size());execute(&actual);double worst=0;
    for(std::size_t i=0;i<actual.size();++i){if((actual[i]&0x7c00)==0x7c00)check(hyle::Status::Internal("page P1 nonfinite output"));worst=std::max(worst,std::abs(decode(actual[i])-expected[i]));}
    if(worst>2e-3)check(hyle::Status::Internal("page P1 independent double oracle differs"));
    const auto bytes=keys.page_bytes();auto host=take(c::LeasePool::create(staging.ref(),2*bytes+512));
    const std::array specs{c::CopyResourceSpec{device.ref(),bytes},c::CopyResourceSpec{staging.ref(),bytes}};
    auto cp=take(n::CopyBatchProgram::prepare(context,specs));auto copy=take(n::CopyBatchInvocation::create(cp));
    for(unsigned port=1;port<3;++port){const auto& reads=port==1?key_reads:value_reads;
      for(unsigned page=0;page<reads.size();++page){check(copy.reset());auto writer=take(host.allocate(bytes,64));
        check(copy.bind_source({0},reads[page]));check(copy.bind_mutable({1},std::move(writer)));
        const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,bytes}}};
        check(copy.prepare_replay(phases,ranges));auto done=take(copy.submit());check(done.wait());auto read=take(done.output({0},{1},0,bytes));
        std::vector<std::uint16_t> after(bytes/2);check(read.view()->read(std::as_writable_bytes(std::span(after))));
        if(!std::equal(after.begin(),after.end(),data.values[port].begin()+page*bytes/2))check(hyle::Status::Internal("page P1 readonly physical KV changed"));
      }
    }
    std::fprintf(stderr,"# NATIVE_PAGE_CORRECTNESS max_abs_error=%.9g elements=%zu\n",worst,expected.size());
  }
};
void measure(int q1,int q2,int k1,int k2,int depth,bool causal){Data data(q1,q2,k1,k2,depth,causal);
  const auto* p=std::getenv("HYLE_BENCH_AA_ROUTE");const std::string aa=p?p:"";
  if(!aa.empty() && aa!="native" && aa!="selected")check(hyle::Status::InvalidArgument("page AA route"));
  std::unique_ptr<Harness> candidate,peer;
  if(set("HYLE_BENCH_PEER_ALLOCATION_FIRST")){peer=std::make_unique<Harness>(data,aa!="selected");candidate=std::make_unique<Harness>(data,aa=="native");}
  else{candidate=std::make_unique<Harness>(data,aa=="native");peer=std::make_unique<Harness>(data,aa!="selected");}
  const auto expected=data.oracle();candidate->verify(data,expected);peer->verify(data,expected);
  for(unsigned i=0;i<4;++i){candidate->execute();peer->execute();}if(set("HYLE_BENCH_VERIFY_ONLY"))return;
  nb::Bench bench;bench.title(std::string(backend)+" page16 F16 Q="+std::to_string(q1)+","+std::to_string(q2)+" KV="+std::to_string(k1)+","+std::to_string(k2)+" D="+std::to_string(depth)+" causal="+std::to_string(causal)).batch(1).epochs(16).minEpochIterations(1).minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
  const auto a=[&]{candidate->execute();};const auto b=[&]{peer->execute();};const std::string cn=std::string("HYLE ")+backend+" page16",pn=std::string("Original ")+backend+" gathered page lifecycle";
  if(set("HYLE_BENCH_RIVAL_FIRST"))(void)bench.compare(pn,b,cn,a);else(void)bench.compare(cn,a,pn,b);
}
}
int main(int,char** argv){const auto qos=hyle::bench::perf::configure_thread_qos();
  if(!set("HYLE_BENCH_VERIFY_ONLY") && !set("HYLE_BENCH_AA_ROUTE") && set("HYLE_PERF_RUN_NONCE")){
    std::string ready;std::ifstream(std::filesystem::absolute(argv[0]).parent_path()/"g01_native_page_aa.ready")>>ready;
    if(ready!="PASS")check(hyle::Status::FailedPrecondition("page AA has not qualified"));}
  const auto* aa=std::getenv("HYLE_BENCH_AA_ROUTE");
  std::fprintf(stderr,"# HYLE_NATIVE_PAGE_RUN run_nonce=%s apple_qos=%s qos_ok=%s backend=%s verify_only=%d paired_rounds=16 placement=%d execution_order=%s allocation_order=%s aa_route=%s\n",hyle::bench::perf::run_nonce(),qos.name,qos.ok,backend,set("HYLE_BENCH_VERIFY_ONLY"),HYLE_PAGE_PLACEMENT,set("HYLE_BENCH_RIVAL_FIRST")?"peer-first":"candidate-first",set("HYLE_BENCH_PEER_ALLOCATION_FIRST")?"peer-first":"candidate-first",aa&&*aa?aa:"none");
  measure(1,17,1024,768,256,true);measure(64,128,1024,768,256,true);measure(1,17,31,47,64,false);
}
