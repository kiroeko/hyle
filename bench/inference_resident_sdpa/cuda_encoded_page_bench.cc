#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "cuda_encoded_page_reference.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/flashinfer.h"
#include "../common/perf_process.h"
#include "hyle/compute/batch_execution.h"
#include "hyle/simd/dtype.h"
#include <cstring>
#include <nanobench.h>
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
#ifndef HYLE_RESIDENT_PLACEMENT
#define HYLE_RESIDENT_PLACEMENT 0
#endif
namespace c=hyle::compute;namespace nb=ankerl::nanobench;
[[noreturn]] void fail(hyle::Status status){std::fprintf(stderr,"CUDA encoded: %.*s\n",int(status.message().size()),status.message().data());std::exit(2);}
void check(hyle::Status status){if(!status.ok())fail(status);}
template<class T>T take(hyle::Result<T> value){if(!value)fail(value.error());return std::move(*value);}
bool set(const char *name){const auto *p=std::getenv(name);return p&&*p;}
float decode(std::uint16_t x){return std::bit_cast<float>(std::uint32_t(x)<<16);}
std::uint16_t encode(float x){return std::uint16_t(std::bit_cast<std::uint32_t>(x)>>16);}
double fp8(std::uint8_t code){const unsigned e=(code&127)>>3,m=code&7;
  const double value=e?std::ldexp(1.+double(m)/8.,int(e)-7):std::ldexp(double(m)/8.,-6);
  return code&128?-value:value;
}
struct Data {
  c::BatchCapacity capacity;
  std::array<std::int32_t,3> qp,kp;
  std::vector<std::int32_t> ids;std::array<std::int32_t,2> tails;
  std::vector<std::uint16_t> query;
  std::array<std::vector<std::byte>,2> packed;
  std::array<std::vector<float>,2> decoded;
  float scale;bool causal;std::int32_t window;
  Data(int q1,int q2,int k1,int k2,int dk,int dv,bool masked,int left):
    capacity{.storage=c::BatchStorage::kPaged,.query_type=c::DataType::kBFloat16,
      .output_type=c::DataType::kBFloat16,.key_storage=c::CacheStorage::float8(.5F).value(),
      .value_storage=c::CacheStorage::float8(2.F).value(),.max_requests=2,
      .max_query_rows=unsigned(q1+q2+3),.max_kv_rows=unsigned(k1+k2+13),
      .max_pages=unsigned((k1+15)/16+(k2+15)/16+2),.max_page_refs=unsigned((k1+15)/16+(k2+15)/16),
      .query_heads=8,.kv_heads=2,.qk_head_dim=unsigned(dk),.value_head_dim=unsigned(dv),.page_size=16},
    qp{0,q1,q1+q2},kp{0,(k1+15)/16,(k1+15)/16+(k2+15)/16},
    scale(1.F/std::sqrt(float(dk))),causal(masked),window(left){
    tails={(k1-1)%16+1,(k2-1)%16+1};
    for(unsigned i=0;i<capacity.max_page_refs;++i)ids.push_back(static_cast<std::int32_t>(capacity.max_pages-1-i));
    ids[static_cast<unsigned>(kp[1])]=ids[0];
    query.resize(std::size_t(capacity.max_query_rows)*8*dk);
    for(std::size_t i=0;i<query.size();++i)query[i]=encode(.125F+float((i/dk)%8%3)*.015625F);
    for(unsigned port=0;port<2;++port){
      const auto depth=port?dv:dk;
      auto &bytes=packed[port];auto &values=decoded[port];
      values.resize(std::size_t(capacity.max_pages)*16*2*depth);
      bytes.resize(values.size());
      // Independent finite FP8 fixture; the oracle decodes these actual bytes.
      for(std::size_t i=0;i<bytes.size();++i){
        const auto row=i/(2*depth),head=(i/depth)%2;
        constexpr std::uint8_t key_codes[]{0,56,64,68,72},value_codes[]{16,24,28,32,34};
        const auto code=(port?value_codes:key_codes)[(row+head)%5];
        bytes[i]=std::byte(code);values[i]=float(fp8(code)*(port?2.:.5));
      }
      for(unsigned page=0;page<capacity.max_pages;++page){
        unsigned valid=0;
        for(unsigned r=0;r<2;++r)for(auto ref=kp[r];ref<kp[r+1];++ref)
          if(ids[unsigned(ref)]==int(page))valid=std::max(valid,ref+1==kp[r+1]?unsigned(tails[r]):16u);
        for(auto i=std::size_t(page*16+valid)*2*depth;i<std::size_t((page+1)*16)*2*depth;++i){
          bytes[i]=std::byte(0x78);values[i]=float(fp8(0x78)*(port?2.:.5));
        }
      }
    }
  }
  std::size_t row_bytes(unsigned port)const{return std::size_t(2)*(port?capacity.value_head_dim:capacity.qk_head_dim);}
  std::vector<double> oracle()const{
    const auto dk=capacity.qk_head_dim,dv=capacity.value_head_dim;
    std::vector<double> output(std::size_t(qp[2])*8*dv),weights;double uniform_difference=0;
    for(unsigned r=0;r<2;++r){const int nq=qp[r+1]-qp[r],nk=(kp[r+1]-kp[r]-1)*16+tails[r];weights.resize(nk);
      for(int row=0;row<nq;++row)for(unsigned h=0;h<8;++h){double sum=0;
        const auto position=nk-nq+row;
        for(int j=0;j<nk;++j){
          if(causal&&(j>position||(window>=0&&j<std::max(0,position-window)))){weights[j]=0;continue;}
          double dot=0;for(unsigned d=0;d<dk;++d)dot+=double(decode(query[(std::size_t(qp[r]+row)*8+h)*dk+d]))*
              decoded[0][((std::size_t(ids[static_cast<unsigned>(kp[r]+j/16)])*16+j%16)*2+h/4)*dk+d];
          weights[j]=std::exp(dot*scale);sum+=weights[j];
        }
        for(unsigned d=0;d<dv;++d)for(int j=0;j<nk;++j)output[(std::size_t(qp[r]+row)*8+h)*dv+d]+=
            weights[j]/sum*decoded[1][((std::size_t(ids[static_cast<unsigned>(kp[r]+j/16)])*16+j%16)*2+h/4)*dv+d];
        double uniform=0;unsigned eligible=0;
        for(int j=0;j<nk;++j)if(weights[j]>0){++eligible;
          uniform+=decoded[1][((std::size_t(ids[unsigned(kp[r]+j/16)])*16+j%16)*2+h/4)*dv];}
        uniform_difference=std::max(uniform_difference,std::abs(
            output[(std::size_t(qp[r]+row)*8+h)*dv]-uniform/eligible));
      }
    }
    if(uniform_difference<=.01)fail(hyle::Status::Internal("FP8 oracle fixture cannot distinguish missing QK loads"));
    std::fprintf(stderr,"# CUDA_ENCODED_ORACLE uniform_difference=%.9g\n",uniform_difference);
    return output;
  }
};
template<class A,class B>
nb::CompareResult episode_compare(nb::Bench&bench,const char*an,A&&a,const char*bn,B&&b){
  constexpr unsigned groups=16,pairs=2048;
  // Interleave whole episodes to share the WSL API/observation time window.
  // Sum every elapsed sample, including tails, into the original mean groups.
  const auto micro=bench.output(nullptr).epochs(groups*pairs).epochIterations(1).compare(an,a,bn,b);
  std::vector<nb::Result> results;results.reserve(2);
  for(unsigned side=0;side<2;++side){
    auto config=micro[side].result.config();config.mNumEpochs=groups;
    config.mEpochIterations=pairs;results.emplace_back(config);
    if(micro[side].result.size()!=groups*pairs)std::abort();
    for(unsigned group=0;group<groups;++group){
      double seconds=0;
      for(unsigned pair=0;pair<pairs;++pair){
        const auto index=group*pairs+pair;
        if(micro[side].result.get(index,nb::Result::Measure::iterations)!=1)std::abort();
        seconds+=micro[side].result.get(index,nb::Result::Measure::elapsed);
      }
      std::printf("# CUDA_EPISODE_GROUP side=%u group=%u count=%u seconds=%.17g\n",side,group,pairs,seconds);
      results[side].add(std::chrono::duration_cast<nb::Clock::duration>(std::chrono::duration<double>(seconds)),pairs,nb::detail::performanceCounters());
    }
  }
  const auto ratios=nb::detail::pairedLogRatios(results[0],results[1]);
  const auto ci=nb::detail::medianInterval(ratios,nb::detail::bonferroniConfidence(1));
  std::vector<nb::CompareResult::Entry> entries;
  entries.emplace_back(an,std::move(results[0]),1.,1.,1.,0);
  entries.emplace_back(bn,std::move(results[1]),std::exp(nb::detail::medianOf(ratios)),std::exp(ci.first),std::exp(ci.second),nb::detail::countTiedRounds(ratios));
  nb::CompareResult result(std::move(entries),groups);std::cout<<result;std::fflush(stdout);return result;
}
struct Harness {
  std::unique_ptr<hyle::bench::resident::CudaEncodedReferenceData> reference;
  c::Context context;c::cuda::MemoryResource memory,host;c::Queue queue;c::BatchSchema schema;c::BatchLayoutVersion layout;
  c::BatchProgram program;std::array<c::BatchInvocation,2> slots;c::LeasePool pool;
  c::PagedArena keys,values;c::PagedArenaRead key_frame,value_frame;
  std::vector<c::ReadLease> key_pages,value_pages;std::vector<unsigned> physical_ids;c::ReadLease query;std::size_t next=0;
  Harness(const Data &data,bool original){
    if(original)reference=std::make_unique<hyle::bench::resident::CudaEncodedReferenceData>();
    context=take(c::Context::create(take(c::cuda::device(0)),original?
        hyle::bench::resident::cuda_encoded_reference_provider(reference.get()):c::flashinfer::provider()));
    const auto env=take(c::cuda::environment(context));
    if(env.sm_major!=8||env.sm_minor!=9)fail(hyle::Status::Unimplemented("finite actual SM89 profile"));
    std::fprintf(stderr,"# CUDA_ENCODED_ENV device=%s sm=%d%d driver=%s runtime=%d toolkit=%d original=%d\n",
        env.device_name.data(),env.sm_major,env.sm_minor,env.driver_release.data(),env.runtime_version,env.build_toolkit_version,original);
    memory=take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
    host=take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
    queue=take(context.create_queue());
    schema=take(c::BatchSchema::create(data.capacity));layout=take(schema.version({data.qp,data.kp,data.ids,data.tails}));
    const auto key_bytes=16*data.row_bytes(0),value_bytes=16*data.row_bytes(1);
    keys=take(c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory.ref(),{.max_backing_bytes=key_bytes*data.capacity.max_pages+1}));
    values=take(c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory.ref(),{.max_backing_bytes=value_bytes*data.capacity.max_pages+1}));
    for(unsigned page=0;page<data.capacity.max_pages;++page)if(std::find(data.ids.begin(),data.ids.end(),static_cast<std::int32_t>(page))!=data.ids.end()){
      auto kw=take(keys.acquire(page)),vw=take(values.acquire(page));
      check(kw.lease().view()->write(std::span(data.packed[0]).subspan(page*key_bytes,key_bytes)));
      check(vw.lease().view()->write(std::span(data.packed[1]).subspan(page*value_bytes,value_bytes)));
      physical_ids.push_back(page);key_pages.push_back(take(kw.publish(16)));value_pages.push_back(take(vw.publish(16)));
    }
    key_frame=take(keys.read(layout,key_pages));value_frame=take(values.read(layout,value_pages));
    if(original)check(hyle::bench::resident::prepare_cuda_encoded_reference(*reference,queue,layout,key_frame,value_frame,memory.ref()));
    program=take(c::BatchProgram::prepare(context,schema,{.resource=memory.ref(),.host_workspace=host.ref(),
        .max_float_workspace_bytes=32u<<20,.max_int_workspace_bytes=4096,.scale=data.scale,.causal=data.causal,.window_left=data.window}));
    for(auto &slot:slots){slot=take(c::BatchInvocation::create(program,queue));auto prepared=take(slot.prepare_batch(layout));}
    pool=take(c::LeasePool::create(memory.ref(),data.query.size()*2+2048));
    auto writer=take(pool.allocate(data.query.size()*2));check(writer.view()->write(std::as_bytes(std::span(data.query))));query=take(writer.publish());
    std::fprintf(stderr,"# CUDA_ENCODED_COLD original=%d known_float_bytes=%zu known_int_bytes=%zu tile=%u split=%d private_pool_upper=UNKNOWN requests=2 immutable_history=1\n",original,slots[0].report().float_workspace_bytes,slots[0].report().int_workspace_bytes,slots[0].report().cta_tile_q,slots[0].report().split_kv);
  }
  ~Harness(){const auto start=std::chrono::steady_clock::now();
    for(auto &slot:slots){check(slot.close());slot={};}
    program={};query.reset();pool={};key_frame={};value_frame={};
    key_pages.clear();value_pages.clear();keys={};values={};reference.reset();
    check(queue.drain());queue={};host.reset();memory.reset();check(context.close());
    std::fprintf(stderr,"# CUDA_ENCODED_TEARDOWN seconds=%.9f\n",
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
  }
  c::ReadLease execute(){auto &slot=slots[next++%slots.size()];auto prepared=take(slot.update_batch(layout));if(!slot.report().reused_native_plan)fail(hyle::Status::Internal("paged warm topology rebuilt"));
    auto completion=take(prepared.run(query,key_frame,value_frame));check(completion.wait());return take(slot.output());}
  std::vector<std::uint16_t> verify(const Data &data,const std::vector<double> &expected){auto output=execute();std::vector<std::uint16_t> actual(std::size_t(data.capacity.max_query_rows)*8*data.capacity.value_head_dim);check(output.view()->read(std::as_writable_bytes(std::span(actual))));
    double worst=0;for(std::size_t i=0;i<expected.size();++i){if(!std::isfinite(decode(actual[i])))fail(hyle::Status::Internal("paged nonfinite output"));worst=std::max(worst,std::abs(double(decode(actual[i]))-expected[i]));}
    if(worst>8e-3)fail(hyle::Status::Internal("paged independent double oracle exceeded 8e-3"));
    for(unsigned port=1;port<3;++port)for(unsigned index=0;index<key_pages.size();++index){const auto &lease=port==1?key_pages[index]:value_pages[index];auto view=take(lease.view());
      std::vector<std::byte> after(view.size());check(view.read(after));
      const auto first=data.packed[port-1].begin()+std::size_t(physical_ids[index])*16*data.row_bytes(port-1);
      if(!std::equal(after.begin(),after.end(),first))fail(hyle::Status::Internal("paged readonly KV changed"));
    }
    std::fprintf(stderr,"# CUDA_ENCODED_CORRECTNESS max_abs_error=%.9g elements=%zu\n",worst,expected.size());return actual;
  }
};
void measure(int q1,int q2,int k1,int k2,int dk,int dv,bool causal,int window){Data data(q1,q2,k1,k2,dk,dv,causal,window);std::unique_ptr<Harness> candidate,peer;
  const auto *aa=std::getenv("HYLE_BENCH_AA_ROUTE");const std::string route=aa?aa:"";if(!route.empty()&&route!="native"&&route!="selected")fail(hyle::Status::InvalidArgument("ragged AA route"));
  const auto make_candidate=[&]{candidate=std::make_unique<Harness>(data,route=="native");};const auto make_peer=[&]{peer=std::make_unique<Harness>(data,route!="selected");};
  if(set("HYLE_BENCH_PEER_ALLOCATION_FIRST")){make_peer();make_candidate();}else{make_candidate();make_peer();}
  const auto expected=data.oracle();std::vector<std::uint16_t> candidate_values,peer_values;
  for(unsigned replay=0;replay<4;++replay){candidate_values=candidate->verify(data,expected);peer_values=peer->verify(data,expected);}
  double disagreement=0;for(std::size_t i=0;i<expected.size();++i)disagreement=std::max(disagreement,std::abs(double(decode(candidate_values[i]))-decode(peer_values[i])));
  if(disagreement>8e-3)fail(hyle::Status::Internal("FP8 peer disagreement exceeded 8e-3"));
  std::fprintf(stderr,"# CUDA_FP8_PAIR max_abs_difference=%.9g\n",disagreement);
  for(int i=0;i<4;++i){auto a=candidate->execute();auto b=peer->execute();nb::doNotOptimizeAway(a);nb::doNotOptimizeAway(b);}
  if(set("HYLE_BENCH_VERIFY_ONLY"))return;
  // The workload-local adapter groups single-episode timings without dropping
  // slow observations; the fixed epoch count overrides duration calibration.
  nb::Bench bench;bench.title("CUDA paged FP8 BF16 Q="+std::to_string(q1)+","+std::to_string(q2)+" KV="+std::to_string(k1)+","+std::to_string(k2)+" D="+std::to_string(dk)+"/"+std::to_string(dv)+" causal="+std::to_string(causal)+" window="+std::to_string(window)).batch(1).epochs(16).minEpochIterations(1).minEpochTime(std::chrono::milliseconds(40)).performanceCounters(false);
  const auto selected=[&]{auto result=candidate->execute();nb::doNotOptimizeAway(result);};const auto original=[&]{auto result=peer->execute();nb::doNotOptimizeAway(result);};
  if(set("HYLE_BENCH_RIVAL_FIRST"))(void)episode_compare(bench,"FI69ff sync-fixed cold-expanded immutable lifecycle",original,"HYLE CUDA encoded paged",selected);
  else (void)episode_compare(bench,"HYLE CUDA encoded paged",selected,"FI69ff sync-fixed cold-expanded immutable lifecycle",original);
}
}
int main(int,char **argv){const auto qos=hyle::bench::perf::configure_thread_qos();const auto affinity=hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool verify_only=set("HYLE_BENCH_VERIFY_ONLY");
  if(!verify_only&&!set("HYLE_BENCH_AA_ROUTE")&&set("HYLE_PERF_RUN_NONCE")){
    std::string ready;std::ifstream(std::filesystem::absolute(argv[0]).parent_path()/"g04_cuda_encoded_aa.ready")>>ready;
    if(ready!="PASS")fail(hyle::Status::FailedPrecondition("ragged finite workload AA not passed for this build"));
  }
  std::string parent_mask="not-applicable",priority="not-applicable";
#if defined(_WIN32)
  DWORD_PTR mask=0,system_mask=0;if(!GetProcessAffinityMask(GetCurrentProcess(),&mask,&system_mask))fail(hyle::Status::Internal("ragged process affinity observation"));
  char text[40]{};std::snprintf(text,sizeof(text),"0x%llx",static_cast<unsigned long long>(mask));parent_mask=text;
  const auto actual_priority=GetPriorityClass(GetCurrentProcess());if(!actual_priority)fail(hyle::Status::Internal("ragged process priority observation"));priority=std::to_string(actual_priority);
#endif
  const auto *aa=std::getenv("HYLE_BENCH_AA_ROUTE");
  std::fprintf(stderr,"# HYLE_CUDA_ENCODED_RUN run_nonce=%s apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s verify_only=%d parent_mask=%s priority=%s placement=%d execution_order=%s allocation_order=%s aa_route=%s paired_mode=nanobench-compare paired_rounds=16 timing_schedule=episode-interleaved arithmetic_group_size=2048\n",hyle::bench::perf::run_nonce(),qos.name,qos.ok,affinity.topology,affinity.cpu,affinity.ok,verify_only,parent_mask.c_str(),priority.c_str(),HYLE_RESIDENT_PLACEMENT,set("HYLE_BENCH_RIVAL_FIRST")?"peer-first":"candidate-first",set("HYLE_BENCH_PEER_ALLOCATION_FIRST")?"peer-first":"candidate-first",aa&&*aa?aa:"none");
  measure(1,17,1024,768,256,256,true,16);measure(64,128,1024,768,256,256,true,-1);measure(17,1,7,47,64,128,false,-1);
}
