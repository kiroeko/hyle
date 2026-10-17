#define main original_benchmark_main
#include "native_ragged_bench.cc"
#undef main
#include <mlx/compile.h>
#include <mlx/transforms.h>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/graph_utils.h>
#include <iostream>
#include <fstream>
namespace mx=mlx::core;
using NativeFunction=std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;
NativeFunction raw_program(const Data&d,mx::Stream stream,bool hyle,bool force){
  return mx::compile([cap=d.cap,qp=d.qp,kp=d.kp,scale=d.scale,causal=d.causal,stream,hyle,force](const std::vector<mx::array>&in){
    std::vector<mx::array>parts;
    for(unsigned r=0;r<2;++r){const int nq=qp[r+1]-qp[r],nk=kp[r+1]-kp[r];
      const auto view=[&](unsigned p,int start,int n,int heads){return mx::transpose(mx::reshape(mx::slice(in[p],{start,0,0},{start+n,heads,int(cap.qk_head_dim)},stream),{1,n,heads,int(cap.qk_head_dim)},stream),{0,2,1,3},stream);};
      auto q=view(0,qp[r],nq,8),k=view(1,kp[r],nk,2),v=view(2,kp[r],nk,2);
      auto o=hyle?mx::fast::hyle_ragged_attention(q,k,v,scale,causal,stream,false,-1):mx::fast::scaled_dot_product_attention(q,k,v,scale,causal?"causal":"",std::nullopt,std::nullopt,force,stream);
      parts.push_back(mx::reshape(mx::transpose(o,{0,2,1,3},stream),{nq,8,int(cap.qk_head_dim)},stream));
    }
    return std::vector{mx::concatenate(parts,0,stream)};
  },false);
}
void raw_execute(const NativeFunction&fn,const std::vector<mx::array>&inputs,mx::Stream stream){
  const std::vector<mx::array>bound{inputs[0],inputs[1],inputs[2]};
  auto outputs=fn(bound);mx::async_eval(outputs);mx::synchronize(stream);for(auto&o:outputs)o.wait();const auto owner=outputs[0];nb::doNotOptimizeAway(owner);
}
void raw_verify(const NativeFunction&fn,const std::vector<mx::array>&inputs,mx::Stream stream,const std::vector<double>&expected,const char*name){
  auto outputs=fn(inputs);std::fprintf(stderr,"# RAW fn returned n=%zu\n",outputs.size());mx::async_eval(outputs);std::fprintf(stderr,"# RAW async returned\n");mx::synchronize(stream);std::fprintf(stderr,"# RAW sync returned\n");for(auto&o:outputs)o.wait();
  const auto*bytes=reinterpret_cast<const std::uint16_t*>(outputs[0].data<mx::float16_t>());double worst=0;
  for(std::size_t i=0;i<expected.size();++i){if((bytes[i]&0x7c00u)==0x7c00u)std::exit(3);worst=std::max(worst,std::abs(decode(bytes[i])-expected[i]));}
  std::fprintf(stderr,"# RAW_VERIFY %s max_abs=%.9g\n",name,worst);if(worst>2e-3&&!set("HYLE_RAGGED_ADVERSARIAL"))std::exit(4);
}
void phase_sample(Harness&h,const char*name,int index){using Clock=std::chrono::steady_clock;const auto ns=[](auto a,auto b){return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();};auto&slot=h.slots[h.next++%h.slots.size()];
  const auto a=Clock::now();{const std::array bound{*h.inputs[0],*h.inputs[1],*h.inputs[2]};check(slot.bind(bound));}const auto b=Clock::now();
  auto submitted=take(slot.submit());const auto c=Clock::now();{auto result=take(submitted.wait());const auto d=Clock::now();{const auto output=result->arrays()[0];nb::doNotOptimizeAway(output);}const auto e=Clock::now();result={};const auto f=Clock::now();
  std::printf("{\"phase\":\"%s\",\"i\":%d,\"bind_ns\":%lld,\"submit_ns\":%lld,\"wait_ns\":%lld,\"own_ns\":%lld,\"release_ns\":%lld}\n",name,index,(long long)ns(a,b),(long long)ns(b,c),(long long)ns(c,d),(long long)ns(d,e),(long long)ns(e,f));}
}
int main(){auto qos=hyle::bench::perf::configure_thread_qos();std::fprintf(stderr,"# RAW qos=%s ok=%s forced=%d order=%s\n",qos.name,qos.ok,set("HYLE_RAGGED_FORCE_FUSED"),set("HYLE_BENCH_RIVAL_FIRST")?"BA":"AB");
  Data d(64,128,1024,768,256,true);
  if(set("HYLE_RAGGED_ADVERSARIAL")){
    for(auto&port:d.values)std::fill(port.begin(),port.end(),encode(0.F));
    for(unsigned row=0;row<d.cap.max_query_rows;++row)for(unsigned h=0;h<8;++h)d.values[0][(row*8+h)*256]=encode(1.F);
    for(unsigned row=0;row<d.cap.max_kv_rows;++row)for(unsigned h=0;h<2;++h){d.values[1][(row*2+h)*256]=encode(-256.F);for(unsigned z=0;z<256;++z)d.values[2][(row*2+h)*256+z]=encode(60000.F);}
    for(auto first:{d.kp[0],d.kp[1]})for(unsigned h=0;h<2;++h){d.values[1][(first*2+h)*256]=encode(0.F);for(unsigned z=0;z<256;++z)d.values[2][(first*2+h)*256+z]=encode(0.F);}
  }
  Harness holder(d,false),peer(d,true);auto expected=d.oracle();if(!set("HYLE_RAGGED_ADVERSARIAL")){holder.verify(d,expected);peer.verify(d,expected);}
  const std::vector<mx::array>inputs{*holder.inputs[0],*holder.inputs[1],*holder.inputs[2]};auto stream=mx::new_stream(mx::Device::gpu);auto a=raw_program(d,stream,true,false),b=raw_program(d,stream,false,set("HYLE_RAGGED_FORCE_FUSED"));
  std::fprintf(stderr,"# RAW graph candidate start\n");{auto o=a(inputs);std::fprintf(stderr,"# RAW graph candidate built\n");std::ofstream f("raw-candidate.dot");mx::export_to_dot(f,o);}std::fprintf(stderr,"# RAW graph peer start\n");{auto o=b(inputs);std::fprintf(stderr,"# RAW graph peer built\n");std::ofstream f(set("HYLE_RAGGED_FORCE_FUSED")?"raw-fast-forced.dot":"raw-fast.dot");mx::export_to_dot(f,o);}
  std::fprintf(stderr,"# RAW verify start\n");raw_verify(a,inputs,stream,expected,"hyle");raw_verify(b,inputs,stream,expected,"fast");if(set("HYLE_RAGGED_ADVERSARIAL"))return 0;for(int i=0;i<4;++i){raw_execute(a,inputs,stream);raw_execute(b,inputs,stream);}
  nb::Bench bench;bench.title("Direct compiled resident D256 Q64,128 KV1024,768 F16").batch(1).epochs(24).minEpochIterations(1).minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
  const auto runa=[&]{raw_execute(a,inputs,stream);};const auto runb=[&]{raw_execute(b,inputs,stream);};
  if(set("HYLE_BENCH_RIVAL_FIRST"))bench.compare("raw patched fast",runb,"raw hyle native",runa);else bench.compare("raw hyle native",runa,"raw patched fast",runb);
  for(int i=0;i<24;++i){if(set("HYLE_BENCH_RIVAL_FIRST")){phase_sample(peer,"fast",i);phase_sample(holder,"hyle",i);}else{phase_sample(holder,"hyle",i);phase_sample(peer,"fast",i);}}
}
