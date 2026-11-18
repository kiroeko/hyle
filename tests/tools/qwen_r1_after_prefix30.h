#pragma once

// EXPERIMENT ONLY, UNBUILT. Every added device read is AFTER the original
// PrefixSampling Completion and packet read, BEFORE original input release.
// All earlier hooks only assign bounded CPU POD fields: no formatting, heap,
// lease/storage query, MemoryAccess, device read, or additional wait.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"
#include "../compute_models/sampling_fixture.h"

namespace hyle::experimental::r1_after_prefix30 {
namespace c=hyle::compute;
namespace vn=hyle::compute::vulkan::detail;
using hyle::inference::SamplingCursor;
using qwen35::take;
using qwen35::check;
inline constexpr std::size_t max_vocabulary=248320, max_float_words=14*max_vocabulary;
struct Floats final {std::vector<float> value;};
struct Call final {
  unsigned request=0,role=0;
  std::size_t episode=0,generated=0,credit=0,quota=0,planned=0,length=0;
  std::int32_t pending=-1;
  SamplingCursor cursor{};
};
struct Prior final {std::size_t quota=0,planned=0,verified=0,consumed=0,emitted=0;SamplingCursor before{},after{};};
struct Draft final {
  qwen35::DraftProgram *program=nullptr; // Borrowed original cold owner, dereferenced only on the original post stack.
  std::size_t rows=0,sampler=0,steps=0;
  std::int32_t id=-1;
  std::array<std::uint32_t,4> cpu_key{},post_gpu_key{};
  bool retired=false,post_key_ok=false,superseded=false;
};
struct Sample final {
  bool armed=false,eligible=false,selected=false,before=false,after=false,target_ok=false,fixed_ok=false;
  unsigned request=0;
  const void *session=nullptr;
  const std::string *source=nullptr; // Borrowed for immediate post text only, never dereferenced at shutdown.
  std::size_t vocabulary=0,quota=0,planned=0,actual=0,draft_count=0,prior_count=0;
  std::size_t profile=0,sampler=0,capacity=0,lag=0,frontier=0,members=0,physical=0,slot=0;
  SamplingCursor cursor{},prefix_cursor{},published_cursor{};
  bool published=false,target_retired=false,conditional=false;
  std::array<Prior,4> prior{};
  std::array<Draft,3> draft{};
  std::array<std::int32_t,3> cpu_ids{},gpu_ids{};
  std::array<std::uint32_t,16> gpu_keys{};
  std::array<float,3> continuation{};
  std::array<std::int32_t,8> packet{};
  std::size_t packet_count=0;
  std::array<Floats,3> post_q;
  std::array<bool,3> q_ok{};
  Floats post_target_raw;
};
struct State final {
  std::array<Sample,2> samples;
  std::array<Call,32> calls{};
  std::size_t call_count=0,float_words=0,used=0;
  std::array<char,128*1024> text{};
  std::uint32_t pre_error=0;
  bool incomplete=false,overflow=false,compared=false;
  void note(const char *format,...) noexcept {
    char line[2048];va_list args;va_start(args,format);
    const auto n=std::vsnprintf(line,sizeof(line),format,args);va_end(args);
    if(n<0 || std::size_t(n)>=sizeof(line) || std::size_t(n)>text.size()-used){incomplete=overflow=true;return;}
    std::memcpy(text.data()+used,line,std::size_t(n));used+=std::size_t(n);
  }
  ~State() noexcept {
    for(std::size_t i=0;i<call_count;++i){const auto &v=calls[i];note("AFTER30_CALL request=%u episode=%zu generated=%zu output_credit=%zu quota=%zu planned_K=%zu role=%u length=%zu pending=%d seed=%llu stream=%llu draw=%llu\n",
        v.request,v.episode,v.generated,v.credit,v.quota,v.planned,v.role,v.length,v.pending,
        (unsigned long long)v.cursor.seed,(unsigned long long)v.cursor.stream,(unsigned long long)v.cursor.draw);}
    for(const auto &s:samples)note("AFTER30_COMPONENT request=%u post=%d q=%d,%d,%d target=%d fixed=%d published=%d after_draw=%llu\n",
        s.request,int(s.after),int(s.q_ok[0]),int(s.q_ok[1]),int(s.q_ok[2]),int(s.target_ok),int(s.fixed_ok),int(s.published),(unsigned long long)s.published_cursor.draw);
    note("R1_AFTER_PREFIX30_STATUS captured12=%d captured14=%d compared=%d incomplete=%d overflow=%d pre_error=%u float_words=%zu max_float_words=%zu post_only=1 timing_perturbed=1 fresh_or_pre_stability_observed=0 internal_P_observed=0\n",
        int(samples[0].after),int(samples[1].after),int(compared),int(incomplete || !samples[0].after || !samples[1].after),int(overflow),pre_error,float_words,max_float_words);
    if(used)(void)std::fwrite(text.data(),1,used,stderr);
  }
};
inline State state;
inline thread_local Sample *draft_sample=nullptr;
inline void pre_failure(std::uint32_t bit) noexcept {state.pre_error|=bit;state.incomplete=true;}

// BEGIN PRE-DECISION HOOKS: deliberately no format/throw/alloc/view/wait/read.
template<class Model,class PreparedCall> inline void prepared(Model &,const PreparedCall &call,std::size_t credit) noexcept {
  const auto &r=*call.request;
  const unsigned id=r.id=="12"?12:r.id=="14"?14:0;
  if(!id || !r.session || !r.session->head)return;
  if(state.call_count==state.calls.size()){pre_failure(1);return;}
  state.calls[state.call_count++]={id,unsigned(call.role),r.rounds.size()+1,r.progress.generated.size(),credit,call.quota,call.maximum,
      r.session->head->length,call.pending,r.session->head->sampling};
}
struct ProposalScope final {
  Sample *previous=draft_sample;
  template<class Model,class Request> ProposalScope(Model &model,const Request &r,std::size_t quota,std::size_t planned) noexcept {
    draft_sample=nullptr;
    const unsigned id=r.id=="12"?12:r.id=="14"?14:0;
    if(!id || r.rounds.size()!=2)return;
    auto &s=state.samples[id==12?0:1];
    if(s.armed){pre_failure(2);return;}
    s.armed=true;s.request=id;s.session=r.session.get();s.source=&model.source_identity;
    if(!r.session || !r.session->head || !r.session->head->draft || !r.ledger || model.cache_plan.active_sequences!=2 ||
        planned>3 || r.ledger->committed_count>r.ledger->records.size() || r.ledger->committed_count>s.prior.size()) {pre_failure(4);return;}
    const auto &head=*r.session->head;
    const auto &sampler=r.session->sampling_configuration().configuration;
    s.cursor=head.sampling;s.quota=quota;s.planned=planned;s.vocabulary=std::size_t(model.vocabulary);
    s.profile=r.execution_profile;s.sampler=r.sampling_index;s.capacity=model.capacity;s.lag=head.draft->lag;s.frontier=head.draft->frontier;
    if(!s.vocabulary || s.vocabulary>max_vocabulary || !sampler.enabled || !sampler.modifiers.empty() ||
        sampler.policy.temperature!=.8F || sampler.policy.top_k!=40 || sampler.policy.top_p!=.95F || s.cursor.seed!=42 || s.cursor.stream!=0) {pre_failure(8);return;}
    constexpr std::array<std::int32_t,5> expected{1,2,3,2,471};std::size_t at=0;
    for(std::size_t i=0;i<r.ledger->committed_count;++i){const auto &v=r.ledger->records[i];
      if(!v.committed){pre_failure(16);return;}
      for(const auto token:v.committed_ids()){if(at==expected.size() || token!=expected[at++]){pre_failure(16);return;}}
      s.prior[s.prior_count++]={v.quota,v.draft_limit,v.verified_rows,v.consumed_drafts,v.emitted_count,v.before.sampling,v.after.sampling};
    }
    constexpr std::array<std::int32_t,3> delivered{2,471,220};
    if(at!=expected.size() || head.length!=5 || head.next_id!=220 || r.progress.generated.size()!=delivered.size()){pre_failure(16);return;}
    for(std::size_t i=0;i<delivered.size();++i)if(r.progress.generated[i]!=delivered[i]){pre_failure(16);return;}
    s.eligible=true;draft_sample=&s;
  }
  ~ProposalScope(){draft_sample=previous;}
};
inline void proposal(qwen35::DraftProgram &program,const c::ReadLease &,std::int32_t id,
    const std::array<std::uint32_t,4> &cpu_key,hyle::inference::Backend &) noexcept {
  if(!draft_sample)return;
  auto &s=*draft_sample;
  if(s.draft_count==s.draft.size()){pre_failure(32);return;}
  auto &d=s.draft[s.draft_count++];d.program=&program;d.rows=program.rows;d.sampler=program.sampling_index;
  d.steps=program.steps;d.id=id;d.cpu_key=cpu_key;d.retired=program.carrier.retired();
}
struct TargetScope final {
  inline static thread_local TargetScope *current=nullptr;
  TargetScope *previous=current;
  Sample *sample=nullptr;
  const c::ReadLease *generated=nullptr;
  qwen35::Tensor *caller=nullptr;
  const hyle::inference::ProgramCarrier<qwen35::ColdResources> *carrier=nullptr;
  c::ConstTensorView exact;
  TargetScope(const void *session,const c::ReadLease &lease,qwen35::Tensor &tensor,
      const hyle::inference::ProgramCarrier<qwen35::ColdResources> *original,c::ConstTensorView view,
      std::size_t members,std::size_t physical,std::size_t slot) noexcept
      :generated(&lease),caller(&tensor),carrier(original),exact(view) {
    for(auto &s:state.samples)if(s.eligible && s.session==session && !s.selected){s.selected=true;sample=&s;break;}
    if(sample){sample->members=members;sample->physical=physical;sample->slot=slot;sample->target_retired=original && original->retired();}
    current=this;
  }
  ~TargetScope(){current=previous;}
};
inline void prefix_before(qwen35::PrefixSamplingProgram &,std::span<const c::ReadLease> q,
    std::span<const std::int32_t> ids,SamplingCursor cursor,std::size_t quota,
    const hyle::inference::LogitsInputs &inputs,std::span<const hyle::inference::LogitsInputs> conditional,
    hyle::inference::Backend &) noexcept {
  auto *scope=TargetScope::current;if(!scope || !scope->sample)return;auto &s=*scope->sample;
  s.actual=q.size();s.prefix_cursor=cursor;s.conditional=inputs.bias.lease.valid() || inputs.allowed.lease.valid() || inputs.counts.lease.valid() || !conditional.empty();
  if(s.before || q.size()!=ids.size() || q.size()>3 || s.draft_count!=q.size() || cursor!=s.cursor || quota!=s.quota ||
      s.members!=1 || !s.target_retired || s.conditional){pre_failure(64);return;}
  for(std::size_t i=0;i<ids.size();++i){s.cpu_ids[i]=ids[i];if(ids[i]!=s.draft[i].id){pre_failure(64);return;}}
  s.before=true;
}
template<class Model,class PreparedCall> inline void published(Model &,std::span<PreparedCall> calls) noexcept {
  for(const auto &call:calls){
    const unsigned id=call.request->id=="12"?12:call.request->id=="14"?14:0;
    if(!id)continue;
    for(auto &s:state.samples)if(s.after && s.request==id && s.session==call.request->session.get() && call.request->rounds.size()==2){
      s.published_cursor=call.request->session->head->sampling;s.published=true;
    }
  }
}
// END PRE-DECISION HOOKS. Formatting/allocation/backend access only below.

inline void failure(const char *where,std::string_view message) noexcept {state.incomplete=true;state.note("AFTER30_UNAVAILABLE stage=%s message=%.*s\n",where,int((std::min)(message.size(),std::size_t{512})),message.data());}
template<class F> inline bool observe(const char *where,F &&f) noexcept {
  try{f();return true;}catch(const hyle::inference::EvaluationFailure &e){failure(where,e.status().message());}
  catch(const std::exception &e){failure(where,e.what());}catch(...){failure(where,"unknown post observation error");}return false;
}
inline void identity(const char *label,Sample &s,std::size_t row,mem::ConstBufferView view,const c::TensorDesc &desc,
    std::uint64_t generation,const char *source,bool generation_valid,hyle::inference::Backend &backend) {
  const auto span=vn::MemoryAccess::copy_span(vn::Access::runtime(*take(backend.vulkan_context())),view);
  if(!span){failure("physical_span",span.error().message());return;}
  const auto dims=desc.dimensions();
  state.note("AFTER30_STORAGE label=%s request=%u row=%zu dtype=%u rank=%zu dim0=%lld dim1=%lld bytes=%zu view_offset=%zu backing=%llu physical_offset=%zu source=%s generation=%llu lease_generation_valid=%d no_lease_generation=%d post_only=1\n",
      label,s.request,row,unsigned(desc.data_type()),dims.size(),(long long)(dims.empty()?-1:dims[0]),(long long)(dims.size()>1?dims[1]:-1),view.size(),view.offset(),
      (unsigned long long)span->physical.backing,span->physical.origin,source,(unsigned long long)generation,int(generation_valid),int(!generation_valid));
}
inline void read_float_view(const char *label,Sample &s,std::size_t row,mem::ConstBufferView view,const c::TensorDesc &desc,
    Floats &result,std::uint64_t generation,const char *source,bool generation_valid,hyle::inference::Backend &backend) {
  const auto bytes=take(desc.required_bytes());
  if(desc.data_type()!=c::DataType::kFloat32 || view.size()!=bytes || bytes%sizeof(float))throw std::runtime_error("post F32 exact range/type differs");
  const auto words=bytes/sizeof(float);if(words>max_float_words-state.float_words)throw std::runtime_error("post RAM payload bound");
  result.value.resize(words);state.float_words+=words;
  identity(label,s,row,view,desc,generation,source,generation_valid,backend);
  check(view.read(std::as_writable_bytes(std::span(result.value))));
}
struct OriginalTensorView final {mem::ConstBufferView view;std::uint64_t generation=0;const char *source=nullptr;bool generation_valid=false;};
inline OriginalTensorView original_tensor_view(qwen35::Tensor &tensor) {
  if(unsigned(tensor.buffer.valid())+unsigned(tensor.writer.valid())+unsigned(tensor.ready.valid())!=1)throw std::runtime_error("original Tensor missing/mixed payload owner");
  OriginalTensorView owned;
  if(tensor.ready.valid())owned={take(tensor.ready.view()),tensor.ready.generation(),"original_read_lease",true};
  else if(tensor.writer.valid())owned={take(tensor.writer.view()),tensor.writer.generation(),"original_exclusive_writer",true};
  else owned={tensor.buffer.view(),0,"original_owned_buffer",false};
  owned.view=take(owned.view.subview(0,take(tensor.description.required_bytes())));return owned;
}
template<class T> inline void read_fixed(const char *label,Sample &s,qwen35::Tensor &tensor,
    const hyle::inference::ProgramCarrier<qwen35::ColdResources> &carrier,std::span<T> result,hyle::inference::Backend &backend) {
  if(!carrier.retired())throw std::runtime_error("post fixed original carrier not retired");
  const auto owned=original_tensor_view(tensor);if(owned.view.size()!=result.size_bytes())throw std::runtime_error("post fixed exact span differs");
  identity(label,s,0,owned.view,tensor.description,owned.generation,owned.source,owned.generation_valid,backend);
  check(owned.view.read(std::as_writable_bytes(result)));
}
inline void read_target(TargetScope &scope,Sample &s,hyle::inference::Backend &backend) {
  if(!scope.carrier || !scope.carrier->retired())throw std::runtime_error("post target original carrier not retired");
  const auto owned=scope.generated->valid()?OriginalTensorView{take(scope.generated->view()),scope.generated->generation(),"original_read_lease",true}:original_tensor_view(*scope.caller);
  const auto wanted=scope.exact.storage();const auto bytes=take(scope.exact.description().required_bytes());
  if(owned.view.resource()!=wanted.resource() || owned.view.native_handle()!=wanted.native_handle() || wanted.offset()<owned.view.offset())throw std::runtime_error("post target owner root differs");
  const auto offset=wanted.offset()-owned.view.offset();if(offset>owned.view.size() || bytes>owned.view.size()-offset || bytes>wanted.size())throw std::runtime_error("post target exact view exceeds original owner");
  read_float_view("target_RAW_logits_NOT_internal_P",s,0,take(owned.view.subview(offset,bytes)),scope.exact.description(),s.post_target_raw,owned.generation,owned.source,owned.generation_valid,backend);
}
inline void difference(const char *label,std::span<const float> a,std::span<const float> b,std::size_t row) {
  if(a.size()!=b.size())throw std::runtime_error("post pair shapes differ");std::size_t count=0,nonfinite=0,shown=0;double maximum=0;
  for(std::size_t i=0;i<a.size();++i)if(std::bit_cast<std::uint32_t>(a[i])!=std::bit_cast<std::uint32_t>(b[i])){++count;
    if(std::isfinite(a[i]) && std::isfinite(b[i]))maximum=(std::max)(maximum,std::abs(double(a[i])-double(b[i])));else ++nonfinite;
    if(shown++<8)state.note("AFTER30_DIFF label=%s row=%zu word=%zu request12_bits=%08x request14_bits=%08x a=%a b=%a\n",label,row,i,std::bit_cast<std::uint32_t>(a[i]),std::bit_cast<std::uint32_t>(b[i]),double(a[i]),double(b[i]));
  }
  state.note("AFTER30_DIFF_SUMMARY label=%s row=%zu words=%zu bit_diff=%zu max_finite_abs=%.17g nonfinite_diff=%zu after_decision_only=1\n",label,row,a.size(),count,maximum,nonfinite);
}

// BEGIN POST-ONLY INDEPENDENT SCALAR DIAGNOSTIC (no Graph/Session/test runner).
// Philox is reused from the independent fixture, never production RNG/math.
// CPU policy/residual/Gumbel is a RECONSTRUCTION, not observed internal P.
template<class Real> struct HostPolicy final {
  struct Entry {std::uint32_t id=0;Real value=-std::numeric_limits<Real>::infinity(),logp=-std::numeric_limits<Real>::infinity();};
  std::array<Entry,40> top{};std::size_t count=0,kept=0;
  Real at(std::size_t token) const noexcept {for(std::size_t i=0;i<kept;++i)if(top[i].id==token)return top[i].logp;return -std::numeric_limits<Real>::infinity();}
};
template<class Real> inline HostPolicy<Real> host_policy(std::span<const float> raw) {
  HostPolicy<Real> p;float maximum=-INFINITY;
  for(const auto x:raw){if(std::isnan(x) || x==INFINITY)throw std::runtime_error("post raw invalid for CPU policy diagnostic");maximum=(std::max)(maximum,x);}
  if(!std::isfinite(maximum))throw std::runtime_error("post raw has no CPU policy support");
  for(std::size_t token=0;token<raw.size();++token){const Real value=(Real(raw[token])-Real(maximum))/Real(.8F);if(!std::isfinite(value))continue;
    std::size_t at=0;while(at<p.count && (p.top[at].value>value || (p.top[at].value==value && p.top[at].id<token)))++at;
    if(at==p.top.size())continue;const auto next=(std::min)(p.top.size(),p.count+1);
    for(auto j=next-1;j>at;--j)p.top[j]=p.top[j-1];p.top[at]={std::uint32_t(token),value,-std::numeric_limits<Real>::infinity()};p.count=next;
  }
  Real mass=0;for(std::size_t i=0;i<p.count;++i)mass+=std::exp(p.top[i].value-p.top[0].value);
  Real prior=0,kept_mass=0;
  for(std::size_t i=0;i<p.count;++i){const Real weight=std::exp(p.top[i].value-p.top[0].value);const bool keep=i==0 || prior<Real(.95F);
    prior+=weight/mass;if(keep){++p.kept;kept_mass+=weight;}else break;}
  for(std::size_t i=0;i<p.kept;++i)p.top[i].logp=(p.top[i].value-p.top[0].value)-std::log(kept_mass);
  return p;
}
template<class Real> inline Real host_gumbel(std::uint32_t token,const std::array<std::uint32_t,4> &key) {
  const auto words=sampling_test::philox({token/2,0,key[2],key[3]},{key[0],key[1]});auto hi=words[(token%2)*2],lo=words[(token%2)*2+1];
  const bool upper=(hi&0x80000000U)!=0;if(upper){hi=~hi;lo=~lo;}
  const Real small=Real(hi)*Real(0x1p-32)+ (Real(lo)+Real(.5))*Real(0x1p-64);
  return -std::log(upper?-std::log1p(-small):-std::log(small));
}
template<class Real> inline Real host_residual(Real lp,Real lq) {
  if(!std::isfinite(lp) || (std::isfinite(lq) && lp<=lq))return -std::numeric_limits<Real>::infinity();
  if(!std::isfinite(lq))return lp;
  // Mathematical stable identity; float mode is still HOST libm, not the GPU's softplus/tanh lowering.
  return lp+std::log(-std::expm1(lq-lp));
}
template<class Real> inline void scalar_diagnostic(const char *mode,const Sample &s) {
  if(!s.actual || s.vocabulary<=19 || s.post_target_raw.value.size()!=(s.actual+1)*s.vocabulary ||
      s.post_q[0].value.size()!=s.vocabulary)throw std::runtime_error("post policy diagnostic exact row shape missing");
  const auto raw=std::span<const float>(s.post_target_raw.value).first(s.vocabulary);const auto p=host_policy<Real>(raw);
  const auto &q=s.post_q[0].value;bool q_support=false;
  for(const auto value:q){
    if(std::isnan(value) || value==INFINITY)throw std::runtime_error("post q NaN/+Inf: CPU distribution diagnostic unavailable; original result unchanged");
    q_support=q_support || std::isfinite(value);
  }
  if(!q_support)throw std::runtime_error("post q has no finite support: CPU distribution diagnostic unavailable; original result unchanged");
  const auto proposed=std::size_t(s.gpu_ids[0]);if(proposed>=q.size())throw std::runtime_error("post proposal outside CPU diagnostic q");
  const auto key=[&](std::size_t row){std::array<std::uint32_t,4> k{};std::copy_n(s.gpu_keys.begin()+4*row,4,k.begin());return k;};
  const Real log_accept=(std::min)(Real(0),p.at(proposed)-Real(q[proposed]));
  const Real log_reject=log_accept==Real(0)?-std::numeric_limits<Real>::infinity():std::log(-std::expm1(log_accept));
  const auto accept_key=key(0),final_key=key(s.actual);
  const Real score_accept=log_accept+host_gumbel<Real>(0,accept_key),score_reject=log_reject+host_gumbel<Real>(1,accept_key);
  const bool host_accept=score_accept>=score_reject;
  const bool gpu_first_reject=s.packet_count>=4 && s.packet[3]==0;
  state.note("AFTER30_CPU_DIAGNOSTIC mode=%s request=%u policy=temperature0.8_topk40_topp0.95 kept=%zu proposal=%zu cpu_accept=%d gpu_first_agrees=%d acceptance_margin=%.17g mathematical_reconstruction=1 actual_internal_P_observed=0 numerical_error_bound=UNKNOWN correctness_gate=0\n",
      mode,s.request,p.kept,proposed,int(host_accept),s.packet[3],double(score_accept-score_reject));
  Real maximum=-std::numeric_limits<Real>::infinity();
  for(std::size_t i=0;i<p.kept;++i)maximum=(std::max)(maximum,host_residual(p.top[i].logp,Real(q[p.top[i].id])));
  if(!std::isfinite(maximum)){state.note("AFTER30_CPU_RESIDUAL mode=%s request=%u support=empty gpu_first_reject=%d correctness_conclusion=NONE\n",mode,s.request,int(gpu_first_reject));return;}
  Real best=-std::numeric_limits<Real>::infinity(),runner=best;std::uint32_t best_id=UINT32_MAX;
  for(std::size_t i=0;i<p.kept;++i){const auto token=p.top[i].id;const Real weight=host_residual(p.top[i].logp,Real(q[token]));if(!std::isfinite(weight))continue;
    const Real score=(weight-maximum)+host_gumbel<Real>(token,final_key);
    if(score>best || (score==best && token<best_id)){runner=best;best=score;best_id=token;}else runner=(std::max)(runner,score);
  }
  std::array<Real,2> scores{};
  for(std::size_t at=0;at<2;++at){const std::uint32_t token=at?19:16;const Real lp=p.at(token),lq=Real(q[token]);
    const Real residual=host_residual(lp,lq),gumbel=host_gumbel<Real>(token,final_key);scores[at]=(residual-maximum)+gumbel;
    state.note("AFTER30_CPU_CANDIDATE mode=%s request=%u token=%u reconstructed_logP=%.17g observed_post_logQ=%.17g reconstructed_logResidual=%.17g reconstructed_Gumbel=%.17g centered_score=%.17g gpu_first_reject=%d\n",
        mode,s.request,token,double(lp),double(lq),double(residual),double(gumbel),double(scores[at]),int(gpu_first_reject));
  }
  state.note("AFTER30_CPU_MARGIN mode=%s request=%u score16_minus19=%.17g reconstructed_best=%u best_minus_runner=%.17g actual_GPU_packet_first=%d residual_is_selected_path=%d correctness_conclusion=NONE\n",
      mode,s.request,double(scores[0]-scores[1]),best_id,double(best-runner),s.packet_count>4?s.packet[4]:-1,int(gpu_first_reject));
}
// END CPU DIAGNOSTIC. F64/F32 disagreement or agreement is NOT an acceptance criterion.

inline void prefix_after(qwen35::PrefixSamplingProgram &program,std::span<const c::ReadLease> q,hyle::inference::Backend &backend) noexcept {
  auto *scope=TargetScope::current;if(!scope || !scope->sample)return;auto &s=*scope->sample;
  // FIRST: preserve the already-read original GPU decision, before any extra
  // formatting, allocation, backend qualification or device transfer.
  if(program.packet_host.size()>s.packet.size()){pre_failure(128);return;}
  s.packet_count=program.packet_host.size();for(std::size_t i=0;i<s.packet_count;++i)s.packet[i]=program.packet_host[i];
  observe("post_context",[&] {
    if(!s.before || s.after || !scope->carrier || !scope->carrier->retired() || !program.carrier.retired() || q.size()!=s.actual)
      throw std::runtime_error("post original target/prefix completion or exact cohort missing");
    s.after=true;
    state.note("AFTER30_CAPTURE request=%u source_identity=%s actual_K=%zu quota=%zu planned_K=%zu seed=%llu stream=%llu draw=%llu lag=%zu frontier=%zu members=%zu physical_query_rows=%zu slot=%zu packet_saved_before_extra_GPU_reads=1 snapshot=AFTER_DECISION_NOT_FRESH_OR_PRE\n",
        s.request,s.source?s.source->c_str():"UNKNOWN",s.actual,s.quota,s.planned,(unsigned long long)s.cursor.seed,(unsigned long long)s.cursor.stream,(unsigned long long)s.cursor.draw,s.lag,s.frontier,s.members,s.physical,s.slot);
    for(std::size_t i=0;i<s.packet_count;++i)state.note("AFTER30_ORIGINAL_PACKET request=%u word=%zu value=%d\n",s.request,i,s.packet[i]);
    for(std::size_t i=0;i<s.prior_count;++i){const auto &v=s.prior[i];state.note("AFTER30_PRIOR request=%u ordinal=%zu quota=%zu planned_K=%zu verified=%zu consumed_drafts=%zu emitted=%zu before_draw=%llu after_draw=%llu\n",s.request,i,v.quota,v.planned,v.verified,v.consumed,v.emitted,(unsigned long long)v.before.draw,(unsigned long long)v.after.draw);}
    s.target_ok=observe("post_target_RAW_logits",[&]{read_target(*scope,s,backend);});
    for(std::size_t i=0;i<q.size();++i)s.q_ok[i]=observe("post_q",[&]{read_float_view("q_AFTER_DECISION",s,i,take(q[i].view()),program.io.input_facts(program.q_inputs[i]).description,s.post_q[i],q[i].generation(),"original_read_lease",true,backend);});
    s.fixed_ok=observe("post_prefix_fixed",[&]{
      read_fixed("prefix_GPU_keys_AFTER",s,program.identity_values,program.carrier,std::span(s.gpu_keys).first(4*(s.actual+1)),backend);
      if(s.actual){read_fixed("prefix_GPU_ids_AFTER",s,program.id_values,program.carrier,std::span(s.gpu_ids).first(s.actual),backend);
        read_fixed("prefix_GPU_continuation_AFTER",s,program.continuation_values,program.carrier,std::span(s.continuation).first(s.actual),backend);}
      for(std::size_t i=0;i<=s.actual;++i)state.note("AFTER30_GPU_KEY request=%u row=%zu words=%u,%u,%u,%u\n",s.request,i,s.gpu_keys[4*i],s.gpu_keys[4*i+1],s.gpu_keys[4*i+2],s.gpu_keys[4*i+3]);
      for(std::size_t i=0;i<s.actual;++i)state.note("AFTER30_GPU_FIXED request=%u row=%zu gpu_id=%d cpu_id=%d continuation_bits=%08x\n",s.request,i,s.gpu_ids[i],s.cpu_ids[i],std::bit_cast<std::uint32_t>(s.continuation[i]));
    });
    for(std::size_t i=0;i<s.draft_count;++i){auto &d=s.draft[i];for(std::size_t j=i+1;j<s.draft_count;++j)if(s.draft[j].program==d.program)d.superseded=true;
      state.note("AFTER30_DRAFT_CPU request=%u row=%zu rows=%zu sampler=%zu steps=%zu id=%d cpu_argument_key=%u,%u,%u,%u actual_GPU_historical_key_observed=0 superseded=%d\n",s.request,i,d.rows,d.sampler,d.steps,d.id,d.cpu_key[0],d.cpu_key[1],d.cpu_key[2],d.cpu_key[3],int(d.superseded));
      if(!d.superseded)d.post_key_ok=observe("latest_draft_GPU_key_post",[&]{
        read_fixed("latest_draft_GPU_key_POST_ONLY",s,d.program->identity_values,d.program->carrier,std::span(d.post_gpu_key),backend);
        state.note("AFTER30_LATEST_DRAFT_GPU_KEY request=%u last_row=%zu words=%u,%u,%u,%u earlier_overwritten_rows_not_inferred=1\n",s.request,i,d.post_gpu_key[0],d.post_gpu_key[1],d.post_gpu_key[2],d.post_gpu_key[3]);
      });
      d.program=nullptr; // Never carry this borrowed original owner beyond the post stack.
    }
    if(s.actual && s.fixed_ok && s.target_ok && s.q_ok[0]){
      observe("CPU_F64_reconstruction",[&]{scalar_diagnostic<double>("host_F64_equations_not_internal_P",s);});
      observe("CPU_F32_reconstruction",[&]{scalar_diagnostic<float>("host_scalar_F32_libm_not_GPU",s);});
    }
    const auto &a=state.samples[0],&b=state.samples[1];if(!a.after || !b.after)return;
    if(a.vocabulary!=b.vocabulary || a.actual!=b.actual)throw std::runtime_error("post pair differs in actual shape; not claimed equal");
    bool complete=a.target_ok && b.target_ok && a.fixed_ok && b.fixed_ok;
    if(a.target_ok && b.target_ok)difference("request12_vs14_target_RAW_POST",a.post_target_raw.value,b.post_target_raw.value,0);
    for(std::size_t i=0;i<a.actual;++i){complete=complete && a.q_ok[i] && b.q_ok[i];if(a.q_ok[i] && b.q_ok[i])difference("request12_vs14_q_POST",a.post_q[i].value,b.post_q[i].value,i);}
    if(a.fixed_ok && b.fixed_ok)state.note("AFTER30_CROSS_FIXED cursor_equal=%d planned_K_equal=%d quota_equal=%d actual_GPU_keys_equal=%d GPU_proposals_equal=%d continuation_bits_equal=%d post_only=1\n",
        int(a.cursor==b.cursor),int(a.planned==b.planned),int(a.quota==b.quota),int(a.gpu_keys==b.gpu_keys),int(a.gpu_ids==b.gpu_ids),int(std::memcmp(a.continuation.data(),b.continuation.data(),sizeof(a.continuation))==0));
    state.compared=complete;
  });
}
} // namespace hyle::experimental::r1_after_prefix30
