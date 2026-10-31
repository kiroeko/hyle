// Finite original KDA0 -> NoPE MLA3 component consumer, not token generation.
// The independent source-only reference is test code, never a runtime fallback.
#include "../../engine/models/kimi_k3/component.h"
#include "../compute_models/kimi_original_attention_reference.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace component=hyle::inference::kimi_k3;
namespace c=hyle::compute;
namespace inference=hyle::inference;
namespace original=component::roles;
namespace reference=kimi_original_attention_reference;
using component::Bytes;
using component::Epoch;
using component::kEpoch;
using component::kHistory;
using component::kRecurrent;
using component::kMla;
using component::kCursor;
using component::kFeature;
using original::detail::take;

namespace {
void require(bool value,const char *where) {if(!value)throw std::runtime_error(where);}
void check(hyle::Status status,const char *where) {
  if(!status.ok())throw inference::EvaluationFailure(status.WithContext(where));
}
template<class T> std::vector<T> values(const c::ReadLease &lease) {
  const auto raw=component::read(lease);require(raw.size()%sizeof(T)==0,"typed output extent");
  std::vector<T> result(raw.size()/sizeof(T));std::memcpy(result.data(),raw.data(),raw.size());return result;
}
template<class T> Bytes bytes(const std::vector<T> &input) {
  const auto raw=std::as_bytes(std::span(input));return {raw.begin(),raw.end()};
}
std::vector<float> upstream(int seed) {
  // Same explicit normalized-upstream fixture as the original component test.
  std::vector<float> result(7168);
  for(std::size_t i=0;i<result.size();++i) {
    const auto wave=std::cos(static_cast<double>(i+31*seed)*0.017);
    result[i]=static_cast<float>(wave*0.0625+double((i+seed)%7)*0.002-0.006);
  }
  double square=0;for(const auto value:result)square+=double(value)*value;
  const auto factor=1/std::sqrt(square/double(result.size())+double(1.e-5F));
  for(auto &value:result)value=float(double(value)*factor);
  return result;
}
void compare(std::span<const c::ReadLease> output,const std::vector<std::vector<float>> &expected,
    bool attention,std::size_t visible=0) {
  require(output.size()>=expected.size(),"independent output inventory");
  for(std::size_t port=0;port<expected.size();++port) {
    const auto actual=values<float>(output[port]);require(actual.size()==expected[port].size(),"independent output extent");
    const auto checked=visible&&port==0?visible*576:actual.size();
    const bool bfloat=attention&&(port==0||port==1||port==2||port==3||port==4||port==5||port==6||port==8);
    const float absolute=bfloat?2.0e-5F:8.0e-5F,relative=bfloat?0.025F:4.0e-5F;
    for(std::size_t index=0;index<checked;++index)
      if(!std::isfinite(actual[index])||std::abs(double(actual[index])-expected[port][index])>
          double(absolute+relative*std::abs(expected[port][index])))
        throw std::runtime_error("independent Kimi output differs at port="+std::to_string(port)+" element="+std::to_string(index));
    for(std::size_t index=checked;index<actual.size();++index)
      require(std::bit_cast<std::uint32_t>(actual[index])==std::bit_cast<std::uint32_t>(expected[port][index]),"unwritten cache sentinel changed");
  }
}
std::array<Bytes,kEpoch> snapshot(const component::State &state,const c::StateVersion &version) {
  std::array<Bytes,kEpoch> result;
  for(std::size_t index=0;index<result.size();++index)result[index]=component::read(state.member(version,static_cast<component::Entry>(index)));
  return result;
}
void verify_reset(component::Program &program) {
  auto &run=*take(program.carrier.borrow(),"Kimi original reset Invocation");
  auto stale=run.submit(program.carrier.queues());
  if(stale) {
    const auto retired=program.carrier.retire(*program.backend);
    check(retired.status,"Kimi unexpected stale submit retirement");
    require(retired.retired,"Kimi unexpected stale submit is still retiring");
    check(program.carrier.reset_retired(),"Kimi unexpected stale submit reset");
    throw std::runtime_error("Kimi reset retained stale replay");
  }
}
void certify(float actual,const reference::Dot &expected) {
  require(std::isfinite(actual)&&reference::scalar::distance(double(actual),expected.center)<=expected.bound,"actual-producer original-row certificate failed");
}
void save(const std::filesystem::path &directory,std::int32_t position,const char *name,std::span<const c::ReadLease> outputs) {
  if(directory.empty())return;
  std::filesystem::create_directories(directory);
  for(std::size_t port=0;port<outputs.size();++port) {
    const auto raw=component::read(outputs[port]);
    std::ofstream stream(directory/(std::to_string(position)+"-"+name+"-"+std::to_string(port)+".bin"),std::ios::binary);
    if(!stream||!stream.write(reinterpret_cast<const char *>(raw.data()),static_cast<std::streamsize>(raw.size())))throw std::runtime_error("component vector write failed");
  }
}
} // namespace

int main(int argc,char **argv) {
  if(argc==2&&std::string_view(argv[1])=="--help") {
    std::cout<<"kimi_kda_mla_probe INPUT_DIRECTORY [VECTOR_DIRECTORY]\nRead-only fixed original component input; no downloads, tokens, FFN or full-model claim.\n";return 0;
  }
  try {
    require(argc==2||argc==3,"usage: kimi_kda_mla_probe INPUT_DIRECTORY [VECTOR_DIRECTORY]");
    const auto input=original::load_input(argv[1]);
    const auto source_kda=original::Data::load(input.directory,original::Component::kKda,input.pins());
    const auto source_mla=original::Data::load(input.directory,original::Component::kMla,input.pins());
    auto backend=std::make_shared<inference::Backend>(take(inference::Backend::create({.primary=inference::BackendKind::kCuda}),"Kimi Backend"));
    const auto &owner=backend->storage_owner();
    const auto kda_operands=original::prepare_kda(source_kda,owner);
    const auto mla_operands=original::prepare_mla(source_mla,owner);
    auto recurrent=component::Program::prepare(component::build_kda_graph(kda_operands,owner),backend);
    component::LiteralBank small_literals;
    std::array<std::shared_ptr<component::Program>,2> attention;
    for(std::int32_t position=0;position<2;++position)
      attention[std::size_t(position)]=component::Program::prepare(component::build_mla_graph(position,mla_operands,owner,small_literals),backend);
    small_literals.release_host_bytes();
    std::size_t budget=recurrent->pool.stats().capacity_bytes+32u*1024u*1024u;
    for(const auto &program:attention)budget+=program->pool.stats().capacity_bytes;
    auto state=component::State::create(owner,budget);
    std::size_t committed=0,stale=0,cancelled=0,refused=0;
    {
      const auto initial=state.initial(owner,0);const auto old_initial=snapshot(state,initial);
      auto head=take(c::StateHead::create(state.store,initial),"complete Head");c::StateVersion first;
      for(std::int32_t position=0;position<2;++position) {
        auto before=take(head.current(),"expected base");const auto old=snapshot(state,before);
        require(values<std::int32_t>(state.member(before,kCursor))==std::vector<std::int32_t>{position},"old cursor");
        const auto old_epoch=state.epoch(before);require(old_epoch.step==std::uint64_t(position)&&old_epoch.position==position&&old_epoch.session==0,"old Epoch");
        const auto activation=upstream(3+26*position);
        const std::array args{component::upload(owner,bytes(activation)),state.member(before,kHistory),state.member(before,kRecurrent)};
        auto &program=attention[std::size_t(position)];
        auto hidden_writer=take(recurrent->pool.allocate(take(recurrent->output(5).description.required_bytes(),"hidden extent")),"hidden writer");
        check(take(hidden_writer.view(),"initial hidden view").fill(std::byte{}),"initialize hidden");
        std::array mla_args{take(hidden_writer.publish(),"future hidden lease"),state.member(before,kMla),state.member(before,kCursor)};
        if(position==0) {
          auto *original_run=take(program->carrier.borrow(),"original unsubmitted Invocation");
          auto invalid=mla_args;invalid[1]=component::upload(owner,bytes(std::vector<float>{0}));
          try {(void)program->prepare_inputs(invalid);throw std::runtime_error("short MLA cache was accepted");}
          catch(const inference::EvaluationFailure &failure) {
            if(!failure.status().IsOutOfRange()||failure.status().has_native_error()||
                !failure.retirement_observed()||!failure.retirement().retired) {
              std::cerr<<"short-cache original status="<<static_cast<unsigned>(failure.status().code())
                  <<" native="<<failure.status().has_native_error()<<" retirement_observed="<<failure.retirement_observed()
                  <<" retired="<<failure.retirement().retired<<" message="<<failure.status().message()<<'\n';
              inference::EvaluationFailure unexpected(failure.status().WithContext("Kimi short-cache requires original range refusal and retirement"));
              if(failure.retirement_observed())unexpected.observe_retirement(failure.retirement());
              throw unexpected;
            }
            ++refused;
          }
          require(!program->pending&&!program->submit_attempted&&program->carrier.initialized()&&
              take(program->carrier.borrow(),"same Invocation after bind refusal")==original_run,"bind refusal replaced/retained Invocation");
          require(take(head.current(),"Head after refusal").identity()==before.identity(),"refusal published state");
        }
        auto submitted_mla=program->prepare_inputs(mla_args);
        hidden_writer=program->prepare_source_reuse(submitted_mla,mla_args[0]);
        auto submitted_kda=recurrent->prepare_inputs(args,&hidden_writer);
        recurrent->submit(submitted_kda,nullptr,true);program->submit(submitted_mla,&submitted_kda);
        require(!submitted_kda.observed,"producer was observed before Source consumer submission");
        auto actual_mla=program->finish(submitted_mla);verify_reset(*program);
        auto actual_kda=recurrent->finish(submitted_kda);verify_reset(*recurrent);
        compare(actual_kda,reference::kda(*source_kda,activation,values<float>(args[1]),values<float>(args[2])),false);
        const auto history=values<float>(actual_kda[0]);const auto current=std::span(history).last(std::size_t(component::kChannels));
        constexpr std::array projections{original::Role::kKdaQ,original::Role::kKdaK,original::Role::kKdaV};
        for(std::size_t part=0;part<projections.size();++part)for(const auto row:std::array<std::size_t,3>{0,reference::width/2,reference::width-1})
          certify(current[part*reference::width+row],reference::original_row(*source_kda,projections[part],row,activation));
        const auto kda_gated=values<float>(actual_kda[4]),kda_output=values<float>(actual_kda[5]);
        for(const auto row:std::array<std::size_t,3>{0,reference::hidden/2,reference::hidden-1})
          certify(kda_output[row],reference::original_row(*source_kda,original::Role::kKdaOutput,row,kda_gated));
        compare(actual_mla,reference::mla(*source_mla,kda_output,values<float>(mla_args[1]),std::size_t(component::kCapacity),std::size_t(position)),true,std::size_t(position+1));
        require(values<std::int32_t>(actual_mla[9])==std::vector<std::int32_t>{position+1},"actual cursor successor");
        const auto query=values<float>(actual_mla[1]),absorbed=values<float>(actual_mla[2]);
        const auto attended=values<float>(actual_mla[3]),projected=values<float>(actual_mla[4]);
        for(const auto h:std::array<std::size_t,3>{0,reference::heads/2,reference::heads-1}) {
          for(const auto row:std::array<std::size_t,3>{0,256,511})certify(absorbed[h*512+row],reference::original_absorbed_row(*source_mla,h,row,std::span(query).subspan(h*192,128),true));
          for(const auto row:std::array<std::size_t,3>{0,64,127})certify(projected[h*128+row],reference::original_absorbed_row(*source_mla,h,row,std::span(attended).subspan(h*512,512),false));
        }
        const auto normalized=values<float>(actual_mla[7]),gate=values<float>(actual_mla[8]),gated=values<float>(actual_mla[5]),output=values<float>(actual_mla[6]);
        for(const auto row:std::array<std::size_t,3>{0,6144,12287})certify(gate[row],reference::original_row(*source_mla,original::Role::kMlaGate,row,normalized));
        for(const auto row:std::array<std::size_t,3>{0,reference::hidden/2,reference::hidden-1})certify(output[row],reference::original_row(*source_mla,original::Role::kMlaOutput,row,gated));
        const Epoch epoch{std::uint64_t(position+1),position+1,0};
        const auto candidate=state.candidate(before,actual_kda,actual_mla,epoch);const auto actual_epoch=state.epoch(candidate);
        require(actual_epoch.step==epoch.step&&actual_epoch.position==epoch.position&&actual_epoch.session==epoch.session,"complete candidate Epoch");
        check(head.publish(before.identity(),candidate),"complete publication");++committed;
        require(!head.publish(before.identity(),candidate).ok(),"stale publication accepted");++stale;
        require(snapshot(state,before)==old,"old complete state changed");if(position==0)first=candidate;
        if(argc==3) {save(argv[2],position,"kda",actual_kda);save(argv[2],position,"mla",actual_mla);}
      }
      const auto retained=snapshot(state,first);auto fork=take(c::StateHead::create(state.store,first),"old fork");
      const auto current=take(head.current(),"continued Head");auto abandoned=take(state.store.begin(first),"unpublished candidate");
      check(abandoned.replace(state.entries[kFeature],state.member(current,kFeature)),"unpublished feature");check(abandoned.cancel(),"cancel candidate");++cancelled;
      require(take(head.current(),"Head after cancel").identity()==current.identity()&&take(fork.current(),"fork after cancel").identity()==first.identity(),"cancel changed Heads");
      require(snapshot(state,first)==retained&&snapshot(state,initial)==old_initial,"old readers changed");
    }
    check(state.store.collect(),"state collection");const auto stats=state.store.stats();
    require(stats.versions==0&&stats.drafts==0&&stats.backing_bytes==0,"StateStore did not release complete candidates");
    require(committed==2&&stale==2&&cancelled==1&&refused==1,"finite transaction counters");
    require(recurrent->submissions==2&&attention[0]->submissions==1&&attention[1]->submissions==1,"native submission counters");
    require(attention[0]->source_submissions_before_host_observation+attention[1]->source_submissions_before_host_observation==2,"Source ordering counters");
    require(attention[0]->cold_capture_preparations+attention[1]->cold_capture_preparations==2&&
        attention[0]->recorded_capture_ranges+attention[1]->recorded_capture_ranges==0&&attention[0]->host_waits+attention[1]->host_waits==0,"consumer cold reuse counters");
    std::cout<<"{\"result\":\"pass\",\"component\":\"KDA0-to-NoPE-MLA3\",\"full_model\":false,\"steps\":2,\"committed\":"<<committed
        <<",\"stale_refused\":"<<stale<<",\"cancelled\":"<<cancelled<<",\"bind_refused\":"<<refused<<",\"state_versions\":"<<stats.versions
        <<",\"source_before_host_observation\":2,\"consumer_recorded_ranges\":0,\"consumer_host_waits\":0}\n";
    return 0;
  } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
