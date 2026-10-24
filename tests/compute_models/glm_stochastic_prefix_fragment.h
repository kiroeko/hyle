#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "hyle/compute/state_versions.h"
#include "model_sampling_fragment.h"
#include "sampling_fixture.h"
#include "../../engine/inference/mtp_acceptance.h"
#include "../../engine/inference/sampling_cursor.h"

// Finite GLM component adapters. Sampling and accepted-prefix mathematics stay
// in the inference helpers; model-specific target/NextN replay stays with the
// caller. This declares no executor, supplier, state schema or runtime route.
namespace glm_stochastic_prefix_fragment {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
using Cursor = hyle::inference::SamplingCursor;
using Value = sem::ValueId;

inline void append(sem::Fragment &fragment, std::int64_t vocabulary) {
  model_sampling_fragment::Builder r(fragment);
  const auto logits = fragment.input(sem::dense({3,vocabulary}));
  const auto identities = fragment.input(sem::dense({3,4},c::DataType::kUInt32));
  const auto proposals = fragment.input(sem::dense({2},c::DataType::kInt32));
  const auto selected = hyle::inference::sampling::speculative_prefix(r,logits,identities,proposals);
  for(std::size_t i=0;i<2;++i) {
    fragment.outputs.push_back(r.copy(selected.accepted[i]));
    fragment.outputs.push_back(r.copy(selected.final_ids[i]));
    fragment.outputs.push_back(r.copy(selected.valid[i]));
  }
}

struct Decision final {
  std::array<std::uint8_t,2> accepts{};
  std::array<std::int32_t,2> final_ids{};
  std::array<c::ReadLease,2> final_leases;
};

// Independent Philox/F64 equations choose only explicit fixture addresses.
// The real common Graph still consumes unmodified p/q and verifies the branch.
inline bool reference_accept(double log_alpha,std::array<std::uint32_t,4> key,bool cuda) {
  const auto bits=sampling_test::philox({0,0,key[2],key[3]},{key[0],key[1]});
  if(!cuda) {
    const auto u=(double(bits[0])*0x1p20+double(bits[1]>>12)+.5)*0x1p-52;
    return u<std::exp(log_alpha);
  }
  const auto gumbel=[](std::uint32_t hi,std::uint32_t lo) {
    const bool upper=(hi&0x80000000u)!=0;if(upper){hi=~hi;lo=~lo;}
    const auto tail=std::ldexp(double(hi),-32)+std::ldexp(double(lo)+.5,-64);
    return -std::log(upper?-std::log1p(-tail):-std::log(tail));
  };
  return log_alpha+gumbel(bits[0],bits[1])>
      std::log(-std::expm1(log_alpha))+gumbel(bits[2],bits[3]);
}
inline double log_alpha(std::span<const float> p_logits,std::span<const float> log_q,std::int32_t proposal) {
  if(p_logits.size()!=log_q.size()||proposal<0||std::size_t(proposal)>=p_logits.size())
    throw std::runtime_error("GLM prefix probability geometry differs");
  const auto maximum=*std::max_element(p_logits.begin(),p_logits.end());
  double sum=0;for(const auto value:p_logits)sum+=std::exp(double(value)-maximum);
  return std::min(0.,double(p_logits[std::size_t(proposal)])-maximum-std::log(sum)-double(log_q[std::size_t(proposal)]));
}

template<class T> T take(hyle::Result<T> result,const char *label) {
  if(!result)throw std::runtime_error(std::string(label)+": "+std::string(result.error().message()));
  return std::move(*result);
}
inline void check(hyle::Status status,const char *label) {
  if(!status.ok())throw std::runtime_error(std::string(label)+": "+std::string(status.message()));
}

// Real complete checkpoints are produced lazily by replay(count), count0/1/2.
// Independent bounded Heads reuse the same retired authentic result leases.
// Default draws are q0=0,q1=1,accept0=2,accept1=3,final=4. Controlled branch
// addresses are explicit and reserve through their largest used address+1.
// Caller admission requires quota/context >=3 before executing T3. Smaller
// requests use its existing T2/K1 (three draws) or target-only (one draw) path.
// The independent policy Heads below reuse already verified T3 leases; they
// do not authorize speculative execution beyond an original request's limit.
template<class Decide,class Replay,class Observe,class Continue>
void exercise(c::StateStore &store,const c::StateVersion &base,c::StateEntryId pending_entry,
    c::StateEntryId cursor_entry,std::int32_t current,std::array<std::int32_t,2> proposals,
    const std::array<c::ReadLease,2> &proposal_leases,const c::ReadLease &wrong_type,
    Cursor initial,std::int32_t vocabulary,bool cuda,std::array<double,2> alpha,
    hyle::inference::mtp::SamplingLimits limits,Decide decide,Replay replay,
    Observe observe,Continue continued) {
  const std::array inputs{current,proposals[0],proposals[1]};
  const auto member=[&](const c::StateVersion &version,c::StateEntryId entry) {
    return take(version.member(entry),"GLM K2 actual state member");
  };
  const auto publish=[&](const Decision &decision,std::array<std::uint64_t,3> addresses,
      hyle::inference::mtp::SamplingLimits boundary,const char *name,bool next,bool negative) {
    std::vector<std::int32_t> emitted;
    const auto selected=hyle::inference::mtp::select_sampled_prefix(inputs,decision.accepts,
        decision.final_ids,boundary.quota,boundary.eos,vocabulary,emitted);
    const auto index=selected.consumed-1;
    const auto pending=index<2&&decision.accepts[index]?proposal_leases[index]:decision.final_leases[std::min(index,std::size_t(1))];
    const auto checkpoint=replay(selected.consumed_drafts);
    const auto rng=initial.advanced(*std::max_element(addresses.begin(),addresses.end())+1);
    auto head=take(c::StateHead::create(store,base),"GLM K2 expected-base Head");
    const auto stage=[&] {
      auto draft=take(store.begin(checkpoint),"GLM K2 complete checkpoint draft");
      check(draft.replace(pending_entry,pending),"GLM K2 actual pending result lease");
      check(draft.replace_metadata(cursor_entry,std::as_bytes(std::span(&rng,1))),"GLM K2 atomic reserved draw cursor");
      return draft;
    };
    if(negative) {
      auto discarded=stage();EXPECT_FALSE(discarded.replace(pending_entry,wrong_type).ok());
      check(discarded.cancel(),"GLM K2 failure cancels all staged metadata");
      EXPECT_EQ(take(head.current(),"unchanged K2 cancelled Head").identity(),base.identity());
    }
    auto draft=stage();const auto version=take(draft.finalize(),"GLM K2 full resource/pending/RNG result");
    EXPECT_FALSE(head.publish(checkpoint.identity(),version).ok());
    EXPECT_EQ(take(head.current(),"unchanged K2 stale Head").identity(),base.identity());
    check(head.publish(base.identity(),version),"GLM K2 retry publishes same original decision without new draws");
    const auto published=take(head.current(),"GLM K2 published actual Head");
    EXPECT_EQ(published.identity(),version.identity());
    EXPECT_EQ(member(published,pending_entry).size(),proposal_leases[0].size());
    observe(published,checkpoint,selected,rng);
    std::printf("GLM K2 BOUNDARY name=%s proposal0=%d proposal1=%d accept0=%u accept1=%u consumed=%zu consumed_drafts=%zu emitted=%zu pending=%d stopped=%d accept_addresses=%llu,%llu final_address=%llu draw=%llu\n",
        name,proposals[0],proposals[1],unsigned(decision.accepts[0]),unsigned(decision.accepts[1]),
        selected.consumed,selected.consumed_drafts,emitted.size(),selected.next_id,int(selected.stopped),
        static_cast<unsigned long long>(addresses[0]),static_cast<unsigned long long>(addresses[1]),
        static_cast<unsigned long long>(addresses[2]),static_cast<unsigned long long>(rng.draw));
    if(next&&!selected.stopped)continued(head,published,selected.consumed_drafts,rng);
  };
  const std::array<std::uint64_t,3> natural_addresses{2,3,4};
  const auto natural=decide(natural_addresses);
  const bool unlimited=limits.quota==std::numeric_limits<std::size_t>::max()&&
      limits.remaining_context==std::numeric_limits<std::size_t>::max()&&limits.eos.empty();
  publish(natural,natural_addresses,limits,"natural",unlimited,true);
  check(store.collect(),"retire natural GLM K2 Head");
  if(!unlimited)return;
  // A fixture can configure EOS/quota but never rewrite p/q or force flags.
  std::array<std::uint64_t,3> accepted_addresses{};
  constexpr std::uint64_t bound=262144;
  for(std::size_t row=0;row<2;++row) {
    for(std::uint64_t address=2;address<bound;++address)
      if((row==0||address!=accepted_addresses[0])&&reference_accept(alpha[row],initial.identity(address),cuda)) {
        accepted_addresses[row]=address;break;
      }
    if(!accepted_addresses[row])throw std::runtime_error("GLM K2 finite controlled acceptance address was not found");
  }
  accepted_addresses[2]=std::max(accepted_addresses[0],accepted_addresses[1])+1;
  const auto accepted=decide(accepted_addresses);
  if(!accepted.accepts[0]||!accepted.accepts[1])
    throw std::runtime_error("GLM K2 actual decision disagrees with independent finite accepted-address fixture");
  for(std::size_t quota=1;quota<=3;++quota) {
    publish(accepted,accepted_addresses,{quota,3,{}},quota==1?"quota1":quota==2?"quota2":"all_accepted_bonus",quota==3,false);
    check(store.collect(),"retire independent GLM K2 prefix Head");
  }
  if(proposals[1]!=current&&proposals[1]!=proposals[0]) {
    const std::array eos{proposals[1]};publish(accepted,accepted_addresses,{3,3,eos},"accepted_second_EOS",false,false);
    check(store.collect(),"retire independent GLM K2 EOS Head");
  }
  ::testing::Test::RecordProperty("k2_default_draw_schedule","proposal0,proposal1,accept2,accept3,final4; successful complete Head advances5");
  ::testing::Test::RecordProperty("k2_controlled_address_bound",std::to_string(bound));
  ::testing::Test::RecordProperty("k2_scope","Actual two recursive NextN proposals, three true target rows, common stochastic acceptance, complete target-conditioned prefix0/1/2 replay, original result leases and atomic RNG. Controlled-address and EOS/quota fixtures are explicit; not natural acceptance rate or whole-model task quality.");
}
} // namespace glm_stochastic_prefix_fragment
