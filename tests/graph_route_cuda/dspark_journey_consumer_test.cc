#include <gtest/gtest.h>

#include "dspark_cuda_consumer_fixture.h"
#include "dspark_head_checks.h"
#include "../compute_state/dspark_accepted_prefix_fixture.h"
#include "hyle/compute/batch_transfer.h"

#include <numeric>

namespace {
namespace f = dspark_cuda_fixture;
namespace d = dspark_complete_graph;
namespace p = dspark_accepted_prefix_fixture;
namespace s = deepseek_sparse_prefix_fixture;
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace original = dspark_original_roles;
using hyle::Status;
template<class T> using Result = hyle::Result<T>;

// Actual public byte transfer on the original CUDA Context/Queue. CPU Graph
// never receives a device view. System<->pinned is an explicit host copy;
// pinned<->device uses one of these two prepared native copy slots.
struct Transfer final {
  c::StorageOwner pinned;
  c::LeasePool host, device;
  c::CopyBatchProgram up_program, down_program;
  c::CopyBatchInvocation up, down;
  static Transfer create(c::Context &context,c::Queue &queue,const c::StorageOwner &device_owner) {
    Transfer result;
    result.pinned=f::take(c::StorageOwner::adopt(context,f::take(c::cuda::MemoryResource::create(
        context,c::cuda::MemoryKind::kPinnedHost),"original pinned resource")),"pinned allocation owner");
    result.host=f::take(c::LeasePool::create(result.pinned.ref(),8u<<20,{.max_blocks=128}),"pinned transfer permissions");
    result.device=f::take(c::LeasePool::create(device_owner.ref(),8u<<20,{.max_blocks=128}),"device transfer permissions");
    constexpr std::size_t capacity=1u<<20;
    const std::array uploads{c::CopyResourceSpec{result.pinned.ref(),capacity},c::CopyResourceSpec{device_owner.ref(),capacity}};
    const std::array downloads{c::CopyResourceSpec{device_owner.ref(),capacity},c::CopyResourceSpec{result.pinned.ref(),capacity}};
    const c::CopyBatchOptions options{.max_resources=2,.max_copies=1,.max_phases=1,.max_bytes=capacity};
    result.up_program=f::take(c::CopyBatchProgram::prepare(context,uploads,options),"fixed native upload program");
    result.down_program=f::take(c::CopyBatchProgram::prepare(context,downloads,options),"fixed native download program");
    result.up=f::take(c::CopyBatchInvocation::create(result.up_program,queue),"original upload slot");
    result.down=f::take(c::CopyBatchInvocation::create(result.down_program,queue),"original download slot");
    return result;
  }
  c::ReadLease copy(c::CopyBatchInvocation &slot,c::LeasePool &pool,const c::ReadLease &source) {
    f::check(slot.bind_source({0},source),"retain actual transfer source");
    f::check(slot.bind_mutable({1},f::take(pool.allocate(source.size()),"real transfer destination")),"bind transfer writer");
    const std::array<c::CopyPhase,1> phases{{{}}};
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,source.size()}}};
    f::check(slot.prepare_replay(phases,ranges),"seal actual copy ranges");
    c::ReadLease output;
    {
      auto submitted=f::take(slot.submit(),"native copy accepted");
      f::check(submitted.wait(),"native copy actually retired");
      output=f::take(submitted.output({0},{1},0,source.size()),"copy successful readonly output");
    }
    f::check(slot.reset(),"retire and withdraw copy slot permissions");
    return output;
  }
  c::ReadLease to_device(std::span<const std::byte> bytes) {
    auto writer=f::take(host.allocate(bytes.size()),"pinned upload writer");
    f::check(f::take(writer.view(),"pinned writable range").write(bytes),"copy actual host bytes to pinned source");
    auto source=f::take(writer.publish(),"publish actual pinned source");
    return copy(up,device,source);
  }
  c::ReadLease to_device(s::Route &route,const c::ReadLease &source) {
    return to_device(f::take(route.read<std::byte>(source),"read actual published system member"));
  }
  c::ReadLease to_system(s::Route &route,const c::ReadLease &source) {
    auto pinned_output=copy(down,host,source);
    return f::take(route.load(f::read(pinned_output)),"copy actual pinned bytes into CPU system permission");
  }
};

Status compare_features(s::Route &route,const s::Case &mode,
    std::span<const c::ReadLease> actual,std::span<const std::int32_t> ids,int position) {
  const auto expected=s::token_feature_reference(mode,ids,position);
  HYLE_RETURN_IF_ERROR(s::require(actual.size()==expected.size(),"actual ID feature port count"));
  for (std::size_t i=0;i<actual.size();++i) {
    HYLE_ASSIGN_OR_RETURN(auto bytes,route.read<std::byte>(actual[i]));
    HYLE_RETURN_IF_ERROR(s::require(bytes==expected[i],"Graph token projection differs from actual-ID affine reference"));
  }
  return Status::Ok();
}

// One-row target reference consumes the SAME actual proposed IDs/features.
// It independently advances every target member and all three main rings;
// it never seeds a later row with an oracle or unconsumed verify suffix.
Result<c::StateVersion> serial_reference(s::Route &route,p::CompleteState &state,
    const c::StateVersion &before,std::span<const c::ReadLease> features,
    std::span<const c::ReadLease> refreshed_rings,int consumed,s::GraphProgram &single_bridge) {
  auto version=before;
  const auto all_rows=std::size_t(p::kVerifyRows);
  for (int row=0;row<consumed;++row) {
    HYLE_ASSIGN_OR_RETURN(auto boundary,state.boundary(version));
    HYLE_ASSIGN_OR_RETURN(auto target,route.prepare(s::make_fragment(state.mode,1,boundary.target)));
    HYLE_ASSIGN_OR_RETURN(auto arguments,state.target(version));
    for (const auto &feature:features) {
      const auto row_bytes=feature.size()/all_rows;
      HYLE_ASSIGN_OR_RETURN(auto single,feature.subview(std::size_t(row)*row_bytes,row_bytes));
      arguments.push_back(std::move(single));
    }
    HYLE_ASSIGN_OR_RETURN(auto result,route.execute(target,arguments));
    const auto attention=state.target_members+(state.mode.indexed()?1u:0u);
    const std::array bridge_input{result[attention]};
    HYLE_ASSIGN_OR_RETURN(auto main,route.execute(single_bridge,bridge_input));
    std::vector<c::ReadLease> catchup_input{main[0]};
    if (row==0) catchup_input.insert(catchup_input.end(),refreshed_rings.begin(),refreshed_rings.end());
    else {
      HYLE_ASSIGN_OR_RETURN(auto rings,state.rings(version));
      catchup_input.insert(catchup_input.end(),rings.begin(),rings.end());
    }
    HYLE_ASSIGN_OR_RETURN(auto catchup,route.prepare(p::main_catchup(1,int(boundary.target_frontier))));
    HYLE_ASSIGN_OR_RETURN(auto caught,route.execute(catchup,catchup_input));
    HYLE_ASSIGN_OR_RETURN(auto ids,route.read<std::int32_t>(main[1]));
    result.resize(state.target_members);
    HYLE_ASSIGN_OR_RETURN(version,state.advance(version,result,caught,1,ids.at(0)));
  }
  return version;
}

Status run_mode(f::Draft &draft,Transfer &transfer,const original::Roles &roles,
                const s::Case &mode) {
  HYLE_ASSIGN_OR_RETURN(auto context,g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(m::system_resource(),256u<<20,{.max_blocks=2048}));
  s::Route route{std::move(context),std::move(queue),std::move(pool)};
  HYLE_ASSIGN_OR_RETURN(auto initial,s::prepare_inputs(route,mode));
  std::array<c::ReadLease,p::kDraftStages> rings;
  const auto &seed=draft.fragment.replays.at(0);
  for (std::size_t stage=0;stage<rings.size();++stage) {
    HYLE_ASSIGN_OR_RETURN(rings[stage],route.load(seed.inputs[d::kCache0+stage]));
  }
  HYLE_ASSIGN_OR_RETURN(auto carry,route.load(seed.inputs[d::kMain]));
  // Establish the initial caught-up invariant with a real Graph too: carry
  // denotes the last verified projected main row at T-1, already in each
  // persistent ring. The imported target state/history is a finite boundary.
  HYLE_ASSIGN_OR_RETURN(auto bootstrap,route.prepare(p::main_catchup(1,mode.base_position-1)));
  std::vector<c::ReadLease> bootstrap_input{carry};
  bootstrap_input.insert(bootstrap_input.end(),rings.begin(),rings.end());
  HYLE_ASSIGN_OR_RETURN(auto bootstrap_state,route.execute(bootstrap,bootstrap_input));
  for (std::size_t stage=0;stage<rings.size();++stage) rings[stage]=bootstrap_state[stage];
  carry=bootstrap_state[p::kDraftStages];
  HYLE_ASSIGN_OR_RETURN(auto state,p::CompleteState::create(mode));
  HYLE_ASSIGN_OR_RETURN(auto base,state.import(initial.base_values,rings,carry,3));
  HYLE_ASSIGN_OR_RETURN(auto head,c::StateHead::create(state.store,base));
  HYLE_ASSIGN_OR_RETURN(auto old,p::image(route,state,base));
  HYLE_ASSIGN_OR_RETURN(auto bridge,route.prepare(p::main_bridge(p::kVerifyRows)));
  HYLE_ASSIGN_OR_RETURN(auto single_bridge,route.prepare(p::main_bridge(1)));
  std::vector<c::ReadLease> oldest_proposal;
  std::vector<f::Bytes> oldest_proposal_bytes;
  std::vector<std::int32_t> consumed_ledger;
  int total_consumed=0, natural_matches=0;
  for (int round=0;round<2;++round) {
    HYLE_ASSIGN_OR_RETURN(auto before,head.current());
    HYLE_ASSIGN_OR_RETURN(auto boundary,state.boundary(before));
    HYLE_ASSIGN_OR_RETURN(auto previous,p::image(route,state,before));
    HYLE_RETURN_IF_ERROR(s::require(boundary.target_frontier==boundary.draft_frontier &&
        boundary.target_frontier>0,"proposal starts from caught-up complete frontier"));
    HYLE_ASSIGN_OR_RETURN(auto previous_rings,state.rings(before));
    HYLE_ASSIGN_OR_RETURN(auto previous_carry,before.member(state.entries[state.carry_entry()]));
    const auto position=std::int32_t(boundary.target_frontier-1);
    std::vector<std::int32_t> draft_positions(d::rows);
    std::iota(draft_positions.begin(),draft_positions.end(),position+1);
    std::vector<std::uint8_t> writing(p::kWindow); writing[std::size_t(position)%p::kWindow]=1;
    const std::vector<float> mask(d::rows*(p::kWindow+d::rows),0);
    std::vector<c::ReadLease> proposal_input;
    proposal_input.push_back(transfer.to_device(route,previous_carry));
    proposal_input.push_back(transfer.to_device(s::sem::bytes(std::vector<std::int32_t>{boundary.pending})));
    proposal_input.push_back(transfer.to_device(s::sem::bytes(std::vector<std::int32_t>{position})));
    proposal_input.push_back(transfer.to_device(s::sem::bytes(draft_positions)));
    proposal_input.push_back(transfer.to_device(s::sem::bytes(writing)));
    proposal_input.push_back(transfer.to_device(s::sem::bytes(mask)));
    for (const auto &ring:previous_rings) proposal_input.push_back(transfer.to_device(route,ring));
    auto proposal=draft.execute(proposal_input);
    // IDs are a genuine Graph output transferred through the original native
    // copy chain. The host only reads them for acceptance/checking policies.
    const auto actual_ids=transfer.to_system(route,proposal[d::kIds]);
    HYLE_ASSIGN_OR_RETURN(auto proposed,route.read<std::int32_t>(actual_ids));
    HYLE_RETURN_IF_ERROR(s::require(proposed.size()==p::kVerifyRows && proposed[0]==boundary.pending,
        "actual proposal includes the current pending input"));
    for (const auto id:proposed) HYLE_RETURN_IF_ERROR(s::require(id>=0 && id<d::vocabulary,"actual full-vocabulary proposal ID"));
    const auto hidden=f::decode<float>(f::read(proposal[d::kHidden]));
    const auto normalized=f::decode<float>(f::read(proposal[d::kNormalized]));
    const auto lm=d::lm_reference(normalized);
    const auto expected=original::reference(roles,lm,hidden,boundary.pending);
    HYLE_RETURN_IF_ERROR(s::require(proposed==expected.tokens,"original head greedy IDs"));
    HYLE_RETURN_IF_ERROR(s::require(f::decode<std::uint16_t>(f::read(proposal[d::kMarkov]))==expected.markov,"actual preceding ID feeds original Markov row"));
    HYLE_RETURN_IF_ERROR(s::require(dspark_head_checks::adjusted_logits_equal(
        f::decode<float>(f::read(proposal[d::kLogits])),lm,expected),"original composed logit gate"));
    HYLE_RETURN_IF_ERROR(s::require(dspark_head_checks::projection_equal(
        f::decode<float>(f::read(proposal[d::kConfidence])),expected.confidence,"journey raw confidence"),"original raw confidence gate"));
    std::array<c::ReadLease,p::kDraftStages> refreshed;
    for (std::size_t stage=0;stage<refreshed.size();++stage) {
      refreshed[stage]=transfer.to_system(route,proposal[d::kNextCache0+stage]);
      HYLE_ASSIGN_OR_RETURN(auto old_ring,route.read<std::byte>(previous_rings[stage]));
      HYLE_ASSIGN_OR_RETURN(auto new_ring,route.read<std::byte>(refreshed[stage]));
      for (int row=0;row<p::kWindow;++row) if (row!=position%p::kWindow) {
        const auto row_bytes=std::size_t(s::kDepth)*sizeof(float);
        const auto first=std::size_t(row)*row_bytes;
        HYLE_RETURN_IF_ERROR(s::require(std::equal(old_ring.begin()+std::ptrdiff_t(first),
            old_ring.begin()+std::ptrdiff_t(first+row_bytes),new_ring.begin()+std::ptrdiff_t(first)),
            "proposal temporary KV escaped into a persistent ring row"));
      }
    }
    HYLE_ASSIGN_OR_RETURN(auto feature_graph,route.prepare(s::token_features(mode,p::kVerifyRows,int(boundary.target_frontier))));
    const std::array feature_input{actual_ids};
    HYLE_ASSIGN_OR_RETURN(auto features,route.execute(feature_graph,feature_input));
    HYLE_RETURN_IF_ERROR(compare_features(route,mode,features,proposed,int(boundary.target_frontier)));
    HYLE_ASSIGN_OR_RETURN(auto verify,route.prepare(s::make_fragment(mode,p::kVerifyRows,boundary.target)));
    HYLE_ASSIGN_OR_RETURN(auto verify_input,state.target(before));
    verify_input.insert(verify_input.end(),features.begin(),features.end());
    HYLE_ASSIGN_OR_RETURN(auto verified,route.execute(verify,verify_input));
    const auto attention=std::size_t(p::kVerifyRows)*state.target_members+(mode.indexed()?1u:0u);
    const std::array bridge_input{verified[attention]};
    HYLE_ASSIGN_OR_RETURN(auto mains,route.execute(bridge,bridge_input));
    HYLE_ASSIGN_OR_RETURN(auto predicted,route.read<std::int32_t>(mains[1]));
    HYLE_RETURN_IF_ERROR(s::require(predicted.size()==p::kVerifyRows,"target produces all verify/fresh rows"));
    int matched=0;
    while (matched<p::kProposalRows && proposed[std::size_t(matched+1)]==predicted[std::size_t(matched)]) ++matched;
    const int accepted_rows=matched+1;
    const auto pending=predicted[std::size_t(accepted_rows-1)];
    HYLE_RETURN_IF_ERROR(s::require(pending>=0 && pending<17,"actual finite projected-target fresh/bonus ID"));
    const auto main_bytes=std::size_t(accepted_rows)*p::kMain*sizeof(float);
    HYLE_ASSIGN_OR_RETURN(auto accepted_main,mains[0].subview(0,main_bytes));
    HYLE_ASSIGN_OR_RETURN(auto catchup,route.prepare(p::main_catchup(accepted_rows,int(boundary.target_frontier))));
    std::vector<c::ReadLease> catchup_input{accepted_main};
    catchup_input.insert(catchup_input.end(),refreshed.begin(),refreshed.end());
    HYLE_ASSIGN_OR_RETURN(auto caught,route.execute(catchup,catchup_input));
    const auto target_prefix=std::span(verified).subspan(std::size_t(accepted_rows-1)*state.target_members,state.target_members);
    const auto ring_prefix=std::span(caught).subspan(std::size_t(accepted_rows-1)*p::kCatchupStride,p::kCatchupStride);
    HYLE_ASSIGN_OR_RETURN(auto candidate,state.advance(before,target_prefix,ring_prefix,accepted_rows,pending));
    HYLE_ASSIGN_OR_RETURN(auto reference,serial_reference(route,state,before,features,refreshed,accepted_rows,single_bridge));
    HYLE_RETURN_IF_ERROR(p::compare(route,state,candidate,reference));
    HYLE_ASSIGN_OR_RETURN(auto unpublished,head.current());
    HYLE_RETURN_IF_ERROR(s::require(unpublished.identity()==before.identity(),"partial target/ring/carry publication"));
    HYLE_RETURN_IF_ERROR(head.publish(before.identity(),candidate));
    HYLE_ASSIGN_OR_RETURN(auto published,head.current());
    HYLE_ASSIGN_OR_RETURN(auto next,state.boundary(published));
    total_consumed+=accepted_rows; natural_matches+=matched;
    HYLE_RETURN_IF_ERROR(s::require(next.target_frontier==boundary.target_frontier+accepted_rows &&
        next.target_frontier==next.draft_frontier && next.pending==pending &&
        next.accepted_rows==std::uint32_t(total_consumed),"relative complete-state continuation boundary"));
    consumed_ledger.insert(consumed_ledger.end(),proposed.begin(),proposed.begin()+accepted_rows);
    HYLE_ASSIGN_OR_RETURN(auto old_after,p::image(route,state,before));
    HYLE_RETURN_IF_ERROR(s::require(previous==old_after,"journey mutated the previous complete snapshot"));
    HYLE_ASSIGN_OR_RETURN(auto base_after,p::image(route,state,base));
    HYLE_RETURN_IF_ERROR(s::require(base_after==old,"journey mutated the initial complete snapshot"));
    if (!round) {
      oldest_proposal=proposal;
      for (const auto &value:oldest_proposal) oldest_proposal_bytes.push_back(f::read(value));
    } else for (std::size_t i=0;i<oldest_proposal.size();++i)
      HYLE_RETURN_IF_ERROR(s::require(f::read(oldest_proposal[i])==oldest_proposal_bytes[i],"second real proposal mutated its predecessor lease"));
    std::printf("DSpark actual journey ratio%d round%d matched_proposals=%d accepted_inputs=%d next_frontier=%u pending=%d fullV=true\n",
        mode.ratio,round,matched,accepted_rows,next.target_frontier,pending);
  }
  HYLE_ASSIGN_OR_RETURN(auto final,head.current());
  HYLE_ASSIGN_OR_RETURN(auto facts,state.boundary(final));
  HYLE_RETURN_IF_ERROR(s::require(consumed_ledger.size()==std::size_t(total_consumed) &&
      facts.target_frontier==std::uint32_t(mode.base_position+total_consumed),"natural token ledger and complete frontier"));
  std::printf("DSpark actual journey ratio%d two rounds: consumed=%d naturally_matched=%d; projected-target/generated-stage-LM boundary only PASS\n",
      mode.ratio,total_consumed,natural_matches);
  return Status::Ok();
}

TEST(CudaDSparkJourney, ActualProposalVerifyCatchupPublishesAndContinuesTwice) {
  const auto markov=f::environment("HYLE_DSPARK_MARKOV_DIR");
  const auto confidence=f::environment("HYLE_DSPARK_CONFIDENCE_BF16");
  if (markov.empty() || confidence.empty()) GTEST_SKIP()<<"Set authenticated original DSpark Markov/confidence paths";
  const auto roles=original::load(markov,confidence);
  auto context=f::take(g::cuda::create_context(),"one default CUDA context");
  const auto owner=f::take(c::StorageOwner::adopt(context,f::take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"original CUDA allocation domain")),"retained CUDA owner");
  auto queue=f::take(context.create_queue(),"one original CUDA queue");
  auto transfer=Transfer::create(context,queue,owner);
  auto draft=f::Draft::create(context,owner,queue,{roles.w1,roles.w2,roles.confidence});
  // Exactly one shared cold parameter preparation for both finite target
  // compression modes. Four actual full-shape proposal submissions total.
  for (const auto &mode:std::array{s::Case{4,4,1024,516,511,2047},s::kWide}) {
    const auto status=run_mode(draft,transfer,roles,mode);
    ASSERT_TRUE(status.ok())<<status.message();
    ASSERT_EQ(draft.outputs.stats().live_blocks,0u);
    ASSERT_EQ(transfer.host.stats().live_blocks,0u);
    ASSERT_EQ(transfer.device.stats().live_blocks,0u);
  }
  RecordProperty("scope","actual full Flash DSpark3stage/original W1-W2-confidence; finite untrained projected target and generated stage/LM weights; not complete original43layer model");
  RecordProperty("actual_proposal_submissions",4);
  RecordProperty("proposal_parameter_cold_graphs",1);
  RecordProperty("acceptance","natural greedy comparison only; accepted_inputs=matched+1 includes pending, all5 means6 including fresh bonus");
}
} // namespace
