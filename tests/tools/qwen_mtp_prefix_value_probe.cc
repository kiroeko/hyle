// Explicit cached Q35 correctness check. No diagnostic graph ports, downloads,
// alternative kernels or cross-width bit-equality criterion are introduced.
#define main retained_mtp_session_probe_entry
#include "../../engine/models/qwen35/mtp_session_probe.cc"
#undef main

namespace {
struct PrefixValueCase final {
  HeadImage selected, continued;
  std::size_t graph_prefix_bytes = 0, kv_prefix_bytes = 0, draft_reference_bytes = 0;
};
Bytes original_result(q::Model &model, q::StepState &step, q::V port) {
  auto result = q::take(step.run->result(port));
  if (auto tensor = result.tensor(); tensor) {
    Bytes value(q::take(tensor->description().required_bytes()));
    q::check(model.execution.read(tensor->storage(), value)); return value;
  }
  return read(q::take(result.storage_lease()));
}
const Blob &named(const HeadImage &captured, std::string_view name) {
  const auto found = std::ranges::find(captured.blobs, name, &Blob::name);
  require(found != captured.blobs.end(), "missing committed image member"); return *found;
}
void exact_bytes(const Bytes &expected, const Bytes &actual, std::string_view label) {
  if (expected != actual) throw std::runtime_error("prefix-value mismatch: " + std::string(label) +
      " expected_bytes=" + std::to_string(expected.size()) + " actual_bytes=" + std::to_string(actual.size()) +
      " expected_hash=" + fingerprint(expected) + " actual_hash=" + fingerprint(actual));
}
void exact_payload(const HeadImage &expected, const HeadImage &actual) {
  require(expected.metadata == actual.metadata && expected.blobs.size() == actual.blobs.size(),
      "same-width equal-consumed-prefix metadata changed");
  for (const auto &blob : expected.blobs) exact_bytes(blob.bytes, named(actual, blob.name).bytes, blob.name);
}
PrefixValueCase check_prefix(q::Model &model, const q::cache::SnapshotRef &base,
    std::span<const std::int32_t> ids, std::size_t consumed, const HeadImage *full_reference,
    bool continue_one, q::cache::SnapshotRef *saved = nullptr) {
  require(consumed && consumed <= ids.size(), "invalid finite prefix test");
  auto session = model.create_session({}, base, q::GraphRole::kMtpSingle);
  const auto old = image(model, *base);
  const std::array requests{q::EvaluationRequest{session, ids}};
  const std::array outputs_io{q::EvaluationIO{.name = "value_prefix", .trace_states = false}};
  const std::array outputs{q::evaluation_output(model, session, outputs_io[0])};
  model.prepare_evaluations(requests, nullptr, q::GraphRole::kVerify); model.submit_evaluations();
  std::vector<Blob> expected; std::vector<Bytes> carry;
  std::array<std::int32_t, 1> pending{16};
  model.finish_selected_evaluations(outputs, pending,
      [&](auto, q::StepState &step, auto selected) {
        // Observe only ports that the unmodified verifier already publishes.
        // These values belong to this exact completed graph, not T1 or another
        // row width whose reduction arithmetic is allowed to differ.
        for (std::size_t layer = 0; layer < model.geometry.layers.size(); ++layer)
          if (model.geometry.is_gdn(layer)) {
            const auto &ports = step.phase->layers[layer].sequences[0];
            expected.push_back({"layer"+std::to_string(layer)+".conv.bf16",true,
                original_result(model,step,ports.history_prefix.at(consumed-1))});
            expected.push_back({"layer"+std::to_string(layer)+".gdn.f32",false,
                original_result(model,step,ports.recurrent_prefix.at(consumed-1))});
          }
        for (std::size_t row = 0; row < consumed; ++row)
          carry.push_back(original_result(model,step,step.phase->carry_ports[0].at(row)));
        expected.push_back({"carry.bf16",true,carry.back()});
        expected.push_back({"prior.bf16",true,consumed > 1 ? carry[consumed-2] : Bytes{}});
        step.members[0].accepted_tokens = consumed; selected[0] = 16;
      }, [&](q::StepState &step) { model.repair_recursive_draft(step); }, [&](std::size_t j, const q::CompletedEvaluationView &view) {
        q::observe_completed_evaluation(model, outputs[j], outputs_io[j], view);
      });
  PrefixValueCase result; result.selected = image(model,*session->head);
  require(session->length() == base->length+consumed && session->head->next_id == 16 &&
      session->head->sampling == base->sampling && session->head->draft->lag == (consumed==1?0u:1u),
      "selected consumed/pending/RNG/lag boundary differs");
  for (const auto &blob : expected) {
    exact_bytes(blob.bytes,named(result.selected,blob.name).bytes,blob.name);
    result.graph_prefix_bytes += blob.bytes.size();
  }
  if (full_reference) for (const auto &blob : full_reference->blobs)
    if (blob.name.starts_with("target")) {
      const auto &actual = named(result.selected,blob.name).bytes;
      require(actual.size() == session->length()*1024 && actual.size() <= blob.bytes.size(),
          "target visible rows exceed the committed original full-width peer");
      exact_bytes(Bytes(blob.bytes.begin(),blob.bytes.begin()+static_cast<std::ptrdiff_t>(actual.size())),actual,blob.name);
      result.kv_prefix_bytes += actual.size();
    }
  // An independent input assembly runs the existing cold KV-only NextN graph:
  // committed target h[p-1], never recursive approximate draft hidden. Its
  // actual row width matches repair, so this introduces no cross-width oracle.
  {
    q::DraftWork reference; reference.state = *base->draft;
    std::vector<std::int32_t> tokens; std::vector<c::ReadLease> shifted;
    std::vector<q::Tensor> retained; retained.reserve(consumed);
    if (base->draft->lag) {tokens.push_back(base->draft->catchup_token);shifted.push_back(base->prior_hidden);}
    const auto retained_rows = consumed == 1 ? 1 : consumed-1;
    for (std::size_t row=0;row<retained_rows;++row) {
      tokens.push_back(ids[row]);
      if (!row) shifted.push_back(base->carry_hidden);
      else {
        retained.emplace_back(model.execution.device(),q::dense(q::Type::kBFloat16,{1,model.hidden}));
        retained.back().write<std::byte>(carry[row-1]);
        q::check(model.execution.freeze(retained.back().buffer, retained.back().ready));
        shifted.push_back(retained.back().ready);
      }
    }
    (void)model.execute_draft(reference,tokens,shifted,false);
    require(reference.state.frontier==session->head->draft->frontier,
        "repaired draft retained another number of original input rows");
    const auto key=logical_rows(model.draft_inventory.keys[0],reference.state.key,reference.state.frontier),
        value=logical_rows(model.draft_inventory.values[0],reference.state.value,reference.state.frontier);
    exact_bytes(key,named(result.selected,"draft.key.bf16").bytes,"draft.key.bf16");
    exact_bytes(value,named(result.selected,"draft.value.bf16").bytes,"draft.value.bf16");
    result.draft_reference_bytes=key.size()+value.size();
  }
  require(same(old,image(model,*base)),"verification or draft repair mutated the immutable complete source");
  if (saved) *saved=session->head;
  if (continue_one) {
    const auto role=session->head->draft->lag?q::GraphRole::kMtpCatchupSingle:q::GraphRole::kMtpSingle;
    const std::array<std::int32_t,1> one{16};
    model.prepare_evaluation(*session,one,nullptr,role);model.submit_evaluation(*session);
    q::finish_evaluation(model, *session,"same_T1_continuation",{},false);
    result.continued=image(model,*session->head);
  }
  q::check(session->stop());q::check(session->drain());require(q::take(session->try_retire()),"prefix test Session did not retire");
  return result;
}
}
int main(int argc,char**argv) {
  try {
    if (argc!=2) throw std::runtime_error("qwen_mtp_prefix_value_probe EXISTING_Q35_PREPARED_INPUTS");
    const std::filesystem::path inputs=argv[1];auto prompt=q::read_ids(inputs/"prompt.ids");
    require(prompt.size()>=15,"prefix-value probe requires15 existing prompt IDs");prompt.resize(15);
    const auto geometry=q::probe::read_two_b_geometry(inputs);constexpr std::size_t context=64;
    const auto plan=q::cache::plan(geometry,context,4,15,q::recurrent_bytes(q::state_schema_for(geometry,context),geometry),1,5,true);
    q::Model model(inputs/"weights.tsv",geometry,context,plan,{},false,q::ReplayPolicy::kCommands,1024ULL*1024*1024);
    model.configure_draft_policy({4,0});
    const std::array prompt_shape{std::size_t{15}},single{std::size_t{1}};const std::array<std::size_t,2> wide{3,5};
    model.prepare_family(prompt_shape,false,1,q::GraphRole::kMtpPrefill);
    model.plan_replay_uses(15,1,4,q::GraphRole::kMtpPrefill);
    model.prepare_family(wide,false,1,q::GraphRole::kVerify);
    for(auto width:wide)model.plan_replay_uses(width,1,24,q::GraphRole::kVerify);
    for(std::size_t width=1;width<=5;++width){model.prepare_draft_program(width,false);if(width<=2)model.prepare_draft_program(width,true);}
    for(auto role:{q::GraphRole::kMtpSingle,q::GraphRole::kMtpCatchupSingle}){
      model.prepare_family(single,false,1,role);model.plan_replay_uses(1,1,24,role);
    }
    auto seed=model.create_session(prompt_shape,{},q::GraphRole::kMtpPrefill);
    model.prepare_evaluation(*seed,prompt,nullptr,q::GraphRole::kMtpPrefill);model.submit_evaluation(*seed);q::finish_evaluation(model, *seed,"seed",{},false);
    auto base=seed->head;const auto before=image(model,*base);
    const std::array<std::int32_t,2> eos{248044,248046};std::size_t cases=0,perturbed=0,checked=0;
    q::cache::SnapshotRef lagged;
    for(auto width:wide){
      auto ids=model.propose_recursive(*seed,width-1,eos);require(ids.size()==width,"finite fixture proposal hit EOS early");
      auto full=check_prefix(model,base,ids,width,nullptr,false,width==3?&lagged:nullptr);++cases;
      for(std::size_t consumed=1;consumed<width;++consumed){
        if(width==5&&consumed==2)continue; // finite first/middle/near-tail cases
        const auto original=check_prefix(model,base,ids,consumed,&full.selected,true);++cases;
        auto changed=ids;for(std::size_t row=consumed;row<changed.size();++row)changed[row]=static_cast<std::int32_t>((changed[row]+101+row)%model.vocabulary);
        const auto other=check_prefix(model,base,changed,consumed,&full.selected,true);++cases;++perturbed;
        exact_payload(original.selected,other.selected);exact_payload(original.continued,other.continued);
        checked+=original.graph_prefix_bytes+original.kv_prefix_bytes+original.draft_reference_bytes;
        std::cout<<"PASS width="<<width<<" consumed="<<consumed<<" same_graph_state=true visible_KV_prefix=true draft_target_hidden_reference=true suffix_independent=true same_T1_continuation=true\n";
      }
    }
    require(lagged&&lagged->draft->lag==1,"full consumed prefix did not provide the retained lag1 control");
    const std::vector<std::int32_t> lagged_ids{16,32,64,96,128};
    const auto lagged_full=check_prefix(model,lagged,lagged_ids,5,nullptr,false);++cases;
    for(const auto consumed:{std::size_t{1},std::size_t{3}}){
      const auto first=check_prefix(model,lagged,lagged_ids,consumed,&lagged_full.selected,true);++cases;
      auto changed=lagged_ids;for(std::size_t row=consumed;row<changed.size();++row)changed[row]+=1009;
      const auto second=check_prefix(model,lagged,changed,consumed,&lagged_full.selected,true);++cases;++perturbed;
      exact_payload(first.selected,second.selected);exact_payload(first.continued,second.continued);
      checked+=first.graph_prefix_bytes+first.kv_prefix_bytes+first.draft_reference_bytes;
      std::cout<<"PASS lag1_base=true width=5 consumed="<<consumed<<" same_graph_state=true visible_KV_prefix=true draft_target_hidden_reference=true suffix_independent=true same_T1_continuation=true\n";
    }
    require(same(before,image(model,*base)),"finite prefix checks mutated the old complete head");
    lagged.reset();base.reset();q::check(seed->stop());q::check(seed->drain());require(q::take(seed->try_retire()),"seed did not retire");seed.reset();
    require(!model.active_step&&!model.carrier_borrowed&&model.snapshot_versions.live==0,"prefix-value test leaked an accepted model state");
    std::cout<<"{\"prefix_cases\":"<<cases<<",\"perturbed_suffix_pairs\":"<<perturbed<<",\"exact_checked_bytes\":"<<checked<<",\"old_complete_state_unchanged\":true,\"final_live_versions\":0,\"new_graph_ports\":0,\"cross_width_bit_equality_required\":false,\"quality_claim\":false,\"performance_claim\":false}\n";
    return 0;
  }catch(const std::exception&error){std::cerr<<"MTP prefix values: "<<error.what()<<'\n';return 1;}
}
