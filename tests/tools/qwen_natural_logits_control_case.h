#pragma once

// Finite controls compose the original request, proposal, Verify and retirement
// owners. Singleton constraints are real sampling inputs, never selectors.
namespace qwen_natural_logits_control_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace k = qwen_conditional_logits_case;
namespace g = qwen_mtp_group_return_case;
namespace s = qwen_mtp_batch_state_case;
namespace n = qwen_mtp_natural_relay_case;
namespace p = qwen_packed_prefill_case;
namespace t = qwen_mtp_stop_case;
namespace e = qwen_logits_preflight_case;
namespace c = qwen_history_control_case;
namespace l = qwen_natural_logits_case;
using h::require;

inline void cold(q::Model &model) {
  const std::array three{std::size_t{3}},two{std::size_t{2}};
  for (const auto width : {std::size_t{1},std::size_t{2}})
    for (const auto role : {q::GraphRole::kMtpPrefill,q::GraphRole::kVerify}) {
      model.prepare_family(three,false,width,role); model.plan_replay_uses(3,width,6,role);
    }
  model.prepare_family(two,false,1,q::GraphRole::kVerify); model.plan_replay_uses(2,1,1,q::GraphRole::kVerify);
  model.prepare_recursive_round_programs(0,false); model.prepare_recursive_round_programs(0,true);
  model.freeze_shape_plan();
}

inline void q_rows(const q::PreparedGenerationCall &call,std::size_t count) {
  require(call.draft_log_probabilities.size() == count &&
      std::ranges::all_of(call.draft_log_probabilities,[](const auto &lease) { return lease.valid() && lease.size() == 64*sizeof(float); }),
      "conditional controls lost their actual normalized stochastic q leases");
}

// An independent exact R1 consumer uses the same checkpoint and explicitly
// frozen policy rows. It does not invoke the request supplier or select output.
inline q::MtpRoundResult control_round(q::Model &model,const std::shared_ptr<q::Session> &session,
    const q::LogitsInputs &shared,std::span<const q::LogitsInputs> rows,std::size_t &submissions) {
  const auto cursor = session->head->sampling; const auto pending = session->head->next_id;
  std::vector<q::c::ReadLease> probabilities;
  const auto ids = model.propose_recursive(*session,2,q::model_eos_tokens,&probabilities,nullptr,shared,rows);
  require(ids.size() == 3 && probabilities.size() == 2,"conditional R1 control did not run both actual q rows");
  q::GenerationLedger ledger(1,3); ledger.bind_stop_policy({},model.vocabulary);
  q::prepare_generation_record(ledger,*session,q::GraphRole::kVerify,std::span(&pending,1),true,3,0,{},2);
  const i::ConditionalLogitsInputs conditional{rows};
  const std::array inputs{q::EvaluationRequest{session,ids,&ledger.stop_policy(),shared,conditional}};
  model.prepare_evaluations(inputs,nullptr,q::GraphRole::kVerify); model.submit_evaluations(); ++submissions;
  const std::array outputs{q::evaluation_output(model,session,{.name = "conditional-control-R1",.trace_states = false})};
  q::GenerationPublicationBatch publication({{&ledger,nullptr}});
  const std::array selections{q::MtpGenerationSelection{session,pending,3,probabilities,2}};
  std::array<q::MtpRoundResult,1> result;
  model.finish_recursive_mtp(selections,q::model_eos_tokens,outputs,result,[](auto,const auto &) {},publication);
  require(ledger.committed_count == 1 && session->head->sampling == cursor.advanced(5),
      "conditional R1 control changed the original reserved draw interval");
  probabilities.clear(); t::idle(model); return std::move(result[0]);
}

inline void stop(q::Model &model,std::size_t &submissions) {
  const auto first = k::row(model,{8},-.25F),ended = k::row(model,{9},.25F);
  const std::array<std::int32_t,3> prompt{1,2,3}; const std::array<std::int32_t,1> stop_ids{9};
  std::vector<q::GenerationRequest> owners; owners.reserve(1);
  owners.emplace_back("conditional-stop",prompt,std::filesystem::path{},4,0,i::RequestStopPolicy(stop_ids));
  auto &owner = owners[0]; t::prepare_request(owner,16); owner.progress.initial_sampling = {171,301,0};
  std::size_t calls = 0;
  owner.logits_provider = [&](std::span<const std::int32_t> outputs,std::span<const std::int32_t> future) -> hyle::Result<q::LogitsInputs> {
    ++calls;
    require(future.empty(),"terminated conditional grammar was queried with its absorbing stop hypothesis");
    if (outputs.empty()) return first.inputs;
    require(std::ranges::equal(outputs,std::array<std::int32_t,1>{8}),"stop supplier received unpublished or terminal output");
    return ended.inputs;
  };
  i::RequestWindow window; window.prepare(1,1); owner.request_key = window.enqueue(0); const auto key = owner.request_key; window.admitted(key);
  std::array prefill{q::prepare_generation_call(model,owner,4,nullptr,owners,4)};
  q::execute_generation_calls(model,prefill,false,nullptr,owners,[&] { ++submissions; });
  require(calls == 1 && owner.progress.generated == std::vector<std::int32_t>{8} &&
      owner.session->head->sampling == q::cache::SamplingCursor{171,301,1},"stop prefill changed its actual constrained output/RNG");
  auto checkpoint = owner.session->checkpoint(); const auto old = s::image(model,*checkpoint);
  const auto epoch = owner.session->head.epoch();
  std::array ending{q::prepare_generation_call(model,owner,4,nullptr,owners,3)};
  q::execute_generation_calls(model,ending,false,nullptr,owners,[&] {
    ++submissions; const auto &step = *model.active_step;
    require(step.submitted && step.members.size() == 1 && step.input_tokens == 2 &&
        step.phase->family->role == q::GraphRole::kVerify && ending[0].verification_ids == std::vector<std::int32_t>({8,9}),
        "stop supplier did not reach an actual first-draft-shortened original Verify2");
    q_rows(ending[0],1); const auto rows = ending[0].conditional_logits.rows();
    require(calls == 2 && rows.size() == 2 && g::identity(rows[0].bias.lease) == g::identity(rows[1].bias.lease) &&
        g::identity(rows[0].allowed.lease) == g::identity(rows[1].allowed.lease),
        "absorbing unreachable bonus did not retain its last live immutable policy row");
  });
  const auto &record = owner.ledger->records[1];
  require(calls == 2 && owner.ledger->committed_count == 2 && owner.progress.observed_commits() == 2 &&
      owner.progress.generated == std::vector<std::int32_t>({8,9}) && owner.progress.turn_done && record.stopped &&
      record.role == q::GraphRole::kVerify && record.verified_rows == 2 && record.emitted_count == 1 && record.emitted[0] == 9 &&
      record.before.sampling == old.sampling && record.after.sampling == old.sampling.advanced(3) &&
      record.after == q::generation_boundary(*owner.session->head) && owner.session->head.epoch() == epoch+1 &&
      owner.session->length() == old.length+1 && owner.session->head->draft->lag == 0 &&
      owner.session->head->draft->frontier == owner.session->length() && s::image(model,*checkpoint) == old &&
      ending[0].draft_log_probabilities.empty() && ending[0].conditional_logits.empty(),
      "stop publication changed its actual accepted boundary, original checkpoint, RNG or retired q ownership");
  const auto drafted = model.draft_program_graph_calls, terminals = model.prefix_decision_graph_calls;
  bool refused = false;
  try { (void)q::prepare_generation_call(model,owner,4,nullptr,owners,3); }
  catch (const std::runtime_error &) { refused = true; }
  require(refused && calls == 2 && model.draft_program_graph_calls == drafted && model.prefix_decision_graph_calls == terminals &&
      owner.request_key == key && window.matches(key),"ended request queried its supplier or executed another proposal during prepare");
  window.retire_active(key); h::retire(owner.session); window.release(key); checkpoint.reset();
  require(!model.snapshot_versions.live,"stop journey retained an immutable checkpoint or committed version"); t::idle(model);
}

inline std::vector<p::StepReport> mixed(q::Model &model,const std::shared_ptr<i::HostStateStorage> &storage,
    std::size_t &submissions) {
  const auto shared = k::row(model,{12},.375F); const auto values = l::payloads(model,8);
  for (std::size_t at = values->size(); at != 10; ++at)
    values->push_back(k::row(model,{8+static_cast<std::int32_t>(at)},.125F*static_cast<float>(at)-.25F));
  const auto trace = std::make_shared<l::Trace>();
  const std::array<std::array<std::int32_t,3>,2> prompts{{{1,2,3},{3,2,1}}};
  std::vector<q::GenerationRequest> owners; owners.reserve(2); i::RequestWindow window; window.prepare(2,2);
  for (std::size_t member = 0; member != 2; ++member) {
    owners.emplace_back(member ? "conditional-dynamic" : "conditional-static",prompts[member],std::filesystem::path{},10);
    auto &owner = owners.back(); t::prepare_request(owner,16); owner.progress.initial_sampling = {171,401+member,0};
    if (member) owner.logits_provider = l::provider(trace,values); else owner.logits = shared.inputs;
    owner.request_key = window.enqueue(member); window.admitted(owner.request_key);
  }
  const std::array keys{owners[0].request_key,owners[1].request_key};
  const auto prepare = [&](std::size_t member) { return q::prepare_generation_call(model,owners[member],10,nullptr,owners,3); };
  const auto observe = [&](const q::GenerationBatchFacts &facts) {
    for (const auto &call : facts.members) if (call.request == &owners[1] && call.disposition == q::EvaluationDisposition::kPublished)
      l::audit(*trace,owners[1],owners[1].ledger->records[owners[1].ledger->committed_count-1]);
  };
  {
    std::array prefills{prepare(0),prepare(1)};
    q::execute_generation_calls(model,prefills,false,nullptr,owners,[&] {
      ++submissions; require(model.active_step->submitted && model.active_step->members.size() == 2 &&
          model.active_step->phase->family->role == q::GraphRole::kMtpPrefill,"mixed policies split their original R2 prefill");
    },nullptr,observe);
  }
  for (std::size_t member = 0; member != 2; ++member)
    require(owners[member].ledger->committed_count == 1 && owners[member].progress.generated == std::vector<std::int32_t>{member ? 8 : 12} &&
        owners[member].session->head->sampling == q::cache::SamplingCursor{171,401+member,1},
        "mixed prefill changed an independent request's actual seed, stream, draw or output");
  std::array<q::cache::SnapshotRef,2> initial{owners[0].session->checkpoint(),owners[1].session->checkpoint()};
  const std::array old{s::image(model,*initial[0]),s::image(model,*initial[1])};
  std::array<std::shared_ptr<q::Session>,2> controls;
  std::array<q::MtpRoundResult,2> expected;
  const std::array<q::LogitsInputs,3> rows{values->at(1).inputs,values->at(2).inputs,values->at(3).inputs};
  const std::array three{std::size_t{3}};
  for (std::size_t member = 0; member != 2; ++member) {
    controls[member] = model.create_session(three,initial[member],q::GraphRole::kVerify);
    expected[member] = control_round(model,controls[member],member ? q::LogitsInputs{} : shared.inputs,
        member ? std::span<const q::LogitsInputs>(rows) : std::span<const q::LogitsInputs>{},submissions);
  }
  std::vector<p::StepReport> checks;
  {
    std::array calls{prepare(0),prepare(1)};
    q::execute_generation_calls(model,calls,false,nullptr,owners,[&] {
      ++submissions; const auto &step = *model.active_step;
      require(step.submitted && step.members.size() == 2 && step.input_tokens == 3 && step.active_input_rows == 6 &&
          step.phase->family->requests == 2 && step.phase->family->role == q::GraphRole::kVerify,
          "dynamic/static policies did not share one actual R2 Verify3 episode");
      for (std::size_t member = 0; member != 2; ++member) {
        require(step.members[member].session == owners[member].session && calls[member].verification_ids.size() == 3,
            "mixed Verify lost an original member's Session or proposal prefix"); q_rows(calls[member],2);
      }
      require(step.members[0].conditional_logits.empty() && step.members[1].conditional_logits.rows().size() == 3 &&
          calls[0].conditional_logits.empty() && calls[1].conditional_logits.rows().size() == 3,
          "mixed episode converted the static member or dropped the dynamic member's frozen rows");
      for (std::size_t row = 0; row != 3; ++row) {
        const auto &actual = step.members[1].conditional_logits.rows()[row];
        require(g::identity(actual.bias.lease) == g::identity(rows[row].bias.lease) &&
            g::identity(actual.allowed.lease) == g::identity(rows[row].allowed.lease),
            "mixed target did not reuse its actual private q/bonus policy snapshots");
      }
    },nullptr,observe);
    for (const auto &call : calls) require(call.draft_log_probabilities.empty() && call.conditional_logits.empty(),
        "successful mixed episode retained completed q or conditional owner");
  }
  for (std::size_t member = 0; member != 2; ++member) {
    const auto &record = owners[member].ledger->records[1];
    require(record.requests == 2 && record.member_index == member && record.role == q::GraphRole::kVerify &&
        record.verified_rows == 3 && record.before.sampling == old[member].sampling &&
        record.after.sampling == old[member].sampling.advanced(5) && record.after == q::generation_boundary(*owners[member].session->head) &&
        owners[member].request_key == keys[member] && window.matches(keys[member]),
        "mixed R2 commit changed its original identity, actual draw interval or complete boundary");
    h::same_round(expected[member],owners[member].rounds.back());
    auto check = n::compare_state(model,s::image(model,*controls[member]->head),s::image(model,*owners[member].session->head),3);
    check.sequence = member; check.prefix = old[member].length; checks.push_back(std::move(check));
    require(s::image(model,*initial[member]) == old[member],"mixed R2 mutated an immutable prefill checkpoint");
    h::retire(controls[member]); initial[member].reset();
  }
  std::array before{e::save(model,owners[0],storage),e::save(model,owners[1],storage)};
  const std::array before_images{s::image(model,*before[0].checkpoint),s::image(model,*before[1].checkpoint)};
  const std::array<q::LogitsInputs,3> cancel_rows{values->at(4).inputs,values->at(5).inputs,values->at(6).inputs};
  const auto cancel_control = [&] {
    auto control = model.create_session(three,before[1].checkpoint,q::GraphRole::kVerify);
    auto result = control_round(model,control,{},std::span<const q::LogitsInputs>(cancel_rows),submissions);
    auto image = s::image(model,*control->head); h::retire(control);
    return std::pair{std::move(result),std::move(image)};
  }();
  const auto provider_begin = trace->calls.size();
  std::size_t decisions = 0,repairs = 0;
  {
    std::array calls{prepare(0),prepare(1)}; bool submitted = false;
    q::execute_generation_calls(model,calls,false,nullptr,owners,[&] {
      ++submissions; const auto &step = *model.active_step;
      require(step.submitted && step.members.size() == 2 && step.input_tokens == 3 && step.active_input_rows == 6 &&
          step.phase->family->role == q::GraphRole::kVerify && !step.discard_stopped_members,
          "cancel did not reach its original shared R2 recursive episode");
      for (const auto &call : calls) q_rows(call,2);
      require(calls[1].conditional_logits.rows().size() == 3 && trace->calls.size() == provider_begin+3,
          "cancelled dynamic member did not finish its actual bounded q/bonus supplier prefix");
      for (std::size_t row = 0; row != 3; ++row) {
        const auto &fact = trace->calls[provider_begin+row];
        require(std::ranges::equal(fact.outputs,before[1].generated) &&
            std::ranges::equal(fact.hypotheses,std::span<const std::int32_t>(calls[1].verification_ids).subspan(1,row)),
            "cancelled supplier leaked pending or uncommitted output into its immutable current-turn prefix");
      }
      decisions = model.prefix_decision_graph_calls; repairs = model.draft_program_graph_calls;
      submitted = true; (void)owners[0].session->stop();
      require(!model.active_step->cancelled && model.active_step->discard_stopped_members &&
          model.active_step->execution.has_completion() && model.active_step->members.size() == 2,
          "conditional member stop cancelled the surviving member's original shared verifier");
      for (const auto &call : calls) q_rows(call,2);
      require(calls[1].conditional_logits.rows().size() == 3,
          "member stop released its policy snapshots before the original complete cohort retired");
    },nullptr,observe);
    require(submitted,"conditional cancellation never submitted its target episode");
    for (std::size_t member = 0; member != calls.size(); ++member)
      require(calls[member].disposition == (member ? q::EvaluationDisposition::kPublished : q::EvaluationDisposition::kCancelled) &&
          calls[member].draft_log_probabilities.empty() && calls[member].conditional_logits.empty(),
          "member cancellation lost the original survivor publication or retained retired q/conditional authority");
  }
  t::idle(model);
  require(model.prefix_decision_graph_calls == decisions+1 && model.draft_program_graph_calls == repairs+1,
      "cancelled conditional K2 member executed acceptance sampling or draft repair");
  c::unchanged_committed(model,owners[0],before[0],storage);
  require(s::image(model,*owners[0].session->head) == before_images[0],
      "cancelled member changed complete target/GDN/conv/draft/hidden state");
  const auto &published = owners[1].ledger->records.at(before[1].commits);
  require(published.committed && published.requests == 2 && published.member_index == 1 && published.verified_rows == 3 &&
      published.before.sampling == before_images[1].sampling &&
      published.after.sampling == before_images[1].sampling.advanced(5) &&
      published.after == q::generation_boundary(*owners[1].session->head) &&
      owners[1].session->head.epoch() == before[1].epoch+1 &&
      owners[1].ledger->committed_count == before[1].commits+1 &&
      owners[1].progress.observed_commits() == before[1].observed+1 &&
      owners[1].progress.generated == std::vector<std::int32_t>({8,9,10,11,12,13,14}) &&
      trace->audited == trace->calls.size(),
      "conditional survivor did not publish its original R2 member and independent RNG exactly once");
  for (std::size_t at = 0; at != before[1].commits; ++at)
    require(e::same_record(owners[1].ledger->records[at],before[1].records[at]),
        "conditional survivor changed an older committed record");
  h::same_round(cancel_control.first,owners[1].rounds.back());
  auto cancelled_check = n::compare_state(model,cancel_control.second,s::image(model,*owners[1].session->head),3);
  cancelled_check.sequence = 2; cancelled_check.prefix = before_images[1].length; checks.push_back(std::move(cancelled_check));
  const auto published_image = s::image(model,*owners[1].session->head);
  const auto published_generated = owners[1].progress.generated;
  const auto published_commits = owners[1].ledger->committed_count,published_observed = owners[1].progress.observed_commits();
  for (std::size_t member = 0; member != 2; ++member) {
    require(s::image(model,*before[member].checkpoint) == before_images[member],
        "member cancellation or peer publication changed an immutable complete checkpoint"); before[member].checkpoint.reset();
  }
  const auto stopped = owners[0].session->try_retire();
  require(stopped.retired && (stopped.status.ok() || (stopped.status.IsCancelled() && !stopped.status.has_native_error())),
      "stopped static member did not actually retire its original shared work");
  owners[0].session.reset(); window.retire_active(keys[0]); window.release(keys[0]);
  auto &survivor = owners[1];
  require(!survivor.session->stopping && !survivor.session->failed && survivor.session->retirement_status.ok() &&
      survivor.request_key == keys[1] && window.matches(keys[1]),"cancelled peer poisoned the dynamic survivor's original key or retirement");
  const auto calls_before_park = trace->calls.size();
  auto image = model.export_session(*survivor.session,storage); const auto parked = h::read_image(*image);
  const auto boundary = q::generation_boundary(*survivor.session->head);
  survivor.last_idle = window.pause(keys[1]); h::retire(survivor.session);
  require(!model.snapshot_versions.live && image->ready(),"idle conditional park retained a resident version");
  survivor.session = model.restore_session(image); image.reset();
  require(q::generation_boundary(*survivor.session->head) == boundary && survivor.request_key == keys[1] &&
      window.matches(keys[1]) && survivor.progress.generated == published_generated &&
      survivor.ledger->committed_count == published_commits && survivor.progress.observed_commits() == published_observed &&
      trace->calls.size() == calls_before_park && s::image(model,*survivor.session->head) == published_image,
      "HostImage restore queried policy, resampled pending/RNG or changed the original request/state");
  h::same_image(parked,p::capture(model,*survivor.session,storage));
  window.wake(keys[1]); window.admitted(keys[1]);
  auto restored_checkpoint = survivor.session->checkpoint(); const auto restored_old = s::image(model,*restored_checkpoint);
  auto control = model.create_session(three,restored_checkpoint,q::GraphRole::kVerify);
  const std::array<q::LogitsInputs,3> next_rows{values->at(7).inputs,values->at(8).inputs,values->at(9).inputs};
  const auto expected_next = control_round(model,control,{},std::span<const q::LogitsInputs>(next_rows),submissions);
  const auto epoch = survivor.session->head.epoch();
  std::array next{prepare(1)};
  q::execute_generation_calls(model,next,false,nullptr,owners,[&] {
    ++submissions; require(model.active_step->submitted && model.active_step->members.size() == 1 &&
        model.active_step->phase->family->role == q::GraphRole::kVerify && model.active_step->input_tokens == 3,
        "restored conditional survivor did not begin its next R1 round after the original R2 publication"); q_rows(next[0],2);
  },nullptr,observe);
  h::same_round(expected_next,survivor.rounds.back());
  auto checked = n::compare_state(model,s::image(model,*control->head),s::image(model,*survivor.session->head),3);
  checked.sequence = 3; checked.prefix = restored_old.length; checks.push_back(std::move(checked));
  require(survivor.request_key == keys[1] && window.matches(keys[1]) && survivor.session->head.epoch() == epoch+1 &&
      survivor.session->head->sampling == boundary.sampling.advanced(5) &&
      survivor.progress.generated == std::vector<std::int32_t>({8,9,10,11,12,13,14,15,16,17}) &&
      survivor.ledger->committed_count == published_commits+1 && survivor.progress.observed_commits() == published_observed+1 &&
      next[0].draft_log_probabilities.empty() && next[0].conditional_logits.empty() && trace->audited == trace->calls.size() &&
      s::image(model,*restored_checkpoint) == restored_old,"same-key restored survivor changed its real output/commit/RNG or retained a q/row owner");
  window.retire_active(keys[1]); h::retire(survivor.session); window.release(keys[1]); h::retire(control); restored_checkpoint.reset();
  require(window.active().empty() && !window.waiting_front() && !storage->live_images(),
      "conditional controls retained a logical key, active request or host image"); t::idle(model); return checks;
}

inline int run(const std::filesystem::path &input,q::BackendKind kind) {
  auto source = q::prepare_weight_source(input/"weights.tsv",true); const auto geometry = source.roles.geometry;
  require((kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda) && geometry.hidden == 64 && geometry.vocabulary == 64 &&
      geometry.context_limit >= 16 && geometry.mtp_layers == 1 && geometry.recurrent_layers() && geometry.attention_layers(),
      "conditional controls require cached untrained GA/H64/V64/C16 CPU/CUDA weights");
  const auto plan = q::cache::plan(geometry,source.cache_row_bytes,16,8,3,
      q::recurrent_bytes(q::state_schema_for(geometry,16),geometry),2,3,true,true);
  q::Model model(std::move(source),geometry,16,plan,{},false,q::ReplayPolicy::kCommands,SIZE_MAX,
      {.enabled = true,.policy = {.8F,0,1.F},.seed = 123,.modifiers = {.bias = true,.allowed = true}},0,{.primary = kind});
  model.configure_draft_policy({.max_drafts = 2}); cold(model);
  const auto storage = q::take(i::HostStateStorage::create(model.execution,1,4u << 20));
  std::size_t submissions = 0; stop(model,submissions); const auto checks = mixed(model,storage,submissions);
  require(submissions == 10 && checks.size() == 4 && !model.snapshot_versions.live && !storage->live_images() &&
      model.page_inventory.free_ids.size() == plan.max_pages && model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
      "conditional controls retained a state, initialized page, native episode or host image"); t::idle(model);
  std::cout << "{\"status\":\"PASS\",\"case\":\"natural-logits-control\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"configurations\":1,\"original_GenerationRequest\":true,\"natural_RequestLoop\":false,"
         "\"extra_stop_first_draft\":true,\"terminated_hypothesis_callbacks\":0,\"absorbing_bonus_reuses_last_row\":true,"
         "\"ended_prepare_provider_calls\":0,\"stop_provider_calls\":2,\"stop_verified_rows\":2,\"stop_reserved_draws\":3,"
         "\"mixed_R2_dynamic_static\":true,\"fully_published_R2_Verify_submissions\":1,\"member_cancel_R2_Verify_submissions\":1,"
         "\"survivor_same_R2_published\":true,\"survivor_original_member_index\":1,\"postrestore_next_R1_Verify_submissions\":1,"
         "\"cancelled_member_postverify_auxiliary_calls\":0,\"real_q_leases_per_R2\":4,\"q_conditional_owners_cleared\":true,"
         "\"whole_episode_retired\":true,\"cancelled_member_head_ledger_RNG_unchanged\":true,\"independent_request_RNG\":true,"
         "\"HostImage_same_key_restore\":true,\"parked_resident_versions\":0,\"old_checkpoints_unchanged\":true,"
         "\"target_submissions\":" << submissions << ",\"synthetic_untrained\":true,\"quality_claim\":false,\"performance_claim\":false,"
         "\"final_versions\":0,\"state_checks\":[";
  for (std::size_t at = 0; at != checks.size(); ++at) { if (at) std::cout << ','; p::write_step(checks[at]); }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen_natural_logits_control_case
