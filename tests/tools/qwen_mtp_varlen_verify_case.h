#pragma once

#include "inference_packed_prefix_rows_case.h"

// Actual bounded Verify3 carrier, exact R1 controls and original NextN repair.
// Controlled prefixes establish state semantics; the final Engine episode
// uses the original sampled-prefix decision and real normalized q leases.
namespace qwen_mtp_varlen_verify_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace s = qwen_mtp_batch_state_case;
namespace p = qwen_packed_prefill_case;
namespace r = qwen_mtp_ragged_prefill_case;
namespace m = qwen_mtp_logits_case;
namespace g = qwen_mtp_group_return_case;
using h::require;

inline void cold(q::Model &model) {
  const std::array three{std::size_t{3}}, one{std::size_t{1}};
  model.prepare_family(three,false,1,q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3,1,2,q::GraphRole::kMtpPrefill);
  for (const auto rows : {std::size_t{1},std::size_t{3}}) {
    const std::array width{rows}; model.prepare_family(width,false,1,q::GraphRole::kVerify);
    model.plan_replay_uses(rows,1,4,q::GraphRole::kVerify);
  }
  model.prepare_family(three,false,2,q::GraphRole::kVerify,false,q::TextCompletion::kEmitNext,0,q::RowPacking::kVerify);
  model.plan_replay_uses(3,2,3,q::GraphRole::kVerify,q::TextCompletion::kEmitNext,0,q::RowPacking::kVerify);
  model.prepare_recursive_round_programs();
  model.prepare_prefix_sampling(0); // Real Li1 verifier: no proposal/q, one final target draw.
  model.freeze_shape_plan();
  model.prepare_execution(three,1,q::GraphRole::kMtpPrefill);
  model.prepare_execution(one,1,q::GraphRole::kVerify);
  model.prepare_execution(three,1,q::GraphRole::kVerify);
  model.prepare_execution(three,2,q::GraphRole::kVerify,q::TextCompletion::kEmitNext,0,q::RowPacking::kVerify);
}

struct Proposal final {
  std::vector<std::int32_t> ids;
  std::vector<q::c::ReadLease> probabilities;
  std::vector<g::QIdentity> identities;
};

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  inference_packed_prefix_rows_case::run();
  auto source = q::prepare_weight_source(input / "weights.tsv",true);
  const auto geometry = source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.hidden == 64 && geometry.vocabulary == 64 &&
      geometry.recurrent_layers() && geometry.attention_layers() && geometry.mtp_layers == 1,
      "varlen verifier requires the original cached C16 GA/H64/V64 MTP fixture");
  const auto plan = q::cache::plan(geometry,source.cache_row_bytes,16,6,3,
      q::recurrent_bytes(q::state_schema_for(geometry,16),geometry),2,3,true,true);
  q::Model model(std::move(source),geometry,16,plan,{},false,q::ReplayPolicy::kCommands,SIZE_MAX,
      {.enabled = true,.policy = {.8F,0,1.F},.seed = 123,.modifiers = {.bias = true,.allowed = true}},
      0,{.primary = kind});
  model.configure_draft_policy({.max_drafts = 2}); cold(model);
  const auto payload = m::payload(model,0);
  const auto storage = q::take(i::HostStateStorage::create(model.execution,1,4u << 20));
  std::array<std::shared_ptr<q::Session>,2> sessions, controls;
  std::array<q::cache::SnapshotRef,2> original;
  std::array<s::StateImage,2> original_images;
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    (void)model.abort_episode();
    for (auto *group : {&controls,&sessions}) for (auto &session : *group) if (session) {
      session->stop().IgnoreError(); (void)session->drain(); (void)session->try_retire(); session.reset();
    }
    original = {};
  }};
  std::size_t target_submissions = 0, bounded_submissions = 0;
  const std::array three{std::size_t{3}};
  const std::array<std::array<std::int32_t,3>,2> prompts{{{1,2,3},{3,2,1}}};
  for (std::size_t member = 0; member < 2; ++member) {
    sessions[member] = model.create_session(three,{},q::GraphRole::kMtpPrefill,q::cache::SamplingCursor{123,7+member,0});
    model.prepare_evaluation(*sessions[member],prompts[member],nullptr,q::GraphRole::kMtpPrefill,
        q::TextCompletion::kEmitNext,nullptr,payload.inputs);
    model.submit_evaluations(); ++target_submissions;
    (void)q::finish_evaluation(model,*sessions[member],"varlen-verify-prefill",{},false);
    original[member] = sessions[member]->checkpoint(); original_images[member] = s::image(model,*original[member]);
    controls[member] = model.create_session(three,original[member],q::GraphRole::kVerify);
  }
  const auto old_unchanged = [&] {
    for (std::size_t member = 0; member < 2; ++member)
      require(s::image(model,*original[member]) == original_images[member],
          "varlen verification mutated an immutable complete initial checkpoint");
  };
  const auto propose = [&](const std::shared_ptr<q::Session> &session, std::size_t length) {
    const auto before = s::image(model,*session->head); const auto epoch = session->head.epoch();
    Proposal proposal;
    proposal.ids = model.propose_recursive(*session,length-1,q::model_eos_tokens,&proposal.probabilities,nullptr,payload.inputs);
    require(proposal.ids.size() == length && proposal.probabilities.size()+1 == length &&
        proposal.ids.front() == before.pending && session->head.epoch() == epoch && s::image(model,*session->head) == before,
        "actual recursive proposal changed its head/RNG or lost a real verifier/q row");
    m::allowed(std::span(proposal.ids).subspan(1),payload); m::q_rows(model,proposal.probabilities,payload);
    for (const auto &lease : proposal.probabilities) proposal.identities.push_back(g::identity(lease));
    return proposal;
  };
  const auto same_q = [&](const Proposal &proposal) {
    require(proposal.probabilities.size() == proposal.identities.size(),"varlen proposal lost its original q inventory");
    for (std::size_t row = 0; row < proposal.probabilities.size(); ++row)
      require(g::identity(proposal.probabilities[row]) == proposal.identities[row],
          "varlen preparation replaced the original normalized q block/generation");
    m::q_rows(model,proposal.probabilities,payload);
  };
  const q::Phase *bounded_carrier = nullptr;
  const auto inspect = [&](const std::array<std::size_t,2> &lengths,
      const std::array<Proposal,2> &proposals, bool submitted, bool owns_q = true) {
    const auto &step = *model.active_step;
    require(step.prepared && step.submitted == submitted && step.members.size() == 2 &&
        step.input_tokens == 3 && step.active_input_rows == 4 && step.phase->query_rows == 3 &&
        step.phase->prefix_rows == 3 && step.phase->family->requests == 2 &&
        step.phase->family->role == q::GraphRole::kVerify && step.phase->family->packing == q::RowPacking::kVerify,
        "varlen verifier split members or replaced the one bounded physical Verify3 carrier");
    if (bounded_carrier) require(bounded_carrier == step.phase,"actual lengths compiled a different bounded verification carrier");
    else bounded_carrier = step.phase;
    auto &io = step.phase->io[step.state_slot];
    const auto layout = io.attention_layout.host();
    const std::array<std::int32_t,3> offsets{0,static_cast<std::int32_t>(lengths[0]),4};
    require(std::ranges::equal(layout.query_indptr,offsets) && io.attention_layout.facts().query_rows == 4,
        "varlen verification layout counted physical padding as real target input");
    std::vector<std::int32_t> tokens,positions,prefixes,history;
    const auto h = static_cast<std::size_t>(geometry.conv_kernel-1);
    for (std::size_t member = 0; member < 2; ++member) {
      const auto &part = step.members[member]; const auto begin = static_cast<std::size_t>(offsets[member]);
      require(part.session == sessions[member] && part.input_tokens == lengths[member] &&
          part.input_row_begin == begin && part.layout.facts().query_rows == lengths[member] &&
          part.layout.facts().kv_rows == part.base->length+lengths[member] &&
          part.verification_inputs == proposals[member].ids,
          "varlen member owns another sequence's tokens, prefix rows or append frontier");
      tokens.insert(tokens.end(),proposals[member].ids.begin(),proposals[member].ids.end());
      for (std::size_t row = 0; row < lengths[member]; ++row) positions.push_back(static_cast<std::int32_t>(part.base->length+row));
      for (std::size_t slot = 0; slot < 3; ++slot) {
        const auto count = (std::min)(slot+1,lengths[member]);
        prefixes.push_back(static_cast<std::int32_t>(begin+count-1));
        for (std::size_t row = 0; row < h; ++row) {
          const auto source = count+row < h ? member*h+count+row : 2*h+begin+count+row-h;
          history.push_back(static_cast<std::int32_t>(source));
        }
      }
      if (owns_q) same_q(proposals[member]);
    }
    tokens.resize(6,0); positions.resize(6,0);
    require(r::read_indices(model,io.tokens) == tokens && r::read_indices(model,io.positions) == positions &&
        r::read_indices(model,io.prefix_ids) == prefixes && io.prefix_id_values == prefixes &&
        r::read_indices(model,io.history_select) == history,
        "varlen compact tokens/positions or clamped real-prefix hidden/history selectors differ");
  };
  std::vector<p::StepReport> checks; checks.reserve(6);
  const auto compare = [&](std::size_t sequence,const std::array<std::size_t,2> &lengths) {
    for (std::size_t member = 0; member < 2; ++member) {
      p::StepReport check{sequence,original_images[member].length,lengths[member],3};
      check.epoch = sessions[member]->head.epoch(); check.exact_next = controls[member]->head->next_id;
      check.bucket_next = sessions[member]->head->next_id;
      p::compare(model,p::capture(model,*controls[member],storage),p::capture(model,*sessions[member],storage),check);
      require(check.numerically_equivalent && check.exact_next == check.bucket_next,
          "bounded varlen verifier changed complete state versus its independent exact R1 consumer");
      checks.push_back(std::move(check));
    }
    old_unchanged();
  };
  const auto controlled = [&](std::size_t sequence,const std::array<std::size_t,2> &lengths,
      const std::array<std::size_t,2> &accepted) {
    std::array<Proposal,2> proposals;
    std::array<q::cache::SnapshotRef,2> before;
    std::array<s::StateImage,2> bytes;
    std::array<std::uint64_t,2> epochs;
    std::array<std::int32_t,2> expected{-1,-1},actual{-1,-1};
    for (std::size_t member = 0; member < 2; ++member) {
      before[member] = sessions[member]->checkpoint(); bytes[member] = s::image(model,*before[member]);
      epochs[member] = sessions[member]->head.epoch(); proposals[member] = propose(sessions[member],lengths[member]);
      const auto old_control = controls[member]->checkpoint(); const auto old_bytes = s::image(model,*old_control);
      model.prepare_evaluation(*controls[member],proposals[member].ids,nullptr,q::GraphRole::kVerify,
          q::TextCompletion::kEmitNext,nullptr,payload.inputs);
      model.submit_evaluations(); ++target_submissions;
      const std::array outputs{q::EvaluationOutput{.session = controls[member]}};
      model.finish_selected_evaluations(outputs,std::span(&expected[member],1),
          [&](auto candidates,q::StepState &step,auto selected) {
            require(step.members.size() == 1 && step.members[0].input_tokens == lengths[member] &&
                candidates.size() == lengths[member],"exact R1 verifier used physical bounded rows");
            step.members[0].accepted_tokens = accepted[member]; selected[0] = candidates[accepted[member]-1];
          },[&](q::StepState &step) { model.repair_recursive_draft(step); },[](auto,const auto &) {});
      require(s::image(model,*old_control) == old_bytes,"exact R1 verification mutated its original checkpoint");
    }
    const std::array inputs{q::EvaluationRequest{sessions[0],proposals[0].ids,nullptr,payload.inputs},
        q::EvaluationRequest{sessions[1],proposals[1].ids,nullptr,payload.inputs}};
    const std::array outputs{q::EvaluationOutput{.session = sessions[0]},q::EvaluationOutput{.session = sessions[1]}};
    model.prepare_verification_evaluations(inputs,3); inspect(lengths,proposals,false);
    model.submit_evaluations(); ++target_submissions; ++bounded_submissions;
    std::array<std::size_t,2> observed{};
    model.finish_selected_evaluations(outputs,actual,[&](auto candidates,q::StepState &step,auto selected) {
      require(candidates.size() == 6,"bounded verifier did not retain its physical compact logits carrier");
      for (std::size_t member = 0; member < 2; ++member) {
        auto &part = step.members[member]; part.accepted_tokens = accepted[member];
        selected[member] = candidates[part.input_row_begin+accepted[member]-1];
      }
    },[&](q::StepState &step) { model.repair_recursive_draft(step); },[&](std::size_t member,const q::CompletedEvaluationView &view) {
      require(member < 2 && view.requests == 2 && view.member_index == member &&
          view.observation.input_tokens == lengths[member] && view.observation.graph_input_tokens == lengths[member],
          "varlen observer charged a padded row or another member's verification work"); ++observed[member];
    });
    for (std::size_t member = 0; member < 2; ++member) {
      const auto after = s::image(model,*sessions[member]->head); const auto n = accepted[member];
      require(observed[member] == 1 && actual[member] == expected[member] &&
          sessions[member]->head.epoch() == epochs[member]+1 && after.length == bytes[member].length+n &&
          after.lag == (n == 1 ? 0u : 1u) && after.frontier+after.lag == after.length && after.pending == actual[member] &&
          after.catchup == (n == 1 ? -1 : proposals[member].ids[n-1]) && after.sampling == bytes[member].sampling,
          "controlled varlen prefix changed its actual accepted boundary, repair, pending or RNG");
      r::preserved_prefix(bytes[member],after); same_q(proposals[member]);
      require(s::image(model,*before[member]) == bytes[member],"bounded prefix selection mutated the retained pre-step snapshot");
    }
    compare(sequence,lengths);
  };
  controlled(0,{1,3},{1,2});
  controlled(1,{3,1},{3,1});

  // Original real-q sampled acceptance, through the Engine batch entry. A
  // maximum-zero proposal is a supported actual pending-only verifier, not a
  // fabricated normalized q row or a mutation of the sampler's RNG address.
  const std::array<std::size_t,2> lengths{1,3};
  std::array<Proposal,2> proposals;
  std::array<q::cache::SnapshotRef,2> before;
  std::array<s::StateImage,2> bytes;
  std::array<std::uint64_t,2> epochs;
  std::array<q::MtpRoundResult,2> expected,actual;
  std::size_t ledger_storage_checks = 0;
  for (std::size_t member = 0; member < 2; ++member) {
    before[member] = sessions[member]->checkpoint(); bytes[member] = s::image(model,*before[member]);
    epochs[member] = sessions[member]->head.epoch(); proposals[member] = propose(sessions[member],lengths[member]);
    auto control = propose(controls[member],lengths[member]);
    require(control.ids == proposals[member].ids,"varlen and exact R1 controls proposed different real verification tokens");
    model.prepare_evaluation(*controls[member],control.ids,nullptr,q::GraphRole::kVerify,
        q::TextCompletion::kEmitNext,nullptr,payload.inputs);
    model.submit_evaluations(); ++target_submissions;
    const std::array outputs{q::EvaluationOutput{.session = controls[member]}};
    const std::array selections{q::MtpGenerationSelection{controls[member],control.ids.front(),3,control.probabilities,2}};
    model.finish_recursive_mtp(selections,q::model_eos_tokens,outputs,std::span(&expected[member],1),[](auto,const auto &) {});
  }
  {
    struct RecordStorage final {
      std::array<const std::int32_t *,3> data;
      std::array<std::size_t,3> capacity;
    };
    const auto storage_of = [](const q::GenerationCommitRecord &record) {
      return RecordStorage{{record.emitted.data(),record.accepted_inputs.data(),record.verification_inputs.data()},
          {record.emitted.capacity(),record.accepted_inputs.capacity(),record.verification_inputs.capacity()}};
    };
    std::array<std::array<RecordStorage,2>,2> cold_storage{};
    std::vector<q::GenerationRequest> owners; owners.reserve(2);
    const auto unchanged_storage = [&](std::size_t member) {
      const auto &ledger = *owners[member].ledger;
      for (std::size_t ordinal = 0; ordinal < ledger.records.size(); ++ordinal) {
        const auto actual_storage = storage_of(ledger.records[ordinal]);
        require(actual_storage.data == cold_storage[member][ordinal].data &&
            actual_storage.capacity == cold_storage[member][ordinal].capacity,
            "actual Engine preparation/publication discarded a cold ledger token buffer");
        ++ledger_storage_checks;
      }
    };
    for (std::size_t member = 0; member < 2; ++member) {
      owners.emplace_back("varlen-frontier"+std::to_string(member),std::span<const std::int32_t>{},std::filesystem::path{},4);
      auto &request = owners.back(); request.session = sessions[member]; request.logits = payload.inputs;
      request.ledger = std::make_unique<q::GenerationLedger>(2,8);
      for (std::size_t ordinal = 0; ordinal < request.ledger->records.size(); ++ordinal) {
        cold_storage[member][ordinal] = storage_of(request.ledger->records[ordinal]);
        for (const auto capacity : cold_storage[member][ordinal].capacity)
          require(capacity >= 8,"cold ledger did not reserve all three token buffers beyond inline capacity");
      }
      request.round_emitted.resize(3); for (auto &emitted : request.round_emitted) emitted.reserve(3);
      request.turn_observations.resize(1); request.progress.initial_sampling = bytes[member].sampling;
      (void)request.progress.begin_turn(); request.progress.apply_outputs(std::span(&bytes[member].pending,1),true,true,false);
    }
    // The same pending record may be prepared again after a withdrawn host
    // admission. Exercise original prepare/discard owners, including a shorter
    // draft bound, before the actual sampled width-one/width-three batch.
    for (std::size_t member = 0; member < 2; ++member) {
      auto &ledger = *owners[member].ledger;
      for (const auto maximum : {std::size_t{2},std::size_t{0},std::size_t{2}}) {
        const auto &pending = q::prepare_generation_record(ledger,*sessions[member],q::GraphRole::kVerify,
            std::span(&bytes[member].pending,1),true,3,0,{},maximum);
        require(pending.emitted.size() == 2 && pending.accepted_inputs.size() == 2 &&
            pending.verification_inputs.empty() && pending.emitted[0] == -1 && pending.emitted[1] == -1 &&
            pending.accepted_inputs[0] == -1 && pending.accepted_inputs[1] == -1 &&
            pending.draft_limit == maximum && !ledger.committed_count,
            "reprepared actual pending ledger retained stale token values or published a commit");
        unchanged_storage(member);
      }
      ledger.discard_uncommitted(); unchanged_storage(member);
      const auto &pending = ledger.next();
      require(pending.emitted.empty() && pending.accepted_inputs.empty() && pending.verification_inputs.empty() &&
          !pending.generation && !pending.committed && !ledger.committed_count,
          "withdrawing the prepared host record retained a previous episode's token facts");
    }
    std::array calls{q::prepare_generation_call(model,owners[0],4,nullptr,owners,3),
        q::prepare_generation_call(model,owners[1],4,nullptr,owners,3)};
    for (std::size_t member = 0; member < 2; ++member) {
      auto &call = calls[member];
      require(call.role == q::GraphRole::kVerify && call.maximum == 2,"real request did not select its bounded K2 verifier");
      call.verification_ids = proposals[member].ids;
      call.draft_log_probabilities = std::move(proposals[member].probabilities);
    }
    std::size_t submissions = 0;
    const auto drafts_before = model.draft_program_graph_calls;
    q::execute_generation_calls(model,calls,false,nullptr,owners,[&] {
      // Borrow the exact original q wrappers for the observation only; their
      // identity is checked without manufacturing or retaining a new owner.
      for (std::size_t member = 0; member < 2; ++member) {
        require(calls[member].draft_log_probabilities.size() == lengths[member]-1,"Engine lost an actual q prefix");
        for (std::size_t row = 0; row < calls[member].draft_log_probabilities.size(); ++row)
          require(g::identity(calls[member].draft_log_probabilities[row]) == proposals[member].identities[row],
              "Engine replaced a normalized q lease before its actual target submission");
        m::q_rows(model,calls[member].draft_log_probabilities,payload);
        unchanged_storage(member);
      }
      // The original Call owns q until retirement; inspect borrows metadata.
      inspect(lengths,proposals,true,false); ++submissions; ++target_submissions; ++bounded_submissions;
      require(submissions == 1,"Engine regrouped unequal verifier lengths instead of one bounded B2");
    });
    require(submissions == 1 && model.draft_program_graph_calls == drafts_before+2,
        "Engine varlen verifier split its batch or regenerated an already supplied real proposal");
    for (std::size_t member = 0; member < 2; ++member) {
      const auto &request = owners[member]; const auto &record = request.ledger->records[0];
      unchanged_storage(member);
      require(calls[member].disposition == q::EvaluationDisposition::kPublished && calls[member].draft_log_probabilities.empty() &&
          request.ledger->committed_count == 1 && request.progress.observed_commits() == 1 && request.rounds.size() == 1 &&
          record.committed && record.role == q::GraphRole::kVerify && record.requests == 2 && record.member_index == member &&
          record.verified_rows == lengths[member] && record.verification_inputs.size() == lengths[member],
          "Engine varlen publication retained q or charged physical rows to the committed member ledger");
      actual[member] = request.rounds.front(); qwen_mtp_mixed_lag_case::same_round(expected[member],actual[member]);
      require(record.accepted == actual[member].processed_target && record.accepted <= proposals[member].ids.size() &&
          std::ranges::equal(record.accepted_inputs,std::span(proposals[member].ids).first(record.accepted)) &&
          std::ranges::equal(record.verification_inputs,proposals[member].ids) &&
          record.emitted_count == actual[member].emitted.size() &&
          std::ranges::equal(record.emitted,actual[member].emitted),
          "reused ledger storage changed the actual accepted/verified/emitted token values");
      const auto after = s::image(model,*sessions[member]->head);
      const auto retained = actual[member].processed_target == 1 ? 1 : actual[member].processed_target-1;
      const auto proposal_rows = lengths[member]-1 + (lengths[member] > 1 ? bytes[member].lag : 0);
      const auto draft_rows = proposal_rows + bytes[member].lag + retained;
      require(sessions[member]->head.epoch() == epochs[member]+1 &&
          after.length == bytes[member].length+actual[member].processed_target && after.pending == actual[member].next_id &&
          after.sampling == bytes[member].sampling.advanced(2*(lengths[member]-1)+1) &&
          record.after.sampling == after.sampling && actual[member].verified_rows == lengths[member] &&
          actual[member].proposal_count+1 == lengths[member] && actual[member].processed_draft == draft_rows &&
          record.draft_rows == draft_rows && actual[member].auxiliary_graph_calls == lengths[member]+1,
          "sampled varlen member used physical-width RNG addresses, work counts or another pending frontier");
      r::preserved_prefix(bytes[member],after);
      require(s::image(model,*before[member]) == bytes[member],"sampled verifier changed its retained original checkpoint");
    }
  }
  require(ledger_storage_checks == 24,"ledger storage checks missed a prepare, withdrawal, submission or publication");
  compare(2,lengths); before = {}; original = {};
  for (auto *group : {&controls,&sessions}) for (auto &session : *group) h::retire(session);
  for (const auto &[key,program] : model.draft_programs) {
    require(!program->guard.execution.has_completion() && !program->guard.execution.cleanup_pending &&
        !program->hidden_transactions.bound() && !program->carrier.pending(),"varlen repair/proposal retained original native resources");
    if (program->head) require(program->probability_pool.stats().live_blocks == 0,"retired varlen calls retained a private q owner");
  }
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      !storage->live_images() && model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == plan.max_pages && target_submissions == 11 && bounded_submissions == 3,
      "varlen journey split a bounded submission or retained a state/target/draft page owner");
  bool equivalent = true, exact = true;
  for (const auto &check : checks) { equivalent = equivalent && check.numerically_equivalent; exact = exact && check.payloads_equal; }
  std::cout << "{\"status\":\"PASS\",\"mode\":\"mtp_varlen_verify\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"requests\":2,\"physical_rows_per_member\":3,\"active_rows\":4,\"physical_rows\":6,"
      "\"length_orders\":[[1,3],[3,1],[1,3]],\"controlled_accepted\":[[1,2],[3,1]],"
      "\"target_submissions\":" << target_submissions << ",\"bounded_target_submissions\":" << bounded_submissions
      << ",\"natural_proposal_rows\":[0,2],\"natural_rng_draws\":[1,5],\"natural_processed_target\":["
      << actual[0].processed_target << ',' << actual[1].processed_target << "],\"natural_processed_draft\":["
      << actual[0].processed_draft << ',' << actual[1].processed_draft << "],\"state_checks\":[";
  for (std::size_t index = 0; index < checks.size(); ++index) { if (index) std::cout << ','; p::write_step(checks[index]); }
  std::cout << "],\"payloads_byte_equal\":" << (exact ? "true" : "false")
      << ",\"numerically_equivalent\":" << (equivalent ? "true" : "false")
      << ",\"old_checkpoints_unchanged\":true,\"accepted_prefixes_1_2_3\":true,\"fixed_carrier_reused\":true,"
      "\"engine_unequal_lengths_one_submission\":true,\"original_q_identity_and_release\":true,"
      "\"natural_R1_results_equal\":true,\"ledger_reserved_width\":8,\"ledger_record_storage_checks\":"
      << ledger_storage_checks << ",\"ledger_cold_storage_reused\":true,\"ledger_host_prepare_withdraw_reprepare\":true,"
      "\"full_request_history_import_coverage\":false,\"synthetic_untrained\":true,"
      "\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0}\n";
  cleanup.dismiss(); return 0;
}
} // namespace qwen_mtp_varlen_verify_case
