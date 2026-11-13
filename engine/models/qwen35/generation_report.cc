#include "generation_diagnostics.h"
#include "cache_report.h"
#include "weight_report.h"
#include "../../inference/request_owner.h"

namespace qwen35 {
void print_generation_rebuild(std::ostream &out, const GenerationRestoreStats &stats, bool draft_cache) {
  out << ",\"restore_contract\":\"recompute_committed_inputs\",\"restored_graph_calls\":" << stats.graph_calls
      << ",\"restored_target_rows\":" << stats.target_rows;
  if (draft_cache) out << ",\"restored_draft_rows\":" << stats.draft_rows
      << ",\"actual_restored_draft_rows\":" << stats.submitted_draft_rows
      << ",\"normalized_lag_rows\":" << stats.normalized_lag_rows;
  out << ",\"decision_changes\":[";
  for (std::size_t i = 0; i < stats.decision_changes.size(); ++i) {
    const auto &change = stats.decision_changes[i];
    out << (i ? "," : "") << "{\"episode\":" << change.episode << ",\"historical_next\":" << change.historical
        << ",\"fresh_next\":" << change.fresh << '}';
  }
  out << ']';
}

void write_request_sampling(std::ostream &out, const Model &model, const GenerationRequest &request) {
  const auto &configuration = model.sampling_configuration(request.sampling_index).configuration;
  if (model.sampling_configuration_count() == 1 && configuration.modifiers.history.empty()) return;
  out << ",\"sampling_configuration_index\":" << request.sampling_index
      << ",\"request_sampling\":\"" << (configuration.enabled ? "random" : "greedy") << '\"'
      << ",\"request_temperature\":" << configuration.policy.temperature
      << ",\"request_top_k\":" << configuration.policy.top_k << ",\"request_top_p\":" << configuration.policy.top_p
      << ",\"request_history_repetition\":" << configuration.modifiers.history.repetition
      << ",\"request_history_presence\":" << configuration.modifiers.history.presence
      << ",\"request_history_frequency\":" << configuration.modifiers.history.frequency
      << ",\"request_mtp_acceptance_policy\":\"" << (!model.cache_plan.draft_cache ? "disabled" : configuration.enabled ? "min_one_p_over_q" : "equal_greedy_ids") << '\"';
}

void print_generation_request(std::ostream &out, const Model &model, const GenerationRequest &request, std::size_t steps,
                       Clock::time_point startup, Clock::time_point ready,
                       std::array<std::size_t, 3> packed_prefill_offset = {},
                       const qwen35::RouteDecisionFacts &routes = {}) {
  const auto &prompt = request.prompt;
  const auto &generated = request.progress.generated;
  const auto &rounds = request.rounds;
  const auto prefill_start = request.turn_observations.empty() ? request.prefill_start : request.turn_observations.front().start;
  const auto prefill_endpoint = request.turn_observations.empty() ? request.prefill_endpoint : request.turn_observations.front().prefill_endpoint;
  const auto endpoint = request.endpoint;
  const auto prefill_step_seconds = request.prefill_step_seconds,
      decode_step_sum_seconds = request.decode_step_sum_seconds;
  const auto verified_rows = request.verified_rows, draft_rows = request.draft_rows,
      draft_rounds = request.draft_rounds, agreements = request.agreements, accepts = request.accepts;
  const auto &trace = request.trace;
  // The original finish endpoint follows publication and host token visibility,
  // before its trailing trace/log output. Inter-call work remains in elapsed.
  auto prefill_seconds = std::chrono::duration<double>(prefill_endpoint - prefill_start).count();
  auto decode_seconds = std::chrono::duration<double>(endpoint - prefill_endpoint).count();
  std::size_t logical_prompt = prompt.size(), turn_count = 1;
  if (!request.turn_observations.empty()) {
    prefill_seconds = 0; decode_seconds = 0; logical_prompt = request.progress.turns().size() - 1; turn_count = request.progress.turns().size();
    for (std::size_t i = 0; i < request.progress.turns().size(); ++i) {
      logical_prompt = cache::add(logical_prompt, request.progress.turns()[i].prompt.size());
      const auto &turn = request.turn_observations[i];
      prefill_seconds += std::chrono::duration<double>(turn.prefill_endpoint - turn.start).count();
      decode_seconds += std::chrono::duration<double>(turn.endpoint - turn.prefill_endpoint).count();
    }
  }
  const auto session_seconds = std::chrono::duration<double>(endpoint - prefill_start).count();
  out << std::setprecision(17)
            << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
            << "\",\"mtp\":" << (model.cache_plan.draft_cache ? "true" : "false") << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes)
            << hyle::engine::reporting::qwen35_cache_report(model, packed_prefill_offset) << ",\"visible_tokens\":" << request.visible_tokens;
  write_request_sampling(out, model, request);
  if (model.cache_plan.draft_cache) out << ",\"draft_frontier\":" << request.draft_frontier
      << ",\"draft_lag\":" << request.draft_lag;
  out << ",\"generated_ids\":[";
  for (std::size_t i = 0; i < generated.size(); ++i) out << (i ? "," : "") << generated[i];
  out << "],\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds
            << ",\"session_seconds\":" << session_seconds
            << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
            << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(prefill_endpoint - startup).count()
            << ",\"prefill_step_seconds\":" << prefill_step_seconds + request.prefix_sampling_seconds
            << ",\"decode_step_sum_seconds\":" << decode_step_sum_seconds
            << ",\"step_sum_seconds\":" << prefill_step_seconds + decode_step_sum_seconds + request.restored.step_seconds + request.prefix_sampling_seconds
            << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? logical_prompt / prefill_seconds : 0)
            << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? (generated.size() - turn_count) / decode_seconds : 0)
            << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated.size() / session_seconds : 0)
            << ",\"requested_steps\":" << steps << ",\"generated_tokens\":" << generated.size()
            << ",\"generated_decode_tokens\":" << generated.size() - turn_count << ",\"verified_target_rows\":" << verified_rows
            << (request.turn_observations.empty() ? ",\"actual_graph_calls\":" : ",\"graph_participations\":")
            << rounds.size() + request.prefill_graph_calls + request.restored.graph_calls + request.prefix_sampling_calls + request.auxiliary_graph_calls
            << ",\"prefix_sampling_graph_calls\":" << request.prefix_sampling_calls
            << ",\"sampling_stream\":" << request.progress.initial_sampling.stream
            << ",\"sampling_draw_cursor\":" << (request.ledger && request.ledger->committed_count
                ? request.ledger->records[request.ledger->committed_count - 1].after.sampling.draw
                : request.session ? request.session->head->sampling.draw : 0)
            << ",\"prefill_graph_calls\":" << request.prefill_graph_calls << ",\"decode_graph_calls\":" << rounds.size()
            << ",\"prefill_target_rows\":" << request.prefill_graph_tokens
            << ",\"actual_target_rows\":" << request.prefill_graph_tokens + verified_rows + request.restored.target_rows
            << ",\"committed_target_rows\":" << request.visible_tokens
            << ",\"committed_target_decode_rows\":" << request.visible_tokens - logical_prompt;
  if (model.cache_plan.draft_cache) out << ",\"prefill_draft_rows\":" << request.prefill_draft_rows
      << ",\"actual_draft_rows\":" << request.actual_draft_rows()
      << ",\"actual_draft_rows_scope\":\"admitted_active_input_rows\""
      << ",\"draft_rows\":" << draft_rows << ",\"draft_rounds\":" << draft_rounds
      << ",\"agreements\":" << agreements << ",\"consumed_drafts\":" << accepts;
  out << ",\"diagnostic\":" << (trace.empty() ? "false" : "true");
  if (!request.turn_observations.empty()) {
    out << ",\"emitted_pending\":" << generated.back() << ",\"fresh_next_id\":" << request.fresh_next_id;
    print_generation_rebuild(out, request.restored, model.cache_plan.draft_cache);
  }
  if (!request.id.empty())
    out << ",\"logical_prompt_tokens\":" << logical_prompt << ",\"reused_tokens\":" << request.reused_tokens
              << ",\"prefix_hits\":" << request.prefix_hits << ",\"prefix_misses\":" << request.prefix_misses
              << ",\"prefix_full_hits\":" << request.prefix_full_hits;
  if (!request.turn_observations.empty()) {
    out << ",\"between_turn_seconds\":" << (std::max)(0.0, session_seconds - prefill_seconds - decode_seconds)
              << ",\"phase_timing\":\"sum_per_turn_latency_including_interleaving\",\"turns\":[";
    for (std::size_t i = 0; i < request.turn_observations.size(); ++i) {
      const auto &turn = request.turn_observations[i];
      out << (i ? "," : "") << "{\"turn\":" << i + 1 << ",\"requested_steps\":" << request.progress.turns()[i].max_outputs
          << ",\"new_user_tokens\":" << request.progress.turns()[i].prompt.size() << ",\"pending_flushed\":" << (i ? 1 : 0)
          << ",\"context_before\":" << turn.context_before << ",\"context_after\":" << turn.context_after
          << ",\"output_begin\":" << turn.output_begin << ",\"output_end\":" << turn.output_end
          << ",\"prefill_seconds\":" << std::chrono::duration<double>(turn.prefill_endpoint - turn.start).count()
          << ",\"decode_seconds\":" << std::chrono::duration<double>(turn.endpoint - turn.prefill_endpoint).count()
          << ",\"prefill_shape_admission_seconds\":" << turn.prefill_shape_admission_seconds
          << ",\"prefill_phase_prepare_seconds\":" << turn.prefill_phase_prepare_seconds
          << ",\"prefill_slot_initialize_seconds\":" << turn.prefill_slot_initialize_seconds
          << ",\"prefill_phase_creations\":" << turn.prefill_phase_creations
          << ",\"prefill_slot_initializations\":" << turn.prefill_slot_initializations;
      if (turn.arrival != Clock::time_point{})
        out << ",\"arrival_elapsed_seconds\":" << std::chrono::duration<double>(turn.arrival - startup).count()
                  << ",\"request_to_first_token_seconds\":" << std::chrono::duration<double>(turn.prefill_endpoint - turn.arrival).count();
      if (i == 0) out << ",\"after_ready_first_token_seconds\":"
                           << std::chrono::duration<double>(turn.prefill_endpoint - ready).count();
      out << ",\"input_ids\":[";
      for (std::size_t k = 0; k < request.warm_inputs[i]->size(); ++k)
        out << (k ? "," : "") << (*request.warm_inputs[i])[k];
      out << "]}";
    }
    out << ']';
  }
  out << ",\"rounds\":[";
  for (std::size_t i = 0; i < rounds.size(); ++i) {
    const auto &row = rounds[i];
    out << (i ? "," : "") << "{\"pending\":" << row.pending;
    if (!model.cache_plan.draft_cache) {
      out << ",\"next_id\":" << row.next_id << ",\"committed_target_rows\":" << row.processed_target << '}';
      continue;
    }
    out << ",\"draft\":" << row.draft
              << ",\"target_first\":" << row.target_first << ",\"target_second\":" << row.target_second
              << ",\"accepted_draft\":" << (row.accepted_draft ? "true" : "false")
              << ",\"proposals\":" << row.proposal_count << ",\"verified_rows\":" << row.verified_rows
              << ",\"consumed_drafts\":" << row.consumed_drafts << ",\"auxiliary_graph_calls\":" << row.auxiliary_graph_calls
              << ",\"committed_target_rows\":" << row.processed_target << ",\"draft_rows\":" << row.processed_draft << '}';
  }
  out << "]}";
}


void write_generation_report(std::ostream &report_out, const GenerationDiagnosticView &view) {
  const auto &model = view.model;
  const auto &requests = view.requests;
  const auto &output_deliveries = view.output_deliveries;
  const auto &prefix = view.prefix;
  const auto &dispatch = view.dispatch;
  const auto &dispatch_turns = view.dispatch_turns;
  const auto &evictions = view.evictions;
  const auto &batches = view.batches;
  const auto &startup = view.startup;
  const auto &ready = view.ready;
  const auto &start = view.start;
  const auto &endpoint = view.endpoint;
  const auto &total_turns = view.total_turns;
  const auto &requested_outputs = view.requested_outputs;
  const auto &admitted = view.admitted;
  const auto &completed = view.completed;
  const auto &peak_resident = view.peak_resident;
  const auto &peak_committed = view.peak_committed;
  const auto &actual_prefill_calls = view.actual_prefill_calls;
  const auto &actual_decode_calls = view.actual_decode_calls;
  const auto &actual_fresh_participations = view.actual_fresh_participations;
  const auto &actual_fresh_target_rows = view.actual_fresh_target_rows;
  const auto &frontier_before = view.frontier_before;
  const auto &draft_before = view.draft_before;
  const auto &prefix_before = view.prefix_before;
  const auto &submitted_draft_begin = view.submitted_draft_begin;
  const auto &prefill_seconds = view.prefill_seconds;
  const auto &decode_seconds = view.decode_seconds;
  const auto &actual_step_sum = view.actual_step_sum;
  const auto &frontier_seconds_before = view.frontier_seconds_before;
  const auto &cold_prepare_seconds = view.cold_prepare_seconds;
  const auto &mtp = view.mtp;
  const auto &reuse_ready = view.reuse_ready;
  const auto &decode_overlap = view.decode_overlap;
  const auto &prefix_enabled = view.prefix_enabled;
  const auto &first_turn_only = view.first_turn_only;
  const auto &startup_prefill_warmup = view.startup_prefill_warmup;
  const auto &packed_prefill_offset = view.packed_prefill_offset;
  const auto &continuation = view.continuation;
  const auto &routes = view.routes;
  const auto &trace = view.trace;
  if (output_deliveries) {
    const auto cancelled = output_deliveries->cancelled_count();
    if (cancelled) {
      // Planned-turn timing and decode totals are not meaningful for an
      // interrupted workload. Preserve actual outputs without inventing them.
      report_out << "{\"cancelled_requests\":" << cancelled << ",\"sessions\":[";
      for (std::size_t index = 0; index < requests.size(); ++index) {
        report_out << (index ? "," : "") << "{\"slot\":" << index << ",\"cancelled\":"
            << (output_deliveries->cancelled(index) ? "true" : "false") << ",\"generated_ids\":[";
        const auto &ids = requests[index].progress.generated;
        for (std::size_t at = 0; at < ids.size(); ++at) report_out << (at ? "," : "") << ids[at];
        report_out << "]}";
      }
      report_out << "]}\n";
      return;
    }
  }
  const auto post_endpoint_seconds = std::chrono::duration<double>(Clock::now() - endpoint).count();
  const auto session_seconds = std::chrono::duration<double>(endpoint - start).count();
  if ((actual_prefill_calls && !(prefill_seconds > 0)) || (actual_decode_calls && !(decode_seconds > 0)))
    throw std::logic_error("an actual generation phase has no observed completion interval");
  std::size_t logical_prompt = 0, user_prompt = 0, prefill_tokens = 0, prefill_draft = 0;
  std::size_t generated = 0, verified = 0, drafted = 0, committed = 0;
  std::size_t draft_rounds = 0, agreements = 0, accepts = 0, hits = 0, misses = 0, full_hits = 0, reused = 0;
  std::size_t restores = 0, recomputed_changes = 0; GenerationRestoreStats restored; double step_sum_seconds = 0;
  for (const auto &request : requests) {
    for (const auto &turn : request.progress.turns()) user_prompt = cache::add(user_prompt, turn.prompt.size());
    prefill_tokens = cache::add(prefill_tokens, request.prefill_graph_tokens);
    prefill_draft = cache::add(prefill_draft, request.prefill_draft_rows);
    generated = cache::add(generated, request.progress.generated.size());
    verified = cache::add(verified, request.verified_rows); drafted = cache::add(drafted, request.draft_rows);
    committed = cache::add(committed, request.visible_tokens); draft_rounds = cache::add(draft_rounds, request.draft_rounds);
    agreements = cache::add(agreements, request.agreements); accepts = cache::add(accepts, request.accepts);
    hits = cache::add(hits, request.prefix_hits); misses = cache::add(misses, request.prefix_misses);
    full_hits = cache::add(full_hits, request.prefix_full_hits); reused = cache::add(reused, request.reused_tokens);
    step_sum_seconds += request.prefill_step_seconds + request.decode_step_sum_seconds;
    restores = cache::add(restores, request.restore_count);
    restored.graph_calls = cache::add(restored.graph_calls, request.restored.graph_calls);
    restored.target_rows = cache::add(restored.target_rows, request.restored.target_rows);
    restored.draft_rows = cache::add(restored.draft_rows, request.restored.draft_rows);
    restored.submitted_draft_rows = cache::add(restored.submitted_draft_rows, request.restored.submitted_draft_rows);
    restored.normalized_lag_rows = cache::add(restored.normalized_lag_rows, request.restored.normalized_lag_rows);
    recomputed_changes = cache::add(recomputed_changes, request.restored.decision_changes.size());
    restored.step_seconds += request.restored.step_seconds; restored.seconds += request.restored.seconds;
  }
  logical_prompt = cache::add(user_prompt, total_turns - requests.size());
  const auto frontier_calls = model.frontier_sampling_graph_calls - frontier_before;
  const auto draft_calls = model.draft_program_graph_calls - draft_before;
  const auto prefix_calls = model.prefix_decision_graph_calls - prefix_before;
  const auto graph_calls = cache::add(cache::add(cache::add(actual_prefill_calls, actual_decode_calls), restored.graph_calls),
      cache::add(frontier_calls, cache::add(draft_calls, prefix_calls)));
  report_out << std::setprecision(17) << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
      << "\",\"mtp\":" << (model.cache_plan.draft_cache ? "true" : "false") << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes)
      << (reuse_ready ? ",\"model_counters_scope\":\"lifetime_except_packed_prefill_and_aggregate_calls\"" : "")
      << ",\"session_count\":" << requests.size() << ",\"turn_count\":" << total_turns << ",\"active_sequences\":" << model.cache_plan.active_sequences << ",\"max_sessions\":" << model.cache_plan.resident_sessions
      << ",\"decode_overlap\":" << (decode_overlap ? "true" : "false")
      << ",\"decode_overlap_available\":" << (decode_overlap && !continuation.unavailable ? "true" : "false")
      << ",\"overlap_successor_submissions\":" << continuation.successor_submissions
      << ",\"overlap_successor_publications\":" << continuation.successor_publications
      << ",\"overlap_successor_discards\":" << continuation.successor_discards
      << ",\"overlap_successor_refusals\":" << continuation.successor_refusals
      << hyle::engine::reporting::qwen35_cache_report(model, packed_prefill_offset) << ",\"prefix_cache\":" << (prefix_enabled ? "true" : "false")
      << ",\"admitted_requests\":" << requests.size() << ",\"admitted_turns\":" << admitted << ",\"retired_requests\":" << completed
      << ",\"peak_resident_requests\":" << peak_resident << ",\"peak_committed_versions\":" << peak_committed
      << ",\"live_committed_versions\":" << model.snapshot_versions.live << ",\"idle_evictions\":" << evictions.size()
      << ",\"prefix_hits\":" << hits << ",\"prefix_misses\":" << misses << ",\"prefix_full_hits\":" << full_hits << ",\"reused_tokens\":" << reused
      << ",\"prefix_entries\":" << (prefix ? prefix->size() : 0) << ",\"prefix_evictions\":" << (prefix ? prefix->evictions() : 0)
      << ",\"requested_output_tokens\":" << requested_outputs << ",\"generated_tokens\":" << generated << ",\"generated_decode_tokens\":" << generated - total_turns
      << ",\"actual_graph_calls\":" << graph_calls << ",\"prefill_graph_calls\":" << actual_prefill_calls << ",\"decode_graph_calls\":" << actual_decode_calls
      << ",\"graph_participations\":" << actual_fresh_participations + restored.graph_calls + frontier_calls + draft_calls + prefix_calls
      << ",\"cold_planning\":\"" << (first_turn_only ? "configured_capacity" : "full_schedule_preknowledge_invalid")
      << "\",\"future_shape_hints\":" << (first_turn_only ? 0 : 1)
      << ",\"configured_prefill_rows\":" << model.bounded_prefill_rows
      << ",\"shared_execution_scratch_bytes\":" << model.execution_workspace.bytes()
      << ",\"shared_prefill_scratch_bytes\":" << model.execution_workspace.bytes()
      << ",\"cold_prepare_seconds\":" << cold_prepare_seconds
      << ",\"startup_prefill_warmup_calls\":" << startup_prefill_warmup.calls
      << ",\"startup_prefill_warmup_target_rows\":" << startup_prefill_warmup.target_rows
      << ",\"startup_prefill_warmup_draft_rows\":" << startup_prefill_warmup.draft_rows
      << ",\"startup_prefill_warmup_physical_rows\":" << startup_prefill_warmup.physical_rows
      << ",\"startup_prefill_warmup_retired\":" << (startup_prefill_warmup.retired ? "true" : "false")
      << ",\"startup_prefill_warmup_seconds\":" << startup_prefill_warmup.seconds
      << ",\"logical_dispatches\":" << dispatch.size() << ",\"logical_prompt_tokens\":" << logical_prompt << ",\"logical_user_prompt_tokens\":" << user_prompt
      << ",\"pending_flushed_tokens\":" << total_turns - requests.size() << ",\"prefill_target_rows\":" << prefill_tokens
      << ",\"verified_target_rows\":" << verified
      << ",\"actual_target_rows\":" << cache::add(actual_fresh_target_rows, restored.target_rows)
      << ",\"committed_target_rows\":" << committed << ",\"committed_target_decode_rows\":" << committed - logical_prompt
      << ",\"restore_count\":" << restores << ",\"restored_graph_calls\":" << restored.graph_calls << ",\"restored_target_rows\":" << restored.target_rows
      << ",\"restore_step_seconds\":" << restored.step_seconds << ",\"restore_seconds\":" << restored.seconds
      << ",\"restore_contract\":\"recompute_committed_inputs\"";
  if (mtp) report_out << ",\"prefill_draft_rows\":" << prefill_draft << ",\"draft_rows\":" << drafted
      << ",\"actual_draft_rows\":" << model.submitted_draft_rows - submitted_draft_begin
      << ",\"actual_draft_rows_scope\":\"admitted_active_input_rows\""
      << ",\"draft_rounds\":" << draft_rounds << ",\"agreements\":" << agreements << ",\"consumed_drafts\":" << accepts
      << ",\"restored_draft_rows\":" << restored.draft_rows
      << ",\"actual_restored_draft_rows\":" << restored.submitted_draft_rows
      << ",\"normalized_lag_rows\":" << restored.normalized_lag_rows;
  report_out << ",\"recomputed_decision_changes\":" << recomputed_changes
      << ",\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds << ",\"session_seconds\":" << session_seconds
      << ",\"step_sum_seconds\":" << actual_step_sum + restored.step_seconds + model.frontier_sampling_seconds - frontier_seconds_before
      << ",\"participation_step_sum_seconds\":" << step_sum_seconds << ",\"post_endpoint_seconds\":" << post_endpoint_seconds
      << ",\"phase_timing\":\"disjoint_continuous_episode_intervals\",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
      << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count() << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
      << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(requests.front().turn_observations.front().prefill_endpoint - startup).count()
      << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? logical_prompt / prefill_seconds : 0)
      << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? (generated - total_turns) / decode_seconds : 0)
      << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated / session_seconds : 0)
      << ",\"diagnostic\":" << (trace.empty() ? "false" : "true") << ",\"dispatch\":[";
  for (std::size_t i = 0; i < dispatch.size(); ++i)
    report_out << (i ? "," : "") << "{\"session_id\":\"" << requests[dispatch[i]].id << "\",\"turn\":" << dispatch_turns[i] + 1 << '}';
  report_out << "],\"batches\":[";
  for (std::size_t i = 0; i < batches.size(); ++i) {
    const auto &batch = batches[i];
    report_out << (i ? "," : "") << "{\"index\":" << i << ",\"requests\":" << batch.members.size()
        << ",\"tokens\":" << batch.tokens << ",\"role\":" << static_cast<int>(batch.role)
        << ",\"generation\":" << (batch.generation ? "true" : "false") << ",\"session_ids\":[";
    for (std::size_t j = 0; j < batch.members.size(); ++j)
      report_out << (j ? "," : "") << '"' << requests[batch.members[j]].id << '"';
    report_out << ']';
    report_out << ",\"input_rows\":[";
    for (std::size_t j = 0; j < batch.input_rows.size(); ++j) report_out << (j ? "," : "") << batch.input_rows[j];
    report_out << "],\"active_rows\":" << batch.active_rows << ",\"physical_rows\":" << batch.physical_rows << '}';
  }
  report_out << "],\"sessions\":[";
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto &request = requests[i]; std::size_t quota = 0;
    for (const auto &turn : request.progress.turns()) quota = cache::add(quota, turn.max_outputs);
    report_out << (i ? "," : "") << "{\"session_id\":\"" << request.id << "\",\"result\":";
    print_generation_request(report_out, model, request, quota, startup, ready, packed_prefill_offset, routes);
    report_out << ",\"restore_count\":" << request.restore_count << ",\"idle_evictions\":" << request.idle_evictions
        << ",\"checkpoint_resume_hits\":" << request.checkpoint_resume_hits
        << ",\"checkpoint_resume_misses\":" << request.checkpoint_resume_misses << ",\"ledger\":[";
    std::size_t round = 0;
    for (std::size_t j = 0; j < request.ledger->committed_count; ++j) {
      const auto &record = request.ledger->records[j]; if (record.generation) ++round;
      std::uint64_t hash = 1469598103934665603ULL;
      for (const auto token : record.ids()) hash = (hash ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
      const auto prefix_name = record.turn ? "round" + std::to_string(record.turn + 1) + "_" : "";
      const auto name = prefix_name + (record.generation ? (mtp ? "mtp" : "decode") + std::to_string(round) : "prefill");
      report_out << (j ? "," : "") << "{\"index\":" << j << ",\"turn\":" << record.turn + 1 << ",\"role\":" << static_cast<int>(record.role)
          << ",\"requests\":" << record.requests << ",\"member_index\":" << record.member_index
          << ",\"generation\":" << (record.generation ? "true" : "false") << ",\"input_rows\":" << record.ids().size()
          << ",\"verified_rows\":" << record.verified_rows << ",\"draft_limit\":" << record.draft_limit
          << ",\"consumed_drafts\":" << record.consumed_drafts << ",\"auxiliary_graph_calls\":" << record.auxiliary_graph_calls
          << ",\"input_hash\":" << hash << ",\"first_input\":" << record.ids().front() << ",\"quota\":" << record.quota
          << ",\"before_length\":" << record.before.length << ",\"before_frontier\":" << record.before.frontier << ",\"before_lag\":" << record.before.lag
          << ",\"before_next_id\":" << record.before.next_id << ",\"draft\":" << record.draft << ",\"target_first\":" << record.target_first << ",\"target_second\":" << record.target_second
          << ",\"sampling_seed\":" << record.before.sampling.seed << ",\"sampling_stream\":" << record.before.sampling.stream
          << ",\"sampling_draw_before\":" << record.before.sampling.draw << ",\"sampling_draw_after\":" << record.after.sampling.draw
          << ",\"accepted\":" << record.accepted << ",\"draft_rows\":" << record.draft_rows << ",\"after_length\":" << record.after.length
          << ",\"after_frontier\":" << record.after.frontier << ",\"after_lag\":" << record.after.lag << ",\"next_id\":" << record.after.next_id
          << ",\"committed_input_ids\":[";
      if (record.generation) for (std::size_t k = 0; k < record.accepted; ++k)
        report_out << (k ? "," : "") << record.accepted_inputs[k];
      report_out << "],\"verification_input_ids\":[";
      for (std::size_t k = 0; k < record.verification_inputs.size(); ++k)
        report_out << (k ? "," : "") << record.verification_inputs[k];
      report_out << "],\"imported\":" << (record.imported ? "true" : "false")
          << ",\"source_geometry\":\"" << (record.imported ? "unknown" : "recorded")
          << "\",\"trace_name\":\"" << (record.imported ? "" : name) << "\"}";
    }
    report_out << "]}";
  }
  report_out << "]}\n"; return;
}
} // namespace qwen35
