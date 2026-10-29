#pragma once

// Finite conditional bias/mask consumers on the cached untrained GA fixture.
// Real private NextN q leases are borrowed unchanged by the original terminal.
// This does not claim a natural grammar supplier or full target-state commit.
namespace qwen_conditional_logits_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace m = qwen_mtp_logits_case;
namespace s = qwen_mtp_batch_state_case;
namespace t = qwen_mtp_stop_case;
namespace g = qwen_mtp_group_return_case;
using h::require;
constexpr float temperature = .8F;

inline m::Payload row(q::Model &model, std::initializer_list<std::int32_t> support, float offset) {
  m::Payload result;
  result.bias.resize(static_cast<std::size_t>(model.vocabulary)); result.allowed.resize(result.bias.size());
  for (const auto token : support) {
    const auto at = static_cast<std::size_t>(token);
    result.allowed.at(at) = 1; result.bias.at(at) = offset + static_cast<float>(at) * (.375F-offset);
  }
  result.inputs = {m::ready<float>(model,q::Type::kFloat32,result.bias),
      m::ready<std::uint8_t>(model,q::Type::kBool,result.allowed)};
  return result;
}
inline std::vector<float> read(q::Model &model, const q::c::ReadLease &lease) {
  std::vector<float> values(static_cast<std::size_t>(model.vocabulary));
  require(lease.valid() && lease.size() == values.size()*sizeof(float),"conditional q lost its actual full-vocabulary owner");
  q::check(model.execution.read(q::take(lease.view()),std::as_writable_bytes(std::span(values)))); return values;
}
inline std::vector<double> probabilities(std::span<const float> values) {
  std::vector<double> result; result.reserve(values.size()); double total = 0;
  for (const auto value : values) { result.push_back(std::exp(double(value))); total += result.back(); }
  require(std::abs(total-1) < .00003,"conditional q is not its original normalized distribution"); return result;
}
inline void q_support(q::Model &model, const q::c::ReadLease &lease, const m::Payload &payload) {
  const auto values = read(model,lease); (void)probabilities(values);
  for (std::size_t at = 0; at < values.size(); ++at)
    require(payload.allowed[at] ? std::isfinite(values[at]) : values[at] == -(std::numeric_limits<float>::infinity)(),
        "private draft ignored the bias/mask snapshot for its actual proposal prefix");
}
struct Reference final {
  std::vector<std::int32_t> emitted;
  std::size_t consumed_drafts = 0;
  bool first_agrees = false, stopped = false;
};
inline Reference reference(std::span<const std::vector<double>> p,
    std::span<const std::vector<double>> proposal_laws, std::span<const std::int32_t> ids,
    q::cache::SamplingCursor cursor, std::size_t quota, std::span<const std::int32_t> eos, q::BackendKind kind) {
  const auto count = ids.size();
  require(p.size() == count+1 && proposal_laws.size() == count && quota,
      "conditional reference needs its actual proposal laws, target rows and a positive quota");
  Reference result;
  for (std::size_t at = 0; at <= count; ++at) {
    bool accepted = false;
    if (at != count) {
      const auto token = static_cast<std::size_t>(ids[at]);
      require(proposal_laws[at][token] > 0,"conditional reference proposal has no original q support");
      const auto ratio = (std::min)(1.,p[at][token]/proposal_laws[at][token]);
      const std::array weights{ratio,1.-ratio};
      accepted = m::draw(weights,cursor.identity(2*at+1),kind) == 0;
      if (!at) result.first_agrees = accepted;
    }
    std::int32_t token;
    if (accepted) token = ids[at];
    else {
      auto weights = p[at];
      if (at != count) for (std::size_t column = 0; column != weights.size(); ++column)
        weights[column] = (std::max)(0.,weights[column]-proposal_laws[at][column]);
      token = m::draw(weights,cursor.identity(2*count),kind);
    }
    result.emitted.push_back(token);
    result.stopped = std::ranges::find(eos,token) != eos.end();
    if (!accepted || result.stopped || result.emitted.size() == quota) return result;
    ++result.consumed_drafts;
  }
  throw std::runtime_error("conditional independent reference did not terminate");
}
inline void equal(const q::PrefixSamplingResult &actual, const Reference &expected) {
  require(actual.final_ids == expected.emitted && actual.prefix.consumed == expected.emitted.size() &&
      actual.prefix.consumed_drafts == expected.consumed_drafts && actual.prefix.next_id == expected.emitted.back() &&
      actual.prefix.first_agrees == expected.first_agrees && actual.prefix.stopped == expected.stopped,
      "conditional device prefix differs from independent per-row acceptance/residual/bonus equations");
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  require(kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda,
      "conditional logits finite consumer is currently checked on CPU/CUDA only");
  auto source = q::prepare_weight_source(input/"weights.tsv",true); const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.mtp_layers == 1 && geometry.recurrent_layers() && geometry.attention_layers(),
      "conditional logits require the existing cached untrained GA/H64/V64/C16 MTP fixture");
  const auto plan = q::cache::plan(geometry,source.cache_row_bytes,16,6,3,
      q::recurrent_bytes(q::state_schema_for(geometry,16),geometry),2,3,true,true);
  constexpr float frequency = .375F;
  const std::array additional{
      q::SamplingConfiguration{true,{temperature,0,1.F},{.bias = true,.allowed = true,.history = {.frequency = frequency}}},
      q::SamplingConfiguration{false,{}, {.bias = true,.allowed = true}}};
  q::Model model(std::move(source),geometry,16,plan,{},false,q::ReplayPolicy::kCommands,SIZE_MAX,
      {.enabled = true,.policy = {temperature,0,1.F},.seed = 123,.modifiers = {.bias = true,.allowed = true}},
      0,{.primary = kind},additional);
  model.configure_draft_policy({.max_drafts = 2});
  const std::array three{std::size_t{3}};
  model.prepare_family(three,false,1,q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3,1,1,q::GraphRole::kMtpPrefill);
  model.prepare_recursive_round_programs(0,true);
  model.prepare_prefix_sampling(2,1,true); model.prepare_prefix_sampling(2,2,true);
  model.freeze_shape_plan(); model.prepare_execution(three,1,q::GraphRole::kMtpPrefill);
  const std::array payloads{row(model,{0,1},-.5F),row(model,{1,2},.25F),row(model,{2,3},-.75F)};
  const std::array<q::LogitsInputs,3> inputs{payloads[0].inputs,payloads[1].inputs,payloads[2].inputs};
  const auto conditional = std::span<const q::LogitsInputs>(inputs);
  std::array<std::vector<float>,3> raw;
  const std::array<std::array<double,2>,3> masses{{{.25,.75},{.375,.625},{.25,.75}}};
  for (std::size_t at = 0; at != raw.size(); ++at) {
    raw[at].assign(static_cast<std::size_t>(model.vocabulary),-40.F);
    for (std::size_t column = 0; column != 2; ++column) {
      const auto token = at+column;
      raw[at][token] = static_cast<float>(std::log(masses[at][column]))*temperature-payloads[at].bias[token];
    }
  }
  std::array<std::vector<double>,3> expected;
  for (std::size_t at = 0; at != expected.size(); ++at)
    expected[at] = m::normalized(raw[at],payloads[at].bias,payloads[at].allowed,temperature);
  auto target = model.caller_tensor(model.execution.device(),q::dense(q::Type::kFloat32,{3,model.vocabulary}));
  auto target_values = raw[0]; target_values.insert(target_values.end(),raw[1].begin(),raw[1].end());
  target_values.insert(target_values.end(),raw[2].begin(),raw[2].end()); target.write<float>(target_values);
  std::array<std::vector<float>,2> q_log;
  for (auto &values : q_log) values.assign(static_cast<std::size_t>(model.vocabulary),-(std::numeric_limits<float>::infinity)());
  q_log[0][0] = std::log(.5F); q_log[0][1] = std::log(.5F);
  q_log[1][1] = std::log(.75F); q_log[1][2] = std::log(.25F);
  const std::array<q::c::ReadLease,2> q_rows{m::ready<float>(model,q::Type::kFloat32,q_log[0]).lease,
      m::ready<float>(model,q::Type::kFloat32,q_log[1]).lease};
  const std::array laws{probabilities(q_log[0]),probabilities(q_log[1])};
  const std::array<std::int32_t,2> ids{0,2};
  std::array<q::cache::SamplingCursor,2> cursors;
  for (std::size_t branch = 0; branch != cursors.size(); ++branch) {
    bool found = false;
    for (std::uint64_t seed = 1; seed != 513 && !found; ++seed) {
      const q::cache::SamplingCursor cursor{seed,71,0};
      const auto ratio = expected[0][0]/laws[0][0]; const std::array acceptance{ratio,1.-ratio};
      const auto choice = m::draw_reference(acceptance,cursor.identity(1),kind);
      if ((choice.selected == 0) == (branch == 1) && choice.margin > .05 &&
          m::separated_draw(expected[2],cursor.identity(4),kind)) { cursors[branch] = cursor; found = true; }
    }
    require(found,"conditional finite corpus could not find separated fixed acceptance addresses");
    const auto actual = model.sample_recursive_prefix(target.read(),ids,q_rows,cursors[branch],3,{},nullptr,0,{},conditional);
    const auto wanted = reference(expected,laws,ids,cursors[branch],3,{},kind); equal(actual,wanted);
    require(actual.final_ids.size() == (branch ? 3u : 1u),"finite conditional accept/reject branch was not actually exercised");
  }
  const std::array eos{ids[0]};
  equal(model.sample_recursive_prefix(target.read(),ids,q_rows,cursors[1],3,eos,nullptr,0,{},conditional),
      reference(expected,laws,ids,cursors[1],3,eos,kind));
  equal(model.sample_recursive_prefix(target.read(),ids,q_rows,cursors[1],1,{},nullptr,0,{},conditional),
      reference(expected,laws,ids,cursors[1],1,{},kind));
  auto one_target = model.caller_tensor(model.execution.device(),q::dense(q::Type::kFloat32,{1,model.vocabulary}));
  one_target.write<float>(raw[2]);
  const std::array one_input{inputs[2]};
  const auto sole = model.sample_recursive_prefix(one_target.read(),{},{},cursors[1],1,{},nullptr,0,{},
      std::span<const q::LogitsInputs>(one_input));
  require(sole.final_ids == std::vector<std::int32_t>{m::draw(expected[2],cursors[1].identity(0),kind)} &&
      sole.prefix.consumed == 1 && sole.prefix.consumed_drafts == 0,
      "conditional zero-proposal final row used a shared modifier or wrong RNG address");

  // One C0 belongs to the history owner. Later rows add the actual
  // hypotheses, while their external bias/mask snapshots stay independent.
  std::vector<std::int32_t> counts(static_cast<std::size_t>(model.vocabulary));
  counts[0] = 2; counts[1] = 1; counts[2] = 3; counts[3] = 1; counts[7] = 5;
  const q::LogitsInputs base_counts{.counts = m::ready<std::int32_t>(model,q::Type::kInt32,counts)};
  auto advanced = counts;
  const std::vector<float> zero_bias(counts.size());
  std::array<std::vector<double>,3> history_expected, stale_history;
  for (std::size_t at = 0; at != 3; ++at) {
    auto processed = raw[at], stale = raw[at];
    for (std::size_t column = 0; column != counts.size(); ++column) {
      processed[column] += payloads[at].bias[column]; stale[column] += payloads[at].bias[column];
      processed[column] -= frequency*static_cast<float>(advanced[column]);
      stale[column] -= frequency*static_cast<float>(counts[column]);
    }
    history_expected[at] = m::normalized(processed,zero_bias,payloads[at].allowed,temperature);
    stale_history[at] = m::normalized(stale,zero_bias,payloads[at].allowed,temperature);
    if (at != 2) ++advanced[static_cast<std::size_t>(ids[at])];
  }
  q::cache::SamplingCursor history_cursor;
  bool history_address = false;
  for (std::uint64_t seed = 1; seed != 8193 && !history_address; ++seed) {
    const q::cache::SamplingCursor candidate{seed,71,0};
    const auto correct = reference(history_expected,laws,ids,candidate,3,{},kind);
    const auto stale = reference(stale_history,laws,ids,candidate,3,{},kind);
    const auto ratio = history_expected[0][0]/laws[0][0]; const std::array acceptance{ratio,1.-ratio};
    if (correct.emitted.size() == 3 && stale.emitted.size() == 3 && correct.emitted != stale.emitted &&
        m::separated_draw(acceptance,candidate.identity(1),kind) &&
        m::separated_draw(history_expected[2],candidate.identity(4),kind) &&
        m::separated_draw(stale_history[2],candidate.identity(4),kind)) {
      history_cursor = candidate; history_address = true;
    }
  }
  require(history_address,"conditional history corpus did not distinguish real prefix counts from stale C0");
  equal(model.sample_recursive_prefix(target.read(),ids,q_rows,history_cursor,3,{},nullptr,1,base_counts,conditional),
      reference(history_expected,laws,ids,history_cursor,3,{},kind));
  std::vector<std::int32_t> counts_after(counts.size());
  q::check(model.execution.read(q::take(base_counts.counts.lease.view()),std::as_writable_bytes(std::span(counts_after))));
  require(counts_after == counts,"conditional prefix rewrote its owner's readonly C0 history");
  std::vector<std::int32_t> greedy_ids, wrong_shared_bias_ids;
  for (std::size_t at = 0; at != 3; ++at) {
    float maximum = -(std::numeric_limits<float>::infinity)(), wrong_maximum = maximum;
    std::int32_t selected = -1, wrong_selected = -1;
    for (std::size_t column = 0; column != counts.size(); ++column) if (payloads[at].allowed[column]) {
      const float value = raw[at][column]+payloads[at].bias[column];
      const float wrong_value = raw[at][column]+payloads[0].bias[column];
      if (value > maximum) { maximum = value; selected = static_cast<std::int32_t>(column); }
      if (wrong_value > wrong_maximum) { wrong_maximum = wrong_value; wrong_selected = static_cast<std::int32_t>(column); }
    }
    require(selected >= 0 && wrong_selected >= 0,"conditional greedy oracle has no allowed column");
    greedy_ids.push_back(selected); wrong_shared_bias_ids.push_back(wrong_selected);
  }
  require(greedy_ids != wrong_shared_bias_ids,"conditional bias corpus cannot distinguish an incorrect shared row0 bias binding");
  const auto greedy = model.sample_recursive_prefix(target.read(),ids,{},cursors[1],3,{},nullptr,2,{},conditional);
  require(greedy.final_ids == greedy_ids,"conditional greedy rows differ from independent F32 masked argmax");

  const std::array<std::int32_t,3> prompt{1,2,3};
  auto session = model.create_session(three,{},q::GraphRole::kMtpPrefill,q::cache::SamplingCursor{123,17,0});
  const auto prefill_payload = row(model,{8,9},.5F);
  (void)m::raw(model,session,prompt,q::GraphRole::kMtpPrefill,prefill_payload.inputs);
  auto checkpoint = session->checkpoint(); const auto head = session->head.get();
  const auto epoch = session->head.epoch(); const auto cursor = session->head->sampling;
  const auto image = s::image(model,*checkpoint);
  const auto unchanged = [&] {
    require(session->head.get() == head && session->head.epoch() == epoch && session->head->sampling == cursor &&
        !session->pending && s::image(model,*session->head) == image && s::image(model,*checkpoint) == image,
        "private conditional consumer changed an authoritative head, RNG or immutable complete checkpoint");
    t::idle(model);
  };
  std::vector<q::c::ReadLease> original_q;
  const auto verifier_ids = model.propose_recursive(*session,2,{},&original_q,nullptr,{},conditional);
  require(verifier_ids.size() == 3 && verifier_ids.front() == session->head->next_id && original_q.size() == 2,
      "conditional recursive proposer lost an actual verifier input or normalized q row");
  const std::array<std::int32_t,2> original_ids{verifier_ids[1],verifier_ids[2]};
  for (std::size_t at = 0; at != original_q.size(); ++at) {
    require(original_ids[at] >= 0 && payloads[at].allowed.at(static_cast<std::size_t>(original_ids[at])),
        "actual private conditional NextN proposal ignored its own prefix mask");
    q_support(model,original_q[at],payloads[at]);
  }
  unchanged();
  const std::array q_identities{g::identity(original_q[0]),g::identity(original_q[1])};
  const std::array actual_laws{probabilities(read(model,original_q[0])),probabilities(read(model,original_q[1]))};
  const auto draft_calls = model.draft_program_graph_calls;
  equal(model.sample_recursive_prefix(target.read(),original_ids,original_q,cursor,3,{},nullptr,0,{},conditional),
      reference(expected,actual_laws,original_ids,cursor,3,{},kind)); unchanged();
  for (std::size_t at = 0; at != 2; ++at)
    require(g::identity(original_q[at]) == q_identities[at],"conditional target terminal replaced or reprocessed an original q owner");
  require(model.draft_program_graph_calls == draft_calls,"conditional target terminal rebuilt the actual proposal distribution");
  auto invalid_proposal = inputs;
  invalid_proposal.back().allowed.description = q::dense(q::Type::kBool,{model.vocabulary});
  const std::array early_eos{original_ids[0]};
  bool proposal_preflight = false;
  try { (void)model.propose_recursive(*session,2,early_eos,&original_q,nullptr,{},
      std::span<const q::LogitsInputs>(invalid_proposal)); }
  catch (const std::runtime_error &) { proposal_preflight = true; }
  require(proposal_preflight && model.draft_program_graph_calls == draft_calls && original_q.size() == 2,
      "conditional proposer read a malformed later row or cleared original q before complete preflight");
  for (std::size_t at = 0; at != 2; ++at)
    require(g::identity(original_q[at]) == q_identities[at],"failed conditional proposal preflight replaced an original q owner");
  unchanged();
  {
    std::vector<q::c::ReadLease> early_q;
    const auto early = model.propose_recursive(*session,2,early_eos,&early_q,nullptr,{},conditional);
    require(early.size() == 2 && early[0] == verifier_ids[0] && early[1] == verifier_ids[1] && early_q.size() == 1,
        "conditional proposer did not retain its actual EOS-shortened width");
    q_support(model,early_q[0],payloads[0]);
    auto pair_target = model.caller_tensor(model.execution.device(),q::dense(q::Type::kFloat32,{2,model.vocabulary}));
    auto pair_values = raw[0]; pair_values.insert(pair_values.end(),raw[1].begin(),raw[1].end()); pair_target.write<float>(pair_values);
    const std::array early_law{probabilities(read(model,early_q[0]))};
    const auto early_ids = std::span<const std::int32_t>(early).subspan(1);
    equal(model.sample_recursive_prefix(pair_target.read(),early_ids,early_q,cursor,3,early_eos,nullptr,0,{},conditional.first(2)),
        reference(std::span<const std::vector<double>>(expected).first(2),early_law,early_ids,cursor,3,early_eos,kind));
    unchanged();
  }
  auto bad = inputs; bad[1].allowed.description = q::dense(q::Type::kBool,{model.vocabulary});
  const auto decisions = model.prefix_decision_graph_calls;
  bool bad_descriptor = false;
  try { (void)model.sample_recursive_prefix(target.read(),original_ids,original_q,cursor,3,{},nullptr,0,{},
      std::span<const q::LogitsInputs>(bad)); } catch (const std::runtime_error &) { bad_descriptor = true; }
  require(bad_descriptor && model.prefix_decision_graph_calls == decisions,"non-first conditional descriptor reached native execution"); unchanged();
  for (std::size_t malformed = 0; malformed != 3; ++malformed) {
    bad = inputs;
    if (malformed == 1) bad[1].counts = inputs[0].bias;
    if (malformed == 2) bad[1].allowed = {};
    bool rejected = false;
    try { (void)model.sample_recursive_prefix(target.read(),original_ids,original_q,cursor,3,{},nullptr,0,{},
        std::span<const q::LogitsInputs>(bad).first(malformed ? 3u : 2u)); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected && model.prefix_decision_graph_calls == decisions,
        "conditional missing-row/counts/presence mismatch reached native execution"); unchanged();
  }
  auto masked = row(model,{},0.F); bad = inputs; bad[1] = masked.inputs;
  bool numeric_failure = false;
  try { (void)model.sample_recursive_prefix(target.read(),original_ids,original_q,cursor,3,{},nullptr,0,{},
      std::span<const q::LogitsInputs>(bad)); }
  catch (const i::EvaluationFailure &failure) {
    numeric_failure = failure.status().IsInvalidArgument() && !failure.status().has_native_error() &&
        failure.retirement_observed() && failure.retirement().retired && failure.retirement().status.ok();
  }
  require(numeric_failure && model.prefix_decision_graph_calls == decisions+1,
      "non-first all-masked conditional row did not fail after actual native retirement"); unchanged();
  equal(model.sample_recursive_prefix(target.read(),original_ids,original_q,cursor,3,{},nullptr,0,{},conditional),
      reference(expected,actual_laws,original_ids,cursor,3,{},kind)); unchanged();
  original_q = {}; checkpoint.reset(); h::retire(session);
  require(!model.snapshot_versions.live && model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
      "conditional logits consumer retained a state or private append page");
  t::idle(model);
  std::cout << "{\"status\":\"PASS\",\"case\":\"conditional-logits\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"configurations\":3,\"conditional_rows\":3,\"finite_accept_reject_residual_bonus\":true,\"EOS_quota\":true,"
      "\"zero_proposal_bonus\":true,\"conditional_history_counts\":true,\"readonly_C0_unchanged\":true,\"greedy_conditional\":true,\"actual_conditional_proposer\":true,\"actual_private_q_rows\":2,\"same_q_leases\":true,\"head_rng_checkpoint_unchanged\":true,"
      "\"proposal_preflight_preserves_q\":true,\"EOS_shortened_prefix\":true,\"nonfirst_descriptor_zero_native\":true,\"conditional_metadata_failures\":4,\"nonfirst_numeric_retired\":true,\"same_terminal_retry\":true,"
      "\"full_target_publication\":false,\"natural_conditional_supplier\":false,\"synthetic_untrained\":true,"
      "\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0}\n";
  return 0;
}
} // namespace qwen_conditional_logits_case
