#include "../../engine/inference/mtp_policy.h"
#include "../../engine/models/qwen35/mtp_default_policy.h"
#include "../../engine/models/qwen35/decode_route_policy.h"

#include <array>
#include <cstdio>
#include <stdexcept>

using RouteReason = hyle::inference::RouteReason;

void check_decode_routes() {
  using namespace hyle::inference;
  using namespace qwen35;
  for (unsigned bits=0; bits<(1u<<8); ++bits)
    for (const auto drafts : {std::size_t{0},std::size_t{1},std::size_t{2},std::size_t{3},std::size_t{4}})
      for (const auto active : {std::size_t{0},std::size_t{1},std::size_t{2}}) {
        const auto f=[bits](unsigned bit){return (bits & (1u<<bit)) != 0;};
        const ColdWorkloadFacts workload{.natural_generation=f(1), .state_trace=f(6)};
        const DecodeRouteFacts x{workload,f(0),f(2),drafts,3,f(3),f(4),RouteReason::kNone,f(5),f(7),active};
        const bool overlap=x.overlap_requested && x.workload.natural_generation &&
          (!x.mtp_enabled || (drafts>=1 && drafts<=3 &&
                              (drafts==1 || x.stored_draft_head))) && x.relay_eligible &&
          !x.commands_replay && !x.workload.state_trace && x.has_decode;
        const bool verify=x.mtp_enabled && drafts>1 && drafts<=3 &&
          x.relay_eligible && active==1 && !x.commands_replay && x.has_decode;
        const auto [o,v]=plan_decode_routes(x);
        if (o.enabled!=overlap || v.enabled!=verify || o.reason == RouteReason::kNone || v.reason == RouteReason::kNone ||
            (o.reason==RouteReason::kSelected)!=o.enabled || (v.reason==RouteReason::kSelected)!=v.enabled)
          throw std::runtime_error("cold decode route choice differs from its existing conditions");
        if (v.enabled && !x.relay_eligible)
          throw std::runtime_error("device verification requires relay capability");
        for (unsigned flags=0; flags<16; ++flags) for (const auto cached : {std::size_t{0},std::size_t{1},std::size_t{2}}) {
          const bool enabled=(flags&1)!=0, prefix=(flags&2)!=0, attention=(flags&4)!=0, exact=(flags&8)!=0;
          auto continuous_facts=x;
          continuous_facts.workload.prefix_enabled=prefix;
          const auto c=select_continuous_mtp(continuous_facts,enabled,cached,attention,exact);
          const bool expected=enabled && x.mtp_enabled && !prefix && cached==1 && active==1 && attention && exact;
          if (c.enabled!=expected || c.reason == RouteReason::kNone || (c.reason==RouteReason::kSelected)!=c.enabled)
            throw std::runtime_error("continuous MTP route differs from its existing conditions");
          const auto planned=select_continuous_mtp(continuous_facts,o.enabled,cached,attention,exact);
          if (planned.enabled && !o.enabled)
            throw std::runtime_error("continuous MTP requires a selected overlap route");
        }
      }
  DecodeRouteFacts facts{.workload={.natural_generation=true},
      .overlap_requested=true, .mtp_enabled=true, .max_drafts=3, .max_device_drafts=3,
      .stored_draft_head=true, .relay_eligible=false, .relay_refusal=RouteReason::kScheduler, .has_decode=true,
      .resolved_active_sequences=1};
  if (select_decode_overlap(facts).reason!=RouteReason::kScheduler || select_device_verification(facts).reason!=RouteReason::kScheduler)
    throw std::runtime_error("relay capability refusal reason was lost");
  facts.relay_eligible=true; facts.max_drafts=4;
  if (select_decode_overlap(facts).reason!=RouteReason::kDeviceChainCapacity ||
      select_device_verification(facts).reason!=RouteReason::kDeviceChainCapacity)
    throw std::runtime_error("unsupported chain must retain its fallback reason");

  // The verifier packet can be useful without overlap or the compound chain.
  facts.max_drafts=3; facts.stored_draft_head=false;
  if (select_decode_overlap(facts).reason!=RouteReason::kDraftProjection || !select_device_verification(facts).enabled)
    throw std::runtime_error("packet verification must remain independent of the compound projection");
  facts.stored_draft_head=true; facts.workload.state_trace=true;
  if (select_decode_overlap(facts).reason!=RouteReason::kStateTrace || !select_device_verification(facts).enabled)
    throw std::runtime_error("tracing must retain capacity for an independent draft chain");
  facts.workload.state_trace=false;
  facts.overlap_requested=false;
  const auto independent = plan_decode_routes(facts);
  if (independent.overlap.reason!=RouteReason::kUserDisabled || !independent.verification.enabled)
    throw std::runtime_error("disabling overlap must retain independent device verification");
  facts.overlap_requested=true;
  facts.has_decode=false;
  if (select_decode_overlap(facts).reason!=RouteReason::kNoDecode || select_device_verification(facts).reason!=RouteReason::kNoDecode)
    throw std::runtime_error("a finite workload without decode must not plan unused verification");

  DecodeRouteFacts resident{.workload={.natural_generation=true, .greedy=false,
      .max_sessions=1, .active_sequences=1, .capacity=128},
      .overlap_requested=true, .relay_eligible=true, .has_decode=true, .resolved_active_sequences=1};
  if (!select_resident_decode_overlap(resident).enabled)
    throw std::runtime_error("ordinary random sampling should use available single-session relay");
  resident.workload.greedy=true;
  if (!select_resident_decode_overlap(resident).enabled)
    throw std::runtime_error("ordinary greedy single-session relay changed");
  resident.workload.max_sessions=2;
  if (select_resident_decode_overlap(resident).reason!=RouteReason::kMultipleSessionsOrSequences)
    throw std::runtime_error("multi-session resident route must not select single-session relay");
  resident.workload.max_sessions=1; resident.workload.active_sequences=2;
  if (select_resident_decode_overlap(resident).reason!=RouteReason::kMultipleSessionsOrSequences)
    throw std::runtime_error("multi-sequence resident route must not select single-session relay");
  resident.workload.active_sequences=1; resident.relay_eligible=false;
  resident.relay_refusal=RouteReason::kSampling;
  if (select_resident_decode_overlap(resident).reason!=RouteReason::kSampling)
    throw std::runtime_error("resident route lost its backend sampling refusal");

  BoundedGenerationFacts bounded{.workload={.natural_generation=true, .greedy=true,
      .capacity=128}, .supported_backend=true, .mtp_enabled=true,
      .max_drafts=3};
  if (!select_bounded_generation(bounded).enabled) throw std::runtime_error("bounded route unavailable");
  bounded.workload.planned_shapes=true;
  if (select_bounded_generation(bounded).reason!=RouteReason::kPlannedShapes)
    throw std::runtime_error("planned shapes need an observable finite route");
  bounded.workload.planned_shapes=false; bounded.workload.active_sequences=2;
  if (select_bounded_generation(bounded).reason!=RouteReason::kMultipleSessionsOrSequences)
    throw std::runtime_error("bounded route admitted multiple active sequences");
  bounded.workload.active_sequences=1;
  bounded.mtp_enabled=false;
  if (!select_bounded_generation(bounded).enabled)
    throw std::runtime_error("ordinary bounded generation was not selected");

  const DecodeBudgetRoutes initial{{true,RouteReason::kSelected},{true,RouteReason::kSelected},3,RouteReason::kSelected};
  // A synthetic monotonic capacity exposes each ordered fallback, without
  // depending on a particular model's byte formula.
  for (const int budget : {6,5,4,3,2,1}) {
    const auto routes=select_decode_budget_routes(initial,[budget](const auto &r) {
      return (r.verification.enabled ? 2 : 0) + (r.overlap.enabled ? int(r.overlap_depth) : 1) <= budget;
    });
    if (routes.verification.enabled!=(budget>=5) || routes.overlap.enabled!=(budget>=2) ||
        routes.overlap_depth!=(budget>=3 ? std::size_t{3} : std::size_t{2}))
      throw std::runtime_error("cache budget fallback order changed");
  }
  const auto single=select_decode_budget_routes(initial,[](const auto &r) { return !r.overlap.enabled; });
  if (single.overlap.enabled || single.verification.enabled || single.overlap_depth!=2 ||
      single.overlap.reason!=RouteReason::kCacheBudget || single.verification.reason!=RouteReason::kCacheBudget ||
      single.overlap_depth_reason!=RouteReason::kCacheBudget)
    throw std::runtime_error("cache budget failed to preserve fallback reasons");
}

int main() {
  namespace m = hyle::inference::mtp;
  if (!hyle::inference::route_reason_name(RouteReason::kNone).empty() ||
      hyle::inference::route_reason_name(RouteReason::kSampling) != "sampling" ||
      hyle::inference::route_reason_name(RouteReason::kCacheBudget) != "cache_budget" ||
      hyle::inference::route_reason_name(RouteReason::kSelected) != "selected")
    throw std::runtime_error("route reason serialization changed");
  std::size_t checks = 0;
  constexpr std::array capacities{std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{32768}, std::size_t{32769}};
  constexpr std::array counts{std::size_t{0}, std::size_t{1}, std::size_t{2}};
  constexpr std::array drafts{std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{4}};
  constexpr std::array modes{m::RequestedMode::kOff, m::RequestedMode::kAuto, m::RequestedMode::kOn};
  for (unsigned bits = 0; bits < (1u << 10); ++bits)
    for (const auto capacity : capacities)
      for (const auto sessions : counts)
        for (const auto active : counts)
          for (const auto depth : drafts)
            for (const auto mode : modes) {
    const auto flag = [bits](unsigned bit) { return (bits & (1u << bit)) != 0; };
    const hyle::inference::ColdWorkloadFacts workload{
        .natural_generation=flag(0), .multiple_schedules=flag(1), .prefix_enabled=flag(2),
        .scheduler_progress=flag(3), .cpu_assist=flag(5), .greedy=flag(6),
        .max_sessions=sessions, .active_sequences=active, .capacity=capacity};
    const qwen35::MtpDefaultRequest request{mode, workload, flag(4), depth, flag(9)};
    const bool inspect = mode != m::RequestedMode::kOff && depth > 0 && request.automatic_backend_eligible &&
        request.workload.greedy && request.automatic_vocabulary;
    if (qwen35::inspect_prepared_vocabulary(request) != inspect)
      throw std::runtime_error("prepared asset inspection changed");
    for (const bool exists : {false, true}) for (const bool original : {false, true})
      for (const bool explicit_asset : {false, true}) {
      const bool prepared = inspect && exists;
      const bool source_allowed = mode != m::RequestedMode::kOff && depth > 0 &&
          request.automatic_backend_eligible && request.workload.greedy;
      const bool route = request.workload.natural_generation && !request.workload.multiple_schedules &&
          !request.workload.prefix_enabled && !request.workload.scheduler_progress &&
          sessions <= 1 && active <= 1 && capacity >= 2 &&
          request.automatic_backend_eligible && !request.workload.cpu_assist && request.workload.greedy;
      const bool automatic = (prepared || (source_allowed && (original || explicit_asset))) && route;
      const bool enabled = mode == m::RequestedMode::kOn || automatic;
      const auto selected = qwen35::select_mtp_default(request, exists, original, explicit_asset);
      if (selected.automatic != automatic || selected.enabled != enabled ||
          selected.prepared_assets != (enabled && prepared))
        throw std::runtime_error("cold MTP policy changed an existing default or fallback");
      const auto bounded = qwen35::select_bounded_generation(
          {workload, request.automatic_backend_eligible, selected.enabled, depth});
      if (bounded.enabled && (workload.multiple_schedules || sessions > 1 || active > 1))
        throw std::runtime_error("bounded generation requires a single session");
      ++checks;
    }
  }
  qwen35::MtpDefaultRequest typical{.mode=m::RequestedMode::kAuto,
      .workload={.natural_generation=true, .greedy=true, .capacity=32768},
      .automatic_backend_eligible=true, .max_drafts=1, .automatic_vocabulary=true};
  if (qwen35::select_mtp_default(typical,false).reason != RouteReason::kNoEligibleSource ||
      qwen35::select_mtp_default(typical,false,true).reason != RouteReason::kOriginalProjection ||
      qwen35::select_mtp_default(typical,true).reason != RouteReason::kPreparedVocabulary ||
      qwen35::select_mtp_default(typical,false,false,true).reason != RouteReason::kExplicitDraftAsset)
    throw std::runtime_error("automatic source reason differs");
  const auto automatic=qwen35::select_mtp_default(typical,true);
  const auto recommended=m::select_draft_depth(automatic,false,1,2);
  const auto wider=m::select_draft_depth(automatic,false,1,8);
  const auto explicit_depth=m::select_draft_depth(automatic,true,1,2);
  const auto unspecified=m::select_draft_depth(automatic,false,1,0);
  const m::DefaultSelection explicit_on{false,true,true,RouteReason::kExplicitOn};
  const auto explicit_on_recommendation=m::select_draft_depth(explicit_on,false,1,2);
  if (recommended.max_drafts!=2 || recommended.source!=RouteReason::kArtifactRecommendation ||
      wider.max_drafts!=8 || wider.source!=RouteReason::kArtifactRecommendation ||
      explicit_depth.max_drafts!=1 || explicit_depth.source!=RouteReason::kExplicit ||
      unspecified.max_drafts!=1 || unspecified.source!=RouteReason::kConfiguredDefault ||
      explicit_on_recommendation.max_drafts!=2 || explicit_on_recommendation.source!=RouteReason::kArtifactRecommendation)
    throw std::runtime_error("source recommendation, capability or explicit depth priority differs");
  struct ConstraintCase final {
    void (*apply)(qwen35::MtpDefaultRequest &);
    RouteReason reason;
  };
  constexpr std::array constraints{
      ConstraintCase{+[](qwen35::MtpDefaultRequest &r) { r.workload.natural_generation=false; }, RouteReason::kNonNaturalRequest},
      ConstraintCase{+[](qwen35::MtpDefaultRequest &r) { r.automatic_backend_eligible=false; }, RouteReason::kBackendPolicy},
      ConstraintCase{+[](qwen35::MtpDefaultRequest &r) { r.workload.greedy=false; }, RouteReason::kSampling},
      ConstraintCase{+[](qwen35::MtpDefaultRequest &r) { r.workload.active_sequences=2; }, RouteReason::kMultipleSessionsOrSequences},
      ConstraintCase{+[](qwen35::MtpDefaultRequest &r) { r.workload.prefix_enabled=true; }, RouteReason::kPrefixCache},
      ConstraintCase{+[](qwen35::MtpDefaultRequest &r) { r.workload.scheduler_progress=true; }, RouteReason::kScheduler},
      ConstraintCase{+[](qwen35::MtpDefaultRequest &r) { r.workload.cpu_assist=true; }, RouteReason::kCpuAssist},
      ConstraintCase{+[](qwen35::MtpDefaultRequest &r) { r.workload.capacity=1; }, RouteReason::kInsufficientCapacity},
  };
  for (const auto &test : constraints) {
    auto request=typical;
    test.apply(request);
    const auto result=qwen35::select_mtp_default(request,true);
    if (result.automatic || result.enabled || result.reason!=test.reason)
      throw std::runtime_error("automatic decision and constraint reason disagree");
  }
  const m::DefaultRequest other_provider{m::RequestedMode::kAuto, RouteReason::kNone};
  if (!m::select_default(other_provider, true).automatic)
    throw std::runtime_error("common selection imposed a model-specific automatic restriction");
  const m::DefaultRequest explicit_provider{m::RequestedMode::kOn, RouteReason::kMultipleSessionsOrSequences};
  if (!m::select_default(explicit_provider, true).enabled)
    throw std::runtime_error("default preference blocked explicit MTP");
  auto multiple=typical;
  multiple.workload.greedy=false; multiple.workload.prefix_enabled=true;
  if (qwen35::select_mtp_default(multiple,true).reason!=RouteReason::kSampling)
    throw std::runtime_error("first automatic constraint changed");
  typical.workload.capacity=1;
  if (qwen35::select_mtp_default(typical,true).enabled ||
      qwen35::select_mtp_default(typical,true).reason != RouteReason::kInsufficientCapacity)
    throw std::runtime_error("short-context auto policy differs");
  typical.mode=m::RequestedMode::kOn;
  if (!qwen35::select_mtp_default(typical,true).enabled ||
      qwen35::select_mtp_default(typical,true).reason != RouteReason::kExplicitOn)
    throw std::runtime_error("short-context explicit route was blocked by auto policy");
  typical.mode=m::RequestedMode::kAuto; typical.workload.capacity=2;
  if (!qwen35::select_mtp_default(typical,true).enabled)
    throw std::runtime_error("two-row auto policy was blocked by the old K3 preference");
  typical.mode=m::RequestedMode::kAuto; typical.workload.capacity=32768;
  typical.max_drafts=0;
  if (qwen35::select_mtp_default(typical,true).reason != RouteReason::kDraftingDisabled)
    throw std::runtime_error("zero drafts should report the disabled policy");
  typical.workload.greedy=false;
  if (qwen35::select_mtp_default(typical,true).reason != RouteReason::kDraftingDisabled)
    throw std::runtime_error("explicit zero should explain a disabled automatic route");
  typical.workload.greedy=true;
  typical.max_drafts=1;
  typical.workload.greedy=false;
  if (qwen35::select_mtp_default(typical,true).reason != RouteReason::kSampling)
    throw std::runtime_error("sampling fallback reason differs");
  typical.workload.greedy=true; typical.max_drafts=4;
  if (!qwen35::select_mtp_default(typical,true).automatic)
    throw std::runtime_error("explicit depth should retain the finite automatic route");
  typical.max_drafts=1;
  typical.mode=m::RequestedMode::kOff;
  if (qwen35::select_mtp_default(typical,true).reason != RouteReason::kExplicitOff)
    throw std::runtime_error("explicit off reason differs");
  check_decode_routes();
  {
    using namespace qwen35;
    DecodeRouteFacts greedy{.workload = {.natural_generation = true, .greedy = true,
        .max_sessions = 1, .active_sequences = 1, .capacity = 128},
        .overlap_requested = true, .mtp_enabled = true, .max_drafts = 3,
        .max_device_drafts = 3, .stored_draft_head = true, .relay_eligible = true,
        .has_decode = true, .resolved_active_sequences = 1};
    auto random = greedy;
    random.workload.greedy = false; random.relay_eligible = false; random.relay_refusal = RouteReason::kSampling;
    for (const auto choices : {std::array{greedy, random}, std::array{random, greedy}}) {
      const auto prepared = plan_prepared_decode_routes(choices);
      if (!prepared.overlap || !prepared.verification || prepared.by_sampling.size() != 2)
        throw std::runtime_error("default sampling order changed the cold resource union");
      for (std::size_t index = 0; index < choices.size(); ++index)
        if (prepared.by_sampling[index].overlap.enabled != choices[index].relay_eligible ||
            prepared.by_sampling[index].verification.enabled != choices[index].relay_eligible)
          throw std::runtime_error("one sampling program inherited another program's route");
      const auto disabled = plan_prepared_decode_routes(choices, false);
      if (disabled.overlap || !disabled.verification)
        throw std::runtime_error("explicit overlap disable changed independent verification capacity");
    }
    const auto sampled_only = plan_prepared_decode_routes(std::array{random}, true);
    if (sampled_only.overlap || sampled_only.verification)
      throw std::runtime_error("explicit overlap enable waived actual route eligibility");
  }
  std::printf("MTP cold policy: %zu configuration checks and decode route combinations passed\n", checks);
}
