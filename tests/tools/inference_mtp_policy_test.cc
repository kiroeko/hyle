#include "../../engine/inference/mtp_policy.h"

#include <array>
#include <cstdio>
#include <stdexcept>

int main() {
  namespace m = hyle::inference::mtp;
  std::size_t checks = 0;
  constexpr std::array capacities{std::size_t{3}, std::size_t{4}, std::size_t{32768}, std::size_t{32769}};
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
    const m::DefaultRequest request{mode, flag(0), flag(1), flag(2), flag(3),
        sessions, active, capacity, flag(4), flag(5), flag(6),
        depth, flag(9)};
    const bool inspect = mode != m::RequestedMode::kOff && depth > 0 && request.automatic_backend_eligible &&
        request.greedy && request.automatic_vocabulary;
    if (m::inspect_prepared_vocabulary(request) != inspect)
      throw std::runtime_error("prepared asset inspection changed");
    for (const bool exists : {false, true}) for (const bool original : {false, true})
      for (const bool explicit_asset : {false, true}) {
      const bool prepared = inspect && exists;
      const bool source_allowed = mode != m::RequestedMode::kOff && depth > 0 &&
          request.automatic_backend_eligible && request.greedy;
      const bool route = request.natural_generation && !request.multiple_schedules &&
          !request.prefix_enabled && !request.scheduler_progress &&
          sessions <= 1 && active <= 1 && capacity >= 4 &&
          request.automatic_backend_eligible && !request.cpu_ffn && request.greedy;
      const bool automatic = (prepared || (source_allowed && (original || explicit_asset))) && route;
      const bool enabled = mode == m::RequestedMode::kOn || automatic;
      const auto selected = m::select_default(request, exists, original, explicit_asset);
      if (selected.automatic != automatic || selected.enabled != enabled ||
          selected.prepared_assets != (enabled && prepared))
        throw std::runtime_error("cold MTP policy changed an existing default or fallback");
      ++checks;
    }
  }
  m::DefaultRequest typical{.mode=m::RequestedMode::kAuto, .natural_generation=true,
      .capacity=32768, .automatic_backend_eligible=true, .greedy=true,
      .max_drafts=1, .automatic_vocabulary=true};
  if (m::select_default(typical,false).reason != "no_eligible_source" ||
      m::select_default(typical,false,true).reason != "original_projection" ||
      m::select_default(typical,true).reason != "prepared_vocabulary" ||
      m::select_default(typical,false,false,true).reason != "explicit_draft_asset")
    throw std::runtime_error("automatic source reason differs");
  const auto automatic=m::select_default(typical,true);
  const auto recommended=m::select_draft_depth(automatic,false,1,2);
  const auto wider=m::select_draft_depth(automatic,false,1,8);
  const auto explicit_depth=m::select_draft_depth(automatic,true,1,2);
  const auto unspecified=m::select_draft_depth(automatic,false,1,0);
  const m::DefaultSelection explicit_on{false,true,true,"explicit_on"};
  const auto explicit_on_recommendation=m::select_draft_depth(explicit_on,false,1,2);
  if (recommended.max_drafts!=2 || recommended.source!="artifact_recommendation" ||
      wider.max_drafts!=8 || wider.source!="artifact_recommendation" ||
      explicit_depth.max_drafts!=1 || explicit_depth.source!="explicit" ||
      unspecified.max_drafts!=1 || unspecified.source!="configured_default" ||
      explicit_on_recommendation.max_drafts!=2 || explicit_on_recommendation.source!="artifact_recommendation")
    throw std::runtime_error("source recommendation, capability or explicit depth priority differs");
  struct ConstraintCase final {
    void (*apply)(m::DefaultRequest &);
    std::string_view reason;
  };
  constexpr std::array constraints{
      ConstraintCase{+[](m::DefaultRequest &r) { r.natural_generation=false; }, "non_natural_request"},
      ConstraintCase{+[](m::DefaultRequest &r) { r.automatic_backend_eligible=false; }, "backend_policy"},
      ConstraintCase{+[](m::DefaultRequest &r) { r.greedy=false; }, "sampling_or_draft_threshold"},
      ConstraintCase{+[](m::DefaultRequest &r) { r.active_sequences=2; }, "multiple_sessions_or_sequences"},
      ConstraintCase{+[](m::DefaultRequest &r) { r.prefix_enabled=true; }, "prefix_cache"},
      ConstraintCase{+[](m::DefaultRequest &r) { r.scheduler_progress=true; }, "scheduler"},
      ConstraintCase{+[](m::DefaultRequest &r) { r.cpu_ffn=true; }, "cpu_ffn"},
      ConstraintCase{+[](m::DefaultRequest &r) { r.capacity=3; }, "insufficient_capacity"},
  };
  for (const auto &test : constraints) {
    auto request=typical;
    test.apply(request);
    const auto result=m::select_default(request,true);
    if (result.automatic || result.enabled || result.reason!=test.reason)
      throw std::runtime_error("automatic decision and constraint reason disagree");
  }
  auto multiple=typical;
  multiple.greedy=false; multiple.prefix_enabled=true;
  if (m::select_default(multiple,true).reason!="sampling_or_draft_threshold")
    throw std::runtime_error("first automatic constraint changed");
  typical.capacity=3;
  if (m::select_default(typical,true).enabled ||
      m::select_default(typical,true).reason != "insufficient_capacity")
    throw std::runtime_error("short-context auto policy differs");
  typical.mode=m::RequestedMode::kOn;
  if (!m::select_default(typical,true).enabled ||
      m::select_default(typical,true).reason != "explicit_on")
    throw std::runtime_error("short-context explicit route was blocked by auto policy");
  typical.mode=m::RequestedMode::kAuto; typical.capacity=32768;
  typical.max_drafts=0;
  if (m::select_default(typical,true).reason != "drafting_disabled")
    throw std::runtime_error("zero drafts should report the disabled policy");
  typical.greedy=false;
  if (m::select_default(typical,true).reason != "drafting_disabled")
    throw std::runtime_error("explicit zero should explain a disabled automatic route");
  typical.greedy=true;
  typical.max_drafts=1;
  typical.greedy=false;
  if (m::select_default(typical,true).reason != "sampling_or_draft_threshold")
    throw std::runtime_error("sampling fallback reason differs");
  typical.greedy=true; typical.max_drafts=4;
  if (!m::select_default(typical,true).automatic)
    throw std::runtime_error("explicit depth should retain the finite automatic route");
  typical.max_drafts=1;
  typical.mode=m::RequestedMode::kOff;
  if (m::select_default(typical,true).reason != "explicit_off")
    throw std::runtime_error("explicit off reason differs");
  std::printf("MTP cold policy: %zu configuration checks passed\n", checks);
}
