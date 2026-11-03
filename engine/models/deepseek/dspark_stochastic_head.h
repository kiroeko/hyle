#pragma once

#include <limits>

#include "dspark_original_draft_program.h"
#include "../detail/model_sampling_fragment.h"
#include "../../inference/mtp_acceptance.h"

// Source-specific Markov head wiring only. All distribution, draws and
// speculative acceptance mathematics are shared with the inference engine.
namespace dspark_stochastic_head {
namespace own = dspark_original_formal_roles;
namespace sem = hyle::inference::model_definition;
namespace c = hyle::compute;
namespace s = hyle::inference::sampling;
using Value = sem::ValueId;
using s::fixed;
struct Proposal final {
  sem::Fragment fragment;
  std::vector<dspark_original_draft_program::Constant> constants;
  std::size_t ids = 0;
  std::vector<std::size_t> q, valid, adjusted, markov, confidence;
};
// Physical role storage and visible/projection activation types are separate.
// The source adapter selects its real store boundary, never a provider label.
struct HeadSource final {
  std::int64_t hidden=0,vocabulary=0;
  c::DataType hidden_type=c::DataType::kUnknown;
  c::DataType projection_input_type=c::DataType::kUnknown;
  std::array<own::ReadonlyRole,3> roles; // W1,W2,confidence: original BF16
};
Proposal proposal(const HeadSource &source,std::size_t count,s::Policy policy,
    const c::ModelArtifactProfileDescription *profile);
Proposal proposal(const own::Bound &bound,std::size_t count,s::Policy policy,
    const c::ModelArtifactProfileDescription *profile);

struct Verification final {
  sem::Fragment fragment;
  std::vector<dspark_original_draft_program::Constant> constants;
  std::vector<std::size_t> accepted, final_ids, valid;
};
Verification verification(std::size_t count,s::Policy policy,
    const c::ModelArtifactProfileDescription *profile);
} // namespace dspark_stochastic_head
