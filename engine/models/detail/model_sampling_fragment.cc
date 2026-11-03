#include "model_sampling_fragment.h"

namespace model_sampling_fragment {
RandomProposalPorts append_proposal(sem::Fragment &fragment,
    Value actual_nextn_logits, Value proposal_identity,
    hyle::inference::sampling::Policy policy ) {
  Builder r(fragment);
  const auto q = hyle::inference::sampling::distribution(r, actual_nextn_logits, policy);
  return {r.copy(q.log_probabilities),
      r.copy(r.one(c::sampling_ops::categorical(), {q.log_probabilities, proposal_identity})),
      r.copy(q.valid)};
}
}

namespace model_sampling_fragment {
RandomDecisionPorts append_decision(sem::Fragment &fragment,
    Value actual_target_logits, Value actual_log_q, Value actual_proposal,
    Value accept_identity, Value final_identity,
    hyle::inference::sampling::Policy policy ) {
  Builder r(fragment);
  const auto target = hyle::inference::sampling::distribution(r,
      r.copy(r.slice(actual_target_logits, 0, hyle::inference::sampling::fixed(0),
          hyle::inference::sampling::fixed(1))), policy);
  const auto bonus = hyle::inference::sampling::distribution(r,
      r.copy(r.slice(actual_target_logits, 0, hyle::inference::sampling::fixed(1),
          hyle::inference::sampling::fixed(1))), policy);
  const auto selected = hyle::inference::sampling::decision(r,
      target.log_probabilities, actual_log_q, bonus.log_probabilities,
      actual_proposal, accept_identity, final_identity);
  const auto no = r.cast(r.scalar(0), c::DataType::kBool);
  const auto valid = r.one(c::Select{}, {target.valid,
      r.one(c::Select{}, {bonus.valid, selected.valid, no}), no});
  return {r.copy(selected.accepted), r.copy(selected.final_id), r.copy(valid),
      r.copy(target.log_probabilities), r.copy(bonus.log_probabilities)};
}
}
