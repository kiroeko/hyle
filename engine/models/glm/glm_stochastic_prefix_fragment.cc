#include "glm_stochastic_prefix_fragment.h"
#include "../detail/model_sampling_fragment.h"

namespace glm_stochastic_prefix_fragment {
namespace c=hyle::compute;
namespace sem=hyle::inference::model_definition;
void append(sem::Fragment &fragment, std::int64_t vocabulary) {
  model_sampling_fragment::Builder r(fragment);
  const auto logits = fragment.input(sem::dense({3,vocabulary}));
  const auto identities = fragment.input(sem::dense({3,4},c::DataType::kUInt32));
  const auto proposals = fragment.input(sem::dense({2},c::DataType::kInt32));
  const auto selected = hyle::inference::sampling::speculative_prefix(r,logits,identities,proposals);
  for(std::size_t i=0;i<2;++i) {
    fragment.outputs.push_back(r.copy(selected.accepted[i]));
    fragment.outputs.push_back(r.copy(selected.final_ids[i]));
    fragment.outputs.push_back(r.copy(selected.valid[i]));
  }
}

}
