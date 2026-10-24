#pragma once

#include <stdexcept>
#include <vector>

#include "dspark_complete_graph_fixture.h"
#include "dspark_target_binding.h"
#include "dspark_vocabulary_fragment.h"

// Matching target-owned vocabulary, borrowed by the complete Flash draft.
// Original BF16 and canonical main-GGUF packed roles retain their own actual
// descriptors and source pins. The Q8 DSpark sidecar contains neither role;
// this shared math seam does not certify the complete target model artifact.
namespace dspark_shared_target_roles {
namespace b=dspark_target_binding;
namespace d=dspark_complete_graph;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
using hyle::Result;
using hyle::Status;

struct Roles final {b::ReadonlyRole embedding,head;};
struct Constant final {sem::ValueId value;b::ReadonlyRole role;};

inline Status qualify(const Roles &roles,const b::TargetIdentity &target) {
  if(target.model_repository!=b::kModelRepository || target.model_revision!=b::kModelRevision)
    return Status::InvalidArgument("DSpark shared vocabulary requires matching Flash0731 architecture");
  if(!dspark_vocabulary_fragment::embedding_table(roles.embedding.description,d::vocabulary,d::hidden)||
      (!dspark_vocabulary_fragment::bfloat_table(roles.head.description,d::vocabulary,d::hidden)&&
       !dspark_vocabulary_fragment::packed_head(roles.head.description,d::vocabulary,d::hidden)))
    return Status::InvalidArgument("DSpark shared vocabulary has a different full typed row/head layout");
  HYLE_RETURN_IF_ERROR(b::qualify_role(roles.embedding,"embed.weight",roles.embedding.description));
  return b::qualify_role(roles.head,"head.weight",roles.head.description);
}

// The fixed source RMSNorm returns BF16. The existing model fixture exposes
// its exact F32 widening for diagnostics. Recover that REAL producer value;
// never silently narrow a general F32 activation to make the head fit.
inline sem::ValueId bfloat_source(const sem::Fragment &f,sem::ValueId normalized) {
  const auto bf16=sem::dense({d::rows,d::hidden},c::DataType::kBFloat16);
  if(normalized.ordinal()>=f.descriptions.size())
    throw std::runtime_error("DSpark shared LM input is absent");
  if(f.description(normalized)==bf16) return normalized;
  if(f.description(normalized)==sem::dense({d::rows,d::hidden})) {
    for(auto node=f.nodes.rbegin();node!=f.nodes.rend();++node) {
      if(node->outputs.size()!=1 || node->outputs[0].ordinal()!=normalized.ordinal()) continue;
      const auto *cast=node->operation.get_if<c::Cast>();
      if(cast && cast->output_type==c::DataType::kFloat32 && node->inputs.size()==1 &&
          f.description(node->inputs[0])==bf16) return node->inputs[0];
      break;
    }
  }
  throw std::runtime_error("DSpark shared LM needs actual BF16 RMS output or its exact F32 widening");
}

// Callbacks are used only while build_flash constructs this one Fragment.
// The caller keeps constants alive through construction, then binds each
// ordinary Graph constant with its original StorageOwner and ReadLease.
inline Result<d::SharedParameters> parameters(const Roles &roles,
    const b::TargetIdentity &target,std::vector<Constant> &constants) {
  HYLE_RETURN_IF_ERROR(qualify(roles,target));
  d::SharedParameters supplied;
  supplied.embedding_table=[roles,&constants](sem::Fragment &f) {
    const auto table=f.constant(roles.embedding.description,{});
    constants.push_back({table,roles.embedding});
    return table;
  };
  supplied.logits=[roles,&constants](sem::Fragment &f,sem::ValueId normalized) {
    const auto input=bfloat_source(f,normalized);
    const auto weight=f.constant(roles.head.description,{});
    constants.push_back({weight,roles.head});
    // Dense BF16 keeps DenseLinearF32; encoded weights keep their native
    // packed RHS and quantize only the exact-widened actual activation.
    return dspark_vocabulary_fragment::append_head(f,input,weight,d::vocabulary,d::hidden);
  };
  return supplied;
}
} // namespace dspark_shared_target_roles
