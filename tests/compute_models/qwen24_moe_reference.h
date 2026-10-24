#pragma once

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>

#include "qwen24_original_moe_fragment.h"
#include "qwen24_original_nextn_reference.h"

// Same Qwen512/Top10 mathematical contract for distinct target/NextN original
// banks. Original-byte/actual-codec projections use the existing AQ certificate.
namespace qwen24_moe_reference {
namespace packed=qwen24_original_nextn_reference;
namespace model=qwen24_original_moe_fragment;
namespace numeric=d5_qwen35_gguf_fragment;
struct Parameters final {
  std::span<const std::byte> router,shared_router;
  std::array<packed::PackedMatrix,3> shared; // gate,up,down
};
template<class ReadFloat,class ReadInt,class ExpertMatrix>
void observe(const Parameters &parameters,const model::Ports &ports,ReadFloat &&get,ReadInt &&integers,
    ExpertMatrix &&expert_matrix,packed::ActivationCodec codec,bool full,std::size_t &full_values) {
  const auto input=get(ports.input),logits=get(ports.logits),probabilities=get(ports.probabilities);
  const auto selected=get(ports.selected_scores),sum=get(ports.sum),denominator=get(ports.denominator),weights=get(ports.weights);
  const auto ids=integers(ports.ids),local_ids=integers(ports.local_ids),mapping=integers(ports.map);
  ASSERT_TRUE(packed::expect_f32_projection(logits,parameters.router,512,input,"original Q24 F32 router"));
  if(full)full_values+=logits.size();
  ASSERT_TRUE(packed::expect_routing({logits,probabilities,selected,sum,denominator,weights,ids}));
  ASSERT_TRUE(packed::expect_map(ids,local_ids,mapping));
  const auto grouped=[&](std::size_t leg,std::span<const float> actual,std::span<const float> actual_input,const char *label) {
    const auto source_row=[&](std::int32_t global){return expert_matrix(leg,global);};
    const auto passed=packed::expect_selected_projection(actual,actual_input,ids,source_row,codec,label,full);
    if(passed&&full)full_values+=actual.size();return passed;
  };
  const auto gate=get(ports.routed.gate),up=get(ports.routed.up),activation=get(ports.routed.activation),down=get(ports.routed.down);
  ASSERT_TRUE(grouped(0,gate,input,"original Q24 selected gate"));ASSERT_TRUE(grouped(1,up,input,"original Q24 selected up"));
  ASSERT_TRUE(packed::expect_swiglu(activation,gate,up,"original Q24 actual routed SwiGLU"));
  ASSERT_TRUE(grouped(2,down,activation,"original Q24 selected down"));
  const auto weighted=get(ports.weighted),routed_sum=get(ports.routed_sum);
  ASSERT_TRUE(packed::expect_product(weighted,down,weights,8192,"original Q24 down-post route weight"));
  ASSERT_TRUE(packed::expect_sum(routed_sum,weighted,8192,"original Q24 selected10 reduction"));
  const auto matrix=[&](std::size_t leg,std::span<const float> actual,std::span<const float> actual_input) {
    const auto &source=parameters.shared[leg];
    const auto passed=packed::expect_projection(actual,source.bytes,source.outputs,source.width,source.type,actual_input,codec,"original Q24 shared projection",full);
    if(passed&&full)full_values+=actual.size();return passed;
  };
  const auto shared_gate=get(ports.shared.gate),shared_up=get(ports.shared.up),shared_activation=get(ports.shared.activation),shared_down=get(ports.shared.down);
  ASSERT_TRUE(matrix(0,shared_gate,input));ASSERT_TRUE(matrix(1,shared_up,input));
  ASSERT_TRUE(packed::expect_swiglu(shared_activation,shared_gate,shared_up,"original Q24 actual shared SwiGLU"));
  ASSERT_TRUE(matrix(2,shared_down,shared_activation));
  const auto raw=get(ports.shared_gate_raw),sigmoid=get(ports.shared_gate_sigmoid),shared_gated=get(ports.shared_gated);
  ASSERT_TRUE(packed::expect_f32_projection(raw,parameters.shared_router,1,input,"original Q24 F32 shared gate"));
  if(full)full_values+=raw.size();ASSERT_EQ(raw.size(),1u);ASSERT_EQ(sigmoid.size(),1u);
  const std::array<float,1> expected{float(1.0/(1.0+std::exp(-double(raw[0]))))};numeric::expect_close(sigmoid,expected,8e-5F,4e-5F);
  ASSERT_TRUE(packed::expect_product(shared_gated,shared_down,sigmoid,8192,"original Q24 gated shared down"));
  auto terms=routed_sum;terms.insert(terms.end(),shared_gated.begin(),shared_gated.end());
  ASSERT_TRUE(packed::expect_sum(get(ports.merged),terms,8192,"original Q24 routed plus shared return"));
}
} // namespace qwen24_moe_reference
