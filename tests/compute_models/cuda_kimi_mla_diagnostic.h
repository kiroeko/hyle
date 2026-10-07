#pragma once
#include <cstdio>

namespace cuda_kimi_mla_diagnostic {
template<class Run>
void check(const hyle::compute::ModelArtifactProfileDescription *profile,Run run) {
  namespace sem=d5_qwen38_24t_semantic_fragments;
  auto fixture=d5_kimi_remaining_fragments::mla(profile,false);
  const auto node=std::ranges::find_if(fixture.nodes,[](const auto &node) {
    return node.operation.template get_if<hyle::compute::ScaledDotProductAttention>()!=nullptr;
  });
  ASSERT_NE(node,fixture.nodes.end());
  const std::array extra{node->inputs[0],node->inputs[1]};
  fixture.outputs.insert(fixture.outputs.begin(),extra.begin(),extra.end());
  for(auto &port:fixture.bfloat_chain_outputs)port+=2;
  for(auto &ports:fixture.state_ports)ports.second+=2;
  for(auto &replay:fixture.replays)replay.outputs.insert(replay.outputs.begin(),2,std::vector<float>{});
  for(auto &replay:fixture.replays) {
    replay.outputs[0].resize(fixture.description(extra[0]).element_count().value());
    replay.outputs[1].resize(fixture.description(extra[1]).element_count().value());
  }
  fixture.validate_intermediate=[](std::size_t port,std::span<const float>) {return port<2;};
  std::vector<std::vector<float>> references;
  for(const auto &sample:fixture.replays)references.push_back(sample.outputs[5]);
  std::vector<float> query,cache;std::size_t replay=0;
  fixture.observe=[&](std::size_t port,std::span<const float> values) {
    if(port==0)query.assign(values.begin(),values.end());
    if(port==1)cache.assign(values.begin(),values.end());
    if(port!=5)return;
    const auto &expected=references[replay];
    double source_error=0,reference_error=0;std::size_t worst=0;
    for(std::size_t head=0;head<96;++head) {
      std::array<double,4> scores{};
      for(std::size_t token=0;token<4;++token) {
        for(std::size_t col=0;col<576;++col)scores[token]+=double(query[head*576+col])*cache[token*576+col];
        scores[token]*=1.0F/std::sqrt(192.0F);
      }
      const auto maximum=*std::max_element(scores.begin(),scores.end());double total=0;
      for(auto &score:scores)total+=score=std::exp(score-maximum);
      for(std::size_t col=0;col<512;++col) {
        double source=0;for(std::size_t token=0;token<4;++token)source+=scores[token]/total*cache[token*576+col];
        const auto index=head*512+col;
        const auto error=std::abs(double(values[index])-source);
        if(error>source_error){source_error=error;worst=index;}
        reference_error=std::max(reference_error,std::abs(double(expected[index])-source));
      }
    }
    std::fprintf(stderr,"KIMI_MLA_SOURCE profile=%.*s replay=%zu source_max=%.12g reference_max=%.12g worst=%zu\n",
      int(profile->variant.size()),profile->variant.data(),replay++,source_error,reference_error,worst);
  };
  run(std::move(fixture));
}
}
