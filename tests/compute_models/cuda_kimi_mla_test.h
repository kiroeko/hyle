#pragma once

namespace cuda_kimi_mla_test {
template<class Run>
void check(const hyle::compute::ModelArtifactProfileDescription *profile,Run run) {
  namespace sem=d5_qwen38_24t_semantic_fragments;
  d5_kimi_remaining_fragments::MlaReference source;
  auto fixture=d5_kimi_remaining_fragments::mla(profile,false,&source);
  // Use the identical public inputs on both sides of this stateful program.
  // StateFeedback feeds actual cache bytes back; a precomputed independent
  // trajectory is not the per-replay oracle for those different bytes.
  std::vector<std::vector<float>> inputs;
  for(const auto &sample:fixture.replays) {
    std::vector<float> input(sample.inputs[0].size()/sizeof(float));
    std::memcpy(input.data(),sample.inputs[0].data(),sample.inputs[0].size());
    inputs.push_back(std::move(input));
  }
  std::vector<float> cache(fixture.replays[0].inputs[1].size()/sizeof(float));
  std::memcpy(cache.data(),fixture.replays[0].inputs[1].data(),fixture.replays[0].inputs[1].size());
  sem::Fragment policy{.quantized=fixture.quantized,.absolute_tolerance=fixture.absolute_tolerance};
  policy.bfloat_chain_outputs=fixture.bfloat_chain_outputs;
  policy.exact_outputs=fixture.exact_outputs;
  std::vector<std::vector<float>> expected;std::size_t replay=0;
  fixture.observe=[&](std::size_t port,std::span<const float> values) {
    if(port!=0)return;
    ASSERT_LT(replay,inputs.size());
    expected=source(inputs[replay],cache);
    cache.assign(values.begin(),values.end());
  };
  fixture.validate_intermediate=[&](std::size_t port,std::span<const float> values) {
    EXPECT_LT(port,expected.size());
    if(port>=expected.size())return true;
    sem::expect_output(policy,values,expected[port],{},policy.exact_output(port),
                       nullptr,policy.bfloat_chain_output(port),port);
    if(port+1==expected.size())++replay;
    return true;
  };
  run(std::move(fixture));
  EXPECT_EQ(replay,inputs.size());
}
}
