#include <gtest/gtest.h>

#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "../compute_models/sampling_math_fixture.h"
#include "../../src/compute_route_apple/apple_sampling_math.h"

TEST(AppleSamplingMath, ActualProductionTransformAtOpen64EndpointsAndCorpus) {
  namespace mx=mlx::core;
  const auto words=sampling_test::endpoint_words();
  const auto count=int(words.size()/2);
  const auto input=mx::array(words.data(),{count,2},mx::uint32);
  const auto kernel=mx::fast::metal_kernel("hyle_sampling_math_probe",{"words"},{"output"},R"METAL(
    uint i=thread_position_in_grid.x;
    if(i<uint(words_shape[0]))output[i]=hyle_gumbel(words[i*2],words[i*2+1]);
  )METAL",hyle::compute::apple::detail::kSamplingMetalHeader,true,false,mx::CompileOptions{});
  auto result=kernel({input},{{count}},{mx::float32},
      {(count+127)/128*128,1,1},{128,1,1},{},std::nullopt,false,mx::default_stream(mx::Device::gpu))[0];
  mx::async_eval({result});mx::synchronize(mx::default_stream(mx::Device::gpu));result.wait();
  EXPECT_NO_THROW(sampling_test::check_endpoints(words,{result.data<float>(),std::size_t(count)}));
}
