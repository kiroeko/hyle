#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

#include "../compute_models/d5_quantization_roundtrip.h"
#include "tensor_test_support.h"

namespace apple_fp4_fidelity_test {
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;
using Bf = d5_qwen38_24t_semantic_fragments::q::BF16;
struct Case {
  const char *name;
  std::vector<std::uint16_t> input, expected;
};
inline std::uint16_t encode(float value) {
  return d5_qwen38_24t_semantic_fragments::q::encode(value).bits;
}
inline void expected(Case &value) {
  std::vector<float> decoded;
  decoded.reserve(value.input.size());
  for (auto bits : value.input) decoded.push_back(std::bit_cast<float>(std::uint32_t(bits) << 16));
  const auto output = d5_quantization_roundtrip::reference(decoded, true);
  value.expected.reserve(output.size());
  for (float x : output) value.expected.push_back(encode(x));
}
inline std::vector<Case> cases() {
  std::vector<Case> result{{"all-finite-mixed-groups",{},{}},
      {"all-finite-uniform-groups",{},{}},{"midpoints-signed-zeros-tiny-mixed-max",{},{}}};
  for (std::uint32_t bits = 0; bits <= 0xffff; ++bits) {
    if ((bits & 0x7f80) == 0x7f80) continue;
    result[0].input.push_back(std::uint16_t(bits));
    for (int i = 0; i < 32; ++i) result[1].input.push_back(std::uint16_t(bits));
  }
  constexpr std::array<float,7> midpoint{.25F,.75F,1.25F,1.75F,2.5F,3.5F,5};
  for (int exponent : {-126,-125,-124,-20,0,8,125}) for (float mid : midpoint)
    for (bool negative : {false,true}) for (int neighbor : {-1,0,1}) {
      const float scale = std::ldexp(1.0F,exponent);
      const auto center = encode(mid*scale);
      const auto sample = std::uint16_t((int(center)+neighbor) | (negative?0x8000:0));
      for(int i=0;i<32;++i) result[2].input.push_back(i==31?encode(6*scale):sample);
    }
  for(int maximum : {0x0000,0x0001,0x0040,0x007f,0x0080,0x01c0,0x0240,0x7f7f}) {
    for(int i=0;i<32;++i) {
      const std::array<std::uint16_t,8> tiny{0,0x8000,1,0x8001,0x40,0x8040,0x7f,0x807f};
      result[2].input.push_back(i==31?std::uint16_t(maximum):tiny[i%tiny.size()]);
    }
  }
  for(auto &value:result) expected(value);
  return result;
}
inline void check_bits(const mx::array &output,std::span<const std::uint16_t> expected) {
  ASSERT_EQ(output.dtype(),mx::bfloat16);
  ASSERT_EQ(output.size(),expected.size());
  const auto *raw=output.data<std::uint16_t>();
  std::size_t mismatch=0,first=0;
  for(std::size_t i=0;i<expected.size();++i) if(raw[i]!=expected[i]) {if(!mismatch)first=i;++mismatch;}
  EXPECT_EQ(mismatch,0u) << "first=" << first << " actual=" << raw[first] << " expected=" << expected[first];
}
template<class Compile> void check(Compile compile) {
  for(auto value:cases()) {
    SCOPED_TRACE(value.name);
    auto context=a::Context::create().value();
    const auto description=c::TensorDesc::dense(c::DataType::kBFloat16,
        std::array<std::int64_t,2>{std::int64_t(value.input.size()/32),32}).value();
    const auto operation=c::model_ops::block_quantization_roundtrip(
        c::model_ops::BlockQuantizationFormat::kFloat4E2M1Block32);
    auto program=compile(context,operation,description).value();
    std::optional<c::ResultLease<a::Results>> old;
    std::vector<std::uint16_t> previous;
    for(int pass=0;pass<2;++pass) {
      if(pass) {std::reverse(value.input.begin(),value.input.end());std::reverse(value.expected.begin(),value.expected.end());}
      const mx::Shape shape{int(value.input.size()/32),32};
      auto input=apple_tensor_test::ready(mx::view(mx::array(value.input.data(),shape,mx::uint16),
          mx::bfloat16,apple_tensor_test::stream()));
      auto invocation=program.create_invocation().value();
      ASSERT_TRUE(invocation.bind(std::array{input}).ok());
      auto work=invocation.submit();ASSERT_TRUE(work) << work.error().message();
      auto result=work->wait();ASSERT_TRUE(result) << result.error().message();
      check_bits(result->get()->arrays()[0],value.expected);
      if(old) check_bits(old->get()->arrays()[0],previous);
      old=std::move(*result);previous=value.expected;
    }
    ASSERT_TRUE(context.close().ok());
    check_bits(old->get()->arrays()[0],previous);
  }
}
} // namespace apple_fp4_fidelity_test
