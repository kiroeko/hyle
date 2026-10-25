#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "original_readonly_role.h"
#include "qwen_nvfp4_original_reference.h"
#include "static_fp8_reference_fixture.h"

// One connected ordinary source fragment: layer40 expert0 down contribution,
// the original ordered BF16 FFN tail/residual, then layer41 norm + GDN in_proj_a.
// The other nine ALREADY WEIGHTED expert contributions, shared-gated result,
// residual and effective F32 gamma are explicit upstream semantic boundaries.
// This neither constructs the router/SwiGLU/full GDN nor changes their policy.
namespace q24_modelopt_mixed_source_case {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace source = original_named_role_source;
namespace nv = qwen_nvfp4_original_reference;
namespace fp8 = static_fp8_reference_test;
namespace bf = qwen_bfloat_original_reference;
using Type = c::DataType;
using Value = g::ValueId;
inline constexpr std::int64_t hidden = 8192, intermediate = 2048, columns = 128;
inline constexpr std::size_t roles = 7, context_inputs = 14;
inline constexpr std::string_view repository = "nvidia/Qwen3.8-2.4T-A95B-NVFP4";
inline constexpr std::string_view revision = "74eb734297318ecf7c48016a4ac8ce1b47cc74a9";
enum Role : std::size_t {kCodes, kBlockScale, kWeightGlobal, kInputGlobal, kFp8Codes, kSa, kSw};
inline constexpr std::array<std::string_view, roles> names{
    "model.layers.40.mlp.experts.0.down_proj.weight",
    "model.layers.40.mlp.experts.0.down_proj.weight_scale",
    "model.layers.40.mlp.experts.0.down_proj.weight_scale_2",
    "model.layers.40.mlp.experts.0.down_proj.input_scale",
    "model.layers.41.linear_attn.in_proj_a.weight",
    "model.layers.41.linear_attn.in_proj_a.input_scale",
    "model.layers.41.linear_attn.in_proj_a.weight_scale"};
inline std::array<c::TensorDesc, roles> descriptions() {
  return {fp8::dense(Type::kUInt8,{hidden,intermediate/2}),
      fp8::dense(Type::kFloat8E4M3FN,{hidden,intermediate/16}),
      fp8::dense(Type::kFloat32,{1}),fp8::dense(Type::kFloat32,{1}),
      fp8::dense(Type::kFloat8E4M3FN,{columns,hidden}),
      fp8::dense(Type::kFloat32,{1}),fp8::dense(Type::kFloat32,{1})};
}

struct Case final {
  g::Graph graph;
  std::array<g::ConstantBinding,roles> constants;
  std::vector<g::ValueId> inputs, outputs;
  std::array<c::TensorDesc,context_inputs> input_descriptions;
  std::array<c::TensorDesc,5> output_descriptions;

  // Actual readonly leases from existing Data::load/bind or the explicit test
  // fixture. No role upload, source discovery, conversion, resource owner,
  // map, resident table or loader is added here. Source authentication remains
  // with the original caller. Generated fixture Identity MUST say fixture.
  Case(std::span<const source::ReadonlyRole,roles> parameters,std::int64_t rows=1) {
    const auto expected = descriptions();
    for(std::size_t i=0;i<roles;++i) {
      if(parameters[i].name!=names[i]||parameters[i].description!=expected[i]||
          !parameters[i].owner||!parameters[i].lease.valid())
        throw std::runtime_error("mixed ModelOpt source role differs from its independent typed operand");
      const auto tensor=fp8::take(parameters[i].lease.tensor(expected[i]),"mixed source permission");
      if(tensor.storage().resource()!=parameters[i].owner.ref())
        throw std::runtime_error("mixed source role permission differs from actual resource owner");
    }
    auto b=fp8::take(g::GraphBuilder::create(),"connected ordinary source Graph");
    const auto one=[&](c::Operation operation,std::initializer_list<Value> values) {
      const auto node=fp8::take(b.add_node(operation,{values.begin(),values.size()}),"mixed original node");
      return fp8::take(b.result(node),"mixed original result");
    };
    const auto binary=[&](c::BinaryKind kind,Value left,Value right) {
      return one(c::Operation(c::Binary{kind}),{left,right});
    };
    const auto input=[&](c::TensorDesc d) {
      const auto value=fp8::take(b.input(d),"mixed actual boundary");
      input_descriptions[inputs.size()]=std::move(d);inputs.push_back(value);return value;
    };
    const auto down_input=input(fp8::dense(Type::kBFloat16,{rows,intermediate}));
    const auto selected_weight=input(fp8::dense(Type::kFloat32,{rows,1}));
    std::array<Value,9> later;
    for(auto &v:later)v=input(fp8::dense(Type::kBFloat16,{rows,hidden}));
    const auto shared=input(fp8::dense(Type::kBFloat16,{rows,hidden}));
    const auto residual=input(fp8::dense(Type::kBFloat16,{rows,hidden}));
    const auto gamma=input(fp8::dense(Type::kFloat32,{hidden}));
    std::array<Value,roles> p;
    for(std::size_t i=0;i<roles;++i) {
      p[i]=fp8::take(b.constant(expected[i]),"mixed original source constant");
      constants[i]=fp8::take(parameters[i].binding(p[i]),"mixed actual source binding");
    }
    const auto cast=[&](Value v,Type type) {
      return one(c::Operation(c::Cast{type}),{v});
    };
    // Same ModelOpt operation as Recipe, with independent WG and IG slots.
    const auto down=one(c::model_ops::nvfp4_linear(),
        {down_input,p[kCodes],p[kBlockScale],p[kWeightGlobal],p[kInputGlobal]});
    const auto weighted=cast(binary(c::BinaryKind::kMultiply,
        cast(down,Type::kFloat32),selected_weight),Type::kBFloat16);
    // Exact existing BF16 MoE tail, global expert0 first, then ascending1..9.
    // Keep the zero+first add as well, including its signed-zero semantics.
    auto total=one(c::Operation(c::Fill{.output_shape=fp8::take(c::TensorShape::create(std::array{rows,hidden}),"sum extent"),
        .output_type=Type::kBFloat16,.value=c::ScalarValue::from_bits(Type::kBFloat16,0)}),{});
    total=binary(c::BinaryKind::kAdd,total,weighted);
    for(auto v:later)total=binary(c::BinaryKind::kAdd,total,v);
    const auto merged=binary(c::BinaryKind::kAdd,total,shared);
    const auto post=binary(c::BinaryKind::kAdd,residual,merged);
    // Existing Qwen normalize: effective gamma is already F32. Never add1
    // again or reinterpret these provided gamma values as raw HF deltas.
    const auto raw_norm=one(c::Operation(c::RmsNorm{.epsilon=1.e-6F}),{cast(post,Type::kFloat32),gamma});
    const auto norm=cast(raw_norm,Type::kBFloat16);
    // Exact existing stored-SA/SW twelve-node reader, not HF block128 FP8.
    const auto a=cast(norm,Type::kFloat32);
    const auto one_value=one(c::Operation(c::Fill{.output_shape=fp8::take(c::TensorShape::create(std::array<std::int64_t,1>{1}),"SA shape"),
        .output_type=Type::kFloat32,.value=c::ScalarValue(1.F)}),{});
    const auto inverse=binary(c::BinaryKind::kDivide,one_value,p[kSa]);
    const auto normalized=binary(c::BinaryKind::kMultiply,a,inverse);
    const auto encoded=cast(normalized,Type::kFloat8E4M3FN);
    const auto q=cast(encoded,Type::kFloat32),w=cast(p[kFp8Codes],Type::kFloat32);
    const auto rhs=one(c::Operation(fp8::take(c::View::permute(std::array<std::int32_t,2>{1,0}),"FP8 weight transpose")),{w});
    const auto dot=one(c::Operation(c::MatMul{.accumulation_type=Type::kFloat32}),{q,rhs});
    const auto scaled=binary(c::BinaryKind::kMultiply,
        binary(c::BinaryKind::kMultiply,dot,p[kSa]),p[kSw]);
    const auto result=cast(scaled,Type::kBFloat16);
    outputs={down,post,raw_norm,norm,result};
    for(auto value:outputs)fp8::check(b.publish(value),"publish checked mixed boundary");
    graph=fp8::take(std::move(b).finish(),"finish connected original source Graph");
    for(std::size_t i=0;i<outputs.size();++i)
      output_descriptions[i]=fp8::take(graph.value_description(outputs[i]),"actual mixed boundary description");
  }
};

// A generated, deliberately sparse fixture in the ORIGINAL dimensions. This
// is not publisher payload. Keep full K2048/K8192 and every real scale axis.
struct Fixture final {
  static constexpr std::int64_t rows=1;
  inline static constexpr source::Identity identity{
      "hyle-fixture/q24-modelopt-mixed", "0000000000000000000000000000000000000000",
      "generated-full-K", "not-original-publisher-values"};
  std::array<std::vector<std::byte>,roles> parameters;
  std::array<std::vector<std::byte>,context_inputs> inputs;
  float wg=1.00390625F,ig=.13F,sa=1.00390625F,sw=1.125F,weight=.25F;
  std::vector<std::uint16_t> activation;
  std::array<std::vector<std::uint16_t>,9> later;
  std::vector<std::uint16_t> shared,residual;
  std::vector<float> gamma;
  explicit Fixture(unsigned phase=0) {
    activation.resize(std::size_t(intermediate));
    for(std::size_t i=0;i<16;++i)activation[i]=bf::round(float(int(i)-7)*.25F+float(phase)*.125F);
    std::vector<std::uint8_t> codes(std::size_t(hidden*(intermediate/2)));
    std::vector<std::uint8_t> scales(std::size_t(hidden*(intermediate/16)),0x38);
    for(std::size_t n=0;n<std::size_t(hidden);++n) {
      const auto lane=n%16;codes[n*std::size_t(intermediate/2)+lane/2]=std::uint8_t((1u+(n+phase)%7u)<<(4*(lane%2)));
    }
    std::vector<std::uint8_t> fp8_codes(std::size_t(columns*hidden));
    for(std::size_t n=0;n<std::size_t(columns);++n)
      fp8_codes[n*std::size_t(hidden)+(n*37)%std::size_t(hidden)]=std::uint8_t((n+phase)%2?0xb8:0x38);
    const auto raw=[]<class T>(const std::vector<T> &v){return nvfp4_reference_test::bytes<T>(v);};
    parameters={raw(codes),raw(scales),raw(std::vector<float>{wg}),raw(std::vector<float>{ig}),
        raw(fp8_codes),raw(std::vector<float>{sa}),raw(std::vector<float>{sw})};
    inputs[0]=raw(activation);inputs[1]=raw(std::vector<float>{weight});
    for(std::size_t i=0;i<9;++i) {
      later[i].resize(std::size_t(hidden));
      for(std::size_t n=0;n<later[i].size();++n)later[i][n]=bf::round(float(int((n+i+phase)%7)-3)*.015625F);
      inputs[i+2]=raw(later[i]);
    }
    shared.resize(std::size_t(hidden));residual.resize(std::size_t(hidden));gamma.resize(std::size_t(hidden));
    for(std::size_t n=0;n<std::size_t(hidden);++n) {
      shared[n]=bf::round(float(int(n%5)-2)*.03125F);
      residual[n]=bf::round(.5F+float(n%3)*.03125F);gamma[n]=1.F+float(n%4)*.00390625F;
    }
    inputs[11]=raw(shared);inputs[12]=raw(residual);inputs[13]=raw(gamma);
  }
  // Direct independent scalar checks are the existing original ModelOpt
  // reference; subsequent exact BF16 additions use its already checked down.
  void check(std::span<const std::vector<std::byte>,5> actual) const {
    const auto u16=[](const std::vector<std::byte> &v) {
      std::vector<std::uint16_t> r(v.size()/2);std::memcpy(r.data(),v.data(),v.size());return r;
    };
    const auto down=u16(actual[0]),post=u16(actual[1]),norm=u16(actual[3]),output=u16(actual[4]);
    ASSERT_TRUE(nv::expect_projection(down,activation,parameters[kCodes],parameters[kBlockScale],
        parameters[kWeightGlobal],parameters[kInputGlobal],std::size_t(hidden),std::size_t(intermediate),"fixture layer40 expert0 down",true));
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(hidden));
    for(std::size_t n=0;n<std::size_t(hidden);++n) {
      auto x=bf::round(nvfp4_reference_test::multiply(bf::widen(down[n]),weight));
      x=bf::round(0.F+bf::widen(x));
      for(const auto &v:later)x=bf::round(bf::widen(x)+bf::widen(v[n]));
      x=bf::round(bf::widen(x)+bf::widen(shared[n]));
      expected[n]=bf::round(bf::widen(residual[n])+bf::widen(x));
    }
    ASSERT_EQ(post,expected);
    std::vector<float> raw_norm(actual[2].size()/4);std::memcpy(raw_norm.data(),actual[2].data(),actual[2].size());
    ASSERT_TRUE(bf::expect_norm(raw_norm,norm,post,gamma,"fixture layer41 effective-gamma norm"));
    for(std::size_t n=0;n<std::size_t(columns);++n) {
      const auto lane=(n*37)%std::size_t(hidden);
      const auto code=fp8::encode_e4m3(fp8::multiply(bf::widen(norm[lane]),fp8::reciprocal(sa)));
      const auto wc=std::to_integer<std::uint8_t>(parameters[kFp8Codes][n*std::size_t(hidden)+lane]);
      const auto product=fp8::multiply(fp8::e4m3(code),fp8::e4m3(wc));
      const auto dot=nvfp4_reference_test::f32(0.0+double(product));
      EXPECT_EQ(output[n],bf::round(fp8::multiply(fp8::multiply(dot,sa),sw)))<<n;
    }
  }
};
} // namespace q24_modelopt_mixed_source_case
