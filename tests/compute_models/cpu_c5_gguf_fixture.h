#pragma once
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS)
#include "cpu_expanded_gguf_fixture.h"
#include "cpu_c4_model_branches.h"
#include "cpu_c5_gguf_inputs.h"
#include <limits>

namespace cpu_c5_gguf_fixture {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace in=cpu_c5_inputs;
using sem::Fragment;
using sem::bytes;
using sem::dense;
inline constexpr std::int64_t prototypes=8;

inline c::ModelArtifactProfileDescription profile(unsigned index) {
  const auto &p=in::profiles[index];
  constexpr std::array bases{
      &d5_qwen35_fixture::kPrimaryGgufProfile,
      &d5_qwen38_gdn_fixture::kPrimaryGgufProfile,
      &d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,
      &d5_moe_profile_cases::kimi_k3_primary_gguf,
      &d5_moe_profile_cases::glm53_primary_gguf,
      &d5_moe_profile_cases::glm53_flash_primary_gguf,
      &cpu_c4_model_branches::pro_profile,
      &cpu_c4_model_branches::flash_profile};
  auto result=*bases[p.family];
  result.artifact_repository=p.repository;result.artifact_revision=p.revision;
  result.variant=p.variant;result.manifest_digest=p.digest;result.format_family="gguf";
  return result;
}

struct Bank {
  ggml_type type;
  std::int64_t k;
  std::size_t stride;
  std::vector<std::byte> packed;
  std::vector<float> decoded;
  float l1=0;

  explicit Bank(const in::Case &item):type(item.native),k(item.k),stride(ggml_row_size(type,k)),
      packed(prototypes*stride),decoded(prototypes*k) {
    ggml_cpu_init();
    const auto *traits=ggml_get_type_traits(type);
    if(item.packed && static_cast<unsigned>(item.quant)>=static_cast<unsigned>(c::GgufQuantizedType::kQ4_0)) {
      const auto seed=cpu_expanded_gguf_fixture::weights({item.quant,type},prototypes);
      const auto seed_stride=ggml_row_size(type,256);
      for(std::int64_t row=0;row<prototypes;++row)
        for(std::size_t offset=0;offset<stride;) {
          const auto count=std::min(seed_stride,stride-offset);
          std::memcpy(packed.data()+row*stride+offset,seed.data()+row*seed_stride,count);
          offset+=count;
        }
    } else if(item.packed) {
      auto source=d5_gguf_test::values(prototypes,k,37);
      packed=d5_gguf_test::pack(item.quant,source,prototypes,k,packed.size()).bytes;
    } else {
      std::vector<float> source(k);
      for(std::int64_t row=0;row<prototypes;++row) {
        for(std::int64_t col=0;col<k;++col)source[col]=float((row*13+col*7)%23-11)/32;
        auto *destination=packed.data()+row*stride;
        if(type==GGML_TYPE_F32)std::memcpy(destination,source.data(),stride);
        else if(type==GGML_TYPE_I32) {
          for(std::int64_t col=0;col<k;++col) {
            const auto value=std::int32_t((row*13+col*7)%256);
            std::memcpy(destination+col*4,&value,4);
          }
        } else ggml_get_type_traits_cpu(type)->from_float(source.data(),destination,k);
      }
    }
    for(std::int64_t row=0;row<prototypes;++row) {
      auto *result=decoded.data()+row*k;const auto *source=packed.data()+row*stride;
      if(type==GGML_TYPE_F32)std::memcpy(result,source,stride);
      else if(type==GGML_TYPE_I32) {
        for(std::int64_t col=0;col<k;++col) {
          std::int32_t value;std::memcpy(&value,source+col*4,4);result[col]=float(value);
        }
      } else traits->to_float(source,result,k);
      double norm=0;for(std::int64_t col=0;col<k;++col)norm+=std::abs(result[col]);
      l1=std::max(l1,float(norm));
    }
  }
  std::vector<std::byte> rows(std::int64_t count,std::int64_t first=0) const {
    std::vector<std::byte> result(count*stride);
    for(std::int64_t row=0;row<count;++row)
      std::memcpy(result.data()+row*stride,packed.data()+((first+row)%prototypes)*stride,stride);
    return result;
  }
  std::array<float,prototypes> dots(std::span<const float> activation) const {
    const auto *traits=ggml_get_type_traits_cpu(type);
    const auto bytes=ggml_row_size(traits->vec_dot_type,k);
    std::vector<std::uint64_t> encoded((bytes+7)/8);
    const auto *codec=ggml_get_type_traits_cpu(traits->vec_dot_type);
    if(codec->from_float)codec->from_float(activation.data(),encoded.data(),k);
    else std::memcpy(encoded.data(),activation.data(),bytes);
    std::array<float,prototypes> result{};
    for(std::int64_t row=0;row<prototypes;++row)
      traits->vec_dot(int(k),&result[row],0,packed.data()+row*stride,0,encoded.data(),0,1);
    return result;
  }
};

inline std::vector<float> activation(std::int64_t rows,std::int64_t k,int seed) {
  std::vector<float> result(rows*k);
  for(std::int64_t row=0;row<rows;++row)for(std::int64_t col=0;col<k;++col)
    result[row*k+col]=float((row*3+col*7+seed)%17-8)/16;
  return result;
}

inline Fragment projection(const in::Case &item,const Bank &bank,
    const c::ModelArtifactProfileDescription *identity,std::int64_t n,std::int64_t first=0) {
  Fragment f{.profile=identity,.region=0x0c050001};
  f.absolute_tolerance=std::max(8.0e-5f,16*std::numeric_limits<float>::epsilon()*bank.l1);
  auto a=f.input(dense({1,item.k}));sem::ValueId result;
  if(item.native==GGML_TYPE_BF16) {
    auto rounded=f.one(c::Operation(c::Cast{.output_type=c::DataType::kBFloat16}),{a});
    auto w=f.constant(dense({n,item.k},c::DataType::kBFloat16),bank.rows(n,first));
    result=f.one(c::model_ops::bfloat_linear_f32(),{rounded,w});
  } else if(item.native==GGML_TYPE_F16) {
    // Explicit cold conversion preserves original F16 multiplicands. The hot
    // activation takes its original F16 rounding before the mature F32 dot.
    auto cold=std::make_shared<Fragment>(Fragment{.profile=identity,.region=0x0c050002});
    auto w=cold->constant(dense({n,item.k},c::DataType::kFloat16),bank.rows(n,first));
    cold->outputs={cold->one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{w})};
    std::vector<float> expected(n*item.k);
    for(std::int64_t row=0;row<n;++row)
      std::copy_n(bank.decoded.data()+((first+row)%prototypes)*item.k,item.k,expected.data()+row*item.k);
    cold->replays.push_back({{}, {std::move(expected)}});
    f.preparation=cold;
    auto converted=f.constant(dense({n,item.k}),{});f.prepared_constants={0};
    auto transposed=f.one(c::Operation(c::View::permute(std::array<std::int32_t,2>{1,0}).value()),{converted});
    auto rounded=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat16}),{a});
    auto widened=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{rounded});
    result=f.one(c::Operation(c::MatMul{}),{widened,transposed});
  } else {
    auto desc=item.packed?c::gguf_weight_description(item.quant,c::DataType::kFloat32,item.k,n).value():
        dense({n,item.k}).transposed().value();
    auto w=f.constant(desc,bank.rows(n,first));result=f.one(c::Operation(c::MatMul{}),{a,w});
  }
  f.outputs={result};
  for(int seed:{3,19}) {
    auto input=activation(1,item.k,seed);auto dot=bank.dots(input);std::vector<float> expected(n);
    for(std::int64_t row=0;row<n;++row)expected[row]=dot[(first+row)%prototypes];
    f.replays.push_back({{bytes(input)},{std::move(expected)}});
  }
  return f;
}

inline Fragment lookup(const in::Case &item,const Bank &bank,const c::ModelArtifactProfileDescription *identity) {
  Fragment f{.profile=identity,.region=0x0c050003};
  auto ids=f.input(dense({4},c::DataType::kInt32));
  const auto type=item.native==GGML_TYPE_I32?c::DataType::kInt32:
      (item.native==GGML_TYPE_F16?c::DataType::kFloat16:(item.native==GGML_TYPE_BF16?c::DataType::kBFloat16:c::DataType::kFloat32));
  auto desc=item.packed?c::gguf_weight_description(item.quant,c::DataType::kFloat32,item.k,item.n).value().transposed().value():dense({item.n,item.k},type);
  auto w=f.constant(desc,bank.rows(item.n));auto gathered=f.one(c::Operation(c::Gather{}),{w,ids});
  f.outputs={type==c::DataType::kFloat32||type==c::DataType::kInt32||item.packed?gathered:f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{gathered})};
  f.exact_outputs={0};f.invalid_index_input=0;
  for(int pass:{0,1}) {
    const std::array<std::int32_t,4> selected{pass?1:0,std::int32_t(item.n-1),std::int32_t(item.n/2),std::int32_t(item.n/2)};
    std::vector<float> expected(4*item.k);
    for(int row=0;row<4;++row)
      std::copy_n(bank.decoded.data()+(selected[row]%prototypes)*item.k,item.k,expected.data()+row*item.k);
    f.replays.push_back({{bytes<std::int32_t>(selected)},{std::move(expected)}});
  }
  return f;
}

inline Fragment grouped(const in::Case &item,const Bank &bank,const c::ModelArtifactProfileDescription *identity) {
  const auto e=item.kind==in::Kind::kExpert?std::min<std::int64_t>(item.e,8):item.e;
  const auto slots=item.kind==in::Kind::kExpert?3:e;
  Fragment f{.profile=identity,.region=0x0c050004};
  f.absolute_tolerance=std::max(8.0e-5f,16*std::numeric_limits<float>::epsilon()*bank.l1);
  auto a=f.input(dense({1,slots,item.k})),ids=f.input(dense({1,slots},c::DataType::kInt32));
  const auto type=item.native==GGML_TYPE_BF16?c::DataType::kBFloat16:c::DataType::kFloat32;
  auto desc=item.packed?c::gguf_grouped_weight_description(item.quant,c::DataType::kFloat32,e,item.n,item.k).value():dense({e,item.n,item.k},type);
  auto w=f.constant(desc,bank.rows(e*item.n));f.outputs={f.one(c::model_ops::grouped_matmul(),{w,a,ids})};
  f.invalid_index_input=1;
  for(int pass:{0,1}) {
    auto input=activation(slots,item.k,3+pass*16);std::vector<std::int32_t> selected(slots);
    std::vector<float> expected(slots*item.n);
    for(std::int64_t slot=0;slot<slots;++slot) {
      selected[slot]=std::int32_t(item.kind==in::Kind::kExpert?(slot?e-1:pass):((slot+pass)%e));
      auto dot=bank.dots(std::span(input).subspan(slot*item.k,item.k));
      for(std::int64_t row=0;row<item.n;++row)expected[slot*item.n+row]=dot[(selected[slot]*item.n+row)%prototypes];
    }
    f.replays.push_back({{bytes(input),bytes(selected)},{std::move(expected)}});
  }
  return f;
}
} // namespace cpu_c5_gguf_fixture
#endif
