#pragma once

#include "dspark_original_shared_reference.h"

// Original main-GGUF Q8_0 vocabulary rows only. BF16 state/head-normalization
// belongs to the existing formal consumer. These references use actual stored
// operands and original LE bytes, with the existing explicit CPU/CUDA Q8
// activation contract; no native oracle or expanded vocabulary is prepared.
namespace dspark_gguf_vocabulary_reference {
namespace q8=dspark_original_shared_reference;
using Codec=q8::ActivationCodec;
namespace detail {
inline std::size_t bank_bytes(std::size_t vocabulary,std::size_t hidden) {
  if(!vocabulary||vocabulary>std::size_t(std::numeric_limits<std::int32_t>::max())||
      !hidden||hidden%32||vocabulary>std::numeric_limits<std::size_t>::max()/(hidden/32)/34)
    throw std::runtime_error("original GGUF Q8 vocabulary geometry/byte span overflows");
  return vocabulary*(hidden/32)*34;
}
inline float bfloat_store(float value) {
  if(!std::isfinite(value))throw std::runtime_error("original Q8 embedding has nonfinite arithmetic");
  const auto raw=std::bit_cast<std::uint32_t>(value);
  const auto stored=std::uint16_t((raw+0x7fffU+((raw>>16U)&1U))>>16U);
  return std::bit_cast<float>(std::uint32_t(stored)<<16U);
}
inline std::vector<std::size_t> candidates(std::span<const float> row) {
  std::vector<std::size_t> result;
  for(std::size_t token=0;token<row.size();++token) {
    if(!std::isfinite(row[token]))throw std::runtime_error("original GGUF Q8 head has nonfinite actual logits");
    auto where=result.begin();
    while(where!=result.end()&&(row[*where]>row[token]||(row[*where]==row[token]&&*where<token)))++where;
    if(where!=result.end()||result.size()<4)result.insert(where,token);
    if(result.size()>4)result.pop_back();
  }
  return result;
}
inline void codec(Codec value) {
  switch(value) {
    case Codec::kCpuAvxQ8_0:case Codec::kCpuScalarQ8_0:case Codec::kCudaMmvqQ8_1Fast:return;
    default:throw std::runtime_error("original GGUF Q8 head requires its actual supported activation codec");
  }
}
} // namespace detail

// Gather decodes each selected original Q8_0 row to F32, then the formal
// consumer explicitly stores BF16. Half-scale times signed8 code has at most
// 18 significant bits and is an exact finite normal F32 value (or zero).
inline bool expect_embedding(std::span<const float> actual_raw_f32,
    std::span<const float> actual_bfloat_widen,std::span<const std::byte> original,
    std::span<const std::int32_t> actual_ids,std::size_t vocabulary,std::size_t hidden) {
  if(original.size()!=detail::bank_bytes(vocabulary,hidden)||actual_ids.empty()||
      actual_ids.size()>std::numeric_limits<std::size_t>::max()/hidden||
      actual_raw_f32.size()!=actual_ids.size()*hidden||actual_bfloat_widen.size()!=actual_raw_f32.size()) {
    ADD_FAILURE()<<"original GGUF Q8 embedding actual extent";return false;
  }
  const auto blocks=hidden/32,row_bytes=blocks*34;
  for(std::size_t row=0;row<actual_ids.size();++row) {
    const auto id=actual_ids[row];
    if(id<0||std::size_t(id)>=vocabulary){ADD_FAILURE()<<"actual Q8 embedding ID outside original vocabulary";return false;}
    const auto packed=original.subspan(std::size_t(id)*row_bytes,row_bytes);
    for(std::size_t block=0;block<blocks;++block) {
      const float scale=q8::detail::scale(packed,block*34);
      for(std::size_t column=0;column<32;++column) {
        const auto index=row*hidden+block*32+column;
        const float expected=float(double(scale)*q8::detail::code(packed[block*34+2+column]));
        if(std::bit_cast<std::uint32_t>(actual_raw_f32[index])!=std::bit_cast<std::uint32_t>(expected)) {
          ADD_FAILURE()<<"original GGUF Q8 embedding F32 decode row="<<row<<" column="<<block*32+column;return false;
        }
        const auto stored=detail::bfloat_store(expected);
        if(std::bit_cast<std::uint32_t>(actual_bfloat_widen[index])!=std::bit_cast<std::uint32_t>(stored)) {
          ADD_FAILURE()<<"original GGUF Q8 embedding actual BF16 store row="<<row<<" column="<<block*32+column;return false;
        }
      }
    }
  }
  return true;
}

// Native dispatch remains the consumer/provider's responsibility. The fixed
// NVIDIA Q8_0 M5/M6 path is MMVQ/Q8_1 (cap8); CPU uses Q8_0/Q8_0 with the
// actual worker's AVX or scalar quantizer. Their SAME existing bound and CUDA
// fast-division ambiguity envelope are reused without a new tolerance.
// First call checks every row/token. Later calls scan all actual logits but
// evaluate only stable top4 plus each supplied actual selected ID.
inline bool expect_logits(std::span<const float> actual,std::span<const float> actual_bfloat_input,
    std::span<const std::byte> original_head,std::size_t vocabulary,std::size_t hidden,
    Codec activation_codec,bool full,std::span<const std::int32_t> selected_ids={}) {
  detail::codec(activation_codec);
  if(original_head.size()!=detail::bank_bytes(vocabulary,hidden)||actual_bfloat_input.empty()||
      actual_bfloat_input.size()%hidden||actual_bfloat_input.size()/hidden>std::numeric_limits<std::size_t>::max()/vocabulary||
      actual.size()!=actual_bfloat_input.size()/hidden*vocabulary||
      (!selected_ids.empty()&&selected_ids.size()!=actual_bfloat_input.size()/hidden)) {
    ADD_FAILURE()<<"original GGUF Q8 head actual extent";return false;
  }
  const auto rows=actual_bfloat_input.size()/hidden,row_bytes=(hidden/32)*34;
  for(std::size_t row=0;row<selected_ids.size();++row)
    if(selected_ids[row]<0||std::size_t(selected_ids[row])>=vocabulary){ADD_FAILURE()<<"actual Q8 head selected ID outside original vocabulary";return false;}
  std::size_t checked=0;double maximum_error=0,maximum_ratio=0,maximum_ambiguity=0;
  const auto check=[&](float value,const q8::ProjectionReference &reference,std::size_t index,std::size_t row,std::size_t token) {
    if(!std::isfinite(value)){ADD_FAILURE()<<"original Q8 head nonfinite row="<<row<<" token="<<token;return false;}
    const auto center=reference.center[index],bound=reference.bound[index];
    const double error=double(value)==center?0:q8::detail::upper(std::abs(double(value)-center));
    if(error>bound) {
      ADD_FAILURE()<<"original GGUF Q8 head row="<<row<<" token="<<token<<" actual="<<value
          <<" native-A-quantized F64="<<center<<" bound="<<bound
          <<" quantization_ambiguity="<<reference.quantization_ambiguity[index]
          <<" arithmetic_bound="<<reference.arithmetic_bound[index]<<" sumabs="<<reference.sumabs[index];return false;
    }
    ++checked;maximum_error=std::max(maximum_error,error);
    maximum_ratio=std::max(maximum_ratio,bound?error/bound:0.0);
    maximum_ambiguity=std::max(maximum_ambiguity,reference.quantization_ambiguity[index]);return true;
  };
  if(full) {
    const auto reference=q8::evaluate(original_head,vocabulary,hidden,actual_bfloat_input,activation_codec);
    for(std::size_t row=0;row<rows;++row)for(std::size_t token=0;token<vocabulary;++token)
      if(!check(actual[row*vocabulary+token],reference,row*vocabulary+token,row,token))return false;
  } else {
    for(std::size_t row=0;row<rows;++row) {
      const auto logits=actual.subspan(row*vocabulary,vocabulary);auto chosen=detail::candidates(logits);
      if(!selected_ids.empty()) {
        const auto selected=std::size_t(selected_ids[row]);
        if(std::find(chosen.begin(),chosen.end(),selected)==chosen.end())chosen.push_back(selected);
      }
      for(const auto token:chosen) {
        const auto reference=q8::evaluate(original_head.subspan(token*row_bytes,row_bytes),1,hidden,
            actual_bfloat_input.subspan(row*hidden,hidden),activation_codec);
        if(!check(logits[token],reference,0,row,token))return false;
      }
    }
  }
  std::printf("DSpark original GGUF Q8 LM: codec=%d full=%d rows=%zu values=%zu K=%zu maxAbs=%.17g maxBoundRatio=%.17g maxQuantizationAmbiguity=%.17g\n",
      int(activation_codec),int(full),rows,checked,hidden,maximum_error,maximum_ratio,maximum_ambiguity);
  return true;
}
} // namespace dspark_gguf_vocabulary_reference
