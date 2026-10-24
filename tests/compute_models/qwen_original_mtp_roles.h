#pragma once

#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "original_role_input.h"
#include "original_named_role_source.h"
#include "d5_qwen35_gguf_fragment_support.h"

// Authenticated test-role input and independent equations only. No executor,
// product loader or expanded execution weight bank. The caller creates its
// ordinary Graph constants and shares these actual immutable native leases.
namespace qwen_original_mtp_roles {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
using Bytes=std::vector<std::byte>;
using Identity=original_named_role_source::Identity;
using ReadonlyRole=original_named_role_source::ReadonlyRole;
namespace detail {
using original_named_role_source::detail::take;
inline c::TensorDesc dense(std::initializer_list<std::int64_t> dimensions) {
  return take(c::TensorDesc::dense(c::DataType::kFloat32,{dimensions.begin(),dimensions.size()}),"original F32 role description");
}
template<class T> std::span<const std::byte> bytes(const std::vector<T> &source) {
  return std::as_bytes(std::span(source));
}
using original_named_role_source::detail::bind;
struct Sum final {
  double value=0,correction=0;
  void add(double term) {
    const auto next=value+term;
    correction+=std::abs(value)>=std::abs(term)?(value-next)+term:(term-next)+value;
    value=next;
  }
  float rounded() const {return float(value+correction);}
};
} // namespace detail

namespace q27 {
// Embedded blk64 of the selected MAIN artifact, not the C4 draft sidecar.
// F32 separate norms, embedding-before-hidden concat, then packed Q6_K EH.
inline constexpr std::int64_t kHidden=5120,kWidth=2*kHidden;
inline constexpr std::size_t kProjectionBytes=43'008'000;
inline constexpr float kEpsilon=1.0e-6F;
inline constexpr auto kProjectionSha="5e72ecfbdff8f87139874ff85031dc18bc275fcd1639a3b124690c73ca3b9e07";
inline constexpr auto kEmbeddingNormSha="d96abe72c938e866b5a8bdaa8aa9f8590577f55bef6281c2e9fce892db518d97";
inline constexpr auto kHiddenNormSha="e9247e9fc08ebd0ca8f14a610cfd1046a0115143f6967de9d520647149890ac2";
inline constexpr Identity kIdentity{"unsloth/Qwen3.8-27B-GGUF","4ca720788d1e01f1bff70c033e0d0028fd02e502",
    "UD-Q4_K_M","Qwen3.8-27B-UD-Q4_K_M.gguf"};
enum Role : std::size_t {kEmbeddingNorm,kHiddenNorm,kProjection};
using Roles=std::array<ReadonlyRole,3>;
struct Data final {
  Bytes projection;
  std::vector<float> embedding_norm,hidden_norm;
  static Data load(const std::filesystem::path &path) {
    return {original_role_input::read_bytes(path/"eh_proj.weight.q6_k",kProjectionBytes,kProjectionSha),
      original_role_input::read_float(path/"enorm.weight.f32",kHidden,kEmbeddingNormSha),
      original_role_input::read_float(path/"hnorm.weight.f32",kHidden,kHiddenNormSha)};
  }
};
inline Roles bind(const Data &data,const c::StorageOwner &owner) {
  const auto projection=detail::take(c::gguf_weight_description(c::GgufQuantizedType::kQ6K,
      c::DataType::kFloat32,kWidth,kHidden),"Q27 original packed Q6_K role");
  return {detail::bind("blk.64.nextn.enorm.weight",detail::dense({kHidden}),owner,detail::bytes(data.embedding_norm),&kIdentity),
    detail::bind("blk.64.nextn.hnorm.weight",detail::dense({kHidden}),owner,detail::bytes(data.hidden_norm),&kIdentity),
    detail::bind("blk.64.nextn.eh_proj.weight",projection,owner,data.projection,&kIdentity)};
}
struct InputReference final {std::vector<float> hidden,embedding,joined;};
inline InputReference normalize(const Data &data,std::span<const float> hidden,std::span<const float> embedding) {
  if(hidden.empty() || hidden.size()%std::size_t(kHidden) || embedding.size()!=hidden.size())
    throw std::runtime_error("original Q27 input reference geometry");
  InputReference result{d5_qwen35_gguf_fragment::rms_norm_rows(hidden,data.hidden_norm,kHidden,kEpsilon),
      d5_qwen35_gguf_fragment::rms_norm_rows(embedding,data.embedding_norm,kHidden,kEpsilon),{}};
  result.joined.resize(hidden.size()*2);
  for(std::size_t first=0;first<hidden.size();first+=std::size_t(kHidden)) {
    // Concat axis1 is e-before-h PER ROW, also for actual verification M>1.
    std::copy_n(result.embedding.begin()+std::ptrdiff_t(first),std::size_t(kHidden),result.joined.begin()+std::ptrdiff_t(2*first));
    std::copy_n(result.hidden.begin()+std::ptrdiff_t(first),std::size_t(kHidden),result.joined.begin()+std::ptrdiff_t(2*first+kHidden));
  }
  return result;
}
struct ProjectionReference final {std::vector<float> local,whole;};
inline ProjectionReference project(const Bytes &packed,std::span<const float> local,std::span<const float> whole) {
  const auto *codec=ggml_get_type_traits(GGML_TYPE_Q6_K);
  const auto row_bytes=std::size_t(ggml_row_size(GGML_TYPE_Q6_K,kWidth));
  if(!codec || !codec->to_float || row_bytes*std::size_t(kHidden)!=packed.size() ||
      local.empty() || local.size()%std::size_t(kWidth) || whole.size()!=local.size())
    throw std::runtime_error("original Q6_K equation reference geometry");
  const auto count=local.size()/std::size_t(kWidth);
  ProjectionReference result{std::vector<float>(count*std::size_t(kHidden)),std::vector<float>(count*std::size_t(kHidden))};
  std::vector<float> decoded(std::size_t(kWidth),0.F);
  for(std::size_t out=0;out<std::size_t(kHidden);++out) {
    codec->to_float(packed.data()+out*row_bytes,decoded.data(),kWidth);
    for(const auto coefficient:decoded) if(!std::isfinite(coefficient))
      throw std::runtime_error("original Q6_K coefficient is nonfinite");
    for(std::size_t row=0;row<count;++row) {
      detail::Sum local_dot,whole_dot;
      for(std::size_t col=0;col<decoded.size();++col) {
        local_dot.add(double(decoded[col])*local[row*std::size_t(kWidth)+col]);
        whole_dot.add(double(decoded[col])*whole[row*std::size_t(kWidth)+col]);
      }
      result.local[row*std::size_t(kHidden)+out]=local_dot.rounded();
      result.whole[row*std::size_t(kHidden)+out]=whole_dot.rounded();
    }
  }
  return result;
}
} // namespace q27

namespace q24 {
// Embedded blk92 K/V of the selected MAIN artifact. H is already attention
// normalized; K alone gets per-head norm and text HalfSplit64 RoPE.
inline constexpr std::int64_t kHidden=8192,kHeads=4,kDepth=256,kWidth=kHeads*kDepth;
inline constexpr std::size_t kProjectionBytes=5'767'168;
inline constexpr float kEpsilon=1.0e-6F,kTheta=10'000'000.F;
inline constexpr auto kKeySha="a75fa3037cd5ae6b6747e9f886f698661698c61eb11b3873aa3eddce232f3fb5";
inline constexpr auto kValueSha="0d2b7efd67d8cfbf3a42ef9a7553c3d3a01ce690f91e97e08770c96eb2c4174e";
inline constexpr auto kNormSha="815bc7300aad77fecbbb494b15e70949ecb0e43b8ccaf2958a40afdc39737125";
inline constexpr Identity kIdentity{"unsloth/Qwen3.8-2.4T-A95B-GGUF","567d3e6ac26c5474b18311e619c04350fb9a5556",
    "UD-IQ4_XS","UD-IQ4_XS/Qwen3.8-2.4T-A95B-UD-IQ4_XS-00029-of-00029.gguf"};
enum Role : std::size_t {kKey,kValue,kNorm,kFrequency};
using Roles=std::array<ReadonlyRole,4>;
struct Data final {
  Bytes key,value;
  std::vector<float> norm;
  // Optional authenticated packed views. The strong actual source owner keeps
  // these mappings alive; primary acquisition still owns the vectors above.
  c::GgufQuantizedType codec=c::GgufQuantizedType::kQ5K;
  ggml_type native=GGML_TYPE_Q5_K;
  const Identity *identity=&kIdentity;
  std::shared_ptr<const void> original_owner;
  std::span<const std::byte> key_source,value_source;
  std::span<const float> norm_source;
  std::span<const std::byte> key_bytes()const{return original_owner?key_source:std::span<const std::byte>(key);}
  std::span<const std::byte> value_bytes()const{return original_owner?value_source:std::span<const std::byte>(value);}
  std::span<const float> norm_values()const{return original_owner?norm_source:std::span<const float>(norm);}
  static Data load(const std::filesystem::path &path) {
    return {original_role_input::read_bytes(path/"attn_k.weight.q5_k",kProjectionBytes,kKeySha),
      original_role_input::read_bytes(path/"attn_v.weight.q5_k",kProjectionBytes,kValueSha),
      original_role_input::read_float(path/"attn_k_norm.weight.f32",kDepth,kNormSha)};
  }
};
inline std::vector<float> frequencies() {
  std::vector<float> result(32);
  for(std::size_t i=0;i<result.size();++i) result[i]=1.F/std::pow(kTheta,float(i)/32.F);
  return result;
}
inline Roles bind(const Data &data,const c::StorageOwner &owner) {
  if(!data.identity||original_named_role_source::native_codec(data.codec)!=data.native)throw std::runtime_error("actual Q24 KV native/codec/identity differs");
  const auto packed=detail::take(c::gguf_weight_description(data.codec,
      c::DataType::kFloat32,kHidden,kWidth),"Q24 actual original packed KV role");
  const auto frequency=frequencies();
  // The fourth binding is derived from pinned theta metadata, not another
  // downloaded original weight. The three authenticated payloads stay packed.
  return {detail::bind("blk.92.attn_k.weight",packed,owner,data.key_bytes(),data.identity),
    detail::bind("blk.92.attn_v.weight",packed,owner,data.value_bytes(),data.identity),
    detail::bind("blk.92.attn_k_norm.weight",detail::dense({kDepth}),owner,std::as_bytes(data.norm_values()),data.identity),
    detail::bind("text.rope.frequencies",detail::dense({32}),owner,detail::bytes(frequency),data.identity)};
}
inline std::vector<float> project(std::span<const std::byte> packed,std::span<const float> activation,int count,ggml_type native=GGML_TYPE_Q5_K) {
  const auto *codec=ggml_get_type_traits(native);
  const auto row_bytes=std::size_t(ggml_row_size(native,kHidden));
  if(!codec || !codec->to_float || row_bytes*std::size_t(kWidth)!=packed.size() || count<=0 ||
      activation.size()!=std::size_t(count)*std::size_t(kHidden))
    throw std::runtime_error("original packed KV reference geometry");
  std::vector<float> result(std::size_t(count)*std::size_t(kWidth)),decoded(std::size_t(kHidden),0.F);
  for(std::size_t out=0;out<std::size_t(kWidth);++out) {
    codec->to_float(packed.data()+out*row_bytes,decoded.data(),kHidden);
    for(const auto coefficient:decoded) if(!std::isfinite(coefficient))
      throw std::runtime_error("original packed KV coefficient is nonfinite");
    for(int row=0;row<count;++row) {
      detail::Sum dot;
      for(std::size_t col=0;col<decoded.size();++col)
        dot.add(double(decoded[col])*activation[std::size_t(row)*std::size_t(kHidden)+col]);
      result[std::size_t(row)*std::size_t(kWidth)+out]=dot.rounded();
    }
  }
  return result;
}
inline std::vector<float> normalize(const Data &data,std::span<const float> key) {
  if(key.empty() || key.size()%std::size_t(kWidth)) throw std::runtime_error("original Q24 K norm reference geometry");
  return d5_qwen35_gguf_fragment::rms_norm_rows(key,data.norm_values(),kDepth,kEpsilon);
}
inline std::vector<float> rotate(std::span<const float> normalized,int count,int first_position) {
  if(count<=0 || first_position<0 || first_position>std::numeric_limits<std::int32_t>::max()-(count-1) ||
      normalized.size()!=std::size_t(count)*std::size_t(kWidth))
    throw std::runtime_error("original Q24 rotary reference geometry");
  auto result=std::vector<float>(normalized.begin(),normalized.end());const auto frequency=frequencies();
  for(int row=0;row<count;++row) for(int head=0;head<kHeads;++head) for(int pair=0;pair<32;++pair) {
    const auto first=std::size_t(row*kWidth+head*kDepth+pair),second=first+32;
    const float angle=float(first_position+row)*frequency[std::size_t(pair)];
    const double cosine=std::cos(double(angle)),sine=std::sin(double(angle));
    result[first]=float(double(normalized[first])*cosine-double(normalized[second])*sine);
    result[second]=float(double(normalized[first])*sine+double(normalized[second])*cosine);
  }
  return result;
}
} // namespace q24
} // namespace qwen_original_mtp_roles
