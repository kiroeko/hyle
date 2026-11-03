#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "glm_gguf_compute_fragment.h"

// Explicit converted F32 runtime. A plain index cache contains ONLY F32 keys;
// no fake scale member, Scatter1, E4 quantizer, BF16 store or implicit codec.
// These are cold descriptions/pure Graph producers, not another StateStore,
// StateHead, execution loop, acceptance policy or artifact loader.
namespace glm_gguf_f32_cache_fragment {
namespace c=hyle::compute;
namespace sem=hyle::inference::model_definition;
namespace model=glm_gguf_model_types;
namespace compute=glm_gguf_compute_fragment;
using Value=sem::ValueId;

struct IndexDescription final {
  c::TensorDesc keys;
  std::optional<c::TensorDesc> scales;
  static IndexDescription from(c::TensorDesc actual_keys,std::optional<c::TensorDesc> actual_scales={}) {
    if(!actual_keys.valid()||!actual_keys.is_plain()||!actual_keys.is_dense()||actual_keys.rank()!=2||
        actual_keys.dimensions()[0]<=0||actual_keys.dimensions()[0]>std::numeric_limits<std::int32_t>::max()||
        actual_keys.dimensions()[1]<=0||!actual_keys.required_bytes())
      throw std::runtime_error("GLM index cache requires its positive dense actual row descriptor");
    if(actual_keys.data_type()==c::DataType::kFloat32) {
      if(actual_scales)throw std::runtime_error("plain F32 index cache has no scale state");
    } else if(actual_keys.data_type()==c::DataType::kFloat8E4M3FN) {
      if(!actual_scales||*actual_scales!=sem::dense({actual_keys.dimensions()[0],1}))
        throw std::runtime_error("encoded E4 index cache requires its real F32 row actual_scales");
    } else throw std::runtime_error("unsupported explicitly declared GLM index encoding");
    return {actual_keys,actual_scales};
  }
};
// This small layout replaces only hardcoded ordinals/descriptions in the
// existing UNPOOLED controller. Its same loops use resources.size()/epoch().
// Flash retains its real pooled history/pool layout; it does not use this
// Pro-specific entry list. All compute/rotary/store leaves still take Geometry.
struct UnpooledLayout final {
  std::vector<c::TensorDesc> resources;
  std::size_t main,index_keys,hidden,pending,cursor,index_cursor,selection;
  std::size_t journal,journal_positions,lag;
  std::optional<std::size_t> index_scales;
  std::size_t epoch()const noexcept{return resources.size();}
  std::size_t first_target()const noexcept{return resources.size()+1;}
};
UnpooledLayout unpooled_layout(const model::Geometry &g,std::int64_t capacity,
    std::int64_t selected,std::int64_t journal_rows,IndexDescription index);
std::array<c::TensorDesc,3> target_descriptions(const model::Geometry &g,std::int64_t capacity);

void require(sem::Fragment &f,Value value,const c::TensorDesc &d);
// Rows use the same absolute positions as main-cache writes and journal
// entries. The frequency Value is caller-provided original metadata, not a
// hardcoded model/theta or a cache-slot-derived position.
Value rotate(sem::Fragment &f,Value actual,Value positions,Value frequencies,
    const model::Geometry &g);
struct MainWrite final {
  Value row,main;
  template<class Visitor> void visit_values(Visitor &&v){v(row);v(main);}
};
MainWrite append_main_row(sem::Fragment &f,Value old_main,Value actual_latent,
    std::optional<Value> actual_rotated_kpe,Value positions,const model::Geometry &g);
struct IndexHistory final {
  glm_gguf_model_types::Projection projection;
  Value input,positions,normalized,rotated,transformed,keys;
  template<class Visitor> void visit_values(Visitor &&v) {
    projection.visit_values(v);for(auto *value:{&input,&positions,&normalized,&rotated,&transformed,&keys})v(*value);
  }
};
// The exact same producer serves fresh append, finite initial prefix and
// accepted lag catchup. A skip step must not call it. No implicit A/row Cast.
template<class Projections>
IndexHistory append_index_history(sem::Fragment &f,Value actual_normalized_input,Value positions,
    Value old_keys,const model::Block &p,const model::Geometry &g,Value frequencies,const Projections &project) {
  model::require_geometry(g);
  const auto &input=f.description(actual_normalized_input),&old=f.description(old_keys);
  if(g.index_pool!=1||input.rank()!=2||input.dimensions()[0]<=0||
      input!=sem::dense({input.dimensions()[0],g.hidden})||old.rank()!=2||old.dimensions()[0]<=0||
      old!=sem::dense({old.dimensions()[0],g.index_depth})||
      old.dimensions()[0]>std::numeric_limits<std::int32_t>::max())
    throw std::runtime_error("unpooled F32 index history requires actual F32 features and plain F32 cache");
  const auto rows=input.dimensions()[0];require(f,positions,sem::dense({rows},c::DataType::kInt32));
  require(f,p.index_norm,sem::dense({g.index_depth}));require(f,p.index_bias,sem::dense({g.index_depth}));
  IndexHistory result;result.input=actual_normalized_input;result.positions=positions;
  result.projection=project.linear(f,actual_normalized_input,p.index_key);
  require(f,result.projection.output,sem::dense({rows,g.index_depth}));
  result.normalized=f.one(c::Operation(c::LayerNorm{.epsilon=g.index_epsilon}),
      {result.projection.output,p.index_norm,p.index_bias});
  result.rotated=compute::reshape(f,rotate(f,compute::reshape(f,result.normalized,{rows,1,g.index_depth}),
      positions,frequencies,g),{rows,g.index_depth});
  result.transformed=f.one(c::Operation(c::Hadamard{}),{result.rotated});
  result.keys=f.one(c::Operation(c::Scatter{.axis=0}),{old_keys,positions,result.transformed});
  return result;
}
struct Selection final {
  Value query_transformed,dots,weights,scores,ids;
  template<class Visitor> void visit_values(Visitor &&v) {
    for(auto *value:{&query_transformed,&dots,&weights,&scores,&ids})v(*value);
  }
};
// Fresh ONLY: original query projection/RoPE are supplied as actual values.
// Normalized Hadamard and the original 1/sqrt(Hindex*Dindex) gate factor stay.
// F32 storage is unscaled: there is no QDQ, qscale, keyscale or dummy state.
Selection select(sem::Fragment &f,Value actual_rotated_query,Value keys,
    Value actual_raw_weights,const model::Geometry &g,std::int64_t visible,std::int64_t selected);
} // namespace glm_gguf_f32_cache_fragment
