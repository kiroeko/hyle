#include "glm_gguf_f32_cache_fragment.h"

namespace glm_gguf_f32_cache_fragment {
UnpooledLayout unpooled_layout(const model::Geometry &g,std::int64_t capacity,
    std::int64_t selected,std::int64_t journal_rows,IndexDescription index) {
  model::require_geometry(g);
  index=IndexDescription::from(index.keys,index.scales);
  if(g.index_pool!=1||capacity<=0||selected<=0||selected>capacity||selected>g.index_topk||journal_rows<=0||
      index.keys.dimensions()[0]!=capacity||index.keys.dimensions()[1]!=g.index_depth)
    throw std::runtime_error("unpooled GLM actual cache/layout extent differs");
  UnpooledLayout result;
  result.resources.reserve(index.scales?11:10);
  const auto add=[&](c::TensorDesc d) {
    if(!d.required_bytes())throw std::runtime_error("GLM actual state byte extent overflow");
    const auto ordinal=result.resources.size();result.resources.push_back(d);return ordinal;
  };
  const auto scalar=sem::dense({1},c::DataType::kInt32);
  result.main=add(sem::dense({capacity,g.latent+g.rope}));
  result.index_keys=add(index.keys);
  if(index.scales)result.index_scales=add(*index.scales);
  result.hidden=add(sem::dense({1,g.hidden}));result.pending=add(scalar);
  result.cursor=add(scalar);result.index_cursor=add(scalar);
  result.selection=add(sem::dense({selected},c::DataType::kInt32));
  result.journal=add(sem::dense({journal_rows,g.hidden}));
  result.journal_positions=add(sem::dense({journal_rows},c::DataType::kInt32));
  result.lag=add(scalar);
  return result;
}
}

namespace glm_gguf_f32_cache_fragment {
std::array<c::TensorDesc,3> target_descriptions(const model::Geometry &g,std::int64_t capacity) {
  model::require_geometry(g);
  if(capacity<=0)throw std::runtime_error("GLM target cache capacity must be positive");
  return {sem::dense({capacity,g.latent+g.rope}),sem::dense({1},c::DataType::kInt32),
      sem::dense({1,g.hidden})};
}
}

namespace glm_gguf_f32_cache_fragment {
void require(sem::Fragment &f,Value value,const c::TensorDesc &d) {
  if(value.ordinal()>=f.descriptions.size()||f.description(value)!=d)
    throw std::runtime_error("GLM F32 runtime must consume its actual declared predecessor");
}
}

namespace glm_gguf_f32_cache_fragment {
Value rotate(sem::Fragment &f,Value actual,Value positions,Value frequencies,
    const model::Geometry &g) {
  model::require_geometry(g);
  const auto &d=f.description(actual);
  if(d.rank()!=3||d.dimensions()[0]<=0||d.dimensions()[1]<=0||d.dimensions()[2]<g.rope||
      !d.is_plain()||!d.is_dense()||d.data_type()!=c::DataType::kFloat32)
    throw std::runtime_error("GLM F32 rotary requires actual [R,H,D] rows");
  const auto rows=d.dimensions()[0],heads=d.dimensions()[1],depth=d.dimensions()[2];
  require(f,positions,sem::dense({rows},c::DataType::kInt32));
  if(!g.rope)return actual;
  require(f,frequencies,sem::dense({g.rope/2}));
  const auto rotated=f.one(c::Operation(c::RoPE{.rotary_dimensions=std::size_t(g.rope),
      .sequence_axis=1,.layout=c::RopeLayout::kInterleaved}),
      {compute::reshape(f,actual,{1,rows,heads,depth}),positions,frequencies});
  return compute::reshape(f,rotated,{rows,heads,depth});
}
}

namespace glm_gguf_f32_cache_fragment {
MainWrite append_main_row(sem::Fragment &f,Value old_main,Value actual_latent,
    std::optional<Value> actual_rotated_kpe,Value positions,const model::Geometry &g) {
  model::require_geometry(g);
  const auto &latent=f.description(actual_latent),&old=f.description(old_main);
  if(latent.rank()!=2||latent.dimensions()[0]<=0||latent!=sem::dense({latent.dimensions()[0],g.latent})||
      old.rank()!=2||old.dimensions()[0]<=0||old!=sem::dense({old.dimensions()[0],g.latent+g.rope}))
    throw std::runtime_error("GLM F32 main cache/CKV descriptor differs");
  const auto rows=latent.dimensions()[0];require(f,positions,sem::dense({rows},c::DataType::kInt32));
  Value row=actual_latent;
  if(g.rope) {
    if(!actual_rotated_kpe)throw std::runtime_error("GLM main row needs its actual rotated KPE suffix");
    require(f,*actual_rotated_kpe,sem::dense({rows,g.rope}));
    row=f.one(c::Operation(c::Concat{.axis=1}),{actual_latent,*actual_rotated_kpe});
  } else if(actual_rotated_kpe)throw std::runtime_error("NoPE-only GLM cache has no KPE suffix");
  return {row,f.one(c::Operation(c::Scatter{.axis=0}),{old_main,positions,row})};
}
}

namespace glm_gguf_f32_cache_fragment {
Selection select(sem::Fragment &f,Value actual_rotated_query,Value keys,
    Value actual_raw_weights,const model::Geometry &g,std::int64_t visible,std::int64_t selected) {
  model::require_geometry(g);
  const auto &bank=f.description(keys);
  if(g.index_pool!=1||bank.rank()!=2||bank.dimensions()[0]<=0||
      bank!=sem::dense({bank.dimensions()[0],g.index_depth})||visible<=0||visible>bank.dimensions()[0]||
      selected<=0||selected>visible||selected>g.index_topk||
      visible>std::numeric_limits<std::int32_t>::max())
    throw std::runtime_error("F32 unpooled index selection extent differs");
  require(f,actual_rotated_query,sem::dense({g.index_heads,g.index_depth}));
  require(f,actual_raw_weights,sem::dense({1,g.index_heads}));
  std::vector<std::int32_t> prefix(std::size_t(visible),0);std::iota(prefix.begin(),prefix.end(),0);
  const auto ids=f.constant(sem::dense({visible},c::DataType::kInt32),sem::bytes(prefix));
  const auto rows=f.one(c::Operation(c::Gather{}),{keys,ids});
  constexpr std::array<std::int32_t,2> axes{1,0};
  const auto transpose=c::View::permute(axes);if(!transpose)throw std::runtime_error("F32 index transpose failed");
  Selection result;
  result.query_transformed=f.one(c::Operation(c::Hadamard{}),{actual_rotated_query});
  result.dots=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),
      {result.query_transformed,f.one(c::Operation(*transpose),{rows})});
  const auto relu=f.one(c::Operation(c::Unary{.kind=c::UnaryKind::kRelu}),{result.dots});
  const auto factor=f.constant(sem::dense({1}),sem::bytes(std::vector<float>{
      1.F/std::sqrt(float(g.index_heads*g.index_depth))}));
  result.weights=f.one(c::Operation(c::Binary{.kind=c::BinaryKind::kMultiply}),{actual_raw_weights,factor});
  const auto weighted=f.one(c::Operation(c::Binary{.kind=c::BinaryKind::kMultiply}),
      {relu,compute::reshape(f,result.weights,{g.index_heads,1})});
  constexpr std::array<std::int32_t,1> reduce_axis{0};const auto axis=c::Axes::from(reduce_axis);
  if(!axis)throw std::runtime_error("F32 index head reduction axis failed");
  result.scores=compute::reshape(f,f.one(c::Operation(c::Reduce{.kind=c::ReductionKind::kSum,
      .axes=*axis,.accumulation_type=c::DataType::kFloat32}),{weighted}),{1,visible});
  const auto top=f.append(c::Operation(c::TopK{.count=std::size_t(selected),.stable=true}),{result.scores},2);
  result.ids=compute::copy(f,compute::reshape(f,top[1],{selected}));
  return result;
}
}
