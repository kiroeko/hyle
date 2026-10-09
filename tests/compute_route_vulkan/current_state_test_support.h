#pragma once

#include "catalogue_gguf_test_support.h"
#include "../../src/compute_ggml_runtime/explicit_rope.h"

namespace vulkan_catalogue_gguf {
inline std::vector<float> projection(c::GgufQuantizedType quant,
    std::span<const std::byte> packed,std::int64_t n,std::int64_t k,
    std::span<const float> input) {
  if(k<=0 || n<=0 || input.size()%static_cast<std::size_t>(k))
    throw std::runtime_error("original projection reference shape differs");
  static Backend backend;auto active=backend.activate();Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*4+ggml_graph_overhead_custom(8,false),nullptr,true});
  if(!storage.context) throw std::runtime_error("original projection reference metadata OOM");
  auto *w=ggml_new_tensor_2d(storage.context,vulkan_gguf_test::native_type(quant),k,n);
  auto *a=ggml_new_tensor_2d(storage.context,GGML_TYPE_F32,k,static_cast<std::int64_t>(input.size()/k));
  auto *out=ggml_mul_mat(storage.context,w,a);
  if(!ggml_prec_set_acc(out,GGML_PREC_F32)||ggml_nbytes(w)!=packed.size())
    throw std::runtime_error("original projection precision/bytes differ");
  auto *graph=ggml_new_graph_custom(storage.context,8,false);ggml_build_forward_expand(graph,out);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  if(!storage.buffer) throw std::runtime_error("original projection storage OOM");
  ggml_backend_tensor_set(w,packed.data(),0,packed.size());ggml_backend_tensor_set(a,input.data(),0,input.size_bytes());
  if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)
    throw std::runtime_error("original projection native completion failed");
  std::vector<float> result(input.size()/k*n);ggml_backend_tensor_get(out,result.data(),0,result.size()*sizeof(float));return result;
}
inline std::vector<float> grouped(const c::TensorDesc &description,
    std::span<const std::byte> packed,std::int64_t n,std::int64_t k,
    std::span<const float> input,std::span<const std::int32_t> ids,std::int64_t tokens) {
  if(tokens<=0 || k<=0 || n<=0 || ids.size()%static_cast<std::size_t>(tokens) || input.size()!=ids.size()*k)
    throw std::runtime_error("original grouped reference shape differs");
  const auto slots=static_cast<std::int64_t>(ids.size()/tokens),experts=description.dimensions()[0];
  const auto type=description.is_plain()?GGML_TYPE_BF16:vulkan_gguf_test::native_type(c::gguf_quantized_type(description.storage_format()).value());
  static Backend backend;auto active=backend.activate();Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*6+ggml_graph_overhead_custom(8,false),nullptr,true});
  if(!storage.context) throw std::runtime_error("original grouped reference metadata OOM");
  auto *w=ggml_new_tensor_3d(storage.context,type,k,n,experts);
  auto *a=ggml_new_tensor_3d(storage.context,GGML_TYPE_F32,k,slots,tokens);
  auto *indices=ggml_new_tensor_2d(storage.context,GGML_TYPE_I32,slots,tokens);
  auto *out=ggml_mul_mat_id(storage.context,w,a,indices);
  if(!ggml_prec_set_acc(out,GGML_PREC_F32)||ggml_nbytes(w)!=packed.size())
    throw std::runtime_error("original grouped precision/bytes differ");
  auto *graph=ggml_new_graph_custom(storage.context,8,false);ggml_build_forward_expand(graph,out);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  if(!storage.buffer) throw std::runtime_error("original grouped storage OOM");
  ggml_backend_tensor_set(w,packed.data(),0,packed.size());ggml_backend_tensor_set(a,input.data(),0,input.size_bytes());
  ggml_backend_tensor_set(indices,ids.data(),0,ids.size_bytes());
  if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)
    throw std::runtime_error("original grouped native completion failed");
  std::vector<float> result(ids.size()*n);ggml_backend_tensor_get(out,result.data(),0,result.size()*sizeof(float));return result;
}
} // namespace vulkan_catalogue_gguf

namespace vulkan_current_state_test {
inline std::vector<float> native_norm(std::span<const float> input,std::span<const float> scale) {
  using namespace vulkan_catalogue_gguf;
  if(scale.empty() || input.size()%scale.size()) throw std::runtime_error("native norm shape differs");
  static Backend backend;auto active=backend.activate();Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead_custom(16,false),nullptr,true});
  if(!storage.context) throw std::runtime_error("native norm metadata OOM");
  auto *x=ggml_new_tensor_2d(storage.context,GGML_TYPE_F32,static_cast<std::int64_t>(scale.size()),static_cast<std::int64_t>(input.size()/scale.size()));
  auto *gamma=ggml_new_tensor_1d(storage.context,GGML_TYPE_F32,static_cast<std::int64_t>(scale.size()));
  auto *output=ggml_mul(storage.context,ggml_rms_norm(storage.context,x,1e-6f),gamma);
  auto *graph=ggml_new_graph_custom(storage.context,16,false);ggml_build_forward_expand(graph,output);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  if(!storage.buffer) throw std::runtime_error("native norm storage OOM");
  ggml_backend_tensor_set(x,input.data(),0,input.size_bytes());ggml_backend_tensor_set(gamma,scale.data(),0,scale.size_bytes());
  if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("native norm terminal failure");
  std::vector<float> result(input.size());ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));return result;
}
inline std::vector<float> native_rotary(std::span<const float>,std::int64_t,
    std::span<const std::int32_t>,bool,bool,std::int64_t,bool,bool);
inline std::vector<float> model_sink(std::span<const float>,std::span<const float>,
    std::span<const float>,std::span<const float>,std::span<const float>,
    std::int64_t,std::int64_t,float);
}
#define HYLE_C4_QUANTIZED_REFERENCE vulkan_catalogue_gguf::projection
#define HYLE_C4_GROUPED_REFERENCE vulkan_catalogue_gguf::grouped
#define HYLE_C4_ROTARY_REFERENCE(input,heads,positions,inverse) vulkan_current_state_test::native_rotary(input,heads,positions,false,inverse,512,false,true)
#define HYLE_C4_SINK_REFERENCE vulkan_current_state_test::model_sink
#include "../compute_models/cpu_c4_qwen_mtp.h"
#include "../compute_models/cpu_c4_flash_sidecars.h"
#undef HYLE_C4_QUANTIZED_REFERENCE
#undef HYLE_C4_GROUPED_REFERENCE
#undef HYLE_C4_ROTARY_REFERENCE
#undef HYLE_C4_SINK_REFERENCE

namespace vulkan_current_state_test {
inline std::vector<float> native_rotary(std::span<const float> input,std::int64_t heads,
    std::span<const std::int32_t> positions,bool compressed,bool inverse,std::int64_t depth=512,
    bool legacy=false,bool bfloat=true) {
  using namespace vulkan_catalogue_gguf;
  if(depth<64 || input.size()!=positions.size()*heads*depth) throw std::runtime_error("native rotary shape differs");
  static Backend backend;auto active=backend.activate();Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(128,false),nullptr,true});
  if(!storage.context) throw std::runtime_error("native rotary metadata OOM");
  auto *x=ggml_new_tensor_3d(storage.context,GGML_TYPE_F32,64,heads,static_cast<std::int64_t>(positions.size()));
  auto *p=ggml_new_tensor_1d(storage.context,GGML_TYPE_I32,static_cast<std::int64_t>(positions.size()));
  auto *f=ggml_new_tensor_1d(storage.context,GGML_TYPE_F32,32);
  auto *out=hyle::compute::ggml_runtime::explicit_rope(storage.context,x,p,f,64,true);
  auto *graph=ggml_new_graph_custom(storage.context,128,false);ggml_build_forward_expand(graph,out);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  if(!storage.buffer) throw std::runtime_error("native rotary storage OOM");
  std::vector<float> tail(positions.size()*heads*64);
  for(std::size_t row=0;row<positions.size()*heads;++row)
    std::copy_n(input.data()+row*depth+depth-64,64,tail.data()+row*64);
  const auto frequency=legacy?d5_deepseek_remaining_fragments::rope_frequencies(compressed,inverse):
      cpu_c4_model_branches::ds::rope_frequencies(compressed,inverse);
  ggml_backend_tensor_set(x,tail.data(),0,tail.size()*sizeof(float));
  ggml_backend_tensor_set(p,positions.data(),0,positions.size_bytes());
  ggml_backend_tensor_set(f,frequency.data(),0,frequency.size()*sizeof(float));
  if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("native rotary terminal failure");
  ggml_backend_tensor_get(out,tail.data(),0,tail.size()*sizeof(float));std::vector<float> result(input.begin(),input.end());
  for(std::size_t row=0;row<positions.size()*heads;++row)
    std::copy_n(tail.data()+row*64,64,result.data()+row*depth+depth-64);
  return bfloat?cpu_c4_model_branches::ds::bfloat_boundary(std::move(result)):result;
}
inline d5_deepseek_remaining_fragments::RotaryReference gguf_reference() {
  return [](auto input,auto depth,std::int32_t position,bool compressed,bool inverse) {
    const std::array positions{position};
    return native_rotary(input,static_cast<std::int64_t>(input.size()/depth),positions,compressed,inverse,depth,true,false);
  };
}
inline std::vector<float> rounded_sink(const std::array<std::vector<float>,5> &data,
    std::int64_t tokens,std::int64_t keys,float scale) {
  const auto heads=static_cast<std::int64_t>(data[4].size());
  auto half=[](float value){return static_cast<float>(half_test::decode(half_test::encode<hyle::simd::float16_t>(value)));};
  std::vector<float> result(tokens*heads*512),score(keys),probabilities(keys);
  for(std::int64_t token=0;token<tokens;++token) for(std::int64_t head=0;head<heads;++head) {
    float maximum=data[4][head];const auto offset=(token*heads+head)*512;
    for(std::int64_t row=0;row<keys;++row) {
      double sum=0;
      for(std::int64_t d=0;d<512;++d) sum+=static_cast<double>(half(data[0][offset+d]))*half(data[1][row*512+d]);
      score[row]=static_cast<float>(sum)*scale+data[3][token*keys+row];maximum=std::max(maximum,score[row]);
    }
    double denominator=std::exp(static_cast<double>(data[4][head])-maximum);
    for(std::int64_t row=0;row<keys;++row) denominator+=std::exp(static_cast<double>(score[row])-maximum);
    for(std::int64_t row=0;row<keys;++row) probabilities[row]=half(static_cast<float>(std::exp(static_cast<double>(score[row])-maximum)/denominator));
    for(std::int64_t d=0;d<512;++d) {
      double sum=0;
      for(std::int64_t row=0;row<keys;++row) sum+=static_cast<double>(probabilities[row])*half(data[2][row*512+d]);
      result[offset+d]=static_cast<float>(sum);
    }
  }
  return result;
}
inline std::vector<float> native_sink(std::span<const hyle::compute::TensorDesc> descriptions,
    const std::array<std::vector<float>,5> &data,float scale) {
  using namespace vulkan_catalogue_gguf;
  static Backend backend;auto active=backend.activate();Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(64,false),nullptr,true});
  if(!storage.context) throw std::runtime_error("native sink metadata OOM");
  std::array<ggml_tensor *,5> inputs{};
  for(std::size_t port=0;port<5;++port) {
    std::array<std::int64_t,4> dims{1,1,1,1};
    std::ranges::reverse_copy(descriptions[port].dimensions(),dims.begin());
    inputs[port]=ggml_new_tensor(storage.context,GGML_TYPE_F32,static_cast<int>(descriptions[port].rank()),dims.data());
    if(ggml_nbytes(inputs[port])!=data[port].size()*sizeof(float)) throw std::runtime_error("native sink bytes differ");
  }
  auto *q=ggml_permute(storage.context,inputs[0],0,2,1,3);
  auto *k=ggml_permute(storage.context,inputs[1],0,2,1,3);
  auto *v=ggml_permute(storage.context,inputs[2],0,2,1,3);
  auto *scores=ggml_mul_mat(storage.context,k,q);ggml_prec_set_acc(scores,GGML_PREC_F32);
  auto *probability=ggml_soft_max_ext(storage.context,scores,inputs[3],scale,0.f);
  ggml_soft_max_add_sinks(probability,inputs[4]);
  v=ggml_cont(storage.context,ggml_transpose(storage.context,v));
  auto *output=ggml_mul_mat(storage.context,v,probability);ggml_prec_set_acc(output,GGML_PREC_F32);
  output=ggml_cont(storage.context,ggml_permute(storage.context,output,0,2,1,3));
  auto *graph=ggml_new_graph_custom(storage.context,64,false);ggml_build_forward_expand(graph,output);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  if(!storage.buffer) throw std::runtime_error("native sink storage OOM");
  for(std::size_t port=0;port<5;++port) ggml_backend_tensor_set(inputs[port],data[port].data(),0,data[port].size()*sizeof(float));
  if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("native sink terminal failure");
  std::vector<float> result(ggml_nelements(output));ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));return result;
}
inline std::vector<float> model_sink(std::span<const float> query,std::span<const float> key,
    std::span<const float> value,std::span<const float> mask,std::span<const float> sinks,
    std::int64_t tokens,std::int64_t keys,float scale) {
  namespace c=hyle::compute;
  const auto heads=static_cast<std::int64_t>(sinks.size());
  const std::array descriptions{
      c::TensorDesc::dense(c::DataType::kFloat32,std::array{std::int64_t{1},tokens,heads,std::int64_t{512}}).value(),
      c::TensorDesc::dense(c::DataType::kFloat32,std::array{std::int64_t{1},keys,std::int64_t{1},std::int64_t{512}}).value(),
      c::TensorDesc::dense(c::DataType::kFloat32,std::array{std::int64_t{1},keys,std::int64_t{1},std::int64_t{512}}).value(),
      c::TensorDesc::dense(c::DataType::kFloat32,std::array{tokens,keys}).value(),
      c::TensorDesc::dense(c::DataType::kFloat32,std::array{heads}).value()};
  const std::array<std::vector<float>,5> data{
      std::vector<float>(query.begin(),query.end()),std::vector<float>(key.begin(),key.end()),
      std::vector<float>(value.begin(),value.end()),std::vector<float>(mask.begin(),mask.end()),
      std::vector<float>(sinks.begin(),sinks.end())};
  return native_sink(descriptions,data,scale);
}
inline auto models() {
  auto result=cpu_c4_model_branches::models;
  // Fixed original IREE block equations materialize multiply before add,
  // matching the already qualified source program on this route.
  for(auto &model:result) {
    model.fused_scale_accumulation=false;
    model.attention_reference=[](auto query,auto key,auto value,auto sinks,float scale) {
      const auto tokens=static_cast<std::int64_t>(query.size()/sinks.size()/512),keys=static_cast<std::int64_t>(key.size()/512);
      const std::vector<float> mask(tokens*keys,0.f);
      return model_sink(query,key,value,mask,sinks,tokens,keys,scale);
    };
    model.rotation_reference=[](auto input,auto heads,auto positions,bool inverse) {
      return native_rotary(input,heads,positions,false,inverse);
    };
  }
  return result;
}
inline d5_deepseek_official_attention::NativeReference main_reference() {
  d5_deepseek_official_attention::NativeReference reference;
  reference.fused_scale_accumulation=false;
  reference.rotation=[](auto input,auto depth,std::int32_t position,bool compressed,bool inverse) {
    const std::array positions{position};
    return native_rotary(input,static_cast<std::int64_t>(input.size()/depth),positions,compressed,inverse,depth);
  };
  reference.sink=[](auto query,auto key,auto value,auto mask,auto sinks,float scale) {
    return model_sink(query,key,value,mask,sinks,1,static_cast<std::int64_t>(key.size()/512),scale);
  };
  return reference;
}
} // namespace vulkan_current_state_test
