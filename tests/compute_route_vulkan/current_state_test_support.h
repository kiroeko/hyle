#pragma once

#include "catalogue_gguf_test_support.h"

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

#define HYLE_C4_QUANTIZED_REFERENCE vulkan_catalogue_gguf::projection
#define HYLE_C4_GROUPED_REFERENCE vulkan_catalogue_gguf::grouped
#include "../compute_models/cpu_c4_qwen_mtp.h"
#include "../compute_models/cpu_c4_flash_sidecars.h"
#undef HYLE_C4_QUANTIZED_REFERENCE
#undef HYLE_C4_GROUPED_REFERENCE

namespace vulkan_current_state_test {
inline auto models() {
  auto result=cpu_c4_model_branches::models;
  // Fixed original IREE block equations materialize multiply before add,
  // matching the already qualified source program on this route.
  for(auto &model:result) model.fused_scale_accumulation=false;
  return result;
}
} // namespace vulkan_current_state_test
