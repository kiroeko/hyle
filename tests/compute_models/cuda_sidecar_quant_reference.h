#pragma once

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "d5_gguf_test_support.h"

namespace cuda_sidecar_quant_reference {
inline ggml_type native_type(hyle::compute::GgufQuantizedType type) {
  using Q=hyle::compute::GgufQuantizedType;
  switch(type) {
  case Q::kIq1Xs:return GGML_TYPE_IQ1_XS;
  case Q::kIq1Xxs:return GGML_TYPE_IQ1_XXS;
  case Q::kIq1Xxxs:return GGML_TYPE_IQ1_XXXS;
  default:return d5_gguf_test::native_type(type);
  }
}
// Integration oracle for a single mature CUDA carrier: original native
// ggml invocation, matching physical weights/activation codec/precision.
// Independent equation/codec gates remain in the qualified primitive tests;
// this is not a new reference codec, a CPU execution fallback or a P1 peer.
class Backend final {
public:
  Backend():value_(ggml_backend_cuda_init(0)) {
    if(!value_ || !ggml_backend_is_cuda(value_))
      throw std::runtime_error("sidecar reference requires the original CUDA backend");
  }
  ~Backend(){ggml_backend_free(value_);}
  ggml_backend_t get() const noexcept{return value_;}
private:
  ggml_backend_t value_;
};
struct Storage final {
  ggml_context *context=nullptr;
  ggml_backend_buffer_t buffer=nullptr;
  ~Storage(){ggml_backend_buffer_free(buffer);ggml_free(context);}
};
inline std::vector<float> projection(hyle::compute::GgufQuantizedType type,
    std::span<const std::byte> packed,std::int64_t n,std::int64_t k,std::span<const float> input) {
  if(k<=0 || n<=0 || input.size()%std::size_t(k)!=0)
    throw std::runtime_error("sidecar reference has invalid original geometry");
  thread_local Backend backend;
  Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*4+ggml_graph_overhead_custom(8,false),nullptr,true});
  if(!storage.context)throw std::runtime_error("sidecar reference context allocation failed");
  auto *w=ggml_new_tensor_2d(storage.context,native_type(type),k,n);
  auto *a=ggml_new_tensor_2d(storage.context,GGML_TYPE_F32,k,std::int64_t(input.size()/k));
  auto *output=ggml_mul_mat(storage.context,w,a);
  if(!ggml_prec_set_acc(output,GGML_PREC_F32))
    throw std::runtime_error("sidecar reference F32 accumulation is unsupported");
  if(ggml_nbytes(w)!=packed.size())throw std::runtime_error("sidecar reference packed byte size differs");
  auto *graph=ggml_new_graph_custom(storage.context,8,false);
  ggml_build_forward_expand(graph,output);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  if(!storage.buffer)throw std::runtime_error("sidecar reference device allocation failed");
  ggml_backend_tensor_set(w,packed.data(),0,packed.size());
  ggml_backend_tensor_set(a,input.data(),0,input.size_bytes());
  if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)
    throw std::runtime_error("sidecar reference native CUDA program failed");
  std::vector<float> result(input.size()/k*n);
  ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));
  return result;
}
inline std::vector<float> grouped(const hyle::compute::TensorDesc &description,
    std::span<const std::byte> packed,std::int64_t n,std::int64_t k,std::span<const float> input,
    std::span<const std::int32_t> ids,std::int64_t tokens) {
  if(tokens<=0 || ids.size()%std::size_t(tokens)!=0 || input.size()!=ids.size()*k)
    throw std::runtime_error("sidecar grouped reference geometry differs");
  const auto slots=std::int64_t(ids.size()/tokens);
  const auto experts=description.dimensions()[0];
  for(const auto id:ids)if(id<0 || id>=experts)
    throw std::runtime_error("sidecar grouped reference id is outside its source");
  const auto type=description.is_plain()?GGML_TYPE_BF16:
      native_type(hyle::compute::gguf_quantized_type(description.storage_format()).value());
  thread_local Backend backend;
  Storage storage;
  storage.context=ggml_init({ggml_tensor_overhead()*6+ggml_graph_overhead_custom(8,false),nullptr,true});
  if(!storage.context)throw std::runtime_error("sidecar grouped reference context allocation failed");
  auto *w=ggml_new_tensor_3d(storage.context,type,k,n,experts);
  auto *a=ggml_new_tensor_3d(storage.context,GGML_TYPE_F32,k,slots,tokens);
  auto *i=ggml_new_tensor_2d(storage.context,GGML_TYPE_I32,slots,tokens);
  // Match the declared plain-BF16/F32 multiplication contract, not the
  // original BF16 MMF's implicit cast of the activation to BF16.
  auto *weights=description.is_plain()?ggml_cast(storage.context,w,GGML_TYPE_F32):w;
  auto *output=ggml_mul_mat_id(storage.context,weights,a,i);
  if(!ggml_prec_set_acc(output,GGML_PREC_F32))
    throw std::runtime_error("sidecar grouped reference F32 accumulation is unsupported");
  if(ggml_nbytes(w)!=packed.size())throw std::runtime_error("sidecar grouped reference byte size differs");
  auto *graph=ggml_new_graph_custom(storage.context,8,false);ggml_build_forward_expand(graph,output);
  storage.buffer=ggml_backend_alloc_ctx_tensors(storage.context,backend.get());
  if(!storage.buffer)throw std::runtime_error("sidecar grouped reference device allocation failed");
  ggml_backend_tensor_set(w,packed.data(),0,packed.size());
  ggml_backend_tensor_set(a,input.data(),0,input.size_bytes());
  ggml_backend_tensor_set(i,ids.data(),0,ids.size_bytes());
  if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)
    throw std::runtime_error("sidecar grouped reference original CUDA program failed");
  std::vector<float> result(ids.size()*n);ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));
  return result;
}

// Exact physical prototype rows from the fixed catalogue. Native quantized
// dots use the original GPU activation codec; plain dots remain an independent
// source equation with the declared input rounding boundary.
inline std::array<float,8> catalogue_dots(ggml_type type,std::int64_t k,
    std::span<const std::byte> packed,std::span<const float> decoded,
    std::span<const float> input,bool f32_activation) {
  if(input.size()!=std::size_t(k) || decoded.size()!=8*std::size_t(k))
    throw std::runtime_error("catalogue prototype geometry differs");
  std::array<float,8> result{};
  if(ggml_is_quantized(type)) {
    using Q=hyle::compute::GgufQuantizedType;
    auto quant=Q::kQ4_0;
    for(unsigned value=0;value<=unsigned(Q::kIq1Xs);++value)
      if(native_type(Q(value))==type){quant=Q(value);break;}
    if(native_type(quant)!=type)
      throw std::runtime_error("catalogue native quantized type is unmapped");
    const auto values=projection(quant,packed,8,k,input);
    std::copy(values.begin(),values.end(),result.begin());return result;
  }
  for(std::size_t row=0;row<8;++row) {
    double dot=0;
    for(std::int64_t col=0;col<k;++col) {
      float value=input[col];
      if(!f32_activation && type==GGML_TYPE_BF16)
        value=ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
      if(!f32_activation && type==GGML_TYPE_F16)
        value=ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
      dot+=double(decoded[row*k+col])*value;
    }
    result[row]=float(dot);
  }
  return result;
}
} // namespace cuda_sidecar_quant_reference

#define HYLE_C4_QUANTIZED_REFERENCE cuda_sidecar_quant_reference::projection
#define HYLE_C4_GROUPED_REFERENCE cuda_sidecar_quant_reference::grouped
#define HYLE_C5_GGUF_DOT_REFERENCE cuda_sidecar_quant_reference::catalogue_dots
