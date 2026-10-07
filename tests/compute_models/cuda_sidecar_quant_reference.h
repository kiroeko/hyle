#pragma once

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "d5_gguf_test_support.h"

namespace cuda_sidecar_quant_reference {
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
  auto *w=ggml_new_tensor_2d(storage.context,d5_gguf_test::native_type(type),k,n);
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
} // namespace cuda_sidecar_quant_reference

#define HYLE_C4_QUANTIZED_REFERENCE cuda_sidecar_quant_reference::projection
