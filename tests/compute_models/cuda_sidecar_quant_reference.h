#pragma once

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "d5_gguf_test_support.h"
#include "../../src/compute_provider_ggml_cuda/native_bridge.h"

namespace cuda_sidecar_quant_reference {
// Integration oracle for a single mature CUDA carrier: original native
// ggml invocation, matching physical weights/activation codec/precision.
// Independent equation/codec gates remain in the qualified primitive tests;
// this is not a new reference codec, a CPU execution fallback or a P1 peer.
class Backend final {
public:
  Backend():value_(ggml_backend_cuda_init(0)) {
    // Preserve a fatal source diagnostic even if another runtime has installed
    // an abort callback. This is a test-process hook, not product recovery.
    ggml_set_abort_callback([](const char *message) {
      std::fprintf(stderr,"SIDE_REFERENCE_GGML_ABORT: %s\n",message);std::fflush(stderr);
    });
    std::set_terminate([] {
      try {
        if(auto error=std::current_exception())std::rethrow_exception(error);
      } catch(const hyle::compute::ggml_cuda::bridge::NativeFailure<CUresult> &error) {
        std::fprintf(stderr,"SIDE_REFERENCE_TERMINATE driver: %d %s\n",int(error.code),error.operation);
      } catch(const hyle::compute::ggml_cuda::bridge::NativeFailure<cudaError_t> &error) {
        std::fprintf(stderr,"SIDE_REFERENCE_TERMINATE runtime: %d %s\n",int(error.code),error.operation);
      } catch(const std::exception &error) {
        std::fprintf(stderr,"SIDE_REFERENCE_TERMINATE: %s\n",error.what());
      } catch(const std::string &error) {
        std::fprintf(stderr,"SIDE_REFERENCE_TERMINATE string: %s\n",error.c_str());
      } catch(const char *error) {
        std::fprintf(stderr,"SIDE_REFERENCE_TERMINATE text: %s\n",error);
      } catch(int error) {
        std::fprintf(stderr,"SIDE_REFERENCE_TERMINATE code: %d\n",error);
      } catch(...) {
        std::fprintf(stderr,"SIDE_REFERENCE_TERMINATE: unknown exception\n");
      }
      std::fflush(stderr);std::abort();
    });
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
  std::fprintf(stderr,"SIDE_REFERENCE projection n=%lld k=%lld rows=%zu type=%u\n",
      static_cast<long long>(n),static_cast<long long>(k),input.size()/k,static_cast<unsigned>(type));
  std::fflush(stderr);
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
      d5_gguf_test::native_type(hyle::compute::gguf_quantized_type(description.storage_format()).value());
  thread_local Backend backend;
  std::fprintf(stderr,"SIDE_REFERENCE grouped n=%lld k=%lld tokens=%lld slots=%lld type=%d\n",
      static_cast<long long>(n),static_cast<long long>(k),static_cast<long long>(tokens),
      static_cast<long long>(slots),int(type));std::fflush(stderr);
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
} // namespace cuda_sidecar_quant_reference

#define HYLE_C4_QUANTIZED_REFERENCE cuda_sidecar_quant_reference::projection
#define HYLE_C4_GROUPED_REFERENCE cuda_sidecar_quant_reference::grouped
