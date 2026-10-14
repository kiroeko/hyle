#include "native_cache_encoder.h"
#include "../compute_backend_cuda/device_status_kernel.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>

namespace hyle::compute::flashinfer::native {
namespace {
template<class T> __global__ void encode(const T *input,std::uint8_t *output,
    std::uint64_t count,float scale,unsigned *status){
  for(auto index=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;index<count;
      index+=std::uint64_t(gridDim.x)*blockDim.x){
    const auto value=static_cast<float>(input[index]);
    if(!isfinite(value)){
      atomicCAS(status,0u,static_cast<unsigned>(cuda::detail::DeviceStatusCode::kInvalidNumericInput));
      output[index]=0;
    }else output[index]=__nv_cvt_float_to_fp8(value/scale,__NV_SATFINITE,__NV_E4M3);
  }
}
template<class T>cudaError_t launch(EncoderSpec spec,const void *input,void *output,unsigned *status,cudaStream_t stream){
  const auto blocks=static_cast<unsigned>((std::min)(std::uint64_t(65535),(spec.elements+255)/256));
  encode<<<blocks,256,0,stream>>>(static_cast<const T*>(input),static_cast<std::uint8_t*>(output),spec.elements,spec.scale,status);
  return cudaGetLastError();
}
}
cudaError_t encode_cache(EncoderSpec spec,const void *input,void *output,unsigned *status,cudaStream_t stream){
  if(!spec.elements)return cudaSuccess;
  switch(spec.input){
  case EncoderInput::kFloat16:return launch<half>(spec,input,output,status,stream);
  case EncoderInput::kBFloat16:return launch<__nv_bfloat16>(spec,input,output,status,stream);
  case EncoderInput::kFloat32:return launch<float>(spec,input,output,status,stream);
  }
  return cudaErrorInvalidValue;
}
}
