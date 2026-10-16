#include "../../src/compute_provider_cuda/sampling_math.cuh"

namespace {
__global__ void probe(const std::uint32_t *words,float *output,std::size_t count) {
  const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count) output[i]=hyle::compute::cuda_native::sampling::math::gumbel(words[i*2],words[i*2+1]);
}
}
// Private test bridge; no HYLE product API or test-only operation is added.
cudaError_t hyle_g08_cuda_math_probe(const std::uint32_t *words,float *output,std::size_t count) {
  std::uint32_t *input=nullptr;float *result=nullptr;
  auto status=cudaMalloc(&input,count*2*sizeof(std::uint32_t));
  if(status!=cudaSuccess)return status;
  status=cudaMalloc(&result,count*sizeof(float));
  if(status==cudaSuccess)status=cudaMemcpy(input,words,count*2*sizeof(std::uint32_t),cudaMemcpyHostToDevice);
  if(status==cudaSuccess) {
    void *arguments[]{&input,&result,&count};
    status=cudaLaunchKernel(reinterpret_cast<const void *>(&probe),
        dim3(unsigned((count+255)/256)),dim3(256),arguments,0,nullptr);
  }
  if(status==cudaSuccess)status=cudaMemcpy(output,result,count*sizeof(float),cudaMemcpyDeviceToHost);
  const auto free_result=result?cudaFree(result):cudaSuccess;
  const auto free_input=cudaFree(input);
  return status!=cudaSuccess?status:free_result!=cudaSuccess?free_result:free_input;
}
