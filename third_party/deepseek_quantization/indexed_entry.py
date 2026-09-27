"""Index-only CUDA entry adapter. The generated original GEMM body is unchanged."""

def adapt(source, n, k):
    old = 'fp4_gemm_kernel__kernel'
    body = 'deepseek_original_fp4_gemm'
    prototype = 'extern "C" __global__ void ' + old
    definition = 'extern "C" __global__ void __launch_bounds__(128, 1) ' + old
    assert source.count(prototype) == 1 and source.count(definition) == 1
    source = source.replace(prototype, '__device__ __forceinline__ void ' + body)
    source = source.replace(definition, '__device__ __forceinline__ void ' + body)
    return source + f'''
// HYLE integration only: validated expert addressing, then original math.
static_assert(sizeof(fp4_e2_t) == 1);
extern "C" __global__ void __launch_bounds__(128, 1) deepseek_indexed_fp4(
    const fp8_e4_t* A, const fp4_e2_t* B, bfloat16_t* C,
    const float* scales_a, const float* scales_b, const int* ids,
    unsigned* status, int experts, int slots_per_token, int activation_slots) {{
  const unsigned slot = blockIdx.z;
  const int expert = ids[slot];
  if (expert < 0 || expert >= experts) {{
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicCAS(status, 0u, 1u);
    return;
  }}
  const auto row = static_cast<unsigned long long>(
      activation_slots == 1 ? slot / slots_per_token : slot);
  const auto selected = static_cast<unsigned long long>(expert);
  deepseek_original_fp4_gemm(A + row * {k}, B + selected * {n * k // 2}ULL,
      C + static_cast<unsigned long long>(slot) * {n},
      scales_a + row * {k // 128}, scales_b + selected * {n * k // 32}ULL, 1);
}}
'''
