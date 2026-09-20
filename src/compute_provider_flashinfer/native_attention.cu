#include "native_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>

namespace hyle::compute::flashinfer::native {
namespace {

template <class T>
cudaError_t execute_typed(const Attention &attention, const void *query,
                          const void *key, const void *value, void *output,
                          void *workspace, cudaStream_t stream) {
  // Upstream params use mutable pointer types for read-only Q/K/V operands.
  auto *q = const_cast<T *>(static_cast<const T *>(query));
  auto *k = const_cast<T *>(static_cast<const T *>(key));
  auto *v = const_cast<T *>(static_cast<const T *>(value));
  auto *o = static_cast<T *>(output);
  auto *temporary = static_cast<T *>(workspace);
  using Variant = ::flashinfer::DefaultAttention<false, false, false, false>;
  if (attention.query_length == 1) {
    ::flashinfer::SingleDecodeParams<T, T, T> params(
        q, k, v, o, nullptr, attention.kv_length, 8, 2,
        ::flashinfer::QKVLayout::kNHD, 256, -1, 0.0F, attention.scale, 1.0F,
        10000.0F);
    return ::flashinfer::SingleDecodeWithKVCacheDispatched<
        256, ::flashinfer::PosEncodingMode::kNone, Variant>(params, temporary,
                                                            stream);
  }
  ::flashinfer::SinglePrefillParams<T, T, T> params(
      q, k, v, nullptr, o, nullptr, nullptr, 8, 2, attention.query_length,
      attention.kv_length, 8 * 256, 256, 2 * 256, 256, 256, -1, 0.0F,
      attention.scale, 1.0F, 10000.0F);
  return ::flashinfer::SinglePrefillWithKVCacheDispatched<
      256, 256, ::flashinfer::PosEncodingMode::kNone, false,
      ::flashinfer::MaskMode::kCausal, Variant>(params, temporary, stream);
}

} // namespace

cudaError_t execute(const Attention &attention, const void *query,
                    const void *key, const void *value, void *output,
                    void *workspace, cudaStream_t stream) {
  if (attention.bfloat16)
    return execute_typed<__nv_bfloat16>(attention, query, key, value, output,
                                        workspace, stream);
  return execute_typed<half>(attention, query, key, value, output, workspace,
                             stream);
}

} // namespace hyle::compute::flashinfer::native
