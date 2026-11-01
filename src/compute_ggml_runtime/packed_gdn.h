#pragma once

#include <cstdint>

// Same-build private R1/K1, all-F32 native GDN length contract. Src[8] is the
// admitted, status-checked I32[2] offset carrier: [0, L], 0 <= L <= physical T.
// Invalid/admitted-error controls use a negative sentinel and cause no native
// Y/state writes. Metadata alone confers no read/write permission.
//
// Physical tensor strides and the state-output tail retain physical T. Only
// the original recurrent loop uses L; Y[L:T] is +0 without reading padding.
// L == 0 copies initial state bits, including NaNs and signed zero, without
// evaluating recurrent arithmetic. Existing ordinary/indexed programs do not
// opt in. The combined indexed variant requires both original pool/address
// controls at src[5:8] and this separate length marker at parameter 5.
namespace hyle::compute::ggml_cuda::internal {
inline constexpr std::int32_t kPackedGdnR1Contract = 0x47504c31;
inline constexpr std::int32_t kPackedGdnR1IndexedContract = 0x47504931;
inline constexpr int kPackedGdnR1Parameter = 5;
} // namespace hyle::compute::ggml_cuda::internal

// Cold callers verify the actual linked native implementation, not just a
// marker from this header. This private bridge is not an installed API.
extern "C" std::int32_t ggml_hyle_cuda_gdn_r1_packed_contract();
extern "C" std::int32_t ggml_hyle_cuda_gdn_r1_indexed_packed_contract();
