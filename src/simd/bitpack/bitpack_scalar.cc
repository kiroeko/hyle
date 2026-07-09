// Scalar bitpack floor -- the pure contiguous bit-stream codec (bitstream.h). ALWAYS compiled
// (every arch, the portable TU in the module base SOURCES); it is BOTH the runtime fallback when
// no SIMD tier is available (old x86 without AVX2, or a non-x86/non-arm64 target) AND the tail
// helper the SIMD tiers reuse for sub-block leftovers. NOT a stub.
//
// The whole-array scalar codec and the SIMD tiers' tail use the SAME bitstream_pack/unpack, so a
// same-host round trip is exact regardless of which tier ran (see bitpack.h layout contract).
#include "src/simd/bitpack/kernels.h"

#include <cstddef>
#include <cstdint>

#include "src/simd/bitpack/bitstream.h"

namespace hyle::simd::detail {

#define HYLE_BITPACK_SCALAR_DEFINE(T, tag)                                              \
  void pack_##tag##_scalar(const T* raw, std::size_t count, T* packed,                  \
                           unsigned bits) noexcept {                                    \
    bitstream_pack<T>(raw, count, packed, bits);                                        \
  }                                                                                     \
  void unpack_##tag##_scalar(const T* packed, std::size_t count, T* raw,                \
                             unsigned bits) noexcept {                                  \
    bitstream_unpack<T>(packed, count, raw, bits);                                      \
  }

HYLE_BITPACK_TYPES(HYLE_BITPACK_SCALAR_DEFINE)
#undef HYLE_BITPACK_SCALAR_DEFINE

}  // namespace hyle::simd::detail
