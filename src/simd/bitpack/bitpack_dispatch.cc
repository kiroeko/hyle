#include "hyle/simd/bitpack.h"

#include <cstddef>
#include <cstdint>

#include "hyle/core/compiler.h"
#include "src/simd/bitpack/kernels.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/runtime_internal.h"

namespace hyle::simd {

#if defined(HYLE_SIMD_EXACT)
#define HYLE_BITPACK_CALL(name, ...) HYLE_SIMD_EXACT_SYMBOL(name)(__VA_ARGS__)
#else
#define HYLE_BITPACK_CALL(name, ...) runtime().bitpack_ops.name(__VA_ARGS__)
#endif

#define HYLE_BITPACK_PUBLIC(T, tag)                                            \
  void pack(const T *raw, std::size_t count, T *packed,                        \
            unsigned bits) noexcept {                                          \
    HYLE_FATAL_ASSERT(bits >= 1 && bits <= 8 * sizeof(T));                     \
    if (count == 0)                                                            \
      return;                                                                  \
    HYLE_BITPACK_CALL(pack_##tag, raw, count, packed, bits);                   \
  }                                                                            \
  void unpack(const T *packed, std::size_t count, T *raw,                      \
              unsigned bits) noexcept {                                        \
    HYLE_FATAL_ASSERT(bits >= 1 && bits <= 8 * sizeof(T));                     \
    if (count == 0)                                                            \
      return;                                                                  \
    HYLE_BITPACK_CALL(unpack_##tag, packed, count, raw, bits);                 \
  }

HYLE_BITPACK_TYPES(HYLE_BITPACK_PUBLIC)
#undef HYLE_BITPACK_PUBLIC
#undef HYLE_BITPACK_CALL

} // namespace hyle::simd
