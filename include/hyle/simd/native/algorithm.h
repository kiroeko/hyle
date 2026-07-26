#pragma once

#include <cstddef>

#include "hyle/simd/vec.h"

// Target-bound whole-range algorithm helpers. These are intentionally part of
// the native/expert surface: Func receives target-specific vector values and is
// inlined into the selected ISA translation unit. To publish one kernel for an
// unknown CPU, put the call in a hyle_add_fat_kernel implementation header.
//
// Every helper owns the unaligned whole-vector loop and the final partial
// vector. Func is also invoked once for a zero-filled final vector when n is not
// a multiple of Lanes(d); StoreN / the explicit tail mask ensure that inactive
// lanes are never observed in memory or counted. Func must therefore be pure
// with respect to inactive lanes.

namespace hyle::simd::native {

// Writes func(d, indices) to out[0, n). indices has the unsigned lane type
// corresponding to T and starts at the current element offset.
template <typename T, class Func>
static HYLE_FORCE_INLINE void generate(T* out, std::size_t n,
                                       const Func& func) {
  const ScalableTag<T> d;
  const RebindToUnsigned<decltype(d)> du;
  using TU = TFromD<decltype(du)>;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    StoreU(func(d, Iota(du, static_cast<TU>(i))), d, out + i);
  }
  if (i != n) {
    StoreN(func(d, Iota(du, static_cast<TU>(i))), d, out + i, n - i);
  }
}

// Exact in-place unary transform. Offset overlaps are not supported.
template <typename T, class Func>
static HYLE_FORCE_INLINE void transform(T* inout, std::size_t n,
                                        const Func& func) {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    StoreU(func(d, LoadU(d, inout + i)), d, inout + i);
  }
  if (i != n) {
    const std::size_t remaining = n - i;
    StoreN(func(d, LoadN(d, inout + i, remaining)), d, inout + i,
           remaining);
  }
}

// Unary out-of-place transform. output may exactly equal input; other overlap
// is not supported.
template <typename T, class Func>
static HYLE_FORCE_INLINE void transform(const T* input, T* output,
                                        std::size_t n, const Func& func) {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    StoreU(func(d, LoadU(d, input + i)), d, output + i);
  }
  if (i != n) {
    const std::size_t remaining = n - i;
    StoreN(func(d, LoadN(d, input + i, remaining)), d, output + i,
           remaining);
  }
}

// Binary transform. output may exactly equal either input; other overlap is
// not supported.
template <typename T, class Func>
static HYLE_FORCE_INLINE void transform(const T* a, const T* b, T* output,
                                        std::size_t n, const Func& func) {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    StoreU(func(d, LoadU(d, a + i), LoadU(d, b + i)), d, output + i);
  }
  if (i != n) {
    const std::size_t remaining = n - i;
    StoreN(func(d, LoadN(d, a + i, remaining),
                LoadN(d, b + i, remaining)),
           d, output + i, remaining);
  }
}

// Ternary transform for fused expressions. output may exactly equal an input;
// other overlap is not supported.
template <typename T, class Func>
static HYLE_FORCE_INLINE void transform(const T* a, const T* b, const T* c,
                                        T* output, std::size_t n,
                                        const Func& func) {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    StoreU(func(d, LoadU(d, a + i), LoadU(d, b + i), LoadU(d, c + i)), d,
           output + i);
  }
  if (i != n) {
    const std::size_t remaining = n - i;
    StoreN(func(d, LoadN(d, a + i, remaining),
                LoadN(d, b + i, remaining), LoadN(d, c + i, remaining)),
           d, output + i, remaining);
  }
}

template <typename T>
static HYLE_FORCE_INLINE void replace(T* inout, std::size_t n, T old_value,
                                      T new_value) {
  transform(inout, n, [old_value, new_value](auto d, auto v) {
    return IfThenElse(Eq(v, Set(d, old_value)), Set(d, new_value), v);
  });
}

template <typename T, class Predicate>
static HYLE_FORCE_INLINE void replace_if(T* inout, std::size_t n, T new_value,
                                         const Predicate& predicate) {
  transform(inout, n, [new_value, &predicate](auto d, auto v) {
    return IfThenElse(predicate(d, v), Set(d, new_value), v);
  });
}

// Returns n when no active element satisfies predicate(d, vector).
template <typename T, class Predicate>
[[nodiscard]] static HYLE_FORCE_INLINE std::size_t find_if(
    const T* input, std::size_t n, const Predicate& predicate) {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto match = predicate(d, LoadU(d, input + i));
    const std::ptrdiff_t lane = FindFirstTrue(d, match);
    if (lane >= 0) return i + static_cast<std::size_t>(lane);
  }
  if (i != n) {
    const std::size_t remaining = n - i;
    const auto active = FirstN(d, remaining);
    const auto match = And(active, predicate(d, LoadN(d, input + i, remaining)));
    const std::ptrdiff_t lane = FindFirstTrue(d, match);
    if (lane >= 0) return i + static_cast<std::size_t>(lane);
  }
  return n;
}

template <typename T, class Predicate>
[[nodiscard]] static HYLE_FORCE_INLINE std::size_t count_if(
    const T* input, std::size_t n, const Predicate& predicate) {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t total = 0;
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    total += CountTrue(d, predicate(d, LoadU(d, input + i)));
  }
  if (i != n) {
    const std::size_t remaining = n - i;
    const auto active = FirstN(d, remaining);
    total += CountTrue(
        d, And(active, predicate(d, LoadN(d, input + i, remaining))));
  }
  return total;
}

// Stable compaction into output[0, result). output must have room for n items.
// Exact in-place compaction (output == input) is supported; offset overlaps are
// not. The returned count is the STL-style output size.
template <typename T, class Predicate>
[[nodiscard]] static HYLE_FORCE_INLINE std::size_t copy_if(
    const T* input, T* output, std::size_t n, const Predicate& predicate) {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t written = 0;
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto v = LoadU(d, input + i);
    written += CompressStore(v, predicate(d, v), d, output + written);
  }
  if (i != n) {
    const std::size_t remaining = n - i;
    const auto active = FirstN(d, remaining);
    const auto v = LoadN(d, input + i, remaining);
    written += CompressStore(v, And(active, predicate(d, v)), d,
                             output + written);
  }
  return written;
}

}  // namespace hyle::simd::native
