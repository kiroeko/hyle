#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>

// Scalar math helpers (clay math/utils.h, distilled). clay's 353-line utils.h is
// overwhelmingly superseded by the C++20/23 standard library -- the same kind of
// modernization win as result.h -> std::expected:
//   * bit tricks (is_power_of_2 / next_pow2 / log2_* / ilog2 / clz /
//     round_up_to_pot) -> <bit> (has_single_bit / bit_ceil / bit_floor /
//     bit_width / countl_zero);
//   * clamp / min3 / max3 / minn / maxn -> std::clamp / std::max({...});
//   * pow / is_normal / abs -> <cmath> (constexpr in C++23);
//   * numbers/consts.h (clay itself flags "will be replaced") -> <numbers> +
//     std::numeric_limits;
//   * align_up -> hyle::mem::align_up (mem/alignment.h, power-of-two).
// What the standard genuinely lacks, and only these, live here. Domain-specific
// graphics helpers (deg2rad/rad2deg/fract/hypot2) are out of scope for general
// infra -- they belong to a consumer, not the base library.

namespace hyle::math {

// Ceiling of a/b for non-negative integers, b > 0. The standard has floor
// division (operator/) but no ceil. Computed as a/b + (a%b != 0) rather than
// clay's (a+b-1)/b: the latter overflows when a is near the type max, this never
// does (and the compiler folds both / and % into one div on x86).
template <std::unsigned_integral T>
constexpr T div_ceil(T a, T b) noexcept {
  return a / b + (a % b != 0 ? T{1} : T{0});
}

// Smallest multiple of `m` that is >= n, for ANY positive m (not just powers of
// two -- that case is hyle::mem::align_up, which is cheaper via bit masking).
// m > 0 required. Uses div_ceil so it shares the overflow-safe path.
template <std::unsigned_integral T>
constexpr T round_up_to_multiple(T n, T m) noexcept {
  return div_ceil(n, m) * m;
}

// Sign of `x` with a dead zone: -1 below -eps, +1 above +eps, else 0. Distinct
// from a bare (x>0)-(x<0): the eps band treats near-zero as zero, which is what
// you want for floating-point geometry/thresholds. eps defaults small but is a
// parameter because the right band is domain-specific.
template <std::floating_point T>
constexpr int sgn(T x, T eps = static_cast<T>(1e-6)) noexcept {
  if (x < -eps) return -1;
  if (x > eps) return 1;
  return 0;
}

// Relative-epsilon float equality (clay equal_adjusted_eps): two values are
// "equal" if their gap is within tol scaled by the SMALLER magnitude. A fixed
// absolute eps is wrong across scales (1e9 vs 1e9+1 should compare equal; 1e-9
// vs 2e-9 should not) -- this scales the tolerance to the operands.
template <std::floating_point T>
constexpr bool approx_eq(
    T a, T b, T tol = std::numeric_limits<T>::epsilon()) noexcept {
  T fa = a < 0 ? -a : a;
  T fb = b < 0 ? -b : b;
  T diff = a > b ? a - b : b - a;
  return diff <= (fa < fb ? fa : fb) * (tol < 0 ? -tol : tol);
}

}  // namespace hyle::math
