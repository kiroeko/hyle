#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/dtype.h"

namespace hyle::simd {

class RandomState;

struct Runtime {
  // Version of this C++ convenience table, independent of the pre-release FFI
  // schema candidate under include/hyle/c/.
  static constexpr std::uint32_t kTableVersion = 7;

  struct Ops {
    float (*dot_f32)(const float*, const float*, std::size_t) noexcept;
    double (*dot_f64)(const double*, const double*, std::size_t) noexcept;
    float (*dot_bf16_bf16)(const bfloat16_t*, const bfloat16_t*,
                           std::size_t) noexcept;
    float (*dot_f32_bf16)(const float*, const bfloat16_t*, std::size_t) noexcept;
    float (*dot_f16_f16)(const float16_t*, const float16_t*,
                         std::size_t) noexcept;
    std::int64_t (*dot_i16_i16)(const std::int16_t*, const std::int16_t*,
                                std::size_t) noexcept;
    std::int64_t (*dot_i8_i8)(const std::int8_t*, const std::int8_t*,
                              std::size_t) noexcept;
    void (*saxpy_f32)(float, const float*, float*, std::size_t) noexcept;
    void (*saxpy_f64)(double, const double*, double*, std::size_t) noexcept;
    void (*add_f32)(const float*, const float*, float*, std::size_t) noexcept;
    void (*add_f64)(const double*, const double*, double*, std::size_t) noexcept;
  };

  struct Random {
    void (*fill_u64)(RandomState*, std::uint64_t*, std::size_t) noexcept;
    void (*fill_f32)(RandomState*, float*, std::size_t) noexcept;
    void (*fill_f64)(RandomState*, double*, std::size_t) noexcept;
  };

  struct Search {
#define HYLE_RUNTIME_SEARCH_FIELDS(T, tag)                                    \
    std::size_t (*count_equal_##tag)(const T*, std::size_t, T) noexcept;      \
    std::size_t (*find_equal_##tag)(const T*, std::size_t, T) noexcept;       \
    T (*min_value_##tag)(const T*, std::size_t) noexcept;                     \
    T (*max_value_##tag)(const T*, std::size_t) noexcept;                     \
    void (*minmax_value_##tag)(const T*, std::size_t, T*, T*) noexcept;       \
    bool (*is_sorted_asc_##tag)(const T*, std::size_t) noexcept;              \
    bool (*is_sorted_desc_##tag)(const T*, std::size_t) noexcept;

    HYLE_RUNTIME_SEARCH_FIELDS(std::int8_t, i8)
    HYLE_RUNTIME_SEARCH_FIELDS(std::uint8_t, u8)
    HYLE_RUNTIME_SEARCH_FIELDS(std::int16_t, i16)
    HYLE_RUNTIME_SEARCH_FIELDS(std::uint16_t, u16)
    HYLE_RUNTIME_SEARCH_FIELDS(std::int32_t, i32)
    HYLE_RUNTIME_SEARCH_FIELDS(std::uint32_t, u32)
    HYLE_RUNTIME_SEARCH_FIELDS(std::int64_t, i64)
    HYLE_RUNTIME_SEARCH_FIELDS(std::uint64_t, u64)
    HYLE_RUNTIME_SEARCH_FIELDS(float, f32)
    HYLE_RUNTIME_SEARCH_FIELDS(double, f64)

#undef HYLE_RUNTIME_SEARCH_FIELDS
  };

  struct Narrow {
#define HYLE_RUNTIME_NARROW_FIELDS(T, tag)                                    \
    std::size_t (*count_equal_##tag)(const T*, std::size_t, T) noexcept;      \
    std::size_t (*find_equal_##tag)(const T*, std::size_t, T) noexcept;       \
    std::size_t (*count_equal_bits_##tag)(const T*, std::size_t, T) noexcept; \
    std::size_t (*find_equal_bits_##tag)(const T*, std::size_t, T) noexcept;  \
    float (*sum_##tag)(const T*, std::size_t) noexcept;                       \
    bool (*is_sorted_total_asc_##tag)(const T*, std::size_t) noexcept;        \
    bool (*is_sorted_total_desc_##tag)(const T*, std::size_t) noexcept;

    HYLE_RUNTIME_NARROW_FIELDS(float16_t, f16)
    HYLE_RUNTIME_NARROW_FIELDS(bfloat16_t, bf16)

#undef HYLE_RUNTIME_NARROW_FIELDS
  };

  // Module-wide selection metadata. Some modules choose individual implementations by precise
  // feature (for example F16C or AVX512-BF16), so `isa` is the broad tier and `features` records
  // the non-hierarchical choices. This remains a C++ diagnostic/control surface, not the C ABI.
  struct Module {
    Isa isa;
    std::uint64_t features;
  };

  enum Feature : std::uint64_t {
    kFeatureF16c = 1ull << 0,
    kFeatureAvx512Bf16 = 1ull << 1,
    kFeatureAvx10_2 = 1ull << 2,
    kFeatureAvxVnni = 1ull << 3,
    kFeatureAvx512Vnni = 1ull << 4,
    // Module contract/diagnostic flags. `Module::isa` is the broad product tier; these bits make
    // intentional per-op fallbacks and scalable-model constraints observable.
    kFeatureMathExactLibmScalar = 1ull << 16,
    kFeatureDtypeFp8Scalar = 1ull << 17,
    kFeaturePerOpLowerTier = 1ull << 18,
    kFeatureSortCappedNetwork = 1ull << 19,
    kFeatureBitpackLogical128 = 1ull << 20,
  };

  std::uint32_t table_version;
  std::uint32_t table_size;
  Isa isa;
  Ops ops;
  Random random;
  Search search;
  Narrow narrow;
  Module ops_module;
  Module scan;
  Module dtype;
  Module math;
  Module sort;
  Module bitpack;
};

// Automatic process-wide choice, resolved once. The returned table is immutable and lives for
// the process lifetime.
HYLE_SIMD_API const Runtime& runtime() noexcept;

// Exact compiled tier for tests, power policy or explicit pinning. Returns nullptr if the tier is
// not compiled for this architecture or the current CPU/OS cannot safely execute it. Scalar is
// always available.
HYLE_SIMD_API const Runtime* runtime_for(Isa isa) noexcept;

}  // namespace hyle::simd
