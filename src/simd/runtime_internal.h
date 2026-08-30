#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dtype.h"
#include "hyle/simd/runtime.h"

namespace hyle::simd {

class RandomState;

// Private implementation table. Vendor vectors never cross this boundary and
// ordinary calls pay one function-pointer hop per whole buffer. Its layout and
// per-tier accessors are deliberately not part of the installed C++ surface.
struct Runtime {
  static constexpr std::uint32_t kTableVersion = 11;

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

#define HYLE_RUNTIME_ROW_EXTRA_FIELDS(T, tag)                                  \
    void (*sub_##tag)(const T*, const T*, T*, std::size_t) noexcept;           \
    void (*mul_##tag)(const T*, const T*, T*, std::size_t) noexcept;           \
    void (*div_##tag)(const T*, const T*, T*, std::size_t) noexcept;           \
    void (*scale_##tag)(const T*, T, T*, std::size_t) noexcept;                \
    void (*add_scalar_##tag)(const T*, T, T*, std::size_t) noexcept;           \
    void (*affine_##tag)(const T*, T, T, T*, std::size_t) noexcept;            \
    T (*sum_##tag)(const T*, std::size_t) noexcept;                            \
    void (*abs_##tag)(const T*, T*, std::size_t) noexcept;                     \
    void (*neg_##tag)(const T*, T*, std::size_t) noexcept;                     \
    void (*square_##tag)(const T*, T*, std::size_t) noexcept;                  \
    void (*sqrt_##tag)(const T*, T*, std::size_t) noexcept;

    HYLE_RUNTIME_ROW_EXTRA_FIELDS(float, f32)
    HYLE_RUNTIME_ROW_EXTRA_FIELDS(double, f64)

#undef HYLE_RUNTIME_ROW_EXTRA_FIELDS
  };

  struct Random {
    void (*fill_u64)(RandomState*, std::uint64_t*, std::size_t) noexcept;
    void (*fill_f32)(RandomState*, float*, std::size_t) noexcept;
    void (*fill_f64)(RandomState*, double*, std::size_t) noexcept;
    void (*advance)(RandomState*, std::size_t) noexcept;
  };

  struct Search {
#define HYLE_RUNTIME_SEARCH_FIELDS(T, tag)                                \
    std::size_t (*count_equal_##tag)(const T*, std::size_t, T) noexcept;  \
    std::size_t (*find_equal_##tag)(const T*, std::size_t, T) noexcept;   \
    T (*min_value_##tag)(const T*, std::size_t) noexcept;                 \
    T (*max_value_##tag)(const T*, std::size_t) noexcept;                 \
    std::size_t (*index_of_min_##tag)(const T*, std::size_t) noexcept;    \
    std::size_t (*index_of_max_##tag)(const T*, std::size_t) noexcept;    \
    void (*minmax_value_##tag)(const T*, std::size_t, T*, T*) noexcept;   \
    bool (*is_sorted_asc_##tag)(const T*, std::size_t) noexcept;          \
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

  struct Scan {
#define HYLE_RUNTIME_SCAN_FIELDS(T, tag)                                  \
    void (*inclusive_scan_##tag)(const T*, T*, std::size_t) noexcept;     \
    void (*exclusive_scan_##tag)(const T*, T*, std::size_t, T) noexcept;

    HYLE_RUNTIME_SCAN_FIELDS(std::uint32_t, u32)
    HYLE_RUNTIME_SCAN_FIELDS(std::int32_t, i32)
    HYLE_RUNTIME_SCAN_FIELDS(std::uint64_t, u64)
    HYLE_RUNTIME_SCAN_FIELDS(std::int64_t, i64)
    HYLE_RUNTIME_SCAN_FIELDS(float, f32)
    HYLE_RUNTIME_SCAN_FIELDS(double, f64)

#undef HYLE_RUNTIME_SCAN_FIELDS
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

  struct DtypeConvert {
    void (*f16_to_f32)(const float16_t*, float*, std::size_t) noexcept;
    void (*f32_to_f16)(const float*, float16_t*, std::size_t) noexcept;
    void (*bf16_to_f32)(const bfloat16_t*, float*, std::size_t) noexcept;
    void (*f32_to_bf16)(const float*, bfloat16_t*, std::size_t) noexcept;
    void (*f16_to_bf16)(const float16_t*, bfloat16_t*, std::size_t) noexcept;
    void (*bf16_to_f16)(const bfloat16_t*, float16_t*, std::size_t) noexcept;
  };

  struct Bitpack {
    void (*pack_u8)(const std::uint8_t*, std::size_t, std::uint8_t*,
                    unsigned) noexcept;
    void (*unpack_u8)(const std::uint8_t*, std::size_t, std::uint8_t*,
                      unsigned) noexcept;
    void (*pack_u16)(const std::uint16_t*, std::size_t, std::uint16_t*,
                     unsigned) noexcept;
    void (*unpack_u16)(const std::uint16_t*, std::size_t, std::uint16_t*,
                       unsigned) noexcept;
    void (*pack_u32)(const std::uint32_t*, std::size_t, std::uint32_t*,
                     unsigned) noexcept;
    void (*unpack_u32)(const std::uint32_t*, std::size_t, std::uint32_t*,
                       unsigned) noexcept;
  };

  struct Sort {
#define HYLE_RUNTIME_SORT_FIELDS(T, tag)                                      \
    void (*sort_##tag)(T*, std::size_t, bool) noexcept;                       \
    void (*partial_sort_##tag)(T*, std::size_t, std::size_t, bool) noexcept;  \
    void (*select_##tag)(T*, std::size_t, std::size_t, bool) noexcept;

    HYLE_RUNTIME_SORT_FIELDS(std::uint16_t, u16)
    HYLE_RUNTIME_SORT_FIELDS(std::int16_t, i16)
    HYLE_RUNTIME_SORT_FIELDS(std::uint32_t, u32)
    HYLE_RUNTIME_SORT_FIELDS(std::int32_t, i32)
    HYLE_RUNTIME_SORT_FIELDS(std::uint64_t, u64)
    HYLE_RUNTIME_SORT_FIELDS(std::int64_t, i64)
    HYLE_RUNTIME_SORT_FIELDS(float, f32)
    HYLE_RUNTIME_SORT_FIELDS(double, f64)
    HYLE_RUNTIME_SORT_FIELDS(UInt128, u128)
    HYLE_RUNTIME_SORT_FIELDS(KeyValue32, kv32)
    HYLE_RUNTIME_SORT_FIELDS(KeyValue64, kv64)

#undef HYLE_RUNTIME_SORT_FIELDS
  };

  struct Module {
    Isa isa;
    std::uint64_t features;
  };

  static constexpr std::uint64_t kFeatureF16c =
      target_feature_bits(TargetFeature::kF16c);
  static constexpr std::uint64_t kFeatureAvx512Bf16 =
      target_feature_bits(TargetFeature::kAvx512Bf16);
  static constexpr std::uint64_t kFeatureAvx10_2 =
      target_feature_bits(TargetFeature::kAvx10_2);
  static constexpr std::uint64_t kFeatureAvxVnni =
      target_feature_bits(TargetFeature::kAvxVnni);
  static constexpr std::uint64_t kFeatureAvx512Vnni =
      target_feature_bits(TargetFeature::kAvx512Vnni);
  static constexpr std::uint64_t kFeatureMathExactLibmScalar =
      target_feature_bits(TargetFeature::kMathExactLibmScalar);
  static constexpr std::uint64_t kFeatureDtypeFp8Scalar =
      target_feature_bits(TargetFeature::kDtypeFp8Scalar);
  static constexpr std::uint64_t kFeaturePerOpLowerTier =
      target_feature_bits(TargetFeature::kPerOpLowerTier);
  static constexpr std::uint64_t kFeatureSortCappedNetwork =
      target_feature_bits(TargetFeature::kSortCappedNetwork);
  static constexpr std::uint64_t kFeatureBitpackLogical128 =
      target_feature_bits(TargetFeature::kBitpackLogical128);

  std::uint32_t table_version;
  std::uint32_t table_size;
  Isa isa;
  Ops ops;
  Random random;
  Search search;
  Scan scan_ops;
  Narrow narrow;
  DtypeConvert dtype_convert;
  Bitpack bitpack_ops;
  Sort sort_ops;
  Module ops_module;
  Module random_module;
  Module search_module;
  Module scan;
  Module narrow_module;
  Module dtype;
  Module math;
  Module sort;
  Module bitpack;
};

#if defined(HYLE_SIMD_INTERNAL_TESTING)
#  define HYLE_SIMD_INTERNAL_TEST_API HYLE_SIMD_API
#else
#  define HYLE_SIMD_INTERNAL_TEST_API
#endif

HYLE_SIMD_INTERNAL_TEST_API const Runtime& runtime() noexcept;
HYLE_SIMD_INTERNAL_TEST_API const Runtime* runtime_for(Isa isa) noexcept;

#undef HYLE_SIMD_INTERNAL_TEST_API

}  // namespace hyle::simd
