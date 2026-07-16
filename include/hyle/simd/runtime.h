#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dispatch.h"

namespace hyle::simd {

class RandomState;

struct Runtime {
  // Version of this C++ convenience table, not the stable C ABI from include/hyle/c/.
  static constexpr std::uint32_t kTableVersion = 2;

  struct Ops {
    float (*dot_f32)(const float*, const float*, std::size_t) noexcept;
    double (*dot_f64)(const double*, const double*, std::size_t) noexcept;
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
  };

  std::uint32_t table_version;
  std::uint32_t table_size;
  Isa isa;
  Ops ops;
  Random random;
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
