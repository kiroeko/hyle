#pragma once

#include <utility>

#include "hyle/compute/result.h"

#if defined(_WIN32)
#if defined(HYLE_RESULT_FIXTURE_BUILD)
#define HYLE_RESULT_FIXTURE_API __declspec(dllexport)
#else
#define HYLE_RESULT_FIXTURE_API __declspec(dllimport)
#endif
#else
#define HYLE_RESULT_FIXTURE_API __attribute__((visibility("default")))
#endif

namespace hyle::compute::test {

struct SharedResultOwner final {
  int value = 0;
  int *destroyed = nullptr;
  SharedResultOwner(int number, int &counter) noexcept
      : value(number), destroyed(&counter) {}
  SharedResultOwner(SharedResultOwner &&other) noexcept
      : value(other.value), destroyed(std::exchange(other.destroyed, nullptr)) {
  }
  HYLE_RESULT_FIXTURE_API ~SharedResultOwner() noexcept;
};

HYLE_RESULT_FIXTURE_API Result<OwnedResult<SharedResultOwner>>
make_shared_result(int value, int &destroyed) noexcept;

} // namespace hyle::compute::test

#undef HYLE_RESULT_FIXTURE_API
