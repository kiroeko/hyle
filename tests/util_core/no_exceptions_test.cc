#include "hyle/util/scope_guard.h"
#include "hyle/util/unique_resource.h"

namespace {

[[maybe_unused]] void compile_no_exception_util_core() noexcept {
  bool exit = false;
  hyle::util::ScopeGuard guard{[&]() noexcept { exit = true; }};
  hyle::util::ScopeFail fail{[]() noexcept {}};
  hyle::util::ScopeSuccess success{[]() noexcept {}};
  hyle::util::UniqueResource resource{1, [](int) noexcept {}};
  fail.dismiss();
  success.dismiss();
  resource.reset();
  if (exit) guard.dismiss();
}

}  // namespace
