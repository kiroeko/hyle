#include "hyle/util/scope_guard.h"
#include "hyle/util/unique_resource.h"

namespace {

struct MoveOnlyDeleter {
  explicit MoveOnlyDeleter(int* released) noexcept : released(released) {}
  MoveOnlyDeleter(MoveOnlyDeleter&&) noexcept = default;
  MoveOnlyDeleter& operator=(MoveOnlyDeleter&&) noexcept = default;
  MoveOnlyDeleter(const MoveOnlyDeleter&) = delete;
  MoveOnlyDeleter& operator=(const MoveOnlyDeleter&) = delete;

  void operator()(int value) const noexcept { *released += value; }

  int* released;
};

[[maybe_unused]] void compile_no_exception_util_core() noexcept {
  bool exit = false;
  hyle::util::ScopeGuard guard{[&]() noexcept { exit = true; }};
  hyle::util::ScopeFail fail{[]() noexcept {}};
  hyle::util::ScopeSuccess success{[]() noexcept {}};
  hyle::util::UniqueResource resource{1, [](int) noexcept {}};
  int released = 0;
  auto move_only = hyle::util::make_unique_resource_checked(
      2, -1, MoveOnlyDeleter{&released});
  hyle::util::UniqueResource moved{std::move(move_only)};
  fail.dismiss();
  success.dismiss();
  resource.reset();
  moved.reset();
  if (exit) guard.dismiss();
}

}  // namespace
