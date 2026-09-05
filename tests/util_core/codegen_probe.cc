#include "hyle/core/compiler.h"
#include "hyle/util/borrow.h"
#include "hyle/util/function_ref.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"
#include "hyle/util/unique_resource.h"

namespace {

struct AtomicObject : hyle::util::RefCounted<AtomicObject> {};
struct LocalObject : hyle::util::ThreadUnsafeRefCounted<LocalObject> {};

int increment(int value) noexcept { return value + 1; }

struct Scorer {
  int score(int value) const noexcept { return value + bias; }
  int bias;
};

struct MoveOnlyProbeDeleter {
  explicit MoveOnlyProbeDeleter(int* released) noexcept : released(released) {}
  MoveOnlyProbeDeleter(MoveOnlyProbeDeleter&&) noexcept = default;
  MoveOnlyProbeDeleter(const MoveOnlyProbeDeleter&) = delete;

  void operator()(int value) const noexcept { *released += value; }

  int* released;
};

}  // namespace

// Stable names for maintenance-only object inspection. These symbols are not
// linked into an installed artifact and do not create a Native FFI surface.
extern "C" {

HYLE_NOINLINE void hyle_util_core_probe_atomic_incref(
    AtomicObject* object) noexcept {
  object->incref();
}

HYLE_NOINLINE void hyle_util_core_probe_local_incref(
    LocalObject* object) noexcept {
  object->incref();
}

HYLE_NOINLINE int* hyle_util_core_probe_not_null_identity(
    hyle::util::NotNull<int*> pointer) noexcept {
  return pointer.get();
}

HYLE_NOINLINE void hyle_util_core_probe_cpu_relax() noexcept {
  hyle::util::cpu_relax();
}

HYLE_NOINLINE int hyle_util_core_probe_function_ref_const(
    hyle::util::FunctionRef<int(int) const noexcept> function,
    int value) noexcept {
  return function(value);
}

HYLE_NOINLINE int hyle_util_core_probe_function_ref_nontype(
    int value) noexcept {
  hyle::util::FunctionRef reference{hyle::util::nontype<&increment>};
  return reference(value);
}

HYLE_NOINLINE int hyle_util_core_probe_function_ref_bound_ctad(
    const Scorer& scorer, int value) noexcept {
  hyle::util::FunctionRef function{hyle::util::nontype<&Scorer::score>, scorer};
  return function(value);
}

HYLE_NOINLINE void hyle_util_core_probe_unique_resource_move_only(
    int value, int* released) noexcept {
  hyle::util::UniqueResource owner{value, MoveOnlyProbeDeleter{released}};
}

}  // extern "C"
