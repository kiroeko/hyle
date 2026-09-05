#include "hyle/core/compiler.h"
#include "hyle/util/borrow.h"
#include "hyle/util/function_ref.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

namespace {

struct AtomicObject : hyle::util::RefCounted<AtomicObject> {};
struct LocalObject : hyle::util::ThreadUnsafeRefCounted<LocalObject> {};

int increment(int value) noexcept { return value + 1; }

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

}  // extern "C"
