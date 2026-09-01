#include <cstdint>
#include <unordered_set>

#include "hyle/util/borrow.h"
#include "hyle/util/function_ref.h"
#include "hyle/util/random.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/scope_guard.h"
#include "hyle/util/spin_lock.h"
#include "hyle/util/type_name.h"

namespace {

struct Item : hyle::util::RefCounted<Item> {
  explicit Item(int initial) : value(initial) {}
  int value;
};

int apply(hyle::util::FunctionRef<int(int)> function, int value) {
  return function(value);
}

}  // namespace

int main() {
  hyle::util::Rng rng{7};
  const std::uint64_t value = rng.range_u64(1, 9);
  if (value < 1 || value > 9) return 1;

  auto item = hyle::util::make_ref<Item>(3);
  std::unordered_set<hyle::util::Ref<Item>> owners{item};
  if (!owners.contains(item)) return 2;
  hyle::util::NotNull<Item*> required = item.get();
  if (apply([required](int input) { return input + required->value; }, 4) != 7) {
    return 3;
  }

  hyle::util::SpinLock lock;
  lock.lock();
  lock.unlock();

  bool cleaned = false;
  {
    hyle::util::ScopeGuard cleanup{[&]() noexcept { cleaned = true; }};
  }
  if (!cleaned || hyle::util::type_name<Item>().empty()) return 4;
  return 0;
}
