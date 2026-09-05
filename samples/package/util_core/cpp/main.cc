#include <cstdint>
#include <unordered_set>

#include "hyle/util/borrow.h"
#include "hyle/util/function_ref.h"
#include "hyle/util/random.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/scope_guard.h"
#include "hyle/util/spin_lock.h"
#include "hyle/util/type_name.h"
#include "hyle/util/unique_resource.h"

namespace {

struct Item : hyle::util::RefCounted<Item> {
  explicit Item(int initial) : value(initial) {}
  int value;
};

int apply(hyle::util::FunctionRef<int(int) const> function, int value) {
  return function(value);
}

int increment(int value) noexcept { return value + 1; }

struct Decoder {
  int score(int input) const noexcept { return input + bias; }
  int bias;
};

struct MoveOnlyDeleter {
  MoveOnlyDeleter(int* released, int weight) noexcept
      : released(released), weight(weight) {}
  MoveOnlyDeleter(MoveOnlyDeleter&&) noexcept = default;
  MoveOnlyDeleter(const MoveOnlyDeleter&) = delete;

  void operator()(int token) const noexcept { *released += token * weight; }

  int* released;
  int weight;
};

}  // namespace

int main() {
  hyle::util::Rng rng{7};
  const std::uint64_t value = rng.range_u64(1, 9);
  if (value < 1 || value > 9) return 1;

  auto item = hyle::util::make_ref<Item>(3);
  std::unordered_set<hyle::util::Ref<Item>> owners{item};
  if (!owners.contains(item)) return 2;
  hyle::util::NotNull<Item*> required = item.get();
  if (apply([required](int input) { return input + required->value; }, 4) !=
      7) {
    return 3;
  }
  hyle::util::FunctionRef static_call{hyle::util::nontype<&increment>};
  if (static_call(7) != 8) return 4;
  Decoder decoder{3};
  hyle::util::FunctionRef bound_call{hyle::util::nontype<&Decoder::score>,
                                     decoder};
  if (bound_call(8) != 11) return 4;

  hyle::util::SpinLock lock;
  lock.lock();
  lock.unlock();

  bool cleaned = false;
  bool succeeded = false;
  {
    hyle::util::ScopeGuard cleanup{[&]() noexcept { cleaned = true; }};
    hyle::util::ScopeSuccess success{[&]() noexcept { succeeded = true; }};
  }
  if (!cleaned || !succeeded || hyle::util::type_name<Item>().empty()) return 5;

  int released = 0;
  {
    hyle::util::UniqueResource token{6, MoveOnlyDeleter{&released, 1}};
  }
  auto invalid = hyle::util::make_unique_resource_checked(
      -1, -1, MoveOnlyDeleter{&released, 2});
  if (released != 6 || invalid.allocated()) return 6;
  return 0;
}
