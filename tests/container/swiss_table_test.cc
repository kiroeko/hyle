#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "hyle/container/swiss_table.h"
#include "hyle/mem/allocator.h"

using hyle::container::SwissTable;
using hyle::container::FlatHashMap;
using hyle::container::FlatHashSet;
using hyle::mem::system_allocator;

static_assert(std::forward_iterator<FlatHashMap<int, int>::iterator>);
static_assert(std::forward_iterator<FlatHashMap<int, int>::const_iterator>);
static_assert(std::forward_iterator<FlatHashSet<int>::iterator>);
static_assert(std::is_copy_constructible_v<FlatHashMap<int, std::string>>);
static_assert(!std::is_copy_constructible_v<
              FlatHashMap<int, std::unique_ptr<int>>>);

namespace {

void* fail_alloc(void*, std::size_t, std::size_t) noexcept { return nullptr; }
void fail_free(void*, void*, std::size_t, std::size_t) noexcept {}
void* fail_realloc(void*, void*, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr;
}

struct TransparentStringHash {
  std::uint64_t operator()(const std::string& value) const noexcept {
    return operator()(std::string_view(value));
  }
  std::uint64_t operator()(std::string_view value) const noexcept {
    return hyle::container::hash_bytes(value.data(), value.size());
  }
};

struct TransparentStringEq {
  bool operator()(const std::string& left,
                  const std::string& right) const noexcept {
    return left == right;
  }
  bool operator()(const std::string& left,
                  std::string_view right) const noexcept {
    return left == right;
  }
};

struct CountedMapValue {
  static int constructions;
  int number;

  explicit CountedMapValue(int value) noexcept : number(value) {
    ++constructions;
  }
  CountedMapValue() noexcept : CountedMapValue(0) {}
  CountedMapValue(CountedMapValue&& other) noexcept : number(other.number) {
    ++constructions;
  }
  CountedMapValue& operator=(CountedMapValue&&) noexcept = default;
};

int CountedMapValue::constructions = 0;

struct ThrowingMapValue {
  explicit ThrowingMapValue(int input) noexcept : value(input) { ++alive; }
  ThrowingMapValue(const ThrowingMapValue& other) : value(other.value) {
    if (copies_before_throw == 0)
      throw std::runtime_error("copy failed");
    --copies_before_throw;
    ++alive;
  }
  ThrowingMapValue(ThrowingMapValue&& other) noexcept : value(other.value) {
    ++alive;
  }
  ThrowingMapValue& operator=(ThrowingMapValue&& other) noexcept {
    value = other.value;
    return *this;
  }
  ~ThrowingMapValue() noexcept { --alive; }

  static inline int alive = 0;
  static inline int copies_before_throw = 0;
  int value;
};

}  // namespace

TEST(SwissTable, EmptyFindMisses) {
  SwissTable<int, int> t;
  EXPECT_EQ(t.size(), 0u);
  EXPECT_EQ(t.find(42), nullptr);
  EXPECT_FALSE(t.contains(7));
}

TEST(SwissTable, InsertFindBasic) {
  SwissTable<int, int> t;
  EXPECT_TRUE(t.insert(1, 10));
  EXPECT_TRUE(t.insert(2, 20));
  EXPECT_TRUE(t.insert(3, 30));
  EXPECT_EQ(t.size(), 3u);
  ASSERT_NE(t.find(2), nullptr);
  EXPECT_EQ(*t.find(2), 20);
  EXPECT_EQ(*t.find(1), 10);
  EXPECT_EQ(*t.find(3), 30);
  EXPECT_EQ(t.find(99), nullptr);
}

TEST(SwissTable, InsertDoesNotOverwriteAndInsertOrAssignDoes) {
  SwissTable<int, int> t;
  EXPECT_TRUE(t.insert(5, 50));
  EXPECT_FALSE(t.insert(5, 51));
  EXPECT_EQ(t.size(), 1u);
  EXPECT_EQ(*t.find(5), 50);
  EXPECT_FALSE(t.insert_or_assign(5, 51));
  EXPECT_EQ(*t.find(5), 51);
}

TEST(SwissTable, GrowsAcrossManyInserts) {
  SwissTable<int, int> t;
  for (int i = 0; i < 10000; ++i) ASSERT_TRUE(t.insert(i, i * 2));
  EXPECT_EQ(t.size(), 10000u);
  EXPECT_GT(t.capacity(), 10000u);  // grew past load factor
  for (int i = 0; i < 10000; ++i) {
    ASSERT_NE(t.find(i), nullptr) << "lost " << i;
    EXPECT_EQ(*t.find(i), i * 2);
  }
}

// Differential test vs std::unordered_map (the std map is an oracle ONLY in
// tests; production never uses it -- conventions §1). Random insert/find over
// a shared key space must agree at every step.
TEST(SwissTable, MatchesUnorderedMapOracle) {
  SwissTable<std::uint64_t, std::uint64_t> t;
  std::unordered_map<std::uint64_t, std::uint64_t> oracle;
  std::mt19937_64 rng(12345);
  for (int op = 0; op < 50000; ++op) {
    std::uint64_t k = rng() % 5000;  // dense space -> updates + collisions
    std::uint64_t v = rng();
    bool ins = t.insert_or_assign(k, v);
    bool oracle_new = oracle.find(k) == oracle.end();
    EXPECT_EQ(ins, oracle_new) << "insert/update disagreement at op " << op;
    oracle[k] = v;
    // spot-check a random key each iteration
    std::uint64_t q = rng() % 5000;
    auto oit = oracle.find(q);
    std::uint64_t* tv = t.find(q);
    if (oit == oracle.end()) {
      EXPECT_EQ(tv, nullptr) << "phantom key " << q;
    } else {
      ASSERT_NE(tv, nullptr) << "missing key " << q;
      EXPECT_EQ(*tv, oit->second) << "value mismatch for " << q;
    }
  }
  EXPECT_EQ(t.size(), oracle.size());
  // full sweep: every oracle key present with the right value
  for (auto& [k, v] : oracle) {
    std::uint64_t* tv = t.find(k);
    ASSERT_NE(tv, nullptr) << "final missing " << k;
    EXPECT_EQ(*tv, v);
  }
}

TEST(SwissTable, StringKeys) {
  SwissTable<std::string_view, int> t;
  std::vector<std::string> keep;
  for (int i = 0; i < 1000; ++i) keep.push_back("key_" + std::to_string(i));
  for (int i = 0; i < 1000; ++i) t.insert(keep[i], i);
  EXPECT_EQ(t.size(), 1000u);
  for (int i = 0; i < 1000; ++i) {
    ASSERT_NE(t.find(keep[i]), nullptr);
    EXPECT_EQ(*t.find(keep[i]), i);
  }
}

TEST(SwissTable, EraseBasic) {
  SwissTable<int, int> t;
  t.insert(1, 10);
  t.insert(2, 20);
  t.insert(3, 30);
  EXPECT_TRUE(t.erase(2));
  EXPECT_EQ(t.size(), 2u);
  EXPECT_EQ(t.find(2), nullptr);
  EXPECT_EQ(*t.find(1), 10);   // others still findable past the tombstone
  EXPECT_EQ(*t.find(3), 30);
  EXPECT_FALSE(t.erase(2));    // already gone
  EXPECT_FALSE(t.erase(99));   // never present
}

TEST(SwissTable, EraseThenReinsertReusesTombstone) {
  SwissTable<int, int> t;
  for (int i = 0; i < 50; ++i) t.insert(i, i);
  for (int i = 0; i < 50; ++i) EXPECT_TRUE(t.erase(i));
  EXPECT_EQ(t.size(), 0u);
  for (int i = 0; i < 50; ++i) EXPECT_TRUE(t.insert(i, i * 100));
  for (int i = 0; i < 50; ++i) {
    ASSERT_NE(t.find(i), nullptr);
    EXPECT_EQ(*t.find(i), i * 100);
  }
}

// Churn: alternate insert/erase heavily. The grow() same-size rehash must keep
// capacity bounded (tombstones reclaimed), not grow unboundedly.
TEST(SwissTable, ChurnBoundedCapacity) {
  SwissTable<std::uint64_t, std::uint64_t> t;
  std::mt19937_64 rng(99);
  for (int i = 0; i < 200000; ++i) {
    std::uint64_t k = rng() % 256;  // tiny live set, heavy churn
    if (rng() & 1) t.insert(k, i); else t.erase(k);
  }
  EXPECT_LE(t.size(), 256u);
  EXPECT_LT(t.capacity(), 4096u) << "tombstones grew capacity unboundedly";
}

TEST(SwissTable, EraseMatchesOracleUnderMixedOps) {
  SwissTable<std::uint64_t, std::uint64_t> t;
  std::unordered_map<std::uint64_t, std::uint64_t> oracle;
  std::mt19937_64 rng(7);
  for (int op = 0; op < 50000; ++op) {
    std::uint64_t k = rng() % 3000;
    int action = rng() % 3;  // 0/1 insert, 2 erase
    if (action == 2) {
      EXPECT_EQ(t.erase(k), oracle.erase(k) == 1);
    } else {
      std::uint64_t v = rng();
      t.insert_or_assign(k, v);
      oracle[k] = v;
    }
  }
  EXPECT_EQ(t.size(), oracle.size());
  for (auto& [k, v] : oracle) {
    auto* tv = t.find(k);
    ASSERT_NE(tv, nullptr) << "missing " << k;
    EXPECT_EQ(*tv, v);
  }
}

// erase/re-insert churn on a sparse table stays correct and capacity-bounded.
// (The reclaim-to-empty path's CORRECTNESS is exercised hard by the two oracle
// tests above -- 50k/200k erases; if reclaim ever set a slot empty that a later
// key probed through, those would diverge. This test pins the bounded-capacity
// behavior: survivors intact, the churned key gone, no unbounded growth.)
TEST(SwissTable, EraseReinsertChurnStaysBounded) {
  SwissTable<int, int> t;
  t.insert(1, 100);
  t.insert(2, 200);  // a couple of live keys so the table isn't trivially empty
  for (int i = 0; i < 100000; ++i) {
    ASSERT_TRUE(t.insert(7, i));   // new key each cycle (erased below)
    ASSERT_TRUE(t.erase(7));
  }
  EXPECT_LT(t.capacity(), 64u);    // bounded (tombstones never accumulate)
  EXPECT_EQ(t.size(), 2u);
  EXPECT_EQ(*t.find(1), 100);      // survivors intact
  EXPECT_EQ(*t.find(2), 200);
  EXPECT_EQ(t.find(7), nullptr);
}

// for_each visits EXACTLY the live set, once each, with correct key->value --
// across empty / after inserts / after erases (tombstones present) / after growth.
// Guards the group-scan for_each (match_full over kGroup bytes): a wrong mask would
// miss live slots, double-visit (ctrl tail-mirror), or visit tombstones/empties.
TEST(SwissTable, ForEachVisitsExactlyLiveSet) {
  SwissTable<std::uint64_t, std::uint64_t> t;
  // Empty: for_each is a no-op.
  { std::size_t n = 0; t.for_each([&](std::uint64_t, std::uint64_t&) { ++n; }); EXPECT_EQ(n, 0u); }

  std::mt19937_64 rng(2024);
  std::unordered_map<std::uint64_t, std::uint64_t> oracle;
  constexpr int kN = 5000;  // forces several growths past kMinCap
  for (int i = 0; i < kN; ++i) {
    std::uint64_t k = rng();
    t.insert(k, k ^ 0x9e3779b97f4a7c15ull);
    oracle[k] = k ^ 0x9e3779b97f4a7c15ull;
  }
  // Erase ~1/3 to leave tombstones interleaved with live slots.
  {
    std::vector<std::uint64_t> ks;
    for (auto& kv : oracle) ks.push_back(kv.first);
    for (std::size_t i = 0; i < ks.size(); i += 3) { t.erase(ks[i]); oracle.erase(ks[i]); }
  }
  // for_each must reproduce the oracle EXACTLY: same keys, same values, no dup.
  std::unordered_map<std::uint64_t, std::uint64_t> seen;
  t.for_each([&](std::uint64_t k, std::uint64_t& v) {
    EXPECT_TRUE(seen.emplace(k, v).second) << "duplicate visit of key " << k;
  });
  EXPECT_EQ(seen.size(), oracle.size());
  EXPECT_EQ(t.size(), oracle.size());
  for (auto& kv : oracle) {
    auto it = seen.find(kv.first);
    ASSERT_NE(it, seen.end()) << "for_each missed live key " << kv.first;
    EXPECT_EQ(it->second, kv.second) << "wrong value for key " << kv.first;
  }
}

TEST(SwissTable, InsertValueMayAliasSlotAcrossGrowth) {
  SwissTable<std::uint64_t, std::uint64_t> table;
  for (std::uint64_t i = 0; i < 14; ++i) table.insert(i, 1000 + i);
  const std::uint64_t* aliased = table.find(3);
  ASSERT_NE(aliased, nullptr);
  ASSERT_TRUE(table.insert(99, *aliased));  // triggers the 16 -> 32 grow
  ASSERT_NE(table.find(99), nullptr);
  EXPECT_EQ(*table.find(99), 1003u);
}

TEST(SwissTable, TryEmplaceArgumentsMayAliasSlotAcrossGrowth) {
  SwissTable<std::uint64_t, std::uint64_t> table;
  for (std::uint64_t i = 0; i < 14; ++i) table.insert(i, 1000 + i);
  const std::uint64_t* aliased = table.find(3);
  ASSERT_NE(aliased, nullptr);
  auto [value, inserted] = table.try_emplace(99, *aliased);
  ASSERT_TRUE(inserted);
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 1003u);
  EXPECT_EQ(*table.find(99), 1003u);
}

TEST(SwissTable, MoveConstructionPreservesProbeMask) {
  SwissTable<std::uint64_t, std::uint64_t> source;
  for (std::uint64_t i = 0; i < 1000; ++i) source.insert(i, i * 3);
  const std::size_t capacity = source.capacity();
  SwissTable<std::uint64_t, std::uint64_t> moved(std::move(source));
  EXPECT_EQ(moved.capacity(), capacity);
  EXPECT_EQ(source.size(), 0u);
  for (std::uint64_t i = 0; i < 1000; ++i) {
    ASSERT_NE(moved.find(i), nullptr);
    EXPECT_EQ(*moved.find(i), i * 3);
  }
}

TEST(SwissTable, ReserveClearMoveAssignAndConstTraversal) {
  SwissTable<int, int> source;
  ASSERT_TRUE(source.reserve(1000));
  EXPECT_GE(source.capacity() - source.capacity() / 8, 1000u);
  for (int i = 0; i < 100; ++i) source.insert(i, i + 1);

  SwissTable<int, int> target;
  target.insert(-1, -1);
  target = std::move(source);
  const SwissTable<int, int>& view = target;
  std::size_t visits = 0;
  view.for_each([&](const int& key, const int& value) {
    EXPECT_EQ(value, key + 1);
    ++visits;
  });
  EXPECT_EQ(visits, 100u);
  EXPECT_TRUE(view.contains(42));

  target.clear();
  EXPECT_TRUE(target.empty());
  EXPECT_EQ(target.capacity(), view.capacity());
  EXPECT_EQ(target.find(42), nullptr);
  EXPECT_TRUE(target.insert(7, 9));
  EXPECT_EQ(*target.find(7), 9);
}

TEST(SwissTable, FallibleReserveRejectsOverflowWithoutChangingState) {
  SwissTable<int, int> table;
  ASSERT_TRUE(table.insert(7, 9));
  const std::size_t old_capacity = table.capacity();
  EXPECT_FALSE(table.reserve(static_cast<std::size_t>(-1)));
  EXPECT_EQ(table.capacity(), old_capacity);
  ASSERT_NE(table.find(7), nullptr);
  EXPECT_EQ(*table.find(7), 9);
}

TEST(SwissTable, PrepareInsertIsFallibleAndMakesInsertAllocationFree) {
  hyle::mem::Allocator failing{fail_alloc, fail_free, fail_realloc, nullptr};
  SwissTable<int, int, hyle::container::Hash<int>, std::equal_to<>,
             hyle::mem::Allocator>
      table(failing);
  EXPECT_FALSE(table.try_prepare_insert());
  EXPECT_TRUE(table.empty());

  SwissTable<int, int> ready;
  ASSERT_TRUE(ready.try_prepare_insert());
  const std::size_t capacity = ready.capacity();
  EXPECT_TRUE(ready.insert(7, 9));
  EXPECT_EQ(ready.capacity(), capacity);
  ASSERT_NE(ready.find(7), nullptr);
  EXPECT_EQ(*ready.find(7), 9);
}

TEST(SwissTable, OwnsMoveOnlyNonTrivialValuesAcrossRehashEraseAndClear) {
  SwissTable<int, std::unique_ptr<int>> table;
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(table.insert(i, std::make_unique<int>(i * 3)));
  }
  ASSERT_EQ(table.size(), 1000u);
  for (int i = 0; i < 1000; ++i) {
    auto* value = table.find(i);
    ASSERT_NE(value, nullptr);
    ASSERT_NE(*value, nullptr);
    EXPECT_EQ(**value, i * 3);
  }
  EXPECT_FALSE(table.insert(17, std::make_unique<int>(99)));
  EXPECT_EQ(**table.find(17), 51);
  EXPECT_FALSE(table.insert_or_assign(17, std::make_unique<int>(99)));
  EXPECT_EQ(**table.find(17), 99);
  EXPECT_TRUE(table.erase(17));
  EXPECT_EQ(table.find(17), nullptr);
  table.clear();
  EXPECT_TRUE(table.empty());
}

TEST(FlatHashMap, TryEmplaceConstructsOnlyForAbsentKeysAndSubscriptIsFamiliar) {
  CountedMapValue::constructions = 0;
  FlatHashMap<int, CountedMapValue> table;
  auto [first, inserted] = table.try_emplace(7, 11);
  ASSERT_TRUE(inserted);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->number, 11);
  const int after_insert = CountedMapValue::constructions;

  auto [same, inserted_again] = table.try_emplace(7, 99);
  EXPECT_FALSE(inserted_again);
  EXPECT_EQ(same, first);
  EXPECT_EQ(same->number, 11);
  EXPECT_EQ(CountedMapValue::constructions, after_insert);

  table[9].number = 42;
  ASSERT_NE(table.find(9), nullptr);
  EXPECT_EQ(table.find(9)->number, 42);
}

TEST(SwissTable, OwnsNonTrivialStringKeys) {
  SwissTable<std::string, int, TransparentStringHash, TransparentStringEq>
      table;
  for (int i = 0; i < 500; ++i) {
    table.insert("key-" + std::to_string(i), i);
  }
  ASSERT_NE(table.find(std::string("key-301")), nullptr);
  EXPECT_EQ(*table.find(std::string("key-301")), 301);
}

TEST(SwissTable, HeterogeneousLookupAndEraseAvoidKeyConstruction) {
  SwissTable<std::string, int, TransparentStringHash, TransparentStringEq>
      table;
  table.insert("alpha", 7);
  const std::string_view query = "alpha";
  ASSERT_NE(table.find(query), nullptr);
  EXPECT_EQ(*table.find(query), 7);
  EXPECT_TRUE(table.contains(query));
  EXPECT_TRUE(table.erase(query));
  EXPECT_FALSE(table.contains(query));
}

TEST(SwissTable, FallibleInsertReportsAllocationFailure) {
  hyle::mem::Allocator failing{fail_alloc, fail_free, fail_realloc, nullptr};
  using FallibleTable =
      SwissTable<int, int, hyle::container::Hash<int>, std::equal_to<>,
                 hyle::mem::Allocator>;
  FallibleTable table(failing);
  EXPECT_EQ(table.try_insert_or_assign(1, 2),
            FallibleTable::InsertResult::kOutOfMemory);
  EXPECT_TRUE(table.empty());
}

TEST(FlatHashMap, IterationKeepsKeysConstAndValuesMutable) {
  FlatHashMap<int, int> table;
  for (int i = 0; i < 1000; ++i) ASSERT_TRUE(table.insert(i, i * 2));

  std::vector<bool> seen(1000, false);
  for (auto& entry : table) {
    const int key = entry.key();
    ASSERT_GE(key, 0);
    ASSERT_LT(key, 1000);
    EXPECT_FALSE(seen[static_cast<std::size_t>(key)]);
    seen[static_cast<std::size_t>(key)] = true;
    entry.value() += 1;
  }
  for (bool value : seen) EXPECT_TRUE(value);

  const FlatHashMap<int, int>& view = table;
  std::size_t count = 0;
  for (const auto& entry : view) {
    EXPECT_EQ(entry.value(), entry.key() * 2 + 1);
    ++count;
  }
  EXPECT_EQ(count, table.size());
  ASSERT_NE(table.find_key(87), nullptr);
  EXPECT_EQ(*table.find_key(87), 87);
}

TEST(FlatHashSet, InsertFindEraseAndIterate) {
  FlatHashSet<int> set;
  EXPECT_EQ(set.try_insert(7), FlatHashSet<int>::InsertResult::kInserted);
  EXPECT_EQ(set.try_insert(7),
            FlatHashSet<int>::InsertResult::kAlreadyPresent);
  for (int i = 0; i < 1000; ++i) set.insert(i);
  EXPECT_EQ(set.size(), 1000u);
  EXPECT_TRUE(set.contains(511));
  ASSERT_NE(set.find(511), nullptr);
  EXPECT_EQ(*set.find(511), 511);

  std::vector<bool> seen(1000, false);
  for (const int key : set) {
    ASSERT_GE(key, 0);
    ASSERT_LT(key, 1000);
    seen[static_cast<std::size_t>(key)] = true;
  }
  for (bool value : seen) EXPECT_TRUE(value);
  EXPECT_TRUE(set.erase(511));
  EXPECT_FALSE(set.contains(511));
}

TEST(FlatHashSet, HeterogeneousLookupReturnsCanonicalStoredKey) {
  FlatHashSet<std::string, TransparentStringHash, TransparentStringEq> set;
  ASSERT_TRUE(set.insert(std::string("alpha")));
  const std::string_view query = "alpha";
  const std::string* stored = set.find(query);
  ASSERT_NE(stored, nullptr);
  EXPECT_EQ(*stored, query);
  EXPECT_TRUE(set.erase(query));
  EXPECT_TRUE(set.empty());
}

TEST(FlatHashSet, FallibleInsertDoesNotChangeStateOnAllocationFailure) {
  hyle::mem::Allocator failing{fail_alloc, fail_free, fail_realloc, nullptr};
  using FallibleSet =
      FlatHashSet<int, hyle::container::Hash<int>, std::equal_to<>,
                  hyle::mem::Allocator>;
  FallibleSet set(failing);
  EXPECT_EQ(set.try_insert(1), FallibleSet::InsertResult::kOutOfMemory);
  EXPECT_TRUE(set.empty());
}

TEST(FlatHashMap, CopyRangeSwapAndEqualityMatchOwningContainerShape) {
  std::vector<std::pair<int, std::string>> input = {
      {1, "one"}, {2, "two"}, {1, "ignored"}};
  FlatHashMap<int, std::string> values(input.begin(), input.end());
  values.insert({{3, "three"}, {4, "four"}});
  ASSERT_EQ(values.size(), 4u);
  EXPECT_EQ(*values.find(1), "one");

  FlatHashMap<int, std::string> copy(values);
  EXPECT_EQ(copy, values);
  *copy.find(1) = "changed";
  EXPECT_NE(copy, values);
  EXPECT_EQ(*values.find(1), "one");

  FlatHashMap<int, std::string> assigned;
  assigned = values;
  EXPECT_EQ(assigned, values);

  FlatHashMap<int, std::string> other({{9, "nine"}});
  swap(assigned, other);
  EXPECT_EQ(assigned.size(), 1u);
  EXPECT_EQ(*assigned.find(9), "nine");
  EXPECT_EQ(other, values);
}

TEST(FlatHashMap, ThrowingCopyDestroysCompletedEntries) {
  ThrowingMapValue::alive = 0;
  {
    FlatHashMap<int, ThrowingMapValue> source;
    source.try_emplace(1, 10);
    source.try_emplace(2, 20);
    source.try_emplace(3, 30);
    ASSERT_EQ(ThrowingMapValue::alive, 3);

    ThrowingMapValue::copies_before_throw = 1;
    using ThrowingMap = FlatHashMap<int, ThrowingMapValue>;
    EXPECT_THROW(
        {
          ThrowingMap copy(source);
          (void)copy;
        },
        std::runtime_error);
    EXPECT_EQ(ThrowingMapValue::alive, 3);
  }
  EXPECT_EQ(ThrowingMapValue::alive, 0);
}

TEST(FlatHashSet, CopyRangeSwapAndEqualityMatchOwningContainerShape) {
  const std::vector<int> input = {1, 2, 3, 2};
  FlatHashSet<int> values(input.begin(), input.end());
  values.insert({3, 4, 5});
  ASSERT_EQ(values.size(), 5u);

  FlatHashSet<int> copy(values);
  EXPECT_EQ(copy, values);
  copy.erase(3);
  EXPECT_NE(copy, values);
  EXPECT_TRUE(values.contains(3));

  FlatHashSet<int> assigned;
  assigned = values;
  FlatHashSet<int> other({9});
  swap(assigned, other);
  EXPECT_EQ(assigned, (FlatHashSet<int>{9}));
  EXPECT_EQ(other, values);
}
