#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hyle/container/swiss_table.h"
#include "hyle/mem/allocator.h"

using hyle::container::SwissTable;
using hyle::mem::system_allocator;

TEST(SwissTable, EmptyFindMisses) {
  SwissTable<int, int> t(system_allocator());
  EXPECT_EQ(t.size(), 0u);
  EXPECT_EQ(t.find(42), nullptr);
  EXPECT_FALSE(t.contains(7));
}

TEST(SwissTable, InsertFindBasic) {
  SwissTable<int, int> t(system_allocator());
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

TEST(SwissTable, InsertExistingUpdates) {
  SwissTable<int, int> t(system_allocator());
  EXPECT_TRUE(t.insert(5, 50));
  EXPECT_FALSE(t.insert(5, 51));  // update, not insert
  EXPECT_EQ(t.size(), 1u);
  EXPECT_EQ(*t.find(5), 51);
}

TEST(SwissTable, GrowsAcrossManyInserts) {
  SwissTable<int, int> t(system_allocator());
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
  SwissTable<std::uint64_t, std::uint64_t> t(system_allocator());
  std::unordered_map<std::uint64_t, std::uint64_t> oracle;
  std::mt19937_64 rng(12345);
  for (int op = 0; op < 50000; ++op) {
    std::uint64_t k = rng() % 5000;  // dense space -> updates + collisions
    std::uint64_t v = rng();
    bool ins = t.insert(k, v);
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
  SwissTable<std::string_view, int> t(system_allocator());
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
  SwissTable<int, int> t(system_allocator());
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
  SwissTable<int, int> t(system_allocator());
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
  SwissTable<std::uint64_t, std::uint64_t> t(system_allocator());
  std::mt19937_64 rng(99);
  for (int i = 0; i < 200000; ++i) {
    std::uint64_t k = rng() % 256;  // tiny live set, heavy churn
    if (rng() & 1) t.insert(k, i); else t.erase(k);
  }
  EXPECT_LE(t.size(), 256u);
  EXPECT_LT(t.capacity(), 4096u) << "tombstones grew capacity unboundedly";
}

TEST(SwissTable, EraseMatchesOracleUnderMixedOps) {
  SwissTable<std::uint64_t, std::uint64_t> t(system_allocator());
  std::unordered_map<std::uint64_t, std::uint64_t> oracle;
  std::mt19937_64 rng(7);
  for (int op = 0; op < 50000; ++op) {
    std::uint64_t k = rng() % 3000;
    int action = rng() % 3;  // 0/1 insert, 2 erase
    if (action == 2) {
      EXPECT_EQ(t.erase(k), oracle.erase(k) == 1);
    } else {
      std::uint64_t v = rng();
      t.insert(k, v);
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
  SwissTable<int, int> t(system_allocator());
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
  SwissTable<std::uint64_t, std::uint64_t> t(system_allocator());
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
  SwissTable<std::uint64_t, std::uint64_t> table(system_allocator());
  for (std::uint64_t i = 0; i < 14; ++i) table.insert(i, 1000 + i);
  const std::uint64_t* aliased = table.find(3);
  ASSERT_NE(aliased, nullptr);
  ASSERT_TRUE(table.insert(99, *aliased));  // triggers the 16 -> 32 grow
  ASSERT_NE(table.find(99), nullptr);
  EXPECT_EQ(*table.find(99), 1003u);
}

TEST(SwissTable, MoveConstructionPreservesProbeMask) {
  SwissTable<std::uint64_t, std::uint64_t> source(system_allocator());
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
  SwissTable<int, int> source(system_allocator());
  ASSERT_TRUE(source.reserve(1000));
  EXPECT_GE(source.capacity() - source.capacity() / 8, 1000u);
  for (int i = 0; i < 100; ++i) source.insert(i, i + 1);

  SwissTable<int, int> target(system_allocator());
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
  SwissTable<int, int> table(system_allocator());
  ASSERT_TRUE(table.insert(7, 9));
  const std::size_t old_capacity = table.capacity();
  EXPECT_FALSE(table.reserve(static_cast<std::size_t>(-1)));
  EXPECT_EQ(table.capacity(), old_capacity);
  ASSERT_NE(table.find(7), nullptr);
  EXPECT_EQ(*table.find(7), 9);
}
