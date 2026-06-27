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
