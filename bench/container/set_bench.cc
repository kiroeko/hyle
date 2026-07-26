#include <nanobench.h>

#include <bit>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "hyle/container/dynamic_bitset.h"
#include "hyle/container/sparse_set.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::container::DynamicBitSet;
using hyle::container::SparseSet;

constexpr std::size_t kBits = 1u << 20;
constexpr std::size_t kWords = kBits / 64;
constexpr std::size_t kUniverse = 1u << 20;
constexpr std::size_t kActive = 4096;

std::uint32_t id_at(std::size_t i) {
  return static_cast<std::uint32_t>((i * 2654435761u) & (kUniverse - 1));
}
}  // namespace

int main() {
  // Graph-frontier shape: merge a 1M-bit discovered mask then count the active
  // frontier. std::bitset is the mature fixed-size peer; the uint64 block loop
  // is the portable dynamic-bitset implementation floor.
  DynamicBitSet hyle_left(kBits);
  DynamicBitSet hyle_right(kBits);
  std::bitset<kBits> std_left;
  std::bitset<kBits> std_right;
  std::vector<std::uint64_t> block_left(kWords);
  std::vector<std::uint64_t> block_right(kWords);
  for (std::size_t i = 0; i < kBits; ++i) {
    if (i % 7 == 0) {
      hyle_left.set(i);
      std_left.set(i);
      block_left[i / 64] |= std::uint64_t{1} << (i % 64);
    }
    if (i % 11 == 0) {
      hyle_right.set(i);
      std_right.set(i);
      block_right[i / 64] |= std::uint64_t{1} << (i % 64);
    }
  }
  {
    nb::Bench b;
    b.title("graph frontier: OR + count (1M bits)")
        .relative(true)
        .batch(kBits)
        .minEpochIterations(500);
    b.run("HYLE DynamicBitSet (runtime ISA)", [&] {
      hyle_left |= hyle_right;
      nb::doNotOptimizeAway(hyle_left.count());
    });
    b.run("std::bitset<1M> (fixed-size peer)", [&] {
      std_left |= std_right;
      nb::doNotOptimizeAway(std_left.count());
    });
    b.run("uint64 dynamic blocks (portable floor)", [&] {
      std::size_t count = 0;
      for (std::size_t i = 0; i < kWords; ++i) {
        block_left[i] |= block_right[i];
        count += std::popcount(block_left[i]);
      }
      nb::doNotOptimizeAway(count);
    });
  }

  // Frame-local active-ID shape: clear the previous set, insert a sparse batch,
  // then answer membership queries. A byte membership array is the strongest
  // dense-domain peer but pays O(universe) clear; unordered_set is the generic
  // unbounded-ID peer and pays hashing/allocation.
  SparseSet<std::uint32_t> sparse(kUniverse);
  std::vector<std::uint8_t> dense_membership(kUniverse);
  std::unordered_set<std::uint32_t> hashed;
  hashed.reserve(kActive * 2);
  {
    nb::Bench b;
    b.title("frame-local active IDs: clear + insert 4096 / 1M universe")
        .relative(true)
        .batch(kActive)
        .minEpochIterations(200);
    b.run("HYLE SparseSet<uint32_t>", [&] {
      sparse.clear();
      for (std::size_t i = 0; i < kActive; ++i) sparse.insert(id_at(i));
      nb::doNotOptimizeAway(sparse.data());
    });
    b.run("dense byte membership", [&] {
      std::fill(dense_membership.begin(), dense_membership.end(), 0);
      for (std::size_t i = 0; i < kActive; ++i)
        dense_membership[id_at(i)] = 1;
      nb::doNotOptimizeAway(dense_membership.data());
    });
    b.run("std::unordered_set<uint32_t>", [&] {
      hashed.clear();
      for (std::size_t i = 0; i < kActive; ++i) hashed.insert(id_at(i));
      nb::doNotOptimizeAway(hashed.size());
    });
  }
  for (std::size_t i = 0; i < kActive; ++i) {
    const std::uint32_t id = id_at(i);
    sparse.insert(id);
    dense_membership[id] = 1;
    hashed.insert(id);
  }
  {
    constexpr std::size_t kQueries = 65536;
    nb::Bench b;
    b.title("active-ID membership: 64K mixed queries")
        .relative(true)
        .batch(kQueries)
        .minEpochIterations(200);
    b.run("HYLE SparseSet<uint32_t>", [&] {
      std::size_t hits = 0;
      for (std::size_t i = 0; i < kQueries; ++i)
        hits += sparse.contains(id_at(i * 17));
      nb::doNotOptimizeAway(hits);
    });
    b.run("dense byte membership", [&] {
      std::size_t hits = 0;
      for (std::size_t i = 0; i < kQueries; ++i)
        hits += dense_membership[id_at(i * 17)];
      nb::doNotOptimizeAway(hits);
    });
    b.run("std::unordered_set<uint32_t>", [&] {
      std::size_t hits = 0;
      for (std::size_t i = 0; i < kQueries; ++i)
        hits += hashed.contains(id_at(i * 17));
      nb::doNotOptimizeAway(hits);
    });
  }
}
