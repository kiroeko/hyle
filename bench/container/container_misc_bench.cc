// container misc bench (task #19 breadth): the two claim-bearing containers that had no
// benchmark. performance.md 桶A: SmallVector "小 N 省 malloc（大赢）"; intrusive list
// "O(1) 摘除、无分配". Rivals are the std equivalents (std::vector / std::list) -- no
// external dep. Win or lose, printed.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <vector>

#include "hyle/container/intrusive_list.h"
#include "hyle/container/small_vector.h"
#include "hyle/mem/allocator.h"

using hyle::container::IntrusiveList;
using hyle::container::IntrusiveListHook;
using hyle::container::SmallVector;

namespace {
namespace nb = ankerl::nanobench;
constexpr int kSmall = 8;     // fits inline (no spill) -- the SmallVector design point
constexpr int kRounds = 1000;  // construct+fill+destroy churn per epoch
constexpr int kNodes = 4096;   // intrusive vs std::list node count
}  // namespace

int main() {
  // ---- SmallVector vs std::vector: small-N churn (the "省 malloc" claim) ----
  // Each round: build a fresh container, push kSmall ints, destroy. SmallVector<int,16>
  // (default compile-time SystemAllocator) stays inline (zero heap); std::vector mallocs.
  {
    nb::Bench b;
    b.title("small-N churn: build+fill 8 ints+destroy")
        .relative(true)
        .batch(kRounds * kSmall)
        .minEpochIterations(200);
    b.run("HYLE SmallVector<int,16> (inline)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        SmallVector<int, 16> v;
        for (int i = 0; i < kSmall; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
    b.run("std::vector<int> (heap)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        std::vector<int> v;
        for (int i = 0; i < kSmall; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
    b.run("std::vector<int> reserve(8) (heap, 1 malloc)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        std::vector<int> v;
        v.reserve(kSmall);
        for (int i = 0; i < kSmall; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
  }

  // ---- IntrusiveList vs std::list: push-all + remove-all (the "O(1) 无分配" claim) ----
  // Intrusive: links live in the node -> push_back/remove are pure pointer ops, ZERO
  // allocation. std::list allocates a node per push_back, frees per remove.
  {
    struct Node {
      int v;
      IntrusiveListHook hook;
    };
    std::vector<Node> nodes(kNodes);  // pre-allocated storage (intrusive owns no memory)
    for (int i = 0; i < kNodes; ++i) nodes[i].v = i;

    nb::Bench b;
    b.title("list churn: push 4096 + pop_front all")
        .relative(true)
        .batch(kNodes)
        .minEpochIterations(2000);
    b.run("HYLE IntrusiveList (zero alloc)", [&] {
      IntrusiveList<Node, &Node::hook> list;
      for (int i = 0; i < kNodes; ++i) list.push_back(&nodes[i]);
      while (list.pop_front() != nullptr) {
      }
      nb::doNotOptimizeAway(&list);
    });
    b.run("std::list<int> (node alloc per push)", [&] {
      std::list<int> list;
      for (int i = 0; i < kNodes; ++i) list.push_back(i);
      while (!list.empty()) list.pop_front();
      nb::doNotOptimizeAway(&list);
    });
  }

  return 0;
}
