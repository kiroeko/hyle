// container misc bench (task #19 breadth + gap-fill #34/#35): the claim-bearing containers.
// performance.md 桶A: SmallVector "小 N 省 malloc（大赢）"; intrusive list "O(1) 摘除、
// 无分配"; CArray = the dynamic-array (std::vector equivalent, C-ABI header-before-data).
//
// SOTA rivals (not just std -- "只赢 std 不算价值"):
//   * SmallVector  -> absl::InlinedVector (THE small-vector SOTA), gated -DHYLE_BENCH_ABSL
//   * IntrusiveList -> boost::intrusive::list (canonical intrusive SOTA), gated -DHYLE_BENCH_BOOST
//   * CArray       -> std::vector (the drop-in equivalent, always on)
// std equivalents always run as the floor. Win or lose, printed.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <vector>

#include "hyle/container/carray.h"
#include "hyle/container/intrusive_list.h"
#include "hyle/container/small_vector.h"
#include "hyle/mem/allocator.h"

#if defined(HYLE_BENCH_ABSL)
#  include "absl/container/inlined_vector.h"
#endif
#if defined(HYLE_BENCH_BOOST)
#  include <boost/intrusive/list.hpp>
#endif

using hyle::container::CArray;
using hyle::container::IntrusiveList;
using hyle::container::IntrusiveListHook;
using hyle::container::SmallVector;
using hyle::mem::system_allocator;

namespace {
namespace nb = ankerl::nanobench;
constexpr int kSmall = 8;      // fits inline (no spill) -- the SmallVector design point
constexpr int kRounds = 1000;  // construct+fill+destroy churn per epoch
constexpr int kNodes = 4096;   // intrusive vs std::list node count
constexpr int kBig = 4096;     // CArray/vector growth element count
}  // namespace

int main() {
  // ---- SmallVector vs absl::InlinedVector vs std::vector: small-N churn -------
  // Each round: build fresh, push kSmall ints, destroy. SmallVector<int,16> and
  // absl::InlinedVector<int,16> both stay inline (zero heap); std::vector mallocs.
  // absl::InlinedVector is the SOTA small-vector (same inline-storage design) ->
  // the apples-to-apples rival; std::vector is the floor.
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
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::InlinedVector<int,16> (SOTA)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        absl::InlinedVector<int, 16> v;
        for (int i = 0; i < kSmall; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
#endif
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

  // ---- SmallVector SPILL path: push kSpill > inline cap (heap grow) ----------
  // The small-N block above stays inline (the no-malloc design point). This block
  // exercises the OTHER half: push 64 ints into SmallVector<int,16> so it spills
  // inline->heap and then grows. absl::InlinedVector<int,16> spills the same way
  // (the SOTA rival); std::vector mallocs from the first push (the floor). Tests
  // whether HYLE's spill/grow path is competitive once inline storage is exceeded.
  {
    constexpr int kSpill = 64;  // > 16 inline cap -> forces spill + ~2 grows
    nb::Bench b;
    b.title("spill churn: build+fill 64 ints+destroy (>inline cap)")
        .relative(true)
        .batch(kRounds * kSpill)
        .minEpochIterations(100);
    b.run("HYLE SmallVector<int,16> (spill->heap)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        SmallVector<int, 16> v;
        for (int i = 0; i < kSpill; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::InlinedVector<int,16> (SOTA, spill->heap)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        absl::InlinedVector<int, 16> v;
        for (int i = 0; i < kSpill; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
#endif
    b.run("std::vector<int> (heap from push 1)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        std::vector<int> v;
        for (int i = 0; i < kSpill; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
  }

  // ---- IntrusiveList vs boost::intrusive::list vs std::list ------------------
  // push-all + pop-all (the "O(1) 无分配" claim). Intrusive: links live in the
  // node -> push/pop are pure pointer ops, ZERO allocation. boost::intrusive is
  // the canonical SOTA intrusive container (same hook-in-node model) -> the real
  // rival; std::list (node alloc per push) is the floor.
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
#if defined(HYLE_BENCH_BOOST)
    {
      namespace bi = boost::intrusive;
      struct BNode : bi::list_base_hook<> {
        int v;
      };
      using BList = bi::list<BNode>;
      std::vector<BNode> bnodes(kNodes);
      for (int i = 0; i < kNodes; ++i) bnodes[i].v = i;
      b.run("boost::intrusive::list (SOTA, zero alloc)", [&] {
        BList list;
        for (int i = 0; i < kNodes; ++i) list.push_back(bnodes[i]);
        while (!list.empty()) list.pop_front();
        nb::doNotOptimizeAway(&list);
      });
    }
#endif
    b.run("std::list<int> (node alloc per push)", [&] {
      std::list<int> list;
      for (int i = 0; i < kNodes; ++i) list.push_back(i);
      while (!list.empty()) list.pop_front();
      nb::doNotOptimizeAway(&list);
    });
  }

  // ---- IntrusiveList::remove(node) vs std::list::erase(it): O(1) ANY-position-
  // unlink. This is the intrusive headline ("O(1) 任意摘除, no traversal") that
  // push/pop alone don't exercise. Workload: build, then remove every node in a
  // SCRAMBLED order (not front/back) -- pure arbitrary unlink. Intrusive removes
  // by node pointer (links in the node); std::list needs the iterator (kept in a
  // side vector, valid across erases). boost::intrusive is the same-model SOTA.
  {
    struct Node {
      int v;
      IntrusiveListHook hook;
    };
    std::vector<Node> nodes(kNodes);
    for (int i = 0; i < kNodes; ++i) nodes[i].v = i;
    // Scrambled visit order (LCG-ish stride, coprime to kNodes=4096 via odd step).
    std::vector<int> order(kNodes);
    for (int i = 0; i < kNodes; ++i) order[i] = (i * 2654435761u) & (kNodes - 1);

    nb::Bench b;
    b.title("list arbitrary unlink: remove 4096 in scrambled order")
        .relative(true)
        .batch(kNodes)
        .minEpochIterations(2000);
    b.run("HYLE IntrusiveList::remove (O(1) by ptr)", [&] {
      IntrusiveList<Node, &Node::hook> list;
      for (int i = 0; i < kNodes; ++i) list.push_back(&nodes[i]);
      for (int i = 0; i < kNodes; ++i) list.remove(&nodes[order[i]]);
      nb::doNotOptimizeAway(&list);
    });
#if defined(HYLE_BENCH_BOOST)
    {
      namespace bi = boost::intrusive;
      struct BNode : bi::list_base_hook<> {
        int v;
      };
      using BList = bi::list<BNode>;
      std::vector<BNode> bnodes(kNodes);
      for (int i = 0; i < kNodes; ++i) bnodes[i].v = i;
      b.run("boost::intrusive::list::erase (SOTA, O(1))", [&] {
        BList list;
        for (int i = 0; i < kNodes; ++i) list.push_back(bnodes[i]);
        // erase(iterator_to(node)) -- O(1) unlink by node, same as HYLE remove.
        for (int i = 0; i < kNodes; ++i) list.erase(BList::s_iterator_to(bnodes[order[i]]));
        nb::doNotOptimizeAway(&list);
      });
    }
#endif
    b.run("std::list::erase(it) (O(1) given the iterator)", [&] {
      std::list<int> list;
      std::vector<std::list<int>::iterator> its;
      its.reserve(kNodes);
      for (int i = 0; i < kNodes; ++i) its.push_back(list.insert(list.end(), i));
      for (int i = 0; i < kNodes; ++i) list.erase(its[order[i]]);
      nb::doNotOptimizeAway(&list);
    });
  }

  // ---- CArray vs std::vector: large-N push_back growth + indexed read --------
  // CArray is the dynamic-array (std::vector equivalent) with a C-ABI header-
  // before-data layout: arr[i] is one load, the element ptr hands straight to a C
  // API. For trivially-copyable T both memcpy-grow, so this measures whether the
  // type-erased core + header layout costs anything vs std::vector's inline growth.
  // std::vector IS the drop-in SOTA here (an excellent, heavily-tuned dynamic array).
  {
    nb::Bench b;
    b.title("dynamic-array growth: push_back 4096 ints (no reserve)")
        .relative(true)
        .batch(kBig)
        .minEpochIterations(5000);
    b.run("HYLE CArray<int> (header-before-data)", [&] {
      CArray<int> a(system_allocator());
      for (int i = 0; i < kBig; ++i) a.push_back(i);
      nb::doNotOptimizeAway(a.data());
    });
    b.run("std::vector<int> (SOTA drop-in)", [&] {
      std::vector<int> v;
      for (int i = 0; i < kBig; ++i) v.push_back(i);
      nb::doNotOptimizeAway(v.data());
    });
  }
  {
    // Indexed read hot path: sequential sum over a prebuilt array (the arr[i]
    // single-load claim). Build once, sum many times.
    CArray<int> ha(system_allocator());
    for (int i = 0; i < kBig; ++i) ha.push_back(i);
    std::vector<int> hv;
    for (int i = 0; i < kBig; ++i) hv.push_back(i);
    nb::Bench b;
    b.title("dynamic-array sequential read sum (4096 ints)")
        .relative(true)
        .batch(kBig)
        .minEpochIterations(2000);
    b.run("HYLE CArray<int>", [&] {
      std::int64_t sum = 0;
      for (int i = 0; i < kBig; ++i) sum += ha[static_cast<std::uint64_t>(i)];
      nb::doNotOptimizeAway(sum);
    });
    b.run("std::vector<int>", [&] {
      std::int64_t sum = 0;
      for (int i = 0; i < kBig; ++i) sum += hv[static_cast<std::size_t>(i)];
      nb::doNotOptimizeAway(sum);
    });
  }

  return 0;
}
