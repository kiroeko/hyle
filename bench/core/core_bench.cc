#include <nanobench.h>

#include <cstddef>
#include <list>
#include <memory>
#include <vector>

#include "hyle/core/intrusive_list.h"

#if defined(HYLE_BENCH_BOOST)
#  include <boost/intrusive/list.hpp>
#endif

namespace {
namespace nb = ankerl::nanobench;
using hyle::core::IntrusiveList;
using hyle::core::IntrusiveListHook;

constexpr int kNodes = 4096;

struct Node {
  int value = 0;
  IntrusiveListHook hook;
};
using NodeList = IntrusiveList<Node, &Node::hook>;

#if defined(HYLE_BENCH_BOOST)
namespace bi = boost::intrusive;
using BoostHook = bi::list_member_hook<bi::link_mode<bi::auto_unlink>>;
struct BoostNode {
  int value = 0;
  BoostHook hook;
};
using BoostList = bi::list<BoostNode, bi::member_hook<BoostNode, BoostHook, &BoostNode::hook>,
                           bi::constant_time_size<false>>;
#endif

}  // namespace

int main() {
  auto nodes = std::make_unique<Node[]>(kNodes);
  for (int index = 0; index < kNodes; ++index) nodes[index].value = index;

  std::vector<int> order(kNodes);
  for (int index = 0; index < kNodes; ++index) {
    order[index] = (index * 2654435761u) & (kNodes - 1);
  }

#if defined(HYLE_BENCH_BOOST)
  auto boost_nodes = std::make_unique<BoostNode[]>(kNodes);
  for (int index = 0; index < kNodes; ++index) {
    boost_nodes[index].value = index;
  }
#endif

  {
    nb::Bench bench;
    bench.title("intrusive FIFO batch: push 4096 + pop all")
        .relative(true)
        .batch(kNodes)
        .minEpochIterations(2000);
    bench.run("HYLE IntrusiveList", [&] {
      NodeList list;
      for (int index = 0; index < kNodes; ++index) {
        list.push_back(&nodes[index]);
      }
      while (list.pop_front() != nullptr) {
      }
      nb::doNotOptimizeAway(&list);
    });
#if defined(HYLE_BENCH_BOOST)
    bench.run("boost::intrusive auto_unlink", [&] {
      BoostList list;
      for (int index = 0; index < kNodes; ++index) {
        list.push_back(boost_nodes[index]);
      }
      while (!list.empty()) list.pop_front();
      nb::doNotOptimizeAway(&list);
    });
#endif
    bench.run("std::list (allocating floor)", [&] {
      std::list<int> list;
      for (int index = 0; index < kNodes; ++index) list.push_back(index);
      while (!list.empty()) list.pop_front();
      nb::doNotOptimizeAway(&list);
    });
  }

  {
    nb::Bench bench;
    bench.title("intrusive arbitrary unlink: 4096 nodes")
        .relative(true)
        .batch(kNodes)
        .minEpochIterations(2000);
    bench.run("HYLE IntrusiveList::remove", [&] {
      NodeList list;
      for (int index = 0; index < kNodes; ++index) {
        list.push_back(&nodes[index]);
      }
      for (int index : order) list.remove(&nodes[index]);
      nb::doNotOptimizeAway(&list);
    });
#if defined(HYLE_BENCH_BOOST)
    bench.run("boost::intrusive::erase", [&] {
      BoostList list;
      for (int index = 0; index < kNodes; ++index) {
        list.push_back(boost_nodes[index]);
      }
      for (int index : order) {
        list.erase(BoostList::s_iterator_to(boost_nodes[index]));
      }
      nb::doNotOptimizeAway(&list);
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("intrusive batch discard: destroy non-empty list")
        .relative(true)
        .batch(kNodes)
        .minEpochIterations(2000);
    bench.run("HYLE IntrusiveList auto detach", [&] {
      NodeList list;
      for (int index = 0; index < kNodes; ++index) {
        list.push_back(&nodes[index]);
      }
      nb::doNotOptimizeAway(&list);
    });
#if defined(HYLE_BENCH_BOOST)
    bench.run("boost::intrusive auto_unlink detach", [&] {
      BoostList list;
      for (int index = 0; index < kNodes; ++index) {
        list.push_back(boost_nodes[index]);
      }
      nb::doNotOptimizeAway(&list);
    });
#endif
  }
}
