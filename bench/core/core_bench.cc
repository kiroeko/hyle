#include <nanobench.h>

#include <cstddef>
#include <cstdlib>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "hyle/core/intrusive_list.h"

#if defined(HYLE_BENCH_BOOST)
#  include <boost/intrusive/list.hpp>
#endif

namespace {
namespace nb = ankerl::nanobench;
using hyle::core::IntrusiveList;
using hyle::core::IntrusiveListHook;

bool rival_first_requested() {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, "HYLE_BENCH_RIVAL_FIRST") == 0 &&
      value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
#endif
}

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
using BoostList =
    bi::list<BoostNode,
             bi::member_hook<BoostNode, BoostHook, &BoostNode::hook>,
             bi::constant_time_size<false>>;
#endif

template <std::size_t Count>
void bench_fifo(bool rival_first) {
  auto nodes = std::make_unique<Node[]>(Count);
  for (std::size_t index = 0; index < Count; ++index)
    nodes[index].value = static_cast<int>(index);
#if defined(HYLE_BENCH_BOOST)
  auto boost_nodes = std::make_unique<BoostNode[]>(Count);
  for (std::size_t index = 0; index < Count; ++index)
    boost_nodes[index].value = static_cast<int>(index);
#endif

  nb::Bench bench;
  bench.title("intrusive FIFO batch N=" + std::to_string(Count))
      .relative(true)
      .batch(Count)
      .minEpochIterations(Count < 256 ? 200000 : 2000);
  auto run_hyle = [&] {
    NodeList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(&nodes[index]);
    while (list.pop_front() != nullptr) {
    }
    nb::doNotOptimizeAway(&list);
  };
#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&] {
    BoostList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(boost_nodes[index]);
    while (!list.empty())
      list.pop_front();
    nb::doNotOptimizeAway(&list);
  };
  if (rival_first) {
    bench.run("boost::intrusive auto_unlink", run_boost);
    bench.run("HYLE IntrusiveList", run_hyle);
  } else {
    bench.run("HYLE IntrusiveList", run_hyle);
    bench.run("boost::intrusive auto_unlink", run_boost);
  }
#else
  bench.run("HYLE IntrusiveList", run_hyle);
#endif
  bench.run("std::list allocating floor", [&] {
    std::list<int> list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(static_cast<int>(index));
    while (!list.empty())
      list.pop_front();
    nb::doNotOptimizeAway(&list);
  });
}

template <std::size_t Count>
void bench_unlink(bool rival_first) {
  static_assert((Count & (Count - 1)) == 0,
                "multiplicative permutation requires a power of two");
  auto nodes = std::make_unique<Node[]>(Count);
  std::vector<std::size_t> order(Count);
  for (std::size_t index = 0; index < Count; ++index)
    order[index] = (index * 2654435761u) & (Count - 1);
#if defined(HYLE_BENCH_BOOST)
  auto boost_nodes = std::make_unique<BoostNode[]>(Count);
#endif

  nb::Bench bench;
  bench.title("intrusive arbitrary cancel N=" + std::to_string(Count))
      .relative(true)
      .batch(Count)
      .minEpochIterations(Count < 256 ? 200000 : 2000);
  auto run_hyle = [&] {
    NodeList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(&nodes[index]);
    for (std::size_t index : order)
      list.remove(&nodes[index]);
    nb::doNotOptimizeAway(&list);
  };
#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&] {
    BoostList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(boost_nodes[index]);
    for (std::size_t index : order)
      list.erase(BoostList::s_iterator_to(boost_nodes[index]));
    nb::doNotOptimizeAway(&list);
  };
  if (rival_first) {
    bench.run("boost::intrusive::erase", run_boost);
    bench.run("HYLE IntrusiveList::remove", run_hyle);
  } else {
    bench.run("HYLE IntrusiveList::remove", run_hyle);
    bench.run("boost::intrusive::erase", run_boost);
  }
#else
  bench.run("HYLE IntrusiveList::remove", run_hyle);
#endif
}

template <std::size_t Count>
void bench_iteration(bool rival_first) {
  auto nodes = std::make_unique<Node[]>(Count);
  NodeList list;
  for (std::size_t index = 0; index < Count; ++index) {
    nodes[index].value = static_cast<int>(index);
    list.push_back(&nodes[index]);
  }
#if defined(HYLE_BENCH_BOOST)
  auto boost_nodes = std::make_unique<BoostNode[]>(Count);
  BoostList boost_list;
  for (std::size_t index = 0; index < Count; ++index) {
    boost_nodes[index].value = static_cast<int>(index);
    boost_list.push_back(boost_nodes[index]);
  }
#endif

  nb::Bench bench;
  bench.title("intrusive read-only traversal N=" + std::to_string(Count))
      .relative(true)
      .batch(Count)
      .minEpochIterations(Count < 256 ? 200000 : 2000);
  auto run_hyle = [&] {
    int sum = 0;
    for (const Node &node : list)
      sum += node.value;
    nb::doNotOptimizeAway(sum);
  };
#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&] {
    int sum = 0;
    for (const BoostNode &node : boost_list)
      sum += node.value;
    nb::doNotOptimizeAway(sum);
  };
  if (rival_first) {
    bench.run("boost::intrusive iteration", run_boost);
    bench.run("HYLE IntrusiveList iteration", run_hyle);
  } else {
    bench.run("HYLE IntrusiveList iteration", run_hyle);
    bench.run("boost::intrusive iteration", run_boost);
  }
#else
  bench.run("HYLE IntrusiveList iteration", run_hyle);
#endif
}

template <std::size_t Count>
void bench_splice(bool rival_first) {
  auto nodes = std::make_unique<Node[]>(Count);
  NodeList source;
  NodeList destination;
  for (std::size_t index = 0; index < Count; ++index)
    source.push_back(&nodes[index]);
#if defined(HYLE_BENCH_BOOST)
  auto boost_nodes = std::make_unique<BoostNode[]>(Count);
  BoostList boost_source;
  BoostList boost_destination;
  for (std::size_t index = 0; index < Count; ++index)
    boost_source.push_back(boost_nodes[index]);
#endif

  nb::Bench bench;
  bench.title("intrusive whole-batch handoff N=" + std::to_string(Count))
      .relative(true)
      .batch(2)
      .minEpochIterations(200000);
  auto run_hyle = [&] {
    destination.append(source);
    source.append(destination);
    nb::doNotOptimizeAway(source.front());
  };
#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&] {
    boost_destination.splice(boost_destination.end(), boost_source);
    boost_source.splice(boost_source.end(), boost_destination);
    nb::doNotOptimizeAway(&boost_source.front());
  };
  if (rival_first) {
    bench.run("boost::intrusive whole-list splice", run_boost);
    bench.run("HYLE IntrusiveList::append", run_hyle);
  } else {
    bench.run("HYLE IntrusiveList::append", run_hyle);
    bench.run("boost::intrusive whole-list splice", run_boost);
  }
#else
  bench.run("HYLE IntrusiveList::append", run_hyle);
#endif
}

template <std::size_t Count>
void bench_discard(bool rival_first) {
  auto nodes = std::make_unique<Node[]>(Count);
#if defined(HYLE_BENCH_BOOST)
  auto boost_nodes = std::make_unique<BoostNode[]>(Count);
#endif
  nb::Bench bench;
  bench.title("intrusive owner teardown N=" + std::to_string(Count))
      .relative(true)
      .batch(Count)
      .minEpochIterations(2000);
  auto run_hyle = [&] {
    NodeList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(&nodes[index]);
    nb::doNotOptimizeAway(&list);
  };
#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&] {
    BoostList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(boost_nodes[index]);
    nb::doNotOptimizeAway(&list);
  };
  if (rival_first) {
    bench.run("boost::intrusive auto_unlink detach", run_boost);
    bench.run("HYLE IntrusiveList auto detach", run_hyle);
  } else {
    bench.run("HYLE IntrusiveList auto detach", run_hyle);
    bench.run("boost::intrusive auto_unlink detach", run_boost);
  }
#else
  bench.run("HYLE IntrusiveList auto detach", run_hyle);
#endif
}

} // namespace

int main() {
  const bool rival_first = rival_first_requested();
  bench_fifo<32>(rival_first);
  bench_fifo<4096>(rival_first);
  bench_unlink<32>(rival_first);
  bench_unlink<4096>(rival_first);
  bench_iteration<32>(rival_first);
  bench_iteration<4096>(rival_first);
  bench_splice<32>(rival_first);
  bench_splice<4096>(rival_first);
  bench_discard<4096>(rival_first);
}
