#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#endif

#include "hyle/container/intrusive_list.h"

#if defined(HYLE_BENCH_BOOST)
#include <boost/intrusive/list.hpp>
#endif

namespace {
namespace nb = ankerl::nanobench;
using hyle::container::IntrusiveList;
using hyle::container::IntrusiveListHook;
using hyle::container::SafeIntrusiveList;
using hyle::container::SafeIntrusiveListHook;

extern "C" std::uintptr_t hyle_container_intrusive_observe_storage_phase(
    const void *storage, std::size_t byte_count) noexcept;

nb::Bench make_bench() {
  nb::Bench bench;
#if defined(__APPLE__)
  // Darwin has no supported hard P/E-core affinity. Keep an epoch long enough
  // that a scheduler burst cannot dominate these sub-nanosecond primitives.
  bench.minEpochTime(std::chrono::milliseconds(8));
#endif
  return bench;
}

struct ThreadQosState {
  const char *name;
  const char *ok;
};

// Make SSH-launched Apple runs explicit without claiming physical-core
// pinning. Interactive is the product certificate; utility is a separate
// lower-priority audit selected by HYLE_BENCH_APPLE_QOS=utility.
ThreadQosState configure_main_benchmark_thread() {
#if defined(__APPLE__)
  const char *requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  const char *name = "user-interactive";
  if (requested != nullptr && std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
    name = "utility";
  } else if (requested != nullptr &&
             std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr,
                 "hyle_container_intrusive_bench: HYLE_BENCH_APPLE_QOS must be "
                 "interactive or utility (got %s)\n",
                 requested);
    std::abort();
  }
  const int result = pthread_set_qos_class_self_np(qos, 0);
  if (result != 0)
    std::abort();
  return {name, "1"};
#else
  return {"not-applicable", "not-applicable"};
#endif
}

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

bool peer_allocation_first_requested() {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size,
                "HYLE_BENCH_PEER_ALLOCATION_FIRST") == 0 &&
      value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv("HYLE_BENCH_PEER_ALLOCATION_FIRST") != nullptr;
#endif
}

bool formal_only_requested() {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, "HYLE_CONTAINER_INTRUSIVE_BENCH_FORMAL_ONLY") ==
          0 &&
      value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv("HYLE_CONTAINER_INTRUSIVE_BENCH_FORMAL_ONLY") != nullptr;
#endif
}

struct Node {
  int value = 0;
  IntrusiveListHook hook;
};
using NodeList = IntrusiveList<Node, &Node::hook>;

struct SafeNode {
  int value = 0;
  SafeIntrusiveListHook hook;
};
using SafeNodeList = SafeIntrusiveList<SafeNode, &SafeNode::hook>;

static_assert(sizeof(Node) == sizeof(SafeNode));
static_assert(alignof(Node) == alignof(SafeNode));
static_assert(sizeof(NodeList) == sizeof(SafeNodeList));
static_assert(alignof(NodeList) == alignof(SafeNodeList));

#if defined(HYLE_BENCH_BOOST)
namespace bi = boost::intrusive;
using BoostHook = bi::list_member_hook<bi::link_mode<bi::auto_unlink>>;
struct BoostNode {
  int value = 0;
  BoostHook hook;
};
using BoostList =
    bi::list<BoostNode, bi::member_hook<BoostNode, BoostHook, &BoostNode::hook>,
             bi::constant_time_size<false>>;

using BoostSafeHook = bi::list_member_hook<bi::link_mode<bi::safe_link>>;
struct BoostSafeNode {
  int value = 0;
  BoostSafeHook hook;
};
using BoostSafeList = bi::list<
    BoostSafeNode,
    bi::member_hook<BoostSafeNode, BoostSafeHook, &BoostSafeNode::hook>,
    bi::constant_time_size<false>>;

static_assert(sizeof(BoostNode) == sizeof(BoostSafeNode));
static_assert(alignof(BoostNode) == alignof(BoostSafeNode));
static_assert(sizeof(BoostList) == sizeof(BoostSafeList));
static_assert(alignof(BoostList) == alignof(BoostSafeList));
static_assert(sizeof(Node) == sizeof(BoostNode));
static_assert(alignof(Node) == alignof(BoostNode));
static_assert(offsetof(Node, hook) == offsetof(BoostNode, hook));
static_assert(sizeof(SafeNode) == sizeof(BoostSafeNode));
static_assert(alignof(SafeNode) == alignof(BoostSafeNode));
static_assert(offsetof(SafeNode, hook) == offsetof(BoostSafeNode, hook));
static_assert(sizeof(NodeList) == sizeof(BoostList));
static_assert(alignof(NodeList) == alignof(BoostList));
static_assert(sizeof(SafeNodeList) == sizeof(BoostSafeList));
static_assert(alignof(SafeNodeList) == alignof(BoostSafeList));
#endif

template <class T> class RawNodeStorage {
public:
  explicit RawNodeStorage(std::size_t count)
      : count_(count), data_(allocator_.allocate(count)) {}

  RawNodeStorage(const RawNodeStorage &) = delete;
  RawNodeStorage &operator=(const RawNodeStorage &) = delete;

  ~RawNodeStorage() { allocator_.deallocate(data_, count_); }

  [[nodiscard]] T *data() noexcept { return data_; }
  [[nodiscard]] T &operator[](std::size_t index) noexcept {
    return data_[index];
  }

private:
  std::allocator<T> allocator_;
  std::size_t count_;
  T *data_;
};

#if defined(HYLE_BENCH_BOOST)
// Allocation order is an independent stratum from benchmark execution order.
// Reversing which implementation receives the allocator's first/second
// address stratum prevents a stable heap-placement advantage from being
// misreported as list performance.
template <class HyleNode, class PeerNode> struct PairedNodeArrays {
  PairedNodeArrays(std::size_t count, bool peer_first) {
    if (peer_first) {
      peer = std::make_unique<PeerNode[]>(count);
      hyle = std::make_unique<HyleNode[]>(count);
    } else {
      hyle = std::make_unique<HyleNode[]>(count);
      peer = std::make_unique<PeerNode[]>(count);
    }
  }

  std::unique_ptr<HyleNode[]> hyle;
  std::unique_ptr<PeerNode[]> peer;
};

template <class HyleNode, class PeerNode> struct PairedRawNodeArrays {
  PairedRawNodeArrays(std::size_t count, bool peer_first) {
    if (peer_first) {
      peer = std::make_unique<RawNodeStorage<PeerNode>>(count);
      hyle = std::make_unique<RawNodeStorage<HyleNode>>(count);
    } else {
      hyle = std::make_unique<RawNodeStorage<HyleNode>>(count);
      peer = std::make_unique<RawNodeStorage<PeerNode>>(count);
    }
  }

  std::unique_ptr<RawNodeStorage<HyleNode>> hyle;
  std::unique_ptr<RawNodeStorage<PeerNode>> peer;
};
#endif

struct AutoPolicy {
  using HyleNode = Node;
  using HyleList = NodeList;
#if defined(HYLE_BENCH_BOOST)
  using PeerNode = BoostNode;
  using PeerList = BoostList;
#endif
  static constexpr const char *label = "auto-unlink";
  static constexpr const char *fifo_hyle = "HYLE IntrusiveList";
  static constexpr const char *fifo_peer = "boost::intrusive auto_unlink";
  static constexpr const char *cancel_hyle = "HYLE IntrusiveList::remove";
  static constexpr const char *cancel_peer = "boost::intrusive::erase";
  static constexpr const char *iteration_hyle = "HYLE IntrusiveList iteration";
  static constexpr const char *iteration_peer = "boost::intrusive iteration";
  static constexpr const char *splice_hyle = "HYLE IntrusiveList::append";
  static constexpr const char *splice_peer =
      "boost::intrusive whole-list splice";
  static constexpr const char *discard_hyle = "HYLE IntrusiveList auto detach";
  static constexpr const char *discard_peer =
      "boost::intrusive auto_unlink detach";
};

struct SafePolicy {
  using HyleNode = SafeNode;
  using HyleList = SafeNodeList;
#if defined(HYLE_BENCH_BOOST)
  using PeerNode = BoostSafeNode;
  using PeerList = BoostSafeList;
#endif
  static constexpr const char *label = "safe-link";
  static constexpr const char *fifo_hyle = "HYLE SafeIntrusiveList";
  static constexpr const char *fifo_peer = "boost::intrusive safe_link";
  static constexpr const char *cancel_hyle = "HYLE SafeIntrusiveList::remove";
  static constexpr const char *cancel_peer = "boost::intrusive safe_link erase";
  static constexpr const char *iteration_hyle =
      "HYLE SafeIntrusiveList iteration";
  static constexpr const char *iteration_peer =
      "boost::intrusive safe_link iteration";
  static constexpr const char *splice_hyle = "HYLE SafeIntrusiveList::append";
  static constexpr const char *splice_peer =
      "boost::intrusive safe_link whole-list splice";
  static constexpr const char *discard_hyle =
      "HYLE SafeIntrusiveList owner detach";
  static constexpr const char *discard_peer =
      "boost::intrusive safe_link owner detach";
};

template <std::size_t Count, class Policy>
void bench_fifo([[maybe_unused]] bool rival_first,
                [[maybe_unused]] bool peer_allocation_first,
                bool include_allocating_floor = true) {
  using HyleNode = typename Policy::HyleNode;
  using HyleList = typename Policy::HyleList;
#if defined(HYLE_BENCH_BOOST)
  using PeerNode = typename Policy::PeerNode;
  using PeerList = typename Policy::PeerList;
  PairedNodeArrays<HyleNode, PeerNode> storage(Count,
                                               peer_allocation_first);
  auto &nodes = storage.hyle;
  auto &boost_nodes = storage.peer;
#else
  auto nodes = std::make_unique<HyleNode[]>(Count);
#endif
  for (std::size_t index = 0; index < Count; ++index)
    nodes[index].value = static_cast<int>(index);
#if defined(HYLE_BENCH_BOOST)
  for (std::size_t index = 0; index < Count; ++index)
    boost_nodes[index].value = static_cast<int>(index);
#endif

  nb::Bench bench = make_bench();
  bench
      .title(std::string("intrusive ") + Policy::label +
             " FIFO batch N=" + std::to_string(Count))
      .relative(true)
      .batch(Count)
      .minEpochIterations(Count < 256 ? 200000 : 2000);
  auto run_hyle = [&] {
    HyleList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(&nodes[index]);
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        nodes.get(), Count * sizeof(HyleNode)));
    std::size_t checksum = 0;
    while (HyleNode *node = list.pop_front())
      checksum += static_cast<std::size_t>(node->value);
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        nodes.get(), Count * sizeof(HyleNode)));
    nb::doNotOptimizeAway(checksum);
  };
#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&] {
    PeerList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(boost_nodes[index]);
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        boost_nodes.get(), Count * sizeof(PeerNode)));
    std::size_t checksum = 0;
    while (!list.empty()) {
      checksum += static_cast<std::size_t>(list.front().value);
      list.pop_front();
    }
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        boost_nodes.get(), Count * sizeof(PeerNode)));
    nb::doNotOptimizeAway(checksum);
  };
  if (rival_first) {
    bench.run(Policy::fifo_peer, run_boost);
    bench.run(Policy::fifo_hyle, run_hyle);
  } else {
    bench.run(Policy::fifo_hyle, run_hyle);
    bench.run(Policy::fifo_peer, run_boost);
  }
#else
  bench.run(Policy::fifo_hyle, run_hyle);
#endif
  if (include_allocating_floor) {
    bench.run("std::list allocating floor", [&] {
      std::list<int> list;
      for (std::size_t index = 0; index < Count; ++index)
        list.push_back(static_cast<int>(index));
      std::size_t checksum = 0;
      while (!list.empty()) {
        checksum += static_cast<std::size_t>(list.front());
        list.pop_front();
      }
      nb::doNotOptimizeAway(checksum);
    });
  }
}

template <std::size_t Count, class Policy>
void bench_unlink([[maybe_unused]] bool rival_first,
                  [[maybe_unused]] bool peer_allocation_first) {
  using HyleNode = typename Policy::HyleNode;
  using HyleList = typename Policy::HyleList;
  static_assert((Count & (Count - 1)) == 0,
                "multiplicative permutation requires a power of two");
#if defined(HYLE_BENCH_BOOST)
  using PeerNode = typename Policy::PeerNode;
  using PeerList = typename Policy::PeerList;
  PairedNodeArrays<HyleNode, PeerNode> storage(Count,
                                               peer_allocation_first);
  auto &nodes = storage.hyle;
  auto &boost_nodes = storage.peer;
#else
  auto nodes = std::make_unique<HyleNode[]>(Count);
#endif
  std::vector<std::size_t> order(Count);
  for (std::size_t index = 0; index < Count; ++index)
    order[index] = (index * 2654435761u) & (Count - 1);

  nb::Bench bench = make_bench();
  bench
      .title(std::string("intrusive ") + Policy::label +
             " refill + arbitrary cancel N=" + std::to_string(Count))
      .relative(true)
      .batch(Count)
      .minEpochIterations(Count < 256 ? 200000 : 2000);
  auto run_hyle = [&] {
    HyleList list;
    // Production queues receive a runtime-sized batch. Using the runtime-sized
    // order view here prevents MSVC from fully unrolling the N=32 refill and
    // turning the result into a code-placement test of two kilobyte-sized
    // lambdas instead of a list-operation comparison.
    for (std::size_t index = 0; index < order.size(); ++index)
      list.push_back(&nodes[index]);
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        nodes.get(), Count * sizeof(HyleNode)));
    for (std::size_t index : order)
      list.remove(&nodes[index]);
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        nodes.get(), Count * sizeof(HyleNode)));
  };
#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&] {
    PeerList list;
    for (std::size_t index = 0; index < order.size(); ++index)
      list.push_back(boost_nodes[index]);
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        boost_nodes.get(), Count * sizeof(PeerNode)));
    for (std::size_t index : order)
      list.erase(PeerList::s_iterator_to(boost_nodes[index]));
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        boost_nodes.get(), Count * sizeof(PeerNode)));
  };
  if (rival_first) {
    bench.run(Policy::cancel_peer, run_boost);
    bench.run(Policy::cancel_hyle, run_hyle);
  } else {
    bench.run(Policy::cancel_hyle, run_hyle);
    bench.run(Policy::cancel_peer, run_boost);
  }
#else
  bench.run(Policy::cancel_hyle, run_hyle);
#endif
}

template <std::size_t Count, class Policy>
void bench_iteration([[maybe_unused]] bool rival_first,
                     [[maybe_unused]] bool peer_allocation_first) {
  using HyleNode = typename Policy::HyleNode;
  using HyleList = typename Policy::HyleList;
#if defined(HYLE_BENCH_BOOST)
  using PeerNode = typename Policy::PeerNode;
  using PeerList = typename Policy::PeerList;
  PairedNodeArrays<HyleNode, PeerNode> storage(Count,
                                               peer_allocation_first);
  auto &nodes = storage.hyle;
  auto &boost_nodes = storage.peer;
#else
  auto nodes = std::make_unique<HyleNode[]>(Count);
#endif
  HyleList list;
  for (std::size_t index = 0; index < Count; ++index) {
    nodes[index].value = static_cast<int>(index);
    list.push_back(&nodes[index]);
  }
#if defined(HYLE_BENCH_BOOST)
  PeerList boost_list;
  for (std::size_t index = 0; index < Count; ++index) {
    boost_nodes[index].value = static_cast<int>(index);
    boost_list.push_back(boost_nodes[index]);
  }
#endif

  nb::Bench bench = make_bench();
  bench
      .title(std::string("intrusive ") + Policy::label +
             " read-only traversal N=" + std::to_string(Count))
      .relative(true)
      .batch(Count)
      .minEpochIterations(Count < 256 ? 200000 : 2000);
  auto run_hyle = [&] {
    int sum = 0;
    for (const HyleNode &node : list)
      sum += node.value;
    nb::doNotOptimizeAway(sum);
  };
#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&] {
    int sum = 0;
    for (const PeerNode &node : boost_list)
      sum += node.value;
    nb::doNotOptimizeAway(sum);
  };
  if (rival_first) {
    bench.run(Policy::iteration_peer, run_boost);
    bench.run(Policy::iteration_hyle, run_hyle);
  } else {
    bench.run(Policy::iteration_hyle, run_hyle);
    bench.run(Policy::iteration_peer, run_boost);
  }
#else
  bench.run(Policy::iteration_hyle, run_hyle);
#endif
}

template <std::size_t Count, class Policy>
void bench_splice([[maybe_unused]] bool rival_first,
                  [[maybe_unused]] bool peer_allocation_first) {
  using HyleNode = typename Policy::HyleNode;
  using HyleList = typename Policy::HyleList;
#if defined(HYLE_BENCH_BOOST)
  using PeerNode = typename Policy::PeerNode;
  using PeerList = typename Policy::PeerList;
  PairedNodeArrays<HyleNode, PeerNode> storage(Count,
                                               peer_allocation_first);
  auto &nodes = storage.hyle;
  auto &boost_nodes = storage.peer;
#else
  auto nodes = std::make_unique<HyleNode[]>(Count);
#endif
  HyleList source;
  HyleList destination;
  for (std::size_t index = 0; index < Count; ++index)
    source.push_back(&nodes[index]);
#if defined(HYLE_BENCH_BOOST)
  PeerList boost_source;
  PeerList boost_destination;
  for (std::size_t index = 0; index < Count; ++index)
    boost_source.push_back(boost_nodes[index]);
#endif

  nb::Bench bench = make_bench();
  bench
      .title(std::string("intrusive ") + Policy::label +
             " whole-batch handoff N=" + std::to_string(Count))
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
    bench.run(Policy::splice_peer, run_boost);
    bench.run(Policy::splice_hyle, run_hyle);
  } else {
    bench.run(Policy::splice_hyle, run_hyle);
    bench.run(Policy::splice_peer, run_boost);
  }
#else
  bench.run(Policy::splice_hyle, run_hyle);
#endif
}

template <std::size_t Count>
HYLE_NOINLINE void hyle_auto_node_retirement_workload(
    RawNodeStorage<Node> &nodes,
    const std::vector<std::size_t> &order) noexcept {
  NodeList list;
  for (std::size_t index = 0; index < Count; ++index) {
    Node *node = std::construct_at(nodes.data() + index);
    node->value = static_cast<int>(index);
    list.push_back(node);
  }
  nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
      nodes.data(), Count * sizeof(Node)));
  std::size_t checksum = 0;
  for (std::size_t index : order) {
    checksum += static_cast<std::size_t>(nodes[index].value);
    std::destroy_at(&nodes[index]);
  }
  nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
      nodes.data(), Count * sizeof(Node)));
  nb::doNotOptimizeAway(checksum);
  nb::doNotOptimizeAway(list.empty());
}

#if defined(HYLE_BENCH_BOOST)
template <std::size_t Count>
HYLE_NOINLINE void boost_auto_node_retirement_workload(
    RawNodeStorage<BoostNode> &nodes,
    const std::vector<std::size_t> &order) noexcept {
  BoostList list;
  for (std::size_t index = 0; index < Count; ++index) {
    BoostNode *node = std::construct_at(nodes.data() + index);
    node->value = static_cast<int>(index);
    list.push_back(*node);
  }
  nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
      nodes.data(), Count * sizeof(BoostNode)));
  std::size_t checksum = 0;
  for (std::size_t index : order) {
    checksum += static_cast<std::size_t>(nodes[index].value);
    std::destroy_at(&nodes[index]);
  }
  nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
      nodes.data(), Count * sizeof(BoostNode)));
  nb::doNotOptimizeAway(checksum);
  nb::doNotOptimizeAway(list.empty());
}
#endif

template <std::size_t Count>
void bench_auto_node_retirement(
    [[maybe_unused]] bool rival_first,
    [[maybe_unused]] bool peer_allocation_first) {
  static_assert((Count & (Count - 1)) == 0,
                "multiplicative permutation requires a power of two");
  static_assert(std::is_nothrow_default_constructible_v<Node>);
#if defined(HYLE_BENCH_BOOST)
  static_assert(std::is_nothrow_default_constructible_v<BoostNode>);
  PairedRawNodeArrays<Node, BoostNode> storage(Count,
                                               peer_allocation_first);
  auto &nodes = *storage.hyle;
  auto &boost_nodes = *storage.peer;
#else
  RawNodeStorage<Node> nodes(Count);
#endif
  std::vector<std::size_t> order(Count);
  for (std::size_t index = 0; index < Count; ++index)
    order[index] = (index * 2654435761u) & (Count - 1);

  nb::Bench bench = make_bench();
  bench
      .title("intrusive auto-unlink linked-node retirement N=" +
             std::to_string(Count))
      .relative(true)
      .batch(Count)
      .minEpochIterations(2000);

  // Raw storage allocation is outside the epoch, and there are no live objects
  // at any callback boundary. The noexcept callback therefore preserves the
  // owner's all-dead invariant even if nanobench fails between epochs.
  auto run_hyle = [&]() noexcept {
    hyle_auto_node_retirement_workload<Count>(nodes, order);
  };

#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&]() noexcept {
    boost_auto_node_retirement_workload<Count>(boost_nodes, order);
  };
  if (rival_first) {
    bench.run("boost::intrusive auto_unlink node retirement", run_boost);
    bench.run("HYLE IntrusiveList linked node retirement", run_hyle);
  } else {
    bench.run("HYLE IntrusiveList linked node retirement", run_hyle);
    bench.run("boost::intrusive auto_unlink node retirement", run_boost);
  }
#else
  bench.run("HYLE IntrusiveList linked node retirement", run_hyle);
#endif
}

template <std::size_t Count, class HyleNode, class HyleList>
HYLE_NOINLINE void hyle_owner_teardown_workload(HyleNode *nodes) noexcept {
  {
    HyleList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(&nodes[index]);
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        nodes, Count * sizeof(HyleNode)));
  }
  nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
      nodes, Count * sizeof(HyleNode)));
}

#if defined(HYLE_BENCH_BOOST)
template <std::size_t Count, class PeerNode, class PeerList>
HYLE_NOINLINE void boost_owner_teardown_workload(PeerNode *nodes) noexcept {
  {
    PeerList list;
    for (std::size_t index = 0; index < Count; ++index)
      list.push_back(nodes[index]);
    nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
        nodes, Count * sizeof(PeerNode)));
  }
  nb::doNotOptimizeAway(hyle_container_intrusive_observe_storage_phase(
      nodes, Count * sizeof(PeerNode)));
}
#endif

template <std::size_t Count, class Policy>
void bench_discard([[maybe_unused]] bool rival_first,
                   [[maybe_unused]] bool peer_allocation_first) {
  using HyleNode = typename Policy::HyleNode;
  using HyleList = typename Policy::HyleList;
#if defined(HYLE_BENCH_BOOST)
  using PeerNode = typename Policy::PeerNode;
  using PeerList = typename Policy::PeerList;
  PairedNodeArrays<HyleNode, PeerNode> storage(Count,
                                               peer_allocation_first);
  auto &nodes = storage.hyle;
  auto &boost_nodes = storage.peer;
#else
  auto nodes = std::make_unique<HyleNode[]>(Count);
#endif
  nb::Bench bench = make_bench();
  bench
      .title(std::string("intrusive ") + Policy::label +
             " owner teardown N=" + std::to_string(Count))
      .relative(true)
      .batch(Count)
      .minEpochIterations(2000);
  auto run_hyle = [&] {
    hyle_owner_teardown_workload<Count, HyleNode, HyleList>(nodes.get());
  };
#if defined(HYLE_BENCH_BOOST)
  auto run_boost = [&] {
    boost_owner_teardown_workload<Count, PeerNode, PeerList>(boost_nodes.get());
  };
  if (rival_first) {
    bench.run(Policy::discard_peer, run_boost);
    bench.run(Policy::discard_hyle, run_hyle);
  } else {
    bench.run(Policy::discard_hyle, run_hyle);
    bench.run(Policy::discard_peer, run_boost);
  }
#else
  bench.run(Policy::discard_hyle, run_hyle);
#endif
}

template <class Policy>
void run_policy_benches(bool rival_first, bool formal_only,
                        bool peer_allocation_first,
                        bool include_allocating_floor) {
  if (!formal_only)
    bench_fifo<32, Policy>(rival_first, peer_allocation_first,
                           include_allocating_floor);
  bench_fifo<4096, Policy>(rival_first, peer_allocation_first,
                           include_allocating_floor && !formal_only);
  if (!formal_only)
    bench_unlink<32, Policy>(rival_first, peer_allocation_first);
  bench_unlink<4096, Policy>(rival_first, peer_allocation_first);
  if (!formal_only)
    bench_iteration<32, Policy>(rival_first, peer_allocation_first);
  bench_iteration<4096, Policy>(rival_first, peer_allocation_first);
  if (!formal_only) {
    // Whole-list splice is an O(1) mechanism diagnostic rather than a
    // scalable product comparison, so the formal certificate excludes it.
    bench_splice<32, Policy>(rival_first, peer_allocation_first);
    bench_splice<4096, Policy>(rival_first, peer_allocation_first);
  }
  bench_discard<4096, Policy>(rival_first, peer_allocation_first);
}

} // namespace

int main() {
  const ThreadQosState qos = configure_main_benchmark_thread();
  const bool rival_first = rival_first_requested();
  const bool peer_allocation_first = peer_allocation_first_requested();
  const bool formal_only = formal_only_requested();
  std::fprintf(
      stderr,
      "# HYLE_CONTAINER_INTRUSIVE_RUN formal_only=%d execution_order=%s "
      "allocation_order=%s apple_qos=%s qos_ok=%s affinity=external\n",
      formal_only ? 1 : 0, rival_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-allocation-first"
                            : "candidate-allocation-first",
      qos.name, qos.ok);
  run_policy_benches<AutoPolicy>(rival_first, formal_only,
                                 peer_allocation_first, true);
  run_policy_benches<SafePolicy>(rival_first, formal_only,
                                 peer_allocation_first, false);
  bench_auto_node_retirement<4096>(rival_first, peer_allocation_first);
}
