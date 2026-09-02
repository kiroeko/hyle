#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <forward_list>
#include <random>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hyle/container/intrusive_forward_list.h"
#include "hyle/container/intrusive_hlist.h"
#include "hyle/container/intrusive_list.h"

namespace {

struct CountedNode {
  int value = 0;
  hyle::container::SafeIntrusiveListHook hook;
  CountedNode(int input = 0) noexcept : value(input) {}
  friend bool operator==(const CountedNode &left,
                         const CountedNode &right) noexcept {
    return left.value == right.value;
  }
};
using CountedList =
    hyle::container::CountedIntrusiveList<CountedNode, &CountedNode::hook>;

struct ForwardNode {
  int value = 0;
  hyle::container::IntrusiveForwardListHook hook;
  ForwardNode(int input = 0) noexcept : value(input) {}
  friend bool operator==(const ForwardNode &left,
                         const ForwardNode &right) noexcept {
    return left.value == right.value;
  }
};
using ForwardList =
    hyle::container::IntrusiveForwardList<ForwardNode, &ForwardNode::hook>;

struct HNode {
  int value = 0;
  hyle::container::IntrusiveHListHook hook;
  HNode(int input = 0) noexcept : value(input) {}
};
using HList = hyle::container::IntrusiveHList<HNode, &HNode::hook>;

struct SafeHNode {
  int value = 0;
  hyle::container::SafeIntrusiveHListHook hook;
  SafeHNode(int input = 0) noexcept : value(input) {}
};
using SafeHList =
    hyle::container::SafeIntrusiveHList<SafeHNode, &SafeHNode::hook>;

template <class List> std::vector<int> values(const List &list) {
  std::vector<int> result;
  for (const auto &node : list)
    result.push_back(node.value);
  return result;
}

static_assert(std::ranges::bidirectional_range<CountedList>);
static_assert(std::ranges::sized_range<CountedList>);
static_assert(sizeof(CountedList) == 3 * sizeof(void *));
static_assert(std::ranges::forward_range<ForwardList>);
static_assert(!std::ranges::sized_range<ForwardList>);
static_assert(sizeof(ForwardList) == 2 * sizeof(void *));
static_assert(sizeof(hyle::container::IntrusiveForwardListHook) ==
              sizeof(void *));
static_assert(std::ranges::forward_range<HList>);
static_assert(sizeof(HList) == sizeof(void *));
static_assert(sizeof(SafeHList) == sizeof(void *));

TEST(CountedIntrusiveList, CountTracksEveryMutationAndTransferShape) {
  CountedNode nodes[8]{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}};
  CountedList left;
  CountedList right;
  for (int index = 0; index != 5; ++index)
    left.push_back(&nodes[index]);
  right.push_back(&nodes[5]);
  EXPECT_EQ(left.size(), 5u);
  EXPECT_EQ(right.size(), 1u);

  right.splice(right.end(), left, left.begin());
  EXPECT_EQ(left.size(), 4u);
  EXPECT_EQ(right.size(), 2u);

  auto first = left.begin();
  ++first;
  auto last = first;
  ++last;
  ++last;
  right.splice(right.end(), left, first, last);
  EXPECT_EQ(left.size(), 2u);
  EXPECT_EQ(right.size(), 4u);

  left.push_back(&nodes[6]);
  left.push_back(&nodes[7]);
  first = left.begin();
  last = first;
  ++last;
  ++last;
  right.splice(right.end(), left, first, last, 2);
  EXPECT_EQ(left.size(), 2u);
  EXPECT_EQ(right.size(), 6u);

  auto same = right.begin();
  auto same_last = same;
  ++same_last;
  right.splice(right.end(), right, same, same_last, 99);
  EXPECT_EQ(right.size(), 6u);

  left.append(right);
  EXPECT_EQ(left.size(), 8u);
  EXPECT_TRUE(right.empty());
  EXPECT_EQ(right.size(), 0u);
  EXPECT_NE(left.pop_front(), nullptr);
  EXPECT_NE(left.pop_back(), nullptr);
  EXPECT_EQ(left.size(), 6u);

  CountedList moved(std::move(left));
  EXPECT_EQ(moved.size(), 6u);
  EXPECT_EQ(left.size(), 0u);
  EXPECT_EQ(moved.consume_all([](CountedNode *) noexcept {}), 6u);
  EXPECT_EQ(moved.size(), 0u);
}

TEST(IntrusiveSequences, AlgorithmsAreStableAllocationFreeShapes) {
  CountedNode counted_nodes[8]{{4}, {1}, {3}, {1}, {2}, {2}, {5}, {0}};
  CountedList counted;
  for (auto &node : counted_nodes)
    counted.push_back(&node);
  counted.sort([](const auto &left, const auto &right) {
    return left.value < right.value;
  });
  const CountedNode absent_counted{99};
  EXPECT_EQ(counted.remove(absent_counted), 0u);
  EXPECT_EQ(values(counted), (std::vector<int>{0, 1, 1, 2, 2, 3, 4, 5}));
  EXPECT_EQ(counted.unique([](const auto &left, const auto &right) {
    return left.value == right.value;
  }),
            2u);
  EXPECT_EQ(
      counted.remove_if([](const auto &node) { return (node.value & 1) == 0; }),
      3u);
  counted.reverse();
  EXPECT_EQ(values(counted), (std::vector<int>{5, 3, 1}));
  EXPECT_EQ(counted.size(), 3u);

  CountedNode other_nodes[3]{{0}, {2}, {6}};
  CountedList other;
  for (auto &node : other_nodes)
    other.push_back(&node);
  counted.reverse();
  counted.merge(other, [](const auto &left, const auto &right) {
    return left.value < right.value;
  });
  EXPECT_EQ(values(counted), (std::vector<int>{0, 1, 2, 3, 5, 6}));
  EXPECT_TRUE(other.empty());
  EXPECT_EQ(counted.size(), 6u);
  counted.clear();

  ForwardNode forward_nodes[8]{{4}, {1}, {3}, {1}, {2}, {2}, {5}, {0}};
  ForwardList forward;
  for (auto &node : forward_nodes)
    forward.push_back(&node);
  forward.sort([](const auto &left, const auto &right) {
    return left.value < right.value;
  });
  const ForwardNode absent_forward{99};
  EXPECT_EQ(forward.remove(absent_forward), 0u);
  EXPECT_EQ(values(forward), (std::vector<int>{0, 1, 1, 2, 2, 3, 4, 5}));
  EXPECT_EQ(forward.unique([](const auto &left, const auto &right) {
    return left.value == right.value;
  }),
            2u);
  EXPECT_EQ(
      forward.remove_if([](const auto &node) { return (node.value & 1) == 0; }),
      3u);
  forward.reverse();
  EXPECT_EQ(values(forward), (std::vector<int>{5, 3, 1}));
}

TEST(IntrusiveSequences, ThrowingSortKeepsEveryNodeInOneValidList) {
  CountedNode counted_nodes[16];
  CountedList counted;
  for (int index = 0; index != 16; ++index) {
    counted_nodes[index].value = 15 - index;
    counted.push_back(&counted_nodes[index]);
  }
  int calls = 0;
  EXPECT_THROW(counted.sort([&](const auto &left, const auto &right) {
    if (++calls == 9) throw std::runtime_error("stop");
    return left.value < right.value;
  }),
               std::runtime_error);
  EXPECT_EQ(counted.size(), 16u);
  std::vector<const CountedNode *> seen;
  for (const auto &node : counted)
    seen.push_back(&node);
  std::ranges::sort(seen);
  std::vector<const CountedNode *> expected;
  for (const auto &node : counted_nodes)
    expected.push_back(&node);
  std::ranges::sort(expected);
  EXPECT_EQ(seen, expected);
  counted.clear();

  ForwardNode forward_nodes[16];
  ForwardList forward;
  for (int index = 0; index != 16; ++index) {
    forward_nodes[index].value = 15 - index;
    forward.push_back(&forward_nodes[index]);
  }
  calls = 0;
  EXPECT_THROW(forward.sort([&](const auto &left, const auto &right) {
    if (++calls == 9) throw std::runtime_error("stop");
    return left.value < right.value;
  }),
               std::runtime_error);
  std::vector<const ForwardNode *> forward_seen;
  for (const auto &node : forward)
    forward_seen.push_back(&node);
  std::ranges::sort(forward_seen);
  std::vector<const ForwardNode *> forward_expected;
  for (const auto &node : forward_nodes)
    forward_expected.push_back(&node);
  std::ranges::sort(forward_expected);
  EXPECT_EQ(forward_seen, forward_expected);
}

TEST(IntrusiveForwardList, MutationAndSpliceMatchForwardSequenceSemantics) {
  ForwardNode nodes[8]{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}};
  ForwardList source;
  ForwardList destination;
  source.push_back(&nodes[0]);
  source.push_back(&nodes[1]);
  source.push_back(&nodes[2]);
  source.push_back(&nodes[3]);
  destination.push_back(&nodes[4]);

  auto before_one = source.before_begin();
  destination.splice_after(destination.before_begin(), source, before_one);
  EXPECT_EQ(values(source), (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(values(destination), (std::vector<int>{0, 4}));

  auto before_first = source.before_begin();
  auto last = source.begin();
  ++last;
  ++last;
  destination.splice_after(destination.begin(), source, before_first, last);
  EXPECT_EQ(values(source), (std::vector<int>{3}));
  EXPECT_EQ(values(destination), (std::vector<int>{0, 1, 2, 4}));

  source.push_back(&nodes[5]);
  destination.append(source);
  EXPECT_TRUE(source.empty());
  EXPECT_EQ(destination.back(), &nodes[5]);
  EXPECT_EQ(values(destination), (std::vector<int>{0, 1, 2, 4, 3, 5}));

  auto before_two = destination.begin();
  ++before_two;
  destination.splice_after(destination.iterator_to(&nodes[5]), destination,
                           before_two);
  EXPECT_EQ(destination.back(), &nodes[2]);
  EXPECT_EQ(values(destination), (std::vector<int>{0, 1, 4, 3, 5, 2}));
  EXPECT_EQ(destination.pop_front(), &nodes[0]);
  EXPECT_FALSE(nodes[0].hook.linked());
}

TEST(IntrusiveHList, AutoSafeReplaceSwapAndConsumePreserveTopology) {
  HNode nodes[5]{{0}, {1}, {2}, {3}, {4}};
  HList list;
  list.push_front(&nodes[2]);
  list.push_front(&nodes[1]);
  list.push_front(&nodes[0]);
  EXPECT_EQ(values(list), (std::vector<int>{0, 1, 2}));

  list.insert_after(list.iterator_to(&nodes[1]), &nodes[3]);
  list.insert_before(list.iterator_to(&nodes[2]), &nodes[4]);
  EXPECT_EQ(values(list), (std::vector<int>{0, 1, 3, 4, 2}));
  nodes[3].hook.unlink();
  EXPECT_EQ(values(list), (std::vector<int>{0, 1, 4, 2}));
  list.replace(list.iterator_to(&nodes[4]), &nodes[3]);
  EXPECT_FALSE(nodes[4].hook.linked());
  EXPECT_EQ(values(list), (std::vector<int>{0, 1, 3, 2}));

  HList other;
  other.push_front(&nodes[4]);
  list.swap(other);
  EXPECT_EQ(values(list), (std::vector<int>{4}));
  EXPECT_EQ(values(other), (std::vector<int>{0, 1, 3, 2}));
  EXPECT_EQ(other.consume_all([](HNode *node) noexcept {
    EXPECT_FALSE(node->hook.linked());
  }),
            4u);
  EXPECT_TRUE(other.empty());

  SafeHNode safe_nodes[3]{{0}, {1}, {2}};
  SafeHList safe;
  for (auto &node : safe_nodes)
    safe.push_front(&node);
  safe.remove(&safe_nodes[1]);
  EXPECT_FALSE(safe_nodes[1].hook.linked());
  EXPECT_EQ(safe.pop_front(), &safe_nodes[2]);
  safe.clear();
}

TEST(IntrusiveExtensions, CountedForwardAndHListMatchFixedSeedModels) {
  constexpr std::size_t kCount = 32;
  std::array<CountedNode, kCount> counted_nodes;
  std::array<ForwardNode, kCount> forward_nodes;
  std::array<HNode, kCount> hnodes;
  for (std::size_t index = 0; index < kCount; ++index) {
    counted_nodes[index].value = static_cast<int>(index);
    forward_nodes[index].value = static_cast<int>(index);
    hnodes[index].value = static_cast<int>(index);
  }
  CountedList counted[2];
  ForwardList forward[2];
  HList hlists[2];
  std::vector<int> counted_model[2];
  std::vector<int> forward_model[2];
  std::vector<int> hlist_model[2];
  std::mt19937 random(0x48594c45u);

  auto erase_value = [](std::vector<int> &model, int value) {
    auto position = std::ranges::find(model, value);
    if (position != model.end()) model.erase(position);
  };
  auto counted_at = [&](int list, std::size_t index) {
    if (index == counted_model[list].size()) return counted[list].end();
    return counted[list].iterator_to(
        &counted_nodes[static_cast<std::size_t>(counted_model[list][index])]);
  };
  auto forward_before = [&](int list, std::size_t index) {
    auto position = forward[list].before_begin();
    for (std::size_t step = 0; step < index; ++step)
      ++position;
    return position;
  };
  auto verify = [&] {
    for (int list = 0; list != 2; ++list) {
      EXPECT_EQ(values(counted[list]), counted_model[list]);
      EXPECT_EQ(counted[list].size(), counted_model[list].size());
      EXPECT_EQ(values(forward[list]), forward_model[list]);
      EXPECT_EQ(values(hlists[list]), hlist_model[list]);
    }
    for (std::size_t index = 0; index < kCount; ++index) {
      const int value = static_cast<int>(index);
      EXPECT_EQ(counted_nodes[index].hook.linked(),
                std::ranges::find(counted_model[0], value) !=
                        counted_model[0].end() ||
                    std::ranges::find(counted_model[1], value) !=
                        counted_model[1].end());
      EXPECT_EQ(forward_nodes[index].hook.linked(),
                std::ranges::find(forward_model[0], value) !=
                        forward_model[0].end() ||
                    std::ranges::find(forward_model[1], value) !=
                        forward_model[1].end());
      EXPECT_EQ(
          hnodes[index].hook.linked(),
          std::ranges::find(hlist_model[0], value) != hlist_model[0].end() ||
              std::ranges::find(hlist_model[1], value) != hlist_model[1].end());
    }
  };

  for (std::size_t step = 0; step < 1200; ++step) {
    const int destination = static_cast<int>(random() & 1u);
    const int source = 1 - destination;
    const int value = static_cast<int>(random() % kCount);
    switch (random() % 6) {
    case 0:
      if (!counted_nodes[static_cast<std::size_t>(value)].hook.linked()) {
        counted[destination].push_back(
            &counted_nodes[static_cast<std::size_t>(value)]);
        counted_model[destination].push_back(value);
      }
      break;
    case 1:
      if (!counted_model[source].empty()) {
        const std::size_t index = random() % counted_model[source].size();
        counted[destination].splice(counted[destination].end(), counted[source],
                                    counted_at(source, index));
        counted_model[destination].push_back(counted_model[source][index]);
        counted_model[source].erase(counted_model[source].begin() + index);
      }
      break;
    case 2:
      if (!forward_nodes[static_cast<std::size_t>(value)].hook.linked()) {
        forward[destination].push_back(
            &forward_nodes[static_cast<std::size_t>(value)]);
        forward_model[destination].push_back(value);
      }
      break;
    case 3:
      if (!forward_model[source].empty()) {
        const std::size_t index = random() % forward_model[source].size();
        forward[destination].splice_after(forward[destination].before_begin(),
                                          forward[source],
                                          forward_before(source, index));
        forward_model[destination].insert(forward_model[destination].begin(),
                                          forward_model[source][index]);
        forward_model[source].erase(forward_model[source].begin() + index);
      }
      break;
    case 4:
      if (!hnodes[static_cast<std::size_t>(value)].hook.linked()) {
        hlists[destination].push_front(
            &hnodes[static_cast<std::size_t>(value)]);
        hlist_model[destination].insert(hlist_model[destination].begin(),
                                        value);
      }
      break;
    default:
      if (hnodes[static_cast<std::size_t>(value)].hook.linked()) {
        hnodes[static_cast<std::size_t>(value)].hook.unlink();
        erase_value(hlist_model[0], value);
        erase_value(hlist_model[1], value);
      }
      break;
    }
    verify();
  }
  counted[0].clear();
  counted[1].clear();
  forward[0].clear();
  forward[1].clear();
  hlists[0].clear();
  hlists[1].clear();
}

#if !defined(NDEBUG)
TEST(IntrusiveExtensions, ExplicitLifetimeHooksFailFastWhenStillLinked) {
  EXPECT_DEATH(
      {
        ForwardList list;
        auto *node = new ForwardNode{1};
        list.push_front(node);
        delete node;
      },
      "HYLE fatal");
  EXPECT_DEATH(
      {
        SafeHList list;
        auto *node = new SafeHNode{1};
        list.push_front(node);
        delete node;
      },
      "HYLE fatal");
}
#endif

} // namespace
