#include "hyle/container/intrusive_list.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <concepts>
#include <iterator>
#include <list>
#include <memory>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

using hyle::container::IntrusiveList;
using hyle::container::IntrusiveListHook;
using hyle::container::SafeIntrusiveList;
using hyle::container::SafeIntrusiveListHook;

namespace {
struct Node {
  int value;
  IntrusiveListHook hook;
  explicit Node(int input) : value(input) {}

  void increment() noexcept { ++value; }
};
using NodeList = IntrusiveList<Node, &Node::hook>;

struct SafeNode {
  int value;
  SafeIntrusiveListHook hook;
  explicit SafeNode(int input) : value(input) {}

  void increment() noexcept { ++value; }
};
using SafeNodeList = SafeIntrusiveList<SafeNode, &SafeNode::hook>;

template <class Hook>
concept HasPublicUnlink = requires(Hook &hook) { hook.unlink(); };

static_assert(std::bidirectional_iterator<NodeList::iterator>);
static_assert(std::bidirectional_iterator<NodeList::const_iterator>);
static_assert(std::ranges::bidirectional_range<NodeList>);
static_assert(std::ranges::bidirectional_range<const NodeList>);
static_assert(std::bidirectional_iterator<SafeNodeList::iterator>);
static_assert(std::bidirectional_iterator<SafeNodeList::const_iterator>);
static_assert(std::ranges::bidirectional_range<SafeNodeList>);
static_assert(std::ranges::bidirectional_range<const SafeNodeList>);
static_assert(!std::ranges::sized_range<NodeList>);
static_assert(!std::ranges::sized_range<const NodeList>);
static_assert(!std::ranges::sized_range<SafeNodeList>);
static_assert(!std::ranges::sized_range<const SafeNodeList>);
static_assert(HasPublicUnlink<IntrusiveListHook>);
static_assert(!HasPublicUnlink<SafeIntrusiveListHook>);
static_assert(sizeof(IntrusiveListHook) == 2 * sizeof(void *));
static_assert(sizeof(SafeIntrusiveListHook) == 2 * sizeof(void *));
static_assert(sizeof(NodeList) == 2 * sizeof(void *));
static_assert(sizeof(SafeNodeList) == 2 * sizeof(void *));
static_assert(sizeof(NodeList::iterator) == sizeof(void *));
static_assert(sizeof(SafeNodeList::iterator) == sizeof(void *));
volatile bool g_throw_from_visitor = false;
} // namespace

TEST(IntrusiveList, ConventionalRangeInsertEraseAndSpliceAreUsable) {
  Node a(1), b(2), c(3), d(4), e(5);
  NodeList destination;
  NodeList source;
  destination.push_back(&a);
  destination.push_back(&d);
  source.push_back(&b);
  source.push_back(&c);

  const NodeList &destination_view = destination;
  auto before_d = destination_view.cend();
  --before_d;
  const NodeList &source_view = source;
  auto b_position = source_view.cbegin();
  destination.splice(before_d, source);
  EXPECT_TRUE(source.empty());
  EXPECT_EQ(&*b_position, &b); // node iterators survive whole-list splice

  auto inserted = destination.insert(destination_view.cend(), &e);
  EXPECT_EQ(&*inserted, &e);
  EXPECT_FALSE(destination.try_insert(destination.end(), &e));

  std::vector<int> forward;
  for (const Node &node : destination)
    forward.push_back(node.value);
  EXPECT_EQ(forward, (std::vector<int>{1, 2, 3, 4, 5}));
  const NodeList &const_destination = destination;
  EXPECT_EQ(destination.begin(), const_destination.begin());

  std::vector<int> reverse;
  for (auto it = destination.rbegin(); it != destination.rend(); ++it)
    reverse.push_back(it->value);
  EXPECT_EQ(reverse, (std::vector<int>{5, 4, 3, 2, 1}));

  auto erase_position = destination_view.cbegin();
  ++erase_position;
  auto after_erased = destination.erase(erase_position);
  EXPECT_EQ(&*after_erased, &c);
  EXPECT_FALSE(b.hook.linked());

  NodeList replacement;
  replacement.push_back(&b);
  destination.swap(replacement);
  EXPECT_EQ(destination.front(), &b);
  EXPECT_EQ(replacement.front(), &a);
  EXPECT_EQ(replacement.back(), &e);
}

TEST(IntrusiveList, SingleNodeSpliceSupportsSameListReordering) {
  Node a(1), b(2), c(3);
  NodeList list;
  list.push_back(&a);
  list.push_back(&b);
  list.push_back(&c);

  const NodeList &view = list;
  auto b_position = view.cbegin();
  ++b_position;
  auto moved = list.splice(view.cend(), list, b_position);
  EXPECT_EQ(&*moved, &b);

  std::vector<int> values;
  for (const Node &node : list)
    values.push_back(node.value);
  EXPECT_EQ(values, (std::vector<int>{1, 3, 2}));

  auto successor = view.cbegin();
  ++successor;
  auto unchanged = list.splice(successor, list, view.cbegin());
  EXPECT_EQ(&*unchanged, &a);
  EXPECT_EQ(list.front(), &a);
}

TEST(IntrusiveList, IteratorToAndRangeSplicePreserveNodeIterators) {
  Node a(1), b(2), c(3), d(4), e(5), f(6), g(7);
  NodeList source;
  NodeList destination;
  for (Node *node : {&a, &b, &c, &d, &e})
    source.push_back(node);
  destination.push_back(&f);
  destination.push_back(&g);

  auto b_position = source.iterator_to(&b);
  auto after_range = source.iterator_to(&e);
  auto before_g = destination.iterator_to(&g);
  auto moved = destination.splice(before_g, source, b_position, after_range);
  EXPECT_EQ(&*moved, &b);
  EXPECT_EQ(&*b_position, &b);
  EXPECT_EQ((std::vector<int>{a.value, e.value}),
            (std::vector<int>{source.front()->value, source.back()->value}));

  std::vector<int> values;
  for (const Node &node : destination)
    values.push_back(node.value);
  EXPECT_EQ(values, (std::vector<int>{6, 2, 3, 4, 7}));

  // Same-list transfer remains O(1); moving [b, d) before f produces b,c,f,d,g.
  auto d_position = destination.iterator_to(&d);
  destination.splice(destination.begin(), destination, b_position, d_position);
  values.clear();
  for (const Node &node : destination)
    values.push_back(node.value);
  EXPECT_EQ(values, (std::vector<int>{2, 3, 6, 4, 7}));

  // A range already immediately before its position is a documented no-op.
  auto unchanged = destination.splice(destination.iterator_to(&f), destination,
                                      destination.iterator_to(&b),
                                      destination.iterator_to(&f));
  EXPECT_EQ(&*unchanged, &f);
  values.clear();
  for (const Node &node : destination)
    values.push_back(node.value);
  EXPECT_EQ(values, (std::vector<int>{2, 3, 6, 4, 7}));

  const NodeList &view = destination;
  EXPECT_EQ(&*view.iterator_to(&g), &g);
}

TEST(IntrusiveList, SwapAndMoveKeepNodeIteratorsButNotSentinelIdentity) {
  Node a(1), b(2), c(3);
  NodeList first;
  NodeList second;
  first.push_back(&a);
  first.push_back(&b);
  second.push_back(&c);
  auto a_position = first.iterator_to(&a);
  auto old_first_end = first.end();

  using std::swap;
  swap(first, second);
  EXPECT_EQ(first.front(), &c);
  EXPECT_EQ(second.front(), &a);
  EXPECT_EQ(&*a_position, &a);
  EXPECT_EQ(old_first_end, first.end());
  ++a_position;
  EXPECT_EQ(&*a_position, &b);
  ++a_position;
  EXPECT_EQ(a_position, second.end());

  auto old_second_end = second.end();
  auto b_position = second.iterator_to(&b);
  NodeList moved(std::move(second));
  EXPECT_TRUE(second.empty());
  EXPECT_EQ(old_second_end, second.end());
  EXPECT_EQ(&*b_position, &b);
  ++b_position;
  EXPECT_EQ(b_position, moved.end());
}

TEST(IntrusiveList, SwapHandlesEmptyListsWithoutChangingMembership) {
  Node node(1);
  NodeList empty;
  NodeList populated;
  populated.push_back(&node);
  empty.swap(populated);
  EXPECT_EQ(empty.front(), &node);
  EXPECT_TRUE(populated.empty());
  EXPECT_TRUE(node.hook.linked());

  populated.swap(empty);
  EXPECT_EQ(populated.front(), &node);
  EXPECT_TRUE(empty.empty());

  empty.swap(empty);
  NodeList another_empty;
  empty.swap(another_empty);
  EXPECT_TRUE(empty.empty());
  EXPECT_TRUE(another_empty.empty());
}

TEST(IntrusiveList, PushPopAndRemovePreserveOrder) {
  NodeList list;
  Node a(1), b(2), c(3), d(4);
  list.push_front(&b);
  list.push_front(&a);
  list.push_back(&c);
  list.push_back(&d);
  EXPECT_EQ(list.front(), &a);
  EXPECT_EQ(list.back(), &d);

  list.remove(&b);
  EXPECT_FALSE(b.hook.linked());
  EXPECT_EQ(list.pop_front(), &a);
  EXPECT_EQ(list.pop_back(), &d);
  EXPECT_EQ(list.pop_front(), &c);
  EXPECT_TRUE(list.empty());
  EXPECT_EQ(list.pop_front(), nullptr);
}

TEST(IntrusiveList, SingleAndWholeListSpliceWithoutAllocation) {
  NodeList source;
  NodeList destination;
  Node a(1), b(2), c(3), d(4);
  source.push_back(&a);
  source.push_back(&b);
  destination.push_back(&c);

  auto moved = destination.splice(destination.end(), source, source.begin());
  EXPECT_EQ(&*moved, &a);
  EXPECT_EQ(source.front(), &b);
  EXPECT_EQ(destination.pop_front(), &c);
  EXPECT_EQ(destination.front(), &a);

  NodeList tail;
  tail.push_back(&d);
  destination.append(tail);
  EXPECT_TRUE(tail.empty());
  EXPECT_EQ(destination.pop_front(), &a);
  EXPECT_EQ(destination.pop_front(), &d);
  EXPECT_EQ(source.pop_front(), &b);
}

TEST(IntrusiveList, MoveConstructionAndAssignmentTransferMembership) {
  Node a(1), b(2), c(3);
  NodeList source;
  source.push_back(&a);
  source.push_back(&b);

  NodeList moved(std::move(source));
  EXPECT_TRUE(source.empty());
  EXPECT_EQ(moved.front(), &a);

  NodeList destination;
  destination.push_back(&c);
  destination = std::move(moved);
  EXPECT_EQ(moved.front(), &c);
  EXPECT_TRUE(c.hook.linked());
  EXPECT_EQ(destination.pop_front(), &a);
  EXPECT_EQ(destination.pop_front(), &b);
  moved.clear();
  EXPECT_FALSE(c.hook.linked());
}

TEST(IntrusiveList, RangeSpliceAndConsumeSupportBatchHandoff) {
  NodeList source;
  NodeList destination;
  Node a(1), b(2), c(3), d(4);
  source.push_back(&a);
  source.push_back(&b);
  source.push_back(&c);
  destination.push_back(&d);
  auto first = source.begin();
  auto after = first;
  ++after;
  ++after;
  destination.splice(destination.end(), source, first, after);
  EXPECT_EQ(source.front(), &c);

  NodeList consumed;
  int sum = 0;
  EXPECT_EQ(destination.consume_all([&](Node *node) noexcept {
    EXPECT_FALSE(node->hook.linked());
    sum += node->value;
    consumed.push_back(node);
  }),
            3u);
  EXPECT_EQ(sum, 7);
  EXPECT_TRUE(destination.empty());
  EXPECT_EQ(consumed.pop_front(), &d);
  EXPECT_EQ(consumed.pop_front(), &a);
  EXPECT_EQ(consumed.pop_front(), &b);
  EXPECT_EQ(destination.consume_all([](Node *) noexcept {}), 0u);
}

TEST(IntrusiveList, InvokeCompatibleCallbacksAndDeleteCurrentAreSupported) {
  NodeList list;
  Node a(1), b(2), c(4);
  list.push_back(&a);
  list.push_back(&b);
  list.push_back(&c);
  list.for_each(&Node::increment);
  EXPECT_EQ(a.value, 2);
  EXPECT_EQ(b.value, 3);
  EXPECT_EQ(c.value, 5);

  NodeList owned;
  owned.push_back(new Node(10));
  owned.push_back(new Node(20));
  int sum = 0;
  EXPECT_EQ(owned.consume_all([&](Node *node) noexcept {
    EXPECT_FALSE(node->hook.linked());
    sum += node->value;
    delete node;
  }),
            2u);
  EXPECT_EQ(sum, 30);
  EXPECT_TRUE(owned.empty());
}

TEST(IntrusiveList, IndependentHooksSupportSimultaneousMembership) {
  struct MultiNode {
    int value;
    IntrusiveListHook ready_hook;
    IntrusiveListHook timeout_hook;
  } node{42, {}, {}};
  using ReadyList = IntrusiveList<MultiNode, &MultiNode::ready_hook>;
  using TimeoutList = IntrusiveList<MultiNode, &MultiNode::timeout_hook>;

  ReadyList ready;
  TimeoutList timeout;
  ready.push_back(&node);
  timeout.push_back(&node);
  EXPECT_TRUE(node.ready_hook.linked());
  EXPECT_TRUE(node.timeout_hook.linked());
  EXPECT_EQ(ready.pop_front(), &node);
  EXPECT_FALSE(node.ready_hook.linked());
  EXPECT_TRUE(node.timeout_hook.linked());
  EXPECT_EQ(timeout.pop_front(), &node);
}

TEST(IntrusiveList, ForEachVisitsMutableAndConstNodes) {
  NodeList list;
  Node a(1), b(2), c(3);
  list.push_back(&a);
  list.push_back(&b);
  list.push_back(&c);

  list.for_each([](Node *node) { node->value *= 2; });
  const NodeList &view = list;
  int sum = 0;
  view.for_each([&sum](const Node *node) { sum += node->value; });
  EXPECT_EQ(sum, 12);
}

TEST(IntrusiveList, DestroyingNodeAutomaticallyUnlinksIt) {
  NodeList list;
  Node first(1);
  list.push_back(&first);
  {
    Node transient(2);
    list.push_back(&transient);
    EXPECT_EQ(list.back(), &transient);
  }
  EXPECT_EQ(list.front(), &first);
  EXPECT_EQ(list.back(), &first);
}

TEST(IntrusiveList, DestroyingOrClearingListResetsEveryHook) {
  Node a(1), b(2);
  {
    NodeList list;
    list.push_back(&a);
    list.push_back(&b);
  }
  EXPECT_FALSE(a.hook.linked());
  EXPECT_FALSE(b.hook.linked());

  NodeList list;
  list.push_back(&a);
  list.push_back(&b);
  list.clear();
  EXPECT_TRUE(list.empty());
  EXPECT_FALSE(a.hook.linked());
  EXPECT_FALSE(b.hook.linked());
}

TEST(IntrusiveList, CopyingOrMovingHookDoesNotCopyMembership) {
  NodeList list, destination_list;
  Node source(1), destination(2);
  list.push_back(&source);
  destination_list.push_back(&destination);

  IntrusiveListHook copied(source.hook);
  IntrusiveListHook moved(std::move(source.hook));
  EXPECT_FALSE(copied.linked());
  EXPECT_FALSE(moved.linked());
  EXPECT_TRUE(source.hook.linked());

  destination.hook = source.hook;
  EXPECT_TRUE(source.hook.linked());
  EXPECT_TRUE(destination.hook.linked());
  EXPECT_EQ(list.front(), &source);
  EXPECT_EQ(destination_list.front(), &destination);

  destination.hook = std::move(source.hook);
  EXPECT_TRUE(source.hook.linked());
  EXPECT_TRUE(destination.hook.linked());
  EXPECT_EQ(list.front(), &source);
  EXPECT_EQ(destination_list.front(), &destination);
}

TEST(SafeIntrusiveList, ExplicitMembershipOperationsResetHooks) {
  SafeNode a(1), b(2), c(3), d(4), e(5);
  SafeNodeList source;
  SafeNodeList destination;
  source.push_back(&a);
  source.push_back(&b);
  source.push_back(&c);
  destination.push_back(&d);

  EXPECT_EQ(std::ranges::distance(source), 3);
  auto moved = destination.splice(destination.end(), source, source.begin());
  EXPECT_EQ(&*moved, &a);
  EXPECT_TRUE(a.hook.linked());
  EXPECT_EQ(source.front(), &b);

  // Same-list splice is shared with the auto-unlink policy and keeps hooks
  // linked throughout the constant-time relink.
  auto b_position = source.iterator_to(&b);
  source.splice(source.end(), source, b_position);
  EXPECT_EQ(source.back(), &b);

  SafeNodeList tail;
  tail.push_back(&e);
  destination.append(tail);
  EXPECT_TRUE(tail.empty());
  EXPECT_EQ(destination.back(), &e);

  destination.remove(&a);
  EXPECT_FALSE(a.hook.linked());
  EXPECT_EQ(destination.pop_front(), &d);
  EXPECT_FALSE(d.hook.linked());
  destination.clear();
  EXPECT_FALSE(e.hook.linked());
  source.clear();
  EXPECT_FALSE(b.hook.linked());
  EXPECT_FALSE(c.hook.linked());
}

TEST(SafeIntrusiveList, BatchTraversalSpliceAndSwapShareTheNaturalApi) {
  SafeNode a(1), b(2), c(4), d(8);
  SafeNodeList source;
  SafeNodeList destination;
  for (SafeNode *node : {&a, &b, &c})
    source.push_back(node);
  destination.push_back(&d);

  source.for_each(&SafeNode::increment);
  auto first = source.begin();
  auto after = first;
  ++after;
  ++after;
  destination.splice(destination.end(), source, first, after);
  EXPECT_EQ(source.front(), &c);
  EXPECT_EQ(destination.front(), &d);

  SafeNodeList exchanged;
  exchanged.swap(destination);
  EXPECT_TRUE(destination.empty());
  EXPECT_EQ(exchanged.front(), &d);
  EXPECT_EQ(exchanged.consume_all([](SafeNode *node) noexcept {
              EXPECT_FALSE(node->hook.linked());
            }),
            3u);
  EXPECT_TRUE(exchanged.empty());
  source.clear();
}

TEST(SafeIntrusiveList, EmptyAndNonEmptyListDestructionAreSafe) {
  { SafeNodeList empty; }

  SafeNode a(1), b(2);
  {
    SafeNodeList list;
    list.push_back(&a);
    list.push_back(&b);
  }
  EXPECT_FALSE(a.hook.linked());
  EXPECT_FALSE(b.hook.linked());
}

TEST(SafeIntrusiveList, CopyingOrMovingHookNeverCopiesMembership) {
  SafeNodeList source_list, destination_list;
  SafeNode source(1), destination(2);
  source_list.push_back(&source);
  destination_list.push_back(&destination);

  SafeIntrusiveListHook copied(source.hook);
  SafeIntrusiveListHook moved(std::move(source.hook));
  EXPECT_FALSE(copied.linked());
  EXPECT_FALSE(moved.linked());

  destination.hook = source.hook;
  EXPECT_TRUE(source.hook.linked());
  EXPECT_TRUE(destination.hook.linked());
  destination.hook = std::move(source.hook);
  EXPECT_TRUE(source.hook.linked());
  EXPECT_TRUE(destination.hook.linked());

  source_list.clear();
  destination_list.clear();
}

TEST(SafeIntrusiveList, MoveConstructionAndAssignmentTransferMembership) {
  SafeNode a(1), b(2), c(3);
  SafeNodeList source;
  source.push_back(&a);
  source.push_back(&b);

  SafeNodeList moved(std::move(source));
  EXPECT_TRUE(source.empty());
  EXPECT_EQ(moved.front(), &a);
  EXPECT_EQ(moved.back(), &b);

  SafeNodeList destination;
  destination.push_back(&c);
  destination = std::move(moved);
  EXPECT_EQ(destination.front(), &a);
  EXPECT_EQ(destination.back(), &b);
  // O(1) move assignment swaps memberships; the moved-from list receives the
  // old destination nodes rather than walking them to reset every hook.
  EXPECT_EQ(moved.front(), &c);
  EXPECT_TRUE(c.hook.linked());

  destination.clear();
  moved.clear();
}

TEST(IntrusiveList, ForEachPropagatesExceptionsWithoutChangingList) {
  NodeList list;
  Node a(1), b(2);
  list.push_back(&a);
  list.push_back(&b);

  g_throw_from_visitor = true;
  EXPECT_THROW(list.for_each([](Node *) {
    if (g_throw_from_visitor)
      throw 7;
  }),
               int);
  g_throw_from_visitor = false;
  EXPECT_EQ(list.front(), &a);
  EXPECT_EQ(list.back(), &b);
}

TEST(IntrusiveList, TryInsertionRejectsInvalidMembership) {
  NodeList list;
  Node node(1);
  EXPECT_FALSE(list.try_push_back(nullptr));
  EXPECT_TRUE(list.try_push_back(&node));
  EXPECT_FALSE(list.try_push_front(&node));
  EXPECT_EQ(list.pop_front(), &node);
  EXPECT_TRUE(list.try_push_front(&node));
  EXPECT_EQ(list.pop_back(), &node);
}

TEST(IntrusiveList, RandomizedOperationsMatchOwningListModel) {
  constexpr std::size_t kNodeCount = 32;
  constexpr std::size_t kSteps = 2000;
  std::vector<std::unique_ptr<Node>> storage;
  storage.reserve(kNodeCount);
  for (std::size_t index = 0; index < kNodeCount; ++index)
    storage.push_back(std::make_unique<Node>(static_cast<int>(index)));

  NodeList actual[2];
  std::list<Node *> model[2];
  std::mt19937 random(0x48594c45u);

  auto model_at = [&](int list_index, std::size_t index) {
    auto position = model[list_index].begin();
    std::advance(position, static_cast<std::ptrdiff_t>(index));
    return position;
  };
  auto actual_at = [&](int list_index, std::size_t index) {
    if (index == model[list_index].size())
      return actual[list_index].end();
    return actual[list_index].iterator_to(*model_at(list_index, index));
  };
  auto verify = [&] {
    for (int list_index = 0; list_index < 2; ++list_index) {
      auto expected = model[list_index].begin();
      for (Node &node : actual[list_index]) {
        ASSERT_NE(expected, model[list_index].end());
        EXPECT_EQ(&node, *expected);
        ++expected;
      }
      EXPECT_EQ(expected, model[list_index].end());
      auto reverse_expected = model[list_index].rbegin();
      for (auto position = actual[list_index].rbegin();
           position != actual[list_index].rend(); ++position) {
        ASSERT_NE(reverse_expected, model[list_index].rend());
        EXPECT_EQ(&*position, *reverse_expected);
        ++reverse_expected;
      }
      EXPECT_EQ(reverse_expected, model[list_index].rend());
      EXPECT_EQ(actual[list_index].empty(), model[list_index].empty());
    }
    for (const auto &node : storage) {
      const bool expected = std::find(model[0].begin(), model[0].end(),
                                      node.get()) != model[0].end() ||
                            std::find(model[1].begin(), model[1].end(),
                                      node.get()) != model[1].end();
      EXPECT_EQ(node->hook.linked(), expected);
    }
  };

  for (std::size_t step = 0; step < kSteps; ++step) {
    const unsigned operation = random() % 7;
    const int first_index = static_cast<int>(random() & 1u);
    const int second_index = 1 - first_index;
    if (operation == 0) {
      std::vector<Node *> available;
      for (const auto &node : storage) {
        if (!node->hook.linked())
          available.push_back(node.get());
      }
      if (!available.empty()) {
        Node *node = available[random() % available.size()];
        if ((random() & 1u) == 0) {
          actual[first_index].push_front(node);
          model[first_index].push_front(node);
        } else {
          actual[first_index].push_back(node);
          model[first_index].push_back(node);
        }
      }
    } else if (operation == 1) {
      if (!model[first_index].empty()) {
        const bool front = (random() & 1u) == 0;
        Node *expected =
            front ? model[first_index].front() : model[first_index].back();
        Node *removed = front ? actual[first_index].pop_front()
                              : actual[first_index].pop_back();
        EXPECT_EQ(removed, expected);
        if (front)
          model[first_index].pop_front();
        else
          model[first_index].pop_back();
      }
    } else if (operation == 2) {
      if (!model[first_index].empty()) {
        const std::size_t item = random() % model[first_index].size();
        auto expected = model_at(first_index, item);
        actual[first_index].erase(actual[first_index].iterator_to(*expected));
        model[first_index].erase(expected);
      }
    } else if (operation == 3) {
      if (!model[first_index].empty()) {
        const std::size_t item = random() % model[first_index].size();
        const std::size_t position =
            random() % (model[second_index].size() + 1);
        auto expected_item = model_at(first_index, item);
        auto expected_position = model_at(second_index, position);
        actual[second_index].splice(
            actual_at(second_index, position), actual[first_index],
            actual[first_index].iterator_to(*expected_item));
        model[second_index].splice(expected_position, model[first_index],
                                   expected_item);
      }
    } else if (operation == 4) {
      if (!model[first_index].empty()) {
        const std::size_t begin = random() % model[first_index].size();
        const std::size_t end =
            begin + 1 + random() % (model[first_index].size() - begin);
        const std::size_t position =
            random() % (model[second_index].size() + 1);
        auto expected_first = model_at(first_index, begin);
        auto expected_last = model_at(first_index, end);
        auto expected_position = model_at(second_index, position);
        actual[second_index].splice(
            actual_at(second_index, position), actual[first_index],
            actual_at(first_index, begin), actual_at(first_index, end));
        model[second_index].splice(expected_position, model[first_index],
                                   expected_first, expected_last);
      }
    } else if (operation == 5) {
      actual[second_index].splice(actual[second_index].end(),
                                  actual[first_index]);
      model[second_index].splice(model[second_index].end(), model[first_index]);
    } else if (!model[first_index].empty()) {
      const std::size_t item = random() % model[first_index].size();
      Node *node = *model_at(first_index, item);
      node->hook.unlink();
      model[first_index].remove(node);
    }
    verify();
  }
}

TEST(SafeIntrusiveList, RandomizedOperationsMatchOwningListModel) {
  constexpr std::size_t kNodeCount = 24;
  constexpr std::size_t kSteps = 1200;
  std::vector<std::unique_ptr<SafeNode>> storage;
  storage.reserve(kNodeCount);
  for (std::size_t index = 0; index < kNodeCount; ++index)
    storage.push_back(std::make_unique<SafeNode>(static_cast<int>(index)));

  SafeNodeList actual[2];
  std::list<SafeNode *> model[2];
  std::mt19937 random(0x53414645u);

  auto model_at = [&](int list_index, std::size_t index) {
    auto position = model[list_index].begin();
    std::advance(position, static_cast<std::ptrdiff_t>(index));
    return position;
  };
  auto actual_at = [&](int list_index, std::size_t index) {
    if (index == model[list_index].size())
      return actual[list_index].end();
    return actual[list_index].iterator_to(*model_at(list_index, index));
  };
  auto verify = [&] {
    for (int list_index = 0; list_index < 2; ++list_index) {
      auto expected = model[list_index].begin();
      for (SafeNode &node : actual[list_index]) {
        ASSERT_NE(expected, model[list_index].end());
        EXPECT_EQ(&node, *expected);
        ++expected;
      }
      EXPECT_EQ(expected, model[list_index].end());
      auto reverse_expected = model[list_index].rbegin();
      for (auto position = actual[list_index].rbegin();
           position != actual[list_index].rend(); ++position) {
        ASSERT_NE(reverse_expected, model[list_index].rend());
        EXPECT_EQ(&*position, *reverse_expected);
        ++reverse_expected;
      }
      EXPECT_EQ(reverse_expected, model[list_index].rend());
      EXPECT_EQ(actual[list_index].empty(), model[list_index].empty());
      EXPECT_EQ(std::ranges::distance(actual[list_index]),
                static_cast<std::ptrdiff_t>(model[list_index].size()));
    }
    for (const auto &node : storage) {
      const bool expected = std::find(model[0].begin(), model[0].end(),
                                      node.get()) != model[0].end() ||
                            std::find(model[1].begin(), model[1].end(),
                                      node.get()) != model[1].end();
      EXPECT_EQ(node->hook.linked(), expected);
    }
  };

  for (std::size_t step = 0; step < kSteps; ++step) {
    const unsigned operation = random() % 6;
    const int first_index = static_cast<int>(random() & 1u);
    const int second_index = 1 - first_index;
    if (operation == 0) {
      std::vector<SafeNode *> available;
      for (const auto &node : storage) {
        if (!node->hook.linked())
          available.push_back(node.get());
      }
      if (!available.empty()) {
        SafeNode *node = available[random() % available.size()];
        if ((random() & 1u) == 0) {
          actual[first_index].push_front(node);
          model[first_index].push_front(node);
        } else {
          actual[first_index].push_back(node);
          model[first_index].push_back(node);
        }
      }
    } else if (operation == 1 && !model[first_index].empty()) {
      const bool front = (random() & 1u) == 0;
      SafeNode *expected =
          front ? model[first_index].front() : model[first_index].back();
      SafeNode *removed = front ? actual[first_index].pop_front()
                                : actual[first_index].pop_back();
      EXPECT_EQ(removed, expected);
      if (front)
        model[first_index].pop_front();
      else
        model[first_index].pop_back();
    } else if (operation == 2 && !model[first_index].empty()) {
      const std::size_t item = random() % model[first_index].size();
      auto expected = model_at(first_index, item);
      actual[first_index].erase(actual[first_index].iterator_to(*expected));
      model[first_index].erase(expected);
    } else if (operation == 3 && !model[first_index].empty()) {
      const std::size_t item = random() % model[first_index].size();
      const std::size_t position = random() % (model[second_index].size() + 1);
      auto expected_item = model_at(first_index, item);
      auto expected_position = model_at(second_index, position);
      actual[second_index].splice(
          actual_at(second_index, position), actual[first_index],
          actual[first_index].iterator_to(*expected_item));
      model[second_index].splice(expected_position, model[first_index],
                                 expected_item);
    } else if (operation == 4 && !model[first_index].empty()) {
      const std::size_t begin = random() % model[first_index].size();
      const std::size_t end =
          begin + 1 + random() % (model[first_index].size() - begin);
      const std::size_t position = random() % (model[second_index].size() + 1);
      auto expected_first = model_at(first_index, begin);
      auto expected_last = model_at(first_index, end);
      auto expected_position = model_at(second_index, position);
      actual[second_index].splice(
          actual_at(second_index, position), actual[first_index],
          actual_at(first_index, begin), actual_at(first_index, end));
      model[second_index].splice(expected_position, model[first_index],
                                 expected_first, expected_last);
    } else if (operation == 5) {
      actual[second_index].splice(actual[second_index].end(),
                                  actual[first_index]);
      model[second_index].splice(model[second_index].end(), model[first_index]);
    }
    verify();
  }
}

#if !defined(NDEBUG)
TEST(SafeIntrusiveList, DestroyingLinkedNodeFailsFastInDebug) {
  EXPECT_DEATH(
      {
        SafeNodeList list;
        auto *node = new SafeNode(1);
        list.push_back(node);
        delete node;
      },
      "HYLE fatal");
}

TEST(IntrusiveList, DiagnosesUncheckedMembershipViolationsInDebug) {
  EXPECT_DEATH(
      {
        NodeList list;
        Node node(1);
        list.push_back(&node);
        list.push_back(&node);
      },
      "");
  EXPECT_DEATH(
      {
        NodeList list;
        (void)list.erase(list.end());
      },
      "");
  EXPECT_DEATH(
      {
        NodeList list;
        Node node(1);
        list.remove(&node);
      },
      "");
}

TEST(IntrusiveList, DiagnosesSelfWholeListTransfersInDebug) {
  EXPECT_DEATH(
      {
        NodeList list;
        Node node(1);
        list.push_back(&node);
        list.splice(list.end(), list);
      },
      "");
  EXPECT_DEATH(
      {
        NodeList list;
        Node node(1);
        list.push_back(&node);
        list.append(list);
      },
      "");
}
#endif
