#include "hyle/container/intrusive_list.h"

#include <gtest/gtest.h>

#include <concepts>
#include <iterator>
#include <ranges>
#include <utility>
#include <vector>

using hyle::container::IntrusiveList;
using hyle::container::IntrusiveListHook;

namespace {
struct Node {
  int value;
  IntrusiveListHook hook;
  explicit Node(int input) : value(input) {}
};
using NodeList = IntrusiveList<Node, &Node::hook>;
static_assert(std::bidirectional_iterator<NodeList::iterator>);
static_assert(std::bidirectional_iterator<NodeList::const_iterator>);
static_assert(std::ranges::bidirectional_range<NodeList>);
static_assert(std::ranges::bidirectional_range<const NodeList>);
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
  EXPECT_FALSE(c.hook.linked());
  EXPECT_TRUE(moved.empty());
  EXPECT_EQ(destination.pop_front(), &a);
  EXPECT_EQ(destination.pop_front(), &b);
}

TEST(IntrusiveList, PrefixAndConsumeSupportBatchHandoff) {
  NodeList source;
  NodeList destination;
  Node a(1), b(2), c(3), d(4);
  source.push_back(&a);
  source.push_back(&b);
  source.push_back(&c);
  destination.push_back(&d);
  int moved = 0;

  EXPECT_EQ(source.move_prefix_to(
                destination,
                [](const Node *node) noexcept { return node->value <= 2; },
                [&moved](Node *) noexcept { ++moved; }),
            2u);
  EXPECT_EQ(moved, 2);
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

TEST(IntrusiveList, PrefixCallbacksObserveUnchangedSourceMembership) {
  NodeList source, destination;
  Node a(1), b(2), c(3);
  source.push_back(&a);
  source.push_back(&b);
  source.push_back(&c);

  EXPECT_EQ(source.move_prefix_to(
                destination,
                [&](const Node *node) noexcept {
                  EXPECT_TRUE(node->hook.linked());
                  EXPECT_EQ(source.front(), &a);
                  return node->value <= 2;
                },
                [&](Node *node) noexcept {
                  EXPECT_TRUE(node->hook.linked());
                  EXPECT_EQ(source.front(), &a);
                }),
            2u);
  EXPECT_EQ(source.front(), &c);
  EXPECT_EQ(destination.front(), &a);
  EXPECT_EQ(destination.back(), &b);
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

#if !defined(NDEBUG)
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
  EXPECT_DEATH(
      {
        NodeList list;
        Node node(1);
        list.push_back(&node);
        (void)list.move_prefix_to(
            list, [](Node *) noexcept { return true; },
            [](Node *) noexcept {});
      },
      "");
}
#endif
