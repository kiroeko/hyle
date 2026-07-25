#include "hyle/core/intrusive_list.h"

#include <gtest/gtest.h>

#include <utility>

using hyle::core::IntrusiveList;
using hyle::core::IntrusiveListHook;

namespace {
struct Node {
  int value;
  IntrusiveListHook hook;
  explicit Node(int input) : value(input) {}
};
using NodeList = IntrusiveList<Node, &Node::hook>;
}  // namespace

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

TEST(IntrusiveList, MoveFrontAndAppendSpliceWithoutAllocation) {
  NodeList source;
  NodeList destination;
  Node a(1), b(2), c(3), d(4);
  source.push_back(&a);
  source.push_back(&b);
  destination.push_back(&c);

  EXPECT_EQ(source.move_front_to(destination), &a);
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
                destination, [](const Node* node) { return node->value <= 2; },
                [&moved](Node*) { ++moved; }),
            2u);
  EXPECT_EQ(moved, 2);
  EXPECT_EQ(source.front(), &c);

  NodeList consumed;
  int sum = 0;
  EXPECT_EQ(destination.consume_all([&](Node* node) {
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
  EXPECT_EQ(destination.consume_all([](Node*) {}), 0u);
}

TEST(IntrusiveList, ForEachVisitsMutableAndConstNodes) {
  NodeList list;
  Node a(1), b(2), c(3);
  list.push_back(&a);
  list.push_back(&b);
  list.push_back(&c);

  list.for_each([](Node* node) { node->value *= 2; });
  const NodeList& view = list;
  int sum = 0;
  view.for_each([&sum](const Node* node) { sum += node->value; });
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
  NodeList list;
  Node source(1);
  list.push_back(&source);

  IntrusiveListHook copied(source.hook);
  IntrusiveListHook moved(std::move(source.hook));
  EXPECT_FALSE(copied.linked());
  EXPECT_FALSE(moved.linked());
  EXPECT_TRUE(source.hook.linked());
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

TEST(IntrusiveList, RejectsSelfTransfers) {
  EXPECT_DEATH(
      {
        NodeList list;
        Node node(1);
        list.push_back(&node);
        (void)list.move_front_to(list);
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
