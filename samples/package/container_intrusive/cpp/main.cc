#include <hyle/container/intrusive_list.h>

struct Node {
  int value;
  hyle::container::IntrusiveListHook hook{};

  explicit Node(int input) noexcept : value(input) {}
};

struct SafeNode {
  int value;
  hyle::container::SafeIntrusiveListHook hook{};

  explicit SafeNode(int input) noexcept : value(input) {}
};

int main() {
  using List = hyle::container::IntrusiveList<Node, &Node::hook>;
  using SafeList =
      hyle::container::SafeIntrusiveList<SafeNode, &SafeNode::hook>;
  Node first{20};
  Node second{22};
  Node cancelled{-1};
  List ready;
  List incoming;
  SafeNode explicit_retirement{7};
  SafeList owned;

  ready.push_back(&first);
  const auto cancelled_position = ready.insert(ready.end(), &cancelled);
  ready.erase(cancelled_position);
  incoming.push_back(&second);
  ready.splice(ready.end(), incoming);
  owned.push_back(&explicit_retirement);
  owned.remove(&explicit_retirement);

  int sum = 0;
  for (const Node& node : ready) {
    sum += node.value;
  }
  return sum == 42 && ready.pop_front() == &first &&
                 ready.pop_front() == &second && ready.empty() &&
                 !cancelled.hook.linked() &&
                 !explicit_retirement.hook.linked() && owned.empty()
             ? 0
             : 1;
}
