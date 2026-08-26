#include <hyle/container/intrusive_list.h>

struct Node {
  int value;
  hyle::container::IntrusiveListHook hook{};
};

int main() {
  using List = hyle::container::IntrusiveList<Node, &Node::hook>;
  Node first{20};
  Node second{22};
  Node cancelled{-1};
  List ready;
  List incoming;

  ready.push_back(&first);
  const auto cancelled_position = ready.insert(ready.end(), &cancelled);
  ready.erase(cancelled_position);
  incoming.push_back(&second);
  ready.splice(ready.end(), incoming);

  int sum = 0;
  for (const Node& node : ready) {
    sum += node.value;
  }
  return sum == 42 && ready.pop_front() == &first &&
                 ready.pop_front() == &second && ready.empty() &&
                 !cancelled.hook.linked()
             ? 0
             : 1;
}
