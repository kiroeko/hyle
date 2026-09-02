#include <hyle/container/intrusive_forward_list.h>
#include <hyle/container/intrusive_hlist.h>
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

struct ForwardNode {
  int value;
  hyle::container::IntrusiveForwardListHook hook{};
};

struct CallbackNode {
  int value;
  hyle::container::SafeIntrusiveHListHook hook{};
};

int main() {
  using List = hyle::container::IntrusiveList<Node, &Node::hook>;
  using SafeList =
      hyle::container::SafeIntrusiveList<SafeNode, &SafeNode::hook>;
  using CountedList =
      hyle::container::CountedIntrusiveList<SafeNode, &SafeNode::hook>;
  using ForwardList =
      hyle::container::IntrusiveForwardList<ForwardNode, &ForwardNode::hook>;
  using CallbackList =
      hyle::container::SafeIntrusiveHList<CallbackNode, &CallbackNode::hook>;
  Node first{20};
  Node second{22};
  Node cancelled{-1};
  List ready;
  List incoming;
  SafeNode explicit_retirement{7};
  SafeList owned;
  SafeNode counted_first{8};
  SafeNode counted_second{6};
  CountedList counted;
  ForwardNode work_first{9};
  ForwardNode work_second{11};
  ForwardList work_queue;
  CallbackNode callback{10};
  CallbackList callbacks;

  ready.push_back(&first);
  const auto cancelled_position = ready.insert(ready.end(), &cancelled);
  ready.erase(cancelled_position);
  incoming.push_back(&second);
  ready.splice(ready.end(), incoming);
  owned.push_back(&explicit_retirement);
  owned.remove(&explicit_retirement);
  counted.push_back(&counted_first);
  counted.push_back(&counted_second);
  counted.sort([](const SafeNode &left, const SafeNode &right) noexcept {
    return left.value < right.value;
  });
  work_queue.push_back(&work_first);
  work_queue.push_back(&work_second);
  callbacks.push_front(&callback);

  int sum = 0;
  for (const Node& node : ready) {
    sum += node.value;
  }
  return sum == 42 && ready.pop_front() == &first &&
                 ready.pop_front() == &second && ready.empty() &&
                 !cancelled.hook.linked() &&
                 !explicit_retirement.hook.linked() && owned.empty() &&
                 counted.size() == 2 &&
                 counted.pop_front() == &counted_second &&
                 counted.pop_front() == &counted_first &&
                 work_queue.back() == &work_second &&
                 work_queue.pop_front() == &work_first &&
                 work_queue.pop_front() == &work_second &&
                 callbacks.pop_front() == &callback
             ? 0
             : 1;
}
