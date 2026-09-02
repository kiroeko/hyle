#include "hyle/container/intrusive_forward_list.h"
#include "hyle/container/intrusive_list.h"
#include "hyle/core/environment.h"

#if HYLE_HAS_EXCEPTIONS
#  error "container_intrusive no-exception consumer unexpectedly has exceptions enabled"
#endif

namespace {

struct ListNode {
  int value;
  hyle::container::SafeIntrusiveListHook hook;

  ListNode(int input) noexcept : value(input) {}
};

struct ForwardNode {
  int value;
  hyle::container::IntrusiveForwardListHook hook;

  ForwardNode(int input) noexcept : value(input) {}
};

} // namespace

int main() {
  ListNode list_nodes[2]{{2}, {1}};
  hyle::container::SafeIntrusiveList<ListNode, &ListNode::hook> list;
  list.push_back(&list_nodes[0]);
  list.push_back(&list_nodes[1]);
  list.sort([](const ListNode &left, const ListNode &right) noexcept {
    return left.value < right.value;
  });

  ForwardNode forward_nodes[2]{{2}, {1}};
  hyle::container::IntrusiveForwardList<ForwardNode, &ForwardNode::hook>
      forward;
  forward.push_back(&forward_nodes[0]);
  forward.push_back(&forward_nodes[1]);
  forward.sort([](const ForwardNode &left,
                  const ForwardNode &right) noexcept {
    return left.value < right.value;
  });

  const bool ordered = list.front() == &list_nodes[1] &&
                       forward.front() == &forward_nodes[1];
  list.clear();
  forward.clear();
  return ordered ? 0 : 1;
}
