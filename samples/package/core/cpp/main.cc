#include <hyle/core/cache.h>
#include <hyle/core/intrusive_list.h>
#include <hyle/core/version.h>
#include <hyle/macros.h>

struct Node {
  int value;
  hyle::core::IntrusiveListHook hook;
};

HYLE_NOINLINE int consume(int value) {
  return HYLE_LIKELY(value == 42) ? value : 0;
}

int main() {
  using List = hyle::core::IntrusiveList<Node, &Node::hook>;
  List queue;
  Node item{42};
  queue.push_back(&item);
  hyle::core::prefetch_read(&item);
  return hyle::version() == hyle::kVersion && queue.pop_front() == &item &&
                 queue.empty() && consume(item.value) == 42
             ? 0
             : 1;
}
