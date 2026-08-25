#include <hyle/core/cache.h>
#include <hyle/core/intrusive_list.h>
#include <hyle/core/version.h>
#include <hyle/macros.h>

#include <cstddef>

struct Node {
  int value;
  hyle::core::IntrusiveListHook hook;
};

HYLE_NOINLINE int consume(int value) {
  return HYLE_LIKELY(value == 42) ? value : 0;
}

HYLE_NODISCARD int sum_values(const int* HYLE_RESTRICT values,
                              std::size_t count) {
  HYLE_ASSUME(values != nullptr);
  int total = 0;
  HYLE_UNROLL(4)
  for (std::size_t index = 0; index < count; ++index) total += values[index];
  return total;
}

int main() {
  using List = hyle::core::IntrusiveList<Node, &Node::hook>;
  List queue;
  Node first{20};
  Node second{22};
  const int values[] = {20, 22};
  queue.push_back(&first);
  queue.push_back(&second);
  int sum = 0;
  for (const Node& node : queue) sum += node.value;
  hyle::core::prefetch_read(&first);
  return hyle::version() == hyle::kVersion && queue.pop_front() == &first &&
                 queue.pop_front() == &second && queue.empty() &&
                 consume(sum) == 42 && sum_values(values, 2) == 42
             ? 0
             : 1;
}
