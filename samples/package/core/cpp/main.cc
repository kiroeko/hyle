#include <hyle/core/cache.h>
#include <hyle/core/intrusive_list.h>
#include <hyle/core/version.h>
#include <hyle/macros.h>

#include <cstddef>

static_assert(hyle::core::kMinConstructiveInterferenceSize == 64);
static_assert(hyle::core::kMaxDestructiveInterferenceSize == 128);
static_assert(hyle::core::kConstructiveInterferenceSize >=
              hyle::core::kMinConstructiveInterferenceSize);
static_assert(hyle::core::kDestructiveInterferenceSize <=
              hyle::core::kMaxDestructiveInterferenceSize);
static_assert(hyle::core::kConstructiveInterferenceSize <=
              hyle::core::kDestructiveInterferenceSize);
#if defined(__APPLE__) &&                                                \
    (defined(__aarch64__) || defined(__arm64__))
static_assert(hyle::core::kConstructiveInterferenceSize == 128);
static_assert(hyle::core::kDestructiveInterferenceSize == 128);
#elif defined(__x86_64__) || defined(_M_X64)
static_assert(hyle::core::kConstructiveInterferenceSize == 64);
static_assert(hyle::core::kDestructiveInterferenceSize == 64);
#else
static_assert(hyle::core::kConstructiveInterferenceSize == 64);
static_assert(hyle::core::kDestructiveInterferenceSize == 128);
#endif

struct Node {
  int value;
  hyle::core::IntrusiveListHook hook{};
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
  List incoming;
  List deferred;
  Node first{20};
  Node second{22};
  Node cancelled{-1};
  const int values[] = {20, 22};
  queue.push_back(&first);
  auto cancelled_position = queue.insert(queue.end(), &cancelled);
  queue.erase(cancelled_position);
  incoming.push_back(&second);
  queue.splice(queue.end(), incoming);
  deferred.splice(deferred.end(), queue, queue.begin());
  queue.swap(deferred);
  queue.splice(queue.end(), deferred);
  int sum = 0;
  for (const Node& node : queue) sum += node.value;
  hyle::core::prefetch_read(&first);
  return hyle::version() == hyle::kVersion && queue.pop_front() == &first &&
                 queue.pop_front() == &second && queue.empty() &&
                 consume(sum) == 42 && sum_values(values, 2) == 42
             ? 0
             : 1;
}
