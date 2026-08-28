#include <nanobench.h>

#include <array>
#include <bit>
#include <bitset>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#if defined(__APPLE__)
#include <pthread.h>
#endif

#include "hyle/time/timer_queue.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::time::TimerEntry;
using hyle::time::TimerQueue;

constexpr std::size_t kTimerCount = 64;
constexpr std::size_t kCancelledCount = 45;
constexpr std::uint64_t kTickNs = 100'000;

struct ThreadQosState {
  const char* name;
  const char* ok;
};

ThreadQosState configure_main_benchmark_thread() {
#if defined(__APPLE__)
  const char* requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  const char* name = "user-interactive";
  if (requested != nullptr && std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
    name = "utility";
  } else if (requested != nullptr &&
             std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr,
                 "hyle_time_consumer_bench: HYLE_BENCH_APPLE_QOS must be "
                 "interactive or utility (got %s)\n",
                 requested);
    std::abort();
  }
  const int result = pthread_set_qos_class_self_np(qos, 0);
  if (result != 0) std::abort();
  return {name, "1"};
#else
  return {"not-applicable", "not-applicable"};
#endif
}

bool environment_present(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

struct Payload {
  std::uint64_t id = 0;
  std::uint64_t* checksum = nullptr;
};

void observe_payload(Payload* payload) noexcept {
  *payload->checksum += (payload->id + 1u) * 0x9e3779b1u;
}

void hyle_fire(TimerEntry* entry) noexcept {
  observe_payload(static_cast<Payload*>(entry->data()));
}

// Faithful dependency-free port of Seastar timer_set's 64 radix buckets for a
// signed 64-bit chrono duration,
// deferred sorting, boost::intrusive default circular-list/count shape and
// lazy minimum recomputation. The Timer node carries the same persistent
// callback/data/owner metadata as HYLE's public TimerEntry; execution policy is
// intentionally outside both containers.
class SeastarTimerSet {
 public:
  static constexpr std::size_t kBucketCount = 64;

  struct Link {
    Link* previous = nullptr;
    Link* next = nullptr;
  };

  struct Node {
    Link hook;
    std::int64_t deadline_ns = 0;
    void* data = nullptr;
    void (*fire_callback)(Node*) noexcept = nullptr;
    const void* owner = nullptr;
  };

 private:
  class List {
   public:
    List() noexcept {
      root_.previous = &root_;
      root_.next = &root_;
    }

    [[nodiscard]] bool empty() const noexcept {
      return root_.next == &root_;
    }

    void push_back(Node* node) noexcept {
      Link* hook = &node->hook;
      HYLE_FATAL_ASSERT(hook->previous == nullptr && hook->next == nullptr);
      hook->previous = root_.previous;
      hook->next = &root_;
      root_.previous->next = hook;
      root_.previous = hook;
      ++size_;
    }

    Node* pop_front() noexcept {
      if (empty()) return nullptr;
      Link* hook = root_.next;
      hook->next->previous = &root_;
      root_.next = hook->next;
      hook->previous = nullptr;
      hook->next = nullptr;
      --size_;
      return reinterpret_cast<Node*>(hook);
    }

    void remove(Node* node) noexcept {
      Link* hook = &node->hook;
      hook->previous->next = hook->next;
      hook->next->previous = hook->previous;
      hook->previous = nullptr;
      hook->next = nullptr;
      --size_;
    }

    void append(List& source) noexcept {
      if (source.empty()) return;
      Link* first = source.root_.next;
      Link* last = source.root_.previous;
      Link* destination_last = root_.previous;
      destination_last->next = first;
      first->previous = destination_last;
      last->next = &root_;
      root_.previous = last;
      size_ += source.size_;
      source.root_.previous = &source.root_;
      source.root_.next = &source.root_;
      source.size_ = 0;
    }

    template <class Function>
    void for_each(Function&& function) noexcept {
      for (Link* hook = root_.next; hook != &root_; hook = hook->next) {
        function(reinterpret_cast<Node*>(hook));
      }
    }

   private:
    Link root_;
    std::size_t size_ = 0;
  };

 public:
  using ReadyList = List;

  void schedule(Node* node, std::uint64_t deadline_ns) noexcept {
    HYLE_FATAL_ASSERT(node != nullptr);
    HYLE_FATAL_ASSERT(
        deadline_ns <=
        static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()));
    if (node->owner != nullptr) {
      HYLE_FATAL_ASSERT(node->owner == this);
      cancel(node);
    }
    node->deadline_ns = static_cast<std::int64_t>(deadline_ns);
    node->owner = this;
    insert(node);
  }

  bool cancel(Node* node) noexcept {
    HYLE_FATAL_ASSERT(node != nullptr);
    if (node->owner == nullptr) return false;
    HYLE_FATAL_ASSERT(node->owner == this);
    const std::size_t index = index_for(node->deadline_ns);
    buckets_[index].remove(node);
    if (buckets_[index].empty()) nonempty_.reset(index);
    node->owner = nullptr;
    return true;
  }

  void advance(std::uint64_t now_ns, ReadyList& output) noexcept {
    HYLE_FATAL_ASSERT(
        now_ns <=
        static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()));
    const std::int64_t timestamp = static_cast<std::int64_t>(now_ns);
    HYLE_FATAL_ASSERT(timestamp >= last_ns_);
    const std::size_t index = index_for(timestamp);
    ReadyList ready;

    // Seastar bitsets::for_each_set(nonempty, index + 1): visit only populated
    // buckets via trailing-zero scans, not a benchmark-only linear 64-bucket
    // walk.
    std::uint64_t known_due = nonempty_.to_ullong();
    known_due = index == kBucketCount - 1
                    ? 0
                    : known_due &
                          (~std::uint64_t{0} << (index + 1));
    while (known_due != 0) {
      const std::size_t bucket =
          static_cast<std::size_t>(std::countr_zero(known_due));
      ready.append(buckets_[bucket]);
      nonempty_.reset(bucket);
      known_due &= known_due - 1;
    }

    ReadyList inspect;
    inspect.append(buckets_[index]);
    nonempty_.reset(index);
    last_ns_ = timestamp;
    next_ns_ = (std::numeric_limits<std::int64_t>::max)();
    while (Node* node = inspect.pop_front()) {
      if (node->deadline_ns <= timestamp) {
        ready.push_back(node);
      } else {
        insert(node);
      }
    }

    if (next_ns_ == (std::numeric_limits<std::int64_t>::max)() &&
        nonempty_.any()) {
      const std::size_t nearest = kBucketCount - 1u -
                                  static_cast<std::size_t>(std::countl_zero(
                                      nonempty_.to_ullong()));
      buckets_[nearest].for_each([&](Node* node) noexcept {
        if (node->deadline_ns < next_ns_) next_ns_ = node->deadline_ns;
      });
    }

    ready.for_each([](Node* node) noexcept { node->owner = nullptr; });
    output.append(ready);
  }

  [[nodiscard]] bool empty() const noexcept { return nonempty_.none(); }

 private:
  [[nodiscard]] std::size_t index_for(
      std::int64_t deadline_ns) const noexcept {
    if (deadline_ns <= last_ns_) return kBucketCount - 1;
    return static_cast<std::size_t>(
        std::countl_zero(static_cast<std::uint64_t>(deadline_ns ^ last_ns_)));
  }

  void insert(Node* node) noexcept {
    const std::size_t index = index_for(node->deadline_ns);
    buckets_[index].push_back(node);
    nonempty_.set(index);
    if (node->deadline_ns < next_ns_) next_ns_ = node->deadline_ns;
  }

  std::array<List, kBucketCount> buckets_{};
  std::bitset<kBucketCount> nonempty_{};
  std::int64_t last_ns_ = 0;
  std::int64_t next_ns_ = (std::numeric_limits<std::int64_t>::max)();
};

static_assert(sizeof(SeastarTimerSet::Node) == sizeof(TimerEntry));
static_assert(alignof(SeastarTimerSet::Node) == alignof(TimerEntry));

void seastar_fire(SeastarTimerSet::Node* node) noexcept {
  observe_payload(static_cast<Payload*>(node->data));
}

template <class Operations>
void replay_scheduler_lifecycle(Operations& operations,
                                std::uint64_t base_ns) noexcept {
  // Stable nodes are armed in a permutation rather than backing-address order;
  // deadline is a separate 1..64 tick distribution. This mirrors one worker's
  // heterogeneous sleep/io timers without charging random generation.
  for (std::size_t step = 0; step < kTimerCount; ++step) {
    const std::size_t index = (step * 37u + 11u) & (kTimerCount - 1u);
    operations.schedule(index, base_ns + (index + 1u) * kTickNs);
  }

  // Cancel 45/64 (70.3125%) in another deterministic permutation. Preserve the
  // deadline-64 node so the queue remains live through the complete tick replay.
  std::size_t cancelled = 0;
  for (std::size_t step = 0;
       step < kTimerCount && cancelled < kCancelledCount; ++step) {
    const std::size_t index = (step * 29u + 7u) & (kTimerCount - 1u);
    if (index == kTimerCount - 1u) continue;
    operations.cancel(index);
    ++cancelled;
  }
  HYLE_FATAL_ASSERT(cancelled == kCancelledCount);

  for (std::size_t tick = 1; tick <= kTimerCount; ++tick) {
    HYLE_FATAL_ASSERT(!operations.empty());
    operations.advance(base_ns + tick * kTickNs);
  }
  HYLE_FATAL_ASSERT(operations.empty());
  HYLE_FATAL_ASSERT(operations.drain() ==
                    kTimerCount - kCancelledCount);
}

class HyleReplay {
 public:
  HyleReplay() {
    for (std::size_t index = 0; index < kTimerCount; ++index) {
      payloads_[index] = Payload{index, &checksum_};
      entries_[index] =
          std::make_unique<TimerEntry>(&payloads_[index], &hyle_fire);
    }
  }

  void run() noexcept {
    checksum_ = 0;
    replay_scheduler_lifecycle(*this, base_ns_);
    base_ns_ += (kTimerCount + 1u) * kTickNs;
    nb::doNotOptimizeAway(checksum_);
  }

  void schedule(std::size_t index, std::uint64_t deadline_ns) noexcept {
    queue_.schedule(entries_[index].get(), deadline_ns);
  }
  void cancel(std::size_t index) noexcept {
    (void)queue_.cancel(entries_[index].get());
  }
  void advance(std::uint64_t now_ns) noexcept {
    (void)queue_.advance(now_ns, ready_);
  }
  [[nodiscard]] bool empty() const noexcept { return queue_.empty(); }
  std::size_t drain() noexcept {
    std::size_t fired = 0;
    ready_.consume_all([&](TimerEntry* entry) noexcept {
      entry->fire_callback()(entry);
      ++fired;
    });
    return fired;
  }

 private:
  TimerQueue queue_{0};
  std::array<Payload, kTimerCount> payloads_{};
  std::array<std::unique_ptr<TimerEntry>, kTimerCount> entries_{};
  TimerQueue::EntryList ready_;
  std::uint64_t base_ns_ = 0;
  std::uint64_t checksum_ = 0;
};

class SeastarReplay {
 public:
  SeastarReplay() {
    for (std::size_t index = 0; index < kTimerCount; ++index) {
      payloads_[index] = Payload{index, &checksum_};
      entries_[index] = std::make_unique<SeastarTimerSet::Node>();
      entries_[index]->data = &payloads_[index];
      entries_[index]->fire_callback = &seastar_fire;
    }
  }

  void run() noexcept {
    checksum_ = 0;
    replay_scheduler_lifecycle(*this, base_ns_);
    base_ns_ += (kTimerCount + 1u) * kTickNs;
    nb::doNotOptimizeAway(checksum_);
  }

  void schedule(std::size_t index, std::uint64_t deadline_ns) noexcept {
    queue_.schedule(entries_[index].get(), deadline_ns);
  }
  void cancel(std::size_t index) noexcept {
    (void)queue_.cancel(entries_[index].get());
  }
  void advance(std::uint64_t now_ns) noexcept {
    queue_.advance(now_ns, ready_);
  }
  [[nodiscard]] bool empty() const noexcept { return queue_.empty(); }
  std::size_t drain() noexcept {
    std::size_t fired = 0;
    while (SeastarTimerSet::Node* node = ready_.pop_front()) {
      node->fire_callback(node);
      ++fired;
    }
    return fired;
  }

 private:
  SeastarTimerSet queue_;
  std::array<Payload, kTimerCount> payloads_{};
  std::array<std::unique_ptr<SeastarTimerSet::Node>, kTimerCount> entries_{};
  SeastarTimerSet::ReadyList ready_;
  std::uint64_t base_ns_ = 0;
  std::uint64_t checksum_ = 0;
};

struct Replays {
  std::unique_ptr<HyleReplay> hyle;
  std::unique_ptr<SeastarReplay> seastar;

  explicit Replays(bool peer_allocation_first) {
    if (peer_allocation_first) {
      seastar = std::make_unique<SeastarReplay>();
      hyle = std::make_unique<HyleReplay>();
    } else {
      hyle = std::make_unique<HyleReplay>();
      seastar = std::make_unique<SeastarReplay>();
    }
  }
};

}  // namespace

int main() {
  const ThreadQosState qos = configure_main_benchmark_thread();
  const bool rival_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_TIME_CONSUMER_RUN execution_order=%s allocation_order=%s "
      "apple_qos=%s qos_ok=%s affinity=external run_nonce=%s\n",
      rival_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
      qos.ok, hyle::bench::perf::run_nonce());

  Replays replay(peer_allocation_first);
  nb::Bench bench;
#if defined(__APPLE__)
  bench.minEpochTime(std::chrono::milliseconds(8));
#endif
  bench.title("scheduler timer lifecycle aggregate N=64")
      .relative(true)
      .batch(kTimerCount)
      .minEpochIterations(500);
  if (rival_first) {
    bench.run("Seastar timer_set lifecycle N=64",
              [&] { replay.seastar->run(); });
    bench.run("HYLE TimerQueue lifecycle N=64", [&] { replay.hyle->run(); });
  } else {
    bench.run("HYLE TimerQueue lifecycle N=64", [&] { replay.hyle->run(); });
    bench.run("Seastar timer_set lifecycle N=64",
              [&] { replay.seastar->run(); });
  }
}
