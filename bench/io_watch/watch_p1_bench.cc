// Same-contract IO Watch v1 P1 replay.
//
// Both routes retain one watcher and one event owner outside timing. A charged
// episode publishes a new authoritative value, receives the native change,
// classifies the exact target/direct child and re-reads that value. Reading the
// value prevents a duplicate notification from an earlier publication from
// completing a later episode, without imposing a benchmark-only drain API.

#include <nanobench.h>
#include <uv.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "../common/perf_process.h"
#include "hyle/io/io_context.h"
#include "hyle/io/watch.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifndef HYLE_IO_WATCH_P1_PLACEMENT
#define HYLE_IO_WATCH_P1_PLACEMENT 0
#endif

namespace {

using hyle::Status;
using hyle::io::FileWatchBatchView;
using hyle::io::FileWatchTarget;
using hyle::io::FileWatcher;
using hyle::io::IoContext;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::JoinHandle;
using hyle::sched::Scheduler;
using hyle::sched::Task;

constexpr unsigned kOwner = 0;
constexpr std::size_t kDirectoryFiles = 8;
constexpr std::size_t kExactEpisodes = 8;
constexpr std::size_t kDirectoryEpisodes = 2;
constexpr auto kReadinessRetry = std::chrono::milliseconds(20);
constexpr auto kReadinessTimeout = std::chrono::seconds(5);
constexpr auto kRouteQuiescence = std::chrono::milliseconds(100);

enum class Shape : std::uint8_t { kExactReplace, kDirectoryBurst };

[[noreturn]] void fail(const char* message) {
  std::fprintf(stderr, "hyle_io_watch_p1_bench: %s\n", message);
  std::abort();
}

template <class T>
void wait_at_least(const std::atomic<T>& value, T target) noexcept {
  T observed = value.load(std::memory_order_acquire);
  while (observed < target) {
    value.wait(observed, std::memory_order_acquire);
    observed = value.load(std::memory_order_acquire);
  }
}

template <class T, class Rep, class Period>
bool wait_at_least_for(const std::atomic<T>& value, T target,
                       std::chrono::duration<Rep, Period> timeout) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (value.load(std::memory_order_acquire) < target) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

std::uint64_t process_id() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(::GetCurrentProcessId());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

class TempDirectory {
 public:
  TempDirectory() {
    static std::atomic<std::uint64_t> sequence{0};
    const std::uint64_t suffix =
        static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        (process_id() << 32) ^
        sequence.fetch_add(1, std::memory_order_relaxed);
    path_ = std::filesystem::temp_directory_path() /
            ("hyle_io_watch_p1_" + std::to_string(suffix));
    std::error_code error;
    if (!std::filesystem::create_directory(path_, error) || error)
      fail("could not create the local watcher fixture");
  }

  ~TempDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

bool write_value(const std::string& path, std::uint64_t value) noexcept {
  auto opened = hyle::platform::open_file(
      path.c_str(), hyle::platform::kOpenWrite |
                        hyle::platform::kOpenCreate |
                        hyle::platform::kOpenTruncate);
  if (!opened) return false;
  Status status = opened->write_all(&value, sizeof(value));
  status.Update(opened->close());
  return status.ok();
}

bool read_value(const std::string& path, std::uint64_t* value) noexcept {
  auto opened = hyle::platform::open_file(path.c_str(),
                                          hyle::platform::kOpenRead);
  if (!opened) return false;
  Status status = opened->read_exact(value, sizeof(*value));
  status.Update(opened->close());
  return status.ok();
}

struct WatchState {
  explicit WatchState(Shape selected) : shape(selected) {}

  Shape shape;
  std::string root;
  std::string exact_path;
  std::string replacement_path;
  std::string exact_basename;
  std::array<std::string, kDirectoryFiles> child_paths{};
  std::array<std::string, kDirectoryFiles> child_names{};
  std::atomic<std::uint64_t> ready{0};
  std::atomic<std::uint64_t> armed{0};
  std::atomic<std::uint64_t> requested{0};
  std::atomic<std::uint64_t> completed{0};
  std::atomic<std::uint64_t> checksum{0};
  std::atomic<bool> stopping{false};
  std::atomic<bool> failed{false};
};

void initialize_fixture(WatchState* state, const TempDirectory& temporary) {
  state->root = temporary.path().string();
  state->exact_basename = "model.bin";
  state->exact_path = (temporary.path() / state->exact_basename).string();
  state->replacement_path = (temporary.path() / "model.next").string();
  if (!write_value(state->exact_path, 0))
    fail("could not create the exact-file fixture");
  for (std::size_t index = 0; index < kDirectoryFiles; ++index) {
    state->child_names[index] = "shard-" + std::to_string(index) + ".bin";
    state->child_paths[index] =
        (temporary.path() / state->child_names[index]).string();
    if (!write_value(state->child_paths[index], 0))
      fail("could not create the directory-burst fixture");
  }
}

std::size_t child_index(const WatchState& state,
                        std::string_view name) noexcept {
  for (std::size_t index = 0; index < state.child_names.size(); ++index) {
    if (name == state.child_names[index]) return index;
  }
  return state.child_names.size();
}

std::uint64_t full_directory_mask() noexcept {
  return (std::uint64_t{1} << kDirectoryFiles) - 1;
}

bool observe_exact(const WatchState& state, std::uint64_t expected) noexcept {
  std::uint64_t value = 0;
  return read_value(state.exact_path, &value) && value == expected;
}

void observe_child(const WatchState& state, std::size_t index,
                   std::uint64_t expected, std::uint64_t* mask) noexcept {
  if (index >= kDirectoryFiles) return;
  std::uint64_t value = 0;
  if (read_value(state.child_paths[index], &value) && value == expected)
    *mask |= std::uint64_t{1} << index;
}

void rescan_children(const WatchState& state, std::uint64_t expected,
                     std::uint64_t* mask) noexcept {
  for (std::size_t index = 0; index < kDirectoryFiles; ++index)
    observe_child(state, index, expected, mask);
}

bool publish_exact(WatchState* state, std::uint64_t sequence) noexcept {
  return write_value(state->replacement_path, sequence) &&
         hyle::platform::rename_path(state->replacement_path.c_str(),
                                     state->exact_path.c_str())
             .ok();
}

bool publish_directory(WatchState* state, std::uint64_t sequence) noexcept {
  for (const std::string& path : state->child_paths) {
    if (!write_value(path, sequence)) return false;
  }
  return true;
}

bool publish(WatchState* state, std::uint64_t sequence) noexcept {
  return state->shape == Shape::kExactReplace
             ? publish_exact(state, sequence)
             : publish_directory(state, sequence);
}

void warm_until_observable(WatchState* state, std::uint64_t sequence,
                           const char* failure) {
  const auto deadline = std::chrono::steady_clock::now() + kReadinessTimeout;
  do {
    if (!publish(state, sequence)) fail(failure);
    if (wait_at_least_for(state->completed, sequence, kReadinessRetry)) return;
  } while (std::chrono::steady_clock::now() < deadline);
  fail("watcher did not become observable during setup");
}

void mark_failed(WatchState* state) noexcept {
  state->failed.store(true, std::memory_order_release);
  state->ready.store(1, std::memory_order_release);
  state->ready.notify_all();
  state->armed.store((std::numeric_limits<std::uint64_t>::max)(),
                     std::memory_order_release);
  state->armed.notify_all();
  state->completed.store((std::numeric_limits<std::uint64_t>::max)(),
                         std::memory_order_release);
  state->completed.notify_all();
}

void finish_episode(WatchState* state, std::uint64_t sequence,
                    std::uint64_t contribution) noexcept {
  state->checksum.fetch_add(contribution, std::memory_order_relaxed);
  state->completed.store(sequence, std::memory_order_release);
  state->completed.notify_all();
}

Task<int> hyle_watch_owner(AllocatorRef allocator, IoContext* io,
                           WatchState* state) noexcept {
  const std::array target{
      state->shape == Shape::kExactReplace
          ? FileWatchTarget::file(state->exact_path)
          : FileWatchTarget::directory(state->root)};
  auto made = FileWatcher::create(allocator, *io, kOwner, target);
  if (!made) {
    mark_failed(state);
    co_return -1;
  }
  FileWatcher watcher = std::move(*made);
  state->ready.store(1, std::memory_order_release);
  state->ready.notify_all();

  for (std::uint64_t sequence = 1;; ++sequence) {
    state->armed.store(sequence, std::memory_order_release);
    state->armed.notify_all();
    std::uint64_t directory_mask = 0;
    bool observed = false;
    while (!observed) {
      auto batch = co_await watcher.next();
      if (!batch) {
        mark_failed(state);
        (void)watcher.close();
        co_return -1;
      }
      if (state->shape == Shape::kExactReplace) {
        bool routed = batch->rescan_required();
        for (const auto& event : batch->events())
          routed = routed || (event.target_index == 0);
        observed = routed && observe_exact(*state, sequence);
      } else {
        if (batch->rescan_required())
          rescan_children(*state, sequence, &directory_mask);
        for (const auto& event : batch->events()) {
          if (event.target_index != 0) continue;
          observe_child(*state, child_index(*state, event.relative_path),
                        sequence, &directory_mask);
        }
        observed = directory_mask == full_directory_mask();
      }
    }

    finish_episode(state, sequence,
                   sequence ^ (state->shape == Shape::kExactReplace
                                   ? 0x9e3779b97f4a7c15ULL
                                   : directory_mask));
    if (state->stopping.load(std::memory_order_acquire)) break;
  }

  if (!watcher.close().ok()) {
    mark_failed(state);
    co_return -1;
  }
  co_return 0;
}

class HyleHarness {
 public:
  explicit HyleHarness(Shape shape)
      : state_(shape), allocator_(system_allocator()),
        scheduler_(1, allocator_, 256), io_(scheduler_) {
    initialize_fixture(&state_, temporary_);
    if (!scheduler_.valid() || !io_.valid())
      fail("HYLE watcher runtime construction failed");
    owner_ = hyle::sched::spawn_task_on(
        scheduler_, kOwner, allocator_,
        hyle_watch_owner(allocator_, &io_, &state_));
    if (!owner_.valid()) fail("HYLE watcher owner admission failed");
    wait_at_least(state_.ready, std::uint64_t{1});
    if (state_.failed.load(std::memory_order_acquire))
      fail("HYLE watcher initialization failed");
  }

  ~HyleHarness() {
    if (!owner_.valid()) return;
    if (!state_.failed.load(std::memory_order_acquire)) {
      state_.stopping.store(true, std::memory_order_release);
      run(1);
    }
    auto result = owner_.join();
    if (!result || *result != 0) fail("HYLE watcher owner failed");
    scheduler_.shutdown();
  }

  HyleHarness(const HyleHarness&) = delete;
  HyleHarness& operator=(const HyleHarness&) = delete;

  void run(std::size_t episodes) {
    for (std::size_t count = 0; count < episodes; ++count) {
      const std::uint64_t sequence = ++issued_;
      wait_at_least(state_.armed, sequence);
      state_.requested.store(sequence, std::memory_order_release);
      if (!publish(&state_, sequence)) fail("HYLE fixture publication failed");
      wait_at_least(state_.completed, sequence);
      if (state_.failed.load(std::memory_order_acquire))
        fail("HYLE watcher episode failed");
    }
    ankerl::nanobench::doNotOptimizeAway(
        state_.checksum.load(std::memory_order_relaxed));
  }

  void warmup() {
    const std::uint64_t sequence = ++issued_;
    wait_at_least(state_.armed, sequence);
    state_.requested.store(sequence, std::memory_order_release);
    warm_until_observable(&state_, sequence,
                          "HYLE readiness publication failed");
    if (state_.failed.load(std::memory_order_acquire))
      fail("HYLE watcher readiness failed");
  }

 private:
  TempDirectory temporary_;
  WatchState state_;
  AllocatorRef allocator_;
  Scheduler scheduler_;
  IoContext io_;
  JoinHandle<int> owner_{};
  std::uint64_t issued_ = 0;
};

class LibuvHarness {
 public:
  explicit LibuvHarness(Shape shape) : state_(shape) {
    initialize_fixture(&state_, temporary_);
    worker_ = std::thread([this] { worker_main(); });
    wait_at_least(state_.ready, std::uint64_t{1});
    if (state_.failed.load(std::memory_order_acquire))
      fail("libuv watcher initialization failed");
  }

  ~LibuvHarness() {
    if (worker_.joinable()) {
      if (async_initialized_) {
        if (uv_async_send(&stop_) != 0) fail("libuv stop signal failed");
      }
      worker_.join();
    }
    if (state_.failed.load(std::memory_order_acquire))
      fail("libuv watcher owner failed");
  }

  LibuvHarness(const LibuvHarness&) = delete;
  LibuvHarness& operator=(const LibuvHarness&) = delete;

  void run(std::size_t episodes) {
    for (std::size_t count = 0; count < episodes; ++count) {
      const std::uint64_t sequence = ++issued_;
      wait_at_least(state_.armed, sequence);
      state_.requested.store(sequence, std::memory_order_release);
      if (!publish(&state_, sequence)) fail("libuv fixture publication failed");
      wait_at_least(state_.completed, sequence);
      if (state_.failed.load(std::memory_order_acquire))
        fail("libuv watcher episode failed");
    }
    ankerl::nanobench::doNotOptimizeAway(
        state_.checksum.load(std::memory_order_relaxed));
  }

  void warmup() {
    const std::uint64_t sequence = ++issued_;
    wait_at_least(state_.armed, sequence);
    state_.requested.store(sequence, std::memory_order_release);
    warm_until_observable(&state_, sequence,
                          "libuv readiness publication failed");
    if (state_.failed.load(std::memory_order_acquire))
      fail("libuv watcher readiness failed");
  }

 private:
  static void watch_callback(uv_fs_event_t* handle, const char* filename,
                             int, int status) {
    auto* self = static_cast<LibuvHarness*>(handle->data);
    if (status < 0) {
      mark_failed(&self->state_);
      return;
    }
    const std::uint64_t sequence =
        self->state_.requested.load(std::memory_order_acquire);
    if (sequence == 0 ||
        self->state_.completed.load(std::memory_order_acquire) >= sequence)
      return;

    bool observed = false;
    if (self->state_.shape == Shape::kExactReplace) {
      const bool routed = filename == nullptr ||
                          std::string_view(filename) ==
                              self->state_.exact_basename;
      observed = routed && observe_exact(self->state_, sequence);
    } else {
      if (self->observed_sequence_ != sequence) {
        self->observed_sequence_ = sequence;
        self->directory_mask_ = 0;
      }
      if (filename == nullptr) {
        rescan_children(self->state_, sequence, &self->directory_mask_);
      } else {
        const std::string_view name(filename);
        if (name.find('/') == std::string_view::npos &&
            name.find('\\') == std::string_view::npos) {
          observe_child(self->state_, child_index(self->state_, name),
                        sequence, &self->directory_mask_);
        }
      }
      observed = self->directory_mask_ == full_directory_mask();
    }
    if (!observed) return;

    finish_episode(&self->state_, sequence,
                   sequence ^ (self->state_.shape == Shape::kExactReplace
                                   ? 0x9e3779b97f4a7c15ULL
                                   : self->directory_mask_));
    self->state_.armed.store(sequence + 1, std::memory_order_release);
    self->state_.armed.notify_all();
  }

  static void stop_callback(uv_async_t* async) {
    auto* self = static_cast<LibuvHarness*>(async->data);
    (void)uv_fs_event_stop(&self->watch_);
    uv_close(reinterpret_cast<uv_handle_t*>(&self->watch_), nullptr);
    uv_close(reinterpret_cast<uv_handle_t*>(&self->stop_), nullptr);
  }

  void worker_main() noexcept {
    if (uv_loop_init(&loop_) != 0) {
      mark_failed(&state_);
      return;
    }
    if (uv_fs_event_init(&loop_, &watch_) != 0) {
      (void)uv_loop_close(&loop_);
      mark_failed(&state_);
      return;
    }
    watch_.data = this;
    const int started = uv_fs_event_start(&watch_, &watch_callback,
                                          state_.root.c_str(), 0);
    if (started != 0) {
      uv_close(reinterpret_cast<uv_handle_t*>(&watch_), nullptr);
      (void)uv_run(&loop_, UV_RUN_DEFAULT);
      (void)uv_loop_close(&loop_);
      mark_failed(&state_);
      return;
    }
    if (uv_async_init(&loop_, &stop_, &stop_callback) != 0) {
      (void)uv_fs_event_stop(&watch_);
      uv_close(reinterpret_cast<uv_handle_t*>(&watch_), nullptr);
      (void)uv_run(&loop_, UV_RUN_DEFAULT);
      (void)uv_loop_close(&loop_);
      mark_failed(&state_);
      return;
    }
    stop_.data = this;
    async_initialized_ = true;
    state_.armed.store(1, std::memory_order_release);
    state_.armed.notify_all();
    state_.ready.store(1, std::memory_order_release);
    state_.ready.notify_all();
    (void)uv_run(&loop_, UV_RUN_DEFAULT);
    if (uv_loop_close(&loop_) != 0) mark_failed(&state_);
  }

  TempDirectory temporary_;
  WatchState state_;
  uv_loop_t loop_{};
  uv_fs_event_t watch_{};
  uv_async_t stop_{};
  std::thread worker_;
  std::atomic<bool> async_initialized_{false};
  std::uint64_t issued_ = 0;
  std::uint64_t observed_sequence_ = 0;
  std::uint64_t directory_mask_ = 0;
};

void run_shape(Shape shape, const char* title, std::size_t episodes,
               std::size_t events_per_episode, bool peer_first) {
  ankerl::nanobench::Bench bench;
  bench.title(title)
      .relative(true)
      .batch(static_cast<std::uint64_t>(episodes * events_per_episode))
      .unit("change")
      .minEpochTime(std::chrono::nanoseconds(0))
      .minEpochIterations(1)
      .epochs(1);

  // macOS implements both routes with FSEvents streams whose 50 ms delivery
  // windows share daemon state. Keeping both streams alive simultaneously
  // makes the second measured route inherit the first stream's batching phase
  // and reverses the result with execution order. A real consumer owns one
  // route, so establish, warm, measure, and retire one route at a time. The
  // equal uncharged quiescence also drains readiness traffic before timing.
  auto run_hyle = [&] {
    HyleHarness hyle(shape);
    hyle.warmup();
    std::this_thread::sleep_for(kRouteQuiescence);
    bench.run("HYLE FileWatcher", [&] { hyle.run(episodes); });
  };
  auto run_peer = [&] {
    LibuvHarness peer(shape);
    peer.warmup();
    std::this_thread::sleep_for(kRouteQuiescence);
    bench.run("libuv uv_fs_event", [&] { peer.run(episodes); });
  };

  if (peer_first) {
    run_peer();
    std::this_thread::sleep_for(kRouteQuiescence);
    run_hyle();
  } else {
    run_hyle();
    std::this_thread::sleep_for(kRouteQuiescence);
    run_peer();
  }
}

#if HYLE_IO_WATCH_P1_PLACEMENT
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
std::uint64_t placement_probe(std::uint64_t value) noexcept {
  static volatile std::uint64_t padding[257]{};
  padding[value % 257] = value ^ 0xd6e8feb86659fd93ULL;
  return padding[(value + 17) % 257];
}
#endif

}  // namespace

int main() {
#if HYLE_IO_WATCH_P1_PLACEMENT
  ankerl::nanobench::doNotOptimizeAway(placement_probe(19));
#endif
  const auto qos = hyle::bench::perf::configure_thread_qos();
  const bool peer_first = std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
  std::fprintf(stderr,
               "# HYLE_IO_WATCH_RUN execution_order=%s apple_qos=%s "
               "qos_ok=%s run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               qos.name, qos.ok,
               hyle::bench::perf::run_nonce());
  run_shape(Shape::kExactReplace, "exact-file atomic replace reload",
            kExactEpisodes, 1, peer_first);
  run_shape(Shape::kDirectoryBurst, "nonrecursive directory burst reload",
            kDirectoryEpisodes, kDirectoryFiles, peer_first);
  return 0;
}
