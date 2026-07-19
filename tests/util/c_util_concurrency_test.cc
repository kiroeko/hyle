#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "hyle/c/util.h"

namespace {

void* return_context(void* context, const void*) { return context; }

struct ReentrantFactoryContext {
  hyle_util_registry* registry;
  bool reentered;
};

void* reenter_registry(void* context, const void*) {
  auto* state = static_cast<ReentrantFactoryContext*>(context);
  state->reentered =
      hyle_util_registry_has(state->registry, "reentrant") != 0;
  return state;
}

void count_destroy(void*, void* context) {
  static_cast<std::atomic<int>*>(context)->fetch_add(
      1, std::memory_order_relaxed);
}

}  // namespace

TEST(CUtilRef, ConcurrentRetainReleaseKeepsObjectAlive) {
  int object = 7;
  std::atomic<int> destroys{0};
  hyle_util_ref* root =
      hyle_util_ref_create(&object, &count_destroy, &destroys);
  ASSERT_NE(root, nullptr);
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 8; ++thread) {
    threads.emplace_back([&] {
      for (int i = 0; i < 50000; ++i) {
        hyle_util_ref* retained = hyle_util_ref_retain(root);
        ASSERT_EQ(retained, root);
        EXPECT_EQ(hyle_util_ref_get(retained), &object);
        hyle_util_ref_release(retained);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(hyle_util_ref_use_count(root), 1u);
  EXPECT_EQ(destroys.load(std::memory_order_relaxed), 0);
  hyle_util_ref_release(root);
  EXPECT_EQ(destroys.load(std::memory_order_relaxed), 1);
}

TEST(CUtilRegistry, ConcurrentRegistrationAndQueries) {
  constexpr std::size_t kKeys = 128;
  hyle_util_registry* registry = hyle_util_registry_create();
  ASSERT_NE(registry, nullptr);
  std::array<std::array<char, 16>, kKeys> keys{};
  std::array<int, kKeys> payloads{};
  for (std::size_t i = 0; i < kKeys; ++i) {
    std::snprintf(keys[i].data(), keys[i].size(), "key-%zu", i);
    payloads[i] = static_cast<int>(i);
  }
  std::atomic<bool> start{false};
  std::atomic<bool> done{false};
  std::thread reader([&] {
    while (!start.load(std::memory_order_acquire)) {}
    while (!done.load(std::memory_order_acquire)) {
      (void)hyle_util_registry_has(registry, "missing");
      (void)hyle_util_registry_size(registry);
    }
  });
  std::vector<std::thread> writers;
  for (std::size_t worker = 0; worker < 4; ++worker) {
    writers.emplace_back([&, worker] {
      while (!start.load(std::memory_order_acquire)) {}
      for (std::size_t i = worker; i < kKeys; i += 4) {
        ASSERT_TRUE(hyle_util_registry_register(
            registry, keys[i].data(), &return_context, &payloads[i],
            HYLE_UTIL_REGISTRY_DEFAULT, "payload"));
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& writer : writers) writer.join();
  done.store(true, std::memory_order_release);
  reader.join();

  EXPECT_EQ(hyle_util_registry_size(registry), kKeys);
  for (std::size_t i = 0; i < kKeys; ++i) {
    EXPECT_EQ(hyle_util_registry_create_object(
                  registry, keys[i].data(), nullptr),
              &payloads[i]);
  }
  hyle_util_registry_destroy(registry);
}

TEST(CUtilRegistry, CreatorMayReenterRegistry) {
  hyle_util_registry* registry = hyle_util_registry_create();
  ASSERT_NE(registry, nullptr);
  ReentrantFactoryContext state{registry, false};
  ASSERT_TRUE(hyle_util_registry_register(
      registry, "reentrant", &reenter_registry, &state,
      HYLE_UTIL_REGISTRY_DEFAULT, "reentrant factory"));
  EXPECT_EQ(hyle_util_registry_create_object(
                registry, "reentrant", nullptr),
            &state);
  EXPECT_TRUE(state.reentered);
  hyle_util_registry_destroy(registry);
}

TEST(CUtilRng, SequentialThreadsReceiveDistinctTlsStreams) {
  std::array<std::array<std::uint64_t, 4>, 8> states{};
  for (auto& state : states) {
    std::thread worker([&] {
      hyle_util_rng* rng = hyle_util_thread_rng();
      ASSERT_NE(rng, nullptr);
      for (std::size_t lane = 0; lane < state.size(); ++lane) {
        state[lane] = rng->state[lane];
      }
    });
    worker.join();
  }
  for (std::size_t left = 0; left < states.size(); ++left) {
    for (std::size_t right = left + 1; right < states.size(); ++right) {
      EXPECT_NE(states[left], states[right]);
    }
  }
}

TEST(CUtilErrors, LastErrorIsThreadLocal) {
  ASSERT_NE(hyle_util_thread_rng(), nullptr);
  ASSERT_STREQ(hyle_util_last_error(), "");
  bool worker_saw_error = false;
  std::thread worker([&] {
    EXPECT_FALSE(hyle_util_rng_is_valid(nullptr));
    worker_saw_error = hyle_util_last_error()[0] != '\0';
  });
  worker.join();
  EXPECT_TRUE(worker_saw_error);
  EXPECT_STREQ(hyle_util_last_error(), "");
}

TEST(CUtilSpinLock, ConcurrentCriticalSectionIsExclusive) {
  hyle_util_spin_lock* lock = hyle_util_spin_lock_create();
  ASSERT_NE(lock, nullptr);
  std::uint64_t counter = 0;
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 8; ++thread) {
    threads.emplace_back([&] {
      for (int i = 0; i < 50000; ++i) {
        ASSERT_TRUE(hyle_util_spin_lock_lock(lock));
        ++counter;
        ASSERT_TRUE(hyle_util_spin_lock_unlock(lock));
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(counter, 8ull * 50000ull);
  EXPECT_TRUE(hyle_util_spin_lock_destroy(lock));
}
