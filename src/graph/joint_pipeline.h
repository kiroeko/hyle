#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <vector>

#include "hyle/core/assert.h"
#include "hyle/error/result.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::graph::detail {

// Dependency metadata for the existing joint composition drivers. This owns
// no executor, queue, thread, timer, native object or completion. The native
// root and CPU child use their original Scheduler and remain the sole producer
// of each actual endpoint. All mutable frontier calls are externally serialized
// by the composition owner's short monitor; math/waits run outside that lock.
struct JointPipelinePlan final {
  enum class Lane : std::uint8_t { kNative = 0, kCpu = 1 };
  struct Job final {
    Lane lane = Lane::kNative;
    std::size_t dependency_begin = 0, dependency_count = 0;
  };
  template<class T> using Vector = std::vector<T, mem::StdAllocator<T>>;
  explicit JointPipelinePlan(mem::AllocatorRef allocator)
      : jobs(mem::StdAllocator<Job>(allocator)),
        dependencies(mem::StdAllocator<std::size_t>(allocator)),
        successor_begin(mem::StdAllocator<std::size_t>(allocator)),
        successors(mem::StdAllocator<std::size_t>(allocator)) {}
  Vector<Job> jobs;
  Vector<std::size_t> dependencies, successor_begin, successors;
  bool has_parallel_lanes = false;

  // Cold append, inside the original compile/prepare allocation boundary.
  // Prerequisites name earlier jobs; original lane/memory-reuse constraints
  // must be supplied alongside data/control edges by the actual plan owner.
  void append(Lane lane, std::span<const std::size_t> prerequisites) {
    const auto begin = dependencies.size();
    dependencies.insert(dependencies.end(), prerequisites.begin(), prerequisites.end());
    jobs.push_back({lane, begin, prerequisites.size()});
  }

  Status prepare() noexcept {
    try {
      successor_begin.assign(jobs.size() + 1, 0);
      successors.resize(dependencies.size());
      has_parallel_lanes = false;
      unsigned lanes = 0;
      bool has_unordered_jobs = false;
      for (std::size_t i = 0; i < jobs.size(); ++i) {
        const auto &job = jobs[i];
        if (static_cast<unsigned>(job.lane) >= 2 ||
            job.dependency_begin > dependencies.size() ||
            job.dependency_count > dependencies.size() - job.dependency_begin)
          return Status::InvalidArgument("joint job dependency range is invalid");
        lanes |= 1u << static_cast<unsigned>(job.lane);
        bool follows_previous = false;
        for (auto edge = job.dependency_begin; edge < job.dependency_begin + job.dependency_count; ++edge) {
          const auto source = dependencies[edge];
          if (source >= i) return Status::InvalidArgument("joint dependency must precede its job");
          ++successor_begin[source + 1];
          follows_previous = follows_previous || source == i - 1;
        }
        if (i != 0 && !follows_previous) has_unordered_jobs = true;
      }
      // Only a fully ordered graph is a safe serial fast path. Comparing just
      // adjacent lane changes misses a later ready native branch behind an
      // earlier native job that still depends on CPU work.
      has_parallel_lanes = lanes == 3 && has_unordered_jobs;
      for (std::size_t i = 1; i < successor_begin.size(); ++i)
        successor_begin[i] += successor_begin[i - 1];
      // Reuse the final offsets as insertion cursors, then restore them in one
      // reverse pass. No per-edge search or uncharged second adjacency table.
      for (std::size_t i = 0; i < jobs.size(); ++i) {
        const auto &job = jobs[i];
        for (auto edge = job.dependency_begin; edge < job.dependency_begin + job.dependency_count; ++edge)
          successors[successor_begin[dependencies[edge]]++] = i;
      }
      for (auto i = jobs.size(); i != 0; --i)
        successor_begin[i] = successor_begin[i - 1];
      successor_begin[0] = 0;
      return Status::Ok();
    } catch (const std::bad_alloc &) {
      return Status::ResourceExhausted("joint dependency metadata allocation");
    }
  }

  bool parallel_after(std::span<const std::uint8_t> completed) const noexcept {
    HYLE_FATAL_ASSERT(completed.size() == jobs.size());
    auto previous = std::numeric_limits<std::size_t>::max();
    unsigned lanes = 0;
    bool unordered = false;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      if (completed[i]) continue;
      lanes |= 1u << static_cast<unsigned>(jobs[i].lane);
      bool follows_previous = false;
      const auto &job = jobs[i];
      for (auto edge = job.dependency_begin; edge < job.dependency_begin + job.dependency_count; ++edge)
        follows_previous = follows_previous || dependencies[edge] == previous;
      if (previous != std::numeric_limits<std::size_t>::max() && !follows_previous) unordered = true;
      previous = i;
    }
    return lanes == 3 && unordered;
  }
};

class JointPipelineFrontier final {
public:
  using Lane = JointPipelinePlan::Lane;
  static constexpr auto kNone = std::numeric_limits<std::size_t>::max();
  explicit JointPipelineFrontier(mem::AllocatorRef allocator)
      : remaining_(mem::StdAllocator<std::size_t>(allocator)),
        next_(mem::StdAllocator<std::size_t>(allocator)) {}

  Status prepare(const JointPipelinePlan &plan) noexcept {
    if (plan.successor_begin.size() != plan.jobs.size() + 1 ||
        plan.jobs.size() >= kNone - 1)
      return Status::InvalidArgument("joint plan is not prepared");
    try {
      remaining_.resize(plan.jobs.size());
      next_.resize(plan.jobs.size());
    } catch (const std::bad_alloc &) {
      return Status::ResourceExhausted("joint per-slot frontier allocation");
    }
    reset(plan);
    return Status::Ok();
  }

  // Warm reset cannot allocate. Called only after both original drivers have
  // returned and their accepted native owners have actually retired.
  void reset(const JointPipelinePlan &plan, std::span<const std::uint8_t> completed = {}) noexcept {
    HYLE_FATAL_ASSERT(remaining_.size() == plan.jobs.size());
    HYLE_FATAL_ASSERT(completed.empty() || completed.size() == plan.jobs.size());
    heads_.fill(kNone); tails_.fill(kNone); completed_ = 0;
    std::fill(next_.begin(), next_.end(), kNone);
    for (std::size_t i = 0; i < plan.jobs.size(); ++i) {
      const bool ready = !completed.empty() && completed[i];
      remaining_[i] = ready ? kNone - 1 : plan.jobs[i].dependency_count;
      if (ready) ++completed_;
    }
    // Already-ready host/native inputs are facts of the frozen binding, not
    // queue jobs. Remove their outgoing obligations once without submitting a
    // no-op or making an independent CPU branch wait for native preparation.
    for (std::size_t i = 0; i < plan.jobs.size(); ++i) {
      if (remaining_[i] != kNone - 1) continue;
      for (auto edge = plan.successor_begin[i]; edge < plan.successor_begin[i + 1]; ++edge) {
        auto &count = remaining_[plan.successors[edge]];
        if (count != kNone - 1) { HYLE_FATAL_ASSERT(count != 0); --count; }
      }
    }
    for (std::size_t i = 0; i < plan.jobs.size(); ++i)
      if (remaining_[i] == 0) enqueue(plan, i);
  }
  bool ready(Lane lane) const noexcept { return heads_[static_cast<unsigned>(lane)] != kNone; }
  bool done() const noexcept { return completed_ == remaining_.size(); }
  std::size_t take(Lane lane) noexcept {
    const auto index = static_cast<unsigned>(lane);
    const auto job = heads_[index];
    if (job == kNone) return kNone;
    heads_[index] = next_[job];
    if (heads_[index] == kNone) tails_[index] = kNone;
    HYLE_FATAL_ASSERT(remaining_[job] == 0);
    remaining_[job] = kNone; // claimed, distinct from completed
    return job;
  }
  void complete(const JointPipelinePlan &plan, std::size_t job) noexcept {
    HYLE_FATAL_ASSERT(job < remaining_.size() && remaining_[job] == kNone);
    remaining_[job] = kNone - 1;
    ++completed_;
    for (auto edge = plan.successor_begin[job]; edge < plan.successor_begin[job + 1]; ++edge) {
      const auto successor = plan.successors[edge];
      HYLE_FATAL_ASSERT(remaining_[successor] > 0 && remaining_[successor] < kNone - 1);
      if (--remaining_[successor] == 0) enqueue(plan, successor);
    }
  }
private:
  void enqueue(const JointPipelinePlan &plan, std::size_t job) noexcept {
    const auto lane = static_cast<unsigned>(plan.jobs[job].lane);
    if (tails_[lane] == kNone) heads_[lane] = job;
    else next_[tails_[lane]] = job;
    tails_[lane] = job;
  }
  JointPipelinePlan::Vector<std::size_t> remaining_, next_;
  std::array<std::size_t, 2> heads_{kNone, kNone}, tails_{kNone, kNone};
  std::size_t completed_ = 0;
};

struct NativePipelineStage final {
  enum class Kind : std::uint8_t { kRepack, kUpload, kDownload, kFragment };
  Kind kind = Kind::kFragment;
  std::size_t index = 0, fragment = SIZE_MAX;
};

// The two original native joint owners have the same data/control boundary
// contract. Only their native-use flag differs. Project real conversion jobs
// once; each route still executes its original Program/transfer/page owners.
template<class Plan, class NativeUse>
Status prepare_native_pipeline(Plan &plan, NativeUse native_use) noexcept {
  using Lane = JointPipelinePlan::Lane;
  using Kind = NativePipelineStage::Kind;
  constexpr auto none = std::numeric_limits<std::size_t>::max();
  try {
    JointPipelinePlan::Vector<std::size_t> host_ready(plan.values.size(), none,
        mem::StdAllocator<std::size_t>(plan.allocator));
    JointPipelinePlan::Vector<std::size_t> native_ready(plan.values.size(), none,
        mem::StdAllocator<std::size_t>(plan.allocator));
    JointPipelinePlan::Vector<std::size_t> fragments(plan.fragments.size(), none,
        mem::StdAllocator<std::size_t>(plan.allocator));
    JointPipelinePlan::Vector<std::size_t> dependencies{mem::StdAllocator<std::size_t>(plan.allocator)};
    auto append = [&](Kind kind, std::size_t index, std::size_t fragment, Lane lane,
                      std::span<const std::size_t> prerequisites) {
      const auto job = plan.pipeline.jobs.size();
      plan.pipeline.append(lane, prerequisites);
      plan.pipeline_stages.push_back({kind, index, fragment});
      return job;
    };
    for (const auto id : plan.inputs) {
      const auto &value = plan.values[id];
      auto repack = none;
      if (value.repack.valid()) repack = append(Kind::kRepack, id, none, Lane::kCpu, {});
      if (value.cpu_use) host_ready[id] = append(Kind::kDownload, id, none, Lane::kNative, {});
      if (native_use(value)) {
        const std::span<const std::size_t> source = repack == none
            ? std::span<const std::size_t>{} : std::span<const std::size_t>{&repack, 1};
        native_ready[id] = append(Kind::kUpload, id, none, Lane::kNative, source);
      }
    }
    for (std::size_t i = 0; i < plan.fragments.size(); ++i) {
      const auto &fragment = plan.fragments[i];
      const bool cpu = fragment.target.ordinal() == 1;
      dependencies.clear();
      for (const auto source : fragment.dependencies) {
        if (source >= i) return Status::Internal("native pipeline control/data predecessor is not earlier");
        dependencies.push_back(fragments[source]);
      }
      for (const auto input : fragment.inputs) {
        const auto ready = cpu ? host_ready[input] : native_ready[input];
        if (ready != none) dependencies.push_back(ready);
      }
      const auto math = append(Kind::kFragment, i, i, cpu ? Lane::kCpu : Lane::kNative, dependencies);
      fragments[i] = math;
      const std::span<const std::size_t> source{&math, 1};
      for (const auto id : fragment.outputs) {
        const auto &value = plan.values[id];
        host_ready[id] = cpu ? math : none;
        native_ready[id] = cpu ? none : math;
        if (cpu && native_use(value)) native_ready[id] = append(Kind::kUpload, id, i, Lane::kNative, source);
        if (!cpu && value.cpu_use) host_ready[id] = append(Kind::kDownload, id, i, Lane::kNative, source);
      }
    }
    return plan.pipeline.prepare();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("native joint boundary dependency metadata");
  }
}

} // namespace hyle::graph::detail
